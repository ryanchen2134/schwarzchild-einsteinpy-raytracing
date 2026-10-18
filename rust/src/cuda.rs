//! The CUDA backend: the FANTASY kernel in `fantasy.cu`, compiled with NVRTC at
//! start-up and launched through [`cudarc`]. Built only with `--features cuda-12`
//! (or `cuda-11`); needs `libcuda` (the driver) and `libnvrtc` (from the CUDA
//! toolkit or the NVRTC redistributable) at run time, nothing at build time.
//!
//! Rounding: the kernel is compiled with `--fmad=false`, precise division and
//! square root, and no flush-to-zero, so `+ − × ÷ √` round exactly as on the CPU.
//! The device `sin`/`cos`/`sincos` are specified to 2 ulp rather than correctly
//! rounded and are not glibc's code, so GPU and CPU results agree to a few ulp per
//! step, not bit for bit: compare `q`/`p` with a tolerance and expect a few pixels
//! on a classification boundary to differ.
//!
//! Each launch advances the unfinished rays by at most [`STEPS_PER_LAUNCH`] steps
//! and the host relaunches until every ray has exited or used its budget: no
//! single kernel runs long enough to trip a display GPU's watchdog, and progress is
//! reported between rounds.
//!
//! Disk crossings are detected in the kernel with the arithmetic of
//! [`crate::integrator::CrossingDetector`]; the crossing count and the hit live in
//! their own buffers so that a ray's detection state survives relaunches.

use crate::backend::CUDA_KERNEL_SOURCE as KERNEL_SOURCE;
use crate::integrator::{DiskHit, ExitReason, IntegrationResult, IntegratorSettings, Ray};
use anyhow::{anyhow, bail, Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::sync::Arc;

const KERNEL_NAME: &str = "fantasy_integrate";
/// Schwarzschild radius in the kernel's units.
const RS_UNIT: f64 = 2.0;
/// Rays per chunk of device buffers.
pub const CHUNK: usize = 1 << 16;
/// Steps a ray advances per launch; bounds a launch to well under a second.
pub const STEPS_PER_LAUNCH: usize = 4096;
/// Threads per block: the kernel keeps four 4-vectors per copy in registers, so a
/// modest block keeps occupancy up without hitting the register limit.
const THREADS_PER_BLOCK: u32 = 128;

/// Which run-time library is absent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MissingLibrary {
    /// `libcuda`: no CUDA driver on this machine.
    Driver,
    /// `libnvrtc`: the driver is present but the CUDA toolkit's runtime compiler is not.
    Nvrtc,
}

/// Can `libcuda` and `libnvrtc` be loaded? `cudarc` aborts the process when they
/// cannot, so this is checked first.
pub fn probe_libraries() -> std::result::Result<(), MissingLibrary> {
    fn any_loads(names: &[&str]) -> bool {
        names.iter().any(|n| {
            // SAFETY: loading a CUDA library runs its initialisers, which is what
            // cudarc does next anyway; we discard the handle without calling into it.
            unsafe { libloading::Library::new(n) }.is_ok()
        })
    }
    const DRIVER: &[&str] = &["libcuda.so.1", "libcuda.so", "nvcuda.dll", "libcuda.dylib"];
    const NVRTC: &[&str] = &[
        "libnvrtc.so",
        "libnvrtc.so.12",
        "libnvrtc.so.11.2",
        "nvrtc64_120_0.dll",
        "nvrtc64_112_0.dll",
    ];
    if !any_loads(DRIVER) {
        return Err(MissingLibrary::Driver);
    }
    if !any_loads(NVRTC) {
        return Err(MissingLibrary::Nvrtc);
    }
    Ok(())
}

/// Run `f`, turning a panic (cudarc panics when a library or symbol is missing) into
/// an error carrying the panic message, without printing the default panic report.
pub fn catch_cuda_panic<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    std::panic::set_hook(hook);
    match outcome {
        Ok(r) => r,
        Err(payload) => {
            let msg = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "unknown panic".to_string());
            Err(anyhow!("CUDA unavailable: {msg}"))
        }
    }
}

/// Number of CUDA devices, 0 when the driver cannot be initialised.
pub fn device_count() -> usize {
    CudaContext::device_count()
        .map(|c| c.max(0) as usize)
        .unwrap_or(0)
}

pub struct CudaIntegrator {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    func: CudaFunction,
    name: String,
}

impl CudaIntegrator {
    /// Open device `ordinal` and compile the kernel.
    pub fn new(ordinal: usize) -> Result<Self> {
        let count = device_count();
        if ordinal >= count {
            bail!(
                "CUDA device {ordinal} does not exist; {count} device(s) present (valid --cuda-device values: 0..{})",
                count.saturating_sub(1)
            );
        }
        let ctx = CudaContext::new(ordinal)
            .with_context(|| format!("cannot open CUDA device {ordinal}"))?;
        let name = ctx.name().unwrap_or_else(|_| "unknown device".to_string());
        let opts = CompileOptions {
            fmad: Some(false),
            prec_div: Some(true),
            prec_sqrt: Some(true),
            ftz: Some(false),
            name: Some("fantasy.cu".to_string()),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(KERNEL_SOURCE, opts)
            .map_err(|e| anyhow!("NVRTC could not compile the FANTASY kernel: {e:?}"))?;
        let module = ctx.load_module(ptx).context(
            "cannot load the FANTASY kernel module (CUDA_ERROR_UNSUPPORTED_PTX_VERSION means \
             libnvrtc is newer than the CUDA driver: update the driver, or use the libnvrtc of \
             a toolkit no newer than the driver)",
        )?;
        let func = module
            .load_function(KERNEL_NAME)
            .with_context(|| format!("kernel {KERNEL_NAME} not found in the compiled module"))?;
        let stream = ctx.default_stream();
        Ok(Self {
            ctx,
            stream,
            _module: module,
            func,
            name,
        })
    }

    pub fn device_name(&self) -> &str {
        &self.name
    }

    pub fn ordinal(&self) -> usize {
        self.ctx.ordinal()
    }

    /// Integrate `rays` on the device. Inputs and outputs are in the caller's length
    /// unit; the rescaling to units of `M` happens here. `on_ray_done` is called once
    /// per ray, as rays finish between launches.
    pub fn integrate_batch(
        &self,
        rays: &[Ray],
        rs: f64,
        settings: &IntegratorSettings,
        on_ray_done: &(dyn Fn() + Sync),
    ) -> Result<Vec<IntegrationResult>> {
        let mass = rs / RS_UNIT;
        let mut results = Vec::with_capacity(rays.len());
        for chunk in rays.chunks(CHUNK) {
            results.extend(self.integrate_chunk(chunk, mass, settings, on_ray_done)?);
        }
        Ok(results)
    }

    fn integrate_chunk(
        &self,
        rays: &[Ray],
        mass: f64,
        settings: &IntegratorSettings,
        on_ray_done: &(dyn Fn() + Sync),
    ) -> Result<Vec<IntegrationResult>> {
        let n = rays.len();
        let mut q_host = Vec::with_capacity(4 * n);
        let mut p_host = Vec::with_capacity(4 * n);
        let mut phi_host = Vec::with_capacity(n);
        for ray in rays {
            let (q, p) = (ray.q0, ray.p0);
            q_host.extend_from_slice(&[q[0] / mass, q[1] / mass, q[2], q[3]]);
            p_host.extend_from_slice(&[p[0], p[1], p[2] / mass, p[3] / mass]);
            // Without a band no ray detects crossings, whatever azimuth it carries.
            phi_host.push(if settings.disk.is_some() {
                ray.disk_phi
            } else {
                f64::NAN
            });
        }
        let (disk_r_in, disk_r_out) = match settings.disk {
            Some(band) => (band.r_in / mass, band.r_out / mass),
            None => (0.0, 0.0),
        };
        let stream = &self.stream;
        let mut q1 = stream
            .memcpy_stod(&q_host)
            .context("copying rays to the device")?;
        let mut p1 = stream
            .memcpy_stod(&p_host)
            .context("copying rays to the device")?;
        let mut q2 = stream
            .memcpy_stod(&q_host)
            .context("copying rays to the device")?;
        let mut p2 = stream
            .memcpy_stod(&p_host)
            .context("copying rays to the device")?;
        let mut status = stream
            .alloc_zeros::<i32>(n)
            .context("allocating device buffers")?;
        let mut n_steps = stream
            .alloc_zeros::<i32>(n)
            .context("allocating device buffers")?;
        let disk_phi = stream
            .memcpy_stod(&phi_host)
            .context("copying disk azimuths to the device")?;
        let mut crossings = stream
            .alloc_zeros::<i32>(n)
            .context("allocating device buffers")?;
        let mut hit_r = stream
            .alloc_zeros::<f64>(n)
            .context("allocating device buffers")?;
        let mut hit_phi = stream
            .alloc_zeros::<f64>(n)
            .context("allocating device buffers")?;

        let n_i32 = i32::try_from(n).context("too many rays for one launch")?;
        let steps_i32 = i32::try_from(settings.steps).context("--steps does not fit in an int")?;
        let round_i32 = i32::try_from(STEPS_PER_LAUNCH.min(settings.steps))?;
        let delta = settings.delta / mass;
        let r_max = settings.r_max / mass;
        let r_capture = settings.r_capture / mass;
        let rs_unit = RS_UNIT;
        let omega = settings.omega;
        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(THREADS_PER_BLOCK), 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        let max_rounds = settings.steps.div_ceil(STEPS_PER_LAUNCH);
        let mut finished = 0usize;
        for _round in 0..max_rounds {
            let mut launch = stream.launch_builder(&self.func);
            launch
                .arg(&mut q1)
                .arg(&mut p1)
                .arg(&mut q2)
                .arg(&mut p2)
                .arg(&mut status)
                .arg(&mut n_steps)
                .arg(&disk_phi)
                .arg(&mut crossings)
                .arg(&mut hit_r)
                .arg(&mut hit_phi)
                .arg(&n_i32)
                .arg(&steps_i32)
                .arg(&round_i32)
                .arg(&delta)
                .arg(&rs_unit)
                .arg(&r_max)
                .arg(&r_capture)
                .arg(&omega)
                .arg(&disk_r_in)
                .arg(&disk_r_out);
            // SAFETY: the argument list matches the kernel signature in fantasy.cu
            // (four read/write ray buffers of 4n doubles, two read/write int buffers
            // of n, one read-only double buffer of n, one read/write int buffer and
            // two read/write double buffers of n, three ints, seven doubles), and the
            // kernel guards i < n.
            unsafe { launch.launch(cfg) }.context("kernel launch failed")?;
            let st = stream
                .memcpy_dtov(&status)
                .context("reading ray status back (the kernel may have faulted)")?;
            let ns = stream
                .memcpy_dtov(&n_steps)
                .context("reading step counts back")?;
            let now_finished = st
                .iter()
                .zip(&ns)
                .filter(|(s, k)| **s != 0 || **k as usize >= settings.steps)
                .count();
            for _ in finished..now_finished {
                on_ray_done();
            }
            finished = now_finished;
            if finished == n {
                break;
            }
        }

        let q_out = stream.memcpy_dtov(&q1).context("reading results back")?;
        let p_out = stream.memcpy_dtov(&p1).context("reading results back")?;
        let st = stream
            .memcpy_dtov(&status)
            .context("reading results back")?;
        let ns = stream
            .memcpy_dtov(&n_steps)
            .context("reading results back")?;
        let cross = stream
            .memcpy_dtov(&crossings)
            .context("reading results back")?;
        let hr = stream.memcpy_dtov(&hit_r).context("reading results back")?;
        let hp = stream
            .memcpy_dtov(&hit_phi)
            .context("reading results back")?;
        stream.synchronize().context("waiting for the device")?;
        for _ in finished..n {
            on_ray_done();
        }

        Ok((0..n)
            .map(|i| IntegrationResult {
                q: [
                    q_out[4 * i] * mass,
                    q_out[4 * i + 1] * mass,
                    q_out[4 * i + 2],
                    q_out[4 * i + 3],
                ],
                p: [
                    p_out[4 * i],
                    p_out[4 * i + 1],
                    p_out[4 * i + 2] * mass,
                    p_out[4 * i + 3] * mass,
                ],
                exit: match st[i] {
                    1 => ExitReason::Captured,
                    2 => ExitReason::Escaped,
                    3 => ExitReason::Diverged,
                    4 => ExitReason::Disk,
                    _ => ExitReason::StepLimit,
                },
                n_steps: ns[i].max(0) as usize,
                disk_hit: (st[i] == 4).then(|| DiskHit {
                    r: hr[i] * mass,
                    phi: hp[i],
                }),
                plane_crossings: cross[i].max(0) as u32,
            })
            .collect())
    }
}
