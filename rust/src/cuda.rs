//! The CUDA backend: the FANTASY kernel in `fantasy.cu`, compiled with NVRTC at
//! start-up and launched through [`cudarc`]. Built only with `--features cuda`;
//! needs a CUDA driver (and `libnvrtc`) at run time, nothing at build time.
//!
//! The kernel is compiled with `--fmad=false`, so its arithmetic rounds like the
//! CPU integrators and the results agree with them to rounding.

use crate::integrator::{ExitReason, IntegrationResult, IntegratorSettings};
use anyhow::{anyhow, Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::sync::Arc;

use crate::backend::CUDA_KERNEL_SOURCE as KERNEL_SOURCE;
const KERNEL_NAME: &str = "fantasy_integrate";
/// Schwarzschild radius in the kernel's units.
const RS_UNIT: f64 = 2.0;
/// Rays per launch; bounds device memory for very large images.
const CHUNK: usize = 1 << 20;
/// Threads per block: the kernel keeps four 4-vectors per copy in registers, so a
/// modest block keeps occupancy up without hitting the register limit.
const THREADS_PER_BLOCK: u32 = 128;

pub struct CudaIntegrator {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    func: CudaFunction,
    name: String,
}

/// Is at least one CUDA device visible?
pub fn device_count() -> usize {
    CudaContext::device_count()
        .map(|c| c.max(0) as usize)
        .unwrap_or(0)
}

impl CudaIntegrator {
    /// Open device `ordinal` and compile the kernel.
    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal).with_context(|| {
            format!("cannot open CUDA device {ordinal} (is a CUDA driver installed?)")
        })?;
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
        let module = ctx
            .load_module(ptx)
            .context("cannot load the FANTASY kernel module")?;
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
    /// unit; the rescaling to units of `M` happens here.
    pub fn integrate_batch(
        &self,
        rays: &[([f64; 4], [f64; 4])],
        rs: f64,
        settings: &IntegratorSettings,
    ) -> Result<Vec<IntegrationResult>> {
        let mass = rs / RS_UNIT;
        let mut results = Vec::with_capacity(rays.len());
        for chunk in rays.chunks(CHUNK) {
            results.extend(self.integrate_chunk(chunk, mass, settings)?);
        }
        Ok(results)
    }

    fn integrate_chunk(
        &self,
        rays: &[([f64; 4], [f64; 4])],
        mass: f64,
        settings: &IntegratorSettings,
    ) -> Result<Vec<IntegrationResult>> {
        let n = rays.len();
        let mut q_host = Vec::with_capacity(4 * n);
        let mut p_host = Vec::with_capacity(4 * n);
        for (q, p) in rays {
            q_host.extend_from_slice(&[q[0] / mass, q[1] / mass, q[2], q[3]]);
            p_host.extend_from_slice(&[p[0], p[1], p[2] / mass, p[3] / mass]);
        }
        let q_dev = self.stream.memcpy_stod(&q_host)?;
        let p_dev = self.stream.memcpy_stod(&p_host)?;
        let mut out_q = self.stream.alloc_zeros::<f64>(4 * n)?;
        let mut out_p = self.stream.alloc_zeros::<f64>(4 * n)?;
        let mut out_status = self.stream.alloc_zeros::<i32>(n)?;
        let mut out_steps = self.stream.alloc_zeros::<i32>(n)?;

        let n_i32 = i32::try_from(n).context("too many rays for one launch")?;
        let steps_i32 = i32::try_from(settings.steps).context("--steps does not fit in an int")?;
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
        let mut launch = self.stream.launch_builder(&self.func);
        launch
            .arg(&q_dev)
            .arg(&p_dev)
            .arg(&n_i32)
            .arg(&steps_i32)
            .arg(&delta)
            .arg(&rs_unit)
            .arg(&r_max)
            .arg(&r_capture)
            .arg(&omega)
            .arg(&mut out_q)
            .arg(&mut out_p)
            .arg(&mut out_status)
            .arg(&mut out_steps);
        // SAFETY: the argument list matches the kernel signature in fantasy.cu
        // (two const double* inputs, nine scalars, four output buffers), every buffer
        // holds n or 4n elements, and the kernel guards i < n.
        unsafe { launch.launch(cfg) }.context("kernel launch failed")?;

        let q_out = self.stream.memcpy_dtov(&out_q)?;
        let p_out = self.stream.memcpy_dtov(&out_p)?;
        let status = self.stream.memcpy_dtov(&out_status)?;
        let steps = self.stream.memcpy_dtov(&out_steps)?;
        self.stream.synchronize()?;

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
                exit: match status[i] {
                    1 => ExitReason::Captured,
                    2 => ExitReason::Escaped,
                    3 => ExitReason::Diverged,
                    _ => ExitReason::StepLimit,
                },
                n_steps: steps[i].max(0) as usize,
            })
            .collect())
    }
}
