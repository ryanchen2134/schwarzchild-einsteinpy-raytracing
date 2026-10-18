//! Where the rays get integrated: the scalar reference, the SIMD fast path, or a
//! CUDA device.

use crate::integrator::{integrate_batch, IntegrationResult, IntegratorSettings, Ray};
use crate::integrator_fast::integrate_batch_fast;
use anyhow::Result;
use rayon::prelude::*;

/// Source of the FANTASY kernel as CUDA C, available in every build so that it can
/// be inspected and syntax-checked without the `cuda` feature.
pub const CUDA_KERNEL_SOURCE: &str = include_str!("fantasy.cu");

pub trait BatchIntegrator: Sync {
    /// Human-readable name for logs.
    fn name(&self) -> String;

    /// Integrate every ray. `on_ray_done` is called once per ray for progress; a
    /// device backend may report in batches as launches complete.
    fn integrate_batch(
        &self,
        rays: &[Ray],
        rs: f64,
        settings: &IntegratorSettings,
        on_ray_done: &(dyn Fn() + Sync),
    ) -> Result<Vec<IntegrationResult>>;
}

/// The scalar integrator: bit-faithful to the Python kernel, one ray per core at a time.
pub struct CpuExact;

impl BatchIntegrator for CpuExact {
    fn name(&self) -> String {
        format!("cpu-exact ({} threads)", rayon::current_num_threads())
    }

    fn integrate_batch(
        &self,
        rays: &[Ray],
        rs: f64,
        settings: &IntegratorSettings,
        on_ray_done: &(dyn Fn() + Sync),
    ) -> Result<Vec<IntegrationResult>> {
        Ok(integrate_batch(
            rays.par_iter().copied(),
            rs,
            settings,
            on_ray_done,
        ))
    }
}

/// The equatorial four-lane SIMD integrator.
pub struct CpuFast;

impl BatchIntegrator for CpuFast {
    fn name(&self) -> String {
        format!(
            "cpu-simd ({} threads × 4 lanes)",
            rayon::current_num_threads()
        )
    }

    fn integrate_batch(
        &self,
        rays: &[Ray],
        rs: f64,
        settings: &IntegratorSettings,
        on_ray_done: &(dyn Fn() + Sync),
    ) -> Result<Vec<IntegrationResult>> {
        Ok(integrate_batch_fast(rays, rs, settings, on_ray_done))
    }
}

#[cfg(feature = "cuda")]
impl BatchIntegrator for crate::cuda::CudaIntegrator {
    fn name(&self) -> String {
        format!("cuda ({} on device {})", self.device_name(), self.ordinal())
    }

    fn integrate_batch(
        &self,
        rays: &[Ray],
        rs: f64,
        settings: &IntegratorSettings,
        on_ray_done: &(dyn Fn() + Sync),
    ) -> Result<Vec<IntegrationResult>> {
        crate::cuda::CudaIntegrator::integrate_batch(self, rays, rs, settings, on_ray_done)
    }
}

/// Was the binary built with the CUDA backend?
pub const fn cuda_compiled_in() -> bool {
    cfg!(feature = "cuda")
}

/// Open a CUDA backend if the build has one and a device is present.
///
/// `Ok(None)` means there is nothing to use (no driver library, or no device):
/// the caller falls back to the CPU. `Err` means CUDA is there but unusable
/// (missing NVRTC, bad device ordinal, compilation or driver error).
#[cfg(feature = "cuda")]
pub fn try_cuda(ordinal: usize) -> Result<Option<Box<dyn BatchIntegrator>>> {
    use crate::cuda::{
        catch_cuda_panic, device_count, probe_libraries, CudaIntegrator, MissingLibrary,
    };
    match probe_libraries() {
        Err(MissingLibrary::Driver) => return Ok(None),
        Err(MissingLibrary::Nvrtc) => anyhow::bail!(
            "libnvrtc not found: the CUDA driver is present but the runtime compiler is not; \
             install the CUDA toolkit's NVRTC package (it is not part of the display driver)"
        ),
        Ok(()) => {}
    }
    // cudarc panics instead of erroring when a symbol is missing; keep that an error.
    catch_cuda_panic(|| {
        if device_count() == 0 {
            return Ok(None);
        }
        let dev = CudaIntegrator::new(ordinal)?;
        Ok(Some(Box::new(dev) as Box<dyn BatchIntegrator>))
    })
}

/// Open a CUDA backend if the build has one and a device is present.
#[cfg(not(feature = "cuda"))]
pub fn try_cuda(ordinal: usize) -> Result<Option<Box<dyn BatchIntegrator>>> {
    let _ = ordinal;
    Ok(None)
}
