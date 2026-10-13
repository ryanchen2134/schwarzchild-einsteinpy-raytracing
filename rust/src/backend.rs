//! Where the rays get integrated: the scalar reference, the SIMD fast path, or a
//! CUDA device.

use crate::integrator::{integrate_batch, IntegrationResult, IntegratorSettings};
use crate::integrator_fast::integrate_batch_fast;
use anyhow::Result;
use rayon::prelude::*;

/// Source of the FANTASY kernel as CUDA C, available in every build so that it can
/// be inspected and syntax-checked without the `cuda` feature.
pub const CUDA_KERNEL_SOURCE: &str = include_str!("fantasy.cu");

pub trait BatchIntegrator: Sync {
    /// Human-readable name for logs.
    fn name(&self) -> String;

    /// Integrate every ray. `on_ray_done` is called once per ray for progress.
    fn integrate_batch(
        &self,
        rays: &[([f64; 4], [f64; 4])],
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
        rays: &[([f64; 4], [f64; 4])],
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
        rays: &[([f64; 4], [f64; 4])],
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
        rays: &[([f64; 4], [f64; 4])],
        rs: f64,
        settings: &IntegratorSettings,
        on_ray_done: &(dyn Fn() + Sync),
    ) -> Result<Vec<IntegrationResult>> {
        let results = crate::cuda::CudaIntegrator::integrate_batch(self, rays, rs, settings)?;
        for _ in 0..rays.len() {
            on_ray_done();
        }
        Ok(results)
    }
}

/// Was the binary built with the CUDA backend?
pub const fn cuda_compiled_in() -> bool {
    cfg!(feature = "cuda")
}

/// Open a CUDA backend if the build has one and a device is present.
#[cfg(feature = "cuda")]
pub fn try_cuda(ordinal: usize) -> Result<Option<Box<dyn BatchIntegrator>>> {
    if crate::cuda::device_count() == 0 {
        return Ok(None);
    }
    let dev = crate::cuda::CudaIntegrator::new(ordinal)?;
    Ok(Some(Box::new(dev)))
}

/// Open a CUDA backend if the build has one and a device is present.
#[cfg(not(feature = "cuda"))]
pub fn try_cuda(ordinal: usize) -> Result<Option<Box<dyn BatchIntegrator>>> {
    let _ = ordinal;
    Ok(None)
}
