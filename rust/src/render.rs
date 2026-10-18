//! The curved-spacetime render: camera → initial conditions → integrate →
//! shade → diagnostic trajectories. No file I/O here; see [`crate::output`].

use crate::backend::BatchIntegrator;
use crate::camera::Camera;
use crate::coords::{mat_vec, rot_x, spherical_to_cartesian, Vec3};
use crate::disk::Disk;
use crate::initial_conditions::{initial_conditions, RayInit};
use crate::integrator::{
    integrate_trajectory_bounded, DiskBand, IntegrationResult, IntegratorSettings, Ray,
};
use crate::schwarzschild::BlackHole;
use crate::shading::{shade, Scene, ShadedRay};
use crate::sky_patch::SkyPatch;
use crate::texture::{Rgb, Texture};
use anyhow::{ensure, Result};
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct RenderConfig {
    pub camera: Camera,
    pub black_hole: BlackHole,
    pub boundary_radius: f64,
    pub patch: SkyPatch,
    /// Integrator settings; the disk band is filled in from `disk` at render time.
    pub integrator: IntegratorSettings,
    pub disk: Option<Disk>,
    /// Flat pixel indices whose trajectories are recorded.
    pub sample_pixels: Vec<usize>,
    /// Trajectories are thinned to at most this many points.
    pub max_trajectory_points: usize,
}

impl RenderConfig {
    /// The integrator settings with the disk band set from `disk`.
    pub fn integrator_settings(&self) -> IntegratorSettings {
        match &self.disk {
            Some(d) => self.integrator.with_disk(DiskBand {
                r_in: d.r_in,
                r_out: d.r_out,
            }),
            None => IntegratorSettings {
                disk: None,
                ..self.integrator
            },
        }
    }

    /// The ray for pixel `(i, j)`: initial conditions plus the disk's crossing azimuth
    /// in the ray's frame.
    pub fn ray(&self, init: &RayInit) -> Ray {
        match &self.disk {
            Some(d) => Ray::with_disk(init.q0, init.p0, d.ray_frame(init.beta).phi_d),
            None => Ray::new(init.q0, init.p0),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PhotonRecord {
    pub i: usize,
    pub j: usize,
    pub init: RayInit,
    pub result: IntegrationResult,
    pub shaded: ShadedRay,
    /// `alpha0 ≤` the analytic shadow half-angle.
    pub analytic_capture: bool,
}

pub struct RenderResult {
    pub width: usize,
    pub height: usize,
    /// Row-major, row 0 at the bottom (see [`Camera`]).
    pub image: Vec<Rgb>,
    pub photons: Vec<PhotonRecord>,
    /// `(flat pixel index, lab-frame points)` for each sampled ray.
    pub trajectories: Vec<(usize, Vec<Vec3>)>,
}

/// Evenly spaced indices into `0..n` (nearest-rounded), at most `max_points` of them,
/// always including the first and last.
pub fn downsample_indices(n: usize, max_points: usize) -> Vec<usize> {
    if n == 0 || max_points == 0 {
        return Vec::new();
    }
    let k = n.min(max_points);
    if k == 1 {
        return vec![0];
    }
    (0..k)
        .map(|m| ((n - 1) as f64 * m as f64 / (k - 1) as f64).round() as usize)
        .collect()
}

/// Rotate an equatorial-plane trajectory back into the lab frame and thin it.
pub fn trajectory_to_lab(traj: &[[f64; 4]], beta: f64, max_points: usize) -> Vec<Vec3> {
    let rot = rot_x(beta);
    downsample_indices(traj.len(), max_points)
        .into_iter()
        .map(|k| {
            let q = traj[k];
            mat_vec(&rot, spherical_to_cartesian(q[1], q[2], q[3]))
        })
        .collect()
}

/// Render the scene with `backend`. `on_ray_done` is called once per integrated ray
/// (progress). The sampled trajectories always use the scalar integrator, which is
/// the only one that records every step.
pub fn render_curved<F>(
    cfg: &RenderConfig,
    texture: Option<&Texture>,
    backend: &dyn BatchIntegrator,
    on_ray_done: F,
) -> Result<RenderResult>
where
    F: Fn() + Sync,
{
    let cam = &cfg.camera;
    let bh = cfg.black_hole;
    let rs = bh.rs();
    let n = cam.n_pixels();
    if let Some(d) = &cfg.disk {
        // The integrators do not check the first step for a crossing (integrator.rs,
        // "Disk crossings"); that is only safe when the disk ends a step inside the
        // observer.
        ensure!(
            d.r_out < cam.observer_radius() - cfg.integrator.delta,
            "the disk's outer radius ({}) must lie more than one step (delta = {}) inside the \
             observer distance ({}): crossings during the first step are not checked",
            d.r_out,
            cfg.integrator.delta,
            cam.observer_radius()
        );
    }

    let inits: Vec<RayInit> = (0..n)
        .into_par_iter()
        .map(|flat| {
            let (i, j) = cam.pixel_of(flat);
            initial_conditions(cam.observer, cam.pixel_position(i, j), bh.mass)
        })
        .collect();

    let settings = cfg.integrator_settings();
    let rays: Vec<Ray> = inits.iter().map(|init| cfg.ray(init)).collect();
    let results = backend.integrate_batch(&rays, rs, &settings, &on_ray_done)?;
    drop(rays);

    // Diagnostic trajectories first: they only need `inits`, which the photon
    // records consume below.
    let trajectories = cfg
        .sample_pixels
        .par_iter()
        .map(|&flat| {
            let init = inits[flat];
            let (traj, _) = integrate_trajectory_bounded(
                &cfg.ray(&init),
                rs,
                &settings,
                cfg.max_trajectory_points,
            );
            (
                flat,
                trajectory_to_lab(&traj, init.beta, cfg.max_trajectory_points),
            )
        })
        .collect();

    let capture_angle = bh.capture_angle(cam.observer_radius());
    let scene = Scene {
        black_hole: bh,
        r_obs: cam.observer_radius(),
        boundary_radius: cfg.boundary_radius,
        patch: &cfg.patch,
        texture,
        disk: cfg.disk.as_ref(),
    };
    let photons: Vec<PhotonRecord> = inits
        .into_par_iter()
        .zip(results.into_par_iter())
        .enumerate()
        .map(|(flat, (init, result))| {
            let (i, j) = cam.pixel_of(flat);
            let shaded = shade(&result, &init, &scene);
            PhotonRecord {
                i,
                j,
                init,
                result,
                shaded,
                analytic_capture: init.alpha0 <= capture_angle,
            }
        })
        .collect();
    let image = photons.iter().map(|p| p.shaded.colour).collect();

    Ok(RenderResult {
        width: cam.width,
        height: cam.height,
        image,
        photons,
        trajectories,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{CpuExact, CpuFast};
    use crate::integrator::ExitReason;
    use crate::shading::Collision;

    #[test]
    fn downsample_keeps_endpoints() {
        assert_eq!(downsample_indices(10, 4), vec![0, 3, 6, 9]);
        assert_eq!(downsample_indices(3, 10), vec![0, 1, 2]);
        assert_eq!(downsample_indices(0, 10), Vec::<usize>::new());
        assert_eq!(downsample_indices(5, 1), vec![0]);
    }

    fn config(size: usize, sample_pixels: Vec<usize>) -> RenderConfig {
        let camera = Camera::new([30.0, 0.0, 0.0], 80f64.to_radians(), size, size).unwrap();
        let bh = BlackHole::new(1.0);
        RenderConfig {
            camera,
            black_hole: bh,
            boundary_radius: 31.0,
            patch: SkyPatch::full_sky(),
            integrator: IntegratorSettings::new(20_000, 0.01, 0.01, 31.0, bh.rs()),
            disk: None,
            sample_pixels,
            max_trajectory_points: 50,
        }
    }

    #[test]
    fn a_face_on_disk_is_seen_without_doppler_shift_at_any_pointing() {
        // n = +x̂: b_z ≡ 0, so every disk pixel has g = √(1−3M/r)/√(1−2M/r_obs), even
        // with the camera turned.
        let mut cfg = config(15, vec![]);
        cfg.camera = Camera::with_rotation(
            [30.0, 0.0, 0.0],
            80f64.to_radians(),
            15,
            15,
            0.2,
            -0.15,
            0.4,
        )
        .unwrap();
        cfg.disk = Some(Disk::new([1.0, 0.0, 0.0], 6.0, 20.0, 6000.0, 1.0, 30.0).unwrap());
        let res = render_curved(&cfg, None, &CpuExact, || {}).unwrap();
        let disk_pixels: Vec<_> = res
            .photons
            .iter()
            .filter(|p| p.shaded.collision == Collision::Disk)
            .collect();
        assert!(disk_pixels.len() > 30, "{}", disk_pixels.len());
        for p in &disk_pixels {
            let d = p.shaded.disk.unwrap();
            let expect = (1.0 - 3.0 / d.r).sqrt() / (1.0 - 2.0 / 30.0f64).sqrt();
            assert!((d.g - expect).abs() < 1e-14, "g {} vs {expect}", d.g);
            assert!(d.r >= 6.0 && d.r <= 20.0);
            assert!(
                d.position[0].abs() < 1e-9,
                "the hit lies in the plane x = 0: {:?}",
                d.position
            );
            assert_eq!(p.result.exit, ExitReason::Disk);
            assert_ne!(p.shaded.colour, [0, 0, 0]);
        }
        assert!(res
            .photons
            .iter()
            .all(|p| p.shaded.collision != Collision::NumericalError));
        // The same scene through the SIMD backend is identical.
        let fast = render_curved(&cfg, None, &CpuFast, || {}).unwrap();
        assert_eq!(res.image, fast.image);
        for (a, b) in res.photons.iter().zip(&fast.photons) {
            assert_eq!(a.shaded.collision, b.shaded.collision);
            assert_eq!(
                a.shaded.disk.map(|d| (d.r, d.phi, d.g)),
                b.shaded.disk.map(|d| (d.r, d.phi, d.g))
            );
            assert_eq!(a.result.plane_crossings, b.result.plane_crossings);
        }
    }

    #[test]
    fn a_disk_reaching_within_one_step_of_the_observer_is_refused() {
        // The first step is not checked for a crossing (integrator.rs, "Disk
        // crossings"), so the disk must end more than one step inside the observer.
        let mut cfg = config(3, vec![]);
        cfg.disk = Some(Disk::new([1.0, 0.0, 0.0], 6.0, 29.995, 6000.0, 1.0, 30.0).unwrap());
        let err = render_curved(&cfg, None, &CpuExact, || {})
            .err()
            .expect("refused");
        assert!(err.to_string().contains("more than one step"), "{err}");
        cfg.disk = Some(Disk::new([1.0, 0.0, 0.0], 6.0, 29.98, 6000.0, 1.0, 30.0).unwrap());
        assert!(render_curved(&cfg, None, &CpuExact, || {}).is_ok());
    }

    #[test]
    fn sampled_trajectories_end_on_the_disk() {
        let mut cfg = config(9, vec![]);
        cfg.disk = Some(Disk::new([1.0, 0.0, 0.0], 6.0, 20.0, 6000.0, 1.0, 30.0).unwrap());
        let probe = render_curved(&cfg, None, &CpuExact, || {}).unwrap();
        let flat = probe
            .photons
            .iter()
            .position(|p| p.shaded.collision == Collision::Disk)
            .expect("a disk pixel");
        cfg.sample_pixels = vec![flat];
        let res = render_curved(&cfg, None, &CpuExact, || {}).unwrap();
        let (_, pts) = &res.trajectories[0];
        let last = pts.last().unwrap();
        let r_last = crate::coords::norm(*last);
        let hit = res.photons[flat].shaded.disk.unwrap();
        assert!(
            (r_last - hit.r).abs() < 0.02,
            "trajectory ends at r = {r_last}, hit at {}",
            hit.r
        );
        assert!(last[0].abs() < 0.02, "and on the plane x = 0: {last:?}");
    }

    #[test]
    fn small_render_classifies_centre_and_edge() {
        let cfg = config(9, vec![0, 40]);
        let tex = Texture::solid(4, 2, [10, 20, 30]);
        let res = render_curved(&cfg, Some(&tex), &CpuExact, || {}).unwrap();
        let centre = &res.photons[40];
        assert_eq!(centre.shaded.collision, Collision::BlackHole);
        assert!(centre.analytic_capture);
        let corner = &res.photons[0];
        assert_eq!(corner.shaded.collision, Collision::EscapedBackground);
        assert_eq!(corner.shaded.colour, [10, 20, 30]);
        assert!(!corner.analytic_capture);
        assert_eq!(res.trajectories.len(), 2);
        assert!(res.trajectories[1].1.len() <= 50);
        assert_eq!(res.image.len(), 81);
        assert_eq!(res.image[0], [10, 20, 30]);
        assert_eq!(res.image[40], [0, 0, 0]);
    }

    #[test]
    fn escaped_pixels_sample_the_texture_at_their_exit_direction() {
        let cfg = config(5, vec![]);
        let mut tex = Texture::solid(64, 32, [0, 0, 0]);
        for r in 0..32 {
            for c in 0..64 {
                tex.data[r * 64 + c] = [r as u8, c as u8, 1];
            }
        }
        let res = render_curved(&cfg, Some(&tex), &CpuExact, || {}).unwrap();
        let mut checked = 0;
        for p in &res.photons {
            if p.shaded.collision == Collision::EscapedBackground {
                let (u, v) = cfg
                    .patch
                    .texel(p.shaded.final_theta, p.shaded.final_phi, 32, 64);
                assert_eq!(p.shaded.colour, [u as u8, v as u8, 1]);
                checked += 1;
            }
        }
        assert!(checked >= 20, "{checked}");
    }

    #[test]
    fn fast_backend_renders_the_same_image_as_the_exact_one() {
        // 25×25: an odd size, so a lane permutation would pair different rays.
        let cfg = config(25, vec![3, 300]);
        let mut tex = Texture::solid(64, 32, [0, 0, 0]);
        for r in 0..32 {
            for c in 0..64 {
                tex.data[r * 64 + c] = [r as u8, c as u8, 1];
            }
        }
        let exact = render_curved(&cfg, Some(&tex), &CpuExact, || {}).unwrap();
        let fast = render_curved(&cfg, Some(&tex), &CpuFast, || {}).unwrap();
        assert_eq!(exact.image, fast.image);
        assert_eq!(exact.photons.len(), fast.photons.len());
        for (a, b) in exact.photons.iter().zip(&fast.photons) {
            assert_eq!(a.shaded.collision, b.shaded.collision);
            assert_eq!(a.result.exit, b.result.exit);
            assert_eq!(a.result.n_steps, b.result.n_steps);
            assert_eq!(a.result.q, b.result.q, "pixel ({}, {})", a.i, a.j);
            assert_eq!(a.shaded.final_theta, b.shaded.final_theta);
            assert_eq!(a.shaded.final_phi, b.shaded.final_phi);
        }
        assert_eq!(exact.trajectories.len(), fast.trajectories.len());
        for (a, b) in exact.trajectories.iter().zip(&fast.trajectories) {
            assert_eq!(a, b, "trajectories always come from the scalar integrator");
        }
    }
}
