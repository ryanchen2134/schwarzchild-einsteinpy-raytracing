//! The curved-spacetime render: camera → initial conditions → integrate →
//! shade → diagnostic trajectories. No file I/O here; see [`crate::output`].

use crate::camera::Camera;
use crate::coords::{mat_vec, rot_x, spherical_to_cartesian, Vec3};
use crate::initial_conditions::{initial_conditions, RayInit};
use crate::integrator::{
    integrate_batch, integrate_trajectory_bounded, IntegrationResult, IntegratorSettings,
};
use crate::schwarzschild::BlackHole;
use crate::shading::{shade, ShadedRay};
use crate::sky_patch::SkyPatch;
use crate::texture::{Rgb, Texture};
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct RenderConfig {
    pub camera: Camera,
    pub black_hole: BlackHole,
    pub boundary_radius: f64,
    pub patch: SkyPatch,
    pub integrator: IntegratorSettings,
    /// Flat pixel indices whose trajectories are recorded.
    pub sample_pixels: Vec<usize>,
    /// Trajectories are thinned to at most this many points.
    pub max_trajectory_points: usize,
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

/// Render the scene. `on_ray_done` is called once per integrated ray (progress).
pub fn render_curved<F>(
    cfg: &RenderConfig,
    texture: Option<&Texture>,
    on_ray_done: F,
) -> RenderResult
where
    F: Fn() + Sync,
{
    let cam = &cfg.camera;
    let bh = cfg.black_hole;
    let rs = bh.rs();
    let n = cam.n_pixels();

    let inits: Vec<RayInit> = (0..n)
        .into_par_iter()
        .map(|flat| {
            let (i, j) = cam.pixel_of(flat);
            initial_conditions(cam.observer, cam.pixel_position(i, j), bh.mass)
        })
        .collect();

    let results = integrate_batch(
        inits.par_iter().map(|r| (r.q0, r.p0)),
        rs,
        &cfg.integrator,
        on_ray_done,
    );

    // Diagnostic trajectories first: they only need `inits`, which the photon
    // records consume below.
    let trajectories = cfg
        .sample_pixels
        .par_iter()
        .map(|&flat| {
            let init = inits[flat];
            let (traj, _) = integrate_trajectory_bounded(
                init.q0,
                init.p0,
                rs,
                &cfg.integrator,
                cfg.max_trajectory_points,
            );
            (
                flat,
                trajectory_to_lab(&traj, init.beta, cfg.max_trajectory_points),
            )
        })
        .collect();

    let capture_angle = bh.capture_angle(cam.observer_radius());
    let photons: Vec<PhotonRecord> = inits
        .into_par_iter()
        .zip(results.into_par_iter())
        .enumerate()
        .map(|(flat, (init, result))| {
            let (i, j) = cam.pixel_of(flat);
            let shaded = shade(
                &result,
                init.beta,
                rs,
                cfg.boundary_radius,
                &cfg.patch,
                texture,
            );
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

    RenderResult {
        width: cam.width,
        height: cam.height,
        image,
        photons,
        trajectories,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            sample_pixels,
            max_trajectory_points: 50,
        }
    }

    #[test]
    fn small_render_classifies_centre_and_edge() {
        let cfg = config(9, vec![0, 40]);
        let tex = Texture::solid(4, 2, [10, 20, 30]);
        let res = render_curved(&cfg, Some(&tex), || {});
        let centre = &res.photons[40];
        assert_eq!(
            centre.shaded.collision,
            crate::shading::Collision::BlackHole
        );
        assert!(centre.analytic_capture);
        let corner = &res.photons[0];
        assert_eq!(
            corner.shaded.collision,
            crate::shading::Collision::EscapedBackground
        );
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
        let res = render_curved(&cfg, Some(&tex), || {});
        let mut checked = 0;
        for p in &res.photons {
            if p.shaded.collision == crate::shading::Collision::EscapedBackground {
                let (u, v) = cfg
                    .patch
                    .texel(p.shaded.final_theta, p.shaded.final_phi, 32, 64);
                assert_eq!(p.shaded.colour, [u as u8, v as u8, 1]);
                checked += 1;
            }
        }
        assert!(checked >= 20, "{checked}");
    }
}
