//! The no-gravity reference render: straight rays to the boundary sphere (or to
//! the disk, drawn without redshift), shaded through the same camera and sky patch
//! as the curved render.

use crate::camera::Camera;
use crate::coords::{add, dot, linspace, norm, scale, Vec3};
use crate::disk::{disk_pixel, Disk};
use crate::shading::BLACK;
use crate::sky_patch::SkyPatch;
use crate::texture::{Rgb, Texture};
use rayon::prelude::*;

/// Points per straight-line diagnostic trajectory.
pub const FLAT_TRAJECTORY_POINTS: usize = 100;

pub struct FlatResult {
    /// Row-major, row 0 at the bottom (see [`Camera`]).
    pub image: Vec<Rgb>,
    /// Where each ray meets the boundary sphere, row-major; `None` if it misses.
    pub hits: Vec<Option<Vec3>>,
    /// Where each ray meets the disk, if a disk is drawn and the ray reaches it.
    pub disk_hits: Vec<Option<Vec3>>,
    /// `(flat pixel index, points)` for each sampled ray; a ray that hits the disk
    /// ends there.
    pub trajectories: Vec<(usize, Vec<Vec3>)>,
}

/// Intersection of the ray `obs + t·dir` with the sphere of radius `boundary_radius`,
/// taking the far root as the Python code did.
#[inline]
pub fn sphere_hit(obs: Vec3, dir: Vec3, boundary_radius: f64) -> Option<Vec3> {
    let a = dot(dir, dir);
    let b = 2.0 * dot(obs, dir);
    let c = dot(obs, obs) - boundary_radius * boundary_radius;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let t = (-b + disc.sqrt()) / (2.0 * a);
    Some(add(obs, scale(dir, t)))
}

pub fn render_flat(
    camera: &Camera,
    boundary_radius: f64,
    patch: &SkyPatch,
    texture: Option<&Texture>,
    disk: Option<&Disk>,
    sample_pixels: &[usize],
) -> FlatResult {
    let obs = camera.observer;
    let shaded: Vec<(Rgb, Option<Vec3>, Option<Vec3>)> = (0..camera.n_pixels())
        .into_par_iter()
        .map(|flat| {
            let (i, j) = camera.pixel_of(flat);
            let dir = camera.ray_direction(i, j);
            let sphere = sphere_hit(obs, dir, boundary_radius);
            if let Some(d) = disk {
                if let Some(hit) = d.straight_ray_hit(obs, dir) {
                    let colour = disk_pixel(1.0, d.temperature(norm(hit)), d.peak_temperature);
                    return (colour, sphere, Some(hit));
                }
            }
            let Some(hit) = sphere else {
                return (BLACK, None, None);
            };
            let r = norm(hit);
            let theta = (hit[2] / r).acos();
            let phi = hit[1].atan2(hit[0]);
            let colour = match texture {
                Some(tex) if patch.contains(theta, phi) => {
                    let (u, v) = patch.texel(theta, phi, tex.height, tex.width);
                    tex.texel(u, v)
                }
                _ => BLACK,
            };
            (colour, Some(hit), None)
        })
        .collect();
    let image = shaded.iter().map(|s| s.0).collect();
    let hits: Vec<Option<Vec3>> = shaded.iter().map(|s| s.1).collect();
    let disk_hits: Vec<Option<Vec3>> = shaded.iter().map(|s| s.2).collect();
    let trajectories = sample_pixels
        .iter()
        .filter_map(|&flat| {
            disk_hits[flat]
                .or(hits[flat])
                .map(|h| (flat, linspace(obs, h, FLAT_TRAJECTORY_POINTS)))
        })
        .collect();
    FlatResult {
        image,
        hits,
        disk_hits,
        trajectories,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_ray_from_inside_hits_the_sphere() {
        let cam = Camera::new([30.0, 0.0, 0.0], 80f64.to_radians(), 8, 6).unwrap();
        let res = render_flat(&cam, 31.0, &SkyPatch::full_sky(), None, None, &[0, 47]);
        assert!(res.hits.iter().all(|h| h.is_some()));
        for h in res.hits.iter().flatten() {
            assert!((norm(*h) - 31.0).abs() < 1e-9);
        }
        assert_eq!(res.trajectories.len(), 2);
        assert_eq!(res.trajectories[0].1.len(), FLAT_TRAJECTORY_POINTS);
        assert!(res.disk_hits.iter().all(|h| h.is_none()));
    }

    #[test]
    fn straight_rays_draw_an_inclined_disk_as_an_ellipse() {
        let cam = Camera::new([30.0, 0.0, 0.0], 80f64.to_radians(), 41, 41).unwrap();
        let disk = Disk::new(
            Disk::normal_from_angles(1.2, 0.0),
            6.0,
            20.0,
            6000.0,
            1.0,
            30.0,
        )
        .unwrap();
        let centre = cam.flat_index(20, 20);
        let res = render_flat(
            &cam,
            31.0,
            &SkyPatch::full_sky(),
            None,
            Some(&disk),
            &[centre, 0],
        );
        let n_disk = res.disk_hits.iter().flatten().count();
        assert!(n_disk > 100, "{n_disk}");
        for (flat, h) in res.disk_hits.iter().enumerate() {
            if let Some(h) = h {
                let r = norm(*h);
                assert!((6.0..=20.0).contains(&r));
                assert!(crate::coords::dot(disk.normal, *h).abs() < 1e-9);
                assert_ne!(res.image[flat], BLACK);
            }
        }
        // The centre ray passes through the hole in the disk and reaches the sphere.
        assert!(res.disk_hits[centre].is_none());
        assert_eq!(res.image[centre], BLACK);
        // A sampled ray that hits the disk ends there.
        let hit_pixel = res.disk_hits.iter().position(|h| h.is_some()).unwrap();
        let res2 = render_flat(
            &cam,
            31.0,
            &SkyPatch::full_sky(),
            None,
            Some(&disk),
            &[hit_pixel],
        );
        let last = *res2.trajectories[0].1.last().unwrap();
        assert!((norm(last) - norm(res2.disk_hits[hit_pixel].unwrap())).abs() < 1e-9);
    }
}
