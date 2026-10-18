//! From an integrated ray to a pixel: undo the equatorial rotation, classify
//! the exit, look the escape direction up in the sky patch, or shade a disk hit.

use crate::coords::{cartesian_to_spherical, mat_vec, rot_x, spherical_to_cartesian, Vec3};
use crate::disk::{disk_pixel, hit_position_lab, redshift, Disk};
use crate::initial_conditions::RayInit;
use crate::integrator::{ExitReason, IntegrationResult};
use crate::schwarzschild::BlackHole;
use crate::sky_patch::SkyPatch;
use crate::texture::{Rgb, Texture};

/// Rays that reach this radius have left the physical domain by a wide margin.
/// The boundary radius must stay below it (checked by the CLI).
pub const NUMERICAL_ERROR_RADIUS: f64 = 100.0;
/// Classification threshold for capture, slightly outside the integrator's exit radius.
pub const CAPTURE_RADIUS_FACTOR: f64 = 1.2;

pub const BLACK: Rgb = [0, 0, 0];
pub const NUMERICAL_ERROR_COLOUR: Rgb = [255, 0, 0];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Collision {
    BlackHole,
    EscapedBackground,
    EscapedNoPatch,
    InDomain,
    NumericalError,
    Disk,
}

impl Collision {
    /// Labels match the Python `photon_data.csv`.
    pub fn label(self) -> &'static str {
        match self {
            Collision::BlackHole => "bh",
            Collision::EscapedBackground => "escape_bg",
            Collision::EscapedNoPatch => "escape_no_patch",
            Collision::InDomain => "in_domain",
            Collision::NumericalError => "numerical error",
            Collision::Disk => "disk",
        }
    }
}

/// What a ray that hit the disk saw there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiskPixel {
    /// Hit radius.
    pub r: f64,
    /// Hit azimuth in the ray's rotated frame.
    pub phi: f64,
    /// Hit position in the lab frame.
    pub position: Vec3,
    /// Observed/emitted frequency ratio; not finite or not positive on a hit the
    /// physics rules out (reported as a numerical error).
    pub g: f64,
    /// Effective temperature of the material at the hit, in kelvin.
    pub temperature: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct ShadedRay {
    pub colour: Rgb,
    pub collision: Collision,
    /// Exit direction in the lab frame: `θ ∈ [0, π]`, `φ ∈ (−π, π]`.
    pub final_theta: f64,
    pub final_phi: f64,
    /// Set when the ray ended on the disk.
    pub disk: Option<DiskPixel>,
}

/// Everything shading needs besides the ray itself.
#[derive(Clone, Copy)]
pub struct Scene<'a> {
    pub black_hole: BlackHole,
    pub r_obs: f64,
    pub boundary_radius: f64,
    pub patch: &'a SkyPatch,
    pub texture: Option<&'a Texture>,
    pub disk: Option<&'a Disk>,
}

/// Rotate a point `(r, θ, φ)` of the equatorial-plane orbit back into the lab frame
/// (rotation by `+β` about `+x`) and return its `(θ, φ)`.
#[inline]
pub fn unrotate_hit(q: &[f64; 4], beta: f64) -> (f64, f64) {
    let xyz = spherical_to_cartesian(q[1], q[2], q[3]);
    let lab = mat_vec(&rot_x(beta), xyz);
    let (_, th, ph) = cartesian_to_spherical(lab);
    (th, ph)
}

/// Classify by final radius, as the Python renderer did, after rejecting radii that
/// are not physical; the integrator's exit reason is recorded separately.
#[inline]
pub fn classify(r_final: f64, rs: f64, boundary_radius: f64, in_patch: bool) -> Collision {
    if !r_final.is_finite() || r_final <= 0.0 {
        Collision::NumericalError
    } else if r_final <= CAPTURE_RADIUS_FACTOR * rs {
        Collision::BlackHole
    } else if r_final >= NUMERICAL_ERROR_RADIUS {
        Collision::NumericalError
    } else if r_final >= boundary_radius {
        if in_patch {
            Collision::EscapedBackground
        } else {
            Collision::EscapedNoPatch
        }
    } else {
        Collision::InDomain
    }
}

/// Shade one ray. A diverged integration is a numerical error whatever its radius;
/// a disk hit is coloured from its redshift and the local temperature.
pub fn shade(result: &IntegrationResult, init: &RayInit, scene: &Scene) -> ShadedRay {
    let (final_theta, final_phi) = unrotate_hit(&result.q, init.beta);
    let rs = scene.black_hole.rs();
    let in_patch = scene.texture.is_some() && scene.patch.contains(final_theta, final_phi);
    if result.exit == ExitReason::Disk {
        if let (Some(disk), Some(hit)) = (scene.disk, result.disk_hit) {
            let frame = disk.ray_frame(init.beta);
            let g = redshift(
                scene.black_hole.mass,
                hit.r,
                init.p0[0],
                init.p0[3],
                frame.n_z,
                scene.r_obs,
            );
            let temperature = disk.temperature(hit.r);
            let pixel = DiskPixel {
                r: hit.r,
                phi: hit.phi,
                position: hit_position_lab(hit.r, hit.phi, init.beta),
                g,
                temperature,
            };
            let (collision, colour) = if g.is_finite() && g > 0.0 {
                (
                    Collision::Disk,
                    disk_pixel(g, temperature, disk.peak_temperature),
                )
            } else {
                (Collision::NumericalError, NUMERICAL_ERROR_COLOUR)
            };
            return ShadedRay {
                colour,
                collision,
                final_theta,
                final_phi,
                disk: Some(pixel),
            };
        }
    }
    let collision = if result.exit == ExitReason::Diverged || result.exit == ExitReason::Disk {
        // A disk exit without the disk geometry cannot be shaded.
        Collision::NumericalError
    } else {
        classify(result.q[1], rs, scene.boundary_radius, in_patch)
    };
    let colour = match collision {
        Collision::EscapedBackground => {
            let tex = scene.texture.expect("in_patch implies a texture");
            let (u, v) = scene
                .patch
                .texel(final_theta, final_phi, tex.height, tex.width);
            tex.texel(u, v)
        }
        Collision::NumericalError => NUMERICAL_ERROR_COLOUR,
        _ => BLACK,
    };
    ShadedRay {
        colour,
        collision,
        final_theta,
        final_phi,
        disk: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    use crate::integrator::DiskHit;

    fn result(r: f64, phi: f64, exit: ExitReason) -> IntegrationResult {
        IntegrationResult {
            q: [-40.0, r, PI / 2.0, phi],
            p: [1.0, -1.0, 0.0, 0.0],
            exit,
            n_steps: 100,
            disk_hit: None,
            plane_crossings: 0,
        }
    }

    fn init(beta: f64, p_t: f64, p_phi: f64) -> RayInit {
        RayInit {
            q0: [0.0, 30.0, PI / 2.0, 0.0],
            p0: [p_t, -1.0, 0.0, p_phi],
            alpha0: 0.1,
            beta,
            direction: [-1.0, 0.0, 0.0],
        }
    }

    fn scene<'a>(
        patch: &'a SkyPatch,
        texture: Option<&'a Texture>,
        disk: Option<&'a Disk>,
    ) -> Scene<'a> {
        Scene {
            black_hole: BlackHole::new(1.0),
            r_obs: 30.0,
            boundary_radius: 31.0,
            patch,
            texture,
            disk,
        }
    }

    fn shade_plain(
        result: &IntegrationResult,
        beta: f64,
        patch: &SkyPatch,
        texture: Option<&Texture>,
    ) -> ShadedRay {
        shade(result, &init(beta, 1.0, 0.0), &scene(patch, texture, None))
    }

    fn gradient_texture(h: usize, w: usize) -> Texture {
        let mut t = Texture::solid(w, h, [0, 0, 0]);
        for r in 0..h {
            for c in 0..w {
                t.data[r * w + c] = [r as u8, c as u8, 7];
            }
        }
        t
    }

    #[test]
    fn unrotate_is_inverse_of_initial_rotation() {
        // A direction with β = atan2(dz, dy) rotated into the plane and back.
        let dir: [f64; 3] = [-0.8, 0.3, 0.5];
        let beta = dir[2].atan2(dir[1]);
        let flat = mat_vec(&rot_x(-beta), dir);
        let (r, th, ph) = cartesian_to_spherical(flat);
        assert!((th - PI / 2.0).abs() < 1e-12);
        let (th2, ph2) = unrotate_hit(&[0.0, r, th, ph], beta);
        let (_, th_lab, ph_lab) = cartesian_to_spherical(dir);
        assert!((th2 - th_lab).abs() < 1e-12 && (ph2 - ph_lab).abs() < 1e-12);
    }

    #[test]
    fn classification_thresholds() {
        assert_eq!(classify(2.3, 2.0, 31.0, true), Collision::BlackHole);
        assert_eq!(classify(2.5, 2.0, 31.0, true), Collision::InDomain);
        assert_eq!(
            classify(31.0, 2.0, 31.0, true),
            Collision::EscapedBackground
        );
        assert_eq!(classify(31.0, 2.0, 31.0, false), Collision::EscapedNoPatch);
        assert_eq!(classify(150.0, 2.0, 31.0, true), Collision::NumericalError);
        assert_eq!(classify(-3.0, 2.0, 31.0, true), Collision::NumericalError);
        assert_eq!(
            classify(f64::NAN, 2.0, 31.0, true),
            Collision::NumericalError
        );
        assert_eq!(
            classify(f64::INFINITY, 2.0, 31.0, true),
            Collision::NumericalError
        );
    }

    #[test]
    fn shade_colours_each_outcome() {
        let patch = SkyPatch::full_sky();
        let tex = gradient_texture(16, 32);
        let escaped = shade_plain(
            &result(31.0, 2.0, ExitReason::Escaped),
            0.0,
            &patch,
            Some(&tex),
        );
        assert_eq!(escaped.collision, Collision::EscapedBackground);
        let (u, v) = patch.texel(escaped.final_theta, escaped.final_phi, 16, 32);
        assert_eq!(escaped.colour, tex.texel(u, v));
        assert_eq!(escaped.colour, [u as u8, v as u8, 7]);

        let no_texture = shade_plain(&result(31.0, 2.0, ExitReason::Escaped), 0.0, &patch, None);
        assert_eq!(no_texture.collision, Collision::EscapedNoPatch);
        assert_eq!(no_texture.colour, BLACK);

        let narrow = SkyPatch::from_degrees(90.0, 180.0, 10.0, 10.0, 0.0, 0.0, false, false);
        let outside = shade_plain(
            &result(31.0, 0.5, ExitReason::Escaped),
            0.0,
            &narrow,
            Some(&tex),
        );
        assert_eq!(outside.collision, Collision::EscapedNoPatch);

        let far = shade_plain(
            &result(150.0, 1.0, ExitReason::Escaped),
            0.0,
            &patch,
            Some(&tex),
        );
        assert_eq!(far.collision, Collision::NumericalError);
        assert_eq!(far.colour, NUMERICAL_ERROR_COLOUR);

        let diverged = shade_plain(
            &result(10.0, 1.0, ExitReason::Diverged),
            0.0,
            &patch,
            Some(&tex),
        );
        assert_eq!(diverged.collision, Collision::NumericalError);

        let captured = shade_plain(
            &result(2.19, 1.0, ExitReason::Captured),
            0.0,
            &patch,
            Some(&tex),
        );
        assert_eq!(captured.collision, Collision::BlackHole);
        assert_eq!(captured.colour, BLACK);
    }

    #[test]
    fn flipped_patch_mirrors_the_texel_but_not_the_pixel_classification() {
        let tex = gradient_texture(16, 32);
        let plain = SkyPatch::full_sky();
        let flipped = SkyPatch {
            flip_theta: true,
            flip_phi: true,
            ..plain
        };
        let res = result(31.0, 2.5, ExitReason::Escaped);
        let a = shade_plain(&res, 0.4, &plain, Some(&tex));
        let b = shade_plain(&res, 0.4, &flipped, Some(&tex));
        assert_eq!(a.collision, b.collision);
        let (u, v) = plain.texel(PI - a.final_theta, -a.final_phi, 16, 32);
        assert_eq!(b.colour, tex.texel(u, v));
        assert_ne!(a.colour, b.colour);
    }

    #[test]
    fn disk_hits_are_shaded_from_their_redshift() {
        let patch = SkyPatch::full_sky();
        let disk = Disk::new([0.0, 0.0, 1.0], 6.0, 20.0, 6000.0, 1.0, 30.0).unwrap();
        let mut res = result(9.0, 0.1, ExitReason::Disk);
        res.disk_hit = Some(DiskHit { r: 9.0, phi: PI });
        res.plane_crossings = 1;
        // A left-hand pixel (β = π): the frame has n'_z = −1 and p_φ > 0, so g > 1.
        let left = shade(&res, &init(PI, 1.0, 5.0), &scene(&patch, None, Some(&disk)));
        assert_eq!(left.collision, Collision::Disk);
        let px = left.disk.unwrap();
        assert!(px.g > 1.0 && px.g.is_finite());
        assert!((px.temperature - disk.temperature(9.0)).abs() < 1e-9);
        assert!(
            (px.position[0] + 9.0).abs() < 1e-12 && px.position[2].abs() < 1e-9,
            "{:?}",
            px.position
        );
        assert_ne!(left.colour, BLACK);
        let right = shade(
            &res,
            &init(0.0, 1.0, 5.0),
            &scene(&patch, None, Some(&disk)),
        );
        assert!(right.disk.unwrap().g < 1.0);
        // Expected closed form with b_z = −p_φ n'_z / p_t = 5.
        let g = (1.0 - 3.0 / 9.0f64).sqrt()
            / ((1.0 - (1.0 / 729.0f64).sqrt() * 5.0) * (1.0 - 2.0 / 30.0f64).sqrt());
        assert!((px.g - g).abs() < 1e-12);
        // A hit the physics rules out (1 − Ω b_z ≤ 0) is a numerical error, not a pixel.
        let bad = shade(
            &res,
            &init(PI, 1.0, 50.0),
            &scene(&patch, None, Some(&disk)),
        );
        assert_eq!(bad.collision, Collision::NumericalError);
        assert_eq!(bad.colour, NUMERICAL_ERROR_COLOUR);
        // A disk exit without disk geometry cannot be shaded either.
        let orphan = shade(&res, &init(PI, 1.0, 5.0), &scene(&patch, None, None));
        assert_eq!(orphan.collision, Collision::NumericalError);
    }
}
