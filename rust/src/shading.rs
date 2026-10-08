//! From an integrated ray to a pixel: undo the equatorial rotation, classify
//! the exit, and look the escape direction up in the sky patch.

use crate::coords::{cartesian_to_spherical, mat_vec, rot_x, spherical_to_cartesian};
use crate::integrator::{ExitReason, IntegrationResult};
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
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ShadedRay {
    pub colour: Rgb,
    pub collision: Collision,
    /// Exit direction in the lab frame: `θ ∈ [0, π]`, `φ ∈ (−π, π]`.
    pub final_theta: f64,
    pub final_phi: f64,
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

/// Shade one ray. A diverged integration is a numerical error whatever its radius.
pub fn shade(
    result: &IntegrationResult,
    beta: f64,
    rs: f64,
    boundary_radius: f64,
    patch: &SkyPatch,
    texture: Option<&Texture>,
) -> ShadedRay {
    let (final_theta, final_phi) = unrotate_hit(&result.q, beta);
    let in_patch = texture.is_some() && patch.contains(final_theta, final_phi);
    let collision = if result.exit == ExitReason::Diverged {
        Collision::NumericalError
    } else {
        classify(result.q[1], rs, boundary_radius, in_patch)
    };
    let colour = match collision {
        Collision::EscapedBackground => {
            let tex = texture.expect("in_patch implies a texture");
            let (u, v) = patch.texel(final_theta, final_phi, tex.height, tex.width);
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn result(r: f64, phi: f64, exit: ExitReason) -> IntegrationResult {
        IntegrationResult {
            q: [-40.0, r, PI / 2.0, phi],
            p: [1.0, -1.0, 0.0, 0.0],
            exit,
            n_steps: 100,
        }
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
        let escaped = shade(
            &result(31.0, 2.0, ExitReason::Escaped),
            0.0,
            2.0,
            31.0,
            &patch,
            Some(&tex),
        );
        assert_eq!(escaped.collision, Collision::EscapedBackground);
        let (u, v) = patch.texel(escaped.final_theta, escaped.final_phi, 16, 32);
        assert_eq!(escaped.colour, tex.texel(u, v));
        assert_eq!(escaped.colour, [u as u8, v as u8, 7]);

        let no_texture = shade(
            &result(31.0, 2.0, ExitReason::Escaped),
            0.0,
            2.0,
            31.0,
            &patch,
            None,
        );
        assert_eq!(no_texture.collision, Collision::EscapedNoPatch);
        assert_eq!(no_texture.colour, BLACK);

        let narrow = SkyPatch::from_degrees(90.0, 180.0, 10.0, 10.0, 0.0, 0.0, false, false);
        let outside = shade(
            &result(31.0, 0.5, ExitReason::Escaped),
            0.0,
            2.0,
            31.0,
            &narrow,
            Some(&tex),
        );
        assert_eq!(outside.collision, Collision::EscapedNoPatch);

        let far = shade(
            &result(150.0, 1.0, ExitReason::Escaped),
            0.0,
            2.0,
            31.0,
            &patch,
            Some(&tex),
        );
        assert_eq!(far.collision, Collision::NumericalError);
        assert_eq!(far.colour, NUMERICAL_ERROR_COLOUR);

        let diverged = shade(
            &result(10.0, 1.0, ExitReason::Diverged),
            0.0,
            2.0,
            31.0,
            &patch,
            Some(&tex),
        );
        assert_eq!(diverged.collision, Collision::NumericalError);

        let captured = shade(
            &result(2.19, 1.0, ExitReason::Captured),
            0.0,
            2.0,
            31.0,
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
        let a = shade(&res, 0.4, 2.0, 31.0, &plain, Some(&tex));
        let b = shade(&res, 0.4, 2.0, 31.0, &flipped, Some(&tex));
        assert_eq!(a.collision, b.collision);
        let (u, v) = plain.texel(PI - a.final_theta, -a.final_phi, 16, 32);
        assert_eq!(b.colour, tex.texel(u, v));
        assert_ne!(a.colour, b.colour);
    }
}
