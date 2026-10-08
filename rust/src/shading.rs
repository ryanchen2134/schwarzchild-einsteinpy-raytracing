//! From an integrated ray to a pixel: undo the equatorial rotation, classify
//! the exit, and look the escape direction up in the sky patch.

use crate::coords::{cartesian_to_spherical, mat_vec, rot_x, spherical_to_cartesian};
use crate::integrator::{ExitReason, IntegrationResult};
use crate::sky_patch::SkyPatch;
use crate::texture::{Rgb, Texture};

/// Rays that reach this radius have left the physical domain by a wide margin.
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
    /// Exit direction in the lab frame.
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

/// Classify by final radius, as the Python renderer did; the integrator's exit
/// reason is recorded separately in the photon table.
#[inline]
pub fn classify(r_final: f64, rs: f64, boundary_radius: f64, in_patch: bool) -> Collision {
    if r_final <= CAPTURE_RADIUS_FACTOR * rs {
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

/// Shade one ray.
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
    let collision = classify(result.q[1], rs, boundary_radius, in_patch);
    let colour = match collision {
        Collision::EscapedBackground => {
            let tex = texture.expect("in_patch implies a texture");
            let (u, v) = patch.texel(final_theta, final_phi, tex.height, tex.width);
            tex.texel(u, v)
        }
        Collision::NumericalError => NUMERICAL_ERROR_COLOUR,
        _ => BLACK,
    };
    let _ = ExitReason::Escaped; // exit reason is carried by `result`, not re-derived here
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
    }
}
