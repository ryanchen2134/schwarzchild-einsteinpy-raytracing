//! The region of the celestial sphere that carries the background texture,
//! and the one equirectangular rule that maps a direction to a texel.
//!
//! Membership does not depend on the flips: a flip mirrors the texture inside
//! the patch, it never moves the patch. (The Python curved renderer flipped `φ`
//! before testing membership; for the default full-sky patch the two agree.)

use crate::coords::TWO_PI;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkyPatch {
    /// Patch centre, radians, `θ ∈ [0, π]`, `φ ∈ [0, 2π)`.
    pub center_theta: f64,
    pub center_phi: f64,
    /// Full angular extent, radians.
    pub size_theta: f64,
    pub size_phi: f64,
    pub flip_theta: bool,
    pub flip_phi: bool,
}

impl SkyPatch {
    /// Build from the CLI's degree inputs. `dtheta_deg`/`dphi_deg` are offsets added
    /// to the centre (`θ` clamped to `[0, π]`, `φ` wrapped to `[0, 2π)`).
    #[allow(clippy::too_many_arguments)]
    pub fn from_degrees(
        center_theta_deg: f64,
        center_phi_deg: f64,
        size_theta_deg: f64,
        size_phi_deg: f64,
        dtheta_deg: f64,
        dphi_deg: f64,
        flip_theta: bool,
        flip_phi: bool,
    ) -> Self {
        let center_theta = (center_theta_deg.to_radians() + dtheta_deg.to_radians()).clamp(0.0, PI);
        let center_phi = (center_phi_deg.to_radians() + dphi_deg.to_radians()).rem_euclid(TWO_PI);
        Self {
            center_theta,
            center_phi,
            size_theta: size_theta_deg.to_radians(),
            size_phi: size_phi_deg.to_radians(),
            flip_theta,
            flip_phi,
        }
    }

    /// The whole sky, centred on the direction opposite the observer (`φ = π`).
    pub fn full_sky() -> Self {
        Self::from_degrees(90.0, 180.0, 180.0, 360.0, 0.0, 0.0, false, false)
    }

    /// `(θ0, θ1, φ0, φ_span)`.
    #[inline]
    pub fn bounds(&self) -> (f64, f64, f64, f64) {
        (
            self.center_theta - self.size_theta / 2.0,
            self.center_theta + self.size_theta / 2.0,
            self.center_phi - self.size_phi / 2.0,
            self.size_phi,
        )
    }

    /// Does the direction `(θ, φ)` fall inside the patch?
    #[inline]
    pub fn contains(&self, theta: f64, phi: f64) -> bool {
        let th = theta.rem_euclid(TWO_PI);
        let ph = phi.rem_euclid(TWO_PI);
        let dtheta = (th - self.center_theta).abs();
        let dphi = ((ph - self.center_phi + PI).rem_euclid(TWO_PI) - PI).abs();
        dtheta <= self.size_theta / 2.0 && dphi <= self.size_phi / 2.0
    }

    /// Texel `(row, col)` of a texture with `height × width` texels covering the
    /// patch, for a direction inside it. Rounds to the nearest texel and clamps.
    #[inline]
    pub fn texel(&self, theta: f64, phi: f64, height: usize, width: usize) -> (usize, usize) {
        let (theta0, theta1, phi0, phi_span) = self.bounds();
        let th = theta.rem_euclid(TWO_PI);
        let ph = phi.rem_euclid(TWO_PI);
        let theta_map = if self.flip_theta { PI - th } else { th };
        let phi_map = if self.flip_phi { -ph } else { ph };
        let phi_rel = (phi_map - phi0).rem_euclid(TWO_PI);
        let u = ((theta_map - theta0) / (theta1 - theta0) * (height as f64 - 1.0) + 0.5).floor();
        let v = (phi_rel / phi_span * (width as f64 - 1.0) + 0.5).floor();
        (
            u.clamp(0.0, height as f64 - 1.0) as usize,
            v.clamp(0.0, width as f64 - 1.0) as usize,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_sky_contains_everything() {
        let p = SkyPatch::full_sky();
        for th in [0.0, 0.3, 1.5, PI] {
            for ph in [-PI, -1.0, 0.0, 2.0, PI, 5.0] {
                assert!(p.contains(th, ph));
            }
        }
    }

    #[test]
    fn narrow_patch_wraps_around_phi_zero() {
        let p = SkyPatch::from_degrees(90.0, 350.0, 20.0, 40.0, 0.0, 0.0, false, false);
        assert!(p.contains(PI / 2.0, 355f64.to_radians()));
        assert!(p.contains(PI / 2.0, 5f64.to_radians()));
        assert!(p.contains(PI / 2.0, -5f64.to_radians()));
        assert!(!p.contains(PI / 2.0, 15f64.to_radians()));
        assert!(!p.contains(80f64.to_radians() - 1e-6, 350f64.to_radians()));
    }

    #[test]
    fn texel_corners_and_flips() {
        let p = SkyPatch::full_sky();
        // θ = 0 is the top row, φ = φ0 the first column.
        assert_eq!(p.texel(0.0, 0.0, 64, 128), (0, 0));
        assert_eq!(p.texel(PI, 0.0, 64, 128), (63, 0));
        let (u, v) = p.texel(PI / 2.0, PI, 64, 128);
        assert_eq!((u, v), (32, 64));
        let flipped = SkyPatch {
            flip_theta: true,
            flip_phi: true,
            ..p
        };
        let (fu, fv) = flipped.texel(PI / 2.0 - 0.3, 1.0, 64, 128);
        let (gu, gv) = p.texel(PI / 2.0 + 0.3, -1.0, 64, 128);
        assert_eq!(
            (fu, fv),
            (gu, gv),
            "flips mirror the texture about the patch centre"
        );
    }

    #[test]
    fn offsets_apply_in_degrees() {
        let p = SkyPatch::from_degrees(90.0, 180.0, 180.0, 360.0, 5.0, -10.0, false, false);
        assert!((p.center_theta - 95f64.to_radians()).abs() < 1e-12);
        assert!((p.center_phi - 170f64.to_radians()).abs() < 1e-12);
        let q = SkyPatch::from_degrees(90.0, 10.0, 10.0, 10.0, 0.0, -20.0, false, false);
        assert!(
            (q.center_phi - 350f64.to_radians()).abs() < 1e-12,
            "φ wraps"
        );
        let r = SkyPatch::from_degrees(175.0, 0.0, 10.0, 10.0, 10.0, 0.0, false, false);
        assert_eq!(r.center_theta, PI, "θ clamps");
    }
}
