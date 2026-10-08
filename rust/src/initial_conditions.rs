//! From a pixel ray to a null covariant 4-momentum.
//!
//! Each ray's orbit lies in a plane through the origin (spherical symmetry).
//! We rotate that plane onto the equator before integrating, which keeps the
//! integration away from the coordinate poles `θ = 0, π`, and rotate the result
//! back afterwards ([`crate::shading::unrotate_hit`]). The rotation is about
//! `+x`, so it leaves the observer on the `+x` axis fixed.
//!
//! Momentum convention: the integrator's Hamiltonian is `H = ½ g^{μν} p_μ p_ν`,
//! so `p` holds *covariant* components. For a static observer whose local
//! orthonormal frame is `e_r̂ = √f ∂_r`, `e_θ̂ = (1/r) ∂_θ`, `e_φ̂ = (1/(r sin θ)) ∂_φ`,
//! a photon with unit spatial direction `n = (n_r̂, n_θ̂, n_φ̂)` has
//! `p_r = n_r̂ / √f`, `p_θ = r n_θ̂`, `p_φ = r sin θ n_φ̂`.
//!
//! The Python implementation used `p_r = n_r̂ · √f` (the contravariant formula)
//! with the covariant `p_θ`, `p_φ`; that stretched every ray's angle from the
//! optical axis by `1/f ≈ 1.07` at `r = 30`. This port uses the consistent
//! covariant set; the golden tests record the difference.

use crate::coords::{cartesian_to_spherical, mat_vec, normalize, rot_x, sub, Vec3};
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
pub struct RayInit {
    /// `(t, r, θ, φ)` of the observer, with the ray rotated into the equatorial plane.
    pub q0: [f64; 4],
    /// Covariant null momentum `(p_t, p_r, p_θ, p_φ)` in the rotated frame; `p_θ = 0`.
    pub p0: [f64; 4],
    /// Angle between the ray and the optical axis (towards the black hole), `[0, π]`.
    pub alpha0: f64,
    /// Rotation about `+x` that was applied: `β = atan2(d_z, d_y)` of the lab-frame direction.
    pub beta: f64,
    /// Lab-frame unit direction of the ray.
    pub direction: Vec3,
}

/// Future-directed `p_t > 0` that makes `(p_t, p_r, p_θ, p_φ)` null at radius `r`, polar
/// angle `θ`, with lapse `f = 1 − 2M/r`. Operation order follows the Python code.
#[inline]
pub fn null_p_t(f: f64, r: f64, theta: f64, p_r: f64, p_th: f64, p_ph: f64) -> f64 {
    let gtt = -1.0 / f;
    let grr = f;
    let gthth = 1.0 / (r * r);
    let s = theta.sin();
    let gphph = 1.0 / (r * r * (s * s));
    let a = gtt;
    let c = grr * p_r * p_r + gthth * p_th * p_th + gphph * p_ph * p_ph;
    let disc = -4.0 * a * c;
    disc.sqrt() / (2.0 * (-a))
}

/// Covariant momentum of a ray leaving a static observer at `(r_obs, θ = π/2)` with
/// in-plane angle `alpha` from the inward radial direction.
#[inline]
pub fn equatorial_momentum(alpha: f64, r_obs: f64, mass: f64) -> (f64, f64, f64) {
    let n_rhat = -alpha.cos(); // inward
    let n_phhat = alpha.sin();
    let f_sqrt = (1.0 - 2.0 * mass / r_obs).sqrt();
    let p_r = n_rhat / f_sqrt;
    let p_th = 0.0;
    let p_ph = n_phhat * r_obs;
    (p_r, p_th, p_ph)
}

/// Initial conditions for the ray from `observer` through `pixel`.
pub fn initial_conditions(observer: Vec3, pixel: Vec3, mass: f64) -> RayInit {
    let direction = normalize(sub(pixel, observer));

    // Rotate the ray into the x–y plane.
    let beta = direction[2].atan2(direction[1]);
    let dir_xy = mat_vec(&rot_x(-beta), direction);
    debug_assert!(dir_xy[2].abs() < 1e-9);

    let (r_obs, theta_obs, phi_obs) = cartesian_to_spherical(observer);

    // In the rotated frame the ray makes azimuth h_phi with +x; the angle from the
    // inward (−x) direction is π − h_phi.
    let h_phi = dir_xy[1].atan2(dir_xy[0]);
    let alpha = PI - h_phi;

    let (p_r, p_th, p_ph) = equatorial_momentum(alpha, r_obs, mass);
    let f = 1.0 - 2.0 * mass / r_obs;
    let p_t = null_p_t(f, r_obs, theta_obs, p_r, p_th, p_ph);

    RayInit {
        q0: [0.0, r_obs, theta_obs, phi_obs],
        p0: [p_t, p_r, p_th, p_ph],
        alpha0: (alpha.cos()).acos(),
        beta,
        direction,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn momentum_is_null() {
        let obs = [30.0, 0.0, 0.0];
        let init = initial_conditions(obs, [24.0, 1.3, -0.7], 1.0);
        let [p_t, p_r, p_th, p_ph] = init.p0;
        let r = init.q0[1];
        let f = 1.0 - 2.0 / r;
        let h = -p_t * p_t / f + f * p_r * p_r + p_th * p_th / (r * r) + p_ph * p_ph / (r * r);
        assert!(h.abs() < 1e-12, "H = {h}");
        assert_eq!(p_th, 0.0);
        assert!(p_t > 0.0, "future directed");
        assert!(p_r < 0.0, "inward");
    }

    #[test]
    fn alpha0_is_angle_from_optical_axis() {
        let obs = [30.0, 0.0, 0.0];
        for pix in [[24.0, 2.0, 1.0], [24.0, -3.0, 0.5], [24.0, 0.0, -2.0]] {
            let init = initial_conditions(obs, pix, 1.0);
            let expect = (-init.direction[0]).acos();
            assert!((init.alpha0 - expect).abs() < 1e-12);
        }
    }

    #[test]
    fn on_axis_ray_is_radial() {
        let init = initial_conditions([30.0, 0.0, 0.0], [24.0, 0.0, 0.0], 1.0);
        assert!(init.alpha0.abs() < 1e-15);
        assert!(init.p0[3].abs() < 1e-15);
    }
}
