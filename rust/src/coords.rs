//! 3-vector helpers and coordinate conversions.
//!
//! The spherical conventions match `einsteinpy.coordinates.utils` so that
//! results can be compared against the Python implementation bit for bit:
//! `θ` is the polar angle from `+z`, `φ` the azimuth from `+x`.

use std::f64::consts::PI;

pub type Vec3 = [f64; 3];
pub type Mat3 = [[f64; 3]; 3];

pub const TWO_PI: f64 = 2.0 * PI;

#[inline]
pub fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
pub fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub fn scale(a: Vec3, s: f64) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
pub fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub fn norm(a: Vec3) -> f64 {
    dot(a, a).sqrt()
}

#[inline]
pub fn normalize(a: Vec3) -> Vec3 {
    let n = norm(a);
    [a[0] / n, a[1] / n, a[2] / n]
}

#[inline]
pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// `(r, θ, φ)` → `(x, y, z)`.
#[inline]
pub fn spherical_to_cartesian(r: f64, theta: f64, phi: f64) -> Vec3 {
    let st = theta.sin();
    [r * st * phi.cos(), r * st * phi.sin(), r * theta.cos()]
}

/// `(x, y, z)` → `(r, θ, φ)` with `θ ∈ [0, π]` and `φ ∈ (−π, π]`.
#[inline]
pub fn cartesian_to_spherical(v: Vec3) -> (f64, f64, f64) {
    let hxy = v[0].hypot(v[1]);
    let r = hxy.hypot(v[2]);
    (r, hxy.atan2(v[2]), v[1].atan2(v[0]))
}

/// Right-handed rotation about `+x` by `angle`.
///
/// `rot_x(-β)` brings a ray with direction `(dx, dy, dz)` into the x–y plane
/// when `β = atan2(dz, dy)`; `rot_x(β)` undoes it.
#[inline]
pub fn rot_x(angle: f64) -> Mat3 {
    let (s, c) = angle.sin_cos();
    [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
}

#[inline]
pub fn mat_vec(m: &Mat3, v: Vec3) -> Vec3 {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// `n` points from `a` to `b` inclusive (`numpy.linspace` on vectors).
pub fn linspace(a: Vec3, b: Vec3, n: usize) -> Vec<Vec3> {
    if n == 1 {
        return vec![a];
    }
    (0..n)
        .map(|k| {
            let t = k as f64 / (n - 1) as f64;
            add(a, scale(sub(b, a), t))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spherical_roundtrip() {
        for &(r, th, ph) in &[(30.0, 1.0, 0.3), (2.5, 3.0, -2.0), (7.0, 0.1, 3.1)] {
            let v = spherical_to_cartesian(r, th, ph);
            let (r2, th2, ph2) = cartesian_to_spherical(v);
            assert!((r - r2).abs() < 1e-12 && (th - th2).abs() < 1e-12 && (ph - ph2).abs() < 1e-12);
        }
    }

    #[test]
    fn rot_x_inverse_and_handedness() {
        let v: Vec3 = [0.3, 0.5, -0.8];
        let beta = (v[2]).atan2(v[1]);
        let flat = mat_vec(&rot_x(-beta), v);
        assert!(
            flat[2].abs() < 1e-15,
            "rot_x(-β) must put the ray in the x-y plane"
        );
        assert!(
            flat[1] > 0.0,
            "the in-plane y component is the positive radius hypot(dy, dz)"
        );
        let back = mat_vec(&rot_x(beta), flat);
        for k in 0..3 {
            assert!((back[k] - v[k]).abs() < 1e-15);
        }
    }

    #[test]
    fn linspace_endpoints() {
        let pts = linspace([0.0, 0.0, 0.0], [1.0, 2.0, 3.0], 5);
        assert_eq!(pts.len(), 5);
        assert_eq!(pts[0], [0.0, 0.0, 0.0]);
        assert_eq!(pts[4], [1.0, 2.0, 3.0]);
        assert_eq!(pts[2], [0.5, 1.0, 1.5]);
    }
}
