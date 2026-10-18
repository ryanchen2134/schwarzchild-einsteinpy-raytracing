//! The accretion disk: a geometrically thin, optically thick sheet of material on
//! circular Keplerian orbits in a plane through the black hole, and the closed-form
//! physics that turns a ray's hit on it into a pixel.
//!
//! # Geometry
//!
//! The disk plane is `n · x = 0` for a unit normal `n`. The CLI gives `n` by two
//! angles. The *inclination* `i ∈ [0°, 180°]` is the angle between `n` and the
//! observer's position vector `+x̂`: 0 is face-on, 90 edge-on, 180 face-on from the
//! other side, which reverses the sense of rotation on the sky. The *azimuth* `a`
//! turns `n` right-handedly about `+x̂`: `n = R_x(a) · (cos i, 0, sin i)`. At `a = 0`
//! the normal lies in the x–z plane, so an inclined disk's near side is below the
//! black hole in the image and its far side above. Both angles are measured against
//! the observer's position, not the optical axis, so turning the camera does not move
//! the disk.
//!
//! The material orbits right-handedly about `n`: at `x` it moves along `n × x̂`. With
//! `i = 90°, a = 0` (`n = +ẑ`) the material at `(0, −r, 0)`, on the observer's left,
//! moves along `+x̂`, towards the observer: the left side of an edge-on disk
//! approaches and is blueshifted.
//!
//! # Redshift
//!
//! Each ray is integrated in a frame rotated about `+x̂` by `−β`
//! ([`crate::initial_conditions`]). In that frame its orbit is the x–y plane, its
//! angular-momentum vector lies along `+ẑ'` with magnitude `p_φ`, and the disk normal
//! is `n' = R_x(−β) n`. The physical photon is the time reverse of the traced ray, so
//! its energy at infinity is `E = p_t` and its angular momentum about `n` is
//! `L_n = −p_φ n'_z`. With `b_z = L_n / E` and the Keplerian angular velocity
//! `Ω = √(M/r³)`, the ratio of observed to emitted frequency is
//!
//! `g = (k·u_obs)/(k·u_emit) = √(1 − 3M/r) / ((1 − Ω b_z) √(1 − 2M/r_obs))`
//!
//! for the static observer `u_obs = (1,0,0,0)/√(1 − 2M/r_obs)` and the circular
//! orbit `u_emit = (1,0,0,Ω)/√(1 − 3M/r)`. `g` is a ratio of two contractions of the
//! same covector, so the sign convention of the traced momentum cancels. `p_t` and
//! `p_φ` are conserved, so `g` depends only on where the ray hits. The formula needs
//! `r ≥ 3M`, and circular orbits are stable only from the ISCO at `6M`, so the inner
//! radius must be at least `6M`.
//!
//! # Emission
//!
//! The effective temperature follows the Newtonian Shakura–Sunyaev profile with zero
//! torque at the inner edge, `T⁴ ∝ (1 − √(r_in/r)) / r³`, which peaks at
//! `r = 49/36 r_in`; the peak value is the user's `--disk-temperature`. Liouville's
//! theorem (`I_ν/ν³` is invariant) makes the observed spectrum a blackbody at `g·T`:
//! its bolometric intensity is `(gT)⁴` and its colour the chromaticity of a blackbody
//! at `gT`; nothing is multiplied by `g⁴` on top of that. The pixel is that
//! chromaticity scaled by the tone-mapped intensity `(gT/T_peak)⁴`. The temperature is
//! a false-colour parameter: at realistic disk temperatures (10⁵–10⁷ K) every pixel
//! would be white.

use crate::coords::{dot, mat_vec, norm, normalize, rot_x, scale, Vec3};
use crate::texture::Rgb;
use anyhow::{ensure, Result};

/// Innermost stable circular orbit, in units of `M`.
pub const ISCO_PER_M: f64 = 6.0;
/// The temperature profile peaks at this multiple of the inner radius.
pub const PEAK_RADIUS_PER_R_IN: f64 = 49.0 / 36.0;
/// A ray whose orbital plane is within this of the disk plane (`|n'_⊥|`, the part
/// of the disk normal perpendicular to the ray's angular momentum) lies in the
/// disk and has no crossing to detect.
pub const PARALLEL_TOLERANCE: f64 = 1e-9;
/// Below this effective temperature the disk is not drawn (the colour fit is only
/// valid from 1667 K; the profile reaches zero at the inner edge anyway).
pub const MIN_DRAWN_TEMPERATURE: f64 = 1.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Disk {
    /// Unit normal of the disk plane; the material orbits right-handedly about it.
    pub normal: Vec3,
    pub r_in: f64,
    pub r_out: f64,
    /// Peak effective temperature in kelvin (false colour).
    pub peak_temperature: f64,
}

/// The disk as one ray's rotated frame sees it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayDiskFrame {
    /// Azimuth, in the ray's frame, of the line where the disk plane meets the
    /// orbital plane (the plane is crossed at `phi_d + kπ`). `NaN` when the orbit lies
    /// in the disk plane, in which case no crossing is detected.
    pub phi_d: f64,
    /// `n'_z`: the disk normal's component along the ray's angular-momentum axis.
    pub n_z: f64,
}

impl Disk {
    /// A disk with the given plane normal (any length) and radii in the same length
    /// unit as `mass` and `r_obs`.
    pub fn new(
        normal: Vec3,
        r_in: f64,
        r_out: f64,
        peak_temperature: f64,
        mass: f64,
        r_obs: f64,
    ) -> Result<Self> {
        let len = norm(normal);
        ensure!(
            len.is_finite() && len > 0.0,
            "disk normal must be a non-zero vector, got {normal:?}"
        );
        let isco = ISCO_PER_M * mass;
        ensure!(
            r_in.is_finite() && r_in >= isco,
            "the disk's inner radius ({r_in}) must be at least the ISCO, 6M = {isco}: there \
             are no stable circular orbits inside it, and below 3M the Keplerian redshift \
             is undefined"
        );
        ensure!(
            r_out.is_finite() && r_out > r_in,
            "the disk's outer radius ({r_out}) must exceed its inner radius ({r_in})"
        );
        ensure!(
            r_out < r_obs,
            "the disk's outer radius ({r_out}) must be smaller than the observer distance \
             ({r_obs}): every ray starts on the crossing line of an edge-on disk"
        );
        ensure!(
            peak_temperature.is_finite() && peak_temperature > 0.0,
            "the disk temperature must be positive, got {peak_temperature}"
        );
        Ok(Self {
            normal: normalize(normal),
            r_in,
            r_out,
            peak_temperature,
        })
    }

    /// `n = R_x(azimuth) · (cos i, 0, sin i)`; angles in radians (see the module docs).
    pub fn normal_from_angles(inclination: f64, azimuth: f64) -> Vec3 {
        let (si, ci) = inclination.sin_cos();
        mat_vec(&rot_x(azimuth), [ci, 0.0, si])
    }

    /// The disk in the frame of a ray rotated by `β` (see [`crate::initial_conditions`]).
    pub fn ray_frame(&self, beta: f64) -> RayDiskFrame {
        let n = mat_vec(&rot_x(-beta), self.normal);
        let perp = n[0].hypot(n[1]);
        let phi_d = if perp < PARALLEL_TOLERANCE {
            f64::NAN
        } else {
            (-n[0]).atan2(n[1])
        };
        RayDiskFrame { phi_d, n_z: n[2] }
    }

    #[inline]
    pub fn contains_radius(&self, r: f64) -> bool {
        r >= self.r_in && r <= self.r_out
    }

    /// Effective temperature at radius `r`: zero at and inside `r_in`, the peak value
    /// at `49/36 r_in`.
    pub fn temperature(&self, r: f64) -> f64 {
        let peak = temperature_shape(self.r_in, self.r_in * PEAK_RADIUS_PER_R_IN);
        self.peak_temperature * (temperature_shape(self.r_in, r) / peak).powf(0.25)
    }

    /// Where the straight ray `origin + t·dir` (`t > 0`) meets the disk, if it does.
    pub fn straight_ray_hit(&self, origin: Vec3, dir: Vec3) -> Option<Vec3> {
        let denom = dot(self.normal, dir);
        if denom == 0.0 {
            return None;
        }
        let t = -dot(self.normal, origin) / denom;
        if t <= 0.0 {
            return None;
        }
        let hit = crate::coords::add(origin, scale(dir, t));
        self.contains_radius(norm(hit)).then_some(hit)
    }
}

/// `(1 − √(r_in/r)) / r³`, the shape of `T⁴`; zero at and inside `r_in`.
#[inline]
pub fn temperature_shape(r_in: f64, r: f64) -> f64 {
    if r <= r_in {
        0.0
    } else {
        (1.0 - (r_in / r).sqrt()) / (r * r * r)
    }
}

/// Keplerian coordinate angular velocity `Ω = √(M/r³)`.
#[inline]
pub fn keplerian_omega(mass: f64, r: f64) -> f64 {
    (mass / (r * r * r)).sqrt()
}

/// `g = √(1 − 3M/r) / ((1 − Ω b_z) √(1 − 2M/r_obs))` with `b_z = −p_φ n'_z / p_t`
/// (module docs). `p_t` and `p_φ` are the traced ray's conserved momenta in its
/// rotated frame. A result that is not finite and positive means `1 − Ω b_z ≤ 0`,
/// which no photon reaching the observer can produce: treat it as an error.
#[inline]
pub fn redshift(mass: f64, r: f64, p_t: f64, p_phi: f64, n_z: f64, r_obs: f64) -> f64 {
    let omega = keplerian_omega(mass, r);
    let b_z = -p_phi * n_z / p_t;
    let denom = 1.0 - omega * b_z;
    (1.0 - 3.0 * mass / r).sqrt() / (denom * (1.0 - 2.0 * mass / r_obs).sqrt())
}

/// Lab-frame position of a hit at `(r, φ)` in the frame of a ray rotated by `β`.
#[inline]
pub fn hit_position_lab(r: f64, phi: f64, beta: f64) -> Vec3 {
    let (s, c) = phi.sin_cos();
    mat_vec(&rot_x(beta), [r * c, r * s, 0.0])
}

/// Chromaticity (linear sRGB, largest channel 1) of a blackbody at `t` kelvin, from
/// Kang et al. (2002)'s fit of the Planckian locus; `t` is clamped to the fit's
/// 1667–25000 K range.
pub fn blackbody_chromaticity(t: f64) -> [f64; 3] {
    let t = t.clamp(1667.0, 25000.0);
    let inv = 1.0 / t;
    let inv2 = inv * inv;
    let inv3 = inv2 * inv;
    let x = if t <= 4000.0 {
        -0.2661239e9 * inv3 - 0.2343589e6 * inv2 + 0.8776956e3 * inv + 0.179910
    } else {
        -3.0258469e9 * inv3 + 2.1070379e6 * inv2 + 0.2226347e3 * inv + 0.240390
    };
    let x2 = x * x;
    let x3 = x2 * x;
    let y = if t <= 2222.0 {
        -1.1063814 * x3 - 1.34811020 * x2 + 2.18555832 * x - 0.20219683
    } else if t <= 4000.0 {
        -0.9549476 * x3 - 1.37418593 * x2 + 2.09137015 * x - 0.16748867
    } else {
        3.0817580 * x3 - 5.87338670 * x2 + 3.75112997 * x - 0.37001483
    };
    // CIE xyY with Y = 1 → XYZ → linear sRGB (D65).
    let cx = x / y;
    let cz = (1.0 - x - y) / y;
    let r = 3.2406 * cx - 1.5372 - 0.4986 * cz;
    let g = -0.9689 * cx + 1.8758 + 0.0415 * cz;
    let b = 0.0557 * cx - 0.2040 + 1.0570 * cz;
    let m = r.max(g).max(b);
    [(r / m).max(0.0), (g / m).max(0.0), (b / m).max(0.0)]
}

/// Maps a bolometric intensity relative to the disk's peak (`(gT/T_peak)⁴`, which
/// exceeds 1 on the approaching side) to a display luminance in `[0, 1)`.
#[inline]
pub fn tonemap(relative_intensity: f64) -> f64 {
    1.0 - (-2.0 * relative_intensity).exp()
}

/// Linear light in `[0, 1]` to an 8-bit sRGB value.
#[inline]
pub fn srgb_encode(v: f64) -> u8 {
    let v = if v.is_finite() {
        v.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let s = if v <= 0.0031308 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round() as u8
}

/// The pixel for a hit seen with redshift factor `g` on material at effective
/// temperature `t_emit`, relative to the disk's peak temperature.
pub fn disk_pixel(g: f64, t_emit: f64, t_peak: f64) -> Rgb {
    let t_obs = g * t_emit;
    if t_obs.is_nan() || t_obs < MIN_DRAWN_TEMPERATURE {
        return [0, 0, 0];
    }
    let x = t_obs / t_peak;
    let x2 = x * x;
    let l = tonemap(x2 * x2);
    let c = blackbody_chromaticity(t_obs);
    [
        srgb_encode(l * c[0]),
        srgb_encode(l * c[1]),
        srgb_encode(l * c[2]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coords::{cross, mat_vec, normalize, rot_x};
    use crate::initial_conditions::initial_conditions;
    use std::f64::consts::PI;

    const M: f64 = 1.0;
    const R_OBS: f64 = 30.0;

    fn disk(normal: Vec3) -> Disk {
        Disk::new(normal, 6.0, 20.0, 6000.0, M, R_OBS).unwrap()
    }

    /// Independent truth for `g`, in the static orthonormal frame at the hit: the
    /// emitter moves at `v = rΩ/√f` along `n × x̂`, the photon's unit direction is
    /// `k̂`, and `g = √f √(1 − v²) / (√f_obs (1 − v·k̂))`. No rotated frame, no
    /// covariant momenta, no `b_z`.
    fn kinematic_redshift(n: Vec3, x_hit: Vec3, k_hat: Vec3) -> f64 {
        let r = norm(x_hit);
        let f = 1.0 - 2.0 * M / r;
        let omega = keplerian_omega(M, r);
        let v = r * omega / f.sqrt();
        let e_v = cross(n, normalize(x_hit));
        assert!(
            (norm(e_v) - 1.0).abs() < 1e-12,
            "x_hit must lie in the disk"
        );
        f.sqrt() * (1.0 - v * v).sqrt()
            / ((1.0 - 2.0 * M / R_OBS).sqrt() * (1.0 - v * dot(e_v, k_hat)))
    }

    /// The One Thing First: on an oblique disk, the closed form that the renderer
    /// uses (conserved `p_t`, `p_φ`, the rotated normal's `n'_z`) equals the kinematic
    /// Doppler + gravitational factor computed from lab-frame geometry alone. This can
    /// only pass if the spin sense, the inclination convention and the direction of
    /// the frame rotation (`R_x(−β)`, not `R_x(+β)`) are all right together.
    #[test]
    fn oblique_oracle_closed_form_equals_kinematic_redshift() {
        let n = Disk::normal_from_angles(55f64.to_radians(), 30f64.to_radians());
        let d = disk(n);
        let x_axis = [1.0, 0.0, 0.0];
        let mut checked = 0;
        // Generic pixel directions (yawed and pitched views included), hit radii and
        // in-plane photon directions.
        for dir in [
            [-0.9, 0.3, 0.2],
            [-0.8, -0.4, 0.35],
            [-0.95, 0.1, -0.25],
            [-0.7, -0.5, -0.4],
            [-0.99, 0.02, 0.1],
        ] {
            let dir = normalize(dir);
            let beta = dir[2].atan2(dir[1]);
            let frame = d.ray_frame(beta);
            assert!(frame.phi_d.is_finite());
            let orbit_normal = normalize(cross(x_axis, dir));
            // The disk plane meets the orbital plane along this line.
            let line = normalize(cross(n, orbit_normal));
            let in_plane = cross(orbit_normal, line);
            let rot = rot_x(-beta);
            for r_hit in [6.5, 9.0, 13.0, 19.5] {
                for sign in [1.0, -1.0] {
                    let x_hit = scale(line, sign * r_hit);
                    let x_rot = mat_vec(&rot, x_hit);
                    assert!(
                        x_rot[2].abs() < 1e-12,
                        "the hit lies in the rotated orbital plane"
                    );
                    let phi_h = x_rot[1].atan2(x_rot[0]);
                    let off = ((phi_h - frame.phi_d) / PI).round();
                    assert!(
                        (phi_h - frame.phi_d - off * PI).abs() < 1e-12,
                        "the hit sits on the crossing line phi_d + kπ"
                    );
                    for psi in [0.3f64, 1.4, 2.5, 4.0, 5.5] {
                        let k_hat =
                            crate::coords::add(scale(line, psi.cos()), scale(in_plane, psi.sin()));
                        // Covariant traced momentum at the hit, in the rotated frame, for
                        // a photon of unit energy in the static frame: p = −k.
                        let f = 1.0 - 2.0 * M / r_hit;
                        let p_t = f.sqrt();
                        let k_rot = mat_vec(&rot, k_hat);
                        let e_phi = [-phi_h.sin(), phi_h.cos(), 0.0];
                        let p_phi = -r_hit * dot(k_rot, e_phi);
                        let g_closed = redshift(M, r_hit, p_t, p_phi, frame.n_z, R_OBS);
                        let g_kin = kinematic_redshift(n, x_hit, k_hat);
                        assert!(
                            (g_closed - g_kin).abs() < 1e-12,
                            "dir {dir:?} r {r_hit} psi {psi}: closed {g_closed} kinematic {g_kin}"
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert_eq!(checked, 200);
    }

    /// The same oracle with the wrong rotation sense must fail: the test has teeth.
    #[test]
    fn oracle_rejects_the_opposite_frame_rotation() {
        let n = Disk::normal_from_angles(55f64.to_radians(), 30f64.to_radians());
        let dir = normalize([-0.9, 0.3, 0.2]);
        let beta = dir[2].atan2(dir[1]);
        let wrong_n = mat_vec(&rot_x(beta), n);
        let orbit_normal = normalize(cross([1.0, 0.0, 0.0], dir));
        let line = normalize(cross(n, orbit_normal));
        let x_hit = scale(line, 9.0);
        let rot = rot_x(-beta);
        let x_rot = mat_vec(&rot, x_hit);
        let phi_h = x_rot[1].atan2(x_rot[0]);
        let k_hat = crate::coords::add(
            scale(line, 0.3f64.cos()),
            scale(cross(orbit_normal, line), 0.3f64.sin()),
        );
        let p_t = (1.0 - 2.0 * M / 9.0).sqrt();
        let k_rot = mat_vec(&rot, k_hat);
        let p_phi = -9.0 * dot(k_rot, [-phi_h.sin(), phi_h.cos(), 0.0]);
        let g_wrong = redshift(M, 9.0, p_t, p_phi, wrong_n[2], R_OBS);
        let g_kin = kinematic_redshift(n, x_hit, k_hat);
        assert!((g_wrong - g_kin).abs() > 1e-3, "{g_wrong} vs {g_kin}");
    }

    #[test]
    fn face_on_disk_has_no_doppler_shift_whatever_the_pointing() {
        for normal in [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]] {
            let d = disk(normal);
            for beta in [0.0, 0.4, 1.9, -2.3, PI] {
                let frame = d.ray_frame(beta);
                assert_eq!(frame.n_z, 0.0, "n'_z vanishes exactly: b_z ≡ 0");
                assert!(frame.phi_d.is_finite());
                for r in [6.0, 10.0, 20.0] {
                    let g = redshift(M, r, 0.97, 7.3, frame.n_z, R_OBS);
                    let expect = (1.0 - 3.0 * M / r).sqrt() / (1.0 - 2.0 * M / R_OBS).sqrt();
                    assert!((g - expect).abs() < 1e-15);
                }
            }
        }
    }

    /// Written pin, independent of every formula: with `n = +ẑ` the material at
    /// `(0, −r, 0)` moves along `n × x̂ = +x̂`, towards the observer, so a pixel on the
    /// observer's left sees `g > 1` and one on the right `g < 1`.
    #[test]
    fn left_side_of_an_edge_on_disk_approaches_the_observer() {
        let d = disk([0.0, 0.0, 1.0]);
        let left = initial_conditions([R_OBS, 0.0, 0.0], [24.0, -2.0, 0.01], M);
        let right = initial_conditions([R_OBS, 0.0, 0.0], [24.0, 2.0, 0.01], M);
        for (init, approaching) in [(left, true), (right, false)] {
            let frame = d.ray_frame(init.beta);
            let g = redshift(M, 8.0, init.p0[0], init.p0[3], frame.n_z, R_OBS);
            assert!(g.is_finite() && g > 0.0);
            assert_eq!(
                g > 1.0,
                approaching,
                "g = {g} for pixel {:?}",
                init.direction
            );
        }
        // Flipping the normal flips the sense of rotation.
        let flipped = disk([0.0, 0.0, -1.0]);
        let frame = flipped.ray_frame(left.beta);
        assert!(redshift(M, 8.0, left.p0[0], left.p0[3], frame.n_z, R_OBS) < 1.0);
    }

    #[test]
    fn reversing_the_normal_mirrors_the_redshift_map() {
        let n = Disk::normal_from_angles(70f64.to_radians(), 0.0);
        let d = disk(n);
        let anti = disk([-n[0], -n[1], -n[2]]);
        for pix in [[24.0, -3.0, 1.0], [24.0, 1.5, -2.0], [24.0, 4.0, 0.3]] {
            let a = initial_conditions([R_OBS, 0.0, 0.0], pix, M);
            let b = initial_conditions([R_OBS, 0.0, 0.0], [pix[0], -pix[1], pix[2]], M);
            let ga = redshift(M, 9.0, a.p0[0], a.p0[3], d.ray_frame(a.beta).n_z, R_OBS);
            let gb = redshift(M, 9.0, b.p0[0], b.p0[3], anti.ray_frame(b.beta).n_z, R_OBS);
            assert!((ga - gb).abs() < 1e-12, "{ga} vs {gb}");
        }
    }

    #[test]
    fn rays_in_the_disk_plane_are_flagged_parallel() {
        let d = disk([0.0, 0.0, 1.0]);
        assert!(d.ray_frame(0.0).phi_d.is_nan());
        assert!(d.ray_frame(PI).phi_d.is_nan());
        let f = d.ray_frame(0.3);
        assert!(f.phi_d.is_finite());
        // n' = (0, sin β, cos β): the crossing line is the x axis.
        assert!(f.phi_d.abs() < 1e-15 && (f.n_z - 0.3f64.cos()).abs() < 1e-15);
        let back = d.ray_frame(-0.3).phi_d;
        assert!(
            (back.abs() - PI).abs() < 1e-15,
            "crossing line at ±π: {back}"
        );
    }

    #[test]
    fn normal_from_angles_follows_the_documented_convention() {
        let n = Disk::normal_from_angles(0.0, 1.0);
        assert!((n[0] - 1.0).abs() < 1e-15 && n[1].abs() < 1e-15 && n[2].abs() < 1e-15);
        let n = Disk::normal_from_angles(PI / 2.0, 0.0);
        assert!(n[0].abs() < 1e-15 && n[1].abs() < 1e-15 && (n[2] - 1.0).abs() < 1e-15);
        let n = Disk::normal_from_angles(PI / 2.0, PI / 2.0);
        assert!(
            (n[1] + 1.0).abs() < 1e-15,
            "azimuth turns the normal right-handedly about +x̂: ẑ → −ŷ"
        );
        let n = Disk::normal_from_angles(1.0, 0.0);
        let near_side_z = -n[0] / n[2]; // z of the disk point (1, 0, z): below the hole
        assert!(near_side_z < 0.0);
        let anti = Disk::normal_from_angles(PI - 1.0, PI);
        let n1 = Disk::normal_from_angles(1.0, 0.0);
        for k in 0..3 {
            assert!(
                (anti[k] + n1[k]).abs() < 1e-15,
                "(i, a) → (180° − i, a + 180°) reverses n"
            );
        }
    }

    #[test]
    fn temperature_profile_peaks_at_49_over_36_r_in() {
        let d = disk([1.0, 0.0, 0.0]);
        assert_eq!(d.temperature(6.0), 0.0);
        assert_eq!(d.temperature(5.0), 0.0);
        let r_peak = 6.0 * PEAK_RADIUS_PER_R_IN;
        assert!((d.temperature(r_peak) - 6000.0).abs() < 1e-9);
        let mut best = 0.0f64;
        let mut r = 6.0;
        while r < 20.0 {
            best = best.max(d.temperature(r));
            r += 0.001;
        }
        assert!(best <= 6000.0 + 1e-6 && best > 5999.9);
        assert!(d.temperature(20.0) < d.temperature(10.0));
    }

    #[test]
    fn blackbody_colours_run_from_orange_to_blue() {
        let cool = blackbody_chromaticity(3000.0);
        let warm = blackbody_chromaticity(6500.0);
        let hot = blackbody_chromaticity(15000.0);
        for c in [cool, warm, hot] {
            assert!(c.iter().all(|v| (0.0..=1.0).contains(v)));
            assert!((c.iter().cloned().fold(0.0, f64::max) - 1.0).abs() < 1e-12);
        }
        assert!(cool[0] > cool[1] && cool[1] > cool[2]);
        assert!(hot[2] > hot[0]);
        assert!(
            warm.iter().all(|v| *v > 0.85),
            "6500 K is close to white: {warm:?}"
        );
        assert_eq!(
            blackbody_chromaticity(100.0),
            blackbody_chromaticity(1667.0)
        );
        assert_eq!(blackbody_chromaticity(1e6), blackbody_chromaticity(25000.0));
    }

    #[test]
    fn disk_pixel_brightens_with_g_and_is_black_when_cold() {
        let dim = disk_pixel(0.8, 6000.0, 6000.0);
        let unit = disk_pixel(1.0, 6000.0, 6000.0);
        let bright = disk_pixel(1.2, 6000.0, 6000.0);
        let lum = |p: Rgb| p.iter().map(|&c| c as u32).sum::<u32>();
        assert!(
            lum(dim) < lum(unit) && lum(unit) < lum(bright),
            "{dim:?} {unit:?} {bright:?}"
        );
        assert!(
            bright[2] > unit[2],
            "the approaching side is hotter, hence bluer"
        );
        assert_eq!(disk_pixel(1.0, 0.0, 6000.0), [0, 0, 0]);
        assert_eq!(disk_pixel(f64::NAN, 6000.0, 6000.0), [0, 0, 0]);
        assert_eq!(srgb_encode(0.0), 0);
        assert_eq!(srgb_encode(1.0), 255);
        assert_eq!(srgb_encode(f64::NAN), 0);
    }

    #[test]
    fn straight_ray_hits_the_plane_where_geometry_says() {
        let d = disk([0.0, 0.0, 1.0]);
        let hit = d
            .straight_ray_hit([30.0, 0.0, 3.0], normalize([-1.0, 0.0, -0.2]))
            .unwrap();
        assert!(hit[2].abs() < 1e-12 && (hit[0] - 15.0).abs() < 1e-12);
        assert!(
            d.straight_ray_hit([30.0, 0.0, 3.0], normalize([-1.0, 0.0, -1.0]))
                .is_none(),
            "inside r_in"
        );
        assert!(
            d.straight_ray_hit([30.0, 0.0, 3.0], [-1.0, 0.0, 0.0])
                .is_none(),
            "parallel"
        );
        assert!(
            d.straight_ray_hit([30.0, 0.0, 3.0], normalize([-1.0, 0.0, 0.2]))
                .is_none(),
            "behind"
        );
    }

    #[test]
    fn validation_rejects_unphysical_disks() {
        assert!(Disk::new([1.0, 0.0, 0.0], 5.9, 20.0, 6000.0, M, R_OBS).is_err());
        assert!(Disk::new([1.0, 0.0, 0.0], 12.0, 40.0, 6000.0, 2.0, 60.0).is_ok());
        assert!(Disk::new([1.0, 0.0, 0.0], 11.9, 40.0, 6000.0, 2.0, 60.0).is_err());
        assert!(Disk::new([1.0, 0.0, 0.0], 6.0, 30.0, 6000.0, M, R_OBS).is_err());
        assert!(Disk::new([1.0, 0.0, 0.0], 6.0, 6.0, 6000.0, M, R_OBS).is_err());
        assert!(Disk::new([1.0, 0.0, 0.0], 6.0, 20.0, 0.0, M, R_OBS).is_err());
        assert!(Disk::new([0.0, 0.0, 0.0], 6.0, 20.0, 6000.0, M, R_OBS).is_err());
        let d = Disk::new([0.0, 0.0, 2.0], 6.0, 20.0, 6000.0, M, R_OBS).unwrap();
        assert_eq!(d.normal, [0.0, 0.0, 1.0]);
    }
}
