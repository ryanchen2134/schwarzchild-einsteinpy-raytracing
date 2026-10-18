//! The fast CPU path: an equatorial, four-lane SIMD FANTASY integrator.
//!
//! Every pipeline ray is rotated into the equatorial plane before integration
//! ([`crate::initial_conditions`]), so `θ ≡ π/2` and `p_θ ≡ 0` exactly. The general
//! step in [`crate::integrator`] still evaluates `sin θ`, `cos θ` and the `θ`
//! derivative of the metric at every flow: 12 `sin` and 4 `cos` per step on
//! constants. This module specialises the step to the equator (`sin θ = 1`,
//! `cos θ = 0`, `p_θ = 0`), which leaves only arithmetic, and runs four rays per
//! instruction with [`wide::f64x4`]. Operation order matches the scalar step and
//! neither rustc nor `wide` contracts to FMA, so `t`, `r`, `φ`, `p_t`, `p_r` and `p_φ`
//! come out bit-identical to the scalar integrator; the only difference is the
//! `cos(π/2) ≈ 6e-17` kick the scalar code gives `p_θ`, which the tests bound.
//!
//! Rays are sorted by `|p_φ|` (the impact parameter, which sets how long a ray
//! orbits) before being grouped into lanes, so the four rays of a group finish at
//! similar steps. A ray that is not equatorial falls back to the scalar integrator.
//!
//! Disk crossings are detected lane-wise with the arithmetic of
//! [`crate::integrator::CrossingDetector`], in the same operation order, so the hit
//! radius and azimuth are bit-identical to the scalar integrator's. A lane without a
//! disk azimuth carries `NaN`, which fails every comparison and so never crosses.

use crate::integrator::{
    integrate_ray, DiskHit, ExitReason, IntegrationResult, IntegratorSettings, Ray,
};
use rayon::prelude::*;
use std::f64::consts::{FRAC_PI_2, PI};
use wide::{f64x4, CmpEq, CmpGe, CmpGt, CmpLe};

/// Schwarzschild radius in the integrator's units (lengths in units of `M`).
const RS: f64 = 2.0;
const LANES: usize = 4;

#[derive(Clone, Copy)]
struct Lanes {
    t1: f64x4,
    r1: f64x4,
    ph1: f64x4,
    pt1: f64x4,
    pr1: f64x4,
    pph1: f64x4,
    t2: f64x4,
    r2: f64x4,
    ph2: f64x4,
    pt2: f64x4,
    pr2: f64x4,
    pph2: f64x4,
}

/// `A(δ)`: kick `p1` with `∂H/∂q` at `(q1, p2)`, drift `q2` with `g(q1) p2`.
#[inline(always)]
fn flow_a(s: &mut Lanes, delta: f64) {
    let rs = f64x4::splat(RS);
    let r = s.r1;
    let denom = r - rs;
    let r2 = r * r;
    let r3 = r2 * r;
    // ∂g^{μμ}/∂r at θ = π/2: (rs/(r−rs)², rs/r², −2/r³ [θθ, unused], −2/r³ [φφ])
    let dh = f64x4::splat(0.5)
        * ((rs / (denom * denom)) * s.pt2 * s.pt2
            + (rs / r2) * s.pr2 * s.pr2
            + (f64x4::splat(-2.0) / r3) * s.pph2 * s.pph2);
    s.pr1 -= f64x4::splat(delta) * dh;
    let inv_fac = f64x4::splat(1.0) - rs / r;
    let d = f64x4::splat(delta);
    s.t2 += d * ((f64x4::splat(-1.0) / inv_fac) * s.pt2);
    s.r2 += d * (inv_fac * s.pr2);
    s.ph2 += d * ((f64x4::splat(1.0) / r2) * s.pph2);
}

/// `B(δ)`: kick `p2` with `∂H/∂q` at `(q2, p1)`, drift `q1` with `g(q2) p1`.
#[inline(always)]
fn flow_b(s: &mut Lanes, delta: f64) {
    let rs = f64x4::splat(RS);
    let r = s.r2;
    let denom = r - rs;
    let r2 = r * r;
    let r3 = r2 * r;
    let dh = f64x4::splat(0.5)
        * ((rs / (denom * denom)) * s.pt1 * s.pt1
            + (rs / r2) * s.pr1 * s.pr1
            + (f64x4::splat(-2.0) / r3) * s.pph1 * s.pph1);
    s.pr2 -= f64x4::splat(delta) * dh;
    let inv_fac = f64x4::splat(1.0) - rs / r;
    let d = f64x4::splat(delta);
    s.t1 += d * ((f64x4::splat(-1.0) / inv_fac) * s.pt1);
    s.r1 += d * (inv_fac * s.pr1);
    s.ph1 += d * ((f64x4::splat(1.0) / r2) * s.pph1);
}

#[inline(always)]
fn mix_pair(
    q1: &mut f64x4,
    p1: &mut f64x4,
    q2: &mut f64x4,
    p2: &mut f64x4,
    cos: f64x4,
    sin: f64x4,
) {
    let half = f64x4::splat(0.5);
    let q_sum = *q1 + *q2;
    let q_dif = *q1 - *q2;
    let p_sum = *p1 + *p2;
    let p_dif = *p1 - *p2;
    *q1 = half * (q_sum + q_dif * cos + p_dif * sin);
    *p1 = half * (p_sum + p_dif * cos - q_dif * sin);
    *q2 = half * (q_sum - q_dif * cos - p_dif * sin);
    *p2 = half * (p_sum - p_dif * cos + q_dif * sin);
}

#[inline(always)]
fn flow_mixed(s: &mut Lanes, cos: f64x4, sin: f64x4) {
    mix_pair(&mut s.t1, &mut s.pt1, &mut s.t2, &mut s.pt2, cos, sin);
    mix_pair(&mut s.r1, &mut s.pr1, &mut s.r2, &mut s.pr2, cos, sin);
    mix_pair(&mut s.ph1, &mut s.pph1, &mut s.ph2, &mut s.pph2, cos, sin);
}

#[inline(always)]
fn step(s: &mut Lanes, half_delta: f64, cos: f64x4, sin: f64x4) {
    flow_a(s, half_delta);
    flow_b(s, half_delta);
    flow_mixed(s, cos, sin);
    flow_b(s, half_delta);
    flow_a(s, half_delta);
}

/// Is a ray equatorial in the exact sense the fast path needs?
#[inline]
pub fn is_equatorial(q0: &[f64; 4], p0: &[f64; 4]) -> bool {
    q0[2] == FRAC_PI_2 && p0[2] == 0.0
}

/// Integrate up to four equatorial rays (already in units of `M`) together.
fn integrate_group(
    q0s: &[[f64; 4]; LANES],
    p0s: &[[f64; 4]; LANES],
    disk_phis: &[f64; LANES],
    settings: &IntegratorSettings,
    mass: f64,
) -> [IntegrationResult; LANES] {
    let delta = settings.delta / mass;
    let half = 0.5 * delta;
    let (sin_m, cos_m) = (2.0 * settings.omega * delta).sin_cos();
    let (cos_v, sin_v) = (f64x4::splat(cos_m), f64x4::splat(sin_m));
    let r_max = f64x4::splat(settings.r_max / mass);
    let r_cap = f64x4::splat(settings.r_capture / mass);
    let rs = f64x4::splat(RS);
    let zero = f64x4::splat(0.0);
    let one = f64x4::splat(1.0);
    let minus_one = f64x4::splat(-1.0);
    let pi = f64x4::splat(PI);
    // Disk detection: only when a band is set and some lane carries an azimuth.
    let disk = settings
        .disk
        .filter(|_| disk_phis.iter().any(|p| p.is_finite()))
        .map(|band| {
            (
                f64x4::from(*disk_phis),
                f64x4::splat(band.r_in / mass),
                f64x4::splat(band.r_out / mass),
            )
        });
    let mut hits: [Option<DiskHit>; LANES] = [None; LANES];
    let mut crossings = [0u32; LANES];

    let col = |k: usize, src: &[[f64; 4]; LANES]| {
        f64x4::from([src[0][k], src[1][k], src[2][k], src[3][k]])
    };
    let mut s = Lanes {
        t1: col(0, q0s),
        r1: col(1, q0s),
        ph1: col(3, q0s),
        pt1: col(0, p0s),
        pr1: col(1, p0s),
        pph1: col(3, p0s),
        t2: col(0, q0s),
        r2: col(1, q0s),
        ph2: col(3, q0s),
        pt2: col(0, p0s),
        pr2: col(1, p0s),
        pph2: col(3, p0s),
    };

    let mut active: u32 = 0b1111;
    let mut exits = [ExitReason::StepLimit; LANES];
    let mut steps_taken = [settings.steps; LANES];
    let mut snapshot = s;
    let mut n_steps = 0usize;

    // Per-lane exit masks in the scalar integrator's priority: diverged, captured,
    // escaped; the fourth mask (disk) is only raised after a step.
    let classify = |s: &Lanes| -> (u32, u32, u32, u32) {
        let diverged = (!s.r1.is_finite() | s.r1.cmp_le(zero) | !s.r2.is_finite() | s.r2.cmp_le(rs))
            .move_mask() as u32;
        let captured = s.r1.cmp_le(r_cap).move_mask() as u32 & !diverged;
        let escaped = s.r1.cmp_ge(r_max).move_mask() as u32 & !diverged & !captured;
        (captured, escaped, diverged, 0)
    };

    let mut record =
        |s: &Lanes,
         snapshot: &mut Lanes,
         active: &mut u32,
         n_steps: usize,
         (captured, escaped, diverged, disk): (u32, u32, u32, u32)| {
            let any = (captured | escaped | diverged | disk) & *active;
            if any == 0 {
                return;
            }
            for lane in 0..LANES {
                let bit = 1u32 << lane;
                if any & bit == 0 {
                    continue;
                }
                exits[lane] = if diverged & bit != 0 {
                    ExitReason::Diverged
                } else if captured & bit != 0 {
                    ExitReason::Captured
                } else if escaped & bit != 0 {
                    ExitReason::Escaped
                } else {
                    ExitReason::Disk
                };
                steps_taken[lane] = n_steps;
                copy_lane(snapshot, s, lane);
                *active &= !bit;
            }
        };

    for _ in 0..settings.steps {
        record(&s, &mut snapshot, &mut active, n_steps, classify(&s));
        if active == 0 {
            break;
        }
        let (r_prev, ph_prev) = (s.r1, s.ph1);
        step(&mut s, half, cos_v, sin_v);
        n_steps += 1;
        if let Some((phi_d, r_in, r_out)) = disk {
            if n_steps >= 2 {
                // CrossingDetector::check, lane-wise and in the same operation order.
                let k_prev = ((ph_prev - phi_d) / pi).floor();
                let k = ((s.ph1 - phi_d) / pi).floor();
                let d = k - k_prev;
                let crossed = (d.cmp_eq(one) | d.cmp_eq(minus_one)).move_mask() as u32 & active;
                if crossed != 0 {
                    let k_boundary = d.cmp_gt(zero).blend(k, k_prev);
                    let phi_c = phi_d + k_boundary * pi;
                    let frac = (phi_c - ph_prev) / (s.ph1 - ph_prev);
                    let r_hit = r_prev + (s.r1 - r_prev) * frac;
                    let in_band =
                        (r_hit.cmp_ge(r_in) & r_hit.cmp_le(r_out)).move_mask() as u32 & crossed;
                    for lane in 0..LANES {
                        let bit = 1u32 << lane;
                        if crossed & bit == 0 {
                            continue;
                        }
                        crossings[lane] += 1;
                        if in_band & bit != 0 {
                            hits[lane] = Some(DiskHit {
                                r: r_hit.as_array_ref()[lane],
                                phi: phi_c.as_array_ref()[lane],
                            });
                        }
                    }
                    record(&s, &mut snapshot, &mut active, n_steps, (0, 0, 0, in_band));
                    if active == 0 {
                        break;
                    }
                }
            }
        }
    }
    if active != 0 {
        // Lanes that ran out of steps: a boundary crossed on the last step still counts.
        let masks = classify(&s);
        record(&s, &mut snapshot, &mut active, n_steps, masks);
        for (lane, steps) in steps_taken.iter_mut().enumerate() {
            if active & (1 << lane) != 0 {
                *steps = n_steps;
                copy_lane(&mut snapshot, &s, lane);
            }
        }
    }

    let out = |v: f64x4, lane: usize| v.as_array_ref()[lane];
    std::array::from_fn(|lane| IntegrationResult {
        q: [
            out(snapshot.t1, lane) * mass,
            out(snapshot.r1, lane) * mass,
            FRAC_PI_2,
            out(snapshot.ph1, lane),
        ],
        p: [
            out(snapshot.pt1, lane),
            out(snapshot.pr1, lane),
            0.0,
            out(snapshot.pph1, lane) * mass,
        ],
        exit: exits[lane],
        n_steps: steps_taken[lane],
        disk_hit: hits[lane].map(|h| DiskHit {
            r: h.r * mass,
            phi: h.phi,
        }),
        plane_crossings: crossings[lane],
    })
}

#[inline]
fn copy_lane(dst: &mut Lanes, src: &Lanes, lane: usize) {
    macro_rules! cp {
        ($($f:ident),*) => { $( dst.$f.as_array_mut()[lane] = src.$f.as_array_ref()[lane]; )* };
    }
    cp!(t1, r1, ph1, pt1, pr1, pph1, t2, r2, ph2, pt2, pr2, pph2);
}

/// Integrate many rays with the SIMD equatorial step, four at a time, in parallel.
/// Non-equatorial rays go through the scalar integrator. `on_ray_done` is called
/// exactly once per input ray: scalar rays fire it as each finishes, SIMD rays once
/// per real ray in the group when the group finishes (padding copies of a short
/// final group are not counted).
pub fn integrate_batch_fast<F>(
    rays: &[Ray],
    rs: f64,
    settings: &IntegratorSettings,
    on_ray_done: F,
) -> Vec<IntegrationResult>
where
    F: Fn() + Sync,
{
    let mass = rs / RS;
    let n = rays.len();
    let mut results: Vec<Option<IntegrationResult>> = vec![None; n];

    let (fast, slow): (Vec<usize>, Vec<usize>) =
        (0..n).partition(|&i| is_equatorial(&rays[i].q0, &rays[i].p0));
    if !slow.is_empty() {
        let scalar: Vec<(usize, IntegrationResult)> = slow
            .par_iter()
            .map(|&i| {
                let r = integrate_ray(&rays[i], rs, settings);
                on_ray_done();
                (i, r)
            })
            .collect();
        for (i, r) in scalar {
            results[i] = Some(r);
        }
    }

    // Group rays of similar impact parameter so the lanes finish together.
    let mut order = fast;
    // A total order keeps the sort well defined even for a NaN momentum (sorted last).
    order.sort_unstable_by(|&a, &b| rays[a].p0[3].abs().total_cmp(&rays[b].p0[3].abs()));

    let scaled = |i: usize| {
        let (mut q, mut p) = (rays[i].q0, rays[i].p0);
        q[0] /= mass;
        q[1] /= mass;
        p[2] /= mass;
        p[3] /= mass;
        (q, p)
    };
    let groups: Vec<(Vec<usize>, [IntegrationResult; LANES])> = order
        .par_chunks(LANES)
        .map(|chunk| {
            let mut q0s = [[0.0; 4]; LANES];
            let mut p0s = [[0.0; 4]; LANES];
            let mut phis = [f64::NAN; LANES];
            for lane in 0..LANES {
                // Pad a short final group by repeating its last ray; the copies are dropped.
                let i = chunk[lane.min(chunk.len() - 1)];
                let (q, p) = scaled(i);
                q0s[lane] = q;
                p0s[lane] = p;
                phis[lane] = rays[i].disk_phi;
            }
            let res = integrate_group(&q0s, &p0s, &phis, settings, mass);
            for _ in chunk {
                on_ray_done();
            }
            (chunk.to_vec(), res)
        })
        .collect();
    for (idx, res) in groups {
        for (lane, &i) in idx.iter().enumerate() {
            results[i] = Some(res[lane]);
        }
    }
    results
        .into_iter()
        .map(|r| r.expect("every ray integrated"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::Camera;
    use crate::disk::Disk;
    use crate::initial_conditions::initial_conditions;
    use crate::integrator::DiskBand;

    fn pipeline_rays(size: usize, mass: f64) -> Vec<Ray> {
        let cam = Camera::new([30.0 * mass, 0.0, 0.0], 80f64.to_radians(), size, size).unwrap();
        (0..cam.n_pixels())
            .map(|flat| {
                let (i, j) = cam.pixel_of(flat);
                let init = initial_conditions(cam.observer, cam.pixel_position(i, j), mass);
                Ray::new(init.q0, init.p0)
            })
            .collect()
    }

    /// The same rays with the per-ray crossing azimuth of an oblique disk.
    fn disk_rays(size: usize, mass: f64, disk: &Disk) -> Vec<Ray> {
        let cam = Camera::new([30.0 * mass, 0.0, 0.0], 80f64.to_radians(), size, size).unwrap();
        (0..cam.n_pixels())
            .map(|flat| {
                let (i, j) = cam.pixel_of(flat);
                let init = initial_conditions(cam.observer, cam.pixel_position(i, j), mass);
                Ray::with_disk(init.q0, init.p0, disk.ray_frame(init.beta).phi_d)
            })
            .collect()
    }

    /// Four distinct equatorial rays that exit at different steps.
    fn distinct_rays() -> Vec<Ray> {
        [0.5, 4.0, 5.2, 8.0]
            .iter()
            .map(|&y| {
                let init = initial_conditions([30.0, 0.0, 0.0], [24.0, y, 0.0], 1.0);
                Ray::new(init.q0, init.p0)
            })
            .collect()
    }

    fn settings() -> IntegratorSettings {
        IntegratorSettings::new(20_000, 0.01, 0.01, 31.0, 2.0)
    }

    fn integrate_all(rays: &[Ray], rs: f64, s: &IntegratorSettings) -> Vec<IntegrationResult> {
        rays.iter().map(|r| integrate_ray(r, rs, s)).collect()
    }

    /// Exits and step counts must match; `t, r, θ, φ, p_t, p_r, p_φ` must be bit-identical
    /// (same operation order, no FMA); `p_θ` may carry the scalar step's `cos(π/2)` kicks.
    fn assert_same(fast: &[IntegrationResult], exact: &[IntegrationResult]) -> f64 {
        let mut worst_p_theta = 0.0f64;
        for (idx, (f, e)) in fast.iter().zip(exact).enumerate() {
            assert_eq!(f.exit, e.exit, "ray {idx}");
            assert_eq!(f.n_steps, e.n_steps, "ray {idx}");
            assert_eq!(f.plane_crossings, e.plane_crossings, "ray {idx}");
            match (f.disk_hit, e.disk_hit) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    assert_eq!(
                        a.r.to_bits(),
                        b.r.to_bits(),
                        "ray {idx} hit r: {} vs {}",
                        a.r,
                        b.r
                    );
                    assert_eq!(a.phi.to_bits(), b.phi.to_bits(), "ray {idx} hit φ");
                }
                (a, b) => panic!("ray {idx}: hit {a:?} vs {b:?}"),
            }
            for k in [0, 1, 2, 3] {
                assert_eq!(
                    f.q[k].to_bits(),
                    e.q[k].to_bits(),
                    "ray {idx} q[{k}]: {} vs {}",
                    f.q[k],
                    e.q[k]
                );
            }
            for k in [0, 1, 3] {
                assert_eq!(
                    f.p[k].to_bits(),
                    e.p[k].to_bits(),
                    "ray {idx} p[{k}]: {} vs {}",
                    f.p[k],
                    e.p[k]
                );
            }
            worst_p_theta = worst_p_theta.max((f.p[2] - e.p[2]).abs());
        }
        assert!(worst_p_theta < 1e-12, "p_θ {worst_p_theta}");
        worst_p_theta
    }

    #[test]
    fn fast_path_is_bit_identical_to_the_scalar_integrator() {
        // 169 rays: an odd size, so a lane permutation or a wrong move_mask bit order
        // would pair different rays and fail; the final group holds one ray.
        let rays = pipeline_rays(13, 1.0);
        let exact = integrate_all(&rays, 2.0, &settings());
        let fast = integrate_batch_fast(&rays, 2.0, &settings(), || {});
        let worst = assert_same(&fast, &exact);
        eprintln!("fast vs exact: max |Δp_θ| = {worst:e}, everything else bit-identical");
        assert!(exact.iter().any(|r| r.exit == ExitReason::Captured));
        assert!(exact.iter().any(|r| r.exit == ExitReason::Escaped));
    }

    #[test]
    fn disk_hits_are_bit_identical_to_the_scalar_integrator() {
        let disk = Disk::new(
            Disk::normal_from_angles(55f64.to_radians(), 30f64.to_radians()),
            6.0,
            20.0,
            6000.0,
            1.0,
            30.0,
        )
        .unwrap();
        let s = settings().with_disk(DiskBand {
            r_in: 6.0,
            r_out: 20.0,
        });
        let rays = disk_rays(13, 1.0, &disk);
        let exact = integrate_all(&rays, 2.0, &s);
        let fast = integrate_batch_fast(&rays, 2.0, &s, || {});
        assert_same(&fast, &exact);
        let hits = exact.iter().filter(|r| r.exit == ExitReason::Disk).count();
        let passed = exact
            .iter()
            .filter(|r| r.exit != ExitReason::Disk && r.plane_crossings > 0)
            .count();
        assert!(
            hits > 20 && passed > 0,
            "{hits} hits, {passed} rays crossed outside the band"
        );
        // Mixed lanes: some rays detect, some do not; the detector must not leak across lanes.
        let mut mixed = rays.clone();
        for (k, r) in mixed.iter_mut().enumerate() {
            if k % 3 == 0 {
                r.disk_phi = f64::NAN;
            }
        }
        let exact = integrate_all(&mixed, 2.0, &s);
        let fast = integrate_batch_fast(&mixed, 2.0, &s, || {});
        assert_same(&fast, &exact);
        // With a disk band but no azimuths the results equal the disk-off run.
        let plain = pipeline_rays(13, 1.0);
        let off = integrate_batch_fast(&plain, 2.0, &settings(), || {});
        let on = integrate_batch_fast(&plain, 2.0, &s, || {});
        assert_same(&on, &off);
    }

    #[test]
    fn disk_hits_rescale_with_mass() {
        let disk = Disk::new(
            Disk::normal_from_angles(55f64.to_radians(), 30f64.to_radians()),
            15.0,
            50.0,
            6000.0,
            2.5,
            75.0,
        )
        .unwrap();
        let s = IntegratorSettings::new(20_000, 0.025, 0.01, 77.5, 5.0).with_disk(DiskBand {
            r_in: 15.0,
            r_out: 50.0,
        });
        let rays = disk_rays(7, 2.5, &disk);
        let exact = integrate_all(&rays, 5.0, &s);
        let fast = integrate_batch_fast(&rays, 5.0, &s, || {});
        assert_same(&fast, &exact);
        assert!(exact.iter().any(|r| r.exit == ExitReason::Disk));
        for r in exact.iter().filter_map(|r| r.disk_hit) {
            assert!(r.r >= 15.0 && r.r <= 50.0);
        }
    }

    #[test]
    fn one_group_of_distinct_rays_keeps_its_lanes_straight() {
        let rays = distinct_rays();
        let exact = integrate_all(&rays, 2.0, &settings());
        let steps: Vec<usize> = exact.iter().map(|r| r.n_steps).collect();
        assert_eq!(
            steps
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            4,
            "the four rays must exit at four different steps: {steps:?}"
        );
        assert_eq!(exact[0].exit, ExitReason::Captured);
        assert_eq!(exact[3].exit, ExitReason::Escaped);
        let fast = integrate_batch_fast(&rays, 2.0, &settings(), || {});
        assert_same(&fast, &exact);
    }

    #[test]
    fn fast_path_rescales_mass_like_the_scalar_integrator() {
        let rays = pipeline_rays(7, 2.5);
        let settings = IntegratorSettings::new(20_000, 0.025, 0.01, 77.5, 5.0);
        let exact = integrate_all(&rays, 5.0, &settings);
        let fast = integrate_batch_fast(&rays, 5.0, &settings, || {});
        assert_same(&fast, &exact);
    }

    #[test]
    fn non_equatorial_rays_fall_back_to_the_scalar_integrator() {
        let mut rays = pipeline_rays(4, 1.0);
        rays[5].q0[2] = 1.4; // tilt one ray out of the equator
        rays[5].p0[2] = 0.3;
        let settings = IntegratorSettings::new(5_000, 0.01, 0.01, 31.0, 2.0);
        let fast = integrate_batch_fast(&rays, 2.0, &settings, || {});
        let exact = integrate_ray(&rays[5], 2.0, &settings);
        assert_eq!(fast[5].q, exact.q);
        assert_eq!(fast[5].exit, exact.exit);
    }

    #[test]
    fn step_limit_and_last_step_crossing_match_the_scalar_integrator() {
        let rays = distinct_rays();
        let full = settings();
        let exact_full = integrate_all(&rays, 2.0, &full);
        let n = exact_full[3].n_steps;
        for steps in [n, n - 1, 10] {
            let s = IntegratorSettings { steps, ..full };
            let exact = integrate_all(&rays, 2.0, &s);
            let fast = integrate_batch_fast(&rays, 2.0, &s, || {});
            assert_same(&fast, &exact);
        }
    }

    #[test]
    fn a_nan_lane_is_diverged_and_leaves_the_other_lanes_untouched() {
        let mut rays = distinct_rays();
        let clean = integrate_batch_fast(&rays, 2.0, &settings(), || {});
        rays[1].q0[1] = f64::NAN;
        let fast = integrate_batch_fast(&rays, 2.0, &settings(), || {});
        assert_eq!(fast[1].exit, ExitReason::Diverged);
        assert_eq!(fast[1].n_steps, 0);
        for k in [0, 2, 3] {
            assert_eq!(fast[k].exit, clean[k].exit);
            assert_eq!(fast[k].n_steps, clean[k].n_steps);
            assert_eq!(fast[k].q, clean[k].q);
            assert_eq!(fast[k].p, clean[k].p);
        }
        let exact = integrate_ray(&rays[1], 2.0, &settings());
        assert_eq!(exact.exit, ExitReason::Diverged);
    }

    #[test]
    fn a_non_positive_radius_is_diverged_not_captured() {
        let mut rays = distinct_rays();
        rays[2].q0[1] = -1.0;
        let fast = integrate_batch_fast(&rays, 2.0, &settings(), || {});
        assert_eq!(fast[2].exit, ExitReason::Diverged);
        assert_eq!(fast[2].n_steps, 0);
        let exact = integrate_ray(&rays[2], 2.0, &settings());
        assert_eq!(exact.exit, ExitReason::Diverged);
    }

    #[test]
    fn progress_callback_fires_once_per_ray() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        for size in [1, 2, 3, 5] {
            let rays = pipeline_rays(size, 1.0);
            let count = AtomicUsize::new(0);
            let settings = IntegratorSettings::new(100, 0.01, 0.01, 31.0, 2.0);
            let _ = integrate_batch_fast(&rays, 2.0, &settings, || {
                count.fetch_add(1, Ordering::Relaxed);
            });
            assert_eq!(count.load(Ordering::Relaxed), size * size);
        }
    }
}
