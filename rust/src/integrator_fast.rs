//! The fast CPU path: an equatorial, four-lane SIMD FANTASY integrator.
//!
//! Every pipeline ray is rotated into the equatorial plane before integration
//! ([`crate::initial_conditions`]), so `θ ≡ π/2` and `p_θ ≡ 0` exactly. The general
//! step in [`crate::integrator`] still evaluates `sin θ`, `cos θ` and the `θ`
//! derivative of the metric at every flow: 12 `sin` and 4 `cos` per step on
//! constants. This module specialises the step to the equator (`sin θ = 1`,
//! `cos θ = 0`, `p_θ = 0`), which leaves only arithmetic, and runs four rays per
//! instruction with [`wide::f64x4`]. Operation order matches the scalar step, so
//! the results agree with it to rounding (the tests measure the difference); the
//! only dropped terms are the `cos(π/2) ≈ 6e-17` kicks the scalar code gives `p_θ`.
//!
//! Rays are sorted by `|p_φ|` (the impact parameter, which sets how long a ray
//! orbits) before being grouped into lanes, so the four rays of a group finish at
//! similar steps. A ray that is not equatorial falls back to the scalar integrator.

use crate::integrator::{integrate, ExitReason, IntegrationResult, IntegratorSettings};
use rayon::prelude::*;
use std::f64::consts::FRAC_PI_2;
use wide::{f64x4, CmpGe, CmpLe};

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

    let classify = |s: &Lanes| -> (u32, u32, u32) {
        let captured = s.r1.cmp_le(r_cap).move_mask() as u32;
        let escaped = s.r1.cmp_ge(r_max).move_mask() as u32;
        let diverged = (!s.r1.is_finite() | s.r1.cmp_le(zero) | !s.r2.is_finite() | s.r2.cmp_le(rs))
            .move_mask() as u32;
        (captured, escaped, diverged)
    };

    let mut record = |s: &Lanes,
                      snapshot: &mut Lanes,
                      active: &mut u32,
                      n_steps: usize,
                      (captured, escaped, diverged): (u32, u32, u32)| {
        let any = (captured | escaped | diverged) & *active;
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
            } else {
                ExitReason::Escaped
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
        step(&mut s, half, cos_v, sin_v);
        n_steps += 1;
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
/// Non-equatorial rays go through the scalar integrator. `on_ray_done` is called once
/// per ray (for a SIMD group, four times when the group finishes).
pub fn integrate_batch_fast<F>(
    rays: &[([f64; 4], [f64; 4])],
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
        (0..n).partition(|&i| is_equatorial(&rays[i].0, &rays[i].1));
    if !slow.is_empty() {
        let scalar: Vec<(usize, IntegrationResult)> = slow
            .par_iter()
            .map(|&i| {
                let r = integrate(rays[i].0, rays[i].1, rs, settings);
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
    order.sort_unstable_by(|&a, &b| {
        let ka = rays[a].1[3].abs();
        let kb = rays[b].1[3].abs();
        ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal)
    });

    let scaled = |i: usize| {
        let (mut q, mut p) = rays[i];
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
            for lane in 0..LANES {
                // Pad a short final group by repeating its last ray; the copies are dropped.
                let i = chunk[lane.min(chunk.len() - 1)];
                let (q, p) = scaled(i);
                q0s[lane] = q;
                p0s[lane] = p;
            }
            let res = integrate_group(&q0s, &p0s, settings, mass);
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
    use crate::initial_conditions::initial_conditions;

    fn pipeline_rays(size: usize, mass: f64) -> Vec<([f64; 4], [f64; 4])> {
        let cam = Camera::new([30.0 * mass, 0.0, 0.0], 80f64.to_radians(), size, size).unwrap();
        (0..cam.n_pixels())
            .map(|flat| {
                let (i, j) = cam.pixel_of(flat);
                let init = initial_conditions(cam.observer, cam.pixel_position(i, j), mass);
                (init.q0, init.p0)
            })
            .collect()
    }

    fn compare(fast: &[IntegrationResult], exact: &[IntegrationResult]) -> f64 {
        let mut worst = 0.0f64;
        for (f, e) in fast.iter().zip(exact) {
            assert_eq!(f.exit, e.exit);
            assert_eq!(f.n_steps, e.n_steps);
            for k in 0..4 {
                worst = worst
                    .max((f.q[k] - e.q[k]).abs())
                    .max((f.p[k] - e.p[k]).abs());
            }
        }
        worst
    }

    #[test]
    fn fast_path_matches_the_scalar_integrator() {
        let rays = pipeline_rays(13, 1.0); // 169 rays: a partial final group of 1
        let settings = IntegratorSettings::new(20_000, 0.01, 0.01, 31.0, 2.0);
        let exact: Vec<_> = rays
            .iter()
            .map(|(q, p)| integrate(*q, *p, 2.0, &settings))
            .collect();
        let fast = integrate_batch_fast(&rays, 2.0, &settings, || {});
        let worst = compare(&fast, &exact);
        eprintln!("fast vs exact: max |Δ| = {worst:e}");
        assert!(worst < 1e-12, "{worst}");
        assert!(exact.iter().any(|r| r.exit == ExitReason::Captured));
        assert!(exact.iter().any(|r| r.exit == ExitReason::Escaped));
    }

    #[test]
    fn fast_path_rescales_mass_like_the_scalar_integrator() {
        let rays = pipeline_rays(6, 2.5);
        let settings = IntegratorSettings::new(20_000, 0.025, 0.01, 77.5, 5.0);
        let exact: Vec<_> = rays
            .iter()
            .map(|(q, p)| integrate(*q, *p, 5.0, &settings))
            .collect();
        let fast = integrate_batch_fast(&rays, 5.0, &settings, || {});
        assert!(compare(&fast, &exact) < 1e-10);
    }

    #[test]
    fn non_equatorial_rays_fall_back_to_the_scalar_integrator() {
        let mut rays = pipeline_rays(4, 1.0);
        rays[5].0[2] = 1.4; // tilt one ray out of the equator
        rays[5].1[2] = 0.3;
        let settings = IntegratorSettings::new(5_000, 0.01, 0.01, 31.0, 2.0);
        let fast = integrate_batch_fast(&rays, 2.0, &settings, || {});
        let exact = integrate(rays[5].0, rays[5].1, 2.0, &settings);
        assert_eq!(fast[5].q, exact.q);
        assert_eq!(fast[5].exit, exact.exit);
    }

    #[test]
    fn step_limit_and_last_step_crossing_match_the_scalar_integrator() {
        let rays = pipeline_rays(4, 1.0);
        let full = IntegratorSettings::new(20_000, 0.01, 0.01, 31.0, 2.0);
        let exact_full: Vec<_> = rays
            .iter()
            .map(|(q, p)| integrate(*q, *p, 2.0, &full))
            .collect();
        let n = exact_full[0].n_steps;
        for steps in [n, n - 1, 10] {
            let s = IntegratorSettings { steps, ..full };
            let exact: Vec<_> = rays
                .iter()
                .map(|(q, p)| integrate(*q, *p, 2.0, &s))
                .collect();
            let fast = integrate_batch_fast(&rays, 2.0, &s, || {});
            assert!(compare(&fast, &exact) < 1e-12, "steps {steps}");
        }
    }

    #[test]
    fn progress_callback_fires_once_per_ray() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let rays = pipeline_rays(5, 1.0);
        let count = AtomicUsize::new(0);
        let settings = IntegratorSettings::new(100, 0.01, 0.01, 31.0, 2.0);
        let _ = integrate_batch_fast(&rays, 2.0, &settings, || {
            count.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(count.load(Ordering::Relaxed), 25);
    }
}
