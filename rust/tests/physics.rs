//! Checks against closed-form Schwarzschild results, independent of the Python code.

use schwarzschild_rt::initial_conditions::{equatorial_momentum, initial_conditions, null_p_t};
use schwarzschild_rt::integrator::{integrate, ExitReason, IntegratorSettings};
use schwarzschild_rt::schwarzschild::BlackHole;
use std::f64::consts::PI;

/// Observer radius in units of `M`.
const R_OBS_PER_M: f64 = 30.0;

fn settings_for(bh: &BlackHole, steps: usize) -> IntegratorSettings {
    IntegratorSettings::new(steps, 0.01 * bh.mass, 0.01, 31.0 * bh.mass, bh.rs())
}

fn escapes(p_r: f64, p_ph: f64, bh: &BlackHole, settings: &IntegratorSettings) -> bool {
    let r_obs = R_OBS_PER_M * bh.mass;
    let f = bh.lapse(r_obs);
    let p_t = null_p_t(f, r_obs, PI / 2.0, p_r, 0.0, p_ph);
    let res = integrate(
        [0.0, r_obs, PI / 2.0, 0.0],
        [p_t, p_r, 0.0, p_ph],
        bh.rs(),
        settings,
    );
    res.exit == ExitReason::Escaped
}

/// Bisect the angle at which rays stop escaping, for a given momentum construction.
fn shadow_edge(bh: &BlackHole, momentum: impl Fn(f64) -> (f64, f64)) -> f64 {
    let settings = settings_for(bh, 100_000);
    let (mut lo, mut hi) = (0.10, 0.25);
    let (plo, phi_) = (momentum(lo), momentum(hi));
    assert!(!escapes(plo.0, plo.1, bh, &settings) && escapes(phi_.0, phi_.1, bh, &settings));
    for _ in 0..24 {
        let mid = 0.5 * (lo + hi);
        let (p_r, p_ph) = momentum(mid);
        if escapes(p_r, p_ph, bh, &settings) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

fn covariant_edge(bh: &BlackHole) -> f64 {
    let r_obs = R_OBS_PER_M * bh.mass;
    shadow_edge(bh, |alpha| {
        let (p_r, _, p_ph) = equatorial_momentum(alpha, r_obs, bh.mass);
        (p_r, p_ph)
    })
}

/// With the covariant momentum the integrated shadow edge sits at the analytic
/// critical angle `sin α_c = (3√3 M / r) √(1 − 2M/r)` (measured residual ≈5e-6 rad;
/// the 24-step bisection resolves ≈9e-9 rad).
#[test]
fn shadow_edge_matches_critical_angle() {
    let bh = BlackHole::new(1.0);
    let edge = covariant_edge(&bh);
    let expect = bh.capture_angle(R_OBS_PER_M);
    eprintln!("integrated shadow edge {edge:.6} rad, analytic {expect:.6} rad");
    assert!(
        (edge - expect).abs() < 1e-4,
        "edge {edge} vs analytic {expect}"
    );

    // The test can see sub-percent momentum errors: a 1 % error in p_φ moves the edge
    // by more than the tolerance.
    let r_obs = R_OBS_PER_M;
    let perturbed = shadow_edge(&bh, |alpha| {
        let (p_r, _, p_ph) = equatorial_momentum(alpha, r_obs, bh.mass);
        (p_r, p_ph * 1.01)
    });
    assert!(
        (perturbed - expect).abs() > 1e-4,
        "perturbed edge {perturbed}"
    );
}

/// The Python construction `p_r = n_r̂ √f` (instead of `n_r̂ / √f`) stretches every
/// angle by `1/f`, so its shadow edge lands at `atan(f · tan α_c)`: 6.6 % too small at
/// `r = 30`. This pins the size of the discrepancy the port removes.
#[test]
fn python_momentum_convention_shrinks_the_shadow_by_the_lapse() {
    let bh = BlackHole::new(1.0);
    let f = bh.lapse(R_OBS_PER_M);
    let edge = shadow_edge(&bh, |alpha| {
        let f_sqrt = f.sqrt();
        (-alpha.cos() * f_sqrt, alpha.sin() * R_OBS_PER_M)
    });
    let expect = (f * bh.capture_angle(R_OBS_PER_M).tan()).atan();
    eprintln!("python-convention shadow edge {edge:.6} rad, predicted {expect:.6} rad");
    assert!(
        (edge - expect).abs() < 1e-4,
        "edge {edge} vs predicted {expect}"
    );
    assert!(expect < 0.95 * bh.capture_angle(R_OBS_PER_M));
}

/// The integrator rescales every ray to units of `M`, so the shadow edge measured at
/// any mass equals the `M = 1` edge and the analytic angle. (Without the rescaling the
/// FANTASY coupling grows as `ω M²` and the edge at `M = 2.5` sat at 0.18 rad, at
/// `M = 5` the copies lost each other entirely.) Pins README difference 5 as well: the
/// Python's hard-coded `r_s = 2` in the metric derivative puts the `M = 0.5` edge at
/// 0.23 rad.
#[test]
fn shadow_edge_is_the_same_at_every_mass() {
    let unit = covariant_edge(&BlackHole::new(1.0));
    for mass in [0.5, 2.0, 2.5, 5.0] {
        let bh = BlackHole::new(mass);
        let edge = covariant_edge(&bh);
        let expect = bh.capture_angle(R_OBS_PER_M * mass);
        eprintln!("M = {mass}: edge {edge:.6}, analytic {expect:.6}");
        assert!(
            (edge - expect).abs() < 1e-4,
            "M = {mass}: {edge} vs {expect}"
        );
        assert!(
            (edge - unit).abs() < 1e-6,
            "M = {mass}: {edge} vs M=1 {unit}"
        );
    }
}

/// A near-critical ray integrated at masses 0.5, 2, 2.5 and 5 reproduces the `M = 1`
/// trajectory (lengths rescaled) to rounding, including its exit and step count.
#[test]
fn near_critical_ray_is_scale_invariant() {
    let alpha = 1.05 * BlackHole::new(1.0).capture_angle(R_OBS_PER_M);
    let reference = {
        let bh = BlackHole::new(1.0);
        let (p_r, p_th, p_ph) = equatorial_momentum(alpha, R_OBS_PER_M, 1.0);
        let p_t = null_p_t(
            bh.lapse(R_OBS_PER_M),
            R_OBS_PER_M,
            PI / 2.0,
            p_r,
            p_th,
            p_ph,
        );
        integrate(
            [0.0, R_OBS_PER_M, PI / 2.0, 0.0],
            [p_t, p_r, p_th, p_ph],
            bh.rs(),
            &settings_for(&bh, 100_000),
        )
    };
    assert_eq!(reference.exit, ExitReason::Escaped);
    for mass in [0.5, 2.0, 2.5, 5.0] {
        let bh = BlackHole::new(mass);
        let r_obs = R_OBS_PER_M * mass;
        let (p_r, p_th, p_ph) = equatorial_momentum(alpha, r_obs, mass);
        let p_t = null_p_t(bh.lapse(r_obs), r_obs, PI / 2.0, p_r, p_th, p_ph);
        let res = integrate(
            [0.0, r_obs, PI / 2.0, 0.0],
            [p_t, p_r, p_th, p_ph],
            bh.rs(),
            &settings_for(&bh, 100_000),
        );
        assert_eq!(res.exit, reference.exit, "M = {mass}");
        assert_eq!(res.n_steps, reference.n_steps, "M = {mass}");
        assert!(
            (res.q[1] / mass - reference.q[1]).abs() < 1e-9,
            "M = {mass}: r"
        );
        assert!(
            (res.q[0] / mass - reference.q[0]).abs() < 1e-9,
            "M = {mass}: t"
        );
        assert!((res.q[3] - reference.q[3]).abs() < 1e-9, "M = {mass}: φ");
    }
}

/// The full pipeline's initial conditions scale with the mass too.
#[test]
fn pixel_rays_scale_with_mass() {
    let a = initial_conditions([30.0, 0.0, 0.0], [24.0, 1.5, -0.7], 1.0);
    let b = initial_conditions([90.0, 0.0, 0.0], [72.0, 4.5, -2.1], 3.0);
    assert!((a.alpha0 - b.alpha0).abs() < 1e-14);
    assert!((a.beta - b.beta).abs() < 1e-14);
    assert!((a.p0[0] - b.p0[0]).abs() < 1e-14, "p_t is dimensionless");
    assert!((a.p0[1] - b.p0[1]).abs() < 1e-14, "p_r is dimensionless");
    assert!(
        (3.0 * a.p0[3] - b.p0[3]).abs() < 1e-12,
        "p_φ carries one length"
    );
}
