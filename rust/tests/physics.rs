//! Checks against closed-form Schwarzschild results, independent of the Python code.

use schwarzschild_rt::initial_conditions::{equatorial_momentum, null_p_t};
use schwarzschild_rt::integrator::{integrate, ExitReason, IntegratorSettings};
use schwarzschild_rt::schwarzschild::BlackHole;
use std::f64::consts::PI;

const R_OBS: f64 = 30.0;

fn escapes(p_r: f64, p_ph: f64, bh: &BlackHole, settings: &IntegratorSettings) -> bool {
    let f = bh.lapse(R_OBS);
    let p_t = null_p_t(f, R_OBS, PI / 2.0, p_r, 0.0, p_ph);
    let res = integrate(
        [0.0, R_OBS, PI / 2.0, 0.0],
        [p_t, p_r, 0.0, p_ph],
        bh.rs(),
        settings,
    );
    res.exit == ExitReason::Escaped
}

/// Bisect the angle at which rays stop escaping, for a given momentum construction.
fn shadow_edge(bh: &BlackHole, momentum: impl Fn(f64) -> (f64, f64)) -> f64 {
    let settings = IntegratorSettings::new(100_000, 0.01, 0.01, 31.0, bh.rs());
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

/// With the covariant momentum the integrated shadow edge sits at the analytic
/// critical angle `sin α_c = (3√3 M / r) √(1 − 2M/r)`.
#[test]
fn shadow_edge_matches_critical_angle() {
    let bh = BlackHole::new(1.0);
    let edge = shadow_edge(&bh, |alpha| {
        let (p_r, _, p_ph) = equatorial_momentum(alpha, R_OBS, bh.mass);
        (p_r, p_ph)
    });
    let expect = bh.capture_angle(R_OBS);
    eprintln!("integrated shadow edge {edge:.5} rad, analytic {expect:.5} rad");
    assert!(
        (edge - expect).abs() < 1.5e-3,
        "edge {edge} vs analytic {expect}"
    );
}

/// The Python construction `p_r = n_r̂ √f` (instead of `n_r̂ / √f`) stretches every
/// angle by `1/f`, so its shadow edge lands at `atan(f · tan α_c)`: 6.6 % too small at
/// `r = 30`. This pins the size of the discrepancy the port removes.
#[test]
fn python_momentum_convention_shrinks_the_shadow_by_the_lapse() {
    let bh = BlackHole::new(1.0);
    let f = bh.lapse(R_OBS);
    let edge = shadow_edge(&bh, |alpha| {
        let f_sqrt = f.sqrt();
        (-alpha.cos() * f_sqrt, alpha.sin() * R_OBS)
    });
    let expect = (f * bh.capture_angle(R_OBS).tan()).atan();
    eprintln!("python-convention shadow edge {edge:.5} rad, predicted {expect:.5} rad");
    assert!(
        (edge - expect).abs() < 1.5e-3,
        "edge {edge} vs predicted {expect}"
    );
    assert!(expect < 0.95 * bh.capture_angle(R_OBS));
}

/// Rays aimed well inside the critical angle fall in, well outside they escape, and
/// the result does not depend on the mass scale when lengths are rescaled with it.
#[test]
fn mass_scaling_is_consistent() {
    for mass in [0.5, 1.0, 2.0] {
        let bh = BlackHole::new(mass);
        let r_obs = 30.0 * mass;
        let settings = IntegratorSettings::new(200_000, 0.01 * mass, 0.01, 31.0 * mass, bh.rs());
        let f = bh.lapse(r_obs);
        for (alpha, expect_escape) in [
            (0.5 * bh.capture_angle(r_obs), false),
            (1.5 * bh.capture_angle(r_obs), true),
        ] {
            let (p_r, p_th, p_ph) = equatorial_momentum(alpha, r_obs, mass);
            let p_t = null_p_t(f, r_obs, PI / 2.0, p_r, p_th, p_ph);
            let res = integrate(
                [0.0, r_obs, PI / 2.0, 0.0],
                [p_t, p_r, p_th, p_ph],
                bh.rs(),
                &settings,
            );
            assert_eq!(
                res.exit == ExitReason::Escaped,
                expect_escape,
                "mass {mass}, alpha {alpha}"
            );
        }
    }
}
