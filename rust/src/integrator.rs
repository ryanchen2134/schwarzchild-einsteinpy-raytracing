//! FANTASY order-2 symplectic integrator for Schwarzschild null geodesics.
//!
//! Port of the CUDA kernel in `simulation/cuda_geodesic.py`, which mirrors
//! `einsteinpy.integrators.fantasy` with the analytic Schwarzschild metric.
//! Two copies of phase space `(q1, p1)`, `(q2, p2)` are advanced with the
//! split `A(δ/2) B(δ/2) M(δ) B(δ/2) A(δ/2)`; `M` couples the copies with
//! strength `omega`. Arithmetic order follows the Python code so results agree
//! to rounding (the golden tests check this at `M = 1`).
//!
//! The only place the Python differed from the general formula is the
//! `r`-derivative of `g^{tt}` and `g^{rr}`, where it hard-coded `r_s = 2`;
//! this port uses `r_s`, which is identical at `M = 1`.

use rayon::prelude::*;

/// A ray is counted as captured once `r ≤ HORIZON_EXIT_FACTOR · r_s`.
pub const HORIZON_EXIT_FACTOR: f64 = 1.1;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntegratorSettings {
    /// Maximum number of steps per ray.
    pub steps: usize,
    /// Affine-parameter step.
    pub delta: f64,
    /// FANTASY coupling between the two phase-space copies.
    pub omega: f64,
    /// Rays stop once `r ≥ r_max` (escaped).
    pub r_max: f64,
    /// Rays stop once `r ≤ r_capture` (captured).
    pub r_capture: f64,
}

impl IntegratorSettings {
    pub fn new(steps: usize, delta: f64, omega: f64, r_max: f64, rs: f64) -> Self {
        Self {
            steps,
            delta,
            omega,
            r_max,
            r_capture: HORIZON_EXIT_FACTOR * rs,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExitReason {
    /// `r ≤ r_capture`.
    Captured,
    /// `r ≥ r_max`.
    Escaped,
    /// Ran out of steps inside the domain.
    StepLimit,
}

impl ExitReason {
    pub fn label(self) -> &'static str {
        match self {
            ExitReason::Captured => "captured",
            ExitReason::Escaped => "escaped",
            ExitReason::StepLimit => "step_limit",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct IntegrationResult {
    /// Final `(t, r, θ, φ)` of the first phase-space copy.
    pub q: [f64; 4],
    /// Final covariant momentum of the first copy.
    pub p: [f64; 4],
    pub exit: ExitReason,
    /// Steps actually taken.
    pub n_steps: usize,
}

#[derive(Clone, Copy)]
struct State {
    q1: [f64; 4],
    p1: [f64; 4],
    q2: [f64; 4],
    p2: [f64; 4],
}

/// Diagonal of the contravariant metric `g^{μμ}` at `q`.
#[inline]
fn metric_diag(q: &[f64; 4], rs: f64) -> [f64; 4] {
    let r = q[1];
    let th = q[2];
    let inv_fac = 1.0 - rs / r;
    let r_sin = r * th.sin();
    [
        -1.0 / inv_fac,
        inv_fac,
        1.0 / (r * r),
        1.0 / (r_sin * r_sin),
    ]
}

/// Diagonal of `∂g^{μμ}/∂q^{wrt}`; only `wrt = 1 (r)` and `wrt = 2 (θ)` are non-zero.
#[inline]
fn metric_derivative_diag(q: &[f64; 4], rs: f64, wrt: usize) -> [f64; 4] {
    let r = q[1];
    let th = q[2];
    match wrt {
        1 => {
            let denom = r - rs;
            let sin_th = th.sin();
            let sin2 = sin_th * sin_th;
            [
                rs / (denom * denom),
                rs / (r * r),
                -2.0 / (r * r * r),
                -2.0 / (r * r * r * sin2),
            ]
        }
        2 => {
            let sin_th = th.sin();
            let cos_th = th.cos();
            [
                0.0,
                0.0,
                0.0,
                (-2.0 * cos_th) / ((r * r) * sin_th.powf(3.0)),
            ]
        }
        _ => [0.0; 4],
    }
}

/// `½ (∂g^{μν}/∂q^{wrt}) p_μ p_ν`.
#[inline]
fn part_ham_flow(q: &[f64; 4], p: &[f64; 4], rs: f64, wrt: usize) -> f64 {
    let gp = metric_derivative_diag(q, rs, wrt);
    let mut acc = 0.0;
    for a in 0..4 {
        let val = gp[a];
        if val != 0.0 {
            acc += val * p[a] * p[a];
        }
    }
    0.5 * acc
}

/// `g^{μν} p_ν`.
#[inline]
fn metric_vec_mul(q: &[f64; 4], p: &[f64; 4], rs: f64) -> [f64; 4] {
    let g = metric_diag(q, rs);
    [g[0] * p[0], g[1] * p[1], g[2] * p[2], g[3] * p[3]]
}

#[inline]
fn flow_a(s: &mut State, delta: f64, rs: f64) {
    let mut dh1 = [0.0; 4];
    for (wrt, d) in dh1.iter_mut().enumerate() {
        *d = part_ham_flow(&s.q1, &s.p2, rs, wrt);
    }
    for (p, d) in s.p1.iter_mut().zip(dh1) {
        *p -= delta * d;
    }
    let dq2 = metric_vec_mul(&s.q1, &s.p2, rs);
    for (q, d) in s.q2.iter_mut().zip(dq2) {
        *q += delta * d;
    }
}

#[inline]
fn flow_b(s: &mut State, delta: f64, rs: f64) {
    let mut dh2 = [0.0; 4];
    for (wrt, d) in dh2.iter_mut().enumerate() {
        *d = part_ham_flow(&s.q2, &s.p1, rs, wrt);
    }
    for (p, d) in s.p2.iter_mut().zip(dh2) {
        *p -= delta * d;
    }
    let dq1 = metric_vec_mul(&s.q2, &s.p1, rs);
    for (q, d) in s.q1.iter_mut().zip(dq1) {
        *q += delta * d;
    }
}

#[inline]
fn flow_mixed(s: &mut State, cos: f64, sin: f64) {
    let mut q1 = [0.0; 4];
    let mut p1 = [0.0; 4];
    let mut q2 = [0.0; 4];
    let mut p2 = [0.0; 4];
    for k in 0..4 {
        let q_sum = s.q1[k] + s.q2[k];
        let q_dif = s.q1[k] - s.q2[k];
        let p_sum = s.p1[k] + s.p2[k];
        let p_dif = s.p1[k] - s.p2[k];
        q1[k] = 0.5 * (q_sum + q_dif * cos + p_dif * sin);
        p1[k] = 0.5 * (p_sum + p_dif * cos - q_dif * sin);
        q2[k] = 0.5 * (q_sum - q_dif * cos - p_dif * sin);
        p2[k] = 0.5 * (p_sum - p_dif * cos + q_dif * sin);
    }
    s.q1 = q1;
    s.p1 = p1;
    s.q2 = q2;
    s.p2 = p2;
}

#[inline]
fn fantasy_step(s: &mut State, half_delta: f64, rs: f64, cos_m: f64, sin_m: f64) {
    flow_a(s, half_delta, rs);
    flow_b(s, half_delta, rs);
    flow_mixed(s, cos_m, sin_m);
    flow_b(s, half_delta, rs);
    flow_a(s, half_delta, rs);
}

#[inline]
fn exit_reason(r: f64, s: &IntegratorSettings) -> Option<ExitReason> {
    if r <= s.r_capture {
        Some(ExitReason::Captured)
    } else if r >= s.r_max {
        Some(ExitReason::Escaped)
    } else {
        None
    }
}

/// Integrate one ray from `(q0, p0)` until it is captured, escapes, or `settings.steps` run out.
pub fn integrate(
    q0: [f64; 4],
    p0: [f64; 4],
    rs: f64,
    settings: &IntegratorSettings,
) -> IntegrationResult {
    let mut s = State {
        q1: q0,
        p1: p0,
        q2: q0,
        p2: p0,
    };
    let half = 0.5 * settings.delta;
    let (sin_m, cos_m) = (2.0 * settings.omega * settings.delta).sin_cos();
    let mut n_steps = 0;
    let mut exit = ExitReason::StepLimit;
    for _ in 0..settings.steps {
        if let Some(reason) = exit_reason(s.q1[1], settings) {
            exit = reason;
            break;
        }
        fantasy_step(&mut s, half, rs, cos_m, sin_m);
        n_steps += 1;
    }
    if exit == ExitReason::StepLimit {
        // A ray that crossed a boundary on its very last step is still captured/escaped.
        if let Some(reason) = exit_reason(s.q1[1], settings) {
            exit = reason;
        }
    }
    IntegrationResult {
        q: s.q1,
        p: s.p1,
        exit,
        n_steps,
    }
}

/// Like [`integrate`], also returning `(t, r, θ, φ)` *before* each step. When the
/// ray leaves the domain the last entry is the exit state; when it runs out of steps
/// the state after the final step is not recorded (same as the CUDA kernel).
pub fn integrate_trajectory(
    q0: [f64; 4],
    p0: [f64; 4],
    rs: f64,
    settings: &IntegratorSettings,
) -> (Vec<[f64; 4]>, IntegrationResult) {
    let mut s = State {
        q1: q0,
        p1: p0,
        q2: q0,
        p2: p0,
    };
    let half = 0.5 * settings.delta;
    let (sin_m, cos_m) = (2.0 * settings.omega * settings.delta).sin_cos();
    let mut traj = Vec::with_capacity(settings.steps.min(1 << 16) + 1);
    let mut n_steps = 0;
    let mut exit = ExitReason::StepLimit;
    for _ in 0..settings.steps {
        traj.push(s.q1);
        if let Some(reason) = exit_reason(s.q1[1], settings) {
            exit = reason;
            break;
        }
        fantasy_step(&mut s, half, rs, cos_m, sin_m);
        n_steps += 1;
    }
    if exit == ExitReason::StepLimit {
        if let Some(reason) = exit_reason(s.q1[1], settings) {
            exit = reason;
        }
    }
    (
        traj,
        IntegrationResult {
            q: s.q1,
            p: s.p1,
            exit,
            n_steps,
        },
    )
}

/// Integrate many rays in parallel. `on_ray_done` is called once per finished ray.
pub fn integrate_batch<F>(
    rays: &[([f64; 4], [f64; 4])],
    rs: f64,
    settings: &IntegratorSettings,
    on_ray_done: F,
) -> Vec<IntegrationResult>
where
    F: Fn() + Sync,
{
    rays.par_iter()
        .map(|(q0, p0)| {
            let r = integrate(*q0, *p0, rs, settings);
            on_ray_done();
            r
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::initial_conditions::initial_conditions;

    fn settings() -> IntegratorSettings {
        IntegratorSettings::new(20_000, 0.01, 0.01, 31.0, 2.0)
    }

    #[test]
    fn radial_ray_is_captured() {
        let init = initial_conditions([30.0, 0.0, 0.0], [24.0, 0.0, 0.0], 1.0);
        let res = integrate(init.q0, init.p0, 2.0, &settings());
        assert_eq!(res.exit, ExitReason::Captured);
        assert!(res.q[1] <= 2.2);
    }

    #[test]
    fn wide_ray_escapes_and_conserves_energy_and_angular_momentum() {
        let init = initial_conditions([30.0, 0.0, 0.0], [24.0, 4.0, 0.0], 1.0);
        let res = integrate(init.q0, init.p0, 2.0, &settings());
        assert_eq!(res.exit, ExitReason::Escaped);
        assert!(res.q[1] >= 31.0);
        // p_t and p_φ are conserved by the Schwarzschild Hamiltonian. FANTASY conserves
        // the extended Hamiltonian of the two coupled copies, so each copy's p_t drifts
        // at the scheme's second-order level (≈6e-6 here at δ = 0.01).
        assert!(
            (res.p[0] - init.p0[0]).abs() < 1e-4,
            "p_t drift {}",
            res.p[0] - init.p0[0]
        );
        assert!(
            (res.p[3] - init.p0[3]).abs() < 1e-4,
            "p_φ drift {}",
            res.p[3] - init.p0[3]
        );
    }

    #[test]
    fn trajectory_last_entry_is_exit_state() {
        let init = initial_conditions([30.0, 0.0, 0.0], [24.0, 4.0, 0.0], 1.0);
        let (traj, res) = integrate_trajectory(init.q0, init.p0, 2.0, &settings());
        assert_eq!(traj.len(), res.n_steps + 1);
        assert_eq!(traj.last().unwrap(), &res.q);
        assert_eq!(traj[0], init.q0);
    }

    #[test]
    fn step_limit_when_steps_too_few() {
        let init = initial_conditions([30.0, 0.0, 0.0], [24.0, 4.0, 0.0], 1.0);
        let s = IntegratorSettings {
            steps: 10,
            ..settings()
        };
        let res = integrate(init.q0, init.p0, 2.0, &s);
        assert_eq!(res.exit, ExitReason::StepLimit);
        assert_eq!(res.n_steps, 10);
    }
}
