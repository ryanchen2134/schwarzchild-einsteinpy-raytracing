//! FANTASY order-2 symplectic integrator for Schwarzschild null geodesics.
//!
//! Port of the CUDA kernel in `simulation/cuda_geodesic.py`, which mirrors
//! `einsteinpy.integrators.fantasy` with the analytic Schwarzschild metric.
//! Two copies of phase space `(q1, p1)`, `(q2, p2)` are advanced with the
//! split `A(δ/2) B(δ/2) M(δ) B(δ/2) A(δ/2)`; `M` couples the copies with
//! strength `omega`. Arithmetic order follows the Python code so results agree
//! to rounding (the golden tests check this at `M = 1`).
//!
//! # Units
//!
//! The mixing flow `M` rotates `(q1 − q2, p1 − p2)` by one angle `2ωδ`, which
//! mixes components of different dimension (`t`, `r`, `p_θ`, `p_φ` carry one power
//! of length; `θ`, `φ`, `p_t`, `p_r` none). The scheme is therefore only well
//! defined in one unit of length. Every ray is rescaled to units of `M`
//! (`r_s = 2`) before stepping and scaled back afterwards, so a black hole of any
//! mass runs exactly the `M = 1` scheme the golden tests pin. In those units the
//! coupling angle per step is `2 ω δ / M`.
//!
//! The only place the Python differed from the general formula is the
//! `r`-derivative of `g^{tt}` and `g^{rr}`, where it hard-coded `r_s = 2`;
//! in units of `M` the two are the same.

use rayon::prelude::*;

/// A ray is counted as captured once `r ≤ HORIZON_EXIT_FACTOR · r_s`.
pub const HORIZON_EXIT_FACTOR: f64 = 1.1;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntegratorSettings {
    /// Maximum number of steps per ray.
    pub steps: usize,
    /// Affine-parameter step, in the same length unit as `r`.
    pub delta: f64,
    /// FANTASY coupling between the two phase-space copies (see the module docs).
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
    /// The state stopped being physical: `r` of the first copy non-finite or
    /// non-positive, or the second copy inside the horizon where the metric is
    /// singular. The result must not be trusted.
    Diverged,
}

impl ExitReason {
    pub fn label(self) -> &'static str {
        match self {
            ExitReason::Captured => "captured",
            ExitReason::Escaped => "escaped",
            ExitReason::StepLimit => "step_limit",
            ExitReason::Diverged => "diverged",
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
            // The Python used `sin_th ** 3` (libm pow); `s*s*s` is identical at θ = π/2,
            // where every pipeline ray lives, and within 1 ulp elsewhere.
            [
                0.0,
                0.0,
                0.0,
                (-2.0 * cos_th) / ((r * r) * (sin_th * sin_th * sin_th)),
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

/// Schwarzschild radius in the integrator's own units (lengths in units of `M`).
const RS_UNIT: f64 = 2.0;

/// One ray's integration, in units of `M`.
struct Run {
    s: State,
    mass: f64,
    half_delta: f64,
    cos_m: f64,
    sin_m: f64,
    r_max: f64,
    r_capture: f64,
}

/// Divide the length-carrying components by `k`.
#[inline]
fn scale_down(q: &mut [f64; 4], p: &mut [f64; 4], k: f64) {
    q[0] /= k;
    q[1] /= k;
    p[2] /= k;
    p[3] /= k;
}

/// Multiply the length-carrying components by `k`.
#[inline]
fn scale_up(q: &mut [f64; 4], p: &mut [f64; 4], k: f64) {
    q[0] *= k;
    q[1] *= k;
    p[2] *= k;
    p[3] *= k;
}

impl Run {
    fn new(q0: [f64; 4], p0: [f64; 4], rs: f64, settings: &IntegratorSettings) -> Self {
        let mass = rs / RS_UNIT;
        let (mut q, mut p) = (q0, p0);
        scale_down(&mut q, &mut p, mass);
        let delta = settings.delta / mass;
        let (sin_m, cos_m) = (2.0 * settings.omega * delta).sin_cos();
        Self {
            s: State {
                q1: q,
                p1: p,
                q2: q,
                p2: p,
            },
            mass,
            half_delta: 0.5 * delta,
            cos_m,
            sin_m,
            r_max: settings.r_max / mass,
            r_capture: settings.r_capture / mass,
        }
    }

    #[inline]
    fn step(&mut self) {
        flow_a(&mut self.s, self.half_delta, RS_UNIT);
        flow_b(&mut self.s, self.half_delta, RS_UNIT);
        flow_mixed(&mut self.s, self.cos_m, self.sin_m);
        flow_b(&mut self.s, self.half_delta, RS_UNIT);
        flow_a(&mut self.s, self.half_delta, RS_UNIT);
    }

    #[inline]
    fn exit_reason(&self) -> Option<ExitReason> {
        let r1 = self.s.q1[1];
        let r2 = self.s.q2[1];
        if !r1.is_finite() || r1 <= 0.0 || !r2.is_finite() || r2 <= RS_UNIT {
            Some(ExitReason::Diverged)
        } else if r1 <= self.r_capture {
            Some(ExitReason::Captured)
        } else if r1 >= self.r_max {
            Some(ExitReason::Escaped)
        } else {
            None
        }
    }

    /// Current `(t, r, θ, φ)` of the first copy in the caller's units.
    #[inline]
    fn q_out(&self) -> [f64; 4] {
        let mut q = self.s.q1;
        q[0] *= self.mass;
        q[1] *= self.mass;
        q
    }

    fn finish(&self, exit: ExitReason, n_steps: usize) -> IntegrationResult {
        let (mut q, mut p) = (self.s.q1, self.s.p1);
        scale_up(&mut q, &mut p, self.mass);
        IntegrationResult {
            q,
            p,
            exit,
            n_steps,
        }
    }
}

/// Integrate one ray from `(q0, p0)` until it is captured, escapes, diverges, or
/// `settings.steps` run out.
pub fn integrate(
    q0: [f64; 4],
    p0: [f64; 4],
    rs: f64,
    settings: &IntegratorSettings,
) -> IntegrationResult {
    let mut run = Run::new(q0, p0, rs, settings);
    let mut n_steps = 0;
    let mut exit = ExitReason::StepLimit;
    for _ in 0..settings.steps {
        if let Some(reason) = run.exit_reason() {
            exit = reason;
            break;
        }
        run.step();
        n_steps += 1;
    }
    if exit == ExitReason::StepLimit {
        // A ray that crossed a boundary on its very last step is still captured/escaped.
        if let Some(reason) = run.exit_reason() {
            exit = reason;
        }
    }
    run.finish(exit, n_steps)
}

/// Like [`integrate`], also returning `(t, r, θ, φ)` *before* each step, exactly as
/// the CUDA kernel records it: when the ray leaves the domain within the step budget
/// the last entry is the exit state; otherwise `steps` entries are returned and the
/// state after the final step is not among them, even if that last step crossed a
/// boundary (the exit reason still reports it).
pub fn integrate_trajectory(
    q0: [f64; 4],
    p0: [f64; 4],
    rs: f64,
    settings: &IntegratorSettings,
) -> (Vec<[f64; 4]>, IntegrationResult) {
    let mut run = Run::new(q0, p0, rs, settings);
    let mut traj = Vec::with_capacity(settings.steps.min(1 << 16) + 1);
    let mut n_steps = 0;
    let mut exit = ExitReason::StepLimit;
    for _ in 0..settings.steps {
        traj.push(run.q_out());
        if let Some(reason) = run.exit_reason() {
            exit = reason;
            break;
        }
        run.step();
        n_steps += 1;
    }
    if exit == ExitReason::StepLimit {
        if let Some(reason) = run.exit_reason() {
            exit = reason;
        }
    }
    (traj, run.finish(exit, n_steps))
}

/// Like [`integrate_trajectory`], but keeps at most `2 · max_points + 1` states: a
/// stride-doubling reservoir that always holds the start and, when the ray leaves
/// the domain, the exit state. Memory no longer scales with the step budget.
pub fn integrate_trajectory_bounded(
    q0: [f64; 4],
    p0: [f64; 4],
    rs: f64,
    settings: &IntegratorSettings,
    max_points: usize,
) -> (Vec<[f64; 4]>, IntegrationResult) {
    let cap = 2 * max_points.max(1);
    let mut run = Run::new(q0, p0, rs, settings);
    let mut traj = Vec::with_capacity(cap);
    let mut stride = 1usize;
    let mut n_steps = 0;
    let mut exit = ExitReason::StepLimit;
    for _ in 0..settings.steps {
        let exiting = run.exit_reason();
        if n_steps % stride == 0 || exiting.is_some() {
            traj.push(run.q_out());
            if traj.len() == cap {
                traj = traj.iter().step_by(2).copied().collect();
                stride *= 2;
            }
        }
        if let Some(reason) = exiting {
            exit = reason;
            break;
        }
        run.step();
        n_steps += 1;
    }
    if exit == ExitReason::StepLimit {
        if let Some(reason) = run.exit_reason() {
            exit = reason;
        }
    }
    // A boundary crossed on the last permitted step is detected after the loop; the
    // exit state is still recorded.
    let last = run.q_out();
    if exit != ExitReason::StepLimit && traj.last() != Some(&last) {
        traj.push(last);
    }
    (traj, run.finish(exit, n_steps))
}

/// Integrate many rays in parallel. `on_ray_done` is called once per finished ray.
pub fn integrate_batch<I, F>(
    rays: I,
    rs: f64,
    settings: &IntegratorSettings,
    on_ray_done: F,
) -> Vec<IntegrationResult>
where
    I: IndexedParallelIterator<Item = ([f64; 4], [f64; 4])>,
    F: Fn() + Sync,
{
    rays.map(|(q0, p0)| {
        let r = integrate(q0, p0, rs, settings);
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
        // p_t > 0 with covariant momenta means dt/dλ = g^{tt} p_t < 0: the ray is traced
        // backwards in coordinate time, as backward ray tracing needs.
        assert!(res.q[0] < 0.0);
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

    #[test]
    fn crossing_on_the_last_permitted_step_is_still_an_exit() {
        let init = initial_conditions([30.0, 0.0, 0.0], [24.0, 4.0, 0.0], 1.0);
        let full = integrate(init.q0, init.p0, 2.0, &settings());
        let s = IntegratorSettings {
            steps: full.n_steps,
            ..settings()
        };
        let res = integrate(init.q0, init.p0, 2.0, &s);
        assert_eq!(res.exit, ExitReason::Escaped);
        assert_eq!(res.n_steps, full.n_steps);
        assert_eq!(res.q, full.q);
        // The kernel-faithful trajectory holds `steps` states and omits the exit state.
        let (traj, res2) = integrate_trajectory(init.q0, init.p0, 2.0, &s);
        assert_eq!(res2.exit, ExitReason::Escaped);
        assert_eq!(traj.len(), s.steps);
        assert!(traj.last().unwrap()[1] < 31.0);
        // The bounded recorder always includes the exit state.
        let (btraj, res3) = integrate_trajectory_bounded(init.q0, init.p0, 2.0, &s, 50);
        assert_eq!(res3.exit, ExitReason::Escaped);
        assert_eq!(btraj.last().unwrap(), &res3.q);
    }

    #[test]
    fn bounded_trajectory_is_a_thinned_copy_of_the_full_one() {
        let init = initial_conditions([30.0, 0.0, 0.0], [24.0, 4.0, 0.0], 1.0);
        let (full, _) = integrate_trajectory(init.q0, init.p0, 2.0, &settings());
        let (bounded, res) = integrate_trajectory_bounded(init.q0, init.p0, 2.0, &settings(), 100);
        assert!(
            bounded.len() <= 200 && bounded.len() > 50,
            "{}",
            bounded.len()
        );
        assert_eq!(bounded[0], init.q0);
        assert_eq!(bounded.last().unwrap(), &res.q);
        for q in &bounded {
            assert!(full.contains(q));
        }
    }

    #[test]
    fn non_finite_state_is_reported_as_diverged() {
        let init = initial_conditions([30.0, 0.0, 0.0], [24.0, 4.0, 0.0], 1.0);
        let mut q0 = init.q0;
        q0[1] = f64::NAN;
        let res = integrate(q0, init.p0, 2.0, &settings());
        assert_eq!(res.exit, ExitReason::Diverged);
        assert_eq!(res.n_steps, 0);
    }

    #[test]
    fn mass_rescaling_reproduces_the_unit_mass_run_exactly() {
        // M = 2 scales every length by a power of two, so the rescaled run must be
        // bit-identical to the M = 1 run.
        let unit = initial_conditions([30.0, 0.0, 0.0], [24.0, 1.0, 0.5], 1.0);
        let res1 = integrate(unit.q0, unit.p0, 2.0, &settings());
        let two = initial_conditions([60.0, 0.0, 0.0], [48.0, 2.0, 1.0], 2.0);
        let s2 = IntegratorSettings::new(20_000, 0.02, 0.01, 62.0, 4.0);
        let res2 = integrate(two.q0, two.p0, 4.0, &s2);
        assert_eq!(res1.exit, res2.exit);
        assert_eq!(res1.n_steps, res2.n_steps);
        assert_eq!(res2.q[0], 2.0 * res1.q[0]);
        assert_eq!(res2.q[1], 2.0 * res1.q[1]);
        assert_eq!(res2.q[2], res1.q[2]);
        assert_eq!(res2.q[3], res1.q[3]);
    }
}
