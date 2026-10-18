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
//!
//! # Disk crossings
//!
//! A ray may carry the azimuth `phi_d` at which the accretion disk's plane cuts its
//! orbital plane ([`crate::disk::Disk::ray_frame`]). The plane is crossed where
//! `n' · x = r |n'_⊥| sin(φ − phi_d)` changes sign, i.e. where
//! `floor((φ − phi_d)/π)` changes by exactly one between consecutive steps; `φ` is
//! never reduced modulo `2π` inside the integrator, so the floor is monotone along
//! the ray. The crossing radius is interpolated linearly in `φ` between the two
//! states. Crossings are only looked for from the second step on: every ray starts
//! at the observer, which sits on the crossing line of an edge-on disk, where
//! rounding could register a crossing at `r_obs`. A null ray moves at most
//! `|dr/dλ| ≤ p_t < 1` per unit affine parameter, so a crossing during the first
//! step lies within `δ` of `r_obs`; it is neither counted nor able to hit the disk as
//! long as `r_out < r_obs − δ`, which the renderer and the CLI enforce
//! ([`crate::render::render_curved`]). Each step's crossing check runs before the next
//! step's capture/escape check, so a crossing on a ray's last step is not lost. A
//! crossing inside `[r_in, r_out]` ends the ray with [`ExitReason::Disk`]; every
//! crossing is counted. The arithmetic is written out so that the SIMD and CUDA
//! kernels can reproduce it bit for bit.

use rayon::prelude::*;
use std::f64::consts::PI;

/// A ray is counted as captured once `r ≤ HORIZON_EXIT_FACTOR · r_s`.
pub const HORIZON_EXIT_FACTOR: f64 = 1.1;

/// Radial extent of the disk, in the caller's length unit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiskBand {
    pub r_in: f64,
    pub r_out: f64,
}

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
    /// Rays that carry a finite `disk_phi` stop on crossing the disk plane inside
    /// this band. `None` disables crossing detection for every ray.
    pub disk: Option<DiskBand>,
}

impl IntegratorSettings {
    pub fn new(steps: usize, delta: f64, omega: f64, r_max: f64, rs: f64) -> Self {
        Self {
            steps,
            delta,
            omega,
            r_max,
            r_capture: HORIZON_EXIT_FACTOR * rs,
            disk: None,
        }
    }

    pub fn with_disk(mut self, band: DiskBand) -> Self {
        self.disk = Some(band);
        self
    }
}

/// One ray's initial state plus the per-ray disk geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    pub q0: [f64; 4],
    pub p0: [f64; 4],
    /// Azimuth of the disk-plane crossing line in this ray's frame; `NaN` when no
    /// crossing is to be detected (no disk, or the ray lies in the disk plane).
    pub disk_phi: f64,
}

impl Ray {
    pub fn new(q0: [f64; 4], p0: [f64; 4]) -> Self {
        Self {
            q0,
            p0,
            disk_phi: f64::NAN,
        }
    }

    pub fn with_disk(q0: [f64; 4], p0: [f64; 4], disk_phi: f64) -> Self {
        Self { q0, p0, disk_phi }
    }

    #[inline]
    pub fn detects_disk(&self) -> bool {
        self.disk_phi.is_finite()
    }
}

impl From<([f64; 4], [f64; 4])> for Ray {
    fn from((q0, p0): ([f64; 4], [f64; 4])) -> Self {
        Ray::new(q0, p0)
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
    /// Crossed the accretion disk's plane inside its radial band.
    Disk,
}

impl ExitReason {
    pub fn label(self) -> &'static str {
        match self {
            ExitReason::Captured => "captured",
            ExitReason::Escaped => "escaped",
            ExitReason::StepLimit => "step_limit",
            ExitReason::Diverged => "diverged",
            ExitReason::Disk => "disk",
        }
    }
}

/// Where a ray met the disk, in the ray's rotated frame and the caller's length unit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiskHit {
    pub r: f64,
    /// `phi_d + kπ` for the crossing's `k`; not reduced modulo `2π`.
    pub phi: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct IntegrationResult {
    /// Final `(t, r, θ, φ)` of the first phase-space copy (for a disk hit, the state
    /// just after the crossing step).
    pub q: [f64; 4],
    /// Final covariant momentum of the first copy.
    pub p: [f64; 4],
    pub exit: ExitReason,
    /// Steps actually taken.
    pub n_steps: usize,
    /// Set when `exit == Disk`.
    pub disk_hit: Option<DiskHit>,
    /// How many times the ray crossed the disk plane, the terminating hit included.
    pub plane_crossings: u32,
}

/// Crossing detection between two consecutive states, in units of `M`.
#[derive(Clone, Copy, Debug)]
pub struct CrossingDetector {
    pub phi_d: f64,
    pub r_in: f64,
    pub r_out: f64,
}

/// A detected crossing: interpolated `(r, φ)` and whether it lies in the band.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Crossing {
    pub r: f64,
    pub phi: f64,
    pub in_band: bool,
}

impl CrossingDetector {
    /// Did the plane get crossed between `(r_prev, phi_prev)` and `(r, phi)`?
    #[inline]
    pub fn check(&self, r_prev: f64, phi_prev: f64, r: f64, phi: f64) -> Option<Crossing> {
        let k_prev = ((phi_prev - self.phi_d) / PI).floor();
        let k = ((phi - self.phi_d) / PI).floor();
        let d = k - k_prev;
        if d != 1.0 && d != -1.0 {
            return None;
        }
        let k_boundary = if d > 0.0 { k } else { k_prev };
        let phi_c = self.phi_d + k_boundary * PI;
        let frac = (phi_c - phi_prev) / (phi - phi_prev);
        let r_hit = r_prev + (r - r_prev) * frac;
        Some(Crossing {
            r: r_hit,
            phi: phi_c,
            in_band: r_hit >= self.r_in && r_hit <= self.r_out,
        })
    }
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
    disk: Option<CrossingDetector>,
    n_steps: usize,
    crossings: u32,
    hit: Option<DiskHit>,
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
    fn new(ray: &Ray, rs: f64, settings: &IntegratorSettings) -> Self {
        let mass = rs / RS_UNIT;
        let (mut q, mut p) = (ray.q0, ray.p0);
        scale_down(&mut q, &mut p, mass);
        let delta = settings.delta / mass;
        let (sin_m, cos_m) = (2.0 * settings.omega * delta).sin_cos();
        let disk = match settings.disk {
            Some(band) if ray.detects_disk() => Some(CrossingDetector {
                phi_d: ray.disk_phi,
                r_in: band.r_in / mass,
                r_out: band.r_out / mass,
            }),
            _ => None,
        };
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
            disk,
            n_steps: 0,
            crossings: 0,
            hit: None,
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

    /// One step, then the disk-crossing check (from the second step on). Returns
    /// `true` when the ray hit the disk.
    #[inline]
    fn advance(&mut self) -> bool {
        let (r_prev, phi_prev) = (self.s.q1[1], self.s.q1[3]);
        self.step();
        self.n_steps += 1;
        if let Some(det) = &self.disk {
            if self.n_steps >= 2 {
                if let Some(c) = det.check(r_prev, phi_prev, self.s.q1[1], self.s.q1[3]) {
                    self.crossings += 1;
                    if c.in_band {
                        self.hit = Some(DiskHit { r: c.r, phi: c.phi });
                        return true;
                    }
                }
            }
        }
        false
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

    fn finish(&self, exit: ExitReason) -> IntegrationResult {
        let (mut q, mut p) = (self.s.q1, self.s.p1);
        scale_up(&mut q, &mut p, self.mass);
        IntegrationResult {
            q,
            p,
            exit,
            n_steps: self.n_steps,
            disk_hit: self.hit.map(|h| DiskHit {
                r: h.r * self.mass,
                phi: h.phi,
            }),
            plane_crossings: self.crossings,
        }
    }
}

/// Integrate one ray (no disk) from `(q0, p0)` until it is captured, escapes,
/// diverges, or `settings.steps` run out.
pub fn integrate(
    q0: [f64; 4],
    p0: [f64; 4],
    rs: f64,
    settings: &IntegratorSettings,
) -> IntegrationResult {
    integrate_ray(&Ray::new(q0, p0), rs, settings)
}

/// Integrate one ray until it is captured, escapes, hits the disk, diverges, or
/// `settings.steps` run out.
pub fn integrate_ray(ray: &Ray, rs: f64, settings: &IntegratorSettings) -> IntegrationResult {
    let mut run = Run::new(ray, rs, settings);
    let mut exit = ExitReason::StepLimit;
    for _ in 0..settings.steps {
        if let Some(reason) = run.exit_reason() {
            exit = reason;
            break;
        }
        if run.advance() {
            exit = ExitReason::Disk;
            break;
        }
    }
    if exit == ExitReason::StepLimit {
        // A ray that crossed a boundary on its very last step is still captured/escaped.
        if let Some(reason) = run.exit_reason() {
            exit = reason;
        }
    }
    run.finish(exit)
}

/// Like [`integrate_ray`], also returning `(t, r, θ, φ)` *before* each step, exactly as
/// the CUDA kernel records it: when the ray leaves the domain within the step budget
/// the last entry is the exit state; otherwise `steps` entries are returned and the
/// state after the final step is not among them, even if that last step crossed a
/// boundary (the exit reason still reports it).
pub fn integrate_trajectory(
    ray: &Ray,
    rs: f64,
    settings: &IntegratorSettings,
) -> (Vec<[f64; 4]>, IntegrationResult) {
    let mut run = Run::new(ray, rs, settings);
    let mut traj = Vec::with_capacity(settings.steps.min(1 << 16) + 1);
    let mut exit = ExitReason::StepLimit;
    for _ in 0..settings.steps {
        traj.push(run.q_out());
        if let Some(reason) = run.exit_reason() {
            exit = reason;
            break;
        }
        if run.advance() {
            exit = ExitReason::Disk;
            traj.push(run.q_out());
            break;
        }
    }
    if exit == ExitReason::StepLimit {
        if let Some(reason) = run.exit_reason() {
            exit = reason;
        }
    }
    (traj, run.finish(exit))
}

/// Like [`integrate_trajectory`], but keeps at most `2 · max_points + 1` states: a
/// stride-doubling reservoir that always holds the start and, when the ray leaves
/// the domain, the exit state. Memory no longer scales with the step budget.
pub fn integrate_trajectory_bounded(
    ray: &Ray,
    rs: f64,
    settings: &IntegratorSettings,
    max_points: usize,
) -> (Vec<[f64; 4]>, IntegrationResult) {
    let cap = 2 * max_points.max(1);
    let mut run = Run::new(ray, rs, settings);
    let mut traj = Vec::with_capacity(cap);
    let mut stride = 1usize;
    let mut exit = ExitReason::StepLimit;
    for _ in 0..settings.steps {
        let exiting = run.exit_reason();
        if run.n_steps.is_multiple_of(stride) || exiting.is_some() {
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
        if run.advance() {
            exit = ExitReason::Disk;
            break;
        }
    }
    if exit == ExitReason::StepLimit {
        if let Some(reason) = run.exit_reason() {
            exit = reason;
        }
    }
    // A boundary crossed on the last permitted step is detected after the loop, and a
    // disk hit ends the loop after the step; the exit state is still recorded.
    // Compare bit patterns: a diverged NaN state already recorded is not recorded again.
    let last = run.q_out();
    let same = |q: &[f64; 4]| q.map(f64::to_bits) == last.map(f64::to_bits);
    if exit != ExitReason::StepLimit && !traj.last().is_some_and(same) {
        traj.push(last);
    }
    (traj, run.finish(exit))
}

/// Integrate many rays in parallel. `on_ray_done` is called once per finished ray.
pub fn integrate_batch<I, F>(
    rays: I,
    rs: f64,
    settings: &IntegratorSettings,
    on_ray_done: F,
) -> Vec<IntegrationResult>
where
    I: IndexedParallelIterator<Item = Ray>,
    F: Fn() + Sync,
{
    rays.map(|ray| {
        let r = integrate_ray(&ray, rs, settings);
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
        let (traj, res) = integrate_trajectory(&Ray::new(init.q0, init.p0), 2.0, &settings());
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
        let (traj, res2) = integrate_trajectory(&Ray::new(init.q0, init.p0), 2.0, &s);
        assert_eq!(res2.exit, ExitReason::Escaped);
        assert_eq!(traj.len(), s.steps);
        assert!(traj.last().unwrap()[1] < 31.0);
        // The bounded recorder always includes the exit state.
        let (btraj, res3) = integrate_trajectory_bounded(&Ray::new(init.q0, init.p0), 2.0, &s, 50);
        assert_eq!(res3.exit, ExitReason::Escaped);
        assert_eq!(btraj.last().unwrap(), &res3.q);
    }

    #[test]
    fn bounded_trajectory_is_a_thinned_copy_of_the_full_one() {
        let init = initial_conditions([30.0, 0.0, 0.0], [24.0, 4.0, 0.0], 1.0);
        let (full, _) = integrate_trajectory(&Ray::new(init.q0, init.p0), 2.0, &settings());
        let (bounded, res) =
            integrate_trajectory_bounded(&Ray::new(init.q0, init.p0), 2.0, &settings(), 100);
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

    /// The linear-interpolation error of the hit radius is `C h² f (1 − f)` for a
    /// crossing at fraction `f` of the step, so the step is placed to put the crossing
    /// at the same fraction for every `h`; then halving `h` must divide the error by 4.
    /// Taking `r_prev` instead (a first-order rule) divides it by 2, which the test
    /// checks it can tell apart.
    #[test]
    fn crossing_detector_converges_as_the_step_squared_at_a_fixed_fraction() {
        let x0 = [30.0f64, -4.0];
        let d = [-0.96f64, 0.28];
        let phi_d = 2.0f64;
        let det = CrossingDetector {
            phi_d,
            r_in: 0.0,
            r_out: 1e9,
        };
        let (sd, cd) = phi_d.sin_cos();
        let lambda_c = -(x0[0] * sd - x0[1] * cd) / (d[0] * sd - d[1] * cd);
        let at = |l: f64| {
            let (x, y) = (x0[0] + l * d[0], x0[1] + l * d[1]);
            (x.hypot(y), y.atan2(x))
        };
        let exact_r = at(lambda_c).0;
        for f in [0.25, 0.5, 0.8] {
            let errors = |h: f64| {
                let (r0, p0) = at(lambda_c - f * h);
                let (r1, p1) = at(lambda_c + (1.0 - f) * h);
                let c = det
                    .check(r0, p0, r1, p1)
                    .expect("the step straddles the crossing");
                ((c.r - exact_r).abs(), (r0 - exact_r).abs())
            };
            let (e1, z1) = errors(0.4);
            let (e2, z2) = errors(0.2);
            let (e3, z3) = errors(0.1);
            for (a, b) in [(e1, e2), (e2, e3)] {
                assert!(
                    (3.6..4.4).contains(&(a / b)),
                    "f = {f}: ratio {} ({e1} {e2} {e3})",
                    a / b
                );
            }
            for (a, b) in [(z1, z2), (z2, z3)] {
                assert!(
                    (1.8..2.2).contains(&(a / b)),
                    "first-order reference: {}",
                    a / b
                );
            }
        }
    }

    /// A straight line in a flat plane, fed to the detector step by step from a fixed
    /// start: one crossing, at the exact line–line intersection to within the
    /// interpolation error.
    #[test]
    fn crossing_detector_on_a_straight_line_finds_one_crossing() {
        // Line x(λ) = x0 + λ d in the plane, crossing line at azimuth phi_d.
        let x0 = [30.0f64, -4.0];
        let d = [-0.96f64, 0.28];
        let phi_d = 2.0f64;
        let det = CrossingDetector {
            phi_d,
            r_in: 0.0,
            r_out: 1e9,
        };
        // Exact crossing: x(λ) ∥ (cos phi_d, sin phi_d) ⇒ x(λ) sin phi_d − y(λ) cos phi_d = 0.
        let (sd, cd) = phi_d.sin_cos();
        let lambda = -(x0[0] * sd - x0[1] * cd) / (d[0] * sd - d[1] * cd);
        let exact_r = ((x0[0] + lambda * d[0]).powi(2) + (x0[1] + lambda * d[1]).powi(2)).sqrt();
        let run = |h: f64| -> (f64, u32) {
            let mut lam = 0.0;
            let state = |l: f64| {
                let x = x0[0] + l * d[0];
                let y = x0[1] + l * d[1];
                (x.hypot(y), y.atan2(x))
            };
            let (mut r_prev, mut phi_prev) = state(0.0);
            // Unwrap φ by hand as the integrator's monotone φ would be.
            let mut hit = None;
            let mut count = 0;
            while lam < 80.0 {
                lam += h;
                let (r, mut phi) = state(lam);
                while phi - phi_prev > PI {
                    phi -= 2.0 * PI;
                }
                while phi - phi_prev < -PI {
                    phi += 2.0 * PI;
                }
                if let Some(c) = det.check(r_prev, phi_prev, r, phi) {
                    count += 1;
                    if hit.is_none() {
                        hit = Some(c.r);
                    }
                }
                r_prev = r;
                phi_prev = phi;
            }
            (hit.expect("the line crosses"), count)
        };
        let (r1, n1) = run(0.2);
        let (r2, n2) = run(0.1);
        let (r3, n3) = run(0.05);
        assert_eq!(
            (n1, n2, n3),
            (1, 1, 1),
            "a straight line crosses a line through the origin once"
        );
        for r in [r1, r2, r3] {
            assert!((r - exact_r).abs() < 2e-3, "{r} vs {exact_r}");
        }
    }

    /// Both crossing directions on a spiral `r = 9 + 0.1 φ`: forward (d = +1, the
    /// boundary is `k`) and backward (d = −1, the boundary is `k_prev`) find the same
    /// azimuths, and the interpolated radius sits on the spiral.
    #[test]
    fn crossings_on_a_spiral_agree_in_both_directions() {
        let det = CrossingDetector {
            phi_d: 0.7,
            r_in: 0.0,
            r_out: 100.0,
        };
        let spiral = |phi: f64| 9.0 + 0.1 * phi;
        let sweep = |forward: bool| {
            let mut out = Vec::new();
            let (start, step) = if forward {
                (0.0, 0.013)
            } else {
                (4.0 * PI, -0.013)
            };
            let mut prev = start;
            for k in 1..1000 {
                let phi = start + k as f64 * step;
                if let Some(c) = det.check(spiral(prev), prev, spiral(phi), phi) {
                    assert!((c.r - spiral(c.phi)).abs() < 1e-12, "{c:?}");
                    out.push(c.phi);
                }
                prev = phi;
            }
            out
        };
        let fwd = sweep(true);
        let mut back = sweep(false);
        back.reverse();
        assert_eq!(fwd.len(), 4);
        assert_eq!(
            fwd, back,
            "the same boundaries, k on the way up and k_prev on the way down"
        );
        for (k, phi) in fwd.iter().enumerate() {
            assert!((phi - (0.7 + k as f64 * PI)).abs() < 1e-12);
        }
    }

    #[test]
    fn circular_motion_crosses_the_plane_twice_per_revolution_at_the_orbit_radius() {
        let det = CrossingDetector {
            phi_d: 0.7,
            r_in: 5.0,
            r_out: 20.0,
        };
        let r = 9.0;
        let mut phi_prev = 0.0;
        let mut hits = Vec::new();
        let mut phi = 0.0;
        while phi < 4.0 * PI {
            phi += 0.013;
            if let Some(c) = det.check(r, phi_prev, r, phi) {
                assert_eq!(c.r, r);
                assert!(c.in_band);
                hits.push(c.phi);
            }
            phi_prev = phi;
        }
        assert_eq!(hits.len(), 4, "{hits:?}");
        for (k, h) in hits.iter().enumerate() {
            assert!(
                (h - (0.7 + k as f64 * PI)).abs() < 1e-12,
                "crossing {k} at {h}"
            );
        }
        // Going the other way (φ decreasing) finds the same crossings.
        let mut back = Vec::new();
        let (mut phi, mut phi_prev) = (4.0 * PI, 4.0 * PI);
        while phi > 0.0 {
            phi -= 0.013;
            if let Some(c) = det.check(r, phi_prev, r, phi) {
                back.push(c.phi);
            }
            phi_prev = phi;
        }
        assert_eq!(back.len(), 4);
        // A jump of two boundaries in one step is not a crossing.
        assert!(det.check(r, 0.0, r, 0.7 + 2.5 * PI).is_none());
        assert!(det.check(r, 1.0, r, 1.2).is_none());
    }

    fn disk_settings() -> IntegratorSettings {
        settings().with_disk(DiskBand {
            r_in: 6.0,
            r_out: 20.0,
        })
    }

    #[test]
    fn a_ray_aimed_through_the_disk_stops_there() {
        // Face-on disk (plane x = 0, crossing line at φ = ±π/2 in every ray's frame):
        // a ray with impact parameter ≈ 10 crosses it inside the band.
        let init = initial_conditions([30.0, 0.0, 0.0], [24.0, 2.0, 0.0], 1.0);
        let ray = Ray::with_disk(init.q0, init.p0, PI / 2.0);
        let res = integrate_ray(&ray, 2.0, &disk_settings());
        assert_eq!(res.exit, ExitReason::Disk);
        let hit = res.disk_hit.unwrap();
        assert!(hit.r > 6.0 && hit.r < 20.0, "{hit:?}");
        assert!((hit.phi - PI / 2.0).abs() < 1e-12);
        assert_eq!(res.plane_crossings, 1);
        assert!(
            res.q[3] >= hit.phi && res.q[3] - hit.phi < 0.01,
            "stopped just past the crossing"
        );
        // Without the per-ray azimuth, or without the band, the same ray escapes.
        let plain = integrate_ray(&Ray::new(init.q0, init.p0), 2.0, &disk_settings());
        assert_eq!(plain.exit, ExitReason::Escaped);
        assert_eq!(plain.plane_crossings, 0);
        let no_band = integrate_ray(&ray, 2.0, &settings());
        assert_eq!(no_band.exit, ExitReason::Escaped);
        // A radial ray never crosses the plane through the hole: captured, no crossings.
        let radial = initial_conditions([30.0, 0.0, 0.0], [24.0, 0.0, 0.0], 1.0);
        let res = integrate_ray(
            &Ray::with_disk(radial.q0, radial.p0, PI / 2.0),
            2.0,
            &disk_settings(),
        );
        assert_eq!(res.exit, ExitReason::Captured);
        assert_eq!(res.plane_crossings, 0);
        assert!(res.disk_hit.is_none());
    }

    #[test]
    fn crossings_outside_the_band_are_counted_but_do_not_stop_the_ray() {
        // Impact parameter ≈ 24 > r_out: the crossing is counted, the ray escapes.
        let init = initial_conditions([30.0, 0.0, 0.0], [24.0, 6.0, 0.0], 1.0);
        let res = integrate_ray(
            &Ray::with_disk(init.q0, init.p0, PI / 2.0),
            2.0,
            &disk_settings(),
        );
        assert_eq!(res.exit, ExitReason::Escaped);
        assert_eq!(res.plane_crossings, 1);
        assert!(res.disk_hit.is_none());
    }

    #[test]
    fn disk_off_results_are_unchanged_by_the_crossing_code() {
        for pix in [[24.0, 0.5, 0.0], [24.0, 1.1, 0.3], [24.0, 4.0, 0.0]] {
            let init = initial_conditions([30.0, 0.0, 0.0], pix, 1.0);
            let a = integrate(init.q0, init.p0, 2.0, &settings());
            let b = integrate_ray(&Ray::new(init.q0, init.p0), 2.0, &disk_settings());
            assert_eq!(a.q, b.q);
            assert_eq!(a.p, b.p);
            assert_eq!(a.exit, b.exit);
            assert_eq!(a.n_steps, b.n_steps);
            assert_eq!(b.plane_crossings, 0);
        }
    }

    #[test]
    fn edge_on_rays_do_not_register_a_crossing_at_the_observer() {
        // Edge-on disk n = ẑ: the crossing line is the x axis (phi_d = 0 or π) and every
        // ray starts on it. Wide rays leave the plane and never return.
        let s = settings().with_disk(DiskBand {
            r_in: 6.0,
            r_out: 29.5,
        });
        for pix in [[24.0, 6.0, 6.0], [24.0, -5.0, 6.0], [24.0, 0.0, -6.0]] {
            let init = initial_conditions([30.0, 0.0, 0.0], pix, 1.0);
            let phi_d = if init.beta.sin() > 0.0 { 0.0 } else { PI };
            let res = integrate_ray(&Ray::with_disk(init.q0, init.p0, phi_d), 2.0, &s);
            assert_eq!(res.exit, ExitReason::Escaped, "{pix:?}");
            assert_eq!(res.plane_crossings, 0, "{pix:?}");
        }
        // The case the first-step rule exists for: a crossing line a hair ahead of the
        // observer, crossed during the very first step at r ≈ r_obs.
        let init = initial_conditions([30.0, 0.0, 0.0], [24.0, 6.0, 0.0], 1.0);
        let ray = Ray::with_disk(init.q0, init.p0, 1e-12);
        let (traj, _) = integrate_trajectory(&ray, 2.0, &s);
        let det = CrossingDetector {
            phi_d: 1e-12,
            r_in: 6.0,
            r_out: 29.5,
        };
        let first = det.check(traj[0][1], traj[0][3], traj[1][1], traj[1][3]);
        assert!(
            first.is_some_and(|c| c.r > 29.9),
            "the first step does cross: {first:?}"
        );
        let res = integrate_ray(&ray, 2.0, &s);
        assert_eq!(res.exit, ExitReason::Escaped);
        assert_eq!(res.plane_crossings, 0, "but the first step is not checked");
    }

    #[test]
    fn a_crossing_on_the_last_permitted_step_is_a_disk_exit() {
        let init = initial_conditions([30.0, 0.0, 0.0], [24.0, 2.0, 0.0], 1.0);
        let ray = Ray::with_disk(init.q0, init.p0, PI / 2.0);
        let full = integrate_ray(&ray, 2.0, &disk_settings());
        assert_eq!(full.exit, ExitReason::Disk);
        let s = IntegratorSettings {
            steps: full.n_steps,
            ..disk_settings()
        };
        let res = integrate_ray(&ray, 2.0, &s);
        assert_eq!(res.exit, ExitReason::Disk);
        assert_eq!(res.disk_hit, full.disk_hit);
        let short = IntegratorSettings {
            steps: full.n_steps - 1,
            ..disk_settings()
        };
        assert_eq!(integrate_ray(&ray, 2.0, &short).exit, ExitReason::StepLimit);
        // The recorders stop at the hit too and end with the post-crossing state.
        let (traj, r2) = integrate_trajectory(&ray, 2.0, &disk_settings());
        assert_eq!(r2.exit, ExitReason::Disk);
        assert_eq!(traj.len(), full.n_steps + 1);
        assert_eq!(traj.last().unwrap(), &full.q);
        let (btraj, r3) = integrate_trajectory_bounded(&ray, 2.0, &disk_settings(), 40);
        assert_eq!(r3.exit, ExitReason::Disk);
        assert_eq!(btraj.last().unwrap(), &full.q);
        assert!(btraj.len() <= 81);
    }

    /// A near-critical ray loops the hole: it crosses the plane of a face-on disk
    /// several times inside `r_in` (counted, not stopping) before one crossing lands
    /// in the band or the ray leaves.
    #[test]
    fn a_looping_ray_counts_every_crossing() {
        let alpha = 1.002 * crate::schwarzschild::BlackHole::new(1.0).capture_angle(30.0);
        let (p_r, p_th, p_ph) = crate::initial_conditions::equatorial_momentum(alpha, 30.0, 1.0);
        let p_t =
            crate::initial_conditions::null_p_t(1.0 - 2.0 / 30.0, 30.0, PI / 2.0, p_r, p_th, p_ph);
        let s = IntegratorSettings::new(200_000, 0.01, 0.01, 31.0, 2.0).with_disk(DiskBand {
            r_in: 6.0,
            r_out: 20.0,
        });
        let ray = Ray::with_disk([0.0, 30.0, PI / 2.0, 0.0], [p_t, p_r, p_th, p_ph], PI / 2.0);
        let res = integrate_ray(&ray, 2.0, &s);
        let (traj, _) = integrate_trajectory(&ray, 2.0, &s);
        // Count by hand along the recorded states: every boundary of floor((φ−π/2)/π).
        let boundaries = traj
            .windows(2)
            .skip(1)
            .filter(|w| ((w[1][3] - PI / 2.0) / PI).floor() != ((w[0][3] - PI / 2.0) / PI).floor())
            .count();
        assert!(res.plane_crossings >= 2, "{res:?}");
        assert_eq!(res.plane_crossings as usize, boundaries, "{res:?}");
        assert_eq!(res.exit, ExitReason::Disk, "{res:?}");
        let hit = res.disk_hit.unwrap();
        assert!(
            (hit.phi - (PI / 2.0 + (res.plane_crossings - 1) as f64 * PI)).abs() < 1e-9,
            "the hit is the last counted boundary: {hit:?}"
        );
    }

    #[test]
    fn phi_is_never_wrapped() {
        // A near-critical ray loops the hole: its final φ exceeds 2π.
        let alpha = 1.002 * crate::schwarzschild::BlackHole::new(1.0).capture_angle(30.0);
        let (p_r, p_th, p_ph) = crate::initial_conditions::equatorial_momentum(alpha, 30.0, 1.0);
        let p_t =
            crate::initial_conditions::null_p_t(1.0 - 2.0 / 30.0, 30.0, PI / 2.0, p_r, p_th, p_ph);
        let res = integrate(
            [0.0, 30.0, PI / 2.0, 0.0],
            [p_t, p_r, p_th, p_ph],
            2.0,
            &IntegratorSettings::new(200_000, 0.01, 0.01, 31.0, 2.0),
        );
        assert_eq!(res.exit, ExitReason::Escaped);
        assert!(res.q[3] > 2.0 * PI, "φ = {}", res.q[3]);
    }
}
