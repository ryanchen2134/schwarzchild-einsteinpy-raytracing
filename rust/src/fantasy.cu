// FANTASY order-2 symplectic integrator for Schwarzschild null geodesics.
//
// CUDA C port of the kernel in simulation/cuda_geodesic.py, with the same
// operation order as the Rust scalar integrator (integrator.rs). Compiled at
// start-up with NVRTC (see cuda.rs). All quantities are in units of M: the host
// rescales every ray so that rs == 2, and scales t, r and p_phi back afterwards.
// Momenta are covariant: dq^mu/dlambda = g^{mu nu} p_nu.
//
// The kernel is resumable: each launch advances every unfinished ray by at most
// round_steps steps and stores the full state (both FANTASY copies, status, step
// count) back into the buffers, so the host can bound a launch's run time (display
// GPUs kill long kernels) and report progress between rounds.
//
// Exit status codes match integrator::ExitReason: 0 still running / step limit,
// 1 captured, 2 escaped, 3 diverged.

typedef double real;

__device__ __forceinline__ void metric_diag(const real* q, real rs, real* g)
{
    const real r = q[1];
    const real th = q[2];
    const real inv_fac = 1.0 - rs / r;
    const real r_sin = r * sin(th);
    g[0] = -1.0 / inv_fac;
    g[1] = inv_fac;
    g[2] = 1.0 / (r * r);
    g[3] = 1.0 / (r_sin * r_sin);
}

// d g^{mu mu} / d q^{wrt}; only wrt = 1 (r) and wrt = 2 (theta) are non-zero.
__device__ __forceinline__ void metric_derivative_diag(const real* q, real rs, int wrt, real* gp)
{
    const real r = q[1];
    const real th = q[2];
    gp[0] = 0.0; gp[1] = 0.0; gp[2] = 0.0; gp[3] = 0.0;
    if (wrt == 1) {
        const real denom = r - rs;
        const real sin_th = sin(th);
        const real sin2 = sin_th * sin_th;
        gp[0] = rs / (denom * denom);
        gp[1] = rs / (r * r);
        gp[2] = -2.0 / (r * r * r);
        gp[3] = -2.0 / (r * r * r * sin2);
    } else if (wrt == 2) {
        const real sin_th = sin(th);
        const real cos_th = cos(th);
        gp[3] = (-2.0 * cos_th) / ((r * r) * (sin_th * sin_th * sin_th));
    }
}

// 1/2 (d g^{mu nu} / d q^{wrt}) p_mu p_nu
__device__ __forceinline__ real part_ham_flow(const real* q, const real* p, real rs, int wrt)
{
    real gp[4];
    metric_derivative_diag(q, rs, wrt, gp);
    real acc = 0.0;
    for (int a = 0; a < 4; ++a) {
        const real val = gp[a];
        if (val != 0.0) {
            acc += val * p[a] * p[a];
        }
    }
    return 0.5 * acc;
}

__device__ __forceinline__ void metric_vec_mul(const real* q, const real* p, real rs, real* out)
{
    real g[4];
    metric_diag(q, rs, g);
    out[0] = g[0] * p[0];
    out[1] = g[1] * p[1];
    out[2] = g[2] * p[2];
    out[3] = g[3] * p[3];
}

__device__ __forceinline__ void flow_a(real* q1, real* p1, real* q2, real* p2, real delta, real rs)
{
    real dh1[4];
    for (int i = 0; i < 4; ++i) dh1[i] = part_ham_flow(q1, p2, rs, i);
    for (int i = 0; i < 4; ++i) p1[i] = p1[i] - delta * dh1[i];
    real dq2[4];
    metric_vec_mul(q1, p2, rs, dq2);
    for (int i = 0; i < 4; ++i) q2[i] = q2[i] + delta * dq2[i];
}

__device__ __forceinline__ void flow_b(real* q1, real* p1, real* q2, real* p2, real delta, real rs)
{
    real dh2[4];
    for (int i = 0; i < 4; ++i) dh2[i] = part_ham_flow(q2, p1, rs, i);
    for (int i = 0; i < 4; ++i) p2[i] = p2[i] - delta * dh2[i];
    real dq1[4];
    metric_vec_mul(q2, p1, rs, dq1);
    for (int i = 0; i < 4; ++i) q1[i] = q1[i] + delta * dq1[i];
}

__device__ __forceinline__ void flow_mixed(real* q1, real* p1, real* q2, real* p2, real c, real s)
{
    for (int k = 0; k < 4; ++k) {
        const real q_sum = q1[k] + q2[k];
        const real q_dif = q1[k] - q2[k];
        const real p_sum = p1[k] + p2[k];
        const real p_dif = p1[k] - p2[k];
        q1[k] = 0.5 * (q_sum + q_dif * c + p_dif * s);
        p1[k] = 0.5 * (p_sum + p_dif * c - q_dif * s);
        q2[k] = 0.5 * (q_sum - q_dif * c - p_dif * s);
        p2[k] = 0.5 * (p_sum - p_dif * c + q_dif * s);
    }
}

__device__ __forceinline__ void fantasy_step(real* q1, real* p1, real* q2, real* p2,
                                             real half_delta, real rs, real cos_m, real sin_m)
{
    flow_a(q1, p1, q2, p2, half_delta, rs);
    flow_b(q1, p1, q2, p2, half_delta, rs);
    flow_mixed(q1, p1, q2, p2, cos_m, sin_m);
    flow_b(q1, p1, q2, p2, half_delta, rs);
    flow_a(q1, p1, q2, p2, half_delta, rs);
}

__device__ __forceinline__ int exit_reason(const real* q1, const real* q2, real rs, real r_capture, real r_max)
{
    const real r1 = q1[1];
    const real r2 = q2[1];
    if (!isfinite(r1) || r1 <= 0.0 || !isfinite(r2) || r2 <= rs) return 3;
    if (r1 <= r_capture) return 1;
    if (r1 >= r_max) return 2;
    return 0;
}

// q1s, p1s, q2s, p2s: both phase-space copies, 4 doubles per ray, read and written.
// status, n_steps: per ray, read and written. A ray with status != 0 or
// n_steps >= steps is finished and left untouched.
extern "C" __global__ void fantasy_integrate(real* q1s, real* p1s, real* q2s, real* p2s,
                                             int* status, int* n_steps, int n, int steps, int round_steps,
                                             real delta, real rs, real r_max, real r_capture, real omega)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (status[i] != 0) return;
    int done = n_steps[i];
    if (done >= steps) return;

    real q1[4], p1[4], q2[4], p2[4];
    for (int a = 0; a < 4; ++a) {
        q1[a] = q1s[4 * i + a];
        p1[a] = p1s[4 * i + a];
        q2[a] = q2s[4 * i + a];
        p2[a] = p2s[4 * i + a];
    }
    const real half = 0.5 * delta;
    real sin_m, cos_m;
    sincos(2.0 * omega * delta, &sin_m, &cos_m);

    int st = 0;
    const int end = (steps - done < round_steps) ? steps : done + round_steps;
    for (; done < end; ++done) {
        st = exit_reason(q1, q2, rs, r_capture, r_max);
        if (st != 0) break;
        fantasy_step(q1, p1, q2, p2, half, rs, cos_m, sin_m);
    }
    if (st == 0 && done >= steps) {
        // Budget exhausted: a boundary crossed on the last permitted step still counts.
        st = exit_reason(q1, q2, rs, r_capture, r_max);
    }
    for (int a = 0; a < 4; ++a) {
        q1s[4 * i + a] = q1[a];
        p1s[4 * i + a] = p1[a];
        q2s[4 * i + a] = q2[a];
        p2s[4 * i + a] = p2[a];
    }
    status[i] = st;
    n_steps[i] = done;
}
