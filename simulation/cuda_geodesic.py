# cuda_geodesic.py
import numpy as np
from numba import cuda, float64, int32, uint8
import math
import logging
logging.getLogger('numba').setLevel(logging.ERROR)

# ----------------------------------------------------------------------------
# Flat-space ray–tracing kernels (GPU)
# ----------------------------------------------------------------------------
# These utilities are used by simulation.background.save_no_gravity_image_with_background
# to render background-mapped rays on the GPU.  They do *not* integrate curved
# geodesics.  All previous GPU Schwarzschild integrator code has been removed –
# curved geodesics are now handled exclusively through EinsteinPy on the CPU to
# ensure numerical consistency with the reference library.
# ----------------------------------------------------------------------------

@cuda.jit
def _flat_raytrace_kernel(obs_pos, ray_dirs, boundary_radius,
                          patch_center_theta, patch_center_phi,
                          patch_size_theta, patch_size_phi,
                          flip_theta, flip_phi,
                          bg_array, h, w,
                          out_img):
    """GPU kernel: flat-space ray trace without trajectory capture.

    Each thread handles a single ray (pixel).
    * obs_pos        : (3,) observer Cartesian position (float64)
    * ray_dirs       : (N,3) unit direction vectors in Cartesian coords (float64)
    * boundary_radius: scalar – radius of outer sphere (float64)
    * patch_*        : spherical patch definition (float64)
    * bg_array       : (h,w,3) uint8 background image (device)
    * out_img        : (N,3) uint8 – RGB result for each ray (device)
    """
    i = cuda.grid(1)
    n = ray_dirs.shape[0]
    if i >= n:
        return

    # Observer position
    ox, oy, oz = obs_pos[0], obs_pos[1], obs_pos[2]
    # Direction
    dx, dy, dz = ray_dirs[i, 0], ray_dirs[i, 1], ray_dirs[i, 2]

    # ---------- Intersection of ray with sphere (quadratic) ----------
    a = dx * dx + dy * dy + dz * dz
    b = 2.0 * (ox * dx + oy * dy + oz * dz)
    c = ox * ox + oy * oy + oz * oz - boundary_radius * boundary_radius
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        # No intersection -> black pixel
        out_img[i, 0] = 0
        out_img[i, 1] = 0
        out_img[i, 2] = 0
        return
    t = (-b + math.sqrt(disc)) / (2.0 * a)
    hx = ox + t * dx
    hy = oy + t * dy
    hz = oz + t * dz

    # ---------- Convert hit-point to spherical coords ----------
    r = math.sqrt(hx * hx + hy * hy + hz * hz)
    theta = math.acos(hz / r)
    phi = math.atan2(hy, hx)

    # ---------- Check if hit point lies inside the user patch ----------
    theta0 = patch_center_theta - patch_size_theta * 0.5
    theta1 = patch_center_theta + patch_size_theta * 0.5
    phi0   = patch_center_phi   - patch_size_phi   * 0.5
    phi1   = patch_center_phi   + patch_size_phi   * 0.5

    # Handle wrapping in phi
    phi_span = (phi1 - phi0) % (2.0 * math.pi)
    if phi_span == 0.0:
        phi_span = 2.0 * math.pi
    # Normalise angles to [0, 2π)
    phi_mod     = phi % (2.0 * math.pi)
    phi0_mod    = phi0 % (2.0 * math.pi)
    phi1_mod    = phi1 % (2.0 * math.pi)

    in_phi_patch = False
    if phi0_mod <= phi1_mod:
        in_phi_patch = (phi_mod >= phi0_mod) and (phi_mod <= phi1_mod)
    else:
        in_phi_patch = (phi_mod >= phi0_mod) or (phi_mod <= phi1_mod)

    if (theta >= theta0) and (theta <= theta1) and in_phi_patch:
        # Map to background texture coordinates
        theta_map = (math.pi - theta) if flip_theta else theta
        phi_map   = (-phi)           if flip_phi  else phi

        u = int((theta_map - theta0) / (theta1 - theta0) * (h - 1))
        phi_map_mod = (phi_map - phi0) % (2.0 * math.pi)
        v = int(phi_map_mod / phi_span * (w - 1))

        # Clamp
        if u < 0:
            u = 0
        elif u > h - 1:
            u = h - 1
        if v < 0:
            v = 0
        elif v > w - 1:
            v = w - 1

        out_img[i, 0] = bg_array[u, v, 0]
        out_img[i, 1] = bg_array[u, v, 1]
        out_img[i, 2] = bg_array[u, v, 2]
    else:
        # Outside patch -> black pixel
        out_img[i, 0] = 0
        out_img[i, 1] = 0
        out_img[i, 2] = 0

# ------------------------------------------------------------
# Variant with trajectory capture for a subset of rays
# ------------------------------------------------------------
@cuda.jit
def _flat_raytrace_kernel_with_traj(obs_pos, ray_dirs, boundary_radius,
                                    patch_center_theta, patch_center_phi,
                                    patch_size_theta, patch_size_phi,
                                    flip_theta, flip_phi,
                                    bg_array, h, w,
                                    out_img,
                                    sampled_indices, n_sampled, n_points,
                                    out_trajs):
    """Kernel identical to `_flat_raytrace_kernel` but additionally records
    `n_points` equally-spaced samples along the ray for a predefined subset of
    indices (`sampled_indices`)."""
    i = cuda.grid(1)
    n = ray_dirs.shape[0]
    if i >= n:
        return

    # Observer position
    ox, oy, oz = obs_pos[0], obs_pos[1], obs_pos[2]
    dx, dy, dz = ray_dirs[i, 0], ray_dirs[i, 1], ray_dirs[i, 2]

    a = dx * dx + dy * dy + dz * dz
    b = 2.0 * (ox * dx + oy * dy + oz * dz)
    c = ox * ox + oy * oy + oz * oz - boundary_radius * boundary_radius
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        out_img[i, 0] = 0
        out_img[i, 1] = 0
        out_img[i, 2] = 0
        return
    t = (-b + math.sqrt(disc)) / (2.0 * a)
    hx = ox + t * dx
    hy = oy + t * dy
    hz = oz + t * dz

    # Record trajectory for sampled rays
    for sidx in range(n_sampled):
        if i == sampled_indices[sidx]:
            for p in range(n_points):
                alpha = p / (n_points - 1.0)
                px = ox + alpha * (hx - ox)
                py = oy + alpha * (hy - oy)
                pz = oz + alpha * (hz - oz)
                out_trajs[sidx, p, 0] = px
                out_trajs[sidx, p, 1] = py
                out_trajs[sidx, p, 2] = pz

    # ----- Map hit-point to texture (same code as above kernel) -----
    r = math.sqrt(hx * hx + hy * hy + hz * hz)
    theta = math.acos(hz / r)
    phi = math.atan2(hy, hx)

    theta0 = patch_center_theta - patch_size_theta * 0.5
    theta1 = patch_center_theta + patch_size_theta * 0.5
    phi0   = patch_center_phi   - patch_size_phi   * 0.5
    phi1   = patch_center_phi   + patch_size_phi   * 0.5
    phi_span = (phi1 - phi0) % (2.0 * math.pi)
    if phi_span == 0.0:
        phi_span = 2.0 * math.pi

    phi_mod  = phi % (2.0 * math.pi)
    phi0_mod = phi0 % (2.0 * math.pi)
    phi1_mod = phi1 % (2.0 * math.pi)

    in_phi_patch = False
    if phi0_mod <= phi1_mod:
        in_phi_patch = (phi_mod >= phi0_mod) and (phi_mod <= phi1_mod)
    else:
        in_phi_patch = (phi_mod >= phi0_mod) or (phi_mod <= phi1_mod)

    if (theta >= theta0) and (theta <= theta1) and in_phi_patch:
        theta_map = (math.pi - theta) if flip_theta else theta
        phi_map   = (-phi)           if flip_phi  else phi
        u = int((theta_map - theta0) / (theta1 - theta0) * (h - 1))
        phi_map_mod = (phi_map - phi0) % (2.0 * math.pi)
        v = int(phi_map_mod / phi_span * (w - 1))
        if u < 0:
            u = 0
        elif u > h - 1:
            u = h - 1
        if v < 0:
            v = 0
        elif v > w - 1:
            v = w - 1
        out_img[i, 0] = bg_array[u, v, 0]
        out_img[i, 1] = bg_array[u, v, 1]
        out_img[i, 2] = bg_array[u, v, 2]
    else:
        out_img[i, 0] = 0
        out_img[i, 1] = 0
        out_img[i, 2] = 0

# ---------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------

def cuda_flat_raytrace(obs_pos, ray_dirs, boundary_radius,
                       patch_center_theta, patch_center_phi,
                       patch_size_theta, patch_size_phi,
                       flip_theta, flip_phi,
                       bg_array):
    """GPU-accelerated flat-space ray tracing without trajectory capture.

    Parameters match those of the kernel.  Returns a flattened `(N,3)` uint8
    RGB array suitable for reshaping to `(H,W,3)` in the caller.
    """
    h, w, _ = bg_array.shape
    n = ray_dirs.shape[0]

    out_img = np.zeros((n, 3), dtype=np.uint8)

    # Copy arrays to device
    d_obs_pos = cuda.to_device(np.asarray(obs_pos, dtype=np.float64))
    d_ray_dirs = cuda.to_device(np.asarray(ray_dirs, dtype=np.float64))
    d_bg_array = cuda.to_device(np.asarray(bg_array, dtype=np.uint8))
    d_out_img = cuda.to_device(out_img)

    threadsperblock = 32
    blockspergrid = (n + (threadsperblock - 1)) // threadsperblock

    _flat_raytrace_kernel[blockspergrid, threadsperblock](
        d_obs_pos, d_ray_dirs, boundary_radius,
        patch_center_theta, patch_center_phi,
        patch_size_theta, patch_size_phi,
        flip_theta, flip_phi,
        d_bg_array, h, w,
        d_out_img
    )

    d_out_img.copy_to_host(out_img)
    return out_img

def cuda_flat_raytrace_with_trajectories(obs_pos, ray_dirs, boundary_radius,
                                         patch_center_theta, patch_center_phi,
                                         patch_size_theta, patch_size_phi,
                                         flip_theta, flip_phi,
                                         bg_array,
                                         sampled_indices, n_points=100):
    """Same as `cuda_flat_raytrace` but additionally records *n_points* along each
    ray listed in *sampled_indices* (list of flat indices).  Returns the image
    and an array of sampled trajectories with shape `(len(sampled_indices),
    n_points, 3)`.
    """
    h, w, _ = bg_array.shape
    n = ray_dirs.shape[0]
    n_sampled = len(sampled_indices)

    out_img = np.zeros((n, 3), dtype=np.uint8)
    out_trajs = np.zeros((n_sampled, n_points, 3), dtype=np.float64)

    # Device copies
    d_obs_pos = cuda.to_device(np.asarray(obs_pos, dtype=np.float64))
    d_ray_dirs = cuda.to_device(np.asarray(ray_dirs, dtype=np.float64))
    d_bg_array = cuda.to_device(np.asarray(bg_array, dtype=np.uint8))
    d_out_img = cuda.to_device(out_img)
    d_sampled_indices = cuda.to_device(np.asarray(sampled_indices, dtype=np.int32))
    d_out_trajs = cuda.to_device(out_trajs)

    threadsperblock = 32
    blockspergrid = (n + (threadsperblock - 1)) // threadsperblock

    _flat_raytrace_kernel_with_traj[blockspergrid, threadsperblock](
        d_obs_pos, d_ray_dirs, boundary_radius,
        patch_center_theta, patch_center_phi,
        patch_size_theta, patch_size_phi,
        flip_theta, flip_phi,
        d_bg_array, h, w,
        d_out_img,
        d_sampled_indices, n_sampled, n_points,
        d_out_trajs
    )

    d_out_img.copy_to_host(out_img)
    d_out_trajs.copy_to_host(out_trajs)
    return out_img, out_trajs

# =============================================================================
# GPU Schwarzschild null-geodesic integrator (curved rays)
# =============================================================================
# These routines mirror EinsteinPy's GeodesicIntegrator logic but run entirely
# on the GPU with fixed-step (delta) integration.  They are used by
# simulation.raytracing when *use_cuda* is True.
# =============================================================================

# ------------------------- Helper: Christoffel symbols ------------------------
@cuda.jit(device=True)
def _schw_christoffel(q, rs, Gamma):
    """Compute Christoffel symbols Γ^a_{bc} for Schwarzschild metric in
    spherical coordinates (t,r,θ,φ).  Only non-zero components are filled."""
    r = q[1]
    th = q[2]
    if r <= rs:
        r = rs + 1e-12  # avoid divide-by-zero
    sin_th = math.sin(th)
    # Zero all elements first
    for a in range(4):
        for b in range(4):
            for c in range(4):
                Gamma[a, b, c] = 0.0
    # t components
    Gamma[0, 1, 0] = rs / (2.0 * r * (r - rs))
    Gamma[0, 0, 1] = Gamma[0, 1, 0]
    # r components
    Gamma[1, 0, 0] = (r - rs) / (2.0 * r**3)
    Gamma[1, 1, 1] = -rs / (2.0 * r * (r - rs))
    Gamma[1, 2, 2] = -(r - rs)
    Gamma[1, 3, 3] = -(r - rs) * sin_th**2
    # θ components
    Gamma[2, 1, 2] = 1.0 / r
    Gamma[2, 2, 1] = Gamma[2, 1, 2]
    Gamma[2, 3, 3] = -sin_th * math.cos(th)
    # φ components
    Gamma[3, 1, 3] = 1.0 / r
    Gamma[3, 3, 1] = Gamma[3, 1, 3]
    Gamma[3, 2, 3] = math.cos(th) / sin_th
    Gamma[3, 3, 2] = Gamma[3, 2, 3]

# ------------------------- Geodesic RHS (Hamilton) ---------------------------
@cuda.jit(device=True)
def _geodesic_rhs(q, p, rs, dqdt, dpdt):
    # Position ODE: dq^a/dλ = p^a
    for a in range(4):
        dqdt[a] = p[a]
    # Momentum ODE: dp^a/dλ = -Γ^a_{bc} p^b p^c
    Gamma = cuda.local.array((4, 4, 4), dtype=float64)
    _schw_christoffel(q, rs, Gamma)
    for a in range(4):
        dpdt[a] = 0.0
        for b in range(4):
            for c in range(4):
                dpdt[a] -= Gamma[a, b, c] * p[b] * p[c]

# ------------------------- Integration kernels -------------------------------
@cuda.jit
def _integrate_batch(q0s, p0s, steps, delta, rs, out_qs, out_ps):
    """Integrate *steps* Euler steps of the geodesic equations for each ray."""
    i = cuda.grid(1)
    n = q0s.shape[0]
    if i >= n:
        return
    q = cuda.local.array(4, dtype=float64)
    p = cuda.local.array(4, dtype=float64)
    dqdt = cuda.local.array(4, dtype=float64)
    dpdt = cuda.local.array(4, dtype=float64)
    for a in range(4):
        q[a] = q0s[i, a]
        p[a] = p0s[i, a]
    for _ in range(steps):
        _geodesic_rhs(q, p, rs, dqdt, dpdt)
        for a in range(4):
            q[a] += delta * dqdt[a]
            p[a] += delta * dpdt[a]
    for a in range(4):
        out_qs[i, a] = q[a]
        out_ps[i, a] = p[a]

@cuda.jit
def _integrate_batch_full(q0s, p0s, steps, delta, rs, out_qs_traj):
    i = cuda.grid(1)
    n = q0s.shape[0]
    if i >= n:
        return
    q = cuda.local.array(4, dtype=float64)
    p = cuda.local.array(4, dtype=float64)
    dqdt = cuda.local.array(4, dtype=float64)
    dpdt = cuda.local.array(4, dtype=float64)
    for a in range(4):
        q[a] = q0s[i, a]
        p[a] = p0s[i, a]
    for s in range(steps):
        # Store position each step
        for a in range(4):
            out_qs_traj[i, s, a] = q[a]
        _geodesic_rhs(q, p, rs, dqdt, dpdt)
        for a in range(4):
            q[a] += delta * dqdt[a]
            p[a] += delta * dpdt[a]

# ------------------------- Public wrapper class -----------------------------
class CUDASchwarzschildIntegrator:
    """Simple fixed-step GPU integrator for null geodesics in Schwarzschild
    spacetime (G=c=M=1)."""

    def __init__(self, steps=500, delta=0.2, mass=1.0):
        self.steps = steps
        self.delta = delta
        self.rs = 2.0 * mass

    def integrate_batch(self, q0s, p0s):
        n = q0s.shape[0]
        out_qs = np.zeros_like(q0s)
        out_ps = np.zeros_like(p0s)
        threads = 32
        blocks = (n + (threads - 1)) // threads
        _integrate_batch[blocks, threads](q0s, p0s, self.steps, self.delta, self.rs, out_qs, out_ps)
        return out_qs, out_ps

    def integrate_batch_full(self, q0s, p0s):
        n = q0s.shape[0]
        out_qs_traj = np.zeros((n, self.steps, 4), dtype=np.float64)
        threads = 32
        blocks = (n + (threads - 1)) // threads
        _integrate_batch_full[blocks, threads](q0s, p0s, self.steps, self.delta, self.rs, out_qs_traj)
        return out_qs_traj

# ------------------------- Helper: null 4-momentum ---------------------------
def compute_null_4momentum_schwarzschild(q, p_spatial):
    """Given spatial momentum (p^r, p^θ, p^φ) compute p^t so that g_{ab}p^a p^b = 0."""
    r = q[1]
    th = q[2]
    if r <= 2.0:
        r = 2.0 + 1e-8
    g_tt = -(1.0 - 2.0 / r)
    g_rr = 1.0 / (1.0 - 2.0 / r)
    g_thth = r * r
    g_phph = r * r * math.sin(th) ** 2
    pr, pth, pph = p_spatial
    C = g_rr * pr * pr + g_thth * pth * pth + g_phph * pph * pph
    # Null condition: g_tt (p^t)^2 + C = 0  ->  p^t = ±sqrt(-C/g_tt)
    pt = math.sqrt(max(0.0, -C / g_tt))
    return [-pt, pr, pth, pph]  # negative root for consistency with EPy

# -----------------------------------------------------------------------------
#                      FANTASY ORDER-2 SYMPLECTIC INTEGRATOR
# -----------------------------------------------------------------------------
# Mirrors einsteinpy.integrators.fantasy.GeodesicIntegrator (order-2) but with
# analytic Schwarzschild metric, therefore no auto-diff is required.
# -----------------------------------------------------------------------------

@cuda.jit(device=True)
def _metric_contravariant(q, rs, g):
    """Fill 4×4 contravariant Schwarzschild metric in *g*."""
    t = q[0]  # unused but kept for completeness
    r = q[1]
    th = q[2]
    for a in range(4):
        for b in range(4):
            g[a, b] = 0.0
    inv_fac = 1.0 - rs / r / 1.0  # rs = 2M ; but rs passed directly
    inv_fac = 1.0 - rs / r
    g[0, 0] = -1.0 / inv_fac
    g[1, 1] = inv_fac
    g[2, 2] = 1.0 / (r * r)
    sin_th = math.sin(th)
    g[3, 3] = 1.0 / ((r * sin_th) ** 2)

@cuda.jit(device=True)
def _metric_derivative(q, rs, wrt, gprime):
    """Derivative ∂g^{ab}/∂q^{wrt}.  Only r- and θ-derivatives are non-zero."""
    r = q[1]
    th = q[2]
    # zero out first
    for a in range(4):
        for b in range(4):
            gprime[a, b] = 0.0
    inv_fac = 1.0 - rs / r
    if wrt == 1:  # derivative w.r.t r
        # d g^{tt}/dr = 2 / (r - rs)^2
        denom = (r - rs)
        gprime[0, 0] = 2.0 / (denom * denom)
        # d g^{rr}/dr = 2/r^2
        gprime[1, 1] = 2.0 / (r * r)
        # d g^{θθ}/dr = -2 / r^3
        gprime[2, 2] = -2.0 / (r * r * r)
        # d g^{φφ}/dr = -2 /(r^3 sin^2 θ)
        sin2 = math.sin(th) ** 2
        gprime[3, 3] = -2.0 / (r * r * r * sin2)
    elif wrt == 2:  # derivative w.r.t θ
        sin_th = math.sin(th)
        cos_th = math.cos(th)
        # only g^{φφ} depends on θ
        gprime[3, 3] = (-2.0 * cos_th) / ( (r * r) * sin_th ** 3 )

@cuda.jit(device=True)
def _part_ham_flow(q, p, rs, wrt):
    """0.5 * (∂g / ∂q^{wrt})_{ab} p^a p^b ."""
    Gp = cuda.local.array((4,4), dtype=float64)
    _metric_derivative(q, rs, wrt, Gp)
    acc = 0.0
    for a in range(4):
        for b in range(4):
            val = Gp[a, b]
            if val != 0.0:
                acc += val * p[a] * p[b]
    return 0.5 * acc

@cuda.jit(device=True)
def _metric_vec_mul(q, p, rs, out):
    """out = g^{ab} p_b (contravariant metric times contravariant mom)."""
    G = cuda.local.array((4,4), dtype=float64)
    _metric_contravariant(q, rs, G)
    for a in range(4):
        acc = 0.0
        for b in range(4):
            acc += G[a, b] * p[b]
        out[a] = acc

@cuda.jit(device=True)
def _flow_A_dev(q1, p1, q2, p2, delta, rs):
    # Compute dH1 vector (size 4)
    dH1 = cuda.local.array(4, dtype=float64)
    for i in range(4):
        dH1[i] = _part_ham_flow(q1, p2, rs, i)
    # Update p1
    for i in range(4):
        p1[i] = p1[i] - delta * dH1[i]
    # Update q2
    dq2 = cuda.local.array(4, dtype=float64)
    _metric_vec_mul(q1, p2, rs, dq2)
    for i in range(4):
        q2[i] = q2[i] + delta * dq2[i]

@cuda.jit(device=True)
def _flow_B_dev(q1, p1, q2, p2, delta, rs):
    dH2 = cuda.local.array(4, dtype=float64)
    for i in range(4):
        dH2[i] = _part_ham_flow(q2, p1, rs, i)
    for i in range(4):
        p2[i] = p2[i] - delta * dH2[i]
    dq1 = cuda.local.array(4, dtype=float64)
    _metric_vec_mul(q2, p1, rs, dq1)
    for i in range(4):
        q1[i] = q1[i] + delta * dq1[i]

@cuda.jit(device=True)
def _flow_mixed_dev(q1, p1, q2, p2, delta, omega):
    q_sum0 = q1[0] + q2[0]
    q_sum1 = q1[1] + q2[1]
    q_sum2 = q1[2] + q2[2]
    q_sum3 = q1[3] + q2[3]
    q_dif0 = q1[0] - q2[0]
    q_dif1 = q1[1] - q2[1]
    q_dif2 = q1[2] - q2[2]
    q_dif3 = q1[3] - q2[3]
    p_sum0 = p1[0] + p2[0]
    p_sum1 = p1[1] + p2[1]
    p_sum2 = p1[2] + p2[2]
    p_sum3 = p1[3] + p2[3]
    p_dif0 = p1[0] - p2[0]
    p_dif1 = p1[1] - p2[1]
    p_dif2 = p1[2] - p2[2]
    p_dif3 = p1[3] - p2[3]
    cos = math.cos(2.0 * omega * delta)
    sin = math.sin(2.0 * omega * delta)
    # q1 next
    q1[0] = 0.5 * (q_sum0 + q_dif0 * cos + p_dif0 * sin)
    q1[1] = 0.5 * (q_sum1 + q_dif1 * cos + p_dif1 * sin)
    q1[2] = 0.5 * (q_sum2 + q_dif2 * cos + p_dif2 * sin)
    q1[3] = 0.5 * (q_sum3 + q_dif3 * cos + p_dif3 * sin)
    # p1 next
    p1[0] = 0.5 * (p_sum0 + p_dif0 * cos - q_dif0 * sin)
    p1[1] = 0.5 * (p_sum1 + p_dif1 * cos - q_dif1 * sin)
    p1[2] = 0.5 * (p_sum2 + p_dif2 * cos - q_dif2 * sin)
    p1[3] = 0.5 * (p_sum3 + p_dif3 * cos - q_dif3 * sin)
    # q2 next
    q2[0] = 0.5 * (q_sum0 - q_dif0 * cos - p_dif0 * sin)
    q2[1] = 0.5 * (q_sum1 - q_dif1 * cos - p_dif1 * sin)
    q2[2] = 0.5 * (q_sum2 - q_dif2 * cos - p_dif2 * sin)
    q2[3] = 0.5 * (q_sum3 - q_dif3 * cos - p_dif3 * sin)
    # p2 next
    p2[0] = 0.5 * (p_sum0 - p_dif0 * cos + q_dif0 * sin)
    p2[1] = 0.5 * (p_sum1 - p_dif1 * cos + q_dif1 * sin)
    p2[2] = 0.5 * (p_sum2 - p_dif2 * cos + q_dif2 * sin)
    p2[3] = 0.5 * (p_sum3 - p_dif3 * cos + q_dif3 * sin)

@cuda.jit(device=True)
def _fantasy_step_ord2(q1, p1, q2, p2, delta, rs, omega):
    # Make local copies? (arrays are passed by reference already)
    _flow_A_dev(q1, p1, q2, p2, 0.5 * delta, rs)
    _flow_B_dev(q1, p1, q2, p2, 0.5 * delta, rs)
    _flow_mixed_dev(q1, p1, q2, p2, delta, omega)
    _flow_B_dev(q1, p1, q2, p2, 0.5 * delta, rs)
    _flow_A_dev(q1, p1, q2, p2, 0.5 * delta, rs)

# ------------------------- Integration kernels -------------------------------

@cuda.jit
def fantasy_integrate_batch_ord2(q0s, p0s, steps, delta, rs, r_max, omega, out_qs):
    i = cuda.grid(1)
    n = q0s.shape[0]
    if i >= n:
        return
    # local copies
    q1 = cuda.local.array(4, dtype=float64)
    p1 = cuda.local.array(4, dtype=float64)
    q2 = cuda.local.array(4, dtype=float64)
    p2 = cuda.local.array(4, dtype=float64)
    for a in range(4):
        q1[a] = q0s[i, a]
        p1[a] = p0s[i, a]
        q2[a] = q0s[i, a]
        p2[a] = p0s[i, a]
    for _ in range(steps):
        # early exit if outside domain
        if q1[1] <= 1.1 * rs or q1[1] >= r_max:
            break
        _fantasy_step_ord2(q1, p1, q2, p2, delta, rs, omega)
    for a in range(4):
        out_qs[i, a] = q1[a]  # return first copy's position

@cuda.jit
def fantasy_integrate_batch_ord2_full(q0s, p0s, steps, delta, rs, r_max, omega, out_qs_traj):
    i = cuda.grid(1)
    n = q0s.shape[0]
    if i >= n:
        return
    q1 = cuda.local.array(4, dtype=float64)
    p1 = cuda.local.array(4, dtype=float64)
    q2 = cuda.local.array(4, dtype=float64)
    p2 = cuda.local.array(4, dtype=float64)
    for a in range(4):
        q1[a] = q0s[i, a]
        p1[a] = p0s[i, a]
        q2[a] = q0s[i, a]
        p2[a] = p0s[i, a]
    for step in range(steps):
        for a in range(4):
            out_qs_traj[i, step, a] = q1[a]
        # stop storing if out of domain
        if q1[1] <= 1.1 * rs or q1[1] >= r_max:
            break
        _fantasy_step_ord2(q1, p1, q2, p2, delta, rs, omega)

# ------------------------- Wrapper class overwrite ---------------------------

class CUDASchwarzschildIntegrator:
    """GPU Fantasy order-2 symplectic integrator for null geodesics."""

    def __init__(self, steps=500, delta=0.2, mass=1.0, omega=1.0, r_max=1e6):
        self.steps = steps
        self.delta = delta
        self.rs = 2.0 * mass
        self.omega = omega
        self.r_max = r_max

    def integrate_batch(self, q0s, p0s):
        n = q0s.shape[0]
        out_qs = np.zeros((n, 4), dtype=np.float64)
        threads = 32
        blocks = (n + threads - 1) // threads
        fantasy_integrate_batch_ord2[blocks, threads](q0s, p0s, self.steps, self.delta, self.rs, self.r_max, self.omega, out_qs)
        return out_qs, None

    def integrate_batch_full(self, q0s, p0s):
        n = q0s.shape[0]
        out_qs_traj = np.zeros((n, self.steps, 4), dtype=np.float64)
        threads = 32
        blocks = (n + threads - 1) // threads
        fantasy_integrate_batch_ord2_full[blocks, threads](q0s, p0s, self.steps, self.delta, self.rs, self.r_max, self.omega, out_qs_traj)
        return out_qs_traj 