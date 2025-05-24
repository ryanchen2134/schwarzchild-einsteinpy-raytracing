import numpy as np
from numba import cuda, float64, int32, uint8
import math
import logging
logging.getLogger('numba').setLevel(logging.ERROR)

# Constants for Schwarzschild metric (G = c = M = 1)
@cuda.jit(device=True)
def schwarzschild_christoffel(q, rs, Gamma):
    """Compute Christoffel symbols for Schwarzschild metric.

    Parameters
    ----------
    q : array-like
        4-position (t, r, theta, phi)
    rs : float64
        Schwarzschild radius  (2M)  in geometrized units.
    Gamma : 4x4x4 array
        Output array (filled in-place).
    """
    r = q[1]
    th = q[2]
    sin_th = math.sin(th)
    cos_th = math.cos(th)
    # Only nonzero Christoffel symbols in Schwarzschild (spherical coordinates)
    # Fill Gamma^a_{bc} in-place
    for a in range(4):
        for b in range(4):
            for c in range(4):
                Gamma[a, b, c] = 0.0
    # Nonzero components (see e.g. Carroll, eqn 5.50)
    # t
    Gamma[0, 1, 0] = rs / (2. * r * (r - rs))
    Gamma[0, 0, 1] = Gamma[0, 1, 0]
    # r
    Gamma[1, 0, 0] = (r - rs) / (2. * r**3)
    Gamma[1, 1, 1] = -rs / (2. * r * (r - rs))
    Gamma[1, 2, 2] = -(r - rs)
    Gamma[1, 3, 3] = -(r - rs) * (math.sin(th)**2)
    # theta
    Gamma[2, 1, 2] = 1. / r
    Gamma[2, 2, 1] = Gamma[2, 1, 2]
    Gamma[2, 3, 3] = -math.sin(th) * math.cos(th)
    # phi
    Gamma[3, 1, 3] = 1. / r
    Gamma[3, 3, 1] = Gamma[3, 1, 3]
    Gamma[3, 2, 3] = math.cos(th) / math.sin(th)
    Gamma[3, 3, 2] = Gamma[3, 2, 3]

@cuda.jit(device=True)
def geodesic_rhs(q, p, rs, dqdt, dpdt):
    # dqdt^a = p^a
    for a in range(4):
        dqdt[a] = p[a]
    # dpdt^a = -Gamma^a_{bc} p^b p^c
    Gamma = cuda.local.array((4, 4, 4), dtype=float64)
    schwarzschild_christoffel(q, rs, Gamma)
    for a in range(4):
        dpdt[a] = 0.0
        for b in range(4):
            for c in range(4):
                dpdt[a] -= Gamma[a, b, c] * p[b] * p[c]

@cuda.jit
def integrate_schwarzschild_batch(q0s, p0s, steps, delta, rs, out_qs, out_ps):
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
        geodesic_rhs(q, p, rs, dqdt, dpdt)
        for a in range(4):
            q[a] += delta * dqdt[a]
            p[a] += delta * dpdt[a]
    for a in range(4):
        out_qs[i, a] = q[a]
        out_ps[i, a] = p[a]

@cuda.jit
def integrate_schwarzschild_batch_full(q0s, p0s, steps, delta, rs, out_qs_traj, out_ps_traj):
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
        for a in range(4):
            out_qs_traj[i, s, a] = q[a]
            out_ps_traj[i, s, a] = p[a]
        geodesic_rhs(q, p, rs, dqdt, dpdt)
        for a in range(4):
            q[a] += delta * dqdt[a]
            p[a] += delta * dpdt[a]

class CUDASchwarzschildIntegrator:
    """GPU Schwarzschild null-geodesic integrator.

    All distances are expressed in units of the chosen mass (i.e. M).  Internally we
    pass the Schwarzschild radius ``r_s = 2 M`` to the kernel so the same code works
    for any mass.
    """
    def __init__(self, steps=500, delta=0.2, mass=1.0):
        self.steps = steps
        self.delta = delta
        self.mass = mass
        self.rs = 2.0 * mass
    def integrate_batch(self, q0s, p0s):
        n = q0s.shape[0]
        out_qs = np.zeros_like(q0s)
        out_ps = np.zeros_like(p0s)
        threadsperblock = 32
        blockspergrid = (n + (threadsperblock - 1)) // threadsperblock
        integrate_schwarzschild_batch[blockspergrid, threadsperblock](
            q0s, p0s, self.steps, self.delta, self.rs, out_qs, out_ps
        )
        return out_qs, out_ps
    def integrate_batch_full_trajectory(self, q0s, p0s):
        n = q0s.shape[0]
        steps = self.steps
        out_qs_traj = np.zeros((n, steps, 4), dtype=np.float64)
        out_ps_traj = np.zeros((n, steps, 4), dtype=np.float64)
        threadsperblock = 32
        blockspergrid = (n + (threadsperblock - 1)) // threadsperblock
        integrate_schwarzschild_batch_full[blockspergrid, threadsperblock](
            q0s, p0s, steps, self.delta, self.rs, out_qs_traj, out_ps_traj
        )
        return out_qs_traj, out_ps_traj

# Helper: Convert (theta, phi) in observer frame to global momentum (if needed)
def observer_angles_to_global_momentum(observer_pos, theta, phi):
    # For now, assume observer is at (x, 0, 0), BH at (0, 0, 0)
    # The direction vector in global frame is:
    dir_vec = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])
    # Optionally rotate if observer is not on x-axis
    return dir_vec

# Helper: Compute null 4-momentum for Schwarzschild metric (in M=G=c=1 units)
def compute_null_4momentum_schwarzschild(q, p_spatial):
    # q: [t, r, theta, phi]
    # p_spatial: [p^r, p^theta, p^phi]
    r, th = q[1], q[2]
    # Contravariant metric components
    g_tt = -1.0 / (1.0 - 2.0 / r)
    g_rr = 1.0 - 2.0 / r
    g_thth = 1.0 / (r ** 2)
    g_phph = 1.0 / ((r * math.sin(th)) ** 2)
    pr, pth, pph = p_spatial
    # For Schwarzschild, g[0,3] = 0, so B = 0
    A = g_tt
    C = g_rr * pr**2 + g_thth * pth**2 + g_phph * pph**2
    if A == 0:
        raise ValueError("g_tt is zero at r=2M; cannot compute null 4-momentum.")
    pt = -math.sqrt(-C / A)  # Use negative root for consistency with EinsteinPy
    return [pt, pr, pth, pph]

@cuda.jit
def flat_raytrace_kernel_with_traj(obs_pos, ray_dirs, boundary_radius, patch_center_theta, patch_center_phi, patch_size_theta, patch_size_phi, flip_theta, flip_phi, bg_array, h, w, out_img, sampled_indices, n_sampled, n_points, out_trajs):
    i = cuda.grid(1)
    n = ray_dirs.shape[0]
    if i >= n:
        return
    # Unpack observer position
    ox, oy, oz = obs_pos[0], obs_pos[1], obs_pos[2]
    dx, dy, dz = ray_dirs[i, 0], ray_dirs[i, 1], ray_dirs[i, 2]
    # Ray-sphere intersection
    a = dx*dx + dy*dy + dz*dz
    b = 2 * (ox*dx + oy*dy + oz*dz)
    c = ox*ox + oy*oy + oz*oz - boundary_radius*boundary_radius
    disc = b*b - 4*a*c
    if disc < 0:
        out_img[i, 0] = 0
        out_img[i, 1] = 0
        out_img[i, 2] = 0
        return
    t = (-b + math.sqrt(disc)) / (2*a)
    hx = ox + t*dx
    hy = oy + t*dy
    hz = oz + t*dz
    r = math.sqrt(hx*hx + hy*hy + hz*hz)
    theta = math.acos(hz / r)
    phi = math.atan2(hy, hx)
    # Patch bounds
    theta0 = patch_center_theta - patch_size_theta/2
    theta1 = patch_center_theta + patch_size_theta/2
    phi0 = patch_center_phi - patch_size_phi/2
    phi1 = patch_center_phi + patch_size_phi/2
    phi_span = (phi1 - phi0) % (2 * math.pi)
    if phi_span == 0:
        phi_span = 2 * math.pi
    # _in_phi_patch logic
    phi_mod = phi % (2 * math.pi)
    phi0_mod = phi0 % (2 * math.pi)
    phi1_mod = phi1 % (2 * math.pi)
    in_patch = False
    if phi0_mod <= phi1_mod:
        in_patch = (phi_mod >= phi0_mod) and (phi_mod <= phi1_mod)
    else:
        in_patch = (phi_mod >= phi0_mod) or (phi_mod <= phi1_mod)
    if (theta >= theta0) and (theta <= theta1) and in_patch:
        theta_map = (math.pi - theta) if flip_theta else theta
        phi_map = (-phi) if flip_phi else phi
        u = int((theta_map - theta0) / (theta1 - theta0) * (h - 1))
        phi_map_mod = (phi_map - phi0) % (2 * math.pi)
        v = int(phi_map_mod / phi_span * (w - 1))
        if u < 0:
            u = 0
        if u > h - 1:
            u = h - 1
        if v < 0:
            v = 0
        if v > w - 1:
            v = w - 1
        out_img[i, 0] = bg_array[u, v, 0]
        out_img[i, 1] = bg_array[u, v, 1]
        out_img[i, 2] = bg_array[u, v, 2]
    else:
        out_img[i, 0] = 0
        out_img[i, 1] = 0
        out_img[i, 2] = 0
    # If this ray is a sampled index, fill its trajectory
    for sidx in range(n_sampled):
        if i == sampled_indices[sidx]:
            for p in range(n_points):
                alpha = p / (n_points - 1)
                px = ox + alpha * (hx - ox)
                py = oy + alpha * (hy - oy)
                pz = oz + alpha * (hz - oz)
                out_trajs[sidx, p, 0] = px
                out_trajs[sidx, p, 1] = py
                out_trajs[sidx, p, 2] = pz

def cuda_flat_raytrace_with_trajectories(obs_pos, ray_dirs, boundary_radius, patch_center_theta, patch_center_phi, patch_size_theta, patch_size_phi, flip_theta, flip_phi, bg_array, sampled_indices, n_points=100):
    # obs_pos: (3,), ray_dirs: (N,3), bg_array: (h,w,3), sampled_indices: (n_sampled,)
    import numpy as np
    from numba import cuda
    h, w, _ = bg_array.shape
    n = ray_dirs.shape[0]
    n_sampled = len(sampled_indices)
    out_img = np.zeros((n, 3), dtype=np.uint8)
    out_trajs = np.zeros((n_sampled, n_points, 3), dtype=np.float64)
    # Copy arrays to device
    d_obs_pos = cuda.to_device(np.asarray(obs_pos, dtype=np.float64))
    d_ray_dirs = cuda.to_device(np.asarray(ray_dirs, dtype=np.float64))
    d_bg_array = cuda.to_device(np.asarray(bg_array, dtype=np.uint8))
    d_out_img = cuda.to_device(out_img)
    d_sampled_indices = cuda.to_device(np.asarray(sampled_indices, dtype=np.int32))
    d_out_trajs = cuda.to_device(out_trajs)
    threadsperblock = 32
    blockspergrid = (n + (threadsperblock - 1)) // threadsperblock
    flat_raytrace_kernel_with_traj[blockspergrid, threadsperblock](
        d_obs_pos, d_ray_dirs, boundary_radius, patch_center_theta, patch_center_phi, patch_size_theta, patch_size_phi, flip_theta, flip_phi, d_bg_array, h, w, d_out_img, d_sampled_indices, n_sampled, n_points, d_out_trajs
    )
    d_out_img.copy_to_host(out_img)
    d_out_trajs.copy_to_host(out_trajs)
    return out_img, out_trajs 