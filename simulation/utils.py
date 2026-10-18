#utils.py
import numpy as np
from einsteinpy.coordinates.utils import spherical_to_cartesian_fast, cartesian_to_spherical_fast
import math

# -----------------------------------------------------------------------------
# Helper: exact EinsteinPy-style null 4-momentum in Schwarzschild coordinates
# -----------------------------------------------------------------------------
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
    g_phph = r * r * np.sin(th) ** 2
    pr, pth, pph = p_spatial
    C = g_rr * pr * pr + g_thth * pth * pth + g_phph * pph * pph
    # Null condition: g_tt (p^t)^2 + C = 0  ->  p^t = ±sqrt(-C/g_tt)
    pt = np.sqrt(max(0.0, -C / g_tt))
    return [-pt, pr, pth, pph]  # negative root for consistency with EPy


def _apply_relative_offsets(theta_base_deg, phi_base_deg,
                            dtheta_deg=0.0, dphi_deg=0.0):
    """Return new θ,φ after adding small observer-relative offsets (deg)."""
    theta_base = np.deg2rad(theta_base_deg)
    phi_base = np.deg2rad(phi_base_deg)
    dθ = np.deg2rad(dtheta_deg)
    dφ = np.deg2rad(dphi_deg)
    θ  = np.clip(theta_base + dθ, 0.0, np.pi)
    φ  = (phi_base  + dφ) % (2*np.pi)
    return θ, φ


def build_null_4momentum_ep_sph(p_sph, pos_sph, *, mass_bh=1.0, future=False):
    """Return the covariant null 4-momentum p_μ = (p_t, p_r, p_θ, p_φ).

    The FANTASY integrator's Hamiltonian is H = ½ g^{μν} p_μ p_ν, so the momenta
    are covariant throughout: this solves g^{μν} p_μ p_ν = 0 for p_t. The root
    selection mirrors EinsteinPy's internal `_P()` routine but allows an
    arbitrary black-hole mass *mass_bh*.

    Parameters
    ----------
    p_sph : array_like, shape (3,)
        Covariant spatial components `(p_r, p_θ, p_φ)`, as returned by
        `angles_to_p_sph(..., normalise=True)`.
    pos_sph : array_like, shape (3,)
        Observer position `(r, θ, φ)`.
    mass_bh : float, default 1.0
        Black-hole mass *M* (G = c = 1).  The metric becomes flat as
        `mass_bh → 0`.
    future : bool, default False
        If *True* pick `p_t > 0`, i.e. `p^t = g^{tt} p_t = -p_t / f < 0`:
        coordinate time *decreases* along the ray, which is what tracing rays
        backwards from the camera needs. The default *False* returns `p_t < 0`
        (`p^t > 0`) for consistency with earlier code based on
        `compute_null_4momentum_schwarzschild`.
    """

    pr, pth, pph = p_sph  # covariant components
    r, th, _ = pos_sph

    f = 1.0 - 2.0 * mass_bh / r
    if f <= 0.0:
        raise ValueError("Observer must lie outside the event horizon (r > 2M).")

    gtt = -1.0 / f
    grr = f
    gthth = 1.0 / (r * r)
    gphph = 1.0 / (r * r * math.sin(th) ** 2)

    # Null condition: g^{μν} p_μ p_ν = 0  ->  quadratic in the covariant p_t
    A = gtt  # < 0
    C = grr * pr * pr + gthth * pth * pth + gphph * pph * pph

    disc = -4.0 * A * C  # B = 0 in Schwarzschild so Δ = -4AC
    if disc < 0.0:
        raise RuntimeError("Negative discriminant while enforcing null condition.")

    sqrt_disc = math.sqrt(disc)
    p_t = sqrt_disc / (2.0 * (-A))  # always positive
    if not future:
        p_t = -p_t

    return np.array([p_t, pr, pth, pph])

# utils.py  --------------------------------------------------------------
def get_initial_conditions(observer_pos, pixel_pos, *, mass_bh=1.0):
    """
    Returns q0, p0, alpha, beta  (beta = rotation about +x̂).
    """

    # ------------------------------------------------------------------ #
    # 0)  Build the raw ray direction in the *lab* frame
    # ------------------------------------------------------------------ #
    ray_dir = pixel_pos - observer_pos
    ray_dir /= np.linalg.norm(ray_dir)

    # β = angle between ray_dir and the x-y plane.  Positive if ray has +z.
    beta = np.arctan2(ray_dir[2], ray_dir[1])
    # Rotate by −β about +x̂ so the vector lies in x-y plane
    c, s = np.cos(-beta), np.sin(-beta)
    R_x = np.array([[1, 0, 0],
                    [0, c,-s],
                    [0, s, c]])
    ray_xy = R_x @ ray_dir
    # ray_xy = np.matmul(R_x, ray_dir)
    # print(ray_xy)
    
    assert abs(ray_xy[2]) < 1e-6     ,print(f"original ray: {ray_dir}, ray_xy: {ray_xy}, beta: {beta}")    # z ≈ 0 by construction

    # ------------------------------------------------------------------ #
    # 1)  Observer position in spherical → needed for q0
    # ------------------------------------------------------------------ #
    r_obs, theta_obs, phi_obs = cartesian_to_spherical_fast(0,*observer_pos)[1:]

    # ------------------------------------------------------------------ #
    # 2)  Direction vector in *rotated* frame ⇒ camera angles α (=φ_xy)
    # ------------------------------------------------------------------ #
    
    h_r, h_theta, h_phi = cartesian_to_spherical_fast(0, *ray_xy)[1:]
    
    assert abs(h_theta - np.pi/2) < 1e-6, print(f"h_spherical: {h_r,h_theta, h_phi}")
    
    #  p_spatial = angles_to_p_sph(np.pi-h_phi, np.pi/2-h_theta, r_obs)
    p_spatial = angles_to_p_sph(np.pi - h_phi, 0.0, r_obs, mass_bh=mass_bh, normalise=True)

    # ------------------------------------------------------------------ #
    # 3)  Full null 4-momentum and initial 4-position
    # ------------------------------------------------------------------ #
    p0 = build_null_4momentum_ep_sph(p_spatial,
                                     np.array([r_obs, theta_obs, phi_obs]),
                                     mass_bh=mass_bh, future=True)
    q0 = np.array([0.0, r_obs, theta_obs, phi_obs])

    _, h_r, h_theta, h_phi = cartesian_to_spherical_fast(0, *ray_dir)
    # p_r = n_rhat / sqrt(f)  =>  n_rhat = p_r * sqrt(f) = -cos(alpha)
    alpha0 = np.arccos(-p_spatial[0] * np.sqrt(1.0 - 2.0 * mass_bh / r_obs))
    return q0, p0, alpha0, h_r, h_theta, h_phi, beta


# -----------------------------------------------------------------------------
# Helper: angles (α,β) → spherical momentum components
# -----------------------------------------------------------------------------

def angles_to_p_sph(alpha, beta, r_obs, *, mass_bh=1.0, normalise=True):
    """Return *covariant* momentum components `(p_r, p_θ, p_φ)` for a ray
    emitted from a static observer on the **+x axis** (θ = π/2, φ = 0) that makes
    camera angles *(alpha, beta)*:

        • alpha  : right-hand deflection ( +y direction)  [rad]
        • beta   : upward deflection   ( +z direction)  [rad]

    The ray always points **towards** the black hole, i.e. radially *inward*.

    The FANTASY integrator's Hamiltonian is H = ½ g^{μν} p_μ p_ν, so the momenta
    it integrates are covariant. For a static observer whose orthonormal frame is
    e_r̂ = √f ∂_r, e_θ̂ = (1/r) ∂_θ, e_φ̂ = (1/(r sinθ)) ∂_φ with f = 1 − 2M/r, a photon
    with unit spatial direction (n_r̂, n_θ̂, n_φ̂) has

        p_r = n_r̂ / √f,   p_θ = r n_θ̂,   p_φ = r sinθ n_φ̂.

    (An earlier version used the contravariant p^r = n_r̂ √f next to the covariant
    p_θ, p_φ, which stretched every ray's angle from the optical axis by 1/f and
    shrank the shadow by 6.6 % at r = 30M.)

    Parameters
    ----------
    alpha, beta : float
        Camera angles in radians.
    r_obs : float
        Observer radius (distance from BH).
    normalise : bool, default True
        If *True* return the covariant components above. If *False* return the
        orthonormal-frame components (n_r̂, n_θ̂, n_φ̂) themselves.
    """

    # 1) orthonormal components (r̂, θ̂, φ̂) at the observer
    n_rhat  = -math.cos(alpha) * math.cos(beta)   # negative = inward (–x)
    n_phhat =  math.sin(alpha) * math.cos(beta)   # +y
    n_thhat = -math.sin(beta)                     # –z (θ̂)

    if not normalise:
        return np.array([n_rhat, n_thhat, n_phhat])

    # 2) covariant components (θ = π/2, so r sinθ = r)
    f_sqrt = np.sqrt(1.0 - 2.0 * mass_bh / r_obs)   # √f = √(-g_tt) = 1/√g_rr
    p_r  = n_rhat / f_sqrt      # p_r = g_rr p^r = n_r̂ / √f
    p_th = n_thhat * r_obs      # p_θ = r² p^θ = r n_θ̂
    p_ph = n_phhat * r_obs      # p_φ = r² sin²θ p^φ = r sinθ n_φ̂

    return np.array([p_r, p_th, p_ph])


# =============================================================================
# Pure-Python 3-vector helpers (the operation order of the Rust port's coords.rs)
# =============================================================================
# These work on plain floats so that the disk physics and the camera basis are
# computed exactly as the Rust build computes them: no BLAS, no fused multiply-add,
# every sum left to right.

def vec_add(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def vec_sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def vec_scale(a, s):
    return [a[0] * s, a[1] * s, a[2] * s]


def vec_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vec_norm(a):
    return math.sqrt(vec_dot(a, a))


def vec_normalize(a):
    n = vec_norm(a)
    return [a[0] / n, a[1] / n, a[2] / n]


def vec_cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def rot_x(angle):
    """Right-handed rotation about +x by `angle` (3x3 nested lists).

    rot_x(-beta) brings a ray with direction (dx, dy, dz) into the x-y plane when
    beta = atan2(dz, dy); rot_x(beta) undoes it.
    """
    s = math.sin(angle)
    c = math.cos(angle)
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


def rotation_about(axis, angle):
    """Right-handed rotation by `angle` about the unit vector `axis` (Rodrigues)."""
    s = math.sin(angle)
    c = math.cos(angle)
    t = 1.0 - c
    x, y, z = axis
    return [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]


def mat_mul(a, b):
    """a . b for 3x3 nested lists."""
    out = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    for i in range(3):
        for j in range(3):
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j]
    return out


def mat_vec(m, v):
    return [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]


# =============================================================================
# The pinhole camera: pointing and the image-plane grid
# =============================================================================
# The observer sits on the +x axis. Unrotated, the camera looks at the black hole
# at the origin: optical_axis = -x, right = +y, up = +z. The image plane is
# PLANE_DIST_FRACTION * |observer| in front of the observer along the optical axis;
# row i = 0 is the *bottom* row (v = -1/2).
#
# Pointing: the camera can turn (pan) but not move, since the equatorial rotation in
# get_initial_conditions requires the observer on +x. The viewer basis is rotated by
# three intrinsic angles applied in the order yaw -> pitch -> roll, each right-handed
# about the named axis of the camera as it then stands:
#   * yaw, about up: positive turns the view to the observer's left, so the black
#     hole moves right in the image;
#   * pitch, about right: positive turns the view up, so the black hole moves down;
#   * roll, about the optical axis: positive turns the image counter-clockwise as
#     seen by the viewer.
# Intrinsic yaw -> pitch -> roll equals the product R_up(yaw) . R_right(pitch)
# . R_axis(roll) of rotations about the *unrotated* axes (up = +z, right = +y,
# axis = -x), which is how camera_rotation builds it.

PLANE_DIST_FRACTION = 0.2


def camera_rotation(yaw, pitch, roll):
    """The rotation (3x3 nested lists) that takes the unrotated basis to the pointed one."""
    r_yaw = rotation_about([0.0, 0.0, 1.0], yaw)
    r_pitch = rotation_about([0.0, 1.0, 0.0], pitch)
    r_roll = rotation_about([-1.0, 0.0, 0.0], roll)
    return mat_mul(r_yaw, mat_mul(r_pitch, r_roll))


def camera_basis(yaw=0.0, pitch=0.0, roll=0.0):
    """(optical_axis, right, up) of the camera turned by yaw, pitch, roll (radians).

    With all angles zero the basis is exactly (-1, 0, 0), (0, 1, 0), (0, 0, 1).
    """
    for name, v in (("yaw", yaw), ("pitch", pitch), ("roll", roll)):
        if not math.isfinite(v):
            raise ValueError(f"{name} must be finite, got {v}")
    rot = camera_rotation(float(yaw), float(pitch), float(roll))
    optical_axis = mat_vec(rot, [-1.0, 0.0, 0.0])
    right = mat_vec(rot, [0.0, 1.0, 0.0])
    up = mat_vec(rot, [0.0, 0.0, 1.0])
    return (np.array(optical_axis, dtype=np.float64),
            np.array(right, dtype=np.float64),
            np.array(up, dtype=np.float64))


def image_plane(observer_pos, fov, h, w, optical_axis):
    """(plane_center, plane_width, plane_height) of the image plane."""
    obs_pos = np.asarray(observer_pos, dtype=np.float64)
    plane_dist = PLANE_DIST_FRACTION * np.linalg.norm(obs_pos)
    plane_center = obs_pos + optical_axis * plane_dist
    plane_width = 2 * plane_dist * np.tan(fov / 2)
    plane_height = plane_width * (h / w)
    return plane_center, plane_width, plane_height


def pixel_positions(observer_pos, fov, h, w, optical_axis, right, up):
    """World positions of every pixel centre, shape (h, w, 3); row i counts from the
    bottom of the image, column j from the left.

    Pixel (i, j) sits at plane_center + u*plane_width*right + v*plane_height*up with
    u = (j + 0.5)/w - 0.5 and v = (i + 0.5)/h - 0.5, the expression both renderers
    used before the camera could turn.
    """
    plane_center, plane_width, plane_height = image_plane(observer_pos, fov, h, w, optical_axis)
    positions = np.zeros((h, w, 3), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            u = (j + 0.5) / w - 0.5
            v = (i + 0.5) / h - 0.5
            positions[i, j] = plane_center + u * plane_width * right + v * plane_height * up
    return positions


def ray_directions(observer_pos, positions):
    """Unit vectors from the observer through the given positions (any leading shape)."""
    obs_pos = np.asarray(observer_pos, dtype=np.float64)
    dirs = np.asarray(positions, dtype=np.float64) - obs_pos
    return dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
