#utils.py
import numpy as np
from einsteinpy.coordinates.utils import spherical_to_cartesian_fast, cartesian_to_spherical_fast
import math

# -----------------------------------------------------------------------------
# Helper: exact EinsteinPy-style null 4-momentum in Schwarzschild coordinates
# -----------------------------------------------------------------------------

def build_null_4momentum_ep_sph(p_sph, pos_sph, *, mass_bh=1.0, future=False):
    """Return contravariant null 4-momentum p^μ = (p^t, p^r, p^θ, p^φ).

    This faithfully reproduces the root selection used by EinsteinPy's internal
    `_P()` routine but allows an arbitrary black-hole mass *mass_bh*.

    Parameters
    ----------
    p_sph : array_like, shape (3,)
        Spatial contravariant momentum components `(p^r, p^θ, p^φ)` in the
        coordinate basis.
    pos_sph : array_like, shape (3,)
        Observer position `(r, θ, φ)`.
    mass_bh : float, default 1.0
        Black-hole mass *M* (G = c = 1).  The metric becomes flat as
        `mass_bh → 0`.
    future : bool, default False
        Choose the future-directed (`p^t > 0`) solution if *True*.  The default
        *False* returns the past-directed branch (negative p^t) for
        consistency with earlier code based on
        `compute_null_4momentum_schwarzschild`, which supplied the negative
        root.
    """

    pr, pth, pph = p_sph
    r, th, _ = pos_sph

    f = 1.0 - 2.0 * mass_bh / r
    if f <= 0.0:
        raise ValueError("Observer must lie outside the event horizon (r > 2M).")

    gtt = -1.0 / f
    grr = f
    gthth = 1.0 / (r * r)
    gphph = 1.0 / (r * r * math.sin(th) ** 2)

    # Null condition: g^{μν} p_μ p_ν = 0  ->  quadratic in p^t
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

def get_initial_conditions(observer_pos, pixel_pos, *, mass_bh=1.0):
    """Return (`q0`, `p0`) for a photon launched from *observer_pos* towards
    *pixel_pos*.

    • `q0` is the 4-position `(t=0, r, θ, φ)` in Schwarzschild coordinates.
    • `p0` is the **full** contravariant null 4-momentum `(p^t, p^r, …)` that
      satisfies `g_{μν} p^μ p^ν = 0`.

    The routine follows the derivation showcased in *single_ray_cuda_test.py*
    for robustness and mass-dependency, superseding previous uses of
    `cartesian_to_spherical_fast`.
    """

    # 1) direction (unit) vector in Cartesian coords -------------------------
    ray_dir = pixel_pos - observer_pos
    ray_dir = ray_dir / np.linalg.norm(ray_dir)

    # 2) observer position in spherical coords ------------------------------
    x, y, z = observer_pos
    _, r, theta, phi = cartesian_to_spherical_fast(0,x, y, z)
    r_obs = r
    
    # 2.1) ray direction in spherical coords ------------------------------
    x, y, z = ray_dir
    _, h_r, h_theta, h_phi = cartesian_to_spherical_fast(0, x, y, z)
    
    
    # 3) convert direction vector to spherical momentum components
    
    p_spatial = angles_to_p_sph(np.pi-h_phi, np.pi/2-h_theta, r_obs)
    

   
    # 5) full 4-momentum via null condition ---------------------------------
    
    
    
    p0 = build_null_4momentum_ep_sph(p_spatial, np.array([r, theta, phi]), mass_bh=mass_bh, future=False)

    # 6) final 4-position ----------------------------------------------------
    q0 = np.array([0.0, r, theta, phi])

    return q0, p0

# -----------------------------------------------------------------------------
# Helper: angles (α,β) → spherical momentum components
# -----------------------------------------------------------------------------

def angles_to_p_sph(alpha, beta, r_obs, *, normalise=True):
    """Return coordinate-basis momentum components `(p^r, p^θ, p^φ)` for a ray
    emitted from the observer on the **+x axis** (θ = π/2, φ = 0) that makes
    camera angles *(alpha, beta)*:

        • alpha  : right-hand deflection ( +y direction)  [rad]
        • beta   : upward deflection   ( +z direction)  [rad]

    The ray always points **towards** the black hole, i.e. radially *inward*.

    Parameters
    ----------
    alpha, beta : float
        Camera angles in radians.
    r_obs : float
        Observer radius (distance from BH).
    normalise : bool, default True
        If *True* the returned vector is normalised such that the orthonormal
        3-vector has unit length.  Set *False* to keep the raw components.
    """

    # 1) orthonormal components (r̂, θ̂, φ̂) at the observer
    n_rhat  = -math.cos(alpha) * math.cos(beta)   # negative = inward (–x)
    n_phhat =  math.sin(alpha) * math.cos(beta)   # +y
    n_thhat = -math.sin(beta)                     # –z (θ̂)

    # 2) optional normalisation
    if normalise:
        norm = math.sqrt(n_rhat**2 + n_thhat**2 + n_phhat**2)
        n_rhat, n_thhat, n_phhat = n_rhat/norm, n_thhat/norm, n_phhat/norm

    # 3) convert orthonormal → coordinate basis components
    p_r  = n_rhat / r_obs            # radial component (already correct units)
    p_th = n_thhat    # divide by r
    p_ph = n_phhat  # divide by r

    return np.array([p_r, p_th, p_ph]) 