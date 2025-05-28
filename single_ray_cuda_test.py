"""
Single-ray CUDA test script – *fixed* with colour‑gradient time coding
====================================================================

Adds a perceptual colour gradient (viridis) that runs **from λ = 0 to the
final integration step** so you can see the photon’s temporal progress at a
glance.

Changes compared with the previous version
------------------------------------------
* New helper `make_colour_segments()` produces a *Line3DCollection* (3‑D) or
  *LineCollection* (2‑D) whose segments are coloured by normalised λ.
* All four diagnostic panels now use the gradient instead of a single solid
  colour.
* Colormap and normalisation are defined **once** so every subplot shares the
  same colour meaning.

Run exactly as before:
```bash
python single_ray_cuda_test_fixed.py
```

Everything else (physics, CSV export, buffer cut‑off) is unchanged.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – enables 3‑D projection
import pandas as pd

from simulation.cuda_geodesic import CUDASchwarzschildIntegrator

###############################################################################
# Helper utilities                                                            #
###############################################################################



def build_null_4momentum(n_cart: np.ndarray, r0: float, theta0: float, phi0: float, *, mass_bh: float = 1.0) -> np.ndarray:
    """Return contravariant null 4‑momentum that matches *n_cart*."""
    f_r = 1.0 - 2.0 * mass_bh / r0
    sqrt_f = np.sqrt(f_r)

    sin_t, cos_t = np.sin(theta0), np.cos(theta0)
    sin_p, cos_p = np.sin(phi0),  np.cos(phi0)

    e_r     = np.array([sin_t * cos_p, sin_t * sin_p, cos_t])
    e_theta = np.array([cos_t * cos_p, cos_t * sin_p, -sin_t])
    e_phi   = np.array([-sin_p,        cos_p,        0.0])

    n_rhat, n_thhat, n_phihat = map(lambda e: np.dot(n_cart, e), (e_r, e_theta, e_phi))

    E_hat = 1.0  # local photon energy scale (λ‑parametrisation)

    P_t     =  1.0 / sqrt_f
    P_r     =  sqrt_f * E_hat * n_rhat
    P_theta = E_hat * n_thhat / r0
    P_phi   = E_hat * n_phihat / (r0 * sin_t)

    # Sanity – enforce nullness
    H = (
        -f_r * P_t**2
        + (1.0 / f_r) * P_r**2
        + r0**2 * (P_theta**2 + (sin_t**2) * P_phi**2)
    )
    if abs(H) > 1e-10:
        raise RuntimeError(f"Hamiltonian not null (H={H:.3e}).")

    return np.array([P_t, P_r, P_theta, P_phi])


def make_colour_segments(xs, ys, zs=None, cmap=cm.viridis):
    """Return a LineCollection/Line3DCollection colour‑coded by index."""
    pts = np.column_stack((xs, ys)) if zs is None else np.column_stack((xs, ys, zs))
    segments = np.stack([pts[:-1], pts[1:]], axis=1)
    norm = plt.Normalize(0, len(xs) - 1)
    colors = cmap(norm(np.arange(len(xs) - 1)))
    if zs is None:
        lc = LineCollection(segments, colors=colors, linewidth=2)
    else:
        lc = Line3DCollection(segments, colors=colors, linewidth=2)
    return lc, cmap, norm

###############################################################################
# Main driver                                                                  #
###############################################################################

def main():
    # ---------------- parameters -----------------
    mass_bh   = 1.0
    
    r_max =1000
    steps, delta, omega = 10000, 0.05, 0.01 # delta is the affine step size
    
    theta0, phi0 = np.pi/2, 0.0
    
    R_obs = 4.0                     # same as r0
    r0    = R_obs                   # keep a single source of truth

    alpha_deg = 60                  # right-hand deflection (towards +y)
    beta_deg  = 60                  # upward deflection   (towards +z)

    alpha = np.deg2rad(alpha_deg)
    beta  = np.deg2rad(beta_deg)
    
    n_cart = np.array([
    -np.cos(alpha)*np.cos(beta),   # x
     np.sin(alpha)*np.cos(beta),   # y
     np.sin(beta)                  # z
    ])
    n_cart /= np.linalg.norm(n_cart) 
    
    
    # ------------------------------------------------------------
    # 3)  Project n_cart onto the local orthonormal spherical basis
    #     At (θ₀=π/2, φ₀=0):
    #       r̂      = ( 1, 0, 0)
    #       θ̂      = ( 0, 0,-1)
    #       φ̂      = ( 0, 1, 0)
    # ------------------------------------------------------------
    n_rhat   =  n_cart[0]              #  dot(n_cart, r̂)
    n_thhat  = -n_cart[2]              #  dot(n_cart, θ̂)
    n_phihat =  n_cart[1]              #  dot(n_cart, φ̂)


    q0 = np.array([0.0, r0, theta0, phi0])                    # (t,r,θ,φ)
    cart = np.array([n_rhat, n_thhat, n_phihat])     # components
    p0 = build_null_4momentum(cart, r0, theta0, phi0,
                          mass_bh=mass_bh)
    
    print(f"p0: {p0}")

    # -------------- integrate ---------------------
    integrator = CUDASchwarzschildIntegrator(steps=steps, delta=delta, mass=mass_bh, omega=omega, r_max=r_max)
    traj = integrator.integrate_batch_full(q0[np.newaxis, :], p0[np.newaxis, :])[0]

    rs = 2.0 * mass_bh
    safe = traj[:, 1] > 1.1 * rs
    if not np.all(safe):
        traj = traj[: np.argmax(~safe)]

    # -------------- export CSV --------------------
    df = pd.DataFrame(traj, columns=["t", "r", "theta", "phi"])
    df["theta"], df["phi"] = np.degrees(df["theta"]), np.degrees(df["phi"])
    df.to_csv("single_ray_cuda_test.csv", index=False)

    # -------------- Cartesian ---------------------
    t, r, th, ph = traj.T
    xs, ys, zs = r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph), r * np.cos(th)

    # in‑plane basis
    r0_vec = np.array([R_obs, 0.0, 0.0])
    n_hat  = np.cross(r0_vec, n_cart)
    n_hat = n_hat / np.linalg.norm(n_hat) if np.linalg.norm(n_hat) else np.array([0.0, 0.0, 1.0])
    e1 = (r0_vec - np.dot(r0_vec, n_hat) * n_hat) / np.linalg.norm(r0_vec - np.dot(r0_vec, n_hat) * n_hat)
    e2 = np.cross(n_hat, e1)
    u, v = xs*e1[0] + ys*e1[1] + zs*e1[2], xs*e2[0] + ys*e2[1] + zs*e2[2]

    # Global colour map (shared normalisation)
    cmap = cm.viridis
    norm = plt.Normalize(0, len(xs) - 1)

    # -------------- figure ------------------------
    fig = plt.figure(figsize=(10, 8))

    # 3‑D
    ax3d = fig.add_subplot(221, projection="3d")
    lc3d, _, _ = make_colour_segments(xs, ys, zs, cmap)
    ax3d.add_collection3d(lc3d)
    ax3d.scatter([0], [0], [0], c="k", s=40)
    ax3d.scatter([R_obs], [0], [0], c="red", s=25)
    ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
    ax3d.set_title("3‑D trajectory (time‑coded)")

    # x‑y
    ax_xy = fig.add_subplot(222)
    lc_xy, _, _ = make_colour_segments(xs, ys, cmap=cmap)
    ax_xy.add_collection(lc_xy)
    ax_xy.scatter([0], [0], c="k", s=10)
    ax_xy.set_xlabel("x"); ax_xy.set_ylabel("y"); ax_xy.set_title("x‑y")
    ax_xy.axis("equal")

    # x‑z
    ax_xz = fig.add_subplot(223)
    lc_xz, _, _ = make_colour_segments(xs, zs, cmap=cmap)
    ax_xz.add_collection(lc_xz)
    ax_xz.scatter([0], [0], c="k", s=10)
    ax_xz.set_xlabel("x"); ax_xz.set_ylabel("z"); ax_xz.set_title("x‑z")
    ax_xz.axis("equal")

    # polar in‑plane
    r_plane = np.sqrt(u**2 + v**2)
    th_plane = np.arctan2(v, u)
    ax_pol = fig.add_subplot(224, projection="polar")
    sc = ax_pol.scatter(th_plane, r_plane, c=np.arange(len(th_plane)), cmap=cmap, s=4, norm=norm)
    ax_pol.set_title("Orbital‑plane polar")

    # Shared colour‑bar
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label="integration step → time")

    fig.tight_layout(rect=[0,0,0.9,1])
    fig.savefig("single_ray_cuda_test.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
