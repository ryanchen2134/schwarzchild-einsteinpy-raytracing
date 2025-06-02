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

def alpha_from_b(b, r0, M=1.0):
    sin_alpha = b / r0 / np.sqrt(1 - 2*M/r0)
    if sin_alpha >= 1:
        raise ValueError("Chosen b is too large for this r0 (sinα>1).")
    return np.arcsin(sin_alpha)        # radians


def plot_geodesic_df(df, *, mass_bh=1.0, cmap=cm.plasma, step=1000):
    """
    4-panel summary for a geodesic stored in a DataFrame.

    Extra features:
        • horizon sphere / circles at r_s = 2M
        • faint line from BH (origin) to observer
    """
    # ── 0) constants ────────────────────────────────────────────────
    rs = 2.0 * mass_bh                      # Schwarzschild radius

    # ── 1) extract & decimate ───────────────────────────────────────
    t, r, th, ph = df["t"].values, df["r"].values, df["theta"].values, df["phi"].values
    xs = (r * np.sin(th) * np.cos(ph))[::step]
    ys = (r * np.sin(th) * np.sin(ph))[::step]
    zs = (r * np.cos(th))[::step]
    t  = t[::step]

    # BH → observer axis (first point is observer)
    obs_vec = np.array([xs[0], ys[0], zs[0]])

    # ── 2) orbital-plane basis (unchanged) ──────────────────────────
    r0_vec = obs_vec
    v_vec  = np.array([xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0]]) if len(xs) > 1 else r0_vec
    n_hat  = np.cross(r0_vec, v_vec);  n_hat = n_hat/np.linalg.norm(n_hat) if np.linalg.norm(n_hat)>0 else np.array([0.,0.,1.])
    e1     = (r0_vec - np.dot(r0_vec, n_hat)*n_hat) / np.linalg.norm(r0_vec - np.dot(r0_vec,n_hat)*n_hat)
    e2     = np.cross(n_hat, e1)
    u      = xs*e1[0] + ys*e1[1] + zs*e1[2]
    v      = xs*e2[0] + ys*e2[1] + zs*e2[2]

    # colour map normalisation
    norm = plt.Normalize(0, len(xs)-1)

    # ── 3) figure layout ────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 8))

    # ---------- 3-D -------------------------------------------------
    ax3d = fig.add_subplot(221, projection="3d")
    lc3d, _, _ = make_colour_segments(xs, ys, zs, cmap=cmap)
    ax3d.add_collection3d(lc3d)

    # horizon wire-frame
    ugrid = np.linspace(0, 2*np.pi, 40)
    vgrid = np.linspace(0, np.pi, 20)
    x_s = rs*np.outer(np.cos(ugrid), np.sin(vgrid))
    y_s = rs*np.outer(np.sin(ugrid), np.sin(vgrid))
    z_s = rs*np.outer(np.ones_like(ugrid), np.cos(vgrid))
    ax3d.plot_wireframe(x_s, y_s, z_s, color='gray', alpha=0.25, linewidth=0.4)

    # BH + observer + axis
    ax3d.scatter(0,0,0,c='k',s=40,label='BH')
    ax3d.scatter(*obs_vec,c='r',s=25,label='observer')
    ax3d.plot([0,obs_vec[0]],[0,obs_vec[1]],[0,obs_vec[2]], color='gray', linestyle='--', linewidth=0.8, alpha=0.4)

    ax3d.set_xlabel('x'); ax3d.set_ylabel('y'); ax3d.set_zlabel('z')
    ax3d.set_title('3-D trajectory')
    ax3d.legend()

    # ---------- x-y -------------------------------------------------
    ax_xy = fig.add_subplot(222)
    lc_xy, _, _ = make_colour_segments(xs, ys, cmap=cmap)
    ax_xy.add_collection(lc_xy)
    circ = np.linspace(0, 2*np.pi, 400)
    ax_xy.plot(rs*np.cos(circ), rs*np.sin(circ), color='gray', alpha=0.25)
    ax_xy.plot([0,obs_vec[0]],[0,obs_vec[1]], color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    ax_xy.set_ylim(-5,5)
    ax_xy.set_xlim(-5,5)
    ax_xy.set_xlabel('x'); ax_xy.set_ylabel('y'); ax_xy.set_title('x-y')
    ax_xy.axis('equal'); ax_xy.autoscale()

    # ---------- x-z -------------------------------------------------
    ax_xz = fig.add_subplot(223)
    lc_xz, _, _ = make_colour_segments(xs, zs, cmap=cmap)
    ax_xz.add_collection(lc_xz)
    ax_xz.plot(rs*np.cos(circ), rs*np.sin(circ), color='gray', alpha=0.25)
    ax_xz.plot([0,obs_vec[0]],[0,obs_vec[2]], color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    ax_xz.set_ylim(-5,5)
    ax_xz.set_xlim(-5,5)
    ax_xz.set_xlabel('x'); ax_xz.set_ylabel('z'); ax_xz.set_title('x-z')
    ax_xz.axis('equal'); ax_xz.autoscale()

    # # ---------- x-z -------------------------------------------------
    # ax_yz = fig.add_subplot(224)
    # lc_yz, _, _ = make_colour_segments(ys, zs, cmap=cmap)
    # ax_yz.add_collection(lc_yz)
    # ax_yz.plot(rs*np.cos(circ), rs*np.sin(circ), color='gray', alpha=0.25)
    # ax_yz.plot([0,obs_vec[1]],[0,obs_vec[2]], color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    # ax_yz.set_ylim(-5,5)
    # ax_yz.set_xlim(-5,5)
    # ax_yz.set_xlabel('y'); ax_yz.set_ylabel('z'); ax_yz.set_title('y-z')
    # ax_yz.axis('equal'); ax_yz.autoscale()
    
    
    # ---------- polar orbital plane --------------------------------
    r_plane = np.hypot(u, v)
    th_plane = np.arctan2(v, u)
    ax_pol = fig.add_subplot(224, projection='polar')
    sc = ax_pol.scatter(th_plane, r_plane, c=np.arange(len(th_plane)),
                        cmap=cmap, s=4, norm=norm)
    # horizon circle in the plane (constant radius = r_s)
    ax_pol.plot(np.linspace(0, 2*np.pi, 400), np.full(400, rs), color='gray', alpha=0.25)
    ax_pol.set_title("orbital plane (r, θ')")
    ax_pol.set_rlabel_position(45)

    # shared colour-bar
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.68])
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=cax, label='index 0 → … → final (time)')

    fig.tight_layout(rect=[0,0,0.9,1])
    fig.savefig("single_ray_cuda_test.png", dpi=150)



# -------------------------------------------------------------------
# build_null_4momentum_ep
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Einstein-style null 4-momentum from spherical data
# -------------------------------------------------------------------
def build_null_4momentum_ep_sph(
        p_sph: np.ndarray,          # (p^r, p^θ, p^φ)
        pos_sph: np.ndarray,        # (r, θ, φ)
        *,
        mass_bh: float = 1.0,
        future: bool = True,
) -> np.ndarray:
    """
    Return the CONTRAVARIANT null 4-momentum p^μ = (p^t, p^r, p^θ, p^φ)
    exactly the way EinsteinPy’s `_P()` does, but using spherical inputs.

    Parameters
    ----------
    p_sph : (3,) array_like
        Initial 3-momentum **in the coordinate basis**  
        `p_sph = (p^r, p^θ, p^φ)`.  Normalise it any way you like; its
        overall scale sets the affine-parameter normalisation.
    pos_sph : (3,) array_like
        Observer position `(r, θ, φ)` in Boyer–Lindquist coordinates.
    mass_bh : float, default 1
        Black-hole mass `M` (G = c = 1).
    future : bool, default True
        Choose the future-directed root (`p^t > 0`).  Set `False`
        for the past-directed solution.

    Returns
    -------
    p : ndarray, shape (4,)
        Full 4-momentum `(p^t, p^r, p^θ, p^φ)`.
    """
    # unpack inputs ---------------------------------------------------
    pr, pth, pph = p_sph        # contravariant components
    r,  th,  ph  = pos_sph

    # contravariant Schwarzschild metric g^{μν} -----------------------
    f = 1.0 - 2.0 * mass_bh / r
    if f <= 0.0:
        raise ValueError("Emitter must lie outside the horizon (r > 2M).")

    gtt = -1.0 / f
    grr =  f
    gthth = 1.0 / r**2
    gphph = 1.0 / (r**2 * np.sin(th)**2)

    # EinsteinPy quadratic  g^{μν}p_μ p_ν = 0  (null) ----------------
    A = gtt                       # < 0
    B = 0.0                       # g^{tφ}=0 in Schwarzschild
    C = grr * pr**2 + gthth * pth**2 + gphph * pph**2

    disc = B**2 - 4*A*C
    if disc < 0:
        raise RuntimeError("Negative discriminant (should be zero or >0).")

    sqrt_disc = np.sqrt(disc)
    p_t = (-B + sqrt_disc) / (2*A)   # gives p^t > 0 because A < 0
    if not future:
        p_t = (-B - sqrt_disc) / (2*A)

    return np.array([p_t, pr, pth, pph])


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

import math
from simulation.utils import angles_to_p_sph

def main():
    # ---------------- parameters -----------------
    mass_bh = 1.0
    
    r_max =50
    steps, delta, omega = 200_000 , 0.03, 0.01 # delta is the affine step size

    theta0, phi0 = np.pi/2, 0.0
    
    #R_obs = 500   # same as r0
    R_obs = 35
    r0    = R_obs                   # keep a single source of truth
    
    pos_sph = np.array([r0, theta0, phi0]) 

    alpha_deg =  0               # right-hand deflection (towards +y)
    beta_deg  = 90             # upward deflection   (towards +z)
        # ------------------------------------------------------------------
    # 1)  map (α,β) → Cartesian unit vector  (unchanged)
    # ------------------------------------------------------------------
    alpha = np.deg2rad(alpha_deg) #x-axis deflection (right-hand to +y)
    beta  = np.deg2rad(beta_deg) #x-axis deflection (upward to +z)
    
    p_direction=angles_to_p_sph(alpha, beta, r_obs=R_obs)
    
    # p_direction[0] = p_direction[0]/R_obs
    # p_direction = 100*p_direction
    
    
    # p_direction = np.array([1/10,1,1])  # (p^r, p^θ, p^φ)
    p_direction = np.array([-0.026942690335328513,-0.028502831807219468,0.06898831276132347])
    
    


    print("Spherical position:", pos_sph)
    print("Spherical direction:", p_direction)
    


   
    p0 = build_null_4momentum_ep_sph(p_direction, pos_sph,
                                 mass_bh=mass_bh, future=True)
    print("EinsteinPy-style 4-momentum:", p0)
    q0 = np.array([0.0, *pos_sph]) 



    print("Starting integration")
    # -------------- integrate ---------------------
    integrator = CUDASchwarzschildIntegrator(steps=steps, delta=delta, mass=mass_bh, omega=omega, r_max=r_max)
    traj = integrator.integrate_batch_full(q0[np.newaxis, :], p0[np.newaxis, :])[0]
    print("Integration complete")

    rs = 2.0 * mass_bh
    #get traj datapoints
    length = len(traj)
    print(f"Trajectory length: {length} steps")
    
    safe = traj[:, 1] > 1.1 * rs
    if not np.all(safe):
        traj = traj[: np.argmax(~safe)]
    print(f"Safe trajectory length: {len(traj)} steps")

    print("Exporting trajectory to CSV...")
    # -------------- export CSV --------------------
    df = pd.DataFrame(traj, columns=["t", "r", "theta", "phi"])
    # df["r"] = np.linalg.norm(df[["x", "y", "z"]], axis=1)
    
    print("Drawing trajectory plots...")
    plot_geodesic_df(df)
    
    df["theta"], df["phi"] = np.degrees(df["theta"]), np.degrees(df["phi"])
    df.to_csv("single_ray_cuda_test.csv", index=False)
    


    # # -------------- Cartesian ---------------------
    # t, r, th, ph = traj.T
    # xs, ys, zs = r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph), r * np.cos(th)

    # # in‑plane basis
    # r0_vec = np.array([R_obs, 0.0, 0.0])
    # n_hat  = np.cross(r0_vec, n_cart)
    # n_hat = n_hat / np.linalg.norm(n_hat) if np.linalg.norm(n_hat) else np.array([0.0, 0.0, 1.0])
    # e1 = (r0_vec - np.dot(r0_vec, n_hat) * n_hat) / np.linalg.norm(r0_vec - np.dot(r0_vec, n_hat) * n_hat)
    # e2 = np.cross(n_hat, e1)
    # u, v = xs*e1[0] + ys*e1[1] + zs*e1[2], xs*e2[0] + ys*e2[1] + zs*e2[2]

    # # Global colour map (shared normalisation)
    # cmap = cm.viridis
    # norm = plt.Normalize(0, len(xs) - 1)

    # # -------------- figure ------------------------
    # fig = plt.figure(figsize=(10, 8))

    # # 3‑D
    # ax3d = fig.add_subplot(221, projection="3d")
    # lc3d, _, _ = make_colour_segments(xs, ys, zs, cmap)
    # ax3d.add_collection3d(lc3d)
    # ax3d.scatter([0], [0], [0], c="k", s=40)
    # ax3d.scatter([R_obs], [0], [0], c="red", s=25)
    # ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
    # ax3d.set_title("3‑D trajectory (time‑coded)")

    # # x‑y
    # ax_xy = fig.add_subplot(222)
    # lc_xy, _, _ = make_colour_segments(xs, ys, cmap=cmap)
    # ax_xy.add_collection(lc_xy)
    # ax_xy.scatter([0], [0], c="k", s=10)
    # ax_xy.set_xlabel("x"); ax_xy.set_ylabel("y"); ax_xy.set_title("x‑y")
    # ax_xy.axis("equal")

    # # x‑z
    # ax_xz = fig.add_subplot(223)
    # lc_xz, _, _ = make_colour_segments(xs, zs, cmap=cmap)
    # ax_xz.add_collection(lc_xz)
    # ax_xz.scatter([0], [0], c="k", s=10)
    # ax_xz.set_xlabel("x"); ax_xz.set_ylabel("z"); ax_xz.set_title("x‑z")
    # ax_xz.axis("equal")

    # # polar in‑plane
    # r_plane = np.sqrt(u**2 + v**2)
    # th_plane = np.arctan2(v, u)
    # ax_pol = fig.add_subplot(224, projection="polar")
    # sc = ax_pol.scatter(th_plane, r_plane, c=np.arange(len(th_plane)), cmap=cmap, s=4, norm=norm)
    # ax_pol.set_title("Orbital‑plane polar")

    # # Shared colour‑bar
    # cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    # plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label="integration step → time")

    # fig.tight_layout(rect=[0,0,0.9,1])
    # fig.savefig("single_ray_cuda_test.png", dpi=150)
    # plt.show()


if __name__ == "__main__":
    main()
