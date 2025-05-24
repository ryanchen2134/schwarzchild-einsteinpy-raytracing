import numpy as np
import matplotlib.pyplot as plt
import os
from simulation.blackhole import BlackHole, Observer
from einsteinpy.geodesic import Nulllike
from einsteinpy.metric import Schwarzschild
from einsteinpy.coordinates.differential import SphericalDifferential
from einsteinpy.coordinates import Spherical
from astropy import units as u

"""Simple debug script: Launch one ray at a small angle off the optical axis (in the xy-plane, θ = π/2)
from the observer and plot its curved trajectory in a 2-D top-down view, together with the
black hole, observer position, and simulation boundary. This is useful for debugging
coordinate/trajectory issues.
Run with:  python debug_single_ray.py
"""

def main():
    # Geometry setup ---------------------------------------------------------
    bh_mass = 10  # M units (Schwarzschild radius r_s = 2M)
    bh = BlackHole(mass=bh_mass)
    rs = bh.rs
    boundary_radius = 10 * rs

    observer_pos = np.array([8 * rs, 0.0, 0.0])
    observer = Observer(position=observer_pos, fov=np.deg2rad(60), image_size=(100, 100))

    # Choose a single direction in the xy-plane (θ = π/2) at +offset_deg° from the -x axis
    offset_deg = 60
    theta = np.pi / 2  # equatorial plane
    phi = np.pi - np.deg2rad(offset_deg)  # Slightly off -x axis

    # Initial position in spherical coordinates
    r = np.linalg.norm(observer_pos)
    th = theta
    ph = np.arctan2(observer_pos[1], observer_pos[0])

    # Global direction: offset angle from observer-BH axis in x-y plane
    alpha = np.deg2rad(offset_deg)
    dir_vec = np.array([
        -np.cos(alpha),  # x (toward -x axis, offset by alpha)
        np.sin(alpha),   # y
        0.0              # z (xy-plane)
    ])
    dir_vec = dir_vec / np.linalg.norm(dir_vec)
    # Spherical basis at observer
    x, y, z = observer_pos
    r = np.linalg.norm([x, y, z])
    th = np.arccos(z / r)
    ph = np.arctan2(y, x)
    e_r = np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])
    e_theta = np.array([np.cos(th)*np.cos(ph), np.cos(th)*np.sin(ph), -np.sin(th)])
    e_phi = np.array([-np.sin(ph), np.cos(ph), 0])
    pr = np.dot(dir_vec, e_r)
    ptheta = np.dot(dir_vec, e_theta)
    pphi = np.dot(dir_vec, e_phi)
    geod = Nulllike(
        metric="Schwarzschild",
        metric_params=(),
        position=[r, th, ph],
        momentum=[pr, ptheta, pphi],
        steps=5000,
        delta=0.1,
        omega=1,
        return_cartesian=True
    )
    traj = geod.trajectory[1]  # (steps, 6)

    # Convert to cartesian for plotting ---------------------------------------
    xs = traj[:, 1]
    ys = traj[:, 2]

    # Plot --------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))

    # Black hole event horizon
    bh_circle = plt.Circle((0, 0), rs, color='black')
    ax.add_patch(bh_circle)

    # Boundary
    boundary_circle = plt.Circle((0, 0), boundary_radius, color='gray', fill=False, linestyle='--')
    ax.add_patch(boundary_circle)

    # Observer
    ax.plot(observer_pos[0], observer_pos[1], 'ro', label='Observer')

    # Curved trajectory
    ax.plot(xs, ys, color='orange', lw=2, label='Curved ray')

    # Initial momentum arrow (projected to xy)
    arrow_scale = 0.5 * rs
    # Project local direction to global xy for arrow
    # Use the first step of the trajectory for direction
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    ax.arrow(observer_pos[0], observer_pos[1], dx * arrow_scale, dy * arrow_scale,
             head_width=0.1 * rs, head_length=0.2 * rs, fc='green', ec='green', label='Initial momentum')

    # Formatting
    ax.set_aspect('equal')
    lim = boundary_radius * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Single Curved Ray (offset {offset_deg}°)')
    ax.legend()

    os.makedirs('images', exist_ok=True)
    out_path = 'images/single_ray_topdown.png'
    plt.savefig(out_path, bbox_inches='tight')
    print(f'Saved plot to {out_path}')

if __name__ == '__main__':
    main() 