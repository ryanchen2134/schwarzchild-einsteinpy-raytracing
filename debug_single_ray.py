import numpy as np
import matplotlib.pyplot as plt
import os
from simulation.blackhole import BlackHole, Observer
from einsteinpy.geodesic import Nulllike
from einsteinpy.metric import Schwarzschild
from einsteinpy.coordinates.differential import SphericalDifferential
from einsteinpy.coordinates import Spherical
from astropy import units as u
from einsteinpy.plotting import GeodesicPlotter

"""Simple debug script: Launch one ray at a small angle off the optical axis (in the xy-plane, θ = π/2)
from the observer and plot its curved trajectory in a 2-D top-down view, together with the
black hole, observer position, and simulation boundary. This is useful for debugging
coordinate/trajectory issues.
Run with:  python debug_single_ray.py
"""

def main():
    # Observer position
    R = 8.0
    observer_pos = np.array([R, 0.0, 0.0])
    theta = np.pi / 2  # equatorial plane
    offset_deg = 20
    # Direction: 20 deg off -x axis in xy-plane
    phi = np.pi - np.deg2rad(offset_deg)
    # Direction vector in xy-plane
    dir_vec = np.array([np.cos(phi), np.sin(phi), 0.0])
    dir_vec = dir_vec / np.linalg.norm(dir_vec)

    # Generate straight line trajectory
    n_steps = 200
    t = np.linspace(0, 10, n_steps)  # parameter along the line
    traj = observer_pos[None, :] + t[:, None] * dir_vec[None, :]
    xs = traj[:, 0]
    ys = traj[:, 1]

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xs, ys, color='blue', lw=2, label='Straight ray (20° offset)')
    ax.plot(observer_pos[0], observer_pos[1], 'ro', label='Observer')
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Straight Line Trajectory (20° offset from -x)')
    ax.legend()

    os.makedirs('images', exist_ok=True)
    out_path = 'images/straight_ray_20deg.png'
    plt.savefig(out_path, bbox_inches='tight')
    print(f'Saved plot to {out_path}')

if __name__ == '__main__':
    main() 