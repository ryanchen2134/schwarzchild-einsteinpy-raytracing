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
    R = 100
    observer_pos = np.array([R, 0.0, 0.0])
    theta = np.pi / 2  # equatorial plane
    
    #Ray direction offset from -x axis in xy-plane
    offset_deg = 10
    phi = np.pi - np.deg2rad(offset_deg)
    # Direction vector in xy-plane
    dir_vec = np.array([np.cos(phi), np.sin(phi), 0.0])
    dir_vec = dir_vec / np.linalg.norm(dir_vec)

    # Generate a null geodesic trajectory (curved, with gravity)
    r = np.linalg.norm(observer_pos)
    th = theta
    ph = np.arctan2(observer_pos[1], observer_pos[0])
    
    # Spherical basis at observer
    x, y, z = observer_pos
    r = np.linalg.norm([x, y, z])
    th = np.arccos(z / r)
    ph = np.arctan2(y, x)
    # 
    # e_r = np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])
    # e_theta = np.array([np.cos(th)*np.cos(ph), np.cos(th)*np.sin(ph), -np.sin(th)])
    # e_phi = np.array([-np.sin(ph), np.cos(ph), 0])
    # # 
    # pr = np.dot(dir_vec, e_r)
    # ptheta = np.dot(dir_vec, e_theta)
    # pphi = np.dot(dir_vec, e_phi)
    # # 
    geod = Nulllike(
        metric="Schwarzschild",
        metric_params=(),
        position=[r, th, ph],
        momentum=[-1, th, phi],
        steps=200,
        delta=0.5,
        omega=0.5,
        return_cartesian=True
    )
    traj = geod.trajectory[1]  # (steps, 6)
    xs = traj[:, 1]
    ys = traj[:, 2]

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xs, ys, color='orange', lw=2, label='Null geodesic (20° offset)')
    ax.plot(observer_pos[0], observer_pos[1], 'ro', label='Observer')
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Null Geodesic Trajectory (20° offset from -x)')
    ax.legend()

    os.makedirs('images', exist_ok=True)
    out_path = 'images/single_ray_geodesic.png'
    plt.savefig(out_path, bbox_inches='tight')
    print(f'Saved plot to {out_path}')

if __name__ == '__main__':
    main() 