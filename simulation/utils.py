import numpy as np
from simulation.cuda_geodesic import compute_null_4momentum_schwarzschild

def get_initial_conditions(observer_pos, pixel_pos):
    """
    Given observer position and pixel position (both 3D Cartesian),
    compute initial 4-position and null 4-momentum for Schwarzschild geodesic integration.
    Returns (q0, p0), both 4-vectors.
    """
    ray_dir = pixel_pos - observer_pos
    ray_dir = ray_dir / np.linalg.norm(ray_dir)
    x, y, z = observer_pos
    r = np.linalg.norm([x, y, z])
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    # Spherical basis at observer
    e_r = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    e_theta = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)])
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0])
    pr = np.dot(ray_dir, e_r)
    ptheta = np.dot(ray_dir, e_theta)
    pphi = np.dot(ray_dir, e_phi)
    q0 = np.array([0.0, r, theta, phi])
    p0 = np.array(compute_null_4momentum_schwarzschild(q0, [pr, ptheta, pphi]))
    return q0, p0 