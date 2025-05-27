import numpy as np
from einsteinpy.coordinates.utils import cartesian_to_spherical_fast
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
    # Use EinsteinPy's fast cartesian to spherical conversion for the direction
    _, _, _, _, pr, ptheta, pphi = cartesian_to_spherical_fast(0.0, x, y, z, ray_dir[0], ray_dir[1], ray_dir[2], velocities_provided=True)
    q0 = np.array([0.0, r, theta, phi])
    p0 = np.array([0.0, pr, ptheta, pphi])
    return q0, p0 