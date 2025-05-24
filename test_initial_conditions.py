import numpy as np
from einsteinpy.geodesic import Nulllike
from simulation.utils import get_initial_conditions

def test_initial_conditions_vs_einsteinpy():
    # Observer at (8, 0, 0), looking at origin
    observer_pos = np.array([8.0, 0.0, 0.0])
    # Pick a pixel on the image plane (e.g., center)
    fov = np.deg2rad(30)
    h, w = 100, 100
    plane_dist = 0.2 * np.linalg.norm(observer_pos)
    up = np.array([0, 0, 1])
    right = np.cross(up, observer_pos)
    right = right / np.linalg.norm(right)
    up_vec = np.cross(observer_pos, right)
    up_vec = up_vec / np.linalg.norm(up_vec)
    width = 2 * plane_dist * np.tan(fov/2)
    height = width * (h/w)
    plane_center = observer_pos - (observer_pos/np.linalg.norm(observer_pos)) * plane_dist
    i, j = h//2, w//2
    dx = (j + 0.5) / w - 0.5
    dy = (i + 0.5) / h - 0.5
    pixel_pos = plane_center + dx * width * right + dy * height * up_vec

    # Your code
    q0, p0 = get_initial_conditions(observer_pos, pixel_pos)

    # EinsteinPy
    r, th, ph = q0[1], q0[2], q0[3]
    pr, pth, pph = p0[1], p0[2], p0[3]
    geod = Nulllike(
        metric="Schwarzschild",
        metric_params=(),
        position=[r, th, ph],
        momentum=[pr, pth, pph],
        steps=1,
        delta=0.1,
        return_cartesian=False
    )
    # Compare initial positions and momenta
    print("Your q0:", q0)
    print("EinsteinPy q0:", geod.position)
    print("Your p0:", p0)
    print("EinsteinPy p0:", geod.momentum)
    assert np.allclose(q0, geod.position, atol=1e-10)
    assert np.allclose(p0, geod.momentum, atol=1e-10)
    print("Test passed: Initial conditions match EinsteinPy.")

if __name__ == "__main__":
    test_initial_conditions_vs_einsteinpy() 