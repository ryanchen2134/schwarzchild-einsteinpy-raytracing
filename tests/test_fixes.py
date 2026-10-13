"""Regression tests for the covariant momentum, the scale-free integrator and the
flat render. Run from the repository root:

    NUMBA_ENABLE_CUDASIM=1 python -m pytest tests/test_fixes.py

The CUDA simulator runs the kernels on the CPU; these tests use few rays and few
steps so they finish in seconds.
"""
import os
import sys

os.environ.setdefault("NUMBA_ENABLE_CUDASIM", "1")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from PIL import Image

from simulation.utils import angles_to_p_sph, build_null_4momentum_ep_sph, get_initial_conditions

R_OBS = 30.0


# --------------------------------------------------------------------------- momentum
@pytest.mark.parametrize("alpha", [0.0, 0.05, 0.17, 0.4, 1.0])
def test_covariant_momentum_points_along_the_requested_direction(alpha):
    p_r, p_th, p_ph = angles_to_p_sph(alpha, 0.0, R_OBS)
    f = 1.0 - 2.0 / R_OBS
    # orthonormal components recovered from the covariant ones
    n_r = p_r * np.sqrt(f)
    n_ph = p_ph / R_OBS
    assert n_r == pytest.approx(-np.cos(alpha), abs=1e-15)
    assert n_ph == pytest.approx(np.sin(alpha), abs=1e-15)
    assert p_th == 0.0
    assert n_r * n_r + n_ph * n_ph == pytest.approx(1.0, abs=1e-14)


def test_momentum_is_null_and_alpha0_is_the_angle_from_the_optical_axis():
    obs = np.array([R_OBS, 0.0, 0.0])
    for pix in [[24.0, 1.3, -0.7], [24.0, -3.0, 0.5], [24.0, 0.0, 2.0]]:
        q0, p0, alpha0, *_ , beta = get_initial_conditions(obs, np.array(pix), mass_bh=1.0)
        r = q0[1]
        f = 1.0 - 2.0 / r
        p_t, p_r, p_th, p_ph = p0
        H = -p_t * p_t / f + f * p_r * p_r + p_th * p_th / (r * r) + p_ph * p_ph / (r * r)
        assert abs(H) < 1e-12
        direction = np.array(pix) - obs
        direction /= np.linalg.norm(direction)
        assert alpha0 == pytest.approx(np.arccos(-direction[0]), abs=1e-12)


def test_normalise_false_returns_the_orthonormal_components():
    n = angles_to_p_sph(0.3, 0.1, R_OBS, normalise=False)
    assert np.linalg.norm(n) == pytest.approx(1.0, abs=1e-15)


# --------------------------------------------------------------------------- integrator
def _ray(alpha, mass):
    r_obs = R_OBS * mass
    p_sph = angles_to_p_sph(alpha, 0.0, r_obs, mass_bh=mass)
    p0 = build_null_4momentum_ep_sph(p_sph, np.array([r_obs, np.pi / 2, 0.0]), mass_bh=mass, future=True)
    q0 = np.array([0.0, r_obs, np.pi / 2, 0.0])
    return q0[None, :], p0[None, :]


def test_metric_derivative_uses_rs_not_a_hard_coded_2():
    from simulation.cuda_geodesic import _metric_derivative

    q = np.array([0.0, 7.0, np.pi / 2, 0.0])
    gprime = np.zeros((4, 4))
    _metric_derivative(q, 4.0, 1, gprime)  # M = 2
    assert gprime[0, 0] == pytest.approx(4.0 / (7.0 - 4.0) ** 2)
    assert gprime[1, 1] == pytest.approx(4.0 / 49.0)


def test_integrator_is_scale_free_in_the_mass():
    from simulation.cuda_geodesic import CUDASchwarzschildIntegrator

    alpha = 0.2
    q1, p1 = _ray(alpha, 1.0)
    unit = CUDASchwarzschildIntegrator(steps=3000, delta=0.01, mass=1.0, omega=0.01, r_max=31.0)
    out1, _ = unit.integrate_batch(q1, p1)
    # M = 2: lengths double (exact in binary), so the run must be bit-identical.
    q2, p2 = _ray(alpha, 2.0)
    two = CUDASchwarzschildIntegrator(steps=3000, delta=0.02, mass=2.0, omega=0.01, r_max=62.0)
    out2, _ = two.integrate_batch(q2, p2)
    assert out2[0, 0] == 2.0 * out1[0, 0]
    assert out2[0, 1] == 2.0 * out1[0, 1]
    assert out2[0, 2] == out1[0, 2]
    assert out2[0, 3] == out1[0, 3]
    # M = 2.5 is not a power of two: equal to rounding.
    q3, p3 = _ray(alpha, 2.5)
    big = CUDASchwarzschildIntegrator(steps=3000, delta=0.025, mass=2.5, omega=0.01, r_max=77.5)
    out3, _ = big.integrate_batch(q3, p3)
    assert out3[0, 1] / 2.5 == pytest.approx(out1[0, 1], abs=1e-9)
    assert out3[0, 3] == pytest.approx(out1[0, 3], abs=1e-9)
    # The trajectory kernel is rescaled the same way.
    traj1 = unit.integrate_batch_full(q1, p1)
    traj2 = two.integrate_batch_full(q2, p2)
    assert np.array_equal(traj2[:, :, 1], 2.0 * traj1[:, :, 1])
    assert np.array_equal(traj2[:, :, 3], traj1[:, :, 3])


def test_unit_mass_run_is_unchanged_by_the_rescaling():
    """At M = 1 every scale factor is 1.0, so the integrator output must be identical to
    calling the kernel directly, as the pre-fix code did."""
    from simulation.cuda_geodesic import CUDASchwarzschildIntegrator, fantasy_integrate_batch_ord2

    q, p = _ray(0.2, 1.0)
    integ = CUDASchwarzschildIntegrator(steps=2000, delta=0.01, mass=1.0, omega=0.01, r_max=31.0)
    out, _ = integ.integrate_batch(q, p)
    direct = np.zeros((1, 4))
    fantasy_integrate_batch_ord2[1, 32](q, p, 2000, 0.01, 2.0, 31.0, 0.01, direct)
    assert np.array_equal(out, direct)


# --------------------------------------------------------------------------- flat render
class _Observer:
    def __init__(self, position, fov, image_size):
        self.position = np.array(position)
        self.fov = fov
        self.image_size = image_size


def test_full_sky_flat_render_is_textured_and_not_mirrored(tmp_path):
    from simulation.background import _in_phi_patch, save_no_gravity_image_with_background

    # a 360-degree patch contains every azimuth
    assert _in_phi_patch(0.0, 0.0, 2 * np.pi)
    assert _in_phi_patch(3.0, 0.0, 2 * np.pi)
    assert _in_phi_patch(-1.0, 0.0, 2 * np.pi)
    # a narrow patch wraps around phi = 0
    assert _in_phi_patch(np.radians(5), np.radians(350), np.radians(10))
    assert _in_phi_patch(np.radians(-5), np.radians(350), np.radians(10))
    assert not _in_phi_patch(np.radians(20), np.radians(350), np.radians(10))

    tex = tmp_path / "sky.png"
    Image.fromarray(np.full((8, 16, 3), 200, dtype=np.uint8)).save(tex)
    out = tmp_path / "out" / "no_gravity.png"
    h, w = 8, 8
    observer = _Observer([R_OBS, 0.0, 0.0], np.radians(80.0), (h, w))
    trajs = save_no_gravity_image_with_background(
        observer, str(tex), str(out), boundary_radius=31.0,
        patch_center_theta=np.pi / 2, patch_center_phi=np.pi,
        patch_size_theta=np.pi, patch_size_phi=2 * np.pi,
        return_sampled_trajectories=True, n_sampled=h * w)
    img = np.array(Image.open(out))
    assert img.shape == (h, w, 3)
    assert (img == 200).all(), "every straight ray sees the full-sky texture"
    # trajectories are row-major because every pixel is sampled; the right half of the
    # image (large j) must look towards +y, as the curved renderer does
    hits = np.array([t[-1] for t in trajs]).reshape(h, w, 3)
    assert (hits[:, w // 2:, 1] > 0).all()
    assert (hits[:, : w // 2, 1] < 0).all()
    assert (hits[h // 2:, :, 2] > 0).all(), "large i looks up (+z)"
