"""Tests for the accretion disk and the camera pointing, ported from the Rust build
(disk.rs, camera.rs, integrator.rs and tests/physics.rs). Run from the repository
root:

    NUMBA_ENABLE_CUDASIM=1 python -m pytest tests/test_disk.py

The CUDA simulator runs the kernels on the CPU at roughly a millisecond per step, so
the kernel tests use a handful of rays.
"""
import json
import math
import os
import sys

os.environ.setdefault("NUMBA_ENABLE_CUDASIM", "1")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from simulation.disk import (ISCO_PER_M, PARALLEL_TOLERANCE, PEAK_RADIUS_PER_R_IN, Disk,
                             blackbody_chromaticity, crossing_check, disk_pixel,
                             hit_position_lab, keplerian_omega, redshift, srgb_encode,
                             temperature_shape)
from simulation.utils import (camera_basis, camera_rotation, get_initial_conditions, mat_vec,
                              pixel_positions, ray_directions, rot_x, rotation_about, vec_add,
                              vec_cross, vec_dot, vec_norm, vec_normalize, vec_scale)

M = 1.0
R_OBS = 30.0
OBS = np.array([R_OBS, 0.0, 0.0])
PI = math.pi
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "disk_scene.json")


def disk(normal):
    return Disk(normal, 6.0, 20.0, 6000.0, M, R_OBS)


def init(pixel):
    """(q0, p0, alpha0, beta, direction) of the ray from the observer through `pixel`."""
    pix = np.array(pixel, dtype=np.float64)
    q0, p0, alpha0, _, _, _, beta = get_initial_conditions(OBS, pix.copy(), mass_bh=M)
    direction = pix - OBS
    direction /= np.linalg.norm(direction)
    return q0, p0, float(alpha0), float(beta), direction


# --------------------------------------------------------------------------- redshift
def kinematic_redshift(n, x_hit, k_hat):
    """Independent truth for g, in the static orthonormal frame at the hit: the emitter
    moves at v = r Omega / sqrt(f) along n x x_hat, the photon's unit direction is k_hat,
    and g = sqrt(f) sqrt(1 - v^2) / (sqrt(f_obs) (1 - v . k_hat)). No rotated frame, no
    covariant momenta, no b_z."""
    r = vec_norm(x_hit)
    f = 1.0 - 2.0 * M / r
    omega = keplerian_omega(M, r)
    v = r * omega / math.sqrt(f)
    e_v = vec_cross(n, vec_normalize(x_hit))
    assert abs(vec_norm(e_v) - 1.0) < 1e-12, "x_hit must lie in the disk"
    return (math.sqrt(f) * math.sqrt(1.0 - v * v)
            / (math.sqrt(1.0 - 2.0 * M / R_OBS) * (1.0 - v * vec_dot(e_v, k_hat))))


def test_oblique_oracle_closed_form_equals_kinematic_redshift():
    """On an oblique disk, the closed form the renderer uses (conserved p_t, p_phi, the
    rotated normal's n'_z) equals the kinematic Doppler + gravitational factor from
    lab-frame geometry alone. This can only pass if the spin sense, the inclination
    convention and the direction of the frame rotation (R_x(-beta)) are all right."""
    n = Disk.normal_from_angles(math.radians(55.0), math.radians(30.0))
    d = disk(n)
    x_axis = [1.0, 0.0, 0.0]
    checked = 0
    for direction in [[-0.9, 0.3, 0.2], [-0.8, -0.4, 0.35], [-0.95, 0.1, -0.25],
                      [-0.7, -0.5, -0.4], [-0.99, 0.02, 0.1]]:
        direction = vec_normalize(direction)
        beta = math.atan2(direction[2], direction[1])
        phi_d, n_z = d.ray_frame(beta)
        assert math.isfinite(phi_d)
        orbit_normal = vec_normalize(vec_cross(x_axis, direction))
        # The disk plane meets the orbital plane along this line.
        line = vec_normalize(vec_cross(n, orbit_normal))
        in_plane = vec_cross(orbit_normal, line)
        rot = rot_x(-beta)
        for r_hit in [6.5, 9.0, 13.0, 19.5]:
            for sign in [1.0, -1.0]:
                x_hit = vec_scale(line, sign * r_hit)
                x_rot = mat_vec(rot, x_hit)
                assert abs(x_rot[2]) < 1e-12, "the hit lies in the rotated orbital plane"
                phi_h = math.atan2(x_rot[1], x_rot[0])
                off = round((phi_h - phi_d) / PI)
                assert abs(phi_h - phi_d - off * PI) < 1e-12, "the hit sits on phi_d + k pi"
                for psi in [0.3, 1.4, 2.5, 4.0, 5.5]:
                    k_hat = vec_add(vec_scale(line, math.cos(psi)), vec_scale(in_plane, math.sin(psi)))
                    # Covariant traced momentum at the hit, in the rotated frame, for a
                    # photon of unit energy in the static frame: p = -k.
                    f = 1.0 - 2.0 * M / r_hit
                    p_t = math.sqrt(f)
                    k_rot = mat_vec(rot, k_hat)
                    e_phi = [-math.sin(phi_h), math.cos(phi_h), 0.0]
                    p_phi = -r_hit * vec_dot(k_rot, e_phi)
                    g_closed = redshift(M, r_hit, p_t, p_phi, n_z, R_OBS)
                    g_kin = kinematic_redshift(n, x_hit, k_hat)
                    assert abs(g_closed - g_kin) < 1e-12, \
                        f"dir {direction} r {r_hit} psi {psi}: closed {g_closed} kinematic {g_kin}"
                    checked += 1
    assert checked == 200


def test_oracle_rejects_the_opposite_frame_rotation():
    """The same oracle with the wrong rotation sense must fail: the test has teeth."""
    n = Disk.normal_from_angles(math.radians(55.0), math.radians(30.0))
    direction = vec_normalize([-0.9, 0.3, 0.2])
    beta = math.atan2(direction[2], direction[1])
    wrong_n = mat_vec(rot_x(beta), n)
    orbit_normal = vec_normalize(vec_cross([1.0, 0.0, 0.0], direction))
    line = vec_normalize(vec_cross(n, orbit_normal))
    x_hit = vec_scale(line, 9.0)
    rot = rot_x(-beta)
    x_rot = mat_vec(rot, x_hit)
    phi_h = math.atan2(x_rot[1], x_rot[0])
    k_hat = vec_add(vec_scale(line, math.cos(0.3)), vec_scale(vec_cross(orbit_normal, line), math.sin(0.3)))
    p_t = math.sqrt(1.0 - 2.0 * M / 9.0)
    k_rot = mat_vec(rot, k_hat)
    p_phi = -9.0 * vec_dot(k_rot, [-math.sin(phi_h), math.cos(phi_h), 0.0])
    g_wrong = redshift(M, 9.0, p_t, p_phi, wrong_n[2], R_OBS)
    g_kin = kinematic_redshift(n, x_hit, k_hat)
    assert abs(g_wrong - g_kin) > 1e-3, f"{g_wrong} vs {g_kin}"


def test_face_on_disk_has_no_doppler_shift_whatever_the_pointing():
    for normal in [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]:
        d = disk(normal)
        for beta in [0.0, 0.4, 1.9, -2.3, PI]:
            phi_d, n_z = d.ray_frame(beta)
            assert n_z == 0.0, "n'_z vanishes exactly: b_z == 0"
            assert math.isfinite(phi_d)
            for r in [6.0, 10.0, 20.0]:
                g = redshift(M, r, 0.97, 7.3, n_z, R_OBS)
                expect = math.sqrt(1.0 - 3.0 * M / r) / math.sqrt(1.0 - 2.0 * M / R_OBS)
                assert abs(g - expect) < 1e-15


def test_left_side_of_an_edge_on_disk_approaches_the_observer():
    """Written pin, independent of every formula: with n = +z the material at
    (0, -r, 0) moves along n x x_hat = +x, towards the observer, so a pixel on the
    observer's left sees g > 1 and one on the right g < 1."""
    d = disk([0.0, 0.0, 1.0])
    left = init([24.0, -2.0, 0.01])
    right = init([24.0, 2.0, 0.01])
    for (q0, p0, alpha0, beta, direction), approaching in [(left, True), (right, False)]:
        _, n_z = d.ray_frame(beta)
        g = redshift(M, 8.0, p0[0], p0[3], n_z, R_OBS)
        assert math.isfinite(g) and g > 0.0
        assert (g > 1.0) == approaching, f"g = {g} for pixel {direction}"
    # Flipping the normal flips the sense of rotation.
    flipped = disk([0.0, 0.0, -1.0])
    _, p0, _, beta, _ = left
    assert redshift(M, 8.0, p0[0], p0[3], flipped.ray_frame(beta)[1], R_OBS) < 1.0


def test_reversing_the_normal_mirrors_the_redshift_map():
    n = Disk.normal_from_angles(math.radians(70.0), 0.0)
    d = disk(n)
    anti = disk([-n[0], -n[1], -n[2]])
    for pix in [[24.0, -3.0, 1.0], [24.0, 1.5, -2.0], [24.0, 4.0, 0.3]]:
        _, pa, _, ba, _ = init(pix)
        _, pb, _, bb, _ = init([pix[0], -pix[1], pix[2]])
        ga = redshift(M, 9.0, pa[0], pa[3], d.ray_frame(ba)[1], R_OBS)
        gb = redshift(M, 9.0, pb[0], pb[3], anti.ray_frame(bb)[1], R_OBS)
        assert abs(ga - gb) < 1e-12, f"{ga} vs {gb}"


def test_rays_in_the_disk_plane_are_flagged_parallel():
    d = disk([0.0, 0.0, 1.0])
    assert math.isnan(d.ray_frame(0.0)[0])
    assert math.isnan(d.ray_frame(PI)[0])
    phi_d, n_z = d.ray_frame(0.3)
    assert math.isfinite(phi_d)
    # n' = (0, sin beta, cos beta): the crossing line is the x axis.
    assert abs(phi_d) < 1e-15 and abs(n_z - math.cos(0.3)) < 1e-15
    back = d.ray_frame(-0.3)[0]
    assert abs(abs(back) - PI) < 1e-15, f"crossing line at +-pi: {back}"
    assert PARALLEL_TOLERANCE == 1e-9


def test_normal_from_angles_follows_the_documented_convention():
    n = Disk.normal_from_angles(0.0, 1.0)
    assert abs(n[0] - 1.0) < 1e-15 and abs(n[1]) < 1e-15 and abs(n[2]) < 1e-15
    n = Disk.normal_from_angles(PI / 2.0, 0.0)
    assert abs(n[0]) < 1e-15 and abs(n[1]) < 1e-15 and abs(n[2] - 1.0) < 1e-15
    n = Disk.normal_from_angles(PI / 2.0, PI / 2.0)
    assert abs(n[1] + 1.0) < 1e-15, "azimuth turns the normal right-handedly about +x: z -> -y"
    n = Disk.normal_from_angles(1.0, 0.0)
    near_side_z = -n[0] / n[2]  # z of the disk point (1, 0, z): below the hole
    assert near_side_z < 0.0
    anti = Disk.normal_from_angles(PI - 1.0, PI)
    n1 = Disk.normal_from_angles(1.0, 0.0)
    for k in range(3):
        assert abs(anti[k] + n1[k]) < 1e-15, "(i, a) -> (180 - i, a + 180) reverses n"


# --------------------------------------------------------------------------- emission
def test_temperature_profile_peaks_at_49_over_36_r_in():
    d = disk([1.0, 0.0, 0.0])
    assert d.temperature(6.0) == 0.0
    assert d.temperature(5.0) == 0.0
    assert temperature_shape(6.0, 6.0) == 0.0
    r_peak = 6.0 * PEAK_RADIUS_PER_R_IN
    assert abs(d.temperature(r_peak) - 6000.0) < 1e-9
    best = 0.0
    r = 6.0
    while r < 20.0:
        best = max(best, d.temperature(r))
        r += 0.001
    assert best <= 6000.0 + 1e-6 and best > 5999.9
    assert d.temperature(20.0) < d.temperature(10.0)
    assert ISCO_PER_M == 6.0 and PEAK_RADIUS_PER_R_IN == 49.0 / 36.0


def test_blackbody_colours_run_from_orange_to_blue():
    cool = blackbody_chromaticity(3000.0)
    warm = blackbody_chromaticity(6500.0)
    hot = blackbody_chromaticity(15000.0)
    for c in [cool, warm, hot]:
        assert all(0.0 <= v <= 1.0 for v in c)
        assert abs(max(c) - 1.0) < 1e-12
    assert cool[0] > cool[1] > cool[2]
    assert hot[2] > hot[0]
    assert all(v > 0.85 for v in warm), f"6500 K is close to white: {warm}"
    assert blackbody_chromaticity(100.0) == blackbody_chromaticity(1667.0)
    assert blackbody_chromaticity(1e6) == blackbody_chromaticity(25000.0)


def test_disk_pixel_brightens_with_g_and_is_black_when_cold():
    dim = disk_pixel(0.8, 6000.0, 6000.0)
    unit = disk_pixel(1.0, 6000.0, 6000.0)
    bright = disk_pixel(1.2, 6000.0, 6000.0)
    for px in (dim, unit, bright):
        assert isinstance(px, tuple) and len(px) == 3 and all(isinstance(c, int) for c in px)
    assert sum(dim) < sum(unit) < sum(bright), f"{dim} {unit} {bright}"
    assert bright[2] > unit[2], "the approaching side is hotter, hence bluer"
    assert disk_pixel(1.0, 0.0, 6000.0) == (0, 0, 0)
    assert disk_pixel(math.nan, 6000.0, 6000.0) == (0, 0, 0)
    assert srgb_encode(0.0) == 0
    assert srgb_encode(1.0) == 255
    assert srgb_encode(math.nan) == 0
    # f64::round semantics: halves away from zero, not to even
    assert srgb_encode(0.5 / 255.0 / 12.92 * 1.0000000001) == 1


def test_straight_ray_hits_the_plane_where_geometry_says():
    d = disk([0.0, 0.0, 1.0])
    hit = d.straight_ray_hit([30.0, 0.0, 3.0], vec_normalize([-1.0, 0.0, -0.2]))
    assert hit is not None
    assert abs(hit[2]) < 1e-12 and abs(hit[0] - 15.0) < 1e-12
    assert d.straight_ray_hit([30.0, 0.0, 3.0], vec_normalize([-1.0, 0.0, -1.0])) is None, "inside r_in"
    assert d.straight_ray_hit([30.0, 0.0, 3.0], [-1.0, 0.0, 0.0]) is None, "parallel"
    assert d.straight_ray_hit([30.0, 0.0, 3.0], vec_normalize([-1.0, 0.0, 0.2])) is None, "behind"


def test_validation_rejects_unphysical_disks():
    with pytest.raises(ValueError, match="inner radius"):
        Disk([1.0, 0.0, 0.0], 5.9, 20.0, 6000.0, M, R_OBS)
    Disk([1.0, 0.0, 0.0], 12.0, 40.0, 6000.0, 2.0, 60.0)
    with pytest.raises(ValueError, match="inner radius"):
        Disk([1.0, 0.0, 0.0], 11.9, 40.0, 6000.0, 2.0, 60.0)
    with pytest.raises(ValueError, match="outer radius"):
        Disk([1.0, 0.0, 0.0], 6.0, 30.0, 6000.0, M, R_OBS)
    with pytest.raises(ValueError, match="outer radius"):
        Disk([1.0, 0.0, 0.0], 6.0, 6.0, 6000.0, M, R_OBS)
    with pytest.raises(ValueError, match="temperature"):
        Disk([1.0, 0.0, 0.0], 6.0, 20.0, 0.0, M, R_OBS)
    with pytest.raises(ValueError, match="normal"):
        Disk([0.0, 0.0, 0.0], 6.0, 20.0, 6000.0, M, R_OBS)
    d = Disk([0.0, 0.0, 2.0], 6.0, 20.0, 6000.0, M, R_OBS)
    assert tuple(d.normal) == (0.0, 0.0, 1.0)


def test_hit_position_lab_undoes_the_ray_rotation():
    # A hit at (r, phi) in the frame rotated by -beta sits at R_x(beta) (r cos, r sin, 0).
    pos = hit_position_lab(9.0, PI, PI)
    assert abs(pos[0] + 9.0) < 1e-12 and abs(pos[2]) < 1e-9
    pos = hit_position_lab(9.0, PI / 2.0, 0.3)
    assert abs(pos[0]) < 1e-12 and abs(pos[1] - 9.0 * math.cos(0.3)) < 1e-12 \
        and abs(pos[2] - 9.0 * math.sin(0.3)) < 1e-12


# --------------------------------------------------------------------------- crossings
def test_crossing_detector_on_a_straight_line_converges_as_the_step_squared():
    """A straight line in a flat plane, fed to the detector step by step: the hit radius
    converges to the exact line-line intersection as the square of the step."""
    x0 = [30.0, -4.0]
    d = [-0.96, 0.28]
    phi_d = 2.0
    # Exact crossing: x(l) parallel to (cos phi_d, sin phi_d).
    sd, cd = math.sin(phi_d), math.cos(phi_d)
    lam = -(x0[0] * sd - x0[1] * cd) / (d[0] * sd - d[1] * cd)
    exact_r = math.hypot(x0[0] + lam * d[0], x0[1] + lam * d[1])

    def state(l):
        x = x0[0] + l * d[0]
        y = x0[1] + l * d[1]
        return math.hypot(x, y), math.atan2(y, x)

    def run(h):
        lam = 0.0
        r_prev, phi_prev = state(0.0)
        hit = None
        count = 0
        while lam < 80.0:
            lam += h
            r, phi = state(lam)
            # Unwrap phi by hand as the integrator's monotone phi would be.
            while phi - phi_prev > PI:
                phi -= 2.0 * PI
            while phi - phi_prev < -PI:
                phi += 2.0 * PI
            c = crossing_check(phi_d, 0.0, 1e9, r_prev, phi_prev, r, phi)
            if c is not None:
                count += 1
                assert c[2]
                if hit is None:
                    hit = c[0]
            r_prev, phi_prev = r, phi
        assert hit is not None, "the line crosses"
        return hit, count

    (r1, n1), (r2, n2), (r3, n3) = run(0.2), run(0.1), run(0.05)
    assert (n1, n2, n3) == (1, 1, 1), "a straight line crosses a line through the origin once"
    e1, e2, e3 = abs(r1 - exact_r), abs(r2 - exact_r), abs(r3 - exact_r)
    assert e3 < 1e-3 and e2 > e3 and e1 > e2, f"{e1} {e2} {e3}"
    assert e1 / e2 > 3.0 and e2 / e3 > 3.0, f"second order: {e1} {e2} {e3}"


def test_circular_motion_crosses_the_plane_twice_per_revolution_at_the_orbit_radius():
    phi_d, r_in, r_out = 0.7, 5.0, 20.0
    r = 9.0
    hits = []
    phi_prev = phi = 0.0
    while phi < 4.0 * PI:
        phi += 0.013
        c = crossing_check(phi_d, r_in, r_out, r, phi_prev, r, phi)
        if c is not None:
            assert c[0] == r
            assert c[2]
            hits.append(c[1])
        phi_prev = phi
    assert len(hits) == 4, hits
    for k, h in enumerate(hits):
        assert abs(h - (0.7 + k * PI)) < 1e-12, f"crossing {k} at {h}"
    # Going the other way (phi decreasing) finds the same crossings.
    back = []
    phi = phi_prev = 4.0 * PI
    while phi > 0.0:
        phi -= 0.013
        c = crossing_check(phi_d, r_in, r_out, r, phi_prev, r, phi)
        if c is not None:
            back.append(c[1])
        phi_prev = phi
    assert len(back) == 4
    # A jump of two boundaries in one step is not a crossing.
    assert crossing_check(phi_d, r_in, r_out, r, 0.0, r, 0.7 + 2.5 * PI) is None
    assert crossing_check(phi_d, r_in, r_out, r, 1.0, r, 1.2) is None
    # Out of the band: reported, flagged.
    c = crossing_check(phi_d, r_in, r_out, 25.0, 0.6, 25.0, 0.8)
    assert c is not None and c[0] == 25.0 and not c[2]


# --------------------------------------------------------------------------- kernels
def _integrator(steps=20000):
    from simulation.cuda_geodesic import CUDASchwarzschildIntegrator
    return CUDASchwarzschildIntegrator(steps=steps, delta=0.01, mass=M, omega=0.01, r_max=31.0)


def test_disk_kernel_with_the_disk_off_is_bit_identical_to_the_old_kernel():
    """The old kernel is the regression reference: with every disk_phi NaN the new loop
    must reproduce it bit for bit on pipeline rays (captured and escaped alike)."""
    from simulation.cuda_geodesic import fantasy_integrate_batch_ord2

    pixels = [[24.0, 0.0, 0.0], [24.0, 0.4, 0.2], [24.0, -0.9, 0.0], [24.0, 1.4, -1.0],
              [24.0, 2.4, 0.5], [24.0, -3.5, 2.0], [24.0, 5.0, -5.0], [24.0, 0.3, -6.0]]
    q0s = np.array([init(p)[0] for p in pixels])
    p0s = np.array([init(p)[1] for p in pixels])
    n = len(pixels)
    reference = np.zeros((n, 4))
    fantasy_integrate_batch_ord2[1, 32](q0s.copy(), p0s.copy(), 20000, 0.01, 2.0, 31.0, 0.01, reference)
    integ = _integrator()
    out, status, hits, crossings = integ.integrate_batch(q0s, p0s)
    assert np.array_equal(out, reference)
    assert np.all(np.isnan(hits))
    assert np.all(crossings == 0)
    assert set(status.tolist()) <= {1, 2}, status
    assert status[0] == 1 and status[-1] == 2
    # the exit test is the old one: captured rays stopped at r <= 1.1 rs, escaped at r >= 31
    assert np.all(out[status == 1, 1] <= 2.2) and np.all(out[status == 2, 1] >= 31.0)
    # an explicit all-NaN disk_phi with a band is the same thing
    out2, status2, *_ = integ.integrate_batch(q0s, p0s, disk_phi=np.full(n, np.nan), disk_r_in=6.0, disk_r_out=20.0)
    assert np.array_equal(out2, reference) and np.array_equal(status2, status)


def test_a_ray_aimed_through_the_disk_stops_there():
    """Face-on disk (plane x = 0, crossing line at phi = +-pi/2 in every ray's frame): a
    ray with impact parameter ~10 crosses it inside the band; a radial ray never
    crosses the plane through the hole; a crossing outside the band is counted but
    does not stop the ray; an edge-on disk registers no crossing at the observer."""
    aimed = init([24.0, 2.0, 0.0])
    radial = init([24.0, 0.0, 0.0])
    wide = init([24.0, 6.0, 0.0])  # impact parameter ~24 > r_out
    q0s = np.array([aimed[0], radial[0], wide[0]])
    p0s = np.array([aimed[1], radial[1], wide[1]])
    integ = _integrator()
    out, status, hits, crossings = integ.integrate_batch(
        q0s, p0s, disk_phi=[PI / 2.0, PI / 2.0, PI / 2.0], disk_r_in=6.0, disk_r_out=20.0)
    assert status[0] == 4
    r_hit, phi_hit = hits[0]
    assert 6.0 < r_hit < 20.0, hits[0]
    assert abs(phi_hit - PI / 2.0) < 1e-12
    assert crossings[0] == 1
    assert out[0, 3] >= phi_hit and out[0, 3] - phi_hit < 0.01, "stopped just past the crossing"
    assert status[1] == 1 and crossings[1] == 0 and np.isnan(hits[1]).all()
    assert status[2] == 2 and crossings[2] == 1 and np.isnan(hits[2]).all()
    # The trajectory recorder stops at the hit and ends with the post-crossing state.
    traj = integ.integrate_batch_full(q0s[:1], p0s[:1], disk_phi=[PI / 2.0], disk_r_in=6.0, disk_r_out=20.0)
    rows = traj[0]
    last = int(np.flatnonzero(rows.any(axis=1))[-1])
    assert np.array_equal(rows[last], out[0])
    assert rows[last - 1, 3] < phi_hit <= rows[last, 3]
    # Edge-on disk n = z: the crossing line is the x axis and every ray starts on it.
    edge = init([24.0, 6.0, 6.0])
    phi_d = 0.0 if math.sin(edge[3]) > 0.0 else PI
    _, st, _, cr = integ.integrate_batch(edge[0][None], edge[1][None], disk_phi=[phi_d], disk_r_in=6.0, disk_r_out=29.5)
    assert st[0] == 2 and cr[0] == 0


# --------------------------------------------------------------------------- camera
def directions(h, w, yaw=0.0, pitch=0.0, roll=0.0, fov=math.radians(80.0)):
    axis, right, up = camera_basis(yaw, pitch, roll)
    return ray_directions(OBS, pixel_positions(OBS, fov, h, w, axis, right, up))


def hole_pixel(dirs):
    """Where the black hole appears: the pixel whose direction is closest to -x."""
    flat = int(np.argmax(-dirs[..., 0]))
    return divmod(flat, dirs.shape[1])


def test_zero_rotation_gives_the_unrotated_basis_exactly():
    axis, right, up = camera_basis(0.0, 0.0, 0.0)
    assert axis.tolist() == [-1.0, 0.0, 0.0]
    assert right.tolist() == [0.0, 1.0, 0.0]
    assert up.tolist() == [0.0, 0.0, 1.0]
    assert camera_basis()[0].tolist() == [-1.0, 0.0, 0.0]
    # the grid of the unrotated camera is the one both renderers used before
    pos = pixel_positions(OBS, math.radians(80.0), 6, 8, axis, right, up)
    plane_dist = 0.2 * 30.0
    plane_width = 2 * plane_dist * np.tan(math.radians(80.0) / 2)
    plane_height = plane_width * (6 / 8)
    expect = np.array([24.0, 0.0, 0.0]) + ((4 + 0.5) / 8 - 0.5) * plane_width * np.array([0, 1, 0]) \
        + ((3 + 0.5) / 6 - 0.5) * plane_height * np.array([0, 0, 1])
    assert np.array_equal(pos[3, 4], expect)
    assert pos[3, 4][1] > 0.0 and pos[3, 4][2] > 0.0
    d = directions(6, 8)[0, 0]
    assert d[0] < 0.0 and d[1] < 0.0 and d[2] < 0.0, "pixel (0,0) is bottom-left"
    with pytest.raises(ValueError):
        camera_basis(math.nan, 0.0, 0.0)


def test_rotation_about_matches_rot_x_and_composes():
    a = 0.7
    r1 = rotation_about([1.0, 0.0, 0.0], a)
    r2 = rot_x(a)
    for i in range(3):
        for j in range(3):
            assert abs(r1[i][j] - r2[i][j]) < 1e-15
    # Rotating z about y by +90 degrees gives +x (right-handed).
    v = mat_vec(rotation_about([0.0, 1.0, 0.0], PI / 2.0), [0.0, 0.0, 1.0])
    assert abs(v[0] - 1.0) < 1e-15 and abs(v[1]) < 1e-15 and abs(v[2]) < 1e-15


def test_rotated_camera_is_the_rigid_rotation_of_the_unrotated_one():
    fov = math.radians(70.0)
    plain = directions(6, 9, fov=fov)
    for yaw, pitch, roll in [(0.3, 0.0, 0.0), (0.0, -0.4, 0.0), (0.0, 0.0, 1.1), (0.5, -0.2, 0.7)]:
        axis, right, up = camera_basis(yaw, pitch, roll)
        # Orthonormal, right-handed: right x up = -optical_axis.
        h = vec_cross(right, up)
        for hk, ak in zip(h, axis):
            assert abs(hk + ak) < 1e-15
        assert abs(vec_dot(right, up)) < 1e-15
        rot = camera_rotation(yaw, pitch, roll)
        got = directions(6, 9, yaw, pitch, roll, fov=fov)
        for i in range(6):
            for j in range(9):
                expect = mat_vec(rot, plain[i, j].tolist())
                for k in range(3):
                    assert abs(expect[k] - got[i, j, k]) < 1e-14, f"({i},{j}) {yaw} {pitch} {roll}"


def test_yaw_and_pitch_move_the_hole_as_documented():
    fov = math.radians(80.0)
    assert hole_pixel(directions(21, 21)) == (10, 10)
    # The centre ray of a yawed camera is exactly R (-1, 0, 0).
    yawed = directions(21, 21, 0.3, 0.0, 0.0)
    expect = mat_vec(camera_rotation(0.3, 0.0, 0.0), [-1.0, 0.0, 0.0])
    got = yawed[10, 10]
    for k in range(3):
        assert abs(expect[k] - got[k]) < 1e-15
    assert abs(got[1] + math.sin(0.3)) < 1e-15, "positive yaw looks to the left (-y)"
    i, j = hole_pixel(yawed)
    assert j > 10 and i == 10, f"the hole moves right: ({i},{j})"
    pitched = directions(21, 21, 0.0, 0.3, 0.0)
    assert pitched[10, 10][2] > 0.0, "positive pitch looks up"
    i, j = hole_pixel(pitched)
    assert i < 10 and j == 10, f"the hole moves down: ({i},{j})"
    # Yaw by half the field of view puts the hole at the image edge.
    assert hole_pixel(directions(21, 21, fov / 2.0, 0.0, 0.0)) == (10, 20)


def test_roll_by_a_right_angle_permutes_the_pixels():
    fov = math.radians(60.0)
    plain = directions(8, 8, fov=fov)
    rolled = directions(8, 8, 0.0, 0.0, PI / 2.0, fov=fov)
    # Positive roll turns the image counter-clockwise: what the rolled camera sees at
    # (i, j) the plain camera saw at row 7 - j, column i.
    for i in range(8):
        for j in range(8):
            a = rolled[i, j]
            b = plain[7 - j, i]
            for k in range(3):
                assert abs(a[k] - b[k]) < 1e-14, f"({i},{j})"


# --------------------------------------------------------------------------- fixture
def _nan(v):
    return math.nan if v is None else v


def test_disk_scene_fixture_is_self_consistent():
    """The cross-build fixture (tests/gen_disk_fixture.py): its stored g is the closed
    form of its stored (r_hit, p0, n_z), its (phi_d, n_z) are the disk in each ray's
    frame, and every hit sits on the crossing line inside the band. Pins the format
    the Rust build's test reads."""
    with open(FIXTURE) as f:
        doc = json.load(f)
    s = doc["scene"]
    assert s["size"] == 16 and s["mass"] == 1.0 and s["steps"] == 20000
    assert doc["nan_encoding"] == "null"
    normal = Disk.normal_from_angles(math.radians(s["disk_inclination_deg"]), math.radians(s["disk_azimuth_deg"]))
    d = Disk(normal, s["disk_inner"], s["disk_outer"], s["disk_temperature"], s["mass"], s["r_obs"])
    assert list(d.normal) == s["disk_normal"]
    pixels = doc["pixels"]
    assert len(pixels) == 256
    n_hits = 0
    for idx, px in enumerate(pixels):
        assert (px["i"], px["j"]) == divmod(idx, 16)
        assert px["status"] in (0, 1, 2, 4)
        phi_d, n_z = d.ray_frame(px["beta"])
        assert phi_d == _nan(px["phi_d"]) and n_z == px["n_z"]
        if px["status"] == 4:
            n_hits += 1
            r_hit, phi_hit = px["r_hit"], px["phi_hit"]
            assert d.r_in <= r_hit <= d.r_out
            k = round((phi_hit - phi_d) / PI)
            assert abs(phi_hit - (phi_d + k * PI)) < 1e-12
            assert px["plane_crossings"] >= 1
            g = redshift(s["mass"], r_hit, px["p0"][0], px["p0"][3], n_z, s["r_obs"])
            assert abs(g - px["g"]) < 1e-12
            assert math.isfinite(g) and g > 0.0
            assert px["q"][3] >= phi_hit and px["q"][3] - phi_hit < 0.02
        else:
            assert px["r_hit"] is None and px["phi_hit"] is None and px["g"] is None
    assert n_hits > 30, n_hits
    gs = [px["g"] for px in pixels if px["status"] == 4]
    assert max(gs) > 1.0 and min(gs) < 0.9, "Doppler both ways"
