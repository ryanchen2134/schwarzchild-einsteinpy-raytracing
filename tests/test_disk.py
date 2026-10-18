"""Tests for the accretion disk and the camera pointing, ported from the Rust build
(disk.rs, camera.rs, integrator.rs and tests/physics.rs). Run from the repository
root:

    NUMBA_ENABLE_CUDASIM=1 python -m pytest tests/test_disk.py

The CUDA simulator runs the kernels on the CPU at roughly a millisecond per step, so
the kernel tests use a handful of rays; the renderer test (a 4x4 image with and
without the disk) takes about a minute.
"""
import json
import math
import os
import sys

os.environ.setdefault("NUMBA_ENABLE_CUDASIM", "1")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from simulation.disk import (ISCO_PER_M, LUMINANCE_WEIGHTS, PARALLEL_TOLERANCE,
                             PEAK_RADIUS_PER_R_IN, Disk, blackbody_chromaticity,
                             crossing_check, disk_pixel, disk_rgb_linear, hit_position_lab,
                             keplerian_omega, luminance, redshift, srgb_encode,
                             temperature_shape, tonemap)
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
    covariant momenta, no b_z, and nothing from simulation.disk beyond the vector
    helpers."""
    r = vec_norm(x_hit)
    f = 1.0 - 2.0 * M / r
    # Circular-orbit speed measured by a static observer: v^2 = M/(r - 2M).
    v = math.sqrt(M / (r - 2.0 * M))
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
    # T^4 follows the profile shape exactly (pins the quarter power, which the peak
    # normalisation alone would not): T(r)^4/T_peak^4 = shape(r)/shape(49/6).
    for r in (8.0, 12.0, 18.0):
        ratio = (d.temperature(r) / 6000.0) ** 4
        expect = temperature_shape(6.0, r) / temperature_shape(6.0, r_peak)
        assert abs(ratio - expect) < 1e-12, f"r = {r}: {ratio} vs {expect}"
    # One hand value: shape(12) = (1 - sqrt(1/2))/1728, shape(49/6) = (1 - 6/7)/(49/6)^3.
    by_hand = ((1.0 - math.sqrt(0.5)) / 1728.0 / ((1.0 / 7.0) / (49.0 / 6.0) ** 3)) ** 0.25 * 6000.0
    assert abs(d.temperature(12.0) - by_hand) < 1e-9, f"{d.temperature(12.0)} vs {by_hand}"


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


# disk_pixel(g, 6000, 6000) for g = 0.8, 1.0, 1.2, 1.5, pinned by the Rust build's
# disk.rs (PINNED_PIXELS); both builds assert the same values.
PINNED_PIXELS = [(173, 154, 135), (216, 206, 204), (240, 239, 254), (254, 254, 255)]


def test_disk_pixel_brightens_with_g_and_is_black_when_cold():
    # Linear luminance is exactly the tone-mapped intensity, hence monotone in g, across
    # the range a disk produces (the most blueshifted material must be the brightest,
    # not just the bluest).
    last = -1.0
    g = 0.3
    while g < 3.0:
        c = disk_rgb_linear(g, 6000.0, 6000.0)
        x = g
        expect = tonemap(x * x * x * x)
        assert abs(luminance(c) - expect) < 1e-12, f"g = {g}: Y {luminance(c)} vs {expect}"
        assert all(0.0 <= v <= 1.0 for v in c), f"g = {g}: {c}"
        # Strictly increasing until the tone curve reaches 1 to within rounding
        # (gT >~ 2.4 T_peak), never decreasing.
        assert luminance(c) >= last, f"g = {g}"
        assert luminance(c) > last or last > 1.0 - 1e-12, f"g = {g}"
        last = luminance(c)
        g += 0.01
    dim = disk_pixel(0.8, 6000.0, 6000.0)
    unit = disk_pixel(1.0, 6000.0, 6000.0)
    bright = disk_pixel(1.2, 6000.0, 6000.0)
    hot = disk_pixel(1.5, 6000.0, 6000.0)
    for px in (dim, unit, bright, hot):
        assert isinstance(px, tuple) and len(px) == 3 and all(isinstance(c, int) for c in px)

    # Hotter is bluer: the blue-to-red ratio rises with g until the colour washes out to
    # white at the brightest, most blueshifted spots.
    def blue_red(g):
        c = disk_rgb_linear(g, 6000.0, 6000.0)
        return c[2] / c[0]

    assert blue_red(0.8) < blue_red(1.0) < blue_red(1.2), f"{dim} {unit} {bright}"
    assert all(v >= bright[1] for v in hot), f"{hot} {bright}"
    # No extra g^4: the pixel depends on g and T only through gT.
    for g in (0.8, 1.2, 1.5):
        assert disk_pixel(g, 6000.0, 6000.0) == disk_pixel(1.0, g * 6000.0, 6000.0)
    # Values pinned by the Rust build.
    assert [dim, unit, bright, hot] == PINNED_PIXELS
    assert disk_pixel(1.0, 0.0, 6000.0) == (0, 0, 0)
    assert disk_pixel(math.nan, 6000.0, 6000.0) == (0, 0, 0)
    assert srgb_encode(0.0) == 0
    assert srgb_encode(1.0) == 255
    assert srgb_encode(math.nan) == 0


def test_disk_colour_matches_the_rust_build():
    """disk_rgb_linear and disk_pixel against values computed by the Rust build."""
    assert LUMINANCE_WEIGHTS == (0.2126, 0.7152, 0.0722)
    assert tonemap(1.0) == 1.0 - math.exp(-1.0), "luminance 0.63 at the peak temperature"
    for (g, t, t_peak), expect in [
            ((1.0, 6000.0, 6000.0), (0.6875792144809625, 0.6186342714112824, 0.6024097910880322)),
            ((0.9, 3000.0, 5000.0), (0.15727401272660962, 0.06563710250965386, 0.015729557956663195))]:
        got = disk_rgb_linear(g, t, t_peak)
        for k in range(3):
            assert abs(got[k] - expect[k]) < 1e-15, f"{(g, t, t_peak)} channel {k}: {got} vs {expect}"
    assert disk_pixel(0.9, 3000.0, 5000.0) == (110, 72, 34)
    assert disk_pixel(0.6, 1800.0, 6000.0) == (10, 2, 0)
    # A colour too saturated for its luminance is desaturated at constant luminance.
    c = disk_rgb_linear(1.5, 6000.0, 6000.0)
    assert max(c) <= 1.0 and abs(luminance(c) - tonemap(1.5 ** 4)) < 1e-12


def test_srgb_encoding_rounds_half_away_from_zero():
    # 0.5/255/12.92 lies in the linear segment and encodes to exactly 0.5/255*255 = 0.5:
    # f64::round gives 1 (Python's round() would give 0).
    assert srgb_encode(0.5 / 255.0 / 12.92) == 1
    assert srgb_encode(-1.0) == 0
    assert srgb_encode(2.0) == 255


def test_blackbody_chromaticity_matches_independent_values():
    """The Planckian-locus fit evaluated by hand at one temperature per branch, then CIE
    xyY (Y = 1) -> linear sRGB, normalised to the largest channel."""
    def by_hand(x, y):
        cx, cz = x / y, (1.0 - x - y) / y
        r = 3.2406 * cx - 1.5372 - 0.4986 * cz
        g = -0.9689 * cx + 1.8758 + 0.0415 * cz
        b = 0.0557 * cx - 0.2040 + 1.0570 * cz
        m = max(r, g, b)
        return [max(r / m, 0.0), max(g / m, 0.0), max(b / m, 0.0)]

    # 2000 K (both low branches), 3000 K (middle y branch), 6500 K (high branches).
    x2000 = -0.2661239e9 / 2000.0 ** 3 - 0.2343589e6 / 2000.0 ** 2 + 0.8776956e3 / 2000.0 + 0.179910
    y2000 = -1.1063814 * x2000 ** 3 - 1.34811020 * x2000 ** 2 + 2.18555832 * x2000 - 0.20219683
    x3000 = -0.2661239e9 / 3000.0 ** 3 - 0.2343589e6 / 3000.0 ** 2 + 0.8776956e3 / 3000.0 + 0.179910
    y3000 = -0.9549476 * x3000 ** 3 - 1.37418593 * x3000 ** 2 + 2.09137015 * x3000 - 0.16748867
    x6500 = -3.0258469e9 / 6500.0 ** 3 + 2.1070379e6 / 6500.0 ** 2 + 0.2226347e3 / 6500.0 + 0.240390
    y6500 = 3.0817580 * x6500 ** 3 - 5.87338670 * x6500 ** 2 + 3.75112997 * x6500 - 0.37001483
    # The fit puts 6500 K close to the D65 white point (0.3127, 0.3290).
    assert abs(x6500 - 0.3135) < 2e-3 and abs(y6500 - 0.3237) < 6e-3, f"{x6500} {y6500}"
    for t, x, y in [(2000.0, x2000, y2000), (3000.0, x3000, y3000), (6500.0, x6500, y6500)]:
        got = blackbody_chromaticity(t)
        expect = by_hand(x, y)
        for k in range(3):
            assert abs(got[k] - expect[k]) < 1e-9, f"{t} K channel {k}: {got} vs {expect}"


def test_redshift_is_infinite_rather_than_an_exception_when_the_denominator_vanishes():
    """1 - Omega b_z == 0: IEEE division (the Rust build) gives a non-finite g, which the
    renderer reports as a numerical error; Python's float division would raise."""
    r = 9.0
    p_t = 1.0
    p_phi = 1.0 / keplerian_omega(M, r)  # b_z = -p_phi n_z / p_t = 1/Omega with n_z = -1
    assert 1.0 - keplerian_omega(M, r) * (-p_phi * -1.0 / p_t) == 0.0
    g = redshift(M, r, p_t, p_phi, -1.0, R_OBS)
    assert g == math.inf
    assert not (math.isfinite(g) and g > 0.0)


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
    # Going the other way (phi decreasing, d = -1) finds the same crossings at the same
    # azimuths (k_boundary = k_prev).
    back = []
    phi = phi_prev = 4.0 * PI
    while phi > 0.0:
        phi -= 0.013
        c = crossing_check(phi_d, r_in, r_out, r, phi_prev, r, phi)
        if c is not None:
            back.append(c[1])
        phi_prev = phi
    assert len(back) == 4
    for h, b in zip(hits[::-1], back):
        assert abs(h - b) < 1e-12, f"{h} vs {b}"
    # A jump of two boundaries in one step is not a crossing.
    assert crossing_check(phi_d, r_in, r_out, r, 0.0, r, 0.7 + 2.5 * PI) is None
    assert crossing_check(phi_d, r_in, r_out, r, 1.0, r, 1.2) is None
    # Out of the band: reported, flagged.
    c = crossing_check(phi_d, r_in, r_out, 25.0, 0.6, 25.0, 0.8)
    assert c is not None and c[0] == 25.0 and not c[2]


def test_crossing_check_interpolates_in_phi_in_both_directions():
    """On the spiral r = 9 + 0.1 phi the interpolated radius must sit on the spiral at
    the crossing azimuth, phi increasing (d = +1) and decreasing (d = -1) alike."""
    phi_d = 0.7

    def spiral(phi):
        return 9.0 + 0.1 * phi

    for step in (0.013, -0.013):
        hits = []
        phi = phi_prev = 0.0 if step > 0 else 4.0 * PI
        while 0.0 <= phi <= 4.0 * PI:
            phi += step
            c = crossing_check(phi_d, 5.0, 20.0, spiral(phi_prev), phi_prev, spiral(phi), phi)
            if c is not None:
                assert abs(c[0] - spiral(c[1])) < 1e-9, f"step {step}: {c}"
                k = round((c[1] - phi_d) / PI)
                assert abs(c[1] - (phi_d + k * PI)) < 1e-12
                assert min(phi_prev, phi) <= c[1] <= max(phi_prev, phi)
                hits.append(c[1])
            phi_prev = phi
        assert len(hits) == 4, (step, hits)


def test_crossing_check_returns_none_for_non_finite_input():
    """As the kernels' isfinite guard and the Rust detector: a NaN phi_d (a ray in the
    disk plane) or a non-finite azimuth is no crossing, not an exception."""
    assert crossing_check(math.nan, 6.0, 20.0, 9.0, 0.0, 9.0, 0.1) is None
    assert crossing_check(0.7, 6.0, 20.0, 9.0, 0.0, 9.0, math.inf) is None
    assert crossing_check(0.7, 6.0, 20.0, 9.0, math.nan, 9.0, 1.0) is None


# --------------------------------------------------------------------------- kernels
def _integrator(steps=20000, delta=0.01, mass=M, r_max=31.0):
    from simulation.cuda_geodesic import CUDASchwarzschildIntegrator
    return CUDASchwarzschildIntegrator(steps=steps, delta=delta, mass=mass, omega=0.01, r_max=r_max)


def _last_row(traj):
    """Index of the last recorded (non-zero) row of one ray's trajectory."""
    return int(np.flatnonzero(traj.any(axis=1))[-1])


def test_disk_kernel_with_the_disk_off_is_bit_identical_to_the_old_kernel():
    """The old kernels are the regression reference: with every disk_phi NaN the new
    loops must reproduce them bit for bit on pipeline rays (captured and escaped alike)
    that do not diverge. Under the CUDA simulator both run the same Python source, so
    the bit-identity of the compiled kernels is only checked on a GPU."""
    from simulation.cuda_geodesic import fantasy_integrate_batch_ord2, fantasy_integrate_batch_ord2_full

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
    # captured rays stopped at r <= 1.1 rs, escaped at r >= 31
    assert np.all(out[status == 1, 1] <= 2.2) and np.all(out[status == 2, 1] >= 31.0)
    # an explicit all-NaN disk_phi with a band is the same thing
    out2, status2, *_ = integ.integrate_batch(q0s, p0s, disk_phi=np.full(n, np.nan), disk_r_in=6.0, disk_r_out=20.0)
    assert np.array_equal(out2, reference) and np.array_equal(status2, status)

    # The trajectory kernel against the old one: a captured ray, an outward ray that
    # escapes within ~100 steps and one still in the domain when the budget runs out.
    # The new buffer has one row more (for a disk hit on the last step); compare the
    # first `steps` rows and check the extra one is untouched.
    steps = 3000
    sel = [init([24.0, 0.0, 0.0]), init([36.0, 1.0, 0.0]), init([24.0, 0.3, -6.0])]
    tq = np.array([r[0] for r in sel])
    tp = np.array([r[1] for r in sel])
    ref_traj = np.zeros((3, steps, 4))
    fantasy_integrate_batch_ord2_full[1, 32](tq.copy(), tp.copy(), steps, 0.01, 2.0, 31.0, 0.01, ref_traj)
    traj = _integrator(steps).integrate_batch_full(tq, tp)
    assert traj.shape == (3, steps + 1, 4)
    assert np.array_equal(traj[:, :steps], ref_traj)
    assert not traj[:, steps].any()
    assert ref_traj[0, _last_row(ref_traj[0]), 1] <= 2.2, "captured"
    assert ref_traj[1, _last_row(ref_traj[1]), 1] >= 31.0, "escaped"
    assert _last_row(ref_traj[2]) == steps - 1, "still in the domain"


def test_a_boundary_crossed_on_the_last_permitted_step_still_counts():
    """The post-loop exit test: a ray that reaches r_max on its last permitted step is
    escaped, one step fewer leaves it in the domain (status 0); a disk crossing on the
    last permitted step is a disk exit with the identical hit (the Rust
    a_crossing_on_the_last_permitted_step_is_a_disk_exit)."""
    out_ray = init([36.0, 1.0, 0.0])  # looks away from the hole: escapes in ~100 steps
    q0, p0 = out_ray[0][None], out_ray[1][None]
    traj = _integrator(1000).integrate_batch_full(q0, p0)
    n_exit = _last_row(traj[0])
    assert traj[0, n_exit, 1] >= 31.0 and traj[0, n_exit - 1, 1] < 31.0
    out, st, _, _ = _integrator(n_exit).integrate_batch(q0, p0)
    assert st[0] == 2 and np.array_equal(out[0], traj[0, n_exit])
    out, st, _, _ = _integrator(n_exit - 1).integrate_batch(q0, p0)
    assert st[0] == 0 and out[0, 1] < 31.0


def test_a_ray_aimed_through_the_disk_stops_there():
    """Face-on disk (plane x = 0, crossing line at phi = -pi/2 + k pi in every ray's
    frame): a ray with impact parameter ~10 crosses it inside the band; a radial ray
    never crosses the plane through the hole; a crossing outside the band is counted
    but does not stop the ray; an edge-on disk registers no crossing at the observer.
    The kernel's hit is the host crossing_check of its last two states, bit for bit."""
    face_on = Disk([1.0, 0.0, 0.0], 6.0, 20.0, 6000.0, M, R_OBS)
    aimed = init([24.0, 2.0, 0.0])
    radial = init([24.0, 0.0, 0.0])
    wide = init([24.0, 6.0, 0.0])  # impact parameter ~24 > r_out
    phis = [face_on.ray_frame(r[3])[0] for r in (aimed, radial, wide)]
    assert all(phi == -PI / 2.0 for phi in phis), phis
    q0s = np.array([aimed[0], radial[0], wide[0]])
    p0s = np.array([aimed[1], radial[1], wide[1]])
    integ = _integrator()
    out, status, hits, crossings = integ.integrate_batch(q0s, p0s, disk_phi=phis, disk_r_in=6.0, disk_r_out=20.0)
    assert status[0] == 4
    r_hit, phi_hit = hits[0]
    assert 6.0 < r_hit < 20.0, hits[0]
    assert phi_hit == PI / 2.0
    assert crossings[0] == 1
    assert out[0, 3] >= phi_hit and out[0, 3] - phi_hit < 0.01, "stopped just past the crossing"
    assert status[1] == 1 and crossings[1] == 0 and np.isnan(hits[1]).all()
    assert status[2] == 2 and crossings[2] == 1 and np.isnan(hits[2]).all()
    # The trajectory recorder stops at the hit and ends with the post-crossing state.
    traj = integ.integrate_batch_full(q0s[:1], p0s[:1], disk_phi=phis[:1], disk_r_in=6.0, disk_r_out=20.0)
    rows = traj[0]
    last = _last_row(rows)
    assert np.array_equal(rows[last], out[0])
    assert rows[last - 1, 3] < phi_hit <= rows[last, 3]
    # The kernel's crossing arithmetic is crossing_check's, exactly (M = 1: no rescaling).
    host = crossing_check(phis[0], 6.0, 20.0, rows[last - 1, 1], rows[last - 1, 3], rows[last, 1], rows[last, 3])
    assert host is not None and host[2]
    assert host[0] == r_hit and host[1] == phi_hit
    # A hit on the last permitted step (step index last - 1) is still a disk exit, and
    # the trajectory buffer (steps + 1 rows) keeps the post-crossing row; one step
    # fewer and the ray has not reached the disk.
    budget = _integrator(last)
    out_b, st_b, hits_b, cr_b = budget.integrate_batch(q0s[:1], p0s[:1], disk_phi=phis[:1], disk_r_in=6.0, disk_r_out=20.0)
    assert st_b[0] == 4 and cr_b[0] == 1
    assert np.array_equal(hits_b[0], hits[0]) and np.array_equal(out_b[0], out[0])
    traj_b = budget.integrate_batch_full(q0s[:1], p0s[:1], disk_phi=phis[:1], disk_r_in=6.0, disk_r_out=20.0)
    assert traj_b.shape == (1, last + 1, 4) and np.array_equal(traj_b[0], rows[:last + 1])
    _, st_s, _, cr_s = _integrator(last - 1).integrate_batch(q0s[:1], p0s[:1], disk_phi=phis[:1], disk_r_in=6.0, disk_r_out=20.0)
    assert st_s[0] == 0 and cr_s[0] == 0
    # The mirrored ray (p_phi negated) runs with phi decreasing and crosses at -pi/2:
    # the d = -1 branch (k_boundary = k_prev) gives the same radius.
    p_mirror = aimed[1].copy()
    p_mirror[3] = -p_mirror[3]
    _, st_m, hits_m, cr_m = integ.integrate_batch(aimed[0][None], p_mirror[None], disk_phi=[-PI / 2.0],
                                                  disk_r_in=6.0, disk_r_out=20.0)
    assert st_m[0] == 4 and cr_m[0] == 1
    assert hits_m[0, 0] == r_hit and hits_m[0, 1] == -PI / 2.0
    # No crossing is looked for during the first step: the observer sits on a crossing
    # line at phi_d = 1e-9, which the first step passes at r ~ 30 (outside the band).
    _, st_w, _, cr_w = integ.integrate_batch(wide[0][None], wide[1][None], disk_phi=[1e-9],
                                             disk_r_in=6.0, disk_r_out=20.0)
    assert st_w[0] == 2 and cr_w[0] == 0, "no crossing is registered on the first step"
    # Edge-on disk n = z: the crossing line is the x axis and every ray starts on it.
    edge = init([24.0, 6.0, 6.0])
    phi_d = Disk([0.0, 0.0, 1.0], 6.0, 29.5, 6000.0, M, R_OBS).ray_frame(edge[3])[0]
    assert phi_d == 0.0
    _, st, _, cr = integ.integrate_batch(edge[0][None], edge[1][None], disk_phi=[phi_d], disk_r_in=6.0, disk_r_out=29.5)
    assert st[0] == 2 and cr[0] == 0


def test_diverged_rays_stop_with_status_3_as_in_the_rust_build():
    """The disk kernels apply the Rust exit test before capture and escape: a
    non-finite or non-positive r of the first copy, or the second copy inside the
    horizon, stops the ray at once as diverged (3), with no further crossings."""
    from simulation.cuda_geodesic import STATUS_DIVERGED

    assert STATUS_DIVERGED == 3
    q0, p0, *_ = init([24.0, 2.0, 0.0])
    integ = _integrator(50)
    for bad in (math.nan, -5.0):
        q = q0.copy()
        q[1] = bad
        out, st, hits, cr = integ.integrate_batch(q[None], p0[None], disk_phi=[PI / 2.0], disk_r_in=6.0, disk_r_out=20.0)
        assert st[0] == 3 and cr[0] == 0 and np.isnan(hits).all(), (bad, st, cr)
        assert np.array_equal(out[0], q, equal_nan=True), "stopped before the first step"
        traj = integ.integrate_batch_full(q[None], p0[None], disk_phi=[PI / 2.0], disk_r_in=6.0, disk_r_out=20.0)
        assert np.array_equal(traj[0, 0], q, equal_nan=True) and not traj[0, 1:].any()
    # A near-critical pipeline ray at --delta 0.1 (default disk, inclination 75): its
    # second FANTASY copy falls inside the horizon after 401 steps. The Rust build
    # reports it diverged after 2 crossings at r = 2.246; before the exit test was
    # ported it carried on to a spurious disk hit at r = 14.55 after 3 crossings.
    default_disk = Disk(Disk.normal_from_angles(math.radians(75.0), 0.0), 6.0, 20.0, 6000.0, M, R_OBS)
    q0, p0, _, beta, _ = init([24.0, 0.9702178146900375, 0.3001235401006606])
    phi_d = default_disk.ray_frame(beta)[0]
    coarse = _integrator(4000, delta=0.1)
    out, st, hits, cr = coarse.integrate_batch(q0[None], p0[None], disk_phi=[phi_d], disk_r_in=6.0, disk_r_out=20.0)
    assert st[0] == 3 and cr[0] == 2 and np.isnan(hits).all(), (st, cr, hits)
    assert abs(out[0, 1] - 2.246) < 1e-3, out
    traj = coarse.integrate_batch_full(q0[None], p0[None], disk_phi=[phi_d], disk_r_in=6.0, disk_r_out=20.0)
    assert _last_row(traj[0]) == 401 and np.array_equal(traj[0, 401], out[0])


def test_disk_kernel_is_scale_free_in_the_mass():
    """M = 2 (observer 60, band 12-40, delta 0.02, r_max 62) is the M = 1 run with every
    length doubled: identical status and crossings, hit radius exactly doubled, the
    same hit azimuth. Pins the rescaling of the disk band and of the hits."""
    results = []
    for mass in (1.0, 2.0):
        obs = np.array([30.0 * mass, 0.0, 0.0])
        q0s, p0s, phis = [], [], []
        for pix in ([24.0, 2.0, 0.0], [24.0, 6.0, 0.0]):
            q0, p0, _, _, _, _, beta = get_initial_conditions(obs, np.array(pix) * mass, mass_bh=mass)
            q0s.append(q0)
            p0s.append(p0)
            phis.append(Disk([1.0, 0.0, 0.0], 6.0 * mass, 20.0 * mass, 6000.0, mass, 30.0 * mass).ray_frame(beta)[0])
        integ = _integrator(5000, delta=0.01 * mass, mass=mass, r_max=31.0 * mass)
        results.append(integ.integrate_batch(np.array(q0s), np.array(p0s), disk_phi=phis,
                                             disk_r_in=6.0 * mass, disk_r_out=20.0 * mass))
    (out1, st1, hits1, cr1), (out2, st2, hits2, cr2) = results
    assert st1.tolist() == [4, 2] and np.array_equal(st1, st2)
    assert np.array_equal(cr1, cr2) and cr1.tolist() == [1, 1]
    assert hits2[0, 0] == 2.0 * hits1[0, 0] and hits2[0, 1] == hits1[0, 1]
    assert np.isnan(hits2[1]).all()
    assert np.array_equal(out2[:, 1], 2.0 * out1[:, 1]) and np.array_equal(out2[:, 3], out1[:, 3])


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


def test_camera_angles_compose_intrinsically_yaw_then_pitch_then_roll():
    """Hand values, not camera_rotation: pitch turns about the yawed right axis and
    roll about the yawed-and-pitched optical axis. (Under the reversed order a 90-degree
    yaw would annihilate the pitch: z = 0.)"""
    axis, _, _ = camera_basis(PI / 2.0, 0.5, 0.0)
    # yaw 90 about +z: axis -x -> -y, right +y -> -x; pitch 0.5 about -x lifts the axis
    # toward +z
    assert abs(axis[2] - math.sin(0.5)) < 1e-15, axis
    assert abs(axis[1] + math.cos(0.5)) < 1e-15, axis
    assert abs(axis[0]) < 1e-15, axis
    axis, right, up = camera_basis(PI / 2.0, 0.0, PI / 6.0)
    # yaw 90: axis -y, right -x, up +z; roll pi/6 about -y turns up toward -x
    for got, expect in ((axis, (0.0, -1.0, 0.0)), (up, (-math.sin(PI / 6.0), 0.0, math.cos(PI / 6.0)))):
        for k in range(3):
            assert abs(got[k] - expect[k]) < 1e-15, (got, expect)
    assert abs(right[0] + math.cos(PI / 6.0)) < 1e-15 and abs(right[2] + math.sin(PI / 6.0)) < 1e-15, right


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
def test_disk_scene_fixture_is_self_consistent():
    """The cross-build fixture (tests/gen_disk_fixture.py), checked without integrating:
    its initial conditions (p0, beta, alpha0) are what the camera and
    get_initial_conditions give for its scene, its (phi_d, n_z) are the disk in each
    ray's frame, and every hit sits on the crossing line inside the band, in the disk
    plane at |x| = r_hit, with the stored g, temperature and lab position. Pins the
    format the Rust build's test reads.

    The comparisons are exact: the fixture is regenerated and checked on the same
    platform, and every field is a deterministic function of the scene (a different
    libm could move the trig-derived fields by an ulp)."""
    with open(FIXTURE) as f:
        doc = json.load(f)
    s = doc["scene"]
    assert s["size"] == 16 and s["mass"] == 1.0 and s["steps"] == 20000
    assert doc["nan_encoding"] == "null"
    mass, r_obs = s["mass"], s["r_obs"]
    normal = Disk.normal_from_angles(math.radians(s["disk_inclination_deg"]), math.radians(s["disk_azimuth_deg"]))
    d = Disk(normal, s["disk_inner"], s["disk_outer"], s["disk_temperature"], mass, r_obs)
    assert list(d.normal) == s["disk_normal"]
    obs = np.array(s["observer"], dtype=np.float64)
    assert float(np.linalg.norm(obs)) == r_obs
    axis, right, up = camera_basis(math.radians(s["yaw_deg"]), math.radians(s["pitch_deg"]),
                                   math.radians(s["roll_deg"]))
    positions = pixel_positions(obs, math.radians(s["fov_deg"]), s["size"], s["size"], axis, right, up)
    pixels = doc["pixels"]
    assert len(pixels) == 256
    n_hits = 0
    for idx, px in enumerate(pixels):
        i, j = divmod(idx, 16)
        assert (px["i"], px["j"]) == (i, j)
        assert px["status"] in (0, 1, 2, 4)
        _, p0, alpha0, _, _, _, beta = get_initial_conditions(obs, positions[i, j].copy(), mass_bh=mass)
        assert [float(v) for v in p0] == px["p0"], (i, j)
        assert float(beta) == px["beta"] and float(alpha0) == px["alpha0"], (i, j)
        phi_d, n_z = d.ray_frame(px["beta"])
        if px["phi_d"] is None:
            assert math.isnan(phi_d)
        else:
            assert phi_d == px["phi_d"]
        assert n_z == px["n_z"]
        if px["status"] == 4:
            n_hits += 1
            r_hit, phi_hit = px["r_hit"], px["phi_hit"]
            assert d.r_in <= r_hit <= d.r_out
            k = round((phi_hit - phi_d) / PI)
            assert abs(phi_hit - (phi_d + k * PI)) < 1e-12
            assert px["plane_crossings"] >= 1
            g = redshift(mass, r_hit, px["p0"][0], px["p0"][3], n_z, r_obs)
            assert g == px["g"]
            assert math.isfinite(g) and g > 0.0
            assert px["q"][3] >= phi_hit and px["q"][3] - phi_hit < 0.02
            xyz = px["hit_xyz"]
            assert hit_position_lab(r_hit, phi_hit, px["beta"]) == xyz
            assert d.temperature(r_hit) == px["disk_temp"]
            assert abs(vec_dot(d.normal, xyz)) < 1e-12, "the hit lies in the disk plane"
            assert abs(vec_norm(xyz) - r_hit) < 1e-12
        else:
            assert px["r_hit"] is None and px["phi_hit"] is None and px["g"] is None
            assert px["hit_xyz"] == [None, None, None] and px["disk_temp"] is None
    assert n_hits > 30, n_hits
    gs = [px["g"] for px in pixels if px["status"] == 4]
    assert max(gs) > 1.0 and min(gs) < 0.9, "Doppler both ways"


# --------------------------------------------------------------------------- renderer
def _scene_disk():
    """The fixture scene's disk (inclination 55, azimuth 30, band [6, 20])."""
    return Disk(Disk.normal_from_angles(math.radians(55.0), math.radians(30.0)), 6.0, 20.0, 6000.0, M, R_OBS)


DISK_COLUMNS = ["disk_r", "disk_phi", "disk_x", "disk_y", "disk_z", "disk_g", "disk_temp", "plane_crossings"]


def test_renderer_writes_the_disk_columns_and_shades_hits_from_the_closed_form(tmp_path, monkeypatch):
    """run_manual_simulation end to end on a 4x4 image of the fixture scene's disk and
    camera (yaw 6, pitch -4): the photon table ends with the eight disk columns, every
    'disk' row carries the closed-form g of its initial momenta and the temperature at
    its radius, other rows have NaN disk columns (written 'nan'), the PNG is saved
    with up at the top and the sampled trajectories end on their exit state. Without
    the disk the disk columns are NaN and no crossing is counted."""
    import random

    import pandas as pd
    from PIL import Image
    from simulation.blackhole import BlackHole, Observer
    from simulation.raytracing import run_manual_simulation

    monkeypatch.chdir(tmp_path)  # the renderer writes images/, photon_data.csv, sampled_rays.csv
    (tmp_path / "images").mkdir()
    size, fov = 4, math.radians(80.0)
    yaw, pitch, roll = math.radians(6.0), math.radians(-4.0), 0.0
    d = _scene_disk()
    common = dict(steps=10000, delta=0.02, omega=0.01, use_cuda=True, boundary_radius=31.0,
                  yaw=yaw, pitch=pitch, roll=roll)
    random.seed(12345)
    img, trajs = run_manual_simulation(BlackHole(M), Observer(OBS, fov, (size, size)), n_samples=2, disk=d, **common)
    df = pd.read_csv("photon_data.csv", float_precision="round_trip")
    assert list(df.columns[-8:]) == DISK_COLUMNS
    assert len(df) == size * size
    axis, right, up = camera_basis(yaw, pitch, roll)
    positions = pixel_positions(OBS, fov, size, size, axis, right, up)
    n_disk = 0
    for row in df.itertuples(index=False):
        _, p0, _, _, _, _, beta = get_initial_conditions(OBS, positions[row.i, row.j].copy(), mass_bh=M)
        assert [row.p0_t, row.p0_r, row.p0_th, row.p0_ph] == [float(v) for v in p0]
        if row.collision == "disk":
            n_disk += 1
            g = redshift(M, row.disk_r, row.p0_t, row.p0_ph, d.ray_frame(beta)[1], R_OBS)
            assert row.disk_g == g
            assert row.disk_temp == d.temperature(row.disk_r)
            assert [row.disk_x, row.disk_y, row.disk_z] == hit_position_lab(row.disk_r, row.disk_phi, float(beta))
            assert d.r_in <= row.disk_r <= d.r_out and row.plane_crossings >= 1
            assert tuple(int(c) for c in img[row.i, row.j]) == disk_pixel(g, row.disk_temp, d.peak_temperature)
        else:
            assert all(math.isnan(getattr(row, c)) for c in DISK_COLUMNS[:-1])
    assert n_disk >= 1, df["collision"].tolist()
    with open("photon_data.csv") as f:
        text = f.read()
    assert ",nan,nan,nan,nan,nan,nan,nan," in text, "NaN written as 'nan', as the Rust build does"
    # Saved with row i = 0 at the bottom (up at the top).
    saved = np.array(Image.open("images/manual_output.png"))[..., :3]
    assert np.array_equal(saved[::-1], img)
    # The sampled trajectories (same seed, same draw as the renderer) end on the ray's
    # exit state: the disk hit's post-crossing state, or the capture/escape state.
    random.seed(12345)
    sampled = set()
    while len(sampled) < 2:
        sampled.add((random.randint(0, size - 1), random.randint(0, size - 1)))
    assert len(trajs) == 2
    for (i, j), traj in zip(sampled, trajs):
        row = df[(df.i == i) & (df.j == j)].iloc[0]
        radii = np.linalg.norm(traj, axis=1)
        assert np.all(radii > 0.0), "no zero rows from after the exit"
        if row.collision != "in_domain":
            assert abs(radii[-1] - row.final_r) < 1e-9, (i, j, radii[-1], row.final_r)

    # The same scene without the disk.
    run_manual_simulation(BlackHole(M), Observer(OBS, fov, (size, size)), n_samples=0, disk=None, **common)
    off = pd.read_csv("photon_data.csv", float_precision="round_trip")
    assert list(off.columns[-8:]) == DISK_COLUMNS
    assert off[DISK_COLUMNS[:-1]].isna().all().all()
    assert (off["plane_crossings"] == 0).all()
    assert "disk" not in set(off["collision"])


def test_renderer_rejects_a_disk_within_one_step_of_the_observer():
    """Crossings during the first step are not checked, so the disk must end more than
    one step (delta) inside the observer; checked before any integration."""
    from simulation.blackhole import BlackHole, Observer
    from simulation.raytracing import run_manual_simulation

    d = Disk([0.0, 0.0, 1.0], 6.0, 29.5, 6000.0, M, R_OBS)  # valid: r_out < r_obs
    with pytest.raises(ValueError, match="more than one step"):
        run_manual_simulation(BlackHole(M), Observer(OBS, math.radians(80.0), (2, 2)), steps=10, delta=0.6,
                              use_cuda=True, boundary_radius=31.0, disk=d)


# --------------------------------------------------------------------------- CLI
def _validate(*argv):
    """main.validate_args on the parsed flags: ((yaw, pitch, roll) in radians, disk)."""
    from config import parse_args
    import main as cli

    return cli.validate_args(parse_args(list(argv)))


def test_cli_builds_the_disk_from_its_flags():
    (yaw, pitch, roll), d = _validate("--disk", "--yaw", "6", "--pitch", "-4")
    assert (yaw, pitch, roll) == (math.radians(6.0), math.radians(-4.0), 0.0)
    assert (d.r_in, d.r_out, d.peak_temperature) == (6.0, 20.0, 6000.0)
    assert list(d.normal) == Disk.normal_from_angles(math.radians(75.0), 0.0)
    _, d = _validate("--disk", "--bh-mass", "2", "--observer-distance", "60", "--boundary-radius", "62")
    assert (d.r_in, d.r_out) == (12.0, 40.0)
    # The disk flags only take effect with --disk.
    assert _validate("--disk-inner", "5", "--disk-outer", "30")[1] is None


@pytest.mark.parametrize("argv, fragments", [
    (["--disk", "--disk-inner", "5"],
     ["--disk: inner radius 5.0, outer radius 20.0 (default 20M): ", "inner radius (5.0) must be at least the ISCO"]),
    (["--disk", "--disk-outer", "30"],
     ["--disk: inner radius 6.0 (default 6M), outer radius 30.0: ", "outer radius (30.0) must be smaller",
      "pass a smaller --disk-outer or a larger --observer-distance"]),
    (["--disk", "--disk-inclination", "200"], ["--disk-inclination must be in [0, 180] degrees, got 200.0"]),
    (["--yaw", "nan"], ["--yaw must be a finite angle in degrees, got nan"]),
    # r_out < r_obs, but within one step of the observer: the first-step rule.
    (["--disk", "--disk-outer", "29.995"],
     ["--disk-outer must lie more than one step (--delta 0.01) inside --observer-distance (30"]),
    (["--disk", "--disk-outer", "29.5", "--delta", "0.6"], ["--disk-outer", "--delta 0.6"]),
    # The 20M default exceeds the default observer distance from M = 1.5 on.
    (["--disk", "--bh-mass", "1.5"],
     ["--disk: inner radius 9.0 (default 6M), outer radius 30.0 (default 20M): ",
      "pass a smaller --disk-outer or a larger --observer-distance"]),
    (["--disk", "--bh-mass", "2"],
     ["--disk: inner radius 12.0 (default 6M), outer radius 40.0 (default 20M): ", "--observer-distance"]),
])
def test_cli_rejects_invalid_flags_with_a_message_naming_them(argv, fragments):
    with pytest.raises(SystemExit) as exc:
        _validate(*argv)
    message = str(exc.value)
    for fragment in fragments:
        assert fragment in message, f"{fragment!r} not in {message!r}"
