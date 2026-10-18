"""The accretion disk: a geometrically thin, optically thick sheet of material on
circular Keplerian orbits in a plane through the black hole, and the closed-form
physics that turns a ray's hit on it into a pixel.

This is a port of the Rust build's ``disk.rs``; every function keeps its name and
its arithmetic order so the two builds agree to rounding.

Geometry
--------
The disk plane is ``n . x = 0`` for a unit normal ``n``. The CLI gives ``n`` by two
angles. The *inclination* ``i`` in [0, 180] degrees is the angle between ``n`` and
the observer's position vector ``+x``: 0 is face-on, 90 edge-on, 180 face-on from
the other side, which reverses the sense of rotation on the sky. The *azimuth*
``a`` turns ``n`` right-handedly about ``+x``: ``n = R_x(a) . (cos i, 0, sin i)``.
At ``a = 0`` the normal lies in the x-z plane, so a disk inclined by less than 90
degrees has its near side below the black hole in the image and its far side above;
between 90 and 180 degrees (the same plane as ``(180 - i, a + 180)``, turning the
other way) the near side is above. Both angles are measured against the observer's
position, not the optical axis, so turning the camera does not move the disk.

The material orbits right-handedly about ``n``: at ``x`` it moves along ``n x x_hat``.
With ``i = 90, a = 0`` (``n = +z``) the material at ``(0, -r, 0)``, on the observer's
left, moves along ``+x``, towards the observer: the left side of an edge-on disk
approaches and is blueshifted.

Redshift
--------
Each ray is integrated in a frame rotated about ``+x`` by ``-beta``
(``get_initial_conditions``). In that frame its orbit is the x-y plane, its
angular-momentum vector lies along ``+z'`` with magnitude ``p_phi``, and the disk
normal is ``n' = R_x(-beta) n``. The physical photon is the time reverse of the
traced ray, so its energy at infinity is ``E = p_t`` and its angular momentum about
``n`` is ``L_n = -p_phi n'_z``. With ``b_z = L_n / E`` and the Keplerian angular
velocity ``Omega = sqrt(M/r^3)``, the ratio of observed to emitted frequency is

    g = sqrt(1 - 3M/r) / ((1 - Omega b_z) sqrt(1 - 2M/r_obs))

for the static observer and the circular orbit ``u_emit = (1, 0, 0, Omega) /
sqrt(1 - 3M/r)``. ``p_t`` and ``p_phi`` are conserved, so ``g`` depends only on
where the ray hits and on the ray's *initial* momenta. The formula needs
``r >= 3M``, and circular orbits are stable only from the ISCO at ``6M``, so the
inner radius must be at least ``6M``.

Emission
--------
The effective temperature follows the Newtonian Shakura-Sunyaev profile with zero
torque at the inner edge, ``T^4 ~ (1 - sqrt(r_in/r)) / r^3``, which peaks at
``r = 49/36 r_in``; the peak value is the user's ``--disk-temperature``. Liouville's
theorem (``I_nu/nu^3`` is invariant) makes the observed spectrum a blackbody at
``g T``: its bolometric intensity is ``(gT)^4`` and its colour the chromaticity of a
blackbody at ``gT``; nothing is multiplied by ``g^4`` on top of that. The pixel's
linear luminance is the tone-mapped intensity ``1 - exp(-(gT/T_peak)^4)`` (0.63 at
the peak temperature without Doppler shift, leaving headroom for the approaching
side), so brightness rises monotonically with ``g``, and its hue is the blackbody
chromaticity at ``gT``; a colour too saturated to show at that luminance is
desaturated towards white at constant luminance (``disk_rgb_linear``). The
temperature is a false-colour parameter: at realistic disk temperatures (1e5-1e7 K)
every pixel would be white.
"""
import math

from simulation.utils import (mat_vec, rot_x, vec_add, vec_dot, vec_norm, vec_normalize,
                              vec_scale)

# Innermost stable circular orbit, in units of M.
ISCO_PER_M = 6.0
# The temperature profile peaks at this multiple of the inner radius.
PEAK_RADIUS_PER_R_IN = 49.0 / 36.0
# A ray whose orbital plane is within this of the disk plane (|n'_perp|, the part of
# the disk normal perpendicular to the ray's angular momentum) lies in the disk and
# has no crossing to detect.
PARALLEL_TOLERANCE = 1e-9
# Below this effective temperature the disk is not drawn (the colour fit is only
# valid from 1667 K; the profile reaches zero at the inner edge anyway).
MIN_DRAWN_TEMPERATURE = 1.0

# The same double as math.pi, spelled out as the kernels spell it.
PI = 3.141592653589793


class Disk:
    """A thin Keplerian disk with unit `normal`, radii `r_in <= r <= r_out` and peak
    effective temperature `peak_temperature` (kelvin, false colour)."""

    def __init__(self, normal, r_in, r_out, peak_temperature, mass, r_obs):
        normal = [float(normal[0]), float(normal[1]), float(normal[2])]
        r_in = float(r_in)
        r_out = float(r_out)
        peak_temperature = float(peak_temperature)
        length = vec_norm(normal)
        if not (math.isfinite(length) and length > 0.0):
            raise ValueError(f"disk normal must be a non-zero vector, got {normal}")
        isco = ISCO_PER_M * mass
        if not (math.isfinite(r_in) and r_in >= isco):
            raise ValueError(
                f"the disk's inner radius ({r_in}) must be at least the ISCO, 6M = {isco}: "
                "there are no stable circular orbits inside it, and below 3M the Keplerian "
                "redshift is undefined")
        if not (math.isfinite(r_out) and r_out > r_in):
            raise ValueError(
                f"the disk's outer radius ({r_out}) must exceed its inner radius ({r_in})")
        if not r_out < r_obs:
            raise ValueError(
                f"the disk's outer radius ({r_out}) must be smaller than the observer "
                f"distance ({r_obs}): every ray starts on the crossing line of an edge-on disk")
        if not (math.isfinite(peak_temperature) and peak_temperature > 0.0):
            raise ValueError(f"the disk temperature must be positive, got {peak_temperature}")
        self.normal = tuple(vec_normalize(normal))
        self.r_in = r_in
        self.r_out = r_out
        self.peak_temperature = peak_temperature

    def __repr__(self):
        return (f"Disk(normal={list(self.normal)}, r_in={self.r_in}, r_out={self.r_out}, "
                f"peak_temperature={self.peak_temperature})")

    @staticmethod
    def normal_from_angles(inclination, azimuth):
        """n = R_x(azimuth) . (cos i, 0, sin i); angles in radians (see the module doc)."""
        si = math.sin(inclination)
        ci = math.cos(inclination)
        return mat_vec(rot_x(azimuth), [ci, 0.0, si])

    def ray_frame(self, beta):
        """The disk as the frame of a ray rotated by `beta` sees it: (phi_d, n_z).

        phi_d is the azimuth, in the ray's frame, of the line where the disk plane
        meets the orbital plane (the plane is crossed at phi_d + k*pi); NaN when the
        orbit lies in the disk plane, in which case no crossing is detected. n_z is
        the disk normal's component along the ray's angular-momentum axis.
        """
        n = mat_vec(rot_x(-float(beta)), self.normal)
        perp = math.hypot(n[0], n[1])
        if perp < PARALLEL_TOLERANCE:
            phi_d = math.nan
        else:
            phi_d = math.atan2(-n[0], n[1])
        return phi_d, n[2]

    def contains_radius(self, r):
        return r >= self.r_in and r <= self.r_out

    def temperature(self, r):
        """Effective temperature at radius r: zero at and inside r_in, the peak value
        at 49/36 r_in."""
        peak = temperature_shape(self.r_in, self.r_in * PEAK_RADIUS_PER_R_IN)
        return self.peak_temperature * (temperature_shape(self.r_in, r) / peak) ** 0.25

    def straight_ray_hit(self, origin, direction):
        """Where the straight ray origin + t*direction (t > 0) meets the disk, as a
        list [x, y, z], or None."""
        origin = [float(origin[0]), float(origin[1]), float(origin[2])]
        direction = [float(direction[0]), float(direction[1]), float(direction[2])]
        denom = vec_dot(self.normal, direction)
        if denom == 0.0:
            return None
        t = -vec_dot(self.normal, origin) / denom
        if t <= 0.0:
            return None
        hit = vec_add(origin, vec_scale(direction, t))
        if self.contains_radius(vec_norm(hit)):
            return hit
        return None


def temperature_shape(r_in, r):
    """(1 - sqrt(r_in/r)) / r^3, the shape of T^4; zero at and inside r_in."""
    if r <= r_in:
        return 0.0
    return (1.0 - math.sqrt(r_in / r)) / (r * r * r)


def keplerian_omega(mass, r):
    """Keplerian coordinate angular velocity Omega = sqrt(M/r^3)."""
    return math.sqrt(mass / (r * r * r))


def redshift(mass, r, p_t, p_phi, n_z, r_obs):
    """g = sqrt(1 - 3M/r) / ((1 - Omega b_z) sqrt(1 - 2M/r_obs)) with
    b_z = -p_phi n'_z / p_t (module doc). p_t and p_phi are the traced ray's conserved
    initial momenta in its rotated frame. A result that is not finite and positive
    means 1 - Omega b_z <= 0, which no photon reaching the observer can produce:
    treat it as an error. 1 - Omega b_z == 0 gives an infinite g (sign of the
    numerator), as IEEE division does, rather than raising ZeroDivisionError."""
    omega = keplerian_omega(mass, r)
    b_z = -p_phi * n_z / p_t
    denom = 1.0 - omega * b_z
    num = math.sqrt(1.0 - 3.0 * mass / r)
    den = denom * math.sqrt(1.0 - 2.0 * mass / r_obs)
    if den == 0.0:
        # Python raises on a division by zero; return what IEEE (and the Rust build)
        # gives instead, a non-finite g that the caller reports as a numerical error.
        if num == 0.0 or math.isnan(num):
            return math.nan
        return math.copysign(math.inf, num) * math.copysign(1.0, den)
    return num / den


def hit_position_lab(r, phi, beta):
    """Lab-frame position [x, y, z] of a hit at (r, phi) in the frame of a ray rotated
    by beta."""
    s = math.sin(phi)
    c = math.cos(phi)
    return mat_vec(rot_x(beta), [r * c, r * s, 0.0])


def blackbody_chromaticity(t):
    """Chromaticity (linear sRGB, largest channel 1) of a blackbody at t kelvin, from
    Kang et al. (2002)'s fit of the Planckian locus; t is clamped to the fit's
    1667-25000 K range."""
    t = min(max(t, 1667.0), 25000.0)
    inv = 1.0 / t
    inv2 = inv * inv
    inv3 = inv2 * inv
    if t <= 4000.0:
        x = -0.2661239e9 * inv3 - 0.2343589e6 * inv2 + 0.8776956e3 * inv + 0.179910
    else:
        x = -3.0258469e9 * inv3 + 2.1070379e6 * inv2 + 0.2226347e3 * inv + 0.240390
    x2 = x * x
    x3 = x2 * x
    if t <= 2222.0:
        y = -1.1063814 * x3 - 1.34811020 * x2 + 2.18555832 * x - 0.20219683
    elif t <= 4000.0:
        y = -0.9549476 * x3 - 1.37418593 * x2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x3 - 5.87338670 * x2 + 3.75112997 * x - 0.37001483
    # CIE xyY with Y = 1 -> XYZ -> linear sRGB (D65).
    cx = x / y
    cz = (1.0 - x - y) / y
    r = 3.2406 * cx - 1.5372 - 0.4986 * cz
    g = -0.9689 * cx + 1.8758 + 0.0415 * cz
    b = 0.0557 * cx - 0.2040 + 1.0570 * cz
    m = max(max(r, g), b)
    return [max(r / m, 0.0), max(g / m, 0.0), max(b / m, 0.0)]


def tonemap(relative_intensity):
    """Maps a bolometric intensity relative to the disk's peak ((gT/T_peak)^4, which
    exceeds 1 on the approaching side) to a display luminance in [0, 1] (1 only once
    the exponential underflows)."""
    return 1.0 - math.exp(-relative_intensity)


def _round_half_away_from_zero(x):
    """f64::round for non-negative x (Python's round() rounds half to even)."""
    f = math.floor(x)
    if x - f >= 0.5:
        f += 1
    return int(f)


def srgb_encode(v):
    """Linear light in [0, 1] to an 8-bit sRGB value."""
    if math.isfinite(v):
        v = min(max(v, 0.0), 1.0)
    else:
        v = 0.0
    if v <= 0.0031308:
        s = 12.92 * v
    else:
        s = 1.055 * v ** (1.0 / 2.4) - 0.055
    return min(max(_round_half_away_from_zero(s * 255.0), 0), 255)


# Rec. 709 / sRGB luminance weights of linear R, G, B (they sum to 1).
LUMINANCE_WEIGHTS = (0.2126, 0.7152, 0.0722)


def luminance(c):
    """Linear luminance Y of a linear-sRGB colour."""
    return LUMINANCE_WEIGHTS[0] * c[0] + LUMINANCE_WEIGHTS[1] * c[1] + LUMINANCE_WEIGHTS[2] * c[2]


def disk_rgb_linear(g, t_emit, t_peak):
    """Linear-sRGB colour (r, g, b) of a hit seen with redshift factor g on material at
    effective temperature t_emit: luminance exactly tonemap((gT/T_peak)^4), hue the
    blackbody chromaticity at gT. Where that hue cannot reach the luminance without a
    channel above 1, the colour is mixed with grey of the same luminance just enough
    to fit, so the luminance is kept and the hue desaturates towards white. Black when
    gT is not a number or below MIN_DRAWN_TEMPERATURE."""
    t_obs = g * t_emit
    if math.isnan(t_obs) or t_obs < MIN_DRAWN_TEMPERATURE:
        return (0.0, 0.0, 0.0)
    x = t_obs / t_peak
    x2 = x * x
    l = tonemap(x2 * x2)
    c = blackbody_chromaticity(t_obs)
    k = l / luminance(c)
    rgb = (k * c[0], k * c[1], k * c[2])
    m = max(max(rgb[0], rgb[1]), rgb[2])
    if m <= 1.0:
        return rgb
    # m > 1 >= l, so the mixing weight lies in [0, 1).
    s = (1.0 - l) / (m - l)
    return (l + s * (rgb[0] - l), l + s * (rgb[1] - l), l + s * (rgb[2] - l))


def disk_pixel(g, t_emit, t_peak):
    """The 8-bit (R, G, B) pixel, ints in 0..255, of disk_rgb_linear."""
    c = disk_rgb_linear(g, t_emit, t_peak)
    return (srgb_encode(c[0]), srgb_encode(c[1]), srgb_encode(c[2]))


def crossing_check(phi_d, r_in, r_out, r_prev, phi_prev, r, phi):
    """Host copy of the kernel's crossing test between two consecutive states.

    The plane is crossed where floor((phi - phi_d)/pi) changes by exactly one; the
    crossing radius is interpolated linearly in phi. Returns None, or
    (r_hit, phi_c, in_band) with phi_c = phi_d + k*pi the crossing azimuth and
    in_band whether r_in <= r_hit <= r_out. phi must not be wrapped between calls.
    Non-finite inputs (a NaN phi_d marks a ray in the disk plane) give None, as in
    the kernels and the Rust build.
    """
    a_prev = (phi_prev - phi_d) / PI
    a_now = (phi - phi_d) / PI
    if not (math.isfinite(a_prev) and math.isfinite(a_now)):
        return None
    k_prev = math.floor(a_prev)
    k = math.floor(a_now)
    d = k - k_prev
    if d != 1 and d != -1:
        return None
    k_boundary = k if d > 0 else k_prev
    phi_c = phi_d + k_boundary * PI
    frac = (phi_c - phi_prev) / (phi - phi_prev)
    r_hit = r_prev + (r - r_prev) * frac
    return r_hit, phi_c, (r_hit >= r_in and r_hit <= r_out)
