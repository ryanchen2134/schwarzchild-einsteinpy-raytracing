# Schwarzschild Ray Tracer

**Simulating gravitational lensing and photon trajectories in Schwarzschild spacetime using inverse ray tracing and general relativity.**
Read the implementation details [here!](Phys_129L_Final_Project_Report-3.pdf) 
![Ray Tracing Example](images/manual_output.png)

## 📜 Overview

This project numerically integrates null geodesics to simulate how light bends around a non-rotating Schwarzschild black hole. By back-tracing rays from an observer's camera through curved spacetime, we recreate gravitational lensing, photon rings, and black hole shadows.

The final output demonstrates how black holes appear to distant observers—mimicking features captured in telescope imagery and theoretical models.

---

## Key Features

- **Geodesic Integration**: Solves the null geodesic equations with the `FANTASY` order-2 symplectic scheme, implemented as a Numba-CUDA kernel (EinsteinPy is used only by the diagnostic scripts and tests). Run the kernels on the CPU with `NUMBA_ENABLE_CUDASIM=1`.
- **GPU Acceleration**: Massive speedup through Numba-CUDA support
- **Realistic Rendering**: Projects a background panorama to simulate lensing distortions
- **Schwarzschild Geometry**: Pure Schwarzschild spacetime with no spin
- **Equirectangular Mapping**: Converts escape directions to texture coordinates on a celestial sphere

---

##  Technical Details

- **Metric**: Schwarzschild (static, spherically symmetric)
- **Observer**: Positioned far from the black hole on the +x axis
- **Resolution**: Configurable grid of rays (e.g., 400×400)
- **Integration**: Symplectic solver using affine parameter $\lambda$
- **Output**: Rendered RGB image combining captured rays and background sky


---

## Camera pointing

The observer sits on the `+x` axis and, unrotated, looks at the black hole: optical
axis `−x̂`, right `+ŷ`, up `+ẑ`. The camera can turn (pan) but not move, because the
equatorial rotation that brings every ray into the x–y plane keeps only an observer on
`+x` fixed. The viewer basis is turned by three intrinsic angles, in degrees, applied
in this order and each right-handed about the named axis of the camera as it then
stands:

| Flag | Axis | Positive value |
| --- | --- | --- |
| `--yaw` | up | looks to the observer's left, so the black hole moves right in the image |
| `--pitch` | right (after the yaw) | looks up, so the black hole moves down |
| `--roll` | optical axis (after yaw and pitch) | turns the image counter-clockwise as seen by the viewer |

Intrinsic yaw → pitch → roll equals `R_up(yaw) · R_right(pitch) · R_axis(roll)` about
the *unrotated* axes (`simulation.utils.camera_basis`). Ray directions are unit vectors
in the static observer's orthonormal frame, so the field of view is the angle the
observer measures; the rotated directions feed the same momentum construction as
before. With all angles zero the basis is exactly the old one. Both the curved render
and the no-gravity reference use the same camera (`simulation.utils.pixel_positions`);
row `i = 0` of the pixel grid is the bottom row of the image.

## Accretion disk

`--disk` adds a geometrically thin, opaque sheet of material on circular Keplerian
orbits in a plane through the black hole (`simulation/disk.py`, a port of the Rust
build's `disk.rs` with the same operation order).

**Geometry.** The disk plane is `n · x = 0`. `--disk-inclination` (default 75°) is the
angle between `n` and the observer's position vector `+x̂`: 0 face-on, 90 edge-on, 180
face-on from the other side, which reverses the sense of rotation on the sky.
`--disk-azimuth` (default 0°) turns `n` right-handedly about `+x̂`:
`n = R_x(azimuth) · (cos i, 0, sin i)`. At azimuth 0 a disk inclined by less than 90°
has its near side below the black hole and its far side above; between 90° and 180°
(the same plane as `(180° − i, azimuth + 180°)`, turning the other way) the near side
is above. Both angles are measured against the observer's position, not the optical
axis, so turning the camera does not move the disk. The material orbits right-handedly about `n` (at `x` it moves along `n × x̂`):
edge-on with `n = +ẑ`, the left side of the disk approaches the observer and is
blueshifted. `--disk-inner` defaults to the ISCO, `6M`, and may not be smaller;
`--disk-outer` (default `20M`) must lie more than one step inside the observer,
below `--observer-distance − --delta` (see Crossings), so with `--bh-mass` 1.5 or more
and the default `--observer-distance 30` pass `--disk-outer` or a larger
`--observer-distance`. An invalid disk is reported with the radii actually used and
which of them were defaults. The disk flags only take effect with `--disk`.

**Crossings.** Each ray carries the azimuth `phi_d` at which the disk plane cuts its
orbital plane (`Disk.ray_frame`). The integrator kernel checks after every step, from
the second step on, whether `floor((φ − phi_d)/π)` changed by exactly one, interpolates
the crossing radius linearly in `φ`, counts the crossing, and ends the ray (status 4,
collision `disk`) when the radius lies in `[r_in, r_out]`. `φ` is never wrapped. The
first step is skipped because every ray starts at the observer, which sits on the
crossing line of an edge-on disk, where rounding could register a crossing at `r_obs`.
A null ray moves at most `|dr/dλ| ≤ p_t < 1` per unit affine parameter, so a crossing
during the first step lies within `δ` of `r_obs`; skipping it is harmless only while
`r_out < r_obs − δ`, which `main.py` and `run_manual_simulation` enforce (`r_out < r_obs`
alone is not enough). The kernel stops a ray with the Rust build's exit test, checked
before every step: diverged (status 3: `r` of the first copy not finite or `≤ 0`, or
the second FANTASY copy not finite or inside the horizon, `r₂ ≤ r_s`), then captured
(`r ≤ 1.1 r_s`), then escaped (`r ≥ r_max`). With the disk off it reproduces the
previous kernel bit for bit on every ray that does not diverge.

**Redshift.** With `p_t`, `p_φ` the ray's conserved initial momenta in its rotated
frame, `n'_z` the component of the disk normal along the ray's angular momentum,
`b_z = −p_φ n'_z / p_t` and `Ω = √(M/r³)`, the ratio of observed to emitted frequency is

```
g = √(1 − 3M/r) / ((1 − Ω b_z) √(1 − 2M/r_obs))
```

for the static observer and the circular orbit at the hit. A hit with `g` not finite
and positive is physically impossible and is reported as a `numerical error` (red).

**Emission and colour.** The effective temperature follows the Shakura–Sunyaev profile
`T⁴ ∝ (1 − √(r_in/r)) / r³`, which peaks at `r = 49/36 r_in`; the peak value is
`--disk-temperature` (default 6000 K). The observed spectrum is a blackbody at `gT`
(nothing is multiplied by `g⁴` on top of that), so the pixel depends on `g` and `T`
only through `gT`. Its linear luminance (Rec. 709 weights) is exactly
`1 − exp(−(gT/T_peak)⁴)`: 0.63 at the peak temperature without Doppler shift, leaving
headroom for the approaching side, and rising monotonically with `g`. Its hue is the
chromaticity of a blackbody at `gT` (Kang et al. 2002 fit of the Planckian locus →
linear sRGB); a colour too saturated to reach that luminance with every channel at
most 1 is desaturated toward white at constant luminance (`disk_rgb_linear`). The
result is sRGB-encoded (`disk_pixel`). The temperature is a **false-colour**
parameter: it sets the hue, not the physics; at realistic disk temperatures every
pixel would be white. The no-gravity image draws the disk with `g = 1`.

**Output.** `photon_data.csv` keeps its previous columns and gains, in this order,
`disk_r`, `disk_phi` (hit radius and azimuth in the ray's rotated frame), `disk_x`,
`disk_y`, `disk_z` (lab-frame hit position), `disk_g`, `disk_temp` (kelvin), all NaN
unless the ray ended on the disk (written as the literal `nan`, as in the Rust build),
and `plane_crossings`, which counts every crossing of the disk plane, the hit included
(0 without a disk). The summaries report `Hit disk`.

**Notes (behaviour shared with the Rust build).**

- A ray the integrator stops as diverged (status 3) is a `numerical error` (red), with
  or without a disk. Before the disk port the kernel only tested capture and escape, so
  such a ray kept stepping and was classified by its final radius; at `--delta 0.01`
  this almost never happens, at larger steps near-critical rays can diverge.
- The saved PNGs (`images/manual_output.png`, `images/no_gravity.png`) now have up at
  the top: row `i = 0` of the pixel grid is the bottom row of the image, as in the
  `origin='lower'` preview. Earlier PNGs were stored upside down, so the direction
  statements above (pitch, roll, near side of the disk) hold for the new files only.
- The scene plots draw the image plane of the pointed camera (`--yaw`, `--pitch`,
  `--roll`).
- Sampled trajectories are thinned over the recorded states only and always end on
  the exit state (or, for a disk hit, the state just past the disk).

```
NUMBA_ENABLE_CUDASIM=1 python main.py --size 100 --steps 20000 --disk --disk-inclination 75 --yaw 6 --pitch -4
```

Tests: `NUMBA_ENABLE_CUDASIM=1 python -m pytest tests/test_fixes.py tests/test_disk.py`.
`tests/gen_disk_fixture.py` records a 16×16 disk scene (`tests/fixtures/disk_scene.json`)
for comparison with the Rust build.
