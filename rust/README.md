# schwarzschild-rt

A Rust port of the Schwarzschild ray tracer in the parent directory. It integrates
null geodesics with the same FANTASY order-2 symplectic scheme as the Python CUDA
kernel and reproduces the Python results to rounding (see *Verification*). Three
backends share one pipeline: a four-lane SIMD integrator on the CPU (the default),
the scalar reference integrator, and a CUDA kernel. An optional thin accretion disk
is rendered with its Keplerian redshift, and the camera can be pointed away from the
black hole.

```
cd rust
cargo build --release
./target/release/schwarzschild-rt --size 200 --background ../images/backgrounds/milky-way-equirec.jpg --seed 1
./target/release/schwarzschild-rt --size 400 --disk --disk-inclination 75 --yaw 10 --pitch -5
```

`--help` lists every flag. Names follow `config.py` except:

- `--no-flat` replaces `--no-flat-trajectories` (the old spelling is accepted as an alias);
- `--background` has no default: omit it for a black sky (Python defaults to
  `images/backgrounds/milky-way-equirec.jpg`, relative to the repository root);
- `--suppress-warnings` and `--no-cuda` do not exist;
- `--n-sample-rays`, `--seed`, `--max-trajectory-points`, `--out-dir`, `--threads`,
  `--backend`, `--exact` and `--cuda-device` are new;
- `--yaw`, `--pitch`, `--roll` point the camera (*Camera pointing*); `--disk`,
  `--disk-inclination`, `--disk-azimuth`, `--disk-inner`, `--disk-outer`,
  `--disk-temperature` add the accretion disk (*Accretion disk*). The Python build on
  its own branch takes the same flags with the same meaning;
- `--boundary-radius` must stay below 100 (the radius beyond which a ray is reported as
  a numerical error) and `--observer-distance` must exceed the photon sphere, `3M`.

Units: `G = c = 1`; `--bh-mass` sets the unit of length for every radius and for
`--delta`. Every backend rescales each ray to units of `M` before stepping, so a
black hole of any mass runs the same scheme as `M = 1` (see `integrator.rs`); the
FANTASY mixing angle per step is `2·omega·delta/M`.

## Backends and speed

| `--backend` | What runs | 200×200, 200 000-step budget, 4 cores |
| --- | --- | --- |
| `cpu` (default on a CPU-only build) | `integrator_fast`: the step specialised to the equatorial plane every pipeline ray lives in (`θ = π/2`, `p_θ = 0`), four rays per instruction with `wide::f64x4`, rays grouped by impact parameter so lanes finish together | **1.5 s** |
| `cpu --exact` | `integrator`: the scalar step, operation for operation the Python kernel | 10.9 s |
| `cuda` | `fantasy.cu`, the same scalar step as a CUDA kernel, compiled with NVRTC at start-up | not measured here (no GPU) |
| `auto` | `cuda` when the binary has it, the driver and `libnvrtc` load and a device is present, else `cpu` | |

The SIMD path keeps the scalar step's operation order and nothing contracts to FMA, so
`t, r, φ, p_t, p_r, p_φ` come out bit-identical to the scalar integrator; the only
difference is the `cos(π/2) ≈ 6×10⁻¹⁷` kick the general step gives `p_θ`, which the
tests bound at 10⁻¹². The tests assert bit identity on 169 rays and whole-file
identity of the photon table, the trajectories and the PNG for a 25×25 and a 5×5 scene,
using odd sizes so that a lane mix-up could not hide. Sampled trajectories always come
from the scalar integrator, which is the only one that records every step.

### CUDA

```
cargo build --release --features cuda-12      # CUDA 12.x driver; --features cuda-11 for 11.8
./target/release/schwarzschild-rt --backend cuda --size 1000 --background ../images/backgrounds/milky-way-equirec.jpg
```

Nothing from the CUDA toolkit is needed at build time. At run time the backend needs
`libcuda` (the display driver) and `libnvrtc` (from the CUDA toolkit or the NVRTC
redistributable; it is not part of the driver). The kernel source is embedded in the
binary and compiled by NVRTC when the backend opens. `auto` probes both libraries
first: without a driver it uses the CPU; with a driver but no `libnvrtc` it says so
and stops. `libnvrtc` must not be newer than the driver (the error names
`CUDA_ERROR_UNSUPPORTED_PTX_VERSION` when it is).

Rounding: the kernel is compiled with `--fmad=false`, precise division and square
root and no flush-to-zero, so `+ − × ÷ √` round as on the CPU; the device
`sin`/`cos`/`sincos` are specified to 2 ulp and are not glibc's code, so GPU and CPU
results agree to a few ulp per step, not bit for bit, and a few pixels on a
classification boundary may differ. Each launch advances the rays by at most 4 096
steps and the host relaunches until all have exited, so no kernel runs long enough
to trip a display GPU's watchdog and progress is reported between rounds.

The kernel takes the same scaled inputs as the CPU integrators and returns the same
exit codes. It was written and reviewed against `integrator.rs` and the Python kernel
but has not been run in this environment: the test suite checks that it is valid C++
against a shim (`tests/kernel_syntax.rs`, needs `g++` or `clang++`; set
`SCHWARZSCHILD_RT_SKIP_KERNEL_SYNTAX=1` to skip) and that its typed parameter list
matches the host launch, not that it runs. Run `cargo test --release --features
cuda-12` and a small render on a machine with a GPU before trusting it.

## Camera pointing

The observer stays on the `+x` axis (the equatorial rotation of every ray needs
that), so the camera pans but never moves. Its basis (`optical_axis = −x̂`,
`right = +ŷ`, `up = +ẑ` when pointed at the hole) is turned by three intrinsic
angles in this order, each a right-handed rotation about the named axis of the camera
as it then stands:

| Flag | Axis | Positive value |
| --- | --- | --- |
| `--yaw` | up | looks to the observer's left; the black hole moves right in the image |
| `--pitch` | right | looks up; the black hole moves down |
| `--roll` | optical axis | turns the image counter-clockwise |

Ray directions are unit vectors in the static observer's orthonormal frame, so the
rotation acts in that frame and `--fov` is the horizontal angle the observer measures.
Yaw by `fov/2` puts the hole at the image edge. Everything downstream (initial
conditions, the sky patch, the flat render) takes arbitrary directions from a point
on the `x` axis; with all three angles zero the pixel rays are bit-identical to the
unrotated camera's.

## Accretion disk

`--disk` adds a geometrically thin, optically thick disk of material on circular
Keplerian orbits in a plane through the black hole (`disk.rs`). The first crossing
of that plane inside `[r_in, r_out]` ends the ray; crossings outside the band pass
through and are counted.

**Geometry.** The plane is `n · x = 0`. `--disk-inclination` is the angle between
`n` and the observer's position vector `+x̂` (0 face-on, 90 edge-on, 180 face-on with
the opposite sense of rotation); `--disk-azimuth` turns `n` right-handedly about
`+x̂`: `n = R_x(azimuth) · (cos i, 0, sin i)`. At azimuth 0 an inclined disk's near
side is below the hole in the image and its far side above. Both angles are measured
against the observer's position, not the optical axis, so pointing the camera does
not move the disk. The material orbits right-handedly about `n`: at `x` it moves along
`n × x̂`. Edge-on at azimuth 0 (`n = +ẑ`) the material at `(0, −r, 0)`, on the
observer's left, moves towards the observer, so the left side approaches and is
blueshifted. `(i, a) → (180° − i, a + 180°)` reverses the normal and with it the sense
of rotation. `--disk-inner` defaults to the ISCO `6M` and may not be smaller (no stable
circular orbits inside it; below `3M` the redshift is undefined); `--disk-outer`
defaults to `20M` and must stay below `--observer-distance`.

**Detection** (`integrator.rs`, "Disk crossings"). Each ray is integrated in a frame
rotated about `+x̂` by `−β`, where its orbit is the x–y plane and the disk normal is
`n' = R_x(−β) n`. The disk plane meets the orbital plane along the line at azimuth
`phi_d = atan2(−n'_x, n'_y)`, so a crossing is a change of exactly one in
`floor((φ − phi_d)/π)` between consecutive steps (`φ` is never wrapped); the radius
is interpolated linearly in `φ`. The check runs after every step from the second
step on (every ray starts at the observer, which lies on the crossing line of an
edge-on disk) and before the next step's capture/escape test. A ray whose orbit lies
in the disk plane (`|n'_⊥| < 1e-9`) detects nothing. The scalar, SIMD and CUDA
integrators share this arithmetic operation for operation, so hit radii and azimuths
are bit-identical between the two CPU paths.

**Redshift.** The physical photon is the time reverse of the traced ray: its energy
at infinity is `E = p_t`, its angular momentum about `n` is `L_n = −p_φ n'_z`, and
with `b_z = L_n/E`, `Ω = √(M/r³)`:

```
g = ν_obs / ν_emit = (k·u_obs) / (k·u_emit) = √(1 − 3M/r) / ((1 − Ω b_z) √(1 − 2M/r_obs))
```

for the static observer `u_obs = (1,0,0,0)/√(1 − 2M/r_obs)` and the circular orbit
`u_emit = (1,0,0,Ω)/√(1 − 3M/r)`. `g` is a ratio of two contractions of one covector,
so the sign convention of the traced momentum cancels; `p_t`, `p_φ` are conserved, so
`g` depends only on the hit. A hit with `1 − Ω b_z ≤ 0` cannot reach the observer and is
reported as a numerical error.

**Emission.** The effective temperature follows the Newtonian Shakura–Sunyaev profile
with zero torque at the inner edge, `T⁴ ∝ (1 − √(r_in/r))/r³`, peaking at
`49/36 r_in` with the value of `--disk-temperature`. By Liouville's theorem the
observed spectrum is a blackbody at `gT`: the pixel's luminance is the tone-mapped
`(gT/T_peak)⁴` (`1 − exp(−2x)`), its chromaticity that of a blackbody at `gT`
(Kang et al. 2002 fit, linear sRGB, then gamma-encoded). Nothing is multiplied by
`g⁴` on top of the blackbody at `gT`. The temperature is a false-colour parameter:
at realistic disk temperatures every pixel would be white. The no-gravity reference
draws the disk with straight rays and `g = 1`.

## Outputs (`--out-dir`, default `output/`)

| File | Content |
| --- | --- |
| `manual_output.png` | The lensed image. Row 0 is the top of the image; `up` is `+z`. |
| `no_gravity.png` | Straight-ray reference through the same camera and sky patch. |
| `photon_data.csv` | One row per pixel. The first fifteen columns carry the names of the Python table (read them by name); `exit`, `n_steps`, `beta` follow, then `disk_r`, `disk_phi` (hit radius and azimuth in the ray's rotated frame), `disk_x`, `disk_y`, `disk_z` (lab-frame hit position), `disk_g`, `disk_temp` (kelvin), all `nan` unless the ray ended on the disk (`collision = disk`, `exit = disk`), and `plane_crossings`, the number of disk-plane crossings including the hit. `final_th ∈ [0, π]`, `final_ph ∈ (−π, π]` in the lab frame for every row (Python wrote `φ mod 2π`, negated under `--bg-flip-phi`, for escaped rows). `i` counts rows from the bottom. |
| `sampled_rays.csv` | Lab-frame points of `--n-sample-rays` curved trajectories (`--seed` makes the choice reproducible); `pixel = i·width + j` joins a ray to `photon_data.csv`. |
| `flat_rays.csv` | The same pixels' straight rays. |

## Layout

| Module | Owns |
| --- | --- |
| `camera` | The pinhole camera: one owner of the image-plane geometry, including yaw/pitch/roll. |
| `disk` | The accretion disk: geometry, per-ray crossing line, redshift, temperature, colour. |
| `initial_conditions` | Pixel ray → null covariant 4-momentum, rotated into the equatorial plane. |
| `integrator` | The scalar FANTASY step in units of `M`, exit conditions, disk-crossing detection, bounded trajectory recording. |
| `integrator_fast` | The equatorial four-lane SIMD step and its batch driver. |
| `cuda` (feature `cuda`) + `fantasy.cu` | The GPU backend: NVRTC compilation, buffers, launch. |
| `backend` | The `BatchIntegrator` trait and the three implementations. |
| `sky_patch` | Which directions carry the texture and the one equirectangular texel rule. |
| `shading` | Un-rotate the exit, classify, sample the texture or shade the disk hit. |
| `render` | The curved pipeline, no I/O. |
| `flat` | The no-gravity reference, same camera and patch. |
| `output` | PNG and CSV writers, the collision summary. |
| `schwarzschild`, `coords`, `texture`, `sampling` | Closed-form quantities, vector helpers, texture loading, seeded pixel choice. |

## Differences from the Python implementation

All deliberate; each is pinned by a test named in brackets. Items 1, 2 and 5 have
since been ported back into the Python code on `main`.

1. **Radial momentum.** The Python code built `p_r = n_r̂ √f` (the contravariant
   formula) next to the covariant `p_θ`, `p_φ`. That stretched every ray's angle from
   the optical axis by `1/f ≈ 1.07` at `r = 30`, so the integrated shadow came out
   6.6 % too small in angle. The port uses the covariant `p_r = n_r̂ / √f`; the
   integrated shadow edge now sits at the analytic critical angle
   `sin α_c = (3√3 M / r) √(1 − 2M/r)` to within 1e-4 rad (measured ≈5e-6).
   [`physics::shadow_edge_matches_critical_angle`, `physics::python_momentum_convention_shrinks_the_shadow_by_the_lapse`]
2. **One camera, one patch rule, for both renders.** `background.py` built
   `right = cross(up, axis) = −ŷ` while the curved renderer used `+ŷ`, so the
   no-gravity image was mirrored. It also always re-centred the patch on `−x`
   (ignoring `--bg-patch-center-*` and the offsets), tested membership with an
   interval test that degenerates for a 360° patch (`φ0 ≡ φ1 mod 2π`, so the default
   `no_gravity.png` was black except at `φ = 0`; the CUDA flat kernel shares the
   test), and truncated the texel index where the curved renderer rounded. The port's
   flat render uses `camera::Camera` and `sky_patch::SkyPatch` exactly as the curved
   render does, and records the same `--n-sample-rays` pixels (Python: 10 unseeded).
   [`golden::flat_hits_match_python_up_to_its_mirrored_camera`, `cli::writes_every_output_file_and_shades_from_the_texture`]
3. **Texture at native resolution.** The Python code resized the texture to the
   output size before sampling; the port samples the texture as loaded.
   [`render::escaped_pixels_sample_the_texture_at_their_exit_direction`]
4. **Patch membership ignores the flips.** A flip mirrors the texture inside the
   patch; it does not move the patch. (The Python curved renderer flipped `φ` before
   the membership test.) Identical for the default full-sky patch.
   [`shading::flipped_patch_mirrors_the_texel_but_not_the_pixel_classification`, `golden::texel_rule_matches_python_method_b`]
5. **Masses other than 1.** The Python kernel hard-coded `r_s = 2` in `∂g^{tt}/∂r`
   and `∂g^{rr}/∂r`, and its FANTASY coupling grew as `ω M²` because the mixing
   rotation treats length-carrying and dimensionless components alike. The port
   integrates in units of `M`, so every mass reproduces the `M = 1` trajectory.
   [`physics::shadow_edge_is_the_same_at_every_mass`, `physics::near_critical_ray_is_scale_invariant`]
6. **Image orientation.** The PNG has `up` at the top (`matplotlib.imsave` wrote row 0
   at the top, which is the bottom of the camera plane). A positive
   `--bg-patch-center-theta-relobs` therefore moves the patch down in the image.
   [`output::png_has_up_at_the_top`]
7. **Trajectory thinning.** Sampled trajectories are recorded with a bounded reservoir
   and thinned over the recorded points (nearest-rounded, at most
   `--max-trajectory-points`), always keeping the start and the exit state. The Python
   code thinned the full `steps` buffer with `np.linspace(..., dtype=int32)` (floor)
   and wrote the all-zero rows that follow an early exit, so a captured ray kept only
   a handful of real points. [`integrator::bounded_trajectory_is_a_thinned_copy_of_the_full_one`]
8. **Divergence is reported.** A ray whose state stops being physical (non-finite or
   non-positive `r`, or the second FANTASY copy inside the horizon) exits as `diverged`
   and is coloured as a numerical error instead of black.
   [`integrator::non_finite_state_is_reported_as_diverged`, `shading::shade_colours_each_outcome`]

## Verification

`cargo test --release` runs five layers:

- unit tests in each module. `disk.rs` holds the oblique oracle: on a disk at
  inclination 55°, azimuth 30°, the closed-form `g` from `(p_t, p_φ, n'_z)` equals the
  Doppler + gravitational factor computed independently from lab-frame geometry
  (emitter 3-velocity and photon direction in the static frame) to 1e-12 on 200
  configurations, and fails with the opposite frame rotation; `integrator.rs` checks
  the crossing detector on a straight line (second-order convergence of the hit
  radius) and a circle (two crossings per revolution), that edge-on rays register no
  crossing at the observer, that a last-step crossing is still a hit, and that the
  disk-off path is unchanged; `integrator_fast.rs` checks hits, azimuths and crossing
  counts bit-identical to the scalar integrator, with mixed detecting and
  non-detecting lanes. The other unit tests include the SIMD step against the scalar one
  (`integrator_fast`: bit-identical states, exits and step counts on 169 pipeline
  rays and on one group of four distinct rays, a non-equatorial ray falling back to
  the scalar path, step-limit and last-step crossings, mass rescaling, a NaN and a
  negative-radius lane reported as diverged without touching their neighbours) and a
  25×25 render with both CPU backends giving identical photon records
  (`render::fast_backend_renders_the_same_image_as_the_exact_one`);
- `tests/golden.rs`: fixtures captured from the Python code under numba's CUDA
  simulator (`tests/fixtures/`, regenerated with `tests/fixtures/gen_golden.py`, about
  20 minutes). Measured agreement, with the asserted bound in brackets: initial
  conditions to 1e-13 (with the documented `p_r` factor) [1e-12]; FANTASY trajectories
  step by step through capture and escape (8 602 recorded states) to 7e-15 [1e-12] and
  the final states of 48 rays to 1e-13 in `r` and 9e-12 in `t` [1e-11, 1e-10], from
  both the scalar and the SIMD integrator; flat-render hit points to 1e-9 after
  mirroring [1e-9]; patch membership on all 720 "method b" cases and texel indices on
  the 264 directions both rules accept; the 16×16 end-to-end photon table (exit radius
  bit for bit, exit direction to 2e-15 [1e-9], every classification) when integrated
  from the Python momenta; and `tests/fixtures/disk_scene.json`, a 16×16 scene with an
  oblique disk (inclination 55°, azimuth 30°) and a turned camera (yaw 6°, pitch −4°)
  captured with the Python branch's `tests/gen_disk_fixture.py`: initial conditions
  and per-ray disk frames agree to 1e-12, and from the Python momenta the scalar and
  SIMD integrators reproduce every status (a flip is tolerated only on a pixel
  bordering a different status, at most two per image, since status is a discrete
  decision; none occur), hit radii to 4e-15 relative, hit azimuths to 4e-16 and `g` to
  1e-15 [1e-9]; the whole Rust pipeline from the flags alone gives the same;
- `tests/physics.rs`: closed-form checks that need no reference implementation,
  including the disk: every hit in both CPU backends has a physical `g` that the
  momentum carried to the hit confirms, reversing the normal mirrors the image and
  flips the Doppler side, a disk at azimuth `α` seen unrolled equals the disk at
  azimuth 0 seen with roll `α` (pins the disk's handedness against the camera's), and
  a face-on disk shows no Doppler shift at any pointing;
- `tests/cli.rs`: the binary's files, thread-count determinism, flag parity, argument
  validation, the SIMD and exact backends writing byte-identical files, and (in a
  `cuda-*` build) `auto` running on a machine without a device;
- `tests/kernel_syntax.rs`: the CUDA kernel compiles as C++ against a shim and its
  typed parameter list matches the host launch.

The CUDA kernel's disk detection is the same arithmetic as the CPU paths but has
only been syntax-checked here (no device); a GPU run should agree with `--exact` to
the few-ulp level described under *CUDA*, with status flips possible on pixels
within rounding of `r_in`, `r_out` or the shadow edge.
