# schwarzschild-rt

A Rust port of the Schwarzschild ray tracer in the parent directory. It integrates
null geodesics with the same FANTASY order-2 symplectic scheme as the Python CUDA
kernel, on the CPU with [rayon](https://docs.rs/rayon), and reproduces the Python
results to rounding (see *Verification*).

```
cd rust
cargo build --release
./target/release/schwarzschild-rt --size 200 --background ../images/backgrounds/milky-way-equirec.jpg --seed 1
```

200×200 pixels with the default 200 000-step budget take about 9 s on four cores.
`--help` lists every flag. Names follow `config.py` except:

- `--no-flat` replaces `--no-flat-trajectories` (the old spelling is accepted as an alias);
- `--background` has no default: omit it for a black sky (Python defaults to
  `images/backgrounds/milky-way-equirec.jpg`, relative to the repository root);
- `--suppress-warnings` and `--no-cuda` do not exist;
- `--n-sample-rays`, `--seed`, `--max-trajectory-points`, `--out-dir` and `--threads` are new;
- `--boundary-radius` must stay below 100 (the radius beyond which a ray is reported as
  a numerical error) and `--observer-distance` must exceed the photon sphere, `3M`.

Units: `G = c = 1`; `--bh-mass` sets the unit of length for every radius and for
`--delta`. The integrator rescales each ray to units of `M` before stepping, so a
black hole of any mass runs the same scheme as `M = 1` (see `integrator.rs`); the
FANTASY mixing angle per step is `2·omega·delta/M`.

## Outputs (`--out-dir`, default `output/`)

| File | Content |
| --- | --- |
| `manual_output.png` | The lensed image. Row 0 is the top of the image; `up` is `+z`. |
| `no_gravity.png` | Straight-ray reference through the same camera and sky patch. |
| `photon_data.csv` | One row per pixel. The first fifteen columns carry the names of the Python table (read them by name); `exit`, `n_steps`, `beta` follow. `final_th ∈ [0, π]`, `final_ph ∈ (−π, π]` in the lab frame for every row (Python wrote `φ mod 2π`, negated under `--bg-flip-phi`, for escaped rows). `i` counts rows from the bottom. |
| `sampled_rays.csv` | Lab-frame points of `--n-sample-rays` curved trajectories (`--seed` makes the choice reproducible); `pixel = i·width + j` joins a ray to `photon_data.csv`. |
| `flat_rays.csv` | The same pixels' straight rays. |

## Layout

| Module | Owns |
| --- | --- |
| `camera` | The pinhole camera: one owner of the image-plane geometry. |
| `initial_conditions` | Pixel ray → null covariant 4-momentum, rotated into the equatorial plane. |
| `integrator` | FANTASY order-2 step in units of `M`, exit conditions, batch integration, bounded trajectory recording. |
| `sky_patch` | Which directions carry the texture and the one equirectangular texel rule. |
| `shading` | Un-rotate the exit, classify, sample the texture. |
| `render` | The curved pipeline, no I/O. |
| `flat` | The no-gravity reference, same camera and patch. |
| `output` | PNG and CSV writers, the collision summary. |
| `schwarzschild`, `coords`, `texture`, `sampling` | Closed-form quantities, vector helpers, texture loading, seeded pixel choice. |

## Differences from the Python implementation

All deliberate; each is pinned by a test named in brackets.

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

`cargo test --release` runs four layers:

- unit tests in each module;
- `tests/golden.rs`: fixtures captured from the Python code under numba's CUDA
  simulator (`tests/fixtures/`, regenerated with `tests/fixtures/gen_golden.py`, about
  20 minutes). Measured agreement, with the asserted bound in brackets: initial
  conditions to 1e-13 (with the documented `p_r` factor) [1e-12]; FANTASY trajectories
  step by step through capture and escape (8 602 recorded states) to 7e-15 [1e-12] and
  the final states of 48 rays to 1e-13 in `r` and 9e-12 in `t` [1e-11, 1e-10]; flat-render hit points to 1e-9
  after mirroring [1e-9]; patch membership on all 720 "method b" cases and texel
  indices on the 264 directions both rules accept; the 16×16 end-to-end photon table
  (exit radius bit for bit, exit direction to 2e-15 [1e-9], every classification) when
  integrated from the Python momenta;
- `tests/physics.rs`: closed-form checks that need no reference implementation;
- `tests/cli.rs`: the binary's files, thread-count determinism, flag parity and
  argument validation.
