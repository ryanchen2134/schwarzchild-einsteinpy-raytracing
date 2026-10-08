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

200×200 pixels with the default 200 000-step budget take about 14 s on four cores.
`--help` lists every flag; the names match `config.py`.

## Outputs (`--out-dir`, default `output/`)

| File | Content |
| --- | --- |
| `manual_output.png` | The lensed image. Row 0 is the top of the image; `up` is `+z`. |
| `no_gravity.png` | Straight-ray reference through the same camera and sky patch. |
| `photon_data.csv` | One row per pixel: exit state, classification, integrator exit reason and step count, initial momentum, `alpha0`, `beta`, `analytic_capture`. |
| `sampled_rays.csv` | Lab-frame points of `--n-sample-rays` curved trajectories (`--seed` makes the choice reproducible). |
| `flat_rays.csv` | The same pixels' straight rays. |

## Layout

| Module | Owns |
| --- | --- |
| `camera` | The pinhole camera: one owner of the image-plane geometry. |
| `initial_conditions` | Pixel ray → null covariant 4-momentum, rotated into the equatorial plane. |
| `integrator` | FANTASY order-2 step, exit conditions, batch integration. |
| `sky_patch` | Which directions carry the texture and the one equirectangular texel rule. |
| `shading` | Un-rotate the exit, classify, sample the texture. |
| `render` | The curved pipeline, no I/O. |
| `flat` | The no-gravity reference, same camera and patch. |
| `output` | PNG and CSV writers, the collision summary. |
| `schwarzschild`, `coords`, `texture`, `sampling` | Closed-form quantities, vector helpers, texture loading, seeded pixel choice. |

## Differences from the Python implementation

All deliberate; each is pinned by a test.

1. **Radial momentum.** The Python code built `p_r = n_r̂ √f` (the contravariant
   formula) next to the covariant `p_θ`, `p_φ`. That stretched every ray's angle from
   the optical axis by `1/f ≈ 1.07` at `r = 30`, so the integrated shadow came out
   6.6 % too small in angle. The port uses the covariant `p_r = n_r̂ / √f`; the
   integrated shadow edge now sits at the analytic critical angle
   `sin α_c = (3√3 M / r) √(1 − 2M/r)` to better than 1e-4 rad (`tests/physics.rs`).
2. **One camera.** `background.py` built `right = cross(up, axis) = −ŷ` while the
   curved renderer used `+ŷ`, so the no-gravity image was mirrored. Both renders now
   share `camera::Camera`.
3. **Texture at native resolution.** The Python code resized the texture to the
   output size before sampling; the port samples the texture as loaded.
4. **Patch membership ignores the flips.** A flip mirrors the texture inside the
   patch; it does not move the patch. (The Python curved renderer flipped `φ` before
   the membership test.) Identical for the default full-sky patch.
5. **`r_s` in the metric derivative.** The Python kernel hard-coded `2` for `r_s` in
   `∂g^{tt}/∂r` and `∂g^{rr}/∂r`; the port uses `r_s`, identical at `M = 1`.
6. **Image orientation.** The PNG has `up` at the top (`matplotlib.imsave` wrote row 0
   at the top, which is the bottom of the camera plane).

## Verification

`cargo test --release` runs three layers:

- unit tests in each module;
- `tests/golden.rs`: fixtures captured from the Python code under numba's CUDA
  simulator (`tests/fixtures/`, regenerated with `tests/fixtures/gen_golden.py`).
  Initial conditions agree to 1e-13 (with the documented `p_r` factor), FANTASY
  trajectories agree step by step to 4e-15 over 1 500 steps and the final states of
  48 rays to 1e-13 in `r`, the flat-render hit points agree to 1e-9 after mirroring,
  the texel rule agrees with the Python "method b" on 720 cases, and the 16×16
  end-to-end photon table (exit radius, exit direction, classification) matches
  pixel for pixel when integrated from the Python momenta;
- `tests/physics.rs`: closed-form checks that need no reference implementation.
