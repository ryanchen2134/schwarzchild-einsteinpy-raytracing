"""Capture the Python disk pipeline on a small scene as a cross-build fixture.

Usage, from the repository root:

    NUMBA_ENABLE_CUDASIM=1 python3 tests/gen_disk_fixture.py [output.json]

The default output is tests/fixtures/disk_scene.json. Under the CUDA simulator the
256 rays take several minutes.

Scene: 16x16 image, fov 80 deg, observer (30, 0, 0), boundary radius 31, M = 1,
20000 steps of delta 0.01, omega 0.01, disk inclination 55 deg, azimuth 30 deg,
r_in 6, r_out 20, peak temperature 6000 K, camera yaw 6 deg, pitch -4 deg, roll 0.

Per pixel (row i from the bottom, column j, row-major index i*16 + j) the record
holds: i, j, status (0 step limit, 1 captured, 2 escaped, 4 disk; 3, diverged, does
not occur in this scene and is not listed in status_codes), r_hit and phi_hit (the
interpolated crossing in the ray's rotated frame; null when the ray did not end on
the disk), g (the redshift factor from the conserved initial p_t, p_phi; null when no
hit), plane_crossings, p0 (p_t, p_r, p_theta, p_phi), beta, phi_d, n_z (the disk in
the ray's frame), the final q (t, r, theta, phi) of the first FANTASY copy, and three
fields the Rust build does not read but tests/test_disk.py checks: alpha0 (the
angle between the ray and the -x axis, from get_initial_conditions), hit_xyz (the
lab-frame hit position hit_position_lab(r_hit, phi_hit, beta); [null, null, null]
when no hit) and disk_temp (Disk.temperature(r_hit) in kelvin; null when no hit).

NaN is written as JSON null (standard JSON has no NaN token); the pytest that reloads
the fixture maps null back to NaN.
"""
import json
import math
import os
import sys
import time

os.environ.setdefault("NUMBA_ENABLE_CUDASIM", "1")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

import numpy as np

from simulation.cuda_geodesic import CUDASchwarzschildIntegrator
from simulation.disk import Disk, hit_position_lab, redshift
from simulation.utils import camera_basis, get_initial_conditions, pixel_positions

SCENE = dict(
    size=16, fov_deg=80.0, observer=[30.0, 0.0, 0.0], boundary_radius=31.0, mass=1.0,
    steps=20000, delta=0.01, omega=0.01,
    disk_inclination_deg=55.0, disk_azimuth_deg=30.0, disk_inner=6.0, disk_outer=20.0,
    disk_temperature=6000.0, yaw_deg=6.0, pitch_deg=-4.0, roll_deg=0.0,
)


def _num(v):
    v = float(v)
    return None if math.isnan(v) else v


def main(out_path):
    s = SCENE
    h = w = s["size"]
    fov = math.radians(s["fov_deg"])
    obs = np.array(s["observer"], dtype=np.float64)
    mass = s["mass"]
    r_obs = float(np.linalg.norm(obs))
    normal = Disk.normal_from_angles(math.radians(s["disk_inclination_deg"]),
                                     math.radians(s["disk_azimuth_deg"]))
    disk = Disk(normal, s["disk_inner"], s["disk_outer"], s["disk_temperature"], mass, r_obs)
    axis, right, up = camera_basis(math.radians(s["yaw_deg"]), math.radians(s["pitch_deg"]),
                                   math.radians(s["roll_deg"]))
    positions = pixel_positions(obs, fov, h, w, axis, right, up)

    n = h * w
    q0s = np.zeros((n, 4)); p0s = np.zeros((n, 4))
    betas = np.zeros(n); phi_ds = np.zeros(n); n_zs = np.zeros(n); alpha0s = np.zeros(n)
    for i in range(h):
        for j in range(w):
            idx = i * w + j
            q0, p0, alpha0, _, _, _, beta = get_initial_conditions(obs, positions[i, j].copy(), mass_bh=mass)
            q0s[idx] = q0; p0s[idx] = p0; betas[idx] = beta; alpha0s[idx] = alpha0
            phi_ds[idx], n_zs[idx] = disk.ray_frame(beta)
    print("initial conditions done", flush=True)

    integ = CUDASchwarzschildIntegrator(steps=s["steps"], delta=s["delta"], mass=mass,
                                        omega=s["omega"], r_max=s["boundary_radius"])
    t0 = time.time()
    out_qs, status, hits, crossings = integ.integrate_batch(
        q0s, p0s, disk_phi=phi_ds, disk_r_in=disk.r_in, disk_r_out=disk.r_out)
    print("integration done in %.0fs" % (time.time() - t0), flush=True)

    pixels = []
    for idx in range(n):
        i, j = divmod(idx, w)
        st = int(status[idx])
        if st == 4:
            r_hit, phi_hit = float(hits[idx, 0]), float(hits[idx, 1])
            g = redshift(mass, r_hit, float(p0s[idx, 0]), float(p0s[idx, 3]), float(n_zs[idx]), r_obs)
            pos = hit_position_lab(r_hit, phi_hit, float(betas[idx]))
            temp = disk.temperature(r_hit)
        else:
            r_hit = phi_hit = g = temp = math.nan
            pos = [math.nan] * 3
        pixels.append(dict(
            i=i, j=j, status=st, r_hit=_num(r_hit), phi_hit=_num(phi_hit), g=_num(g),
            plane_crossings=int(crossings[idx]),
            p0=[float(v) for v in p0s[idx]], beta=float(betas[idx]),
            phi_d=_num(phi_ds[idx]), n_z=float(n_zs[idx]),
            q=[float(v) for v in out_qs[idx]],
            alpha0=float(alpha0s[idx]), hit_xyz=[_num(v) for v in pos], disk_temp=_num(temp),
        ))
    doc = dict(
        scene=dict(s, disk_normal=list(disk.normal), r_obs=r_obs),
        status_codes={"0": "step_limit", "1": "captured", "2": "escaped", "4": "disk"},
        nan_encoding="null",
        pixel_order="row-major, index = i*size + j, row i = 0 at the bottom of the image",
        pixels=pixels,
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=1)
    n_disk = sum(1 for p in pixels if p["status"] == 4)
    print(f"wrote {out_path}: {n_disk} disk hits of {n} rays", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "tests", "fixtures", "disk_scene.json"))
