"""Capture golden fixtures from the Python implementation for the Rust port's tests.

Usage, from the repository root with the Python dependencies installed:
    NUMBA_ENABLE_CUDASIM=1 python3 rust/tests/fixtures/gen_golden.py <output-dir>
The CUDA simulator makes the kernels run on the CPU; the whole run takes about 20 minutes.
Copy the JSON files and e2e/photon_data.csv (as e2e_photon_data.csv) into rust/tests/fixtures/.
"""
import os, sys, json, time
os.environ.setdefault("NUMBA_ENABLE_CUDASIM", "1")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
import numpy as np
from simulation.utils import get_initial_conditions
from simulation.cuda_geodesic import CUDASchwarzschildIntegrator

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- camera + initial conditions
h, w = 6, 8
fov = np.radians(80.0)
obs = np.array([30.0, 0.0, 0.0])
mass = 1.0
optical_axis = np.array([-1.0, 0, 0]); right = np.array([0.0, 1, 0]); up = np.array([0.0, 0, 1])
plane_dist = 0.2 * np.linalg.norm(obs)
plane_center = obs + optical_axis * plane_dist
plane_width = 2 * plane_dist * np.tan(fov / 2)
plane_height = plane_width * (h / w)
rows, q0s, p0s = [], [], []
for i in range(h):
    for j in range(w):
        u = (j + 0.5) / w - 0.5
        v = (i + 0.5) / h - 0.5
        pix = plane_center + u * plane_width * right + v * plane_height * up
        q0, p0, alpha0, hr, ht, hp, beta = get_initial_conditions(obs, pix.copy(), mass_bh=mass)
        rows.append(dict(i=i, j=j, pixel=pix.tolist(), q0=q0.tolist(), p0=p0.tolist(),
                         alpha0=float(alpha0), beta=float(beta), heading=[float(hr), float(ht), float(hp)]))
        q0s.append(q0); p0s.append(p0)
json.dump(dict(h=h, w=w, fov_deg=80.0, observer=obs.tolist(), mass=mass, rays=rows),
          open(f"{OUT}/initial_conditions.json", "w"))
q0s = np.array(q0s); p0s = np.array(p0s)
print("initial conditions done", flush=True)

# ---------------------------------------------------------------- integrator: final states
steps, delta, omega, r_max = 20000, 0.01, 0.01, 31.0
t = time.time()
integ = CUDASchwarzschildIntegrator(steps=steps, delta=delta, mass=mass, omega=omega, r_max=r_max)
out, _ = integ.integrate_batch(q0s, p0s)
json.dump(dict(steps=steps, delta=delta, omega=omega, r_max=r_max, mass=mass, q_final=out.tolist()),
          open(f"{OUT}/integrator_final.json", "w"))
print("integrator final done in %.0fs" % (time.time() - t), flush=True)

# ---------------------------------------------------------------- integrator: full trajectories
sel = [0, h * w // 2, h * w // 2 + 1, h * w - 1]
integ2 = CUDASchwarzschildIntegrator(steps=1500, delta=delta, mass=mass, omega=omega, r_max=r_max)
traj = integ2.integrate_batch_full(q0s[sel], p0s[sel])
json.dump(dict(steps=1500, delta=delta, omega=omega, r_max=r_max, mass=mass, ray_index=sel, traj=traj.tolist()),
          open(f"{OUT}/integrator_traj.json", "w"))
print("integrator traj done", flush=True)

# ---------------------------------------------------------------- texel rule ("method b", raytracing.py)
def texel_b(th, ph, h, w, pc_th, pc_ph, ps_th, ps_ph, flip_theta=False, flip_phi=False):
    theta0 = pc_th - ps_th / 2; theta1 = pc_th + ps_th / 2; phi0 = pc_ph - ps_ph / 2; phi_span = ps_ph
    th = th % (2 * np.pi); ph = ph % (2 * np.pi)
    dtheta = abs(th - pc_th)
    ph = (-ph) if flip_phi else ph
    phi_rel = (ph - phi0) % (2 * np.pi)
    dphi = abs((ph - pc_ph + np.pi) % (2 * np.pi) - np.pi)
    inside = (dtheta <= ps_th / 2) and (dphi <= phi_span / 2)
    if not inside:
        return [False, -1, -1]
    theta_map = (np.pi - th) if flip_theta else th
    u = int((theta_map - theta0) / (theta1 - theta0) * (h - 1) + 0.5)
    v = int(phi_rel / phi_span * (w - 1) + 0.5)
    return [True, int(np.clip(u, 0, h - 1)), int(np.clip(v, 0, w - 1))]

cases = []
rng = np.random.default_rng(7)
for pc_th, pc_ph, ps_th, ps_ph in [(np.pi / 2, np.pi, np.pi, 2 * np.pi), (np.pi / 2, np.pi, np.radians(40), np.radians(60)), (np.radians(70), np.radians(350), np.radians(30), np.radians(50))]:
    for _ in range(60):
        th = float(rng.uniform(0, np.pi)); ph = float(rng.uniform(-np.pi, np.pi))
        for ft, fp in [(False, False), (True, False), (False, True), (True, True)]:
            cases.append(dict(th=th, ph=ph, h=64, w=128, pc_th=pc_th, pc_ph=pc_ph, ps_th=ps_th, ps_ph=ps_ph,
                              flip_theta=ft, flip_phi=fp, result=texel_b(th, ph, 64, 128, pc_th, pc_ph, ps_th, ps_ph, ft, fp)))
json.dump(cases, open(f"{OUT}/texel_method_b.json", "w"))
print("texel done", flush=True)

# ---------------------------------------------------------------- flat render hit points (background.py loop)
from simulation.blackhole import Observer
from simulation.background import save_no_gravity_image_with_background
hh, ww = 16, 16
observer = Observer(position=obs, fov=fov, image_size=(hh, ww))
flat_dir = f"{OUT}/flat"; os.makedirs(flat_dir, exist_ok=True)
trajs = save_no_gravity_image_with_background(
    observer, f"{REPO}/images/backgrounds/milky-way-equirec.jpg", f"{flat_dir}/no_gravity.png",
    boundary_radius=31.0, patch_center_theta=np.pi / 2, patch_center_phi=np.pi,
    patch_size_theta=np.pi, patch_size_phi=2 * np.pi, flip_theta=False, flip_phi=False,
    return_sampled_trajectories=True, n_sampled=hh * ww, override_patch_center=False)
hits = [t[-1].tolist() for t in trajs]  # row-major (i, j) order because every pixel is sampled
json.dump(dict(h=hh, w=ww, fov_deg=80.0, observer=obs.tolist(), boundary_radius=31.0, hits=hits),
          open(f"{OUT}/flat_hits.json", "w"))
print("flat done", flush=True)

# ---------------------------------------------------------------- end-to-end curved render (16x16)
from simulation.blackhole import BlackHole
from simulation.raytracing import run_manual_simulation
e2e = f"{OUT}/e2e"; os.makedirs(f"{e2e}/images", exist_ok=True)
os.chdir(e2e)
t = time.time()
bh = BlackHole(mass=mass)
img = run_manual_simulation(
    bh, observer, steps=20000, delta=0.01, omega=0.01, background_path=f"{REPO}/images/backgrounds/milky-way-equirec.jpg",
    use_cuda=True, boundary_radius=31.0, patch_center_theta=np.pi / 2, patch_center_phi=np.pi,
    patch_size_theta=np.pi, patch_size_phi=2 * np.pi, flip_theta=False, flip_phi=False, n_samples=0)
print("e2e done in %.0fs" % (time.time() - t), flush=True)
print("ALL GOLDEN DONE")
