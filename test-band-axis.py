# theta_band_demo.py
# ------------------------------------------------------------
# A one-off script that reproduces your main.py scene,          #
# samples 50 rays in θ ∈ [π-10°, π+10°], produces the image    #
# AND a 3-D plot with ≤ 500 points on each trajectory.         #
# ------------------------------------------------------------
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D               # noqa: F401

# -- your project imports ------------------------------------
from simulation.blackhole   import BlackHole, Observer
from simulation.utils       import get_initial_conditions, spherical_to_cartesian_fast
from simulation.cuda_geodesic import CUDASchwarzschildIntegrator
from simulation.raytracing  import run_manual_simulation

# ------------------------------------------------------------
# 1.  CLI options (only the essentials)
# ------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument('--cuda', action='store_true', help='use GPU integrator')
p.add_argument('--size',  type=int, default=500)
p.add_argument('--fov',   type=float, default=90)
p.add_argument('--steps', type=int, default=30_000)
p.add_argument('--delta', type=float, default=0.05)
p.add_argument('--omega', type=float, default=0.001)
args = p.parse_args()

# ------------------------------------------------------------
# 2.  Scene parameters (identical to the earlier run)
# ------------------------------------------------------------
BH_MASS          = 1.0
OBS_X            = 20.0
BOUNDARY_RADIUS  = 21.0
PATCH_THETA_DEG  = 126
PATCH_PHI_DEG    = 224
BG_IMG_PATH      = 'images/backgrounds/milky-way-background.jpeg'

image_size = (args.size, args.size)
fov_rad    = np.radians(args.fov)

bh       = BlackHole(mass=BH_MASS)
observer = Observer(position=np.array([OBS_X, 0, 0]),
                    fov=fov_rad,
                    image_size=image_size)

# ------------------------------------------------------------
# 3.  Produce the curved image via your existing pipeline
# ------------------------------------------------------------
img = run_manual_simulation(
        bh, observer,
        steps=args.steps, delta=args.delta, omega=args.omega,
        background_path=BG_IMG_PATH,
        use_cuda=args.cuda,
        boundary_radius=BOUNDARY_RADIUS,
        patch_center_theta=np.pi/2,
        patch_center_phi=np.pi,
        patch_size_theta=np.deg2rad(PATCH_THETA_DEG),
        patch_size_phi=np.deg2rad(PATCH_PHI_DEG),
        flip_theta=True, flip_phi=True,
        n_samples=0               # we’ll add our own samples next
)

Path('images').mkdir(exist_ok=True, parents=True)
plt.imsave('images/theta_band_image.png', img)
print('✔ wrote images/theta_band_image.png')

# ------------------------------------------------------------
# 4.  Build 50 custom rays in the phi-band and integrate them
# ------------------------------------------------------------
N_RAYS   = 50
phi_lo = np.pi - np.deg2rad(10)
phi_hi = np.pi + np.deg2rad(10)
theta_vals = np.linspace(0, np.pi, N_RAYS, endpoint=False)

q0s, p0s = [], []
for theta in theta_vals:
    phi = np.random.uniform(phi_lo, phi_hi)
    # inward unit vector expressed at the observer position
    dir_vec = np.array([
        -np.sin(theta)*np.cos(phi),
         np.sin(theta)*np.sin(phi),
         np.cos(theta)
    ])
    pixel_pos = observer.position + dir_vec   # fictitious screen pixel
    q0, p0, _ = get_initial_conditions(observer.position, pixel_pos, mass_bh=BH_MASS)
    q0s.append(q0)
    p0s.append(p0)

q0s = np.stack(q0s)
p0s = np.stack(p0s)

integ = CUDASchwarzschildIntegrator(steps=args.steps,
                                    delta=args.delta,
                                    mass=BH_MASS,
                                    r_max=BOUNDARY_RADIUS)
traj = integ.integrate_batch_full(q0s, p0s)      # shape (50, steps, 4)

# ------------------------------------------------------------
# 5.  3-D plot with ≤ 500 points per trajectory
# ------------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(111, projection='3d')

# event horizon
rs = 2*BH_MASS
phi, theta = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
xs = rs*np.sin(theta)*np.cos(phi)
ys = rs*np.sin(theta)*np.sin(phi)
zs = rs*np.cos(theta)
ax.plot_surface(xs, ys, zs, color='black', alpha=1.0)
ax.plot_wireframe(xs, ys, zs, color='yellow', linewidth=0.3)

# observer
ax.scatter([OBS_X], [0], [0], s=60, color='red')

# trajectories
for k in range(N_RAYS):
    t, r, th, ph = traj[k].T
    # decimate to ≤ 500 points
    idx = np.linspace(0, len(r)-1, min(500, len(r)), dtype=int)
    r, th, ph = r[idx], th[idx], ph[idx]
    _, xx, yy, zz = spherical_to_cartesian_fast(0.0, r, th, ph)
    ax.plot(xx, yy, zz, lw=0.8, color='orange')

ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.set_title('θ-band (π ± 10°) – 50 null geodesics')
lim = BOUNDARY_RADIUS * 1.1
for axis in 'xyz':
    getattr(ax, f'set_{axis}lim')([-lim, lim])

plt.tight_layout()
plt.savefig('images/theta_band_trajectories.png', dpi=200)
plt.show()

print('✔ wrote images/theta_band_trajectories.png')
