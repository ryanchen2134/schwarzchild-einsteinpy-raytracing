import argparse
import logging
from config import parse_args
from simulation.blackhole import BlackHole, Observer
from simulation.background import save_no_gravity_image_with_background
from simulation.raytracing import run_manual_simulation
from simulation.shadow import run_shadow_simulation, SHADOW_AVAILABLE
from visualization.plot import plot_scene_topdown, plot_scene_embedding_3d, plot_scene_closeup_3d
import matplotlib.pyplot as plt
import numpy as np

# ---
# GEOMETRIZED UNITS: G = c = 1
# All quantities (mass, length, time) are in the same units (e.g., meters or M)
# Schwarzschild radius: r_s = 2M
# ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def main():
    args = parse_args()
    image_size = (args.size, args.size)
    fov_rad = np.radians(args.fov)
    mass_bh = args.bh_mass
    boundary_radius = args.boundary_radius if args.boundary_radius is not None else 10 * mass_bh
    observer_x = args.observer_distance if args.observer_distance is not None else 20 * mass_bh
    bh = BlackHole(mass=mass_bh)
    observer_pos = np.array([observer_x, 0, 0])
    observer = Observer(position=observer_pos, fov=fov_rad, image_size=image_size)

    logging.info("Saving top-down scene view...")
    plot_scene_topdown(
        bh, observer, image_size, boundary_radius=boundary_radius,
        out_path='images/scene_topdown.png', fov_deg=args.fov,
        patch_center_theta=np.deg2rad(args.bg_patch_center_theta),
        patch_size_theta=np.deg2rad(args.bg_patch_size_theta),
        patch_size_phi=np.deg2rad(args.bg_patch_size_phi)
    )
    
    logging.info("Saving close-up 3D scene view...")
    plot_scene_closeup_3d(
        bh, observer, image_size, out_path='images/scene_closeup_3d.png', fov_deg=args.fov
    )
    
    logging.info("Saving no-gravity image using background...")
    sampled_trajectories = save_no_gravity_image_with_background(
        observer, args.background, 'images/no_gravity.png',
        boundary_radius=boundary_radius,
        patch_size_theta=np.deg2rad(args.bg_patch_size_theta),
        patch_size_phi=np.deg2rad(args.bg_patch_size_phi),
        flip_theta=args.bg_flip_theta,
        flip_phi=args.bg_flip_phi,
        return_sampled_trajectories=True,
        n_sampled=10,
        override_patch_center=False
    )

    # --- Sample 20 random photon trajectories from CUDA simulation for 3D plot ---
    photon_trajectories = None
    if args.cuda:
        h, w = image_size
        # Reconstruct pixel positions as in run_manual_simulation
        obs_pos = np.array(observer.position)
        obs_r = np.linalg.norm(obs_pos)
        fov = observer.fov
        plane_dist = 0.2 * obs_r
        up = np.array([0, 0, 1])
        if np.allclose(np.cross(obs_pos, up), 0):
            up = np.array([0, 1, 0])
        right = np.cross(up, obs_pos)
        right = right / np.linalg.norm(right)
        up_vec = np.cross(obs_pos, right)
        up_vec = up_vec / np.linalg.norm(up_vec)
        width = 2 * plane_dist * np.tan(fov/2)
        height = width * (h/w)
        plane_center = obs_pos - (obs_pos/obs_r) * plane_dist
        pixel_positions = np.zeros((h, w, 3))
        for i in range(h):
            for j in range(w):
                dx = (j + 0.5) / w - 0.5
                dy = (i + 0.5) / h - 0.5
                pixel_positions[i, j] = plane_center + dx * width * right + dy * height * up_vec
        # Randomly sample 20 pixels
        import random
        sampled_pixels = random.sample([(i, j) for i in range(h) for j in range(w)], 20)
        from simulation.utils import get_initial_conditions
        q0s = []
        p0s = []
        for (i, j) in sampled_pixels:
            q0, p0 = get_initial_conditions(obs_pos, pixel_positions[i, j])
            q0s.append(q0)
            p0s.append(p0)
        q0s = np.stack(q0s)
        p0s = np.stack(p0s)
        from simulation.cuda_geodesic import CUDASchwarzschildIntegrator
        cuda_integrator = CUDASchwarzschildIntegrator(steps=args.steps, delta=args.delta, mass=bh.mass)
        out_qs_traj, _ = cuda_integrator.integrate_batch_full_trajectory(q0s, p0s)
        # Convert (steps, 4) -> (steps, 3) in Cartesian for each ray
        photon_trajectories = []
        for ray_traj in out_qs_traj:
            r = ray_traj[:, 1]
            th = ray_traj[:, 2]
            ph = ray_traj[:, 3]
            xs = r * np.sin(th) * np.cos(ph)
            ys = r * np.sin(th) * np.sin(ph)
            zs = r * np.cos(th)
            photon_trajectories.append(np.stack([xs, ys, zs], axis=1))

    logging.info("Saving 3D embedding scene view...")
    plot_scene_embedding_3d(
        bh, observer, image_size, boundary_radius=boundary_radius,
        out_path='images/scene_topdown_3d.png', fov_deg=args.fov,
        photon_trajectories=photon_trajectories,
        patch_size_theta=np.deg2rad(args.bg_patch_size_theta),
        patch_size_phi=np.deg2rad(args.bg_patch_size_phi),
        override_patch_center=False
    )
    
    
    # 3. Run the simulation
    if args.mode == 'shadow' and SHADOW_AVAILABLE:
        logging.info("Using Shadow class for fast simulation...")
        run_shadow_simulation(mass_bh, fov_rad, image_size)
    else:
        logging.info("Using manual simulation (multi-threaded, real-time updates)...")
        img = run_manual_simulation(
            bh, observer,
            real_time=False,  # Do not display real-time
            update_every=max(1, (image_size[0]*image_size[1])//32),
            steps=args.steps,
            delta=args.delta,
            omega=args.omega,
            rtol=args.rtol,
            atol=args.atol,
            order=args.order,
            suppress_warnings=args.suppress_warnings,
            background_path=args.background,
            use_cuda=args.cuda,
            patch_center_theta=np.deg2rad(args.bg_patch_center_theta),
            patch_center_phi=np.deg2rad(args.bg_patch_center_phi),
            patch_size_theta=np.deg2rad(args.bg_patch_size_theta),
            patch_size_phi=np.deg2rad(args.bg_patch_size_phi),
            flip_theta=args.bg_flip_theta,
            flip_phi=args.bg_flip_phi
        )
        plt.imshow(img, origin='lower')
        plt.title('Black Hole Ray Tracing (Manual)')
        plt.axis('off')
        plt.show()
        plt.imsave('images/manual_output.png', img)
        logging.info("Saved manual_output.png")

if __name__ == "__main__":
    main()