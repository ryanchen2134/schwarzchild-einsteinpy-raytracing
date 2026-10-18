#main.py
import argparse
import logging
from config import parse_args
from simulation.blackhole import BlackHole, Observer
from simulation.background import save_no_gravity_image_with_background
from simulation.raytracing import run_manual_simulation
from visualization.plot import plot_scene_topdown, plot_scene_embedding_3d, plot_scene_closeup_3d
from simulation.utils import _apply_relative_offsets
from simulation.disk import Disk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---
# GEOMETRIZED UNITS: G = c = 1
# All quantities (mass, length, time) are in the same units (e.g., meters or M)
# Schwarzschild radius: r_s = 2M
# ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def build_disk(args, mass_bh, observer_x):
    """The simulation.disk.Disk the flags describe, or None without --disk. Invalid
    values end the run with a message naming the flag."""
    if not args.disk:
        return None
    if not (np.isfinite(args.disk_inclination) and 0.0 <= args.disk_inclination <= 180.0):
        raise SystemExit(f"--disk-inclination must be in [0, 180] degrees, got {args.disk_inclination}")
    if not np.isfinite(args.disk_azimuth):
        raise SystemExit(f"--disk-azimuth must be finite, got {args.disk_azimuth}")
    normal = Disk.normal_from_angles(np.radians(args.disk_inclination), np.radians(args.disk_azimuth))
    r_in = args.disk_inner if args.disk_inner is not None else 6.0 * mass_bh
    r_out = args.disk_outer if args.disk_outer is not None else 20.0 * mass_bh
    try:
        return Disk(normal, r_in, r_out, args.disk_temperature, mass_bh, observer_x)
    except ValueError as e:
        raise SystemExit(f"--disk-inner / --disk-outer / --disk-temperature: {e}")


def main():
    args = parse_args()
    image_size = (args.size, args.size)
    fov_rad = np.radians(args.fov)
    mass_bh = args.bh_mass
    boundary_radius = args.boundary_radius
    observer_x = args.observer_distance
    if not mass_bh > 0:
        raise SystemExit(f"--bh-mass must be positive, got {mass_bh}")
    if not observer_x > 3.0 * mass_bh:
        raise SystemExit(f"--observer-distance ({observer_x}) must exceed the photon sphere, 3 M = {3.0 * mass_bh}")
    if not boundary_radius > observer_x:
        raise SystemExit(f"--boundary-radius ({boundary_radius}) must exceed --observer-distance ({observer_x})")
    if not boundary_radius < 100.0 * mass_bh:
        raise SystemExit(f"--boundary-radius ({boundary_radius}) must stay below the numerical-error radius 100 M = {100.0 * mass_bh}")
    for name, v in (('--yaw', args.yaw), ('--pitch', args.pitch), ('--roll', args.roll)):
        if not np.isfinite(v):
            raise SystemExit(f"{name} must be a finite angle in degrees, got {v}")
    yaw_rad, pitch_rad, roll_rad = np.radians(args.yaw), np.radians(args.pitch), np.radians(args.roll)
    disk = build_disk(args, mass_bh, observer_x)
    bh = BlackHole(mass=mass_bh)
    observer_pos = np.array([observer_x, 0, 0])
    observer = Observer(position=observer_pos, fov=fov_rad, image_size=image_size)
    
    
    bg_patch_center_theta_deg = args.bg_patch_center_theta
    bg_patch_center_phi_deg = args.bg_patch_center_phi
    
    bg_patch_center_theta_relobs_deg = args.bg_patch_center_theta_relobs
    bg_patch_center_phi_relobs_deg = args.bg_patch_center_phi_relobs
    
    patch_center_theta_final_rad, patch_center_phi_final_rad = _apply_relative_offsets(bg_patch_center_theta_deg, bg_patch_center_phi_deg, bg_patch_center_theta_relobs_deg, bg_patch_center_phi_relobs_deg)
    
    
    
    logging.info("Saving no-gravity image using background...")
    if not args.no_flat_trajectories:
        sampled_trajectories = save_no_gravity_image_with_background(
            observer, args.background, 'images/no_gravity.png',
            boundary_radius=boundary_radius,
            patch_center_theta=patch_center_theta_final_rad,
            patch_center_phi=patch_center_phi_final_rad,
            patch_size_theta=np.deg2rad(args.bg_patch_size_theta),
            patch_size_phi=np.deg2rad(args.bg_patch_size_phi),
            flip_theta=args.bg_flip_theta,
            flip_phi=args.bg_flip_phi,
            return_sampled_trajectories=True,
            n_sampled=10,
            override_patch_center=False,
            yaw=yaw_rad, pitch=pitch_rad, roll=roll_rad,
            disk=disk
        )
        flat_trajectories = sampled_trajectories
    else:
        flat_trajectories = None

    # --- Sampled relativistic (curved) trajectories for orange rays ---
    photon_trajectories = None
    img = None
    # Always run the manual simulation path
    result = run_manual_simulation(
        bh, observer,
        real_time=False,  # Do not display real-time
        update_every=max(1, (image_size[0]*image_size[1])//32),
        steps=args.steps,
        delta=args.delta,
        omega=args.omega,
        suppress_warnings=args.suppress_warnings,
        background_path=args.background,
        use_cuda=args.cuda,
        boundary_radius=boundary_radius,
        patch_center_theta=patch_center_theta_final_rad,
        patch_center_phi=patch_center_phi_final_rad,
        patch_size_theta=np.deg2rad(args.bg_patch_size_theta),
        patch_size_phi=np.deg2rad(args.bg_patch_size_phi),
        flip_theta=args.bg_flip_theta,
        flip_phi=args.bg_flip_phi,
        n_samples=20,
        yaw=yaw_rad, pitch=pitch_rad, roll=roll_rad,
        disk=disk
    )
    if isinstance(result, tuple):
        img, photon_trajectories = result
    else:
        img = result
        photon_trajectories = None
    plt.imshow(img, origin='lower')
    plt.title('Black Hole Ray Tracing (Manual)')
    # plt.axis('off')
    plt.show()
    plt.imsave('images/manual_output.png', img)
    logging.info("Saved manual_output.png")

    # already saved in raytracing.py

    # --- 3D plotting at the very end ---
    logging.info("Saving 3D embedding scene view...")
    
    #before plotting photon trajectories, clear all entries that have x,y,z=0
    if photon_trajectories is not None:
        # Filter out any trajectory points that are all zeros
        filtered_trajectories = []
        for traj in photon_trajectories:
            filtered_traj = traj[~np.all(traj == 0, axis=1)]
            if len(filtered_traj) > 0:  # Only keep trajectories that have points remaining
                filtered_trajectories.append(filtered_traj)
        photon_trajectories = filtered_trajectories
        print(f"Filtered {len(photon_trajectories)} trajectories")
        
    logging.info("Saving top-down scene view...")
    plot_scene_topdown(
        bh, observer, image_size, boundary_radius=boundary_radius,
        out_path='images/scene_topdown.png', fov_deg=args.fov,
        patch_center_theta=patch_center_theta_final_rad,
        # patch_center_phi=patch_center_phi_final_rad,
        patch_size_theta=np.deg2rad(args.bg_patch_size_theta),
        patch_size_phi=np.deg2rad(args.bg_patch_size_phi),
        photon_trajectories=photon_trajectories
    )
        
    logging.info("Saving close-up 3D scene view...")
    plot_scene_closeup_3d(
        bh, observer, image_size, out_path='images/scene_closeup_3d.png', fov_deg=args.fov, photon_trajectories=photon_trajectories
    )
    plot_scene_embedding_3d(
        bh, observer, image_size, boundary_radius=boundary_radius,
        out_path='images/scene_topdown_3d.png', fov_deg=args.fov,
        photon_trajectories=photon_trajectories,
        flat_trajectories=flat_trajectories,
        patch_center_theta=patch_center_theta_final_rad,
        patch_center_phi=patch_center_phi_final_rad,
        patch_size_theta=np.deg2rad(args.bg_patch_size_theta),
        patch_size_phi=np.deg2rad(args.bg_patch_size_phi),
        override_patch_center=False
    )

    # --- Print photon summary from photon_data.csv ---
    try:
        df = pd.read_csv('photon_data.csv')
        n_captured = (df['collision'] == 'bh').sum()
        n_in_domain = (df['collision'] == 'in_domain').sum()
        n_escaped = ((df['collision'] == 'escape_no_patch') | (df['collision'] == 'escape_bg')).sum()
        n_bg = (df['collision'] == 'escape_bg').sum()
        n_disk = (df['collision'] == 'disk').sum()
        n_err = (df['collision'] == 'numerical error').sum()
        print(f"\nPhoton summary:")
        print(f"  Captured by BH: {n_captured}")
        print(f"  Still in domain: {n_in_domain}")
        print(f"  Escaped: {n_escaped}")
        print(f"  Hit background: {n_bg}")
        print(f"  Hit disk: {n_disk}")
        print(f"  Numerical error: {n_err}")
    except Exception as e:
        print(f"Could not read photon_data.csv for summary: {e}")

if __name__ == "__main__":
    main()