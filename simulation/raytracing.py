# raytracing.py
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import sys
import logging
from tqdm import tqdm
from .blackhole import BlackHole, Observer
from PIL import Image
import pandas as pd
from simulation.utils import get_initial_conditions
import warnings  # CUDA curved geodesic path removed


def emit_photon_worker(args):
    """
    Worker function for manual ray tracing. Returns (i, j), value for the image.
    """
    from einsteinpy.geodesic import Nulllike
    (
        bh_rs, bh_pos, observer_pos, r_shell, pixel_pos, mesh_idx,
        steps, delta, omega, rtol, atol, order, suppress_warnings, bg_array,
        patch_center_theta, patch_center_phi, patch_size_theta, patch_size_phi, flip_theta, flip_phi
    ) = args
    # Use get_initial_conditions for robust direction-to-momentum conversion
    mass_bh = bh_rs / 2.0
    q0, p0 = get_initial_conditions(observer_pos, pixel_pos, mass_bh=mass_bh)
    geod = Nulllike(
        metric="Schwarzschild",
        metric_params=(),
        position=q0[1:],
        momentum=p0[1:],
        steps=steps,
        delta=delta,
        omega=omega,
        rtol=rtol,
        atol=atol,
        order=order,
        suppress_warnings=suppress_warnings,
        return_cartesian=True
    )
    traj = geod.trajectory[1]
    prev_r_bh = None
    approaching = True  # True while radius is still decreasing towards minimum
    for step in traj:
        x, y, z = step[1:4]
        r_bh = np.linalg.norm([x, y, z])

        # Determine if we've reached the point of closest approach
        if prev_r_bh is not None:
            if approaching and r_bh > prev_r_bh + 1e-9:
                # Radius has started increasing; we are now on the outbound leg
                approaching = False

        # Only look for boundary crossing once we are outbound
        crossed_boundary = False
        if (not approaching) and (prev_r_bh is not None):
            crossed_boundary = (prev_r_bh < r_shell) and (r_bh >= r_shell)

        prev_r_bh = r_bh

        # 1. Photon captured by BH -> Black and terminate
        if r_bh <= bh_rs*1.1:
            return mesh_idx, (0, 0, 0)

        # 2. Photon has crossed the outer boundary (r_shell) -> may hit background or be blue
        if crossed_boundary:
            # If a background image is supplied we map only if inside the user patch
            if bg_array is not None:
                h, w, _ = bg_array.shape
                hit_theta = np.arccos(z / r_bh)
                hit_phi = np.arctan2(y, x)
                dtheta = np.abs(hit_theta - patch_center_theta)
                dphi = np.abs((hit_phi - patch_center_phi + np.pi) % (2*np.pi) - np.pi)

                inside_patch = (dtheta <= patch_size_theta/2) and (dphi <= patch_size_phi/2)
                if inside_patch:
                    theta_map = (np.pi - hit_theta) if flip_theta else hit_theta
                    phi_map = (-hit_phi) if flip_phi else hit_phi
                    u = int((theta_map / np.pi) * (h - 1))
                    v = int(((phi_map + np.pi) / (2 * np.pi)) * (w - 1))
                    u = min(max(u, 0), h - 1)
                    v = min(max(v, 0), w - 1)
                    bg_color = tuple(bg_array[u, v])
                    return mesh_idx, bg_color  # Terminate with BG pixel
                else:
                    return mesh_idx, (0, 0, 255)  # Blue – escaped but outside patch
            # No background supplied -> still mark as blue to indicate escape outside patch
            return mesh_idx, (0, 0, 255)

    # 3. Photon still inside domain after all steps -> mark Red
    return mesh_idx, (255, 0, 0)

def run_manual_simulation(
    bh, observer, real_time=False, update_every=32,
    steps=500, delta=0.2, omega=1.0, rtol=1e-2, atol=1e-2, order=2, suppress_warnings=False,
    background_path=None, use_cuda=False,
    boundary_radius=None,
    patch_center_theta=np.pi/2, patch_center_phi=0, patch_size_theta=np.deg2rad(10), patch_size_phi=np.deg2rad(10),
    flip_theta=False, flip_phi=False,
    n_samples=0
):
    """
    Run manual ray tracing simulation with tqdm progress bar.
    Uses ProcessPoolExecutor for true parallelism or CUDA for GPU parallelism.
    Returns the final image as a numpy array.
    Only maps the background image if the boundary hit is within the specified patch (center, size, flip options).
    If return_sampled_trajectories is True, also returns a list of sampled photon trajectories (in Cartesian coordinates).
    """
    logging.info("Starting manual ray tracing simulation...")
    h, w = observer.image_size
    # Always work with RGB image so that diagnostic colours (red, blue) are visible
    if background_path is not None:
        bg_img = Image.open(background_path).convert('RGB').resize((w, h), Image.LANCZOS)
        bg_array = np.array(bg_img)
    else:
        bg_array = None
    img = np.zeros((h, w, 3), dtype=np.uint8)
    sampled_trajectories = []
    sampled_indices = set()
    if n_samples > 0:
        import random
        while len(sampled_indices) < n_samples:
            i = random.randint(0, h-1)
            j = random.randint(0, w-1)
            sampled_indices.add((i, j))

    # Image plane geometry (robust pinhole camera model)
    
    # get position of observer and black hole
    obs_pos = np.array(observer.position)
    bh_pos = np.array(bh.position) if hasattr(bh, 'position') else np.zeros(3)
    #define the optical axis (should be in the (negative) x direction)
    optical_axis = (bh_pos - obs_pos)
    optical_axis = optical_axis / np.linalg.norm(optical_axis)
    #define the up vector (related to the pixels of the image plane) (should be in the z direction)
    up_guess = np.array([0, 0, 1])
    if np.allclose(np.cross(optical_axis, up_guess), 0):
        up_guess = np.array([0, 1, 0])
    right = np.cross(up_guess, optical_axis)
    right = right / np.linalg.norm(right)
    up_vec = np.cross(optical_axis, right)
    up_vec = up_vec / np.linalg.norm(up_vec)
    fov = observer.fov
    h, w = observer.image_size
    plane_dist = 0.2 * np.linalg.norm(obs_pos)
    plane_center = obs_pos + optical_axis * plane_dist
    plane_width = 2 * plane_dist * np.tan(fov/2)
    plane_height = plane_width * (h/w)
    pixel_positions = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            u = (j + 0.5) / w - 0.5
            v = (i + 0.5) / h - 0.5
            pixel_pos = plane_center + u * plane_width * right + v * plane_height * up_vec
            pixel_positions[i, j] = pixel_pos

    # if boundary_radius is None:
    #     boundary_radius = 10 * bh.rs

    if use_cuda:
        logging.info("Using CUDA Schwarzschild integrator for curved rays ...")
        from simulation.cuda_geodesic import CUDASchwarzschildIntegrator

        # Prepare initial conditions for all rays
        q0s = np.zeros((h*w, 4), dtype=np.float64)
        p0s = np.zeros((h*w, 4), dtype=np.float64)
        for idx, (i, j) in tqdm(list(enumerate([(i, j) for i in range(h) for j in range(w)])), desc="CUDA post-processing", unit="ray"):
            q0, p0 = get_initial_conditions(obs_pos, pixel_positions[i, j], mass_bh=bh.mass)
            q0s[idx] = q0
            p0s[idx] = p0

        cuda_integrator = CUDASchwarzschildIntegrator(steps=steps, delta=delta, mass=bh.mass, r_max=boundary_radius)
        out_qs, _ = cuda_integrator.integrate_batch(q0s, p0s)

        # Handle sampled trajectories
        sampled_traj_data = []
        if n_samples > 0 and len(sampled_indices) > 0:
            sample_flat_idx = np.array([i*w + j for (i, j) in sampled_indices], dtype=np.int64)
            q0s_samples = q0s[sample_flat_idx]
            p0s_samples = p0s[sample_flat_idx]
            traj_out = cuda_integrator.integrate_batch_full(q0s_samples, p0s_samples)
            # Convert to Cartesian for each sample
            for s in range(len(sample_flat_idx)):
                traj_cart = []
                for step in range(traj_out.shape[1]):
                    t, r, th, ph = traj_out[s, step]
                    x = r * np.sin(th) * np.cos(ph)
                    y = r * np.sin(th) * np.sin(ph)
                    z = r * np.cos(th)
                    traj_cart.append([x, y, z])
                sampled_traj_data.append(np.array(traj_cart))

        # Build image colours
        photon_rows = []
        for idx, (i, j) in tqdm(list(enumerate([(i, j) for i in range(h) for j in range(w)])), desc="CUDA post-processing", unit="ray"):
            r_bh = out_qs[idx, 1]
            th_hit = out_qs[idx, 2]
            ph_hit = out_qs[idx, 3]

            collision = ''
            bg_u = bg_v = rgb = None
            # photon getting very close or captured, or heading vector (dot) the inward radial vector is less than or
            #equal to the cosine (projection) of the angle generated (arctan) by the current radial distance and the schwarzschild radius
            # or (np.dot(np.array[-1,0,0], out_qs[idx, 1:4]) <= np.cos(np.arctan(bh.rs/2/np.linalg.norm(r_bh)))):
            if r_bh <= bh.rs*1.1 :
                value = (0, 0, 0)
                collision = 'bh'
            elif r_bh >= boundary_radius:
                if bg_array is not None:
                    dtheta = np.abs(th_hit - patch_center_theta)
                    dphi = np.abs((ph_hit - patch_center_phi + np.pi) % (2*np.pi) - np.pi)
                    inside_patch_angle = (dtheta <= patch_size_theta/2) and (dphi <= patch_size_phi/2)
                    # pi/2 <= phi <= 3pi/2 
                    if inside_patch_angle and (np.pi/2 <= ph_hit <= 3*np.pi/2):
                        theta_map = (np.pi - th_hit) if flip_theta else th_hit
                        phi_map = (-ph_hit) if flip_phi else ph_hit
                        u = int((theta_map / np.pi) * (h - 1))
                        v = int(((phi_map + np.pi) / (2 * np.pi)) * (w - 1))
                        u = min(max(u, 0), h - 1)
                        v = min(max(v, 0), w - 1)
                        value = tuple(bg_array[u, v])
                        bg_u, bg_v = u, v
                        rgb = value
                        collision = 'escape_bg'
                    else:
                        value = (0, 0, 255)
                        collision = 'escape_no_patch'
                else:
                    value = (0, 0, 255)
                    collision = 'escape_no_patch'
            else:
                value = (255, 0, 0)
                collision = 'in_domain'
            img[i, j] = value
            photon_rows.append({'i': i, 'j': j, 'final_r': r_bh, 'final_th': th_hit, 'final_ph': ph_hit, 'collision': collision})

        pd.DataFrame(photon_rows).to_csv('photon_data.csv', index=False)

        # Save sampled trajectories CSV
        if n_samples > 0 and sampled_traj_data:
            rows = []
            for ridx, traj in enumerate(sampled_traj_data):
                for pidx, (px, py, pz) in enumerate(traj):
                    rows.append({'ray_id': ridx, 'point_idx': pidx, 'x': px, 'y': py, 'z': pz})
            pd.DataFrame(rows).to_csv('sampled_rays.csv', index=False)
        sampled_trajectories = sampled_traj_data
    else:
        # CPU fallback (original code)
        args_list = [
            (
                bh.rs, bh.position, observer.position, boundary_radius, pixel_positions[i, j], (i, j),
                steps, delta, omega, rtol, atol, order, suppress_warnings, bg_array,
                patch_center_theta, patch_center_phi, patch_size_theta, patch_size_phi, flip_theta, flip_phi
            )
            for i in range(h) for j in range(w)
        ]
        try:
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(emit_photon_worker, args) for args in args_list]
                with tqdm(total=len(futures), desc="Ray tracing", unit="ray") as pbar:
                    for future in as_completed(futures):
                        (i, j), value = future.result()
                        img[i, j] = value
                        pbar.update(1)
            logging.info("Manual ray tracing simulation completed.")
        except KeyboardInterrupt:
            logging.warning("Simulation interrupted by user. Exiting gracefully.")
            print("\nSimulation interrupted by user. Exiting gracefully.")
            sys.exit(0)
    # Save the image to disk
    plt.imsave('images/manual_output.png', img)
    logging.info("Saved manual_output.png")

    # Count summary
    if 'photon_rows' in locals():
        n_captured = sum(1 for row in photon_rows if row['collision'] == 'bh')
        n_escaped = sum(1 for row in photon_rows if row['collision'] in ('escape_no_patch', 'escape_bg'))
        n_bg = sum(1 for row in photon_rows if row['collision'] == 'escape_bg')
        print(f"Summary: {n_captured} rays captured by BH, {n_escaped} rays escaped, {n_bg} rays hit the background image.")
    else:
        print("Summary counts not available (CPU fallback mode).")
    if n_samples > 0:
        return img, sampled_trajectories
    return img 