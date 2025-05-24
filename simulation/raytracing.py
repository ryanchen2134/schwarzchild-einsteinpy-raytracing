import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import sys
import logging
from tqdm import tqdm
from .blackhole import BlackHole, Observer
from PIL import Image
from simulation.cuda_geodesic import CUDASchwarzschildIntegrator, compute_null_4momentum_schwarzschild
import pandas as pd
from simulation.utils import get_initial_conditions

def cartesian_to_spherical_momentum(ray_dir, obs_pos):
    x, y, z = obs_pos
    r = np.linalg.norm([x, y, z])
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    e_r = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    e_theta = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)])
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0])
    pr = np.dot(ray_dir, e_r)
    ptheta = np.dot(ray_dir, e_theta)
    pphi = np.dot(ray_dir, e_phi)
    return pr, ptheta, pphi

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
    ray_dir = pixel_pos - observer_pos
    ray_dir = ray_dir / np.linalg.norm(ray_dir)
    x, y, z = observer_pos
    r = np.linalg.norm([x, y, z])
    th = np.arccos(z / r)
    ph = np.arctan2(y, x)
    pr, ptheta, pphi = cartesian_to_spherical_momentum(ray_dir, observer_pos)
    momentum = np.array([0.0, pr, ptheta, pphi])
    geod = Nulllike(
        metric="Schwarzschild",
        metric_params=(),
        position=[r, th, ph],
        momentum=momentum[1:],
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
        if r_bh <= bh_rs:
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
    patch_center_theta=np.pi/2, patch_center_phi=0, patch_size_theta=np.deg2rad(10), patch_size_phi=np.deg2rad(10),
    flip_theta=False, flip_phi=False
):
    """
    Run manual ray tracing simulation with tqdm progress bar.
    Uses ProcessPoolExecutor for true parallelism or CUDA for GPU parallelism.
    Returns the final image as a numpy array.
    Only maps the background image if the boundary hit is within the specified patch (center, size, flip options).
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

    # Image plane geometry (same as flat version)
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

    # Precompute pixel positions on image plane
    pixel_positions = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            dx = (j + 0.5) / w - 0.5
            dy = (i + 0.5) / h - 0.5
            pixel_positions[i, j] = plane_center + dx * width * right + dy * height * up_vec

    if use_cuda:
        logging.info("Using CUDA Schwarzschild integrator for ray tracing...")
        rel_pos = observer.position - bh.position
        r = np.linalg.norm(rel_pos)
        th = np.arccos(rel_pos[2] / r)
        ph0 = np.arctan2(rel_pos[1], rel_pos[0])
        q0s = np.zeros((h*w, 4), dtype=np.float64)
        p0s = np.zeros((h*w, 4), dtype=np.float64)
        for idx, (i, j) in enumerate([(i, j) for i in range(h) for j in range(w)]):
            q0, p0 = get_initial_conditions(obs_pos, pixel_positions[i, j])
            q0s[i*w+j, :] = q0
            p0s[i*w+j, :] = p0
        cuda_integrator = CUDASchwarzschildIntegrator(steps=steps, delta=delta, mass=bh.mass)
        out_qs, out_ps = cuda_integrator.integrate_batch(q0s, p0s)
        photon_rows = []
        for idx, (i, j) in tqdm(list(enumerate([(i, j) for i in range(h) for j in range(w)])), desc="CUDA Ray tracing", unit="ray"):
            x, y, z = out_qs[idx, 1], out_qs[idx, 2], out_qs[idx, 3]  # r, theta, phi
            r_bh = x  # r
            q0 = q0s[i*w+j]
            p0 = p0s[i*w+j]
            collision = ''
            bg_u = bg_v = rgb = None

            # Category 1: captured by BH -> Black
            if r_bh <= bh.rs:
                value = (0, 0, 0)
                collision = 'bh'

            # Category 2: Escaped the domain (r >= r_shell)
            elif r_bh >= 10 * bh.rs:
                if bg_array is not None:
                    hit_theta = out_qs[idx, 2]
                    hit_phi = out_qs[idx, 3]
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
                        value = tuple(bg_array[u, v])
                        bg_u, bg_v = u, v
                        rgb = value
                        collision = 'escape_bg'
                    else:
                        value = (0, 0, 255)  # Blue
                        collision = 'escape_no_patch'
                else:
                    value = (0, 0, 255)  # Blue even without background
                    collision = 'escape_no_patch'

            # Category 3: Still inside domain after integration steps -> Red
            else:
                value = (255, 0, 0)
                collision = 'in_domain'
            img[i, j] = value
            photon_rows.append({
                'i': i, 'j': j,
                'q0_t': q0[0], 'q0_r': q0[1], 'q0_th': q0[2], 'q0_ph': q0[3],
                'p0_t': p0[0], 'p0_r': p0[1], 'p0_th': p0[2], 'p0_ph': p0[3],
                'final_r': x, 'final_th': y, 'final_ph': z,
                'collision': collision,
                'bg_u': bg_u, 'bg_v': bg_v,
                'rgb': rgb
            })
        df = pd.DataFrame(photon_rows)
        df.to_csv('photon_data.csv', index=False)
        print(f"Saved photon data for {len(df)} photons to photon_data.csv")
        logging.info("CUDA ray tracing simulation completed.")
    else:
        # CPU fallback (original code)
        args_list = [
            (
                bh.rs, bh.position, observer.position, 10 * bh.rs, pixel_positions[i, j], (i, j),
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
    return img 