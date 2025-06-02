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
from simulation.utils import spherical_to_cartesian_fast, cartesian_to_spherical_fast


def run_manual_simulation(
    bh, observer, real_time=False, update_every=32,
    steps=500, delta=0.2, omega=1.0, rtol=1e-2, atol=1e-2, order=2, suppress_warnings=False,
    background_path=None, use_cuda=False,
    boundary_radius=None,
    patch_center_theta=np.pi/2, patch_center_phi=np.pi, patch_size_theta=np.deg2rad(10), patch_size_phi=np.deg2rad(10),
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
    # bh_pos = np.array(bh.position) if hasattr(bh, 'position') else np.zeros(3)
    # bh_pos = np.array([0, 0, 0])
    
    #define the optical axis (should be in the (negative) x direction)
    # optical_axis = (bh_pos - obs_pos)
    # optical_axis = optical_axis / np.linalg.norm(optical_axis)
    optical_axis = np.array([-1, 0, 0])
    #define the up vector (related to the pixels of the image plane) (should be in the z direction)
    # up_guess = np.array([0, 0, 1]) # is always true since the observer is always looking at the black hole
    # if np.allclose(np.cross(optical_axis, up_guess), 0):  #instead, check if norm of cross is close to 0
    # if np.linalg.norm(np.cross(optical_axis, up_guess)) < 1e-4:
    #     up_guess = np.array([0, 1, 0])
    # right = np.cross(up_guess, optical_axis)
    
    # right = right / np.linalg.norm(right)
    right = np.array([0, 1, 0])
    # up_vec = np.cross(optical_axis, right)
    # up_vec = up_vec / np.linalg.norm(up_vec)
    up_vec = np.array([0, 0, 1])
    
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
        alpha0s = np.zeros((h*w, ), dtype=np.float64) # 1 initial alpha per ray
        #ray headings in coordinates relative to origin. (not sphereical basis elements)
        h_rs = np.zeros((h*w, ), dtype=np.float64)
        h_thetas = np.zeros((h*w, ), dtype=np.float64)
        h_phis = np.zeros((h*w, ), dtype=np.float64)
        # rotaton about the optical axis
        betas = np.zeros((h*w, ), dtype=np.float64)
        #
        for idx, (i, j) in tqdm(list(enumerate([(i, j) for i in range(h) for j in range(w)])), desc="Calculating Trajectories", unit="ray"):
            q0, p0, alpha0, h_r, h_theta, h_phi, beta = get_initial_conditions(obs_pos, pixel_positions[i, j], mass_bh=bh.mass)
            q0s[idx] = q0
            p0s[idx] = p0
            alpha0s[idx] = alpha0
            h_rs[idx] = h_r
            h_thetas[idx] = h_theta
            h_phis[idx] = h_phi
            betas[idx] = beta

        cuda_integrator = CUDASchwarzschildIntegrator(steps=steps, delta=delta, mass=bh.mass, r_max=boundary_radius)
        print("Beginning integration (CUDA)...")
        out_qs, _ = cuda_integrator.integrate_batch(q0s, p0s)
        print("Integration complete (CUDA)")

            # ── parameters ────────────────────────────────────────────────────────
    MAX_POINTS = 1000             # cap per ray                ← you can tweak

    # ── Handle sampled trajectories ───────────────────────────────────────
    sampled_traj_data = []
    if n_samples > 0 and sampled_indices:
        print("Handling sample trajectories (CUDA)...")
        sample_flat_idx = np.array([i*w + j for (i, j) in sampled_indices], dtype=np.int64)
        q0s_samples = q0s[sample_flat_idx]
        p0s_samples = p0s[sample_flat_idx]

        print("Beginning integration of samples (CUDA)...")
        traj_out = cuda_integrator.integrate_batch_full(q0s_samples, p0s_samples)
        print("Integration of samples complete (CUDA)")

        # ── down-sample along the integration axis ────────────────────────
        n_steps_full = traj_out.shape[1]
        # choose evenly-spaced indices, but never more than MAX_POINTS
        keep_idx = np.linspace(0, n_steps_full - 1,
                            num=min(MAX_POINTS, n_steps_full),
                            dtype=np.int32)

        for s in tqdm(range(len(sample_flat_idx)),
                    desc="Converting sampled trajectories to Cartesian (CUDA)",
                    unit="ray"):
            traj_cart = []
            for step in keep_idx:
                t, r, th, ph = traj_out[s, step]
                _, x, y, z = spherical_to_cartesian_fast(_, r, th, ph)
                traj_cart.append((x, y, z))
            sampled_traj_data.append(np.array(traj_cart, dtype=np.float64))


        ############################################                                        ##################
        ## Build image colours ## Build image colours ## Build image colours ## Build image colours ## Build image colours ## Build image colours ## Build image colours 
        print("Building image from trajectory data(CUDA)...")
        
        theta0 = patch_center_theta - patch_size_theta/2
        theta1 = patch_center_theta + patch_size_theta/2
        phi0   = patch_center_phi   - patch_size_phi/2
        # phi1   = patch_center_phi   + patch_size_phi/2
        
        # method a:
        # phi_span = (phi1 - phi0) % (2*np.pi) or 2*np.pi  # handle 2π wrap
        #method b:
        phi_span = patch_size_phi               # = ϕ₁ – ϕ₀ (no modulo!)

        

        photon_rows = []
        for idx, (i, j) in tqdm(list(enumerate([(i, j) for i in range(h) for j in range(w)])), desc="Mapping rays to images (CUDA)", unit="ray"):
            
            r_bh = out_qs[idx, 1]
            th_hit = out_qs[idx, 2]
            ph_hit = out_qs[idx, 3]

            # rotate the hit by the x axis: 
            x, y, z = spherical_to_cartesian_fast(0, r_bh, th_hit, ph_hit)[1:]

            c, s = np.cos(betas[idx]), np.sin(betas[idx])
            R_x = np.array([[1, 0, 0],
                            [0, c, -s],
                            [0, s, c]])
            x, y, z = (R_x @ np.array([x, y, z])).tolist()
            
            th_hit, ph_hit = cartesian_to_spherical_fast(0, x, y, z)[2:]

            # ph_hit = (ph_hit + np.pi) % (2*np.pi) - np.pi
            
            collision = ''
            bg_u = bg_v = rgb = None
            # photon getting very close or captured, or initial trajectory angle is less than or equal to max angle 
            # determined by the observer distance and the schwarzschild radius
            
            # bh_angle = np.arctan(bh.rs/2/obs_pos[0]) #this is the disk radius
            #the shadow radius is related to the impact parameter for plunge photons
            b_crit = 3 * np.sqrt(3) * bh.rs
            bh_angle = np.arcsin(b_crit/obs_pos[0])/2
            
            
            if (r_bh <= bh.rs*1.2) or (alpha0s[idx] <= bh_angle):
                value = (0, 0, 0)
                collision = 'bh'
            elif (r_bh >= 100):
                value = (255, 0, 0)
                collision = 'numerical error'
            elif r_bh >= boundary_radius:
                if bg_array is not None:
                    
                    th_hit = th_hit % (2 * np.pi)  # Ensure theta is in [0, 2π]
                    ph_hit = ph_hit % (2 * np.pi)  # Ensure phi is in [0, 2π]

                    dtheta = np.abs(th_hit - patch_center_theta) 

                    
                    # phi_rel = (ph_hit - phi0)
                    # dphi = np.abs(ph_hit - patch_center_phi)

                    # AFTER   --------------------------------------------------------------
                    phi_rel = (ph_hit - phi0) % (2*np.pi)          # force into 0 … 2π
                    dphi     = np.abs((ph_hit - patch_center_phi + np.pi) % (2*np.pi) - np.pi)

                    inside_patch_angle = (dtheta <= patch_size_theta/2) and (dphi <= phi_span/2)
                    
                    if inside_patch_angle:
                        theta_map = (np.pi - th_hit) if flip_theta else th_hit
                        phi_map   = (-ph_hit)        if flip_phi  else ph_hit

                        
                        #method b:
                        # ── θ → vertical index ─────────────────────────────────────────
                        u = int((theta_map - theta0) / (theta1 - theta0) * (h - 1) + 0.5)
                        # ── ϕ → horizontal index (continuous, no wrap seam) ───────────
                        v = int(phi_rel / phi_span * (w - 1) + 0.5)
                        u = np.clip(u, 0, h-1)
                        v = np.clip(v, 0, w-1)
                        value = tuple(bg_array[u, v])
                        
                        #method a:
                        # # ------- map only the patch, not the whole sky -------------
                        # u = int((theta_map - theta0) / (theta1 - theta0) * (h - 1))
                        # phi_mod = (phi_map - phi0) % (2*np.pi)
                        # v = int(phi_mod / phi_span * (w - 1))
                        # # ------------------------------------------------------------

                        # u = np.clip(u, 0, h-1)
                        # v = np.clip(v, 0, w-1)
                        # value = tuple(bg_array[u, v])
                        bg_u, bg_v = u, v
                        rgb = value
                        collision = 'escape_bg'
                    else:
                        value = (0, 0, 255)
                        # value = (0, 0, 0)
                        collision = 'escape_no_patch'
                else:
                    value = (0, 0, 255)
                    # value = (0, 0, 0)
                    collision = 'escape_no_patch'
            else:
                value = (255, 0, 0)
                # value = (0, 0, 0)
                collision = 'in_domain'
            img[i, j] = value
            
            photon_rows.append({'i': i, 'j': j, 
                                'final_r': r_bh, 'final_th': th_hit, 'final_ph': ph_hit,
                                'collision': collision, 
                                'h_r': h_rs[idx].item(), 'h_theta': h_thetas[idx].item(), 'h_phi': h_phis[idx].item(),
                                'p0_r': p0s[idx, 0].item(), 'p0_th': p0s[idx, 1].item(), 'p0_ph': p0s[idx, 2].item()})
            
        plt.imsave('images/manual_output.png', img)
        logging.info("Saved manual_output.png")
        print("Saving ray photon data...")
        pd.DataFrame(photon_rows).to_csv('photon_data.csv', index=False)

        # Save sampled trajectories CSV
        if n_samples > 0 and sampled_traj_data:
            print("Saving diagnostic sampled trajectories ...")
            rows = []
            #use tqdm to show progress
            for ridx, traj in tqdm(enumerate(sampled_traj_data), desc="Saving diagnostic sampled trajectories", unit="ray"):
            # for ridx, traj in enumerate(sampled_traj_data):
                for pidx, (px, py, pz) in enumerate(traj):
                    pr = np.linalg.norm([px, py, pz])
                    rows.append({'ray_id': ridx, 'point_idx': pidx, 'x': px, 'y': py, 'z': pz, 'r': pr, 'h_r': h_rs[ridx], 'h_theta': h_thetas[ridx], 'h_phi': h_phis[ridx]})
            df = pd.DataFrame(rows)
            df.to_csv('sampled_rays.csv', index=False)

        sampled_trajectories = sampled_traj_data
    else: ######## CPU fallback (cooked)
        pass

    # Count summary
    if 'photon_rows' in locals():
        n_captured = sum(1 for row in photon_rows if row['collision'] == 'bh')
        n_escaped = sum(1 for row in photon_rows if row['collision'] == ('escape_no_patch'))
        n_bg = sum(1 for row in photon_rows if row['collision'] == 'escape_bg')
        print(f"Summary: {n_captured} rays captured by BH, {n_escaped} rays escaped, {n_bg} rays hit the background image.")
    else:
        print("Summary counts not available (CPU fallback mode).")
    if n_samples > 0:
        return img, sampled_trajectories
    return img 