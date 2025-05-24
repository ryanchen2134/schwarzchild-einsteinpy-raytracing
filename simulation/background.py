from PIL import Image
import numpy as np
import os
import random
from tqdm import tqdm

def _in_phi_patch(phi, phi0, phi1):
    """Return True if phi is within [phi0, phi1] on the circle, handling wrapping."""
    # Normalize to [0, 2pi)
    phi = phi % (2 * np.pi)
    phi0 = phi0 % (2 * np.pi)
    phi1 = phi1 % (2 * np.pi)
    if phi0 <= phi1:
        return phi0 <= phi <= phi1
    else:
        return phi >= phi0 or phi <= phi1

def save_no_gravity_image_with_background(
    observer, bg_path, out_path, boundary_radius=None,
    patch_center_theta=None, patch_center_phi=None, patch_size_theta=np.deg2rad(10), patch_size_phi=np.deg2rad(10),
    flip_theta=False, flip_phi=False,
    return_sampled_trajectories=False, n_sampled=10,
    override_patch_center=False,
    cuda=False
):
    """
    Save an image using flat-space (no curvature) ray tracing, with optional CUDA acceleration.
    """
    h, w = observer.image_size
    if boundary_radius is None:
        boundary_radius = np.linalg.norm(observer.position) * 2  # fallback
    bg_img = Image.open(bg_path).convert('RGB')
    bg_array = np.array(bg_img.resize((w, h), Image.LANCZOS))
    out_img = np.zeros((h, w, 3), dtype=np.uint8)

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

    if not override_patch_center or patch_center_theta is None or patch_center_phi is None:
        opp_pos = -obs_pos
        r_opp = np.linalg.norm(opp_pos)
        patch_center_theta = np.arccos(opp_pos[2] / r_opp)
        patch_center_phi = np.arctan2(opp_pos[1], opp_pos[0])

    theta0 = patch_center_theta - patch_size_theta/2
    theta1 = patch_center_theta + patch_size_theta/2
    phi0 = patch_center_phi - patch_size_phi/2
    phi1 = patch_center_phi + patch_size_phi/2
    phi_span = (phi1 - phi0) % (2 * np.pi)
    if phi_span == 0:
        phi_span = 2 * np.pi

    sampled_indices = set()
    sampled_trajectories = []
    if return_sampled_trajectories:
        while len(sampled_indices) < n_sampled:
            i = random.randint(0, h-1)
            j = random.randint(0, w-1)
            sampled_indices.add((i, j))

    if cuda:
        from simulation.cuda_geodesic import cuda_flat_raytrace_with_trajectories
        # Vectorized ray direction calculation
        jj, ii = np.meshgrid(np.arange(w), np.arange(h))
        dx = (jj + 0.5) / w - 0.5
        dy = (ii + 0.5) / h - 0.5
        pixel_pos = plane_center[None, None, :] + dx[..., None] * right + dy[..., None] * up_vec
        ray_dirs = pixel_pos - obs_pos[None, None, :]
        ray_dirs = ray_dirs / np.linalg.norm(ray_dirs, axis=2, keepdims=True)
        ray_dirs = ray_dirs.reshape(-1, 3)
        # Prepare sampled_indices as flat indices
        sampled_indices_flat = [i * w + j for (i, j) in sampled_indices] if return_sampled_trajectories else []
        if return_sampled_trajectories and sampled_indices_flat:
            out_img_flat, out_trajs = cuda_flat_raytrace_with_trajectories(
                obs_pos, ray_dirs, boundary_radius, patch_center_theta, patch_center_phi, patch_size_theta, patch_size_phi, flip_theta, flip_phi, bg_array, sampled_indices_flat, n_points=100
            )
            out_img = out_img_flat.reshape((h, w, 3))
            sampled_trajectories = [out_trajs[k] for k in range(len(sampled_indices_flat))]
        else:
            from simulation.cuda_geodesic import cuda_flat_raytrace
            out_img_flat = cuda_flat_raytrace(obs_pos, ray_dirs, boundary_radius, patch_center_theta, patch_center_phi, patch_size_theta, patch_size_phi, flip_theta, flip_phi, bg_array)
            out_img = out_img_flat.reshape((h, w, 3))
    else:
        for i in tqdm(range(h), desc='Flat ray tracing', unit='row'):
            for j in range(w):
                dx = (j + 0.5) / w - 0.5
                dy = (i + 0.5) / h - 0.5
                pixel_pos = plane_center + dx * width * right + dy * height * up_vec
                ray_dir = pixel_pos - obs_pos
                ray_dir = ray_dir / np.linalg.norm(ray_dir)
                a = np.dot(ray_dir, ray_dir)
                b = 2 * np.dot(obs_pos, ray_dir)
                c = np.dot(obs_pos, obs_pos) - boundary_radius**2
                disc = b**2 - 4*a*c
                if disc < 0:
                    out_img[i, j] = (0, 0, 0)
                    continue
                t = (-b + np.sqrt(disc)) / (2*a)
                hit_pos = obs_pos + t * ray_dir
                if return_sampled_trajectories and (i, j) in sampled_indices:
                    traj = np.linspace(obs_pos, hit_pos, 100)
                    sampled_trajectories.append(traj)
                r = np.linalg.norm(hit_pos)
                theta = np.arccos(hit_pos[2] / r)
                phi = np.arctan2(hit_pos[1], hit_pos[0])
                in_patch = (theta >= theta0) and (theta <= theta1) and _in_phi_patch(phi, phi0, phi1)
                if in_patch:
                    theta_map = (np.pi - theta) if flip_theta else theta
                    phi_map = (-phi) if flip_phi else phi
                    u = int((theta_map - theta0) / (theta1 - theta0) * (h - 1))
                    phi_mod = (phi_map - phi0) % (2 * np.pi)
                    v = int(phi_mod / phi_span * (w - 1))
                    u = min(max(u, 0), h - 1)
                    v = min(max(v, 0), w - 1)
                    out_img[i, j] = bg_array[u, v]
                else:
                    out_img[i, j] = (0, 0, 0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(out_img).save(out_path)
    print(f"Saved no-gravity image to {out_path}")
    scene_out_path = os.path.join(os.path.dirname(out_path), 'scene_full.png')
    bg_img_full = Image.open(bg_path).convert('RGB')
    bg_img_full.save(scene_out_path)
    print(f"Saved full scene image to {scene_out_path}")
    if return_sampled_trajectories:
        return sampled_trajectories 