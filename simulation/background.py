#background.py
from PIL import Image
import numpy as np
import os
import random
from tqdm import tqdm
from simulation.utils import camera_basis, pixel_positions, ray_directions, vec_norm
from simulation.disk import disk_pixel

def _in_phi_patch(phi, phi0, phi1):
    """Return True if phi lies on the counter-clockwise arc from phi0 to phi1.

    A full-circle patch (phi1 - phi0 a multiple of 2*pi, as for the default
    360-degree background) contains every phi. The previous interval test
    reduced phi0 and phi1 modulo 2*pi first, so a 360-degree patch degenerated
    to the single value phi == phi0 and the no-gravity image came out black.
    """
    span = (phi1 - phi0) % (2 * np.pi)
    if span == 0.0:
        return True
    return (phi - phi0) % (2 * np.pi) <= span

def save_no_gravity_image_with_background(
    observer, bg_path, out_path, boundary_radius=None,
    patch_center_theta=None, patch_center_phi=None, patch_size_theta=np.deg2rad(10), patch_size_phi=np.deg2rad(10),
    flip_theta=False, flip_phi=False,
    return_sampled_trajectories=False, n_sampled=10,
    override_patch_center=False,
    cuda=False,
    yaw=0.0, pitch=0.0, roll=0.0,
    disk=None
):
    """
    Save an image using flat-space (no curvature) ray tracing, with optional CUDA acceleration.

    yaw, pitch, roll (radians) turn the camera exactly as in the curved render
    (simulation.utils.camera_basis). `disk` is an optional simulation.disk.Disk: a
    straight ray that meets it inside [r_in, r_out] is drawn with the disk's colour at
    g = 1 (no redshift) instead of the sky, and a sampled trajectory ends there.
    """
    h, w = observer.image_size
    if boundary_radius is None:
        boundary_radius = np.linalg.norm(observer.position) * 2  # fallback
    bg_img = Image.open(bg_path).convert('RGB')
    bg_array = np.array(bg_img.resize((w, h), Image.LANCZOS))
    out_img = np.zeros((h, w, 3), dtype=np.uint8)

    obs_pos = np.array(observer.position)
    # Same camera as the curved renderer (raytracing.py): the observer on +x looks
    # along -x with right = +y, up = +z, turned by yaw, pitch, roll. (An earlier
    # cross(up_guess, optical_axis) gave right = -y and mirrored the no-gravity image.)
    optical_axis, right, up_vec = camera_basis(yaw, pitch, roll)
    fov = observer.fov
    pixel_pos_grid = pixel_positions(obs_pos, fov, h, w, optical_axis, right, up_vec)

    def disk_hit_colour(ray_dir):
        """(hit, colour) of the straight ray on the disk, or (None, None)."""
        if disk is None:
            return None, None
        hit = disk.straight_ray_hit(obs_pos, ray_dir)
        if hit is None:
            return None, None
        return np.array(hit), disk_pixel(1.0, disk.temperature(vec_norm(hit)), disk.peak_temperature)

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
        ray_dirs = ray_directions(obs_pos, pixel_pos_grid).reshape(-1, 3)
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
        if disk is not None:
            # The flat kernel knows nothing of the disk: draw its hits on the host.
            for flat_idx in range(h * w):
                hit, colour = disk_hit_colour(ray_dirs[flat_idx])
                if hit is None:
                    continue
                i, j = divmod(flat_idx, w)
                out_img[i, j] = colour
                if return_sampled_trajectories and flat_idx in sampled_indices_flat:
                    sampled_trajectories[sampled_indices_flat.index(flat_idx)] = np.linspace(obs_pos, hit, 100)
    else:
        for i in tqdm(range(h), desc='Flat ray tracing', unit='row'):
            for j in range(w):
                pixel_pos = pixel_pos_grid[i, j]
                ray_dir = pixel_pos - obs_pos
                ray_dir = ray_dir / np.linalg.norm(ray_dir)
                hit, colour = disk_hit_colour(ray_dir)
                if hit is not None:
                    # The disk is opaque: the ray ends there, drawn without redshift.
                    out_img[i, j] = colour
                    if return_sampled_trajectories and (i, j) in sampled_indices:
                        sampled_trajectories.append(np.linspace(obs_pos, hit, 100))
                    continue
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
                    u_bg = int((theta_map - theta0) / (theta1 - theta0) * (h - 1))
                    phi_mod = (phi_map - phi0) % (2 * np.pi)
                    v_bg = int(phi_mod / phi_span * (w - 1))
                    u_bg = min(max(u_bg, 0), h - 1)
                    v_bg = min(max(v_bg, 0), w - 1)
                    out_img[i, j] = bg_array[u_bg, v_bg]
                else:
                    out_img[i, j] = (0, 0, 0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Row i = 0 is the bottom of the image: flip so the saved PNG has up at the top.
    Image.fromarray(np.ascontiguousarray(out_img[::-1])).save(out_path)
    print(f"Saved no-gravity image to {out_path}")
    scene_out_path = os.path.join(os.path.dirname(out_path), 'scene_full.png')
    bg_img_full = Image.open(bg_path).convert('RGB')
    bg_img_full.save(scene_out_path)
    print(f"Saved full scene image to {scene_out_path}")
    if return_sampled_trajectories:
        return sampled_trajectories 