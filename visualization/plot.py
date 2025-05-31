import matplotlib.pyplot as plt
import numpy as np
import os
from einsteinpy.hypersurface import SchwarzschildEmbedding
from einsteinpy.plotting import HypersurfacePlotter
from astropy import units as u
from mpl_toolkits.mplot3d import Axes3D
import random
from simulation.cuda_geodesic import CUDASchwarzschildIntegrator
from simulation.utils import get_initial_conditions

def plot_placeholder():
    """Placeholder for future plotting utilities."""
    pass

def plot_scene_topdown(
    bh, observer, image_plane_size, boundary_radius, out_path='images/scene_topdown.png', fov_deg=50,
    patch_center_theta=np.pi/2, patch_size_theta=np.deg2rad(10), patch_size_phi=np.deg2rad(10)
):
    """
    Plot a top-down (x-y) view of the simulation setup, matching the actual raytracing geometry:
    - Black hole at origin
    - Observer at observer.position
    - Simulation boundary as a circle
    - FOV cone from observer (matching observer.fov)
    - Image plane as an arc at fixed radius from observer, with tick marks for each pixel
    - Background patch as an arc/segment on the boundary, centered on the point directly opposite the observer (along the optical axis)
    - Overlay photon emission directions as rays
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    # Black hole
    bh_circle = plt.Circle((0, 0), bh.rs, color='black', label='Black Hole')
    ax.add_patch(bh_circle)
    # Observer
    obs_x, obs_y = observer.position[0], observer.position[1]
    ax.plot(obs_x, obs_y, 'ro', label='Observer', markersize=10)
    # Simulation boundary
    boundary = plt.Circle((0, 0), boundary_radius, color='gray', fill=False, linestyle='--', label='Boundary')
    ax.add_patch(boundary)
    # FOV cone
    fov = observer.fov
    n_pix = image_plane_size[0]
    # Assume observer is on x axis, looking toward origin
    obs_angle = np.arctan2(-obs_y, -obs_x)  # direction toward BH
    theta0 = obs_angle - fov/2
    theta1 = obs_angle + fov/2
    # Draw FOV lines
    ax.plot([obs_x, obs_x + 2 * boundary_radius * np.cos(theta0)],
            [obs_y, obs_y + 2 * boundary_radius * np.sin(theta0)], 'k--', lw=1, alpha=0.7)
    ax.plot([obs_x, obs_x + 2 * boundary_radius * np.cos(theta1)],
            [obs_y, obs_y + 2 * boundary_radius * np.sin(theta1)], 'k--', lw=1, alpha=0.7)
    # Draw FOV arc on boundary (for reference)
    arc_thetas = np.linspace(theta0, theta1, 200)
    arc_x = boundary_radius * np.cos(arc_thetas)
    arc_y = boundary_radius * np.sin(arc_thetas)
    ax.plot(arc_x, arc_y, color='green', lw=2, alpha=0.3, label='FOV (Boundary Arc)')
    # Draw background patch as an arc/segment on the boundary
    # Patch is centered on the point opposite the observer
    obs_phi = np.arctan2(obs_y, obs_x)
    patch_phi = (obs_phi + np.pi) % (2 * np.pi)
    patch_phi0 = patch_phi - patch_size_phi/2
    patch_phi1 = patch_phi + patch_size_phi/2
    patch_phis = np.linspace(patch_phi0, patch_phi1, 200)
    patch_x = boundary_radius * np.cos(patch_phis)
    patch_y = boundary_radius * np.sin(patch_phis)
    ax.plot(patch_x, patch_y, color='magenta', lw=6, alpha=0.5, label='Background Patch')
    # Draw image plane as arc at fixed radius from observer
    plane_radius = 0.2 * np.linalg.norm([obs_x, obs_y])
    plane_thetas = np.linspace(theta0, theta1, n_pix)
    plane_x = obs_x + plane_radius * np.cos(plane_thetas)
    plane_y = obs_y + plane_radius * np.sin(plane_thetas)
    ax.plot(plane_x, plane_y, color='blue', lw=3, alpha=0.5, label='Image Plane (arc)')
    # Tick marks for each pixel
    for px, py in zip(plane_x, plane_y):
        ax.plot([obs_x, px], [obs_y, py], color='blue', lw=0.5, alpha=0.2)
    # Overlay photon emission directions as rays (sparser for clarity)
    for t in np.linspace(theta0, theta1, min(n_pix, 32)):
        ax.plot([obs_x, obs_x + boundary_radius * np.cos(t)],
                [obs_y, obs_y + boundary_radius * np.sin(t)], color='orange', lw=0.5, alpha=0.3)
    # Formatting
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Top-Down Scene View (Simulation Geometry)')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    # Set limits for better visibility
    obs_dist = np.linalg.norm([obs_x, obs_y])
    lim = max(boundary_radius, obs_dist) * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved top-down scene image to {out_path}")



def plot_scene_embedding_3d(
    bh, observer, image_plane_size, boundary_radius, out_path='images/scene_topdown.png', fov_deg=None, photon_trajectories=None,
    patch_center_theta=None, patch_center_phi=None, patch_size_theta=np.deg2rad(10), patch_size_phi=np.deg2rad(10),
    override_patch_center=False, flat_trajectories=None
):
    """
    Plot a 3D view of the Schwarzschild spatial hypersurface embedding with simulation elements to scale:
    - Black hole event horizon (sphere at r_s)
    - Observer (point)
    - Image plane (rectangle)
    - Simulation boundary (sphere)
    - User-supplied photon trajectories (list of arrays of shape (N,3)), plotted in orange
    - Optionally: flat_trajectories (list of arrays of shape (N,3)), plotted in blue
    - Background patch as a magenta patch/arc/segment on the boundary sphere
    - By default, the patch center is set to the point on the boundary directly opposite the observer (unless override_patch_center=True).
    """
    # 1. Determine the field-of-view for constructing the image plane
    if fov_deg is None:
        fov = observer.fov
    else:
        fov = np.deg2rad(fov_deg)

    # 2. Prepare event horizon (sphere at r_s) coordinates for later plotting
    rs = 2 * bh.mass  # Schwarzschild radius
    u_sphere, v_sphere = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x_s = rs * np.cos(u_sphere) * np.sin(v_sphere)
    y_s = rs * np.sin(u_sphere) * np.sin(v_sphere)
    z_s = rs * np.cos(v_sphere)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 3. Add observer (point)
    obs_pos = observer.position if hasattr(observer, 'position') else np.array([8*rs, 0, 0])
    ax.scatter([obs_pos[0]], [obs_pos[1]], [obs_pos[2]], color='red', s=100, label='Observer')

    # 4. Add image plane (rectangle) using the resolved fov (radians)
    obs_r = np.linalg.norm(obs_pos)
    plane_dist = 0.2 * obs_r
    plane_center = obs_pos - (obs_pos/obs_r) * plane_dist
    up = np.array([0, 0, 1])
    if np.allclose(np.cross(obs_pos, up), 0):
        up = np.array([0, 1, 0])
    right = np.cross(up, obs_pos)
    right = right / np.linalg.norm(right)
    up_vec = np.cross(obs_pos, right)
    up_vec = up_vec / np.linalg.norm(up_vec)
    width = 2 * plane_dist * np.tan(fov/2)
    height = width * (image_plane_size[0]/image_plane_size[1])
    for dx, dy in [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5)]:
        corner = plane_center + dx*width*right + dy*height*up_vec
        if dx == -0.5 and dy == -0.5:
            xs, ys, zs = [corner[0]], [corner[1]], [corner[2]]
        else:
            xs.append(corner[0])
            ys.append(corner[1])
            zs.append(corner[2])
    ax.plot(xs, ys, zs, color='blue', lw=2, label='Image Plane')

    
    # 5. Add simulation boundary (sphere)
    br = boundary_radius
    x_b = br * np.cos(u_sphere) * np.sin(v_sphere)
    y_b = br * np.sin(u_sphere) * np.sin(v_sphere)
    z_b = br * np.cos(v_sphere)
    ax.plot_wireframe(x_b, y_b, z_b, color='gray', alpha=0.05, label='Boundary')

    # 5b. Draw background patch as magenta points/mesh on the boundary sphere (higher fidelity)
    if not override_patch_center or patch_center_theta is None or patch_center_phi is None:
        opp_pos = -obs_pos
        r_opp = np.linalg.norm(opp_pos)
        patch_center_theta = np.arccos(opp_pos[2] / r_opp)
        patch_center_phi = np.arctan2(opp_pos[1], opp_pos[0])
    n_patch_theta = 100
    n_patch_phi = 200
    theta0 = patch_center_theta - patch_size_theta/2
    theta1 = patch_center_theta + patch_size_theta/2
    phi0 = patch_center_phi - patch_size_phi/2
    phi1 = patch_center_phi + patch_size_phi/2
    patch_thetas = np.linspace(theta0, theta1, n_patch_theta)
    patch_phis = np.linspace(phi0, phi1, n_patch_phi)
    patch_theta_grid, patch_phi_grid = np.meshgrid(patch_thetas, patch_phis, indexing='ij')
    patch_x = br * np.sin(patch_theta_grid) * np.cos(patch_phi_grid)
    patch_y = br * np.sin(patch_theta_grid) * np.sin(patch_phi_grid)
    patch_z = br * np.cos(patch_theta_grid)
    ax.plot_surface(patch_x, patch_y, patch_z, color='magenta', alpha=0.5, linewidth=0, antialiased=True, zorder=10)

    # 6. Plot user-supplied photon trajectories (orange)
    if photon_trajectories is not None and len(photon_trajectories) > 0:
        for traj in photon_trajectories:
            # if traj.shape[0] < 4:
            #     # Densify if only endpoints are provided
            #     p0, p1 = traj[0], traj[-1]
            #     n_interp = 50
            #     alphas = np.linspace(0, 1, n_interp)
            #     traj_dense = np.outer(1 - alphas, p0) + np.outer(alphas, p1)
            # else:
            traj_dense = traj
            ax.plot(traj_dense[:, 0], traj_dense[:, 1], traj_dense[:, 2], color='orange', lw=1, alpha=1.0, zorder=15, label='Sampled Rays' if 'Sampled Rays' not in ax.get_legend_handles_labels()[1] else None)
            # Mark start & end points for clarity
            ax.scatter(traj_dense[0,0], traj_dense[0,1], traj_dense[0,2], color='lime', s=20, zorder=16)
            ax.scatter(traj_dense[-1,0], traj_dense[-1,1], traj_dense[-1,2], color='red', s=20, zorder=16)
    else:
        print("[plot_scene_embedding_3d] Warning: photon_trajectories is None or empty. No sampled rays to plot.")

    # 6b. Plot straight-line (no-gravity) photon trajectories (blue)
    if flat_trajectories is not None:
        for traj in flat_trajectories:
            ax.plot(traj[:,0], traj[:,1], traj[:,2], color='blue', lw=1, alpha=0.7, label='Straight Ray' if 'Straight Ray' not in ax.get_legend_handles_labels()[1] else None)

    # 7. Plot event horizon last for visibility: solid + wireframe
    ax.plot_surface(x_s, y_s, z_s, color='black', alpha=1.0, zorder=20)
    ax.plot_wireframe(x_s, y_s, z_s, color='yellow', linewidth=0.1, zorder=21)

    # Formatting
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('3D Scene: Schwarzschild Embedding & Simulation Geometry')
    max_range = max(boundary_radius, np.linalg.norm(observer.position)) * 1.1
    for axis in 'xyz':
        getattr(ax, f'set_{axis}lim')([-max_range, max_range])
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Observer', markerfacecolor='red', markersize=10),
        Line2D([0], [0], color='black', lw=4, label='Event Horizon'),
        Line2D([0], [0], color='orange', lw=2, label='Sampled Rays'),
        Line2D([0], [0], color='blue', lw=2, label='Straight Rays'),
    ]
    legend_elements.append(Line2D([0], [0], color='magenta', lw=2, label='Background Patch'))
    ax.legend(handles=legend_elements)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()

    # --- Output 3 perspectives rotated about z axis ---
    base, ext = os.path.splitext(out_path)
    for i, azim in enumerate([0,90,120, 180 ,240]):
        ax.view_init(elev=30, azim=azim)  # 30 deg elevation, azim rotation
        out_path_rot = f"{base}_azim{azim}{ext}"
        fig.savefig(out_path_rot)
        print(f"Saved 3D embedding scene image to {out_path_rot}")
    plt.close(fig) 

def plot_scene_closeup_3d(bh, observer, image_plane_size, out_path='images/scene_closeup_3d.png', fov_deg=None):
    """
    Plot a close-up 3D view near the observer, showing:
    - Observer (point)
    - Image plane (rectangle)
    - Black hole event horizon (sphere)
    - FOV arc (dashed) on the image plane
    """
    rs = 2 * bh.mass
    obs_pos = np.array(observer.position)
    obs_r = np.linalg.norm(obs_pos)
    if fov_deg is None:
        fov = observer.fov
    else:
        fov = np.deg2rad(fov_deg)
    h, w = image_plane_size
    plane_dist = 0.2 * obs_r
    plane_center = obs_pos - (obs_pos/obs_r) * plane_dist
    up = np.array([0, 0, 1])
    if np.allclose(np.cross(obs_pos, up), 0):
        up = np.array([0, 1, 0])
    right = np.cross(up, obs_pos)
    right = right / np.linalg.norm(right)
    up_vec = np.cross(obs_pos, right)
    up_vec = up_vec / np.linalg.norm(up_vec)
    width = 2 * plane_dist * np.tan(fov/2)
    height = width * (h/w)
    # Rectangle corners
    corners = []
    for dx, dy in [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5)]:
        corner = plane_center + dx*width*right + dy*height*up_vec
        corners.append(corner)
    corners = np.array(corners)
    # Event horizon
    u_sphere, v_sphere = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x_s = rs * np.cos(u_sphere) * np.sin(v_sphere)
    y_s = rs * np.sin(u_sphere) * np.sin(v_sphere)
    z_s = rs * np.cos(v_sphere)
    # Plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Event horizon
    ax.plot_surface(x_s, y_s, z_s, color='black', alpha=1.0, zorder=20)
    ax.plot_wireframe(x_s, y_s, z_s, color='yellow', linewidth=0.7, zorder=21)
    # Observer
    ax.scatter([obs_pos[0]], [obs_pos[1]], [obs_pos[2]], color='red', s=100, label='Observer')
    # Image plane (rectangle)
    ax.plot(corners[:,0], corners[:,1], corners[:,2], color='blue', lw=2, label='Image Plane')
    # FOV arc (dashed) on the image plane
    n_arc = 100
    arc_angles = np.linspace(-fov/2, fov/2, n_arc)
    # The image plane normal is -obs_pos/obs_r
    # The arc is drawn on the image plane, centered at the observer, at distance plane_dist
    arc_points = []
    for angle in arc_angles:
        # Rotate right vector by angle in the image plane
        vec = np.cos(angle) * right + np.sin(angle) * up_vec
        arc_pt = obs_pos + plane_dist * (-obs_pos/obs_r) + (width/2) * vec
        arc_points.append(arc_pt)
    arc_points = np.array(arc_points)
    ax.plot(arc_points[:,0], arc_points[:,1], arc_points[:,2], color='purple', ls='--', lw=2, label='FOV Arc')
    # Set axis limits to include observer, image plane, and black hole (origin), with margin
    all_points = np.vstack([
        corners,
        obs_pos[None, :],
        np.zeros((1, 3)),  # black hole at origin
    ])
    min_xyz = all_points.min(axis=0)
    max_xyz = all_points.max(axis=0)
    center = (min_xyz + max_xyz) / 2
    span = (max_xyz - min_xyz).max()
    margin = 0.15 * span
    half = 0.5 * (span + margin)
    for axis, c in zip('xyz', center):
        getattr(ax, f'set_{axis}lim')(c - half, c + half)
    # Formatting
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Close-up 3D Scene: Observer, Image Plane, Event Horizon')
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Observer', markerfacecolor='red', markersize=10),
        Line2D([0], [0], color='black', lw=4, label='Event Horizon'),
        Line2D([0], [0], color='yellow', lw=2, label='Event Horizon (Wire)'),
        Line2D([0], [0], color='blue', lw=2, label='Image Plane'),
        Line2D([0], [0], color='purple', lw=2, linestyle='--', label='FOV Arc'),
    ]
    ax.legend(handles=legend_elements)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved close-up 3D scene image to {out_path}") 