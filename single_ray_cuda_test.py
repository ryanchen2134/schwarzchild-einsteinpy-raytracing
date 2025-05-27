import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, needed for 3-D projection

from simulation.cuda_geodesic import CUDASchwarzschildIntegrator, compute_null_4momentum_schwarzschild
from simulation.utils import get_initial_conditions


def main():
    """Generate and plot a single null geodesic integrated on the GPU."""
    
    ##### Physical Conditions #####
    # Black-hole parameters
    mass_bh = 1.0  # geometrized units (M)
    # Observer position (on +x axis)
    R_obs = 200
    observer_pos = np.array([R_obs, 0.0, 0.0])
    # Tangential offsets in the image plane
    alpha_deg = -50.0  # angle away from inward radial direction (-x)
    beta_deg  = 0  # rotation around optical axis
    #simulation boundary
    rmax = 1000
  
    ##### Numerical Parameters #####
    steps = 5000
    delta = 0.2
    omega = 0.1
    
    
    
    

    # Build orthonormal camera basis at observer
    optical_axis = np.array([-1.0, 0.0, 0.0])  # toward BH from observer
    up_guess = np.array([0.0, 0.0, 1.0])
    # ensure right vector is not zero
    if np.allclose(np.cross(optical_axis, up_guess), 0):
        up_guess = np.array([0.0, 1.0, 0.0])
    e_right = np.cross(up_guess, optical_axis)
    e_right = e_right / np.linalg.norm(e_right)
    e_up = np.cross(optical_axis, e_right)
    e_up = e_up / np.linalg.norm(e_up)

    alpha = np.radians(alpha_deg)
    beta = np.radians(beta_deg)

    ray_dir = (np.cos(alpha) * optical_axis +
               np.sin(alpha) * (np.cos(beta) * e_right + np.sin(beta) * e_up))
    ray_dir = ray_dir / np.linalg.norm(ray_dir)

    # Fake pixel position one unit away along that direction to reuse helper
    pixel_pos = observer_pos + ray_dir

    # Build initial 4-position in spherical coords
    r0 = np.linalg.norm(observer_pos)
    theta0 = np.arccos(observer_pos[2] / r0)
    phi0 = np.arctan2(observer_pos[1], observer_pos[0])
    q0 = np.array([0.0, r0, theta0, phi0])

    # Analytic null–momentum (Pt, Pr, Pθ, Pφ)
    M = mass_bh
    f_r = 1.0 - 2.0 * M / r0
    E_loc = 1.0  # choose local energy scale
    E = E_loc / np.sqrt(f_r)

    alpha = np.radians(alpha_deg)  # Θ
    beta  = np.radians(beta_deg)   # Φ

    p_t = E / f_r
    p_r = -E * np.cos(alpha)
    p_theta = E / r0 * np.sin(alpha) * np.cos(beta)
    # sin(theta0) may be zero if observer on z-axis; here θ=π/2 so sin=1
    sin_theta0 = np.sin(theta0) if np.abs(np.sin(theta0)) > 1e-12 else 1.0
    p_phi = E / (r0 * sin_theta0) * np.sin(alpha) * np.sin(beta)

    p0 = np.array([p_t, p_r, p_theta, p_phi])

    # Integrate on GPU (Fantasy order-2)
    
    integrator = CUDASchwarzschildIntegrator(steps=steps, delta=delta, mass=mass_bh, omega=omega, r_max=rmax)
    traj = integrator.integrate_batch_full(q0[np.newaxis, :], p0[np.newaxis, :])[0]  # (steps,4)
    #save this to a df, then a csv
    import pandas as pd 
    df = pd.DataFrame(traj, columns=['t', 'r', 'theta', 'phi'])
    df.to_csv('single_ray_cuda_test.csv', index=True)

    # Discard points if they cross 1.1 r_s
    rs = 2.0 * mass_bh
    mask = traj[:,1] > 1.1 * rs
    discarded = 0
    if not np.all(mask):
        first_bad = np.argmax(~mask)
        discarded = len(traj) - first_bad
        traj = traj[:first_bad]

    # Convert to Cartesian for plotting
    t, r, th, ph = traj.T
    xs = r * np.sin(th) * np.cos(ph)
    ys = r * np.sin(th) * np.sin(ph)
    zs = r * np.cos(th)

    # Compute plane of motion (defined by initial r x p)
    r0_vec = observer_pos
    p_spatial = ray_dir  # unit vector
    n_hat = np.cross(r0_vec, p_spatial)
    n_norm = np.linalg.norm(n_hat)
    if n_norm == 0:
        n_hat = np.array([0.0, 0.0, 1.0])
        n_norm = 1.0
    n_hat = n_hat / n_norm
    # In-plane basis e1 (projection of initial position onto plane) & e2 = n × e1
    e1 = r0_vec - np.dot(r0_vec, n_hat) * n_hat
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n_hat, e1)

    # Coordinates in plane for each point
    u_coords = xs * e1[0] + ys * e1[1] + zs * e1[2]
    v_coords = xs * e2[0] + ys * e2[1] + zs * e2[2]

    # 3-D plot
    fig = plt.figure(figsize=(10, 8))
    ax3d = fig.add_subplot(221, projection='3d')
    ax3d.plot(xs, ys, zs, color='orange', lw=2)
    ax3d.scatter([0], [0], [0], color='black', s=50, label='BH')
    ax3d.scatter([observer_pos[0]], [observer_pos[1]], [observer_pos[2]], color='red', s=30, label='Observer')

    # # Draw wireframe sphere at Schwarzschild radius
    # rs = 2.0 * mass_bh
    # u = np.linspace(0, 2 * np.pi, 20)
    # v = np.linspace(0, np.pi, 20)
    # x = rs * np.outer(np.cos(u), np.sin(v))
    # y = rs * np.outer(np.sin(u), np.sin(v))
    # z = rs * np.outer(np.ones(np.size(u)), np.cos(v))
    # ax3d.plot_wireframe(x, y, z, color='gray', alpha=0.3, label='Event horizon')

    ax3d.set_xlabel('x'); ax3d.set_ylabel('y'); ax3d.set_zlabel('z')
    ax3d.set_title('3-D trajectory')
    ax3d.legend()

    # x-y cross-section
    ax_xy = fig.add_subplot(222)
    ax_xy.plot(xs, ys, color='blue')
    ax_xy.set_xlabel('x'); ax_xy.set_ylabel('y'); ax_xy.set_title('x-y')
    #put dot at 0,0
    ax_xy.scatter([0], [0], color='black', s=5)
    #auto-scale the axes based on the data
    ax_xy.autoscale(axis='both')
    
    ax_xy.axis('equal')

    # x-z cross-section
    ax_xz = fig.add_subplot(223)
    ax_xz.plot(xs, zs, color='green')
    ax_xz.set_xlabel('x'); ax_xz.set_ylabel('z'); ax_xz.set_title('x-z')
    ax_xz.scatter([0], [0], color='black', s=5)
    ax_xz.autoscale(axis='both')
    ax_xz.set_ylim(-1.5, 1.5)
    ax_xz.axis('equal')

    # Planar polar coordinates (r_plane, theta_plane')
    r_plane = np.sqrt(u_coords**2 + v_coords**2)
    theta_plane = np.arctan2(v_coords, u_coords)

    ax_plane = fig.add_subplot(224, projection='polar')
    ax_plane.plot(theta_plane, r_plane, color='purple')
    ax_plane.set_title("Planar (r, θ')")
    ax_plane.set_rlabel_position(45)

    plt.tight_layout()
    plt.savefig('single_ray_cuda_test.png', dpi=150)
    plt.show()

    print(f"Discarded {discarded} points that were within 1.1 r_s of the BH.")


if __name__ == "__main__":
    main() 