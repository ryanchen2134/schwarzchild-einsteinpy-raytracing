import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Black Hole Ray Tracing Simulation")
    parser.add_argument('--size', type=int, default=512, help='Image size (NxN)')
    parser.add_argument('--fov', type=float, default=30.0, help='Field of view in degrees')
    parser.add_argument('--background', type=str, default='images/backgrounds/milky-way-background.jpeg', help='Background image path')
    parser.add_argument('--steps', type=int, default=500, help='Number of integration steps for each geodesic (default: 500)')
    parser.add_argument('--delta', type=float, default=0.1, help='Initial integration step size (default: 0.2)')
    parser.add_argument('--omega', type=float, default=0.01, help='Hamiltonian flow coupling omega (default: 1.0)')
    parser.add_argument('--rtol', type=float, default=1e-2, help='Relative tolerance for integration (default: 1e-2)')
    parser.add_argument('--atol', type=float, default=1e-2, help='Absolute tolerance for integration (default: 1e-2)')
    parser.add_argument('--order', type=int, default=2, choices=[2,4,6,8], help='Integration order (2, 4, 6, or 8; default: 2)')
    parser.add_argument('--suppress-warnings', action='store_true', help='Suppress numerical warnings during integration')
    parser.add_argument('--cuda', action='store_true', help='Enable CUDA GPU acceleration for ray tracing')
    # New simulation configurables
    parser.add_argument('--bh-mass', type=float, default=1, help='Black hole mass (default: 1)')
    parser.add_argument('--boundary-radius', type=float, default=10, help='Simulation boundary radius (default: 10*bh_mass)')
    parser.add_argument('--observer-distance', type=float, default=9, help='Observer distance from BH (default: 20*bh_mass)')
    # Background patch configurables
    parser.add_argument('--bg-patch-center-theta', type=float, default=90, help='Background patch center theta (deg, default: 90)')
    parser.add_argument('--bg-patch-center-phi', type=float, default=0, help='Background patch center phi (deg, default: 0)')
    parser.add_argument('--bg-patch-size-theta', type=float, default=10, help='Background patch size theta (deg, default: 10)')
    parser.add_argument('--bg-patch-size-phi', type=float, default=10, help='Background patch size phi (deg, default: 10)')
    parser.add_argument('--bg-flip-theta', action='store_true', help='Flip theta mapping for background patch')
    parser.add_argument('--bg-flip-phi', action='store_true', help='Flip phi mapping for background patch')
    parser.add_argument('--no-flat-trajectories', action='store_true', help='Disable calculation and plotting of flat (no-gravity) trajectories in the 3D scene')
    return parser.parse_args() 