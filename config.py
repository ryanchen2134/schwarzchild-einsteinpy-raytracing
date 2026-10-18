#config.py
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Black Hole Ray Tracing Simulation")
    parser.add_argument('--size', type=int, default=200, help='Image size (NxN) (default: 200)')
    parser.add_argument('--fov', type=float, default=80.0, help='Field of view in degrees (default: 80)')
    parser.add_argument('--background', type=str, default='images/backgrounds/milky-way-equirec.jpg', help='Background image path')
    parser.add_argument('--steps', type=int, default=200000, help='Number of integration steps for each geodesic (default: 200000)')
    parser.add_argument('--delta', type=float, default=0.01, help='Affine-parameter step size (default: 0.01)')
    parser.add_argument('--omega', type=float, default=0.01, help='FANTASY Hamiltonian flow coupling omega (default: 0.01)')
    parser.add_argument('--suppress-warnings', action='store_true', help='Suppress numerical warnings during integration')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='Disable CUDA GPU acceleration (no CPU integrator exists yet; this only makes the failure explicit)')
    # New simulation configurables
    parser.add_argument('--bh-mass', type=float, default=1.0, help='Black hole mass (default: 1)')
    parser.add_argument('--boundary-radius', type=float, default=31, help='Simulation boundary radius (default: 31)')
    parser.add_argument('--observer-distance', type=float, default=30, help='Observer distance from BH (default: 30)')
    # Background patch configurables
    parser.add_argument('--bg-patch-center-theta', type=float, default=90, help='Background patch center theta in degrees(deg, default: 90)')
    parser.add_argument('--bg-patch-center-phi', type=float, default=180, help='Background patch center phi in degrees (deg, default: 180)')

    parser.add_argument('--bg-patch-center-theta-relobs', type=float, default=0, help='Background patch center theta offset (+ up / - down) in degrees relative to optical axis (deg, default: 0)')
    parser.add_argument('--bg-patch-center-phi-relobs', type=float, default=0, help='Background patch center phi offset (+ right / - left) in degrees relative to optical axis (deg, default: 0)')

    parser.add_argument('--bg-patch-size-theta', type=float, default=180, help='Background patch size theta (deg, default: 180)')
    parser.add_argument('--bg-patch-size-phi', type=float, default=360, help='Background patch size phi (deg, default: 360)')
    parser.add_argument('--bg-flip-theta', action='store_true', help='Flip theta mapping for background patch')
    parser.add_argument('--bg-flip-phi', action='store_true', help='Flip phi mapping for background patch')
    parser.add_argument('--no-flat-trajectories', action='store_true', default=False, help='Disable calculation and plotting of flat (no-gravity) trajectories in the 3D scene')
    # Camera pointing (degrees). The camera pans; it never moves off the +x axis.
    parser.add_argument('--yaw', type=float, default=0.0, help="Camera yaw in degrees, a right-handed turn about the camera's up axis: positive looks to the observer's left, so the black hole moves right in the image (default: 0)")
    parser.add_argument('--pitch', type=float, default=0.0, help="Camera pitch in degrees, about the camera's right axis after the yaw: positive looks up, so the black hole moves down in the image (default: 0)")
    parser.add_argument('--roll', type=float, default=0.0, help='Camera roll in degrees about the optical axis after yaw and pitch: positive turns the image counter-clockwise (default: 0)')
    # Accretion disk (README, "Accretion disk").
    parser.add_argument('--disk', action='store_true', help='Render a thin, opaque, Keplerian accretion disk')
    parser.add_argument('--disk-inclination', type=float, default=75.0, help='Angle in degrees between the disk normal and the observer position vector +x: 0 face-on, 90 edge-on, 180 face-on with the opposite sense of rotation. Measured against the observer position, not the optical axis (default: 75)')
    parser.add_argument('--disk-azimuth', type=float, default=0.0, help='Right-handed rotation of the disk normal about +x, in degrees. At 0 the near side of an inclined disk is below the black hole and, edge-on, the left side approaches (default: 0)')
    parser.add_argument('--disk-inner', type=float, default=None, help='Inner radius of the disk, in the same length unit as the other radii. Defaults to the ISCO, 6M, and may not be smaller')
    parser.add_argument('--disk-outer', type=float, default=None, help='Outer radius of the disk (default 20M); must stay below --observer-distance')
    parser.add_argument('--disk-temperature', type=float, default=6000.0, help='Peak effective temperature of the disk in kelvin. A false-colour parameter: it sets the hue, not the physics (default: 6000)')
    return parser.parse_args()
