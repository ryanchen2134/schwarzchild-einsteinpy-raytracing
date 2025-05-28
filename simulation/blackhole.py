#blackhole.py
import numpy as np

class BlackHole:
    """
    Represents a Schwarzschild black hole.
    mass: in geometrized units (e.g., M = 1)
    position: 3-vector, same units as mass
    """
    def __init__(self, mass=1.0, position=np.zeros(3)):
        self.mass = mass
        self.position = np.array(position)
        self.rs = 2 * mass  # Schwarzschild radius (r_s = 2M)

class Observer:
    """
    Represents the observer in the simulation.
    position: 3-vector, in same units as BH mass
    fov: field of view in radians
    image_size: (height, width)
    """
    def __init__(self, position, fov, image_size):
        self.position = np.array(position)
        self.fov = fov
        self.image_size = image_size

class Photon:
    """
    Represents a photon for ray tracing.
    position: 3-vector
    direction: unit vector
    mesh_idx: (i, j) pixel indices
    """
    def __init__(self, position, direction, mesh_idx):
        self.position = np.array(position)
        self.direction = np.array(direction)
        self.mesh_idx = mesh_idx
        self.collision = None
        self.collision_pos = None 