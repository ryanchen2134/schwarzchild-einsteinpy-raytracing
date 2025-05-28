import numpy as np
import pandas as pd
from einsteinpy.geodesic import Geodesic, Timelike, Nulllike
from einsteinpy.plotting import GeodesicPlotter, StaticGeodesicPlotter, InteractiveGeodesicPlotter
from einsteinpy.coordinates.utils import spherical_to_cartesian_fast

r = 10
theta = np.pi/2 # equatorial x-y plane 
phi = 0         # on x-axis

alpha = 60 # degree deflection of -r hat direction in the x-y plane (toward the y axis) -> phi
beta = 60 # degree deflection of -r hat direction in the x-z plane (toward the z axis) -> theta
# p_theta = pi/2 - beta
# p_phi = pi - alpha
# p_r: 1

p_r = 1
p_theta = np.pi/2 - np.deg2rad(beta)
p_phi = np.pi - np.deg2rad(alpha)

# Double check: p_r, p_theta, p_phi should be unit vectors
_ , p_x, p_y, p_z = spherical_to_cartesian_fast(0, p_r, p_theta, p_phi)
print(f"p_r: {p_r}, p_theta: {np.degrees(p_theta)}, p_phi: {np.degrees(p_phi)}, norm: {np.linalg.norm([p_x, p_y, p_z])}")


q0 = [r, theta, phi]
p0 = [p_r, p_theta, p_phi]
a = 0. # Schwarzschild Black Hole


print("Starting geodesic integration...")
# geodesic object (contains trajectory data)
geod = Nulllike(
    metric = "Schwarzschild",
    metric_params = (a,),
    position=q0, 
    momentum=p0, #thing will automatically calculate the time-componenet, given our 3-momentum to satisfy the null condition
    steps=10000, # As close as we can get before the integration becomes highly unstable
    delta=0.05,
    return_cartesian=True,
    omega=0.01, # Small omega values lead to more stable integration
    suppress_warnings=True, # Uncomment to view the tolerance warning
)

#save the trajectory data
traj = geod.trajectory




# plotting
gpl = GeodesicPlotter()
gpl.plot(geod, color="green")
gpl.show()