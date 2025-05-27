import numpy as np

from einsteinpy.geodesic import Geodesic, Timelike, Nulllike
from einsteinpy.plotting import GeodesicPlotter, StaticGeodesicPlotter, InteractiveGeodesicPlotter

# Initial Conditions
# r , theta, phi

r = 5
theta = np.pi/2 # equatorial x-y plane 
phi = 0         # on x-axis
q0 = [4, np.pi, 0.]
p0 = [0., 0., -1.5]
a = 0. # Schwarzschild Black Hole


geod = Nulllike(
    metric = "Schwarzschild",
    position=q0, 
    momentum=p0, #thing will automatically calculate the time-componenet, given our 3-momentum to satisfy the null condition
    steps=10000, # As close as we can get before the integration becomes highly unstable
    delta=0.0005,
    return_cartesian=True,
    omega=0.01, # Small omega values lead to more stable integration
    suppress_warnings=True, # Uncomment to view the tolerance warning
)