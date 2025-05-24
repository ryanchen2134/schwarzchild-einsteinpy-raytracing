import numpy as np
try:
    from einsteinpy.rays import Shadow
    from einsteinpy.plotting import ShadowPlotter
    import astropy.units as u
    SHADOW_AVAILABLE = True
except ImportError:
    SHADOW_AVAILABLE = False

def run_shadow_simulation(mass_bh, fov_rad, image_size):
    """
    Run the fast shadow simulation using EinsteinPy's Shadow class.
    Converts angular FOV (radians) to physical FOV (km) at observer's distance.
    Visualizes and saves the result.
    """
    if not SHADOW_AVAILABLE:
        raise RuntimeError("Shadow class not available.")
    # Assume 1M = 1.477 km for 1 solar mass, observer at 100M
    M_in_km = 1.477
    observer_distance_M = 100
    observer_distance_km = observer_distance_M * mass_bh * M_in_km
    fov_km = 2 * observer_distance_km * np.tan(fov_rad / 2)
    n_rays = image_size[0] * image_size[1]
    shadow = Shadow(mass=mass_bh * u.kg, fov=fov_km * u.km, n_rays=n_rays)
    obj = ShadowPlotter(shadow=shadow, is_line_plot=False)
    obj.plot()
    import matplotlib.pyplot as plt
    plt.title('Black Hole Shadow (Shadow class)')
    plt.axis('off')
    plt.savefig('images/shadow_output.png', bbox_inches='tight', pad_inches=0)
    plt.show()
    print('Saved shadow_output.png') 