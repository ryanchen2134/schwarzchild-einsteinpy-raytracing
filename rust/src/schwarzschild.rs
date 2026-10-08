//! The black hole and the closed-form Schwarzschild quantities the pipeline needs.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlackHole {
    /// Mass `M` in geometrised units; the Schwarzschild radius is `2M`.
    pub mass: f64,
}

impl BlackHole {
    pub fn new(mass: f64) -> Self {
        Self { mass }
    }

    /// Schwarzschild radius `r_s = 2M`.
    #[inline]
    pub fn rs(&self) -> f64 {
        2.0 * self.mass
    }

    /// Lapse `f(r) = 1 − 2M/r = −g_tt = 1/g_rr`.
    #[inline]
    pub fn lapse(&self, r: f64) -> f64 {
        1.0 - 2.0 * self.mass / r
    }

    /// Critical impact parameter `b_c = 3√3 M` of the photon sphere.
    #[inline]
    pub fn critical_impact_parameter(&self) -> f64 {
        3.0 * 3f64.sqrt() * self.mass
    }

    /// Half-angle of the shadow seen by a static observer at `r_obs ≥ 3M`, measured from
    /// the inward direction: `sin α_c = (b_c / r_obs) · √(1 − 2M/r_obs)`. Inside the photon
    /// sphere the shadow is the complement of the outward escape cone, which this does not
    /// model; the CLI rejects such observers.
    pub fn capture_angle(&self, r_obs: f64) -> f64 {
        (self.critical_impact_parameter() / r_obs * self.lapse(r_obs).sqrt())
            .clamp(-1.0, 1.0)
            .asin()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_angle_at_30m() {
        let bh = BlackHole::new(1.0);
        let a = bh.capture_angle(30.0);
        assert!((a - 0.16812).abs() < 1e-4, "got {a}");
    }

    #[test]
    fn capture_angle_is_finite_at_the_photon_sphere() {
        let bh = BlackHole::new(1.0);
        let a = bh.capture_angle(3.0);
        assert!(a.is_finite());
        assert!((a - std::f64::consts::FRAC_PI_2).abs() < 1e-7, "got {a}");
    }
}
