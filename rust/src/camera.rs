//! The pinhole camera: one owner of the image-plane geometry.
//!
//! The observer sits on the `+x` axis looking at the black hole at the
//! origin. The image plane is `PLANE_DIST_FRACTION · |observer|` in front of
//! the observer; `right = +y`, `up = +z`. Row `i = 0` is the *bottom* row
//! (`v = −½`), matching the Python implementation; [`crate::output::write_png`]
//! flips rows so that `up` is at the top of the saved image.

use crate::coords::{add, normalize, scale, sub, Vec3};
use anyhow::{bail, Result};

/// Image-plane distance as a fraction of the observer radius.
pub const PLANE_DIST_FRACTION: f64 = 0.2;

#[derive(Clone, Debug)]
pub struct Camera {
    pub observer: Vec3,
    /// Full horizontal field of view in radians.
    pub fov: f64,
    pub width: usize,
    pub height: usize,
    pub optical_axis: Vec3,
    pub right: Vec3,
    pub up: Vec3,
    pub plane_center: Vec3,
    pub plane_width: f64,
    pub plane_height: f64,
}

impl Camera {
    /// `observer` must lie on the `+x` axis: the momentum construction in
    /// [`crate::initial_conditions`] rotates every ray about `+x`, which only
    /// keeps the observer fixed when it sits on that axis.
    pub fn new(observer: Vec3, fov: f64, width: usize, height: usize) -> Result<Self> {
        if observer[0] <= 0.0 || observer[1] != 0.0 || observer[2] != 0.0 {
            bail!("observer must lie on the +x axis, got {observer:?}");
        }
        if width == 0 || height == 0 {
            bail!("image size must be at least 1×1");
        }
        if !(fov > 0.0 && fov < std::f64::consts::PI) {
            bail!("field of view must be in (0, π) radians, got {fov}");
        }
        let optical_axis = [-1.0, 0.0, 0.0];
        let right = [0.0, 1.0, 0.0];
        let up = [0.0, 0.0, 1.0];
        let plane_dist = PLANE_DIST_FRACTION * observer[0];
        let plane_center = add(observer, scale(optical_axis, plane_dist));
        let plane_width = 2.0 * plane_dist * (fov / 2.0).tan();
        let plane_height = plane_width * (height as f64 / width as f64);
        Ok(Self {
            observer,
            fov,
            width,
            height,
            optical_axis,
            right,
            up,
            plane_center,
            plane_width,
            plane_height,
        })
    }

    pub fn observer_radius(&self) -> f64 {
        self.observer[0]
    }

    pub fn n_pixels(&self) -> usize {
        self.width * self.height
    }

    /// Normalised image-plane coordinates of pixel centre `(i, j)`, both in `[−½, ½]`.
    #[inline]
    pub fn uv(&self, i: usize, j: usize) -> (f64, f64) {
        let u = (j as f64 + 0.5) / self.width as f64 - 0.5;
        let v = (i as f64 + 0.5) / self.height as f64 - 0.5;
        (u, v)
    }

    /// World position of the centre of pixel `(i, j)`; `i` is the row from the bottom.
    pub fn pixel_position(&self, i: usize, j: usize) -> Vec3 {
        let (u, v) = self.uv(i, j);
        let x = add(self.plane_center, scale(self.right, u * self.plane_width));
        add(x, scale(self.up, v * self.plane_height))
    }

    /// Unit vector from the observer through pixel `(i, j)`.
    pub fn ray_direction(&self, i: usize, j: usize) -> Vec3 {
        normalize(sub(self.pixel_position(i, j), self.observer))
    }

    /// Flat pixel index in row-major order.
    #[inline]
    pub fn flat_index(&self, i: usize, j: usize) -> usize {
        i * self.width + j
    }

    #[inline]
    pub fn pixel_of(&self, flat: usize) -> (usize, usize) {
        (flat / self.width, flat % self.width)
    }

    /// Corners of the image plane: bottom-left, bottom-right, top-right, top-left.
    pub fn plane_corners(&self) -> [Vec3; 4] {
        let c = |u: f64, v: f64| {
            add(
                add(self.plane_center, scale(self.right, u * self.plane_width)),
                scale(self.up, v * self.plane_height),
            )
        };
        [c(-0.5, -0.5), c(0.5, -0.5), c(0.5, 0.5), c(-0.5, 0.5)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basis_for_plus_x_observer() {
        let cam = Camera::new([30.0, 0.0, 0.0], 80f64.to_radians(), 8, 6).unwrap();
        assert_eq!(cam.optical_axis, [-1.0, 0.0, 0.0]);
        assert_eq!(cam.right, [0.0, 1.0, 0.0]);
        assert_eq!(cam.up, [0.0, 0.0, 1.0]);
        assert_eq!(cam.plane_center, [24.0, 0.0, 0.0]);
        let centre = cam.pixel_position(3, 4);
        // the pixel grid is symmetric, so (3,4) is a quarter pixel above-right of the axis
        assert!(centre[1] > 0.0 && centre[2] > 0.0);
        let dir = cam.ray_direction(0, 0);
        assert!(
            dir[0] < 0.0 && dir[1] < 0.0 && dir[2] < 0.0,
            "pixel (0,0) is bottom-left"
        );
    }

    #[test]
    fn rejects_off_axis_observer() {
        assert!(Camera::new([30.0, 1.0, 0.0], 1.0, 4, 4).is_err());
        assert!(Camera::new([-30.0, 0.0, 0.0], 1.0, 4, 4).is_err());
    }
}
