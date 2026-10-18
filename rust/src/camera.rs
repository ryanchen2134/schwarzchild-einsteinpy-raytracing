//! The pinhole camera: one owner of the image-plane geometry.
//!
//! The observer sits on the `+x` axis. Unrotated, the camera looks at the black
//! hole at the origin: `optical_axis = −x̂`, `right = +ŷ`, `up = +ẑ`. The image plane
//! is `PLANE_DIST_FRACTION · |observer|` in front of the observer along the optical
//! axis. Row `i = 0` is the *bottom* row (`v = −½`), matching the Python
//! implementation; [`crate::output::write_png`] flips rows so that `up` is at the
//! top of the saved image.
//!
//! # Pointing
//!
//! The camera can turn (pan) but not move: the observer stays on `+x`, which the
//! equatorial rotation in [`crate::initial_conditions`] requires. The viewer basis is
//! rotated by three intrinsic angles applied in this order, each right-handed about
//! the named axis of the camera as it then stands:
//!
//! * yaw, about `up`: positive turns the view to the observer's left, so the black
//!   hole moves right in the image;
//! * pitch, about `right`: positive turns the view up, so the black hole moves down;
//! * roll, about the optical axis: positive turns the image counter-clockwise as seen
//!   by the viewer.
//!
//! Ray directions are unit vectors in the static observer's orthonormal frame (whose
//! axes coincide with the lab axes on the `+x` axis), so the rotation acts in that
//! frame and the field of view is the angle the observer measures. The horizontal
//! field of view spans the image width. The rotated directions feed the same
//! momentum construction as before, which handles any direction from a point on the
//! `x` axis.

use crate::coords::{add, mat_mul, mat_vec, normalize, rotation_about, scale, sub, Mat3, Vec3};
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
    /// Pointing angles in radians, in the order they are applied.
    pub yaw: f64,
    pub pitch: f64,
    pub roll: f64,
    pub optical_axis: Vec3,
    pub right: Vec3,
    pub up: Vec3,
    pub plane_center: Vec3,
    pub plane_width: f64,
    pub plane_height: f64,
}

impl Camera {
    /// A camera looking straight at the black hole. `observer` must lie on the `+x`
    /// axis: the momentum construction in [`crate::initial_conditions`] rotates every
    /// ray about `+x`, which only keeps the observer fixed when it sits on that axis.
    pub fn new(observer: Vec3, fov: f64, width: usize, height: usize) -> Result<Self> {
        Self::with_rotation(observer, fov, width, height, 0.0, 0.0, 0.0)
    }

    /// A camera turned by `yaw`, `pitch`, `roll` (radians; see the module docs).
    pub fn with_rotation(
        observer: Vec3,
        fov: f64,
        width: usize,
        height: usize,
        yaw: f64,
        pitch: f64,
        roll: f64,
    ) -> Result<Self> {
        if observer[0] <= 0.0 || observer[1] != 0.0 || observer[2] != 0.0 {
            bail!("observer must lie on the +x axis, got {observer:?}");
        }
        if width == 0 || height == 0 {
            bail!("image size must be at least 1×1");
        }
        if !(fov > 0.0 && fov < std::f64::consts::PI) {
            bail!("field of view must be in (0, π) radians, got {fov}");
        }
        if !(yaw.is_finite() && pitch.is_finite() && roll.is_finite()) {
            bail!("yaw, pitch and roll must be finite, got {yaw}, {pitch}, {roll}");
        }
        let axis0: Vec3 = [-1.0, 0.0, 0.0];
        let right0: Vec3 = [0.0, 1.0, 0.0];
        let up0: Vec3 = [0.0, 0.0, 1.0];
        let rot = Self::rotation_matrix(yaw, pitch, roll);
        let optical_axis = mat_vec(&rot, axis0);
        let right = mat_vec(&rot, right0);
        let up = mat_vec(&rot, up0);
        let plane_dist = PLANE_DIST_FRACTION * observer[0];
        let plane_center = add(observer, scale(optical_axis, plane_dist));
        let plane_width = 2.0 * plane_dist * (fov / 2.0).tan();
        let plane_height = plane_width * (height as f64 / width as f64);
        Ok(Self {
            observer,
            fov,
            width,
            height,
            yaw,
            pitch,
            roll,
            optical_axis,
            right,
            up,
            plane_center,
            plane_width,
            plane_height,
        })
    }

    /// The rotation that takes the unrotated basis to the pointed one: intrinsic
    /// yaw → pitch → roll equals the product of the three rotations about the
    /// *unrotated* axes in that order.
    pub fn rotation_matrix(yaw: f64, pitch: f64, roll: f64) -> Mat3 {
        let r_yaw = rotation_about([0.0, 0.0, 1.0], yaw);
        let r_pitch = rotation_about([0.0, 1.0, 0.0], pitch);
        let r_roll = rotation_about([-1.0, 0.0, 0.0], roll);
        mat_mul(&r_yaw, &mat_mul(&r_pitch, &r_roll))
    }

    pub fn is_rotated(&self) -> bool {
        self.yaw != 0.0 || self.pitch != 0.0 || self.roll != 0.0
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
    use crate::coords::{cross, dot};
    use std::f64::consts::FRAC_PI_2;

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
        assert!(!cam.is_rotated());
    }

    #[test]
    fn rejects_off_axis_observer() {
        assert!(Camera::new([30.0, 1.0, 0.0], 1.0, 4, 4).is_err());
        assert!(Camera::new([-30.0, 0.0, 0.0], 1.0, 4, 4).is_err());
        assert!(Camera::with_rotation([30.0, 0.0, 0.0], 1.0, 4, 4, f64::NAN, 0.0, 0.0).is_err());
    }

    #[test]
    fn zero_rotation_is_bit_identical_to_the_unrotated_camera() {
        let a = Camera::new([30.0, 0.0, 0.0], 80f64.to_radians(), 7, 5).unwrap();
        let b = Camera::with_rotation([30.0, 0.0, 0.0], 80f64.to_radians(), 7, 5, 0.0, 0.0, 0.0)
            .unwrap();
        assert_eq!(a.optical_axis, b.optical_axis);
        assert_eq!(a.right, b.right);
        assert_eq!(a.up, b.up);
        for flat in 0..a.n_pixels() {
            let (i, j) = a.pixel_of(flat);
            assert_eq!(a.pixel_position(i, j), b.pixel_position(i, j));
            assert_eq!(a.ray_direction(i, j), b.ray_direction(i, j));
        }
    }

    #[test]
    fn rotated_camera_is_the_rigid_rotation_of_the_unrotated_one() {
        let plain = Camera::new([30.0, 0.0, 0.0], 70f64.to_radians(), 9, 6).unwrap();
        for (yaw, pitch, roll) in [
            (0.3, 0.0, 0.0),
            (0.0, -0.4, 0.0),
            (0.0, 0.0, 1.1),
            (0.5, -0.2, 0.7),
        ] {
            let cam =
                Camera::with_rotation([30.0, 0.0, 0.0], 70f64.to_radians(), 9, 6, yaw, pitch, roll)
                    .unwrap();
            let rot = Camera::rotation_matrix(yaw, pitch, roll);
            assert!(cam.is_rotated());
            // Orthonormal, right-handed: right × up = −optical_axis.
            let h = cross(cam.right, cam.up);
            for (hk, ak) in h.iter().zip(&cam.optical_axis) {
                assert!((hk + ak).abs() < 1e-15);
            }
            assert!(dot(cam.right, cam.up).abs() < 1e-15);
            for flat in 0..cam.n_pixels() {
                let (i, j) = cam.pixel_of(flat);
                let expect = mat_vec(&rot, plain.ray_direction(i, j));
                let got = cam.ray_direction(i, j);
                for k in 0..3 {
                    assert!(
                        (expect[k] - got[k]).abs() < 1e-15,
                        "({i},{j}) {yaw} {pitch} {roll}"
                    );
                }
            }
        }
    }

    /// Where the black hole appears: the pixel whose direction is closest to −x̂.
    fn hole_pixel(cam: &Camera) -> (usize, usize) {
        let mut best = (0, 0);
        let mut best_dot = -2.0;
        for flat in 0..cam.n_pixels() {
            let (i, j) = cam.pixel_of(flat);
            let d = -cam.ray_direction(i, j)[0];
            if d > best_dot {
                best_dot = d;
                best = (i, j);
            }
        }
        best
    }

    #[test]
    fn yaw_and_pitch_move_the_hole_as_documented() {
        let fov = 80f64.to_radians();
        let obs = [30.0, 0.0, 0.0];
        let centre = Camera::new(obs, fov, 21, 21).unwrap();
        assert_eq!(hole_pixel(&centre), (10, 10));
        // The centre ray of a yawed camera is exactly R·(−1, 0, 0).
        let yawed = Camera::with_rotation(obs, fov, 21, 21, 0.3, 0.0, 0.0).unwrap();
        let expect = mat_vec(&Camera::rotation_matrix(0.3, 0.0, 0.0), [-1.0, 0.0, 0.0]);
        let got = yawed.ray_direction(10, 10);
        for k in 0..3 {
            assert!((expect[k] - got[k]).abs() < 1e-15);
        }
        assert!(
            (got[1] + 0.3f64.sin()).abs() < 1e-15,
            "positive yaw looks to the left (−y)"
        );
        let (i, j) = hole_pixel(&yawed);
        assert!(j > 10 && i == 10, "the hole moves right: ({i},{j})");
        let pitched = Camera::with_rotation(obs, fov, 21, 21, 0.0, 0.3, 0.0).unwrap();
        assert!(
            pitched.ray_direction(10, 10)[2] > 0.0,
            "positive pitch looks up"
        );
        let (i, j) = hole_pixel(&pitched);
        assert!(i < 10 && j == 10, "the hole moves down: ({i},{j})");
        // Yaw by half the field of view puts the hole at the image edge.
        let edge = Camera::with_rotation(obs, fov, 21, 21, fov / 2.0, 0.0, 0.0).unwrap();
        assert_eq!(hole_pixel(&edge), (10, 20));
    }

    #[test]
    fn roll_by_a_right_angle_permutes_the_pixels() {
        let fov = 60f64.to_radians();
        let plain = Camera::new([30.0, 0.0, 0.0], fov, 8, 8).unwrap();
        let rolled =
            Camera::with_rotation([30.0, 0.0, 0.0], fov, 8, 8, 0.0, 0.0, FRAC_PI_2).unwrap();
        // Positive roll turns the image counter-clockwise: what the rolled camera sees
        // at (i, j) the plain camera saw at row 7 − j, column i.
        for i in 0..8 {
            for j in 0..8 {
                let a = rolled.ray_direction(i, j);
                let b = plain.ray_direction(7 - j, i);
                for k in 0..3 {
                    assert!((a[k] - b[k]).abs() < 1e-15, "({i},{j})");
                }
            }
        }
    }
}
