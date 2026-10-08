//! The background texture, sampled at its native resolution.
//!
//! The Python implementation resized the texture to the output image size
//! before sampling, which tied sky detail to the render size. Here the
//! [`crate::sky_patch::SkyPatch`] rule maps a direction straight to a texel
//! of the texture as loaded.

use anyhow::{Context, Result};
use std::path::Path;

pub type Rgb = [u8; 3];

#[derive(Clone, Debug)]
pub struct Texture {
    pub width: usize,
    pub height: usize,
    /// Row-major, row 0 at the top of the image (`θ = θ0`).
    pub data: Vec<Rgb>,
}

impl Texture {
    pub fn load(path: &Path) -> Result<Self> {
        let img = image::open(path)
            .with_context(|| format!("cannot open background texture {}", path.display()))?
            .to_rgb8();
        let (w, h) = img.dimensions();
        let data = img.pixels().map(|p| p.0).collect();
        Ok(Self {
            width: w as usize,
            height: h as usize,
            data,
        })
    }

    /// A uniform texture, for tests and dry runs.
    pub fn solid(width: usize, height: usize, rgb: Rgb) -> Self {
        Self {
            width,
            height,
            data: vec![rgb; width * height],
        }
    }

    #[inline]
    pub fn texel(&self, row: usize, col: usize) -> Rgb {
        self.data[row * self.width + col]
    }
}
