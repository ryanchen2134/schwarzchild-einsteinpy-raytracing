//! Files: the rendered images, the photon table, the diagnostic trajectories.

use crate::coords::{norm, Vec3};
use crate::render::{PhotonRecord, RenderResult};
use crate::shading::Collision;
use crate::texture::Rgb;
use anyhow::{Context, Result};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Write a row-major, bottom-row-first image as a PNG with `up` at the top.
pub fn write_png(path: &Path, width: usize, height: usize, pixels: &[Rgb]) -> Result<()> {
    assert_eq!(pixels.len(), width * height);
    let mut raw = Vec::with_capacity(width * height * 3);
    for i in (0..height).rev() {
        for p in &pixels[i * width..(i + 1) * width] {
            raw.extend_from_slice(p);
        }
    }
    let img =
        image::RgbImage::from_raw(width as u32, height as u32, raw).expect("buffer size matches");
    img.save(path)
        .with_context(|| format!("cannot write {}", path.display()))
}

/// One row per pixel, columns compatible with the Python `photon_data.csv` plus
/// `beta`, `exit`, `n_steps`.
pub fn write_photon_csv(path: &Path, photons: &[PhotonRecord]) -> Result<()> {
    let mut w = BufWriter::new(
        File::create(path).with_context(|| format!("cannot create {}", path.display()))?,
    );
    writeln!(
        w,
        "i,j,final_r,final_th,final_ph,collision,exit,n_steps,h_r,h_theta,h_phi,p0_t,p0_r,p0_th,p0_ph,alpha0,beta,analytic_capture"
    )?;
    for p in photons {
        let d = p.init.direction;
        let (h_r, h_th, h_ph) = crate::coords::cartesian_to_spherical(d);
        writeln!(
            w,
            "{},{},{:.17e},{:.17e},{:.17e},{},{},{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{}",
            p.i,
            p.j,
            p.result.q[1],
            p.shaded.final_theta,
            p.shaded.final_phi,
            p.shaded.collision.label(),
            p.result.exit.label(),
            p.result.n_steps,
            h_r,
            h_th,
            h_ph,
            p.init.p0[0],
            p.init.p0[1],
            p.init.p0[2],
            p.init.p0[3],
            p.init.alpha0,
            p.init.beta,
            p.analytic_capture
        )?;
    }
    w.flush()?;
    Ok(())
}

/// `ray_id,pixel,point_idx,x,y,z,r`, one row per recorded point.
pub fn write_trajectories_csv(path: &Path, trajectories: &[(usize, Vec<Vec3>)]) -> Result<()> {
    let mut w = BufWriter::new(
        File::create(path).with_context(|| format!("cannot create {}", path.display()))?,
    );
    writeln!(w, "ray_id,pixel,point_idx,x,y,z,r")?;
    for (ray_id, (pixel, points)) in trajectories.iter().enumerate() {
        for (k, p) in points.iter().enumerate() {
            writeln!(
                w,
                "{},{},{},{:.10e},{:.10e},{:.10e},{:.10e}",
                ray_id,
                pixel,
                k,
                p[0],
                p[1],
                p[2],
                norm(*p)
            )?;
        }
    }
    w.flush()?;
    Ok(())
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    pub captured: usize,
    pub escaped_background: usize,
    pub escaped_no_patch: usize,
    pub in_domain: usize,
    pub numerical_error: usize,
}

impl Summary {
    pub fn escaped(&self) -> usize {
        self.escaped_background + self.escaped_no_patch
    }
}

pub fn summarize(result: &RenderResult) -> Summary {
    let mut s = Summary::default();
    for p in &result.photons {
        match p.shaded.collision {
            Collision::BlackHole => s.captured += 1,
            Collision::EscapedBackground => s.escaped_background += 1,
            Collision::EscapedNoPatch => s.escaped_no_patch += 1,
            Collision::InDomain => s.in_domain += 1,
            Collision::NumericalError => s.numerical_error += 1,
        }
    }
    s
}
