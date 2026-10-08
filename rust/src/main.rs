use anyhow::{Context, Result};
use clap::Parser;
use schwarzschild_rt::camera::Camera;
use schwarzschild_rt::flat::render_flat;
use schwarzschild_rt::integrator::IntegratorSettings;
use schwarzschild_rt::output::{summarize, write_photon_csv, write_png, write_trajectories_csv};
use schwarzschild_rt::render::{render_curved, RenderConfig};
use schwarzschild_rt::sampling::choose_sample_pixels;
use schwarzschild_rt::schwarzschild::BlackHole;
use schwarzschild_rt::sky_patch::SkyPatch;
use schwarzschild_rt::texture::Texture;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// Schwarzschild black-hole ray tracer (G = c = 1, lengths in units of M).
#[derive(Parser, Debug)]
#[command(name = "schwarzschild-rt", version, about)]
struct Args {
    /// Image size (N×N pixels).
    #[arg(long, default_value_t = 200)]
    size: usize,
    /// Horizontal field of view in degrees.
    #[arg(long, default_value_t = 80.0)]
    fov: f64,
    /// Background texture (equirectangular). Omit for a black sky.
    #[arg(long)]
    background: Option<PathBuf>,
    /// Maximum integration steps per ray.
    #[arg(long, default_value_t = 200_000)]
    steps: usize,
    /// Affine-parameter step size.
    #[arg(long, default_value_t = 0.01)]
    delta: f64,
    /// FANTASY coupling between the two phase-space copies.
    #[arg(long, default_value_t = 0.01)]
    omega: f64,
    /// Black-hole mass M.
    #[arg(long, default_value_t = 1.0)]
    bh_mass: f64,
    /// Radius at which rays count as escaped.
    #[arg(long, default_value_t = 31.0)]
    boundary_radius: f64,
    /// Observer distance from the black hole, on the +x axis.
    #[arg(long, default_value_t = 30.0)]
    observer_distance: f64,
    /// Sky-patch centre θ in degrees.
    #[arg(long, default_value_t = 90.0)]
    bg_patch_center_theta: f64,
    /// Sky-patch centre φ in degrees.
    #[arg(long, default_value_t = 180.0)]
    bg_patch_center_phi: f64,
    /// θ offset of the patch centre in degrees (+ up / − down).
    #[arg(long, default_value_t = 0.0)]
    bg_patch_center_theta_relobs: f64,
    /// φ offset of the patch centre in degrees (+ right / − left).
    #[arg(long, default_value_t = 0.0)]
    bg_patch_center_phi_relobs: f64,
    /// Sky-patch extent in θ, degrees.
    #[arg(long, default_value_t = 180.0)]
    bg_patch_size_theta: f64,
    /// Sky-patch extent in φ, degrees.
    #[arg(long, default_value_t = 360.0)]
    bg_patch_size_phi: f64,
    /// Mirror the texture in θ.
    #[arg(long)]
    bg_flip_theta: bool,
    /// Mirror the texture in φ.
    #[arg(long)]
    bg_flip_phi: bool,
    /// Skip the no-gravity reference render.
    #[arg(long)]
    no_flat: bool,
    /// Number of rays whose full trajectories are written out.
    #[arg(long, default_value_t = 20)]
    n_sample_rays: usize,
    /// Seed for the choice of sample rays (random when omitted).
    #[arg(long)]
    seed: Option<u64>,
    /// Maximum points kept per sampled trajectory.
    #[arg(long, default_value_t = 1000)]
    max_trajectory_points: usize,
    /// Directory for all outputs.
    #[arg(long, default_value = "output")]
    out_dir: PathBuf,
    /// Worker threads (default: all cores).
    #[arg(long)]
    threads: Option<usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(t) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()?;
    }
    anyhow::ensure!(
        args.boundary_radius > args.observer_distance,
        "--boundary-radius must exceed --observer-distance"
    );

    let camera = Camera::new(
        [args.observer_distance, 0.0, 0.0],
        args.fov.to_radians(),
        args.size,
        args.size,
    )?;
    let bh = BlackHole::new(args.bh_mass);
    anyhow::ensure!(
        args.observer_distance > bh.rs(),
        "observer must sit outside the horizon"
    );
    let patch = SkyPatch::from_degrees(
        args.bg_patch_center_theta,
        args.bg_patch_center_phi,
        args.bg_patch_size_theta,
        args.bg_patch_size_phi,
        args.bg_patch_center_theta_relobs,
        args.bg_patch_center_phi_relobs,
        args.bg_flip_theta,
        args.bg_flip_phi,
    );
    let texture = match &args.background {
        Some(p) => Some(Texture::load(p)?),
        None => None,
    };
    let sample_pixels = choose_sample_pixels(args.seed, camera.n_pixels(), args.n_sample_rays)?;
    std::fs::create_dir_all(&args.out_dir)
        .with_context(|| format!("cannot create {}", args.out_dir.display()))?;

    let n = camera.n_pixels();
    eprintln!(
        "{}×{} rays, fov {}°, observer at r = {}, M = {}, steps ≤ {}, δ = {}, ω = {}, texture {}",
        args.size,
        args.size,
        args.fov,
        args.observer_distance,
        args.bh_mass,
        args.steps,
        args.delta,
        args.omega,
        texture
            .as_ref()
            .map_or("none".to_string(), |t| format!("{}×{}", t.width, t.height))
    );

    if !args.no_flat {
        let t0 = Instant::now();
        let flat = render_flat(
            &camera,
            args.boundary_radius,
            &patch,
            texture.as_ref(),
            &sample_pixels,
        );
        write_png(
            &args.out_dir.join("no_gravity.png"),
            camera.width,
            camera.height,
            &flat.image,
        )?;
        write_trajectories_csv(&args.out_dir.join("flat_rays.csv"), &flat.trajectories)?;
        eprintln!("flat render: {:.2?}", t0.elapsed());
    }

    let cfg = RenderConfig {
        camera,
        black_hole: bh,
        boundary_radius: args.boundary_radius,
        patch,
        integrator: IntegratorSettings::new(
            args.steps,
            args.delta,
            args.omega,
            args.boundary_radius,
            bh.rs(),
        ),
        sample_pixels,
        max_trajectory_points: args.max_trajectory_points,
    };
    let done = AtomicUsize::new(0);
    let tick = (n / 20).max(1);
    let t0 = Instant::now();
    let result = render_curved(&cfg, texture.as_ref(), || {
        let k = done.fetch_add(1, Ordering::Relaxed) + 1;
        if k.is_multiple_of(tick) || k == n {
            eprintln!("integrated {k}/{n} rays ({:.1?})", t0.elapsed());
        }
    });
    eprintln!("curved render: {:.2?}", t0.elapsed());

    write_png(
        &args.out_dir.join("manual_output.png"),
        result.width,
        result.height,
        &result.image,
    )?;
    write_photon_csv(&args.out_dir.join("photon_data.csv"), &result.photons)?;
    write_trajectories_csv(&args.out_dir.join("sampled_rays.csv"), &result.trajectories)?;

    let s = summarize(&result);
    println!("Photon summary:");
    println!("  Captured by BH:   {}", s.captured);
    println!("  Still in domain:  {}", s.in_domain);
    println!("  Escaped:          {}", s.escaped());
    println!("  Hit background:   {}", s.escaped_background);
    println!("  Numerical error:  {}", s.numerical_error);
    println!("Outputs in {}", args.out_dir.display());
    Ok(())
}
