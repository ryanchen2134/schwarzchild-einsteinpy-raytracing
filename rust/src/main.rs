use anyhow::{bail, ensure, Context, Result};
use clap::{Parser, ValueEnum};
use schwarzschild_rt::backend::{cuda_compiled_in, try_cuda, BatchIntegrator, CpuExact, CpuFast};
use schwarzschild_rt::camera::Camera;
use schwarzschild_rt::disk::Disk;
use schwarzschild_rt::flat::render_flat;
use schwarzschild_rt::integrator::IntegratorSettings;
use schwarzschild_rt::output::{summarize, write_photon_csv, write_png, write_trajectories_csv};
use schwarzschild_rt::render::{render_curved, RenderConfig};
use schwarzschild_rt::sampling::choose_sample_pixels;
use schwarzschild_rt::schwarzschild::BlackHole;
use schwarzschild_rt::shading::NUMERICAL_ERROR_RADIUS;
use schwarzschild_rt::sky_patch::SkyPatch;
use schwarzschild_rt::texture::Texture;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum BackendChoice {
    /// CUDA if the binary was built with `--features cuda` and a device is present, else CPU.
    Auto,
    /// The CPU: the four-lane SIMD integrator, or the scalar one with `--exact`.
    Cpu,
    /// A CUDA device (requires a build with `--features cuda`).
    Cuda,
}

/// Schwarzschild black-hole ray tracer (G = c = 1; lengths in the unit of --bh-mass).
#[derive(Parser, Debug)]
#[command(
    name = "schwarzschild-rt",
    version,
    about,
    allow_negative_numbers = true
)]
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
    /// Affine-parameter step size, in the same length unit as the radii.
    #[arg(long, default_value_t = 0.01)]
    delta: f64,
    /// FANTASY coupling between the two phase-space copies; the mixing angle per step
    /// is 2·omega·delta/M.
    #[arg(long, default_value_t = 0.01)]
    omega: f64,
    /// Black-hole mass M.
    #[arg(long, default_value_t = 1.0)]
    bh_mass: f64,
    /// Radius at which rays count as escaped (must exceed --observer-distance and stay
    /// below 100, the radius beyond which a ray is reported as a numerical error).
    #[arg(long, default_value_t = 31.0)]
    boundary_radius: f64,
    /// Observer distance from the black hole, on the +x axis (must exceed 3M).
    #[arg(long, default_value_t = 30.0)]
    observer_distance: f64,
    /// Sky-patch centre θ in degrees (polar angle from +z).
    #[arg(long, default_value_t = 90.0)]
    bg_patch_center_theta: f64,
    /// Sky-patch centre φ in degrees (azimuth from +x; 180 is behind the black hole).
    #[arg(long, default_value_t = 180.0)]
    bg_patch_center_phi: f64,
    /// θ offset of the patch centre in degrees. θ is the polar angle from +z, so a
    /// positive value moves the patch down in the image, a negative one up.
    #[arg(long, default_value_t = 0.0)]
    bg_patch_center_theta_relobs: f64,
    /// φ offset of the patch centre in degrees. A positive value moves the patch to the
    /// observer's left (right is +y), a negative one to the right.
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
    #[arg(long, alias = "no-flat-trajectories")]
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
    /// Where to integrate the rays.
    #[arg(long, value_enum, default_value_t = BackendChoice::Auto)]
    backend: BackendChoice,
    /// On the CPU, use the scalar reference integrator (operation for operation the
    /// Python kernel, which it matches to rounding) instead of the SIMD one. Several
    /// times slower; the two give the same photon table (see README, "Backends and speed").
    #[arg(long)]
    exact: bool,
    /// CUDA device ordinal.
    #[arg(long, default_value_t = 0)]
    cuda_device: usize,
    /// Camera yaw in degrees, a right-handed turn about the camera's up axis: positive
    /// looks to the observer's left, so the black hole moves right in the image. The
    /// camera pans; it never moves off the +x axis.
    #[arg(long, default_value_t = 0.0)]
    yaw: f64,
    /// Camera pitch in degrees, about the camera's right axis after the yaw: positive
    /// looks up, so the black hole moves down in the image.
    #[arg(long, default_value_t = 0.0)]
    pitch: f64,
    /// Camera roll in degrees about the optical axis after yaw and pitch: positive
    /// turns the image counter-clockwise.
    #[arg(long, default_value_t = 0.0)]
    roll: f64,
    /// Render a thin, opaque, Keplerian accretion disk (README, "Accretion disk").
    #[arg(long)]
    disk: bool,
    /// Angle in degrees between the disk normal and the observer's position vector +x:
    /// 0 face-on, 90 edge-on, 180 face-on with the opposite sense of rotation. Measured
    /// against the observer's position, not the optical axis.
    #[arg(long, default_value_t = 75.0)]
    disk_inclination: f64,
    /// Right-handed rotation of the disk normal about +x, in degrees. At 0 the near side
    /// of an inclined disk is below the black hole and, edge-on, the left side approaches.
    #[arg(long, default_value_t = 0.0)]
    disk_azimuth: f64,
    /// Inner radius of the disk, in the same length unit as the other radii. Defaults to
    /// the ISCO, 6M, and may not be smaller.
    #[arg(long)]
    disk_inner: Option<f64>,
    /// Outer radius of the disk (default 20M); must stay below --observer-distance.
    #[arg(long)]
    disk_outer: Option<f64>,
    /// Peak effective temperature of the disk in kelvin. A false-colour parameter: it
    /// sets the hue, not the physics.
    #[arg(long, default_value_t = 6000.0)]
    disk_temperature: f64,
}

/// The disk the flags describe, if `--disk` is set.
fn build_disk(args: &Args) -> Result<Option<Disk>> {
    if !args.disk {
        return Ok(None);
    }
    ensure!(
        args.disk_inclination.is_finite() && (0.0..=180.0).contains(&args.disk_inclination),
        "--disk-inclination must be in [0, 180] degrees, got {}",
        args.disk_inclination
    );
    ensure!(
        args.disk_azimuth.is_finite(),
        "--disk-azimuth must be finite, got {}",
        args.disk_azimuth
    );
    let normal = Disk::normal_from_angles(
        args.disk_inclination.to_radians(),
        args.disk_azimuth.to_radians(),
    );
    let r_in = args.disk_inner.unwrap_or(6.0 * args.bh_mass);
    let r_out = args.disk_outer.unwrap_or(20.0 * args.bh_mass);
    Disk::new(
        normal,
        r_in,
        r_out,
        args.disk_temperature,
        args.bh_mass,
        args.observer_distance,
    )
    .map(Some)
    .context("--disk-inner / --disk-outer / --disk-temperature")
}

fn validate(args: &Args) -> Result<()> {
    ensure!(
        args.bh_mass.is_finite() && args.bh_mass > 0.0,
        "--bh-mass must be a positive finite number, got {}",
        args.bh_mass
    );
    let bh = BlackHole::new(args.bh_mass);
    let r_min = 3.0 * bh.mass;
    ensure!(
        args.observer_distance.is_finite() && args.observer_distance > r_min,
        "--observer-distance must be finite and exceed the photon sphere, 3M = {r_min}; got {}",
        args.observer_distance
    );
    ensure!(
        args.boundary_radius.is_finite() && args.boundary_radius > args.observer_distance,
        "--boundary-radius ({}) must exceed --observer-distance ({})",
        args.boundary_radius,
        args.observer_distance
    );
    ensure!(
        args.boundary_radius < NUMERICAL_ERROR_RADIUS,
        "--boundary-radius ({}) must stay below the numerical-error radius {NUMERICAL_ERROR_RADIUS}",
        args.boundary_radius
    );
    ensure!(
        args.delta.is_finite() && args.delta > 0.0,
        "--delta must be a positive finite number, got {}",
        args.delta
    );
    ensure!(
        args.omega.is_finite(),
        "--omega must be finite, got {}",
        args.omega
    );
    ensure!(args.steps > 0, "--steps must be at least 1");
    ensure!(
        args.fov.is_finite() && args.fov > 0.0 && args.fov < 180.0,
        "--fov must be in (0, 180) degrees, got {}",
        args.fov
    );
    ensure!(args.size > 0, "--size must be at least 1");
    ensure!(
        args.max_trajectory_points > 0,
        "--max-trajectory-points must be at least 1"
    );
    if args.exact && args.backend == BackendChoice::Cuda {
        bail!(
            "--exact selects the scalar CPU integrator; it cannot be combined with --backend cuda"
        );
    }
    for (name, v) in [
        ("--yaw", args.yaw),
        ("--pitch", args.pitch),
        ("--roll", args.roll),
    ] {
        ensure!(
            v.is_finite(),
            "{name} must be a finite angle in degrees, got {v}"
        );
    }
    build_disk(args)?;
    Ok(())
}

fn choose_backend(args: &Args) -> Result<Box<dyn BatchIntegrator>> {
    let cpu: Box<dyn BatchIntegrator> = if args.exact {
        Box::new(CpuExact)
    } else {
        Box::new(CpuFast)
    };
    match args.backend {
        BackendChoice::Cpu => Ok(cpu),
        BackendChoice::Cuda => {
            if !cuda_compiled_in() {
                bail!("this binary was built without the CUDA backend; rebuild with `cargo build --release --features cuda-12` (or cuda-11)");
            }
            match try_cuda(args.cuda_device)? {
                Some(dev) => Ok(dev),
                None => bail!(
                    "--backend cuda: no CUDA driver library or no CUDA device on this machine"
                ),
            }
        }
        BackendChoice::Auto => {
            if args.exact {
                return Ok(cpu);
            }
            match try_cuda(args.cuda_device) {
                Ok(Some(dev)) => Ok(dev),
                Ok(None) => Ok(cpu),
                Err(e) => {
                    eprintln!("CUDA backend unavailable ({e:#}); using the CPU");
                    Ok(cpu)
                }
            }
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    validate(&args)?;
    if let Some(t) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()?;
    }

    let camera = Camera::with_rotation(
        [args.observer_distance, 0.0, 0.0],
        args.fov.to_radians(),
        args.size,
        args.size,
        args.yaw.to_radians(),
        args.pitch.to_radians(),
        args.roll.to_radians(),
    )?;
    let bh = BlackHole::new(args.bh_mass);
    let disk = build_disk(&args)?;
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
    let sample_pixels = choose_sample_pixels(args.seed, camera.n_pixels(), args.n_sample_rays)
        .with_context(|| {
            format!(
                "--n-sample-rays {} exceeds the {} pixels of a {}×{} image (pass a smaller value or 0)",
                args.n_sample_rays,
                camera.n_pixels(),
                camera.width,
                camera.height
            )
        })?;
    let backend = choose_backend(&args)?;
    std::fs::create_dir_all(&args.out_dir)
        .with_context(|| format!("cannot create {}", args.out_dir.display()))?;

    let n = camera.n_pixels();
    eprintln!(
        "{}×{} rays, fov {}°, yaw/pitch/roll {}/{}/{}°, observer at r = {}, M = {}, steps ≤ {}, δ = {}, ω = {}, texture {}, disk {}, backend {}",
        args.size,
        args.size,
        args.fov,
        args.yaw,
        args.pitch,
        args.roll,
        args.observer_distance,
        args.bh_mass,
        args.steps,
        args.delta,
        args.omega,
        texture
            .as_ref()
            .map_or("none".to_string(), |t| format!("{}×{}", t.width, t.height)),
        disk.as_ref().map_or("none".to_string(), |d| format!(
            "inclination {}°, azimuth {}°, r {}–{}, T_peak {} K",
            args.disk_inclination, args.disk_azimuth, d.r_in, d.r_out, d.peak_temperature
        )),
        backend.name()
    );

    if !args.no_flat {
        let t0 = Instant::now();
        let flat = render_flat(
            &camera,
            args.boundary_radius,
            &patch,
            texture.as_ref(),
            disk.as_ref(),
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
        disk,
        sample_pixels,
        max_trajectory_points: args.max_trajectory_points,
    };
    let done = AtomicUsize::new(0);
    let tick = (n / 20).max(1);
    let t0 = Instant::now();
    let result = render_curved(&cfg, texture.as_ref(), backend.as_ref(), || {
        let k = done.fetch_add(1, Ordering::Relaxed) + 1;
        if k.is_multiple_of(tick) || k == n {
            eprintln!("integrated {k}/{n} rays ({:.1?})", t0.elapsed());
        }
    })?;
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
    println!("  Hit disk:         {}", s.disk);
    println!("  Numerical error:  {}", s.numerical_error);
    println!("Outputs in {}", args.out_dir.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(extra: &[&str]) -> Args {
        let mut argv = vec!["schwarzschild-rt"];
        argv.extend_from_slice(extra);
        Args::try_parse_from(argv).unwrap()
    }

    #[test]
    fn negative_offsets_parse() {
        let a = parse(&[
            "--bg-patch-center-phi-relobs",
            "-10",
            "--bg-patch-center-theta-relobs",
            "-2.5",
        ]);
        assert_eq!(a.bg_patch_center_phi_relobs, -10.0);
        assert_eq!(a.bg_patch_center_theta_relobs, -2.5);
        assert!(Args::try_parse_from(["x", "--size", "-3"]).is_err());
    }

    #[test]
    fn python_flat_flag_spelling_is_accepted() {
        assert!(parse(&["--no-flat-trajectories"]).no_flat);
        assert!(parse(&["--no-flat"]).no_flat);
        assert!(!parse(&[]).no_flat);
    }

    #[test]
    fn backend_flags_parse_and_combine_sensibly() {
        assert_eq!(parse(&[]).backend, BackendChoice::Auto);
        assert_eq!(parse(&["--backend", "cpu"]).backend, BackendChoice::Cpu);
        assert_eq!(
            parse(&["--backend", "cuda", "--cuda-device", "1"]).cuda_device,
            1
        );
        assert!(validate(&parse(&["--backend", "cuda", "--exact"])).is_err());
        assert!(validate(&parse(&["--backend", "cpu", "--exact"])).is_ok());
        let cpu = choose_backend(&parse(&["--backend", "cpu"])).unwrap();
        assert!(cpu.name().starts_with("cpu-simd"));
        let exact = choose_backend(&parse(&["--backend", "cpu", "--exact"])).unwrap();
        assert!(exact.name().starts_with("cpu-exact"));
        if !cuda_compiled_in() {
            let err = choose_backend(&parse(&["--backend", "cuda"]))
                .err()
                .expect("cuda backend must be refused without the feature");
            assert!(err.to_string().contains("--features cuda"));
        }
    }

    #[test]
    fn validation_rejects_bad_physics_inputs() {
        assert!(validate(&parse(&[])).is_ok());
        for bad in [
            ["--delta", "0"],
            ["--delta", "-0.01"],
            ["--delta", "nan"],
            ["--omega", "inf"],
            ["--bh-mass", "0"],
            ["--bh-mass", "-1"],
            ["--observer-distance", "2.5"],
            ["--observer-distance", "3"],
            ["--boundary-radius", "30"],
            ["--boundary-radius", "100"],
            ["--steps", "0"],
            ["--fov", "180"],
            ["--size", "0"],
            ["--max-trajectory-points", "0"],
        ] {
            assert!(
                validate(&parse(&bad)).is_err(),
                "{bad:?} should be rejected"
            );
        }
        assert!(validate(&parse(&[
            "--bh-mass",
            "2",
            "--observer-distance",
            "60",
            "--boundary-radius",
            "62"
        ]))
        .is_ok());
    }

    #[test]
    fn disk_and_pointing_flags_are_validated() {
        assert!(validate(&parse(&["--disk"])).is_ok());
        assert!(validate(&parse(&["--yaw", "-15", "--pitch", "10", "--roll", "400"])).is_ok());
        let d = build_disk(&parse(&[
            "--disk",
            "--bh-mass",
            "2",
            "--observer-distance",
            "60",
            "--boundary-radius",
            "62",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(
            (d.r_in, d.r_out),
            (12.0, 40.0),
            "defaults scale with the mass"
        );
        assert!(build_disk(&parse(&[])).unwrap().is_none());
        for bad in [
            vec!["--disk", "--disk-inner", "5.9"],
            vec!["--disk", "--disk-outer", "30"],
            vec!["--disk", "--disk-inner", "10", "--disk-outer", "10"],
            vec!["--disk", "--disk-temperature", "0"],
            vec!["--disk", "--disk-inclination", "181"],
            vec!["--disk", "--disk-inclination", "-1"],
            vec!["--disk", "--disk-azimuth", "nan"],
            vec!["--yaw", "nan"],
            vec!["--pitch", "inf"],
        ] {
            assert!(
                validate(&parse(&bad)).is_err(),
                "{bad:?} should be rejected"
            );
        }
        // Without --disk the disk flags are inert.
        assert!(validate(&parse(&["--disk-inner", "1"])).is_ok());
    }
}
