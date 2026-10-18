//! The CUDA backend against the scalar reference, run only in a `cuda-*` build and
//! only when a device is present (otherwise each test reports that it skipped). The
//! device's `sin`/`cos` differ from the CPU's by a few ulp (see `cuda.rs`), so radii and
//! redshifts are compared to 1e-9 and a class flip is tolerated only on a pixel that
//! borders a different class, at most two per image.
#![cfg(feature = "cuda")]

use schwarzschild_rt::backend::{try_cuda, BatchIntegrator, CpuExact};
use schwarzschild_rt::camera::Camera;
use schwarzschild_rt::cuda::STEPS_PER_LAUNCH;
use schwarzschild_rt::disk::Disk;
use schwarzschild_rt::integrator::{IntegratorSettings, Ray};
use schwarzschild_rt::render::{render_curved, RenderConfig, RenderResult};
use schwarzschild_rt::schwarzschild::BlackHole;
use schwarzschild_rt::shading::Collision;
use schwarzschild_rt::sky_patch::SkyPatch;

fn device() -> Option<Box<dyn BatchIntegrator>> {
    match try_cuda(0) {
        Ok(Some(dev)) => Some(dev),
        Ok(None) => {
            eprintln!("no CUDA device: skipped");
            None
        }
        Err(e) => panic!("CUDA is present but unusable: {e:#}"),
    }
}

fn scene(delta: f64, steps: usize) -> RenderConfig {
    let bh = BlackHole::new(1.0);
    RenderConfig {
        camera: Camera::with_rotation(
            [30.0, 0.0, 0.0],
            80f64.to_radians(),
            21,
            21,
            6f64.to_radians(),
            (-4f64).to_radians(),
            0.0,
        )
        .unwrap(),
        black_hole: bh,
        boundary_radius: 31.0,
        patch: SkyPatch::full_sky(),
        integrator: IntegratorSettings::new(steps, delta, 0.01, 31.0, bh.rs()),
        disk: Some(
            Disk::new(
                Disk::normal_from_angles(55f64.to_radians(), 30f64.to_radians()),
                6.0,
                20.0,
                6000.0,
                1.0,
                30.0,
            )
            .unwrap(),
        ),
        sample_pixels: vec![],
        max_trajectory_points: 10,
    }
}

fn compare(cpu: &RenderResult, gpu: &RenderResult, n: usize) -> usize {
    let mut flips = 0;
    let mut hits = 0;
    for (k, (a, b)) in cpu.photons.iter().zip(&gpu.photons).enumerate() {
        if a.shaded.collision != b.shaded.collision {
            flips += 1;
            let (i, j) = (k / n, k % n);
            let border = (i.saturating_sub(1)..=(i + 1).min(n - 1)).any(|ni| {
                (j.saturating_sub(1)..=(j + 1).min(n - 1))
                    .any(|nj| cpu.photons[ni * n + nj].shaded.collision != a.shaded.collision)
            });
            assert!(
                border,
                "({i},{j}): {:?} vs {:?}",
                a.shaded.collision, b.shaded.collision
            );
            continue;
        }
        assert_eq!(
            a.result.plane_crossings, b.result.plane_crossings,
            "pixel {k}"
        );
        if let (Some(da), Some(db)) = (a.shaded.disk, b.shaded.disk) {
            assert!(
                (da.r - db.r).abs() < 1e-9 * da.r,
                "pixel {k}: {} vs {}",
                da.r,
                db.r
            );
            assert!((da.phi - db.phi).abs() < 1e-9, "pixel {k}");
            assert!((da.g - db.g).abs() < 1e-9 * da.g, "pixel {k}");
            hits += 1;
        }
    }
    assert!(flips <= 2, "{flips} flips");
    hits
}

#[test]
fn cuda_disk_hits_match_the_scalar_integrator() {
    let Some(dev) = device() else { return };
    let cfg = scene(0.01, 20_000);
    let cpu = render_curved(&cfg, None, &CpuExact, || {}).unwrap();
    let gpu = render_curved(&cfg, None, dev.as_ref(), || {}).unwrap();
    let hits = compare(&cpu, &gpu, 21);
    assert!(hits > 60, "{hits}");
}

/// A small step puts most disk hits after the first launch, so the crossing count and
/// the detection state must survive the relaunch.
#[test]
fn cuda_disk_hits_survive_relaunches() {
    let Some(dev) = device() else { return };
    let cfg = scene(0.0025, 40_000);
    let cpu = render_curved(&cfg, None, &CpuExact, || {}).unwrap();
    let late = cpu
        .photons
        .iter()
        .filter(|p| p.shaded.collision == Collision::Disk && p.result.n_steps > STEPS_PER_LAUNCH)
        .count();
    assert!(late > 30, "{late} hits after the first launch");
    let gpu = render_curved(&cfg, None, dev.as_ref(), || {}).unwrap();
    compare(&cpu, &gpu, 21);
}

/// Without a disk band the device ignores any azimuth a ray carries.
#[test]
fn cuda_ignores_disk_azimuths_without_a_band() {
    let Some(dev) = device() else { return };
    let cfg = scene(0.01, 20_000);
    let settings = IntegratorSettings {
        disk: None,
        ..cfg.integrator
    };
    let disk = cfg.disk.unwrap();
    let rays: Vec<Ray> = (0..cfg.camera.n_pixels())
        .map(|flat| {
            let (i, j) = cfg.camera.pixel_of(flat);
            let init = schwarzschild_rt::initial_conditions::initial_conditions(
                cfg.camera.observer,
                cfg.camera.pixel_position(i, j),
                1.0,
            );
            Ray::with_disk(init.q0, init.p0, disk.ray_frame(init.beta).phi_d)
        })
        .collect();
    let plain: Vec<Ray> = rays.iter().map(|r| Ray::new(r.q0, r.p0)).collect();
    let with = dev.integrate_batch(&rays, 2.0, &settings, &|| {}).unwrap();
    let without = dev.integrate_batch(&plain, 2.0, &settings, &|| {}).unwrap();
    for (a, b) in with.iter().zip(&without) {
        assert_eq!(a.exit, b.exit);
        assert_eq!(a.q, b.q);
        assert_eq!(a.plane_crossings, 0);
        assert!(a.disk_hit.is_none());
    }
}
