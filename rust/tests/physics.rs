//! Checks against closed-form Schwarzschild results, independent of the Python code.

use schwarzschild_rt::initial_conditions::{equatorial_momentum, initial_conditions, null_p_t};
use schwarzschild_rt::integrator::{integrate, ExitReason, IntegratorSettings};
use schwarzschild_rt::schwarzschild::BlackHole;
use std::f64::consts::PI;

/// Observer radius in units of `M`.
const R_OBS_PER_M: f64 = 30.0;

fn settings_for(bh: &BlackHole, steps: usize) -> IntegratorSettings {
    IntegratorSettings::new(steps, 0.01 * bh.mass, 0.01, 31.0 * bh.mass, bh.rs())
}

fn escapes(p_r: f64, p_ph: f64, bh: &BlackHole, settings: &IntegratorSettings) -> bool {
    let r_obs = R_OBS_PER_M * bh.mass;
    let f = bh.lapse(r_obs);
    let p_t = null_p_t(f, r_obs, PI / 2.0, p_r, 0.0, p_ph);
    let res = integrate(
        [0.0, r_obs, PI / 2.0, 0.0],
        [p_t, p_r, 0.0, p_ph],
        bh.rs(),
        settings,
    );
    res.exit == ExitReason::Escaped
}

/// Bisect the angle at which rays stop escaping, for a given momentum construction.
fn shadow_edge(bh: &BlackHole, momentum: impl Fn(f64) -> (f64, f64)) -> f64 {
    let settings = settings_for(bh, 100_000);
    let (mut lo, mut hi) = (0.10, 0.25);
    let (plo, phi_) = (momentum(lo), momentum(hi));
    assert!(!escapes(plo.0, plo.1, bh, &settings) && escapes(phi_.0, phi_.1, bh, &settings));
    for _ in 0..24 {
        let mid = 0.5 * (lo + hi);
        let (p_r, p_ph) = momentum(mid);
        if escapes(p_r, p_ph, bh, &settings) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

fn covariant_edge(bh: &BlackHole) -> f64 {
    let r_obs = R_OBS_PER_M * bh.mass;
    shadow_edge(bh, |alpha| {
        let (p_r, _, p_ph) = equatorial_momentum(alpha, r_obs, bh.mass);
        (p_r, p_ph)
    })
}

/// With the covariant momentum the integrated shadow edge sits at the analytic
/// critical angle `sin α_c = (3√3 M / r) √(1 − 2M/r)` (measured residual ≈5e-6 rad;
/// the 24-step bisection resolves ≈9e-9 rad).
#[test]
fn shadow_edge_matches_critical_angle() {
    let bh = BlackHole::new(1.0);
    let edge = covariant_edge(&bh);
    let expect = bh.capture_angle(R_OBS_PER_M);
    eprintln!("integrated shadow edge {edge:.6} rad, analytic {expect:.6} rad");
    assert!(
        (edge - expect).abs() < 1e-4,
        "edge {edge} vs analytic {expect}"
    );

    // The test can see sub-percent momentum errors: a 1 % error in p_φ moves the edge
    // by more than the tolerance.
    let r_obs = R_OBS_PER_M;
    let perturbed = shadow_edge(&bh, |alpha| {
        let (p_r, _, p_ph) = equatorial_momentum(alpha, r_obs, bh.mass);
        (p_r, p_ph * 1.01)
    });
    assert!(
        (perturbed - expect).abs() > 1e-4,
        "perturbed edge {perturbed}"
    );
}

/// The Python construction `p_r = n_r̂ √f` (instead of `n_r̂ / √f`) stretches every
/// angle by `1/f`, so its shadow edge lands at `atan(f · tan α_c)`: 6.6 % too small at
/// `r = 30`. This pins the size of the discrepancy the port removes.
#[test]
fn python_momentum_convention_shrinks_the_shadow_by_the_lapse() {
    let bh = BlackHole::new(1.0);
    let f = bh.lapse(R_OBS_PER_M);
    let edge = shadow_edge(&bh, |alpha| {
        let f_sqrt = f.sqrt();
        (-alpha.cos() * f_sqrt, alpha.sin() * R_OBS_PER_M)
    });
    let expect = (f * bh.capture_angle(R_OBS_PER_M).tan()).atan();
    eprintln!("python-convention shadow edge {edge:.6} rad, predicted {expect:.6} rad");
    assert!(
        (edge - expect).abs() < 1e-4,
        "edge {edge} vs predicted {expect}"
    );
    assert!(expect < 0.95 * bh.capture_angle(R_OBS_PER_M));
}

/// The integrator rescales every ray to units of `M`, so the shadow edge measured at
/// any mass equals the `M = 1` edge and the analytic angle. (Without the rescaling the
/// FANTASY coupling grows as `ω M²` and the edge at `M = 2.5` sat at 0.18 rad, at
/// `M = 5` the copies lost each other entirely.) Pins README difference 5 as well: the
/// Python's hard-coded `r_s = 2` in the metric derivative puts the `M = 0.5` edge at
/// 0.23 rad.
#[test]
fn shadow_edge_is_the_same_at_every_mass() {
    let unit = covariant_edge(&BlackHole::new(1.0));
    for mass in [0.5, 2.0, 2.5, 5.0] {
        let bh = BlackHole::new(mass);
        let edge = covariant_edge(&bh);
        let expect = bh.capture_angle(R_OBS_PER_M * mass);
        eprintln!("M = {mass}: edge {edge:.6}, analytic {expect:.6}");
        assert!(
            (edge - expect).abs() < 1e-4,
            "M = {mass}: {edge} vs {expect}"
        );
        assert!(
            (edge - unit).abs() < 1e-6,
            "M = {mass}: {edge} vs M=1 {unit}"
        );
    }
}

/// A near-critical ray integrated at masses 0.5, 2, 2.5 and 5 reproduces the `M = 1`
/// trajectory (lengths rescaled) to rounding, including its exit and step count.
#[test]
fn near_critical_ray_is_scale_invariant() {
    let alpha = 1.05 * BlackHole::new(1.0).capture_angle(R_OBS_PER_M);
    let reference = {
        let bh = BlackHole::new(1.0);
        let (p_r, p_th, p_ph) = equatorial_momentum(alpha, R_OBS_PER_M, 1.0);
        let p_t = null_p_t(
            bh.lapse(R_OBS_PER_M),
            R_OBS_PER_M,
            PI / 2.0,
            p_r,
            p_th,
            p_ph,
        );
        integrate(
            [0.0, R_OBS_PER_M, PI / 2.0, 0.0],
            [p_t, p_r, p_th, p_ph],
            bh.rs(),
            &settings_for(&bh, 100_000),
        )
    };
    assert_eq!(reference.exit, ExitReason::Escaped);
    for mass in [0.5, 2.0, 2.5, 5.0] {
        let bh = BlackHole::new(mass);
        let r_obs = R_OBS_PER_M * mass;
        let (p_r, p_th, p_ph) = equatorial_momentum(alpha, r_obs, mass);
        let p_t = null_p_t(bh.lapse(r_obs), r_obs, PI / 2.0, p_r, p_th, p_ph);
        let res = integrate(
            [0.0, r_obs, PI / 2.0, 0.0],
            [p_t, p_r, p_th, p_ph],
            bh.rs(),
            &settings_for(&bh, 100_000),
        );
        assert_eq!(res.exit, reference.exit, "M = {mass}");
        assert_eq!(res.n_steps, reference.n_steps, "M = {mass}");
        assert!(
            (res.q[1] / mass - reference.q[1]).abs() < 1e-9,
            "M = {mass}: r"
        );
        assert!(
            (res.q[0] / mass - reference.q[0]).abs() < 1e-9,
            "M = {mass}: t"
        );
        assert!((res.q[3] - reference.q[3]).abs() < 1e-9, "M = {mass}: φ");
    }
}

/// The full pipeline's initial conditions scale with the mass too.
#[test]
fn pixel_rays_scale_with_mass() {
    let a = initial_conditions([30.0, 0.0, 0.0], [24.0, 1.5, -0.7], 1.0);
    let b = initial_conditions([90.0, 0.0, 0.0], [72.0, 4.5, -2.1], 3.0);
    assert!((a.alpha0 - b.alpha0).abs() < 1e-14);
    assert!((a.beta - b.beta).abs() < 1e-14);
    assert!((a.p0[0] - b.p0[0]).abs() < 1e-14, "p_t is dimensionless");
    assert!((a.p0[1] - b.p0[1]).abs() < 1e-14, "p_r is dimensionless");
    assert!(
        (3.0 * a.p0[3] - b.p0[3]).abs() < 1e-12,
        "p_φ carries one length"
    );
}

// ---------------------------------------------------------------------------
// Accretion disk and camera pointing
// ---------------------------------------------------------------------------

use schwarzschild_rt::backend::{BatchIntegrator, CpuExact, CpuFast};
use schwarzschild_rt::camera::Camera;
use schwarzschild_rt::disk::{keplerian_omega, redshift, Disk};
use schwarzschild_rt::render::{render_curved, RenderConfig, RenderResult};
use schwarzschild_rt::shading::Collision;
use schwarzschild_rt::sky_patch::SkyPatch;

fn disk_scene(
    size: usize,
    inclination_deg: f64,
    azimuth_deg: f64,
    (yaw, pitch, roll): (f64, f64, f64),
) -> RenderConfig {
    let bh = BlackHole::new(1.0);
    let camera = Camera::with_rotation(
        [30.0, 0.0, 0.0],
        80f64.to_radians(),
        size,
        size,
        yaw.to_radians(),
        pitch.to_radians(),
        roll.to_radians(),
    )
    .unwrap();
    let normal = Disk::normal_from_angles(inclination_deg.to_radians(), azimuth_deg.to_radians());
    RenderConfig {
        camera,
        black_hole: bh,
        boundary_radius: 31.0,
        patch: SkyPatch::full_sky(),
        integrator: IntegratorSettings::new(20_000, 0.01, 0.01, 31.0, bh.rs()),
        disk: Some(Disk::new(normal, 6.0, 20.0, 6000.0, 1.0, 30.0).unwrap()),
        sample_pixels: vec![],
        max_trajectory_points: 10,
    }
}

fn render(cfg: &RenderConfig, backend: &dyn BatchIntegrator) -> RenderResult {
    render_curved(cfg, None, backend, || {}).unwrap()
}

/// Every disk hit, in every integrator, must give a physical redshift, and `g`
/// recomputed from the momentum the integrator actually carried to the hit must
/// agree with the closed form from the conserved initial momenta.
#[test]
fn every_disk_hit_has_a_physical_redshift_that_the_integrated_momentum_confirms() {
    let cfg = disk_scene(21, 55.0, 30.0, (6.0, -4.0, 0.0));
    let disk = cfg.disk.as_ref().unwrap();
    for backend in [&CpuExact as &dyn BatchIntegrator, &CpuFast] {
        let res = render(&cfg, backend);
        let mut hits = 0;
        let (mut g_min, mut g_max) = (f64::INFINITY, 0.0f64);
        for p in &res.photons {
            if p.shaded.collision == Collision::NumericalError {
                panic!("numerical error at ({}, {}): {:?}", p.i, p.j, p.shaded.disk);
            }
            let Some(d) = p.shaded.disk else { continue };
            hits += 1;
            assert_eq!(p.shaded.collision, Collision::Disk);
            assert!(d.g.is_finite() && d.g > 0.0);
            let frame = disk.ray_frame(p.init.beta);
            let b_z = -p.init.p0[3] * frame.n_z / p.init.p0[0];
            assert!(
                1.0 - keplerian_omega(1.0, d.r) * b_z > 0.0,
                "1 − Ω b_z must be positive"
            );
            // Contraction with the momentum at the hit (p_t, p_φ drift at the scheme's
            // second-order level) against the closed form from the initial momenta.
            let g_hit = redshift(1.0, d.r, p.result.p[0], p.result.p[3], frame.n_z, 30.0);
            assert!(
                (g_hit - d.g).abs() < 1e-4 * d.g,
                "({}, {}): {g_hit} vs {}",
                p.i,
                p.j,
                d.g
            );
            // The hit lies in the disk plane and in the band.
            assert!(schwarzschild_rt::coords::dot(disk.normal, d.position).abs() < 1e-9);
            assert!((6.0..=20.0).contains(&d.r));
            g_min = g_min.min(d.g);
            g_max = g_max.max(d.g);
        }
        assert!(hits > 60, "{hits} hits with {}", backend.name());
        assert!(
            g_max > 1.0 && g_min < 0.9,
            "Doppler both ways: g in [{g_min}, {g_max}]"
        );
    }
}

/// Reversing the normal reverses the sense of rotation: for an edge-on disk the
/// mirror pixel sees the same hit radius and the same `g`.
#[test]
fn reversing_the_disk_normal_mirrors_the_image() {
    let n = 21;
    let a = render(&disk_scene(n, 90.0, 0.0, (0.0, 0.0, 0.0)), &CpuExact);
    let b = render(&disk_scene(n, 90.0, 180.0, (0.0, 0.0, 0.0)), &CpuExact);
    let mut compared = 0;
    for i in 0..n {
        for j in 0..n {
            let pa = &a.photons[i * n + j];
            let pb = &b.photons[i * n + (n - 1 - j)];
            assert_eq!(pa.shaded.collision, pb.shaded.collision, "({i},{j})");
            if let (Some(da), Some(db)) = (pa.shaded.disk, pb.shaded.disk) {
                assert!(
                    (da.r - db.r).abs() < 1e-9 && (da.g - db.g).abs() < 1e-9,
                    "({i},{j})"
                );
                compared += 1;
            }
        }
    }
    assert!(compared > 15, "{compared}");
    // And the left side approaches for the +ẑ normal (azimuth 0): its Doppler factor
    // lifts g above the purely gravitational value at the same radius, the right side's
    // lowers it below. (g itself can be below 1 on both sides near the inner edge, where
    // the gravitational redshift wins.)
    let gravitational = |r: f64| (1.0 - 3.0 / r).sqrt() / (1.0 - 2.0 / 30.0f64).sqrt();
    let side = |left: bool| -> Vec<(f64, f64)> {
        a.photons
            .iter()
            .filter(|p| if left { p.j < n / 2 } else { p.j > n / 2 })
            .filter_map(|p| p.shaded.disk.map(|d| (d.g, gravitational(d.r))))
            .collect()
    };
    let (left, right) = (side(true), side(false));
    assert!(!left.is_empty() && !right.is_empty());
    assert!(left.iter().all(|(g, g0)| g > g0), "{left:?}");
    assert!(right.iter().all(|(g, g0)| g < g0), "{right:?}");
}

/// Rotating the scene about the x axis is an exact symmetry with the observer fixed:
/// a disk at azimuth α seen by an unrolled camera is the disk at azimuth 0 seen by a
/// camera rolled by α. Pins the disk's azimuth handedness against the camera's roll.
#[test]
fn disk_azimuth_and_camera_roll_are_the_same_rotation() {
    let n = 21;
    let a = render(&disk_scene(n, 55.0, 30.0, (0.0, 0.0, 0.0)), &CpuFast);
    let b = render(&disk_scene(n, 55.0, 0.0, (0.0, 0.0, 30.0)), &CpuFast);
    let mut compared = 0;
    for (pa, pb) in a.photons.iter().zip(&b.photons) {
        assert_eq!(
            pa.shaded.collision, pb.shaded.collision,
            "({}, {})",
            pa.i, pa.j
        );
        if let (Some(da), Some(db)) = (pa.shaded.disk, pb.shaded.disk) {
            assert!(
                (da.r - db.r).abs() < 1e-9 && (da.g - db.g).abs() < 1e-9,
                "({}, {})",
                pa.i,
                pa.j
            );
            compared += 1;
        }
    }
    assert!(compared > 60, "{compared}");
    // The rotation is not a no-op: azimuth 30 differs from azimuth 0 unrolled.
    let c = render(&disk_scene(n, 55.0, 0.0, (0.0, 0.0, 0.0)), &CpuFast);
    assert!(a
        .photons
        .iter()
        .zip(&c.photons)
        .any(|(x, y)| x.shaded.collision != y.shaded.collision));
}

/// Turning the camera moves the hole and the disk together: the pixel directions of a
/// yawed/pitched camera are the rigid rotation of the unrotated ones, so a rotated render
/// contains the same scene content as the unrotated render read at the rotated directions.
#[test]
fn a_turned_camera_sees_the_same_scene_from_the_same_place() {
    // Yaw by exactly the angle of a pixel column in a narrow, finely sampled view is not
    // a pixel permutation (pinhole tangents), so compare through the closed form instead:
    // every disk hit in the turned view obeys the same g(r, β, p) relation, and the
    // hole stays where the camera test says it goes.
    let n = 21;
    let cfg = disk_scene(n, 0.0, 0.0, (20.0, 0.0, 0.0));
    let res = render(&cfg, &CpuExact);
    let captured: Vec<(usize, usize)> = res
        .photons
        .iter()
        .filter(|p| p.shaded.collision == Collision::BlackHole)
        .map(|p| (p.i, p.j))
        .collect();
    assert!(!captured.is_empty());
    assert!(
        captured.iter().all(|&(_, j)| j > n / 2),
        "yaw left moves the hole right: {captured:?}"
    );
    for p in &res.photons {
        if let Some(d) = p.shaded.disk {
            let expect = (1.0 - 3.0 / d.r).sqrt() / (1.0 - 2.0 / 30.0f64).sqrt();
            assert!(
                (d.g - expect).abs() < 1e-14,
                "face-on g is pointing-independent"
            );
        }
    }
}
