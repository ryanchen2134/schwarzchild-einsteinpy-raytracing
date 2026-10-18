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
use schwarzschild_rt::coords::{cross, dot, mat_vec, norm, normalize, rot_x, Vec3};
use schwarzschild_rt::disk::{keplerian_omega, Disk};
use schwarzschild_rt::integrator::{integrate_ray, DiskBand, Ray};
use schwarzschild_rt::render::{render_curved, RenderConfig, RenderResult};
use schwarzschild_rt::shading::Collision;
use schwarzschild_rt::sky_patch::SkyPatch;

fn disk_scene_at(
    mass: f64,
    size: usize,
    inclination_deg: f64,
    azimuth_deg: f64,
    (yaw, pitch, roll): (f64, f64, f64),
) -> RenderConfig {
    let bh = BlackHole::new(mass);
    let camera = Camera::with_rotation(
        [30.0 * mass, 0.0, 0.0],
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
        boundary_radius: 31.0 * mass,
        patch: SkyPatch::full_sky(),
        integrator: IntegratorSettings::new(20_000, 0.01 * mass, 0.01, 31.0 * mass, bh.rs()),
        disk: Some(Disk::new(normal, 6.0 * mass, 20.0 * mass, 6000.0, mass, 30.0 * mass).unwrap()),
        sample_pixels: vec![],
        max_trajectory_points: 10,
    }
}

fn disk_scene(
    size: usize,
    inclination_deg: f64,
    azimuth_deg: f64,
    angles: (f64, f64, f64),
) -> RenderConfig {
    disk_scene_at(1.0, size, inclination_deg, azimuth_deg, angles)
}

fn render(cfg: &RenderConfig, backend: &dyn BatchIntegrator) -> RenderResult {
    render_curved(cfg, None, backend, || {}).unwrap()
}

/// Two renders that are equal only to rounding (different trig arguments, mirrored
/// pixels) may disagree on the class of a pixel within rounding of a boundary. Such a
/// flip is tolerated only on a pixel bordering one of a different class in the first
/// render, and on at most `max` pixels.
fn check_flips(n: usize, a: &[Collision], b: &[Collision], max: usize, what: &str) {
    let mut flips = 0;
    for i in 0..n {
        for j in 0..n {
            let k = i * n + j;
            if a[k] == b[k] {
                continue;
            }
            flips += 1;
            let on_boundary = (i.saturating_sub(1)..=(i + 1).min(n - 1)).any(|ni| {
                (j.saturating_sub(1)..=(j + 1).min(n - 1)).any(|nj| a[ni * n + nj] != a[k])
            });
            assert!(
                on_boundary,
                "{what}: ({i},{j}) is {:?} vs {:?} away from any boundary",
                a[k], b[k]
            );
        }
    }
    assert!(flips <= max, "{what}: {flips} flips");
}

/// Independent truth for `g` (shares no code with `disk.rs`): the emitter moves at the
/// static-frame speed `v = √(M/(r − 2M))` along `n × x̂` and the photon travels along
/// the unit vector `k̂`, so `g = √f √(1 − v²) / (√f_obs (1 − v·k̂))`.
fn kinematic_redshift(mass: f64, r_obs: f64, n: Vec3, x_hit: Vec3, k_hat: Vec3) -> f64 {
    let r = norm(x_hit);
    let f = 1.0 - 2.0 * mass / r;
    let v = (mass / (r - 2.0 * mass)).sqrt();
    let e_v = normalize(cross(n, x_hit));
    f.sqrt() * (1.0 - v * v).sqrt()
        / ((1.0 - 2.0 * mass / r_obs).sqrt() * (1.0 - v * dot(e_v, k_hat)))
}

/// Every disk hit, in both CPU integrators, has a physical `g`, and that `g` agrees with
/// the kinematic Doppler + gravitational factor built from the momentum the integrator
/// actually carried to the hit (the photon is the time reverse of the traced ray) and
/// the lab-frame normal: this ties the closed form's `b_z` sign, the frame rotation
/// and the backward-tracing convention to the integrated rays, on an oblique disk.
#[test]
fn every_disk_hit_agrees_with_the_kinematic_redshift_of_its_integrated_photon() {
    let cfg = disk_scene(21, 55.0, 30.0, (6.0, -4.0, 0.0));
    let disk = cfg.disk.as_ref().unwrap();
    let delta = cfg.integrator.delta;
    for backend in [&CpuExact as &dyn BatchIntegrator, &CpuFast] {
        let res = render(&cfg, backend);
        let (mut hits, mut blue, mut red) = (0, 0, 0);
        let mut worst = 0.0f64;
        for p in &res.photons {
            assert_ne!(
                p.shaded.collision,
                Collision::NumericalError,
                "({}, {})",
                p.i,
                p.j
            );
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
            // The hit lies in the disk plane, in the band, and one step from the state
            // where the integrator stopped.
            assert!(dot(disk.normal, d.position).abs() < 1e-9);
            assert!((6.0..=20.0).contains(&d.r));
            let q = p.result.q;
            let stop = mat_vec(
                &rot_x(p.init.beta),
                [q[1] * q[3].cos(), q[1] * q[3].sin(), 0.0],
            );
            let gap = norm([
                stop[0] - d.position[0],
                stop[1] - d.position[1],
                stop[2] - d.position[2],
            ]);
            assert!(gap < 2.0 * delta, "({}, {}): {gap}", p.i, p.j);
            // Photon direction at the stop: k = −p; static orthonormal components
            // k^r̂ = √f k_r, k^φ̂ = k_φ / r in the ray frame, rotated to the lab.
            let f = 1.0 - 2.0 / q[1];
            let (kr, kph) = (-f.sqrt() * p.result.p[1], -p.result.p[3] / q[1]);
            let (s, c) = q[3].sin_cos();
            let k_rot = [kr * c - kph * s, kr * s + kph * c, 0.0];
            let k_hat = normalize(mat_vec(&rot_x(p.init.beta), k_rot));
            let g_kin = kinematic_redshift(1.0, 30.0, disk.normal, d.position, k_hat);
            let rel = (g_kin - d.g).abs() / d.g;
            assert!(
                rel < 2e-3,
                "({}, {}): kinematic {g_kin} vs closed form {}",
                p.i,
                p.j,
                d.g
            );
            worst = worst.max(rel);
            // Doppler both ways, measured against the gravitational value at that radius.
            let g0 = (1.0 - 3.0 / d.r).sqrt() / (1.0 - 2.0 / 30.0f64).sqrt();
            if d.g > 1.05 * g0 {
                blue += 1;
            }
            if d.g < 0.95 * g0 {
                red += 1;
            }
        }
        eprintln!(
            "{}: {hits} hits, {blue} blueshifted, {red} redshifted, worst kinematic Δg/g {worst:e}",
            backend.name()
        );
        assert!(hits > 60, "{hits} hits with {}", backend.name());
        assert!(
            blue > 5 && red > 5,
            "Doppler both ways: {blue} blue, {red} red"
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
    let mirrored: Vec<_> = (0..n * n)
        .map(|k| b.photons[(k / n) * n + (n - 1 - k % n)].shaded.collision)
        .collect();
    let classes: Vec<_> = a.photons.iter().map(|p| p.shaded.collision).collect();
    check_flips(n, &classes, &mirrored, 2, "mirror");
    let mut compared = 0;
    for i in 0..n {
        for j in 0..n {
            let pa = &a.photons[i * n + j];
            let pb = &b.photons[i * n + (n - 1 - j)];
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
    let ca: Vec<_> = a.photons.iter().map(|p| p.shaded.collision).collect();
    let cb: Vec<_> = b.photons.iter().map(|p| p.shaded.collision).collect();
    check_flips(n, &ca, &cb, 2, "roll");
    let mut compared = 0;
    for (pa, pb) in a.photons.iter().zip(&b.photons) {
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
    // The opposite roll is a different picture: the test can see the sign.
    let wrong = render(&disk_scene(n, 55.0, 0.0, (0.0, 0.0, -30.0)), &CpuFast);
    let differ = a
        .photons
        .iter()
        .zip(&wrong.photons)
        .filter(|(x, y)| x.shaded.collision != y.shaded.collision)
        .count();
    assert!(differ > 20, "{differ}");
}

/// Turning the camera turns the view, not the scene. A face-on disk is axisymmetric
/// about the line of sight, so the hit radius of a ray depends only on its angle from
/// the hole; every hit of a yawed and pitched render must equal the hit of a ray with
/// the same angle sent in the x–y plane by the unturned pipeline.
#[test]
fn a_turned_camera_sees_the_same_scene_from_the_same_place() {
    let n = 21;
    let cfg = disk_scene(n, 0.0, 0.0, (20.0, -10.0, 0.0));
    let disk = cfg.disk.as_ref().unwrap();
    let settings = cfg.integrator.with_disk(DiskBand {
        r_in: 6.0,
        r_out: 20.0,
    });
    let res = render(&cfg, &CpuExact);
    let hole: Vec<(usize, usize)> = res
        .photons
        .iter()
        .filter(|p| p.shaded.collision == Collision::BlackHole)
        .map(|p| (p.i, p.j))
        .collect();
    assert!(!hole.is_empty());
    let mean = |pick: fn(&(usize, usize)) -> usize| {
        hole.iter().map(pick).sum::<usize>() as f64 / hole.len() as f64
    };
    let (hole_row, hole_col) = (mean(|p| p.0), mean(|p| p.1));
    assert!(
        hole_col > 12.0 && hole_row > 11.0,
        "yaw left and pitch down put the hole right of and above the centre: {hole:?}"
    );
    let (mut hits, mut left, mut right, mut mismatched) = (0, 0, 0, 0);
    for p in &res.photons {
        let a = p.init.alpha0;
        let flat = initial_conditions([30.0, 0.0, 0.0], [30.0 - a.cos(), a.sin(), 0.0], 1.0);
        assert!((flat.alpha0 - a).abs() < 1e-14);
        let ray = Ray::with_disk(flat.q0, flat.p0, disk.ray_frame(flat.beta).phi_d);
        let r = integrate_ray(&ray, 2.0, &settings);
        match (p.shaded.disk, r.disk_hit) {
            (Some(d), Some(h)) => {
                assert!(
                    (d.r - h.r).abs() < 1e-9,
                    "({}, {}): {} vs {}",
                    p.i,
                    p.j,
                    d.r,
                    h.r
                );
                hits += 1;
                if (p.j as f64) < hole_col {
                    left += 1
                } else if (p.j as f64) > hole_col {
                    right += 1
                }
            }
            (None, None) => assert_eq!(p.result.exit, r.exit, "({}, {})", p.i, p.j),
            _ => mismatched += 1,
        }
    }
    assert!(
        mismatched <= 1,
        "{mismatched} pixels classified differently"
    );
    assert!(
        hits > 40 && left > 5 && right > 5,
        "{hits} hits, {left} left and {right} right of the hole"
    );
}

/// The whole disk render is scale invariant: at M = 2 (observer, boundary, band and
/// step doubled) every pixel reproduces M = 1 bit for bit, lengths doubled. Exercises
/// the mass rescaling of the band and the hit, and the shading at a physical r_obs.
#[test]
fn the_disk_render_is_scale_invariant() {
    let n = 15;
    let angles = (6.0, -4.0, 10.0);
    let a = render(&disk_scene_at(1.0, n, 55.0, 30.0, angles), &CpuFast);
    let b = render(&disk_scene_at(2.0, n, 55.0, 30.0, angles), &CpuFast);
    let mut hits = 0;
    for (pa, pb) in a.photons.iter().zip(&b.photons) {
        assert_eq!(pa.shaded.collision, pb.shaded.collision);
        assert_eq!(pa.result.exit, pb.result.exit);
        assert_eq!(pa.result.plane_crossings, pb.result.plane_crossings);
        assert_eq!(pa.shaded.colour, pb.shaded.colour);
        assert_eq!((2.0 * pa.result.q[1]).to_bits(), pb.result.q[1].to_bits());
        if let (Some(da), Some(db)) = (pa.shaded.disk, pb.shaded.disk) {
            assert_eq!((2.0 * da.r).to_bits(), db.r.to_bits());
            assert_eq!(da.phi.to_bits(), db.phi.to_bits());
            assert_eq!(da.g.to_bits(), db.g.to_bits(), "g is dimensionless");
            assert_eq!(da.temperature.to_bits(), db.temperature.to_bits());
            for k in 0..3 {
                assert_eq!((2.0 * da.position[k]).to_bits(), db.position[k].to_bits());
            }
            hits += 1;
        }
    }
    assert!(hits > 30, "{hits}");
}
