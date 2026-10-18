//! Equivalence with the Python implementation (`simulation/`), using fixtures
//! captured from it under numba's CUDA simulator at commit 7dcfb9f
//! (`tests/fixtures/`, regenerated with `tests/fixtures/gen_golden.py`).

use schwarzschild_rt::backend::CpuFast;
use schwarzschild_rt::camera::Camera;
use schwarzschild_rt::coords::{cartesian_to_spherical, spherical_to_cartesian, TWO_PI};
use schwarzschild_rt::disk::{redshift, Disk};
use schwarzschild_rt::flat::render_flat;
use schwarzschild_rt::initial_conditions::{initial_conditions, null_p_t};
use schwarzschild_rt::integrator::{
    integrate, integrate_ray, integrate_trajectory, DiskBand, ExitReason, IntegratorSettings, Ray,
};
use schwarzschild_rt::integrator_fast::integrate_batch_fast;
use schwarzschild_rt::render::{render_curved, RenderConfig};
use schwarzschild_rt::schwarzschild::BlackHole;
use schwarzschild_rt::shading::{classify, unrotate_hit, Collision};
use schwarzschild_rt::sky_patch::SkyPatch;
use serde::Deserialize;
use std::f64::consts::PI;
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn load<T: for<'de> Deserialize<'de>>(name: &str) -> T {
    let text = std::fs::read_to_string(fixture(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn angle_diff(a: f64, b: f64) -> f64 {
    ((a - b + PI).rem_euclid(TWO_PI) - PI).abs()
}

#[derive(Deserialize)]
struct PyRay {
    i: usize,
    j: usize,
    pixel: [f64; 3],
    q0: [f64; 4],
    p0: [f64; 4],
    alpha0: f64,
    beta: f64,
    heading: [f64; 3],
}

#[derive(Deserialize)]
struct PyInitial {
    h: usize,
    w: usize,
    fov_deg: f64,
    observer: [f64; 3],
    mass: f64,
    rays: Vec<PyRay>,
}

fn python_initial() -> (Camera, PyInitial) {
    let py: PyInitial = load("initial_conditions.json");
    let cam = Camera::new(py.observer, py.fov_deg.to_radians(), py.w, py.h).unwrap();
    (cam, py)
}

#[test]
fn camera_and_initial_conditions_match_python() {
    let (cam, py) = python_initial();
    let bh = BlackHole::new(py.mass);
    for ray in &py.rays {
        let pixel = cam.pixel_position(ray.i, ray.j);
        for (k, (a, b)) in pixel.iter().zip(&ray.pixel).enumerate() {
            assert!(
                (a - b).abs() < 1e-13,
                "pixel ({},{}) axis {k}",
                ray.i,
                ray.j
            );
        }
        let init = initial_conditions(cam.observer, pixel, py.mass);
        for k in 0..4 {
            assert!((init.q0[k] - ray.q0[k]).abs() < 1e-14, "q0[{k}]");
        }
        assert!((init.beta - ray.beta).abs() < 1e-13, "beta");
        assert!(
            (init.alpha0 - ray.alpha0).abs() < 1e-12,
            "alpha0 {} vs {}",
            init.alpha0,
            ray.alpha0
        );
        let (hr, hth, hph) = cartesian_to_spherical(init.direction);
        assert!(
            (hr - ray.heading[0]).abs() < 1e-13
                && (hth - ray.heading[1]).abs() < 1e-13
                && (hph - ray.heading[2]).abs() < 1e-13
        );

        // p_θ and p_φ agree exactly; p_r differs by the lapse because the Python code
        // used the contravariant formula for p_r (see initial_conditions.rs).
        assert_eq!(init.p0[2], 0.0);
        assert!(ray.p0[2].abs() < 1e-300);
        assert!(
            (init.p0[3] - ray.p0[3]).abs() < 1e-12 * ray.p0[3].abs().max(1.0),
            "p_φ"
        );
        let f = bh.lapse(ray.q0[1]);
        assert!(
            (init.p0[1] * f - ray.p0[1]).abs() < 1e-12,
            "p_r·f = {} vs python {}",
            init.p0[1] * f,
            ray.p0[1]
        );
        // With the Python p_r the Python p_t follows from the same null condition.
        let p_t = null_p_t(f, ray.q0[1], ray.q0[2], ray.p0[1], ray.p0[2], ray.p0[3]);
        assert!(
            (p_t - ray.p0[0]).abs() < 1e-13,
            "p_t {} vs {}",
            p_t,
            ray.p0[0]
        );
    }
}

#[derive(Deserialize)]
struct PyFinal {
    steps: usize,
    delta: f64,
    omega: f64,
    r_max: f64,
    mass: f64,
    q_final: Vec<[f64; 4]>,
}

/// Tolerances are a few orders above the measured agreement (≈1e-13 in r, 9e-12 in t,
/// see the README) so that fixtures regenerated on a real GPU, where FMA contraction
/// changes rounding, still pass, while a change of operation order would not.
#[test]
fn integrator_final_states_match_python() {
    let (_, py) = python_initial();
    let fin: PyFinal = load("integrator_final.json");
    let rs = BlackHole::new(fin.mass).rs();
    let settings = IntegratorSettings::new(fin.steps, fin.delta, fin.omega, fin.r_max, rs);
    let mut worst = [0.0f64; 4];
    for (ray, expect) in py.rays.iter().zip(&fin.q_final) {
        let res = integrate(ray.q0, ray.p0, rs, &settings);
        for k in 0..4 {
            worst[k] = worst[k].max((res.q[k] - expect[k]).abs());
        }
    }
    eprintln!("max |Δ(t, r, θ, φ)| vs Python = {worst:?}");
    assert!(
        worst[1] < 1e-11 && worst[2] < 1e-11 && worst[3] < 1e-12,
        "{worst:?}"
    );
    assert!(worst[0] < 1e-10, "{worst:?}");
}

/// The SIMD equatorial integrator against the same Python final states.
#[test]
fn fast_integrator_final_states_match_python() {
    let (_, py) = python_initial();
    let fin: PyFinal = load("integrator_final.json");
    let rs = BlackHole::new(fin.mass).rs();
    let settings = IntegratorSettings::new(fin.steps, fin.delta, fin.omega, fin.r_max, rs);
    let rays: Vec<Ray> = py.rays.iter().map(|r| Ray::new(r.q0, r.p0)).collect();
    let fast = integrate_batch_fast(&rays, rs, &settings, || {});
    let mut worst = [0.0f64; 4];
    for (res, expect) in fast.iter().zip(&fin.q_final) {
        for k in 0..4 {
            worst[k] = worst[k].max((res.q[k] - expect[k]).abs());
        }
    }
    eprintln!("fast: max |Δ(t, r, θ, φ)| vs Python = {worst:?}");
    assert!(
        worst[1] < 1e-11 && worst[2] < 1e-11 && worst[3] < 1e-12,
        "{worst:?}"
    );
    assert!(worst[0] < 1e-10, "{worst:?}");
}

#[derive(Deserialize)]
struct PyTrajRay {
    ray_index: usize,
    /// Every state the CUDA kernel recorded, up to and including the exit state.
    rows: Vec<[f64; 4]>,
}

#[derive(Deserialize)]
struct PyTraj {
    steps: usize,
    delta: f64,
    omega: f64,
    r_max: f64,
    mass: f64,
    rays: Vec<PyTrajRay>,
}

#[test]
fn integrator_trajectories_match_python_step_by_step_through_the_exit() {
    let (_, py) = python_initial();
    let tr: PyTraj = load("integrator_traj.json");
    let rs = BlackHole::new(tr.mass).rs();
    let settings = IntegratorSettings::new(tr.steps, tr.delta, tr.omega, tr.r_max, rs);
    let mut worst = 0.0f64;
    let mut exits = Vec::new();
    for pyray in &tr.rays {
        let ray = &py.rays[pyray.ray_index];
        let (traj, res) = integrate_trajectory(&Ray::new(ray.q0, ray.p0), rs, &settings);
        assert_ne!(res.exit, ExitReason::StepLimit, "fixture ray must exit");
        assert_eq!(
            traj.len(),
            pyray.rows.len(),
            "ray {}: exit recorded at a different step",
            pyray.ray_index
        );
        for (k, (a, b)) in traj.iter().zip(&pyray.rows).enumerate() {
            for c in 0..4 {
                let d = (a[c] - b[c]).abs();
                assert!(
                    d < 1e-12,
                    "ray {} step {k} component {c}: {} vs {}",
                    pyray.ray_index,
                    a[c],
                    b[c]
                );
                worst = worst.max(d);
            }
        }
        exits.push(res.exit);
    }
    eprintln!(
        "max |Δq| over {} recorded rows = {worst:e}",
        tr.rays.iter().map(|r| r.rows.len()).sum::<usize>()
    );
    assert!(exits.contains(&ExitReason::Captured) && exits.contains(&ExitReason::Escaped));
}

#[derive(Deserialize)]
struct TexelCase {
    th: f64,
    ph: f64,
    h: usize,
    w: usize,
    pc_th: f64,
    pc_ph: f64,
    ps_th: f64,
    ps_ph: f64,
    flip_theta: bool,
    flip_phi: bool,
    /// `[inside, u, v]`.
    result: (bool, i64, i64),
}

/// The Python "method b" flipped φ before testing membership; the port flips only at
/// the texel lookup (README difference 4). So membership is compared on the direction
/// Python tested, and texel indices wherever both rules accept the direction.
#[test]
fn texel_rule_matches_python_method_b() {
    let cases: Vec<TexelCase> = load("texel_method_b.json");
    let (mut compared, mut narrow) = (0, 0);
    for c in &cases {
        let patch = SkyPatch {
            center_theta: c.pc_th,
            center_phi: c.pc_ph,
            size_theta: c.ps_th,
            size_phi: c.ps_ph,
            flip_theta: c.flip_theta,
            flip_phi: c.flip_phi,
        };
        let python_tested_phi = if c.flip_phi { -c.ph } else { c.ph };
        assert_eq!(
            patch.contains(c.th, python_tested_phi),
            c.result.0,
            "membership for {:?}",
            (c.th, c.ph, c.pc_ph, c.ps_ph, c.flip_phi)
        );
        if c.result.0 && patch.contains(c.th, c.ph) {
            let (u, v) = patch.texel(c.th, c.ph, c.h, c.w);
            assert_eq!(
                (u as i64, v as i64),
                (c.result.1, c.result.2),
                "texel for {:?}",
                (c.th, c.ph, c.flip_theta, c.flip_phi)
            );
            compared += 1;
            if c.ps_ph < TWO_PI {
                narrow += 1;
            }
        }
    }
    eprintln!(
        "texel: {compared} of {} cases compared, {narrow} on narrow patches",
        cases.len()
    );
    assert!(
        compared >= 240 && narrow >= 10,
        "{compared} compared, {narrow} narrow"
    );
}

#[derive(Deserialize)]
struct PyFlat {
    h: usize,
    w: usize,
    fov_deg: f64,
    observer: [f64; 3],
    boundary_radius: f64,
    hits: Vec<[f64; 3]>,
}

#[test]
fn flat_hits_match_python_up_to_its_mirrored_camera() {
    let py: PyFlat = load("flat_hits.json");
    let cam = Camera::new(py.observer, py.fov_deg.to_radians(), py.w, py.h).unwrap();
    let res = render_flat(
        &cam,
        py.boundary_radius,
        &SkyPatch::full_sky(),
        None,
        None,
        &[],
    );
    assert_eq!(py.hits.len(), cam.n_pixels());
    for i in 0..py.h {
        for j in 0..py.w {
            let rust = res.hits[cam.flat_index(i, j)].expect("hit");
            // background.py built `right = cross(up, optical_axis) = −ŷ`, mirroring columns.
            let python = py.hits[i * py.w + (py.w - 1 - j)];
            for k in 0..3 {
                assert!(
                    (rust[k] - python[k]).abs() < 1e-9,
                    "({i},{j}) axis {k}: {} vs {}",
                    rust[k],
                    python[k]
                );
            }
        }
    }
}

struct PyPhoton {
    i: usize,
    j: usize,
    final_r: f64,
    final_th: f64,
    final_ph: f64,
    collision: String,
    heading: [f64; 3],
    p0: [f64; 4],
    alpha0: f64,
    analytic_capture: bool,
}

fn load_photon_csv() -> Vec<PyPhoton> {
    let text =
        std::fs::read_to_string(fixture("e2e_photon_data.csv")).expect("e2e_photon_data.csv");
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap().split(',').collect();
    let col = |name: &str| {
        header
            .iter()
            .position(|h| *h == name)
            .unwrap_or_else(|| panic!("column {name}"))
    };
    let (ci, cj, cr, cth, cph, ccol) = (
        col("i"),
        col("j"),
        col("final_r"),
        col("final_th"),
        col("final_ph"),
        col("collision"),
    );
    let (chr, chth, chph) = (col("h_r"), col("h_theta"), col("h_phi"));
    let (cpt, cpr, cpth, cpph, ca, cac) = (
        col("p0_t"),
        col("p0_r"),
        col("p0_th"),
        col("p0_ph"),
        col("alpha0"),
        col("analytic_capture"),
    );
    lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            let num = |k: usize| f[k].parse::<f64>().unwrap();
            PyPhoton {
                i: f[ci].parse().unwrap(),
                j: f[cj].parse().unwrap(),
                final_r: num(cr),
                final_th: num(cth),
                final_ph: num(cph),
                collision: f[ccol].to_string(),
                heading: [num(chr), num(chth), num(chph)],
                p0: [num(cpt), num(cpr), num(cpth), num(cpph)],
                alpha0: num(ca),
                analytic_capture: f[cac] == "True",
            }
        })
        .collect()
}

/// End-to-end against `run_manual_simulation` (16×16, steps 20000, δ 0.01, ω 0.01,
/// observer 30, boundary 31, full-sky patch): integrate from the Python momenta and
/// compare exit radius, exit direction and classification per pixel.
#[test]
fn end_to_end_pipeline_matches_python_photon_table() {
    let photons = load_photon_csv();
    assert_eq!(photons.len(), 256);
    let bh = BlackHole::new(1.0);
    let cam = Camera::new([30.0, 0.0, 0.0], 80f64.to_radians(), 16, 16).unwrap();
    let settings = IntegratorSettings::new(20_000, 0.01, 0.01, 31.0, bh.rs());
    let patch = SkyPatch::full_sky();
    let capture_angle = bh.capture_angle(30.0);
    let mut worst_r = 0.0f64;
    let mut worst_ang = 0.0f64;
    for p in &photons {
        let res = integrate([0.0, 30.0, PI / 2.0, 0.0], p.p0, bh.rs(), &settings);
        worst_r = worst_r.max((res.q[1] - p.final_r).abs());
        let dir = spherical_to_cartesian(p.heading[0], p.heading[1], p.heading[2]);
        let beta = dir[2].atan2(dir[1]);
        let (th, ph) = unrotate_hit(&res.q, beta);
        worst_ang = worst_ang
            .max(angle_diff(th, p.final_th))
            .max(angle_diff(ph, p.final_ph));
        let collision = classify(res.q[1], bh.rs(), 31.0, patch.contains(th, ph));
        assert_eq!(collision.label(), p.collision, "pixel ({},{})", p.i, p.j);
        if collision == Collision::EscapedBackground {
            assert!(res.q[1] >= 31.0);
        }
        // alpha0 is the same geometric angle in both implementations.
        let init = initial_conditions(cam.observer, cam.pixel_position(p.i, p.j), 1.0);
        assert!(
            (init.alpha0 - p.alpha0).abs() < 1e-12,
            "alpha0 ({},{})",
            p.i,
            p.j
        );
        assert_eq!(
            init.alpha0 <= capture_angle,
            p.analytic_capture,
            "analytic_capture ({},{})",
            p.i,
            p.j
        );
    }
    eprintln!("e2e: max |Δr| = {worst_r:e}, max angular Δ = {worst_ang:e}");
    assert!(worst_r < 1e-9 && worst_ang < 1e-9);
}

// ---------------------------------------------------------------------------
// Accretion disk and camera pointing against the Python build
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct DiskSceneParams {
    size: usize,
    fov_deg: f64,
    observer: [f64; 3],
    boundary_radius: f64,
    mass: f64,
    steps: usize,
    delta: f64,
    omega: f64,
    disk_inclination_deg: f64,
    disk_azimuth_deg: f64,
    disk_inner: f64,
    disk_outer: f64,
    disk_temperature: f64,
    yaw_deg: f64,
    pitch_deg: f64,
    roll_deg: f64,
    disk_normal: [f64; 3],
    r_obs: f64,
}

#[derive(Deserialize)]
struct DiskScenePixel {
    i: usize,
    j: usize,
    /// 0 step limit, 1 captured, 2 escaped, 4 disk.
    status: i32,
    r_hit: Option<f64>,
    phi_hit: Option<f64>,
    g: Option<f64>,
    plane_crossings: u32,
    p0: [f64; 4],
    beta: f64,
    phi_d: Option<f64>,
    n_z: f64,
    q: [f64; 4],
}

#[derive(Deserialize)]
struct DiskScene {
    scene: DiskSceneParams,
    pixels: Vec<DiskScenePixel>,
}

fn status_exit(code: i32) -> ExitReason {
    match code {
        1 => ExitReason::Captured,
        2 => ExitReason::Escaped,
        4 => ExitReason::Disk,
        _ => ExitReason::StepLimit,
    }
}

/// Status is a discrete decision at a floor boundary, so one ulp can flip a pixel on
/// the edge of the disk or the shadow between the builds. A flip is tolerated only on a
/// pixel that touches (8-neighbourhood) a pixel of a different Python status, and on at
/// most two pixels per image; any interior flip fails.
fn check_status_flips(size: usize, py: &[ExitReason], rust: &[ExitReason], what: &str) -> usize {
    let mut flips = 0;
    for i in 0..size {
        for j in 0..size {
            let k = i * size + j;
            if py[k] == rust[k] {
                continue;
            }
            flips += 1;
            let mut on_boundary = false;
            for di in -1i64..=1 {
                for dj in -1i64..=1 {
                    let (ni, nj) = (i as i64 + di, j as i64 + dj);
                    if ni < 0 || nj < 0 || ni >= size as i64 || nj >= size as i64 {
                        continue;
                    }
                    if py[ni as usize * size + nj as usize] != py[k] {
                        on_boundary = true;
                    }
                }
            }
            assert!(
                on_boundary,
                "{what}: pixel ({i},{j}) is {:?} in Rust but {:?} in Python, away from any status boundary",
                rust[k], py[k]
            );
        }
    }
    assert!(
        flips <= 2,
        "{what}: {flips} status flips (at most 2 allowed on boundary pixels)"
    );
    flips
}

/// A 16×16 scene with an oblique disk (inclination 55°, azimuth 30°) and a turned camera
/// (yaw 6°, pitch −4°) captured from the Python build (`tests/gen_disk_fixture.py` on the
/// Python branch, under numba's CUDA simulator). Two comparisons: the integrators and
/// crossing detection from the Python momenta, and the whole Rust pipeline from the
/// flags alone.
#[test]
fn disk_scene_matches_python() {
    let fx: DiskScene = load("disk_scene.json");
    let sc = &fx.scene;
    let n = sc.size;
    assert_eq!(fx.pixels.len(), n * n);
    let bh = BlackHole::new(sc.mass);
    let normal = Disk::normal_from_angles(
        sc.disk_inclination_deg.to_radians(),
        sc.disk_azimuth_deg.to_radians(),
    );
    for (a, b) in normal.iter().zip(&sc.disk_normal) {
        assert!((a - b).abs() < 1e-15, "disk normal");
    }
    let disk = Disk::new(
        normal,
        sc.disk_inner,
        sc.disk_outer,
        sc.disk_temperature,
        sc.mass,
        sc.r_obs,
    )
    .unwrap();
    let cam = Camera::with_rotation(
        sc.observer,
        sc.fov_deg.to_radians(),
        n,
        n,
        sc.yaw_deg.to_radians(),
        sc.pitch_deg.to_radians(),
        sc.roll_deg.to_radians(),
    )
    .unwrap();
    let settings =
        IntegratorSettings::new(sc.steps, sc.delta, sc.omega, sc.boundary_radius, bh.rs())
            .with_disk(DiskBand {
                r_in: sc.disk_inner,
                r_out: sc.disk_outer,
            });
    let py_status: Vec<ExitReason> = fx.pixels.iter().map(|p| status_exit(p.status)).collect();
    assert!(py_status.iter().filter(|e| **e == ExitReason::Disk).count() > 100);

    // 1. Camera, initial conditions and the per-ray disk frame.
    for p in &fx.pixels {
        let init = initial_conditions(cam.observer, cam.pixel_position(p.i, p.j), sc.mass);
        for k in 0..4 {
            assert!(
                (init.p0[k] - p.p0[k]).abs() < 1e-12 * p.p0[k].abs().max(1.0),
                "({},{}) p0[{k}]: {} vs {}",
                p.i,
                p.j,
                init.p0[k],
                p.p0[k]
            );
        }
        assert!((init.beta - p.beta).abs() < 1e-12, "({},{}) beta", p.i, p.j);
        let frame = disk.ray_frame(init.beta);
        assert!((frame.n_z - p.n_z).abs() < 1e-12, "({},{}) n_z", p.i, p.j);
        match p.phi_d {
            Some(phi_d) => assert!(
                (frame.phi_d - phi_d).abs() < 1e-12,
                "({},{}) phi_d",
                p.i,
                p.j
            ),
            None => assert!(
                frame.phi_d.is_nan(),
                "({},{}) phi_d should be NaN",
                p.i,
                p.j
            ),
        }
    }

    // 2. Integration and crossing detection from the Python momenta, both CPU paths.
    let rays: Vec<Ray> = fx
        .pixels
        .iter()
        .map(|p| {
            Ray::with_disk(
                [0.0, sc.r_obs, PI / 2.0, 0.0],
                p.p0,
                p.phi_d.unwrap_or(f64::NAN),
            )
        })
        .collect();
    let scalar: Vec<_> = rays
        .iter()
        .map(|r| integrate_ray(r, bh.rs(), &settings))
        .collect();
    let fast = integrate_batch_fast(&rays, bh.rs(), &settings, || {});
    for (label, res) in [("scalar", &scalar), ("simd", &fast)] {
        let status: Vec<ExitReason> = res.iter().map(|r| r.exit).collect();
        let flips = check_status_flips(n, &py_status, &status, label);
        let (mut worst_r, mut worst_phi, mut worst_g, mut worst_q, mut compared) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0);
        for (p, r) in fx.pixels.iter().zip(res) {
            if r.exit != status_exit(p.status) {
                continue;
            }
            assert_eq!(
                r.plane_crossings, p.plane_crossings,
                "({},{}) crossings",
                p.i, p.j
            );
            worst_q = worst_q.max((r.q[1] - p.q[1]).abs());
            if r.exit == ExitReason::Disk {
                let hit = r.disk_hit.unwrap();
                let (py_r, py_phi, py_g) = (p.r_hit.unwrap(), p.phi_hit.unwrap(), p.g.unwrap());
                worst_r = worst_r.max((hit.r - py_r).abs() / py_r);
                worst_phi = worst_phi.max((hit.phi - py_phi).abs());
                let g = redshift(sc.mass, hit.r, p.p0[0], p.p0[3], p.n_z, sc.r_obs);
                worst_g = worst_g.max((g - py_g).abs() / py_g);
                assert!(hit.r >= sc.disk_inner && hit.r <= sc.disk_outer);
                compared += 1;
            }
        }
        eprintln!(
            "{label} vs Python: {flips} status flips, {compared} hits compared, max rel |Δr_hit| = {worst_r:e}, |Δφ_hit| = {worst_phi:e}, rel |Δg| = {worst_g:e}, |Δr_final| = {worst_q:e}"
        );
        assert!(compared > 100);
        assert!(worst_r < 1e-9 && worst_phi < 1e-9 && worst_g < 1e-9 && worst_q < 1e-9);
    }

    // 3. The whole Rust pipeline from the scene parameters alone (camera, initial
    // conditions, disk frame, integration, shading), through the SIMD backend.
    let cfg = RenderConfig {
        camera: cam.clone(),
        black_hole: bh,
        boundary_radius: sc.boundary_radius,
        patch: SkyPatch::full_sky(),
        integrator: IntegratorSettings::new(
            sc.steps,
            sc.delta,
            sc.omega,
            sc.boundary_radius,
            bh.rs(),
        ),
        disk: Some(disk),
        sample_pixels: vec![],
        max_trajectory_points: 10,
    };
    let res = render_curved(&cfg, None, &CpuFast, || {}).unwrap();
    let status: Vec<ExitReason> = res.photons.iter().map(|p| p.result.exit).collect();
    let flips = check_status_flips(n, &py_status, &status, "pipeline");
    let mut worst_g = 0.0f64;
    let mut compared = 0;
    for (p, ph) in fx.pixels.iter().zip(&res.photons) {
        assert_eq!((ph.i, ph.j), (p.i, p.j));
        if ph.result.exit != status_exit(p.status) {
            continue;
        }
        if let (Some(d), Some(py_g)) = (ph.shaded.disk, p.g) {
            assert_eq!(ph.shaded.collision, Collision::Disk);
            worst_g = worst_g.max((d.g - py_g).abs() / py_g);
            compared += 1;
        }
    }
    eprintln!("pipeline vs Python: {flips} status flips, {compared} hits compared, max rel |Δg| = {worst_g:e}");
    assert!(compared > 100 && worst_g < 1e-9);
}
