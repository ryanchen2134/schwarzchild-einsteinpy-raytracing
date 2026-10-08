//! Equivalence with the Python implementation (`simulation/`), using fixtures
//! captured from it under numba's CUDA simulator at commit 7dcfb9f
//! (`tests/fixtures/`, produced by `tests/fixtures/README.md`'s script).

use schwarzschild_rt::camera::Camera;
use schwarzschild_rt::coords::{cartesian_to_spherical, spherical_to_cartesian, TWO_PI};
use schwarzschild_rt::flat::render_flat;
use schwarzschild_rt::initial_conditions::{initial_conditions, null_p_t};
use schwarzschild_rt::integrator::{integrate, integrate_trajectory, IntegratorSettings};
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
        worst[1] < 1e-8 && worst[2] < 1e-8 && worst[3] < 1e-8,
        "{worst:?}"
    );
    assert!(worst[0] < 1e-7, "{worst:?}");
}

#[derive(Deserialize)]
struct PyTraj {
    steps: usize,
    delta: f64,
    omega: f64,
    r_max: f64,
    mass: f64,
    ray_index: Vec<usize>,
    traj: Vec<Vec<[f64; 4]>>,
}

#[test]
fn integrator_trajectories_match_python_step_by_step() {
    let (_, py) = python_initial();
    let tr: PyTraj = load("integrator_traj.json");
    let rs = BlackHole::new(tr.mass).rs();
    let settings = IntegratorSettings::new(tr.steps, tr.delta, tr.omega, tr.r_max, rs);
    let mut worst = 0.0f64;
    for (&idx, expect) in tr.ray_index.iter().zip(&tr.traj) {
        let ray = &py.rays[idx];
        let (traj, _) = integrate_trajectory(ray.q0, ray.p0, rs, &settings);
        assert_eq!(expect.len(), tr.steps);
        for (k, e) in expect.iter().enumerate() {
            if k >= traj.len() {
                assert!(
                    e.iter().all(|x| *x == 0.0),
                    "python buffer past exit must be zero"
                );
                continue;
            }
            for c in 0..4 {
                worst = worst.max((traj[k][c] - e[c]).abs());
            }
        }
    }
    eprintln!("max |Δq| over {} recorded steps = {worst:e}", tr.steps);
    assert!(worst < 1e-9, "{worst}");
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
    /// `[inside, u, v]` as `[bool, i64, i64]` encoded as JSON values.
    result: (bool, i64, i64),
}

#[test]
fn texel_rule_matches_python_method_b() {
    let cases: Vec<TexelCase> = load("texel_method_b.json");
    let mut compared = 0;
    for c in &cases {
        let patch = SkyPatch {
            center_theta: c.pc_th,
            center_phi: c.pc_ph,
            size_theta: c.ps_th,
            size_phi: c.ps_ph,
            flip_theta: c.flip_theta,
            flip_phi: c.flip_phi,
        };
        let inside = patch.contains(c.th, c.ph);
        if !c.flip_phi {
            // Python flipped φ before testing membership; without that flip the rules agree.
            assert_eq!(
                inside,
                c.result.0,
                "membership for {:?}",
                (c.th, c.ph, c.pc_ph, c.ps_ph)
            );
        }
        if inside && c.result.0 {
            let (u, v) = patch.texel(c.th, c.ph, c.h, c.w);
            assert_eq!(
                (u as i64, v as i64),
                (c.result.1, c.result.2),
                "texel for {:?}",
                (c.th, c.ph, c.flip_theta, c.flip_phi)
            );
            compared += 1;
        }
    }
    // 240 full-sky cases are always inside; the two narrow patches add a few more.
    assert!(
        compared >= 240,
        "only {compared} of {} cases compared",
        cases.len()
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
    let res = render_flat(&cam, py.boundary_radius, &SkyPatch::full_sky(), None, &[]);
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
    assert!(worst_r < 1e-7 && worst_ang < 1e-7);
}
