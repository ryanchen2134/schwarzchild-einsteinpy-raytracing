//! The binary: output files, determinism across thread counts, flag parity, validation.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_schwarzschild-rt"))
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "schwarzschild-rt-cli-{}-{name}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn run(out: &Path, extra: &[&str]) -> Output {
    bin()
        .args([
            "--size",
            "4",
            "--steps",
            "8000",
            "--seed",
            "1",
            "--n-sample-rays",
            "2",
            "--out-dir",
        ])
        .arg(out)
        .args(extra)
        .output()
        .expect("binary runs")
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

/// A 32×16 texture whose texel (row, col) is (row, col, 9).
fn write_texture(dir: &Path) -> PathBuf {
    let path = dir.join("sky.png");
    let img = image::RgbImage::from_fn(32, 16, |x, y| image::Rgb([y as u8, x as u8, 9]));
    img.save(&path).unwrap();
    path
}

#[test]
fn writes_every_output_file_and_shades_from_the_texture() {
    let dir = tmp("files");
    let out = dir.join("out");
    let tex = write_texture(&dir);
    let o = run(&out, &["--background", tex.to_str().unwrap()]);
    assert!(o.status.success(), "{}", stderr(&o));
    for f in [
        "manual_output.png",
        "no_gravity.png",
        "photon_data.csv",
        "sampled_rays.csv",
        "flat_rays.csv",
    ] {
        assert!(out.join(f).exists(), "{f} missing");
    }
    let csv = std::fs::read_to_string(out.join("photon_data.csv")).unwrap();
    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(lines.len(), 17);
    assert_eq!(lines[0].split(',').count(), 18);
    assert!(lines[0].starts_with("i,j,final_r,final_th,final_ph,collision,h_r,h_theta,h_phi,p0_t,p0_r,p0_th,p0_ph,alpha0,analytic_capture,exit,n_steps,beta"));
    assert!(lines[1..]
        .iter()
        .all(|l| l.contains(",True,") || l.contains(",False,")));
    assert!(
        lines[1..].iter().any(|l| l.contains(",escape_bg,")),
        "{csv}"
    );
    let img = image::open(out.join("manual_output.png"))
        .unwrap()
        .to_rgb8();
    assert_eq!(img.dimensions(), (4, 4));
    assert!(
        img.pixels().any(|p| p.0[2] == 9),
        "escaped pixels carry texture texels"
    );
    let flat = image::open(out.join("no_gravity.png")).unwrap().to_rgb8();
    assert!(
        flat.pixels().all(|p| p.0[2] == 9),
        "every straight ray hits the full-sky texture"
    );
    let stdout = String::from_utf8_lossy(&o.stdout);
    assert!(stdout.contains("Photon summary"));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn results_do_not_depend_on_the_thread_count() {
    let dir = tmp("threads");
    let a = run(&dir.join("a"), &["--threads", "1", "--no-flat"]);
    let b = run(&dir.join("b"), &["--threads", "2", "--no-flat"]);
    assert!(
        a.status.success() && b.status.success(),
        "{}\n{}",
        stderr(&a),
        stderr(&b)
    );
    for f in ["photon_data.csv", "sampled_rays.csv"] {
        let x = std::fs::read(dir.join("a").join(f)).unwrap();
        let y = std::fs::read(dir.join("b").join(f)).unwrap();
        assert_eq!(x, y, "{f} differs between thread counts");
    }
    assert!(!dir.join("a").join("no_gravity.png").exists());
    assert!(!dir.join("a").join("flat_rays.csv").exists());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn python_flag_spellings_and_negative_offsets_are_accepted() {
    let dir = tmp("flags");
    let o = run(
        &dir.join("out"),
        &[
            "--no-flat-trajectories",
            "--bg-patch-center-phi-relobs",
            "-10",
            "--bg-patch-center-theta-relobs",
            "-5",
        ],
    );
    assert!(o.status.success(), "{}", stderr(&o));
    assert!(!dir.join("out").join("no_gravity.png").exists());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn invalid_arguments_fail_with_a_message_naming_the_flag() {
    let dir = tmp("invalid");
    for (args, needle) in [
        (vec!["--boundary-radius", "20"], "--boundary-radius"),
        (vec!["--n-sample-rays", "100"], "--n-sample-rays"),
        (vec!["--delta", "0"], "--delta"),
        (vec!["--observer-distance", "2.9"], "--observer-distance"),
        (vec!["--bh-mass", "-1"], "--bh-mass"),
        (
            vec!["--boundary-radius", "150", "--observer-distance", "120"],
            "--boundary-radius",
        ),
    ] {
        let o = run(&dir.join("out"), &args);
        assert!(!o.status.success(), "{args:?} should fail");
        assert!(stderr(&o).contains(needle), "{args:?}: {}", stderr(&o));
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
