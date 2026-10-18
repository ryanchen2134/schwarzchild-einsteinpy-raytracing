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
    run_sized(out, "4", extra)
}

fn run_sized(out: &Path, size: &str, extra: &[&str]) -> Output {
    bin()
        .args([
            "--size",
            size,
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
    assert_eq!(lines[0].split(',').count(), 26);
    assert!(lines[0].starts_with("i,j,final_r,final_th,final_ph,collision,h_r,h_theta,h_phi,p0_t,p0_r,p0_th,p0_ph,alpha0,analytic_capture,exit,n_steps,beta,disk_r,disk_phi,disk_x,disk_y,disk_z,disk_g,disk_temp,plane_crossings"));
    assert!(
        lines[1..]
            .iter()
            .all(|l| l.ends_with(",nan,nan,nan,nan,nan,nan,nan,0")),
        "no disk: nan columns, zero crossings"
    );
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

#[test]
fn exact_and_simd_backends_agree() {
    let dir = tmp("exact");
    // 5×5: an odd size, so a lane permutation in the SIMD path would pair different rays.
    let a = run_sized(&dir.join("fast"), "5", &["--no-flat", "--backend", "cpu"]);
    let b = run_sized(
        &dir.join("exact"),
        "5",
        &["--no-flat", "--backend", "cpu", "--exact"],
    );
    assert!(
        a.status.success() && b.status.success(),
        "{}\n{}",
        stderr(&a),
        stderr(&b)
    );
    assert!(stderr(&a).contains("backend cpu-simd"), "{}", stderr(&a));
    assert!(stderr(&b).contains("backend cpu-exact"), "{}", stderr(&b));
    for f in ["photon_data.csv", "sampled_rays.csv", "manual_output.png"] {
        let x = std::fs::read(dir.join("fast").join(f)).unwrap();
        let y = std::fs::read(dir.join("exact").join(f)).unwrap();
        assert_eq!(x, y, "{f} differs between the SIMD and the exact backend");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Every disk row of `photon_data.csv` is consistent with the image and the scene:
/// the hit lies on the disk plane at `disk_r` inside the band, `disk_temp` is a
/// temperature of this disk, `disk_phi` and `disk_g` are finite, the pixel is drawn
/// (neither black nor the red of a numerical error), and the summary counts match.
fn check_disk_outputs(
    out: &Path,
    o: &Output,
    (incl, azim): (f64, f64),
    (r_in, r_out): (f64, f64),
    t_peak: f64,
) {
    let n = schwarzschild_rt::disk::Disk::normal_from_angles(incl.to_radians(), azim.to_radians());
    let csv = std::fs::read_to_string(out.join("photon_data.csv")).unwrap();
    let header: Vec<&str> = csv.lines().next().unwrap().split(',').collect();
    assert_eq!(
        &header[header.len() - 8..],
        [
            "disk_r",
            "disk_phi",
            "disk_x",
            "disk_y",
            "disk_z",
            "disk_g",
            "disk_temp",
            "plane_crossings"
        ]
    );
    let col = |name: &str| header.iter().position(|h| *h == name).unwrap();
    let img = image::open(out.join("manual_output.png"))
        .unwrap()
        .to_rgb8();
    let size = img.height() as usize;
    let mut disk_rows = 0;
    let mut red = 0;
    for line in csv.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        let num = |name: &str| f[col(name)].parse::<f64>().unwrap();
        let (i, j) = (
            f[col("i")].parse::<usize>().unwrap(),
            f[col("j")].parse::<usize>().unwrap(),
        );
        // Row i counts from the bottom; the PNG's row 0 is the top.
        let px = img.get_pixel(j as u32, (size - 1 - i) as u32).0;
        if px == [255, 0, 0] {
            red += 1;
        }
        if f[col("collision")] == "disk" {
            disk_rows += 1;
            let (r, x, y, z) = (num("disk_r"), num("disk_x"), num("disk_y"), num("disk_z"));
            assert!((r_in..=r_out).contains(&r), "{line}");
            assert!(
                ((x * x + y * y + z * z).sqrt() - r).abs() < 1e-9 * r,
                "{line}"
            );
            assert!(
                (n[0] * x + n[1] * y + n[2] * z).abs() < 1e-9 * r,
                "on the plane: {line}"
            );
            assert!(num("disk_phi").is_finite());
            let g = num("disk_g");
            assert!(g > 0.0 && g.is_finite(), "{line}");
            let t = num("disk_temp");
            assert!(t > 0.0 && t <= t_peak * (1.0 + 1e-12), "{line}");
            assert!(f[col("plane_crossings")].parse::<u32>().unwrap() >= 1);
            assert_eq!(f[col("exit")], "disk");
            assert!(
                px != [0, 0, 0] && px != [255, 0, 0],
                "pixel ({i},{j}) = {px:?}"
            );
        } else {
            for c in [
                "disk_r",
                "disk_phi",
                "disk_x",
                "disk_y",
                "disk_z",
                "disk_g",
                "disk_temp",
            ] {
                assert_eq!(f[col(c)], "nan", "{line}");
            }
        }
    }
    assert!(disk_rows > 10, "{disk_rows} disk rows\n{csv}");
    let stdout = String::from_utf8_lossy(&o.stdout);
    assert!(
        stdout.contains(&format!("Hit disk:         {disk_rows}")),
        "{stdout}"
    );
    assert!(
        stdout.contains(&format!("Numerical error:  {red}")),
        "{stdout}"
    );
    let flat = image::open(out.join("no_gravity.png")).unwrap().to_rgb8();
    assert!(
        flat.pixels().any(|p| p.0 != [0, 0, 0]),
        "the reference render draws the disk too"
    );
}

#[test]
fn disk_and_pointing_flags_render_and_are_reported() {
    let dir = tmp("disk");
    let out = dir.join("out");
    let o = run_sized(
        &out,
        "12",
        &[
            "--disk",
            "--disk-inclination",
            "60",
            "--disk-azimuth",
            "20",
            "--disk-temperature",
            "5000",
            "--yaw",
            "4",
            "--pitch",
            "-3",
            "--roll",
            "10",
        ],
    );
    assert!(o.status.success(), "{}", stderr(&o));
    let err = stderr(&o);
    assert!(err.contains("yaw/pitch/roll 4/-3/10°"), "{err}");
    assert!(
        err.contains("disk inclination 60°, azimuth 20°, r 6–20, T_peak 5000 K"),
        "{err}"
    );
    check_disk_outputs(&out, &o, (60.0, 20.0), (6.0, 20.0), 5000.0);
    // The same scene at M = 2 with every length doubled.
    let out2 = dir.join("out2");
    let o2 = run_sized(
        &out2,
        "12",
        &[
            "--disk",
            "--disk-inclination",
            "60",
            "--disk-azimuth",
            "20",
            "--disk-temperature",
            "5000",
            "--yaw",
            "4",
            "--pitch",
            "-3",
            "--roll",
            "10",
            "--bh-mass",
            "2",
            "--observer-distance",
            "60",
            "--boundary-radius",
            "62",
            "--delta",
            "0.02",
        ],
    );
    assert!(o2.status.success(), "{}", stderr(&o2));
    check_disk_outputs(&out2, &o2, (60.0, 20.0), (12.0, 40.0), 5000.0);
    // Bad disk and pointing flags fail with a message naming the problem.
    for (args, needle) in [
        (vec!["--disk", "--disk-inner", "5"], "inner radius"),
        (vec!["--disk", "--disk-outer", "30"], "outer radius"),
        (
            vec!["--disk", "--disk-inclination", "200"],
            "--disk-inclination",
        ),
        (vec!["--yaw", "nan"], "--yaw"),
    ] {
        let o = run(&dir.join("bad"), &args);
        assert!(!o.status.success(), "{args:?} should fail");
        assert!(stderr(&o).contains(needle), "{args:?}: {}", stderr(&o));
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn cuda_backend_without_the_feature_explains_how_to_build_it() {
    if schwarzschild_rt::backend::cuda_compiled_in() {
        return;
    }
    let dir = tmp("cuda");
    let o = run(&dir.join("out"), &["--backend", "cuda"]);
    assert!(!o.status.success());
    assert!(stderr(&o).contains("--features cuda"), "{}", stderr(&o));
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A CUDA build must still run on a machine without a driver: `auto` falls back to
/// the CPU instead of aborting inside cudarc's loader.
#[cfg(feature = "cuda")]
#[test]
fn auto_backend_runs_with_or_without_a_device() {
    let dir = tmp("auto");
    let o = run(&dir.join("out"), &["--no-flat", "--backend", "auto"]);
    assert!(o.status.success(), "{}", stderr(&o));
    let err = stderr(&o);
    assert!(
        err.contains("backend cpu-simd") || err.contains("backend cuda"),
        "{err}"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}
