//! Without a GPU the CUDA kernel cannot run here, but its C++ can be checked: compile
//! `fantasy.cu` with the host compiler against a shim that defines the CUDA keywords
//! and built-ins (the shim supplies `sincos` itself where the host libm lacks it).
//! Needs `g++` or `clang++` on the PATH; set `SCHWARZSCHILD_RT_SKIP_KERNEL_SYNTAX=1`
//! to skip on a machine without one.

use schwarzschild_rt::backend::CUDA_KERNEL_SOURCE;
use std::process::Command;

const SHIM: &str = r#"
#include <math.h>
#if !defined(__GLIBC__)
static inline void sincos(double x, double* s, double* c) { *s = sin(x); *c = cos(x); }
#endif
#define __global__
#define __device__
#define __forceinline__ inline
struct dim3_t { unsigned x, y, z; };
static dim3_t threadIdx, blockIdx, blockDim;
"#;

/// The kernel's parameter list, as the host launch in `cuda.rs` pushes it: this is
/// the only ABI check available without a device, so it includes the types.
const EXPECTED_SIGNATURE: &str = "fantasy_integrate(real* q1s, real* p1s, real* q2s, real* p2s, \
int* status, int* n_steps, const real* disk_phi, int* crossings, real* hit_r, real* hit_phi, \
int n, int steps, int round_steps, \
real delta, real rs, real r_max, real r_capture, real omega, real disk_r_in, real disk_r_out)";

#[test]
fn cuda_kernel_is_valid_cplusplus() {
    let compiler = ["g++", "clang++"].into_iter().find(|c| {
        Command::new(c)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    });
    let Some(compiler) = compiler else {
        if std::env::var_os("SCHWARZSCHILD_RT_SKIP_KERNEL_SYNTAX").is_some() {
            eprintln!("no C++ compiler on PATH; skipping the kernel syntax check");
            return;
        }
        panic!("no g++/clang++ on PATH; install one or set SCHWARZSCHILD_RT_SKIP_KERNEL_SYNTAX=1 to skip the CUDA kernel syntax check");
    };
    let dir = std::env::temp_dir().join(format!("schwarzschild-rt-kernel-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("fantasy_check.cpp");
    std::fs::write(&src, format!("{SHIM}\n{CUDA_KERNEL_SOURCE}")).unwrap();
    let out = Command::new(compiler)
        .args([
            "-std=c++14",
            "-fsyntax-only",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-x",
            "c++",
        ])
        .arg(&src)
        .output()
        .expect("compiler runs");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        out.status.success(),
        "kernel does not compile as C++:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn kernel_signature_matches_the_host_launch() {
    assert!(
        CUDA_KERNEL_SOURCE.contains("typedef double real;"),
        "the host pushes f64; `real` must stay double"
    );
    let sig_start = CUDA_KERNEL_SOURCE
        .find("fantasy_integrate(")
        .expect("kernel entry point");
    let sig_end = sig_start + CUDA_KERNEL_SOURCE[sig_start..].find(')').unwrap() + 1;
    let normalised = CUDA_KERNEL_SOURCE[sig_start..sig_end]
        .replace("__restrict__", "")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    assert_eq!(normalised, EXPECTED_SIGNATURE);
}
