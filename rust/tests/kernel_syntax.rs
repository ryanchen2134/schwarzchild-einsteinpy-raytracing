//! Without a GPU the CUDA kernel cannot run here, but its C++ can be checked: compile
//! `fantasy.cu` with the host compiler against a shim that defines the CUDA keywords
//! and built-ins. Skips when no `g++`/`clang++` is on the PATH.

use schwarzschild_rt::backend::CUDA_KERNEL_SOURCE;
use std::process::Command;

const SHIM: &str = r#"
#include <math.h>
#define __global__
#define __device__
#define __forceinline__ inline
struct dim3_t { unsigned x, y, z; };
static dim3_t threadIdx, blockIdx, blockDim;
"#;

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
        eprintln!("no C++ compiler on PATH; skipping the kernel syntax check");
        return;
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
    // The host passes: q0s, p0s, n, steps, delta, rs, r_max, r_capture, omega, out_q, out_p,
    // out_status, out_steps. Keep the kernel's parameter list in that order.
    let sig_start = CUDA_KERNEL_SOURCE
        .find("fantasy_integrate(")
        .expect("kernel entry point");
    let sig_end = sig_start + CUDA_KERNEL_SOURCE[sig_start..].find(')').unwrap();
    let sig = &CUDA_KERNEL_SOURCE[sig_start..sig_end];
    let params: Vec<&str> = sig
        .split(',')
        .map(|p| p.trim().rsplit(' ').next().unwrap().trim_start_matches('*'))
        .collect();
    assert_eq!(
        params,
        [
            "q0s",
            "p0s",
            "n",
            "steps",
            "delta",
            "rs",
            "r_max",
            "r_capture",
            "omega",
            "out_q",
            "out_p",
            "out_status",
            "out_steps"
        ]
    );
}
