//! Build-time provenance for the `record` feature's run manifest.
//!
//! Emits four `cargo:rustc-env` values consumed via `option_env!` in
//! `record::reporter::RecordingReporter::with_build_provenance`:
//! `GIT_COMMIT`, `GIT_DIRTY` (`"1"`/`"0"`/`""`), `RUSTC_VERSION`, and
//! `BURN_VERSION`. Everything here is **non-fatal**: outside a git
//! checkout, or with no `git`/`rustc` on `PATH`, the value is emitted as an
//! empty string and the corresponding manifest field resolves to `None`. A
//! recording must never fail to *build* because provenance was unavailable.

use std::process::Command;

/// Runs `cmd args...`, returning trimmed stdout on success, or an empty
/// string on any failure (missing binary, non-zero exit, non-UTF-8).
fn capture(cmd: &str, args: &[&str]) -> String {
    Command::new(cmd)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

fn main() {
    // Re-run when the build script itself changes. We deliberately do not
    // chase git HEAD/index here: provenance staleness across incremental
    // rebuilds is an accepted minor cost (see ADR 0014 reversal note).
    println!("cargo:rerun-if-changed=build.rs");

    let git_commit = capture("git", &["rev-parse", "HEAD"]);
    println!("cargo:rustc-env=GIT_COMMIT={git_commit}");

    // `--porcelain` prints one line per change; empty output ⇒ clean tree.
    // Emit "" when git is unavailable so the reporter records `None`.
    let git_dirty = if git_commit.is_empty() {
        String::new()
    } else if capture("git", &["status", "--porcelain"]).is_empty() {
        "0".to_string()
    } else {
        "1".to_string()
    };
    println!("cargo:rustc-env=GIT_DIRTY={git_dirty}");

    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let rustc_version = capture(&rustc, &["-V"]);
    println!("cargo:rustc-env=RUSTC_VERSION={rustc_version}");

    // Pinned to the workspace `burn` dependency. Extracting this from the
    // resolved lockfile is an open question (ADR 0014 §10); a pinned const
    // is the agreed starting point.
    println!("cargo:rustc-env=BURN_VERSION=0.21.0");
}
