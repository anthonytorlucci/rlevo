//! The consumer probes: measurements, not walks.
//!
//! A consumer probe has no steps. It records what a dependency list costs
//! the person who writes it: how many packages their build compiles, how
//! many of those are Burn's, and how long a first build takes. The numbers
//! never fail a run. A probe that does not resolve or build is the one
//! failure, and it is a finding.

use std::fs;
use std::path::Path;
use std::thread;
use std::time::Instant;

use anyhow::{Result, bail};
use tracel_xtask::prelude::*;

use super::stage::{self, Resolved};

/// Measure the probe at `probe_dir`, whose graph has already resolved from
/// the stage. Returns `(what, value)` lines in the order they are printed.
///
/// # Errors
///
/// Returns an error if `cargo tree` fails or the probe does not build.
pub fn measure(
    probe_dir: &Path,
    run_dir: &Path,
    resolved: &Resolved,
) -> Result<Vec<(&'static str, String)>> {
    // Resolution and the tree never build, so any directory serves them.
    let scratch_target = run_dir.join("target");
    let mut lines = Vec::new();

    let packages = stage::cone(probe_dir, &scratch_target, None)?;
    lines.push((
        "packages",
        format!(
            "{} in the probe's graph (unique names, cargo tree -e normal)",
            packages.len()
        ),
    ));
    if resolved
        .rlevo
        .iter()
        .any(|entry| entry.starts_with("rlevo-core "))
    {
        let under_core = stage::cone(probe_dir, &scratch_target, Some("rlevo-core"))?;
        lines.push((
            "under-rlevo-core",
            format!(
                "{} with the features this probe's graph unifies",
                under_core.len()
            ),
        ));
    }
    let burn = packages
        .iter()
        .filter(|name| name.starts_with("burn"))
        .count();
    lines.push(("burn-packages", burn.to_string()));

    // A fresh directory, so the time is a consumer's first build. A warm
    // build measures nothing they see.
    let cold_target = run_dir.join("cold-target");
    let started = Instant::now();
    let built = stage::run(stage::probe_cargo(probe_dir, &cold_target).arg("build"))?;
    let seconds = started.elapsed().as_secs_f64();
    // The build output is recreated from the probe and its lock in seconds.
    let _ = fs::remove_dir_all(&cold_target);
    if !built.ok {
        let error = built
            .stderr
            .lines()
            .find(|line| line.starts_with("error"))
            .unwrap_or("the build failed with no error message");
        bail!("the probe does not build: {error}");
    }
    lines.push((
        "cold-build",
        format!(
            "{seconds:.1} s (n = 1, dev profile, {}, {})",
            stage::toolchain(probe_dir),
            machine()
        ),
    ));
    Ok(lines)
}

/// The machine a timing was taken on, as far as the process can name it.
fn machine() -> String {
    let cores = thread::available_parallelism().map_or(0, std::num::NonZero::get);
    format!(
        "{}/{}, {cores} cores",
        std::env::consts::OS,
        std::env::consts::ARCH
    )
}
