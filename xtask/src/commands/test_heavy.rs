//! Heavy tier — the `#[ignore]`d long-running tests that a CPU can run.
//!
//! `cargo xtask test-fast` runs each crate's *default* test set, which by
//! definition excludes everything carrying `#[ignore]`. This tier is the
//! complement: one `cargo test --release ... -- --ignored` per test target that
//! hosts ignored tests, mirroring the `regression` matrix in
//! `.github/workflows/weekly-tests.yml` and extending it to the ignored tests
//! outside `rlevo` that the weekly job never reached.
//!
//! Three properties are load-bearing:
//!
//! 1. **CPU only.** A handful of ignored tests exist solely to compare a wgpu
//!    device against Flex. Initializing a wgpu device on a host with no adapter
//!    *aborts* — cubecl-wgpu panics on a worker thread and the caller then
//!    panics on the severed channel — so these cannot be run on a CI runner.
//!    They are the `test-gpu` tier, and this one reads that tier's table to know
//!    what to leave alone rather than keeping a second list of its own.
//! 2. **The tier is total over the workspace's ignored tests.** A `#[ignore]`d
//!    test in a target absent from the tier is run by *nothing*: the fast tier
//!    skips it for being ignored, and this tier skips it for being unlisted.
//!    [`tiers::assert_ignored_tests_are_claimed`] scans every member's sources
//!    on each invocation, so a new heavy test fails the tier instead of quietly
//!    falling out of it.
//! 3. **Targets, not crates.** `rlevo` alone contributes nine binaries with
//!    wildly different budgets, and the GPU exclusions are per-target too, so
//!    the tier's unit is the test binary. That is also the selection key:
//!    `cargo xtask test-heavy dqn_integration`, not `-p rlevo`.
//!
//! `--release` is not optional. Several entries run hundreds of thousands of
//! environment steps; the weekly workflow uses `--release` for the same reason,
//! and a debug build puts them far outside any usable budget.
//!
//! # Examples
//!
//! ```ignore
//! cargo xtask test-heavy
//! cargo xtask test-heavy dqn_integration
//! cargo xtask test-heavy -x td3_integration
//! cargo xtask test-heavy --with-acceptance qrdqn_integration
//! ```

use anyhow::Result;
use tracel_xtask::prelude::*;

use super::tiers;

/// A test target in the heavy tier.
pub struct HeavyTarget {
    /// Package name, as `cargo -p` spells it.
    package: &'static str,
    /// The `tests/<name>.rs` binary, or `None` for the crate's unit tests
    /// (`--lib`).
    test: Option<&'static str>,
    /// Ignored tests skipped unless `--with-acceptance` is passed.
    ///
    /// These are the hours-long full-solve runs that do not fit a normal CI
    /// budget. `weekly-tests.yml` makes the same split, giving them a separate
    /// job with a 350-minute timeout.
    deferred: &'static [&'static str],
}

impl HeavyTarget {
    /// The tier's key for this target: what the user names to select it, and
    /// what the group header and failure message report.
    fn label(&self) -> String {
        tiers::target_label(self.package, self.test)
    }
}

/// The tier. Together with `test_gpu::GPU_TARGETS` this must be total over the
/// workspace's `#[ignore]`d tests — `tiers::assert_ignored_tests_are_claimed`
/// enforces it.
pub const HEAVY_TARGETS: &[HeavyTarget] = &[
    HeavyTarget {
        package: "rlevo",
        test: Some("c51_integration"),
        deferred: &[],
    },
    HeavyTarget {
        package: "rlevo",
        test: Some("ddpg_integration"),
        deferred: &[],
    },
    HeavyTarget {
        package: "rlevo",
        test: Some("dqn_integration"),
        deferred: &[],
    },
    // `calibration_explorer` is an observation harness: it prints a multi-seed
    // eval-count sweep and asserts nothing about the numbers. Its value here is
    // narrower than the rest of the tier — it proves the sweep still runs
    // without panicking — but it is a CPU-bound ignored test, so leaving it out
    // would put it in the same run-by-nothing position the tier exists to
    // prevent.
    HeavyTarget {
        package: "rlevo",
        test: Some("memetic_rastrigin"),
        deferred: &[],
    },
    HeavyTarget {
        package: "rlevo",
        test: Some("ppg_integration"),
        deferred: &[],
    },
    HeavyTarget {
        package: "rlevo",
        test: Some("ppo_integration"),
        deferred: &[],
    },
    // The 500k-step acceptance run takes hours; the rest of the binary fits a
    // normal budget. Same split as the weekly workflow's `regression` and
    // `acceptance` jobs.
    HeavyTarget {
        package: "rlevo",
        test: Some("qrdqn_integration"),
        deferred: &["qrdqn_cartpole_acceptance"],
    },
    HeavyTarget {
        package: "rlevo",
        test: Some("sac_integration"),
        deferred: &[],
    },
    HeavyTarget {
        package: "rlevo",
        test: Some("td3_integration"),
        deferred: &[],
    },
    // Also an observation-only harness; see `memetic_rastrigin` above.
    HeavyTarget {
        package: "rlevo-evolution",
        test: Some("coevolution_forgetting"),
        deferred: &[],
    },
    HeavyTarget {
        package: "rlevo-examples",
        test: Some("neuroevolution_santa_fe_ant"),
        deferred: &[],
    },
];

/// Whether the heavy tier covers `(package, test)`.
///
/// `tiers::assert_ignored_tests_are_claimed` asks both ignored-test tiers this,
/// so each owns its own table and neither restates the other's.
pub fn claims(package: &str, test: Option<&str>) -> bool {
    HEAVY_TARGETS
        .iter()
        .any(|entry| entry.package == package && entry.test == test)
}

#[macros::declare_command_args(None, None)]
pub struct TestHeavyCmdArgs {
    /// Test binaries to run. Defaults to every target in the tier.
    #[arg(value_name = "TARGET")]
    pub only: Vec<String>,
    /// Comma-separated list of test binaries to skip.
    #[arg(
        short = 'x',
        long,
        value_name = "TARGET,TARGET,...",
        value_delimiter = ',',
        required = false
    )]
    pub exclude: Vec<String>,
    /// Also run the deferred hours-long acceptance tests.
    #[arg(long, default_value_t = false)]
    pub with_acceptance: bool,
    /// Print the tier's targets as a JSON array and exit, running nothing.
    ///
    /// `weekly-tests.yml` builds its matrix from this with `fromJSON`, so the
    /// tier table is the workflow's single source of truth rather than a list
    /// the YAML restates and then drifts from.
    #[arg(long, default_value_t = false)]
    pub list: bool,
}

/// Run the heavy tier, stopping at the first target that fails.
///
/// `args` is taken by reference, unlike the by-value base `tracel-xtask`
/// handlers — nothing here consumes it, and `clippy::pedantic` is warned
/// workspace-wide.
///
/// # Errors
///
/// Returns an error if the tier has drifted from the workspace's ignored tests,
/// if the caller named a target outside the tier, or if any target's tests fail.
pub fn handle_command(args: &TestHeavyCmdArgs, _env: Environment, _ctx: Context) -> Result<()> {
    tiers::assert_ignored_tests_are_claimed()?;

    if args.list {
        tiers::print_json_list(HEAVY_TARGETS.iter().map(HeavyTarget::label));
        return Ok(());
    }

    reject_unknown_selection(args)?;

    for entry in HEAVY_TARGETS {
        run_heavy_test(entry, args)?;
    }
    Ok(())
}

/// Fail if the user named a target the tier does not contain.
///
/// Without this, a typo selects nothing, every target is skipped, and the tier
/// exits zero having tested nothing at all.
fn reject_unknown_selection(args: &TestHeavyCmdArgs) -> Result<()> {
    let labels: Vec<String> = HEAVY_TARGETS.iter().map(HeavyTarget::label).collect();
    let unknown: Vec<&str> = args
        .only
        .iter()
        .chain(args.exclude.iter())
        .map(String::as_str)
        .filter(|name| !labels.iter().any(|label| label == *name))
        .collect();

    if unknown.is_empty() {
        return Ok(());
    }

    anyhow::bail!(
        "not in the heavy tier: {}\nAvailable targets: {}",
        unknown.join(", "),
        labels.join(", ")
    )
}

fn run_heavy_test(entry: &HeavyTarget, args: &TestHeavyCmdArgs) -> Result<()> {
    let label = entry.label();
    group!("Heavy Tests: {}", label);

    let mut cmd_args = vec![
        "test".to_string(),
        "--release".to_string(),
        "-p".to_string(),
        entry.package.to_string(),
    ];
    match entry.test {
        Some(name) => cmd_args.extend(["--test".to_string(), name.to_string()]),
        None => cmd_args.push("--lib".to_string()),
    }
    // `--ignored` runs *only* the ignored tests; the fast tier already covers
    // the rest, and re-running them here would double the tier's cost.
    cmd_args.extend([
        "--".to_string(),
        "--ignored".to_string(),
        "--color=always".to_string(),
    ]);
    if !args.with_acceptance {
        for name in entry.deferred {
            cmd_args.extend(["--skip".to_string(), (*name).to_string()]);
        }
    }

    // `ignore_log` is deliberately `None`, as in the fast tier: every message
    // cargo emits here is one the tier should fail on.
    run_process_for_package(
        "cargo",
        &label,
        &cmd_args.iter().map(String::as_str).collect::<Vec<_>>(),
        None,
        &args.exclude,
        &args.only,
        &format!("Heavy tests failed for '{label}'"),
        None,
        None,
    )?;

    endgroup!();
    Ok(())
}
