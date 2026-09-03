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
//!    panics on the severed channel — so these cannot be run on a CI runner and
//!    are excused in [`GPU_ONLY`] rather than skipped by name at the call site.
//! 2. **The tier is total over the workspace's ignored tests.** A `#[ignore]`d
//!    test in a target absent from the tier is run by *nothing*: the fast tier
//!    skips it for being ignored, and this tier skips it for being unlisted.
//!    `assert_tier_is_total` scans every member's sources on each invocation, so
//!    a new heavy test fails the tier instead of quietly falling out of it.
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

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;
use tracel_xtask::prelude::*;
use tracel_xtask::utils::workspace::{WorkspaceMemberType, get_workspace_members};

/// A test target in the heavy tier.
struct HeavyTarget {
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
        match self.test {
            Some(name) => name.to_string(),
            None => format!("{}:lib", self.package),
        }
    }
}

/// The tier. Keep total over the workspace's `#[ignore]`d tests —
/// `assert_tier_is_total` enforces it.
const HEAVY_TARGETS: &[HeavyTarget] = &[
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

/// Targets whose ignored tests need a GPU, each with its reason.
///
/// These are excused from the tier rather than skipped by name inside it: the
/// whole point is that they must not be *reached* on a machine with no adapter,
/// and a `--skip` list keyed on test names is exactly the thing that drifts when
/// a test is renamed.
const GPU_ONLY: &[(&str, Option<&str>, &str)] = &[
    (
        "rlevo-evolution",
        Some("backend_parity"),
        "wgpu-vs-Flex parity on sphere/d10; needs a wgpu adapter",
    ),
    (
        "rlevo-reinforcement-learning",
        Some("c51_projection_backend_parity"),
        "wgpu-vs-Flex C51 projection parity; needs a wgpu adapter",
    ),
    (
        "rlevo-reinforcement-learning",
        None,
        "`clamp_preserving_nan_matches_flex_on_wgpu` in `algorithms::shared`; \
         needs a wgpu adapter",
    ),
];

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
    assert_tier_is_total()?;

    if args.list {
        crate::commands::print_json_list(HEAVY_TARGETS.iter().map(HeavyTarget::label));
        return Ok(());
    }

    reject_unknown_selection(args)?;

    for entry in HEAVY_TARGETS {
        run_heavy_test(entry, args)?;
    }
    Ok(())
}

/// Fail if any target hosting `#[ignore]`d tests is neither in the tier nor
/// excused as GPU-only.
///
/// The fast tier answers the same question with `cargo metadata`, because there
/// "is this crate covered?" is a property of the member list. Here the unit is a
/// test target that *contains an ignored test*, which no cargo command reports,
/// so the check reads the sources: any line whose first non-whitespace is
/// `#[ignore` is an attribute (a `//`/`///`/`//!` mention is not), including the
/// ones passed through `rl_learning_test!` and `rl_reproducibility_test!`.
///
/// Scope: a crate's `src/` tree maps to its `--lib` target and each top-level
/// `tests/*.rs` to the integration binary cargo auto-discovers from it. Ignored
/// tests reached only through a `#[path]`-included module outside those two
/// trees are not seen.
fn assert_tier_is_total() -> Result<()> {
    let mut unclaimed: Vec<String> = Vec::new();

    for member in get_workspace_members(WorkspaceMemberType::Crate) {
        let root = Path::new(&member.path);

        if let Some(file) = find_ignored_test(&root.join("src"))?
            && !is_claimed(&member.name, None)
        {
            unclaimed.push(format!("{} --lib ({})", member.name, file.display()));
        }

        for (stem, file) in integration_targets_with_ignored_tests(&root.join("tests"))? {
            if !is_claimed(&member.name, Some(&stem)) {
                unclaimed.push(format!(
                    "{} --test {stem} ({})",
                    member.name,
                    file.display()
                ));
            }
        }
    }

    if unclaimed.is_empty() {
        return Ok(());
    }

    anyhow::bail!(
        "test targets with #[ignore]d tests that the heavy tier does not run:\n  {}\n\
         An ignored test outside this tier is run by nothing: the fast tier skips it\n\
         for being ignored, and this tier skips it for being unlisted.\n\
         Add the target to HEAVY_TARGETS, or to GPU_ONLY with a reason.",
        unclaimed.join("\n  ")
    )
}

/// Whether the tier or the GPU excuse list covers `(package, test)`.
fn is_claimed(package: &str, test: Option<&str>) -> bool {
    HEAVY_TARGETS
        .iter()
        .any(|entry| entry.package == package && entry.test == test)
        || GPU_ONLY
            .iter()
            .any(|(excused, target, _)| *excused == package && *target == test)
}

/// The first file under `dir` carrying an `#[ignore]` attribute, if any.
///
/// Recursive: a crate's unit tests are one cargo target no matter how deeply
/// its modules nest.
fn find_ignored_test(dir: &Path) -> Result<Option<PathBuf>> {
    if !dir.is_dir() {
        return Ok(None);
    }

    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            if let Some(found) = find_ignored_test(&path)? {
                return Ok(Some(found));
            }
        } else if has_ignored_test(&path)? {
            return Ok(Some(path));
        }
    }
    Ok(None)
}

/// The `tests/*.rs` files under `dir` that carry an `#[ignore]` attribute,
/// paired with the target name cargo derives from each.
///
/// Deliberately not recursive: only top-level files in `tests/` become test
/// targets, and a nested `tests/common/` is a helper module of one of them.
fn integration_targets_with_ignored_tests(dir: &Path) -> Result<Vec<(String, PathBuf)>> {
    if !dir.is_dir() {
        return Ok(Vec::new());
    }

    let mut targets = Vec::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().is_none_or(|ext| ext != "rs") || !has_ignored_test(&path)? {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        targets.push((stem.to_string(), path.clone()));
    }
    targets.sort();
    Ok(targets)
}

/// Whether `path` is a Rust source file containing an `#[ignore]` attribute.
fn has_ignored_test(path: &Path) -> Result<bool> {
    if path.extension().is_none_or(|ext| ext != "rs") {
        return Ok(false);
    }
    let source = fs::read_to_string(path)?;
    Ok(source
        .lines()
        .any(|line| line.trim_start().starts_with("#[ignore")))
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
