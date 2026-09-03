//! Fast tier — every crate's default test set, the pull-request gate.
//!
//! One `cargo test -p <crate>` per workspace crate. [`FAST_CRATES`] is the
//! gate's source of truth: `.github/workflows/crate-tests.yml` builds its `test`
//! matrix from `--list` rather than restating the crates in YAML. Two properties
//! are load-bearing and are enforced here rather than left to convention:
//!
//! 1. **The tier is total over the workspace.** A crate absent from the tier
//!    has its tests run by nothing. `assert_tier_is_total` compares the table
//!    below against `cargo metadata` on every invocation, so a newly added
//!    crate fails the tier instead of quietly falling out of it.
//! 2. **Features are per-crate.** A bare `cargo test -p <crate>` *skips* a test
//!    target whose `required-features` are unmet, and compiles a
//!    `#![cfg(feature = ...)]` file to nothing — in both cases silently, with a
//!    zero exit code. The `features` column records what each crate needs.
//!
//! # Examples
//!
//! ```ignore
//! cargo xtask test-fast
//! cargo xtask test-fast rlevo-core
//! cargo xtask test-fast -x rlevo-examples
//! cargo xtask test-fast --list
//! ```

use anyhow::Result;
use tracel_xtask::prelude::*;
use tracel_xtask::utils::workspace::{WorkspaceMemberType, get_workspace_members};

/// A crate in the fast tier and the features its test targets need.
struct FastCrate {
    /// Package name, as `cargo -p` spells it.
    name: &'static str,
    /// Features to enable on top of the crate's defaults.
    ///
    /// Empty means every test target is reachable from the default feature set.
    /// A non-empty list means a bare `cargo test -p <crate>` under-tests the
    /// crate without saying so.
    features: &'static [&'static str],
}

/// The tier. Keep total over `crates/*/` — `assert_tier_is_total` enforces it.
const FAST_CRATES: &[FastCrate] = &[
    FastCrate {
        name: "rlevo-core",
        features: &[],
    },
    // `tests/wire_format_compat.rs` (native `EpisodeRecord` vs. browser-client
    // wire-format mirror parity) is `#![cfg(feature = "record")]`, and `record`
    // is not among this crate's defaults (`json` only). Without the override
    // the file compiles to nothing and the test never runs.
    FastCrate {
        name: "rlevo-benchmarks",
        features: &["record"],
    },
    FastCrate {
        name: "rlevo-benchmarks-report-client",
        features: &[],
    },
    FastCrate {
        name: "rlevo-environments",
        features: &[],
    },
    FastCrate {
        name: "rlevo-evolution",
        features: &[],
    },
    // The `report_*_with_client` examples carry `required-features` that this
    // tier does not enable, so `cargo test` compiles none of them. Widening the
    // tier to cover them is a deliberate open question, not an oversight — the
    // pre-existing CI matrix has the same gap.
    FastCrate {
        name: "rlevo-examples",
        features: &[],
    },
    FastCrate {
        name: "rlevo-hybrid",
        features: &[],
    },
    FastCrate {
        name: "rlevo-metrics-registry",
        features: &[],
    },
    FastCrate {
        name: "rlevo-reinforcement-learning",
        features: &[],
    },
    FastCrate {
        name: "rlevo-test-support",
        features: &[],
    },
    // `tests/cartpole_report_smoke.rs` and `tests/recording_episode_count.rs`
    // are gated behind `required-features = ["viz-report"]`, which is not among
    // `rlevo`'s defaults. A bare `cargo test -p rlevo` skips both binaries.
    FastCrate {
        name: "rlevo",
        features: &["viz-report"],
    },
];

/// Workspace members deliberately outside the tier, each with its reason.
///
/// Add here rather than dropping a member from `FAST_CRATES` — a silent
/// omission is the failure mode `assert_tier_is_total` exists to prevent.
const NOT_TESTED: &[(&str, &str)] = &[("xtask", "the build tooling itself; it carries no tests")];

#[macros::declare_command_args(None, None)]
pub struct TestFastCmdArgs {
    /// Crates to test. Defaults to every crate in the tier.
    #[arg(value_name = "CRATE")]
    pub only: Vec<String>,
    /// Comma-separated list of crates to skip.
    #[arg(
        short = 'x',
        long,
        value_name = "CRATE,CRATE,...",
        value_delimiter = ',',
        required = false
    )]
    pub exclude: Vec<String>,
    /// Print the tier's crates as a JSON array and exit, running nothing.
    ///
    /// `crate-tests.yml` builds its matrix from this with `fromJSON`, so the
    /// tier table is the workflow's single source of truth rather than a list
    /// the YAML restates and then drifts from.
    #[arg(long, default_value_t = false)]
    pub list: bool,
}

/// Run the fast tier, stopping at the first crate that fails.
///
/// `args` is taken by reference, unlike the by-value base `tracel-xtask`
/// handlers — nothing here consumes it, and `clippy::pedantic` is warned
/// workspace-wide.
///
/// # Errors
///
/// Returns an error if the tier has drifted from the workspace, if the caller
/// named a crate outside the tier, or if any crate's tests fail.
pub fn handle_command(args: &TestFastCmdArgs, _env: Environment, _ctx: Context) -> Result<()> {
    assert_tier_is_total()?;

    if args.list {
        crate::commands::print_json_list(FAST_CRATES.iter().map(|entry| entry.name.to_string()));
        return Ok(());
    }

    reject_unknown_selection(args)?;

    for entry in FAST_CRATES {
        run_fast_test(entry, args)?;
    }
    Ok(())
}

/// Fail if any workspace member is neither in the tier nor explicitly excused.
///
/// This replaced a `crate-tests.yml` job that grepped the workflow YAML for each
/// `crates/*/` directory name. Reading `cargo metadata` makes the check exact:
/// it sees the real member list, not a directory listing that happens to
/// resemble one — `xtask` itself is a member and `crates/*/` never saw it.
fn assert_tier_is_total() -> Result<()> {
    let missing: Vec<String> = get_workspace_members(WorkspaceMemberType::Crate)
        .into_iter()
        .map(|member| member.name)
        .filter(|name| {
            !FAST_CRATES.iter().any(|entry| entry.name == name)
                && !NOT_TESTED.iter().any(|(excused, _)| excused == name)
        })
        .collect();

    if missing.is_empty() {
        return Ok(());
    }

    anyhow::bail!(
        "workspace crates absent from the fast tier: {}\n\
         A crate outside the tier has its tests run by no pull-request check.\n\
         Add it to FAST_CRATES, or to NOT_TESTED with a reason.",
        missing.join(", ")
    )
}

/// Fail if the user named a crate the tier does not contain.
///
/// Without this, a typo selects nothing, every crate is skipped, and the tier
/// exits zero having tested nothing at all.
fn reject_unknown_selection(args: &TestFastCmdArgs) -> Result<()> {
    let unknown: Vec<&str> = args
        .only
        .iter()
        .chain(args.exclude.iter())
        .map(String::as_str)
        .filter(|name| !FAST_CRATES.iter().any(|entry| entry.name == *name))
        .collect();

    if unknown.is_empty() {
        return Ok(());
    }

    anyhow::bail!(
        "not in the fast tier: {}\nAvailable crates: {}",
        unknown.join(", "),
        FAST_CRATES
            .iter()
            .map(|entry| entry.name)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn run_fast_test(entry: &FastCrate, args: &TestFastCmdArgs) -> Result<()> {
    group!("Fast Tests: {}", entry.name);

    let mut cmd_args = vec!["test".to_string(), "-p".to_string(), entry.name.to_string()];
    if !entry.features.is_empty() {
        cmd_args.extend(["--features".to_string(), entry.features.join(",")]);
    }
    cmd_args.extend(["--".to_string(), "--color=always".to_string()]);

    // `ignore_log` is deliberately `None`. The base `test` command tolerates
    // "no library targets found" so that a bin-only member does not fail the
    // run; in this tier every member has a library and that message would mean
    // the crate stopped being tested.
    run_process_for_package(
        "cargo",
        &entry.name.to_string(),
        &cmd_args.iter().map(String::as_str).collect::<Vec<_>>(),
        None,
        &args.exclude,
        &args.only,
        &format!("Fast tests failed for '{}'", entry.name),
        None,
        None,
    )?;

    endgroup!();
    Ok(())
}
