//! GPU tier — the `#[ignore]`d tests that need a wgpu adapter. **Local only.**
//!
//! These are the targets `test-heavy` deliberately does not reach. Nothing in
//! `.github/workflows/` invokes this command and nothing should: every runner is
//! `ubuntu-latest` with no GPU, and initializing a wgpu device there *aborts*
//! rather than failing — cubecl-wgpu panics on a worker thread and the calling
//! thread then panics on the severed channel. There is no clean in-process probe
//! across the cubecl boundary, so this tier cannot guard itself; running it on a
//! GPU-less host produces an abort, not a skip.
//!
//! What that costs is worth stating plainly: the defects these tests pin have no
//! automated protection at all. Each one is the only thing in the workspace that
//! can observe a backend divergence, and it fires only when a human remembers to
//! run it. That is what this command is for — making "remembering" one line.
//!
//! [`GPU_TARGETS`] is the single list of GPU-only targets. `test-heavy` reads it
//! through [`claims`] to know what to leave alone, so a target added here is
//! excluded there automatically; the two tiers cannot disagree about which side
//! of the line a target sits on.
//!
//! # Examples
//!
//! ```ignore
//! cargo xtask test-gpu
//! cargo xtask test-gpu backend_parity
//! cargo xtask test-gpu -x rlevo-reinforcement-learning:lib
//! ```

use anyhow::Result;
use tracel_xtask::prelude::*;

use super::tiers;

/// A test target whose `#[ignore]`d tests need a wgpu adapter.
pub struct GpuTarget {
    /// Package name, as `cargo -p` spells it.
    package: &'static str,
    /// The `tests/<name>.rs` binary, or `None` for the crate's unit tests
    /// (`--lib`).
    test: Option<&'static str>,
    /// What the target's ignored tests pin, and why only a GPU can show it.
    reason: &'static str,
}

impl GpuTarget {
    /// The tier's selection key for this target.
    fn label(&self) -> String {
        tiers::target_label(self.package, self.test)
    }
}

/// The tier — and the workspace's only list of GPU-only test targets.
///
/// `test-heavy` excludes exactly these, so adding an entry here is what moves a
/// target out of the CI tier. Keep the pair total over the workspace's
/// `#[ignore]`d tests; `tiers::assert_ignored_tests_are_claimed` enforces it.
pub const GPU_TARGETS: &[GpuTarget] = &[
    GpuTarget {
        package: "rlevo-evolution",
        test: Some("backend_parity"),
        reason: "GA and PSO on Sphere-D10 must reach a non-trivial optimum on \
                 wgpu as well as Flex — the pure-tensor operators composing \
                 correctly on a real device",
    },
    GpuTarget {
        package: "rlevo-reinforcement-learning",
        test: Some("c51_projection_backend_parity"),
        reason: "a NaN reward must stay loud through the C51 projection on \
                 Metal, where the WGSL `clamp` builtin rescues NaN to `lo` and \
                 once produced a well-formed, wrongly-confident probability row",
    },
    GpuTarget {
        package: "rlevo-reinforcement-learning",
        test: None,
        reason: "`clamp_preserving_nan_matches_flex_on_wgpu` in \
                 `algorithms::shared` — the unit-level half of the same NaN \
                 divergence: Flex propagates it, Metal rescues it to `lo`",
    },
];

/// Whether the GPU tier covers `(package, test)`.
///
/// `tiers::assert_ignored_tests_are_claimed` asks both ignored-test tiers this,
/// so each owns its own table and neither restates the other's.
pub fn claims(package: &str, test: Option<&str>) -> bool {
    GPU_TARGETS
        .iter()
        .any(|entry| entry.package == package && entry.test == test)
}

#[macros::declare_command_args(None, None)]
pub struct TestGpuCmdArgs {
    /// Test targets to run. Defaults to every target in the tier.
    #[arg(value_name = "TARGET")]
    pub only: Vec<String>,
    /// Comma-separated list of test targets to skip.
    #[arg(
        short = 'x',
        long,
        value_name = "TARGET,TARGET,...",
        value_delimiter = ',',
        required = false
    )]
    pub exclude: Vec<String>,
}

/// Run the GPU tier, stopping at the first target that fails.
///
/// Requires a working wgpu adapter (Metal or Vulkan). On a host without one this
/// aborts inside cubecl rather than returning an error — see the module docs.
///
/// `args` is taken by reference, unlike the by-value base `tracel-xtask`
/// handlers — nothing here consumes it, and `clippy::pedantic` is warned
/// workspace-wide.
///
/// # Errors
///
/// Returns an error if an ignored test is claimed by no tier, if the caller
/// named a target outside this tier, or if any target's tests fail.
pub fn handle_command(args: &TestGpuCmdArgs, _env: Environment, _ctx: Context) -> Result<()> {
    tiers::assert_ignored_tests_are_claimed()?;
    reject_unknown_selection(args)?;

    log::info!(
        "The GPU tier needs a wgpu adapter. On a host without one cubecl aborts \
         the process instead of failing a test — there is no in-process probe to \
         turn that into a clean skip."
    );

    for entry in GPU_TARGETS {
        run_gpu_test(entry, args)?;
    }
    Ok(())
}

/// Fail if the user named a target the tier does not contain.
///
/// Without this, a typo selects nothing, every target is skipped, and the tier
/// exits zero having tested nothing at all.
fn reject_unknown_selection(args: &TestGpuCmdArgs) -> Result<()> {
    let labels: Vec<String> = GPU_TARGETS.iter().map(GpuTarget::label).collect();
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
        "not in the GPU tier: {}\nAvailable targets: {}",
        unknown.join(", "),
        labels.join(", ")
    )
}

fn run_gpu_test(entry: &GpuTarget, args: &TestGpuCmdArgs) -> Result<()> {
    let label = entry.label();
    group!("GPU Tests: {} — {}", label, entry.reason);

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
    // `--ignored` runs *only* the ignored tests. For the `--lib` entry that
    // matters twice over: the crate's non-ignored unit tests are the fast tier's
    // job, and running them here would put hundreds of CPU tests in front of the
    // one test this tier exists for.
    cmd_args.extend([
        "--".to_string(),
        "--ignored".to_string(),
        "--color=always".to_string(),
    ]);

    run_process_for_package(
        "cargo",
        &label,
        &cmd_args.iter().map(String::as_str).collect::<Vec<_>>(),
        None,
        &args.exclude,
        &args.only,
        &format!("GPU tests failed for '{label}'"),
        None,
        None,
    )?;

    endgroup!();
    Ok(())
}
