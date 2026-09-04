//! Machinery shared by the test tiers.
//!
//! The three tiers partition the workspace's tests: `test-fast` runs each
//! crate's default set, and `test-heavy` and `test-gpu` split the `#[ignore]`d
//! remainder between what a CI runner can execute and what needs a wgpu adapter.
//! That partition is the invariant [`assert_ignored_tests_are_claimed`] checks,
//! and it is why the check lives here rather than inside either tier: an ignored
//! test claimed by *neither* is run by nothing at all.

use std::fs;
use std::path::{Path, PathBuf};

// `anyhow` reaches this crate through the prelude re-export, not a direct
// dependency — the same import the other command modules rely on.
use anyhow::Result;
use tracel_xtask::prelude::*;
use tracel_xtask::utils::workspace::{WorkspaceMemberType, get_workspace_members};

/// A tier's key for one cargo test target: what the user names to select it, and
/// what group headers and failure messages report.
///
/// `test` is the `tests/<name>.rs` binary, or `None` for the crate's unit tests.
pub fn target_label(package: &str, test: Option<&str>) -> String {
    match test {
        Some(name) => name.to_string(),
        None => format!("{package}:lib"),
    }
}

/// Print `labels` as a JSON array on stdout, for a GitHub Actions matrix built
/// with `fromJSON`.
///
/// Every label a tier produces is a package or test-target name, so no JSON
/// string needs escaping. xtask's log goes to stderr, leaving stdout clean for
/// `$(cargo xtask <tier> --list)`.
pub fn print_json_list(labels: impl IntoIterator<Item = String>) {
    let quoted: Vec<String> = labels
        .into_iter()
        .map(|label| format!("\"{label}\""))
        .collect();
    println!("[{}]", quoted.join(","));
}

/// Fail if any target hosting `#[ignore]`d tests is claimed by neither the heavy
/// tier nor the GPU tier.
///
/// The fast tier answers the analogous question with `cargo metadata`, because
/// there "is this crate covered?" is a property of the member list. Here the
/// unit is a test target that *contains an ignored test*, which no cargo command
/// reports, so the check reads the sources: any line whose first non-whitespace
/// is `#[ignore` is an attribute (a `//`/`///`/`//!` mention is not), including
/// the ones passed through `rl_learning_test!` and `rl_reproducibility_test!`.
///
/// Scope: a crate's `src/` tree maps to its `--lib` target and each top-level
/// `tests/*.rs` to the integration binary cargo auto-discovers from it. Ignored
/// tests reached only through a `#[path]`-included module outside those two
/// trees are not seen.
///
/// # Errors
///
/// Returns an error naming every unclaimed target, or any I/O error from reading
/// a workspace member's sources.
pub fn assert_ignored_tests_are_claimed() -> Result<()> {
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
        "test targets with #[ignore]d tests that no tier runs:\n  {}\n\
         An ignored test outside both tiers is run by nothing: the fast tier skips\n\
         it for being ignored, and neither remaining tier lists it.\n\
         Add the target to HEAVY_TARGETS (test_heavy.rs) if a CPU can run it, or\n\
         to GPU_TARGETS (test_gpu.rs) with a reason if it needs a wgpu adapter.",
        unclaimed.join("\n  ")
    )
}

/// Whether either of the two ignored-test tiers covers `(package, test)`.
fn is_claimed(package: &str, test: Option<&str>) -> bool {
    super::test_heavy::claims(package, test) || super::test_gpu::claims(package, test)
}

/// The first file under `dir` carrying an `#[ignore]` attribute, if any.
///
/// Recursive: a crate's unit tests are one cargo target no matter how deeply its
/// modules nest.
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
