//! Layering guards: the rules about which crate may depend on which, checked
//! on every pull request instead of left to review.
//!
//! The toolchain enforces none of these rules. A crate can add a dependency,
//! a feature or an import that breaks one and still build, test and lint
//! clean, so a rule that is only written down goes quiet the day it is
//! broken. Each guard reads one rule where it can be broken (a manifest, the
//! resolved graph or a source tree) and says what breaks it: a file and
//! line, a crate and its feature, the copy that differs, or, for a viz
//! package, the `cargo tree -i` command that prints the path to it.
//!
//! | Guard | Rule | Reads |
//! |---|---|---|
//! | `evolution-names-no-environment` | `rlevo-evolution` never names the environment contract (ADR 0002) | its manifest and every `.rs` file under it |
//! | `harness-edge-is-record-only` | no published crate depends on `rlevo-harness` but the umbrella, and `rlevo-environments` under `record` (`docs/rules.md` section 10, ADR 0084) | `cargo metadata` |
//! | `umbrella-defaults-carry-no-viz` | `rlevo`'s default features turn on no viz, recording or `rerun` feature (ADR 0084) | `cargo metadata` |
//! | `published-graphs-reach-no-viz` | no published crate reaches a viz package at its default features, in its build or its test build (`docs/rules.md` section 10) | `cargo tree` |
//! | `katex-headers-identical` | the `katex-header.html` copies stay byte-identical (`CLAUDE.md`) | the files |
//!
//! One line per guard goes to stderr, and the last line is the verdict:
//!
//! ```text
//! GUARD PASS evolution-names-no-environment :: 81 source files, none names the environment contract
//! GUARD FAIL katex-headers-identical :: crates/rlevo-hybrid/katex-header.html differs from crates/rlevo-core/katex-header.html
//! 1 of 5 guards failed.
//! ```
//!
//! A failed guard exits non-zero. Fix the code, or change the rule, its
//! record and this file together: a guard that is edited to pass has stopped
//! guarding.
//!
//! # Examples
//!
//! ```ignore
//! cargo xtask guards
//! ```

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context as _, Result, bail};
use serde_json::Value;
use tracel_xtask::prelude::*;

/// Packages that are viz or viewer stacks. No published crate reaches one at
/// its default features: the harness's are behind `tui`, `report` and
/// `record`, and `rerun` belongs to an adapter crate that is never a default.
const VIZ_PACKAGES: &[&str] = &[
    "ratatui",
    "crossterm",
    "leptos",
    "axum",
    "wasm-bindgen",
    "rerun",
];

/// The crate whose sources must not name the environment contract.
const EVOLUTION: &str = "rlevo-evolution";

/// The harness, and the two published crates allowed a normal edge to it.
const HARNESS: &str = "rlevo-harness";
const UMBRELLA: &str = "rlevo";
const ENVIRONMENTS: &str = "rlevo-environments";

/// The one feature of `rlevo-environments` that may enable its harness edge.
const RECORD: &str = "record";

#[macros::declare_command_args(None, None)]
pub struct GuardsCmdArgs {}

/// A guard's result: the violations it found, or what it checked.
struct Verdict {
    name: &'static str,
    /// Empty when the guard passes.
    violations: Vec<String>,
    /// What a pass covered, so a guard that checked nothing is visible.
    checked: String,
}

/// Run every guard and report each.
///
/// # Errors
///
/// Returns an error if `cargo metadata` or `cargo tree` cannot be run, a
/// source file cannot be read, or any guard fails.
pub fn handle_command(_args: &GuardsCmdArgs, _env: Environment, _ctx: Context) -> Result<()> {
    let metadata = workspace_metadata()?;
    let root = PathBuf::from(
        metadata["workspace_root"]
            .as_str()
            .context("cargo metadata reported no workspace root")?,
    );

    let verdicts = vec![
        evolution_names_no_environment(&metadata)?,
        harness_edge_is_record_only(&metadata),
        umbrella_defaults_carry_no_viz(&metadata)?,
        published_graphs_reach_no_viz(&metadata)?,
        katex_headers_identical(&root)?,
    ];

    let failed = verdicts.iter().filter(|v| !v.violations.is_empty()).count();
    for verdict in &verdicts {
        if verdict.violations.is_empty() {
            eprintln!("GUARD PASS {} :: {}", verdict.name, verdict.checked);
        } else {
            eprintln!(
                "GUARD FAIL {} :: {}",
                verdict.name,
                verdict.violations.join("; ")
            );
        }
    }
    if failed > 0 {
        bail!("{failed} of {} guards failed.", verdicts.len());
    }
    eprintln!("all {} guards pass.", verdicts.len());
    Ok(())
}

/// `cargo metadata --no-deps`: every member's manifest as cargo reads it.
fn workspace_metadata() -> Result<Value> {
    let output = Command::new("cargo")
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .output()
        .context("could not run cargo metadata")?;
    if !output.status.success() {
        bail!(
            "cargo metadata failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn packages(metadata: &Value) -> impl Iterator<Item = &Value> {
    metadata["packages"].as_array().into_iter().flatten()
}

fn package<'a>(metadata: &'a Value, name: &str) -> Option<&'a Value> {
    packages(metadata).find(|p| p["name"] == name)
}

/// `publish = false` is reported as an empty registry list.
fn is_published(package: &Value) -> bool {
    !package["publish"].as_array().is_some_and(Vec::is_empty)
}

/// Every string a feature reaches through the package's own feature table:
/// the features it turns on, and the `dep:` and `crate/feature` entries they
/// name.
fn feature_closure(package: &Value, feature: &str) -> BTreeSet<String> {
    let table = &package["features"];
    let mut reached = BTreeSet::new();
    let mut pending = vec![feature.to_string()];
    while let Some(next) = pending.pop() {
        if !reached.insert(next.clone()) {
            continue;
        }
        for value in table[next.as_str()].as_array().into_iter().flatten() {
            if let Some(value) = value.as_str() {
                pending.push(value.to_string());
            }
        }
    }
    reached
}

// ---- evolution-names-no-environment ----------------------------------------

/// Whether a source line names a path into the environment contract or the
/// environments crate.
///
/// Matches paths, not words: NEAT has sensor nodes and a comment may
/// snapshot something, but no line of `rlevo-evolution` should reach
/// `rlevo_core::environment` (where `Environment`, `Snapshot` and
/// `EpisodeStatus` live) or `rlevo_environments`, in code or in a doc link.
fn names_environment_path(line: &str) -> bool {
    if line.contains("rlevo_environments") || line.contains("rlevo_core::environment") {
        return true;
    }
    // `environment::X` after a grouped import, `use rlevo_core::{environment, ..}`.
    line.match_indices("environment::").any(|(at, _)| {
        !line[..at]
            .chars()
            .next_back()
            .is_some_and(|c| c.is_alphanumeric() || c == '_')
    })
}

fn evolution_names_no_environment(metadata: &Value) -> Result<Verdict> {
    let name = "evolution-names-no-environment";
    let evolution = package(metadata, EVOLUTION).context("no rlevo-evolution member")?;
    let mut violations = Vec::new();

    for dep in evolution["dependencies"].as_array().into_iter().flatten() {
        if dep["name"] == ENVIRONMENTS {
            violations.push(format!("{EVOLUTION}'s manifest declares {ENVIRONMENTS}"));
        }
    }

    let dir = Path::new(
        evolution["manifest_path"]
            .as_str()
            .context("no manifest path")?,
    )
    .parent()
    .context("manifest path has no directory")?
    .to_path_buf();
    let mut files = Vec::new();
    rust_files(&dir, &mut files)?;
    files.sort();
    for file in &files {
        let text = fs::read_to_string(file)
            .with_context(|| format!("could not read {}", file.display()))?;
        for (index, line) in text.lines().enumerate() {
            if names_environment_path(line) {
                violations.push(format!(
                    "{}:{} names the environment contract",
                    file.strip_prefix(&dir).unwrap_or(file).display(),
                    index + 1
                ));
            }
        }
    }

    Ok(Verdict {
        name,
        violations,
        checked: format!(
            "{} source files, none names the environment contract",
            files.len()
        ),
    })
}

/// Every `.rs` file under `dir`, skipping build output.
fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir).with_context(|| format!("could not list {}", dir.display()))? {
        let path = entry?.path();
        if path.is_dir() {
            if path.file_name().is_some_and(|n| n == "target") {
                continue;
            }
            rust_files(&path, out)?;
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
    Ok(())
}

// ---- harness-edge-is-record-only -------------------------------------------

/// Whether a feature-table entry turns on the optional harness dependency.
/// A weak `rlevo-harness?/x` does not; it applies only once something else
/// has.
fn enables_harness(value: &str) -> bool {
    value == HARNESS
        || value == format!("dep:{HARNESS}")
        || value.starts_with(&format!("{HARNESS}/"))
}

fn harness_edge_is_record_only(metadata: &Value) -> Verdict {
    let name = "harness-edge-is-record-only";
    let mut violations = Vec::new();
    let mut edges = 0;

    for pkg in packages(metadata).filter(|p| is_published(p) && p["name"] != HARNESS) {
        let pkg_name = pkg["name"].as_str().unwrap_or_default();
        // A dev-dependency never reaches a consumer.
        for dep in pkg["dependencies"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|d| d["name"] == HARNESS && d["kind"] != "dev")
        {
            edges += 1;
            match pkg_name {
                UMBRELLA => {}
                ENVIRONMENTS => {
                    if dep["optional"] != true {
                        violations.push(format!(
                            "{ENVIRONMENTS} depends on {HARNESS} unconditionally; only `{RECORD}` may enable it"
                        ));
                    }
                    let enabling: BTreeSet<&str> = pkg["features"]
                        .as_object()
                        .into_iter()
                        .flatten()
                        .filter(|(_, values)| {
                            values
                                .as_array()
                                .into_iter()
                                .flatten()
                                .filter_map(Value::as_str)
                                .any(enables_harness)
                        })
                        .map(|(feature, _)| feature.as_str())
                        .collect();
                    for feature in enabling.iter().filter(|f| **f != RECORD) {
                        violations.push(format!(
                            "{ENVIRONMENTS}'s `{feature}` feature enables {HARNESS}; only `{RECORD}` may"
                        ));
                    }
                    if feature_closure(pkg, "default").contains(RECORD) {
                        violations.push(format!(
                            "{ENVIRONMENTS}'s default features reach `{RECORD}`"
                        ));
                    }
                }
                other => violations.push(format!(
                    "{other} depends on {HARNESS}; no production crate does but {UMBRELLA} and {ENVIRONMENTS} under `{RECORD}`"
                )),
            }
        }
    }

    Verdict {
        name,
        violations,
        checked: format!("{edges} published edges to {HARNESS}, each allowed"),
    }
}

// ---- umbrella-defaults-carry-no-viz ----------------------------------------

/// Whether a default-feature entry of the umbrella turns on a viz, recording
/// or `rerun` gate.
fn is_viz_gate(entry: &str) -> bool {
    let gated = |feature: &str| {
        [HARNESS, ENVIRONMENTS]
            .iter()
            .any(|dep| entry == format!("{dep}/{feature}") || entry == format!("{dep}?/{feature}"))
    };
    entry.starts_with("viz-")
        || entry.contains("rerun")
        || gated("tui")
        || gated("report")
        || gated(RECORD)
}

fn umbrella_defaults_carry_no_viz(metadata: &Value) -> Result<Verdict> {
    let name = "umbrella-defaults-carry-no-viz";
    let umbrella = package(metadata, UMBRELLA).context("no rlevo member")?;
    let reached = feature_closure(umbrella, "default");
    let violations = reached
        .iter()
        .filter(|entry| is_viz_gate(entry))
        .map(|entry| format!("{UMBRELLA}'s default features reach `{entry}`"))
        .collect();
    Ok(Verdict {
        name,
        violations,
        checked: format!(
            "{UMBRELLA}'s default features reach {} entries, none a viz gate",
            reached.len()
        ),
    })
}

// ---- published-graphs-reach-no-viz -----------------------------------------

/// The package names in one crate's resolved graph at its default features,
/// normal and dev edges: what a consumer builds, and what its own tests build.
fn resolved_names(crate_name: &str) -> Result<BTreeSet<String>> {
    let output = Command::new("cargo")
        .args(["tree", "-e", "normal,dev", "-p", crate_name])
        .args(["--prefix", "none", "-f", "{p}"])
        .output()
        .context("could not run cargo tree")?;
    if !output.status.success() {
        // An empty tree would pass; a failed one must not.
        bail!(
            "cargo tree -p {crate_name} failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .map(str::to_string)
        .collect())
}

fn published_graphs_reach_no_viz(metadata: &Value) -> Result<Verdict> {
    let name = "published-graphs-reach-no-viz";
    let mut violations = Vec::new();
    let mut crates = 0;
    for pkg in packages(metadata).filter(|p| is_published(p)) {
        let crate_name = pkg["name"].as_str().unwrap_or_default();
        crates += 1;
        let names = resolved_names(crate_name)?;
        for viz in VIZ_PACKAGES.iter().filter(|viz| names.contains(**viz)) {
            violations.push(format!(
                "{crate_name} reaches {viz} at its default features (`cargo tree -e normal,dev -p {crate_name} -i {viz}`)"
            ));
        }
    }
    if crates == 0 {
        violations.push("no published crate found; this guard checks nothing".to_string());
    }
    Ok(Verdict {
        name,
        violations,
        checked: format!(
            "{crates} published crates, none reaches {}",
            VIZ_PACKAGES.join(", ")
        ),
    })
}

// ---- katex-headers-identical -----------------------------------------------

fn katex_headers_identical(root: &Path) -> Result<Verdict> {
    let name = "katex-headers-identical";
    let mut headers = Vec::new();
    for entry in fs::read_dir(root.join("crates"))? {
        let header = entry?.path().join("katex-header.html");
        if header.is_file() {
            headers.push(header);
        }
    }
    headers.sort();

    let mut violations = Vec::new();
    if let Some((first, rest)) = headers.split_first() {
        let reference = fs::read(first)?;
        for header in rest {
            if fs::read(header)? != reference {
                violations.push(format!(
                    "{} differs from {}",
                    header.strip_prefix(root).unwrap_or(header).display(),
                    first.strip_prefix(root).unwrap_or(first).display()
                ));
            }
        }
    } else {
        violations
            .push("no crates/*/katex-header.html found; this guard checks nothing".to_string());
    }

    Ok(Verdict {
        name,
        violations,
        checked: format!("{} copies, byte-identical", headers.len()),
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn package_with(name: &str, deps: &Value, features: &Value) -> Value {
        json!({
            "name": name,
            "publish": null,
            "dependencies": deps,
            "features": features,
        })
    }

    fn harness_dep(optional: bool) -> Value {
        json!({ "name": "rlevo-harness", "kind": null, "optional": optional })
    }

    fn environments(optional: bool, features: &Value) -> Value {
        package_with(
            "rlevo-environments",
            &json!([harness_dep(optional)]),
            features,
        )
    }

    fn workspace(packages: &[Value]) -> Value {
        json!({ "packages": packages })
    }

    #[test]
    fn feature_closure_follows_features_and_keeps_dependency_entries() {
        let pkg = package_with(
            "x",
            &json!([]),
            &json!({
                "default": ["a"],
                "a": ["b", "dep:y", "z/feat"],
                "b": [],
                "c": ["z/other"],
            }),
        );
        let reached = feature_closure(&pkg, "default");
        let expected: BTreeSet<String> = ["default", "a", "b", "dep:y", "z/feat"]
            .into_iter()
            .map(str::to_string)
            .collect();
        assert_eq!(reached, expected);
    }

    #[test]
    fn environment_paths_are_found_and_words_are_not() {
        for line in [
            "use rlevo_core::environment::Snapshot;",
            "use rlevo_core::{environment::Environment, fitness::Landscape};",
            "/// [`Snapshot`](rlevo_core::environment::Snapshot)",
            "use rlevo_environments::classic::CartPole;",
            "    let s: environment::Snapshot<f32> = todo!();",
        ] {
            assert!(names_environment_path(line), "{line}");
        }
        for line in [
            "/// Sensor node: holds an input value verbatim.",
            "// Snapshot the archives under the lock.",
            "/// An environment rollout scores a genome.",
            "let my_environment::x = 1;",
        ] {
            assert!(!names_environment_path(line), "{line}");
        }
    }

    #[test]
    fn harness_edge_passes_for_the_umbrella_and_a_record_gated_environments_edge() {
        let metadata = workspace(&[
            package_with("rlevo", &json!([harness_dep(false)]), &json!({})),
            environments(
                true,
                &json!({ "default": ["box2d"], "box2d": [], "record": ["dep:rlevo-harness", "rlevo-harness/record"] }),
            ),
            // A dev-dependency never reaches a consumer.
            package_with(
                "rlevo-core",
                &json!([{ "name": "rlevo-harness", "kind": "dev", "optional": false }]),
                &json!({}),
            ),
        ]);
        assert_eq!(
            harness_edge_is_record_only(&metadata).violations,
            Vec::<String>::new()
        );
    }

    #[test]
    fn harness_edge_fails_for_a_third_crate() {
        let metadata = workspace(&[package_with(
            "rlevo-reinforcement-learning",
            &json!([harness_dep(false)]),
            &json!({}),
        )]);
        assert_eq!(harness_edge_is_record_only(&metadata).violations.len(), 1);
    }

    #[test]
    fn harness_edge_fails_for_a_required_or_widely_enabled_environments_edge() {
        let required = workspace(&[environments(false, &json!({ "record": [] }))]);
        assert_eq!(harness_edge_is_record_only(&required).violations.len(), 1);

        let second_feature = workspace(&[environments(
            true,
            &json!({ "record": ["dep:rlevo-harness"], "bench": ["rlevo-harness/record"] }),
        )]);
        assert_eq!(
            harness_edge_is_record_only(&second_feature)
                .violations
                .len(),
            1
        );

        let record_by_default = workspace(&[environments(
            true,
            &json!({ "default": ["record"], "record": ["dep:rlevo-harness"] }),
        )]);
        assert_eq!(
            harness_edge_is_record_only(&record_by_default)
                .violations
                .len(),
            1
        );
    }

    #[test]
    fn a_weak_feature_entry_does_not_enable_the_harness() {
        assert!(!enables_harness("rlevo-harness?/record"));
        assert!(enables_harness("rlevo-harness/record"));
        assert!(enables_harness("dep:rlevo-harness"));
    }

    #[test]
    fn umbrella_defaults_fail_on_any_viz_gate_they_reach() {
        let clean = workspace(&[package_with(
            "rlevo",
            &json!([]),
            &json!({ "default": ["flex"], "flex": ["burn/flex"], "viz-tui": ["rlevo-harness/tui"] }),
        )]);
        assert_eq!(
            umbrella_defaults_carry_no_viz(&clean).unwrap().violations,
            Vec::<String>::new()
        );

        for default in [
            json!(["viz-tui"]),
            json!(["rlevo-harness/report"]),
            json!(["rlevo-environments/record"]),
            json!(["rerun"]),
        ] {
            let dirty = workspace(&[package_with(
                "rlevo",
                &json!([]),
                &json!({ "default": default, "viz-tui": ["rlevo-harness/tui"], "rerun": [] }),
            )]);
            assert!(
                !umbrella_defaults_carry_no_viz(&dirty)
                    .unwrap()
                    .violations
                    .is_empty(),
                "{default}"
            );
        }
    }
}
