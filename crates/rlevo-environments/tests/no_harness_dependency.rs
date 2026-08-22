//! Pins the direction of the `rlevo-environments` ↔ `rlevo-benchmarks` edge.
//!
//! The harness consumes environments. Environments must not consume the
//! harness — not unconditionally, and not behind an optional feature either
//! (ADR 0080, superseding the module-placement half of ADR 0001).
//!
//! # What this covers, and what cargo already covers
//!
//! Cargo covers more of this than it first appears, and the reason is worth
//! knowing: `rlevo-benchmarks` declares `rlevo-environments` as a normal
//! dependency (optional, behind `fixtures`), and **cargo's cycle detection
//! counts optional normal dependencies**. Re-adding
//! `rlevo-benchmarks = { …, optional = true }` here does not fail this test —
//! it fails the whole workspace, at resolution, before a single crate
//! compiles: *"cyclic package dependency: package `rlevo-benchmarks` depends
//! on itself"*. Verified 2026-08-22 by doing exactly that.
//!
//! Two gaps remain, and they are what this file is for:
//!
//! 1. **Dev-dependencies do not form a cycle.**
//!    `[dev-dependencies] rlevo-benchmarks = …` resolves fine and compiles
//!    fine. It is also the arrangement ADR 0001 rejected by name as "fragile"
//!    — it pulls the harness into `cargo test -p rlevo-environments`,
//!    `cargo doc`, and IDE indexing for anyone who only wanted an env.
//!    Verified: the mutant fails
//!    [`environments_does_not_depend_on_the_harness`] with the dependency
//!    list in the message.
//! 2. **Cargo's protection is contingent on `fixtures` existing.** It is a
//!    side effect of the edge this ADR created, not a rule anyone stated. If
//!    the `fixtures` feature were ever removed, the cycle detector would stop
//!    covering case 1's sibling and nothing else would notice.
//!
//! It reads the raw manifest text rather than parsing TOML on purpose: `toml`
//! is not a dependency of this crate, and adding one to police a dependency
//! rule would be its own small joke.

use std::path::Path;

/// Cargo sets `CARGO_MANIFEST_DIR` for integration tests, so this points at
/// the crate root regardless of the directory the test runner was invoked in.
const MANIFEST: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");

/// Every crate this manifest names as a dependency of any kind.
///
/// Deliberately includes `[dev-dependencies]` and `[build-dependencies]`: a
/// dev-dep on the harness would not form a cargo cycle, but it would still
/// drag the harness into `cargo test -p rlevo-environments` and `cargo doc`,
/// which is the "fragile" outcome ADR 0001 rejected under a different name.
fn declared_dependencies(manifest: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut in_deps = false;
    for line in manifest.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_deps = trimmed.contains("dependencies]");
            continue;
        }
        if !in_deps || trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some((name, _)) = trimmed.split_once('=') {
            out.push(name.trim().to_string());
        }
    }
    out
}

#[test]
fn environments_does_not_depend_on_the_harness() {
    let manifest = std::fs::read_to_string(Path::new(MANIFEST)).expect("read own Cargo.toml");
    let deps = declared_dependencies(&manifest);

    assert!(
        !deps.is_empty(),
        "parsed zero dependencies out of {MANIFEST} — the section-header \
         heuristic has drifted from the manifest layout, so this test would \
         pass vacuously",
    );

    assert!(
        !deps.iter().any(|d| d == "rlevo-benchmarks"),
        "`rlevo-environments` declares a dependency on `rlevo-benchmarks`. \
         The arrow points the other way: the harness owns `Suite`, \
         `EvaluatorConfig`, and `RecordedEnvFamily`, so its glue for the \
         built-in envs belongs in `rlevo-benchmarks::fixtures` (ADR 0080). \
         Declared dependencies: {deps:?}",
    );
}

/// The features that carried the inverted edge are gone, not merely emptied.
///
/// An empty `bench = []` would satisfy the test above while leaving the name
/// in place for someone to refill.
#[test]
fn the_bench_and_record_features_are_gone() {
    let manifest = std::fs::read_to_string(Path::new(MANIFEST)).expect("read own Cargo.toml");
    let mut in_features = false;
    for line in manifest.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_features = trimmed == "[features]";
            continue;
        }
        if !in_features || trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some((name, _)) = trimmed.split_once('=') else {
            continue;
        };
        let name = name.trim();
        assert!(
            name != "bench" && name != "record",
            "`{name}` is back in `rlevo-environments`'s [features]. Harness \
             wiring lives in `rlevo-benchmarks::fixtures` behind that crate's \
             `fixtures` feature (ADR 0080)",
        );
    }
}
