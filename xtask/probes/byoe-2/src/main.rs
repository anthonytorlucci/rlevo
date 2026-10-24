//! BYOE-2, well siting: a landscape author's walk from an empty directory to
//! a harness run of a shipped optimiser on their own cost surface, using
//! published crates and public API only.
//!
//! `cargo xtask probe byoe-2` builds this crate against the packaged
//! workspace and runs it. The steps a build can observe are `xtask`'s; this
//! binary reports the steps that need the code to run, one line each on
//! stderr, by label:
//!
//! ```text
//! PROBE-STEP <label> <PASS|FAIL> :: <detail>
//! ```
//!
//! Each step's code sits behind a cargo feature that includes the one before
//! it (`engine`, then `strategy`, then `harness`), so a build failure can be
//! reported at the step whose code broke. The features gate this crate's own
//! modules and never a dependency's feature.

// The lower levels compile a prefix of the walk, which leaves the items the
// later steps use without a caller.
#![cfg_attr(not(feature = "harness"), allow(dead_code))]

mod well_siting;

#[cfg(feature = "engine")]
mod engine;

#[cfg(feature = "strategy")]
mod strategy;

#[cfg(feature = "harness")]
mod harness;

fn main() {
    #[cfg(feature = "engine")]
    report("wrap-landscape", engine::check());
    #[cfg(feature = "strategy")]
    report("build-loop", strategy::check());
    #[cfg(feature = "harness")]
    report("run-trials", harness::run());
}

fn report(label: &str, outcome: Result<String, String>) {
    match outcome {
        Ok(detail) => eprintln!("PROBE-STEP {label} PASS :: {detail}"),
        Err(detail) => eprintln!("PROBE-STEP {label} FAIL :: {detail}"),
    }
}
