//! BYOA-1a: an algorithm author's walk from an empty directory to a learner
//! of their own, trained and measured on a shipped environment, using
//! published crates and public API only.
//!
//! `cargo xtask probe byoa-1a` builds this crate against the packaged
//! workspace and runs it. The steps a build can observe are `xtask`'s; this
//! binary reports the steps that need the code to run, one line each on
//! stderr, by label:
//!
//! ```text
//! PROBE-STEP <label> <PASS|FAIL> :: <detail>
//! ```
//!
//! Each step's code sits behind a cargo feature that includes the one before
//! it (`env`, then `learn`, then `harness`), so a build failure can be
//! reported at the step whose code broke. The features gate this crate's own
//! modules and never a dependency's feature.

// The lower levels compile a prefix of the walk, which leaves the items the
// later steps use without a caller.
#![cfg_attr(not(feature = "harness"), allow(dead_code))]

mod q_learner;

#[cfg(feature = "env")]
mod frozen_lake;

#[cfg(feature = "learn")]
mod learn;

#[cfg(feature = "harness")]
mod harness;

fn main() {
    #[cfg(feature = "learn")]
    learn::run(report);
    #[cfg(feature = "harness")]
    harness::run(report);
}

fn report(label: &str, outcome: Result<String, String>) {
    match outcome {
        Ok(detail) => eprintln!("PROBE-STEP {label} PASS :: {detail}"),
        Err(detail) => eprintln!("PROBE-STEP {label} FAIL :: {detail}"),
    }
}
