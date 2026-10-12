//! BYOE-1, the thermostat: an environment author's walk from an empty
//! directory to a harness run of their own environment, using published
//! crates and public API only.
//!
//! `cargo xtask probe byoe-1-thermostat` builds this crate against the
//! packaged workspace and runs it. The steps a build can observe are
//! `xtask`'s; this binary reports the steps that need the code to run, one
//! line each on stderr, by label:
//!
//! ```text
//! PROBE-STEP <label> <PASS|FAIL> :: <detail>
//! ```
//!
//! Each step's code sits behind a cargo feature that includes the one before
//! it (`agent`, then `harness`), so a build failure can be reported at the
//! step whose code broke. The features gate this crate's own modules and
//! never a dependency's feature.

// The lower levels compile a prefix of the walk, which leaves the items the
// later steps use without a caller.
#![cfg_attr(not(feature = "harness"), allow(dead_code))]

mod thermostat;

#[cfg(feature = "agent")]
mod policy;

#[cfg(feature = "harness")]
mod harness;

use rlevo::core::environment::{Environment, EnvironmentError, EpisodeStatus, Snapshot};

use thermostat::{Boiler, HORIZON, Thermostat};

fn main() {
    report("post-terminal", post_terminal());
    #[cfg(feature = "harness")]
    report("run-suite", harness::run());
}

fn report(label: &str, outcome: Result<String, String>) {
    match outcome {
        Ok(detail) => eprintln!("PROBE-STEP {label} PASS :: {detail}"),
        Err(detail) => eprintln!("PROBE-STEP {label} FAIL :: {detail}"),
    }
}

/// A step after the episode ended must be the specific error the contract
/// names, carry the status that ended the episode, and change nothing.
fn post_terminal() -> Result<String, String> {
    let mut env = Thermostat::with_seed(2026);
    env.reset().map_err(|e| format!("reset failed: {e}"))?;

    // Bounded: a guard that never ends the episode fails here, it does not hang.
    let mut steps = 0;
    let ended = loop {
        if steps > HORIZON + 2 {
            return Err(format!("no done snapshot within {} steps", HORIZON + 2));
        }
        let snapshot = env
            .step(Boiler::On)
            .map_err(|e| format!("step {steps} failed: {e}"))?;
        steps += 1;
        if snapshot.is_done() {
            break snapshot.status();
        }
    };
    if steps != HORIZON || ended != EpisodeStatus::Terminated {
        return Err(format!(
            "the episode ended {ended:?} after {steps} steps; expected Terminated after {HORIZON}"
        ));
    }

    let before = env.state();
    for attempt in 1..=2 {
        match env.step(Boiler::On) {
            Err(EnvironmentError::StepAfterEpisodeEnd { status })
                if status == EpisodeStatus::Terminated => {}
            Err(EnvironmentError::StepAfterEpisodeEnd { status }) => {
                return Err(format!(
                    "the error carries {status:?}; the episode ended Terminated"
                ));
            }
            Err(other) => {
                return Err(format!(
                    "post-terminal step {attempt} returned `{other}`, not StepAfterEpisodeEnd"
                ));
            }
            Ok(_) => {
                return Err(format!("post-terminal step {attempt} was accepted"));
            }
        }
    }
    if env.state() != before {
        return Err("a rejected step changed the environment's state".to_string());
    }

    // `reset` must clear the guard, or the environment is single-use.
    env.reset()
        .map_err(|e| format!("second reset failed: {e}"))?;
    env.step(Boiler::Off)
        .map_err(|e| format!("the first step after a second reset failed: {e}"))?;

    Ok(
        "a step after the end returns StepAfterEpisodeEnd carrying Terminated and changes \
         nothing; guard from rlevo::core"
            .to_string(),
    )
}
