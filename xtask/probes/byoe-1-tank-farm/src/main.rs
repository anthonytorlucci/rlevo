//! BYOE-1, the tank farm: a second environment author's walk from an empty
//! directory to a harness run of their own environment, using published
//! crates and public API only.
//!
//! The thermostat's walk covers a fixed-horizon task with a discrete action.
//! This one covers what it does not: a rank-2 observation over a rank-1
//! state, a bounded continuous action that can be invalid, and a task with
//! two ways to end. An overflow terminates an episode; otherwise the shipped
//! `TimeLimit` wrapper truncates it.
//!
//! `cargo xtask probe byoe-1-tank-farm` builds this crate against the
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

mod tank_farm;

#[cfg(feature = "agent")]
mod policy;

#[cfg(feature = "harness")]
mod harness;

use rlevo::core::base::{Action, Observation};
use rlevo::core::environment::{Environment, EnvironmentError, EpisodeStatus, Snapshot};
use rlevo::envs::wrappers::TimeLimit;

use tank_farm::{HISTORY, PumpCommand, TankFarm, TankObs, TankState};

/// The step cap `TimeLimit` puts on an episode.
const TIME_LIMIT: usize = 60;

/// The seed every check here builds its farm from.
const SEED: u64 = 2026;

// The ranks are written out and checked when the base level compiles: a
// rank-2 observation and a rank-1 action, through the wrapper the harness
// runs. Flattening the observation to rank 1 to make
// something compile breaks this line, at step 3, and not a later step.
const fn has_ranks<E: Environment<2, 1>>() {}
const _: () = has_ranks::<TimeLimit<TankFarm>>();

fn main() {
    report("post-terminal", contract());
    #[cfg(feature = "harness")]
    report("run-suite", harness::run());
}

fn report(label: &str, outcome: Result<String, String>) {
    match outcome {
        Ok(detail) => eprintln!("PROBE-STEP {label} PASS :: {detail}"),
        Err(detail) => eprintln!("PROBE-STEP {label} FAIL :: {detail}"),
    }
}

/// Step 4: the shapes are the declared ones; an invalid command and a step
/// after either ending are the errors the contract names, and change
/// nothing.
fn contract() -> Result<String, String> {
    let shapes = (TankObs::shape(), PumpCommand::shape());
    if shapes != ([2, HISTORY], [2]) {
        return Err(format!(
            "observation and action shapes are {shapes:?}; expected ([2, {HISTORY}], [2])"
        ));
    }

    invalid_commands_change_nothing()?;

    // Both pumps full on overflows a tank. The farm's own guard holds that
    // ending, so it is checked on the bare farm.
    let full = PumpCommand { rates: [1.0, 1.0] };
    let mut farm = TankFarm::with_seed(SEED);
    let (overflow_at, ended) = run_to_end(&mut farm, full, TIME_LIMIT)?;
    if ended != EpisodeStatus::Terminated {
        return Err(format!(
            "the overflow ended the episode {ended:?}, not Terminated"
        ));
    }
    rejects_after_end(&mut farm, full, ended, TankFarm::state)?;

    // Both pumps off overflows nothing, so only the wrapper ends the episode.
    // A truncation is the wrapper's status, so its guard holds that ending.
    let off = PumpCommand { rates: [0.0, 0.0] };
    let mut limited = TimeLimit::new(TankFarm::with_seed(SEED), TIME_LIMIT);
    let (truncated_at, ended) = run_to_end(&mut limited, off, TIME_LIMIT + 2)?;
    if truncated_at != TIME_LIMIT || ended != EpisodeStatus::Truncated {
        return Err(format!(
            "the time limit ended the episode {ended:?} after {truncated_at} steps; \
             expected Truncated after {TIME_LIMIT}"
        ));
    }
    rejects_after_end(&mut limited, off, ended, |env| env.inner().state())?;

    Ok(format!(
        "an invalid command returns InvalidAction and changes nothing; a step after the end \
         returns StepAfterEpisodeEnd carrying Terminated after an overflow (step {overflow_at}) \
         and Truncated after the time limit (step {truncated_at}); guard from rlevo::core"
    ))
}

/// Out-of-range and non-finite rates are refused before anything moves.
///
/// The state alone would miss the level history the observation carries, so
/// the next valid step is compared with the same step on a fresh farm.
fn invalid_commands_change_nothing() -> Result<(), String> {
    let mut farm = TankFarm::with_seed(SEED);
    farm.reset().map_err(|e| format!("reset failed: {e}"))?;
    for rates in [[1.5, 0.0], [f32::NAN, 0.0]] {
        match farm.step(PumpCommand { rates }) {
            Err(EnvironmentError::InvalidAction(_)) => {}
            Err(other) => {
                return Err(format!(
                    "the command {rates:?} returned `{other}`, not InvalidAction"
                ));
            }
            Ok(_) => return Err(format!("the command {rates:?} was accepted")),
        }
    }

    let valid = PumpCommand { rates: [0.3, 0.3] };
    let after = farm
        .step(valid)
        .map_err(|e| format!("the step after a rejected command failed: {e}"))?;
    let mut fresh = TankFarm::with_seed(SEED);
    fresh.reset().map_err(|e| format!("reset failed: {e}"))?;
    let expected = fresh
        .step(valid)
        .map_err(|e| format!("the first step on a fresh farm failed: {e}"))?;
    if after.observation != expected.observation
        || after.reward != expected.reward
        || after.status != expected.status
    {
        return Err(
            "a rejected command changed the farm: the next step differs from the same step \
             on a fresh farm"
                .to_string(),
        );
    }
    Ok(())
}

/// Reset `env` and send `command` until the episode ends, at most `bound`
/// steps. Returns the step it ended on and how.
fn run_to_end<E>(
    env: &mut E,
    command: PumpCommand,
    bound: usize,
) -> Result<(usize, EpisodeStatus), String>
where
    E: Environment<2, 1, ActionType = PumpCommand>,
{
    env.reset().map_err(|e| format!("reset failed: {e}"))?;
    // Bounded: an episode that never ends fails here, it does not hang.
    for steps in 1..=bound {
        let snapshot = env
            .step(command)
            .map_err(|e| format!("step {steps} failed: {e}"))?;
        if snapshot.status.is_done() {
            return Ok((steps, snapshot.status));
        }
    }
    Err(format!(
        "{command:?} did not end the episode within {bound} steps"
    ))
}

/// A step after the end must be the specific error the contract names, carry
/// the status that ended the episode, and change nothing; a `reset` must
/// make the environment usable again.
fn rejects_after_end<E>(
    env: &mut E,
    command: PumpCommand,
    ended: EpisodeStatus,
    state: impl Fn(&E) -> TankState,
) -> Result<(), String>
where
    E: Environment<2, 1, ActionType = PumpCommand>,
{
    let before = state(env);
    match env.step(command) {
        Err(EnvironmentError::StepAfterEpisodeEnd { status }) if status == ended => {}
        Err(EnvironmentError::StepAfterEpisodeEnd { status }) => {
            return Err(format!(
                "the error carries {status:?}; the episode ended {ended:?}"
            ));
        }
        Err(other) => {
            return Err(format!(
                "a step after the end returned `{other}`, not StepAfterEpisodeEnd"
            ));
        }
        Ok(_) => {
            return Err(format!(
                "a step after an episode ended {ended:?} was accepted"
            ));
        }
    }
    if state(env) != before {
        return Err("a rejected step changed the environment's state".to_string());
    }

    // `reset` must clear the guard, or the environment is single-use.
    env.reset()
        .map_err(|e| format!("reset after the end failed: {e}"))?;
    env.step(command)
        .map_err(|e| format!("the first step after a reset failed: {e}"))?;
    Ok(())
}
