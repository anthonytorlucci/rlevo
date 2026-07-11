//! Composition canary: a **guarded** environment inside the **guarded**
//! [`TimeLimit`] wrapper.
//!
//! The unit tests in `wrappers/time_limit.rs` deliberately wrap an *unguarded*
//! stub, which isolates the wrapper's own [`EpisodeGuard`] and proves it works
//! alone. This file covers the case those tests cannot: the double guard, where
//! the inner environment carries a guard of its own (issue #105 / ADR 0044).
//!
//! [`CliffWalking`] is the inner env because it is fully deterministic in its
//! default (non-slippery) config, so a fixed action script is an exact oracle:
//! from the start `(3, 0)`, `Up` then eleven `Right`s then `Down` lands on the
//! goal `(3, 11)` and terminates. No seed-dependence, no flake.
//!
//! Two terminal paths, two very different stories:
//!
//! - **Truncation** — the wrapper synthesises `Truncated` *after* the inner env
//!   has already stepped, so `CliffWalking`'s own guard still reads `Running`
//!   and provably cannot reject the next call. Only the wrapper's guard can.
//! - **Termination** — *both* guards are shut, and both would produce the same
//!   `StepAfterEpisodeEnd { status: Terminated }`. That the errors agree is the
//!   point: the wrapper must not mask, re-wrap, or mislabel the inner status.
//!   `test_time_limit_over_cliff_walking_wrapper_guard_rejects_with_inner_guard_open`
//!   separates the two by re-opening *only* the inner guard through
//!   [`TimeLimit::inner_mut`] and showing the wrapper still rejects.
//!
//! Placement (rules.md §5): this exercises two modules of one crate —
//! `wrappers::time_limit` and `toy_text::cliff_walking` — entirely through
//! `rlevo-environments`'s public surface, and needs no private access. That is a
//! single-crate integration test, so it lives here rather than in either
//! module's in-source `#[cfg(test)]` block.

use rlevo_core::environment::{Environment, EnvironmentError, EpisodeStatus, Snapshot};
use rlevo_environments::toy_text::cliff_walking::{
    CliffWalking, CliffWalkingAction, CliffWalkingConfig,
};
use rlevo_environments::wrappers::TimeLimit;

/// Deterministic optimal-ish route from the start `(3, 0)` to the goal `(3, 11)`:
/// step up out of the cliff row, run the length of row 2, then drop onto the goal.
/// Thirteen steps, every one of them legal and cliff-free.
const PATH_TO_GOAL: [CliffWalkingAction; 13] = [
    CliffWalkingAction::Up,
    CliffWalkingAction::Right,
    CliffWalkingAction::Right,
    CliffWalkingAction::Right,
    CliffWalkingAction::Right,
    CliffWalkingAction::Right,
    CliffWalkingAction::Right,
    CliffWalkingAction::Right,
    CliffWalkingAction::Right,
    CliffWalkingAction::Right,
    CliffWalkingAction::Right,
    CliffWalkingAction::Right,
    CliffWalkingAction::Down,
];

/// Builds a `TimeLimit` over a deterministic (non-slippery) `CliffWalking`.
fn wrapped(max_steps: usize) -> TimeLimit<CliffWalking> {
    let inner = CliffWalking::with_config(CliffWalkingConfig::default())
        .expect("the default CliffWalking config must validate");
    TimeLimit::new(inner, max_steps)
}

/// Drives `env` along `PATH_TO_GOAL` until a done snapshot appears, and returns
/// the status that ended the episode.
///
/// # Panics
///
/// Panics if the script runs out before the episode ends, or if any step inside
/// the episode is rejected.
fn drive_until_done(env: &mut TimeLimit<CliffWalking>) -> EpisodeStatus {
    for action in PATH_TO_GOAL {
        let snapshot = env
            .step(action)
            .expect("every step inside the episode must succeed");
        if snapshot.is_done() {
            return snapshot.status();
        }
    }
    panic!(
        "the scripted path must end the episode within {} steps",
        PATH_TO_GOAL.len()
    );
}

#[test]
fn test_time_limit_over_cliff_walking_rejects_step_after_truncation() {
    // A cap of 3 fires long before the 13-step route reaches the goal, so the
    // only way this episode can end is by the wrapper's own truncation.
    let mut env = wrapped(3);
    env.reset().expect("reset must succeed");

    let ended = drive_until_done(&mut env);
    assert_eq!(
        ended,
        EpisodeStatus::Truncated,
        "with a cap of 3 the step budget, not the goal tile, must end this episode"
    );

    let err = env
        .step(CliffWalkingAction::Up)
        .expect_err("a step after truncation must be rejected");
    assert!(
        matches!(
            err,
            EnvironmentError::StepAfterEpisodeEnd {
                status: EpisodeStatus::Truncated
            }
        ),
        "only the wrapper's guard can catch this: the inner env was never told it \
         was truncated and still reads Running. Expected StepAfterEpisodeEnd \
         {{ status: Truncated }}, got {err:?}"
    );
}

#[test]
fn test_time_limit_over_cliff_walking_rejects_step_after_termination() {
    // A cap of 50 never fires: the inner env's goal tile ends the episode, and
    // both guards are shut by the time the next step arrives.
    let mut env = wrapped(50);
    env.reset().expect("reset must succeed");

    let ended = drive_until_done(&mut env);
    assert_eq!(
        ended,
        EpisodeStatus::Terminated,
        "with a cap of 50 the goal tile, not the step budget, must end this episode"
    );

    let err = env
        .step(CliffWalkingAction::Left)
        .expect_err("a step after the inner env terminated must be rejected");
    assert!(
        matches!(
            err,
            EnvironmentError::StepAfterEpisodeEnd {
                status: EpisodeStatus::Terminated
            }
        ),
        "the composed pair must report the inner env's Terminated, not the \
         wrapper's Truncated — the wrapper must neither mask nor mislabel it; got {err:?}"
    );
}

#[test]
fn test_time_limit_over_cliff_walking_wrapper_guard_rejects_with_inner_guard_open() {
    // Post-termination both guards are shut and produce identical errors, so the
    // previous test cannot say *which* one fired. Re-open only the inner guard —
    // via a direct reset through `inner_mut`, behind the wrapper's back — and the
    // rejection can now come from exactly one place.
    let mut env = wrapped(50);
    env.reset().expect("reset must succeed");
    assert_eq!(
        drive_until_done(&mut env),
        EpisodeStatus::Terminated,
        "the inner env must terminate on the goal tile"
    );

    env.inner_mut()
        .reset()
        .expect("resetting the inner env directly must succeed");

    let err = env
        .step(CliffWalkingAction::Up)
        .expect_err("the wrapper's guard must reject even with the inner guard re-opened");
    assert!(
        matches!(
            err,
            EnvironmentError::StepAfterEpisodeEnd {
                status: EpisodeStatus::Terminated
            }
        ),
        "the inner guard is Running again, so this rejection can only be the \
         wrapper's own guard — proving it is checked before delegating; got {err:?}"
    );
}

#[test]
fn test_time_limit_over_cliff_walking_reset_reopens_after_truncation() {
    let mut env = wrapped(3);
    env.reset().expect("reset must succeed");
    assert_eq!(
        drive_until_done(&mut env),
        EpisodeStatus::Truncated,
        "with a cap of 3 the episode must end by truncation"
    );
    assert!(
        env.step(CliffWalkingAction::Up).is_err(),
        "the episode is truncated; a further step must be rejected"
    );

    env.reset().expect("reset must succeed");
    assert_eq!(
        env.steps(),
        0,
        "reset() must clear the wrapper's step counter"
    );

    let snapshot = env
        .step(CliffWalkingAction::Up)
        .expect("reset() must re-open the composed pair for a new episode");
    assert_eq!(
        snapshot.status(),
        EpisodeStatus::Running,
        "the fresh episode gets the full budget again, so its first step is Running"
    );
}

#[test]
fn test_time_limit_over_cliff_walking_reset_reopens_after_termination() {
    let mut env = wrapped(50);
    env.reset().expect("reset must succeed");
    assert_eq!(
        drive_until_done(&mut env),
        EpisodeStatus::Terminated,
        "the inner env must terminate on the goal tile"
    );
    assert!(
        env.step(CliffWalkingAction::Left).is_err(),
        "the episode has terminated; a further step must be rejected"
    );

    // One reset must re-open *both* guards — the wrapper's and, by delegation,
    // the inner env's — and put the agent back on the start tile.
    env.reset().expect("reset must succeed");
    assert_eq!(
        drive_until_done(&mut env),
        EpisodeStatus::Terminated,
        "the second episode must run the whole route to the goal again, which is \
         only possible if reset() re-opened both guards and restored the start tile"
    );
}
