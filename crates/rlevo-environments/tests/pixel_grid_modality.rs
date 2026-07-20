// Exact comparison is intentional: the asserted rewards are literal
// constants produced without arithmetic.
#![allow(clippy::float_cmp)]

//! Single-crate proof that [`PixelGridEnv`] exposes a modality-changing
//! observation: the snapshot the agent receives is a rank-3 `[20, 20, 3]` RGB
//! image while the underlying state is rank-1 `[2]` (issue #65, ADR 0019/0020).
//!
//! This is the production counterpart to the `MockRam` mock in
//! `crates/rlevo/tests/observable_modality_change.rs`: a real environment in
//! `rlevo-environments` driving `R(3) != SR(1)` end-to-end through the public
//! [`Environment`] API, with every snapshot built from `Observable::project`.

use rlevo_core::action::DiscreteAction;
use rlevo_core::base::{Observation, State};
use rlevo_core::environment::{Environment, EpisodeStatus, Snapshot};

use rlevo_environments::pixel_grid::{
    PixelGridAction, PixelGridConfig, PixelGridEnv, PixelGridState, PixelObservation,
};

/// The observation is rank 3 while the state is rank 1 — the modality change is
/// visible in the very first snapshot the agent receives.
#[test]
fn reset_yields_rank3_observation_over_rank1_state() {
    let mut env = PixelGridEnv::with_config(PixelGridConfig::new(100, 0, false), false)
        .expect("valid config");
    let snap = env.reset().expect("reset succeeds");

    assert_eq!(snap.status(), EpisodeStatus::Running);
    assert_eq!(
        <PixelObservation as Observation<3>>::shape(),
        [20, 20, 3],
        "observation is rank 3"
    );
    assert_eq!(
        <PixelGridState as State<1>>::shape(),
        [2],
        "underlying state is rank 1"
    );
    assert_eq!(
        snap.observation().pixels().len(),
        20 * 20 * 3,
        "snapshot carries the full rendered image"
    );
}

/// A full `reset -> step* -> done` loop that reaches the goal terminates with a
/// positive reward — proving the env is solvable through `project()`-built
/// snapshots.
#[test]
fn optimal_rollout_reaches_goal_with_positive_reward() {
    // Fixed placement: agent at cell 0 (row 0, col 0), goal at cell 24 (row 4,
    // col 4). Optimal path is 4 Downs + 4 Rights = 8 steps.
    let mut env = PixelGridEnv::with_config(PixelGridConfig::new(100, 0, false), false)
        .expect("valid config");
    env.reset().expect("reset succeeds");

    let script = [
        PixelGridAction::Down,
        PixelGridAction::Down,
        PixelGridAction::Down,
        PixelGridAction::Down,
        PixelGridAction::Right,
        PixelGridAction::Right,
        PixelGridAction::Right,
        PixelGridAction::Right,
    ];

    let mut last = None;
    for action in script {
        last = Some(env.step(action).expect("step succeeds"));
    }
    let snap = last.expect("at least one step ran");

    assert!(snap.is_done(), "reaching the goal terminates the episode");
    assert_eq!(snap.status(), EpisodeStatus::Terminated);
    let reward: f32 = (*snap.reward()).into();
    assert!(reward > 0.0, "goal reward must be positive, got {reward}");
    // The terminal observation is still rank-3.
    assert_eq!(snap.observation().pixels().len(), 20 * 20 * 3);
}

/// Exhausting the step budget without reaching the goal truncates with reward
/// `0.0`.
#[test]
fn step_limit_truncates_with_zero_reward() {
    // Budget of 3 steps; bump the top-left wall (Up is a no-op) so the goal is
    // never reached.
    let mut env =
        PixelGridEnv::with_config(PixelGridConfig::new(3, 0, false), false).expect("valid config");
    env.reset().expect("reset succeeds");

    env.step(PixelGridAction::Up).unwrap();
    env.step(PixelGridAction::Up).unwrap();
    let snap = env.step(PixelGridAction::Up).unwrap();

    assert!(snap.is_done());
    assert_eq!(snap.status(), EpisodeStatus::Truncated);
    let reward: f32 = (*snap.reward()).into();
    assert_eq!(reward, 0.0);
}

/// `step` is total under every discrete action from any starting cell — a
/// random policy never errors.
#[test]
fn random_policy_never_errors() {
    let mut env =
        PixelGridEnv::with_config(PixelGridConfig::new(50, 7, true), false).expect("valid config");
    env.reset().expect("reset succeeds");
    for i in 0..50 {
        let action = PixelGridAction::from_index(i % PixelGridAction::ACTION_COUNT);
        let snap = env.step(action).expect("step succeeds");
        if snap.is_done() {
            break;
        }
    }
}
