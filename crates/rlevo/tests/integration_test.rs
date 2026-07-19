//! End-to-end integration tests across `rlevo-core` and `rlevo-reinforcement-learning`.
//!
//! These tests exercise combinations of public items (environment from
//! `rlevo-core`; replay buffer and metrics from `rlevo-reinforcement-learning`) against a small
//! toy `RandomWalkEnv` defined inline. They can only see the public API —
//! unlike the in-crate `#[cfg(test)]` mocks — so they double as a smoke
//! test that the public surface is sufficient to build a working training
//! loop scaffold.

use burn::tensor::Tensor;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rlevo_core::action::DiscreteAction;
use rlevo_core::base::{
    Action, Observation, Reward, State, TensorConversionError, TensorConvertible,
};
use rlevo_core::environment::{
    Environment, EnvironmentError, EpisodeStatus, Sensor, Snapshot, SnapshotBase,
};
use rlevo_reinforcement_learning::metrics::{AgentStats, PerformanceRecord};
use rlevo_reinforcement_learning::replay::{
    DiscreteTransition, PrioritizedReplay, PrioritizedReplaySettings, ReplayStrategy,
};
use serde::{Deserialize, Serialize};
use std::ops::Add;

// ---------------------------------------------------------------------------
// Toy environment: 1-D random walk on positions [0, 6]. Start at 3. Reward +1
// for reaching position 6 (Terminated), -1 for falling below 0 (Terminated),
// 0 otherwise. Truncated after 20 steps.
// ---------------------------------------------------------------------------

/// Agent-visible observation for `RandomWalkEnv`: a single integer position on `[0, 6]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct WalkObservation {
    position: i32,
}

impl Observation<1> for WalkObservation {
    fn shape() -> [usize; 1] {
        [1]
    }
}

impl<B: burn::tensor::backend::Backend> TensorConvertible<1, B> for WalkObservation {
    fn row_shape() -> [usize; 1] {
        [1]
    }
    #[allow(clippy::cast_precision_loss)]
    fn write_host_row(&self, buf: &mut Vec<f32>) {
        buf.push(self.position as f32);
    }
    fn from_tensor(_t: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        // Not exercised by these tests; included only for trait completeness.
        Err(TensorConversionError {
            message: "from_tensor not implemented for WalkObservation".into(),
        })
    }
}

/// Full internal state for `RandomWalkEnv`: position on the 1-D lattice `[0, 6]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WalkState {
    position: i32,
}

impl State<1> for WalkState {
    fn shape() -> [usize; 1] {
        [1]
    }

    fn is_valid(&self) -> bool {
        (0..=6).contains(&self.position)
    }
}

/// Two-move action space for `RandomWalkEnv`: step left (index 0) or right (index 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WalkAction {
    Left,
    Right,
}

impl Action<1> for WalkAction {
    fn shape() -> [usize; 1] {
        [1]
    }
    fn is_valid(&self) -> bool {
        true
    }
}

impl DiscreteAction<1> for WalkAction {
    const ACTION_COUNT: usize = 2;

    fn from_index(index: usize) -> Self {
        match index {
            0 => WalkAction::Left,
            1 => WalkAction::Right,
            _ => panic!("invalid WalkAction index: {index}"),
        }
    }

    fn to_index(&self) -> usize {
        match self {
            WalkAction::Left => 0,
            WalkAction::Right => 1,
        }
    }
}

impl<B: burn::tensor::backend::Backend> TensorConvertible<1, B> for WalkAction {
    fn row_shape() -> [usize; 1] {
        [1]
    }
    #[allow(clippy::cast_precision_loss)]
    fn write_host_row(&self, buf: &mut Vec<f32>) {
        buf.push(self.to_index() as f32);
    }
    fn from_tensor(_t: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        Err(TensorConversionError {
            message: "from_tensor not implemented for WalkAction".into(),
        })
    }
}

/// Local scalar reward newtype. Mirrors `rlevo_core::reward::ScalarReward`
/// but is defined in this crate so we can implement the foreign
/// `TensorConvertible` trait on it without tripping the orphan rule.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct WalkReward(f32);

impl Reward for WalkReward {
    fn zero() -> Self {
        WalkReward(0.0)
    }
}

impl Add for WalkReward {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        WalkReward(self.0 + rhs.0)
    }
}

impl From<WalkReward> for f32 {
    fn from(r: WalkReward) -> f32 {
        r.0
    }
}

impl<B: burn::tensor::backend::Backend> TensorConvertible<1, B> for WalkReward {
    fn row_shape() -> [usize; 1] {
        [1]
    }
    fn write_host_row(&self, buf: &mut Vec<f32>) {
        buf.push(self.0);
    }
    fn from_tensor(_t: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        Err(TensorConversionError {
            message: "from_tensor not implemented for WalkReward".into(),
        })
    }
}

/// Toy 1-D random-walk environment used across all integration tests.
///
/// The agent starts at position 3 on a lattice `[0, 6]`. Reaching position 6
/// terminates the episode with reward `+1`; falling below 0 terminates with
/// reward `-1`. All other steps yield reward `0`. The episode is truncated
/// after `MAX_STEPS` steps regardless of outcome.
struct RandomWalkEnv {
    state: WalkState,
    steps: usize,
}

impl RandomWalkEnv {
    const START: i32 = 3;
    const GOAL: i32 = 6;
    const MAX_STEPS: usize = 20;

    /// Creates a new `RandomWalkEnv`. The `_render` flag is accepted for API
    /// symmetry with production environments but has no effect in tests.
    fn new(_render: bool) -> Self {
        Self {
            state: WalkState {
                position: Self::START,
            },
            steps: 0,
        }
    }
}

impl Sensor<1, 1, 1> for RandomWalkEnv {
    type Action = WalkAction;
    type State = WalkState;
    type Observation = WalkObservation;

    fn observe(&self, _action: &WalkAction, next_state: &WalkState) -> WalkObservation {
        WalkObservation {
            position: next_state.position,
        }
    }

    fn observe_reset(&self, state: &WalkState) -> WalkObservation {
        WalkObservation {
            position: state.position,
        }
    }
}

impl Environment<1, 1, 1> for RandomWalkEnv {
    type StateType = WalkState;
    type ObservationType = WalkObservation;
    type ActionType = WalkAction;
    type RewardType = WalkReward;
    type SnapshotType = SnapshotBase<1, WalkObservation, WalkReward>;

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.state = WalkState {
            position: Self::START,
        };
        self.steps = 0;
        Ok(SnapshotBase::running(
            self.observe_reset(&self.state),
            WalkReward::zero(),
        ))
    }

    fn step(&mut self, action: WalkAction) -> Result<Self::SnapshotType, EnvironmentError> {
        let delta = match action {
            WalkAction::Left => -1,
            WalkAction::Right => 1,
        };
        let next = self.state.position + delta;
        self.steps += 1;

        if next < 0 {
            self.state = WalkState { position: 0 };
            return Ok(SnapshotBase::terminated(
                self.observe(&action, &self.state),
                WalkReward(-1.0),
            ));
        }
        if next >= Self::GOAL {
            self.state = WalkState {
                position: Self::GOAL,
            };
            return Ok(SnapshotBase::terminated(
                self.observe(&action, &self.state),
                WalkReward(1.0),
            ));
        }

        self.state = WalkState { position: next };

        if self.steps >= Self::MAX_STEPS {
            Ok(SnapshotBase::truncated(
                self.observe(&action, &self.state),
                WalkReward::zero(),
            ))
        } else {
            Ok(SnapshotBase::running(
                self.observe(&action, &self.state),
                WalkReward::zero(),
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Verifies a random-walk episode loop terminates and accumulates a positive total reward.
#[test]
#[allow(clippy::float_cmp)]
fn full_episode_loop_reaches_goal() {
    let mut env = RandomWalkEnv::new(false);

    let initial = env.reset().expect("reset");
    assert_eq!(initial.status(), EpisodeStatus::Running);
    assert_eq!(f32::from(*initial.reward()), 0.0);
    assert_eq!(initial.observation().position, RandomWalkEnv::START);

    // Three right-moves take position from 3 to 6 (goal).
    let mut last = env.step(WalkAction::Right).expect("step 1");
    assert!(!last.is_done());
    last = env.step(WalkAction::Right).expect("step 2");
    assert!(!last.is_done());
    last = env.step(WalkAction::Right).expect("step 3 -> goal");

    assert!(last.is_done());
    assert!(last.is_terminated());
    assert!(!last.is_truncated());
    assert_eq!(f32::from(*last.reward()), 1.0);
    assert_eq!(last.observation().position, RandomWalkEnv::GOAL);
}

/// Exercises the opt-in prioritized-replay path (ADR 0050) end to end through
/// the public seam: build [`PrioritizedReplaySettings`], construct a
/// [`PrioritizedReplay`], fill it from the toy env, then draw a stratified
/// prioritized minibatch and confirm it carries one importance-sampling weight
/// per drawn id and that every drawn id resolves back to a stored transition.
///
/// This is the cross-crate smoke test that the public replay surface is
/// sufficient to scaffold a value-based training loop; it replaces the
/// pre-ADR-0050 `PrioritizedExperienceReplay::sample_batch` shape test, whose
/// type no longer exists.
#[test]
fn prioritized_replay_samples_a_weighted_minibatch() {
    let settings = PrioritizedReplaySettings::default();
    let mut buffer: PrioritizedReplay<DiscreteTransition<i32>> =
        PrioritizedReplay::new(settings.buffer_config(32)).expect("valid replay config");

    let mut env = RandomWalkEnv::new(false);
    let mut snapshot = env.reset().expect("reset");

    // Fill the buffer with 20 transitions, resetting whenever the episode ends.
    for i in 0..20 {
        let action = if i % 2 == 0 {
            WalkAction::Right
        } else {
            WalkAction::Left
        };
        let obs_before = snapshot.observation().position;
        let next = env.step(action).expect("step");
        buffer.push(DiscreteTransition {
            obs: obs_before,
            action: i % 2,
            reward: f32::from(*next.reward()),
            next_obs: next.observation().position,
            terminated: next.is_terminated(),
        });
        snapshot = if next.is_done() {
            env.reset().expect("reset after done")
        } else {
            next
        };
    }

    assert_eq!(
        buffer.len(),
        20,
        "twenty pushes under capacity 32: no eviction"
    );

    let mut rng = StdRng::seed_from_u64(42);
    // beta(0) is the schedule start; the caller owns the RNG (ADR 0029).
    let batch = buffer
        .sample(8, settings.beta(0), &mut rng)
        .expect("twenty transitions stored, eight requested");

    assert_eq!(batch.ids().len(), 8, "one id per requested draw");
    let weights = batch
        .weights()
        .expect("prioritized replay emits importance-sampling weights");
    assert_eq!(weights.len(), 8, "one importance weight per drawn id");
    assert!(
        weights
            .iter()
            .all(|w| w.is_finite() && *w > 0.0 && *w <= 1.0),
        "max-normalized IS weights lie in (0, 1]"
    );
    for &id in batch.ids() {
        assert!(
            buffer.get(id).is_some(),
            "a freshly drawn id always resolves to a stored transition"
        );
    }
}

/// Minimal `PerformanceRecord` implementation used to exercise `AgentStats` in isolation.
#[derive(Debug, Clone, Copy)]
struct EpisodeResult {
    score: f32,
    duration: usize,
}

impl PerformanceRecord for EpisodeResult {
    fn score(&self) -> f32 {
        self.score
    }
    fn duration(&self) -> usize {
        self.duration
    }
}

/// Verifies that `AgentStats` correctly tracks episode count and the sliding-window score average.
#[test]
fn agent_stats_tracks_episodes_and_sliding_window() {
    let mut stats = AgentStats::<EpisodeResult>::new(2);

    stats.record(EpisodeResult {
        score: 5.0,
        duration: 10,
    });
    stats.record(EpisodeResult {
        score: 8.0,
        duration: 15,
    });
    stats.record(EpisodeResult {
        score: 3.0,
        duration: 7,
    });

    assert_eq!(stats.total_episodes, 3);
    assert_eq!(stats.total_steps, 32);
    assert_eq!(stats.best_score, Some(8.0));

    // window_size = 2 means the first record is evicted; average is over the
    // last two episodes: (8.0 + 3.0) / 2 = 5.5.
    let avg = stats.avg_score().expect("avg_score present after records");
    assert!((avg - 5.5).abs() < 1e-6, "expected 5.5, got {avg}");
    assert_eq!(stats.recent_history.len(), 2);
}
