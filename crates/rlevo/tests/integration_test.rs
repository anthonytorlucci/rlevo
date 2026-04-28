//! End-to-end integration tests across `rlevo-core` and `rlevo-rl`.
//!
//! These tests exercise combinations of public items (environment from
//! `rlevo-core`; replay buffer and metrics from `rlevo-rl`) against a small
//! toy `RandomWalkEnv` defined inline. They can only see the public API —
//! unlike the in-crate `#[cfg(test)]` mocks — so they double as a smoke
//! test that the public surface is sufficient to build a working training
//! loop scaffold.

use burn::backend::NdArray;
use burn::tensor::Tensor;
use rlevo_core::action::DiscreteAction;
use rlevo_core::base::{
    Action, Observation, Reward, State, TensorConversionError, TensorConvertible,
};
use rlevo_core::environment::{
    Environment, EnvironmentError, EpisodeStatus, Snapshot, SnapshotBase,
};
use rlevo_reinforcement_learning::memory::PrioritizedExperienceReplayBuilder;
use rlevo_reinforcement_learning::metrics::{AgentStats, PerformanceRecord};
use serde::{Deserialize, Serialize};
use std::ops::Add;

// ---------------------------------------------------------------------------
// Toy environment: 1-D random walk on positions [0, 6]. Start at 3. Reward +1
// for reaching position 6 (Terminated), -1 for falling below 0 (Terminated),
// 0 otherwise. Truncated after 20 steps.
// ---------------------------------------------------------------------------

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
    fn to_tensor(&self, device: &B::Device) -> Tensor<B, 1> {
        Tensor::from_floats([self.position as f32], device)
    }
    fn from_tensor(_t: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        // Not exercised by these tests; included only for trait completeness.
        Err(TensorConversionError {
            message: "from_tensor not implemented for WalkObservation".into(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WalkState {
    position: i32,
}

impl State<1> for WalkState {
    type Observation = WalkObservation;

    fn shape() -> [usize; 1] {
        [1]
    }

    fn is_valid(&self) -> bool {
        (0..=6).contains(&self.position)
    }

    fn observe(&self) -> Self::Observation {
        WalkObservation {
            position: self.position,
        }
    }
}

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
    fn to_tensor(&self, device: &B::Device) -> Tensor<B, 1> {
        Tensor::from_floats([self.to_index() as f32], device)
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
    fn to_tensor(&self, device: &B::Device) -> Tensor<B, 1> {
        Tensor::from_floats([self.0], device)
    }
    fn from_tensor(_t: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        Err(TensorConversionError {
            message: "from_tensor not implemented for WalkReward".into(),
        })
    }
}

struct RandomWalkEnv {
    state: WalkState,
    steps: usize,
}

impl RandomWalkEnv {
    const START: i32 = 3;
    const GOAL: i32 = 6;
    const MAX_STEPS: usize = 20;
}

impl Environment<1, 1, 1> for RandomWalkEnv {
    type StateType = WalkState;
    type ObservationType = WalkObservation;
    type ActionType = WalkAction;
    type RewardType = WalkReward;
    type SnapshotType = SnapshotBase<1, WalkObservation, WalkReward>;

    fn new(_render: bool) -> Self {
        Self {
            state: WalkState {
                position: Self::START,
            },
            steps: 0,
        }
    }

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.state = WalkState {
            position: Self::START,
        };
        self.steps = 0;
        Ok(SnapshotBase::running(
            self.state.observe(),
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
                self.state.observe(),
                WalkReward(-1.0),
            ));
        }
        if next >= Self::GOAL {
            self.state = WalkState {
                position: Self::GOAL,
            };
            return Ok(SnapshotBase::terminated(
                self.state.observe(),
                WalkReward(1.0),
            ));
        }

        self.state = WalkState { position: next };

        if self.steps >= Self::MAX_STEPS {
            Ok(SnapshotBase::truncated(
                self.state.observe(),
                WalkReward::zero(),
            ))
        } else {
            Ok(SnapshotBase::running(
                self.state.observe(),
                WalkReward::zero(),
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
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

#[test]
fn replay_buffer_sample_batch_tensor_shapes() {
    type B = NdArray;
    let device = Default::default();

    let mut buffer =
        PrioritizedExperienceReplayBuilder::<1, 1, WalkObservation, WalkAction, WalkReward>::new()
            .with_capacity(32)
            .with_alpha(0.6)
            .build();

    let mut env = RandomWalkEnv::new(false);
    let mut snapshot = env.reset().expect("reset");

    // Fill the buffer with 20 transitions, resetting whenever the episode ends.
    for i in 0..20 {
        let action = if i % 2 == 0 {
            WalkAction::Right
        } else {
            WalkAction::Left
        };
        let obs_before = *snapshot.observation();
        let next = env.step(action).expect("step");
        buffer.add(
            obs_before,
            action,
            *next.reward(),
            *next.observation(),
            next.is_done(),
            Some(1.0),
        );
        snapshot = if next.is_done() {
            env.reset().expect("reset after done")
        } else {
            next
        };
    }

    let batch = buffer
        .sample_batch::<2, 2, B>(8, &device)
        .expect("sample_batch");

    assert_eq!(batch.observations.shape().dims, [8, 1]);
    assert_eq!(batch.actions.shape().dims, [8, 1]);
    assert_eq!(batch.rewards.shape().dims, [8]);
    assert_eq!(batch.next_observations.shape().dims, [8, 1]);
    assert_eq!(batch.dones.shape().dims, [8]);
}

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
