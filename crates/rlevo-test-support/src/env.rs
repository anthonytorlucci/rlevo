//! Synthetic 1-D continuous tracking environment shared across the
//! continuous-control algorithm tests (DDPG / TD3 / SAC).
//!
//! Each step emits an observation `x ∈ [-1, 1]`. The optimal action is `a = x`
//! and the reward is `-(a - x)²`, peaking at `0`. Episodes last a fixed number
//! of steps (typically 20). A uniform-random policy over `U(-1, 1)` averages
//! `≈ -episode_len · 1/3 ≈ -6.67` per 20-step episode, so a learned policy that
//! clears a lax `-1.0` threshold demonstrates real convergence.
//!
//! The fixture intentionally avoids any physics simulator so `cargo test` stays
//! tractable: tiny networks and modest step budgets converge in seconds.

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use rlevo_core::action::{BoundedAction, ContinuousAction};
use rlevo_core::base::{Action, Observation, State, TensorConversionError, TensorConvertible};
use rlevo_core::environment::{Environment, EnvironmentError, EpisodeStatus, Sensor, SnapshotBase};
use rlevo_core::reward::ScalarReward;
use rlevo_environments::classic::cartpole::{CartPole, CartPoleConfig};

/// Builds a `CartPole` environment seeded with `seed`, leaving every other
/// config field at its default.
///
/// This is the canonical discrete-control fixture, shared by the value-based
/// algorithm tests (DQN / C51 / QR-DQN) the way [`LinearEnv`] is shared by the
/// continuous-control tests. Callers still wrap it in a [`TimeLimit`] or pass
/// it straight to `train` as their test requires.
///
/// [`TimeLimit`]: rlevo_environments::wrappers::TimeLimit
///
/// # Panics
///
/// Panics if the seeded config fails validation, which cannot happen for the
/// default field values.
#[must_use]
pub fn cartpole_seeded(seed: u64) -> CartPole {
    CartPole::with_config(CartPoleConfig {
        seed,
        ..CartPoleConfig::default()
    })
    .expect("valid config")
}

/// Full state of [`LinearEnv`]: the current target `x` and the step counter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearState {
    /// Current target value, sampled from `U(-1, 1)` each step.
    pub x: f32,
    /// Number of steps elapsed in the current episode.
    pub steps: usize,
}

impl State<1> for LinearState {
    fn shape() -> [usize; 1] {
        [1]
    }
    fn numel(&self) -> usize {
        1
    }
    fn is_valid(&self) -> bool {
        self.x.is_finite()
    }
}

/// Agent's view of [`LinearState`]: just the scalar target `x`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LinearObservation {
    /// The target value the actor must reproduce.
    pub x: f32,
}

impl Observation<1> for LinearObservation {
    fn shape() -> [usize; 1] {
        [1]
    }
}

impl<B: Backend> TensorConvertible<1, B> for LinearObservation {
    fn row_shape() -> [usize; 1] {
        [1]
    }
    fn write_host_row(&self, buf: &mut Vec<f32>) {
        buf.push(self.x);
    }
    fn from_tensor(tensor: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let v = tensor.into_data().convert::<f32>();
        Ok(Self {
            x: v.as_slice::<f32>().unwrap()[0],
        })
    }
}

/// A single continuous action in `[-1, 1]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LinearAction(pub f32);

impl Action<1> for LinearAction {
    fn shape() -> [usize; 1] {
        [1]
    }
    fn is_valid(&self) -> bool {
        self.0.is_finite() && self.0.abs() <= 1.0
    }
}

impl ContinuousAction<1> for LinearAction {
    const COMPONENTS: usize = 1;

    fn as_slice(&self) -> &[f32] {
        std::slice::from_ref(&self.0)
    }
    fn clip(&self, min: f32, max: f32) -> Self {
        Self(self.0.clamp(min, max))
    }
    fn from_slice(values: &[f32]) -> Self {
        assert_eq!(values.len(), 1);
        Self(values[0])
    }
}

impl BoundedAction<1> for LinearAction {
    fn low() -> [f32; 1] {
        [-1.0]
    }
    fn high() -> [f32; 1] {
        [1.0]
    }
}

/// The synthetic 1-D continuous tracking environment.
///
/// Construct with [`LinearEnv::with_seed`], then drive it through the standard
/// [`Environment`] interface (`reset` / `step`). The internal RNG is seeded so
/// that two runs with the same seed produce identical observation streams.
#[derive(Debug)]
pub struct LinearEnv {
    state: LinearState,
    rng: StdRng,
    episode_len: usize,
}

impl LinearEnv {
    /// Builds a fresh environment seeded with `seed`, truncating episodes after
    /// `episode_len` steps.
    #[must_use]
    pub fn with_seed(seed: u64, episode_len: usize) -> Self {
        Self {
            state: LinearState { x: 0.0, steps: 0 },
            rng: StdRng::seed_from_u64(seed),
            episode_len,
        }
    }

    fn sample_x(rng: &mut StdRng) -> f32 {
        use rand::RngExt;
        rng.random_range(-1.0_f32..=1.0_f32)
    }
}

impl Sensor<1, 1, 1> for LinearEnv {
    type Action = LinearAction;
    type State = LinearState;
    type Observation = LinearObservation;

    fn observe(&self, _action: &LinearAction, next_state: &LinearState) -> LinearObservation {
        LinearObservation { x: next_state.x }
    }

    fn observe_reset(&self, state: &LinearState) -> LinearObservation {
        LinearObservation { x: state.x }
    }
}

impl Environment<1, 1, 1> for LinearEnv {
    type StateType = LinearState;
    type ObservationType = LinearObservation;
    type ActionType = LinearAction;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, LinearObservation, ScalarReward>;

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.state = LinearState {
            x: Self::sample_x(&mut self.rng),
            steps: 0,
        };
        Ok(SnapshotBase {
            observation: self.observe_reset(&self.state),
            reward: ScalarReward::new(0.0),
            status: EpisodeStatus::Running,
            metadata: None,
        })
    }

    fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> {
        let a = action.0.clamp(-1.0, 1.0);
        let err = a - self.state.x;
        let reward = -(err * err);
        let next_x = Self::sample_x(&mut self.rng);
        self.state = LinearState {
            x: next_x,
            steps: self.state.steps + 1,
        };
        let status = if self.state.steps >= self.episode_len {
            EpisodeStatus::Truncated
        } else {
            EpisodeStatus::Running
        };
        Ok(SnapshotBase {
            observation: self.observe(&action, &self.state),
            reward: ScalarReward::new(reward),
            status,
            metadata: None,
        })
    }
}
