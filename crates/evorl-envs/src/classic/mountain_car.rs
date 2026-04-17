//! MountainCar-v0 environment.
//!
//! A car must escape a valley by building momentum. Reward is `-1` each step;
//! the episode ends when the car reaches the goal position with sufficient
//! velocity. No intrinsic step limit — compose with
//! [`crate::wrappers::TimeLimit::new(env, 200)`] for the standard 200-step cap.
use std::fmt;

use evorl_core::{
    action::DiscreteAction,
    base::{Action, Observation, Reward, State},
    environment::{Environment, EnvironmentError, EpisodeStatus, SnapshotBase},
    reward::ScalarReward,
};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Uniform};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for [`MountainCar`].
#[derive(Debug, Clone)]
pub struct MountainCarConfig {
    /// Acceleration applied per step. Default: `0.001`.
    pub force: f32,
    /// Gravity pulling the car back (slope factor). Default: `0.0025`.
    pub gravity: f32,
    /// Left wall position (m). Default: `-1.2`.
    pub min_pos: f32,
    /// Right boundary (m). Default: `0.6`.
    pub max_pos: f32,
    /// Maximum absolute velocity (m/s). Default: `0.07`.
    pub max_speed: f32,
    /// X position considered the goal. Default: `0.5`.
    pub goal_position: f32,
    /// Minimum velocity at goal for termination. Default: `0.0`.
    pub goal_velocity: f32,
    /// RNG seed; `reset()` re-seeds from this value. Default: `0`.
    pub seed: u64,
}

impl Default for MountainCarConfig {
    fn default() -> Self {
        Self {
            force: 0.001,
            gravity: 0.0025,
            min_pos: -1.2,
            max_pos: 0.6,
            max_speed: 0.07,
            goal_position: 0.5,
            goal_velocity: 0.0,
            seed: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// Internal state of the MountainCar.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MountainCarState {
    /// Horizontal position (m).
    pub position: f32,
    /// Velocity (m/s).
    pub velocity: f32,
}

impl State<1> for MountainCarState {
    type Observation = MountainCarObservation;

    fn shape() -> [usize; 1] { [2] }
    fn numel(&self) -> usize { 2 }

    fn is_valid(&self) -> bool {
        self.position.is_finite() && self.velocity.is_finite()
    }

    fn observe(&self) -> MountainCarObservation {
        MountainCarObservation { position: self.position, velocity: self.velocity }
    }
}

// ---------------------------------------------------------------------------
// Observation
// ---------------------------------------------------------------------------

/// Observation returned by [`MountainCar`] at each step.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MountainCarObservation {
    /// Horizontal position (m).
    pub position: f32,
    /// Velocity (m/s).
    pub velocity: f32,
}

impl MountainCarObservation {
    /// Flatten to a `[f32; 2]` array for tensor conversion.
    pub fn to_array(&self) -> [f32; 2] {
        [self.position, self.velocity]
    }
}

impl Observation<1> for MountainCarObservation {
    fn shape() -> [usize; 1] { [2] }
}

// ---------------------------------------------------------------------------
// Action
// ---------------------------------------------------------------------------

/// Discrete action for [`MountainCar`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MountainCarAction {
    /// Accelerate left (index 0).
    Left,
    /// No acceleration (index 1).
    NoAccel,
    /// Accelerate right (index 2).
    Right,
}

impl Action<1> for MountainCarAction {
    fn shape() -> [usize; 1] { [3] }
    fn is_valid(&self) -> bool { true }
}

impl DiscreteAction<1> for MountainCarAction {
    const ACTION_COUNT: usize = 3;

    fn from_index(index: usize) -> Self {
        match index {
            0 => Self::Left,
            1 => Self::NoAccel,
            2 => Self::Right,
            _ => panic!("MountainCarAction index out of range: {index}"),
        }
    }

    fn to_index(&self) -> usize {
        match self {
            Self::Left    => 0,
            Self::NoAccel => 1,
            Self::Right   => 2,
        }
    }
}

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

/// MountainCar-v0: escape the valley by building momentum.
#[derive(Debug)]
pub struct MountainCar {
    state: MountainCarState,
    config: MountainCarConfig,
    rng: StdRng,
    steps: usize,
}

impl MountainCar {
    /// Construct with an explicit config.
    pub fn with_config(config: MountainCarConfig) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        Self {
            state: MountainCarState { position: -0.5, velocity: 0.0 },
            config,
            rng,
            steps: 0,
        }
    }

    /// Current step count within the episode.
    pub fn steps(&self) -> usize { self.steps }

    fn sample_init_state(&mut self) -> MountainCarState {
        let pos = Uniform::new_inclusive(-0.6_f32, -0.4_f32).unwrap().sample(&mut self.rng);
        MountainCarState { position: pos, velocity: 0.0 }
    }

    fn apply_physics(state: MountainCarState, action: MountainCarAction, cfg: &MountainCarConfig) -> MountainCarState {
        let action_val = action.to_index() as f32 - 1.0; // -1, 0, or +1
        let mut vel = state.velocity + action_val * cfg.force - (3.0 * state.position).cos() * cfg.gravity;
        vel = vel.clamp(-cfg.max_speed, cfg.max_speed);
        let mut pos = state.position + vel;
        pos = pos.clamp(cfg.min_pos, cfg.max_pos);
        // Inelastic left wall
        if pos <= cfg.min_pos {
            vel = 0.0;
        }
        MountainCarState { position: pos, velocity: vel }
    }

    fn is_terminal(state: &MountainCarState, cfg: &MountainCarConfig) -> bool {
        state.position >= cfg.goal_position && state.velocity >= cfg.goal_velocity
    }
}

impl fmt::Display for MountainCar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MountainCar(step={}, pos={:.3}, vel={:.4})", self.steps, self.state.position, self.state.velocity)
    }
}

impl Environment<1, 1, 1> for MountainCar {
    type StateType = MountainCarState;
    type ObservationType = MountainCarObservation;
    type ActionType = MountainCarAction;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, MountainCarObservation, ScalarReward>;

    fn new(render: bool) -> Self {
        let _ = render;
        Self::with_config(MountainCarConfig::default())
    }

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.rng = StdRng::seed_from_u64(self.config.seed);
        self.state = self.sample_init_state();
        self.steps = 0;
        Ok(SnapshotBase::running(self.state.observe(), ScalarReward(0.0)))
    }

    fn step(&mut self, action: MountainCarAction) -> Result<Self::SnapshotType, EnvironmentError> {
        self.state = Self::apply_physics(self.state, action, &self.config);
        self.steps += 1;

        let terminated = Self::is_terminal(&self.state, &self.config);
        let snap = if terminated {
            SnapshotBase::terminated(self.state.observe(), ScalarReward(-1.0))
        } else {
            SnapshotBase::running(self.state.observe(), ScalarReward(-1.0))
        };
        Ok(snap)
    }
}

// ---------------------------------------------------------------------------
// ASCII renderer
// ---------------------------------------------------------------------------

impl crate::render::AsciiRenderable for MountainCar {
    fn render_ascii(&self) -> String {
        let width = 60_usize;
        let span = self.config.max_pos - self.config.min_pos;
        let frac = ((self.state.position - self.config.min_pos) / span).clamp(0.0, 1.0);
        let col = (frac * (width as f32 - 1.0)) as usize;
        let mut track = vec!['.'; width];
        track[col] = 'A';
        let track_str: String = track.iter().collect();
        format!("[{track_str}]  pos={:.3}  vel={:.4}  step={}", self.state.position, self.state.velocity, self.steps)
    }
}

// ---------------------------------------------------------------------------
// TensorConvertible
// ---------------------------------------------------------------------------

impl<B: burn::tensor::backend::Backend> evorl_core::base::TensorConvertible<1, B>
    for MountainCarObservation
{
    fn to_tensor(&self, device: &B::Device) -> burn::tensor::Tensor<B, 1> {
        burn::tensor::Tensor::from_floats(self.to_array(), device)
    }

    fn from_tensor(
        tensor: burn::tensor::Tensor<B, 1>,
    ) -> Result<Self, evorl_core::base::TensorConversionError> {
        let dims = tensor.shape().dims;
        if dims.as_slice() != [2] {
            return Err(evorl_core::base::TensorConversionError {
                message: format!("expected shape [2], got {dims:?}"),
            });
        }
        let v = tensor
            .into_data()
            .into_vec::<f32>()
            .map_err(|e| evorl_core::base::TensorConversionError { message: e.to_string() })?;
        Ok(Self { position: v[0], velocity: v[1] })
    }
}

impl<B: burn::tensor::backend::Backend> evorl_core::base::TensorConvertible<1, B>
    for MountainCarAction
{
    fn to_tensor(&self, device: &B::Device) -> burn::tensor::Tensor<B, 1> {
        let mut one_hot = [0.0_f32; 3];
        one_hot[self.to_index()] = 1.0;
        burn::tensor::Tensor::from_floats(one_hot, device)
    }

    fn from_tensor(
        tensor: burn::tensor::Tensor<B, 1>,
    ) -> Result<Self, evorl_core::base::TensorConversionError> {
        let dims = tensor.shape().dims;
        if dims.as_slice() != [3] {
            return Err(evorl_core::base::TensorConversionError {
                message: format!("expected shape [3], got {dims:?}"),
            });
        }
        let v = tensor
            .into_data()
            .into_vec::<f32>()
            .map_err(|e| evorl_core::base::TensorConversionError { message: e.to_string() })?;
        let idx = v
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        Ok(Self::from_index(idx))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use evorl_core::environment::Snapshot;

    fn default_env() -> MountainCar {
        MountainCar::with_config(MountainCarConfig::default())
    }

    #[test]
    fn reset_initialises_correctly() {
        let mut env = default_env();
        let snap = env.reset().unwrap();
        assert_eq!(snap.status(), EpisodeStatus::Running);
        let obs = snap.observation();
        assert!(obs.position >= -0.6 && obs.position <= -0.4, "position {}", obs.position);
        assert_eq!(obs.velocity, 0.0);
    }

    #[test]
    fn observation_shape() {
        assert_eq!(MountainCarObservation::shape(), [2]);
    }

    #[test]
    fn action_count() {
        assert_eq!(MountainCarAction::ACTION_COUNT, 3);
        assert_eq!(MountainCarAction::from_index(0), MountainCarAction::Left);
        assert_eq!(MountainCarAction::from_index(2), MountainCarAction::Right);
    }

    #[test]
    fn left_wall_kills_velocity() {
        let cfg = MountainCarConfig::default();
        let state = MountainCarState { position: -1.19, velocity: -0.05 };
        let next = MountainCar::apply_physics(state, MountainCarAction::Left, &cfg);
        assert_eq!(next.position, cfg.min_pos);
        assert_eq!(next.velocity, 0.0);
    }

    #[test]
    fn goal_terminates() {
        let mut env = default_env();
        env.reset().unwrap();
        env.state = MountainCarState { position: 0.55, velocity: 0.01 };
        let snap = env.step(MountainCarAction::Right).unwrap();
        assert!(snap.is_terminated());
    }

    #[test]
    fn reward_is_minus_one_per_step() {
        let mut env = default_env();
        env.reset().unwrap();
        let snap = env.step(MountainCarAction::NoAccel).unwrap();
        assert_eq!(*snap.reward(), ScalarReward(-1.0));
    }

    #[test]
    fn determinism() {
        let mut a = MountainCar::with_config(MountainCarConfig { seed: 7, ..Default::default() });
        let mut b = MountainCar::with_config(MountainCarConfig { seed: 7, ..Default::default() });
        a.reset().unwrap();
        b.reset().unwrap();
        for action in [MountainCarAction::Right, MountainCarAction::Left, MountainCarAction::NoAccel] {
            let sa = a.step(action).unwrap();
            let sb = b.step(action).unwrap();
            assert_eq!(sa.observation().to_array(), sb.observation().to_array());
        }
    }
}
