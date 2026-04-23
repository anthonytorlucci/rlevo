//! MountainCarContinuous-v0 environment.
//!
//! Continuous-action variant of MountainCar. Reward is shaped:
//! `-0.1 * action²` per step plus `+100` when the goal is reached.
//! No intrinsic step cap — compose with
//! [`crate::wrappers::TimeLimit::new(env, 999)`] for the standard limit.
use std::fmt;

use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Uniform};
use rlevo_core::{
    action::{BoundedAction, ContinuousAction},
    base::{Action, Observation, State, TensorConversionError, TensorConvertible},
    environment::{Environment, EnvironmentError, SnapshotBase},
    reward::ScalarReward,
};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Error type for invalid actions
// ---------------------------------------------------------------------------

/// Returned when constructing an action with an out-of-bounds value.
#[derive(Debug, Clone)]
pub struct InvalidActionError {
    pub message: String,
}

impl fmt::Display for InvalidActionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "InvalidAction: {}", self.message)
    }
}

impl std::error::Error for InvalidActionError {}

// ---------------------------------------------------------------------------
// Named reward-component keys (spec A4)
// ---------------------------------------------------------------------------

/// Named component key for the per-step control cost (`-0.1 * force²`).
pub const REWARD_CTRL: &str = "ctrl";
/// Named component key for the goal-reaching bonus (`+100`).
pub const REWARD_GOAL: &str = "goal";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for [`MountainCarContinuous`].
#[derive(Debug, Clone)]
pub struct MountainCarContinuousConfig {
    /// Minimum valid action value. Default: `-1.0`.
    pub min_action: f32,
    /// Maximum valid action value. Default: `1.0`.
    pub max_action: f32,
    /// Power multiplier applied to the clamped force. Default: `0.0015`.
    pub power: f32,
    /// Goal position (m). Default: `0.45`.
    pub goal_position: f32,
    /// Minimum velocity at goal for termination. Default: `0.0`.
    pub goal_velocity: f32,
    /// Left wall position (m). Default: `-1.2`.
    pub min_pos: f32,
    /// Right boundary (m). Default: `0.6`.
    pub max_pos: f32,
    /// Maximum absolute velocity (m/s). Default: `0.07`.
    pub max_speed: f32,
    /// RNG seed; `reset()` re-seeds from this value. Default: `0`.
    pub seed: u64,
}

impl Default for MountainCarContinuousConfig {
    fn default() -> Self {
        Self {
            min_action: -1.0,
            max_action: 1.0,
            power: 0.0015,
            goal_position: 0.45,
            goal_velocity: 0.0,
            min_pos: -1.2,
            max_pos: 0.6,
            max_speed: 0.07,
            seed: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Action (newtype with invariant — spec B2)
// ---------------------------------------------------------------------------

/// Continuous action for [`MountainCarContinuous`].
///
/// Construct via [`MountainCarContinuousAction::new`] to enforce the
/// `[-1, 1]` invariant at the boundary. The environment additionally
/// clamps the force before applying it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MountainCarContinuousAction(f32);

impl MountainCarContinuousAction {
    /// Construct, returning an error if `force` is not in `[-1, 1]` or is non-finite.
    pub fn new(force: f32) -> Result<Self, InvalidActionError> {
        if force.is_finite() && (-1.0..=1.0).contains(&force) {
            Ok(Self(force))
        } else {
            Err(InvalidActionError {
                message: format!("force {force} not in [-1.0, 1.0] or non-finite"),
            })
        }
    }

    /// The raw force value.
    pub fn force(&self) -> f32 {
        self.0
    }

    /// Unchecked construction for internal use (value already clamped).
    fn unchecked(force: f32) -> Self {
        Self(force)
    }
}

impl Action<1> for MountainCarContinuousAction {
    fn shape() -> [usize; 1] {
        [1]
    }

    fn is_valid(&self) -> bool {
        self.0.is_finite() && self.0.abs() <= 1.0
    }
}

impl ContinuousAction<1> for MountainCarContinuousAction {
    fn as_slice(&self) -> &[f32] {
        std::slice::from_ref(&self.0)
    }

    fn clip(&self, min: f32, max: f32) -> Self {
        Self::unchecked(self.0.clamp(min, max))
    }

    fn from_slice(values: &[f32]) -> Self {
        assert_eq!(
            values.len(),
            1,
            "MountainCarContinuousAction expects a 1-element slice"
        );
        Self::unchecked(values[0])
    }

    fn random() -> Self
    where
        Self: Sized,
    {
        Self::unchecked(0.0) // deterministic fallback; use env.step_with_rng for stochastic
    }
}

impl BoundedAction<1> for MountainCarContinuousAction {
    fn low() -> [f32; 1] {
        [-1.0]
    }

    fn high() -> [f32; 1] {
        [1.0]
    }
}

// ---------------------------------------------------------------------------
// State & observation (shared layout with MountainCar)
// ---------------------------------------------------------------------------

/// Internal state of [`MountainCarContinuous`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MountainCarContinuousState {
    position: f32,
    velocity: f32,
}

/// Observation returned by [`MountainCarContinuous`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MountainCarContinuousObservation {
    /// Horizontal position (m).
    pub position: f32,
    /// Velocity (m/s).
    pub velocity: f32,
}

impl MountainCarContinuousObservation {
    /// Flatten to a `[f32; 2]` array.
    pub fn to_array(&self) -> [f32; 2] {
        [self.position, self.velocity]
    }
}

impl Observation<1> for MountainCarContinuousObservation {
    fn shape() -> [usize; 1] {
        [2]
    }
}

impl State<1> for MountainCarContinuousState {
    type Observation = MountainCarContinuousObservation;

    fn shape() -> [usize; 1] {
        [2]
    }
    fn numel(&self) -> usize {
        2
    }
    fn is_valid(&self) -> bool {
        self.position.is_finite() && self.velocity.is_finite()
    }

    fn observe(&self) -> MountainCarContinuousObservation {
        MountainCarContinuousObservation {
            position: self.position,
            velocity: self.velocity,
        }
    }
}

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

/// MountainCarContinuous-v0: escape the valley with a continuous force.
#[derive(Debug)]
pub struct MountainCarContinuous {
    state: MountainCarContinuousState,
    config: MountainCarContinuousConfig,
    rng: StdRng,
    steps: usize,
}

impl MountainCarContinuous {
    /// Construct with an explicit config.
    pub fn with_config(config: MountainCarContinuousConfig) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        Self {
            state: MountainCarContinuousState {
                position: -0.5,
                velocity: 0.0,
            },
            config,
            rng,
            steps: 0,
        }
    }

    fn sample_init_state(&mut self) -> MountainCarContinuousState {
        let pos = Uniform::new_inclusive(-0.6_f32, -0.4_f32)
            .unwrap()
            .sample(&mut self.rng);
        MountainCarContinuousState {
            position: pos,
            velocity: 0.0,
        }
    }

    fn apply_physics(
        state: MountainCarContinuousState,
        force: f32,
        cfg: &MountainCarContinuousConfig,
    ) -> MountainCarContinuousState {
        let clamped = force.clamp(cfg.min_action, cfg.max_action);
        let mut vel = state.velocity + clamped * cfg.power - 0.0025 * (3.0 * state.position).cos();
        vel = vel.clamp(-cfg.max_speed, cfg.max_speed);
        let mut pos = state.position + vel;
        pos = pos.clamp(cfg.min_pos, cfg.max_pos);
        if pos <= cfg.min_pos {
            vel = 0.0;
        }
        MountainCarContinuousState {
            position: pos,
            velocity: vel,
        }
    }

    fn is_terminal(state: &MountainCarContinuousState, cfg: &MountainCarContinuousConfig) -> bool {
        state.position >= cfg.goal_position && state.velocity >= cfg.goal_velocity
    }
}

impl fmt::Display for MountainCarContinuous {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MountainCarContinuous(step={}, pos={:.3})",
            self.steps, self.state.position
        )
    }
}

impl Environment<1, 1, 1> for MountainCarContinuous {
    type StateType = MountainCarContinuousState;
    type ObservationType = MountainCarContinuousObservation;
    type ActionType = MountainCarContinuousAction;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, MountainCarContinuousObservation, ScalarReward>;

    fn new(render: bool) -> Self {
        let _ = render;
        Self::with_config(MountainCarContinuousConfig::default())
    }

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.rng = StdRng::seed_from_u64(self.config.seed);
        self.state = self.sample_init_state();
        self.steps = 0;
        Ok(SnapshotBase::running(
            self.state.observe(),
            ScalarReward(0.0),
        ))
    }

    fn step(
        &mut self,
        action: MountainCarContinuousAction,
    ) -> Result<Self::SnapshotType, EnvironmentError> {
        let force = action.force();
        self.state = Self::apply_physics(self.state, force, &self.config);
        self.steps += 1;

        let terminated = Self::is_terminal(&self.state, &self.config);
        let ctrl_cost = -0.1 * force * force;
        let goal_bonus = if terminated { 100.0 } else { 0.0 };
        let reward = ScalarReward(ctrl_cost + goal_bonus);

        let snap = if terminated {
            SnapshotBase::terminated(self.state.observe(), reward)
        } else {
            SnapshotBase::running(self.state.observe(), reward)
        };
        Ok(snap)
    }
}

// ---------------------------------------------------------------------------
// TensorConvertible
// ---------------------------------------------------------------------------

impl<B: burn::tensor::backend::Backend> TensorConvertible<1, B>
    for MountainCarContinuousObservation
{
    fn to_tensor(&self, device: &B::Device) -> burn::tensor::Tensor<B, 1> {
        burn::tensor::Tensor::from_floats(self.to_array(), device)
    }

    fn from_tensor(tensor: burn::tensor::Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let dims = tensor.shape().dims;
        if dims.as_slice() != [2] {
            return Err(TensorConversionError {
                message: format!("expected shape [2], got {dims:?}"),
            });
        }
        let v = tensor
            .into_data()
            .into_vec::<f32>()
            .map_err(|e| TensorConversionError {
                message: e.to_string(),
            })?;
        Ok(Self {
            position: v[0],
            velocity: v[1],
        })
    }
}

impl<B: burn::tensor::backend::Backend> TensorConvertible<1, B> for MountainCarContinuousAction {
    fn to_tensor(&self, device: &B::Device) -> burn::tensor::Tensor<B, 1> {
        burn::tensor::Tensor::from_floats([self.0], device)
    }

    fn from_tensor(tensor: burn::tensor::Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let dims = tensor.shape().dims;
        if dims.as_slice() != [1] {
            return Err(TensorConversionError {
                message: format!("expected shape [1], got {dims:?}"),
            });
        }
        let v = tensor
            .into_data()
            .into_vec::<f32>()
            .map_err(|e| TensorConversionError {
                message: e.to_string(),
            })?;
        Ok(Self::unchecked(v[0]))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rlevo_core::environment::Snapshot;

    fn default_env() -> MountainCarContinuous {
        MountainCarContinuous::with_config(MountainCarContinuousConfig::default())
    }

    #[test]
    fn observation_shape() {
        assert_eq!(MountainCarContinuousObservation::shape(), [2]);
    }

    #[test]
    fn action_validation_rejects_out_of_bounds() {
        assert!(MountainCarContinuousAction::new(1.5).is_err());
        assert!(MountainCarContinuousAction::new(-1.5).is_err());
        assert!(MountainCarContinuousAction::new(f32::NAN).is_err());
    }

    #[test]
    fn action_validation_accepts_boundary() {
        assert!(MountainCarContinuousAction::new(1.0).is_ok());
        assert!(MountainCarContinuousAction::new(-1.0).is_ok());
        assert!(MountainCarContinuousAction::new(0.0).is_ok());
    }

    #[test]
    fn zero_action_zero_ctrl_cost() {
        let mut env = default_env();
        env.reset().unwrap();
        let action = MountainCarContinuousAction::new(0.0).unwrap();
        let snap = env.step(action).unwrap();
        // ctrl cost = -0.1 * 0² = 0; no goal bonus
        if !snap.is_done() {
            assert!((snap.reward().0 - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn max_action_has_correct_ctrl_cost() {
        let mut env = default_env();
        env.reset().unwrap();
        let action = MountainCarContinuousAction::new(1.0).unwrap();
        let snap = env.step(action).unwrap();
        if !snap.is_done() {
            let expected = -0.1_f32;
            assert!(
                (snap.reward().0 - expected).abs() < 1e-5,
                "reward={}",
                snap.reward().0
            );
        }
    }

    #[test]
    fn termination_adds_goal_bonus() {
        let mut env = default_env();
        env.reset().unwrap();
        // Force into goal position
        env.state = MountainCarContinuousState {
            position: 0.49,
            velocity: 0.05,
        };
        let action = MountainCarContinuousAction::new(1.0).unwrap();
        let snap = env.step(action).unwrap();
        assert!(
            snap.is_terminated(),
            "expected terminated, got {:?}",
            snap.status()
        );
        // reward = -0.1 * 1² + 100 = 99.9
        assert!(
            snap.reward().0 > 90.0,
            "expected large positive reward, got {}",
            snap.reward().0
        );
    }

    #[test]
    fn determinism() {
        let mut a = MountainCarContinuous::with_config(MountainCarContinuousConfig {
            seed: 3,
            ..Default::default()
        });
        let mut b = MountainCarContinuous::with_config(MountainCarContinuousConfig {
            seed: 3,
            ..Default::default()
        });
        a.reset().unwrap();
        b.reset().unwrap();
        let act = MountainCarContinuousAction::new(0.5).unwrap();
        for _ in 0..5 {
            let sa = a.step(act).unwrap();
            let sb = b.step(act).unwrap();
            assert_eq!(sa.observation().to_array(), sb.observation().to_array());
        }
    }
}
