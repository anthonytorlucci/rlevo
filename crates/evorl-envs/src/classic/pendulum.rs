//! Pendulum-v1 environment.
//!
//! Swing an under-actuated pendulum to the upright position and keep it
//! there. The episode never terminates intrinsically — use
//! [`crate::wrappers::TimeLimit::new(env, 200)`] for the standard limit.
//!
//! Reward is `-cost` where `cost = angle_normalize(θ)² + 0.1θ̇² + 0.001u²`.
//! The minimum per-step reward is ≈ `-16.27`; the maximum is `0`.
use std::fmt;

use evorl_core::{
    action::ContinuousAction,
    base::{Action, Observation, Reward, State},
    environment::{Environment, EnvironmentError, EpisodeStatus, SnapshotBase},
    reward::ScalarReward,
};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Uniform};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Returned when constructing a [`PendulumAction`] with an invalid torque.
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
// Config
// ---------------------------------------------------------------------------

/// Configuration for [`Pendulum`].
#[derive(Debug, Clone)]
pub struct PendulumConfig {
    /// Maximum angular velocity (rad/s). Default: `8.0`.
    pub max_speed: f32,
    /// Maximum torque magnitude (N·m). Default: `2.0`.
    pub max_torque: f32,
    /// Integration time step (s). Default: `0.05`.
    pub dt: f32,
    /// Gravitational acceleration (m/s²). Default: `10.0` (Gymnasium default).
    pub g: f32,
    /// Pendulum mass (kg). Default: `1.0`.
    pub m: f32,
    /// Pendulum length (m). Default: `1.0`.
    pub l: f32,
    /// RNG seed; `reset()` re-seeds from this value. Default: `0`.
    pub seed: u64,
}

impl Default for PendulumConfig {
    fn default() -> Self {
        Self {
            max_speed: 8.0,
            max_torque: 2.0,
            dt: 0.05,
            g: 10.0,
            m: 1.0,
            l: 1.0,
            seed: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Action (newtype with invariant — spec B2)
// ---------------------------------------------------------------------------

/// Continuous torque action for [`Pendulum`].
///
/// Construct via [`PendulumAction::new`] to enforce the `[-2, 2]` invariant.
/// The environment additionally clips the torque to `[-max_torque, max_torque]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PendulumAction(f32);

impl PendulumAction {
    /// Construct, returning an error if `torque` is outside `[-2, 2]` or non-finite.
    ///
    /// The bound `2.0` matches the default `max_torque`; environments with a
    /// non-default `max_torque` will further clip the torque in `step`.
    pub fn new(torque: f32) -> Result<Self, InvalidActionError> {
        if torque.is_finite() && torque.abs() <= 2.0 {
            Ok(Self(torque))
        } else {
            Err(InvalidActionError {
                message: format!("torque {torque} outside [-2.0, 2.0] or non-finite"),
            })
        }
    }

    /// The raw torque value.
    pub fn torque(&self) -> f32 { self.0 }

    fn unchecked(v: f32) -> Self { Self(v) }
}

impl Action<1> for PendulumAction {
    fn shape() -> [usize; 1] { [1] }

    fn is_valid(&self) -> bool {
        self.0.is_finite() && self.0.abs() <= 2.0
    }
}

impl ContinuousAction<1> for PendulumAction {
    fn as_slice(&self) -> &[f32] { std::slice::from_ref(&self.0) }

    fn clip(&self, min: f32, max: f32) -> Self {
        Self::unchecked(self.0.clamp(min, max))
    }

    fn from_slice(values: &[f32]) -> Self {
        assert_eq!(values.len(), 1, "PendulumAction expects a 1-element slice");
        Self::unchecked(values[0])
    }

    fn random() -> Self where Self: Sized {
        Self::unchecked(0.0)
    }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// Internal state of the [`Pendulum`] (angles in radians).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PendulumState {
    /// Angle (rad); 0 = upright, ±π = hanging down.
    pub theta: f32,
    /// Angular velocity (rad/s).
    pub theta_dot: f32,
}

impl State<1> for PendulumState {
    type Observation = PendulumObservation;

    fn shape() -> [usize; 1] { [2] }
    fn numel(&self) -> usize { 2 }
    fn is_valid(&self) -> bool { self.theta.is_finite() && self.theta_dot.is_finite() }

    fn observe(&self) -> PendulumObservation {
        PendulumObservation {
            cos_theta: self.theta.cos(),
            sin_theta: self.theta.sin(),
            theta_dot: self.theta_dot,
        }
    }
}

// ---------------------------------------------------------------------------
// Observation
// ---------------------------------------------------------------------------

/// Observation returned by [`Pendulum`] at each step.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PendulumObservation {
    /// cosine of the pole angle.
    pub cos_theta: f32,
    /// sine of the pole angle.
    pub sin_theta: f32,
    /// Angular velocity (rad/s).
    pub theta_dot: f32,
}

impl PendulumObservation {
    /// Flatten to a `[f32; 3]` array.
    pub fn to_array(&self) -> [f32; 3] {
        [self.cos_theta, self.sin_theta, self.theta_dot]
    }
}

impl Observation<1> for PendulumObservation {
    fn shape() -> [usize; 1] { [3] }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map angle to `(-π, π]`.
///
/// Uses `rem_euclid` to handle negative inputs correctly.
#[inline]
pub fn angle_normalize(x: f32) -> f32 {
    (x + std::f32::consts::PI).rem_euclid(2.0 * std::f32::consts::PI) - std::f32::consts::PI
}

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

/// Pendulum-v1: swing up and balance a pendulum.
///
/// Never terminates — use [`crate::wrappers::TimeLimit`] to cap episodes.
#[derive(Debug)]
pub struct Pendulum {
    state: PendulumState,
    config: PendulumConfig,
    rng: StdRng,
    steps: usize,
}

impl Pendulum {
    /// Construct with an explicit config.
    pub fn with_config(config: PendulumConfig) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        Self {
            state: PendulumState { theta: 0.0, theta_dot: 0.0 },
            config,
            rng,
            steps: 0,
        }
    }

    fn sample_init_state(&mut self) -> PendulumState {
        let theta     = Uniform::new_inclusive(-std::f32::consts::PI, std::f32::consts::PI).unwrap().sample(&mut self.rng);
        let theta_dot = Uniform::new_inclusive(-1.0_f32, 1.0_f32).unwrap().sample(&mut self.rng);
        PendulumState { theta, theta_dot }
    }

    fn step_dynamics(state: PendulumState, torque: f32, cfg: &PendulumConfig) -> (PendulumState, f32) {
        let u = torque.clamp(-cfg.max_torque, cfg.max_torque);
        let theta_ddot = (3.0 * cfg.g / (2.0 * cfg.l)) * state.theta.sin()
            + (3.0 / (cfg.m * cfg.l * cfg.l)) * u;

        let new_theta_dot = (state.theta_dot + theta_ddot * cfg.dt).clamp(-cfg.max_speed, cfg.max_speed);
        let new_theta = angle_normalize(state.theta + new_theta_dot * cfg.dt);

        let cost = angle_normalize(state.theta).powi(2)
            + 0.1 * state.theta_dot.powi(2)
            + 0.001 * u.powi(2);
        let reward = -cost;

        (PendulumState { theta: new_theta, theta_dot: new_theta_dot }, reward)
    }
}

impl fmt::Display for Pendulum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Pendulum(step={}, θ={:.3}°, θ̇={:.3})",
            self.steps,
            self.state.theta.to_degrees(),
            self.state.theta_dot,
        )
    }
}

impl Environment<1, 1, 1> for Pendulum {
    type StateType = PendulumState;
    type ObservationType = PendulumObservation;
    type ActionType = PendulumAction;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, PendulumObservation, ScalarReward>;

    fn new(render: bool) -> Self {
        let _ = render;
        Self::with_config(PendulumConfig::default())
    }

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.rng = StdRng::seed_from_u64(self.config.seed);
        self.state = self.sample_init_state();
        self.steps = 0;
        Ok(SnapshotBase::running(self.state.observe(), ScalarReward(0.0)))
    }

    fn step(&mut self, action: PendulumAction) -> Result<Self::SnapshotType, EnvironmentError> {
        let (next_state, reward_f) = Self::step_dynamics(self.state, action.torque(), &self.config);
        self.state = next_state;
        self.steps += 1;
        Ok(SnapshotBase::running(self.state.observe(), ScalarReward(reward_f)))
    }
}

// ---------------------------------------------------------------------------
// ASCII renderer
// ---------------------------------------------------------------------------

impl crate::render::AsciiRenderable for Pendulum {
    fn render_ascii(&self) -> String {
        let angle_deg = self.state.theta.to_degrees();
        format!(
            "Pendulum  θ={:.1}°  θ̇={:.2} rad/s  step={}",
            angle_deg, self.state.theta_dot, self.steps
        )
    }
}

// ---------------------------------------------------------------------------
// TensorConvertible
// ---------------------------------------------------------------------------

impl<B: burn::tensor::backend::Backend> evorl_core::base::TensorConvertible<1, B>
    for PendulumObservation
{
    fn to_tensor(&self, device: &B::Device) -> burn::tensor::Tensor<B, 1> {
        burn::tensor::Tensor::from_floats(self.to_array(), device)
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
        Ok(Self { cos_theta: v[0], sin_theta: v[1], theta_dot: v[2] })
    }
}

impl<B: burn::tensor::backend::Backend> evorl_core::base::TensorConvertible<1, B>
    for PendulumAction
{
    fn to_tensor(&self, device: &B::Device) -> burn::tensor::Tensor<B, 1> {
        burn::tensor::Tensor::from_floats([self.0], device)
    }

    fn from_tensor(
        tensor: burn::tensor::Tensor<B, 1>,
    ) -> Result<Self, evorl_core::base::TensorConversionError> {
        let dims = tensor.shape().dims;
        if dims.as_slice() != [1] {
            return Err(evorl_core::base::TensorConversionError {
                message: format!("expected shape [1], got {dims:?}"),
            });
        }
        let v = tensor
            .into_data()
            .into_vec::<f32>()
            .map_err(|e| evorl_core::base::TensorConversionError { message: e.to_string() })?;
        Ok(Self::unchecked(v[0]))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use evorl_core::environment::Snapshot;

    fn default_env() -> Pendulum {
        Pendulum::with_config(PendulumConfig::default())
    }

    #[test]
    fn observation_shape() {
        assert_eq!(PendulumObservation::shape(), [3]);
    }

    #[test]
    fn action_validation() {
        assert!(PendulumAction::new(2.0).is_ok());
        assert!(PendulumAction::new(-2.0).is_ok());
        assert!(PendulumAction::new(2.1).is_err());
        assert!(PendulumAction::new(f32::INFINITY).is_err());
    }

    #[test]
    fn upright_zero_reward() {
        let state = PendulumState { theta: 0.0, theta_dot: 0.0 };
        let cfg = PendulumConfig::default();
        let (_, reward) = Pendulum::step_dynamics(state, 0.0, &cfg);
        assert!((reward - 0.0).abs() < 1e-6, "reward={reward}");
    }

    #[test]
    fn worst_case_reward_approx_minus_16() {
        let cfg = PendulumConfig::default();
        let state = PendulumState { theta: std::f32::consts::PI, theta_dot: cfg.max_speed };
        let (_, reward) = Pendulum::step_dynamics(state, cfg.max_torque, &cfg);
        assert!(reward < -15.0, "expected ≈ -16.27, got {reward}");
    }

    #[test]
    fn angle_normalize_examples() {
        let pi = std::f32::consts::PI;
        // 3π and -3π are both ≡ π (mod 2π). The formula maps them to -π (same angle).
        let n3pi = angle_normalize(3.0 * pi);
        assert!(n3pi.abs() - pi < 1e-4, "angle_normalize(3π)={n3pi} should be ±π");
        let nm3pi = angle_normalize(-3.0 * pi);
        assert!(nm3pi.abs() - pi < 1e-4, "angle_normalize(-3π)={nm3pi} should be ±π");
        // 2π → 0; -2π → 0
        assert!(angle_normalize(2.0 * pi).abs() < 1e-4);
        assert!(angle_normalize(0.0).abs() < 1e-4);
        // Result always in (-π, π]
        for x in [-5.0_f32, -pi, -0.5, 0.0, 0.5, pi, 5.0, 7.0] {
            let n = angle_normalize(x);
            assert!(n >= -pi - 1e-4 && n <= pi + 1e-4, "normalize({x}) = {n} out of range");
        }
    }

    #[test]
    fn never_terminates() {
        let mut env = default_env();
        env.reset().unwrap();
        let action = PendulumAction::new(1.0).unwrap();
        for _ in 0..500 {
            let snap = env.step(action).unwrap();
            assert!(!snap.is_terminated(), "should never terminate");
            assert_eq!(snap.status(), EpisodeStatus::Running);
        }
    }

    #[test]
    fn determinism() {
        let mut a = Pendulum::with_config(PendulumConfig { seed: 11, ..Default::default() });
        let mut b = Pendulum::with_config(PendulumConfig { seed: 11, ..Default::default() });
        a.reset().unwrap();
        b.reset().unwrap();
        let action = PendulumAction::unchecked(0.5);
        for _ in 0..10 {
            let sa = a.step(action).unwrap();
            let sb = b.step(action).unwrap();
            assert_eq!(sa.observation().to_array(), sb.observation().to_array());
        }
    }
}
