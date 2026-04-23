//! CartPole-v1 environment.
//!
//! Balance a pole attached to a cart by applying left or right forces.
//! Physics from Barto, Sutton, and Anderson (1983); equations of motion
//! match the Gymnasium `CartPole-v1` reference implementation exactly.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use rlevo_envs::{classic::CartPole, wrappers::TimeLimit};
//! use rlevo_core::environment::{Environment, Snapshot};
//!
//! let env = CartPole::with_config(CartPoleConfig::default());
//! let mut timed = TimeLimit::new(env, 500);
//! let mut snap = timed.reset().unwrap();
//! while !snap.is_done() {
//!     snap = timed.step(CartPoleAction::Right).unwrap();
//! }
//! ```
use std::fmt;

use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Uniform};
use rlevo_core::{
    action::DiscreteAction,
    base::{Action, Observation, State, TensorConversionError, TensorConvertible},
    environment::{Environment, EnvironmentError, SnapshotBase},
    reward::ScalarReward,
};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Integration scheme for the CartPole equations of motion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Integrator {
    /// Forward Euler (Gymnasium default).
    #[default]
    Euler,
    /// Semi-implicit Euler: velocities updated first, then positions.
    SemiImplicit,
}

/// Configuration for [`CartPole`].
///
/// All defaults match `CartPole-v1` in Gymnasium.
#[derive(Debug, Clone)]
pub struct CartPoleConfig {
    /// Gravitational acceleration (m/s²). Default: `9.8`.
    pub gravity: f32,
    /// Mass of the cart (kg). Default: `1.0`.
    pub masscart: f32,
    /// Mass of the pole (kg). Default: `0.1`.
    pub masspole: f32,
    /// Half the pole length (m). Default: `0.5`.
    pub length: f32,
    /// Magnitude of the force applied to the cart (N). Default: `10.0`.
    pub force_mag: f32,
    /// Time step between updates (s). Default: `0.02`.
    pub tau: f32,
    /// Pole angle at which the episode terminates (rad). Default: `12° ≈ 0.20944`.
    pub theta_threshold_radians: f32,
    /// Cart position at which the episode terminates (m). Default: `2.4`.
    pub x_threshold: f32,
    /// Integrator variant. Default: `Euler`.
    pub integrator: Integrator,
    /// Use the Sutton-Barto reward schedule (`0` per step, `-1` on failure).
    /// Default: `false` (+1 every step).
    pub sutton_barto_reward: bool,
    /// RNG seed; `reset()` re-seeds from this value. Default: `0`.
    pub seed: u64,
}

impl Default for CartPoleConfig {
    fn default() -> Self {
        Self {
            gravity: 9.8,
            masscart: 1.0,
            masspole: 0.1,
            length: 0.5,
            force_mag: 10.0,
            tau: 0.02,
            theta_threshold_radians: 12.0_f32.to_radians(),
            x_threshold: 2.4,
            integrator: Integrator::Euler,
            sutton_barto_reward: false,
            seed: 0,
        }
    }
}

/// Builder for [`CartPoleConfig`].
///
/// Construct via [`CartPoleConfig::builder()`] then call setters and `.build()`.
#[derive(Debug, Default)]
pub struct CartPoleConfigBuilder {
    inner: CartPoleConfig,
}

impl CartPoleConfig {
    /// Create a builder seeded from this config's defaults.
    pub fn builder() -> CartPoleConfigBuilder {
        CartPoleConfigBuilder {
            inner: CartPoleConfig::default(),
        }
    }
}

impl CartPoleConfigBuilder {
    pub fn gravity(mut self, v: f32) -> Self {
        self.inner.gravity = v;
        self
    }
    pub fn masscart(mut self, v: f32) -> Self {
        self.inner.masscart = v;
        self
    }
    pub fn masspole(mut self, v: f32) -> Self {
        self.inner.masspole = v;
        self
    }
    pub fn length(mut self, v: f32) -> Self {
        self.inner.length = v;
        self
    }
    pub fn force_mag(mut self, v: f32) -> Self {
        self.inner.force_mag = v;
        self
    }
    pub fn tau(mut self, v: f32) -> Self {
        self.inner.tau = v;
        self
    }
    pub fn theta_threshold_radians(mut self, v: f32) -> Self {
        self.inner.theta_threshold_radians = v;
        self
    }
    pub fn x_threshold(mut self, v: f32) -> Self {
        self.inner.x_threshold = v;
        self
    }
    pub fn integrator(mut self, v: Integrator) -> Self {
        self.inner.integrator = v;
        self
    }
    pub fn sutton_barto_reward(mut self, v: bool) -> Self {
        self.inner.sutton_barto_reward = v;
        self
    }
    pub fn seed(mut self, v: u64) -> Self {
        self.inner.seed = v;
        self
    }

    /// Finalise and return the config.
    pub fn build(self) -> CartPoleConfig {
        self.inner
    }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// Internal state of the CartPole system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CartPoleState {
    /// Cart position (m).
    pub x: f32,
    /// Cart velocity (m/s).
    pub x_dot: f32,
    /// Pole angle (rad), positive = clockwise.
    pub theta: f32,
    /// Pole angular velocity (rad/s).
    pub theta_dot: f32,
}

impl CartPoleState {
    fn new(x: f32, x_dot: f32, theta: f32, theta_dot: f32) -> Self {
        Self {
            x,
            x_dot,
            theta,
            theta_dot,
        }
    }
}

impl State<1> for CartPoleState {
    type Observation = CartPoleObservation;

    fn shape() -> [usize; 1] {
        [4]
    }
    fn numel(&self) -> usize {
        4
    }

    fn is_valid(&self) -> bool {
        self.x.is_finite()
            && self.x_dot.is_finite()
            && self.theta.is_finite()
            && self.theta_dot.is_finite()
    }

    fn observe(&self) -> CartPoleObservation {
        CartPoleObservation {
            cart_pos: self.x,
            cart_vel: self.x_dot,
            pole_angle: self.theta,
            pole_ang_vel: self.theta_dot,
        }
    }
}

// ---------------------------------------------------------------------------
// Observation
// ---------------------------------------------------------------------------

/// Observation returned by [`CartPole`] at each step.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CartPoleObservation {
    /// Cart position (m).
    pub cart_pos: f32,
    /// Cart velocity (m/s).
    pub cart_vel: f32,
    /// Pole angle (rad).
    pub pole_angle: f32,
    /// Pole angular velocity (rad/s).
    pub pole_ang_vel: f32,
}

impl CartPoleObservation {
    /// Flatten to a `[f32; 4]` array for tensor conversion.
    pub fn to_array(&self) -> [f32; 4] {
        [
            self.cart_pos,
            self.cart_vel,
            self.pole_angle,
            self.pole_ang_vel,
        ]
    }
}

impl Observation<1> for CartPoleObservation {
    fn shape() -> [usize; 1] {
        [4]
    }
}

// ---------------------------------------------------------------------------
// Action
// ---------------------------------------------------------------------------

/// Discrete action for [`CartPole`]: push left or push right.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CartPoleAction {
    /// Apply force in the negative-x direction.
    Left,
    /// Apply force in the positive-x direction.
    Right,
}

impl Action<1> for CartPoleAction {
    fn shape() -> [usize; 1] {
        [2]
    }
    fn is_valid(&self) -> bool {
        true
    }
}

impl DiscreteAction<1> for CartPoleAction {
    const ACTION_COUNT: usize = 2;

    fn from_index(index: usize) -> Self {
        match index {
            0 => Self::Left,
            1 => Self::Right,
            _ => panic!("CartPoleAction index out of range: {index}"),
        }
    }

    fn to_index(&self) -> usize {
        match self {
            Self::Left => 0,
            Self::Right => 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

/// CartPole-v1: balance a pole on a moving cart.
///
/// No intrinsic step cap. Wrap with [`crate::wrappers::TimeLimit::new(env, 500)`]
/// to replicate the Gymnasium v1 500-step episode limit.
#[derive(Debug)]
pub struct CartPole {
    state: CartPoleState,
    config: CartPoleConfig,
    rng: StdRng,
    steps: usize,
}

impl CartPole {
    /// Construct with an explicit config.
    pub fn with_config(config: CartPoleConfig) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        Self {
            state: CartPoleState::new(0.0, 0.0, 0.0, 0.0),
            config,
            rng,
            steps: 0,
        }
    }

    /// Current step count within the episode.
    pub fn steps(&self) -> usize {
        self.steps
    }

    fn sample_init_state(&mut self) -> CartPoleState {
        let u = Uniform::new_inclusive(-0.05_f32, 0.05_f32).unwrap();
        CartPoleState::new(
            u.sample(&mut self.rng),
            u.sample(&mut self.rng),
            u.sample(&mut self.rng),
            u.sample(&mut self.rng),
        )
    }

    fn is_terminal(state: &CartPoleState, cfg: &CartPoleConfig) -> bool {
        state.x.abs() > cfg.x_threshold
            || state.theta.abs() > cfg.theta_threshold_radians
            || !state.is_valid()
    }

    /// Apply equations of motion and return the next state.
    fn step_physics(
        state: &CartPoleState,
        action: CartPoleAction,
        cfg: &CartPoleConfig,
    ) -> CartPoleState {
        let force = if action == CartPoleAction::Right {
            cfg.force_mag
        } else {
            -cfg.force_mag
        };
        let total_mass = cfg.masscart + cfg.masspole;
        let pm_l = cfg.masspole * cfg.length;

        let cos_t = state.theta.cos();
        let sin_t = state.theta.sin();

        let temp = (force + pm_l * state.theta_dot * state.theta_dot * sin_t) / total_mass;
        let theta_acc = (cfg.gravity * sin_t - cos_t * temp)
            / (cfg.length * (4.0 / 3.0 - cfg.masspole * cos_t * cos_t / total_mass));
        let x_acc = temp - pm_l * theta_acc * cos_t / total_mass;

        match cfg.integrator {
            Integrator::Euler => CartPoleState {
                x: state.x + cfg.tau * state.x_dot,
                x_dot: state.x_dot + cfg.tau * x_acc,
                theta: state.theta + cfg.tau * state.theta_dot,
                theta_dot: state.theta_dot + cfg.tau * theta_acc,
            },
            Integrator::SemiImplicit => {
                let x_dot_new = state.x_dot + cfg.tau * x_acc;
                let theta_dot_new = state.theta_dot + cfg.tau * theta_acc;
                CartPoleState {
                    x: state.x + cfg.tau * x_dot_new,
                    x_dot: x_dot_new,
                    theta: state.theta + cfg.tau * theta_dot_new,
                    theta_dot: theta_dot_new,
                }
            }
        }
    }
}

impl fmt::Display for CartPole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CartPole(step={}, x={:.3}, θ={:.3}°)",
            self.steps,
            self.state.x,
            self.state.theta.to_degrees(),
        )
    }
}

impl Environment<1, 1, 1> for CartPole {
    type StateType = CartPoleState;
    type ObservationType = CartPoleObservation;
    type ActionType = CartPoleAction;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, CartPoleObservation, ScalarReward>;

    fn new(render: bool) -> Self {
        let _ = render;
        Self::with_config(CartPoleConfig::default())
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

    fn step(&mut self, action: CartPoleAction) -> Result<Self::SnapshotType, EnvironmentError> {
        let next = Self::step_physics(&self.state, action, &self.config);
        self.state = next;
        self.steps += 1;

        let terminated = Self::is_terminal(&self.state, &self.config);
        let reward = if self.config.sutton_barto_reward {
            if terminated {
                ScalarReward(-1.0)
            } else {
                ScalarReward(0.0)
            }
        } else {
            ScalarReward(1.0)
        };

        let snap = if terminated {
            SnapshotBase::terminated(self.state.observe(), reward)
        } else {
            SnapshotBase::running(self.state.observe(), reward)
        };
        Ok(snap)
    }
}

// ---------------------------------------------------------------------------
// ASCII renderer
// ---------------------------------------------------------------------------

impl crate::render::AsciiRenderable for CartPole {
    fn render_ascii(&self) -> String {
        let width = 60_usize;
        let cart_frac = (self.state.x / self.config.x_threshold * 0.5 + 0.5).clamp(0.0, 1.0);
        let cart_col = (cart_frac * (width as f32 - 1.0)) as usize;
        let mut track = vec!['-'; width];
        track[cart_col] = '#';
        let track_str: String = track.iter().collect();
        let angle_deg = self.state.theta.to_degrees();
        format!("[{track_str}]  θ={angle_deg:.1}°  step={}", self.steps)
    }
}

// ---------------------------------------------------------------------------
// TensorConvertible
// ---------------------------------------------------------------------------

impl<B: burn::tensor::backend::Backend> TensorConvertible<1, B> for CartPoleObservation {
    fn to_tensor(&self, device: &B::Device) -> burn::tensor::Tensor<B, 1> {
        burn::tensor::Tensor::from_floats(self.to_array(), device)
    }

    fn from_tensor(tensor: burn::tensor::Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let dims = tensor.shape().dims;
        if dims.as_slice() != [4] {
            return Err(TensorConversionError {
                message: format!("expected shape [4], got {dims:?}"),
            });
        }
        let v = tensor
            .into_data()
            .into_vec::<f32>()
            .map_err(|e| TensorConversionError {
                message: e.to_string(),
            })?;
        Ok(Self {
            cart_pos: v[0],
            cart_vel: v[1],
            pole_angle: v[2],
            pole_ang_vel: v[3],
        })
    }
}

impl<B: burn::tensor::backend::Backend> TensorConvertible<1, B> for CartPoleAction {
    fn to_tensor(&self, device: &B::Device) -> burn::tensor::Tensor<B, 1> {
        let mut one_hot = [0.0_f32; 2];
        one_hot[self.to_index()] = 1.0;
        burn::tensor::Tensor::from_floats(one_hot, device)
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
        let idx = v
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        Ok(Self::from_index(idx))
    }
}

// helper for divergence test
#[cfg(test)]
impl CartPoleState {
    fn to_array(self) -> [f32; 4] {
        [self.x, self.x_dot, self.theta, self.theta_dot]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rlevo_core::environment::Snapshot;

    fn default_env() -> CartPole {
        CartPole::with_config(CartPoleConfig::default())
    }

    #[test]
    fn reset_returns_running_obs_in_range() {
        use rlevo_core::environment::EpisodeStatus;

        let mut env = default_env();
        let snap = env.reset().unwrap();
        assert_eq!(snap.status(), EpisodeStatus::Running);
        assert!(!snap.is_done());
        let obs = snap.observation();
        for v in obs.to_array() {
            assert!(v.abs() <= 0.05 + f32::EPSILON, "init obs {v} out of range");
        }
    }

    #[test]
    fn observation_shape() {
        assert_eq!(CartPoleObservation::shape(), [4]);
    }

    #[test]
    fn action_count() {
        assert_eq!(CartPoleAction::ACTION_COUNT, 2);
        assert_eq!(CartPoleAction::from_index(0), CartPoleAction::Left);
        assert_eq!(CartPoleAction::from_index(1), CartPoleAction::Right);
        assert_eq!(CartPoleAction::Left.to_index(), 0);
        assert_eq!(CartPoleAction::Right.to_index(), 1);
    }

    #[test]
    fn terminates_on_large_angle() {
        use rlevo_core::environment::EpisodeStatus;

        let mut env = default_env();
        env.reset().unwrap();
        // Set pole way past threshold
        env.state.theta = 0.3;
        let snap = env.step(CartPoleAction::Left).unwrap();
        assert_eq!(snap.status(), EpisodeStatus::Terminated);
        assert!(snap.is_terminated());
        assert!(!snap.is_truncated());
    }

    #[test]
    fn terminates_on_large_position() {
        let mut env = default_env();
        env.reset().unwrap();
        env.state.x = 2.5;
        let snap = env.step(CartPoleAction::Left).unwrap();
        assert!(snap.is_terminated());
    }

    #[test]
    fn default_reward_is_one_per_step() {
        let mut env = default_env();
        env.reset().unwrap();
        let snap = env.step(CartPoleAction::Right).unwrap();
        if !snap.is_done() {
            assert_eq!(*snap.reward(), ScalarReward(1.0));
        }
    }

    #[test]
    fn sutton_barto_reward_switch() {
        let config = CartPoleConfig {
            sutton_barto_reward: true,
            ..Default::default()
        };
        let mut env = CartPole::with_config(config);
        env.reset().unwrap();
        // Force termination next step
        env.state.theta = 0.3;
        let snap = env.step(CartPoleAction::Left).unwrap();
        assert!(snap.is_done());
        assert_eq!(*snap.reward(), ScalarReward(-1.0));
    }

    #[test]
    fn sutton_barto_zero_for_non_terminal_step() {
        let config = CartPoleConfig {
            sutton_barto_reward: true,
            ..Default::default()
        };
        let mut env = CartPole::with_config(config);
        env.reset().unwrap();
        let snap = env.step(CartPoleAction::Right).unwrap();
        if !snap.is_done() {
            assert_eq!(*snap.reward(), ScalarReward(0.0));
        }
    }

    #[test]
    fn determinism() {
        let mut env_a = CartPole::with_config(CartPoleConfig {
            seed: 42,
            ..Default::default()
        });
        let mut env_b = CartPole::with_config(CartPoleConfig {
            seed: 42,
            ..Default::default()
        });
        env_a.reset().unwrap();
        env_b.reset().unwrap();

        let actions = [
            CartPoleAction::Right,
            CartPoleAction::Left,
            CartPoleAction::Right,
        ];
        for action in actions {
            let sa = env_a.step(action).unwrap();
            let sb = env_b.step(action).unwrap();
            assert_eq!(sa.observation().to_array(), sb.observation().to_array());
        }
    }

    #[test]
    fn euler_and_semi_implicit_diverge_after_many_steps() {
        let euler_cfg = CartPoleConfig {
            integrator: Integrator::Euler,
            seed: 1,
            ..Default::default()
        };
        let si_cfg = CartPoleConfig {
            integrator: Integrator::SemiImplicit,
            seed: 1,
            ..Default::default()
        };
        let mut euler_env = CartPole::with_config(euler_cfg);
        let mut si_env = CartPole::with_config(si_cfg);
        euler_env.reset().unwrap();
        si_env.reset().unwrap();

        for _ in 0..100 {
            let _ = euler_env.step(CartPoleAction::Right);
            let _ = si_env.step(CartPoleAction::Right);
            if euler_env.state != si_env.state {
                return; // diverged — pass
            }
        }
        // If they somehow match for 100 steps, ensure they're at least close (sanity)
        // Both integrate the same ODE so they may be close but should differ.
        // This test just checks the SemiImplicit branch is wired up, not that
        // they produce wildly different results.
        let diff: f32 = euler_env
            .state
            .to_array()
            .iter()
            .zip(si_env.state.to_array().iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 0.0,
            "Euler and SemiImplicit produced identical states"
        );
    }

    #[test]
    fn config_builder_roundtrip() {
        let cfg = CartPoleConfig::builder()
            .gravity(9.81)
            .masscart(2.0)
            .seed(99)
            .build();
        assert!((cfg.gravity - 9.81).abs() < 1e-6);
        assert!((cfg.masscart - 2.0).abs() < 1e-6);
        assert_eq!(cfg.seed, 99);
    }
}
