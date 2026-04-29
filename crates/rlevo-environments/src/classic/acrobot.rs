//! Acrobot-v1 environment.
//!
//! A two-link pendulum where the second joint is actuated. The goal is to
//! swing the end of the lower link above a target height. Reference:
//! Sutton & Barto §10.
//!
//! ## Dynamics variants
//!
//! Two dynamics implementations are available:
//!
//! | Type | Description |
//! |------|-------------|
//! | [`BookDynamics`] | Sutton & Barto textbook form (Gymnasium default) |
//! | [`NipsDynamics`] | Original NIPS-1995 form — omits certain cross-terms |
//!
//! The environment is generic over the dynamics via [`AcrobotDynamicsFn`]:
//! `Acrobot<BookDynamics>` and `Acrobot<NipsDynamics>`.
//!
//! ## Step limit
//!
//! There is no intrinsic step cap. Compose with a `TimeLimit` wrapper for the
//! standard 500-step v1 limit.
//!
//! ## Quick start
//!
//! ```no_run,ignore
//! use rlevo_environments::classic::acrobot::{Acrobot, AcrobotAction, AcrobotConfig, BookDynamics};
//! use rlevo_core::environment::Environment;
//!
//! let mut env = Acrobot::<BookDynamics>::with_config(AcrobotConfig::default());
//! let _snap = env.reset().unwrap();
//! let snap = env.step(AcrobotAction::TorquePos).unwrap();
//! println!("terminated: {}", snap.is_terminated());
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
// Strategy trait (spec B4)
// ---------------------------------------------------------------------------

/// Computes the time derivative of the Acrobot state vector.
///
/// Both [`BookDynamics`] and [`NipsDynamics`] implement this trait. Supply it
/// as the generic parameter `D` on [`Acrobot<D>`] to select the desired
/// physics model at zero runtime cost (monomorphization).
///
/// Implement this trait only when you need custom dynamics; prefer the
/// provided types for standard use.
pub trait AcrobotDynamicsFn: fmt::Debug + Clone + Send + Sync {
    /// Computes `d/dt [θ1, θ2, θ̇1, θ̇2]` for the given state and torque.
    ///
    /// `s` is `[theta1, theta2, dtheta1, dtheta2]` and `a` is the applied
    /// torque (N·m). The returned array has the same layout.
    fn dsdt(&self, s: [f32; 4], a: f32, cfg: &AcrobotConfig) -> [f32; 4];
}

/// Sutton & Barto textbook dynamics — the Gymnasium default.
///
/// Includes the full `2·dθ2·dθ1` Coriolis cross-term and the `dθ1²`
/// centripetal term in φ₁. This is the recommended choice for reproducing
/// published benchmark results.
#[derive(Debug, Clone, Copy, Default)]
pub struct BookDynamics;

/// Original NIPS-1995 dynamics — omits certain cross-terms.
///
/// φ₁ drops the `2·dθ2·dθ1` Coriolis cross-term and the `dθ1²`
/// centripetal term present in [`BookDynamics`]. Produces subtly different
/// trajectories and is provided for historical reproducibility.
#[derive(Debug, Clone, Copy, Default)]
pub struct NipsDynamics;

impl AcrobotDynamicsFn for BookDynamics {
    fn dsdt(&self, s: [f32; 4], a: f32, cfg: &AcrobotConfig) -> [f32; 4] {
        let [theta1, theta2, dtheta1, dtheta2] = s;
        let m1 = cfg.link_mass_1;
        let m2 = cfg.link_mass_2;
        let lc1 = cfg.link_com_pos_1;
        let lc2 = cfg.link_com_pos_2;
        let l1 = cfg.link_length_1;
        let i1 = cfg.link_moi;
        let i2 = cfg.link_moi;
        let g = cfg.gravity;

        let d1 =
            m1 * lc1 * lc1 + m2 * (l1 * l1 + lc2 * lc2 + 2.0 * l1 * lc2 * theta2.cos()) + i1 + i2;
        let d2 = m2 * (lc2 * lc2 + l1 * lc2 * theta2.cos()) + i2;
        let phi2 = m2 * lc2 * g * (theta1 + theta2 - std::f32::consts::FRAC_PI_2).cos();
        let phi1 = -(m2 * l1 * lc2 * dtheta2 * dtheta2 * theta2.sin())
            - 2.0 * m2 * l1 * lc2 * dtheta2 * dtheta1 * theta2.sin()
            + (m1 * lc1 + m2 * l1) * g * (theta1 - std::f32::consts::FRAC_PI_2).cos()
            + phi2;

        let ddtheta2 =
            (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 * dtheta1 * theta2.sin() - phi2)
                / (m2 * lc2 * lc2 + i2 - d2 * d2 / d1);
        let ddtheta1 = -(d2 * ddtheta2 + phi1) / d1;

        [dtheta1, dtheta2, ddtheta1, ddtheta2]
    }
}

impl AcrobotDynamicsFn for NipsDynamics {
    fn dsdt(&self, s: [f32; 4], a: f32, cfg: &AcrobotConfig) -> [f32; 4] {
        let [theta1, theta2, dtheta1, dtheta2] = s;
        let m1 = cfg.link_mass_1;
        let m2 = cfg.link_mass_2;
        let lc1 = cfg.link_com_pos_1;
        let lc2 = cfg.link_com_pos_2;
        let l1 = cfg.link_length_1;
        let i1 = cfg.link_moi;
        let i2 = cfg.link_moi;
        let g = cfg.gravity;

        let d1 =
            m1 * lc1 * lc1 + m2 * (l1 * l1 + lc2 * lc2 + 2.0 * l1 * lc2 * theta2.cos()) + i1 + i2;
        let d2 = m2 * (lc2 * lc2 + l1 * lc2 * theta2.cos()) + i2;
        let phi2 = m2 * lc2 * g * (theta1 + theta2 - std::f32::consts::FRAC_PI_2).cos();
        // NIPS form omits the dtheta1^2 term and the 2*dtheta2*dtheta1 cross-term
        let phi1 = -(m2 * l1 * lc2 * dtheta2 * dtheta2 * theta2.sin())
            + (m1 * lc1 + m2 * l1) * g * (theta1 - std::f32::consts::FRAC_PI_2).cos()
            + phi2;

        let ddtheta2 = (a + d2 / d1 * phi1 - phi2) / (m2 * lc2 * lc2 + i2 - d2 * d2 / d1);
        let ddtheta1 = -(d2 * ddtheta2 + phi1) / d1;

        [dtheta1, dtheta2, ddtheta1, ddtheta2]
    }
}

// ---------------------------------------------------------------------------
// RK4 integrator
// ---------------------------------------------------------------------------

/// Advances `s` by one step of size `dt` using the classic RK4 method.
fn rk4<F: Fn([f32; 4]) -> [f32; 4]>(dsdt: F, s: [f32; 4], dt: f32) -> [f32; 4] {
    let k1 = dsdt(s);
    let k2 = dsdt(add4(s, scale4(k1, dt / 2.0)));
    let k3 = dsdt(add4(s, scale4(k2, dt / 2.0)));
    let k4 = dsdt(add4(s, scale4(k3, dt)));
    let k = add4(add4(k1, scale4(k2, 2.0)), add4(scale4(k3, 2.0), k4));
    add4(s, scale4(k, dt / 6.0))
}

#[inline]
fn add4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}
#[inline]
fn scale4(a: [f32; 4], s: f32) -> [f32; 4] {
    [a[0] * s, a[1] * s, a[2] * s, a[3] * s]
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for [`Acrobot`].
///
/// All defaults match Gymnasium `Acrobot-v1` with [`BookDynamics`]. Use
/// [`AcrobotConfig::builder()`] for a fluent construction style.
///
/// # Examples
///
/// ```rust
/// use rlevo_environments::classic::acrobot::AcrobotConfig;
///
/// let cfg = AcrobotConfig::builder()
///     .gravity(9.81)
///     .dt(0.05)
///     .seed(42)
///     .build();
/// assert!((cfg.gravity - 9.81).abs() < 1e-5);
/// ```
#[derive(Debug, Clone)]
pub struct AcrobotConfig {
    /// Integration time step (s). Default: `0.2`.
    pub dt: f32,
    /// Length of link 1 (m). Default: `1.0`.
    pub link_length_1: f32,
    /// Length of link 2 (m). Default: `1.0`.
    pub link_length_2: f32,
    /// Mass of link 1 (kg). Default: `1.0`.
    pub link_mass_1: f32,
    /// Mass of link 2 (kg). Default: `1.0`.
    pub link_mass_2: f32,
    /// COM position of link 1 (fraction of length). Default: `0.5`.
    pub link_com_pos_1: f32,
    /// COM position of link 2 (fraction of length). Default: `0.5`.
    pub link_com_pos_2: f32,
    /// Moment of inertia for both links. Default: `1.0`.
    pub link_moi: f32,
    /// Gravitational acceleration (m/s²). Default: `9.8`.
    pub gravity: f32,
    /// Max angular velocity for joint 1 (rad/s). Default: `4π`.
    pub max_vel_1: f32,
    /// Max angular velocity for joint 2 (rad/s). Default: `9π`.
    pub max_vel_2: f32,
    /// Uniform noise added to the applied torque. `0` = deterministic. Default: `0.0`.
    pub torque_noise_max: f32,
    /// RNG seed; `reset()` re-seeds from this value. Default: `0`.
    pub seed: u64,
}

impl Default for AcrobotConfig {
    fn default() -> Self {
        Self {
            dt: 0.2,
            link_length_1: 1.0,
            link_length_2: 1.0,
            link_mass_1: 1.0,
            link_mass_2: 1.0,
            link_com_pos_1: 0.5,
            link_com_pos_2: 0.5,
            link_moi: 1.0,
            gravity: 9.8,
            max_vel_1: 4.0 * std::f32::consts::PI,
            max_vel_2: 9.0 * std::f32::consts::PI,
            torque_noise_max: 0.0,
            seed: 0,
        }
    }
}

/// Fluent builder for [`AcrobotConfig`].
///
/// Obtain via [`AcrobotConfig::builder()`]. All unset fields retain their
/// default values from [`AcrobotConfig::default()`].
#[derive(Debug, Default)]
pub struct AcrobotConfigBuilder {
    inner: AcrobotConfig,
}

impl AcrobotConfig {
    pub fn builder() -> AcrobotConfigBuilder {
        AcrobotConfigBuilder {
            inner: AcrobotConfig::default(),
        }
    }
}

impl AcrobotConfigBuilder {
    pub fn dt(mut self, v: f32) -> Self {
        self.inner.dt = v;
        self
    }
    pub fn link_length_1(mut self, v: f32) -> Self {
        self.inner.link_length_1 = v;
        self
    }
    pub fn link_length_2(mut self, v: f32) -> Self {
        self.inner.link_length_2 = v;
        self
    }
    pub fn link_mass_1(mut self, v: f32) -> Self {
        self.inner.link_mass_1 = v;
        self
    }
    pub fn link_mass_2(mut self, v: f32) -> Self {
        self.inner.link_mass_2 = v;
        self
    }
    pub fn link_com_pos_1(mut self, v: f32) -> Self {
        self.inner.link_com_pos_1 = v;
        self
    }
    pub fn link_com_pos_2(mut self, v: f32) -> Self {
        self.inner.link_com_pos_2 = v;
        self
    }
    pub fn link_moi(mut self, v: f32) -> Self {
        self.inner.link_moi = v;
        self
    }
    pub fn gravity(mut self, v: f32) -> Self {
        self.inner.gravity = v;
        self
    }
    pub fn max_vel_1(mut self, v: f32) -> Self {
        self.inner.max_vel_1 = v;
        self
    }
    pub fn max_vel_2(mut self, v: f32) -> Self {
        self.inner.max_vel_2 = v;
        self
    }
    pub fn torque_noise_max(mut self, v: f32) -> Self {
        self.inner.torque_noise_max = v;
        self
    }
    pub fn seed(mut self, v: u64) -> Self {
        self.inner.seed = v;
        self
    }
    pub fn build(self) -> AcrobotConfig {
        self.inner
    }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// Internal 4-DOF state of the Acrobot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcrobotState {
    /// Angle of link 1 from upright (rad).
    pub theta1: f32,
    /// Angle of link 2 relative to link 1 (rad).
    pub theta2: f32,
    /// Angular velocity of link 1 (rad/s).
    pub theta1_dot: f32,
    /// Angular velocity of link 2 (rad/s).
    pub theta2_dot: f32,
}

impl AcrobotState {
    /// Converts the state to `[theta1, theta2, dtheta1, dtheta2]`.
    fn to_array(self) -> [f32; 4] {
        [self.theta1, self.theta2, self.theta1_dot, self.theta2_dot]
    }

    /// Reconstructs a state from `[theta1, theta2, dtheta1, dtheta2]`.
    fn from_array(a: [f32; 4]) -> Self {
        Self {
            theta1: a[0],
            theta2: a[1],
            theta1_dot: a[2],
            theta2_dot: a[3],
        }
    }
}

impl State<1> for AcrobotState {
    type Observation = AcrobotObservation;

    fn shape() -> [usize; 1] {
        [4]
    }
    fn numel(&self) -> usize {
        4
    }

    fn is_valid(&self) -> bool {
        self.theta1.is_finite()
            && self.theta2.is_finite()
            && self.theta1_dot.is_finite()
            && self.theta2_dot.is_finite()
    }

    fn observe(&self) -> AcrobotObservation {
        AcrobotObservation {
            cos_theta1: self.theta1.cos(),
            sin_theta1: self.theta1.sin(),
            cos_theta2: self.theta2.cos(),
            sin_theta2: self.theta2.sin(),
            theta1_dot: self.theta1_dot,
            theta2_dot: self.theta2_dot,
        }
    }
}

// ---------------------------------------------------------------------------
// Observation
// ---------------------------------------------------------------------------

/// Observation returned by [`Acrobot`] at each step (6-dimensional).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AcrobotObservation {
    pub cos_theta1: f32,
    pub sin_theta1: f32,
    pub cos_theta2: f32,
    pub sin_theta2: f32,
    pub theta1_dot: f32,
    pub theta2_dot: f32,
}

impl AcrobotObservation {
    /// Flattens the observation to `[cos θ1, sin θ1, cos θ2, sin θ2, dθ1, dθ2]`.
    pub fn to_array(&self) -> [f32; 6] {
        [
            self.cos_theta1,
            self.sin_theta1,
            self.cos_theta2,
            self.sin_theta2,
            self.theta1_dot,
            self.theta2_dot,
        ]
    }
}

impl Observation<1> for AcrobotObservation {
    fn shape() -> [usize; 1] {
        [6]
    }
}

// ---------------------------------------------------------------------------
// Action
// ---------------------------------------------------------------------------

/// Discrete torque action for [`Acrobot`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcrobotAction {
    /// Apply torque `-1` N·m.
    TorqueNeg,
    /// Apply torque `0` N·m.
    TorqueZero,
    /// Apply torque `+1` N·m.
    TorquePos,
}

impl AcrobotAction {
    /// Returns the torque value in N·m (`-1.0`, `0.0`, or `+1.0`).
    fn to_torque(self) -> f32 {
        match self {
            Self::TorqueNeg => -1.0,
            Self::TorqueZero => 0.0,
            Self::TorquePos => 1.0,
        }
    }
}

impl Action<1> for AcrobotAction {
    fn shape() -> [usize; 1] {
        [3]
    }
    fn is_valid(&self) -> bool {
        true
    }
}

impl DiscreteAction<1> for AcrobotAction {
    const ACTION_COUNT: usize = 3;

    /// Constructs an action from its integer index.
    ///
    /// # Panics
    ///
    /// Panics if `index` is not in `0..=2`.
    fn from_index(index: usize) -> Self {
        match index {
            0 => Self::TorqueNeg,
            1 => Self::TorqueZero,
            2 => Self::TorquePos,
            _ => panic!("AcrobotAction index out of range: {index}"),
        }
    }

    fn to_index(&self) -> usize {
        match self {
            Self::TorqueNeg => 0,
            Self::TorqueZero => 1,
            Self::TorquePos => 2,
        }
    }
}

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

/// Acrobot-v1 environment — swing a two-link pendulum end above a threshold.
///
/// Generic over `D: AcrobotDynamicsFn`. Use [`BookDynamics`] (the default) to
/// match Gymnasium's reference implementation, or [`NipsDynamics`] for the
/// original 1995 formulation.
///
/// # Examples
///
/// ```rust,ignore
/// use rlevo_environments::classic::acrobot::{Acrobot, AcrobotAction, AcrobotConfig, BookDynamics};
/// use rlevo_core::environment::Environment;
///
/// let mut env = Acrobot::<BookDynamics>::with_config(AcrobotConfig::default());
/// env.reset().unwrap();
/// loop {
///     let snap = env.step(AcrobotAction::TorquePos).unwrap();
///     if snap.is_terminated() { break; }
/// }
/// ```
pub struct Acrobot<D: AcrobotDynamicsFn = BookDynamics> {
    state: AcrobotState,
    config: AcrobotConfig,
    dynamics: D,
    rng: StdRng,
    steps: usize,
}

impl<D: AcrobotDynamicsFn + Default> Acrobot<D> {
    /// Construct with explicit config, using the default dynamics variant.
    pub fn with_config(config: AcrobotConfig) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        Self {
            state: AcrobotState {
                theta1: 0.0,
                theta2: 0.0,
                theta1_dot: 0.0,
                theta2_dot: 0.0,
            },
            config,
            dynamics: D::default(),
            rng,
            steps: 0,
        }
    }
}

impl<D: AcrobotDynamicsFn> Acrobot<D> {
    /// Construct with an explicit config and dynamics instance.
    pub fn with_config_and_dynamics(config: AcrobotConfig, dynamics: D) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        Self {
            state: AcrobotState {
                theta1: 0.0,
                theta2: 0.0,
                theta1_dot: 0.0,
                theta2_dot: 0.0,
            },
            config,
            dynamics,
            rng,
            steps: 0,
        }
    }

    fn sample_init_state(&mut self) -> AcrobotState {
        let u = Uniform::new_inclusive(-0.1_f32, 0.1_f32).unwrap();
        AcrobotState {
            theta1: u.sample(&mut self.rng),
            theta2: u.sample(&mut self.rng),
            theta1_dot: u.sample(&mut self.rng),
            theta2_dot: u.sample(&mut self.rng),
        }
    }

    fn clamp_velocities(state: AcrobotState, cfg: &AcrobotConfig) -> AcrobotState {
        AcrobotState {
            theta1_dot: state.theta1_dot.clamp(-cfg.max_vel_1, cfg.max_vel_1),
            theta2_dot: state.theta2_dot.clamp(-cfg.max_vel_2, cfg.max_vel_2),
            ..state
        }
    }

    fn is_terminal(state: &AcrobotState, cfg: &AcrobotConfig) -> bool {
        let l1 = cfg.link_length_1;
        let l2 = cfg.link_length_2;
        // end-effector height (positive = up): -cos(θ1)*l1 - cos(θ1+θ2)*l2
        -l1 * state.theta1.cos() - l2 * (state.theta1 + state.theta2).cos() > 1.0
    }
}

impl<D: AcrobotDynamicsFn> fmt::Debug for Acrobot<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Acrobot")
            .field("steps", &self.steps)
            .field("state", &self.state)
            .finish()
    }
}

impl<D: AcrobotDynamicsFn> fmt::Display for Acrobot<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Acrobot(step={}, θ1={:.2}°, θ2={:.2}°)",
            self.steps,
            self.state.theta1.to_degrees(),
            self.state.theta2.to_degrees(),
        )
    }
}

impl<D: AcrobotDynamicsFn + Default> Environment<1, 1, 1> for Acrobot<D> {
    type StateType = AcrobotState;
    type ObservationType = AcrobotObservation;
    type ActionType = AcrobotAction;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, AcrobotObservation, ScalarReward>;

    fn new(render: bool) -> Self {
        let _ = render;
        Self::with_config(AcrobotConfig::default())
    }

    /// Resets the environment to a random initial state and returns the first snapshot.
    ///
    /// Re-seeds the internal RNG from `config.seed`, so repeated calls produce
    /// the same initial trajectory for a given seed.
    ///
    /// # Errors
    ///
    /// Currently infallible; returns `Ok` in all cases.
    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.rng = StdRng::seed_from_u64(self.config.seed);
        self.state = self.sample_init_state();
        self.steps = 0;
        Ok(SnapshotBase::running(
            self.state.observe(),
            ScalarReward(0.0),
        ))
    }

    /// Advances the simulation by one time step and returns the resulting snapshot.
    ///
    /// Applies optional torque noise (if `config.torque_noise_max > 0`),
    /// integrates the dynamics with RK4, clamps joint velocities, and checks
    /// the terminal condition. The reward is `-1.0` on non-terminal steps and
    /// `0.0` on the terminal step.
    ///
    /// # Errors
    ///
    /// Currently infallible; returns `Ok` in all cases.
    fn step(&mut self, action: AcrobotAction) -> Result<Self::SnapshotType, EnvironmentError> {
        let mut torque = action.to_torque();
        if self.config.torque_noise_max > 0.0 {
            let noise =
                Uniform::new_inclusive(-self.config.torque_noise_max, self.config.torque_noise_max)
                    .unwrap();
            torque += noise.sample(&mut self.rng);
        }

        let s = self.state.to_array();
        let cfg = &self.config;
        let dyn_ref = &self.dynamics;
        let ns = rk4(|s| dyn_ref.dsdt(s, torque, cfg), s, cfg.dt);
        self.state = Self::clamp_velocities(AcrobotState::from_array(ns), cfg);
        self.steps += 1;

        let terminated = Self::is_terminal(&self.state, &self.config);
        let snap = if terminated {
            SnapshotBase::terminated(self.state.observe(), ScalarReward(0.0))
        } else {
            SnapshotBase::running(self.state.observe(), ScalarReward(-1.0))
        };
        Ok(snap)
    }
}

// ---------------------------------------------------------------------------
// TensorConvertible
// ---------------------------------------------------------------------------

impl<B: burn::tensor::backend::Backend> TensorConvertible<1, B> for AcrobotObservation {
    fn to_tensor(&self, device: &B::Device) -> burn::tensor::Tensor<B, 1> {
        burn::tensor::Tensor::from_floats(self.to_array(), device)
    }

    fn from_tensor(tensor: burn::tensor::Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let dims = tensor.shape().dims;
        if dims.as_slice() != [6] {
            return Err(TensorConversionError {
                message: format!("expected shape [6], got {dims:?}"),
            });
        }
        let v = tensor
            .into_data()
            .into_vec::<f32>()
            .map_err(|e| TensorConversionError {
                message: e.to_string(),
            })?;
        Ok(Self {
            cos_theta1: v[0],
            sin_theta1: v[1],
            cos_theta2: v[2],
            sin_theta2: v[3],
            theta1_dot: v[4],
            theta2_dot: v[5],
        })
    }
}

impl<B: burn::tensor::backend::Backend> TensorConvertible<1, B> for AcrobotAction {
    fn to_tensor(&self, device: &B::Device) -> burn::tensor::Tensor<B, 1> {
        let mut one_hot = [0.0_f32; 3];
        one_hot[self.to_index()] = 1.0;
        burn::tensor::Tensor::from_floats(one_hot, device)
    }

    fn from_tensor(tensor: burn::tensor::Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let dims = tensor.shape().dims;
        if dims.as_slice() != [3] {
            return Err(TensorConversionError {
                message: format!("expected shape [3], got {dims:?}"),
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Unit tests for [`Acrobot`] covering observation shape, action indexing,
    //! episode lifecycle, physics invariants (velocity clamping, termination),
    //! determinism, and dynamics variant divergence.

    use super::*;
    use rlevo_core::environment::Snapshot;

    type DefaultAcrobot = Acrobot<BookDynamics>;

    fn default_env() -> DefaultAcrobot {
        DefaultAcrobot::with_config(AcrobotConfig::default())
    }

    #[test]
    fn observation_shape() {
        assert_eq!(AcrobotObservation::shape(), [6]);
    }

    #[test]
    fn action_count() {
        assert_eq!(AcrobotAction::ACTION_COUNT, 3);
        assert_eq!(AcrobotAction::from_index(0), AcrobotAction::TorqueNeg);
        assert_eq!(AcrobotAction::from_index(2), AcrobotAction::TorquePos);
    }

    #[test]
    fn reset_returns_running() {
        use rlevo_core::environment::EpisodeStatus;

        let mut env = default_env();
        let snap = env.reset().unwrap();
        assert_eq!(snap.status(), EpisodeStatus::Running);
    }

    #[test]
    fn velocity_clamp_applied() {
        let cfg = AcrobotConfig::default();
        let over_state = AcrobotState {
            theta1: 0.0,
            theta2: 0.0,
            theta1_dot: 100.0,
            theta2_dot: -100.0,
        };
        let clamped = DefaultAcrobot::clamp_velocities(over_state, &cfg);
        assert!(clamped.theta1_dot <= cfg.max_vel_1);
        assert!(clamped.theta2_dot >= -cfg.max_vel_2);
    }

    #[test]
    fn termination_condition_at_upright() {
        let cfg = AcrobotConfig::default();
        // θ1 = π (link 1 pointing straight up), θ2 = 0
        // height = -cos(π)*1 - cos(π+0)*1 = 1 + 1 = 2 > 1
        let state = AcrobotState {
            theta1: std::f32::consts::PI,
            theta2: 0.0,
            theta1_dot: 0.0,
            theta2_dot: 0.0,
        };
        assert!(DefaultAcrobot::is_terminal(&state, &cfg));
    }

    #[test]
    fn no_termination_at_rest() {
        let cfg = AcrobotConfig::default();
        let state = AcrobotState {
            theta1: 0.0,
            theta2: 0.0,
            theta1_dot: 0.0,
            theta2_dot: 0.0,
        };
        assert!(!DefaultAcrobot::is_terminal(&state, &cfg));
    }

    #[test]
    fn determinism() {
        let mut a = DefaultAcrobot::with_config(AcrobotConfig {
            seed: 5,
            ..Default::default()
        });
        let mut b = DefaultAcrobot::with_config(AcrobotConfig {
            seed: 5,
            ..Default::default()
        });
        a.reset().unwrap();
        b.reset().unwrap();
        for action in [
            AcrobotAction::TorquePos,
            AcrobotAction::TorqueNeg,
            AcrobotAction::TorqueZero,
        ] {
            let sa = a.step(action).unwrap();
            let sb = b.step(action).unwrap();
            assert_eq!(sa.observation().to_array(), sb.observation().to_array());
        }
    }

    #[test]
    fn book_and_nips_produce_different_trajectories() {
        let cfg = AcrobotConfig::default();
        let mut book = Acrobot::<BookDynamics>::with_config(cfg.clone());
        let mut nips = Acrobot::<NipsDynamics>::with_config(cfg);
        book.reset().unwrap();
        nips.reset().unwrap();
        // Force same initial state
        let init = AcrobotState {
            theta1: 0.1,
            theta2: 0.1,
            theta1_dot: 0.1,
            theta2_dot: 0.1,
        };
        book.state = init;
        nips.state = init;

        let mut any_diff = false;
        for _ in 0..20 {
            let sb = book.step(AcrobotAction::TorquePos).unwrap();
            let sn = nips.step(AcrobotAction::TorquePos).unwrap();
            if sb.observation().to_array() != sn.observation().to_array() {
                any_diff = true;
                break;
            }
        }
        assert!(
            any_diff,
            "Book and NIPS dynamics produced identical trajectories"
        );
    }

    #[test]
    fn config_builder() {
        let cfg = AcrobotConfig::builder()
            .gravity(9.81)
            .dt(0.1)
            .seed(42)
            .build();
        assert!((cfg.gravity - 9.81).abs() < 1e-5);
        assert!((cfg.dt - 0.1).abs() < 1e-5);
        assert_eq!(cfg.seed, 42);
    }

    #[test]
    fn reward_minus_one_until_terminal() {
        let mut env = default_env();
        env.reset().unwrap();
        let snap = env.step(AcrobotAction::TorquePos).unwrap();
        if !snap.is_terminated() {
            assert_eq!(*snap.reward(), ScalarReward(-1.0));
        } else {
            assert_eq!(*snap.reward(), ScalarReward(0.0));
        }
    }
}
