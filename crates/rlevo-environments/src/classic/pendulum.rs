//! Pendulum-v1 environment.
//!
//! Swing an under-actuated pendulum to the upright position and keep it
//! there. The episode never terminates intrinsically — compose with
//! [`crate::wrappers::TimeLimit::new(env, 200)`] for the standard 200-step
//! limit. Equations of motion match the Gymnasium `Pendulum-v1` reference
//! implementation exactly.
//!
//! ## Physical model
//!
//! A rigid pendulum of mass `$m$` ([`PendulumConfig::m`]) and length `$l$`
//! ([`PendulumConfig::l`]) swings about a fixed frictionless pivot under gravity
//! `$g$` ([`PendulumConfig::g`]). The angle `$\theta$` is measured from the
//! **upward** vertical, so `$\theta = 0$` is upright (the unstable equilibrium
//! to balance at) and `$\theta = \pm\pi$` hangs straight down. A continuous
//! torque `$u \in [-u_\text{max}, u_\text{max}]$` ([`PendulumConfig::max_torque`])
//! is applied at the pivot. The state is the 2-vector
//!
//! ```math
//! \mathbf{s} = \left(\theta,\; \frac{d\theta}{dt}\right).
//! ```
//!
//! ## Equations of motion
//!
//! Modelling the pendulum as a point mass on a massless rod, the angular
//! acceleration is
//!
//! ```math
//! \frac{d^2\theta}{dt^2} = \frac{3 g}{2 l}\sin\theta + \frac{3}{m l^2}\,u,
//! ```
//!
//! where `$u$` is the applied torque after clipping to
//! `$[-u_\text{max}, u_\text{max}]$`. The `$\tfrac{3}{m l^2}$` factor is the
//! reciprocal of the rod's moment of inertia about the pivot,
//! `$I = \tfrac{1}{3} m l^2$`, and the gravitational term carries the matching
//! `$\tfrac{3}{2}$`. This is evaluated each step in `Pendulum::step_dynamics`.
//!
//! ## Discrete-time integration
//!
//! The dynamics are advanced by one fixed step `$\Delta t$`
//! ([`PendulumConfig::dt`]) with **semi-implicit** (symplectic) Euler — the
//! angular velocity is updated first and then drives the angle update:
//!
//! ```math
//! \begin{aligned}
//! \frac{d\theta}{dt}\bigg|_{t+1}
//!   &= \operatorname{clip}\!\left(
//!        \frac{d\theta}{dt}\bigg|_t + \Delta t\,\frac{d^2\theta}{dt^2}\bigg|_t,\;
//!        -\omega_\text{max},\; \omega_\text{max}\right), \\[4pt]
//! \theta_{t+1} &= \theta_t + \Delta t\,\frac{d\theta}{dt}\bigg|_{t+1},
//! \end{aligned}
//! ```
//!
//! with the angular velocity clamped to `$\omega_\text{max}$`
//! ([`PendulumConfig::max_speed`]). The stored angle is wrapped to `$(-\pi, \pi]$`
//! via [`angle_normalize`]; this is observationally inert because the
//! observation `$(\cos\theta, \sin\theta, \tfrac{d\theta}{dt})$` and the cost
//! below depend on `$\theta$` only through periodic functions.
//!
//! ## Reward
//!
//! Each step yields `$\text{reward} = -\text{cost}$` with
//!
//! ```math
//! \text{cost} = \big(\operatorname{wrap}(\theta)\big)^2
//!   + 0.1\left(\frac{d\theta}{dt}\right)^{\!2} + 0.001\, u^2,
//! ```
//!
//! where `$\operatorname{wrap}(\theta) \in (-\pi, \pi]$` is [`angle_normalize`].
//! The cost penalises deviation from upright, angular speed, and torque effort.
//! The minimum per-step reward is `$-(\pi^2 + 0.1\,\omega_\text{max}^2 + 0.001\,u_\text{max}^2) \approx -16.27$`
//! (hanging down, max speed, max torque); the maximum is `$0$` (perfectly
//! upright, stationary, no torque).
//!
//! ## References
//!
//! - Gymnasium `Pendulum-v1` — source of the `$\tfrac{3}{ml^2}$` / `$\tfrac{3g}{2l}$`
//!   point-mass-on-rod formulation and the quadratic cost.
//!
//! ## Quick start
//!
//! ```no_run,ignore
//! use rlevo_environments::classic::{Pendulum, PendulumConfig, PendulumAction};
//! use rlevo_environments::wrappers::TimeLimit;
//! use rlevo_core::environment::{ConstructableEnv, Environment};
//!
//! let env = Pendulum::with_config(PendulumConfig::default()).expect("valid config");
//! let mut timed = TimeLimit::new(env, 200);
//! let mut snap = timed.reset().unwrap();
//! while !snap.is_done() {
//!     snap = timed.step(PendulumAction::new(1.0).unwrap()).unwrap();
//! }
//! ```
use std::fmt;

use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Uniform};
use rlevo_core::{
    action::{BoundedAction, ContinuousAction, InvalidActionError},
    base::{Action, Observation, State, TensorConversionError, TensorConvertible},
    config::{self, ConfigError, Validate},
    environment::{ConstructableEnv, Environment, EnvironmentError, SnapshotBase},
    reward::ScalarReward,
};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for [`Pendulum`].
///
/// All defaults match Gymnasium `Pendulum-v1`. Override individual fields with
/// struct update syntax or construct the whole struct directly.
///
/// # Examples
///
/// ```
/// use rlevo_environments::classic::pendulum::PendulumConfig;
///
/// let cfg = PendulumConfig { g: 9.81, seed: 42, ..PendulumConfig::default() };
/// assert!((cfg.g - 9.81).abs() < 1e-5);
/// assert_eq!(cfg.seed, 42);
/// ```
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

impl Validate for PendulumConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "PendulumConfig";
        config::positive(C, "max_speed", f64::from(self.max_speed))?;
        config::positive(C, "max_torque", f64::from(self.max_torque))?;
        config::positive(C, "dt", f64::from(self.dt))?;
        config::positive(C, "m", f64::from(self.m))?;
        config::positive(C, "l", f64::from(self.l))?;
        Ok(())
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
    pub fn torque(&self) -> f32 {
        self.0
    }

    fn unchecked(v: f32) -> Self {
        Self(v)
    }
}

impl Action<1> for PendulumAction {
    fn shape() -> [usize; 1] {
        [1]
    }

    fn is_valid(&self) -> bool {
        self.0.is_finite() && self.0.abs() <= 2.0
    }
}

impl ContinuousAction<1> for PendulumAction {
    fn as_slice(&self) -> &[f32] {
        std::slice::from_ref(&self.0)
    }

    fn clip(&self, min: f32, max: f32) -> Self {
        Self::unchecked(self.0.clamp(min, max))
    }

    /// Constructs a [`PendulumAction`] from a one-element slice.
    ///
    /// # Panics
    ///
    /// Panics if `values.len() != 1`.
    fn from_slice(values: &[f32]) -> Self {
        assert_eq!(values.len(), 1, "PendulumAction expects a 1-element slice");
        Self::unchecked(values[0])
    }

    fn random() -> Self
    where
        Self: Sized,
    {
        Self::unchecked(0.0)
    }
}

impl BoundedAction<1> for PendulumAction {
    fn low() -> [f32; 1] {
        [-2.0]
    }

    fn high() -> [f32; 1] {
        [2.0]
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

    fn shape() -> [usize; 1] {
        [2]
    }
    fn numel(&self) -> usize {
        2
    }
    fn is_valid(&self) -> bool {
        self.theta.is_finite() && self.theta_dot.is_finite()
    }

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
    /// Cosine of the pole angle.
    pub cos_theta: f32,
    /// Sine of the pole angle.
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
    fn shape() -> [usize; 1] {
        [3]
    }
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
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `config` fails [`Validate`] (e.g.
    /// non-positive `max_speed`, `max_torque`, `dt`, mass, or length).
    pub fn with_config(config: PendulumConfig) -> Result<Self, ConfigError> {
        config.validate()?;
        let rng = StdRng::seed_from_u64(config.seed);
        Ok(Self {
            state: PendulumState {
                theta: 0.0,
                theta_dot: 0.0,
            },
            config,
            rng,
            steps: 0,
        })
    }

    /// Re-seed the persistent RNG to `seed`, then [`reset`](Environment::reset).
    ///
    /// Ordinary [`reset`](Environment::reset) advances the persistent stream so
    /// successive episodes differ; use this when you need a *specific* episode
    /// to reproduce bit-for-bit (e.g. replaying a failure). Run-level
    /// reproducibility is already guaranteed by the construction seed.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`reset`](Environment::reset) (currently none).
    pub fn reset_with_seed(
        &mut self,
        seed: u64,
    ) -> Result<SnapshotBase<1, PendulumObservation, ScalarReward>, EnvironmentError> {
        self.rng = StdRng::seed_from_u64(seed);
        self.reset()
    }

    fn sample_init_state(&mut self) -> PendulumState {
        let theta = Uniform::new_inclusive(-std::f32::consts::PI, std::f32::consts::PI)
            .unwrap()
            .sample(&mut self.rng);
        let theta_dot = Uniform::new_inclusive(-1.0_f32, 1.0_f32)
            .unwrap()
            .sample(&mut self.rng);
        PendulumState { theta, theta_dot }
    }

    fn step_dynamics(
        state: PendulumState,
        torque: f32,
        cfg: &PendulumConfig,
    ) -> (PendulumState, f32) {
        let u = torque.clamp(-cfg.max_torque, cfg.max_torque);
        let theta_ddot =
            (3.0 * cfg.g / (2.0 * cfg.l)) * state.theta.sin() + (3.0 / (cfg.m * cfg.l * cfg.l)) * u;

        let new_theta_dot =
            (state.theta_dot + theta_ddot * cfg.dt).clamp(-cfg.max_speed, cfg.max_speed);
        let new_theta = angle_normalize(state.theta + new_theta_dot * cfg.dt);

        let cost = angle_normalize(state.theta).powi(2)
            + 0.1 * state.theta_dot.powi(2)
            + 0.001 * u.powi(2);
        let reward = -cost;

        (
            PendulumState {
                theta: new_theta,
                theta_dot: new_theta_dot,
            },
            reward,
        )
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

impl ConstructableEnv for Pendulum {
    fn new(render: bool) -> Self {
        let _ = render;
        Self::with_config(PendulumConfig::default()).expect("default config must validate")
    }
}

impl Environment<1, 1, 1> for Pendulum {
    type StateType = PendulumState;
    type ObservationType = PendulumObservation;
    type ActionType = PendulumAction;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, PendulumObservation, ScalarReward>;

    /// Resets the environment to a random initial state and returns the first snapshot.
    ///
    /// The initial angle and angular velocity are drawn from the environment's
    /// persistent RNG. The stream **advances** across resets, so successive
    /// episodes see independent initial states. For deterministic replay of a
    /// specific initial state, use [`Pendulum::reset_with_seed`].
    ///
    /// # Errors
    ///
    /// Currently infallible; always returns `Ok`.
    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.state = self.sample_init_state();
        self.steps = 0;
        Ok(SnapshotBase::running(
            self.state.observe(),
            ScalarReward(0.0),
        ))
    }

    /// Advances the simulation by one time step and returns the resulting snapshot.
    ///
    /// Clips the torque to `[-max_torque, max_torque]`, integrates the equations
    /// of motion with semi-implicit (symplectic) Euler, and clamps angular velocity to
    /// `[-max_speed, max_speed]`. The episode never terminates; the snapshot is
    /// always `Running`.
    ///
    /// # Errors
    ///
    /// Currently infallible; always returns `Ok`.
    fn step(&mut self, action: PendulumAction) -> Result<Self::SnapshotType, EnvironmentError> {
        let (next_state, reward_f) = Self::step_dynamics(self.state, action.torque(), &self.config);
        self.state = next_state;
        self.steps += 1;
        Ok(SnapshotBase::running(
            self.state.observe(),
            ScalarReward(reward_f),
        ))
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

    fn render_styled(&self) -> crate::render::StyledFrame {
        let line = self.render_ascii();
        crate::render::StyledFrame {
            lines: vec![style_pendulum_line(&line)],
        }
    }
}

/// Style one `render_ascii` line for [`Pendulum`].
///
/// The leading "Pendulum" label is treated as the agent (no bob glyph is
/// rendered today) and carries [`AGENT_FG`] with [`AGENT_MODIFIER`]. The
/// numeric annotations remain unstyled.
fn style_pendulum_line(line: &str) -> crate::render::StyledLine {
    use crate::render::palette::{AGENT_FG, AGENT_MODIFIER};
    use crate::render::{SpanStyle, StyledLine, StyledSpan};

    const LABEL: &str = "Pendulum";
    let agent_style = SpanStyle::default()
        .fg(AGENT_FG)
        .with_modifier(AGENT_MODIFIER);

    if let Some(rest) = line.strip_prefix(LABEL) {
        StyledLine::from_spans(vec![
            StyledSpan::new(LABEL, agent_style),
            StyledSpan::raw(rest.to_string()),
        ])
    } else {
        StyledLine::unstyled(line)
    }
}

// ---------------------------------------------------------------------------
// TensorConvertible
// ---------------------------------------------------------------------------

impl<B: burn::tensor::backend::Backend> TensorConvertible<1, B> for PendulumObservation {
    fn row_shape() -> [usize; 1] {
        [3]
    }

    fn write_host_row(&self, buf: &mut Vec<f32>) {
        buf.extend_from_slice(&self.to_array());
    }

    fn from_tensor(tensor: burn::tensor::Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let dims = tensor.dims();
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
        Ok(Self {
            cos_theta: v[0],
            sin_theta: v[1],
            theta_dot: v[2],
        })
    }
}

impl<B: burn::tensor::backend::Backend> TensorConvertible<1, B> for PendulumAction {
    fn row_shape() -> [usize; 1] {
        [1]
    }

    fn write_host_row(&self, buf: &mut Vec<f32>) {
        buf.push(self.0);
    }

    fn from_tensor(tensor: burn::tensor::Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let dims = tensor.dims();
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

impl rlevo_core::render::payload::Classic2DPayloadSource for Pendulum {
    fn classic2d_snapshot(&self) -> rlevo_core::render::payload::Classic2DSnapshot {
        use rlevo_core::render::payload::{
            Classic2DBody, Classic2DRole, Classic2DSnapshot, Point2,
        };
        let l = self.config.l;
        let theta = self.state.theta; // 0 = upright, measured from up vertical
        // Tip: upright (+y) at theta=0; +theta rotates clockwise.
        let tip = Point2::new(l * theta.sin(), l * theta.cos());
        let m = l + 0.2;
        Classic2DSnapshot {
            bodies: vec![
                Classic2DBody {
                    points: vec![Point2::new(0.0, 0.0), tip],
                    role: Classic2DRole::Pole,
                    closed: false,
                },
                Classic2DBody {
                    points: vec![Point2::new(0.0, 0.0)],
                    role: Classic2DRole::Hinge,
                    closed: false,
                },
            ],
            bounds: (Point2::new(-m, -m), Point2::new(m, m)),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Unit tests for [`Pendulum`] covering observation shape, action validation,
    //! reward at boundary states, `angle_normalize` correctness, non-termination,
    //! and determinism.

    use super::*;
    use rlevo_core::environment::Snapshot;

    fn default_env() -> Pendulum {
        Pendulum::with_config(PendulumConfig::default()).expect("valid config")
    }

    #[test]
    fn default_config_validates() {
        assert!(PendulumConfig::default().validate().is_ok());
    }

    #[test]
    fn rejects_non_positive_dt() {
        let bad = PendulumConfig {
            dt: 0.0,
            ..Default::default()
        };
        assert!(Pendulum::with_config(bad).is_err());
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
        let state = PendulumState {
            theta: 0.0,
            theta_dot: 0.0,
        };
        let cfg = PendulumConfig::default();
        let (_, reward) = Pendulum::step_dynamics(state, 0.0, &cfg);
        assert!((reward - 0.0).abs() < 1e-6, "reward={reward}");
    }

    #[test]
    fn worst_case_reward_approx_minus_16() {
        let cfg = PendulumConfig::default();
        let state = PendulumState {
            theta: std::f32::consts::PI,
            theta_dot: cfg.max_speed,
        };
        let (_, reward) = Pendulum::step_dynamics(state, cfg.max_torque, &cfg);
        assert!(reward < -15.0, "expected ≈ -16.27, got {reward}");
    }

    #[test]
    fn angle_normalize_examples() {
        let pi = std::f32::consts::PI;
        // 3π and -3π are both ≡ π (mod 2π). The formula maps them to -π (same angle).
        let n3pi = angle_normalize(3.0 * pi);
        assert!(
            n3pi.abs() - pi < 1e-4,
            "angle_normalize(3π)={n3pi} should be ±π"
        );
        let nm3pi = angle_normalize(-3.0 * pi);
        assert!(
            nm3pi.abs() - pi < 1e-4,
            "angle_normalize(-3π)={nm3pi} should be ±π"
        );
        // 2π → 0; -2π → 0
        assert!(angle_normalize(2.0 * pi).abs() < 1e-4);
        assert!(angle_normalize(0.0).abs() < 1e-4);
        // Result always in (-π, π]
        for x in [-5.0_f32, -pi, -0.5, 0.0, 0.5, pi, 5.0, 7.0] {
            let n = angle_normalize(x);
            assert!(
                n >= -pi - 1e-4 && n <= pi + 1e-4,
                "normalize({x}) = {n} out of range"
            );
        }
    }

    #[test]
    fn never_terminates() {
        use rlevo_core::environment::EpisodeStatus;

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
        let mut a = Pendulum::with_config(PendulumConfig {
            seed: 11,
            ..Default::default()
        })
        .expect("valid config");
        let mut b = Pendulum::with_config(PendulumConfig {
            seed: 11,
            ..Default::default()
        })
        .expect("valid config");
        a.reset().unwrap();
        b.reset().unwrap();
        let action = PendulumAction::unchecked(0.5);
        for _ in 0..10 {
            let sa = a.step(action).unwrap();
            let sb = b.step(action).unwrap();
            assert_eq!(sa.observation().to_array(), sb.observation().to_array());
        }
    }

    #[test]
    fn render_styled_matches_ascii() {
        use crate::render::AsciiRenderable;

        let env = Pendulum::new(false);
        let plain = env.render_ascii();
        let styled = env.render_styled();
        assert_eq!(styled.lines.len(), 1);
        assert_eq!(styled.plain_text(), plain);
    }

    #[test]
    fn render_styled_uses_palette_consts() {
        use crate::render::AsciiRenderable;
        use crate::render::palette::{AGENT_FG, AGENT_MODIFIER};

        let env = Pendulum::new(false);
        let styled = env.render_styled();
        let line = &styled.lines[0];
        let label = line
            .spans
            .iter()
            .find(|s| s.text == "Pendulum")
            .expect("Pendulum label span present");
        assert_eq!(label.style.fg, Some(AGENT_FG));
        assert!(label.style.modifier.contains(AGENT_MODIFIER));
    }

    #[test]
    fn render_ascii_within_width_budget() {
        use crate::render::AsciiRenderable;

        let env = Pendulum::new(false);
        for line in env.render_ascii().lines() {
            assert!(
                line.chars().count() <= 80,
                "line exceeds 80 cols: {line:?} ({} chars)",
                line.chars().count()
            );
        }
    }

    #[test]
    fn two_successive_resets_differ() {
        // The persistent RNG advances across resets (reset draws the initial
        // angle and angular velocity), so back-to-back resets must sample
        // independent initial states.
        let mut env = default_env();
        let first = env.reset().unwrap().observation().to_array();
        let second = env.reset().unwrap().observation().to_array();
        assert_ne!(
            first, second,
            "successive resets must draw independent initial states"
        );
    }

    #[test]
    fn reset_with_seed_is_reproducible() {
        let mut env = default_env();
        let a = env.reset_with_seed(999).unwrap().observation().to_array();
        // Advance the stream with an ordinary reset, then re-seed identically.
        env.reset().unwrap();
        let b = env.reset_with_seed(999).unwrap().observation().to_array();
        assert_eq!(
            a, b,
            "reset_with_seed must reproduce the same initial state"
        );
    }
}
