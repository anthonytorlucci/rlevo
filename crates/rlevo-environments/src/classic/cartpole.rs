//! CartPole-v1 environment.
//!
//! Balance a pole attached to a cart by applying left or right forces.
//! Physics from Barto, Sutton, and Anderson (1983); equations of motion
//! match the Gymnasium `CartPole-v1` reference implementation exactly.
//!
//! ## Physical model
//!
//! A cart of mass `$m_c$` slides on a frictionless horizontal track. A rigid,
//! uniform pole of mass `$m_p$` and **full** length `$2\ell$` (so `$\ell$` is the
//! half-length, [`CartPoleConfig::length`]) is hinged to the cart by a
//! frictionless pivot. A horizontal force `$F = \pm F_\text{mag}$` is applied to
//! the cart each step. The state is the 4-vector
//!
//! ```math
//! \mathbf{s} = \left(x,\; \frac{dx}{dt},\; \theta,\; \frac{d\theta}{dt}\right),
//! ```
//!
//! where `$x$` is the cart position and `$\theta$` is the pole angle (here,
//! positive `$\theta$` leans the pole clockwise; `$\theta = 0$` is upright).
//!
//! ## Equations of motion
//!
//! The continuous-time dynamics follow from the Euler–Lagrange equations for the
//! cart–pole. Because the pole is a uniform rod, its moment of inertia about its
//! centre of mass is `$I = \tfrac{1}{12} m_p (2\ell)^2 = \tfrac{1}{3} m_p \ell^2$`;
//! this contributes the `$1 + I/(m_p \ell^2) = \tfrac{4}{3}$` term in the angular
//! denominator below. Writing the total mass as `$m = m_c + m_p$`, the angular
//! and linear accelerations are
//!
//! ```math
//! \frac{d^2\theta}{dt^2} =
//!   \frac{g \sin\theta \;-\; \cos\theta\,\tau}
//!        {\ell \left( \dfrac{4}{3} - \dfrac{m_p \cos^2\theta}{m} \right)},
//! \qquad
//! \frac{d^2 x}{dt^2} = \tau - \frac{m_p \ell}{m}\,\frac{d^2\theta}{dt^2}\,\cos\theta,
//! ```
//!
//! where the shared intermediate term `$\tau$` (the code's `temp`) collects the
//! applied force and the centripetal reaction of the swinging pole:
//!
//! ```math
//! \tau = \frac{F + m_p \ell \left(\dfrac{d\theta}{dt}\right)^{\!2} \sin\theta}{m}.
//! ```
//!
//! Substituting `$\tau$` back gives the equivalent self-contained forms found in
//! Barto, Sutton & Anderson (1983) and Florian (2007):
//!
//! ```math
//! \frac{d^2\theta}{dt^2} =
//!   \frac{g \sin\theta + \cos\theta
//!         \left( \dfrac{-F - m_p \ell \left(\frac{d\theta}{dt}\right)^2 \sin\theta}{m} \right)}
//!        {\ell \left( \dfrac{4}{3} - \dfrac{m_p \cos^2\theta}{m} \right)},
//! \qquad
//! \frac{d^2 x}{dt^2} =
//!   \frac{F + m_p \ell \left[ \left(\dfrac{d\theta}{dt}\right)^2 \sin\theta
//!         - \dfrac{d^2\theta}{dt^2}\cos\theta \right]}{m}.
//! ```
//!
//! These are evaluated each step in `CartPole::step_physics`.
//!
//! ## Discrete-time integration
//!
//! The continuous ODE is advanced with a fixed step `$\Delta t$`
//! ([`CartPoleConfig::tau`], not to be confused with the term `$\tau$` above).
//! [`Integrator::Euler`] uses the explicit forward (positions and velocities both
//! advanced from the *old* velocities/accelerations):
//!
//! ```math
//! \begin{aligned}
//! x_{t+1}            &= x_t + \Delta t\,\frac{dx}{dt}\bigg|_t, &
//! \frac{dx}{dt}\bigg|_{t+1} &= \frac{dx}{dt}\bigg|_t + \Delta t\,\frac{d^2x}{dt^2}\bigg|_t, \\[4pt]
//! \theta_{t+1}       &= \theta_t + \Delta t\,\frac{d\theta}{dt}\bigg|_t, &
//! \frac{d\theta}{dt}\bigg|_{t+1} &= \frac{d\theta}{dt}\bigg|_t + \Delta t\,\frac{d^2\theta}{dt^2}\bigg|_t.
//! \end{aligned}
//! ```
//!
//! [`Integrator::SemiImplicit`] (symplectic Euler) updates the velocities first,
//! then advances the positions with the *new* velocities:
//!
//! ```math
//! \begin{aligned}
//! \frac{dx}{dt}\bigg|_{t+1} &= \frac{dx}{dt}\bigg|_t + \Delta t\,\frac{d^2x}{dt^2}\bigg|_t, &
//! x_{t+1} &= x_t + \Delta t\,\frac{dx}{dt}\bigg|_{t+1}, \\[4pt]
//! \frac{d\theta}{dt}\bigg|_{t+1} &= \frac{d\theta}{dt}\bigg|_t + \Delta t\,\frac{d^2\theta}{dt^2}\bigg|_t, &
//! \theta_{t+1} &= \theta_t + \Delta t\,\frac{d\theta}{dt}\bigg|_{t+1}.
//! \end{aligned}
//! ```
//!
//! The accelerations `$\tfrac{d^2 x}{dt^2}$` and `$\tfrac{d^2\theta}{dt^2}$` are
//! computed once from the state at time `$t$` and reused by both schemes; the
//! variants differ only in whether the position update reads the old or new
//! velocity.
//!
//! ## References
//!
//! - A. G. Barto, R. S. Sutton, C. W. Anderson, "Neuronlike adaptive elements
//!   that can solve difficult learning control problems," *IEEE Trans. SMC*,
//!   1983.
//! - R. V. Florian, "Correct equations of motion for the cart-pole system,"
//!   tech. report, 2007 — source of the `$\tfrac{4}{3}$` rod-inertia term.
//!
//! ## Integrators
//!
//! Two integration schemes are available via [`Integrator`]:
//!
//! | Variant | Description |
//! |---------|-------------|
//! | [`Integrator::Euler`] | Forward Euler — Gymnasium default |
//! | [`Integrator::SemiImplicit`] | Semi-implicit Euler: velocities updated before positions |
//!
//! ## Step limit
//!
//! There is no intrinsic step cap. Compose with [`crate::wrappers::TimeLimit`]
//! to replicate the Gymnasium v1 500-step episode limit.
//!
//! ## Quick start
//!
//! ```no_run,ignore
//! use rlevo_environments::classic::{CartPole, CartPoleConfig, CartPoleAction};
//! use rlevo_environments::wrappers::TimeLimit;
//! use rlevo_core::environment::{ConstructableEnv, Environment};
//!
//! let env = CartPole::with_config(CartPoleConfig::default()).expect("valid config");
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
    config::{self, ConfigError, Validate},
    environment::{ConstructableEnv, Environment, EnvironmentError, SnapshotBase},
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
/// All defaults match `CartPole-v1` in Gymnasium. Use [`CartPoleConfig::builder()`]
/// for a fluent construction style.
///
/// # Examples
///
/// ```rust
/// use rlevo_environments::classic::cartpole::CartPoleConfig;
///
/// let cfg = CartPoleConfig::builder()
///     .gravity(9.81)
///     .masscart(2.0)
///     .seed(42)
///     .build();
/// assert!((cfg.gravity - 9.81).abs() < 1e-5);
/// ```
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

impl Validate for CartPoleConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "CartPoleConfig";
        config::positive(C, "masscart", f64::from(self.masscart))?;
        config::positive(C, "masspole", f64::from(self.masspole))?;
        config::positive(C, "length", f64::from(self.length))?;
        config::positive(C, "force_mag", f64::from(self.force_mag))?;
        config::positive(C, "tau", f64::from(self.tau))?;
        config::positive(
            C,
            "theta_threshold_radians",
            f64::from(self.theta_threshold_radians),
        )?;
        config::positive(C, "x_threshold", f64::from(self.x_threshold))?;
        Ok(())
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
    /// Sets gravitational acceleration (m/s²).
    pub fn gravity(mut self, v: f32) -> Self {
        self.inner.gravity = v;
        self
    }
    /// Sets the mass of the cart (kg).
    pub fn masscart(mut self, v: f32) -> Self {
        self.inner.masscart = v;
        self
    }
    /// Sets the mass of the pole (kg).
    pub fn masspole(mut self, v: f32) -> Self {
        self.inner.masspole = v;
        self
    }
    /// Sets half the pole length (m).
    pub fn length(mut self, v: f32) -> Self {
        self.inner.length = v;
        self
    }
    /// Sets the magnitude of the force applied to the cart (N).
    pub fn force_mag(mut self, v: f32) -> Self {
        self.inner.force_mag = v;
        self
    }
    /// Sets the time step between updates (s).
    pub fn tau(mut self, v: f32) -> Self {
        self.inner.tau = v;
        self
    }
    /// Sets the pole angle termination threshold (rad).
    pub fn theta_threshold_radians(mut self, v: f32) -> Self {
        self.inner.theta_threshold_radians = v;
        self
    }
    /// Sets the cart position termination threshold (m).
    pub fn x_threshold(mut self, v: f32) -> Self {
        self.inner.x_threshold = v;
        self
    }
    /// Sets the integration scheme.
    pub fn integrator(mut self, v: Integrator) -> Self {
        self.inner.integrator = v;
        self
    }
    /// Enables the Sutton-Barto reward schedule (`0` per step, `-1` on failure).
    pub fn sutton_barto_reward(mut self, v: bool) -> Self {
        self.inner.sutton_barto_reward = v;
        self
    }
    /// Sets the RNG seed used by [`CartPole::reset`].
    pub fn seed(mut self, v: u64) -> Self {
        self.inner.seed = v;
        self
    }

    /// Finalises and returns the config.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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

    /// Constructs an action from its integer index.
    ///
    /// # Panics
    ///
    /// Panics if `index` is not `0` (Left) or `1` (Right).
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
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `config` fails [`Validate`] (e.g.
    /// non-positive mass, length, `tau`, or termination thresholds).
    pub fn with_config(config: CartPoleConfig) -> Result<Self, ConfigError> {
        config.validate()?;
        let rng = StdRng::seed_from_u64(config.seed);
        Ok(Self {
            state: CartPoleState::new(0.0, 0.0, 0.0, 0.0),
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
    ) -> Result<SnapshotBase<1, CartPoleObservation, ScalarReward>, EnvironmentError> {
        self.rng = StdRng::seed_from_u64(seed);
        self.reset()
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

    /// Apply the equations of motion and return the next state.
    ///
    /// Implements the cart–pole dynamics documented at the [module
    /// level](crate::classic::cartpole). The local bindings map to the symbols
    /// there as follows:
    ///
    /// | Code            | Symbol                  | Meaning                                         |
    /// |-----------------|-------------------------|-------------------------------------------------|
    /// | `force`         | `$F$`                   | applied force, `$\pm F_\text{mag}$`             |
    /// | `total_mass`    | `$m = m_c + m_p$`       | cart + pole mass                                |
    /// | `pm_l`          | `$m_p \ell$`            | pole mass times half-length                     |
    /// | `temp`          | `$\tau$`                | `$(F + m_p \ell\,(d\theta/dt)^2\sin\theta)/m$` |
    /// | `theta_acc`     | `$d^2\theta/dt^2$`      | angular acceleration                            |
    /// | `x_acc`         | `$d^2x/dt^2$`           | cart linear acceleration                        |
    ///
    /// The `Euler` and `SemiImplicit` arms then apply the corresponding
    /// discretisation with step `$\Delta t$` (`cfg.tau`).
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

impl ConstructableEnv for CartPole {
    fn new(render: bool) -> Self {
        let _ = render;
        Self::with_config(CartPoleConfig::default()).expect("default config must validate")
    }
}

impl Environment<1, 1, 1> for CartPole {
    type StateType = CartPoleState;
    type ObservationType = CartPoleObservation;
    type ActionType = CartPoleAction;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, CartPoleObservation, ScalarReward>;

    /// Resets the environment to a random initial state and returns the first snapshot.
    ///
    /// The initial state is drawn from the environment's persistent RNG. The
    /// stream **advances** across resets, so successive episodes see
    /// independent initial states. For deterministic replay of a specific
    /// initial state, use [`CartPole::reset_with_seed`].
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
    /// Applies the selected integrator, then checks the termination conditions
    /// (pole angle and cart position thresholds). Reward is `+1.0` per step by
    /// default, or `0.0` / `-1.0` under the Sutton-Barto schedule.
    ///
    /// # Errors
    ///
    /// Currently infallible; always returns `Ok`.
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

/// Width of the rendered cart-pole canvas, in columns.
const RENDER_WIDTH: usize = 50;
/// Number of rows the pole occupies above the track.
const RENDER_POLE_H: usize = 6;
/// Fraction of the failure angle past which the pole is drawn as a hazard
/// (red + reversed) rather than balanced (green + bold).
const RENDER_DANGER_TIER: f32 = 0.66;

impl crate::render::AsciiRenderable for CartPole {
    fn render_ascii(&self) -> String {
        self.render_styled().plain_text()
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        crate::render::StyledFrame {
            lines: self.scene_lines(),
        }
    }
}

impl CartPole {
    /// Build the styled scene drawn by the live TUI and report tiers.
    ///
    /// The frame is, top to bottom: [`RENDER_POLE_H`] rows tracing the pole
    /// from tip down to the pivot, one track row carrying the cart glyph,
    /// and a status row with the angle, cart position, and step count.
    ///
    /// The pole's tilt is the primary, hue-independent signal: each row's
    /// horizontal offset and slope glyph (`/`, `|`, `\`) follow
    /// `theta / theta_threshold`, so the tip leans a full [`RENDER_POLE_H`]
    /// columns off vertical at the failure angle. Colour is a redundant
    /// overlay — green/bold while balanced, red/reversed once the pole is
    /// past [`RENDER_DANGER_TIER`] of the way to failing — satisfying the
    /// project's "never rely on hue alone" accessibility contract.
    fn scene_lines(&self) -> Vec<crate::render::StyledLine> {
        use crate::render::palette::{
            AGENT_FG, AGENT_MODIFIER, GOAL_FG, GOAL_MODIFIER, HAZARD_FG, HAZARD_MODIFIER, WALL_FG,
        };
        use crate::render::{SpanStyle, StyledLine, StyledSpan};

        let width = RENDER_WIDTH;
        let cart_frac = (self.state.x / self.config.x_threshold * 0.5 + 0.5).clamp(0.0, 1.0);
        let cart_col = (cart_frac * (width as f32 - 1.0)) as usize;

        let theta = self.state.theta;
        let threshold = self.config.theta_threshold_radians.max(f32::EPSILON);
        let danger = (theta.abs() / threshold).clamp(0.0, 1.0);
        // Horizontal tip offset in columns: a full pole-height lean at the
        // failure angle makes the diagonal unmistakable.
        let tip_frac = (theta / threshold).clamp(-1.0, 1.0);
        let max_tip = RENDER_POLE_H as f32;

        let pole_style = if danger >= RENDER_DANGER_TIER {
            SpanStyle::default()
                .fg(HAZARD_FG)
                .with_modifier(HAZARD_MODIFIER)
        } else {
            SpanStyle::default()
                .fg(GOAL_FG)
                .with_modifier(GOAL_MODIFIER)
        };
        // Positive theta is clockwise (top leans right) → '/'.
        let glyph = if tip_frac > 0.08 {
            '/'
        } else if tip_frac < -0.08 {
            '\\'
        } else {
            '|'
        };

        let wall_style = SpanStyle::default().fg(WALL_FG);
        let agent_style = SpanStyle::default()
            .fg(AGENT_FG)
            .with_modifier(AGENT_MODIFIER);

        let mut lines = Vec::with_capacity(RENDER_POLE_H + 2);

        // Pole rows, tip first. Row `r` sits `RENDER_POLE_H - r` cells above
        // the pivot, so the offset shrinks to zero as we approach the cart.
        for r in 0..RENDER_POLE_H {
            let height_from_base = (RENDER_POLE_H - r) as f32;
            let frac = height_from_base / RENDER_POLE_H as f32;
            let offset = (frac * tip_frac * max_tip).round() as i32;
            let col = (cart_col as i32 + offset).clamp(0, width as i32 - 1) as usize;

            lines.push(StyledLine::from_spans([
                StyledSpan::raw(" ".repeat(col)),
                StyledSpan::new(glyph.to_string(), pole_style),
                StyledSpan::raw(" ".repeat(width - col - 1)),
            ]));
        }

        // Track row: dashes with the cart glyph sitting on it.
        lines.push(StyledLine::from_spans([
            StyledSpan::new("-".repeat(cart_col), wall_style),
            StyledSpan::new("#", agent_style),
            StyledSpan::new("-".repeat(width - cart_col - 1), wall_style),
        ]));

        // Status row — plain text, mirroring the old single-line summary.
        lines.push(StyledLine::unstyled(format!(
            "  θ={:+.1}°  x={:+.2}  step={}",
            theta.to_degrees(),
            self.state.x,
            self.steps
        )));

        lines
    }
}

// ---------------------------------------------------------------------------
// TensorConvertible
// ---------------------------------------------------------------------------

impl<B: burn::tensor::backend::Backend> TensorConvertible<1, B> for CartPoleObservation {
    fn row_shape() -> [usize; 1] {
        [4]
    }

    fn write_host_row(&self, buf: &mut Vec<f32>) {
        buf.extend_from_slice(&self.to_array());
    }

    fn from_tensor(tensor: burn::tensor::Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let dims = tensor.dims();
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
    fn row_shape() -> [usize; 1] {
        [2]
    }

    fn write_host_row(&self, buf: &mut Vec<f32>) {
        let mut one_hot = [0.0_f32; 2];
        one_hot[self.to_index()] = 1.0;
        buf.extend_from_slice(&one_hot);
    }

    fn from_tensor(tensor: burn::tensor::Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let dims = tensor.dims();
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
        let idx = crate::tensor_decode::argmax(&v);
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

impl rlevo_core::render::payload::Classic2DPayloadSource for CartPole {
    fn classic2d_snapshot(&self) -> rlevo_core::render::payload::Classic2DSnapshot {
        use rlevo_core::render::payload::{
            Classic2DBody, Classic2DRole, Classic2DSnapshot, Point2,
        };
        let x = self.state.x;
        let theta = self.state.theta; // 0 = upright, +clockwise
        let xt = self.config.x_threshold;
        let pole_len = 2.0 * self.config.length; // full rod length = 2·half-length
        let (cart_w, cart_h) = (0.4_f32, 0.25_f32);
        let hinge_y = cart_h; // hinge atop the cart
        // Cart rectangle centred at (x, cart_h/2).
        let cy = cart_h * 0.5;
        let cart = vec![
            Point2::new(x - cart_w * 0.5, cy - cart_h * 0.5),
            Point2::new(x + cart_w * 0.5, cy - cart_h * 0.5),
            Point2::new(x + cart_w * 0.5, cy + cart_h * 0.5),
            Point2::new(x - cart_w * 0.5, cy + cart_h * 0.5),
        ];
        let tip = Point2::new(x + pole_len * theta.sin(), hinge_y + pole_len * theta.cos());
        Classic2DSnapshot {
            bodies: vec![
                Classic2DBody {
                    points: vec![Point2::new(-xt, 0.0), Point2::new(xt, 0.0)],
                    role: Classic2DRole::Track,
                    closed: false,
                },
                Classic2DBody {
                    points: cart,
                    role: Classic2DRole::Cart,
                    closed: true,
                },
                Classic2DBody {
                    points: vec![Point2::new(x, hinge_y), tip],
                    role: Classic2DRole::Pole,
                    closed: false,
                },
            ],
            bounds: (
                Point2::new(-xt - 0.2, -0.4),
                Point2::new(xt + 0.2, pole_len + 0.4),
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Unit tests for [`CartPole`] covering observation shape, action indexing,
    //! episode lifecycle, reward schedules, determinism, and integrator divergence.

    use super::*;
    use rlevo_core::environment::Snapshot;

    fn default_env() -> CartPole {
        CartPole::with_config(CartPoleConfig::default()).expect("valid config")
    }

    #[test]
    fn default_config_validates() {
        assert!(CartPoleConfig::default().validate().is_ok());
    }

    #[test]
    fn rejects_non_positive_length() {
        let bad = CartPoleConfig {
            length: 0.0,
            ..Default::default()
        };
        assert!(CartPole::with_config(bad).is_err());
    }

    #[test]
    fn classic2d_snapshot_has_track_cart_pole_and_upright_pole_at_theta_zero() {
        use rlevo_core::render::payload::{Classic2DPayloadSource, Classic2DRole};

        let mut env = default_env();
        env.state.x = 0.0;
        env.state.theta = 0.0; // upright
        let snap = env.classic2d_snapshot();
        // Track, cart, pole.
        let roles: Vec<_> = snap.bodies.iter().map(|b| b.role).collect();
        assert!(roles.contains(&Classic2DRole::Track));
        assert!(roles.contains(&Classic2DRole::Cart));
        let pole = snap
            .bodies
            .iter()
            .find(|b| b.role == Classic2DRole::Pole)
            .expect("pole body present");
        // Upright pole rises straight up from the hinge: same x, higher y.
        assert_eq!(pole.points.len(), 2);
        let (base, tip) = (pole.points[0], pole.points[1]);
        assert!(
            (tip.x - base.x).abs() < 1e-5,
            "upright pole must be vertical"
        );
        assert!(
            tip.y > base.y,
            "pole tip must be above the hinge when upright"
        );
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
    fn action_from_tensor_is_nan_safe() {
        use burn::tensor::{Tensor, TensorData as TD};
        type TestBackend = burn::backend::Flex;
        let device = Default::default();
        // An all-NaN logit vector must not panic; falls back to index 0.
        let data = TD::new(vec![f32::NAN, f32::NAN], [2]);
        let tensor = Tensor::<TestBackend, 1>::from_data(data, &device);
        let back = <CartPoleAction as TensorConvertible<1, TestBackend>>::from_tensor(tensor)
            .expect("all-NaN decodes to fallback");
        assert_eq!(back, CartPoleAction::from_index(0));
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
        let mut env = CartPole::with_config(config).expect("valid config");
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
        let mut env = CartPole::with_config(config).expect("valid config");
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
        })
        .expect("valid config");
        let mut env_b = CartPole::with_config(CartPoleConfig {
            seed: 42,
            ..Default::default()
        })
        .expect("valid config");
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
        let mut euler_env = CartPole::with_config(euler_cfg).expect("valid config");
        let mut si_env = CartPole::with_config(si_cfg).expect("valid config");
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

    #[test]
    fn render_styled_matches_ascii() {
        use crate::render::AsciiRenderable;

        let mut env = CartPole::new(false);
        env.reset().unwrap();
        let plain = env.render_ascii();
        let styled = env.render_styled();
        // Pole rows + track row + status row.
        assert_eq!(styled.lines.len(), RENDER_POLE_H + 2);
        assert_eq!(styled.plain_text(), plain);
        assert!(
            plain.lines().last().unwrap().contains("step="),
            "status row missing: {plain:?}"
        );
    }

    #[test]
    fn render_styled_uses_palette_consts() {
        use crate::render::AsciiRenderable;
        use crate::render::palette::{AGENT_FG, AGENT_MODIFIER, GOAL_FG, WALL_FG};

        let mut env = CartPole::new(false);
        env.reset().unwrap();
        let styled = env.render_styled();
        let spans = || styled.lines.iter().flat_map(|l| l.spans.iter());

        // Cart glyph lives on the track row, carrying the agent style.
        let cart_span = spans()
            .find(|s| s.text == "#")
            .expect("cart glyph span present");
        assert_eq!(cart_span.style.fg, Some(AGENT_FG));
        assert!(cart_span.style.modifier.contains(AGENT_MODIFIER));

        // The track dashes carry the wall style.
        let track_span = spans()
            .find(|s| s.text.starts_with('-'))
            .expect("track dash span present");
        assert_eq!(track_span.style.fg, Some(WALL_FG));

        // At reset the pole is near-vertical, so it renders balanced (goal).
        let pole_span = spans()
            .find(|s| matches!(s.text.as_str(), "|" | "/" | "\\"))
            .expect("pole glyph span present");
        assert_eq!(pole_span.style.fg, Some(GOAL_FG));
    }

    #[test]
    fn render_ascii_within_width_budget() {
        use crate::render::AsciiRenderable;

        let mut env = CartPole::new(false);
        env.reset().unwrap();
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
        // The persistent RNG advances across resets (reset draws U(-0.05, 0.05)
        // on all four state variables), so back-to-back resets must sample
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
