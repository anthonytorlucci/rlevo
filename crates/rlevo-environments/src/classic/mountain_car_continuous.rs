//! MountainCarContinuous-v0 environment.
//!
//! Continuous-action variant of [`crate::classic::mountain_car::MountainCar`].
//! The agent applies a scalar force in `[-1, 1]` at each step; it must swing
//! left to build momentum before cresting the right hill. The valley/momentum
//! problem originates with Moore (1990) and Sutton & Barto §10.1, but both
//! describe only the classic discrete-action task; the continuous-force
//! control scheme is a later Gymnasium/OpenAI Gym addition, and it is that
//! Gymnasium `MountainCarContinuous-v0` reference implementation the update
//! rule below matches exactly.
//!
//! ## Physical model
//!
//! The car moves along a one-dimensional track whose height profile is the
//! sinusoidal hill `$y(x) = \sin(3x)$`, with the position `$x$` confined to
//! `$[x_{\min}, x_{\max}] = [-1.2,\, 0.6]$` (the position bounds
//! [`MountainCarContinuousConfig::pos_bounds`]). The goal flag is at
//! `$x_\text{goal} = 0.45$`. The state is the 2-vector
//!
//! ```math
//! \mathbf{s} = \left(x,\; \frac{dx}{dt}\right),
//! ```
//!
//! the car's position and velocity. Unlike the discrete variant, the agent
//! supplies a **continuous** force `$a \in [-1, 1]$` (the action bounds
//! [`MountainCarContinuousConfig::action_bounds`]) scaled by a power coefficient
//! `$P$` ([`MountainCarContinuousConfig::power`]).
//!
//! ## Equations of motion
//!
//! This is a discrete-time map (no sub-step integrator). Writing the velocity at
//! step `$t$` as `$v_t = \tfrac{dx}{dt}\big|_t$` and the gravity coefficient as
//! `$g = 0.0025$` (fixed, as in Gymnasium), each step updates velocity then
//! position:
//!
//! ```math
//! \begin{aligned}
//! v_{t+1} &= \operatorname{clip}\!\Big(
//!            v_t + P\,\operatorname{clip}(a, -1, 1) - g\cos(3 x_t),\;
//!            -v_{\max},\; v_{\max} \Big), \\[4pt]
//! x_{t+1} &= \operatorname{clip}\!\big( x_t + v_{t+1},\; x_{\min},\; x_{\max} \big).
//! \end{aligned}
//! ```
//!
//! The `$-g\cos(3 x_t)$` term is the gravitational pull along the track (the
//! hill slope is `$\tfrac{dy}{dx} = 3\cos(3x)$`, with the factor of `$3$`
//! absorbed into `$g$`). The speed is capped at `$v_{\max} = 0.07$`
//! ([`MountainCarContinuousConfig::max_speed`]). The left wall at `$x_{\min}$`
//! is fully inelastic: if the position update clamps against it, the velocity is
//! reset to `$0$`. These are evaluated each step in
//! `MountainCarContinuous::apply_physics`.
//!
//! ## Reward and termination
//!
//! An episode **terminates** when `$x \ge x_\text{goal}$` **and**
//! `$v \ge v_\text{goal}$` ([`MountainCarContinuousConfig::goal_velocity`],
//! default `$0$`). The reward shapes a quadratic control cost against a large
//! goal bonus:
//!
//! ```math
//! \text{reward} = -0.1\, a^2 + 100 \cdot \big[\, x \ge x_\text{goal} \;\wedge\; v \ge v_\text{goal} \,\big],
//! ```
//!
//! where `$[\cdot]$` is the Iverson bracket (`$1$` if the goal condition holds,
//! else `$0$`) and `$a$` is the **raw** action (squared *before* clipping, as in
//! Gymnasium). So a non-terminal step scores `$-0.1\,a^2 \le 0$` and the
//! terminal step adds `$+100$`. A zero-force policy scores `$0$` every step; the
//! agent must accept temporary control cost to earn the goal bonus. The two
//! components are also exposed by the keys [`REWARD_CTRL`] and [`REWARD_GOAL`].
//!
//! ## Step limit
//!
//! This environment has **no intrinsic episode limit**. The standard
//! Gymnasium cap of 999 steps should be added externally:
//!
//! ```no_run,ignore
//! use rlevo_environments::{classic::mountain_car_continuous::MountainCarContinuous, wrappers::TimeLimit};
//!
//! let env = TimeLimit::new(MountainCarContinuous::new(false), 999);
//! ```
//!
//! ## Quick start
//!
//! ```rust
//! use rlevo_core::environment::{ConstructableEnv, Environment};
//! use rlevo_environments::classic::mountain_car_continuous::{
//!     MountainCarContinuous, MountainCarContinuousAction,
//! };
//!
//! let mut env = MountainCarContinuous::new(false);
//! let _snap = env.reset().unwrap();
//! let action = MountainCarContinuousAction::new(0.5).unwrap();
//! let snap = env.step(action).unwrap();
//! println!("{snap:?}");
//! ```
use std::fmt;

use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Uniform};
use rlevo_core::{
    action::{BoundedAction, ContinuousAction, InvalidActionError},
    base::{Action, HostRow, Observation, State, TensorConversionError, TensorConvertible},
    bounds::Bounds,
    config::{self, ConfigError, Validate},
    environment::{ConstructableEnv, Environment, EnvironmentError, Sensor, SnapshotBase},
    reward::ScalarReward,
};
use serde::{Deserialize, Serialize};

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
///
/// All fields are public so they can be set with struct-update syntax.
/// Use `MountainCarContinuousConfig::default()` for Gymnasium-compatible
/// defaults.
///
/// # Examples
///
/// ```
/// use rlevo_environments::classic::mountain_car_continuous::MountainCarContinuousConfig;
///
/// let cfg = MountainCarContinuousConfig { seed: 42, ..MountainCarContinuousConfig::default() };
/// assert_eq!(cfg.seed, 42);
/// assert!((cfg.power - 0.0015).abs() < 1e-7);
/// ```
#[derive(Debug, Clone)]
pub struct MountainCarContinuousConfig {
    /// Inclusive valid action bounds `[min, max]`. Default: `[-1.0, 1.0]`.
    pub action_bounds: Bounds,
    /// Power multiplier applied to the clamped force. Default: `0.0015`.
    pub power: f32,
    /// Goal position (m). Default: `0.45`.
    pub goal_position: f32,
    /// Minimum velocity at goal for termination. Default: `0.0`.
    pub goal_velocity: f32,
    /// Inclusive position bounds `[min, max]` (m). Default: `[-1.2, 0.6]`.
    pub pos_bounds: Bounds,
    /// Maximum absolute velocity (m/s). Default: `0.07`.
    pub max_speed: f32,
    /// RNG seed; `reset()` re-seeds from this value. Default: `0`.
    pub seed: u64,
}

impl Default for MountainCarContinuousConfig {
    fn default() -> Self {
        Self {
            action_bounds: Bounds::new(-1.0, 1.0),
            power: 0.0015,
            goal_position: 0.45,
            goal_velocity: 0.0,
            pos_bounds: Bounds::new(-1.2, 0.6),
            max_speed: 0.07,
            seed: 0,
        }
    }
}

impl Validate for MountainCarContinuousConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "MountainCarContinuousConfig";
        config::in_range(
            C,
            "goal_position",
            f64::from(self.pos_bounds.lo()),
            f64::from(self.pos_bounds.hi()),
            f64::from(self.goal_position),
        )?;
        config::positive(C, "max_speed", f64::from(self.max_speed))?;
        config::positive(C, "power", f64::from(self.power))?;
        Ok(())
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
    /// Constructs an action, returning an error if `force` is out of range or non-finite.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidActionError`] if `force` is outside `[-1.0, 1.0]` or is
    /// `NaN` / infinite.
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
    const COMPONENTS: usize = 1;

    fn as_slice(&self) -> &[f32] {
        std::slice::from_ref(&self.0)
    }

    fn clip(&self, min: f32, max: f32) -> Self {
        Self::unchecked(self.0.clamp(min, max))
    }

    /// Constructs a [`MountainCarContinuousAction`] from a one-element slice.
    ///
    /// # Panics
    ///
    /// Panics if `values.len() != 1`.
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
    fn low() -> &'static [f32] {
        &[-1.0]
    }

    fn high() -> &'static [f32] {
        &[1.0]
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
    fn shape() -> [usize; 1] {
        [2]
    }
    fn numel(&self) -> usize {
        2
    }
    fn is_valid(&self) -> bool {
        self.position.is_finite() && self.velocity.is_finite()
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
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `config` fails [`Validate`] (e.g. a
    /// `goal_position` outside `pos_bounds`, or non-positive `max_speed` /
    /// `power`).
    pub fn with_config(config: MountainCarContinuousConfig) -> Result<Self, ConfigError> {
        config.validate()?;
        let rng = StdRng::seed_from_u64(config.seed);
        Ok(Self {
            state: MountainCarContinuousState {
                position: -0.5,
                velocity: 0.0,
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
    ) -> Result<SnapshotBase<1, MountainCarContinuousObservation, ScalarReward>, EnvironmentError>
    {
        self.rng = StdRng::seed_from_u64(seed);
        self.reset()
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
        let clamped = cfg.action_bounds.clamp(force);
        let mut vel = state.velocity + clamped * cfg.power - 0.0025 * (3.0 * state.position).cos();
        vel = vel.clamp(-cfg.max_speed, cfg.max_speed);
        let mut pos = state.position + vel;
        pos = cfg.pos_bounds.clamp(pos);
        if pos <= cfg.pos_bounds.lo() {
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

impl ConstructableEnv for MountainCarContinuous {
    fn new(render: bool) -> Self {
        let _ = render;
        Self::with_config(MountainCarContinuousConfig::default())
            .expect("default config must validate")
    }
}

impl Sensor<1, 1, 1> for MountainCarContinuous {
    type Action = MountainCarContinuousAction;
    type State = MountainCarContinuousState;
    type Observation = MountainCarContinuousObservation;

    /// Projects the resulting state directly onto the 2-D `[position, velocity]`
    /// observation; the observation ignores the applied force.
    fn observe(
        &self,
        _action: &MountainCarContinuousAction,
        next_state: &MountainCarContinuousState,
    ) -> MountainCarContinuousObservation {
        mountain_car_continuous_observation(next_state)
    }

    /// Projects the initial state onto the 2-D observation.
    fn observe_reset(
        &self,
        state: &MountainCarContinuousState,
    ) -> MountainCarContinuousObservation {
        mountain_car_continuous_observation(state)
    }
}

/// Builds the 2-D MountainCarContinuous observation `[position, velocity]` from a
/// state.
fn mountain_car_continuous_observation(
    state: &MountainCarContinuousState,
) -> MountainCarContinuousObservation {
    MountainCarContinuousObservation {
        position: state.position,
        velocity: state.velocity,
    }
}

impl Environment<1, 1, 1> for MountainCarContinuous {
    type StateType = MountainCarContinuousState;
    type ObservationType = MountainCarContinuousObservation;
    type ActionType = MountainCarContinuousAction;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, MountainCarContinuousObservation, ScalarReward>;

    /// Resets the environment to a random initial state and returns the first snapshot.
    ///
    /// The initial position in `[-0.6, -0.4]` is drawn from the environment's
    /// persistent RNG. The stream **advances** across resets, so successive
    /// episodes see independent initial positions. For deterministic replay of
    /// a specific initial state, use [`MountainCarContinuous::reset_with_seed`].
    ///
    /// # Errors
    ///
    /// Currently infallible; always returns `Ok`.
    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.state = self.sample_init_state();
        self.steps = 0;
        Ok(SnapshotBase::running(
            self.observe_reset(&self.state),
            ScalarReward(0.0),
        ))
    }

    /// Advances the simulation by one time step and returns the resulting snapshot.
    ///
    /// Clamps the force to `action_bounds`, integrates physics
    /// (velocity → clamp → position → left-wall inelastic collision), then
    /// computes `reward = -0.1 * force² + 100 * goal_reached`. The episode
    /// terminates when `position ≥ goal_position` and `velocity ≥ goal_velocity`.
    ///
    /// # Errors
    ///
    /// Currently infallible; always returns `Ok`.
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

        let obs = self.observe(&action, &self.state);
        let snap = if terminated {
            SnapshotBase::terminated(obs, reward)
        } else {
            SnapshotBase::running(obs, reward)
        };
        Ok(snap)
    }
}

// ---------------------------------------------------------------------------
// TensorConvertible
// ---------------------------------------------------------------------------

impl HostRow<1> for MountainCarContinuousObservation {
    fn row_shape() -> [usize; 1] {
        [2]
    }

    fn write_host_row(&self, buf: &mut Vec<f32>) {
        buf.extend_from_slice(&self.to_array());
    }
}

impl<B: burn::tensor::backend::Backend> TensorConvertible<1, B>
    for MountainCarContinuousObservation
{
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
        Ok(Self {
            position: v[0],
            velocity: v[1],
        })
    }
}

impl HostRow<1> for MountainCarContinuousAction {
    fn row_shape() -> [usize; 1] {
        [1]
    }

    fn write_host_row(&self, buf: &mut Vec<f32>) {
        buf.push(self.0);
    }
}

impl<B: burn::tensor::backend::Backend> TensorConvertible<1, B> for MountainCarContinuousAction {
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

// ---------------------------------------------------------------------------
// ASCII renderer
// ---------------------------------------------------------------------------

impl crate::render::AsciiRenderable for MountainCarContinuous {
    fn render_ascii(&self) -> String {
        let width = 40_usize;
        let span = self.config.pos_bounds.span();
        let frac = ((self.state.position - self.config.pos_bounds.lo()) / span).clamp(0.0, 1.0);
        let col = (frac * (width as f32 - 1.0)) as usize;
        let mut track = vec!['.'; width];
        track[col] = 'A';
        let track_str: String = track.iter().collect();
        format!(
            "[{track_str}]  pos={:.3}  vel={:.4}  step={}",
            self.state.position, self.state.velocity, self.steps
        )
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        let line = self.render_ascii();
        crate::render::StyledFrame {
            lines: vec![style_track_line(&line, 'A')],
        }
    }
}

/// Style a `[track]  suffix` style line where a single glyph marks the agent.
///
/// Reused by track-style renders (MountainCar, MountainCarContinuous,
/// CartPole). The portion before `]` is treated as the track; the matched
/// `agent_glyph` carries `AGENT_FG | AGENT_MODIFIER`; everything else
/// inside the brackets carries `WALL_FG`; the suffix is unstyled.
fn style_track_line(line: &str, agent_glyph: char) -> crate::render::StyledLine {
    use crate::render::palette::{AGENT_FG, AGENT_MODIFIER, WALL_FG};
    use crate::render::{SpanStyle, StyledLine, StyledSpan};

    let wall_style = SpanStyle::default().fg(WALL_FG);
    let agent_style = SpanStyle::default()
        .fg(AGENT_FG)
        .with_modifier(AGENT_MODIFIER);

    let Some(close_idx) = line.find(']') else {
        return StyledLine::unstyled(line);
    };
    let track_segment = &line[..=close_idx];
    let suffix = &line[close_idx + 1..];

    let Some(agent_col) = track_segment.find(agent_glyph) else {
        return StyledLine::unstyled(line);
    };

    let mut spans = Vec::with_capacity(4);
    spans.push(StyledSpan::new(
        track_segment[..agent_col].to_string(),
        wall_style,
    ));
    spans.push(StyledSpan::new(agent_glyph.to_string(), agent_style));
    spans.push(StyledSpan::new(
        track_segment[agent_col + 1..].to_string(),
        wall_style,
    ));
    if !suffix.is_empty() {
        spans.push(StyledSpan::raw(suffix.to_string()));
    }
    StyledLine::from_spans(spans)
}

impl rlevo_core::render::payload::Classic2DPayloadSource for MountainCarContinuous {
    fn classic2d_snapshot(&self) -> rlevo_core::render::payload::Classic2DSnapshot {
        use rlevo_core::render::payload::{
            Classic2DBody, Classic2DRole, Classic2DSnapshot, Point2,
        };
        let (lo, hi) = (self.config.pos_bounds.lo(), self.config.pos_bounds.hi());
        // Terrain profile y = sin(3x), sampled across the track.
        const SAMPLES: usize = 48;
        let terrain: Vec<Point2> = (0..=SAMPLES)
            .map(|i| {
                let x = lo + (hi - lo) * (i as f32 / SAMPLES as f32);
                Point2::new(x, (3.0 * x).sin())
            })
            .collect();
        let px = self.state.position;
        let py = (3.0 * px).sin();
        // Car as a small square sitting on the hill.
        let r = 0.04;
        let car = vec![
            Point2::new(px - r, py - r),
            Point2::new(px + r, py - r),
            Point2::new(px + r, py + r),
            Point2::new(px - r, py + r),
        ];
        Classic2DSnapshot {
            bodies: vec![
                Classic2DBody {
                    points: terrain,
                    role: Classic2DRole::Track,
                    closed: false,
                },
                Classic2DBody {
                    points: car,
                    role: Classic2DRole::Car,
                    closed: true,
                },
            ],
            bounds: (Point2::new(lo - 0.1, -1.1), Point2::new(hi + 0.1, 1.1)),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Unit tests for [`MountainCarContinuous`] covering observation shape,
    //! action boundary validation, control-cost formula, goal-reaching bonus,
    //! and determinism.

    use super::*;
    use rlevo_core::environment::Snapshot;

    fn default_env() -> MountainCarContinuous {
        MountainCarContinuous::with_config(MountainCarContinuousConfig::default())
            .expect("valid config")
    }

    #[test]
    fn default_config_validates() {
        assert!(MountainCarContinuousConfig::default().validate().is_ok());
    }

    #[test]
    fn rejects_goal_outside_pos_bounds() {
        let bad = MountainCarContinuousConfig {
            goal_position: 5.0,
            ..Default::default()
        };
        let err = bad.validate().unwrap_err();
        assert_eq!(err.field, "goal_position");
        assert!(MountainCarContinuous::with_config(bad).is_err());
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
        })
        .expect("valid config");
        let mut b = MountainCarContinuous::with_config(MountainCarContinuousConfig {
            seed: 3,
            ..Default::default()
        })
        .expect("valid config");
        a.reset().unwrap();
        b.reset().unwrap();
        let act = MountainCarContinuousAction::new(0.5).unwrap();
        for _ in 0..5 {
            let sa = a.step(act).unwrap();
            let sb = b.step(act).unwrap();
            assert_eq!(sa.observation().to_array(), sb.observation().to_array());
        }
    }

    #[test]
    fn render_styled_matches_ascii() {
        use crate::render::AsciiRenderable;

        let mut env = default_env();
        env.reset().unwrap();
        let plain = env.render_ascii();
        let styled = env.render_styled();
        assert_eq!(styled.lines.len(), 1);
        assert_eq!(styled.plain_text(), plain);
    }

    #[test]
    fn render_styled_uses_palette_consts() {
        use crate::render::AsciiRenderable;
        use crate::render::palette::{AGENT_FG, AGENT_MODIFIER, WALL_FG};

        let mut env = default_env();
        env.reset().unwrap();
        let styled = env.render_styled();
        let line = &styled.lines[0];

        let agent = line
            .spans
            .iter()
            .find(|s| s.text == "A")
            .expect("agent glyph span present");
        assert_eq!(agent.style.fg, Some(AGENT_FG));
        assert!(agent.style.modifier.contains(AGENT_MODIFIER));

        let bracket = line
            .spans
            .iter()
            .find(|s| s.text.starts_with('['))
            .expect("track-opening span present");
        assert_eq!(bracket.style.fg, Some(WALL_FG));
    }

    #[test]
    fn render_ascii_within_width_budget() {
        use crate::render::AsciiRenderable;

        let mut env = default_env();
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
        // The persistent RNG advances across resets (reset draws the initial
        // position from U(-0.6, -0.4)), so back-to-back resets must sample
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
