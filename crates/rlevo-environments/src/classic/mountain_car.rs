//! MountainCar-v0 environment.
//!
//! A car starting near the bottom of a valley must reach the flag at the top
//! of the right hill. Direct acceleration is insufficient; the agent must learn
//! to swing left first to build enough momentum. The update rule matches the
//! Gymnasium `MountainCar-v0` reference implementation exactly. Reference:
//! Moore (1990) and Sutton & Barto §10.1.
//!
//! ## Physical model
//!
//! The car moves along a one-dimensional track whose height profile is the
//! sinusoidal hill `$y(x) = \sin(3x)$`, with the position `$x$` confined to
//! `$[x_{\min}, x_{\max}] = [-1.2,\, 0.6]$` ([`MountainCarConfig::min_pos`],
//! [`MountainCarConfig::max_pos`]). The valley floor sits near `$x = -\tfrac{\pi}{6}$`
//! and the goal flag is at `$x_\text{goal} = 0.5$`. The state is the 2-vector
//!
//! ```math
//! \mathbf{s} = \left(x,\; \frac{dx}{dt}\right),
//! ```
//!
//! the car's position and velocity. Each step the agent picks a discrete action
//! `$a \in \{0, 1, 2\}$` mapping to a directional force `$a - 1 \in \{-1, 0, +1\}$`.
//!
//! ## Equations of motion
//!
//! This is a discrete-time map (no sub-step integrator). Writing the velocity at
//! step `$t$` as `$v_t = \tfrac{dx}{dt}\big|_t$`, force magnitude `$F$`
//! ([`MountainCarConfig::force`]) and gravity coefficient `$g$`
//! ([`MountainCarConfig::gravity`]), each step updates velocity then position:
//!
//! ```math
//! \begin{aligned}
//! v_{t+1} &= \operatorname{clip}\!\Big( v_t + (a - 1)\,F - g\cos(3 x_t),\;
//!            -v_{\max},\; v_{\max} \Big), \\[4pt]
//! x_{t+1} &= \operatorname{clip}\!\big( x_t + v_{t+1},\; x_{\min},\; x_{\max} \big).
//! \end{aligned}
//! ```
//!
//! The `$-g\cos(3 x_t)$` term is the gravitational pull along the track: the hill
//! slope is `$\tfrac{dy}{dx} = 3\cos(3x)$`, and the restoring acceleration is
//! taken proportional to `$\cos(3x)$` (the factor of `$3$` absorbed into `$g$`).
//! The speed is capped at `$v_{\max} = 0.07$` ([`MountainCarConfig::max_speed`]).
//! The left wall at `$x_{\min}$` is fully inelastic: if the position update
//! clamps against it, the velocity is reset to `$0$`. These are evaluated each
//! step in `MountainCar::apply_physics`.
//!
//! ## Reward and termination
//!
//! An episode **terminates** when `$x \ge x_\text{goal}$` **and**
//! `$v \ge v_\text{goal}$` ([`MountainCarConfig::goal_velocity`], default `$0$`).
//! The reward is `$-1$` on **every** step, including the terminal one, so the
//! undiscounted return equals the negative number of steps taken — minimising
//! cost is equivalent to reaching the flag as fast as possible. The minimum
//! achievable return depends on the initial position.
//!
//! ## Step limit
//!
//! This environment has **no intrinsic episode limit**. The standard
//! Gymnasium cap of 200 steps should be added externally:
//!
//! ```no_run,ignore
//! use rlevo_environments::{classic::mountain_car::MountainCar, wrappers::TimeLimit};
//!
//! let env = TimeLimit::new(MountainCar::new(false), 200);
//! ```
//!
//! ## Quick start
//!
//! ```rust
//! use rlevo_core::environment::{ConstructableEnv, Environment};
//! use rlevo_environments::classic::mountain_car::{MountainCar, MountainCarAction};
//!
//! let mut env = MountainCar::new(false);
//! let _snap = env.reset().unwrap();
//! let snap = env.step(MountainCarAction::Right).unwrap();
//! println!("{snap:?}");
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

/// Configuration for [`MountainCar`].
///
/// All fields are public so they can be set with struct-update syntax.
/// Use `MountainCarConfig::default()` for Gymnasium-compatible defaults.
///
/// # Examples
///
/// ```
/// use rlevo_environments::classic::mountain_car::MountainCarConfig;
///
/// let cfg = MountainCarConfig { seed: 42, ..MountainCarConfig::default() };
/// assert_eq!(cfg.seed, 42);
/// assert!((cfg.gravity - 0.0025).abs() < 1e-6);
/// ```
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

impl Validate for MountainCarConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "MountainCarConfig";
        config::ordered(C, "min_pos", f64::from(self.min_pos), f64::from(self.max_pos))?;
        config::in_range(C, "goal_position", f64::from(self.min_pos), f64::from(self.max_pos), f64::from(self.goal_position))?;
        config::positive(C, "max_speed", f64::from(self.max_speed))?;
        config::positive(C, "force", f64::from(self.force))?;
        Ok(())
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

    fn shape() -> [usize; 1] {
        [2]
    }
    fn numel(&self) -> usize {
        2
    }

    fn is_valid(&self) -> bool {
        self.position.is_finite() && self.velocity.is_finite()
    }

    fn observe(&self) -> MountainCarObservation {
        MountainCarObservation {
            position: self.position,
            velocity: self.velocity,
        }
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
    fn shape() -> [usize; 1] {
        [2]
    }
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
    fn shape() -> [usize; 1] {
        [3]
    }
    fn is_valid(&self) -> bool {
        true
    }
}

impl DiscreteAction<1> for MountainCarAction {
    const ACTION_COUNT: usize = 3;

    /// Constructs an action from its integer index.
    ///
    /// # Panics
    ///
    /// Panics if `index` is not `0` (Left), `1` (NoAccel), or `2` (Right).
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
            Self::Left => 0,
            Self::NoAccel => 1,
            Self::Right => 2,
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
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `config` fails [`Validate`] (e.g.
    /// `min_pos >= max_pos`, a `goal_position` outside `[min_pos, max_pos]`, or
    /// non-positive `max_speed` / `force`).
    pub fn with_config(config: MountainCarConfig) -> Result<Self, ConfigError> {
        config.validate()?;
        let rng = StdRng::seed_from_u64(config.seed);
        Ok(Self {
            state: MountainCarState {
                position: -0.5,
                velocity: 0.0,
            },
            config,
            rng,
            steps: 0,
        })
    }

    /// Current step count within the episode.
    pub fn steps(&self) -> usize {
        self.steps
    }

    fn sample_init_state(&mut self) -> MountainCarState {
        let pos = Uniform::new_inclusive(-0.6_f32, -0.4_f32)
            .unwrap()
            .sample(&mut self.rng);
        MountainCarState {
            position: pos,
            velocity: 0.0,
        }
    }

    fn apply_physics(
        state: MountainCarState,
        action: MountainCarAction,
        cfg: &MountainCarConfig,
    ) -> MountainCarState {
        let action_val = action.to_index() as f32 - 1.0; // -1, 0, or +1
        let mut vel =
            state.velocity + action_val * cfg.force - (3.0 * state.position).cos() * cfg.gravity;
        vel = vel.clamp(-cfg.max_speed, cfg.max_speed);
        let mut pos = state.position + vel;
        pos = pos.clamp(cfg.min_pos, cfg.max_pos);
        // Inelastic left wall
        if pos <= cfg.min_pos {
            vel = 0.0;
        }
        MountainCarState {
            position: pos,
            velocity: vel,
        }
    }

    fn is_terminal(state: &MountainCarState, cfg: &MountainCarConfig) -> bool {
        state.position >= cfg.goal_position && state.velocity >= cfg.goal_velocity
    }
}

impl fmt::Display for MountainCar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MountainCar(step={}, pos={:.3}, vel={:.4})",
            self.steps, self.state.position, self.state.velocity
        )
    }
}

impl ConstructableEnv for MountainCar {
    fn new(render: bool) -> Self {
        let _ = render;
        Self::with_config(MountainCarConfig::default()).expect("default config must validate")
    }
}

impl Environment<1, 1, 1> for MountainCar {
    type StateType = MountainCarState;
    type ObservationType = MountainCarObservation;
    type ActionType = MountainCarAction;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, MountainCarObservation, ScalarReward>;

    /// Resets the environment to a random initial state and returns the first snapshot.
    ///
    /// Re-seeds the internal RNG from `config.seed` so repeated calls with the
    /// same seed produce identical initial positions in `[-0.6, -0.4]`.
    ///
    /// # Errors
    ///
    /// Currently infallible; always returns `Ok`.
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
    /// Applies the discrete-action physics (velocity update → clamp → position
    /// update → left-wall inelastic collision), then checks whether the car has
    /// reached `goal_position` with at least `goal_velocity`. Reward is `-1.0`
    /// on every step, including the terminal step.
    ///
    /// # Errors
    ///
    /// Currently infallible; always returns `Ok`.
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
        let width = 40_usize;
        let span = self.config.max_pos - self.config.min_pos;
        let frac = ((self.state.position - self.config.min_pos) / span).clamp(0.0, 1.0);
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
            lines: vec![style_mountain_car_line(&line)],
        }
    }
}

/// Convert one `render_ascii` line into a styled line.
///
/// Splits the line into spans: brackets and track dots take [`WALL_FG`],
/// the agent glyph `A` takes [`AGENT_FG`] with [`AGENT_MODIFIER`], and the
/// trailing numeric annotations are emitted unstyled.
fn style_mountain_car_line(line: &str) -> crate::render::StyledLine {
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

    let Some(agent_col) = track_segment.find('A') else {
        return StyledLine::unstyled(line);
    };

    let mut spans = Vec::with_capacity(4);
    spans.push(StyledSpan::new(
        track_segment[..agent_col].to_string(),
        wall_style,
    ));
    spans.push(StyledSpan::new("A", agent_style));
    spans.push(StyledSpan::new(
        track_segment[agent_col + 1..].to_string(),
        wall_style,
    ));
    if !suffix.is_empty() {
        spans.push(StyledSpan::raw(suffix.to_string()));
    }
    StyledLine::from_spans(spans)
}

// ---------------------------------------------------------------------------
// TensorConvertible
// ---------------------------------------------------------------------------

impl<B: burn::tensor::backend::Backend> TensorConvertible<1, B> for MountainCarObservation {
    fn to_tensor(&self, device: &<B as burn::tensor::backend::BackendTypes>::Device) -> burn::tensor::Tensor<B, 1> {
        burn::tensor::Tensor::from_floats(self.to_array(), device)
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
        Ok(Self {
            position: v[0],
            velocity: v[1],
        })
    }
}

impl<B: burn::tensor::backend::Backend> TensorConvertible<1, B> for MountainCarAction {
    fn to_tensor(&self, device: &<B as burn::tensor::backend::BackendTypes>::Device) -> burn::tensor::Tensor<B, 1> {
        let mut one_hot = [0.0_f32; 3];
        one_hot[self.to_index()] = 1.0;
        burn::tensor::Tensor::from_floats(one_hot, device)
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
    //! Unit tests for [`MountainCar`] covering observation shape, action
    //! indexing, reset bounds, left-wall physics, goal termination, reward
    //! value, and determinism.

    use super::*;
    use rlevo_core::environment::Snapshot;

    fn default_env() -> MountainCar {
        MountainCar::with_config(MountainCarConfig::default()).expect("valid config")
    }

    #[test]
    fn default_config_validates() {
        assert!(MountainCarConfig::default().validate().is_ok());
    }

    #[test]
    fn rejects_unordered_bounds() {
        let bad = MountainCarConfig { min_pos: 1.0, max_pos: -1.0, ..Default::default() };
        assert!(MountainCar::with_config(bad).is_err());
    }

    #[test]
    fn reset_initialises_correctly() {
        use rlevo_core::environment::EpisodeStatus;

        let mut env = default_env();
        let snap = env.reset().unwrap();
        assert_eq!(snap.status(), EpisodeStatus::Running);
        let obs = snap.observation();
        assert!(
            obs.position >= -0.6 && obs.position <= -0.4,
            "position {}",
            obs.position
        );
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
        let state = MountainCarState {
            position: -1.19,
            velocity: -0.05,
        };
        let next = MountainCar::apply_physics(state, MountainCarAction::Left, &cfg);
        assert_eq!(next.position, cfg.min_pos);
        assert_eq!(next.velocity, 0.0);
    }

    #[test]
    fn goal_terminates() {
        let mut env = default_env();
        env.reset().unwrap();
        env.state = MountainCarState {
            position: 0.55,
            velocity: 0.01,
        };
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
        let mut a = MountainCar::with_config(MountainCarConfig {
            seed: 7,
            ..Default::default()
        })
        .expect("valid config");
        let mut b = MountainCar::with_config(MountainCarConfig {
            seed: 7,
            ..Default::default()
        })
        .expect("valid config");
        a.reset().unwrap();
        b.reset().unwrap();
        for action in [
            MountainCarAction::Right,
            MountainCarAction::Left,
            MountainCarAction::NoAccel,
        ] {
            let sa = a.step(action).unwrap();
            let sb = b.step(action).unwrap();
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

        let agent_span = line
            .spans
            .iter()
            .find(|s| s.text == "A")
            .expect("agent glyph span present");
        assert_eq!(agent_span.style.fg, Some(AGENT_FG));
        assert!(agent_span.style.modifier.contains(AGENT_MODIFIER));

        let bracket_span = line
            .spans
            .iter()
            .find(|s| s.text.starts_with('['))
            .expect("track-opening span present");
        assert_eq!(bracket_span.style.fg, Some(WALL_FG));
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
}

impl rlevo_core::render::payload::Classic2DPayloadSource for MountainCar {
    fn classic2d_snapshot(&self) -> rlevo_core::render::payload::Classic2DSnapshot {
        use rlevo_core::render::payload::{Classic2DBody, Classic2DRole, Classic2DSnapshot, Point2};
        let (lo, hi) = (self.config.min_pos, self.config.max_pos);
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
                Classic2DBody { points: terrain, role: Classic2DRole::Track, closed: false },
                Classic2DBody { points: car, role: Classic2DRole::Car, closed: true },
            ],
            bounds: (Point2::new(lo - 0.1, -1.1), Point2::new(hi + 0.1, 1.1)),
        }
    }
}
