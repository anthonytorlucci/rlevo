//! Synthetic pixel-over-grid navigation — the first real [`Observable`] consumer.
//!
//! A compact allocentric grid task whose *true latent* is a pair of cell
//! indices (agent, goal) on a `5×5` grid, but whose *observation* is a rendered
//! `20×20×3` RGB image. The agent must recover the latent `(agent, goal)` from
//! pixels — the modality-changing POMDP shape (a small RAM-like state observed
//! as an image) that [`State::observe`] structurally cannot express because it
//! pins the observation to the state's own tensor order.
//!
//! This environment is [`Environment<3, 1, 1>`] — observation rank `3`, state
//! rank `1`, action rank `1`, so `R(3) != SR(1)`. Every snapshot is built from
//! [`Observable::project`] (the rank-3 pixel projection), **never** from
//! [`State::observe`] (the rank-1 latent). It is the production counterpart to
//! the `MockRam` integration test that proved the [`Observable`] contract
//! (issue #62, ADR 0019), and resolves issue #65.
//!
//! ## Why not fold into [`grids`](crate::grids)
//!
//! The `grids` family shares an *egocentric* `7×7×3` observation core with
//! `R == SR` and 7-action turn/forward dynamics. This task is *allocentric*,
//! 4-way Cartesian, and modality-changing — it reuses none of that core, so it
//! lives in its own concept module.
//!
//! ## Dimensions (fixed for v1)
//!
//! | Const | Value | Meaning |
//! |-------|-------|---------|
//! | [`GRID_SIDE`] | `5` | grid is `5×5 = 25` cells |
//! | [`CELL_PX`] | `4` | each cell renders as a `4×4` pixel block |
//! | [`CHANNELS`] | `3` | RGB; image is `[20, 20, 3]` |
//!
//! RGB cell colors (distinct hues, recoverable per channel): background black
//! `[0, 0, 0]`, goal green `[0, 128, 0]`, agent white `[255, 255, 255]`.
//! Agent-on-goal renders white (the terminal frame). RGB rather than grayscale
//! so a future Atari backend differs only in resolution, not channel count.
//!
//! ## Layout (`5×5` grid, fixed placement)
//!
//! ```text
//! @ . . . .   @ = agent (cell 0,  white)
//! . . . . .
//! . . . . .
//! . . . . .
//! . . . . *   * = goal  (cell 24, green)
//! ```
//!
//! ## Example
//!
//! ```rust
//! use rlevo_environments::pixel_grid::{PixelGridConfig, PixelGridEnv};
//! use rlevo_core::environment::{ConstructableEnv, Environment};
//!
//! let mut env = PixelGridEnv::with_config(PixelGridConfig::new(100, 0, false), false)
//!     .expect("valid config");
//! let snapshot = env.reset().unwrap();
//! # let _ = snapshot;
//! ```

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rlevo_core::base::{Observation, State, TensorConversionError, TensorConvertible};
use rlevo_core::config::{self, ConfigError, Validate};
use rlevo_core::environment::{ConstructableEnv, Environment, EnvironmentError, SnapshotBase};
use rlevo_core::reward::ScalarReward;
use rlevo_core::state::Observable;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};

use burn::tensor::{Tensor, TensorData, backend::Backend};

/// Grid side length in cells; the grid is [`GRID_SIDE`]`²` cells.
pub const GRID_SIDE: usize = 5;
/// Pixel side length of one rendered cell block.
pub const CELL_PX: usize = 4;
/// Number of color channels in the rendered image (RGB).
pub const CHANNELS: usize = 3;

/// Image side length in pixels (`GRID_SIDE * CELL_PX`).
pub const IMG_SIDE: usize = GRID_SIDE * CELL_PX;
/// Total number of cells in the grid (`GRID_SIDE²`).
pub const CELL_COUNT: usize = GRID_SIDE * GRID_SIDE;
/// Total number of scalar elements in the rendered image
/// (`IMG_SIDE * IMG_SIDE * CHANNELS`).
pub const PIXEL_COUNT: usize = IMG_SIDE * IMG_SIDE * CHANNELS;

/// RGB color of the goal cell (green).
const GOAL_RGB: [u8; CHANNELS] = [0, 128, 0];
/// RGB color of the agent cell (white).
const AGENT_RGB: [u8; CHANNELS] = [255, 255, 255];

/// Minigrid's canonical success reward: `1 - 0.9 * (step / max_steps)`.
///
/// Reaching the goal early pays close to `1.0`; reaching it on the final legal
/// step pays `0.1`. Returns `0.0` when `max_steps == 0` so the formula is total.
/// Re-implemented locally to keep this module independent of the `grids` family.
#[must_use]
#[allow(clippy::cast_precision_loss)]
fn success_reward(step: usize, max_steps: usize) -> f32 {
    if max_steps == 0 {
        return 0.0;
    }
    1.0 - 0.9 * (step as f32 / max_steps as f32)
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// Compact rank-1 latent: agent and goal cell indices on the `GRID_SIDE × GRID_SIDE` grid.
///
/// Cell indices are row-major flattened into `0..CELL_COUNT` (row `c / GRID_SIDE`,
/// column `c % GRID_SIDE`). This is the RAM analogue — small, fully known, and
/// exactly recoverable — that the rank-3 pixel observation is rendered from.
///
/// Implements both [`State<1>`] (the trivial same-order [`observe`](State::observe)
/// returning the latent indices) and [`Observable<3>`] (the modality-changing
/// [`project`](Observable::project) returning the pixel image). The dual impl is
/// the whole point: `observe()` is rank-locked to the state order, so the
/// rank-3 projection must live on the separate [`Observable`] trait.
///
/// # Examples
///
/// ```rust
/// use rlevo_environments::pixel_grid::PixelGridState;
/// use rlevo_core::base::State;
///
/// let state = PixelGridState::new(0, 24);
/// assert!(state.is_valid());
/// assert_eq!(<PixelGridState as State<1>>::shape(), [2]);
/// ```
#[derive(Debug, Clone)]
pub struct PixelGridState {
    agent: u32,
    goal: u32,
}

impl PixelGridState {
    /// Construct a state from agent and goal cell indices.
    ///
    /// Indices are not validated here; call [`is_valid`](State::is_valid) to
    /// check that both lie in `0..CELL_COUNT`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rlevo_environments::pixel_grid::PixelGridState;
    ///
    /// let state = PixelGridState::new(0, 24);
    /// assert_eq!(state.agent(), 0);
    /// assert_eq!(state.goal(), 24);
    /// ```
    #[must_use]
    pub const fn new(agent: u32, goal: u32) -> Self {
        Self { agent, goal }
    }

    /// Returns the agent's cell index.
    #[must_use]
    pub const fn agent(&self) -> u32 {
        self.agent
    }

    /// Returns the goal's cell index.
    #[must_use]
    pub const fn goal(&self) -> u32 {
        self.goal
    }

    /// `true` when the agent occupies the goal cell.
    #[must_use]
    pub const fn at_goal(&self) -> bool {
        self.agent == self.goal
    }

    /// Apply a 4-way move to the agent, clamping at the grid walls.
    ///
    /// Moving into a wall is a no-op (the agent holds position), matching the
    /// "bump and hold" convention of the grid environments.
    fn apply_move(&mut self, action: PixelGridAction) {
        let agent = self.agent as usize;
        let mut row = agent / GRID_SIDE;
        let mut col = agent % GRID_SIDE;
        match action {
            PixelGridAction::Up => row = row.saturating_sub(1),
            PixelGridAction::Down => row = (row + 1).min(GRID_SIDE - 1),
            PixelGridAction::Left => col = col.saturating_sub(1),
            PixelGridAction::Right => col = (col + 1).min(GRID_SIDE - 1),
        }
        let cell = row * GRID_SIDE + col;
        self.agent = u32::try_from(cell).expect("cell index fits in u32");
    }

    /// Render this state to a `GRID_SIDE × GRID_SIDE` ASCII frame.
    ///
    /// Agent and goal are distinguished by **glyph** (`@` / `*`) as well as
    /// color in the pixel projection — never color alone — so the frame is
    /// legible without color perception. Agent-on-goal renders `@`.
    #[must_use]
    pub fn render_ascii(&self) -> String {
        let mut out = String::with_capacity(CELL_COUNT * 2);
        for cell in 0..CELL_COUNT {
            let cell = u32::try_from(cell).expect("cell index fits in u32");
            let glyph = if cell == self.agent {
                '@'
            } else if cell == self.goal {
                '*'
            } else {
                '.'
            };
            out.push(glyph);
            if (cell as usize) % GRID_SIDE == GRID_SIDE - 1 {
                out.push('\n');
            } else {
                out.push(' ');
            }
        }
        out
    }
}

impl State<1> for PixelGridState {
    type Observation = LatentObservation;

    fn shape() -> [usize; 1] {
        [2]
    }

    fn observe(&self) -> Self::Observation {
        LatentObservation {
            agent: self.agent,
            goal: self.goal,
        }
    }

    fn is_valid(&self) -> bool {
        let count = u32::try_from(CELL_COUNT).expect("cell count fits in u32");
        self.agent < count && self.goal < count
    }
}

impl Observable<3> for PixelGridState {
    type Observation = PixelObservation;

    fn project(&self) -> Self::Observation {
        let mut pixels = vec![0u8; PIXEL_COUNT];
        paint_cell(&mut pixels, self.goal as usize, GOAL_RGB);
        // Agent is painted last so agent-on-goal renders as the agent color.
        paint_cell(&mut pixels, self.agent as usize, AGENT_RGB);
        PixelObservation { pixels }
    }
}

/// Paint a single cell's `CELL_PX × CELL_PX` block in a row-major `[H, W, C]`
/// pixel buffer with the given RGB color.
fn paint_cell(pixels: &mut [u8], cell: usize, color: [u8; CHANNELS]) {
    let crow = cell / GRID_SIDE;
    let ccol = cell % GRID_SIDE;
    for dr in 0..CELL_PX {
        for dc in 0..CELL_PX {
            let h = crow * CELL_PX + dr;
            let w = ccol * CELL_PX + dc;
            let base = (h * IMG_SIDE + w) * CHANNELS;
            pixels[base..base + CHANNELS].copy_from_slice(&color);
        }
    }
}

// ---------------------------------------------------------------------------
// Observations
// ---------------------------------------------------------------------------

/// Rank-1 "full" observation — the latent `(agent, goal)` cell indices.
///
/// Required because [`State<1>`] pins [`observe`](State::observe) to rank 1;
/// the modality change lives on the separate [`Observable<3>`] impl. Useful for
/// scoring a learned belief against ground truth.
///
/// # Examples
///
/// ```rust
/// use rlevo_environments::pixel_grid::LatentObservation;
/// use rlevo_core::base::Observation;
///
/// assert_eq!(<LatentObservation as Observation<1>>::shape(), [2]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LatentObservation {
    /// Agent cell index in `0..CELL_COUNT`.
    pub agent: u32,
    /// Goal cell index in `0..CELL_COUNT`.
    pub goal: u32,
}

impl Observation<1> for LatentObservation {
    fn shape() -> [usize; 1] {
        [2]
    }
}

/// Rank-3 pixel observation: the rendered `[IMG_SIDE, IMG_SIDE, CHANNELS]` RGB image.
///
/// The pixel buffer is row-major `[H, W, C]` with `PIXEL_COUNT` elements. This
/// is the modality the agent actually receives; it must recover the latent
/// `(agent, goal)` from these pixels.
///
/// # Examples
///
/// ```rust
/// use rlevo_environments::pixel_grid::{PixelGridState, PixelObservation, PIXEL_COUNT};
/// use rlevo_core::base::Observation;
/// use rlevo_core::state::Observable;
///
/// assert_eq!(<PixelObservation as Observation<3>>::shape(), [20, 20, 3]);
/// let obs = PixelGridState::new(0, 24).project();
/// assert_eq!(obs.pixels().len(), PIXEL_COUNT);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PixelObservation {
    pixels: Vec<u8>,
}

impl PixelObservation {
    /// Returns the row-major `[H, W, C]` pixel buffer.
    #[must_use]
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }
}

impl Observation<3> for PixelObservation {
    fn shape() -> [usize; 3] {
        [IMG_SIDE, IMG_SIDE, CHANNELS]
    }
}

impl<B: Backend> TensorConvertible<3, B> for PixelObservation {
    fn to_tensor(&self, device: &<B as burn::tensor::backend::BackendTypes>::Device) -> Tensor<B, 3> {
        // Normalize bytes to [0, 1] so a Burn policy can consume the frame directly.
        let flat: Vec<f32> = self.pixels.iter().map(|&b| f32::from(b) / 255.0).collect();
        let data = TensorData::new(flat, [IMG_SIDE, IMG_SIDE, CHANNELS]);
        Tensor::<B, 3>::from_data(data, device)
    }

    /// Reconstructs the image from a normalized `[IMG_SIDE, IMG_SIDE, CHANNELS]`
    /// tensor by scaling back to `0..=255` and rounding.
    ///
    /// Round-trips exactly for any `u8` payload: `b / 255.0 * 255.0` rounds back
    /// to `b`.
    ///
    /// # Errors
    ///
    /// Returns [`TensorConversionError`] if the tensor shape is not
    /// `[IMG_SIDE, IMG_SIDE, CHANNELS]`, the backend fails to materialize its
    /// data, or any value lies outside `[0, 1]` after scaling to the `u8` range.
    fn from_tensor(tensor: Tensor<B, 3>) -> Result<Self, TensorConversionError> {
        let dims = tensor.dims();
        if dims.as_slice() != [IMG_SIDE, IMG_SIDE, CHANNELS] {
            return Err(TensorConversionError {
                message: format!("expected shape [{IMG_SIDE}, {IMG_SIDE}, {CHANNELS}], got {dims:?}"),
            });
        }
        let flat = tensor
            .into_data()
            .into_vec::<f32>()
            .map_err(|e| TensorConversionError {
                message: format!("failed to read tensor data: {e:?}"),
            })?;
        let mut pixels = Vec::with_capacity(PIXEL_COUNT);
        for (idx, &value) in flat.iter().enumerate() {
            let scaled = value * 255.0;
            if !scaled.is_finite() || scaled < -0.5 || scaled > f32::from(u8::MAX) + 0.5 {
                return Err(TensorConversionError {
                    message: format!("value at index {idx} out of u8 range: {value}"),
                });
            }
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            pixels.push(scaled.round() as u8);
        }
        Ok(Self { pixels })
    }
}

// ---------------------------------------------------------------------------
// Action
// ---------------------------------------------------------------------------

/// A 4-way Cartesian move on the grid.
///
/// Indices: `0 = Up`, `1 = Down`, `2 = Left`, `3 = Right`. Moving into a wall is
/// a no-op (the agent holds position).
///
/// # Examples
///
/// ```rust
/// use rlevo_environments::pixel_grid::PixelGridAction;
/// use rlevo_core::action::DiscreteAction;
///
/// assert_eq!(PixelGridAction::from_index(0), PixelGridAction::Up);
/// assert_eq!(PixelGridAction::Right.to_index(), 3);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelGridAction {
    /// Decrease the agent's row (clamped at the top wall).
    Up,
    /// Increase the agent's row (clamped at the bottom wall).
    Down,
    /// Decrease the agent's column (clamped at the left wall).
    Left,
    /// Increase the agent's column (clamped at the right wall).
    Right,
}

impl rlevo_core::base::Action<1> for PixelGridAction {
    fn shape() -> [usize; 1] {
        [4]
    }

    fn is_valid(&self) -> bool {
        true
    }
}

impl rlevo_core::action::DiscreteAction<1> for PixelGridAction {
    const ACTION_COUNT: usize = 4;

    /// Construct an action from its index.
    ///
    /// # Panics
    ///
    /// Panics if `index >= ACTION_COUNT` (`4`), which indicates a programming
    /// error (out-of-bounds action selection).
    fn from_index(index: usize) -> Self {
        match index {
            0 => PixelGridAction::Up,
            1 => PixelGridAction::Down,
            2 => PixelGridAction::Left,
            3 => PixelGridAction::Right,
            _ => panic!("Index out of bounds: {index}"),
        }
    }

    fn to_index(&self) -> usize {
        match self {
            PixelGridAction::Up => 0,
            PixelGridAction::Down => 1,
            PixelGridAction::Left => 2,
            PixelGridAction::Right => 3,
        }
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for [`PixelGridEnv`].
///
/// The grid and image dimensions are fixed compile-time constants ([`GRID_SIDE`],
/// [`CELL_PX`], [`CHANNELS`]); the config carries only the per-episode knobs.
///
/// # Examples
///
/// ```rust
/// use rlevo_environments::pixel_grid::PixelGridConfig;
///
/// let cfg = PixelGridConfig::new(100, 0, false);
/// assert_eq!(cfg.max_steps, 100);
/// assert!(!cfg.random_placement);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PixelGridConfig {
    /// Maximum number of steps before the episode is truncated.
    pub max_steps: usize,
    /// Seed for the environment's RNG (used only when `random_placement`).
    pub seed: u64,
    /// When `true`, agent and goal are placed on random distinct cells each
    /// reset; when `false`, the agent starts at cell `0` and the goal sits at
    /// cell `CELL_COUNT - 1` (deterministic).
    pub random_placement: bool,
}

impl PixelGridConfig {
    /// Construct a config with explicit field values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rlevo_environments::pixel_grid::PixelGridConfig;
    ///
    /// let cfg = PixelGridConfig::new(50, 7, true);
    /// assert_eq!(cfg.seed, 7);
    /// ```
    #[must_use]
    pub const fn new(max_steps: usize, seed: u64, random_placement: bool) -> Self {
        Self {
            max_steps,
            seed,
            random_placement,
        }
    }
}

impl Default for PixelGridConfig {
    fn default() -> Self {
        Self {
            max_steps: 4 * CELL_COUNT,
            seed: 0,
            random_placement: false,
        }
    }
}

impl Validate for PixelGridConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "PixelGridConfig";
        config::nonzero(C, "max_steps", self.max_steps)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

/// Synthetic pixel-over-grid environment: rank-1 latent, rank-3 pixel observation.
///
/// Implements [`Environment<3, 1, 1>`] — `R(3) != SR(1)`. Every snapshot is
/// built from [`Observable::project`] (the rank-3 image), never from
/// [`State::observe`] (the rank-1 latent). Construct via
/// [`PixelGridEnv::with_config`] for full control or
/// [`ConstructableEnv::new`] for defaults.
///
/// # Examples
///
/// ```no_run
/// use rlevo_environments::pixel_grid::{PixelGridConfig, PixelGridEnv};
/// use rlevo_core::environment::{ConstructableEnv, Environment};
///
/// let mut env = PixelGridEnv::with_config(PixelGridConfig::new(100, 0, false), false)
///     .expect("valid config");
/// env.reset().unwrap();
/// ```
#[derive(Debug)]
pub struct PixelGridEnv {
    state: PixelGridState,
    config: PixelGridConfig,
    steps: usize,
    render: bool,
    rng: StdRng,
}

impl PixelGridEnv {
    /// Construct an environment from an explicit configuration.
    ///
    /// Seeds the internal RNG and builds the initial state. Call
    /// [`Environment::reset`] before the first [`Environment::step`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rlevo_environments::pixel_grid::{PixelGridConfig, PixelGridEnv};
    ///
    /// let env = PixelGridEnv::with_config(PixelGridConfig::new(100, 0, false), false)
    ///     .expect("valid config");
    /// assert_eq!(env.steps(), 0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `config` fails [`Validate`] (zero
    /// `max_steps`).
    pub fn with_config(config: PixelGridConfig, render: bool) -> Result<Self, ConfigError> {
        config.validate()?;
        let mut rng = StdRng::seed_from_u64(config.seed);
        let state = Self::initial_state(config, &mut rng);
        Ok(Self {
            state,
            config,
            steps: 0,
            render,
            rng,
        })
    }

    /// Returns a reference to the active configuration.
    #[must_use]
    pub const fn config(&self) -> &PixelGridConfig {
        &self.config
    }

    /// Returns the number of steps taken since the last reset.
    #[must_use]
    pub const fn steps(&self) -> usize {
        self.steps
    }

    /// Returns a reference to the current latent state.
    #[must_use]
    pub const fn state(&self) -> &PixelGridState {
        &self.state
    }

    /// Build the initial state for an episode.
    ///
    /// Fixed placement puts the agent at cell `0` and the goal at the last
    /// cell; random placement samples distinct cells from `rng` (host-RNG
    /// convention — no backend/global RNG).
    fn initial_state(config: PixelGridConfig, rng: &mut StdRng) -> PixelGridState {
        if config.random_placement {
            let agent = rng.random_range(0..CELL_COUNT);
            let mut goal = rng.random_range(0..CELL_COUNT);
            while goal == agent {
                goal = rng.random_range(0..CELL_COUNT);
            }
            PixelGridState::new(
                u32::try_from(agent).expect("cell index fits in u32"),
                u32::try_from(goal).expect("cell index fits in u32"),
            )
        } else {
            PixelGridState::new(0, u32::try_from(CELL_COUNT - 1).expect("cell index fits in u32"))
        }
    }

    /// Build a snapshot from the rank-3 projection, emitting an optional ASCII
    /// debug frame when `render` is set.
    fn snapshot(&self, reward: f32, status: SnapshotStatus) -> SnapshotBase<3, PixelObservation, ScalarReward> {
        if self.render {
            // Render is a debug side effect; drop the string when internal.
            let _ = self.state.render_ascii();
        }
        let obs = self.state.project();
        let reward = ScalarReward::new(reward);
        match status {
            SnapshotStatus::Running => SnapshotBase::running(obs, reward),
            SnapshotStatus::Terminated => SnapshotBase::terminated(obs, reward),
            SnapshotStatus::Truncated => SnapshotBase::truncated(obs, reward),
        }
    }
}

/// Internal snapshot lifecycle selector used by [`PixelGridEnv::snapshot`].
#[derive(Debug, Clone, Copy)]
enum SnapshotStatus {
    Running,
    Terminated,
    Truncated,
}

impl Display for PixelGridEnv {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PixelGridEnv(grid={GRID_SIDE}x{GRID_SIDE}, step={}/{})",
            self.steps, self.config.max_steps
        )
    }
}

impl ConstructableEnv for PixelGridEnv {
    fn new(render: bool) -> Self {
        Self::with_config(PixelGridConfig::default(), render)
            .expect("default config must validate")
    }
}

impl Environment<3, 1, 1> for PixelGridEnv {
    type StateType = PixelGridState;
    type ObservationType = PixelObservation;
    type ActionType = PixelGridAction;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<3, PixelObservation, ScalarReward>;

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.rng = StdRng::seed_from_u64(self.config.seed);
        self.state = Self::initial_state(self.config, &mut self.rng);
        self.steps = 0;
        Ok(self.snapshot(0.0, SnapshotStatus::Running))
    }

    fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> {
        self.steps += 1;
        self.state.apply_move(action);

        let snap = if self.state.at_goal() {
            self.snapshot(success_reward(self.steps, self.config.max_steps), SnapshotStatus::Terminated)
        } else if self.steps >= self.config.max_steps {
            self.snapshot(0.0, SnapshotStatus::Truncated)
        } else {
            self.snapshot(0.0, SnapshotStatus::Running)
        };
        Ok(snap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlevo_core::action::DiscreteAction;
    use rlevo_core::environment::{EpisodeStatus, Snapshot};

    #[test]
    fn default_config_validates() {
        assert!(PixelGridConfig::default().validate().is_ok());
    }

    #[test]
    fn rejects_zero_max_steps() {
        let bad = PixelGridConfig { max_steps: 0, ..Default::default() };
        assert!(PixelGridEnv::with_config(bad, false).is_err());
    }

    #[test]
    fn state_shape_matches_numel() {
        let state = PixelGridState::new(0, 24);
        let shape_product: usize = <PixelGridState as State<1>>::shape().iter().product();
        assert_eq!(state.numel(), shape_product);
        assert_eq!(state.numel(), 2);
    }

    #[test]
    fn is_valid_rejects_out_of_range() {
        assert!(PixelGridState::new(0, 24).is_valid());
        assert!(!PixelGridState::new(25, 0).is_valid());
        assert!(!PixelGridState::new(0, 25).is_valid());
    }

    #[test]
    fn observation_ranks_and_shapes() {
        assert_eq!(<PixelObservation as Observation<3>>::shape(), [20, 20, 3]);
        assert_eq!(<PixelObservation as Observation<3>>::RANK, 3);
        assert_eq!(<LatentObservation as Observation<1>>::shape(), [2]);
        assert_eq!(<LatentObservation as Observation<1>>::RANK, 1);
    }

    #[test]
    fn project_paints_agent_and_goal_blocks() {
        // Agent at cell 0 (top-left block), goal at cell 24 (bottom-right block).
        let obs = PixelGridState::new(0, 24).project();
        assert_eq!(obs.pixels().len(), PIXEL_COUNT);

        let pixel = |h: usize, w: usize| {
            let base = (h * IMG_SIDE + w) * CHANNELS;
            [obs.pixels()[base], obs.pixels()[base + 1], obs.pixels()[base + 2]]
        };
        // Agent block spans rows/cols 0..4.
        assert_eq!(pixel(0, 0), AGENT_RGB);
        assert_eq!(pixel(3, 3), AGENT_RGB);
        // Goal block spans rows/cols 16..20.
        assert_eq!(pixel(16, 16), GOAL_RGB);
        assert_eq!(pixel(19, 19), GOAL_RGB);
        // A middle cell is background (black).
        assert_eq!(pixel(10, 10), [0, 0, 0]);
    }

    #[test]
    fn agent_on_goal_renders_agent_color() {
        let obs = PixelGridState::new(12, 12).project();
        // Cell 12 = row 2, col 2 → pixel block rows/cols 8..12.
        let base = (8 * IMG_SIDE + 8) * CHANNELS;
        assert_eq!(
            [obs.pixels()[base], obs.pixels()[base + 1], obs.pixels()[base + 2]],
            AGENT_RGB
        );
    }

    #[test]
    fn action_index_round_trip() {
        for i in 0..PixelGridAction::ACTION_COUNT {
            assert_eq!(PixelGridAction::from_index(i).to_index(), i);
        }
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn action_from_index_out_of_bounds_panics() {
        let _ = PixelGridAction::from_index(4);
    }

    #[test]
    fn pixel_observation_round_trips_through_tensor() {
        use burn::backend::Flex;
        type TestBackend = Flex;
        let device = Default::default();

        let obs = PixelGridState::new(3, 21).project();
        let tensor = <PixelObservation as TensorConvertible<3, TestBackend>>::to_tensor(&obs, &device);
        let round_tripped =
            <PixelObservation as TensorConvertible<3, TestBackend>>::from_tensor(tensor).unwrap();
        assert_eq!(round_tripped, obs);
    }

    #[test]
    fn from_tensor_rejects_wrong_shape() {
        use burn::backend::Flex;
        type TestBackend = Flex;
        let device = Default::default();

        let data = TensorData::new(vec![0.0f32; IMG_SIDE * IMG_SIDE * 2], [IMG_SIDE, IMG_SIDE, 2]);
        let tensor = Tensor::<TestBackend, 3>::from_data(data, &device);
        let err = <PixelObservation as TensorConvertible<3, TestBackend>>::from_tensor(tensor).unwrap_err();
        assert!(err.message.contains("expected shape"));
    }

    #[test]
    fn reset_is_deterministic_for_fixed_placement() {
        let cfg = PixelGridConfig::new(100, 0, false);
        let mut env = PixelGridEnv::with_config(cfg, false).expect("valid config");
        let snap = env.reset().unwrap();
        assert_eq!(snap.status(), EpisodeStatus::Running);
        assert_eq!(env.state().agent(), 0);
        assert_eq!(env.state().goal(), 24);
    }

    #[test]
    fn random_placement_is_distinct_and_seeded() {
        let cfg = PixelGridConfig::new(100, 42, true);
        let mut a = PixelGridEnv::with_config(cfg, false).expect("valid config");
        let mut b = PixelGridEnv::with_config(cfg, false).expect("valid config");
        a.reset().unwrap();
        b.reset().unwrap();
        assert_ne!(a.state().agent(), a.state().goal());
        // Same seed → same placement.
        assert_eq!(a.state().agent(), b.state().agent());
        assert_eq!(a.state().goal(), b.state().goal());
    }

    #[test]
    fn wall_clamping_holds_position_at_edges() {
        // Agent at cell 0 (top-left). Up and Left are no-ops.
        let mut state = PixelGridState::new(0, 24);
        state.apply_move(PixelGridAction::Up);
        assert_eq!(state.agent(), 0);
        state.apply_move(PixelGridAction::Left);
        assert_eq!(state.agent(), 0);
        // Down moves to row 1 (cell 5); Right moves to col 1 (cell 1).
        let mut state = PixelGridState::new(0, 24);
        state.apply_move(PixelGridAction::Down);
        assert_eq!(state.agent(), 5);
        let mut state = PixelGridState::new(0, 24);
        state.apply_move(PixelGridAction::Right);
        assert_eq!(state.agent(), 1);
        // Agent at cell 24 (bottom-right). Down and Right are no-ops.
        let mut state = PixelGridState::new(24, 0);
        state.apply_move(PixelGridAction::Down);
        assert_eq!(state.agent(), 24);
        state.apply_move(PixelGridAction::Right);
        assert_eq!(state.agent(), 24);
    }

    #[test]
    fn reaching_goal_terminates_with_positive_reward() {
        // Place agent adjacent to goal: agent cell 23, goal cell 24 → one Right.
        let cfg = PixelGridConfig::new(100, 0, false);
        let mut env = PixelGridEnv::with_config(cfg, false).expect("valid config");
        env.reset().unwrap();
        // Override placement for a one-step solve.
        env.state = PixelGridState::new(23, 24);
        let snap = env.step(PixelGridAction::Right).unwrap();
        assert!(snap.is_done());
        assert_eq!(snap.status(), EpisodeStatus::Terminated);
        let reward: f32 = (*snap.reward()).into();
        assert!(reward > 0.0, "goal reward must be positive, got {reward}");
        // 1 step / 100 max → 1 - 0.9 * 0.01 = 0.991.
        assert!((reward - 0.991).abs() < 1e-4, "reward was {reward}");
    }

    #[test]
    fn step_limit_truncates_with_zero_reward() {
        // Goal unreachable in 3 steps from a fixed corner if we move away/idle.
        let cfg = PixelGridConfig::new(3, 0, false);
        let mut env = PixelGridEnv::with_config(cfg, false).expect("valid config");
        env.reset().unwrap();
        // Bump the top-left corner: Up is a no-op, never reaching the goal.
        env.step(PixelGridAction::Up).unwrap();
        env.step(PixelGridAction::Up).unwrap();
        let snap = env.step(PixelGridAction::Up).unwrap();
        assert!(snap.is_done());
        assert_eq!(snap.status(), EpisodeStatus::Truncated);
        let reward: f32 = (*snap.reward()).into();
        assert_eq!(reward, 0.0);
    }

    #[test]
    fn snapshot_observation_is_rank3_over_rank1_state() {
        let cfg = PixelGridConfig::new(100, 0, false);
        let mut env = PixelGridEnv::with_config(cfg, false).expect("valid config");
        let snap = env.reset().unwrap();
        assert_eq!(snap.observation().pixels().len(), PIXEL_COUNT);
        assert_eq!(<PixelObservation as Observation<3>>::shape(), [20, 20, 3]);
        assert_eq!(<PixelGridState as State<1>>::shape(), [2]);
    }

    #[test]
    fn display_contains_step_budget() {
        let env = PixelGridEnv::with_config(PixelGridConfig::new(50, 0, false), false).expect("valid config");
        let s = env.to_string();
        assert!(s.contains("PixelGridEnv"));
        assert!(s.contains("50"));
    }
}
