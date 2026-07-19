//! `GoToDoor`: instruction-conditioned navigation to a colored door.
//!
//! Ports Farama Minigrid's [`GoToDoorEnv`]. Four doors are embedded in the four
//! perimeter walls of an otherwise empty room. **Every episode re-samples the
//! four door colors** (4 distinct colors drawn from the 6-color palette) and
//! then picks one of those four doors as the target. The agent solves the task
//! by issuing [`GridAction::Done`] while facing a door of the target color;
//! doing so anywhere else — including at a door of a different color —
//! terminates the episode with reward `0.0`.
//!
//! ## The mission is observable
//!
//! The instruction is carried **in the observation**, not merely on the env: a
//! [`GoToDoorObservation`] is a `7 × 7 × 4` egocentric view whose first three
//! channels are the usual entity encoding (`type`, `color`, `state` — identical
//! to [`GridObservation`](super::core::GridObservation)) and whose **fourth
//! channel is the mission's color byte, broadcast to every cell**.
//!
//! The mission byte is deliberately the *same* encoding as the perceived door
//! color in channel 1 ([`Color::to_u8`], an ordinal byte in `1..=6`). Sharing the
//! encoding is what lets a policy learn *equality* between "the color I was told"
//! and "the color of the door I can see" — the single comparison this task exists
//! to test. A one-hot mission channel compared against an ordinal perceived color
//! would be an encoding mismatch on the two sides of that comparison.
//!
//! Because the color↔wall mapping is re-sampled every episode, a policy cannot
//! shortcut the perception step ("mission is Red ⇒ walk north"); it must look at
//! the doors. Because the mission rides in channel 3, a policy that ignores that
//! channel is information-theoretically capped at 25% success — the four target
//! hypotheses are otherwise indistinguishable.
//!
//! ## Layout (6 × 6 default)
//!
//! ```text
//! # # # ? # #    ? = a door, one per wall, at the wall midpoint.
//! # . . . . #        The four door colors are re-sampled every episode
//! # . A . . #        (4 distinct colors out of 6) — there is NO fixed
//! ? . . . . ?        wall→color mapping.
//! # . . . . #
//! # # # ? # #    A = agent, start (2, 2) facing East    # = wall
//! ```
//!
//! | Observation | `7 × 7 × 4`: `[type, color, state, mission_color]` per cell     |
//! |-------------|-----------------------------------------------------------------|
//! | Action      | `TurnLeft`, `TurnRight`, `Forward`, `Done`                      |
//! | Reward      | `success_reward(steps, max_steps)` on correct `Done`; else `0.0` |
//!
//! ## Deliberate deviations from canonical Minigrid
//!
//! - **Door positions and the agent's start pose are fixed** (wall midpoints;
//!   agent at `(2, 2)` facing East). Canonical resamples both every episode.
//!   They affect generalization difficulty, not correctness — the perception and
//!   instruction-conditioning properties are already forced by the per-episode
//!   color resampling. Deferred to a follow-up issue.
//! - **Success requires *facing* a target-colored door** when `Done` is issued.
//!   Canonical accepts orthogonal adjacency with any facing. rlevo's criterion is
//!   strictly harder and unambiguous about which door the agent means.
//!
//! # Examples
//!
//! ```rust
//! use rlevo_environments::grids::go_to_door::{GoToDoorConfig, GoToDoorEnv};
//! use rlevo_core::environment::{ConstructableEnv, Environment};
//!
//! let cfg = GoToDoorConfig::new(6, 100, 0);
//! let mut env = GoToDoorEnv::with_config(cfg, false).expect("valid config");
//! let snap = env.reset().unwrap();
//! println!("Mission: {}", env.mission().describe());
//! ```
//!
//! [`GoToDoorEnv`]: https://minigrid.farama.org/environments/minigrid/GoToDoorEnv/

use super::core::{
    action::GridAction,
    agent::AgentState,
    color::Color,
    direction::Direction,
    dynamics::{StepOutcome, apply_action},
    entity::{DoorState, Entity},
    grid::{Grid, egocentric_view},
    observation::VIEW_SIZE,
    render::render_ascii,
    reward::success_reward,
    state::GridState,
};
use burn::tensor::{Tensor, backend::Backend};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rlevo_core::base::{HostRow, Observation, TensorConversionError, TensorConvertible};
use rlevo_core::config::{self, ConfigError, ConstraintKind, Validate};
use rlevo_core::environment::{
    ConstructableEnv, Environment, EnvironmentError, Sensor, SnapshotBase,
};
use rlevo_core::reward::ScalarReward;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

/// Minimum side length; we need at least one interior cell and one
/// central slot on each perimeter wall for a door.
const MIN_SIZE: usize = 5;

/// Rejection text for `size < MIN_SIZE`. Names the *cause*, not just the bound.
///
/// Below `MIN_SIZE` the four wall-midpoint door slots stop being four distinct
/// cells — at `size == 2` the East and South doors collide, and at `size == 1`
/// all four land on `(0, 0)` and three are silently overwritten. The env's
/// "exactly one door carries the mission color" invariant, [`GoToDoorEnv::doors`],
/// and the agent's start pose all break as a result, so the config is refused
/// rather than built.
const SIZE_BELOW_MIN: &str = "GoToDoorEnv requires size >= 5: smaller grids cannot host four \
                              distinct wall-midpoint doors plus an in-bounds agent start";

/// Number of doors: exactly one per perimeter wall.
///
/// Public because it is the array length in [`GoToDoorEnv::doors`]'s signature.
pub const DOOR_COUNT: usize = 4;

/// Index of the mission channel within a [`GoToDoorObservation`] cell.
///
/// Channels `0..MISSION_CHANNEL` are the shared entity encoding
/// (`type`, `color`, `state`); this channel holds the mission color byte.
pub const MISSION_CHANNEL: usize = super::core::OBS_CHANNELS;

/// Per-cell channel count of a [`GoToDoorObservation`]: the three shared
/// entity-encoding channels plus the mission channel.
pub const GO_TO_DOOR_OBS_CHANNELS: usize = super::core::OBS_CHANNELS + 1;

/// Snapshot type produced by [`GoToDoorEnv`].
///
/// A plain [`SnapshotBase`] over the 4-channel [`GoToDoorObservation`] (ADR
/// 0042: `SnapshotBase` is the only `Snapshot` implementation in the workspace;
/// a family that needs a named snapshot shape defines a local alias).
pub type GoToDoorSnapshot = SnapshotBase<3, GoToDoorObservation, ScalarReward>;

/// Instruction the agent must fulfil in a given episode.
///
/// Sampled at every [`reset`](Environment::reset) as the color of one uniformly
/// chosen door (see [`GoToDoorEnv`]), exposed via [`GoToDoorEnv::mission`], and
/// — crucially — broadcast into channel [`MISSION_CHANNEL`] of every
/// [`GoToDoorObservation`] cell so the policy can actually read it.
///
/// # Examples
///
/// ```rust
/// use rlevo_environments::grids::go_to_door::Mission;
/// use rlevo_environments::grids::core::color::Color;
///
/// let m = Mission::new(Color::Blue);
/// assert_eq!(m.target_color, Color::Blue);
/// assert!(m.describe().contains("Blue"));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Mission {
    /// Color of the door the agent must face when emitting [`GridAction::Done`].
    pub target_color: Color,
}

impl Mission {
    /// Constructs a [`Mission`] targeting the given door color.
    #[must_use]
    pub const fn new(target_color: Color) -> Self {
        Self { target_color }
    }

    /// Short plain-text rendering of the mission.
    #[must_use]
    pub fn describe(&self) -> String {
        format!("go to the {:?} door", self.target_color)
    }

    /// The mission color as the byte stamped into [`MISSION_CHANNEL`].
    ///
    /// Same encoding as the perceived door color in channel 1, so a policy can
    /// compare the two directly.
    #[must_use]
    pub const fn color_u8(&self) -> u8 {
        self.target_color.to_u8()
    }
}

/// `7 × 7 × 4` egocentric observation carrying the episode mission.
///
/// Bespoke to [`GoToDoorEnv`]: the shared
/// [`GridObservation`](super::core::GridObservation) has no room for an
/// instruction, and widening it would change all twelve grid environments.
/// `Environment::ObservationType` is per-env by design, so the extra channel
/// stays contained here.
///
/// | Channel            | Meaning                                                     |
/// |--------------------|-------------------------------------------------------------|
/// | 0                  | Entity type ([`Entity::type_u8`])                            |
/// | 1                  | Perceived color ([`Entity::color_u8`], `0` if no color)      |
/// | 2                  | Door state ([`Entity::state_u8`], `0` if no state)           |
/// | 3 ([`MISSION_CHANNEL`]) | Mission color byte ([`Color::to_u8`]), same in every cell |
///
/// Channels 0-2 are byte-identical to the shared grid encoding; only channel 3
/// is new. The mission is broadcast to every cell (rather than to a single
/// corner) so a convolutional policy sees it at every spatial position where it
/// might perceive a door.
///
/// # Examples
///
/// ```rust
/// use rlevo_environments::grids::go_to_door::{GoToDoorConfig, GoToDoorEnv, MISSION_CHANNEL};
/// use rlevo_core::environment::{Environment, Snapshot};
///
/// let mut env = GoToDoorEnv::with_config(GoToDoorConfig::new(6, 100, 7), false).unwrap();
/// let snap = env.reset().unwrap();
/// let mission_byte = env.mission().color_u8();
/// assert_eq!(snap.observation().view[0][0][MISSION_CHANNEL], mission_byte);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GoToDoorObservation {
    /// Encoded view, indexed as `view[row][col][channel]`.
    pub view: [[[u8; GO_TO_DOOR_OBS_CHANNELS]; VIEW_SIZE]; VIEW_SIZE],
    /// Agent's current facing direction, encoded via [`Direction::to_u8`].
    ///
    /// Like [`GridObservation`](super::core::GridObservation), this field is
    /// **not** included in the tensor produced by
    /// [`TensorConvertible::to_tensor`]; round-tripping through a tensor resets
    /// it to [`Direction::North`]. Carry the direction out-of-band if full
    /// fidelity is required.
    pub agent_direction: u8,
}

impl GoToDoorObservation {
    /// Encode a decoded `7 × 7` entity view, the agent's facing, and the episode
    /// mission color into an observation.
    ///
    /// Channels 0-2 come from [`Entity::type_u8`] / [`Entity::color_u8`] /
    /// [`Entity::state_u8`]; channel [`MISSION_CHANNEL`] is `mission.to_u8()` in
    /// every cell.
    #[must_use]
    pub fn from_entity_view(
        view: [[Entity; VIEW_SIZE]; VIEW_SIZE],
        direction: Direction,
        mission: Color,
    ) -> Self {
        let mission_byte = mission.to_u8();
        let mut encoded = [[[0u8; GO_TO_DOOR_OBS_CHANNELS]; VIEW_SIZE]; VIEW_SIZE];
        for (r, row) in view.iter().enumerate() {
            for (c, cell) in row.iter().enumerate() {
                encoded[r][c] = [
                    cell.type_u8(),
                    cell.color_u8(),
                    cell.state_u8(),
                    mission_byte,
                ];
            }
        }
        Self {
            view: encoded,
            agent_direction: direction.to_u8(),
        }
    }

    /// The mission color byte carried by this observation.
    ///
    /// Reads cell `(0, 0)`; the value is identical in every cell by construction.
    #[must_use]
    pub const fn mission_color_u8(&self) -> u8 {
        self.view[0][0][MISSION_CHANNEL]
    }
}

impl Observation<3> for GoToDoorObservation {
    fn shape() -> [usize; 3] {
        [VIEW_SIZE, VIEW_SIZE, GO_TO_DOOR_OBS_CHANNELS]
    }
}

impl HostRow<3> for GoToDoorObservation {
    fn row_shape() -> [usize; 3] {
        [VIEW_SIZE, VIEW_SIZE, GO_TO_DOOR_OBS_CHANNELS]
    }

    fn write_host_row(&self, buf: &mut Vec<f32>) {
        for row in &self.view {
            for cell in row {
                for &channel in cell {
                    buf.push(f32::from(channel));
                }
            }
        }
    }
}

impl<B: Backend> TensorConvertible<3, B> for GoToDoorObservation {
    /// Reconstructs the `7 × 7 × 4` view from a tensor.
    ///
    /// The tensor contains only the view channels (including the mission
    /// channel). `agent_direction` is not encoded and is defaulted to
    /// [`Direction::North`]; callers that need round-trip fidelity for the
    /// direction must carry it out-of-band.
    ///
    /// # Errors
    ///
    /// Returns [`TensorConversionError`] if the tensor shape does not equal
    /// `[VIEW_SIZE, VIEW_SIZE, GO_TO_DOOR_OBS_CHANNELS]`, if the backend fails to
    /// materialize its data, or if any value is outside the `u8` range.
    fn from_tensor(tensor: Tensor<B, 3>) -> Result<Self, TensorConversionError> {
        let dims = tensor.dims();
        if dims.as_slice() != [VIEW_SIZE, VIEW_SIZE, GO_TO_DOOR_OBS_CHANNELS] {
            return Err(TensorConversionError {
                message: format!(
                    "expected shape [{VIEW_SIZE}, {VIEW_SIZE}, {GO_TO_DOOR_OBS_CHANNELS}], got {dims:?}"
                ),
            });
        }
        let flat = tensor
            .into_data()
            .into_vec::<f32>()
            .map_err(|e| TensorConversionError {
                message: format!("failed to read tensor data: {e:?}"),
            })?;
        let mut view = [[[0u8; GO_TO_DOOR_OBS_CHANNELS]; VIEW_SIZE]; VIEW_SIZE];
        let mut idx = 0;
        for row in &mut view {
            for cell in row {
                for channel in cell {
                    let value = flat[idx];
                    if !value.is_finite() || value < 0.0 || value > f32::from(u8::MAX) {
                        return Err(TensorConversionError {
                            message: format!("value at index {idx} out of u8 range: {value}"),
                        });
                    }
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    {
                        *channel = value as u8;
                    }
                    idx += 1;
                }
            }
        }
        Ok(Self {
            view,
            agent_direction: Direction::North.to_u8(),
        })
    }
}

/// Configuration for [`GoToDoorEnv`].
///
/// There is deliberately **no** `target_color` field: pinning the mission in
/// config is what made every episode identical (issue #109). The target — and
/// the four door colors it is drawn from — are sampled from the environment's
/// persistent RNG at every [`reset`](Environment::reset).
///
/// # Examples
///
/// ```rust
/// use rlevo_environments::grids::go_to_door::GoToDoorConfig;
///
/// let cfg = GoToDoorConfig::new(6, 100, 0);
/// assert_eq!(cfg.size, 6);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoToDoorConfig {
    /// Grid side length in cells (width = height = `size`); must be ≥ `MIN_SIZE` (5).
    pub size: usize,
    /// Maximum steps before the episode times out with reward `0.0`.
    pub max_steps: usize,
    /// Seed for the environment's persistent RNG, drawn once at construction.
    ///
    /// The RNG **advances** across resets (ADR 0029), so successive episodes get
    /// independent door colors and targets. A fixed seed therefore reproduces a
    /// fixed *sequence* of episodes, not one repeated episode. For bit-for-bit
    /// replay of a single episode use [`GoToDoorEnv::reset_with_seed`].
    pub seed: u64,
}

impl GoToDoorConfig {
    /// Creates a [`GoToDoorConfig`] with the given parameters.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rlevo_environments::grids::go_to_door::GoToDoorConfig;
    ///
    /// let cfg = GoToDoorConfig::new(8, 200, 42);
    /// assert_eq!(cfg.seed, 42);
    /// ```
    #[must_use]
    pub const fn new(size: usize, max_steps: usize, seed: u64) -> Self {
        Self {
            size,
            max_steps,
            seed,
        }
    }
}

impl Default for GoToDoorConfig {
    fn default() -> Self {
        let size = 6;
        Self {
            size,
            max_steps: 4 * size * size,
            seed: 0,
        }
    }
}

impl Validate for GoToDoorConfig {
    /// Rejects any `size` below `MIN_SIZE` (5) and a zero `max_steps`.
    ///
    /// The `size` guard lives **here**, not only in [`FromStr`]: `GoToDoorConfig`
    /// derives `Deserialize`, so a config loaded from a file is user-supplied
    /// runtime data that never passes through `from_str` (rules.md §4 — "if an
    /// invalid value can arrive via `Deserialize`, it must be an `Err`").
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "GoToDoorConfig";
        if self.size < MIN_SIZE {
            return Err(ConfigError {
                config: C,
                field: "size",
                kind: ConstraintKind::Custom(SIZE_BELOW_MIN),
            });
        }
        config::nonzero(C, "max_steps", self.max_steps)?;
        Ok(())
    }
}

impl FromStr for GoToDoorConfig {
    type Err = String;

    /// Parses `"size=6,max_steps=100,seed=0"` (keys in any order) or the
    /// positional form `"6,100,0"`.
    ///
    /// # Errors
    ///
    /// Returns the offending key/value, or the [`Validate`] rejection — the same
    /// guard [`GoToDoorEnv::with_config`] applies, so this parser cannot admit a
    /// config that construction would refuse.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut cfg = Self::default();
        for (idx, raw) in s.trim().split(',').map(str::trim).enumerate() {
            if raw.is_empty() {
                continue;
            }
            if let Some((key, value)) = raw.split_once('=') {
                match key.trim() {
                    "size" => cfg.size = value.trim().parse().map_err(|e| format!("size: {e}"))?,
                    "max_steps" => {
                        cfg.max_steps = value
                            .trim()
                            .parse()
                            .map_err(|e| format!("max_steps: {e}"))?;
                    }
                    "seed" => cfg.seed = value.trim().parse().map_err(|e| format!("seed: {e}"))?,
                    other => return Err(format!("unknown key `{other}`")),
                }
            } else {
                match idx {
                    0 => cfg.size = raw.parse().map_err(|e| format!("size: {e}"))?,
                    1 => cfg.max_steps = raw.parse().map_err(|e| format!("max_steps: {e}"))?,
                    2 => cfg.seed = raw.parse().map_err(|e| format!("seed: {e}"))?,
                    _ => return Err(format!("unexpected positional value `{raw}`")),
                }
            }
        }
        cfg.validate()
            .map_err(|e| format!("{e} (got size={})", cfg.size))?;
        Ok(cfg)
    }
}

/// Minigrid's `GoToDoor` environment.
///
/// Each episode places one door at the midpoint of each perimeter wall, colors
/// the four doors with **4 distinct colors freshly sampled** from
/// [`Color::ALL`], and adopts the color of one uniformly chosen door as the
/// [`Mission`]. The agent must issue [`GridAction::Done`] while facing a door of
/// that color.
///
/// The mission is delivered to the policy in channel [`MISSION_CHANNEL`] of
/// every [`GoToDoorObservation`] cell — this is the only place it appears, so a
/// policy that ignores that channel cannot beat 25%.
///
/// Implements [`Environment<3, 3, 1>`] with [`GridState`] /
/// [`GoToDoorObservation`] / [`GridAction`] / [`ScalarReward`].
///
/// # Examples
///
/// ```rust
/// use rlevo_environments::grids::go_to_door::GoToDoorEnv;
/// use rlevo_core::environment::{ConstructableEnv, Environment};
///
/// let mut env = GoToDoorEnv::new(false);
/// let snap = env.reset().unwrap();
/// println!("Mission: {}", env.mission().describe());
/// ```
#[derive(Debug)]
pub struct GoToDoorEnv {
    state: GridState,
    config: GoToDoorConfig,
    steps: usize,
    render: bool,
    mission: Mission,
    /// The four doors of the current episode as `(x, y, color)`, in
    /// North / East / South / West wall order.
    doors: [(i32, i32, Color); DOOR_COUNT],
    rng: StdRng,
}

impl GoToDoorEnv {
    /// Constructs a [`GoToDoorEnv`] from an explicit configuration.
    ///
    /// Seeds the persistent RNG once from `config.seed` and immediately builds a
    /// first episode (door colors + mission) from it. Call
    /// [`Environment::reset`] before the first [`Environment::step`] to obtain
    /// the first observation; that reset samples a *fresh* episode, advancing the
    /// stream.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rlevo_environments::grids::go_to_door::{GoToDoorConfig, GoToDoorEnv};
    ///
    /// let env = GoToDoorEnv::with_config(
    ///     GoToDoorConfig::new(6, 100, 0),
    ///     true, // render ASCII grid to stdout
    /// )
    /// .expect("valid config");
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `config` fails [`Validate`]: `size` below
    /// `MIN_SIZE` (5) — below which the four wall-midpoint doors would collide on
    /// the same cells and the agent would start out of bounds — or a zero
    /// `max_steps`. This is the construction chokepoint (rules.md §4), so it also
    /// rejects a config that arrived by `Deserialize` or struct-update syntax
    /// without passing through [`FromStr`].
    pub fn with_config(config: GoToDoorConfig, render: bool) -> Result<Self, ConfigError> {
        config.validate()?;
        let mut rng = StdRng::seed_from_u64(config.seed);
        let (state, doors, mission) = Self::build(&config, &mut rng);
        Ok(Self {
            state,
            config,
            steps: 0,
            render,
            mission,
            doors,
            rng,
        })
    }

    /// Re-seed the persistent RNG to `seed`, then [`reset`](Environment::reset).
    ///
    /// Ordinary [`reset`](Environment::reset) advances the persistent stream so
    /// successive episodes get independent door colors and missions (ADR 0029);
    /// use this when you need a *specific* episode to reproduce bit-for-bit (e.g.
    /// replaying a failure). Run-level reproducibility is already guaranteed by
    /// the construction seed.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`reset`](Environment::reset) (currently none).
    pub fn reset_with_seed(&mut self, seed: u64) -> Result<GoToDoorSnapshot, EnvironmentError> {
        self.rng = StdRng::seed_from_u64(seed);
        self.reset()
    }

    /// Returns the environment's active configuration.
    #[must_use]
    pub const fn config(&self) -> &GoToDoorConfig {
        &self.config
    }

    /// Returns the number of steps taken since the last reset.
    #[must_use]
    pub const fn steps(&self) -> usize {
        self.steps
    }

    /// Returns a reference to the current grid state.
    #[must_use]
    pub const fn state(&self) -> &GridState {
        &self.state
    }

    /// Returns the episode mission specifying the target door color.
    ///
    /// The same information reaches the policy through channel
    /// [`MISSION_CHANNEL`] of every observation cell; this accessor is for
    /// logging, scripted oracles, and tests.
    #[must_use]
    pub const fn mission(&self) -> &Mission {
        &self.mission
    }

    /// The four doors of the current episode as `(x, y, color)`, in
    /// North / East / South / West wall order.
    ///
    /// Colors are re-sampled every [`reset`](Environment::reset) and are always
    /// four *distinct* members of [`Color::ALL`]; exactly one of them equals
    /// [`Mission::target_color`].
    #[must_use]
    pub const fn doors(&self) -> &[(i32, i32, Color); DOOR_COUNT] {
        &self.doors
    }

    /// Renders the current grid state as an ASCII string.
    #[must_use]
    pub fn ascii(&self) -> String {
        render_ascii(&self.state.grid, &self.state.agent)
    }

    /// Build a fresh episode: an empty walled room, four doors with four
    /// distinct freshly-sampled colors, and a mission naming one of them.
    ///
    /// Draws from `rng` and lets it advance (ADR 0029) — never re-seeds.
    fn build(
        config: &GoToDoorConfig,
        rng: &mut StdRng,
    ) -> (GridState, [(i32, i32, Color); DOOR_COUNT], Mission) {
        let mut grid = Grid::new(config.size, config.size);
        grid.draw_walls();
        #[allow(clippy::cast_possible_wrap)]
        let size = config.size as i32;
        let mid_x = size / 2;
        let mid_y = size / 2;

        // Rejection-sample four distinct colors, as canonical Minigrid does.
        let colors = Self::sample_door_colors(rng);

        // One door per wall, at the wall midpoint: North, East, South, West.
        let positions = [(mid_x, 0), (size - 1, mid_y), (mid_x, size - 1), (0, mid_y)];
        let mut doors = [(0, 0, Color::Red); DOOR_COUNT];
        for (slot, (&(x, y), &color)) in doors.iter_mut().zip(positions.iter().zip(colors.iter())) {
            grid.set(x, y, Entity::Door(color, DoorState::Closed));
            *slot = (x, y, color);
        }

        // The mission is the color of one uniformly chosen door.
        let target = doors[rng.random_range(0..DOOR_COUNT)].2;

        let agent_pos = (mid_x - 1).max(1);
        let agent = AgentState::new(agent_pos, agent_pos, Direction::East);
        (GridState::new(grid, agent), doors, Mission::new(target))
    }

    /// Rejection-sample `DOOR_COUNT` distinct colors from [`Color::ALL`].
    ///
    /// Terminates with probability 1: the palette has 6 colors and only 4 are
    /// drawn, so at every draw at least 2 of 6 candidates are still fresh.
    fn sample_door_colors(rng: &mut StdRng) -> [Color; DOOR_COUNT] {
        let mut chosen: Vec<Color> = Vec::with_capacity(DOOR_COUNT);
        while chosen.len() < DOOR_COUNT {
            let candidate = Color::ALL[rng.random_range(0..Color::ALL.len())];
            if chosen.contains(&candidate) {
                continue;
            }
            chosen.push(candidate);
        }
        let mut out = [Color::Red; DOOR_COUNT];
        out.copy_from_slice(&chosen);
        out
    }

    /// Project a grid `state` into a mission-carrying observation.
    ///
    /// The shared body behind the env-side [`Sensor`] impl: it stamps the
    /// current [`Mission::target_color`] into the observation's mission channel,
    /// which is why the emission model lives on the environment rather than on
    /// [`GridState`] (the mission is env context, not state).
    fn observe_impl(&self, state: &GridState) -> GoToDoorObservation {
        let view = egocentric_view(&state.grid, &state.agent);
        GoToDoorObservation::from_entity_view(
            view,
            state.agent.direction,
            self.mission.target_color,
        )
    }

    fn emit(&self, observation: GoToDoorObservation, reward: f32, done: bool) -> GoToDoorSnapshot {
        if self.render {
            println!("{}", self.ascii());
        }
        let reward = ScalarReward::new(reward);
        if done {
            SnapshotBase::terminated(observation, reward)
        } else {
            SnapshotBase::running(observation, reward)
        }
    }

    /// Color of the door currently in front of the agent, if any.
    fn door_in_front_color(&self) -> Option<Color> {
        let (fx, fy) = self.state.agent.front();
        match self.state.grid.get(fx, fy) {
            Entity::Door(color, _) => Some(color),
            _ => None,
        }
    }
}

impl crate::render::AsciiRenderable for GoToDoorEnv {
    fn render_ascii(&self) -> String {
        render_ascii(&self.state.grid, &self.state.agent)
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        super::core::render::render_styled(&self.state.grid, &self.state.agent)
    }
}

impl Display for GoToDoorEnv {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GoToDoorEnv(target={:?}, step={}/{})",
            self.mission.target_color, self.steps, self.config.max_steps
        )
    }
}

impl ConstructableEnv for GoToDoorEnv {
    fn new(render: bool) -> Self {
        Self::with_config(GoToDoorConfig::default(), render).expect("default config must validate")
    }
}

impl Sensor<3, 1, 3> for GoToDoorEnv {
    type Action = GridAction;
    type State = GridState;
    type Observation = GoToDoorObservation;

    /// Emission model `O(a, s')`. The observation ignores the action — it is a
    /// function of the resulting `next_state` and the episode mission — so this
    /// forwards to the same projection as [`observe_reset`](Self::observe_reset).
    fn observe(&self, _action: &GridAction, next_state: &GridState) -> GoToDoorObservation {
        self.observe_impl(next_state)
    }

    fn observe_reset(&self, state: &GridState) -> GoToDoorObservation {
        self.observe_impl(state)
    }
}

impl Environment<3, 3, 1> for GoToDoorEnv {
    type StateType = GridState;
    type ObservationType = GoToDoorObservation;
    type ActionType = GridAction;
    type RewardType = ScalarReward;
    type SnapshotType = GoToDoorSnapshot;

    /// Samples a fresh episode from the **persistent** RNG stream.
    ///
    /// Per ADR 0029 the stream is *not* re-seeded here: successive resets draw
    /// independent door colors and missions. Use
    /// [`reset_with_seed`](Self::reset_with_seed) for deterministic replay.
    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        let (state, doors, mission) = Self::build(&self.config, &mut self.rng);
        self.state = state;
        self.doors = doors;
        self.mission = mission;
        self.steps = 0;
        let observation = self.observe_reset(&self.state);
        Ok(self.emit(observation, 0.0, false))
    }

    fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> {
        self.steps += 1;
        let outcome = apply_action(&mut self.state.grid, &mut self.state.agent, action);
        let (reward, done) = match outcome {
            StepOutcome::DoneAction => {
                if self.door_in_front_color() == Some(self.mission.target_color) {
                    (success_reward(self.steps, self.config.max_steps), true)
                } else {
                    (0.0, true)
                }
            }
            _ => {
                let done = self.steps >= self.config.max_steps;
                (0.0, done)
            }
        };
        let observation = self.observe(&action, &self.state);
        Ok(self.emit(observation, reward, done))
    }
}

impl rlevo_core::render::payload::GridPayloadSource for GoToDoorEnv {
    fn grid_snapshot(&self) -> rlevo_core::render::payload::GridSnapshot {
        crate::grids::core::render::grid_snapshot(&self.state.grid, &self.state.agent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlevo_core::environment::Snapshot;
    use std::collections::HashSet;

    /// Wall order used by `GoToDoorEnv::doors` and by the scripts below.
    const NORTH: usize = 0;
    const EAST: usize = 1;
    const SOUTH: usize = 2;
    const WEST: usize = 3;

    fn env_6x6(seed: u64) -> GoToDoorEnv {
        GoToDoorEnv::with_config(GoToDoorConfig::new(6, 100, seed), false).expect("valid config")
    }

    /// Scripted route from the fixed start pose `(2, 2)` facing East to a pose
    /// *facing* the door on the given wall of a 6 × 6 grid, ending in `Done`.
    fn script_for(wall: usize) -> &'static [GridAction] {
        use GridAction::{Done, Forward, TurnLeft, TurnRight};
        match wall {
            // (2,2)E → (3,2) → face N → (3,1); door at (3,0) in front.
            NORTH => &[Forward, TurnLeft, Forward, Done],
            // (2,2)E → (3,2) → (4,2) → face S → (4,3) → face E; door at (5,3).
            EAST => &[Forward, Forward, TurnRight, Forward, TurnLeft, Done],
            // (2,2)E → (3,2) → face S → (3,3) → (3,4); door at (3,5) in front.
            SOUTH => &[Forward, TurnRight, Forward, Forward, Done],
            // (2,2)E → face S → (2,3) → face W → (1,3); door at (0,3) in front.
            WEST => &[TurnRight, Forward, TurnRight, Forward, Done],
            _ => unreachable!("wall index must be 0..4"),
        }
    }

    /// Run a script to completion and return the terminal reward.
    fn run(env: &mut GoToDoorEnv, script: &[GridAction]) -> f32 {
        let mut last = None;
        for &a in script {
            last = Some(env.step(a).expect("step must succeed"));
        }
        let snap = last.expect("script must be non-empty");
        assert!(
            snap.is_done(),
            "a Done-terminated script must end the episode"
        );
        (*snap.reward()).into()
    }

    /// Index of the wall whose door carries the mission color, read *only* from
    /// the observation's mission channel plus the perceived door colors.
    fn target_wall(env: &GoToDoorEnv, obs: &GoToDoorObservation) -> usize {
        let mission_byte = obs.mission_color_u8();
        let matches: Vec<usize> = env
            .doors()
            .iter()
            .enumerate()
            .filter(|&(_, &(_, _, color))| color.to_u8() == mission_byte)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            matches.len(),
            1,
            "exactly one door must carry the mission color"
        );
        matches[0]
    }

    // ---------------------------------------------------------------- config

    #[test]
    fn test_config_default_validates() {
        assert!(
            GoToDoorConfig::default().validate().is_ok(),
            "the library default config must be valid"
        );
    }

    #[test]
    fn test_config_rejects_zero_size() {
        let bad = GoToDoorConfig {
            size: 0,
            ..Default::default()
        };
        assert!(
            GoToDoorEnv::with_config(bad, false).is_err(),
            "zero size must be rejected by Validate"
        );
    }

    #[test]
    fn test_config_validate_rejects_size_below_min() {
        // A `Deserialize`d `{"size": 1, ...}` never touches `FromStr`, so the
        // guard has to live in `Validate` (rules.md §4). At size 1 all four doors
        // land on (0, 0) and three are silently overwritten by `grid.set`.
        for size in 0..MIN_SIZE {
            let bad = GoToDoorConfig {
                size,
                ..Default::default()
            };
            let err = bad
                .validate()
                .expect_err("size below MIN_SIZE must fail validate()");
            assert_eq!(
                err.field, "size",
                "the rejection must name the offending field"
            );
        }
    }

    #[test]
    fn test_with_config_rejects_size_below_min() {
        let bad = GoToDoorConfig::new(1, 100, 0);
        assert!(
            GoToDoorEnv::with_config(bad, false).is_err(),
            "construction must refuse a sub-MIN_SIZE grid rather than build \
             four colliding doors and an out-of-bounds agent"
        );
    }

    #[test]
    fn test_config_default_is_6x6() {
        let cfg = GoToDoorConfig::default();
        assert_eq!(cfg.size, 6, "default grid is 6 x 6");
        assert_eq!(cfg.max_steps, 144, "default budget is 4 * size^2");
    }

    #[test]
    fn test_config_fromstr_parses_keys() {
        let cfg: GoToDoorConfig = "size=7,max_steps=50,seed=3".parse().expect("valid spec");
        assert_eq!(cfg.size, 7);
        assert_eq!(cfg.max_steps, 50);
        assert_eq!(cfg.seed, 3);
    }

    #[test]
    fn test_config_fromstr_parses_positional() {
        let cfg: GoToDoorConfig = "7, 50, 3".parse().expect("valid spec");
        assert_eq!((cfg.size, cfg.max_steps, cfg.seed), (7, 50, 3));
    }

    #[test]
    fn test_config_fromstr_rejects_target_color_key() {
        // The mission is no longer configurable — the key must not resolve.
        assert!(
            "target_color=blue".parse::<GoToDoorConfig>().is_err(),
            "target_color is no longer a config key"
        );
        assert!(
            "color=blue".parse::<GoToDoorConfig>().is_err(),
            "color is no longer a config key"
        );
    }

    #[test]
    fn test_config_fromstr_rejects_small_size() {
        assert!(
            "3".parse::<GoToDoorConfig>().is_err(),
            "size below MIN_SIZE must be rejected"
        );
    }

    // ------------------------------------------------------------ generation

    #[test]
    fn test_build_places_four_doors_with_distinct_colors() {
        let env = env_6x6(0);
        let mut seen = HashSet::new();
        for &(x, y, color) in env.doors() {
            assert!(
                matches!(env.state().grid.get(x, y), Entity::Door(c, DoorState::Closed) if c == color),
                "door at ({x}, {y}) must be a closed door of the recorded color"
            );
            assert!(
                seen.insert(color),
                "door colors must be distinct: {color:?}"
            );
        }
        assert_eq!(seen.len(), DOOR_COUNT, "there must be four doors");
    }

    #[test]
    fn test_door_colors_stay_distinct_across_many_resets() {
        let mut env = env_6x6(11);
        for _ in 0..200 {
            env.reset().expect("reset must succeed");
            let colors: HashSet<Color> = env.doors().iter().map(|&(_, _, c)| c).collect();
            assert_eq!(
                colors.len(),
                DOOR_COUNT,
                "every episode must sample four distinct door colors"
            );
        }
    }

    #[test]
    fn test_target_color_is_on_exactly_one_door() {
        let mut env = env_6x6(3);
        for _ in 0..100 {
            env.reset().expect("reset must succeed");
            let hits = env
                .doors()
                .iter()
                .filter(|&&(_, _, c)| c == env.mission().target_color)
                .count();
            assert_eq!(
                hits, 1,
                "the target color must be carried by exactly one door"
            );
        }
    }

    // ----------------------------------------------------------- observation

    #[test]
    fn test_observation_shape_is_7x7x4() {
        assert_eq!(
            <GoToDoorObservation as Observation<3>>::shape(),
            [VIEW_SIZE, VIEW_SIZE, GO_TO_DOOR_OBS_CHANNELS],
            "the mission channel widens the view to 4 channels"
        );
        assert_eq!(MISSION_CHANNEL, 3, "the mission rides in channel 3");
    }

    #[test]
    fn test_observation_broadcasts_mission_to_every_cell() {
        let mut env = env_6x6(5);
        for _ in 0..20 {
            let snap = env.reset().expect("reset must succeed");
            let expected = env.mission().target_color.to_u8();
            for (r, row) in snap.observation().view.iter().enumerate() {
                for (c, cell) in row.iter().enumerate() {
                    assert_eq!(
                        cell[MISSION_CHANNEL], expected,
                        "cell ({r}, {c}) must carry the mission color byte"
                    );
                }
            }
        }
    }

    #[test]
    fn test_observation_entity_channels_match_shared_encoding() {
        let env = env_6x6(2);
        let obs = env.observe_reset(env.state());
        let view = egocentric_view(&env.state().grid, &env.state().agent);
        for (r, row) in view.iter().enumerate() {
            for (c, cell) in row.iter().enumerate() {
                assert_eq!(
                    [obs.view[r][c][0], obs.view[r][c][1], obs.view[r][c][2]],
                    [cell.type_u8(), cell.color_u8(), cell.state_u8()],
                    "channels 0..2 must equal the shared grid entity encoding"
                );
            }
        }
    }

    #[test]
    fn test_observation_carries_mission_only_in_channel_three() {
        // Two hypotheses over the same board differ *only* in channel 3 — so a
        // policy that ignores channel 3 sees byte-identical inputs and cannot
        // tell the targets apart. This is the information-theoretic statement
        // behind the 25% cap asserted below.
        let mut env = env_6x6(8);
        env.reset().expect("reset must succeed");
        let before = env.observe_reset(env.state());
        let other = env
            .doors()
            .iter()
            .map(|&(_, _, c)| c)
            .find(|&c| c != env.mission().target_color)
            .expect("four distinct colors means an alternative exists");
        env.mission = Mission::new(other);
        let after = env.observe_reset(env.state());

        for r in 0..VIEW_SIZE {
            for c in 0..VIEW_SIZE {
                assert_eq!(
                    before.view[r][c][..MISSION_CHANNEL],
                    after.view[r][c][..MISSION_CHANNEL],
                    "changing the mission must not perturb the perceived board"
                );
            }
        }
        assert_ne!(
            before.mission_color_u8(),
            after.mission_color_u8(),
            "changing the mission must change channel 3"
        );
    }

    #[test]
    fn test_observation_round_trips_through_tensor() {
        use burn::backend::Flex;
        type TestBackend = Flex;
        let device = Default::default();

        let env = env_6x6(4);
        let obs = env.observe_reset(env.state());

        let tensor =
            <GoToDoorObservation as TensorConvertible<3, TestBackend>>::to_tensor(&obs, &device);
        let back = <GoToDoorObservation as TensorConvertible<3, TestBackend>>::from_tensor(tensor)
            .expect("round-trip must succeed");

        assert_eq!(back.view, obs.view, "all four channels must round-trip");
        assert_eq!(
            back.mission_color_u8(),
            env.mission().target_color.to_u8(),
            "the mission byte must survive the tensor round-trip"
        );
        // agent_direction is not encoded in the tensor; defaults to North.
        assert_eq!(back.agent_direction, Direction::North.to_u8());
    }

    #[test]
    fn test_observation_from_tensor_rejects_wrong_shape() {
        use burn::backend::Flex;
        use burn::tensor::TensorData;
        type TestBackend = Flex;
        let device = Default::default();

        let flat = vec![0.0f32; VIEW_SIZE * VIEW_SIZE * 3];
        let data = TensorData::new(flat, [VIEW_SIZE, VIEW_SIZE, 3]);
        let tensor = Tensor::<TestBackend, 3>::from_data(data, &device);
        let err = <GoToDoorObservation as TensorConvertible<3, TestBackend>>::from_tensor(tensor)
            .expect_err("a 3-channel tensor must be rejected");
        assert!(
            err.message.contains("expected shape"),
            "error must name the expected shape, got: {}",
            err.message
        );
    }

    // ------------------------------------------------------------------- rng

    #[test]
    fn test_reset_samples_target_near_uniformly() {
        // 200 resets, target wall counted. p = 1/4, n = 200 → mean 50, sd ≈ 6.1.
        // The bounds below are ≈ ±4 sd; the seed is fixed, so this cannot flake.
        let mut env = env_6x6(1234);
        let mut counts = [0usize; DOOR_COUNT];
        for _ in 0..200 {
            let snap = env.reset().expect("reset must succeed");
            counts[target_wall(&env, snap.observation())] += 1;
        }
        for (wall, &n) in counts.iter().enumerate() {
            assert!(
                (25..=80).contains(&n),
                "wall {wall} was the target {n}/200 times — target is not ~uniform (counts: {counts:?})"
            );
        }
    }

    #[test]
    fn test_reset_does_not_replay_the_same_episode() {
        // The regression guard for issue #109: `reset()` must draw from the
        // persistent stream, not re-seed it from config.
        let mut env = env_6x6(21);
        let mut episodes = HashSet::new();
        for _ in 0..20 {
            env.reset().expect("reset must succeed");
            let colors: Vec<u8> = env.doors().iter().map(|&(_, _, c)| c.to_u8()).collect();
            episodes.insert((colors, env.mission().target_color.to_u8()));
        }
        assert!(
            episodes.len() > 1,
            "successive resets must sample independent episodes; got {} distinct in 20",
            episodes.len()
        );
    }

    #[test]
    fn test_reset_with_seed_is_reproducible() {
        let mut env = env_6x6(21);
        env.reset_with_seed(999).expect("reset must succeed");
        let first = (*env.doors(), *env.mission());
        // Advance the stream with ordinary resets, then re-seed identically.
        env.reset().expect("reset must succeed");
        env.reset().expect("reset must succeed");
        env.reset_with_seed(999).expect("reset must succeed");
        let second = (*env.doors(), *env.mission());
        assert_eq!(
            first, second,
            "reset_with_seed must reproduce the episode bit-for-bit"
        );
    }

    #[test]
    fn test_reset_with_seed_diverges_across_seeds() {
        let mut env = env_6x6(21);
        env.reset_with_seed(1).expect("reset must succeed");
        let a = (*env.doors(), *env.mission());
        env.reset_with_seed(2).expect("reset must succeed");
        let b = (*env.doors(), *env.mission());
        assert_ne!(a, b, "different seeds must produce different episodes");
    }

    #[test]
    fn test_construction_seed_reproduces_the_reset_sequence() {
        let mut a = env_6x6(77);
        let mut b = env_6x6(77);
        for _ in 0..5 {
            a.reset().expect("reset must succeed");
            b.reset().expect("reset must succeed");
            assert_eq!(a.doors(), b.doors(), "same seed → same reset sequence");
            assert_eq!(
                a.mission(),
                b.mission(),
                "same seed → same mission sequence"
            );
        }
    }

    // ------------------------------------------------------------- gameplay

    #[test]
    fn test_mission_conditioned_oracle_always_succeeds() {
        // Reads the target color *from channel 3 of the observation*, locates the
        // door wearing that color, and walks to it. Must succeed every episode.
        let mut env = env_6x6(31);
        for episode in 0..100 {
            let snap = env.reset().expect("reset must succeed");
            let wall = target_wall(&env, snap.observation());
            let reward = run(&mut env, script_for(wall));
            assert!(
                reward > 0.9,
                "episode {episode}: an instruction-following oracle must reach the target door \
                 (wall {wall}), got reward {reward}"
            );
        }
    }

    #[test]
    fn test_mission_blind_policy_is_capped_near_one_quarter() {
        // The same navigation machinery, but ignoring channel 3: always walk to
        // the North door. Since the target is uniform over the four doors and the
        // colors are re-sampled every episode, no board-only signal can help — the
        // success rate must sit near 1/4.
        let mut env = env_6x6(31);
        let trials = 200;
        let mut wins = 0;
        for _ in 0..trials {
            env.reset().expect("reset must succeed");
            if run(&mut env, script_for(NORTH)) > 0.0 {
                wins += 1;
            }
        }
        let rate = f64::from(wins) / f64::from(trials);
        assert!(
            (0.15..=0.35).contains(&rate),
            "a mission-blind policy must be capped near 25%, got {rate}"
        );
    }

    #[test]
    fn test_done_at_wrong_color_door_scores_zero() {
        let mut env = env_6x6(6);
        let snap = env.reset().expect("reset must succeed");
        let target = target_wall(&env, snap.observation());
        let wrong = (target + 1) % DOOR_COUNT;
        let reward = run(&mut env, script_for(wrong));
        assert_eq!(
            reward, 0.0,
            "Done at a door of the wrong color must pay nothing"
        );
    }

    #[test]
    fn test_done_facing_no_door_scores_zero() {
        let mut env = env_6x6(6);
        env.reset().expect("reset must succeed");
        // From (2, 2) facing East the cell in front is empty interior.
        let snap = env.step(GridAction::Done).expect("step must succeed");
        assert!(snap.is_done(), "Done always terminates the episode");
        let reward: f32 = (*snap.reward()).into();
        assert_eq!(reward, 0.0, "Done facing no door must pay nothing");
    }

    #[test]
    fn test_closed_doors_are_impassable() {
        let mut env = env_6x6(6);
        env.reset().expect("reset must succeed");
        // Walk north into the North-wall door: it is closed, so the agent bumps.
        env.step(GridAction::Forward).expect("step"); // → (3, 2)
        env.step(GridAction::TurnLeft).expect("step"); // face North
        env.step(GridAction::Forward).expect("step"); // → (3, 1)
        let snap = env.step(GridAction::Forward).expect("step"); // bump
        assert!(!snap.is_done(), "bumping a door does not end the episode");
        assert_eq!(
            env.state().agent.y,
            1,
            "a closed door must block forward movement"
        );
    }

    #[test]
    fn test_timeout_terminates_with_zero_reward() {
        let max_steps = 12;
        let mut env = GoToDoorEnv::with_config(GoToDoorConfig::new(6, max_steps, 6), false)
            .expect("valid config");
        env.reset().expect("reset must succeed");
        let mut last = None;
        for _ in 0..max_steps {
            last = Some(env.step(GridAction::TurnLeft).expect("step must succeed"));
        }
        let snap = last.expect("max_steps > 0");
        assert!(snap.is_done(), "the episode must end at the step budget");
        let reward: f32 = (*snap.reward()).into();
        assert_eq!(reward, 0.0, "a timeout must pay nothing");
        assert_eq!(env.steps(), max_steps);
    }

    #[test]
    fn test_reset_clears_the_step_counter() {
        let mut env = env_6x6(6);
        env.reset().expect("reset must succeed");
        env.step(GridAction::TurnLeft).expect("step must succeed");
        assert_eq!(env.steps(), 1);
        env.reset().expect("reset must succeed");
        assert_eq!(env.steps(), 0, "reset must zero the step counter");
    }

    #[test]
    fn test_display_reports_the_sampled_mission() {
        let mut env = env_6x6(6);
        env.reset().expect("reset must succeed");
        let shown = env.to_string();
        let expected = format!("target={:?}", env.mission().target_color);
        assert!(
            shown.contains(&expected),
            "Display must report the sampled mission, got: {shown}"
        );
    }
}
