//! Navigate across sampled obstacle rivers, one opening each, to reach the goal.
//!
//! Ports Farama Minigrid's [`CrossingEnv`] (`MiniGrid-LavaCrossingS9N1-v0` and
//! its `SimpleCrossing` siblings). The room is an `N×N` grid with border walls.
//! Every episode samples **two** *rivers* — full interior lines of [`Lava`] or
//! [`Wall`] — from the even interior rows and columns, then punches exactly one
//! opening per river.
//!
//! ## Openings come from a monotone walk, not from independent draws
//!
//! The load-bearing detail: upstream does **not** draw each opening
//! independently. It shuffles a list of moves — one East move per *vertical*
//! river and one South move per *horizontal* river — and walks the room lattice
//! from the top-left room to the bottom-right one, punching the opening for the
//! river it is about to cross, inside the band it currently occupies. Because
//! the walk is monotone and each opening is punched on the boundary of the room
//! the walk is standing in, the openings are guaranteed to chain into a single
//! corridor from `(1, 1)` to `(N - 2, N - 2)`.
//!
//! Sampling the openings independently would draw from the same *marginal*
//! distribution and still generate unsolvable boards, because two openings can
//! land in bands that do not connect. `crossing_is_solvable_across_resets` in
//! this module's test suite is the guard.
//!
//! ## Layout (one sampled episode at `size = 7`)
//!
//! Here the sample picked a vertical river at column `2` and a horizontal river
//! at row `4`, and the walk drew the order *South, then East*:
//!
//! ```text
//!     0 1 2 3 4 5 6
//!  0  # # # # # # #
//!  1  # A L . . . #   ← column 2 is a vertical river …
//!  2  # . L . . . #
//!  3  # . L . . . #
//!  4  # . L L L L #   ← row 4 is a horizontal river, open at column 1
//!  5  # . . . . G #   ← the vertical river is open at row 5
//!  6  # # # # # # #
//! ```
//!
//! - `A` — agent start `(1, 1)`, facing East (fixed, as upstream)
//! - `G` — goal `(size - 2, size - 2)` (fixed, as upstream)
//! - `L` — the configured obstacle: lava, or wall in [`CrossingKind::Wall`]
//! - `.` — empty passable cell
//!
//! The corridor here is `(1,1) → (1,4) → (1,5) → (5,5)`. A different draw moves
//! the rivers, the openings, or both; [`CrossingEnv::strip_cols`],
//! [`CrossingEnv::strip_rows`], [`CrossingEnv::gap_rows`], and
//! [`CrossingEnv::gap_col`] report the layout actually in force.
//!
//! ## Known simplifications relative to upstream
//!
//! - **`num_crossings` is fixed at 2.** Upstream takes it as a registration
//!   kwarg (`size = 9` with 1/2/3 crossings, `size = 11` with 5).
//!   [`CrossingConfig`] has no such field and is `Serialize`/`Deserialize`, so
//!   adding one is a persisted-config change; it is tracked separately. Which
//!   two of the candidate rows/columns become rivers *is* sampled.
//! - **Even `size` is accepted.** Upstream asserts an odd `width`/`height`;
//!   rlevo's [`Validate`] impl only enforces `size >= 7` (see
//!   [`CrossingConfig::size`]). The generator handles even sizes correctly — the
//!   candidate lines and the room bands stay non-degenerate — so the looser
//!   floor is kept rather than tightened under a change that is about layout
//!   sampling.
//!
//! ## Observation and action spaces
//!
//! | | Dimension | Description |
//! |---|---|---|
//! | Observation | 3 | `[agent_x, agent_y, agent_dir]` |
//! | Action | 3 | `TurnLeft`, `TurnRight`, `Forward` (one-hot) |
//! | Reward | 1 | Scalar; positive only on reaching the goal |
//!
//! ## Example
//!
//! ```rust
//! use rlevo_environments::grids::crossing::{CrossingConfig, CrossingEnv, CrossingKind};
//! use rlevo_core::environment::{ConstructableEnv, Environment};
//!
//! let cfg = CrossingConfig::new(7, 196, 42, CrossingKind::Lava);
//! let mut env = CrossingEnv::with_config(cfg, false).expect("valid config");
//! let _snapshot = env.reset().unwrap();
//! // A fresh layout every episode; ask the env, never assume.
//! let _rivers = (env.strip_cols(), env.strip_rows());
//! ```
//!
//! [`CrossingEnv`]: https://minigrid.farama.org/environments/minigrid/CrossingEnv/
//! [`Lava`]: super::core::entity::Entity::Lava
//! [`Wall`]: super::core::entity::Entity::Wall

use super::core::{
    GridSnapshot, Visibility,
    action::GridAction,
    agent::AgentState,
    build_snapshot,
    direction::Direction,
    dynamics::{StepOutcome, apply_action},
    entity::Entity,
    grid::Grid,
    observation::GridObservation,
    observe_grid,
    render::render_ascii,
    reward::success_reward,
    state::GridState,
};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
// `Rng` is the trait bound (rules.md §8); `RngExt` carries `random_range`,
// which rand 0.10 moved out of `Rng` into a blanket extension implemented for
// every `R: Rng + ?Sized`. Imported anonymously — only its methods are wanted.
use rand::{Rng, RngExt as _, SeedableRng};
use rlevo_core::config::{self, ConfigError, Validate};
use rlevo_core::environment::{ConstructableEnv, Environment, EnvironmentError, Sensor};
use rlevo_core::reward::ScalarReward;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

/// Selects the obstacle type used for the crossing rivers.
///
/// Determines whether the agent pays a hard penalty (lava, episode ends) or
/// is simply blocked (wall, episode continues) when it tries to move through
/// a river cell that is not an opening.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CrossingKind {
    /// Terminal-hazard rivers; stepping onto lava immediately ends the episode.
    #[default]
    Lava,
    /// Impassable wall rivers; the agent is bumped back but the episode continues.
    Wall,
}

impl CrossingKind {
    /// Returns the grid entity placed in river cells that are not openings.
    #[must_use]
    pub const fn entity(self) -> Entity {
        match self {
            Self::Lava => Entity::Lava,
            Self::Wall => Entity::Wall,
        }
    }
}

impl FromStr for CrossingKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "lava" => Ok(Self::Lava),
            "wall" => Ok(Self::Wall),
            other => Err(format!("unknown kind `{other}`")),
        }
    }
}

/// Minimum supported side length; we need at least one interior row per river
/// plus free space for the agent and goal.
const MIN_SIZE: usize = 7;

/// Rivers placed per episode — upstream Minigrid's `num_crossings`.
///
/// Fixed at 2 because [`CrossingConfig`] deliberately exposes no
/// `num_crossings` field (see the module docs). *Which* two candidate lines
/// become rivers is sampled every episode.
const NUM_CROSSINGS: usize = 2;

/// Which axis a river runs along.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Axis {
    /// A full interior **column** of obstacles.
    Vertical,
    /// A full interior **row** of obstacles.
    Horizontal,
}

/// One leg of the monotone walk that punches the openings.
///
/// Upstream names these `h` and `v` after the *move* direction, not after the
/// river being crossed — which is why its `path` list reads
/// `[h] * len(rivers_v) + [v] * len(rivers_h)`. An East move crosses a vertical
/// river; a South move crosses a horizontal one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Leg {
    /// Cross the next vertical river, moving East into the following column band.
    East,
    /// Cross the next horizontal river, moving South into the following row band.
    South,
}

/// The rivers and openings sampled for the current episode.
///
/// `gap_rows[k]` is the open row of the vertical river at `strip_cols[k]`, and
/// `gap_cols[k]` is the open column of the horizontal river at `strip_rows[k]`
/// — both vectors are index-aligned with their river vector, which is sorted
/// ascending. Either pair may be empty when both sampled rivers share an axis.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CrossingLayout {
    strip_cols: Vec<i32>,
    strip_rows: Vec<i32>,
    gap_rows: Vec<i32>,
    gap_cols: Vec<i32>,
}

/// Configuration for [`CrossingEnv`].
///
/// Controls the grid dimensions, episode length, RNG seed, and obstacle type.
/// All fields are public so configurations can be constructed with struct
/// literal syntax or via [`CrossingConfig::new`].
///
/// # Examples
///
/// ```rust
/// use rlevo_environments::grids::crossing::{CrossingConfig, CrossingKind};
///
/// let cfg = CrossingConfig::new(9, 324, 0, CrossingKind::Wall);
/// assert_eq!(cfg.size, 9);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrossingConfig {
    /// Side length of the square grid in cells, including border walls.
    ///
    /// Must be at least `7`. The interior playable area is
    /// `(size - 2) × (size - 2)`.
    ///
    /// Unlike upstream Minigrid, which asserts an odd `width`/`height`, this
    /// crate accepts any `size >= 7`; the layout sampler is total over even
    /// sizes too. See the module docs, "Known simplifications".
    pub size: usize,
    /// Maximum number of steps before the episode is truncated.
    pub max_steps: usize,
    /// Seed for the environment's persistent RNG, drawn once at construction.
    ///
    /// The RNG **advances** across resets (ADR 0029), so successive episodes get
    /// independent rivers and openings. A fixed seed therefore reproduces a
    /// fixed *sequence* of episodes, not one repeated episode. For bit-for-bit
    /// replay of a single episode use [`CrossingEnv::reset_with_seed`].
    ///
    /// Two deliberate deviations from upstream `CrossingEnv` are documented in
    /// the module docs: `num_crossings` is fixed at 2, and even `size` values
    /// are accepted.
    pub seed: u64,
    /// Obstacle type placed in the river cells that are not openings.
    pub kind: CrossingKind,
}

impl CrossingConfig {
    /// Constructs a `CrossingConfig` with explicit field values.
    ///
    /// # Panics
    ///
    /// Does not panic here; validation happens when [`CrossingEnv`] is built
    /// or when parsing via [`FromStr`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rlevo_environments::grids::crossing::{CrossingConfig, CrossingKind};
    ///
    /// let cfg = CrossingConfig::new(7, 196, 0, CrossingKind::Lava);
    /// ```
    #[must_use]
    pub const fn new(size: usize, max_steps: usize, seed: u64, kind: CrossingKind) -> Self {
        Self {
            size,
            max_steps,
            seed,
            kind,
        }
    }
}

impl Default for CrossingConfig {
    fn default() -> Self {
        let size = 7;
        Self {
            size,
            max_steps: 4 * size * size,
            seed: 0,
            kind: CrossingKind::Lava,
        }
    }
}

impl Validate for CrossingConfig {
    /// Rejects any `size` below `MIN_SIZE` (7) and a zero `max_steps`.
    ///
    /// The `size` floor lives **here**, not only in [`FromStr`]: `CrossingConfig`
    /// derives `Deserialize`, so a config loaded from a file is user-supplied
    /// runtime data that never passes through `from_str` (rules.md §4 — "if an
    /// invalid value can arrive via `Deserialize`, it must be an `Err`").
    /// Struct-update syntax on [`Default`] bypasses `from_str` just as freely.
    ///
    /// [`config::at_least`] subsumes a `nonzero` check for `size`, so the zero
    /// case reports [`TooSmall`](rlevo_core::config::ConstraintKind::TooSmall)
    /// rather than [`Zero`](rlevo_core::config::ConstraintKind::Zero);
    /// `max_steps` keeps `nonzero` because its only floor is 1.
    ///
    /// Parity is deliberately **not** checked. Upstream asserts an odd size; the
    /// sampler here is total over even sizes, and tightening a config
    /// constraint is not this environment's change to make.
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "CrossingConfig";
        config::at_least(C, "size", self.size, MIN_SIZE)?;
        config::nonzero(C, "max_steps", self.max_steps)?;
        Ok(())
    }
}

impl FromStr for CrossingConfig {
    type Err = String;

    /// Parses `"size=9,max_steps=324,seed=0,kind=wall"` (keys in any order) or
    /// the positional form `"9,324,0,wall"`.
    ///
    /// # Errors
    ///
    /// Returns the offending key/value, or the [`Validate`] rejection — the same
    /// guard [`CrossingEnv::with_config`] applies, so this parser cannot admit a
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
                    "kind" => cfg.kind = value.trim().parse()?,
                    other => return Err(format!("unknown key `{other}`")),
                }
            } else {
                match idx {
                    0 => cfg.size = raw.parse().map_err(|e| format!("size: {e}"))?,
                    1 => cfg.max_steps = raw.parse().map_err(|e| format!("max_steps: {e}"))?,
                    2 => cfg.seed = raw.parse().map_err(|e| format!("seed: {e}"))?,
                    3 => cfg.kind = raw.parse()?,
                    _ => return Err(format!("unexpected positional value `{raw}`")),
                }
            }
        }
        cfg.validate()
            .map_err(|e| format!("{e} (got size={})", cfg.size))?;
        Ok(cfg)
    }
}

/// Grid environment where the agent crosses obstacle rivers to reach the goal.
///
/// Implements [`Environment<3, 3, 1>`] — observation and action spaces each
/// have three components, reward is a scalar.
///
/// Every [`reset`](Environment::reset) samples a fresh layout from the
/// environment's persistent RNG: which two candidate lines become rivers, and
/// where each river's single opening sits. The agent start `(1, 1)` facing East
/// and the goal `(size - 2, size - 2)` are fixed, as upstream.
///
/// Construct via [`CrossingEnv::with_config`] for full control or via
/// [`ConstructableEnv::new`] for default settings (size 7, lava, seed 0).
///
/// # Examples
///
/// ```rust
/// use rlevo_environments::grids::crossing::{CrossingConfig, CrossingEnv, CrossingKind};
/// use rlevo_core::environment::{ConstructableEnv, Environment};
///
/// let mut env = CrossingEnv::with_config(
///     CrossingConfig::new(7, 196, 0, CrossingKind::Lava),
///     false,
/// )
/// .expect("valid config");
/// env.reset().unwrap();
/// ```
#[derive(Debug)]
pub struct CrossingEnv {
    state: GridState,
    config: CrossingConfig,
    steps: usize,
    render: bool,
    layout: CrossingLayout,
    rng: StdRng,
}

impl CrossingEnv {
    /// Emission-model visibility policy: does this environment's agent see
    /// through opaque cells?
    ///
    /// The rlevo spelling of canonical Minigrid's `see_through_walls`
    /// constructor argument. Read only by this environment's [`Sensor`] impl,
    /// and an inherent const rather than a config field because it is part of
    /// the task definition, not a knob a caller tunes.
    ///
    /// [`Visibility::Occluded`], because canonical `minigrid/envs/crossing.py`
    /// passes `see_through_walls=False` explicitly — which is also
    /// `MiniGridEnv.__init__`'s own default, so the env restates the canonical
    /// behaviour rather than opting into it. See ADR 0063
    /// (`docs/adr/0063-grid-visibility-occlusion.md`) for the whole
    /// twelve-environment table.
    const VISIBILITY: Visibility = Visibility::Occluded;

    /// Constructs a `CrossingEnv` from an explicit configuration.
    ///
    /// Seeds the persistent RNG **once** from `config.seed` and immediately
    /// samples a first layout from it. Call [`Environment::reset`] before the
    /// first [`Environment::step`] to obtain the first observation; that reset
    /// samples a *fresh* layout, advancing the stream.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rlevo_environments::grids::crossing::{CrossingConfig, CrossingEnv, CrossingKind};
    ///
    /// let env = CrossingEnv::with_config(
    ///     CrossingConfig::new(9, 324, 42, CrossingKind::Wall),
    ///     true, // render ASCII grid to stdout
    /// )
    /// .expect("valid config");
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `config` fails [`Validate`]: a `size` below
    /// `MIN_SIZE` (7) — zero included — or a zero `max_steps`. This is the
    /// construction chokepoint (rules.md §4), so it also rejects a config that
    /// arrived by `Deserialize` or struct-update syntax without passing through
    /// [`FromStr`].
    pub fn with_config(config: CrossingConfig, render: bool) -> Result<Self, ConfigError> {
        config.validate()?;
        let mut rng = StdRng::seed_from_u64(config.seed);
        let (state, layout) = Self::build(&config, &mut rng);
        Ok(Self {
            state,
            config,
            steps: 0,
            render,
            layout,
            rng,
        })
    }

    /// Re-seed the persistent RNG to `seed`, then [`reset`](Environment::reset).
    ///
    /// Ordinary [`reset`](Environment::reset) advances the persistent stream so
    /// successive episodes get independent rivers and openings (ADR 0029); use
    /// this when you need a *specific* layout to reproduce bit-for-bit (e.g.
    /// replaying a failure). Run-level reproducibility is already guaranteed by
    /// the construction seed.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`reset`](Environment::reset) (currently none).
    pub fn reset_with_seed(&mut self, seed: u64) -> Result<GridSnapshot, EnvironmentError> {
        self.rng = StdRng::seed_from_u64(seed);
        self.reset()
    }

    /// Returns a reference to the active configuration.
    #[must_use]
    pub const fn config(&self) -> &CrossingConfig {
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

    /// Renders the current grid as an ASCII string.
    ///
    /// Useful for debugging; the same output is printed to stdout on each step
    /// when the environment was constructed with `render = true`.
    #[must_use]
    pub fn ascii(&self) -> String {
        render_ascii(&self.state.grid, &self.state.agent)
    }

    /// Open **columns** of this episode's horizontal rivers, aligned with
    /// [`strip_rows`](Self::strip_rows).
    ///
    /// `gap_col()[k]` is the one passable column of the river at
    /// `strip_rows()[k]`. This returns a `Vec` rather than the single `i32` it
    /// did while the board was fixed: each river now gets its **own** opening
    /// from the layout walk, so no single column is shared by all of them, and
    /// the vector may be empty when both sampled rivers are vertical.
    #[must_use]
    pub fn gap_col(&self) -> Vec<i32> {
        self.layout.gap_cols.clone()
    }

    /// Open **rows** of this episode's vertical rivers, aligned with
    /// [`strip_cols`](Self::strip_cols).
    ///
    /// `gap_rows()[k]` is the one passable row of the river at
    /// `strip_cols()[k]`. Empty when both sampled rivers are horizontal.
    #[must_use]
    pub fn gap_rows(&self) -> Vec<i32> {
        self.layout.gap_rows.clone()
    }

    /// Row indices carrying a horizontal obstacle river this episode, ascending.
    ///
    /// Sampled per [`reset`](Environment::reset) from the even interior rows
    /// `2, 4, …` strictly below `size - 2`. Contains between zero and
    /// `NUM_CROSSINGS` entries — the complement sits in
    /// [`strip_cols`](Self::strip_cols).
    #[must_use]
    pub fn strip_rows(&self) -> Vec<i32> {
        self.layout.strip_rows.clone()
    }

    /// Column indices carrying a vertical obstacle river this episode, ascending.
    ///
    /// The vertical counterpart of [`strip_rows`](Self::strip_rows); vertical
    /// rivers did not exist while the board was fixed, so a reader of the board
    /// needs both to reconstruct the layout.
    #[must_use]
    pub fn strip_cols(&self) -> Vec<i32> {
        self.layout.strip_cols.clone()
    }

    /// Sample one episode: two rivers, and one opening per river.
    ///
    /// Draws from `rng` and lets it advance (ADR 0029) — never re-seeds. The
    /// draws, and their order, reproduce upstream `CrossingEnv._gen_grid`:
    ///
    /// 1. shuffle the candidate river lines and keep the first
    ///    [`NUM_CROSSINGS`];
    /// 2. shuffle the walk legs — one East per vertical river, one South per
    ///    horizontal river;
    /// 3. one uniform draw per leg for the opening's free coordinate.
    ///
    /// Step 3 is what makes the board solvable, and it is not interchangeable
    /// with drawing the openings independently: the walk is monotone from the
    /// top-left room to the bottom-right one, and every opening is punched on
    /// the boundary of the room the walk currently occupies, so the openings
    /// chain into one corridor by construction.
    ///
    /// The shared [`placement`](super::core::placement) sampler is deliberately
    /// **not** used here, and the reason is not oversight. It draws a uniform
    /// *free* cell of a [`Rect`](super::core::Rect), and every draw this env
    /// makes is something else: the rivers are a shuffle over candidate *lines*
    /// rather than cells, and an opening is a hole punched **into** a line that
    /// is already full of obstacles — `is_free` rejects every candidate there,
    /// so `sample_pos` would return `Exhausted` on a correct board. Nothing is
    /// placed on a free cell in this environment: the agent and goal are fixed
    /// by direct assignment, as upstream.
    fn build<R: Rng + ?Sized>(config: &CrossingConfig, rng: &mut R) -> (GridState, CrossingLayout) {
        #[allow(clippy::cast_possible_wrap)]
        let size = config.size as i32;

        // Candidate river lines: even interior indices `2, 4, …` strictly below
        // `size - 2`, per axis. `size >= MIN_SIZE` makes `2..size - 2` non-empty,
        // so there are always at least two candidates per axis.
        let mut candidates: Vec<(Axis, i32)> = (2..size - 2)
            .step_by(2)
            .map(|c| (Axis::Vertical, c))
            .chain((2..size - 2).step_by(2).map(|r| (Axis::Horizontal, r)))
            .collect();
        candidates.shuffle(rng);
        candidates.truncate(NUM_CROSSINGS);

        let mut strip_cols: Vec<i32> = candidates
            .iter()
            .filter(|(axis, _)| *axis == Axis::Vertical)
            .map(|&(_, pos)| pos)
            .collect();
        strip_cols.sort_unstable();
        let mut strip_rows: Vec<i32> = candidates
            .iter()
            .filter(|(axis, _)| *axis == Axis::Horizontal)
            .map(|&(_, pos)| pos)
            .collect();
        strip_rows.sort_unstable();

        let mut grid = Grid::new(config.size, config.size);
        grid.draw_walls();
        // Agent and goal are fixed by direct assignment upstream, not sampled.
        let agent = AgentState::new(1, 1, Direction::East);
        grid.set(size - 2, size - 2, Entity::Goal);

        // Rivers span the full interior. Candidate indices are `< size - 2`, so
        // no river can land on the goal, and all are `>= 2`, so none can land on
        // the agent's start cell.
        let blocker = config.kind.entity();
        for &row in &strip_rows {
            for x in 1..size - 1 {
                grid.set(x, row, blocker);
            }
        }
        for &col in &strip_cols {
            for y in 1..size - 1 {
                grid.set(col, y, blocker);
            }
        }

        // The monotone walk. `limits_*` bracket the room bands along each axis;
        // consecutive entries differ by at least 2 (candidates are distinct even
        // numbers in `[2, size - 3]`), so every `random_range` below is
        // non-empty by construction.
        let mut legs: Vec<Leg> = std::iter::repeat_n(Leg::East, strip_cols.len())
            .chain(std::iter::repeat_n(Leg::South, strip_rows.len()))
            .collect();
        legs.shuffle(rng);

        let limits_v: Vec<i32> = std::iter::once(0)
            .chain(strip_cols.iter().copied())
            .chain(std::iter::once(size - 1))
            .collect();
        let limits_h: Vec<i32> = std::iter::once(0)
            .chain(strip_rows.iter().copied())
            .chain(std::iter::once(size - 1))
            .collect();

        let mut gap_rows = Vec::with_capacity(strip_cols.len());
        let mut gap_cols = Vec::with_capacity(strip_rows.len());
        let (mut room_i, mut room_j) = (0usize, 0usize);
        for leg in legs {
            match leg {
                Leg::East => {
                    let x = limits_v[room_i + 1];
                    debug_assert!(limits_h[room_j] + 1 < limits_h[room_j + 1]);
                    let y = rng.random_range(limits_h[room_j] + 1..limits_h[room_j + 1]);
                    grid.set(x, y, Entity::Empty);
                    gap_rows.push(y);
                    room_i += 1;
                }
                Leg::South => {
                    debug_assert!(limits_v[room_i] + 1 < limits_v[room_i + 1]);
                    let x = rng.random_range(limits_v[room_i] + 1..limits_v[room_i + 1]);
                    let y = limits_h[room_j + 1];
                    grid.set(x, y, Entity::Empty);
                    gap_cols.push(x);
                    room_j += 1;
                }
            }
        }

        let layout = CrossingLayout {
            strip_cols,
            strip_rows,
            gap_rows,
            gap_cols,
        };
        (GridState::new(grid, agent), layout)
    }

    fn emit(&self, observation: GridObservation, reward: f32, done: bool) -> GridSnapshot {
        if self.render {
            println!("{}", self.ascii());
        }
        build_snapshot(observation, reward, done)
    }
}

impl crate::render::AsciiRenderable for CrossingEnv {
    fn render_ascii(&self) -> String {
        render_ascii(&self.state.grid, &self.state.agent)
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        super::core::render::render_styled(&self.state.grid, &self.state.agent)
    }
}

impl Display for CrossingEnv {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CrossingEnv(size={}, kind={:?}, step={}/{})",
            self.config.size, self.config.kind, self.steps, self.config.max_steps
        )
    }
}

impl ConstructableEnv for CrossingEnv {
    fn new(render: bool) -> Self {
        Self::with_config(CrossingConfig::default(), render).expect("default config must validate")
    }
}

impl Sensor<3, 1, 3> for CrossingEnv {
    type Action = GridAction;
    type State = GridState;
    type Observation = GridObservation;

    /// Emission model `O(a, s')`. The observation is a function of the resulting
    /// `next_state` alone, so this forwards to the same projection as
    /// [`observe_reset`](Self::observe_reset).
    fn observe(&self, _action: &GridAction, next_state: &GridState) -> GridObservation {
        observe_grid(next_state, Self::VISIBILITY)
    }

    fn observe_reset(&self, state: &GridState) -> GridObservation {
        observe_grid(state, Self::VISIBILITY)
    }
}

impl Environment<3, 3, 1> for CrossingEnv {
    type StateType = GridState;
    type ObservationType = super::core::GridObservation;
    type ActionType = GridAction;
    type RewardType = ScalarReward;
    type SnapshotType = GridSnapshot;

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        let (state, layout) = Self::build(&self.config, &mut self.rng);
        self.state = state;
        self.layout = layout;
        self.steps = 0;
        let observation = self.observe_reset(&self.state);
        Ok(self.emit(observation, 0.0, false))
    }

    fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> {
        self.steps += 1;
        let outcome = apply_action(&mut self.state.grid, &mut self.state.agent, action);
        let (reward, done) = match outcome {
            StepOutcome::ReachedGoal => (success_reward(self.steps, self.config.max_steps), true),
            StepOutcome::HitLava => (0.0, true),
            _ => {
                let done = self.steps >= self.config.max_steps;
                (0.0, done)
            }
        };
        let observation = self.observe(&action, &self.state);
        Ok(self.emit(observation, reward, done))
    }
}

impl rlevo_core::render::payload::GridPayloadSource for CrossingEnv {
    fn grid_snapshot(&self) -> rlevo_core::render::payload::GridSnapshot {
        crate::grids::core::render::grid_snapshot(&self.state.grid, &self.state.agent)
    }
}

#[cfg(test)]
mod tests {
    // Exact comparison is intentional throughout this test module: the values
    // are literals or seeds read back without arithmetic, or two identically
    // seeded runs that must agree bit-for-bit. A tolerance would let a real
    // regression pass. Reviewed as a class, not site-by-site.
    #![allow(clippy::float_cmp)]

    use super::*;
    use rlevo_core::config::ConstraintKind;
    use rlevo_core::environment::Snapshot;
    use std::collections::{HashMap, HashSet, VecDeque};

    /// Episodes each seed-loop oracle draws from one persistent stream.
    ///
    /// Sized to exercise the two shuffles rather than to make a flaky test
    /// pass: at `size = 7` there are `C(4, 2) = 6` river pairs and up to two
    /// leg orders each, so a few hundred episodes visits every branch many
    /// times over. If one of these loops ever fails, the layout walk is broken
    /// — do not lower this number.
    const EPISODES: usize = 256;

    /// Sizes every loop runs at: the floor, two larger odd sizes matching
    /// upstream's registered boards, and one even size that upstream would
    /// reject but rlevo's [`Validate`] admits.
    const SIZES: [usize; 4] = [7, 8, 9, 11];

    const KINDS: [CrossingKind; 2] = [CrossingKind::Lava, CrossingKind::Wall];

    /// One episode's river set: `(vertical columns, horizontal rows)`.
    type Rivers = (Vec<i32>, Vec<i32>);

    /// One episode's openings, index-aligned with [`Rivers`]:
    /// `(open rows of the vertical rivers, open columns of the horizontal ones)`.
    type Openings = (Vec<i32>, Vec<i32>);

    fn env_at(size: usize, kind: CrossingKind, seed: u64) -> CrossingEnv {
        let cfg = CrossingConfig::new(size, 4 * size * size, seed, kind);
        CrossingEnv::with_config(cfg, false).expect("valid config")
    }

    fn default_env(kind: CrossingKind) -> CrossingEnv {
        env_at(7, kind, 0)
    }

    // -----------------------------------------------------------------------
    // Solvability oracle: a flood fill over cells the agent may stand on
    // *safely*. Lava is `is_passable`, so a reachability check built on
    // `Entity::is_passable` would call a lava-locked board solvable — the whole
    // point of this env. Only `Empty` and `Goal` count.
    // -----------------------------------------------------------------------

    fn is_safe(entity: Entity) -> bool {
        matches!(entity, Entity::Empty | Entity::Goal)
    }

    /// Shortest safe cell path from the agent to the goal, endpoints included.
    fn safe_path(state: &GridState) -> Option<Vec<(i32, i32)>> {
        #[allow(clippy::cast_possible_wrap)]
        let size = state.grid.width() as i32;
        let start = (state.agent.x, state.agent.y);
        let goal = (size - 2, size - 2);
        assert!(
            is_safe(state.grid.get(start.0, start.1)),
            "agent must start on a safe cell, found {:?}",
            state.grid.get(start.0, start.1)
        );

        let mut prev: HashMap<(i32, i32), (i32, i32)> = HashMap::new();
        let mut seen: HashSet<(i32, i32)> = HashSet::from([start]);
        let mut queue = VecDeque::from([start]);
        while let Some(cell) = queue.pop_front() {
            if cell == goal {
                let mut path = vec![cell];
                let mut cursor = cell;
                while let Some(&parent) = prev.get(&cursor) {
                    path.push(parent);
                    cursor = parent;
                }
                path.reverse();
                return Some(path);
            }
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let next = (cell.0 + dx, cell.1 + dy);
                if !state.grid.in_bounds(next.0, next.1)
                    || !is_safe(state.grid.get(next.0, next.1))
                    || !seen.insert(next)
                {
                    continue;
                }
                prev.insert(next, cell);
                queue.push_back(next);
            }
        }
        None
    }

    /// Turn a cell path into the actions that walk it, starting from `facing`.
    fn actions_along(path: &[(i32, i32)], mut facing: Direction) -> Vec<GridAction> {
        let mut actions = Vec::new();
        for step in path.windows(2) {
            let want = match (step[1].0 - step[0].0, step[1].1 - step[0].1) {
                (1, 0) => Direction::East,
                (-1, 0) => Direction::West,
                (0, 1) => Direction::South,
                (0, -1) => Direction::North,
                other => panic!("path took a non-unit step {other:?}"),
            };
            while facing != want {
                actions.push(GridAction::TurnRight);
                facing = facing.right();
            }
            actions.push(GridAction::Forward);
        }
        actions
    }

    /// Every structural invariant a freshly sampled board must satisfy.
    fn assert_board_is_sound(env: &CrossingEnv) {
        let cfg = *env.config();
        #[allow(clippy::cast_possible_wrap)]
        let size = cfg.size as i32;
        let grid = &env.state().grid;
        let blocker = cfg.kind.entity();

        // Perimeter is wall all round.
        for i in 0..size {
            for (x, y) in [(i, 0), (i, size - 1), (0, i), (size - 1, i)] {
                assert_eq!(grid.get(x, y), Entity::Wall, "perimeter ({x},{y})");
            }
        }

        // Exactly one goal, at the fixed upstream cell.
        let goals: Vec<(i32, i32)> = (0..size)
            .flat_map(|y| (0..size).map(move |x| (x, y)))
            .filter(|&(x, y)| grid.get(x, y) == Entity::Goal)
            .collect();
        assert_eq!(goals, vec![(size - 2, size - 2)], "goal placement");

        // Agent start is fixed, and stands on a safe cell.
        assert_eq!((env.state().agent.x, env.state().agent.y), (1, 1));
        assert_eq!(env.state().agent.direction, Direction::East);
        assert!(is_safe(grid.get(1, 1)), "agent start must be safe");

        // Exactly NUM_CROSSINGS rivers, one opening each, index-aligned.
        let (cols, rows) = (env.strip_cols(), env.strip_rows());
        let (gap_rows, gap_cols) = (env.gap_rows(), env.gap_col());
        assert_eq!(cols.len() + rows.len(), NUM_CROSSINGS, "river count");
        assert_eq!(gap_rows.len(), cols.len(), "one opening per vertical river");
        assert_eq!(
            gap_cols.len(),
            rows.len(),
            "one opening per horizontal river"
        );
        assert!(
            cols.windows(2).all(|w| w[0] < w[1]),
            "cols ascending/distinct"
        );
        assert!(
            rows.windows(2).all(|w| w[0] < w[1]),
            "rows ascending/distinct"
        );

        // Every river is a full contiguous line of the configured obstacle,
        // interrupted only at its own opening.
        for (&col, &gap) in cols.iter().zip(gap_rows.iter()) {
            assert!((2..size - 2).contains(&col), "river column {col} interior");
            assert!((1..size - 1).contains(&gap), "opening row {gap} interior");
            for y in 1..size - 1 {
                let expected = if y == gap { Entity::Empty } else { blocker };
                assert_eq!(grid.get(col, y), expected, "vertical river ({col},{y})");
            }
        }
        for (&row, &gap) in rows.iter().zip(gap_cols.iter()) {
            assert!((2..size - 2).contains(&row), "river row {row} interior");
            assert!(
                (1..size - 1).contains(&gap),
                "opening column {gap} interior"
            );
            for x in 1..size - 1 {
                let expected = if x == gap { Entity::Empty } else { blocker };
                assert_eq!(grid.get(x, row), expected, "horizontal river ({x},{row})");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Config surface.
    // -----------------------------------------------------------------------

    #[test]
    fn default_config_validates() {
        assert!(CrossingConfig::default().validate().is_ok());
    }

    #[test]
    fn rejects_zero_size() {
        let bad = CrossingConfig {
            size: 0,
            ..Default::default()
        };
        assert!(CrossingEnv::with_config(bad, false).is_err());
    }

    #[test]
    fn with_config_rejects_size_below_min() {
        // The floor must be enforced at the construction chokepoint, not only in
        // `FromStr`: a `Deserialize`d or struct-updated config skips the parser
        // entirely, and used to reach `build` unchecked. The assertion is on the
        // *policy* (`size >= MIN_SIZE`), not on a geometric claim — some
        // sub-MIN_SIZE boards do build successfully, which is exactly why the
        // floor has to be stated rather than inferred from a build failure.
        let bad = CrossingConfig {
            size: MIN_SIZE - 1,
            ..Default::default()
        };
        let err = CrossingEnv::with_config(bad, false)
            .expect_err("a sub-MIN_SIZE grid must be refused at construction");
        assert_eq!(err.config, "CrossingConfig", "the config must be named");
        assert_eq!(err.field, "size", "the offending field must be named");
        assert_eq!(
            err.kind,
            ConstraintKind::TooSmall {
                min: MIN_SIZE as u64,
                got: (MIN_SIZE - 1) as u64,
            },
            "the structured kind carries the bound; assert it, not the message"
        );
    }

    #[test]
    fn default_config_is_lava() {
        let cfg = CrossingConfig::default();
        assert_eq!(cfg.kind, CrossingKind::Lava);
        assert_eq!(cfg.size, 7);
    }

    #[test]
    fn fromstr_rejects_small_size() {
        assert!("5".parse::<CrossingConfig>().is_err());
    }

    #[test]
    fn fromstr_kind_parses() {
        let cfg: CrossingConfig = "kind=wall".parse().unwrap();
        assert_eq!(cfg.kind, CrossingKind::Wall);
    }

    #[test]
    fn even_size_is_admitted_and_documented() {
        // Upstream Minigrid asserts `width % 2 == 1`; rlevo's `Validate` only
        // enforces the `MIN_SIZE` floor. Pinning that here so the divergence is
        // a tested fact rather than an accident of omission — the loops below
        // build and solve even boards, which is why the floor stays as-is.
        let cfg = CrossingConfig::new(8, 256, 0, CrossingKind::Lava);
        assert!(cfg.validate().is_ok(), "even sizes are accepted by policy");
    }

    // -----------------------------------------------------------------------
    // THE load-bearing test: every sampled board is solvable without lava.
    // -----------------------------------------------------------------------

    #[test]
    fn crossing_is_solvable_across_resets() {
        for size in SIZES {
            for kind in KINDS {
                let mut env = env_at(size, kind, 0);
                for episode in 0..EPISODES {
                    env.reset().expect("reset");
                    assert!(
                        safe_path(env.state()).is_some(),
                        "no lava-free path to the goal at size {size}, {kind:?}, \
                         episode {episode}; rivers cols={:?} rows={:?}, \
                         openings rows={:?} cols={:?}\n{}",
                        env.strip_cols(),
                        env.strip_rows(),
                        env.gap_rows(),
                        env.gap_col(),
                        env.ascii(),
                    );
                }
            }
        }
    }

    #[test]
    fn sampled_boards_are_structurally_sound() {
        for size in SIZES {
            for kind in KINDS {
                let mut env = env_at(size, kind, 7);
                for _ in 0..EPISODES {
                    env.reset().expect("reset");
                    assert_board_is_sound(&env);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Non-degeneracy, per sampled quantity, separately. A whole-observation
    // diff would pass on a board whose rivers move but whose openings never do
    // (and vice versa), which is the trap ADR 0062 names.
    // -----------------------------------------------------------------------

    #[test]
    fn river_positions_vary_across_episodes() {
        for size in SIZES {
            let mut env = env_at(size, CrossingKind::Lava, 11);
            let mut seen = HashSet::new();
            for _ in 0..EPISODES {
                env.reset().expect("reset");
                seen.insert((env.strip_cols(), env.strip_rows()));
            }
            assert!(
                seen.len() > 1,
                "river positions never moved at size {size}: {seen:?}"
            );
        }
    }

    #[test]
    fn opening_positions_vary_for_a_fixed_river_set() {
        // Keyed by the river set so this cannot be satisfied by river movement
        // alone: it demands two episodes with the *same* rivers and different
        // openings.
        for size in SIZES {
            let mut env = env_at(size, CrossingKind::Lava, 23);
            let mut by_rivers: HashMap<Rivers, HashSet<Openings>> = HashMap::new();
            for _ in 0..EPISODES {
                env.reset().expect("reset");
                by_rivers
                    .entry((env.strip_cols(), env.strip_rows()))
                    .or_default()
                    .insert((env.gap_rows(), env.gap_col()));
            }
            assert!(
                by_rivers.values().any(|openings| openings.len() > 1),
                "openings are pinned given the rivers at size {size}: {by_rivers:?}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // The persistent-stream seam (ADR 0029).
    // -----------------------------------------------------------------------

    #[test]
    fn consecutive_resets_differ() {
        // The stream must advance across resets. A re-seeding `reset()` would
        // hand back an identical board every time and still pass a "we call the
        // rng" test — this is the assertion that catches it.
        let mut env = default_env(CrossingKind::Lava);
        let mut seen = HashSet::new();
        for _ in 0..EPISODES {
            env.reset().expect("reset");
            seen.insert((
                env.strip_cols(),
                env.strip_rows(),
                env.gap_rows(),
                env.gap_col(),
            ));
        }
        assert!(seen.len() > 1, "reset() replayed one fixed layout");
    }

    #[test]
    fn reset_with_seed_is_reproducible() {
        let mut a = default_env(CrossingKind::Lava);
        let mut b = default_env(CrossingKind::Lava);
        // Advance the two streams differently first, so the agreement below can
        // only come from the re-seed and not from a shared position in the
        // stream.
        a.reset().expect("reset");
        for _ in 0..5 {
            b.reset().expect("reset");
        }
        let sa = a.reset_with_seed(99).expect("reset_with_seed");
        let sb = b.reset_with_seed(99).expect("reset_with_seed");
        assert_eq!(sa.observation(), sb.observation());
        assert_eq!(a.ascii(), b.ascii(), "same seed, bit-identical board");
    }

    #[test]
    fn reset_with_different_seeds_differs() {
        let mut env = default_env(CrossingKind::Lava);
        let mut seen = HashSet::new();
        for seed in 0..64u64 {
            env.reset_with_seed(seed).expect("reset_with_seed");
            seen.insert(env.ascii());
        }
        assert!(seen.len() > 1, "every replay seed produced the same board");
    }

    #[test]
    fn reset_is_deterministic() {
        let cfg = CrossingConfig::new(7, 100, 5, CrossingKind::Wall);
        let mut a = CrossingEnv::with_config(cfg, false).expect("valid config");
        let mut b = CrossingEnv::with_config(cfg, false).expect("valid config");
        let sa = a.reset().unwrap();
        let sb = b.reset().unwrap();
        assert_eq!(sa.observation(), sb.observation());
    }

    // -----------------------------------------------------------------------
    // Accessors and dynamics.
    // -----------------------------------------------------------------------

    #[test]
    fn gap_and_strip_accessors_read_sampled_state() {
        // The accessors used to be closed forms of `config.size`. They must now
        // report the board that was actually generated, so the check is
        // agreement with the grid, at every size, over many episodes.
        for size in SIZES {
            let mut env = env_at(size, CrossingKind::Wall, 3);
            for _ in 0..EPISODES {
                env.reset().expect("reset");
                let grid = &env.state().grid;
                for (col, gap) in env.strip_cols().into_iter().zip(env.gap_rows()) {
                    assert_eq!(grid.get(col, gap), Entity::Empty, "reported opening");
                }
                for (row, gap) in env.strip_rows().into_iter().zip(env.gap_col()) {
                    assert_eq!(grid.get(gap, row), Entity::Empty, "reported opening");
                }
            }
        }
    }

    #[test]
    fn planned_rollout_solves_both_variants() {
        for kind in KINDS {
            for size in SIZES {
                let mut env = env_at(size, kind, 5);
                for _ in 0..16 {
                    env.reset().expect("reset");
                    let path = safe_path(env.state()).expect("board must be solvable");
                    let mut last = None;
                    for action in actions_along(&path, env.state().agent.direction) {
                        last = Some(env.step(action).expect("step"));
                    }
                    let snap = last.expect("plan must contain at least one action");
                    assert!(snap.is_done(), "plan did not terminate the episode");
                    let reward: f32 = (*snap.reward()).into();
                    assert!(reward > 0.0, "reward was {reward} at size {size}, {kind:?}");
                }
            }
        }
    }

    /// Reset until the cell south of the agent carries the obstacle, or give up.
    ///
    /// The rivers move every episode, so a fixed script can no longer assume a
    /// blocker is adjacent to the start; the board must be searched for.
    fn reset_until_blocker_south(env: &mut CrossingEnv, tries: usize) -> bool {
        let blocker = env.config().kind.entity();
        for _ in 0..tries {
            env.reset().expect("reset");
            if env.state().grid.get(1, 2) == blocker {
                return true;
            }
        }
        false
    }

    #[test]
    fn stepping_onto_lava_ends_episode() {
        let mut env = default_env(CrossingKind::Lava);
        assert!(
            reset_until_blocker_south(&mut env, 512),
            "no episode in 512 put lava directly south of the start"
        );
        // (1,1) E → TurnRight → S → Forward → (1,2) = Lava
        env.step(GridAction::TurnRight).unwrap();
        let snap = env.step(GridAction::Forward).unwrap();
        assert!(snap.is_done());
        let reward: f32 = (*snap.reward()).into();
        assert_eq!(reward, 0.0);
    }

    #[test]
    fn walking_into_wall_only_bumps() {
        let mut env = default_env(CrossingKind::Wall);
        assert!(
            reset_until_blocker_south(&mut env, 512),
            "no episode in 512 put a wall directly south of the start"
        );
        env.step(GridAction::TurnRight).unwrap();
        let snap = env.step(GridAction::Forward).unwrap();
        assert!(!snap.is_done(), "walls should not terminate the episode");
        assert_eq!(env.state().agent.y, 1, "agent should not have moved");
    }

    #[test]
    fn wall_variant_uses_walls_and_lava_variant_uses_lava() {
        for (kind, expected) in [
            (CrossingKind::Wall, Entity::Wall),
            (CrossingKind::Lava, Entity::Lava),
        ] {
            let mut env = default_env(kind);
            env.reset().expect("reset");
            let grid = &env.state().grid;
            #[allow(clippy::cast_possible_wrap)]
            let size = env.config().size as i32;
            let mut river_cells: Vec<Entity> = Vec::new();
            for col in env.strip_cols() {
                river_cells.extend((1..size - 1).map(|y| grid.get(col, y)));
            }
            for row in env.strip_rows() {
                river_cells.extend((1..size - 1).map(|x| grid.get(x, row)));
            }
            assert!(!river_cells.is_empty(), "some river must exist");
            assert!(
                river_cells
                    .iter()
                    .all(|&e| e == expected || e == Entity::Empty),
                "{kind:?} rivers must be built from {expected:?}, saw {river_cells:?}"
            );
            assert!(
                river_cells.contains(&expected),
                "{kind:?} rivers must contain at least one {expected:?} cell"
            );
        }
    }

    /// Hand trace of the walk, kept as executable documentation.
    ///
    /// `size = 7`, rivers `{ vertical column 2, horizontal row 4 }`,
    /// leg order `[South, East]`:
    ///
    /// - `limits_v = [0, 2, 6]`, `limits_h = [0, 4, 6]`, `room = (0, 0)`.
    /// - **South**: `x ∈ (limits_v[0], limits_v[1]) = (0, 2)` → `x = 1`;
    ///   `y = limits_h[1] = 4`. Punch `(1, 4)`; `room = (0, 1)`.
    /// - **East**: `x = limits_v[1] = 2`;
    ///   `y ∈ (limits_h[1], limits_h[2]) = (4, 6)` → `y = 5`.
    ///   Punch `(2, 5)`; `room = (1, 1)`.
    ///
    /// ```text
    ///     0 1 2 3 4 5 6
    ///  0  # # # # # # #
    ///  1  # A L . . . #
    ///  2  # . L . . . #
    ///  3  # . L . . . #
    ///  4  # . L L L L #
    ///  5  # . . . . G #
    ///  6  # # # # # # #
    /// ```
    ///
    /// Corridor: `(1,1) → (1,4) → (1,5) → (5,5)`. The other leg order
    /// `[East, South]` instead punches `(2, y)` with `y ∈ {1,2,3}` and
    /// `(x, 4)` with `x ∈ {3,4,5}` — also connected, and connected for the
    /// *same* reason: each opening is punched on the boundary of the room the
    /// walk is standing in.
    ///
    /// The test replays that trace against the real builder by driving it with
    /// the layout the env reports, rather than by pinning a seed to a board.
    #[test]
    fn hand_traced_board_is_reproduced_when_that_layout_is_sampled() {
        let mut env = default_env(CrossingKind::Lava);
        let mut checked = false;
        for _ in 0..EPISODES {
            env.reset().expect("reset");
            if env.strip_cols() != vec![2] || env.strip_rows() != vec![4] {
                continue;
            }
            checked = true;
            let (gap_row, gap_col) = (env.gap_rows()[0], env.gap_col()[0]);
            // The two reachable leg orders, exactly as traced above.
            let south_first = gap_col == 1 && gap_row == 5;
            let east_first = (1..4).contains(&gap_row) && (3..6).contains(&gap_col);
            assert!(
                south_first || east_first,
                "openings ({gap_col} on row 4, {gap_row} on column 2) match neither leg order"
            );
            assert!(safe_path(env.state()).is_some());
        }
        assert!(
            checked,
            "the traced river set never came up in {EPISODES} episodes"
        );
    }
}
