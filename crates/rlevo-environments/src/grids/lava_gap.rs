//! `LavaGap`: cross a vertical lava column through a single gap.
//!
//! Ports Farama Minigrid's [`LavaGapEnv`] — the `MiniGrid-LavaGapS5-v0`
//! family. A vertical strip of [`Lava`] bisects the room with one empty cell
//! forming the crossing. Stepping into lava terminates the episode with reward
//! `0.0`; reaching the goal pays [`success_reward`].
//!
//! ## Layout: sampled fresh every episode
//!
//! Every [`reset`](rlevo_core::environment::Environment::reset) draws a new
//! board, reproducing the two `_rand_int` draws upstream `_gen_grid` makes and
//! the order it makes them in (ADR 0062 §1):
//!
//! 1. the **lava column** `x`, uniform over `2..size-2`;
//! 2. the **gap row** `y`, uniform over `1..size-1`.
//!
//! Nothing else moves. The agent always starts at `(1, 1)` facing East and the
//! goal always sits at `(size-2, size-2)`, because upstream assigns both
//! directly and never routes them through `place_agent` / `place_obj`.
//! [`LavaGapEnv::lava_col`] and [`LavaGapEnv::gap_row`] read the *current*
//! episode's draw back, so a caller (or a planner) can recover the board it was
//! handed.
//!
//! At `size = 5` the column range `2..3` holds a single value, so only the gap
//! row varies there; from `size = 6` upwards both quantities move.
//!
//! ### Two sampled boards
//!
//! `LavaGapConfig::new(5, 100, 0)` after `reset_with_seed(0)` — the column is
//! pinned to `2` at this size, and this seed puts the gap on row `3`:
//!
//! ```text
//! # # # # #
//! # > L . #    L = Lava
//! # . L . #    > = agent at (1, 1) facing East
//! # . . G #    G = Goal at (size-2, size-2)
//! # # # # #    # = wall
//! ```
//!
//! `LavaGapConfig::new(9, 324, 0)` after `reset_with_seed(0)` — at this size
//! the column moves too; this seed lands it on `6` with the gap on row `6`:
//!
//! ```text
//! # # # # # # # # #
//! # > . . . . L . #
//! # . . . . . L . #
//! # . . . . . L . #
//! # . . . . . L . #
//! # . . . . . L . #
//! # . . . . . . . #
//! # . . . . . L G #
//! # # # # # # # # #
//! ```
//!
//! Both boards are real `ascii()` output, pinned by
//! `tests::module_doc_snapshots_are_current`.
//!
//! | Observation | 7 × 7 egocentric grid encoded as `[type, color, state]` per cell  |
//! |-------------|---------------------------------------------------------------------|
//! | Action      | `TurnLeft`, `TurnRight`, `Forward`                                  |
//! | Reward      | `success_reward(steps, max_steps)` on goal; `0.0` on lava / timeout |
//!
//! # Examples
//!
//! ```rust
//! use rlevo_environments::grids::lava_gap::{LavaGapConfig, LavaGapEnv};
//! use rlevo_core::environment::{ConstructableEnv, Environment};
//!
//! let cfg = LavaGapConfig::new(5, 100, 0);
//! let mut env = LavaGapEnv::with_config(cfg, false).expect("valid config");
//! let snap = env.reset().unwrap();
//! println!("lava col: {}, gap row: {}", env.lava_col(), env.gap_row());
//! ```
//!
//! [`LavaGapEnv`]: https://minigrid.farama.org/environments/minigrid/LavaGapEnv/
//! [`Lava`]: super::core::entity::Entity::Lava

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
// `Rng` is the object-safe core trait in rand 0.10; the `random_range` sugar
// lives on the blanket-implemented `RngExt`. The public bound stays
// `R: Rng + ?Sized` per `rules.md` §8.
use rand::{Rng, RngExt as _, SeedableRng, rngs::StdRng};
use rlevo_core::config::{self, ConfigError, Validate};
use rlevo_core::environment::{ConstructableEnv, Environment, EnvironmentError, Sensor};
use rlevo_core::reward::ScalarReward;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

/// Minimum grid side length; the lava column needs at least one interior
/// cell on each side.
const MIN_SIZE: usize = 5;

/// The lava column is drawn from `2..size-2` (upstream's `_rand_int(2, w-2)`),
/// which is empty for `size < 5` and would make
/// [`random_range`](rand::RngExt::random_range) panic. `MIN_SIZE` is what keeps
/// that range non-empty, so tie the two together at compile time rather than
/// leaving the connection in a comment — this is the assertion ADR 0062's
/// consequences section asks each randomized env to carry.
const _: () = assert!(
    MIN_SIZE >= 5,
    "MIN_SIZE < 5 makes the lava-column range `2..size-2` empty"
);

/// Configuration for [`LavaGapEnv`].
///
/// # Examples
///
/// ```rust
/// use rlevo_environments::grids::lava_gap::LavaGapConfig;
///
/// let cfg = LavaGapConfig::new(7, 200, 42);
/// assert_eq!(cfg.size, 7);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LavaGapConfig {
    /// Grid side length in cells (width = height = `size`); must be ≥ `MIN_SIZE` (5).
    ///
    /// It also bounds the layout draws: the lava column comes from `2..size-2`
    /// and the gap row from `1..size-1`. At the `MIN_SIZE` floor the former is
    /// the single value `2`.
    pub size: usize,
    /// Maximum steps before the episode times out with reward `0.0`.
    pub max_steps: usize,
    /// Seed for the environment's persistent RNG, drawn once at construction.
    ///
    /// Every episode samples the **lava column** from `2..size-2` and the
    /// **gap row** from `1..size-1`, in that order — the two `_rand_int` draws
    /// upstream `MiniGrid-LavaGapS5-v0` makes in `_gen_grid`. The agent's start
    /// pose `(1, 1)` facing East and the goal at `(size-2, size-2)` are *not*
    /// sampled: upstream assigns both directly, so neither is a deviation.
    ///
    /// The RNG **advances** across resets (ADR 0029), so successive episodes
    /// get independent columns and gaps. A fixed seed therefore reproduces a
    /// fixed *sequence* of episodes, not one repeated episode. For bit-for-bit
    /// replay of a single episode use [`LavaGapEnv::reset_with_seed`].
    pub seed: u64,
}

impl LavaGapConfig {
    /// Creates a [`LavaGapConfig`] with the given parameters.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rlevo_environments::grids::lava_gap::LavaGapConfig;
    ///
    /// let cfg = LavaGapConfig::new(5, 100, 0);
    /// assert_eq!(cfg.max_steps, 100);
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

impl Default for LavaGapConfig {
    fn default() -> Self {
        let size = 5;
        Self {
            size,
            max_steps: 4 * size * size,
            seed: 0,
        }
    }
}

impl Validate for LavaGapConfig {
    /// Rejects any `size` below `MIN_SIZE` (5) and a zero `max_steps`.
    ///
    /// The `size` guard lives **here**, not only in [`FromStr`]:
    /// `LavaGapConfig` derives `Deserialize`, so a config loaded from a file is
    /// user-supplied runtime data that never passes through `from_str`
    /// (rules.md §4 — "if an invalid value can arrive via `Deserialize`, it
    /// must be an `Err`"). At `size = 1` the layout builder's `size - 2` goal
    /// placement underflows and panics; sizes between that and `MIN_SIZE` build
    /// without panicking but are outside the supported range this env
    /// documents, so the floor is enforced as policy rather than only as a
    /// crash guard.
    ///
    /// `at_least` subsumes the previous `nonzero` check — `MIN_SIZE >= 1`, so a
    /// zero `size` is still rejected, now as
    /// [`ConstraintKind::TooSmall`](rlevo_core::config::ConstraintKind::TooSmall).
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "LavaGapConfig";
        config::at_least(C, "size", self.size, MIN_SIZE)?;
        config::nonzero(C, "max_steps", self.max_steps)?;
        Ok(())
    }
}

impl FromStr for LavaGapConfig {
    type Err = String;

    /// Parses `"size=5,max_steps=100,seed=0"` (keys in any order) or the
    /// positional form `"5,100,0"`.
    ///
    /// # Errors
    ///
    /// Returns the offending key/value, or the [`Validate`] rejection — the same
    /// guard [`LavaGapEnv::with_config`] applies, so this parser cannot admit a
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

/// Minigrid's `LavaGap` environment.
///
/// A vertical lava strip bisects the room; the agent must navigate
/// through the single gap to reach the goal in the opposite corner.
/// Stepping into lava ends the episode immediately with reward `0.0`.
///
/// The strip's column and the gap's row are **re-sampled every
/// [`reset`](Environment::reset)** from a persistent RNG stream (ADR 0029/0062);
/// see the module docs for the ranges and for what stays fixed.
///
/// Implements [`Environment<3, 3, 1>`] with [`GridState`] /
/// [`GridObservation`](super::core::GridObservation) / [`GridAction`] / [`ScalarReward`].
///
/// # Examples
///
/// ```rust
/// use rlevo_environments::grids::lava_gap::LavaGapEnv;
/// use rlevo_core::environment::{ConstructableEnv, Environment};
///
/// let mut env = LavaGapEnv::new(false);
/// let snap = env.reset().unwrap();
/// println!("lava col: {}, gap row: {}", env.lava_col(), env.gap_row());
/// ```
#[derive(Debug)]
pub struct LavaGapEnv {
    state: GridState,
    config: LavaGapConfig,
    steps: usize,
    render: bool,
    /// Column the current episode's lava strip occupies.
    lava_col: i32,
    /// Row of the current episode's single gap in that strip.
    gap_row: i32,
    rng: StdRng,
}

impl LavaGapEnv {
    /// Emission-model visibility policy: does this environment's agent see
    /// through opaque cells?
    ///
    /// The rlevo spelling of canonical Minigrid's `see_through_walls`
    /// constructor argument. Read only by this environment's [`Sensor`] impl,
    /// and an inherent const rather than a config field because it is part of
    /// the task definition, not a knob a caller tunes.
    ///
    /// [`Visibility::Occluded`], because canonical `minigrid/envs/lavagap.py`
    /// passes `see_through_walls=False` explicitly — which is also
    /// `MiniGridEnv.__init__`'s own default, so the env restates the canonical
    /// behaviour rather than opting into it. Note that lava itself is
    /// *transparent* ([`Entity::see_behind`] follows canonical `WorldObj`, and
    /// only `Wall`/shut `Door` override it), so what this occludes here is the
    /// world behind the border walls, not the world behind the lava strip. See
    /// ADR 0063 (`docs/adr/0063-grid-visibility-occlusion.md`) for the whole
    /// twelve-environment table.
    ///
    /// [`Entity::see_behind`]: super::core::entity::Entity::see_behind
    const VISIBILITY: Visibility = Visibility::Occluded;

    /// Constructs a [`LavaGapEnv`] from an explicit configuration.
    ///
    /// Seeds the persistent RNG once from `config.seed` and immediately builds
    /// a first board from it. Call [`Environment::reset`] before the first
    /// [`Environment::step`] to obtain the first observation; that reset samples
    /// a *fresh* board, advancing the stream.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rlevo_environments::grids::lava_gap::{LavaGapConfig, LavaGapEnv};
    ///
    /// let env = LavaGapEnv::with_config(
    ///     LavaGapConfig::new(7, 200, 42),
    ///     true, // render ASCII grid to stdout
    /// )
    /// .expect("valid config");
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `config` fails [`Validate`]: a `size` below
    /// `MIN_SIZE` (5) or a zero `max_steps`. This is the construction chokepoint
    /// (rules.md §4), so it also rejects a config that arrived by `Deserialize`
    /// or struct-update syntax without passing through [`FromStr`].
    pub fn with_config(config: LavaGapConfig, render: bool) -> Result<Self, ConfigError> {
        config.validate()?;
        let mut rng = StdRng::seed_from_u64(config.seed);
        let (state, lava_col, gap_row) = Self::build(&config, &mut rng);
        Ok(Self {
            state,
            config,
            steps: 0,
            render,
            lava_col,
            gap_row,
            rng,
        })
    }

    /// Re-seed the persistent RNG to `seed`, then [`reset`](Environment::reset).
    ///
    /// Ordinary [`reset`](Environment::reset) advances the persistent stream so
    /// successive episodes get independent lava columns and gap rows (ADR 0029);
    /// use this when you need a *specific* board to reproduce bit-for-bit (e.g.
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

    /// Returns the environment's active configuration.
    #[must_use]
    pub const fn config(&self) -> &LavaGapConfig {
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

    /// Column index where **this episode's** lava strip sits.
    ///
    /// Re-sampled from `2..size-2` at every [`reset`](Environment::reset), so
    /// read it back after the reset rather than deriving it from
    /// [`LavaGapConfig::size`].
    #[must_use]
    pub const fn lava_col(&self) -> i32 {
        self.lava_col
    }

    /// Row index of **this episode's** single gap in the lava strip.
    ///
    /// Re-sampled from `1..size-1` at every [`reset`](Environment::reset), so
    /// read it back after the reset rather than deriving it from
    /// [`LavaGapConfig::size`].
    #[must_use]
    pub const fn gap_row(&self) -> i32 {
        self.gap_row
    }

    /// Renders the current grid state as an ASCII string.
    #[must_use]
    pub fn ascii(&self) -> String {
        render_ascii(&self.state.grid, &self.state.agent)
    }

    /// Build a fresh episode: a walled room, a goal in the far corner, and a
    /// lava column with one gap — both of the latter's coordinates sampled.
    ///
    /// Returns the state together with `(lava_col, gap_row)` so the caller can
    /// cache the draw for [`lava_col`](Self::lava_col) / [`gap_row`](Self::gap_row).
    ///
    /// Draws from `rng` and lets it advance (ADR 0029) — never re-seeds.
    ///
    /// # Fidelity
    ///
    /// Mirrors upstream `LavaGapEnv._gen_grid`: perimeter, agent at `(1, 1)`
    /// facing East, goal at `(size-2, size-2)`, then **column before row** —
    /// `_rand_int(2, w-2)` then `_rand_int(1, h-1)`, both half-open in Python
    /// and so written as Rust `2..size-2` / `1..size-1`. Draw order is part of
    /// the algorithm (ADR 0062 §1); swapping the two lines changes every seeded
    /// board.
    ///
    /// These are index draws over a coordinate range, not free-cell draws, so
    /// they deliberately use [`random_range`](rand::RngExt::random_range)
    /// directly rather than [`sample_pos`](super::core::sample_pos). Routing
    /// them through the placement sampler would fold two independent uniform
    /// draws into one draw over a rectangle and would silently re-weight the
    /// distribution if any cell in that rectangle were ever occupied.
    fn build<R: Rng + ?Sized>(config: &LavaGapConfig, rng: &mut R) -> (GridState, i32, i32) {
        let mut grid = Grid::new(config.size, config.size);
        grid.draw_walls();

        #[allow(clippy::cast_possible_wrap)]
        let size = config.size as i32;

        let agent = AgentState::new(1, 1, Direction::East);
        grid.set(size - 2, size - 2, Entity::Goal);

        // `MIN_SIZE >= 5` (asserted at compile time above) keeps `2..size-2`
        // non-empty; `1..size-1` is non-empty for any `size >= 2`.
        let lava_col = rng.random_range(2..size - 2);
        let gap_row = rng.random_range(1..size - 1);

        for y in 1..size - 1 {
            if y != gap_row {
                grid.set(lava_col, y, Entity::Lava);
            }
        }

        (GridState::new(grid, agent), lava_col, gap_row)
    }

    fn emit(&self, observation: GridObservation, reward: f32, done: bool) -> GridSnapshot {
        if self.render {
            println!("{}", self.ascii());
        }
        build_snapshot(observation, reward, done)
    }
}

impl crate::render::AsciiRenderable for LavaGapEnv {
    fn render_ascii(&self) -> String {
        render_ascii(&self.state.grid, &self.state.agent)
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        super::core::render::render_styled(&self.state.grid, &self.state.agent)
    }
}

impl Display for LavaGapEnv {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LavaGapEnv(size={}, step={}/{})",
            self.config.size, self.steps, self.config.max_steps
        )
    }
}

impl ConstructableEnv for LavaGapEnv {
    fn new(render: bool) -> Self {
        Self::with_config(LavaGapConfig::default(), render).expect("default config must validate")
    }
}

impl Sensor<3, 1, 3> for LavaGapEnv {
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

impl Environment<3, 3, 1> for LavaGapEnv {
    type StateType = GridState;
    type ObservationType = super::core::GridObservation;
    type ActionType = GridAction;
    type RewardType = ScalarReward;
    type SnapshotType = GridSnapshot;

    /// Samples a fresh board from the **persistent** RNG stream.
    ///
    /// Per ADR 0029 the stream is *not* re-seeded here: successive resets draw
    /// independent lava columns and gap rows. Use
    /// [`reset_with_seed`](Self::reset_with_seed) for deterministic replay.
    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        let (state, lava_col, gap_row) = Self::build(&self.config, &mut self.rng);
        self.state = state;
        self.lava_col = lava_col;
        self.gap_row = gap_row;
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

impl rlevo_core::render::payload::GridPayloadSource for LavaGapEnv {
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
    // Test-only: the `Visibility` dispatch point, the raw window extractor, and
    // the byte a masked cell carries — all read by
    // `test_lava_gap_occlusion_masks_only_the_world_outside_the_room`.
    use super::super::core::grid::egocentric_view;
    use super::super::core::{UNSEEN_TYPE, VIEW_SIZE, mask_view};
    use crate::grids::core::is_free;
    use rlevo_core::config::ConstraintKind;
    use rlevo_core::environment::Snapshot;
    use std::collections::HashSet;

    /// The four facings, for the occlusion sweep.
    const DIRECTIONS: [Direction; 4] = [
        Direction::North,
        Direction::East,
        Direction::South,
        Direction::West,
    ];

    /// A same-shaped board of transparent cells, used as an **in-grid mask**.
    ///
    /// [`Grid::get`] reads out of bounds as [`Entity::Wall`] (matching canonical
    /// `Grid.slice`), so the window `egocentric_view` cuts from this board is
    /// [`Entity::Floor`] exactly where that window cell lies inside the world and
    /// [`Entity::Wall`] exactly where it does not. That is the view geometry read
    /// off the real extractor, rather than re-derived by a test-local copy of
    /// `rotate_view_offset` that could drift away from it.
    fn all_floor_like(grid: &Grid) -> Grid {
        let (w, h) = (grid.width(), grid.height());
        let mut probe = Grid::new(w, h);
        let (w, h) = (
            i32::try_from(w).expect("test grids fit in i32"),
            i32::try_from(h).expect("test grids fit in i32"),
        );
        for y in 0..h {
            for x in 0..w {
                probe.set(x, y, Entity::Floor);
            }
        }
        probe
    }

    /// Every seed loop runs at both of these.
    ///
    /// `MIN_SIZE` is where the draw ranges degenerate — the column range
    /// `2..size-2` collapses to the single value `2` there — and `9` is a size
    /// where both quantities move. An invariant that only holds at one of the
    /// two is a bug, and running both is what surfaces it.
    const SIZES: [usize; 2] = [MIN_SIZE, 9];

    /// How many seeds each loop sweeps.
    ///
    /// Sized by the *weakest* draw in the suite rather than picked round: at
    /// `MIN_SIZE` the gap row has exactly three values, so "every seed happened
    /// to draw the same row" has probability `3 * (1/3)^SEEDS ≈ 10^-30` at 64.
    /// That is many orders below the number of times this suite will ever run,
    /// so a failure here is evidence of a pinned draw, not of bad luck.
    const SEEDS: u64 = 64;

    fn env_5x5() -> LavaGapEnv {
        LavaGapEnv::with_config(LavaGapConfig::new(5, 100, 0), false).expect("valid config")
    }

    /// A square env at `size` with the same `4 * size * size` budget
    /// [`LavaGapConfig::default`] picks, so no seed loop is quietly handed a
    /// budget the shipped default would not have.
    fn env_of(size: usize) -> LavaGapEnv {
        LavaGapEnv::with_config(LavaGapConfig::new(size, 4 * size * size, 0), false)
            .expect("valid config")
    }

    /// An optimal route across whatever board `env` is **currently** holding.
    ///
    /// Every coordinate is read back from the environment, so the script stays
    /// correct for every sampled layout: descend column `x = 1` to the gap row,
    /// cross east through the gap to the goal column, then descend to the goal.
    /// A fixed action list cannot state that property once the board moves.
    fn route_to_goal(env: &LavaGapEnv) -> Vec<GridAction> {
        #[allow(clippy::cast_possible_wrap)]
        let goal = (env.config().size - 2) as i32;
        let gap_row = env.gap_row();
        let mut script = Vec::new();
        if gap_row > 1 {
            script.push(GridAction::TurnRight); // face South
            for _ in 1..gap_row {
                script.push(GridAction::Forward);
            }
            script.push(GridAction::TurnLeft); // face East again
        }
        // x: 1 -> goal, passing through the gap cell (lava_col, gap_row).
        for _ in 1..goal {
            script.push(GridAction::Forward);
        }
        if goal > gap_row {
            script.push(GridAction::TurnRight); // face South
            for _ in gap_row..goal {
                script.push(GridAction::Forward);
            }
        }
        script
    }

    /// Step `script` through `env`, stopping at the first terminal snapshot.
    ///
    /// The derived scripts above may overshoot by design (a route that reaches
    /// the goal early, a walk that hits lava early); stopping on `done` keeps
    /// the tests from stepping a finished episode.
    fn run_to_completion(env: &mut LavaGapEnv, script: &[GridAction]) -> GridSnapshot {
        let mut last = None;
        for &action in script {
            let snap = env.step(action).unwrap();
            let done = snap.is_done();
            last = Some(snap);
            if done {
                break;
            }
        }
        last.expect("script must contain at least one action")
    }

    #[test]
    fn default_config_validates() {
        assert!(LavaGapConfig::default().validate().is_ok());
    }

    #[test]
    fn rejects_zero_size() {
        let bad = LavaGapConfig {
            size: 0,
            ..Default::default()
        };
        assert!(LavaGapEnv::with_config(bad, false).is_err());
    }

    #[test]
    fn default_config_values() {
        let cfg = LavaGapConfig::default();
        assert_eq!(cfg.size, 5);
        assert_eq!(cfg.max_steps, 4 * 5 * 5);
    }

    #[test]
    fn fromstr_keyvalue() {
        let cfg: LavaGapConfig = "size=6,max_steps=40".parse().unwrap();
        assert_eq!(cfg.size, 6);
        assert_eq!(cfg.max_steps, 40);
    }

    #[test]
    fn fromstr_rejects_small_size() {
        assert!("3".parse::<LavaGapConfig>().is_err());
    }

    /// Issue #106: `MIN_SIZE` was enforced only in [`FromStr`], so a config
    /// built by `Deserialize` or struct-update syntax reached `build`, where
    /// `size - 2` underflowed and panicked at `size = 1`. The guard now lives
    /// in [`Validate`], which `with_config` runs (ADR 0026 chokepoint), and it
    /// rejects the whole sub-`MIN_SIZE` range — including sizes that happen to
    /// build a well-formed board.
    #[test]
    fn with_config_rejects_size_below_min() {
        let bad = LavaGapConfig {
            size: MIN_SIZE - 1,
            ..Default::default()
        };
        let err = LavaGapEnv::with_config(bad, false).unwrap_err();
        assert_eq!(err.config, "LavaGapConfig");
        assert_eq!(err.field, "size");
        assert_eq!(
            err.kind,
            ConstraintKind::TooSmall {
                min: MIN_SIZE as u64,
                got: (MIN_SIZE - 1) as u64,
            }
        );
    }

    /// The two sampled quantities are asserted **separately**.
    ///
    /// A whole-board "consecutive resets differ" check passes while one of the
    /// two draws is stuck, because the other one alone makes the boards differ.
    /// That is the exact trap #282 and ADR 0062 §4 name, so each quantity gets
    /// its own non-degeneracy assertion.
    #[test]
    fn lava_column_and_gap_row_vary_separately_across_seeds() {
        for size in SIZES {
            let mut env = env_of(size);
            let mut cols = HashSet::new();
            let mut rows = HashSet::new();
            for seed in 0..SEEDS {
                env.reset_with_seed(seed).expect("reset");
                cols.insert(env.lava_col());
                rows.insert(env.gap_row());
            }

            assert!(
                rows.len() >= 2,
                "size {size}: the gap row never moved across {SEEDS} seeds, saw {rows:?}"
            );

            if size == MIN_SIZE {
                // Verified by hand against upstream: the column is drawn from
                // `_rand_int(2, width - 2)` = Rust `2..3` at size 5, a range
                // with one member. The column genuinely *cannot* vary here, so
                // assert the true invariant rather than a variability that does
                // not exist — and assert the value, so a range that silently
                // widens or shifts still fails.
                assert_eq!(
                    cols,
                    HashSet::from([2]),
                    "at MIN_SIZE the lava column range `2..3` admits only 2, saw {cols:?}"
                );
            } else {
                assert!(
                    cols.len() >= 2,
                    "size {size}: the lava column never moved across {SEEDS} seeds, saw {cols:?}"
                );
            }
        }
    }

    /// The draws cover their **whole** range, both endpoints included.
    ///
    /// `2..size-2` and `1..size-1` are the Rust translation of Python's
    /// half-open `_rand_int(2, w-2)` / `_rand_int(1, h-1)`, and that interval
    /// conversion is the easiest thing here to get wrong by one. Membership
    /// (asserted in [`every_sampled_board_is_well_formed`]) catches a range that
    /// came out too *wide*; only coverage catches one that came out too
    /// *narrow*, which is the direction an off-by-one on a half-open bound
    /// actually fails in.
    ///
    /// Run at size 9, where the sets are `{2,3,4,5,6}` and `{1,…,7}`.
    /// `WIDE_SEEDS` is sized so that missing any one of the seven rows by chance
    /// has probability `7 * (6/7)^512 ≈ 5e-34`; the assertion is deterministic
    /// for every practical purpose, and 512 resets of a 9×9 board are free.
    #[test]
    fn the_draws_cover_their_whole_range() {
        const WIDE_SEEDS: u64 = 512;
        let mut env = env_of(9);
        let mut cols = HashSet::new();
        let mut rows = HashSet::new();
        for seed in 0..WIDE_SEEDS {
            env.reset_with_seed(seed).expect("reset");
            cols.insert(env.lava_col());
            rows.insert(env.gap_row());
        }
        assert_eq!(
            cols,
            (2..7).collect::<HashSet<i32>>(),
            "the lava column range is not `2..size-2` at size 9"
        );
        assert_eq!(
            rows,
            (1..8).collect::<HashSet<i32>>(),
            "the gap row range is not `1..size-1` at size 9"
        );
    }

    /// Structural invariants that must hold for **every** sampled board.
    ///
    /// This is what replaces the cell-pinning assertions the fixed layout used
    /// to allow: the same facts, quantified over the whole generated
    /// distribution instead of over one board.
    #[test]
    fn every_sampled_board_is_well_formed() {
        for size in SIZES {
            let mut env = env_of(size);
            #[allow(clippy::cast_possible_wrap)]
            let n = size as i32;

            for seed in 0..SEEDS {
                env.reset_with_seed(seed).expect("reset");
                let col = env.lava_col();
                let row = env.gap_row();
                let state = env.state();
                let grid = &state.grid;
                let agent = &state.agent;
                let board = env.ascii();
                let at = format!("size {size} seed {seed}:\n{board}");

                // The upstream ranges, restated in Rust's half-open form.
                assert!(
                    (2..n - 2).contains(&col),
                    "{at}lava column {col} outside `2..{}`",
                    n - 2
                );
                assert!(
                    (1..n - 1).contains(&row),
                    "{at}gap row {row} outside `1..{}`",
                    n - 1
                );

                // The gap is in bounds and strictly inside the interior — never
                // in the perimeter ring.
                assert!(
                    grid.in_bounds(col, row),
                    "{at}gap ({col}, {row}) out of bounds"
                );
                assert!(
                    (1..=n - 2).contains(&col) && (1..=n - 2).contains(&row),
                    "{at}gap ({col}, {row}) is not strictly interior"
                );
                assert_eq!(
                    grid.get(col, row),
                    Entity::Empty,
                    "{at}the gap must be walkable"
                );

                // The perimeter is all wall.
                for i in 0..n {
                    for (x, y) in [(i, 0), (i, n - 1), (0, i), (n - 1, i)] {
                        assert_eq!(grid.get(x, y), Entity::Wall, "{at}({x}, {y}) is not a wall");
                    }
                }

                // Exactly one goal, in the corner upstream fixes it to.
                let goals: Vec<(i32, i32)> = (0..n)
                    .flat_map(|y| (0..n).map(move |x| (x, y)))
                    .filter(|&(x, y)| grid.get(x, y) == Entity::Goal)
                    .collect();
                assert_eq!(goals, vec![(n - 2, n - 2)], "{at}wrong goal set");

                // The strip is contiguous lava down the sampled column, minus
                // exactly the one gap cell. Comparing the whole cell list also
                // proves no lava escaped to another column.
                let lava: Vec<(i32, i32)> = (0..n)
                    .flat_map(|y| (0..n).map(move |x| (x, y)))
                    .filter(|&(x, y)| grid.get(x, y) == Entity::Lava)
                    .collect();
                let expected: Vec<(i32, i32)> =
                    (1..n - 1).filter(|&y| y != row).map(|y| (col, y)).collect();
                assert_eq!(lava, expected, "{at}lava strip is not one gapped column");

                // The agent's pose is fixed (upstream assigns it directly) and
                // its cell is free by the shared placement predicate.
                assert_eq!((agent.x, agent.y), (1, 1), "{at}agent moved off (1, 1)");
                assert_eq!(
                    agent.direction,
                    Direction::East,
                    "{at}agent is not facing East"
                );
                assert!(
                    is_free(grid, agent.x, agent.y),
                    "{at}agent does not start on a free cell"
                );
                let under = grid.get(agent.x, agent.y);
                assert_ne!(under, Entity::Lava, "{at}agent starts on lava");
                assert_ne!(under, Entity::Goal, "{at}agent starts on the goal");
            }
        }
    }

    /// Plain `reset()` draws from the persistent stream, so the board moves
    /// between episodes without any re-seeding.
    ///
    /// The loop length is the same argument as [`SEEDS`]: at `MIN_SIZE` only the
    /// gap row varies, over three values, so "all `SEEDS` boards identical" has
    /// probability `3 * (1/3)^SEEDS`. Picking a small number here — two resets,
    /// say — would fail once in nine runs at `MIN_SIZE` purely by chance.
    #[test]
    fn consecutive_resets_draw_different_boards() {
        for size in SIZES {
            let mut env = env_of(size);
            let mut boards = HashSet::new();
            for _ in 0..SEEDS {
                env.reset().expect("reset");
                boards.insert(env.ascii());
            }
            assert!(
                boards.len() >= 2,
                "size {size}: {SEEDS} consecutive resets produced one board — \
                 the stream is being re-seeded or the draw is dead"
            );
        }
    }

    /// `reset_with_seed` is the single-episode replay hatch (ADR 0029).
    #[test]
    fn reset_with_seed_replays_one_episode_bit_for_bit() {
        for size in SIZES {
            let mut env = env_of(size);
            let first = env.reset_with_seed(7).expect("reset");
            let board = env.ascii();

            // Advance the stream in between, so a replay cannot succeed merely
            // because nothing moved.
            env.reset().expect("reset");
            env.reset().expect("reset");

            let again = env.reset_with_seed(7).expect("reset");
            assert_eq!(
                first.observation(),
                again.observation(),
                "size {size}: reset_with_seed(7) did not replay its observation"
            );
            assert_eq!(
                board,
                env.ascii(),
                "size {size}: reset_with_seed(7) did not replay its board"
            );
        }
    }

    /// Two different replay seeds give two different boards.
    ///
    /// At `MIN_SIZE` only three distinct boards exist (one column, three gap
    /// rows), so this is a spot check on a pair verified by inspection, not a
    /// law that holds for every pair — one third of seed pairs must collide
    /// there. The non-degeneracy sweep above is what carries the general claim.
    #[test]
    fn different_replay_seeds_give_different_boards() {
        for size in SIZES {
            let mut env = env_of(size);
            env.reset_with_seed(1).expect("reset");
            let one = env.ascii();
            env.reset_with_seed(2).expect("reset");
            let two = env.ascii();
            assert_ne!(one, two, "size {size}: seeds 1 and 2 produced one board");
        }
    }

    /// The module-level ASCII snapshots are real output, and stay real.
    ///
    /// The docs show two named boards, which is what carries the documentation
    /// value the deleted cell-pinning assertions used to. A doc block is not
    /// compiled, so this is what stops it rotting: render the same two episodes
    /// and compare, modulo the trailing space [`render_ascii`] puts after every
    /// glyph.
    #[test]
    fn module_doc_snapshots_are_current() {
        fn board(env: &LavaGapEnv) -> String {
            env.ascii()
                .lines()
                .map(str::trim_end)
                .collect::<Vec<_>>()
                .join("\n")
        }

        // `LavaGapConfig::new(5, 100, 0)`, `reset_with_seed(0)`.
        let mut small = env_of(MIN_SIZE);
        small.reset_with_seed(0).expect("reset");
        assert_eq!((small.lava_col(), small.gap_row()), (2, 3));
        assert_eq!(
            board(&small),
            concat!(
                "# # # # #\n",
                "# > L . #\n",
                "# . L . #\n",
                "# . . G #\n",
                "# # # # #",
            )
        );

        // `LavaGapConfig::new(9, 324, 0)`, `reset_with_seed(0)`.
        let mut large = env_of(9);
        large.reset_with_seed(0).expect("reset");
        assert_eq!((large.lava_col(), large.gap_row()), (6, 6));
        assert_eq!(
            board(&large),
            concat!(
                "# # # # # # # # #\n",
                "# > . . . . L . #\n",
                "# . . . . . L . #\n",
                "# . . . . . L . #\n",
                "# . . . . . L . #\n",
                "# . . . . . L . #\n",
                "# . . . . . . . #\n",
                "# . . . . . L G #\n",
                "# # # # # # # # #",
            )
        );
    }

    #[test]
    fn reset_is_deterministic() {
        let cfg = LavaGapConfig::new(5, 100, 11);
        let mut a = LavaGapEnv::with_config(cfg, false).expect("valid config");
        let mut b = LavaGapEnv::with_config(cfg, false).expect("valid config");
        let sa = a.reset().unwrap();
        let sb = b.reset().unwrap();
        assert_eq!(sa.observation(), sb.observation());
    }

    /// Walking into the strip anywhere other than the gap ends the episode for
    /// nothing — on every sampled board, not just the one that used to be fixed.
    #[test]
    fn stepping_into_lava_terminates_with_zero_reward() {
        for size in SIZES {
            let mut env = env_of(size);
            for seed in 0..SEEDS {
                env.reset_with_seed(seed).expect("reset");
                // Walk east along an interior row that is *not* the gap row.
                // Row 1 is the agent's own row and works unless the gap landed
                // there; row 2 is interior at every size >= MIN_SIZE.
                let blocked_row = if env.gap_row() == 1 { 2 } else { 1 };
                let mut script = Vec::new();
                if blocked_row > 1 {
                    script.push(GridAction::TurnRight); // face South
                    for _ in 1..blocked_row {
                        script.push(GridAction::Forward);
                    }
                    script.push(GridAction::TurnLeft); // face East again
                }
                // x: 1 -> lava_col; the last Forward enters the strip.
                for _ in 1..env.lava_col() {
                    script.push(GridAction::Forward);
                }

                let snap = run_to_completion(&mut env, &script);
                assert!(
                    snap.is_done(),
                    "size {size} seed {seed}: walking into the strip did not terminate:\n{}",
                    env.ascii()
                );
                let reward: f32 = (*snap.reward()).into();
                assert_eq!(reward, 0.0, "size {size} seed {seed}: lava paid {reward}");
            }
        }
    }

    /// A route derived from the board reaches the goal on every sampled layout.
    ///
    /// The script is computed from [`LavaGapEnv::gap_row`] and the config size,
    /// never hard-coded — which is the only form of this test that survives the
    /// board moving per episode.
    #[test]
    fn derived_rollout_crosses_gap_and_reaches_goal() {
        for size in SIZES {
            let mut env = env_of(size);
            for seed in 0..SEEDS {
                env.reset_with_seed(seed).expect("reset");
                let script = route_to_goal(&env);
                let snap = run_to_completion(&mut env, &script);
                assert!(
                    snap.is_done(),
                    "size {size} seed {seed}: the derived route did not terminate:\n{}",
                    env.ascii()
                );
                let reward: f32 = (*snap.reward()).into();
                assert!(
                    reward > 0.0,
                    "size {size} seed {seed}: the derived route paid {reward}:\n{}",
                    env.ascii()
                );
            }
        }
    }

    #[test]
    fn timeout_ends_with_zero_reward() {
        let cfg = LavaGapConfig::new(5, 3, 0);
        let mut env = LavaGapEnv::with_config(cfg, false).expect("valid config");
        env.reset().unwrap();
        env.step(GridAction::TurnLeft).unwrap();
        env.step(GridAction::TurnLeft).unwrap();
        let snap = env.step(GridAction::TurnLeft).unwrap();
        assert!(snap.is_done());
        let reward: f32 = (*snap.reward()).into();
        assert_eq!(reward, 0.0);
    }

    /// The accessors report the board actually built, not a formula.
    ///
    /// They were closed-form functions of `config.size` before #282; a caller
    /// that trusted them would now be reading a different board than the one it
    /// was handed, so pin them against the grid itself.
    #[test]
    fn accessors_agree_with_the_built_grid() {
        for size in SIZES {
            let mut env = env_of(size);
            for seed in 0..SEEDS {
                env.reset_with_seed(seed).expect("reset");
                let (col, row) = (env.lava_col(), env.gap_row());
                let grid = &env.state().grid;
                assert_eq!(
                    grid.get(col, row),
                    Entity::Empty,
                    "size {size} seed {seed}: gap_row does not name the gap"
                );
                let other = if row == 1 { 2 } else { 1 };
                assert_eq!(
                    grid.get(col, other),
                    Entity::Lava,
                    "size {size} seed {seed}: lava_col does not name the strip"
                );
            }
        }
    }

    #[test]
    fn reset_clears_step_counter() {
        let mut env = env_5x5();
        env.reset().unwrap();
        env.step(GridAction::TurnLeft).unwrap();
        env.step(GridAction::TurnLeft).unwrap();
        assert_eq!(env.steps(), 2);
        env.reset().unwrap();
        assert_eq!(env.steps(), 0);
    }

    /// Occlusion is live here, but **nothing it hides is task-relevant**: every
    /// masked cell lies outside the room.
    ///
    /// This environment has no decision-relevant occluded pose, and that is a
    /// measured finding rather than a gap in this test. Lava is *transparent* —
    /// canonical `WorldObj.see_behind` is overridden only by `Wall` and shut
    /// `Door`, which [`LavaGapEnv::VISIBILITY`]'s own doc already says — and the
    /// board is a plain rectangle whose only opaque cells are the border ring.
    /// The window's intersection with the room is therefore a rectangle of
    /// transparent cells containing the agent, and `process_vis` is a forward
    /// flood fill, so it lights every one of them. The lava strip, its gap and
    /// the goal are visible from every pose they fall into.
    ///
    /// The test asserts what is actually true, at both ends:
    ///
    /// 1. **The start pose always masks something.** That is the pose every
    ///    episode begins from, and it is what makes this a regression guard:
    ///    flip [`LavaGapEnv::VISIBILITY`] to [`Visibility::SeeThrough`] and this
    ///    clause fails.
    /// 2. **Every masked cell — at every pose — lies outside the room**, swept
    ///    over sizes 5/7/9 × 16 seeds × every passable cell × all four facings.
    ///    Occlusion cannot cost this agent one cell of the board it must solve.
    ///
    /// Clause 2 on its own is vacuous under `SeeThrough` (nothing is masked, so
    /// nothing violates it). Clause 1 is what makes the pair bite. If clause 2
    /// ever starts failing, the board grew an interior wall and this environment
    /// finally has an occlusion story worth naming a pose for.
    #[test]
    fn test_lava_gap_occlusion_masks_only_the_world_outside_the_room() {
        for size in [5usize, 7, 9] {
            let mut env = env_of(size);
            for seed in 0..16u64 {
                let snapshot = env.reset_with_seed(seed).expect("reset");
                let grid = env.state().grid.clone();
                let in_grid = all_floor_like(&grid);

                // (1) The start pose — fixed at (1, 1) facing East, as upstream.
                let start = env.state().agent;
                assert_eq!(
                    (start.x, start.y, start.direction),
                    (1, 1, Direction::East),
                    "size {size} seed {seed}: the start pose this clause names has moved"
                );
                let masked = mask_view(&grid, &start, LavaGapEnv::VISIBILITY);
                let occluded = *snapshot.observation();
                let see_through = observe_grid(env.state(), Visibility::SeeThrough);
                let hidden = masked.iter().flatten().filter(|c| c.is_none()).count();
                assert!(
                    hidden > 0,
                    "size {size} seed {seed}: the border ring must mask something from \
                     the start pose, or occlusion is switched off"
                );
                // Differ *exactly* on the hidden set, in both directions.
                for (r, c) in (0..VIEW_SIZE).flat_map(|r| (0..VIEW_SIZE).map(move |c| (r, c))) {
                    if masked[r][c].is_none() {
                        assert_eq!(
                            occluded.view[r][c],
                            [UNSEEN_TYPE, 0, 0],
                            "size {size} seed {seed}: hidden cell ({r}, {c}) must encode \
                             as the unseen triple"
                        );
                        assert_ne!(
                            occluded.view[r][c], see_through.view[r][c],
                            "size {size} seed {seed}: hidden cell ({r}, {c}) must not \
                             encode the same as the seen one"
                        );
                    } else {
                        assert_eq!(
                            occluded.view[r][c], see_through.view[r][c],
                            "size {size} seed {seed}: visible cell ({r}, {c}) must be \
                             unaffected by occlusion"
                        );
                    }
                }

                // (2) No pose anywhere hides a cell of the room itself.
                #[allow(clippy::cast_possible_wrap)]
                let side = size as i32;
                for y in 0..side {
                    for x in 0..side {
                        if !grid.get(x, y).is_passable() {
                            continue;
                        }
                        for dir in DIRECTIONS {
                            let agent = AgentState::new(x, y, dir);
                            let masked = mask_view(&grid, &agent, LavaGapEnv::VISIBILITY);
                            let inside = egocentric_view(&in_grid, &agent);
                            for r in 0..VIEW_SIZE {
                                for c in 0..VIEW_SIZE {
                                    assert!(
                                        masked[r][c].is_some() || inside[r][c] == Entity::Wall,
                                        "size {size} seed {seed}: ({x}, {y}) facing {dir:?} \
                                         masks the in-room window cell ({r}, {c}) — lava is \
                                         transparent, so only the border can occlude here"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
