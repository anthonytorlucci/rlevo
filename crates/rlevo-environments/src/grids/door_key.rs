//! Pick up the key, unlock the door, and reach the goal in two rooms.
//!
//! Ports Farama Minigrid's [`DoorKeyEnv`]. A vertical interior wall splits the
//! `N×N` grid into a left room (agent + key) and a right room (goal). The wall
//! carries exactly one [`Locked`] door; the agent must pick up the matching
//! yellow key before it can toggle that door open.
//!
//! ## The board is sampled every episode
//!
//! Upstream's `_gen_grid` makes four draws, and this port makes the same four
//! in the same order (ADR 0062 §1 — draw order is part of the algorithm):
//!
//! | # | quantity | upstream | here |
//! |---|----------|----------|------|
//! | 1 | split column | `_rand_int(2, width - 2)` | `rng.random_range(2..size - 2)` |
//! | 2 | agent cell **and** facing | `place_agent(size=(split, height))` | [`place_agent`] over the left room |
//! | 3 | door row | `_rand_int(1, height - 2)` | `rng.random_range(1..size - 2)` |
//! | 4 | key cell | `place_obj(top=(0, 0), size=(split, height))` | [`place_obj`] over the left room |
//!
//! Only the goal is fixed, at `(size - 2, size - 2)`, and the door colour is
//! hard-coded yellow — both match upstream.
//!
//! The ordering is load-bearing, not stylistic: the agent is placed *before*
//! the key and the key's draw rejects the agent's cell, so the key can never
//! land under the agent and is always reachable without opening the door.
//!
//! ## One episode (`size = 5`, `reset_with_seed(3)`)
//!
//! Exactly what [`ascii`](DoorKeyEnv::ascii) prints for that seed — the test
//! `module_doc_snapshot_is_current` pins it, so this picture cannot rot:
//!
//! ```text
//! # # # # #
//! # ^ * . #
//! # k # . #
//! # . # G #
//! # # # # #
//! ```
//!
//! - `#` — border wall, or the split column, here `x = 2`
//! - `k` — yellow key, at `(1, 2)`, somewhere in the left room
//! - `*` — locked door, at `(2, 1)`: the single non-wall cell of the column
//! - `^` — agent, at `(1, 1)` facing North
//! - `G` — goal, always the bottom-right interior cell
//! - `.` — empty passable cell
//!
//! Nothing in that picture except the perimeter and the `G` survives the next
//! [`reset`](Environment::reset): a different seed moves the split column, the
//! door row, the key, and the agent's pose and facing. Use
//! [`DoorKeyEnv::reset_with_seed`] to pin one episode, and
//! [`DoorKeyEnv::split_col`] / [`DoorKeyEnv::door_row`] to read the board back
//! rather than assuming it.
//!
//! ## Required action sequence
//!
//! 1. Reach a cell facing the key, then `Pickup`
//! 2. Reach the cell west of the door facing east, then `Toggle` twice —
//!    unlock, then open
//! 3. Navigate through the door to the goal
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
//! use rlevo_environments::grids::door_key::{DoorKeyConfig, DoorKeyEnv};
//! use rlevo_core::environment::{ConstructableEnv, Environment};
//!
//! let cfg = DoorKeyConfig::new(5, 100, 0);
//! let mut env = DoorKeyEnv::with_config(cfg, false).expect("valid config");
//! let _snapshot = env.reset().unwrap();
//! ```
//!
//! [`DoorKeyEnv`]: https://minigrid.farama.org/environments/minigrid/DoorKeyEnv/
//! [`Locked`]: super::core::entity::DoorState::Locked
//! [`place_agent`]: super::core::place_agent
//! [`place_obj`]: super::core::place_obj

use super::core::{
    GridSnapshot, Visibility,
    action::GridAction,
    build_snapshot,
    color::Color,
    dynamics::{StepOutcome, apply_action},
    entity::{DoorState, Entity},
    grid::Grid,
    observation::GridObservation,
    observe_grid,
    placement::{PlacementError, Rect, no_reject, place_agent, place_obj},
    render::render_ascii,
    reward::success_reward,
    state::GridState,
};
use crate::episode::EpisodeGuard;
use rand::rngs::StdRng;
use rand::{Rng, RngExt as _, SeedableRng};
use rlevo_core::config::{self, ConfigError, Validate};
use rlevo_core::environment::{ConstructableEnv, Environment, EnvironmentError, Sensor, Snapshot};
use rlevo_core::reward::ScalarReward;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

/// Minimum side length; the split wall needs a left and right sub-room.
const MIN_SIZE: usize = 5;
/// Color used for the key, door, and lock.
const DOOR_COLOR: Color = Color::Yellow;

// `DoorKeyEnv::build` draws the split column from `2..size - 2` and the door
// row from `1..size - 2`; `random_range` panics on an empty range, so both are
// non-empty exactly when `size >= 5`. `MIN_SIZE` is what guarantees that, and
// this assertion is what stops a future contributor from lowering the floor
// without noticing the coupling (ADR 0062 §5's `four_rooms` note, same shape).
const _: () = assert!(
    MIN_SIZE >= 5,
    "MIN_SIZE < 5 leaves an empty split-column or door-row sampling range"
);

/// Configuration for [`DoorKeyEnv`].
///
/// Controls the grid size, episode length, and RNG seed. Every position except
/// the goal is drawn per episode from the seed's stream, so the config carries
/// no positional fields.
///
/// # Examples
///
/// ```rust
/// use rlevo_environments::grids::door_key::DoorKeyConfig;
///
/// let cfg = DoorKeyConfig::new(7, 196, 0);
/// assert_eq!(cfg.size, 7);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoorKeyConfig {
    /// Side length of the square grid in cells, including border walls.
    ///
    /// Must be at least `5`. The split column is sampled from `2..size - 2`
    /// each episode, which keeps both sub-rooms at least one interior column
    /// wide at every legal size.
    pub size: usize,
    /// Maximum number of steps before the episode is truncated.
    pub max_steps: usize,
    /// Seed for the environment's persistent RNG, drawn once at construction.
    ///
    /// The RNG **advances** across resets (ADR 0029), so successive episodes
    /// get an independent split column, door row, agent pose, and key cell. A
    /// fixed seed therefore reproduces a fixed *sequence* of episodes, not one
    /// repeated episode. For bit-for-bit replay of a single episode use
    /// [`DoorKeyEnv::reset_with_seed`].
    pub seed: u64,
}

impl DoorKeyConfig {
    /// Constructs a `DoorKeyConfig` with explicit field values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rlevo_environments::grids::door_key::DoorKeyConfig;
    ///
    /// let cfg = DoorKeyConfig::new(5, 100, 42);
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

impl Default for DoorKeyConfig {
    fn default() -> Self {
        let size = 5;
        Self {
            size,
            max_steps: 4 * size * size,
            seed: 0,
        }
    }
}

impl Validate for DoorKeyConfig {
    /// Rejects any `size` below `MIN_SIZE` (5) and a zero `max_steps`.
    ///
    /// The `size` floor lives **here**, not only in [`FromStr`]: `DoorKeyConfig`
    /// derives `Deserialize`, so a config loaded from a file is user-supplied
    /// runtime data that never passes through `from_str` (rules.md §4 — "if an
    /// invalid value can arrive via `Deserialize`, it must be an `Err`").
    /// Struct-update syntax on [`Default`] bypasses `from_str` just as freely.
    ///
    /// [`config::at_least`] subsumes a `nonzero` check for `size`, so the zero
    /// case reports [`TooSmall`](rlevo_core::config::ConstraintKind::TooSmall)
    /// rather than [`Zero`](rlevo_core::config::ConstraintKind::Zero);
    /// `max_steps` keeps `nonzero` because its only floor is 1.
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "DoorKeyConfig";
        config::at_least(C, "size", self.size, MIN_SIZE)?;
        config::nonzero(C, "max_steps", self.max_steps)?;
        Ok(())
    }
}

impl FromStr for DoorKeyConfig {
    type Err = String;

    /// Parses `"size=7,max_steps=196,seed=0"` (keys in any order) or the
    /// positional form `"7,196,0"`.
    ///
    /// # Errors
    ///
    /// Returns the offending key/value, or the [`Validate`] rejection — the same
    /// guard [`DoorKeyEnv::with_config`] applies, so this parser cannot admit a
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

/// One freshly sampled episode: the board plus the two scalars a caller cannot
/// recover from the board without re-deriving the generator.
///
/// Returned by [`DoorKeyEnv::build`] rather than assigning the fields inside it,
/// so `build` stays a pure function of `(config, rng)` and both call sites —
/// construction and [`reset`](Environment::reset) — commit the same three
/// values together.
#[derive(Debug)]
struct Layout {
    /// The board and the agent's start pose.
    state: GridState,
    /// Column of the vertical interior wall.
    split_col: i32,
    /// Row of the single door in that wall.
    door_row: i32,
}

/// Two-room gridworld requiring key pickup, door toggle, and goal navigation.
///
/// Implements [`Environment<3, 3, 1>`] — observation and action spaces each
/// have three components, reward is a scalar.
///
/// This environment tests long-horizon reasoning: the agent must learn a
/// strict ordering (pick up key → unlock door → navigate to goal) before any
/// reward is available.
///
/// The layout is drawn afresh on every [`reset`](Environment::reset) from a
/// persistent stream (ADR 0029/0062); read it back with [`Self::split_col`],
/// [`Self::door_row`], and [`Self::state`] rather than assuming fixed cells.
///
/// Construct via [`DoorKeyEnv::with_config`] for full control or via
/// [`ConstructableEnv::new`] for default settings (size 5, 100 steps, seed 0).
///
/// Reaching the goal (or lava) ends the episode; a [`step`](Environment::step)
/// taken afterwards is rejected with
/// [`EnvironmentError::StepAfterEpisodeEnd`] — see the [`EpisodeGuard`] field.
///
/// # Examples
///
/// ```rust
/// use rlevo_environments::grids::door_key::{DoorKeyConfig, DoorKeyEnv};
/// use rlevo_core::environment::{ConstructableEnv, Environment};
///
/// let mut env = DoorKeyEnv::with_config(
///     DoorKeyConfig::new(5, 100, 0),
///     false,
/// )
/// .expect("valid config");
/// env.reset().unwrap();
/// ```
#[derive(Debug)]
pub struct DoorKeyEnv {
    state: GridState,
    config: DoorKeyConfig,
    steps: usize,
    render: bool,
    split_col: i32,
    door_row: i32,
    rng: StdRng,
    /// Rejects a `step()` taken after the episode already ended.
    ///
    /// Without it the goal cell is a reward faucet, because nothing here is
    /// consumed on success: `Forward` leaves `Entity::Goal` in the grid and the
    /// agent standing on it, so the agent can walk off the goal and back on and
    /// `apply_action` reports `ReachedGoal` again, paying `success_reward` a
    /// second time on a second `Terminated` snapshot. Measured on
    /// `size = 5, max_steps = 100, reset_with_seed(3)`: the episode terminates
    /// at step 11 with reward `0.901`, then five further steps (`TurnRight`,
    /// `TurnRight`, `Forward`, `TurnRight` ×2, `Forward`) re-terminate at step
    /// 17 with another `0.847` — and the five snapshots in between come back
    /// `Running`, so a finished episode is silently resumed.
    ///
    /// The re-payment also *decays*, because `self.steps` never stops
    /// advancing: it is the numerator of `success_reward(steps, max_steps)`, so
    /// each replay is worth less than the last, and once `steps` passes
    /// `max_steps` the formula goes negative — the case `reward::success_reward`
    /// documents as "no env should call it past termination".
    ///
    /// The key/door mechanic keeps running too. The agent still carries the
    /// yellow key after the goal, and `Pickup` / `Drop` / `Toggle` stay live: in
    /// the same seed-3 rollout a post-terminal `Drop` at step 15 dumps the key
    /// at `(3, 1)` inside the goal room and empties the hand. The board the
    /// terminal snapshot described is then no longer the board the environment
    /// holds.
    guard: EpisodeGuard,
}

impl DoorKeyEnv {
    /// Emission-model visibility policy: does this environment's agent see
    /// through opaque cells?
    ///
    /// The rlevo spelling of canonical Minigrid's `see_through_walls`
    /// constructor argument. Read only by this environment's [`Sensor`] impl,
    /// and an inherent const rather than a config field because it is part of
    /// the task definition, not a knob a caller tunes.
    ///
    /// [`Visibility::Occluded`], because canonical `minigrid/envs/doorkey.py`
    /// **omits** `see_through_walls` entirely and so inherits
    /// `MiniGridEnv.__init__`'s `see_through_walls=False` default: upstream,
    /// occlusion is on unless an env opts out, and this one does not. The split
    /// wall and the shut door therefore hide the goal room until the door is
    /// opened — a locked door is opaque, an open one is not. See ADR 0063
    /// (`docs/adr/0063-grid-visibility-occlusion.md`) for the whole
    /// twelve-environment table.
    const VISIBILITY: Visibility = Visibility::Occluded;

    /// Constructs a `DoorKeyEnv` from an explicit configuration.
    ///
    /// Seeds the persistent RNG once from `config.seed` and immediately samples
    /// a first episode from it. Call [`Environment::reset`] before the first
    /// [`Environment::step`] to obtain the first observation; that reset draws a
    /// *fresh* layout, advancing the stream.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rlevo_environments::grids::door_key::{DoorKeyConfig, DoorKeyEnv};
    ///
    /// let env = DoorKeyEnv::with_config(
    ///     DoorKeyConfig::new(7, 196, 99),
    ///     true, // render ASCII grid to stdout
    /// )
    /// .expect("valid config");
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `config` fails [`Validate`]: a `size` below
    /// `MIN_SIZE` (5) — zero included — or a zero `max_steps`. This is the
    /// construction chokepoint (rules.md §4), so it also rejects a config that
    /// arrived by `Deserialize` or struct-update syntax without passing through
    /// [`FromStr`].
    ///
    /// A [`PlacementError`] from the initial layout draw converts into a
    /// `ConfigError` too. No legal `DoorKeyConfig` can reach that path — the
    /// left room always holds at least three free cells and the draws consume
    /// two — but the sampler is fallible and ADR 0062 §3(a) forbids swallowing
    /// it with `.expect()`.
    pub fn with_config(config: DoorKeyConfig, render: bool) -> Result<Self, ConfigError> {
        config.validate()?;
        let mut rng = StdRng::seed_from_u64(config.seed);
        let layout = Self::build(&config, &mut rng)?;
        Ok(Self {
            state: layout.state,
            config,
            steps: 0,
            render,
            split_col: layout.split_col,
            door_row: layout.door_row,
            rng,
            guard: EpisodeGuard::new(),
        })
    }

    /// Re-seed the persistent RNG to `seed`, then [`reset`](Environment::reset).
    ///
    /// Ordinary [`reset`](Environment::reset) advances the persistent stream so
    /// successive episodes get independent layouts (ADR 0029); use this when you
    /// need a *specific* episode to reproduce bit-for-bit (e.g. replaying a
    /// failure, or pinning a board a scripted rollout was written against).
    /// Run-level reproducibility is already guaranteed by the construction seed.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`reset`](Environment::reset), i.e. a
    /// [`PlacementError`] that exhausted the left room. No legal
    /// [`DoorKeyConfig`] can produce one; see [`with_config`](Self::with_config).
    pub fn reset_with_seed(&mut self, seed: u64) -> Result<GridSnapshot, EnvironmentError> {
        self.rng = StdRng::seed_from_u64(seed);
        self.reset()
    }

    /// Returns a reference to the active configuration.
    #[must_use]
    pub const fn config(&self) -> &DoorKeyConfig {
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

    /// Returns the column index of the vertical interior wall and locked door.
    ///
    /// **Sampled per episode** from `2..size - 2`, so this reads the current
    /// board rather than a function of `size`. At `size == 5` that range holds
    /// the single value `2` and the column cannot vary; from `size == 6` up it
    /// moves on every [`reset`](Environment::reset).
    #[must_use]
    pub const fn split_col(&self) -> i32 {
        self.split_col
    }

    /// Returns the row of the single locked door in the split column.
    ///
    /// **Sampled per episode** from `1..size - 2`. Together with
    /// [`split_col`](Self::split_col) this locates the only cell of the split
    /// wall the agent can ever pass through, which is what a scripted rollout or
    /// an oracle needs in order to read the board back instead of assuming it.
    #[must_use]
    pub const fn door_row(&self) -> i32 {
        self.door_row
    }

    /// Samples one episode's layout, reproducing upstream `_gen_grid`'s draws in
    /// upstream's order (see the module docs).
    ///
    /// Draws from `rng` and lets it advance (ADR 0029) — never re-seeds.
    ///
    /// # Errors
    ///
    /// Returns [`PlacementError::Exhausted`] if the left room has no free cell
    /// for the agent, or none left for the key once the agent stands there.
    /// That cannot happen for a config this crate accepts — the left room holds
    /// `(split_col - 1) * (size - 2) >= 1 * 3` free cells at `MIN_SIZE`, and the
    /// two draws consume at most two of them — but the sampler is fallible and
    /// ADR 0062 §3(a) forbids papering over it with `.expect()`, so it is
    /// propagated.
    fn build<R: Rng + ?Sized>(
        config: &DoorKeyConfig,
        rng: &mut R,
    ) -> Result<Layout, PlacementError> {
        #[allow(clippy::cast_possible_wrap)]
        let size = config.size as i32;

        let mut grid = Grid::new(config.size, config.size);
        grid.draw_walls();
        grid.set(size - 2, size - 2, Entity::Goal);

        // 1. The vertical splitting wall. `2..size - 2` keeps at least one
        //    interior column on each side; `MIN_SIZE` keeps it non-empty.
        let split_col = rng.random_range(2..size - 2);
        for y in 1..size - 1 {
            grid.set(split_col, y, Entity::Wall);
        }

        // The whole left partition, walls included: `place_agent` and
        // `place_obj` filter to the free cells themselves, so the region is
        // upstream's `top=(0, 0), size=(split_col, height)` verbatim.
        let left_room = Rect::new(0, 0, split_col - 1, size - 1);

        // 2. The agent's cell *and* facing, before the key — upstream's order,
        //    and the reason step 4 can reject the agent's cell.
        let agent = place_agent(&grid, rng, left_room, &mut no_reject)?;

        // 3. The single door in the splitting wall.
        let door_row = rng.random_range(1..size - 2);
        grid.set(
            split_col,
            door_row,
            Entity::Door(DOOR_COLOR, DoorState::Locked),
        );

        // 4. The key, anywhere free in the left room except under the agent.
        //    `is_free` cannot see the agent, so the veto is the caller's job.
        place_obj(
            &mut grid,
            rng,
            Entity::Key(DOOR_COLOR),
            left_room,
            &mut |_grid, x, y| (x, y) == (agent.x, agent.y),
        )?;

        Ok(Layout {
            state: GridState::new(grid, agent),
            split_col,
            door_row,
        })
    }

    fn emit(&self, observation: GridObservation, reward: f32, done: bool) -> GridSnapshot {
        if self.render {
            println!("{}", self.ascii());
        }
        build_snapshot(observation, reward, done)
    }
}

impl crate::render::AsciiRenderable for DoorKeyEnv {
    fn render_ascii(&self) -> String {
        render_ascii(&self.state.grid, &self.state.agent)
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        super::core::render::render_styled(&self.state.grid, &self.state.agent)
    }
}

impl Display for DoorKeyEnv {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DoorKeyEnv(size={}, step={}/{})",
            self.config.size, self.steps, self.config.max_steps
        )
    }
}

impl ConstructableEnv for DoorKeyEnv {
    fn new(render: bool) -> Self {
        Self::with_config(DoorKeyConfig::default(), render).expect("default config must validate")
    }
}

impl Sensor<3, 1, 3> for DoorKeyEnv {
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

impl Environment<3, 3, 1> for DoorKeyEnv {
    type StateType = GridState;
    type ObservationType = super::core::GridObservation;
    type ActionType = GridAction;
    type RewardType = ScalarReward;
    type SnapshotType = GridSnapshot;

    /// Draws a fresh layout and re-opens the episode.
    ///
    /// # Errors
    ///
    /// Propagates [`PlacementError`] from [`build`](Self::build) if the left
    /// room has no free cell for the agent or the key. No legal
    /// [`DoorKeyConfig`] reaches that path.
    ///
    /// A failed reset is a **total no-op**: the board, the step counter, and the
    /// guard all keep the previous episode's values, so a finished episode stays
    /// finished and a later `step()` is still rejected. Clearing the guard before
    /// the draw would instead re-open a finished episode on a board that never
    /// returned to an initial state. This follows ADR 0044 §6 and the
    /// [`TimeLimit::reset`](crate::wrappers::TimeLimit) precedent, which clears
    /// its guard only after the delegated reset succeeds.
    ///
    /// That `Err` path is unreachable for any config this crate accepts (see
    /// [`build`](Self::build)), so it carries no test — only this ordering.
    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        // Draw first: only a *successful* build actually starts a new episode.
        let layout = Self::build(&self.config, &mut self.rng)?;
        self.guard.reset();
        self.state = layout.state;
        self.split_col = layout.split_col;
        self.door_row = layout.door_row;
        self.steps = 0;
        let observation = self.observe_reset(&self.state);
        Ok(self.emit(observation, 0.0, false))
    }

    /// Applies `action` to the board and returns the resulting snapshot.
    ///
    /// # Errors
    ///
    /// Returns [`EnvironmentError::StepAfterEpisodeEnd`] if the episode has
    /// already ended; call [`reset`](Environment::reset) first.
    fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> {
        // Guard first: before `steps` advances (it is the numerator of
        // `success_reward`, so an illegal call would silently discount every
        // later payout) and before `apply_action` touches the grid or the
        // agent's hand. `step()` draws no randomness today, but `reset()` shares
        // the persistent stream, so checking any later would let rejected calls
        // reorder future episodes (ADR 0029).
        self.guard.check()?;

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

        // Single exit: the guard is fed the emitted snapshot's own status, so
        // the two cannot disagree about whether the episode is over.
        let snapshot = self.emit(observation, reward, done);
        self.guard.record(snapshot.status());
        Ok(snapshot)
    }
}

impl rlevo_core::render::payload::GridPayloadSource for DoorKeyEnv {
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
    use crate::direction::Direction;
    use crate::episode::assert_rejects_post_terminal_step;
    use crate::grids::core::agent::AgentState;
    use crate::grids::core::{UNSEEN_TYPE, VIEW_SIZE};
    use rlevo_core::config::ConstraintKind;
    use rlevo_core::environment::EpisodeStatus;
    use std::collections::{HashSet, VecDeque};

    /// Every seed loop runs at the `MIN_SIZE` floor and at one larger board.
    ///
    /// `MIN_SIZE` is where the sampling ranges degenerate — the split column has
    /// exactly one legal value there and the left room is a single column — so
    /// a bug that only shows up on a wide board and one that only shows up on a
    /// pinched board are different bugs.
    const SIZES: [usize; 2] = [MIN_SIZE, 9];

    /// Episodes drawn per seed loop. Large enough that a quantity with two legal
    /// values is overwhelmingly unlikely to come back constant by chance.
    const SEEDS: u64 = 64;

    fn env_of(size: usize) -> DoorKeyEnv {
        DoorKeyEnv::with_config(DoorKeyConfig::new(size, 4 * size * size, 0), false)
            .expect("valid config")
    }

    fn env_5x5() -> DoorKeyEnv {
        DoorKeyEnv::with_config(DoorKeyConfig::new(5, 100, 0), false).expect("valid config")
    }

    /// Every cell of the board, row-major, plus the agent's pose.
    ///
    /// The whole sampled layout in one comparable value: two episodes are the
    /// same episode exactly when their fingerprints match.
    fn fingerprint(env: &DoorKeyEnv) -> (Vec<Entity>, i32, i32, Direction) {
        let grid = &env.state().grid;
        let mut cells = Vec::with_capacity(grid.width() * grid.height());
        for (x, y) in coords(grid) {
            cells.push(grid.get(x, y));
        }
        let agent = &env.state().agent;
        (cells, agent.x, agent.y, agent.direction)
    }

    /// Row-major `(x, y)` over every cell of `grid`.
    fn coords(grid: &Grid) -> Vec<(i32, i32)> {
        let width = i32::try_from(grid.width()).expect("test grids fit in i32");
        let height = i32::try_from(grid.height()).expect("test grids fit in i32");
        (0..height)
            .flat_map(|y| (0..width).map(move |x| (x, y)))
            .collect()
    }

    /// Every cell whose entity satisfies `pred`.
    fn cells_matching(env: &DoorKeyEnv, pred: impl Fn(Entity) -> bool) -> Vec<(i32, i32)> {
        let grid = &env.state().grid;
        coords(grid)
            .into_iter()
            .filter(|&(x, y)| pred(grid.get(x, y)))
            .collect()
    }

    /// The one key cell of the current board, asserting there is exactly one.
    fn key_pos(env: &DoorKeyEnv) -> (i32, i32) {
        let keys = cells_matching(env, |e| matches!(e, Entity::Key(_)));
        assert_eq!(keys.len(), 1, "expected exactly one key, found {keys:?}");
        keys[0]
    }

    /// Cells reachable from `start` by walking over passable cells only.
    ///
    /// A `Locked` door is impassable, so this flood fill never crosses the split
    /// wall — which is what makes it a statement about what the agent can do
    /// *before* it has the key.
    fn reachable_from(grid: &Grid, start: (i32, i32)) -> HashSet<(i32, i32)> {
        let mut seen = HashSet::from([start]);
        let mut queue = VecDeque::from([start]);
        while let Some((x, y)) = queue.pop_front() {
            for (nx, ny) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] {
                if grid.in_bounds(nx, ny) && grid.get(nx, ny).is_passable() && seen.insert((nx, ny))
                {
                    queue.push_back((nx, ny));
                }
            }
        }
        seen
    }

    #[test]
    fn default_config_validates() {
        assert!(DoorKeyConfig::default().validate().is_ok());
    }

    #[test]
    fn rejects_zero_size() {
        let bad = DoorKeyConfig {
            size: 0,
            ..Default::default()
        };
        assert!(DoorKeyEnv::with_config(bad, false).is_err());
    }

    #[test]
    fn with_config_rejects_size_below_min() {
        // The floor must be enforced at the construction chokepoint, not only in
        // `FromStr`. Before this guard, size 4 built a board whose goal at (2,2)
        // overwrote the locked door, making the episode trivially winnable.
        let bad = DoorKeyConfig {
            size: MIN_SIZE - 1,
            ..Default::default()
        };
        let err = DoorKeyEnv::with_config(bad, false)
            .expect_err("a sub-MIN_SIZE grid must be refused at construction");
        assert_eq!(err.config, "DoorKeyConfig", "the config must be named");
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
    fn default_config_is_5x5() {
        let cfg = DoorKeyConfig::default();
        assert_eq!(cfg.size, 5);
        assert_eq!(cfg.max_steps, 4 * 5 * 5);
    }

    #[test]
    fn fromstr_rejects_small_size() {
        assert!("4".parse::<DoorKeyConfig>().is_err());
    }

    #[test]
    fn fromstr_parses_keyvalue() {
        let cfg: DoorKeyConfig = "size=7,max_steps=120,seed=1".parse().unwrap();
        assert_eq!(cfg.size, 7);
        assert_eq!(cfg.max_steps, 120);
        assert_eq!(cfg.seed, 1);
    }

    // --- structural invariants: true of every board, every seed, every size ---

    #[test]
    fn every_board_has_one_goal_one_locked_door_and_one_key() {
        for size in SIZES {
            let mut env = env_of(size);
            let far = i32::try_from(size).expect("test sizes fit in i32") - 2;
            for seed in 0..SEEDS {
                env.reset_with_seed(seed).expect("reset");
                let goals = cells_matching(&env, |e| e == Entity::Goal);
                assert_eq!(
                    goals,
                    vec![(far, far)],
                    "size {size} seed {seed}: the goal is the one fixed cell"
                );
                let doors = cells_matching(&env, |e| matches!(e, Entity::Door(..)));
                assert_eq!(doors.len(), 1, "size {size} seed {seed}: doors {doors:?}");
                assert_eq!(
                    env.state().grid.get(doors[0].0, doors[0].1),
                    Entity::Door(DOOR_COLOR, DoorState::Locked),
                    "size {size} seed {seed}: the door starts locked and yellow"
                );
                let keys = cells_matching(&env, |e| matches!(e, Entity::Key(_)));
                assert_eq!(keys.len(), 1, "size {size} seed {seed}: keys {keys:?}");
                assert_eq!(
                    env.state().grid.get(keys[0].0, keys[0].1),
                    Entity::Key(DOOR_COLOR),
                    "size {size} seed {seed}: the key must match the door"
                );
            }
        }
    }

    #[test]
    fn the_split_column_is_solid_wall_except_the_door_cell() {
        for size in SIZES {
            let mut env = env_of(size);
            let last = i32::try_from(size).expect("test sizes fit in i32") - 1;
            for seed in 0..SEEDS {
                env.reset_with_seed(seed).expect("reset");
                let (col, row) = (env.split_col(), env.door_row());
                assert!(
                    (2..last - 1).contains(&col),
                    "size {size} seed {seed}: split column {col} left a room empty"
                );
                assert!(
                    (1..last - 1).contains(&row),
                    "size {size} seed {seed}: door row {row} is out of range"
                );
                for y in 0..=last {
                    let cell = env.state().grid.get(col, y);
                    let expected = if y == row {
                        Entity::Door(DOOR_COLOR, DoorState::Locked)
                    } else {
                        Entity::Wall
                    };
                    assert_eq!(
                        cell, expected,
                        "size {size} seed {seed}: split column cell ({col}, {y})"
                    );
                }
            }
        }
    }

    #[test]
    fn perimeter_is_always_wall() {
        for size in SIZES {
            let mut env = env_of(size);
            let last = i32::try_from(size).expect("test sizes fit in i32") - 1;
            for seed in 0..SEEDS {
                env.reset_with_seed(seed).expect("reset");
                let grid = &env.state().grid;
                for i in 0..=last {
                    for (x, y) in [(i, 0), (i, last), (0, i), (last, i)] {
                        assert_eq!(
                            grid.get(x, y),
                            Entity::Wall,
                            "size {size} seed {seed}: perimeter cell ({x}, {y})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn agent_and_key_start_strictly_inside_the_left_room() {
        for size in SIZES {
            let mut env = env_of(size);
            let last = i32::try_from(size).expect("test sizes fit in i32") - 1;
            for seed in 0..SEEDS {
                env.reset_with_seed(seed).expect("reset");
                let col = env.split_col();
                let key = key_pos(&env);
                let agent = env.state().agent;
                for (label, (x, y)) in [("agent", (agent.x, agent.y)), ("key", key)] {
                    assert!(
                        (1..col).contains(&x) && (1..last).contains(&y),
                        "size {size} seed {seed}: {label} at ({x}, {y}) escaped the left room"
                    );
                }
                assert_ne!(
                    (agent.x, agent.y),
                    key,
                    "size {size} seed {seed}: the key must never land under the agent"
                );
                assert_eq!(
                    env.state().grid.get(agent.x, agent.y),
                    Entity::Empty,
                    "size {size} seed {seed}: the agent starts on an empty cell, so it \
                     can be neither on the goal nor on the door"
                );
                assert_eq!(
                    env.state().agent.carrying,
                    None,
                    "size {size} seed {seed}: a fresh agent carries nothing"
                );
            }
        }
    }

    // --- solvability: the draw order is what makes the task well-posed ---

    #[test]
    fn the_key_is_reachable_before_the_door_is_opened() {
        for size in SIZES {
            let mut env = env_of(size);
            for seed in 0..SEEDS {
                env.reset_with_seed(seed).expect("reset");
                let grid = &env.state().grid;
                let agent = env.state().agent;
                let (kx, ky) = key_pos(&env);
                let reachable = reachable_from(grid, (agent.x, agent.y));
                // The key cell itself is impassable; the agent picks it up from
                // an orthogonally adjacent cell while facing it.
                let approach = [(kx + 1, ky), (kx - 1, ky), (kx, ky + 1), (kx, ky - 1)]
                    .into_iter()
                    .any(|cell| reachable.contains(&cell));
                assert!(
                    approach,
                    "size {size} seed {seed}: no cell adjacent to the key at ({kx}, {ky}) \
                     is reachable from the agent without opening the locked door"
                );
            }
        }
    }

    #[test]
    fn the_goal_is_unreachable_until_the_door_opens() {
        for size in SIZES {
            let mut env = env_of(size);
            let far = i32::try_from(size).expect("test sizes fit in i32") - 2;
            for seed in 0..SEEDS {
                env.reset_with_seed(seed).expect("reset");
                let agent = env.state().agent;
                let reachable = reachable_from(&env.state().grid, (agent.x, agent.y));
                assert!(
                    !reachable.contains(&(far, far)),
                    "size {size} seed {seed}: the goal is reachable without the door, so \
                     the split wall has a hole in it"
                );
            }
        }
    }

    // --- non-degeneracy: each sampled quantity moves, checked separately ---

    #[test]
    fn split_column_is_pinned_at_min_size_and_varies_above_it() {
        // At `MIN_SIZE` the range `2..size - 2` is the single value `2`, so
        // asserting variability there would assert something the algorithm
        // forbids. The honest invariant at the floor is that it is pinned.
        let mut floor = env_of(MIN_SIZE);
        for seed in 0..SEEDS {
            floor.reset_with_seed(seed).expect("reset");
            assert_eq!(
                floor.split_col(),
                2,
                "at size {MIN_SIZE} the split column has exactly one legal value"
            );
        }

        let mut wide = env_of(9);
        let mut seen = HashSet::new();
        for seed in 0..SEEDS {
            wide.reset_with_seed(seed).expect("reset");
            seen.insert(wide.split_col());
        }
        assert!(
            seen.len() > 1,
            "the split column must vary at size 9, saw {seen:?}"
        );
    }

    #[test]
    fn door_row_varies_across_episodes() {
        for size in SIZES {
            let mut env = env_of(size);
            let mut seen = HashSet::new();
            for seed in 0..SEEDS {
                env.reset_with_seed(seed).expect("reset");
                seen.insert(env.door_row());
            }
            assert!(
                seen.len() > 1,
                "size {size}: the door row must vary, saw {seen:?}"
            );
        }
    }

    #[test]
    fn agent_position_varies_across_episodes() {
        for size in SIZES {
            let mut env = env_of(size);
            let mut seen = HashSet::new();
            for seed in 0..SEEDS {
                env.reset_with_seed(seed).expect("reset");
                seen.insert((env.state().agent.x, env.state().agent.y));
            }
            assert!(
                seen.len() > 1,
                "size {size}: the agent's start cell must vary, saw {seen:?}"
            );
        }
    }

    #[test]
    fn agent_direction_varies_across_episodes() {
        // Its own assertion, not folded into the position check: a `place_agent`
        // that drew a cell but pinned the facing would still pass that one.
        for size in SIZES {
            let mut env = env_of(size);
            let mut seen = HashSet::new();
            for seed in 0..SEEDS {
                env.reset_with_seed(seed).expect("reset");
                seen.insert(env.state().agent.direction);
            }
            assert_eq!(
                seen.len(),
                4,
                "size {size}: every facing must be reachable, saw {seen:?}"
            );
        }
    }

    #[test]
    fn key_position_varies_across_episodes() {
        for size in SIZES {
            let mut env = env_of(size);
            let mut seen = HashSet::new();
            for seed in 0..SEEDS {
                env.reset_with_seed(seed).expect("reset");
                seen.insert(key_pos(&env));
            }
            assert!(
                seen.len() > 1,
                "size {size}: the key's cell must vary, saw {seen:?}"
            );
        }
    }

    // --- the persistent-stream contract (ADR 0029) ---

    #[test]
    fn reset_is_deterministic() {
        let cfg = DoorKeyConfig::new(5, 100, 17);
        let mut a = DoorKeyEnv::with_config(cfg, false).expect("valid config");
        let mut b = DoorKeyEnv::with_config(cfg, false).expect("valid config");
        let sa = a.reset().unwrap();
        let sb = b.reset().unwrap();
        assert_eq!(sa.observation(), sb.observation());
    }

    #[test]
    fn consecutive_resets_draw_different_layouts() {
        for size in SIZES {
            let mut env = env_of(size);
            let mut seen = HashSet::new();
            for _ in 0..SEEDS {
                env.reset().expect("reset");
                seen.insert(fingerprint(&env));
            }
            assert!(
                seen.len() > 1,
                "size {size}: the stream must advance across resets — a re-seeding \
                 reset() would replay one layout forever"
            );
        }
    }

    #[test]
    fn reset_with_seed_replays_one_episode_bit_for_bit() {
        for size in SIZES {
            let mut env = env_of(size);
            for seed in 0..8 {
                env.reset_with_seed(seed).expect("reset");
                let first = fingerprint(&env);
                // Advance the stream between the two replays so the second one
                // cannot succeed by accident.
                env.reset().expect("reset");
                env.reset_with_seed(seed).expect("reset");
                assert_eq!(
                    first,
                    fingerprint(&env),
                    "size {size} seed {seed}: reset_with_seed must be exact replay"
                );
            }
        }
    }

    #[test]
    fn different_seeds_give_different_layouts() {
        for size in SIZES {
            let mut env = env_of(size);
            let mut seen = HashSet::new();
            for seed in 0..SEEDS {
                env.reset_with_seed(seed).expect("reset");
                seen.insert(fingerprint(&env));
            }
            assert!(
                seen.len() > 1,
                "size {size}: distinct seeds must reach distinct layouts"
            );
        }
    }

    // --- dynamics on a board that is read back, not assumed ---

    #[test]
    fn toggle_locked_door_without_key_is_noop() {
        for size in SIZES {
            let mut env = env_of(size);
            for seed in 0..8 {
                env.reset_with_seed(seed).expect("reset");
                let (col, row) = (env.split_col(), env.door_row());
                // Stand west of the door facing it, hand empty. Overwriting the
                // pose is legitimate here: the point under test is `Toggle`, and
                // routing there is the planner-driven oracle's job
                // (`tests/grids_solvable.rs`).
                env.state.agent = AgentState::new(col - 1, row, Direction::East);
                env.step(GridAction::Toggle).expect("step");
                assert!(
                    matches!(
                        env.state().grid.get(col, row),
                        Entity::Door(_, DoorState::Locked)
                    ),
                    "size {size} seed {seed}: a locked door opened without the key"
                );
            }
        }
    }

    #[test]
    fn reaching_the_goal_pays_a_positive_reward() {
        for size in SIZES {
            let mut env = env_of(size);
            let far = i32::try_from(size).expect("test sizes fit in i32") - 2;
            env.reset_with_seed(1).expect("reset");
            // The cell directly north of the goal is always in the right room:
            // the split column is at most `size - 3`, so column `size - 2` is
            // east of it for every legal size.
            env.state.agent = AgentState::new(far, far - 1, Direction::South);
            let snap = env.step(GridAction::Forward).expect("step");
            assert!(
                snap.is_done(),
                "size {size}: stepping onto the goal ends it"
            );
            let reward: f32 = (*snap.reward()).into();
            assert!(reward > 0.9, "size {size}: reward was {reward}");
        }
    }

    #[test]
    fn timeout_terminates_without_reward() {
        let cfg = DoorKeyConfig::new(5, 2, 0);
        let mut env = DoorKeyEnv::with_config(cfg, false).expect("valid config");
        env.reset().unwrap();
        env.step(GridAction::TurnLeft).unwrap();
        let snap = env.step(GridAction::TurnLeft).unwrap();
        assert!(snap.is_done());
        let reward: f32 = (*snap.reward()).into();
        assert_eq!(reward, 0.0);
    }

    #[test]
    fn reset_clears_episode_state() {
        let mut env = env_5x5();
        env.reset().unwrap();
        for _ in 0..3 {
            env.step(GridAction::TurnLeft).unwrap();
        }
        assert_eq!(env.steps(), 3);
        env.reset().unwrap();
        assert_eq!(env.steps(), 0);
        // The pose is re-drawn rather than restored, so the honest assertion is
        // that it is a legal start pose — see `agent_and_key_start_strictly_
        // inside_the_left_room` for the full invariant.
        let agent = env.state().agent;
        assert_eq!(agent.carrying, None);
        assert!((1..env.split_col()).contains(&agent.x));
    }

    /// The ASCII board in the module docs is a snapshot of a named seed; if the
    /// generator moves, this fails and the docs get corrected with it.
    #[test]
    fn module_doc_snapshot_is_current() {
        let mut env = env_5x5();
        env.reset_with_seed(3).expect("reset");
        assert_eq!(
            env.ascii(),
            "# # # # # \n# ^ * . # \n# k # . # \n# . # G # \n# # # # # \n",
            "the ASCII snapshot in this module's docs is stale"
        );
    }

    /// Occlusion is not decoration here: a cell that encoded a real entity under
    /// the pre-#281 (see-through) emission model must now be able to encode as
    /// *unseen*.
    ///
    /// The key is the interesting case — it is the object the whole task is
    /// about, and the split wall is exactly what is supposed to withhold it.
    /// Seed 3 puts the key at (1, 2) behind the wall column at x = 2, so an
    /// agent in the east half looking back across that wall sees nothing there.
    #[test]
    fn test_door_key_occlusion_hides_the_key_behind_the_split_wall() {
        let mut env =
            DoorKeyEnv::with_config(DoorKeyConfig::new(8, 100, 3), false).expect("valid config");
        env.reset().expect("reset");
        assert_eq!(
            env.state().grid.get(1, 2),
            Entity::Key(Color::Yellow),
            "seed 3 must place the key at (1, 2) for this pose to be the intended one"
        );

        // Agent at (3, 1) in the east half, facing South down the split wall.
        let state = GridState::new(
            env.state().grid.clone(),
            AgentState::new(3, 1, Direction::South),
        );
        let occluded = observe_grid(&state, DoorKeyEnv::VISIBILITY);
        let see_through = observe_grid(&state, Visibility::SeeThrough);

        // (row 5, col 5) is the key cell: one step forward, two to the right.
        let (row, col) = (5, 5);
        assert_eq!(
            see_through.view[row][col][0],
            Entity::Key(Color::Yellow).type_u8(),
            "the pre-#281 emission model reported the key through the split wall"
        );
        assert_eq!(
            occluded.view[row][col],
            [UNSEEN_TYPE, 0, 0],
            "the key must now encode as unseen — this env inherits canonical's \
             see_through_walls=False default"
        );
        assert_eq!(
            DoorKeyEnv::VISIBILITY,
            Visibility::Occluded,
            "doorkey.py omits see_through_walls, so MiniGridEnv's False default applies"
        );
    }

    // ── post-terminal step guard (ADR 0044, issue #291) ──────────────────────

    /// A 5×5 board whose seed-3 layout the module docs pin, so the scripted
    /// solution below is written against a board that cannot silently move —
    /// `module_doc_snapshot_is_current` fails first if the generator changes.
    fn env_seed_3() -> DoorKeyEnv {
        let mut env = env_5x5();
        env.reset_with_seed(3).expect("reset");
        env
    }

    /// End-to-end reproducer for #1027: `AgentState::carrying` must reach the
    /// emitted observation.
    ///
    /// Driven through the **real dynamics** — no hand-built `GridState`, no
    /// direct `stamp_carried` call. The board is the seed-3 layout the module
    /// docs pin (agent `(1, 1)` facing North, key `(1, 2)`, locked door
    /// `(2, 1)`), so `module_doc_snapshot_is_current` fails first if the
    /// generator moves under this script.
    ///
    /// Before the fix, `grid::stamp_carried` did not exist and the agent's own
    /// view cell reported the world tile it stood on; a `DoorKeyEnv` policy had
    /// no channel that ever showed whether it held the key, or which colour.
    #[test]
    fn test_door_key_carried_key_reaches_the_observation() {
        let mut env = env_seed_3();

        // The colour is read off the board, not assumed: the unlock gate
        // compares against whatever colour the layout drew.
        let key_color = match env.state().grid.get(1, 2) {
            Entity::Key(color) => color,
            other => panic!("seed 3 must place a key at (1, 2), found {other:?}"),
        };

        // North → East → South, which leaves the agent facing the key at (1, 2).
        env.step(GridAction::TurnRight).expect("turn is legal");
        let before = env.step(GridAction::TurnRight).expect("turn is legal");
        assert_eq!(
            env.state().agent.carrying,
            None,
            "the agent must still be empty-handed on the step before the Pickup"
        );

        let after = env.step(GridAction::Pickup).expect("the key is in front");
        assert_eq!(
            env.state().agent.carrying,
            Some(Entity::Key(key_color)),
            "the Pickup must have put the key in the agent's hand"
        );

        // The world tile the agent stands on is `Empty`, so the key byte in the
        // agent's own view cell can only have come from the carry stamp — with
        // an unstamped view that cell reports this tile and the next assertion
        // fails.
        let agent = env.state().agent;
        assert_eq!(
            env.state().grid.get(agent.x, agent.y),
            Entity::Empty,
            "the agent stands on an empty tile, which is what makes this a \
             carry-stamp assertion rather than a world-tile one"
        );

        // The agent's own cell: bottom row, centre column.
        let (arow, acol) = (VIEW_SIZE - 1, VIEW_SIZE / 2);
        let cell = after.observation().view[arow][acol];
        assert_eq!(
            cell,
            [Entity::Key(key_color).type_u8(), key_color.to_u8(), 0],
            "the carried key must be stamped into the agent's own view cell"
        );

        // Byte 1 called out on its own: `dynamics.rs:126` gates unlocking on
        // `agent.carrying == Some(Entity::Key(color))` for the door's *colour*,
        // so this is the exact byte a policy has to read to know whether its
        // key opens the door in front of it. A type byte alone is not enough.
        assert_eq!(
            cell[1],
            key_color.to_u8(),
            "the colour byte is what the unlock gate compares against the door"
        );

        // Byte 2 is a literal 0: carryables inherit `WorldObj.encode`, and
        // `Door` is the only entity that contributes a state byte.
        assert_eq!(
            cell[2], 0,
            "a key has no state byte — only Door encodes one"
        );

        // Empty-handed, the same cell encodes `Entity::Empty`, never the unseen
        // triple: the agent always observes its own cell.
        assert_eq!(
            before.observation().view[arow][acol],
            [Entity::Empty.type_u8(), 0, 0],
            "before the Pickup the agent's own cell must report an empty hand"
        );

        // The reproducer proper: exactly which cells the Pickup moved.
        let changed: Vec<(usize, usize)> = (0..VIEW_SIZE)
            .flat_map(|row| (0..VIEW_SIZE).map(move |col| (row, col)))
            .filter(|&(row, col)| {
                before.observation().view[row][col] != after.observation().view[row][col]
            })
            .collect();
        // Two cells, and only two. `(6, 3)` is the agent's own cell gaining the
        // key. `(5, 3)` is the world cell one step ahead — the key's tile —
        // which the Pickup empties, and which is inside the view window here
        // because the agent is standing right next to the key. Every other cell
        // of the window is untouched by a Pickup.
        assert_eq!(
            changed,
            vec![(VIEW_SIZE - 2, VIEW_SIZE / 2), (arow, acol)],
            "a Pickup must change the agent's own cell and the emptied key tile, \
             and nothing else"
        );
        assert_eq!(
            after.observation().view[VIEW_SIZE - 2][VIEW_SIZE / 2],
            [Entity::Empty.type_u8(), 0, 0],
            "the key's world tile is empty once the key is in hand"
        );
    }

    /// The interaction the other carry tests miss: still *holding* the key on
    /// the exact frame the agent reaches the goal.
    ///
    /// `grid::stamp_carried` made two changes that every other test exercises
    /// separately — a carried item becomes visible at the agent's own cell, and
    /// the world tile under the agent stops leaking. This frame is where the
    /// two collide: the agent stands on `Entity::Goal` *and* holds a key, so
    /// exactly one of three values can appear in that cell. Nothing in
    /// `DoorKeyEnv` forces a `Drop` before the goal, and `dynamics.rs`'s
    /// `step_forward` moves the agent onto the goal cell *before* classifying
    /// `Entity::Goal => StepOutcome::ReachedGoal`, after which `step()` calls
    /// `observe()` unconditionally — so the frame is reachable, not synthetic.
    ///
    /// Driven through the real dynamics via [`drive_to_goal`], whose plan never
    /// drops the key.
    #[test]
    fn test_door_key_carrying_at_the_goal_shows_the_hand_not_the_tile() {
        let mut env = env_seed_3();

        // Colour read off the board, not assumed: the layout draws it.
        let key_color = match env.state().grid.get(1, 2) {
            Entity::Key(color) => color,
            other => panic!("seed 3 must place a key at (1, 2), found {other:?}"),
        };

        let terminal = drive_to_goal(&mut env);

        // The episode really ended at the goal, so this cannot pass on some
        // mid-episode frame that never stood on a terminal tile.
        assert_eq!(
            terminal.status(),
            EpisodeStatus::Terminated,
            "the frame under test must be the terminal one"
        );

        // Precondition 1: the hand is still full. The plan contains no `Drop`,
        // and reaching the goal consumes nothing.
        assert_eq!(
            env.state().agent.carrying,
            Some(Entity::Key(key_color)),
            "the agent must still be carrying the key on the terminal frame"
        );

        // Precondition 2: the world tile under the agent genuinely IS the goal.
        // Without this the tile-leak assertion below would be vacuous — it
        // would pass on any tile that simply isn't a goal.
        let agent = env.state().agent;
        assert_eq!(
            env.state().grid.get(agent.x, agent.y),
            Entity::Goal,
            "the agent must be standing on the goal entity itself"
        );

        // The agent's own cell: bottom row, centre column.
        let (arow, acol) = (VIEW_SIZE - 1, VIEW_SIZE / 2);
        let cell = terminal.observation().view[arow][acol];

        assert_eq!(
            cell,
            [Entity::Key(key_color).type_u8(), key_color.to_u8(), 0],
            "the carried key must win the agent's own cell on the terminal frame"
        );

        // Negative 1: the terminal tile must not leak. `Goal` is directly under
        // the agent on this frame, and canonical Minigrid erases it — the
        // terminal tile is never a channel the policy can read.
        assert_ne!(
            cell,
            [Entity::Goal.type_u8(), 0, 0],
            "the goal tile under the agent must not leak into the observation"
        );

        // Negative 2: the hand must not read as empty. `stamp_carried` writes
        // `Some(Entity::Empty)` only for `carrying == None`; a stamp that
        // ignored the hand — or ran before the pickup state was consulted —
        // would land here while still satisfying negative 1.
        assert_ne!(
            cell,
            [Entity::Empty.type_u8(), 0, 0],
            "the agent holds a key, so its own cell must not encode an empty hand"
        );
    }

    /// Solves the seed-3 board through **real `step()` calls** and returns the
    /// terminal snapshot.
    ///
    /// Termination is by *reaching the goal*, not by exhausting `max_steps`:
    /// `grids/core/mod.rs`'s `build_snapshot` currently maps a step-limit cutoff
    /// to `Terminated` rather than `Truncated` (#1028), so a step-limit ending
    /// would pin the wrong status through this guard.
    ///
    /// The board (see the module docs) is agent `(1, 1)` facing North, key
    /// `(1, 2)`, locked door `(2, 1)`, goal `(3, 3)`.
    fn drive_to_goal(env: &mut DoorKeyEnv) -> GridSnapshot {
        let plan = [
            GridAction::TurnRight, // North → East
            GridAction::TurnRight, // East → South, now facing the key
            GridAction::Pickup,    // take the yellow key
            GridAction::TurnLeft,  // South → East, now facing the locked door
            GridAction::Toggle,    // Locked → Closed
            GridAction::Toggle,    // Closed → Open
            GridAction::Forward,   // (1,1) → (2,1), through the door
            GridAction::Forward,   // (2,1) → (3,1)
            GridAction::TurnRight, // East → South
            GridAction::Forward,   // (3,1) → (3,2)
            GridAction::Forward,   // (3,2) → (3,3), the goal
        ];
        let mut snapshot = None;
        for action in plan {
            let snap = env
                .step(action)
                .expect("the scripted solution must be legal");
            snapshot = Some(snap);
        }
        let terminal = snapshot.expect("the plan is non-empty");
        assert!(
            terminal.is_done(),
            "the scripted plan must reach the goal; the board or the dynamics moved"
        );
        assert_eq!(
            (env.state().agent.x, env.state().agent.y),
            (3, 3),
            "the agent must finish on the goal cell"
        );
        terminal
    }

    #[test]
    /// Verifies `DoorKeyEnv` satisfies the shared post-terminal conformance
    /// check: once the agent reaches the goal, a further *legal* action fails
    /// with `StepAfterEpisodeEnd { status: Terminated }`. `Drop` is chosen as the
    /// replay because the agent is still holding the key — the rejection must be
    /// on call-sequence grounds, never on the action's own validity.
    fn test_door_key_rejects_post_terminal_step() {
        let mut env = env_5x5();
        assert_rejects_post_terminal_step(
            &mut env,
            |env| {
                env.reset_with_seed(3).expect("reset");
                drive_to_goal(env)
            },
            GridAction::Drop,
        );
    }

    #[test]
    /// Regression for the concrete defect the guard prevents. Nothing is
    /// consumed on success — the goal entity stays in the grid, the agent stays
    /// on it, and the key stays in its hand — so an unguarded post-terminal step
    /// resumed the episode: `steps` advanced past the true episode length
    /// (discounting the `success_reward` of any goal re-entry), the agent turned
    /// and walked off the goal, and a later `Drop` emptied the hand.
    ///
    /// A rejected step must mutate none of the three.
    fn test_door_key_post_terminal_step_mutates_nothing() {
        let mut env = env_seed_3();
        let terminal = drive_to_goal(&mut env);
        assert!(
            terminal.is_terminated(),
            "the agent must have reached the goal"
        );

        let steps_at_end = env.steps();
        let agent_at_end = env.state().agent;
        assert_eq!(
            agent_at_end.carrying,
            Some(Entity::Key(DOOR_COLOR)),
            "the agent finishes the episode still holding the key"
        );

        let err = env
            .step(GridAction::TurnRight)
            .expect_err("a step after the goal must return Err, not another Running snapshot");
        match err {
            EnvironmentError::StepAfterEpisodeEnd { status } => assert_eq!(
                status,
                EpisodeStatus::Terminated,
                "the error must carry Terminated, the status that ended the episode"
            ),
            other => panic!("expected StepAfterEpisodeEnd, got {other:?}"),
        }

        assert_eq!(
            env.steps(),
            steps_at_end,
            "a rejected step must not advance the step counter"
        );
        assert_eq!(
            (env.state().agent.x, env.state().agent.y),
            (agent_at_end.x, agent_at_end.y),
            "a rejected step must not move the agent off the goal"
        );
        assert_eq!(
            env.state().agent.direction,
            agent_at_end.direction,
            "a rejected TurnRight must not turn the agent"
        );
        assert_eq!(
            env.state().agent.carrying,
            Some(Entity::Key(DOOR_COLOR)),
            "a rejected step must leave the key in the agent's hand"
        );
        assert_eq!(
            env.guard.status(),
            EpisodeStatus::Terminated,
            "a rejected step must not reopen the episode"
        );
    }

    #[test]
    /// Verifies `reset()` re-opens a terminated episode, so a guard that has
    /// latched cannot strand the environment for the rest of the run.
    fn test_door_key_reset_reopens_terminated_episode() {
        let mut env = env_seed_3();
        drive_to_goal(&mut env);
        assert!(
            env.step(GridAction::TurnRight).is_err(),
            "the episode has terminated; a step must be rejected before reset()"
        );

        env.reset().expect("reset must succeed after termination");
        assert_eq!(
            env.guard.status(),
            EpisodeStatus::Running,
            "reset() must return the guard to Running"
        );

        let snap = env
            .step(GridAction::TurnRight)
            .expect("reset() must re-open the environment for a new episode");
        assert!(
            !snap.is_done(),
            "a turn on the first step of a fresh episode cannot end it"
        );
        assert_eq!(env.steps(), 1, "the step counter must restart at the reset");
    }
}
