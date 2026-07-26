//! `UnlockPickup`: unlock a door and pick up the colored box.
//!
//! Ports Farama Minigrid's [`UnlockPickupEnv`] (`MiniGrid-UnlockPickup-v0`). Two
//! rooms are separated by an interior wall carrying a single locked door. A key
//! whose colour **matches the door** sits in the starting room; a target colored
//! box sits in the far room. Success is reached when the agent is carrying that
//! box.
//!
//! ## The layout is drawn fresh every episode
//!
//! The two-room *topology* is fixed: upstream configures its `RoomGrid` with
//! `num_rows = 1, num_cols = 2`, and rlevo's single board with a wall column at
//! `x = size / 2` is structurally that same shape. Everything *inside* the rooms
//! is sampled on every [`reset`](Environment::reset) from the environment's
//! persistent RNG, in upstream `_gen_grid`'s order:
//!
//! | # | Quantity | Drawn from |
//! |---|----------|-----------|
//! | 1 | door row on the split column | `1 ..= size - 2` (any interior row) |
//! | 2 | box colour | [`Color::ALL`] |
//! | 3 | box cell | the free cells of the right room |
//! | 4 | door colour | [`Color::ALL`] |
//! | 5 | key cell — its colour is **forced** to the door's | the free cells of the left room |
//! | 6 | agent cell, then agent facing | the free cells of the left room; any of four |
//!
//! Step 5 is what makes the task solvable by inspection: [`apply_action`] only
//! unlocks a door for the key of the *same* colour, so a key drawn independently
//! would leave most episodes unsolvable. The box colour, by contrast, *is* drawn
//! independently and may collide with the door's — harmlessly, because
//! [`Entity::Box`] and [`Entity::Key`] carry distinct
//! [`type_u8`](Entity::type_u8) bytes, so neither the observation nor
//! [`target`](UnlockPickupEnv::target) can confuse a box with a key of the same
//! colour.
//!
//! Draw *order* is part of the ported algorithm (ADR 0062 §1), not an
//! implementation detail: reordering these six draws yields a different
//! distribution over boards for the same seed.
//!
//! ## One episode, verbatim
//!
//! `UnlockPickupConfig::new(7, 196, 0)` after a single
//! [`reset_with_seed(11)`](UnlockPickupEnv::reset_with_seed), as
//! [`ascii`](UnlockPickupEnv::ascii) renders it (its trailing spaces trimmed):
//!
//! ```text
//! # # # # # # #
//! # . . # . . #    # = wall; the split column is x = size / 2 = 3
//! # . . * . . #    * = locked door (yellow) at (3, 2)
//! # . . # . [ #    [ = box (green) at (5, 3) — the target
//! # . k # . . #    k = key (yellow, matching the door) at (2, 4)
//! # < . # . . #    < = agent at (1, 5) facing West
//! # # # # # # #
//! ```
//!
//! [`ascii`](UnlockPickupEnv::ascii) does not render colour, so the annotations
//! carry it. A different seed moves every one of those glyphs; this one board is
//! documentation, not a specification.
//!
//! Required action sequence: pick up the key → navigate to the door →
//! toggle twice (unlock then open) → cross into the far room → drop the key →
//! pick up the box.
//!
//! | Observation | 7 × 7 egocentric grid encoded as `[type, color, state]` per cell      |
//! |-------------|------------------------------------------------------------------------|
//! | Action      | `TurnLeft`, `TurnRight`, `Forward`, `Pickup`, `Drop`, `Toggle`         |
//! | Reward      | `success_reward(steps, max_steps)` when box is carried; `0.0` on timeout |
//!
//! # Examples
//!
//! ```rust
//! use rlevo_environments::grids::unlock_pickup::{UnlockPickupConfig, UnlockPickupEnv};
//! use rlevo_core::environment::{ConstructableEnv, Environment};
//!
//! let cfg = UnlockPickupConfig::new(7, 196, 0);
//! let mut env = UnlockPickupEnv::with_config(cfg, false).expect("valid config");
//! let snap = env.reset().unwrap();
//! // Both are re-derived by every `reset`.
//! println!("target: {:?} behind a {:?} door", env.target(), env.door_color());
//! ```
//!
//! [`UnlockPickupEnv`]: https://minigrid.farama.org/environments/minigrid/UnlockPickupEnv/

use super::core::{
    GridSnapshot, Visibility,
    action::GridAction,
    build_snapshot,
    color::Color,
    dynamics::apply_action,
    entity::{DoorState, Entity},
    grid::Grid,
    observation::GridObservation,
    observe_grid,
    placement::{PlacementError, Rect, no_reject, place_agent, place_obj},
    render::render_ascii,
    reward::success_reward,
    state::GridState,
};
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};
use rlevo_core::config::{self, ConfigError, Validate};
use rlevo_core::environment::{ConstructableEnv, Environment, EnvironmentError, Sensor};
use rlevo_core::reward::ScalarReward;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

/// Minimum side length, enforced as this environment's documented policy by
/// [`Validate`].
///
/// At the floor the split column sits at `x = 3` and leaves a `2 × 5` room on
/// each side of it, so the key, the agent, and the box each have room to be
/// drawn without the two rooms degenerating into corridors of a single cell.
const MIN_SIZE: usize = 7;

/// Configuration for [`UnlockPickupEnv`].
///
/// # Examples
///
/// ```rust
/// use rlevo_environments::grids::unlock_pickup::UnlockPickupConfig;
///
/// let cfg = UnlockPickupConfig::new(7, 196, 0);
/// assert_eq!(cfg.size, 7);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnlockPickupConfig {
    /// Grid side length in cells (width = height = `size`); must be ≥ `MIN_SIZE` (7).
    ///
    /// The interior wall sits at `x = size / 2` and splits the board into a left
    /// room spanning `x ∈ 1 ..= size / 2 - 1` and a right room spanning
    /// `x ∈ size / 2 + 1 ..= size - 2`, both covering rows `1 ..= size - 2`.
    /// Nothing inside those rooms is pinned to a coordinate derived from `size`:
    /// the door's row, the key and the agent in the left room, and the box in
    /// the right room are all drawn per episode (see the module docs). A larger
    /// `size` therefore widens both rooms rather than moving a fixed landmark.
    pub size: usize,
    /// Maximum steps before the episode times out with reward `0.0`.
    pub max_steps: usize,
    /// Seed for the environment's persistent RNG, drawn once at construction.
    ///
    /// The RNG **advances** across resets (ADR 0029), so successive episodes get
    /// independent door rows, colours, and object/agent placements. A fixed seed
    /// therefore reproduces a fixed *sequence* of episodes, not one repeated
    /// episode. For bit-for-bit replay of a single episode use
    /// [`UnlockPickupEnv::reset_with_seed`].
    ///
    /// # Deviations from `MiniGrid-UnlockPickup-v0`
    ///
    /// The set of quantities drawn and the order they are drawn in match
    /// upstream `_gen_grid` (module docs). These are the differences, recorded
    /// here because ADR 0062 §1 requires a deviation to be visible at the config
    /// field rather than only in a tracker:
    ///
    /// - **Board shape.** Upstream is an `11 × 6` `RoomGrid` of two `6 × 6`
    ///   rooms; rlevo uses one square `size × size` board with a wall column at
    ///   `x = size / 2`. The two-room topology is the same; the extent is
    ///   configurable.
    /// - **No `reject_next_to` veto on the box and the key.** Upstream's
    ///   `RoomGrid.place_in_room` refuses cells within Manhattan distance 2 of
    ///   `self.agent_pos` — which at that moment still holds `RoomGrid`'s
    ///   placeholder pose at the centre of the right room, because the real
    ///   agent is placed afterwards (ADR 0062 §1 records this ordering). rlevo
    ///   has no placeholder pose for the veto to read, so both objects are drawn
    ///   uniformly over their room.
    /// - **No "agent must not start facing an object" retry.** Upstream's
    ///   `RoomGrid.place_agent` re-draws until the agent's front cell is empty
    ///   or a wall; rlevo accepts the first draw.
    /// - **Placement materialises the free set** and draws a uniform index
    ///   instead of running upstream's unbounded rejection loop — same
    ///   distribution, no retry budget, and exhaustion is an `Err` rather than a
    ///   hang. See [`placement`](super::core::placement).
    pub seed: u64,
}

impl UnlockPickupConfig {
    /// Creates an [`UnlockPickupConfig`] with the given parameters.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rlevo_environments::grids::unlock_pickup::UnlockPickupConfig;
    ///
    /// let cfg = UnlockPickupConfig::new(9, 324, 42);
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

impl Default for UnlockPickupConfig {
    fn default() -> Self {
        let size = 7;
        Self {
            size,
            max_steps: 4 * size * size,
            seed: 0,
        }
    }
}

impl Validate for UnlockPickupConfig {
    /// Rejects any `size` below `MIN_SIZE` (7) and a zero `max_steps`.
    ///
    /// The `size` guard lives **here**, not only in [`FromStr`]:
    /// `UnlockPickupConfig` derives `Deserialize`, so a config loaded from a
    /// file is user-supplied runtime data that never passes through `from_str`
    /// (rules.md §4 — "if an invalid value can arrive via `Deserialize`, it must
    /// be an `Err`"). Below the floor the two rooms the layout sampler draws
    /// into stop being well-formed: at `size ≤ 4` the right room's rectangle is
    /// empty or inverted, which [`UnlockPickupEnv::with_config`] surfaces as a
    /// [`PlacementError`]-derived `ConfigError` rather than a panic. Sizes `5`
    /// and `6` do build a playable board and are rejected anyway, so `MIN_SIZE`
    /// is a documented policy — every shipped `UnlockPickup` board has at least
    /// a `2 × 5` room on each side of the split — not merely a crash guard.
    ///
    /// `at_least` subsumes the previous `nonzero` check — `MIN_SIZE >= 1`, so a
    /// zero `size` is still rejected, now as
    /// [`ConstraintKind::TooSmall`](rlevo_core::config::ConstraintKind::TooSmall).
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "UnlockPickupConfig";
        config::at_least(C, "size", self.size, MIN_SIZE)?;
        config::nonzero(C, "max_steps", self.max_steps)?;
        Ok(())
    }
}

impl FromStr for UnlockPickupConfig {
    type Err = String;

    /// Parses `"size=7,max_steps=196,seed=0"` (keys in any order) or the
    /// positional form `"7,196,0"`.
    ///
    /// # Errors
    ///
    /// Returns the offending key/value, or the [`Validate`] rejection — the same
    /// guard [`UnlockPickupEnv::with_config`] applies, so this parser cannot
    /// admit a config that construction would refuse.
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

/// The per-episode quantities [`UnlockPickupEnv::build`] draws beside the board
/// itself.
///
/// Only the values that stay meaningful for a whole episode live here. The key's
/// and the box's cells deliberately do not: both objects can be picked up and
/// dropped, so a stored coordinate would go stale on the first `Pickup`. Read
/// those back from [`UnlockPickupEnv::state`] instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Layout {
    /// The box the mission names, exactly as written into the right room.
    target: Entity,
    /// Cell of the locked door on the split column.
    door_pos: (i32, i32),
    /// Colour shared by the door and the key that opens it.
    door_color: Color,
}

/// Minigrid's `UnlockPickup` environment.
///
/// Two rooms separated by a locked door. The agent must pick up the key
/// from the starting room, unlock and open the door, cross into the far
/// room, and pick up the target box. Reward is paid the moment the agent
/// is carrying the box; the episode times out if the step budget runs out.
///
/// Every episode draws its own door row, door colour, box colour, key cell, box
/// cell, and agent pose — see the module docs for the draw order and
/// [`UnlockPickupConfig::seed`] for the reproducibility contract.
///
/// Implements [`Environment<3, 3, 1>`] with [`GridState`] /
/// [`GridObservation`](super::core::GridObservation) / [`GridAction`] / [`ScalarReward`].
///
/// # Examples
///
/// ```rust
/// use rlevo_environments::grids::unlock_pickup::UnlockPickupEnv;
/// use rlevo_core::environment::{ConstructableEnv, Environment};
///
/// let mut env = UnlockPickupEnv::new(false);
/// let snap = env.reset().unwrap();
/// println!("target: {:?}", env.target());
/// ```
#[derive(Debug)]
pub struct UnlockPickupEnv {
    state: GridState,
    config: UnlockPickupConfig,
    steps: usize,
    render: bool,
    layout: Layout,
    rng: StdRng,
}

impl UnlockPickupEnv {
    /// Emission-model visibility policy: does this environment's agent see
    /// through opaque cells?
    ///
    /// The rlevo spelling of canonical Minigrid's `see_through_walls`
    /// constructor argument. Read only by this environment's [`Sensor`] impl,
    /// and an inherent const rather than a config field because it is part of
    /// the task definition, not a knob a caller tunes.
    ///
    /// [`Visibility::Occluded`], because canonical `UnlockPickup` derives from
    /// `RoomGrid`, and `minigrid/core/roomgrid.py`'s `RoomGrid.__init__` passes
    /// `see_through_walls=False` up to `MiniGridEnv` for every room-based env.
    /// The room divider and the locked door therefore hide the box until the
    /// door is opened. See ADR 0063
    /// (`docs/adr/0063-grid-visibility-occlusion.md`) for the whole
    /// twelve-environment table.
    const VISIBILITY: Visibility = Visibility::Occluded;

    /// Constructs an [`UnlockPickupEnv`] from an explicit configuration.
    ///
    /// Seeds the persistent RNG once from `config.seed` and immediately builds a
    /// first episode from it. Call [`Environment::reset`] before the first
    /// [`Environment::step`] to obtain the first observation; that reset samples
    /// a *fresh* layout, advancing the stream.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rlevo_environments::grids::unlock_pickup::{UnlockPickupConfig, UnlockPickupEnv};
    ///
    /// let env = UnlockPickupEnv::with_config(
    ///     UnlockPickupConfig::new(7, 196, 0),
    ///     true, // render ASCII grid to stdout
    /// )
    /// .expect("valid config");
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `config` fails [`Validate`]: a `size` below
    /// `MIN_SIZE` (7) or a zero `max_steps`. This is the construction chokepoint
    /// (rules.md §4), so it also rejects a config that arrived by `Deserialize`
    /// or struct-update syntax without passing through [`FromStr`].
    ///
    /// Also returns a [`ConfigError`] if the first layout cannot be placed —
    /// a [`PlacementError`] converted at the boundary. Every `size` this
    /// function accepts leaves both rooms large enough for their draws, so this
    /// arm is unreachable through the public API; it is propagated rather than
    /// unwrapped because exhaustion is a function of the config *and* the draws
    /// already made (ADR 0062 §3a).
    pub fn with_config(config: UnlockPickupConfig, render: bool) -> Result<Self, ConfigError> {
        config.validate()?;
        let mut rng = StdRng::seed_from_u64(config.seed);
        let (state, layout) = Self::build(&config, &mut rng)?;
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
    /// successive episodes get independent layouts (ADR 0029); use this when you
    /// need a *specific* episode to reproduce bit-for-bit (e.g. replaying a
    /// failure, or pinning a board in a doc example). Run-level reproducibility
    /// is already guaranteed by the construction seed.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`reset`](Environment::reset) — that is, an
    /// [`EnvironmentError::Config`] should the fresh layout fail to place.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rlevo_environments::grids::unlock_pickup::UnlockPickupEnv;
    /// use rlevo_core::environment::{ConstructableEnv, Environment};
    ///
    /// let mut env = UnlockPickupEnv::new(false);
    /// env.reset_with_seed(11).unwrap();
    /// let (board, target) = (env.ascii(), env.target());
    ///
    /// env.reset().unwrap(); // advance the persistent stream
    /// env.reset_with_seed(11).unwrap();
    /// assert_eq!(board, env.ascii());
    /// assert_eq!(target, env.target());
    /// ```
    pub fn reset_with_seed(&mut self, seed: u64) -> Result<GridSnapshot, EnvironmentError> {
        self.rng = StdRng::seed_from_u64(seed);
        self.reset()
    }

    /// Returns the environment's active configuration.
    #[must_use]
    pub const fn config(&self) -> &UnlockPickupConfig {
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

    /// Entity the agent must be holding to solve the task.
    ///
    /// This is the box of **this** episode: both its colour and its cell are
    /// re-drawn by every [`reset`](Environment::reset), and the returned value is
    /// re-derived with them. A scripted or planning caller must therefore read it
    /// *after* the reset it belongs to.
    #[must_use]
    pub const fn target(&self) -> Entity {
        self.layout.target
    }

    /// Cell of this episode's locked door, on the split column `x = size / 2`.
    ///
    /// The row is re-drawn by every [`reset`](Environment::reset). The door
    /// itself never moves during an episode — only its
    /// [`DoorState`] changes — so this stays valid until the next reset.
    #[must_use]
    pub const fn door_pos(&self) -> (i32, i32) {
        self.layout.door_pos
    }

    /// Colour of this episode's door, which is also the colour of the key that
    /// opens it.
    ///
    /// [`apply_action`] unlocks a door only for a carried [`Entity::Key`] of the
    /// matching colour, so this single value identifies both halves of the lock.
    #[must_use]
    pub const fn door_color(&self) -> Color {
        self.layout.door_color
    }

    /// Renders the current grid state as an ASCII string.
    #[must_use]
    pub fn ascii(&self) -> String {
        render_ascii(&self.state.grid, &self.state.agent)
    }

    /// Build a fresh episode: two rooms, a locked door on the shared wall, a
    /// matching key and the agent in the left room, and the target box in the
    /// right one.
    ///
    /// Draws from `rng` and lets it advance (ADR 0029) — never re-seeds. The six
    /// draws happen in upstream `_gen_grid`'s order, which the module docs
    /// tabulate and ADR 0062 §1 makes part of the ported algorithm.
    ///
    /// # Errors
    ///
    /// Returns [`PlacementError`] when either room's rectangle is degenerate
    /// (only reachable below `MIN_SIZE`, which [`Validate`] rejects) or when a
    /// room has no free cell left for the object being placed. Both call sites
    /// propagate it with `?`.
    fn build<R: Rng + ?Sized>(
        config: &UnlockPickupConfig,
        rng: &mut R,
    ) -> Result<(GridState, Layout), PlacementError> {
        let mut grid = Grid::new(config.size, config.size);
        grid.draw_walls();

        #[allow(clippy::cast_possible_wrap)]
        let size = config.size as i32;
        #[allow(clippy::cast_possible_wrap)]
        let split_col = (config.size / 2) as i32;

        // Both rooms are derived before any draw, so a degenerate board is an
        // `Err` from geometry rather than a panic from an empty sampling range.
        let left_room = Rect::try_new(1, 1, split_col - 1, size - 2)?;
        let right_room = Rect::try_new(split_col + 1, 1, size - 2, size - 2)?;

        // Interior vertical wall; the door is punched through it below.
        for y in 1..size - 1 {
            grid.set(split_col, y, Entity::Wall);
        }

        // 1. The door's row on the shared wall. Upstream draws this first, in
        //    `RoomGrid._gen_grid`, before the task-specific objects.
        let door_y = rng.random_range(1..size - 1);

        // 2-3. The box: colour first, then a free cell of the right room.
        let box_color = sample_color(rng);
        place_obj(
            &mut grid,
            rng,
            Entity::Box(box_color),
            right_room,
            &mut no_reject,
        )?;

        // 4. The door's colour. Its cell was fixed by draw 1.
        let door_color = sample_color(rng);
        grid.set(
            split_col,
            door_y,
            Entity::Door(door_color, DoorState::Locked),
        );

        // 5. The key. Its colour is *forced* to the door's — upstream passes
        //    `door.color` rather than drawing — and only its cell is sampled.
        place_obj(
            &mut grid,
            rng,
            Entity::Key(door_color),
            left_room,
            &mut no_reject,
        )?;

        // 6. The agent's pose in the left room: cell, then facing.
        let agent = place_agent(&grid, rng, left_room, &mut no_reject)?;

        Ok((
            GridState::new(grid, agent),
            Layout {
                target: Entity::Box(box_color),
                door_pos: (split_col, door_y),
                door_color,
            },
        ))
    }

    fn emit(&self, observation: GridObservation, reward: f32, done: bool) -> GridSnapshot {
        if self.render {
            println!("{}", self.ascii());
        }
        build_snapshot(observation, reward, done)
    }

    fn has_target(&self) -> bool {
        self.state.agent.carrying == Some(self.layout.target)
    }
}

/// Draw one colour uniformly from [`Color::ALL`].
///
/// Deliberately unconstrained: upstream `add_object` / `add_door` call
/// `_rand_color()` independently for the box and the door, so the two may
/// coincide. That is not ambiguous here — see the module docs.
fn sample_color<R: Rng + ?Sized>(rng: &mut R) -> Color {
    Color::ALL[rng.random_range(0..Color::ALL.len())]
}

impl crate::render::AsciiRenderable for UnlockPickupEnv {
    fn render_ascii(&self) -> String {
        render_ascii(&self.state.grid, &self.state.agent)
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        super::core::render::render_styled(&self.state.grid, &self.state.agent)
    }
}

impl Display for UnlockPickupEnv {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "UnlockPickupEnv(size={}, step={}/{})",
            self.config.size, self.steps, self.config.max_steps
        )
    }
}

impl ConstructableEnv for UnlockPickupEnv {
    fn new(render: bool) -> Self {
        Self::with_config(UnlockPickupConfig::default(), render)
            .expect("default config must validate")
    }
}

impl Sensor<3, 1, 3> for UnlockPickupEnv {
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

impl Environment<3, 3, 1> for UnlockPickupEnv {
    type StateType = GridState;
    type ObservationType = super::core::GridObservation;
    type ActionType = GridAction;
    type RewardType = ScalarReward;
    type SnapshotType = GridSnapshot;

    /// Draws a fresh layout from the persistent stream and re-derives every
    /// value that depends on it.
    ///
    /// The per-episode layout is replaced wholesale rather than field by field:
    /// the target box is part of the draw, so leaving
    /// [`target`](UnlockPickupEnv::target) behind would score the episode against
    /// an object that is not on the board.
    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        let (state, layout) = Self::build(&self.config, &mut self.rng)?;
        self.state = state;
        self.layout = layout;
        self.steps = 0;
        let observation = self.observe_reset(&self.state);
        Ok(self.emit(observation, 0.0, false))
    }

    fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> {
        self.steps += 1;
        let _ = apply_action(&mut self.state.grid, &mut self.state.agent, action);
        let (reward, done) = if self.has_target() {
            (success_reward(self.steps, self.config.max_steps), true)
        } else if self.steps >= self.config.max_steps {
            (0.0, true)
        } else {
            (0.0, false)
        };
        let observation = self.observe(&action, &self.state);
        Ok(self.emit(observation, reward, done))
    }
}

impl rlevo_core::render::payload::GridPayloadSource for UnlockPickupEnv {
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
    use crate::grids::core::AgentState;
    use rlevo_core::config::ConstraintKind;
    use rlevo_core::environment::Snapshot;
    use std::collections::{HashSet, VecDeque};

    /// Every seed loop runs at the enforced floor **and** at a larger board, so
    /// a range that happens to be non-degenerate only at one size is caught.
    const SIZES: [usize; 2] = [MIN_SIZE, 9];

    /// Seeds per loop. Large enough that a quantity with support 4 (the agent's
    /// facing) or 6 (a colour) must cover its whole range, small enough that the
    /// structural loops stay sub-second.
    const SEEDS: u64 = 64;

    fn env_of(size: usize, seed: u64) -> UnlockPickupEnv {
        UnlockPickupEnv::with_config(UnlockPickupConfig::new(size, 4 * size * size, seed), false)
            .expect("valid config")
    }

    fn env_7x7() -> UnlockPickupEnv {
        env_of(7, 0)
    }

    fn as_i32(size: usize) -> i32 {
        i32::try_from(size).expect("test sizes fit in i32")
    }

    /// Every cell of `grid` whose entity satisfies `pred`, row-major.
    fn cells_where(grid: &Grid, pred: impl Fn(Entity) -> bool) -> Vec<((i32, i32), Entity)> {
        let mut out = Vec::new();
        for y in 0..as_i32(grid.height()) {
            for x in 0..as_i32(grid.width()) {
                let e = grid.get(x, y);
                if pred(e) {
                    out.push(((x, y), e));
                }
            }
        }
        out
    }

    /// The single cell holding an entity matching `pred`, or a panic naming the
    /// board — every caller below has already asserted the count is one.
    fn only_cell(grid: &Grid, what: &str, pred: impl Fn(Entity) -> bool) -> ((i32, i32), Entity) {
        let found = cells_where(grid, pred);
        assert_eq!(found.len(), 1, "expected exactly one {what}, got {found:?}");
        found[0]
    }

    fn key_cell(grid: &Grid) -> ((i32, i32), Color) {
        let (pos, e) = only_cell(grid, "key", |e| matches!(e, Entity::Key(_)));
        let Entity::Key(c) = e else { unreachable!() };
        (pos, c)
    }

    fn box_cell(grid: &Grid) -> ((i32, i32), Color) {
        let (pos, e) = only_cell(grid, "box", |e| matches!(e, Entity::Box(_)));
        let Entity::Box(c) = e else { unreachable!() };
        (pos, c)
    }

    /// A whole-board fingerprint: the rendered glyphs (which carry every cell
    /// plus the agent's pose and facing) together with the colours the renderer
    /// drops.
    fn fingerprint(env: &UnlockPickupEnv) -> (String, Entity, Color, (i32, i32)) {
        (env.ascii(), env.target(), env.door_color(), env.door_pos())
    }

    // -- config -------------------------------------------------------------

    #[test]
    fn default_config_validates() {
        assert!(UnlockPickupConfig::default().validate().is_ok());
    }

    #[test]
    fn rejects_zero_size() {
        let bad = UnlockPickupConfig {
            size: 0,
            ..Default::default()
        };
        assert!(UnlockPickupEnv::with_config(bad, false).is_err());
    }

    #[test]
    fn default_config_is_7x7() {
        let cfg = UnlockPickupConfig::default();
        assert_eq!(cfg.size, 7);
        assert_eq!(cfg.max_steps, 4 * 7 * 7);
    }

    #[test]
    fn fromstr_rejects_small_size() {
        assert!("5".parse::<UnlockPickupConfig>().is_err());
    }

    /// Issue #106: `MIN_SIZE` was enforced only in [`FromStr`], so a config
    /// built by `Deserialize` or struct-update syntax reached `build`. The guard
    /// now lives in [`Validate`], which `with_config` runs (ADR 0026
    /// chokepoint), and it rejects the whole sub-`MIN_SIZE` range — including
    /// sizes such as `5` that happen to build a playable board.
    #[test]
    fn with_config_rejects_size_below_min() {
        let bad = UnlockPickupConfig {
            size: MIN_SIZE - 1,
            ..Default::default()
        };
        let err = UnlockPickupEnv::with_config(bad, false).unwrap_err();
        assert_eq!(err.config, "UnlockPickupConfig");
        assert_eq!(err.field, "size");
        assert_eq!(
            err.kind,
            ConstraintKind::TooSmall {
                min: MIN_SIZE as u64,
                got: (MIN_SIZE - 1) as u64,
            }
        );
    }

    #[test]
    fn fromstr_parses_all_fields() {
        let cfg: UnlockPickupConfig = "size=8,max_steps=200,seed=4".parse().unwrap();
        assert_eq!(cfg.size, 8);
        assert_eq!(cfg.max_steps, 200);
        assert_eq!(cfg.seed, 4);
    }

    /// The chokepoint makes this unreachable through the public API, so `build`
    /// is called directly: a degenerate room must come back as an `Err` that
    /// converts into `ConfigError`, never as a panic from `Rect::new` or from an
    /// empty `random_range`. This is the ADR 0062 §3a contract — propagate with
    /// `?`, do not `.expect()`.
    #[test]
    fn build_below_the_floor_errors_instead_of_panicking() {
        for size in 1..=4usize {
            let cfg = UnlockPickupConfig::new(size, 10, 0);
            let mut rng = StdRng::seed_from_u64(0);
            let err = UnlockPickupEnv::build(&cfg, &mut rng)
                .expect_err("size {size} has no well-formed right room");
            assert!(
                matches!(err, PlacementError::InvertedRect { .. }),
                "size {size}: expected a degenerate room, got {err:?}"
            );
            let as_config: ConfigError = err.into();
            assert_eq!(as_config.config, "GridPlacement");
        }
    }

    // -- structure ----------------------------------------------------------

    /// Per seed and per size, the board is the `UnlockPickup` board: a sealed
    /// perimeter, a solid split column pierced by exactly one locked door, one
    /// key and the agent strictly left of it, one box strictly right of it.
    ///
    /// Three resets per env, so the invariants are asserted against boards drawn
    /// from an *advanced* stream and not only against the constructor's.
    #[test]
    fn every_episode_is_structurally_well_formed() {
        for size in SIZES {
            let n = as_i32(size);
            let split = as_i32(size / 2);
            for seed in 0..SEEDS {
                let mut env = env_of(size, seed);
                for episode in 0..3 {
                    env.reset().expect("reset");
                    let where_ = format!("size {size}, seed {seed}, episode {episode}");
                    let grid = &env.state().grid;

                    for i in 0..n {
                        for (x, y) in [(i, 0), (i, n - 1), (0, i), (n - 1, i)] {
                            assert_eq!(
                                grid.get(x, y),
                                Entity::Wall,
                                "{where_}: perimeter cell ({x}, {y}) is not a wall"
                            );
                        }
                    }

                    let (dx, dy) = env.door_pos();
                    assert_eq!(dx, split, "{where_}: the door left the split column");
                    for y in 1..n - 1 {
                        let expected = if y == dy {
                            Entity::Door(env.door_color(), DoorState::Locked)
                        } else {
                            Entity::Wall
                        };
                        assert_eq!(
                            grid.get(split, y),
                            expected,
                            "{where_}: split column cell ({split}, {y}) is wrong"
                        );
                    }

                    let doors = cells_where(grid, |e| matches!(e, Entity::Door(..)));
                    assert_eq!(doors.len(), 1, "{where_}: expected one door, got {doors:?}");

                    let ((kx, ky), _) = key_cell(grid);
                    assert!(
                        (1..split).contains(&kx) && (1..n - 1).contains(&ky),
                        "{where_}: key at ({kx}, {ky}) is not strictly inside the left room"
                    );

                    let ((bx, by), _) = box_cell(grid);
                    assert!(
                        (split + 1..n - 1).contains(&bx) && (1..n - 1).contains(&by),
                        "{where_}: box at ({bx}, {by}) is not strictly inside the right room"
                    );

                    let agent = env.state().agent;
                    assert!(
                        (1..split).contains(&agent.x) && (1..n - 1).contains(&agent.y),
                        "{where_}: agent at ({}, {}) is not in the left room",
                        agent.x,
                        agent.y
                    );
                    assert_eq!(
                        agent.carrying, None,
                        "{where_}: a fresh agent carries nothing"
                    );
                }
            }
        }
    }

    /// Upstream forces `add_object(0, 0, "key", door.color)`, and
    /// [`apply_action`] only unlocks a door for the key of its own colour — so a
    /// mismatch would make the episode unsolvable rather than merely odd.
    #[test]
    fn the_key_always_matches_the_door() {
        for size in SIZES {
            for seed in 0..SEEDS {
                let mut env = env_of(size, seed);
                for _ in 0..3 {
                    env.reset().expect("reset");
                    let (_, key_color) = key_cell(&env.state().grid);
                    assert_eq!(
                        key_color,
                        env.door_color(),
                        "size {size}, seed {seed}: the key does not open the door"
                    );
                }
            }
        }
    }

    /// Regression for a stale [`UnlockPickupEnv::target`].
    ///
    /// `target` used to be computed once in `with_config`, which was invisible
    /// while the box colour was a constant. The moment the colour became a draw,
    /// a `reset` that did not re-derive it would score the episode against a box
    /// that is not on the board — and every other test here would still pass.
    #[test]
    fn target_tracks_the_box_of_the_current_episode() {
        for size in SIZES {
            for seed in 0..SEEDS {
                let mut env = env_of(size, seed);
                for episode in 0..4 {
                    env.reset().expect("reset");
                    let (_, color) = box_cell(&env.state().grid);
                    assert_eq!(
                        env.target(),
                        Entity::Box(color),
                        "size {size}, seed {seed}, episode {episode}: target() is stale — \
                         the board holds a {color:?} box"
                    );
                }
            }
        }
    }

    /// The replacement for the old `target_is_box_of_declared_color`: there is
    /// no declared colour any more, but "the target is always a box, and always
    /// one of the palette's colours" is still a true and useful property.
    #[test]
    fn target_is_always_a_box_from_the_palette() {
        for size in SIZES {
            for seed in 0..SEEDS {
                let mut env = env_of(size, seed);
                env.reset().expect("reset");
                let Entity::Box(color) = env.target() else {
                    panic!("size {size}, seed {seed}: target is {:?}", env.target())
                };
                assert!(Color::ALL.contains(&color));
            }
        }
    }

    // -- non-degeneracy -----------------------------------------------------

    /// Each sampled quantity is checked **on its own**.
    ///
    /// A single "two observations differ" assertion would pass while five of the
    /// six draws were pinned — the trap ADR 0062 §4 and issue #282 both name.
    #[test]
    fn every_sampled_quantity_varies_across_seeds() {
        for size in SIZES {
            let mut door_rows = HashSet::new();
            let mut door_colors = HashSet::new();
            let mut box_colors = HashSet::new();
            let mut key_cells = HashSet::new();
            let mut box_cells = HashSet::new();
            let mut agent_cells = HashSet::new();
            let mut agent_dirs = HashSet::new();

            for seed in 0..SEEDS {
                let mut env = env_of(size, seed);
                env.reset().expect("reset");
                let grid = &env.state().grid;
                door_rows.insert(env.door_pos().1);
                door_colors.insert(env.door_color());
                let ((kx, ky), _) = key_cell(grid);
                key_cells.insert((kx, ky));
                let ((bx, by), box_color) = box_cell(grid);
                box_cells.insert((bx, by));
                box_colors.insert(box_color);
                agent_cells.insert((env.state().agent.x, env.state().agent.y));
                agent_dirs.insert(env.state().agent.direction);
            }

            // Two quantities have a small enough support to demand full
            // coverage; the rest only have to move.
            assert_eq!(
                door_rows.len(),
                size - 2,
                "size {size}: the door must reach every interior row, saw {door_rows:?}"
            );
            assert_eq!(
                agent_dirs.len(),
                4,
                "size {size}: the agent's facing must vary, saw {agent_dirs:?}"
            );
            for (label, seen) in [
                ("door colour", door_colors.len()),
                ("box colour", box_colors.len()),
            ] {
                assert_eq!(
                    seen,
                    Color::ALL.len(),
                    "size {size}: {label} must cover the palette, saw {seen}"
                );
            }
            for (label, seen) in [
                ("key cell", key_cells.len()),
                ("box cell", box_cells.len()),
                ("agent cell", agent_cells.len()),
            ] {
                assert!(
                    seen > 1,
                    "size {size}: {label} never varied across {SEEDS} seeds"
                );
            }
        }
    }

    /// The box colour is drawn *independently* of the door's, exactly as
    /// upstream's two `_rand_color()` calls are.
    ///
    /// Collisions are therefore expected (about one episode in six) and are left
    /// alone rather than rejection-sampled away: `Entity::Box` and `Entity::Key`
    /// have different `type_u8` bytes, so an equal colour cannot make the
    /// mission ambiguous — `target()` is a `Box` and can never match a carried
    /// `Key`. Asserting *both* outcomes occur is what pins the independence: a
    /// rejection loop would drive `same` to zero, and a forced tie would drive
    /// `different` to zero.
    #[test]
    fn the_box_colour_is_independent_of_the_door_colour() {
        for size in SIZES {
            let (mut same, mut different) = (0u32, 0u32);
            for seed in 0..SEEDS {
                let mut env = env_of(size, seed);
                env.reset().expect("reset");
                let (_, box_color) = box_cell(&env.state().grid);
                if box_color == env.door_color() {
                    same += 1;
                } else {
                    different += 1;
                }
            }
            assert!(
                same > 0,
                "size {size}: the box colour never coincided with the door's in {SEEDS} \
                 episodes — has it been rejection-sampled?"
            );
            assert!(
                different > 0,
                "size {size}: the box colour always equalled the door's — it must be an \
                 independent draw, unlike the key's"
            );
        }
    }

    // -- solvability --------------------------------------------------------

    /// Cells reachable from `start` by walking four-connected over passable
    /// cells, with `extra` treated as passable regardless of what it holds.
    ///
    /// `extra` is how the caller says "this cell will be free by the time the
    /// agent gets there": the key's cell is emptied by the `Pickup` that has to
    /// happen anyway, and the door's cell becomes passable once unlocked and
    /// opened.
    fn flood(grid: &Grid, start: (i32, i32), extra: &HashSet<(i32, i32)>) -> HashSet<(i32, i32)> {
        let mut seen = HashSet::from([start]);
        let mut queue = VecDeque::from([start]);
        while let Some((x, y)) = queue.pop_front() {
            for (nx, ny) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] {
                if !grid.in_bounds(nx, ny) {
                    continue;
                }
                if !grid.get(nx, ny).is_passable() && !extra.contains(&(nx, ny)) {
                    continue;
                }
                if seen.insert((nx, ny)) {
                    queue.push_back((nx, ny));
                }
            }
        }
        seen
    }

    /// Every drawn board is solvable in the shape the task requires: the key is
    /// reachable *before* the door opens, the locked door genuinely separates
    /// the two rooms, the agent can stand where it must to toggle it, and the
    /// box is reachable once through.
    #[test]
    fn every_episode_is_reachable_in_the_right_order() {
        for size in SIZES {
            let split = as_i32(size / 2);
            for seed in 0..SEEDS {
                let mut env = env_of(size, seed);
                for episode in 0..3 {
                    env.reset().expect("reset");
                    let where_ = format!("size {size}, seed {seed}, episode {episode}");
                    let grid = &env.state().grid;
                    let (key, _) = key_cell(grid);
                    let (boxp, _) = box_cell(grid);
                    let door = env.door_pos();
                    let start = (env.state().agent.x, env.state().agent.y);

                    let locked = flood(grid, start, &HashSet::from([key]));
                    assert!(
                        locked.contains(&key),
                        "{where_}: the key at {key:?} is unreachable from {start:?}"
                    );
                    assert!(
                        locked.iter().all(|&(x, _)| x < split),
                        "{where_}: the locked door did not separate the rooms — the flood \
                         escaped past column {split}"
                    );
                    assert!(
                        locked.contains(&(door.0 - 1, door.1)),
                        "{where_}: the agent cannot reach {:?}, the only cell it can toggle \
                         the door from",
                        (door.0 - 1, door.1)
                    );

                    let opened = flood(grid, door, &HashSet::from([door, key, boxp]));
                    assert!(
                        opened.contains(&boxp),
                        "{where_}: the box at {boxp:?} is unreachable through the open door"
                    );
                }
            }
        }
    }

    // -- reproducibility ----------------------------------------------------

    /// Two envs built from the same config agree on their first episode. Under
    /// the persistent-stream model this is a real assertion rather than the
    /// tautology it was when the board was a constant (ADR 0062, Neutral).
    #[test]
    fn reset_is_deterministic() {
        let cfg = UnlockPickupConfig::new(7, 196, 42);
        let mut a = UnlockPickupEnv::with_config(cfg, false).expect("valid config");
        let mut b = UnlockPickupEnv::with_config(cfg, false).expect("valid config");
        let sa = a.reset().unwrap();
        let sb = b.reset().unwrap();
        assert_eq!(sa.observation(), sb.observation());
    }

    /// The stream advances: `reset` must not re-seed from `config.seed`.
    #[test]
    fn consecutive_resets_differ() {
        for size in SIZES {
            for seed in 0..8u64 {
                let mut env = env_of(size, seed);
                env.reset().expect("reset");
                let first = fingerprint(&env);
                env.reset().expect("reset");
                let second = fingerprint(&env);
                assert_ne!(
                    first, second,
                    "size {size}, seed {seed}: two consecutive episodes are identical — \
                     has reset() started re-seeding?"
                );
            }
        }
    }

    /// `reset_with_seed` is the replay hatch: identical seed, identical episode,
    /// down to the emitted observation and regardless of how far the persistent
    /// stream had advanced.
    #[test]
    fn reset_with_seed_replays_and_distinct_seeds_diverge() {
        for size in SIZES {
            let mut env = env_of(size, 0);
            for replay_seed in [1u64, 2, 99] {
                let a = env.reset_with_seed(replay_seed).expect("reset");
                let board_a = (fingerprint(&env), *a.observation());
                // Advance the persistent stream between the two replays, so the
                // match cannot be an artefact of the env sitting still.
                env.reset().expect("reset");
                env.reset().expect("reset");
                let b = env.reset_with_seed(replay_seed).expect("reset");
                let board_b = (fingerprint(&env), *b.observation());
                assert_eq!(
                    board_a, board_b,
                    "size {size}: reset_with_seed({replay_seed}) did not replay"
                );
            }

            env.reset_with_seed(1).expect("reset");
            let one = fingerprint(&env);
            env.reset_with_seed(2).expect("reset");
            let two = fingerprint(&env);
            assert_ne!(one, two, "size {size}: two seeds produced one episode");
        }
    }

    /// The board printed in the module docs is the board this code draws.
    ///
    /// The deleted `build_places_door_key_and_box` pinned six coordinates that a
    /// sampled layout no longer has; its *documentation* value — "here is what an
    /// `UnlockPickup` board looks like" — moved into the module docs' ASCII
    /// snapshot. This keeps that snapshot honest: change the draws, the draw
    /// order, or the RNG, and the docs fail loudly instead of quietly going
    /// stale. If it fails, re-print the board and update both together.
    #[test]
    fn the_module_doc_snapshot_is_the_board_seed_11_draws() {
        const DOCUMENTED: &str = "\
# # # # # # #
# . . # . . #
# . . * . . #
# . . # . [ #
# . k # . . #
# < . # . . #
# # # # # # #";

        let mut env = UnlockPickupEnv::with_config(UnlockPickupConfig::new(7, 196, 0), false)
            .expect("valid config");
        env.reset_with_seed(11).expect("reset");
        let drawn = env
            .ascii()
            .lines()
            .map(str::trim_end)
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            drawn, DOCUMENTED,
            "module docs are stale; drawn board:\n{drawn}"
        );

        // The annotations beside the board carry the colours the renderer drops.
        assert_eq!(env.door_pos(), (3, 2));
        assert_eq!(env.door_color(), Color::Yellow);
        assert_eq!(env.target(), Entity::Box(Color::Green));
        assert_eq!(key_cell(&env.state().grid), ((2, 4), Color::Yellow));
        assert_eq!(box_cell(&env.state().grid), ((5, 3), Color::Green));
        assert_eq!(
            (
                env.state().agent.x,
                env.state().agent.y,
                env.state().agent.direction
            ),
            (1, 5, Direction::West)
        );
    }

    // -- dynamics -----------------------------------------------------------

    /// Breadth-first search for a shortest action sequence from `state` to the
    /// first board satisfying `solved`.
    ///
    /// A miniature of `tests/common/mod.rs`'s planner, kept here because a
    /// randomized layout has no fixed action script to assert against: the plan
    /// has to be derived from the board the environment actually drew.
    /// `GridAction::Done` is omitted — it never changes the board, so it can
    /// never shorten a plan.
    fn plan(state: &GridState, solved: &dyn Fn(&AgentState) -> bool) -> Option<Vec<GridAction>> {
        const MOVES: [GridAction; 6] = [
            GridAction::Forward,
            GridAction::TurnLeft,
            GridAction::TurnRight,
            GridAction::Pickup,
            GridAction::Drop,
            GridAction::Toggle,
        ];

        /// Hashable snapshot of everything `apply_action` can mutate.
        fn key(
            grid: &Grid,
            agent: &AgentState,
        ) -> (i32, i32, Direction, Option<Entity>, Vec<Entity>) {
            let mut cells = Vec::with_capacity(grid.width() * grid.height());
            for y in 0..as_i32(grid.height()) {
                for x in 0..as_i32(grid.width()) {
                    cells.push(grid.get(x, y));
                }
            }
            (agent.x, agent.y, agent.direction, agent.carrying, cells)
        }

        if solved(&state.agent) {
            return Some(Vec::new());
        }
        // (grid, agent, parent index, action that reached it)
        let mut nodes = vec![(
            state.grid.clone(),
            state.agent,
            usize::MAX,
            GridAction::Done,
        )];
        let mut seen = HashSet::from([key(&state.grid, &state.agent)]);
        let mut frontier = VecDeque::from([0usize]);

        while let Some(idx) = frontier.pop_front() {
            for action in MOVES {
                let mut grid = nodes[idx].0.clone();
                let mut agent = nodes[idx].1;
                apply_action(&mut grid, &mut agent, action);
                if !seen.insert(key(&grid, &agent)) {
                    continue;
                }
                let done = solved(&agent);
                nodes.push((grid, agent, idx, action));
                let next = nodes.len() - 1;
                if done {
                    let mut out = Vec::new();
                    let mut at = next;
                    while nodes[at].2 != usize::MAX {
                        out.push(nodes[at].3);
                        at = nodes[at].2;
                    }
                    out.reverse();
                    return Some(out);
                }
                frontier.push_back(next);
            }
        }
        None
    }

    /// Drive `plan` through the public `step` API and return the last snapshot.
    fn run(env: &mut UnlockPickupEnv, plan: &[GridAction]) -> GridSnapshot {
        let mut last = None;
        for &action in plan {
            last = Some(env.step(action).expect("step"));
        }
        last.expect("a non-empty plan")
    }

    /// End-to-end replacement for the old fixed-script rollout, which pinned
    /// coordinates a sampled layout no longer has.
    ///
    /// It also carries what `picking_up_wrong_object_does_not_terminate` used to
    /// assert: the key is a pickable non-target, and taking it must leave the
    /// episode running with reward `0.0`.
    #[test]
    fn a_planned_rollout_takes_the_key_then_the_box() {
        for size in SIZES {
            let mut env = env_of(size, 3);
            env.reset().expect("reset");

            let key = Entity::Key(env.door_color());
            let to_key = plan(env.state(), &|agent| agent.carrying == Some(key))
                .unwrap_or_else(|| panic!("size {size}: no route to the key:\n{}", env.ascii()));
            let snap = run(&mut env, &to_key);
            assert!(
                !snap.is_done(),
                "size {size}: picking up the key is not the mission"
            );
            assert_eq!(f32::from(*snap.reward()), 0.0);
            assert_eq!(env.state().agent.carrying, Some(key));

            let target = env.target();
            let to_box = plan(env.state(), &|agent| agent.carrying == Some(target))
                .unwrap_or_else(|| panic!("size {size}: no route to the box:\n{}", env.ascii()));
            let snap = run(&mut env, &to_box);
            assert!(
                snap.is_done(),
                "size {size}: carrying the box must terminate"
            );
            let reward = f32::from(*snap.reward());
            assert!(reward > 0.0, "size {size}: reward was {reward}");
            assert_eq!(env.state().agent.carrying, Some(target));
        }
    }

    #[test]
    fn timeout_terminates_with_zero_reward() {
        let cfg = UnlockPickupConfig::new(7, 3, 0);
        let mut env = UnlockPickupEnv::with_config(cfg, false).expect("valid config");
        env.reset().unwrap();
        env.step(GridAction::TurnLeft).unwrap();
        env.step(GridAction::TurnLeft).unwrap();
        let snap = env.step(GridAction::TurnLeft).unwrap();
        assert!(snap.is_done());
        let reward: f32 = (*snap.reward()).into();
        assert_eq!(reward, 0.0);
    }

    #[test]
    fn reset_clears_episode_state() {
        let mut env = env_7x7();
        env.reset().unwrap();
        env.step(GridAction::TurnLeft).unwrap();
        env.step(GridAction::Pickup).unwrap();
        assert_eq!(env.steps(), 2);
        env.reset().unwrap();
        assert_eq!(env.state().agent.carrying, None);
        assert_eq!(env.steps(), 0);
    }
}
