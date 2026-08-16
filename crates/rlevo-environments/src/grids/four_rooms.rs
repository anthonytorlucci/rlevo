//! Navigate a four-quadrant maze to reach the goal.
//!
//! Ports Farama Minigrid's [`FourRoomsEnv`] (`MiniGrid-FourRooms-v0`). An
//! interior cross of walls splits the `N×N` grid into four equal quadrants. The
//! **partition is fixed**; everything else is drawn fresh every episode:
//!
//! - one opening per wall segment — the cross has four segments (the upper and
//!   lower halves of the centre column, the left and right halves of the centre
//!   row), and each gets exactly one gap sampled uniformly from *its own*
//!   interior cells;
//! - the agent's start cell **and** its facing, uniform over free cells and the
//!   four directions;
//! - the goal cell, uniform over the free cells other than the agent's.
//!
//! One opening *per segment* — rather than four openings scattered anywhere on
//! the cross — is what makes every quadrant reachable from every other. Two gaps
//! in one segment and none in another would strand a quadrant.
//!
//! The grid size must be **odd** and at least 11 so each quadrant has enough
//! interior cells to navigate.
//!
//! ## Layout — one episode of many (`size = 11`, `seed = 484`)
//!
//! Literal [`FourRoomsEnv::ascii`] output, not a schematic, and **not** a
//! property of the environment: it is what `seed = 484` happens to draw. The
//! next `reset` of the same environment gives a different board.
//!
//! ```text
//! # # # # # # # # # # #
//! # G . . . # . . . . #   G = goal, drawn at (1, 1)
//! # . . . . # . . . . #
//! # . . . . . . . . . #   upper-column opening at (5, 3)
//! # . . . . # . < . . #   < = agent at (7, 4), facing West
//! # # # . # # # # . # #   row openings at (3, 5) and (8, 5)
//! # . . . . . . . . . #   lower-column opening at (5, 6)
//! # . . . . # . . . . #
//! # . . . . # . . . . #
//! # . . . . # . . . . #
//! # # # # # # # # # # #
//! ```
//!
//! - `<`, `>`, `^`, `v` — the agent, glyph showing its facing
//! - `G` — goal
//! - `.` — free cell; the four cross openings read as free cells
//! - `#` — border or interior wall
//!
//! The four openings above sit at `y = 3` and `y = 6` on column `5`, and at
//! `x = 3` and `x = 8` on row `5` — one in each of the cross's four segments,
//! and none of them at the old fixed `$\text{mid} \pm 3$`. Reading them back:
//!
//! ```rust
//! use rlevo_environments::grids::four_rooms::{FourRoomsConfig, FourRoomsEnv};
//!
//! let env = FourRoomsEnv::with_config(FourRoomsConfig::new(11, 484, 484), false)
//!     .expect("valid config");
//! assert_eq!(*env.openings(), [(5, 3), (3, 5), (8, 5), (5, 6)]);
//! assert_eq!((env.state().agent.x, env.state().agent.y), (7, 4));
//! ```
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
//! use rlevo_environments::grids::four_rooms::{FourRoomsConfig, FourRoomsEnv};
//! use rlevo_core::environment::{ConstructableEnv, Environment};
//!
//! let cfg = FourRoomsConfig::new(11, 484, 0);
//! let mut env = FourRoomsEnv::with_config(cfg, false).expect("valid config");
//! let _snapshot = env.reset().unwrap();
//! ```
//!
//! [`FourRoomsEnv`]: https://minigrid.farama.org/environments/minigrid/FourRoomsEnv/

use super::core::{
    GridSnapshot, Visibility,
    action::GridAction,
    build_snapshot,
    dynamics::{StepOutcome, apply_action},
    entity::Entity,
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
use rlevo_core::config::{self, ConfigError, ConstraintKind, Validate};
use rlevo_core::environment::{
    ConstructableEnv, Environment, EnvironmentError, Sensor, Snapshot as _,
};
use rlevo_core::reward::ScalarReward;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

/// Minimum side length; we need at least three interior cells per quadrant.
///
/// `11` is an `rlevo` convention, not a value inherited from upstream: Sutton,
/// Precup & Singh (1999) do not parameterize four-rooms by size at all, and
/// Farama Minigrid `v3.1.0` hardcodes `self.size = 19` with no `size`
/// constructor argument. It is deliberately *above* the smallest geometrically
/// clean size — every segment's opening range is non-empty and lands inside the
/// border from `5` up — so changing it is a policy decision, not a bug fix.
const MIN_SIZE: usize = 11;

/// Number of openings punched into the interior cross: exactly one per wall
/// segment, and the cross has four segments.
pub const OPENING_COUNT: usize = 4;

// The invariants `build` silently assumes, restated for sampled offsets (ADR
// 0062). Until #282 the four openings sat at a fixed `$\text{mid} \pm 3$` and this block
// asserted that `$\text{mid} \pm 3$` lands strictly inside the border — measured before the
// guard existed, `size = 7` perforated all four perimeter walls (holes at (0,3),
// (3,0), (3,6), (6,3)) and `size = 8` perforated two, at (4,7) and (7,4).
//
// The offsets are now drawn per episode, so that particular arithmetic is gone,
// but the guard's job is not: each of the four segments is sampled from its own
// half-open range, and an *empty* range panics inside `random_range` rather than
// producing a bad board. These assertions state at compile time that `MIN_SIZE`
// keeps all four ranges non-empty, so lowering `MIN_SIZE` breaks the build
// instead of the board — the same contract the `$\pm 3$` version carried. Do not
// delete this block; restate it if the ranges change.
//
// Each assertion below is literally "this half-open range is non-empty", i.e.
// `start < end` for the range `build` hands to `random_range`:
//
//   upper / left segments:  `1 .. mid`             non-empty ⟺ 1 < mid
//   right / lower segments: `mid + 1 .. size - 1`  non-empty ⟺ mid + 1 < size - 1
const _: () = assert!(MIN_SIZE % 2 == 1, "MIN_SIZE must be odd");
const _: () = assert!(
    1 < MIN_SIZE / 2,
    "at MIN_SIZE the upper/left opening range `1..mid` must be non-empty"
);
const _: () = assert!(
    MIN_SIZE / 2 + 1 < MIN_SIZE - 1,
    "at MIN_SIZE the right/lower opening range `mid+1..size-1` must be non-empty"
);

/// Rejection text for an even `size`. Unlike the floor, the parity requirement
/// has no `at_least` analogue, so it is a `Custom` invariant.
const SIZE_NOT_ODD: &str =
    "FourRoomsEnv requires an odd size so the interior cross has a single centre row and column";

/// Configuration for [`FourRoomsEnv`].
///
/// Controls the grid size, episode length, and RNG seed. The interior cross is
/// derived from `size`; its four openings, the agent's pose, and the goal are
/// drawn per episode, so no positional fields are needed.
///
/// # Examples
///
/// ```rust
/// use rlevo_environments::grids::four_rooms::FourRoomsConfig;
///
/// let cfg = FourRoomsConfig::new(11, 484, 0);
/// assert_eq!(cfg.size, 11);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FourRoomsConfig {
    /// Side length of the square grid in cells, including border walls.
    ///
    /// Must be **odd** and at least `11`. The interior cross sits at column and
    /// row `mid = size / 2`, splitting the board into four equal quadrants. Each
    /// of the cross's four wall segments carries exactly one opening, drawn per
    /// episode from that segment's own interior cells — `1..mid` for the upper
    /// and left segments, `mid + 1..size - 1` for the lower and right ones — so
    /// no opening position is a closed form of `size`. Read the current
    /// episode's four back with [`FourRoomsEnv::openings`].
    pub size: usize,
    /// Maximum number of steps before the episode is truncated.
    pub max_steps: usize,
    /// Seed for the environment's persistent RNG, drawn once at construction.
    ///
    /// The RNG **advances** across resets (ADR 0029), so successive episodes get
    /// independent opening positions, agent poses, and goal cells. A fixed seed
    /// therefore reproduces a fixed *sequence* of episodes, not one repeated
    /// episode. For bit-for-bit replay of a single episode use
    /// [`FourRoomsEnv::reset_with_seed`].
    ///
    /// The draws reproduce those of upstream `MiniGrid-FourRooms-v0`'s
    /// `_gen_grid` — four segment openings, then `place_agent`, then
    /// `place_obj(Goal())` — in that order (ADR 0062 §1). The registered id
    /// passes neither `agent_pos` nor `goal_pos`, so both are randomized here
    /// too. The one deliberate deviation is shared by the whole family:
    /// placement materializes the free-cell set and draws a uniform index rather
    /// than running upstream's unbounded rejection loop, which is the same
    /// distribution without a non-terminating failure mode (ADR 0062 §3b).
    pub seed: u64,
}

impl FourRoomsConfig {
    /// Constructs a `FourRoomsConfig` with explicit field values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rlevo_environments::grids::four_rooms::FourRoomsConfig;
    ///
    /// let cfg = FourRoomsConfig::new(11, 484, 7);
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

impl Default for FourRoomsConfig {
    fn default() -> Self {
        let size = 11;
        Self {
            size,
            max_steps: 4 * size * size,
            seed: 0,
        }
    }
}

impl Validate for FourRoomsConfig {
    /// Rejects any `size` below `MIN_SIZE` (11), an even `size`, and a zero
    /// `max_steps`.
    ///
    /// Both `size` guards live **here**, not only in [`FromStr`]:
    /// `FourRoomsConfig` derives `Deserialize`, so a config loaded from a file is
    /// user-supplied runtime data that never passes through `from_str`
    /// (rules.md §4 — "if an invalid value can arrive via `Deserialize`, it must
    /// be an `Err`"). Below the floor the builder has no honest board to make:
    /// at `size < 5` a segment's opening range (`1..mid`, `mid + 1..size - 1`)
    /// is empty and `random_range` panics on it, and at `size` 5 through 9 the
    /// quadrants shrink past the "at least three interior cells" premise
    /// `MIN_SIZE` encodes. (Before #282 the openings were fixed at `$\text{mid} \pm 3$`;
    /// the same floor then bought a different failure — `size` 1 through 6
    /// panicked inside `Grid::set`, and `size` 7 and 8 punched their openings
    /// through the *perimeter* wall. The measurement is historical, the guard is
    /// not.) Enforcing parity anywhere but here would recreate, for one env,
    /// exactly the split enforcement this guard removes.
    ///
    /// `at_least` subsumes the previous `nonzero` check — `MIN_SIZE >= 1`, so a
    /// zero `size` is still rejected, now as
    /// [`ConstraintKind::TooSmall`].
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "FourRoomsConfig";
        config::at_least(C, "size", self.size, MIN_SIZE)?;
        if self.size.is_multiple_of(2) {
            return Err(ConfigError {
                config: C,
                field: "size",
                kind: ConstraintKind::Custom(SIZE_NOT_ODD),
            });
        }
        config::nonzero(C, "max_steps", self.max_steps)?;
        Ok(())
    }
}

impl FromStr for FourRoomsConfig {
    type Err = String;

    /// Parses `"size=11,max_steps=484,seed=0"` (keys in any order) or the
    /// positional form `"11,484,0"`.
    ///
    /// # Errors
    ///
    /// Returns the offending key/value, or the [`Validate`] rejection — the same
    /// guard [`FourRoomsEnv::with_config`] applies, so this parser cannot admit a
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

/// Four-quadrant maze environment requiring multi-room navigation.
///
/// Implements [`Environment<3, 3, 1>`] — observation and action spaces each
/// have three components, reward is a scalar.
///
/// The agent must generally transit two openings to reach the goal, making this
/// a classic long-horizon exploration task. The quadrant partition is fixed, but
/// the four openings, the agent's pose, and the goal are re-drawn every
/// [`reset`](Environment::reset), so a policy is evaluated on a *family* of
/// boards rather than on one memorized route.
///
/// Construct via [`FourRoomsEnv::with_config`] for full control or via
/// [`ConstructableEnv::new`] for default settings (size 11, seed 0).
///
/// Reaching the goal (or lava) ends the episode; a [`step`](Environment::step)
/// taken afterwards is rejected with
/// [`EnvironmentError::StepAfterEpisodeEnd`] — see the [`EpisodeGuard`] field.
///
/// # Examples
///
/// ```rust
/// use rlevo_environments::grids::four_rooms::{FourRoomsConfig, FourRoomsEnv};
/// use rlevo_core::environment::{ConstructableEnv, Environment};
///
/// let mut env = FourRoomsEnv::with_config(
///     FourRoomsConfig::new(11, 484, 0),
///     false,
/// )
/// .expect("valid config");
/// env.reset().unwrap();
/// ```
#[derive(Debug)]
pub struct FourRoomsEnv {
    state: GridState,
    config: FourRoomsConfig,
    steps: usize,
    render: bool,
    /// This episode's four cross openings, in the order [`FourRoomsEnv::openings`]
    /// documents.
    openings: [(i32, i32); OPENING_COUNT],
    rng: StdRng,
    /// Rejects a `step()` taken after the goal (or lava) ended the episode.
    ///
    /// Without it the goal cell stays `Entity::Goal` under the agent and
    /// `apply_action` re-classifies it on every visit, so a finished episode was
    /// not merely resumed — it was *farmable*. Measured on `size = 11`,
    /// `max_steps = 484`, seed 3 before this guard existed: the derived route
    /// reached the goal on step 17 and paid `success_reward(17, 484) = 0.968`;
    /// the next `step()` re-emitted a **`Running`** snapshot (the episode came
    /// back to life), five more steps walked the agent off the goal and back on,
    /// and step 23 paid `success_reward(23, 484) = 0.957` a second time. Because
    /// the payout is step-count-discounted rather than one-shot, each re-entry
    /// pays again at a slightly smaller discount — a two-step oscillation on and
    /// off the goal adds ≈0.95 to the episode return per cycle, for as many
    /// cycles as `max_steps` allows, while `steps` runs past the true episode
    /// length and the zero-reward `Running` snapshots in between enter the
    /// replay buffer as ordinary transitions.
    guard: EpisodeGuard,
}

impl FourRoomsEnv {
    /// Emission-model visibility policy: does this environment's agent see
    /// through opaque cells?
    ///
    /// The rlevo spelling of canonical Minigrid's `see_through_walls`
    /// constructor argument. Read only by this environment's [`Sensor`] impl,
    /// and an inherent const rather than a config field because it is part of
    /// the task definition, not a knob a caller tunes.
    ///
    /// [`Visibility::Occluded`], because canonical
    /// `minigrid/envs/fourrooms.py` **omits** `see_through_walls` entirely and
    /// so inherits `MiniGridEnv.__init__`'s `see_through_walls=False` default:
    /// upstream, occlusion is on unless an env opts out, and this one does not.
    /// The cross walls therefore hide the three rooms the agent is not in until
    /// it crosses an opening. See ADR 0063
    /// (`docs/adr/0063-grid-visibility-occlusion.md`) for the whole
    /// twelve-environment table.
    const VISIBILITY: Visibility = Visibility::Occluded;

    /// Constructs a `FourRoomsEnv` from an explicit configuration.
    ///
    /// Seeds the persistent RNG once from `config.seed` and immediately builds a
    /// first episode from it. Call [`Environment::reset`] before the first
    /// [`Environment::step`] to obtain the first observation; that reset samples
    /// a *fresh* board, advancing the stream.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rlevo_environments::grids::four_rooms::{FourRoomsConfig, FourRoomsEnv};
    ///
    /// let env = FourRoomsEnv::with_config(
    ///     FourRoomsConfig::new(11, 484, 0),
    ///     true, // render ASCII grid to stdout
    /// )
    /// .expect("valid config");
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `config` fails [`Validate`]: a `size` below
    /// `MIN_SIZE` (11), an even `size`, or a zero `max_steps`. This is the
    /// construction chokepoint (rules.md §4), so it also rejects a config that
    /// arrived by `Deserialize` or struct-update syntax without passing through
    /// [`FromStr`].
    ///
    /// Also returns a [`ConfigError`] if placing the agent or the goal exhausts
    /// the free cells — converted from [`PlacementError`] (ADR 0062 §3a). No
    /// `size` that passes [`Validate`] can reach that path, because a validated
    /// board has at least `4 × 4 × 4` free interior cells and only two are
    /// consumed; it is propagated rather than asserted so that the sampler owns
    /// the invariant.
    pub fn with_config(config: FourRoomsConfig, render: bool) -> Result<Self, ConfigError> {
        config.validate()?;
        let mut rng = StdRng::seed_from_u64(config.seed);
        let (state, openings) = Self::build(&config, &mut rng)?;
        Ok(Self {
            state,
            config,
            steps: 0,
            render,
            openings,
            rng,
            guard: EpisodeGuard::new(),
        })
    }

    /// Re-seed the persistent RNG to `seed`, then [`reset`](Environment::reset).
    ///
    /// Ordinary [`reset`](Environment::reset) advances the persistent stream so
    /// successive episodes get independent openings, agent poses, and goals
    /// (ADR 0029); use this when you need a *specific* episode to reproduce
    /// bit-for-bit (e.g. replaying a failure). Run-level reproducibility is
    /// already guaranteed by the construction seed.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`reset`](Environment::reset).
    pub fn reset_with_seed(&mut self, seed: u64) -> Result<GridSnapshot, EnvironmentError> {
        self.rng = StdRng::seed_from_u64(seed);
        self.reset()
    }

    /// Returns a reference to the active configuration.
    #[must_use]
    pub const fn config(&self) -> &FourRoomsConfig {
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

    /// This episode's four cross openings as `(x, y)`, one per wall segment.
    ///
    /// The order is the order they are drawn in, which is upstream's
    /// `for j in 0..2 { for i in 0..2 { … } }` order:
    ///
    /// | index | segment | always satisfies |
    /// |---|---|---|
    /// | `0` | upper half of the centre column | `x == mid`, `1 <= y <= mid - 1` |
    /// | `1` | left half of the centre row | `y == mid`, `1 <= x <= mid - 1` |
    /// | `2` | right half of the centre row | `y == mid`, `mid + 1 <= x <= size - 2` |
    /// | `3` | lower half of the centre column | `x == mid`, `mid + 1 <= y <= size - 2` |
    ///
    /// One opening per *segment* — not four openings anywhere on the cross — is
    /// what makes the four quadrants mutually reachable, so this accessor is the
    /// honest way for a test or a planner to check connectivity without
    /// re-deriving positions from `size` (they are not a function of `size`).
    ///
    /// Re-drawn on every [`reset`](Environment::reset). The cells themselves
    /// read back as passable, but not necessarily as [`Entity::Empty`]: the goal
    /// may land in a doorway, exactly as upstream allows.
    #[must_use]
    pub const fn openings(&self) -> &[(i32, i32); OPENING_COUNT] {
        &self.openings
    }

    /// Renders the current grid as an ASCII string.
    ///
    /// Useful for debugging; the same output is printed to stdout on each step
    /// when the environment was constructed with `render = true`.
    #[must_use]
    pub fn ascii(&self) -> String {
        render_ascii(&self.state.grid, &self.state.agent)
    }

    /// Build a fresh episode: the fixed quadrant cross, one sampled opening per
    /// wall segment, a sampled agent pose, and a sampled goal.
    ///
    /// Draws from `rng` and lets it advance (ADR 0029) — never re-seeds. The
    /// draw *order* is part of the algorithm (ADR 0062 §1) and mirrors upstream
    /// `FourRoomsEnv._gen_grid`: upper-column opening, left-row opening,
    /// right-row opening, lower-column opening, then `place_agent` (position
    /// then facing), then `place_obj(Goal())`.
    ///
    /// Python's `_rand_int(a, b)` excludes `b`, so each `_rand_int` maps to the
    /// half-open Rust range with the same endpoints and no `+ 1` correction.
    ///
    /// # Errors
    ///
    /// Returns [`PlacementError::Exhausted`] if no free cell remains for the
    /// agent or the goal. Unreachable for a [`Validate`]-accepted config, but
    /// propagated rather than asserted (ADR 0062 §3a).
    fn build<R: Rng + ?Sized>(
        config: &FourRoomsConfig,
        rng: &mut R,
    ) -> Result<(GridState, [(i32, i32); OPENING_COUNT]), PlacementError> {
        let mut grid = Grid::new(config.size, config.size);
        grid.draw_walls();
        let whole = Rect::whole_grid(&grid);

        #[allow(clippy::cast_possible_wrap)]
        let size = config.size as i32;
        let mid = size / 2;

        // Interior cross of walls. Upstream draws this as four per-quadrant wall
        // segments interleaved with the openings; drawing the whole cross first
        // and punching afterwards yields the identical board, because no segment
        // extends past the perimeter and no opening range covers the centre.
        for y in 1..size - 1 {
            grid.set(mid, y, Entity::Wall);
        }
        for x in 1..size - 1 {
            grid.set(x, mid, Entity::Wall);
        }

        // One opening per segment, each drawn from that segment's own interior
        // cells — this, not the count, is what keeps all four quadrants mutually
        // reachable. The centre `(mid, mid)` lies in no range, so it is always
        // wall, and the four gaps are therefore always four distinct cells.
        let upper_y = rng.random_range(1..mid);
        let left_x = rng.random_range(1..mid);
        let right_x = rng.random_range(mid + 1..size - 1);
        let lower_y = rng.random_range(mid + 1..size - 1);
        let openings = [
            (mid, upper_y),
            (left_x, mid),
            (right_x, mid),
            (mid, lower_y),
        ];
        for &(x, y) in &openings {
            grid.set(x, y, Entity::Empty);
        }

        // `MiniGrid-FourRooms-v0` registers neither `agent_pos` nor `goal_pos`,
        // so upstream falls through to `place_agent()` / `place_obj(Goal())`,
        // both of which reject-sample over the whole board. Perimeter and cross
        // cells are walls, so `is_free` filters them out for us.
        let agent = place_agent(&grid, rng, whole, &mut no_reject)?;
        // `place_obj` refuses `self.agent_pos`; the shared sampler cannot see the
        // agent (it is not a grid cell), so the veto is spelled out here.
        place_obj(&mut grid, rng, Entity::Goal, whole, &mut |_grid, x, y| {
            (x, y) == (agent.x, agent.y)
        })?;

        Ok((GridState::new(grid, agent), openings))
    }

    fn emit(&self, observation: GridObservation, reward: f32, done: bool) -> GridSnapshot {
        if self.render {
            println!("{}", self.ascii());
        }
        build_snapshot(observation, reward, done)
    }
}

impl crate::render::AsciiRenderable for FourRoomsEnv {
    fn render_ascii(&self) -> String {
        render_ascii(&self.state.grid, &self.state.agent)
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        super::core::render::render_styled(&self.state.grid, &self.state.agent)
    }
}

impl Display for FourRoomsEnv {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FourRoomsEnv(size={}, step={}/{})",
            self.config.size, self.steps, self.config.max_steps
        )
    }
}

impl ConstructableEnv for FourRoomsEnv {
    fn new(render: bool) -> Self {
        Self::with_config(FourRoomsConfig::default(), render)
            .expect("default config must validate and its 11x11 board must place agent and goal")
    }
}

impl Sensor<3, 1, 3> for FourRoomsEnv {
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

impl Environment<3, 3, 1> for FourRoomsEnv {
    type StateType = GridState;
    type ObservationType = super::core::GridObservation;
    type ActionType = GridAction;
    type RewardType = ScalarReward;
    type SnapshotType = GridSnapshot;

    /// Samples a fresh board — four openings, agent pose, goal — from the
    /// persistent RNG, letting the stream advance (ADR 0029). Never re-seeds;
    /// [`FourRoomsEnv::reset_with_seed`] is the replay hatch.
    ///
    /// Re-opens the [`EpisodeGuard`], so an episode ended at the goal (or in
    /// lava) becomes steppable again.
    ///
    /// # Errors
    ///
    /// Propagates [`PlacementError`] from [`FourRoomsEnv::build`] as an
    /// [`EnvironmentError`]. On that path the guard is deliberately left
    /// **latched**: the sole fallible call runs first, and the guard is cleared
    /// only once a whole new episode is in hand. Clearing it first would re-open
    /// a finished episode over the *old* board — the environment would happily
    /// step a state it never returned to its initial condition (ADR 0044 §6,
    /// the same rule `TimeLimit::reset` follows).
    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        let (state, openings) = Self::build(&self.config, &mut self.rng)?;
        self.guard.reset();
        self.state = state;
        self.openings = openings;
        self.steps = 0;
        let observation = self.observe_reset(&self.state);
        Ok(self.emit(observation, 0.0, false))
    }

    /// # Errors
    ///
    /// Returns [`EnvironmentError::StepAfterEpisodeEnd`] if the episode has
    /// already ended; call [`reset`](Environment::reset) first.
    fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> {
        // Guard first: before `steps` advances, before `apply_action` touches
        // the grid or the agent pose, and before any RNG draw. A rejected call
        // must leave the board, the counter, and the stream untouched, or the
        // stream would depend on how many illegal steps a caller made (ADR 0029).
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
        // Single exit: exactly one snapshot is built, and the guard is fed that
        // snapshot's own status, so no branch can forget to record.
        let observation = self.observe(&action, &self.state);
        let snapshot = self.emit(observation, reward, done);
        self.guard.record(snapshot.status());
        Ok(snapshot)
    }
}

impl rlevo_core::render::payload::GridPayloadSource for FourRoomsEnv {
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

    // `ConstraintKind` arrives via `super::*` — this module's `Validate` impl
    // needs it at file scope for the parity `Custom` variant.
    use super::*;
    use crate::direction::Direction;
    use crate::episode::assert_rejects_post_terminal_step;
    use crate::grids::core::UNSEEN_TYPE;
    use crate::grids::core::agent::AgentState;
    use rlevo_core::environment::Snapshot;
    use std::collections::{HashSet, VecDeque};

    /// How many distinct episodes every seed loop must clear.
    ///
    /// Matches `ORACLE_SEEDS` in `tests/grids_solvable.rs`. Each opening range
    /// has `mid - 1` values (4 at `MIN_SIZE`), so 32 seeds exercises every value
    /// of every range many times over rather than sampling a corner of it.
    const SEEDS: u64 = 32;

    /// Every seed loop runs at the `MIN_SIZE` floor **and** at one larger board.
    ///
    /// `MIN_SIZE` is where the sampling ranges are narrowest (`1..5` and `6..10`)
    /// and where an off-by-one in a segment's bounds would first punch through
    /// the perimeter. Both values must stay odd — `FourRoomsConfig::validate`
    /// requires it.
    const SIZES: [usize; 2] = [MIN_SIZE, 13];

    fn env_11() -> FourRoomsEnv {
        FourRoomsEnv::with_config(FourRoomsConfig::new(11, 400, 0), false).expect("valid config")
    }

    /// An environment of side `size` whose stream starts at `seed`.
    fn env_at(size: usize, seed: u64) -> FourRoomsEnv {
        FourRoomsEnv::with_config(FourRoomsConfig::new(size, 4 * size * size, seed), false)
            .expect("valid config")
    }

    /// The single [`Entity::Goal`] cell of `grid`.
    ///
    /// Panics unless there is exactly one, which is itself an assertion every
    /// caller wants.
    fn goal_of(grid: &Grid) -> (i32, i32) {
        let mut found = Vec::new();
        for y in 0..i32::try_from(grid.height()).expect("size fits in i32") {
            for x in 0..i32::try_from(grid.width()).expect("size fits in i32") {
                if grid.get(x, y) == Entity::Goal {
                    found.push((x, y));
                }
            }
        }
        assert_eq!(found.len(), 1, "expected exactly one goal, found {found:?}");
        found[0]
    }

    /// Everything the layout draws, in one comparable value.
    ///
    /// Used only by the reset-semantics tests, which are about *whether* two
    /// episodes agree. The non-degeneracy tests deliberately do **not** use it:
    /// a whole-layout diff passes as soon as any single quantity moves, which is
    /// exactly how a still-pinned agent direction hides behind a moving goal
    /// (ADR 0062, #282).
    type Layout = (
        [(i32, i32); OPENING_COUNT],
        (i32, i32, Direction),
        (i32, i32),
    );

    fn layout_of(env: &FourRoomsEnv) -> Layout {
        let agent = &env.state().agent;
        (
            *env.openings(),
            (agent.x, agent.y, agent.direction),
            goal_of(&env.state().grid),
        )
    }

    /// Every cell reachable from `start` by 4-connected moves over passable
    /// cells.
    ///
    /// Deliberately coarser than the environment's own dynamics: it ignores
    /// facing and turning, so it answers "is this board connected", not "can the
    /// agent get there". Turning is always available and never blocked, so the
    /// two agree on reachability.
    fn reachable(grid: &Grid, start: (i32, i32)) -> HashSet<(i32, i32)> {
        let mut seen = HashSet::from([start]);
        let mut queue = VecDeque::from([start]);
        while let Some((x, y)) = queue.pop_front() {
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let next = (x + dx, y + dy);
                if grid.in_bounds(next.0, next.1)
                    && grid.get(next.0, next.1).is_passable()
                    && seen.insert(next)
                {
                    queue.push_back(next);
                }
            }
        }
        seen
    }

    /// The interior cells of the four quadrants, as four `(x, y)` lists.
    ///
    /// A quadrant's interior excludes the border and the cross, so every cell
    /// listed is guaranteed passable on a well-formed board.
    fn quadrant_interiors(size: i32) -> [Vec<(i32, i32)>; 4] {
        let mid = size / 2;
        let low = 1..mid;
        let high = mid + 1..size - 1;
        let cells = |xs: std::ops::Range<i32>, ys: std::ops::Range<i32>| -> Vec<(i32, i32)> {
            ys.flat_map(|y| xs.clone().map(move |x| (x, y))).collect()
        };
        [
            cells(low.clone(), low.clone()),
            cells(high.clone(), low.clone()),
            cells(low, high.clone()),
            cells(high.clone(), high),
        ]
    }

    #[test]
    fn default_config_validates() {
        assert!(FourRoomsConfig::default().validate().is_ok());
    }

    #[test]
    fn rejects_zero_size() {
        let bad = FourRoomsConfig {
            size: 0,
            ..Default::default()
        };
        assert!(FourRoomsEnv::with_config(bad, false).is_err());
    }

    #[test]
    fn default_config_is_11x11_odd() {
        let cfg = FourRoomsConfig::default();
        assert_eq!(cfg.size, 11);
        assert_eq!(cfg.size % 2, 1);
    }

    #[test]
    fn fromstr_rejects_even_size() {
        assert!("10".parse::<FourRoomsConfig>().is_err());
    }

    #[test]
    fn fromstr_rejects_small_size() {
        assert!("9".parse::<FourRoomsConfig>().is_err());
    }

    /// Issue #106: `MIN_SIZE` was enforced only in [`FromStr`], so a config
    /// built by `Deserialize` or struct-update syntax reached `build`. Measured
    /// consequences under the then-fixed `$\text{mid} \pm 3$` openings: `size` 1–6 panicked
    /// in `Grid::set`; `size = 7` punched a hole in all four perimeter walls and
    /// `size = 8` in two. The openings are sampled now (#282) so those exact
    /// numbers are history, but the floor still guards a real cliff — below
    /// `size = 5` a segment's opening range is empty and `random_range` panics.
    /// The guard lives in [`Validate`], which `with_config` runs (ADR 0026
    /// chokepoint).
    #[test]
    fn with_config_rejects_size_below_min() {
        let bad = FourRoomsConfig {
            size: MIN_SIZE - 1,
            ..Default::default()
        };
        let err = FourRoomsEnv::with_config(bad, false).unwrap_err();
        assert_eq!(err.config, "FourRoomsConfig");
        assert_eq!(err.field, "size");
        assert_eq!(
            err.kind,
            ConstraintKind::TooSmall {
                min: MIN_SIZE as u64,
                got: (MIN_SIZE - 1) as u64,
            }
        );
    }

    /// `9` builds a geometrically clean board — every segment's opening range is
    /// non-empty and lands inside the border — and is still refused. The floor is
    /// `rlevo` policy for this env, not the smallest size the layout code
    /// survives.
    #[test]
    fn with_config_rejects_clean_but_sub_floor_size() {
        let bad = FourRoomsConfig {
            size: 9,
            ..Default::default()
        };
        let err = FourRoomsEnv::with_config(bad, false).unwrap_err();
        assert_eq!(err.field, "size");
        assert_eq!(
            err.kind,
            ConstraintKind::TooSmall {
                min: MIN_SIZE as u64,
                got: 9,
            }
        );
    }

    /// The parity rule the `size` field doc already published. It was enforced
    /// only in [`FromStr`] before #106, so `Deserialize` could hand `build` an
    /// even size whose interior cross has two centre rows.
    ///
    /// The value is `14`, not `MIN_SIZE + 1` (12), and must **not** be a
    /// multiple of 3. `12` is divisible by both 2 and 3, so it cannot tell the
    /// guard's modulus apart: mutating `is_multiple_of(2)` to
    /// `is_multiple_of(3)` in [`FourRoomsConfig::validate`] left this test — and
    /// the whole suite — green. `14` is even, above the floor, and `14 % 3 == 2`,
    /// so it pins the modulus as well as the parity intent. Do not "tidy" it back
    /// to `MIN_SIZE + 1`.
    #[test]
    fn with_config_rejects_even_size() {
        let bad = FourRoomsConfig {
            size: 14,
            ..Default::default()
        };
        let err = FourRoomsEnv::with_config(bad, false).unwrap_err();
        assert_eq!(err.config, "FourRoomsConfig");
        assert_eq!(err.field, "size");
        assert_eq!(err.kind, ConstraintKind::Custom(SIZE_NOT_ODD));
    }

    // --- Structure, connectivity, and non-degeneracy of the sampled layout ---
    //
    // `build_places_cross_with_four_openings` used to live here, asserting the
    // openings at literal `(5, 2)`, `(5, 8)`, `(2, 5)`, `(8, 5)` and the goal at
    // `(9, 9)`. It was the strictest layout lock in the grid family, and #282
    // moved every one of those cells into a draw. What replaces it is the set of
    // properties that lock actually implied — plus connectivity, which it never
    // checked. A worked example of the geometry it used to pin now lives in the
    // module docs as literal `ascii()` output.

    /// The hand trace this file is written against, kept as a comment because it
    /// is the thing a reader needs in order to check the ranges below.
    ///
    /// Upstream `FourRoomsEnv._gen_grid` at `size = 11` — `room_w = room_h = 5`,
    /// `mid = 5` — runs `for j in 0..2 { for i in 0..2 { … } }`:
    ///
    /// | `(i, j)` | `xL, yT, xR, yB` | segment written | `_rand_int` | opening |
    /// |---|---|---|---|---|
    /// | `(0, 0)` | `0, 0, 5, 5` | `vert_wall(5, 0, 5)` → `(5, 0..=4)` | `(1, 5)` → `{1,2,3,4}` | `(5, y)` |
    /// | `(0, 0)` | " | `horz_wall(0, 5, 5)` → `(0..=4, 5)` | `(1, 5)` → `{1,2,3,4}` | `(x, 5)` |
    /// | `(1, 0)` | `5, 0, 10, 5` | `horz_wall(5, 5, 5)` → `(5..=9, 5)` | `(6, 10)` → `{6,7,8,9}` | `(x, 5)` |
    /// | `(0, 1)` | `0, 5, 5, 10` | `vert_wall(5, 5, 5)` → `(5, 5..=9)` | `(6, 10)` → `{6,7,8,9}` | `(5, y)` |
    /// | `(1, 1)` | `5, 5, 10, 10` | neither branch runs | — | — |
    ///
    /// `i + 1 < 2` gates the vertical wall, so only `i = 0` writes one; `j + 1 <
    /// 2` gates the horizontal wall, so only `j = 0` writes one. Four segments,
    /// four openings, **one each** — and the centre `(5, 5)` is in no range, so
    /// it is always wall. The perimeter cells `(5, 10)` and `(10, 5)` come from
    /// the border, which the segments never reach past.
    ///
    /// Python's `_rand_int(a, b)` excludes `b`, hence the half-open Rust ranges
    /// `1..mid` and `mid + 1..size - 1`.
    #[test]
    fn opening_ranges_match_the_hand_trace_at_size_11() {
        let mid = 5;
        for seed in 0..SEEDS {
            let env = env_at(11, seed);
            let [upper, left, right, lower] = *env.openings();
            assert_eq!(upper.0, mid, "seed {seed}: upper opening off the column");
            assert!(
                (1..mid).contains(&upper.1),
                "seed {seed}: upper opening {upper:?} outside `1..5`"
            );
            assert_eq!(left.1, mid, "seed {seed}: left opening off the row");
            assert!(
                (1..mid).contains(&left.0),
                "seed {seed}: left opening {left:?} outside `1..5`"
            );
            assert_eq!(right.1, mid, "seed {seed}: right opening off the row");
            assert!(
                (mid + 1..10).contains(&right.0),
                "seed {seed}: right opening {right:?} outside `6..10`"
            );
            assert_eq!(lower.0, mid, "seed {seed}: lower opening off the column");
            assert!(
                (mid + 1..10).contains(&lower.1),
                "seed {seed}: lower opening {lower:?} outside `6..10`"
            );
        }
    }

    /// Perimeter intact, cross solid except the four openings, exactly one goal,
    /// and the agent standing somewhere that is not the goal.
    #[test]
    fn every_sampled_board_is_structurally_well_formed() {
        for size in SIZES {
            let side = i32::try_from(size).expect("size fits in i32");
            let mid = side / 2;
            for seed in 0..SEEDS {
                let env = env_at(size, seed);
                let grid = &env.state().grid;
                let openings: HashSet<(i32, i32)> = env.openings().iter().copied().collect();
                assert_eq!(
                    openings.len(),
                    OPENING_COUNT,
                    "size {size} seed {seed}: two segments shared an opening cell"
                );

                for i in 0..side {
                    for &(x, y) in &[(i, 0), (i, side - 1), (0, i), (side - 1, i)] {
                        assert_eq!(
                            grid.get(x, y),
                            Entity::Wall,
                            "size {size} seed {seed}: perimeter hole at ({x}, {y})"
                        );
                    }
                }

                for i in 1..side - 1 {
                    for cell in [(mid, i), (i, mid)] {
                        if openings.contains(&cell) {
                            assert!(
                                grid.get(cell.0, cell.1).is_passable(),
                                "size {size} seed {seed}: opening {cell:?} is not passable"
                            );
                        } else {
                            assert_eq!(
                                grid.get(cell.0, cell.1),
                                Entity::Wall,
                                "size {size} seed {seed}: unsanctioned gap at {cell:?}"
                            );
                        }
                    }
                }

                // `goal_of` panics unless there is exactly one goal cell.
                let goal = goal_of(grid);
                let agent = &env.state().agent;
                assert_ne!(
                    (agent.x, agent.y),
                    goal,
                    "size {size} seed {seed}: the agent starts on the goal"
                );
            }
        }
    }

    /// The load-bearing invariant: one opening **per segment** keeps every
    /// quadrant reachable.
    ///
    /// "Four holes somewhere on the cross" satisfies every count-based assertion
    /// above and still strands a quadrant — two gaps in the upper column and
    /// none in the lower one seals the bottom-left room off from the bottom-right
    /// one. Flood-filling from the agent is what tells the two apart.
    #[test]
    fn every_quadrant_and_the_goal_are_reachable_from_the_agent() {
        for size in SIZES {
            let side = i32::try_from(size).expect("size fits in i32");
            for seed in 0..SEEDS {
                let env = env_at(size, seed);
                let grid = &env.state().grid;
                let agent = &env.state().agent;
                let seen = reachable(grid, (agent.x, agent.y));

                for (q, cells) in quadrant_interiors(side).iter().enumerate() {
                    for &cell in cells {
                        assert!(
                            seen.contains(&cell),
                            "size {size} seed {seed}: quadrant {q} cell {cell:?} is cut off \
                             from the agent at ({}, {}); openings {:?}",
                            agent.x,
                            agent.y,
                            env.openings()
                        );
                    }
                }
                assert!(
                    seen.contains(&goal_of(grid)),
                    "size {size} seed {seed}: the goal is unreachable"
                );
            }
        }
    }

    /// Each sampled quantity varies **on its own**.
    ///
    /// Asserted per quantity rather than as one whole-layout diff: a diff over
    /// the full observation goes green the moment *anything* moves, so a still-
    /// pinned agent facing (or a fourth opening that never left `mid + 3`) hides
    /// behind the three quantities that do move. ADR 0062 and #282 both name
    /// that trap explicitly.
    #[test]
    fn each_sampled_quantity_varies_across_seeds() {
        for size in SIZES {
            let mut upper = HashSet::new();
            let mut left = HashSet::new();
            let mut right = HashSet::new();
            let mut lower = HashSet::new();
            let mut agent_pos = HashSet::new();
            let mut agent_dir = HashSet::new();
            let mut goal_pos = HashSet::new();

            for seed in 0..SEEDS {
                let env = env_at(size, seed);
                let o = *env.openings();
                upper.insert(o[0].1);
                left.insert(o[1].0);
                right.insert(o[2].0);
                lower.insert(o[3].1);
                let agent = &env.state().agent;
                agent_pos.insert((agent.x, agent.y));
                agent_dir.insert(agent.direction);
                goal_pos.insert(goal_of(&env.state().grid));
            }

            for (name, count) in [
                ("upper-column opening", upper.len()),
                ("left-row opening", left.len()),
                ("right-row opening", right.len()),
                ("lower-column opening", lower.len()),
                ("agent position", agent_pos.len()),
                ("agent direction", agent_dir.len()),
                ("goal position", goal_pos.len()),
            ] {
                assert!(
                    count > 1,
                    "size {size}: {name} took a single value across {SEEDS} seeds — it is pinned"
                );
            }
        }
    }

    #[test]
    fn reset_is_deterministic() {
        let cfg = FourRoomsConfig::new(11, 400, 5);
        let mut a = FourRoomsEnv::with_config(cfg, false).expect("valid config");
        let mut b = FourRoomsEnv::with_config(cfg, false).expect("valid config");
        let sa = a.reset().unwrap();
        let sb = b.reset().unwrap();
        assert_eq!(sa.observation(), sb.observation());
    }

    /// The persistent stream advances, so successive episodes are fresh boards.
    ///
    /// Stated as "more than one distinct layout in eight resets" rather than
    /// "reset 1 differs from reset 2": consecutive draws *may* collide, and a
    /// test that forbids a legitimate collision is a flake waiting for a seed
    /// change. A `reset()` that re-seeded would produce one layout eight times.
    #[test]
    fn consecutive_resets_sample_fresh_layouts() {
        for size in SIZES {
            let mut env = env_at(size, 0);
            let mut seen = HashSet::new();
            for _ in 0..8 {
                env.reset().expect("reset");
                seen.insert(layout_of(&env));
            }
            assert!(
                seen.len() > 1,
                "size {size}: eight consecutive resets produced {} distinct layout(s) — \
                 the stream is not advancing",
                seen.len()
            );
        }
    }

    /// `reset_with_seed` is the replay hatch: same seed, same episode, twice.
    #[test]
    fn reset_with_seed_replays_bit_for_bit() {
        for size in SIZES {
            let mut env = env_at(size, 0);
            let first = env.reset_with_seed(7).expect("reset");
            let first_layout = layout_of(&env);
            // Advance the stream so a re-seed is the only way back.
            env.reset().expect("reset");
            let second = env.reset_with_seed(7).expect("reset");
            assert_eq!(
                first.observation(),
                second.observation(),
                "size {size}: reset_with_seed(7) did not replay"
            );
            assert_eq!(first_layout, layout_of(&env));
        }
    }

    #[test]
    fn different_replay_seeds_give_different_layouts() {
        for size in SIZES {
            let mut env = env_at(size, 0);
            let mut seen = HashSet::new();
            for seed in 0..SEEDS {
                env.reset_with_seed(seed).expect("reset");
                seen.insert(layout_of(&env));
            }
            assert!(
                seen.len() > 1,
                "size {size}: {SEEDS} replay seeds all produced the same layout"
            );
        }
    }

    /// The interior cross blocks movement, on whatever board was drawn.
    ///
    /// Replaces the old fixed-board version, which walked East from `(1, 1)` and
    /// asserted the agent halted at `x == 4` — both of those are draws now. The
    /// property survives: walk forward until the agent stops moving, and the cell
    /// it is facing must be impassable.
    #[test]
    fn walls_block_forward_movement() {
        for size in SIZES {
            for seed in 0..SEEDS {
                let mut env = env_at(size, seed);
                env.reset_with_seed(seed).expect("reset");
                let mut verdict = None;
                for _ in 0..2 * size {
                    let before = (env.state().agent.x, env.state().agent.y);
                    let snap = env.step(GridAction::Forward).expect("step");
                    let agent = env.state().agent;
                    if snap.is_done() {
                        verdict = Some(true); // walked onto the goal; nothing to block
                        break;
                    }
                    if (agent.x, agent.y) == before {
                        let (dx, dy) = agent.direction.delta();
                        assert!(
                            !env.state()
                                .grid
                                .get(agent.x + dx, agent.y + dy)
                                .is_passable(),
                            "size {size} seed {seed}: the agent stopped at {before:?} facing \
                             {:?} with a passable cell ahead",
                            agent.direction
                        );
                        verdict = Some(true);
                        break;
                    }
                }
                assert!(
                    verdict.is_some(),
                    "size {size} seed {seed}: the agent walked {} cells without ever being \
                     blocked or finishing — the board is not bounded",
                    2 * size
                );
            }
        }
    }

    /// A shortest 4-connected cell route from `start` to `goal`, or `None`.
    ///
    /// The same BFS as [`reachable`] with parent links kept. Cell-level, not
    /// pose-level: turning is always legal and never blocked, so a cell route is
    /// always realizable as an action sequence — [`route_actions`] does that
    /// conversion.
    fn cell_route(grid: &Grid, start: (i32, i32), goal: (i32, i32)) -> Option<Vec<(i32, i32)>> {
        let mut parent = std::collections::HashMap::from([(start, start)]);
        let mut queue = VecDeque::from([start]);
        while let Some(cell) = queue.pop_front() {
            if cell == goal {
                let mut route = vec![goal];
                let mut at = goal;
                while at != start {
                    at = parent[&at];
                    route.push(at);
                }
                route.reverse();
                return Some(route);
            }
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let next = (cell.0 + dx, cell.1 + dy);
                if grid.in_bounds(next.0, next.1)
                    && grid.get(next.0, next.1).is_passable()
                    && !parent.contains_key(&next)
                {
                    parent.insert(next, cell);
                    queue.push_back(next);
                }
            }
        }
        None
    }

    /// Turn `route` into actions, starting from `facing`.
    ///
    /// At most two turns per hop (a reversal), then one `Forward`.
    fn route_actions(route: &[(i32, i32)], mut facing: Direction) -> Vec<GridAction> {
        let mut actions = Vec::new();
        for pair in route.windows(2) {
            let delta = (pair[1].0 - pair[0].0, pair[1].1 - pair[0].1);
            let target = [
                Direction::East,
                Direction::South,
                Direction::West,
                Direction::North,
            ]
            .into_iter()
            .find(|d| d.delta() == delta)
            .expect("consecutive route cells are 4-adjacent");
            while facing != target {
                if facing.right() == target {
                    actions.push(GridAction::TurnRight);
                    facing = facing.right();
                } else {
                    actions.push(GridAction::TurnLeft);
                    facing = facing.left();
                }
            }
            actions.push(GridAction::Forward);
        }
        actions
    }

    /// End-to-end through the public API: every sampled board is winnable, and
    /// the win pays.
    ///
    /// Replaces `optimal_rollout_through_two_openings_reaches_goal`, whose
    /// twenty-action script encoded the old fixed board's exact geometry. The
    /// route is now *derived* from the board the env built, so this test cannot
    /// be right by luck on one seed and silently wrong on the rest. The
    /// integration oracle in `tests/grids_solvable.rs` states the same property
    /// against a full pose-level planner; this one keeps it inside the module
    /// that owns the generator.
    #[test]
    fn a_derived_route_reaches_the_goal_and_pays() {
        for size in SIZES {
            for seed in 0..SEEDS {
                let mut env = env_at(size, seed);
                env.reset_with_seed(seed).expect("reset");
                let agent = env.state().agent;
                let goal = goal_of(&env.state().grid);
                let route = cell_route(&env.state().grid, (agent.x, agent.y), goal)
                    .unwrap_or_else(|| panic!("size {size} seed {seed}: no route to the goal"));
                let actions = route_actions(&route, agent.direction);
                assert!(
                    !actions.is_empty(),
                    "size {size} seed {seed}: the agent must not start on the goal"
                );

                let mut last = None;
                for action in actions {
                    last = Some(env.step(action).expect("step"));
                }
                let snap = last.expect("non-empty route");
                assert!(
                    snap.is_done(),
                    "size {size} seed {seed}: the derived route did not end the episode"
                );
                let reward: f32 = (*snap.reward()).into();
                assert!(
                    reward > 0.0,
                    "size {size} seed {seed}: reward was {reward}, expected > 0.0"
                );
            }
        }
    }

    #[test]
    fn timeout_ends_with_zero_reward() {
        let cfg = FourRoomsConfig::new(11, 3, 0);
        let mut env = FourRoomsEnv::with_config(cfg, false).expect("valid config");
        env.reset().unwrap();
        env.step(GridAction::TurnLeft).unwrap();
        env.step(GridAction::TurnLeft).unwrap();
        let snap = env.step(GridAction::TurnLeft).unwrap();
        assert!(snap.is_done());
        let reward: f32 = (*snap.reward()).into();
        assert_eq!(reward, 0.0);
    }

    #[test]
    fn reset_clears_step_counter() {
        let mut env = env_11();
        env.reset().unwrap();
        env.step(GridAction::TurnLeft).unwrap();
        env.step(GridAction::TurnLeft).unwrap();
        assert_eq!(env.steps(), 2);
        env.reset().unwrap();
        assert_eq!(env.steps(), 0);
    }

    // ── post-terminal step guard (ADR 0044) ───────────────────────────────────
    //
    // The guard is a real correctness fix, not step-count hygiene: without it,
    // a caller could `step()` again after a terminal snapshot without calling
    // `reset()`, and internal state would mutate against a stale,
    // already-terminal snapshot (see `dynamic_obstacles.rs` for a concrete
    // instance). `EpisodeGuard::check()?` as the first statement of `step()`
    // makes that invariant hold unconditionally rather than depending on
    // callers never stepping past a terminal snapshot.

    /// Drives `env` to a **goal** termination through real `step()` calls.
    ///
    /// Ends the episode by reaching the goal rather than by exhausting
    /// `max_steps`: the step-limit path currently maps a cutoff to `Terminated`
    /// rather than `Truncated` (`grids/core/mod.rs`, `build_snapshot`; GitHub
    /// #1028), and no guard test should be written against a status that a bug
    /// fix is expected to change. The route is derived from the board the env
    /// actually drew, so this does not encode one seed's geometry.
    fn drive_to_goal(env: &mut FourRoomsEnv) -> GridSnapshot {
        env.reset_with_seed(3).expect("reset must succeed");
        let agent = env.state().agent;
        let goal = goal_of(&env.state().grid);
        let route = cell_route(&env.state().grid, (agent.x, agent.y), goal)
            .expect("every sampled board is connected");
        let actions = route_actions(&route, agent.direction);
        assert!(
            !actions.is_empty(),
            "the agent must not start on the goal, or there is nothing to drive"
        );

        let mut last = None;
        for action in actions {
            last = Some(env.step(action).expect("every step en route must succeed"));
        }
        let snapshot = last.expect("the route is non-empty");
        assert!(
            snapshot.is_done(),
            "walking the derived route must end the episode at the goal"
        );
        snapshot
    }

    /// The shared post-terminal conformance check: once the agent has walked
    /// onto the goal, a further legal action fails with `StepAfterEpisodeEnd`
    /// carrying the status that ended the episode.
    ///
    /// `TurnLeft` is the replayed action precisely because it is always legal
    /// and never blocked — the rejection has to come from the call sequence, not
    /// from the action.
    #[test]
    fn test_four_rooms_rejects_post_terminal_step() {
        let mut env = env_11();
        assert_rejects_post_terminal_step(&mut env, drive_to_goal, GridAction::TurnLeft);
    }

    /// Regression for the measured defect: the goal cell keeps its
    /// `Entity::Goal` under the agent, so before the guard a post-terminal step
    /// re-emitted a **`Running`** snapshot and walking back onto the goal paid
    /// `success_reward` a second time (seed 3: `0.968` on step 17, then `0.957`
    /// on step 23). A rejected step must mutate nothing at all.
    #[test]
    fn test_four_rooms_post_terminal_step_does_not_resume_the_episode() {
        let mut env = env_11();
        let terminal = drive_to_goal(&mut env);
        let ended = terminal.status();

        let steps_at_end = env.steps();
        let agent_at_end = env.state().agent;
        let openings_at_end = *env.openings();

        let err = env
            .step(GridAction::Forward)
            .expect_err("a step after the goal must return Err, not a resurrected episode");
        match err {
            EnvironmentError::StepAfterEpisodeEnd { status } => assert_eq!(
                status, ended,
                "the error must carry the status that ended the episode"
            ),
            other => panic!("expected StepAfterEpisodeEnd, got {other:?}"),
        }

        assert_eq!(
            env.steps(),
            steps_at_end,
            "a rejected step must not advance the step counter"
        );
        assert_eq!(
            (
                env.state().agent.x,
                env.state().agent.y,
                env.state().agent.direction
            ),
            (agent_at_end.x, agent_at_end.y, agent_at_end.direction),
            "a rejected step must not move or turn the agent"
        );
        assert_eq!(
            *env.openings(),
            openings_at_end,
            "a rejected step must not redraw the board"
        );
        assert_eq!(
            env.guard.status(),
            ended,
            "a rejected step must not reopen the episode"
        );
    }

    /// `reset()` re-opens a terminated environment, so a latched guard cannot
    /// strand it for the rest of the run.
    #[test]
    fn test_four_rooms_reset_reopens_terminated_episode() {
        use rlevo_core::environment::EpisodeStatus;

        let mut env = env_11();
        drive_to_goal(&mut env);
        assert!(
            env.step(GridAction::TurnLeft).is_err(),
            "the episode has ended; a step must be rejected before reset()"
        );

        env.reset().expect("reset must succeed after termination");
        assert_eq!(
            env.guard.status(),
            EpisodeStatus::Running,
            "reset() must return the guard to Running"
        );

        let snap = env
            .step(GridAction::TurnLeft)
            .expect("reset() must re-open the environment for a new episode");
        assert!(
            !snap.is_done(),
            "a single turn on a fresh board must not end the episode"
        );
    }

    /// A rejected step draws no randomness. `step()` does not sample today, but
    /// `reset()` shares the persistent stream: checking the guard after any
    /// future draw would let illegal calls shift every subsequent episode's
    /// board and desynchronise replay (ADR 0029).
    #[test]
    fn test_four_rooms_rejected_step_does_not_advance_rng() {
        let mut rejected = env_11();
        let mut untouched = env_11();
        drive_to_goal(&mut rejected);
        drive_to_goal(&mut untouched);

        rejected
            .step(GridAction::TurnLeft)
            .expect_err("a step after the goal must be rejected");

        rejected.reset().expect("reset");
        untouched.reset().expect("reset");
        assert_eq!(
            layout_of(&rejected),
            layout_of(&untouched),
            "a rejected step must draw no randomness; the next episode must be the board it \
             would have been"
        );
    }

    /// Occlusion is not decoration here: a cell that encoded a real entity under
    /// the pre-#281 (see-through) emission model must now be able to encode as
    /// *unseen*.
    ///
    /// The interesting case is the goal, because that is task-relevant
    /// information the cross walls are supposed to withhold until the agent
    /// crosses an opening. Seed 8's layout puts the goal in a room the agent can
    /// stand five cells from and — under occlusion — still not see.
    ///
    /// Note how *weak* the occlusion turns out to be: the four openings let the
    /// flood fill spread sideways into the neighbouring rooms, so from most poses
    /// the goal stays visible. This test therefore names one concrete pose rather
    /// than asserting a general "walls hide the goal" property that does not
    /// hold.
    #[test]
    fn test_four_rooms_occlusion_hides_the_goal_from_an_adjacent_room() {
        let mut env = FourRoomsEnv::with_config(FourRoomsConfig::new(13, 100, 8), false)
            .expect("valid config");
        env.reset().expect("reset");

        // Agent in the NW room's SE corner, looking East across the wall column.
        let state = GridState::new(
            env.state().grid.clone(),
            AgentState::new(5, 5, Direction::East),
        );
        assert_eq!(
            env.state().grid.get(6, 8),
            Entity::Goal,
            "seed 8 must place the goal at (6, 8) for this pose to be the intended one"
        );

        let occluded = observe_grid(&state, FourRoomsEnv::VISIBILITY);
        let see_through = observe_grid(&state, Visibility::SeeThrough);

        // (row 5, col 6) is the goal cell: one step forward, three to the right.
        let (row, col) = (5, 6);
        assert_eq!(
            see_through.view[row][col][0],
            Entity::Goal.type_u8(),
            "the pre-#281 emission model reported the goal from this pose"
        );
        assert_eq!(
            occluded.view[row][col],
            [UNSEEN_TYPE, 0, 0],
            "the goal must now encode as unseen — this env inherits canonical's \
             see_through_walls=False default"
        );
        assert_eq!(
            FourRoomsEnv::VISIBILITY,
            Visibility::Occluded,
            "fourrooms.py omits see_through_walls, so MiniGridEnv's False default applies"
        );
    }
}
