//! Navigate a four-quadrant maze to reach the goal in the bottom-right room.
//!
//! Ports Farama Minigrid's [`FourRoomsEnv`]. An interior cross of walls splits
//! the `N×N` grid into four equal quadrants. Each arm of the cross has one
//! opening, positioned symmetrically at `±3` cells from the centre. The agent
//! starts in the top-left quadrant; the goal sits in the bottom-right.
//!
//! The grid size must be **odd** and at least 11 so each quadrant has enough
//! interior cells to navigate.
//!
//! ## Layout (default `size = 11`, mid = 5)
//!
//! ```text
//! # # # # # # # # # # #
//! # A . . # . . . . . #   A = agent (1, 1)
//! # . . . O . . . . . #   opening at (5, 2)
//! # . . . # . . . . . #
//! # . . . # . . . . . #
//! # O . . # . . . O . #   openings at (2, 5) and (8, 5)
//! # . . . # . . . . . #
//! # . . . # . . . . . #
//! # . . . O . . . . . #   opening at (5, 8)
//! # . . . # . . . . G #   G = goal (9, 9)
//! # # # # # # # # # # #
//! ```
//!
//! - `A` — agent start (1, 1), facing East
//! - `G` — goal (9, 9)
//! - `O` — wall opening (passable gap)
//! - `#` — border or interior wall
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
    GridSnapshot,
    action::GridAction,
    agent::AgentState,
    build_snapshot,
    direction::Direction,
    dynamics::{StepOutcome, apply_action},
    entity::Entity,
    grid::Grid,
    render::render_ascii,
    reward::success_reward,
    state::GridState,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rlevo_core::config::{self, ConfigError, ConstraintKind, Validate};
use rlevo_core::environment::{ConstructableEnv, Environment, EnvironmentError};
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
/// clean size — the openings land inside the border from `9` up — so changing it
/// is a policy decision, not a bug fix.
const MIN_SIZE: usize = 11;

// The two invariants `build` silently assumed. Its four openings are written at
// `mid ± 3` where `mid = size / 2`, with no bounds check of their own; when they
// fall outside the interior they land on the *border* and punch holes in it.
// Measured before the guard existed: `size = 7` perforated all four perimeter
// walls — holes at (0,3), (3,0), (3,6), (6,3) — and `size = 8` perforated two,
// at (4,7) and (7,4). Written as `const _` so lowering `MIN_SIZE` breaks the
// build instead of the board.
//
//   `mid - 3 >= 1`         (top / left opening stays off the border) → mid >= 4
//   `mid + 3 <= size - 2`  (bottom / right opening stays off the border)
const _: () = assert!(MIN_SIZE % 2 == 1, "MIN_SIZE must be odd");
const _: () = assert!(
    MIN_SIZE / 2 + 3 <= MIN_SIZE - 2 && MIN_SIZE / 2 >= 4,
    "the mid±3 openings must land strictly inside the border"
);

/// Rejection text for an even `size`. Unlike the floor, the parity requirement
/// has no `at_least` analogue, so it is a `Custom` invariant.
const SIZE_NOT_ODD: &str =
    "FourRoomsEnv requires an odd size so the interior cross has a single centre row and column";

/// Configuration for [`FourRoomsEnv`].
///
/// Controls the grid size, episode length, and RNG seed. The interior wall
/// positions and door openings are derived from `size`, so no positional
/// fields are needed.
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
    /// Must be **odd** and at least `11`. The interior cross sits
    /// at column and row `size / 2`; openings are at `±3` from that centre.
    pub size: usize,
    /// Maximum number of steps before the episode is truncated.
    pub max_steps: usize,
    /// Seed for the internal random-number generator.
    ///
    /// Using the same seed always produces the same episode layout.
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
    /// be an `Err`"). Measured below the floor: `size` 1 through 6 panic inside
    /// `Grid::set` (out of bounds), and `size` 7 and 8 build a board whose
    /// `mid ± 3` openings are punched through the *perimeter* wall rather than
    /// the interior cross. Enforcing parity anywhere but here would recreate,
    /// for one env, exactly the split enforcement this guard removes.
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
/// The agent must transit through at least two openings to reach the goal,
/// making this a classic long-horizon exploration task. Because the layout is
/// fixed and deterministic, the environment is suitable for evaluating
/// systematic search and planning policies.
///
/// Construct via [`FourRoomsEnv::with_config`] for full control or via
/// [`ConstructableEnv::new`] for default settings (size 11, seed 0).
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
    // Never sampled: this env's layout builder is fully deterministic and
    // ignores `config.seed`, so the field is written but never read. Kept
    // as-is rather than renamed — see #397, which decides whether these
    // envs become genuinely stochastic or drop the seed entirely.
    #[allow(clippy::used_underscore_binding)]
    _rng: StdRng,
}

impl FourRoomsEnv {
    /// Constructs a `FourRoomsEnv` from an explicit configuration.
    ///
    /// Immediately builds the initial grid state and seeds the internal RNG.
    /// Call [`Environment::reset`] before the first [`Environment::step`] to
    /// obtain the first observation.
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
    pub fn with_config(config: FourRoomsConfig, render: bool) -> Result<Self, ConfigError> {
        config.validate()?;
        let rng = StdRng::seed_from_u64(config.seed);
        let state = Self::build(&config);
        Ok(Self {
            state,
            config,
            steps: 0,
            render,
            _rng: rng,
        })
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

    /// Renders the current grid as an ASCII string.
    ///
    /// Useful for debugging; the same output is printed to stdout on each step
    /// when the environment was constructed with `render = true`.
    #[must_use]
    pub fn ascii(&self) -> String {
        render_ascii(&self.state.grid, &self.state.agent)
    }

    fn build(config: &FourRoomsConfig) -> GridState {
        let mut grid = Grid::new(config.size, config.size);
        grid.draw_walls();

        #[allow(clippy::cast_possible_wrap)]
        let size = config.size as i32;
        let mid = size / 2;

        // Interior cross of walls.
        for y in 1..size - 1 {
            grid.set(mid, y, Entity::Wall);
        }
        for x in 1..size - 1 {
            grid.set(x, mid, Entity::Wall);
        }

        // Four openings at fixed offsets from the center wall.
        // Vertical wall openings on rows (mid - 3) and (mid + 3).
        grid.set(mid, mid - 3, Entity::Empty);
        grid.set(mid, mid + 3, Entity::Empty);
        // Horizontal wall openings on cols (mid - 3) and (mid + 3).
        grid.set(mid - 3, mid, Entity::Empty);
        grid.set(mid + 3, mid, Entity::Empty);

        grid.set(size - 2, size - 2, Entity::Goal);
        let agent = AgentState::new(1, 1, Direction::East);
        GridState::new(grid, agent)
    }

    fn emit(&self, reward: f32, done: bool) -> GridSnapshot {
        if self.render {
            println!("{}", self.ascii());
        }
        build_snapshot(&self.state, reward, done)
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
        Self::with_config(FourRoomsConfig::default(), render).expect("default config must validate")
    }
}

impl Environment<3, 3, 1> for FourRoomsEnv {
    type StateType = GridState;
    type ObservationType = super::core::GridObservation;
    type ActionType = GridAction;
    type RewardType = ScalarReward;
    type SnapshotType = GridSnapshot;

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.state = Self::build(&self.config);
        self.steps = 0;
        self._rng = StdRng::seed_from_u64(self.config.seed);
        Ok(self.emit(0.0, false))
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
        Ok(self.emit(reward, done))
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
    use rlevo_core::environment::Snapshot;

    fn env_11() -> FourRoomsEnv {
        FourRoomsEnv::with_config(FourRoomsConfig::new(11, 400, 0), false).expect("valid config")
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
    /// consequences: `size` 1–6 panicked in `Grid::set`; `size = 7` punched a
    /// hole in all four perimeter walls and `size = 8` in two, because `build`
    /// writes its openings at `mid ± 3` without checking where they land. The
    /// guard now lives in [`Validate`], which `with_config` runs (ADR 0026
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

    /// `9` builds a geometrically clean board — the `mid ± 3` openings land
    /// inside the border — and is still refused. The floor is `rlevo` policy for
    /// this env, not the smallest size the layout code survives.
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

    #[test]
    fn build_places_cross_with_four_openings() {
        let env = env_11();
        let grid = &env.state().grid;
        // mid = 5; openings at rows 2, 8 on col 5 and cols 2, 8 on row 5.
        assert_eq!(grid.get(5, 2), Entity::Empty);
        assert_eq!(grid.get(5, 8), Entity::Empty);
        assert_eq!(grid.get(2, 5), Entity::Empty);
        assert_eq!(grid.get(8, 5), Entity::Empty);
        // Other cells on the cross remain walls.
        assert_eq!(grid.get(5, 1), Entity::Wall);
        assert_eq!(grid.get(5, 9), Entity::Wall);
        assert_eq!(grid.get(1, 5), Entity::Wall);
        assert_eq!(grid.get(9, 5), Entity::Wall);
        // Goal at bottom-right corner.
        assert_eq!(grid.get(9, 9), Entity::Goal);
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

    #[test]
    fn central_walls_block_movement() {
        let mut env = env_11();
        env.reset().unwrap();
        // Walk east from (1, 1) until we bump into the vertical wall at (5, 1).
        for _ in 0..3 {
            env.step(GridAction::Forward).unwrap();
        }
        assert_eq!(env.state().agent.x, 4);
        // Next forward should bump into the wall.
        let snap = env.step(GridAction::Forward).unwrap();
        assert!(!snap.is_done());
        assert_eq!(env.state().agent.x, 4);
    }

    #[test]
    fn optimal_rollout_through_two_openings_reaches_goal() {
        let mut env = env_11();
        env.reset().unwrap();
        // Agent (1, 1) facing East → TurnRight, Forward to (1, 2), TurnLeft, go east
        // through the opening at (5, 2), continue east to (8, 2), TurnRight,
        // go south through (8, 5) opening, continue south to (8, 9), TurnLeft,
        // forward to (9, 9) goal.
        let script = [
            GridAction::TurnRight,
            GridAction::Forward, // (1, 2)
            GridAction::TurnLeft,
            GridAction::Forward, // (2, 2)
            GridAction::Forward, // (3, 2)
            GridAction::Forward, // (4, 2)
            GridAction::Forward, // (5, 2) opening
            GridAction::Forward, // (6, 2)
            GridAction::Forward, // (7, 2)
            GridAction::Forward, // (8, 2)
            GridAction::TurnRight,
            GridAction::Forward, // (8, 3)
            GridAction::Forward, // (8, 4)
            GridAction::Forward, // (8, 5) opening
            GridAction::Forward, // (8, 6)
            GridAction::Forward, // (8, 7)
            GridAction::Forward, // (8, 8)
            GridAction::Forward, // (8, 9)
            GridAction::TurnLeft,
            GridAction::Forward, // (9, 9) goal
        ];
        let mut last = None;
        for a in script {
            last = Some(env.step(a).unwrap());
        }
        let snap = last.unwrap();
        assert!(snap.is_done(), "should reach the goal");
        let reward: f32 = (*snap.reward()).into();
        assert!(reward > 0.9, "reward was {reward}");
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
}
