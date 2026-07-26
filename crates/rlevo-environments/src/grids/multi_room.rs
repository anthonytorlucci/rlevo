//! `MultiRoom`: traverse a sequence of rooms connected by closed doors.
//!
//! Ports Farama Minigrid's [`MultiRoomEnv`]. The world is a horizontal
//! strip divided into `num_rooms` equal-width rooms by vertical walls.
//! Each dividing wall contains a closed (but unlocked) door at the
//! corridor row; the agent must toggle each door open in turn to reach
//! the goal in the last room.
//!
//! This is the first grid environment whose rollout length scales with
//! a configurable structural knob (`num_rooms`), giving it a long-horizon
//! planning flavor without depending on randomization.
//!
//! ## Known deviation from upstream `MultiRoom`
//!
//! **This is a simplified fixed layout, not a port of upstream's procedural
//! generator.** Farama's `MiniGrid-MultiRoom-*` resamples the room count, each
//! room's size and position (recursive placement with backtracking), each
//! door's wall, position and color, and the agent and goal placements — every
//! episode is a different board. rlevo builds one uniform horizontal strip of
//! equal-width rooms, every door at `height / 2`, every door [`Color::Grey`].
//! The board is identical on every reset, so a policy can memorize it instead
//! of learning to explore; treat results here as a probe of planning and credit
//! assignment, not as comparable to published `MultiRoom` numbers. Procedural
//! generation is tracked in issue #1021.
//!
//! ## Layout (3 rooms, `room_width` = 5, height = 5 — default)
//!
//! ```text
//! # # # # # # # # # # # # # # # #
//! # . . . # . . . . # . . . . . #
//! # A . . D . . . . D . . . . G #   D = Door (grey, closed)
//! # . . . # . . . . # . . . . . #   A = agent, start (1, 2) facing East
//! # # # # # # # # # # # # # # # #   G = goal (14, 2);  # = wall
//! ```
//!
//! Dividing walls sit at `x = i × room_width` for `i` in `1..num_rooms`.
//! Each wall has a single closed door at the corridor row (`height / 2`).
//!
//! | Observation | 7 × 7 egocentric grid encoded as `[type, color, state]` per cell |
//! |-------------|------------------------------------------------------------------|
//! | Action      | `TurnLeft`, `TurnRight`, `Forward`, `Toggle`                     |
//! | Reward      | `success_reward(steps, max_steps)` on goal; `0.0` on timeout     |
//!
//! # Examples
//!
//! ```rust
//! use rlevo_environments::grids::multi_room::{MultiRoomConfig, MultiRoomEnv};
//! use rlevo_core::environment::{ConstructableEnv, Environment};
//!
//! let cfg = MultiRoomConfig::new(3, 5, 5, 300, 0);
//! let mut env = MultiRoomEnv::with_config(cfg, false).expect("valid config");
//! let snap = env.reset().unwrap();
//! println!("walls: {:?}", env.wall_columns());
//! ```
//!
//! [`MultiRoomEnv`]: https://minigrid.farama.org/environments/minigrid/MultiRoomEnv/

use super::core::{
    GridSnapshot, Visibility,
    action::GridAction,
    agent::AgentState,
    build_snapshot,
    color::Color,
    direction::Direction,
    dynamics::{StepOutcome, apply_action},
    entity::{DoorState, Entity},
    grid::Grid,
    observation::GridObservation,
    observe_grid,
    render::render_ascii,
    reward::success_reward,
    state::GridState,
};
use rlevo_core::config::{self, ConfigError, Validate};
use rlevo_core::environment::{ConstructableEnv, Environment, EnvironmentError, Sensor};
use rlevo_core::reward::ScalarReward;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

/// Minimum per-room width; needs at least one interior column in each room.
const MIN_ROOM_WIDTH: usize = 3;
/// Minimum number of rooms.
///
/// Below this the env has no dividing wall and therefore no door to toggle —
/// the task it exists to pose disappears.
const MIN_NUM_ROOMS: usize = 2;
/// Minimum strip height.
///
/// Was a bare `5` literal in [`FromStr`] before it became a [`Validate`] guard;
/// named so the floor is stated once and reachable from the field docs.
const MIN_HEIGHT: usize = 5;
const DOOR_COLOR: Color = Color::Grey;

/// Configuration for [`MultiRoomEnv`].
///
/// # Examples
///
/// ```rust
/// use rlevo_environments::grids::multi_room::MultiRoomConfig;
///
/// let cfg = MultiRoomConfig::new(3, 5, 5, 300, 0);
/// assert_eq!(cfg.total_width(), 16);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultiRoomConfig {
    /// Number of rooms along the horizontal axis; must be ≥ `MIN_NUM_ROOMS` (2).
    pub num_rooms: usize,
    /// Width of each individual room, including its right-hand wall; must be ≥ `MIN_ROOM_WIDTH` (3).
    pub room_width: usize,
    /// Height of the strip in cells; must be ≥ `MIN_HEIGHT` (5).
    pub height: usize,
    /// Maximum steps before the episode times out with reward `0.0`.
    pub max_steps: usize,
    /// Accepted but unused: this environment makes no random draws.
    ///
    /// rlevo's `MultiRoom` is a **simplified fixed layout that deviates from
    /// upstream** `MiniGrid-MultiRoom-*`, which resamples the room count, every
    /// room's size and position, and every door's wall, position and color on
    /// each reset. Here the strip, the doors (all at `height / 2`, all
    /// [`Color::Grey`]), the agent and the goal are the same on every reset, so
    /// a policy can memorize a single board. Procedural generation is tracked in
    /// issue #1021; until then this value cannot change any observation, reward,
    /// or transition.
    ///
    /// The field is kept so every grid env presents the same config surface.
    pub seed: u64,
}

impl MultiRoomConfig {
    /// Creates a [`MultiRoomConfig`] with the given parameters.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rlevo_environments::grids::multi_room::MultiRoomConfig;
    ///
    /// let cfg = MultiRoomConfig::new(2, 4, 5, 160, 0);
    /// assert_eq!(cfg.total_width(), 9);
    /// ```
    #[must_use]
    pub const fn new(
        num_rooms: usize,
        room_width: usize,
        height: usize,
        max_steps: usize,
        seed: u64,
    ) -> Self {
        Self {
            num_rooms,
            room_width,
            height,
            max_steps,
            seed,
        }
    }

    /// Total grid width including the outer walls.
    #[must_use]
    pub const fn total_width(&self) -> usize {
        self.num_rooms * self.room_width + 1
    }
}

impl Default for MultiRoomConfig {
    fn default() -> Self {
        let num_rooms = 3;
        let room_width = 5;
        let height = 5;
        Self {
            num_rooms,
            room_width,
            height,
            max_steps: 4 * num_rooms * room_width * height,
            seed: 0,
        }
    }
}

impl Validate for MultiRoomConfig {
    /// Rejects `num_rooms` below `MIN_NUM_ROOMS` (2), `room_width` below
    /// `MIN_ROOM_WIDTH` (3), `height` below `MIN_HEIGHT` (5), and a zero
    /// `max_steps`.
    ///
    /// The three floors live **here**, not only in [`FromStr`]: `MultiRoomConfig`
    /// derives `Deserialize`, so a config loaded from a file is user-supplied
    /// runtime data that never passes through `from_str` (rules.md §4 — "if an
    /// invalid value can arrive via `Deserialize`, it must be an `Err`").
    /// Measured below the floors: `num_rooms = 1` silently built a
    /// **single-room** env with no dividing wall and no door at all
    /// (`wall_columns()` returned `[]`); `room_width = 2` built rooms of a single
    /// interior column; `height = 2` put the corridor row, its doors and the goal
    /// on the bottom border wall.
    ///
    /// `at_least` subsumes the previous `nonzero` checks — every floor is `>= 1`,
    /// so zeros are still rejected, now as
    /// [`ConstraintKind::TooSmall`](rlevo_core::config::ConstraintKind::TooSmall).
    /// `max_steps` keeps `nonzero`: it has no floor above 1.
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "MultiRoomConfig";
        config::at_least(C, "num_rooms", self.num_rooms, MIN_NUM_ROOMS)?;
        config::at_least(C, "room_width", self.room_width, MIN_ROOM_WIDTH)?;
        config::at_least(C, "height", self.height, MIN_HEIGHT)?;
        config::nonzero(C, "max_steps", self.max_steps)?;
        Ok(())
    }
}

impl FromStr for MultiRoomConfig {
    type Err = String;

    /// Parses `"num_rooms=3,room_width=5,height=5,max_steps=300,seed=0"` (keys in
    /// any order) or the positional form `"3,5,5,300,0"`.
    ///
    /// # Errors
    ///
    /// Returns the offending key/value, or the [`Validate`] rejection — the same
    /// guard [`MultiRoomEnv::with_config`] applies, so this parser cannot admit a
    /// config that construction would refuse.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut cfg = Self::default();
        for (idx, raw) in s.trim().split(',').map(str::trim).enumerate() {
            if raw.is_empty() {
                continue;
            }
            if let Some((key, value)) = raw.split_once('=') {
                match key.trim() {
                    "num_rooms" => {
                        cfg.num_rooms = value
                            .trim()
                            .parse()
                            .map_err(|e| format!("num_rooms: {e}"))?;
                    }
                    "room_width" => {
                        cfg.room_width = value
                            .trim()
                            .parse()
                            .map_err(|e| format!("room_width: {e}"))?;
                    }
                    "height" => {
                        cfg.height = value.trim().parse().map_err(|e| format!("height: {e}"))?;
                    }
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
                    0 => cfg.num_rooms = raw.parse().map_err(|e| format!("num_rooms: {e}"))?,
                    1 => cfg.room_width = raw.parse().map_err(|e| format!("room_width: {e}"))?,
                    2 => cfg.height = raw.parse().map_err(|e| format!("height: {e}"))?,
                    3 => cfg.max_steps = raw.parse().map_err(|e| format!("max_steps: {e}"))?,
                    4 => cfg.seed = raw.parse().map_err(|e| format!("seed: {e}"))?,
                    _ => return Err(format!("unexpected positional value `{raw}`")),
                }
            }
        }
        // Unlike the square-grid envs there is no single `size` worth appending:
        // `ConfigError`'s own `Display` already names the config, the field and
        // the offending count.
        cfg.validate().map_err(|e| e.to_string())?;
        Ok(cfg)
    }
}

/// Minigrid's `MultiRoom` environment.
///
/// The world is a horizontal strip of rooms connected by closed (unlocked)
/// doors; the agent must toggle each door open in sequence to reach the goal
/// in the last room. Rollout length scales with [`MultiRoomConfig::num_rooms`],
/// making this suitable for testing long-horizon planning and credit assignment.
///
/// Implements [`Environment<3, 3, 1>`] with [`GridState`] /
/// [`GridObservation`](super::core::GridObservation) / [`GridAction`] / [`ScalarReward`].
///
/// # Examples
///
/// ```rust
/// use rlevo_environments::grids::multi_room::MultiRoomEnv;
/// use rlevo_core::environment::{ConstructableEnv, Environment};
///
/// let mut env = MultiRoomEnv::new(false);
/// let snap = env.reset().unwrap();
/// println!("wall cols: {:?}", env.wall_columns());
/// ```
#[derive(Debug)]
pub struct MultiRoomEnv {
    state: GridState,
    config: MultiRoomConfig,
    steps: usize,
    render: bool,
}

impl MultiRoomEnv {
    /// Emission-model visibility policy: does this environment's agent see
    /// through opaque cells?
    ///
    /// The rlevo spelling of canonical Minigrid's `see_through_walls`
    /// constructor argument. Read only by this environment's [`Sensor`] impl,
    /// and an inherent const rather than a config field because it is part of
    /// the task definition, not a knob a caller tunes.
    ///
    /// [`Visibility::Occluded`], because canonical
    /// `minigrid/envs/multiroom.py` **omits** `see_through_walls` entirely and
    /// so inherits `MiniGridEnv.__init__`'s `see_through_walls=False` default:
    /// upstream, occlusion is on unless an env opts out, and this one does not.
    /// See ADR 0063 (`docs/adr/0063-grid-visibility-occlusion.md`) for the
    /// whole twelve-environment table.
    const VISIBILITY: Visibility = Visibility::Occluded;

    /// Constructs a [`MultiRoomEnv`] from an explicit configuration.
    ///
    /// Immediately builds the initial grid state. Call [`Environment::reset`]
    /// before the first [`Environment::step`] to obtain the first observation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rlevo_environments::grids::multi_room::{MultiRoomConfig, MultiRoomEnv};
    ///
    /// let env = MultiRoomEnv::with_config(
    ///     MultiRoomConfig::new(3, 5, 5, 300, 0),
    ///     true, // render ASCII grid to stdout
    /// )
    /// .expect("valid config");
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `config` fails [`Validate`]: `num_rooms`
    /// below `MIN_NUM_ROOMS` (2), `room_width` below `MIN_ROOM_WIDTH` (3),
    /// `height` below `MIN_HEIGHT` (5), or a zero `max_steps`. This is the
    /// construction chokepoint (rules.md §4), so it also rejects a config that
    /// arrived by `Deserialize` or struct-update syntax without passing through
    /// [`FromStr`].
    pub fn with_config(config: MultiRoomConfig, render: bool) -> Result<Self, ConfigError> {
        config.validate()?;
        let state = Self::build(&config);
        Ok(Self {
            state,
            config,
            steps: 0,
            render,
        })
    }

    /// Returns the environment's active configuration.
    #[must_use]
    pub const fn config(&self) -> &MultiRoomConfig {
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

    /// Renders the current grid state as an ASCII string.
    #[must_use]
    pub fn ascii(&self) -> String {
        render_ascii(&self.state.grid, &self.state.agent)
    }

    /// World x coordinates of each interior dividing wall column.
    #[must_use]
    pub fn wall_columns(&self) -> Vec<i32> {
        #[allow(clippy::cast_possible_wrap)]
        let rw = self.config.room_width as i32;
        (1..self.config.num_rooms)
            .map(|i| {
                #[allow(clippy::cast_possible_wrap)]
                let ii = i as i32;
                ii * rw
            })
            .collect()
    }

    fn build(config: &MultiRoomConfig) -> GridState {
        let total_w = config.total_width();
        let mut grid = Grid::new(total_w, config.height);
        grid.draw_walls();

        #[allow(clippy::cast_possible_wrap)]
        let height = config.height as i32;
        #[allow(clippy::cast_possible_wrap)]
        let room_width = config.room_width as i32;
        let corridor_row = height / 2;

        // Interior vertical dividing walls at col i * room_width for
        // i in 1..num_rooms. Each wall has a single closed door at the
        // corridor row.
        for i in 1..config.num_rooms {
            #[allow(clippy::cast_possible_wrap)]
            let wall_col = (i as i32) * room_width;
            for y in 1..height - 1 {
                grid.set(wall_col, y, Entity::Wall);
            }
            grid.set(
                wall_col,
                corridor_row,
                Entity::Door(DOOR_COLOR, DoorState::Closed),
            );
        }

        // Goal in the right-most room, at the corridor row one cell west
        // of the outer wall.
        #[allow(clippy::cast_possible_wrap)]
        let goal_x = (total_w - 2) as i32;
        grid.set(goal_x, corridor_row, Entity::Goal);

        let agent = AgentState::new(1, corridor_row, Direction::East);
        GridState::new(grid, agent)
    }

    fn emit(&self, observation: GridObservation, reward: f32, done: bool) -> GridSnapshot {
        if self.render {
            println!("{}", self.ascii());
        }
        build_snapshot(observation, reward, done)
    }
}

impl crate::render::AsciiRenderable for MultiRoomEnv {
    fn render_ascii(&self) -> String {
        render_ascii(&self.state.grid, &self.state.agent)
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        super::core::render::render_styled(&self.state.grid, &self.state.agent)
    }
}

impl Display for MultiRoomEnv {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MultiRoomEnv(num_rooms={}, step={}/{})",
            self.config.num_rooms, self.steps, self.config.max_steps
        )
    }
}

impl ConstructableEnv for MultiRoomEnv {
    fn new(render: bool) -> Self {
        Self::with_config(MultiRoomConfig::default(), render).expect("default config must validate")
    }
}

impl Sensor<3, 1, 3> for MultiRoomEnv {
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

impl Environment<3, 3, 1> for MultiRoomEnv {
    type StateType = GridState;
    type ObservationType = super::core::GridObservation;
    type ActionType = GridAction;
    type RewardType = ScalarReward;
    type SnapshotType = GridSnapshot;

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.state = Self::build(&self.config);
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

impl rlevo_core::render::payload::GridPayloadSource for MultiRoomEnv {
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
    // Test-only: the `Visibility` dispatch point and the byte a masked cell
    // carries, both read by
    // `test_multi_room_occlusion_hides_the_goal_behind_the_next_closed_door`.
    use super::super::core::{UNSEEN_TYPE, VIEW_SIZE, mask_view};
    use rlevo_core::config::ConstraintKind;
    use rlevo_core::environment::Snapshot;

    fn default_env() -> MultiRoomEnv {
        MultiRoomEnv::with_config(MultiRoomConfig::default(), false).expect("valid config")
    }

    #[test]
    fn default_config_validates() {
        assert!(MultiRoomConfig::default().validate().is_ok());
    }

    #[test]
    fn rejects_zero_num_rooms() {
        let bad = MultiRoomConfig {
            num_rooms: 0,
            ..Default::default()
        };
        assert!(MultiRoomEnv::with_config(bad, false).is_err());
    }

    #[test]
    fn default_config() {
        let cfg = MultiRoomConfig::default();
        assert_eq!(cfg.num_rooms, 3);
        assert_eq!(cfg.room_width, 5);
        assert_eq!(cfg.total_width(), 16);
    }

    #[test]
    fn fromstr_rejects_too_few_rooms() {
        assert!("num_rooms=1".parse::<MultiRoomConfig>().is_err());
    }

    #[test]
    fn fromstr_rejects_narrow_rooms() {
        assert!(
            "num_rooms=3,room_width=2"
                .parse::<MultiRoomConfig>()
                .is_err()
        );
    }

    /// Issue #106: the three structural floors were enforced only in
    /// [`FromStr`], so a config built by `Deserialize` or struct-update syntax
    /// reached `build`. Measured: `num_rooms = 1` produced a silent single-room
    /// env whose `wall_columns()` was empty — no dividing wall, no door, the
    /// whole task gone. The guard now lives in [`Validate`], which `with_config`
    /// runs (ADR 0026 chokepoint).
    #[test]
    fn with_config_rejects_num_rooms_below_min() {
        let bad = MultiRoomConfig {
            num_rooms: MIN_NUM_ROOMS - 1,
            ..Default::default()
        };
        let err = MultiRoomEnv::with_config(bad, false).unwrap_err();
        assert_eq!(err.config, "MultiRoomConfig");
        assert_eq!(err.field, "num_rooms");
        assert_eq!(
            err.kind,
            ConstraintKind::TooSmall {
                min: MIN_NUM_ROOMS as u64,
                got: (MIN_NUM_ROOMS - 1) as u64,
            }
        );
    }

    /// `room_width = 2` used to build rooms with a single interior column.
    #[test]
    fn with_config_rejects_room_width_below_min() {
        let bad = MultiRoomConfig {
            room_width: MIN_ROOM_WIDTH - 1,
            ..Default::default()
        };
        let err = MultiRoomEnv::with_config(bad, false).unwrap_err();
        assert_eq!(err.config, "MultiRoomConfig");
        assert_eq!(err.field, "room_width");
        assert_eq!(
            err.kind,
            ConstraintKind::TooSmall {
                min: MIN_ROOM_WIDTH as u64,
                got: (MIN_ROOM_WIDTH - 1) as u64,
            }
        );
    }

    /// `height = 2` used to build a degenerate strip whose corridor row — agent,
    /// doors and goal — sat on the bottom border wall.
    #[test]
    fn with_config_rejects_height_below_min() {
        let bad = MultiRoomConfig {
            height: MIN_HEIGHT - 1,
            ..Default::default()
        };
        let err = MultiRoomEnv::with_config(bad, false).unwrap_err();
        assert_eq!(err.config, "MultiRoomConfig");
        assert_eq!(err.field, "height");
        assert_eq!(
            err.kind,
            ConstraintKind::TooSmall {
                min: MIN_HEIGHT as u64,
                got: (MIN_HEIGHT - 1) as u64,
            }
        );
    }

    #[test]
    fn build_places_doors_at_corridor_row() {
        let env = default_env();
        let grid = &env.state().grid;
        // num_rooms=3, room_width=5, height=5 → walls at cols 5, 10.
        // corridor_row = 5/2 = 2.
        assert_eq!(grid.get(5, 2), Entity::Door(DOOR_COLOR, DoorState::Closed));
        assert_eq!(grid.get(10, 2), Entity::Door(DOOR_COLOR, DoorState::Closed));
        // Walls above and below the door on each dividing column.
        assert_eq!(grid.get(5, 1), Entity::Wall);
        assert_eq!(grid.get(5, 3), Entity::Wall);
        assert_eq!(grid.get(10, 1), Entity::Wall);
        // Goal at (14, 2).
        assert_eq!(grid.get(14, 2), Entity::Goal);
        // Agent at (1, 2) facing east.
        assert_eq!(env.state().agent.x, 1);
        assert_eq!(env.state().agent.y, 2);
        assert_eq!(env.state().agent.direction, Direction::East);
    }

    #[test]
    fn closed_door_blocks_movement() {
        let mut env = default_env();
        env.reset().unwrap();
        // Walk east three steps then bump the closed door at (5, 2).
        env.step(GridAction::Forward).unwrap(); // (2, 2)
        env.step(GridAction::Forward).unwrap(); // (3, 2)
        env.step(GridAction::Forward).unwrap(); // (4, 2)
        let snap = env.step(GridAction::Forward).unwrap();
        assert!(!snap.is_done());
        assert_eq!(env.state().agent.x, 4);
    }

    #[test]
    fn optimal_rollout_opens_both_doors_and_reaches_goal() {
        let mut env = default_env();
        env.reset().unwrap();
        // Room 1 → Room 2 → Room 3 rollout.
        let script = [
            GridAction::Forward, // (2, 2)
            GridAction::Forward, // (3, 2)
            GridAction::Forward, // (4, 2)
            GridAction::Toggle,  // open door at (5, 2)
            GridAction::Forward, // (5, 2)
            GridAction::Forward, // (6, 2)
            GridAction::Forward, // (7, 2)
            GridAction::Forward, // (8, 2)
            GridAction::Forward, // (9, 2)
            GridAction::Toggle,  // open door at (10, 2)
            GridAction::Forward, // (10, 2)
            GridAction::Forward, // (11, 2)
            GridAction::Forward, // (12, 2)
            GridAction::Forward, // (13, 2)
            GridAction::Forward, // (14, 2) goal
        ];
        let mut last = None;
        for a in script {
            last = Some(env.step(a).unwrap());
        }
        let snap = last.unwrap();
        assert!(snap.is_done());
        let reward: f32 = (*snap.reward()).into();
        assert!(reward > 0.9, "reward was {reward}");
    }

    #[test]
    fn reset_is_deterministic() {
        let cfg = MultiRoomConfig::new(3, 5, 5, 300, 2);
        let mut a = MultiRoomEnv::with_config(cfg, false).expect("valid config");
        let mut b = MultiRoomEnv::with_config(cfg, false).expect("valid config");
        let sa = a.reset().unwrap();
        let sb = b.reset().unwrap();
        assert_eq!(sa.observation(), sb.observation());
    }

    #[test]
    fn wall_columns_match_num_rooms() {
        let env = default_env();
        assert_eq!(env.wall_columns(), vec![5, 10]);
    }

    #[test]
    fn timeout_terminates_with_zero_reward() {
        let cfg = MultiRoomConfig::new(3, 5, 5, 3, 0);
        let mut env = MultiRoomEnv::with_config(cfg, false).expect("valid config");
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
        let mut env = default_env();
        env.reset().unwrap();
        env.step(GridAction::TurnLeft).unwrap();
        assert_eq!(env.steps(), 1);
        env.reset().unwrap();
        assert_eq!(env.steps(), 0);
    }

    /// Occlusion is not decoration here: a cell that encoded a real entity under
    /// the pre-#281 (see-through) emission model must now be able to encode as
    /// *unseen*.
    ///
    /// The pose is the one that decides the task. The agent is driven through
    /// the real dynamics — open door one, walk into room two, stop nose-to-nose
    /// with door two — and the assertion is on the observation the environment
    /// itself **emitted** on that step, not on a hand-built state. From there the
    /// goal is five cells straight ahead, inside the view window, and the closed
    /// door plus its wall column stand between: the whole reason to toggle is
    /// invisible until it is toggled.
    ///
    /// The occlusion here is structural rather than lucky: every dividing wall
    /// spans the full interior height with only the door punched through it, so
    /// unlike `FourRooms` — whose cross openings let the flood fill leak sideways
    /// into the next room, leaving its goal maskable from any pose in only 2 of
    /// 12 seeds — there is no route around it from any pose in this room.
    #[test]
    fn test_multi_room_occlusion_hides_the_goal_behind_the_next_closed_door() {
        let mut env = default_env();
        env.reset().expect("reset");
        assert_eq!(
            env.wall_columns(),
            vec![5, 10],
            "the default strip's dividers"
        );
        assert_eq!(env.state().grid.get(14, 2), Entity::Goal, "goal at (14, 2)");

        // Open door one, cross into room two, stop in front of door two.
        let script = [
            GridAction::Forward, // (2, 2)
            GridAction::Forward, // (3, 2)
            GridAction::Forward, // (4, 2)
            GridAction::Toggle,  // open the door at (5, 2)
            GridAction::Forward, // (5, 2)
            GridAction::Forward, // (6, 2)
            GridAction::Forward, // (7, 2)
            GridAction::Forward, // (8, 2)
            GridAction::Forward, // (9, 2) — nose to nose with the door at (10, 2)
        ];
        let mut last = None;
        for action in script {
            last = Some(env.step(action).expect("step"));
        }
        let snapshot = last.expect("the script is non-empty");
        let agent = env.state().agent;
        assert_eq!(
            (agent.x, agent.y, agent.direction),
            (9, 2, Direction::East),
            "the script must land the agent in front of the second door"
        );
        assert_eq!(
            env.state().grid.get(10, 2),
            Entity::Door(DOOR_COLOR, DoorState::Closed),
            "door two must still be shut — it is the occluder under test"
        );

        let masked = mask_view(&env.state().grid, &agent, MultiRoomEnv::VISIBILITY);
        let occluded = *snapshot.observation();
        let see_through = observe_grid(env.state(), Visibility::SeeThrough);

        // (row 1, col 3) is the goal: five cells straight ahead.
        let (row, col) = (1, 3);
        assert_eq!(
            see_through.view[row][col][0],
            Entity::Goal.type_u8(),
            "the pre-#281 emission model reported the goal through two rooms of wall"
        );
        assert_eq!(
            occluded.view[row][col],
            [UNSEEN_TYPE, 0, 0],
            "the goal must now encode as unseen — this env inherits canonical's \
             see_through_walls=False default"
        );
        assert_eq!(
            masked[5][3],
            Some(Entity::Door(DOOR_COLOR, DoorState::Closed)),
            "the shut door one cell ahead is itself visible — the agent sees what blocks it"
        );

        // Differ *exactly* on the hidden set, in both directions: a bare
        // "something is masked" would also pass if occlusion leaked into cells it
        // must not touch.
        for (r, c) in (0..VIEW_SIZE).flat_map(|r| (0..VIEW_SIZE).map(move |c| (r, c))) {
            if masked[r][c].is_none() {
                assert_eq!(
                    occluded.view[r][c],
                    [UNSEEN_TYPE, 0, 0],
                    "hidden cell ({r}, {c}) must encode as the unseen triple"
                );
                assert_ne!(
                    occluded.view[r][c], see_through.view[r][c],
                    "hidden cell ({r}, {c}) must not encode the same as the seen one"
                );
            } else {
                assert_eq!(
                    occluded.view[r][c], see_through.view[r][c],
                    "visible cell ({r}, {c}) must be unaffected by occlusion"
                );
            }
        }

        assert_eq!(
            MultiRoomEnv::VISIBILITY,
            Visibility::Occluded,
            "multiroom.py omits see_through_walls, so MiniGridEnv's False default applies"
        );
    }
}
