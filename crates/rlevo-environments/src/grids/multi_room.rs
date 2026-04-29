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
//! ## Layout (3 rooms, room_width = 5, height = 5 — default)
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
//! ```no_run
//! use rlevo_envs::grids::multi_room::{MultiRoomConfig, MultiRoomEnv};
//! use rlevo_core::environment::Environment;
//!
//! let cfg = MultiRoomConfig::new(3, 5, 5, 300, 0);
//! let mut env = MultiRoomEnv::with_config(cfg, false);
//! let snap = env.reset().unwrap();
//! println!("walls: {:?}", env.wall_columns());
//! ```
//!
//! [`MultiRoomEnv`]: https://minigrid.farama.org/environments/minigrid/MultiRoomEnv/

use super::core::{
    GridSnapshot,
    action::GridAction,
    agent::AgentState,
    build_snapshot,
    color::Color,
    direction::Direction,
    dynamics::{StepOutcome, apply_action},
    entity::{DoorState, Entity},
    grid::Grid,
    render::render_ascii,
    reward::success_reward,
    state::GridState,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rlevo_core::environment::{Environment, EnvironmentError};
use rlevo_core::reward::ScalarReward;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

/// Minimum per-room width; needs at least one interior column in each room.
const MIN_ROOM_WIDTH: usize = 3;
/// Minimum number of rooms.
const MIN_NUM_ROOMS: usize = 2;
const DOOR_COLOR: Color = Color::Grey;

/// Configuration for [`MultiRoomEnv`].
///
/// # Examples
///
/// ```no_run
/// use rlevo_envs::grids::multi_room::MultiRoomConfig;
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
    /// Height of the strip in cells; must be ≥ 5.
    pub height: usize,
    /// Maximum steps before the episode times out with reward `0.0`.
    pub max_steps: usize,
    /// RNG seed; reserved for future stochastic variants.
    pub seed: u64,
}

impl MultiRoomConfig {
    /// Creates a [`MultiRoomConfig`] with the given parameters.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rlevo_envs::grids::multi_room::MultiRoomConfig;
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

impl FromStr for MultiRoomConfig {
    type Err = String;

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
        if cfg.num_rooms < MIN_NUM_ROOMS {
            return Err(format!(
                "num_rooms must be >= {MIN_NUM_ROOMS}, got {}",
                cfg.num_rooms
            ));
        }
        if cfg.room_width < MIN_ROOM_WIDTH {
            return Err(format!(
                "room_width must be >= {MIN_ROOM_WIDTH}, got {}",
                cfg.room_width
            ));
        }
        if cfg.height < 5 {
            return Err(format!("height must be >= 5, got {}", cfg.height));
        }
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
/// [`GridObservation`] / [`GridAction`] / [`ScalarReward`].
///
/// # Examples
///
/// ```no_run
/// use rlevo_envs::grids::multi_room::MultiRoomEnv;
/// use rlevo_core::environment::Environment;
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
    _rng: StdRng,
}

impl MultiRoomEnv {
    /// Constructs a [`MultiRoomEnv`] from an explicit configuration.
    #[must_use]
    pub fn with_config(config: MultiRoomConfig, render: bool) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        let state = Self::build(&config);
        Self {
            state,
            config,
            steps: 0,
            render,
            _rng: rng,
        }
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

    fn emit(&self, reward: f32, done: bool) -> GridSnapshot {
        if self.render {
            println!("{}", self.ascii());
        }
        build_snapshot(&self.state, reward, done)
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

impl Environment<3, 3, 1> for MultiRoomEnv {
    type StateType = GridState;
    type ObservationType = super::core::GridObservation;
    type ActionType = GridAction;
    type RewardType = ScalarReward;
    type SnapshotType = GridSnapshot;

    fn new(render: bool) -> Self {
        Self::with_config(MultiRoomConfig::default(), render)
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use rlevo_core::environment::Snapshot;

    fn default_env() -> MultiRoomEnv {
        MultiRoomEnv::with_config(MultiRoomConfig::default(), false)
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
        let mut a = MultiRoomEnv::with_config(cfg, false);
        let mut b = MultiRoomEnv::with_config(cfg, false);
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
        let mut env = MultiRoomEnv::with_config(cfg, false);
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
}
