//! `DoorKey`: pick up the key, unlock the door, reach the goal.
//!
//! Ports Farama Minigrid's [`DoorKeyEnv`]. The room is bisected by a
//! vertical interior wall containing a single [`Locked`] door. A matching
//! key sits in the left room; the goal sits in the right room.
//!
//! [`DoorKeyEnv`]: https://minigrid.farama.org/environments/minigrid/DoorKeyEnv/
//! [`Locked`]: super::core::entity::DoorState::Locked

use super::core::{
    action::GridAction,
    agent::AgentState,
    build_snapshot,
    color::Color,
    direction::Direction,
    dynamics::{apply_action, StepOutcome},
    entity::{DoorState, Entity},
    grid::Grid,
    render::render_ascii,
    reward::success_reward,
    state::GridState,
    GridSnapshot,
};
use evorl_core::environment::{Environment, EnvironmentError};
use evorl_core::reward::ScalarReward;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

/// Minimum side length; the split wall needs a left and right sub-room.
const MIN_SIZE: usize = 5;
/// Color used for the key, door, and lock.
const DOOR_COLOR: Color = Color::Yellow;

/// Configuration for [`DoorKeyEnv`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoorKeyConfig {
    pub size: usize,
    pub max_steps: usize,
    pub seed: u64,
}

impl DoorKeyConfig {
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

impl FromStr for DoorKeyConfig {
    type Err = String;

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
        if cfg.size < MIN_SIZE {
            return Err(format!("size must be >= {MIN_SIZE}, got {}", cfg.size));
        }
        Ok(cfg)
    }
}

/// Minigrid's `DoorKey` environment.
#[derive(Debug)]
pub struct DoorKeyEnv {
    state: GridState,
    config: DoorKeyConfig,
    steps: usize,
    render: bool,
    _rng: StdRng,
}

impl DoorKeyEnv {
    #[must_use]
    pub fn with_config(config: DoorKeyConfig, render: bool) -> Self {
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

    #[must_use]
    pub const fn config(&self) -> &DoorKeyConfig {
        &self.config
    }

    #[must_use]
    pub const fn steps(&self) -> usize {
        self.steps
    }

    #[must_use]
    pub const fn state(&self) -> &GridState {
        &self.state
    }

    #[must_use]
    pub fn ascii(&self) -> String {
        render_ascii(&self.state.grid, &self.state.agent)
    }

    /// Column where the vertical interior wall and door sit.
    #[must_use]
    pub fn split_col(&self) -> i32 {
        #[allow(clippy::cast_possible_wrap)]
        let col = (self.config.size / 2) as i32;
        col
    }

    fn build(config: &DoorKeyConfig) -> GridState {
        let mut grid = Grid::new(config.size, config.size);
        grid.draw_walls();

        #[allow(clippy::cast_possible_wrap)]
        let size = config.size as i32;
        #[allow(clippy::cast_possible_wrap)]
        let split_col = (config.size / 2) as i32;

        // Fill the interior column with walls, then replace one cell with
        // the locked door.
        for y in 1..size - 1 {
            grid.set(split_col, y, Entity::Wall);
        }
        grid.set(
            split_col,
            split_col,
            Entity::Door(DOOR_COLOR, DoorState::Locked),
        );

        // Key in the left room and goal in the right room.
        grid.set(1, 1, Entity::Key(DOOR_COLOR));
        grid.set(size - 2, size - 2, Entity::Goal);

        // Agent stands just south of the key, facing north so that
        // `Pickup` grabs the key immediately.
        let agent = AgentState::new(1, 2, Direction::North);
        GridState::new(grid, agent)
    }

    fn emit(&self, reward: f32, done: bool) -> GridSnapshot {
        if self.render {
            println!("{}", self.ascii());
        }
        build_snapshot(&self.state, reward, done)
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

impl Environment<3, 3, 1> for DoorKeyEnv {
    type StateType = GridState;
    type ObservationType = super::core::GridObservation;
    type ActionType = GridAction;
    type RewardType = ScalarReward;
    type SnapshotType = GridSnapshot;

    fn new(render: bool) -> Self {
        Self::with_config(DoorKeyConfig::default(), render)
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
    use evorl_core::environment::Snapshot;

    fn env_5x5() -> DoorKeyEnv {
        DoorKeyEnv::with_config(DoorKeyConfig::new(5, 100, 0), false)
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

    #[test]
    fn build_places_door_key_and_goal() {
        let env = env_5x5();
        let grid = &env.state().grid;
        // Split column is 2; door at (2, 2); walls at (2, 1) and (2, 3).
        assert_eq!(grid.get(2, 1), Entity::Wall);
        assert_eq!(grid.get(2, 2), Entity::Door(DOOR_COLOR, DoorState::Locked));
        assert_eq!(grid.get(2, 3), Entity::Wall);
        assert_eq!(grid.get(1, 1), Entity::Key(DOOR_COLOR));
        assert_eq!(grid.get(3, 3), Entity::Goal);
        assert_eq!(env.state().agent.x, 1);
        assert_eq!(env.state().agent.y, 2);
        assert_eq!(env.state().agent.direction, Direction::North);
    }

    #[test]
    fn reset_is_deterministic() {
        let cfg = DoorKeyConfig::new(5, 100, 17);
        let mut a = DoorKeyEnv::with_config(cfg, false);
        let mut b = DoorKeyEnv::with_config(cfg, false);
        let sa = a.reset().unwrap();
        let sb = b.reset().unwrap();
        assert_eq!(sa.observation(), sb.observation());
    }

    #[test]
    fn toggle_locked_door_without_key_is_noop() {
        let mut env = env_5x5();
        env.reset().unwrap();
        // Turn east to face the door at (2, 2) without picking up the key.
        env.step(GridAction::TurnRight).unwrap();
        env.step(GridAction::Toggle).unwrap();
        assert!(matches!(
            env.state().grid.get(2, 2),
            Entity::Door(_, DoorState::Locked)
        ));
    }

    #[test]
    fn optimal_rollout_solves_env() {
        let mut env = env_5x5();
        env.reset().unwrap();
        // (1,2) N. Key (1,1). Door (2,2). Goal (3,3).
        let script = [
            GridAction::Pickup,    // grab key
            GridAction::TurnRight, // face east toward door
            GridAction::Toggle,    // unlock (Locked → Closed)
            GridAction::Toggle,    // open (Closed → Open)
            GridAction::Forward,   // (2, 2) onto door
            GridAction::Forward,   // (3, 2) into right room
            GridAction::TurnRight, // face south
            GridAction::Forward,   // (3, 3) goal
        ];
        let mut last = None;
        for a in script {
            last = Some(env.step(a).unwrap());
        }
        let snap = last.unwrap();
        assert!(snap.is_done(), "scripted rollout should reach the goal");
        let reward: f32 = (*snap.reward()).into();
        assert!(reward > 0.9, "reward was {reward}");
    }

    #[test]
    fn timeout_terminates_without_reward() {
        let cfg = DoorKeyConfig::new(5, 2, 0);
        let mut env = DoorKeyEnv::with_config(cfg, false);
        env.reset().unwrap();
        env.step(GridAction::TurnLeft).unwrap();
        let snap = env.step(GridAction::TurnLeft).unwrap();
        assert!(snap.is_done());
        let reward: f32 = (*snap.reward()).into();
        assert_eq!(reward, 0.0);
    }

    #[test]
    fn split_col_matches_config() {
        let env = env_5x5();
        assert_eq!(env.split_col(), 2);
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
        // After reset, agent is back at (1, 2) facing North.
        assert_eq!(env.state().agent.x, 1);
        assert_eq!(env.state().agent.y, 2);
        assert_eq!(env.state().agent.direction, Direction::North);
    }
}
