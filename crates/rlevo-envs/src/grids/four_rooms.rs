//! `FourRooms`: navigate a four-quadrant maze to reach the goal.
//!
//! Ports Farama Minigrid's [`FourRoomsEnv`]. The room is split into four
//! quadrants by a vertical and a horizontal interior wall, each containing
//! a single opening. The openings are positioned so a straight-line path
//! around one corner leads from the top-left quadrant to the goal in the
//! bottom-right quadrant.
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
use rlevo_core::environment::{Environment, EnvironmentError};
use rlevo_core::reward::ScalarReward;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

/// Minimum side length; we need at least three interior cells per quadrant.
const MIN_SIZE: usize = 11;

/// Configuration for [`FourRoomsEnv`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FourRoomsConfig {
    pub size: usize,
    pub max_steps: usize,
    pub seed: u64,
}

impl FourRoomsConfig {
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

impl FromStr for FourRoomsConfig {
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
        if cfg.size % 2 == 0 {
            return Err(format!("size must be odd, got {}", cfg.size));
        }
        Ok(cfg)
    }
}

/// Minigrid's `FourRooms` environment.
#[derive(Debug)]
pub struct FourRoomsEnv {
    state: GridState,
    config: FourRoomsConfig,
    steps: usize,
    render: bool,
    _rng: StdRng,
}

impl FourRoomsEnv {
    #[must_use]
    pub fn with_config(config: FourRoomsConfig, render: bool) -> Self {
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
    pub const fn config(&self) -> &FourRoomsConfig {
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

impl Display for FourRoomsEnv {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FourRoomsEnv(size={}, step={}/{})",
            self.config.size, self.steps, self.config.max_steps
        )
    }
}

impl Environment<3, 3, 1> for FourRoomsEnv {
    type StateType = GridState;
    type ObservationType = super::core::GridObservation;
    type ActionType = GridAction;
    type RewardType = ScalarReward;
    type SnapshotType = GridSnapshot;

    fn new(render: bool) -> Self {
        Self::with_config(FourRoomsConfig::default(), render)
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

    fn env_11() -> FourRoomsEnv {
        FourRoomsEnv::with_config(FourRoomsConfig::new(11, 400, 0), false)
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
        let mut a = FourRoomsEnv::with_config(cfg, false);
        let mut b = FourRoomsEnv::with_config(cfg, false);
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
        let mut env = FourRoomsEnv::with_config(cfg, false);
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
