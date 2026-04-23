//! `DynamicObstacles`: reach the goal while avoiding randomly moving balls.
//!
//! Ports Farama Minigrid's [`DynamicObstaclesEnv`]. The room is otherwise
//! empty, but a configurable number of [`Ball`] obstacles random-walk
//! each step. Stepping onto an obstacle (bumping into one) is impossible
//! because balls are not passable; the failure mode is an obstacle
//! *moving onto the agent's cell*, which yields a reward of `-1.0` and
//! terminates the episode.
//!
//! This is the first grid environment with a genuinely stochastic
//! post-step hook driven by the env's own `StdRng`. Seeded resets are
//! deterministic, so the default `with_config` + `reset` pair is fully
//! reproducible for tests and benchmarks.
//!
//! [`DynamicObstaclesEnv`]: https://minigrid.farama.org/environments/minigrid/DynamicObstaclesEnv/
//! [`Ball`]: super::core::entity::Entity::Ball

use super::core::{
    GridSnapshot,
    action::GridAction,
    agent::AgentState,
    build_snapshot,
    color::Color,
    direction::Direction,
    dynamics::{StepOutcome, apply_action},
    entity::Entity,
    grid::Grid,
    render::render_ascii,
    reward::success_reward,
    state::GridState,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rlevo_core::environment::{Environment, EnvironmentError};
use rlevo_core::reward::ScalarReward;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

/// Minimum side length; we need empty cells to spawn obstacles in.
const MIN_SIZE: usize = 5;
/// Color used for every obstacle.
const OBSTACLE_COLOR: Color = Color::Blue;
/// Reward paid when an obstacle collides with the agent.
const COLLISION_REWARD: f32 = -1.0;

/// Configuration for [`DynamicObstaclesEnv`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DynamicObstaclesConfig {
    pub size: usize,
    pub num_obstacles: usize,
    pub max_steps: usize,
    pub seed: u64,
}

impl DynamicObstaclesConfig {
    #[must_use]
    pub const fn new(size: usize, num_obstacles: usize, max_steps: usize, seed: u64) -> Self {
        Self {
            size,
            num_obstacles,
            max_steps,
            seed,
        }
    }
}

impl Default for DynamicObstaclesConfig {
    fn default() -> Self {
        let size = 6;
        Self {
            size,
            num_obstacles: 1,
            max_steps: 4 * size * size,
            seed: 0,
        }
    }
}

impl FromStr for DynamicObstaclesConfig {
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
                    "num_obstacles" => {
                        cfg.num_obstacles = value
                            .trim()
                            .parse()
                            .map_err(|e| format!("num_obstacles: {e}"))?;
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
                    0 => cfg.size = raw.parse().map_err(|e| format!("size: {e}"))?,
                    1 => {
                        cfg.num_obstacles =
                            raw.parse().map_err(|e| format!("num_obstacles: {e}"))?
                    }
                    2 => cfg.max_steps = raw.parse().map_err(|e| format!("max_steps: {e}"))?,
                    3 => cfg.seed = raw.parse().map_err(|e| format!("seed: {e}"))?,
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

/// Minigrid's `DynamicObstacles` environment.
#[derive(Debug)]
pub struct DynamicObstaclesEnv {
    state: GridState,
    config: DynamicObstaclesConfig,
    steps: usize,
    render: bool,
    obstacles: Vec<(i32, i32)>,
    rng: StdRng,
}

impl DynamicObstaclesEnv {
    #[must_use]
    pub fn with_config(config: DynamicObstaclesConfig, render: bool) -> Self {
        let mut rng = StdRng::seed_from_u64(config.seed);
        let (state, obstacles) = Self::build(&config, &mut rng);
        Self {
            state,
            config,
            steps: 0,
            render,
            obstacles,
            rng,
        }
    }

    #[must_use]
    pub const fn config(&self) -> &DynamicObstaclesConfig {
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

    /// Current obstacle positions.
    #[must_use]
    pub fn obstacles(&self) -> &[(i32, i32)] {
        &self.obstacles
    }

    #[must_use]
    pub fn ascii(&self) -> String {
        render_ascii(&self.state.grid, &self.state.agent)
    }

    fn build(config: &DynamicObstaclesConfig, rng: &mut StdRng) -> (GridState, Vec<(i32, i32)>) {
        let mut grid = Grid::new(config.size, config.size);
        grid.draw_walls();

        #[allow(clippy::cast_possible_wrap)]
        let size = config.size as i32;
        let goal_xy = size - 2;
        grid.set(goal_xy, goal_xy, Entity::Goal);

        let agent = AgentState::new(1, 1, Direction::East);

        // Spawn obstacles in random empty interior cells, avoiding the
        // agent's starting square and the goal.
        let mut obstacles = Vec::with_capacity(config.num_obstacles);
        let mut candidates: Vec<(i32, i32)> = (1..size - 1)
            .flat_map(|x| (1..size - 1).map(move |y| (x, y)))
            .filter(|&(x, y)| (x, y) != (agent.x, agent.y) && (x, y) != (goal_xy, goal_xy))
            .collect();

        for _ in 0..config.num_obstacles {
            if candidates.is_empty() {
                break;
            }
            let idx = rng.random_range(0..candidates.len());
            let pos = candidates.swap_remove(idx);
            grid.set(pos.0, pos.1, Entity::Ball(OBSTACLE_COLOR));
            obstacles.push(pos);
        }

        (GridState::new(grid, agent), obstacles)
    }

    fn emit(&self, reward: f32, done: bool) -> GridSnapshot {
        if self.render {
            println!("{}", self.ascii());
        }
        build_snapshot(&self.state, reward, done)
    }

    /// Perform one random-walk step for each obstacle and return `true`
    /// if any obstacle landed on the agent's cell.
    fn move_obstacles(&mut self) -> bool {
        let mut collision = false;
        let agent_pos = (self.state.agent.x, self.state.agent.y);

        // Decide each obstacle's new position before mutating the grid
        // so later obstacles see a stable snapshot.
        let mut new_positions: Vec<(i32, i32)> = Vec::with_capacity(self.obstacles.len());
        for &pos in &self.obstacles {
            let candidates: Vec<(i32, i32)> = [
                (pos.0 + 1, pos.1),
                (pos.0 - 1, pos.1),
                (pos.0, pos.1 + 1),
                (pos.0, pos.1 - 1),
            ]
            .into_iter()
            .filter(|&p| Self::is_obstacle_target(&self.state.grid, p, agent_pos))
            .collect();

            if candidates.is_empty() {
                new_positions.push(pos);
                continue;
            }
            let idx = self.rng.random_range(0..candidates.len());
            new_positions.push(candidates[idx]);
        }

        // Clear old obstacle cells; place at new cells (unless collision).
        for &old in &self.obstacles {
            self.state.grid.set(old.0, old.1, Entity::Empty);
        }
        for (slot, &new_pos) in self.obstacles.iter_mut().zip(new_positions.iter()) {
            if new_pos == agent_pos {
                collision = true;
                // Don't place the ball on the agent's cell — episode ends.
            } else {
                self.state
                    .grid
                    .set(new_pos.0, new_pos.1, Entity::Ball(OBSTACLE_COLOR));
            }
            *slot = new_pos;
        }

        collision
    }

    fn is_obstacle_target(grid: &Grid, pos: (i32, i32), agent_pos: (i32, i32)) -> bool {
        if !grid.in_bounds(pos.0, pos.1) {
            return false;
        }
        if pos == agent_pos {
            return true;
        }
        matches!(grid.get(pos.0, pos.1), Entity::Empty | Entity::Floor)
    }
}

impl Display for DynamicObstaclesEnv {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DynamicObstaclesEnv(size={}, num_obstacles={}, step={}/{})",
            self.config.size, self.config.num_obstacles, self.steps, self.config.max_steps
        )
    }
}

impl Environment<3, 3, 1> for DynamicObstaclesEnv {
    type StateType = GridState;
    type ObservationType = super::core::GridObservation;
    type ActionType = GridAction;
    type RewardType = ScalarReward;
    type SnapshotType = GridSnapshot;

    fn new(render: bool) -> Self {
        Self::with_config(DynamicObstaclesConfig::default(), render)
    }

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.rng = StdRng::seed_from_u64(self.config.seed);
        let (state, obstacles) = Self::build(&self.config, &mut self.rng);
        self.state = state;
        self.obstacles = obstacles;
        self.steps = 0;
        Ok(self.emit(0.0, false))
    }

    fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> {
        self.steps += 1;
        let outcome = apply_action(&mut self.state.grid, &mut self.state.agent, action);

        // Terminal state from agent step takes priority over obstacle motion.
        if let StepOutcome::ReachedGoal = outcome {
            return Ok(self.emit(success_reward(self.steps, self.config.max_steps), true));
        }
        if let StepOutcome::HitLava = outcome {
            return Ok(self.emit(0.0, true));
        }

        let collided = self.move_obstacles();
        let done = collided || self.steps >= self.config.max_steps;
        let reward = if collided { COLLISION_REWARD } else { 0.0 };
        Ok(self.emit(reward, done))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlevo_core::environment::Snapshot;

    fn env_no_obstacles() -> DynamicObstaclesEnv {
        DynamicObstaclesEnv::with_config(DynamicObstaclesConfig::new(5, 0, 100, 0), false)
    }

    #[test]
    fn default_config_has_one_obstacle() {
        let cfg = DynamicObstaclesConfig::default();
        assert_eq!(cfg.size, 6);
        assert_eq!(cfg.num_obstacles, 1);
    }

    #[test]
    fn fromstr_parses_num_obstacles() {
        let cfg: DynamicObstaclesConfig = "size=7,num_obstacles=3".parse().unwrap();
        assert_eq!(cfg.size, 7);
        assert_eq!(cfg.num_obstacles, 3);
    }

    #[test]
    fn fromstr_rejects_small_size() {
        assert!("3".parse::<DynamicObstaclesConfig>().is_err());
    }

    #[test]
    fn build_places_correct_number_of_obstacles() {
        let env =
            DynamicObstaclesEnv::with_config(DynamicObstaclesConfig::new(8, 4, 200, 0), false);
        assert_eq!(env.obstacles().len(), 4);
        for &(x, y) in env.obstacles() {
            assert_eq!(env.state().grid.get(x, y), Entity::Ball(OBSTACLE_COLOR));
        }
    }

    #[test]
    fn zero_obstacles_degenerates_to_empty_env() {
        let mut env = env_no_obstacles();
        env.reset().unwrap();
        assert!(env.obstacles().is_empty());
        // Scripted rollout reaches the goal at (3, 3) from (1, 1).
        let script = [
            GridAction::Forward,   // (2, 1)
            GridAction::Forward,   // (3, 1)
            GridAction::TurnRight, // S
            GridAction::Forward,   // (3, 2)
            GridAction::Forward,   // (3, 3) goal
        ];
        let mut last = None;
        for a in script {
            last = Some(env.step(a).unwrap());
        }
        let snap = last.unwrap();
        assert!(snap.is_done());
        let reward: f32 = (*snap.reward()).into();
        assert!(reward > 0.9);
    }

    #[test]
    fn reset_is_deterministic() {
        let cfg = DynamicObstaclesConfig::new(6, 3, 200, 42);
        let mut a = DynamicObstaclesEnv::with_config(cfg, false);
        let mut b = DynamicObstaclesEnv::with_config(cfg, false);
        let _ = a.reset().unwrap();
        let _ = b.reset().unwrap();
        assert_eq!(a.obstacles(), b.obstacles());
    }

    #[test]
    fn ball_blocks_forward_step() {
        let mut env = env_no_obstacles();
        env.reset().unwrap();
        // Manually place a ball directly in front of the agent.
        env.state.grid.set(2, 1, Entity::Ball(OBSTACLE_COLOR));
        let snap = env.step(GridAction::Forward).unwrap();
        assert!(!snap.is_done());
        assert_eq!(env.state().agent.x, 1, "ball should have blocked movement");
    }

    #[test]
    fn episode_terminates_within_budget_with_obstacles() {
        // Even with a small budget, the episode must cleanly terminate.
        let cfg = DynamicObstaclesConfig::new(6, 2, 50, 7);
        let mut env = DynamicObstaclesEnv::with_config(cfg, false);
        env.reset().unwrap();
        let mut done = false;
        for _ in 0..60 {
            let snap = env.step(GridAction::TurnLeft).unwrap();
            if snap.is_done() {
                done = true;
                break;
            }
        }
        assert!(done, "episode must terminate within budget");
    }

    #[test]
    fn obstacle_move_can_collide_with_static_agent() {
        // Craft a configuration where a ball sits next to the agent and
        // has no legal move except onto the agent's cell.
        let cfg = DynamicObstaclesConfig::new(5, 0, 100, 0);
        let mut env = DynamicObstaclesEnv::with_config(cfg, false);
        env.reset().unwrap();
        // Box the ball in with walls on 3 sides so the only legal move
        // is onto the agent.
        env.state.grid.set(2, 1, Entity::Ball(OBSTACLE_COLOR));
        env.state.grid.set(3, 1, Entity::Wall);
        env.state.grid.set(2, 2, Entity::Wall);
        env.obstacles = vec![(2, 1)];
        // Face the agent east so (1, 1) → front (2, 1) is the ball — but
        // for collision we need the ball to step west onto the agent at
        // (1, 1). The agent stays put with TurnLeft.
        let snap = env.step(GridAction::TurnLeft).unwrap();
        assert!(snap.is_done(), "collision should terminate the episode");
        let reward: f32 = (*snap.reward()).into();
        assert_eq!(reward, COLLISION_REWARD);
    }

    #[test]
    fn reset_clears_step_counter_and_obstacles() {
        let cfg = DynamicObstaclesConfig::new(6, 2, 200, 5);
        let mut env = DynamicObstaclesEnv::with_config(cfg, false);
        env.reset().unwrap();
        let initial = env.obstacles().to_vec();
        env.step(GridAction::TurnLeft).unwrap();
        assert_eq!(env.steps(), 1);
        env.reset().unwrap();
        assert_eq!(env.steps(), 0);
        // Deterministic: obstacles should be at the same starting positions.
        assert_eq!(env.obstacles(), initial.as_slice());
    }

    #[test]
    fn display_contains_obstacle_count() {
        let env =
            DynamicObstaclesEnv::with_config(DynamicObstaclesConfig::new(6, 3, 100, 0), false);
        let s = env.to_string();
        assert!(s.contains("num_obstacles=3"));
    }
}
