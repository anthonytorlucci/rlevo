//! `Empty`: reach the green goal tile in an empty walled room.
//!
//! Ports Farama Minigrid's [`EmptyEnv`]. The grid is a `size × size` room
//! with a wall perimeter, a [`Goal`] tile at `(size - 2, size - 2)`, and
//! the agent starting at `(1, 1)` facing East. Stepping onto the goal
//! terminates the episode and pays [`success_reward`]; exceeding
//! `max_steps` terminates with reward `0.0`.
//!
//! [`EmptyEnv`]: https://minigrid.farama.org/environments/minigrid/EmptyEnv/
//! [`Goal`]: super::core::entity::Entity::Goal

use super::core::{
    action::GridAction,
    agent::AgentState,
    direction::Direction,
    dynamics::{StepOutcome, apply_action},
    entity::Entity,
    grid::Grid,
    observation::GridObservation,
    reward::success_reward,
    state::GridState,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rlevo_core::base::State;
use rlevo_core::environment::{Environment, EnvironmentError, SnapshotBase};
use rlevo_core::reward::ScalarReward;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

/// Minimum grid side length; smaller grids can't host both an agent
/// start cell and a distinct goal.
const MIN_SIZE: usize = 4;

/// Configuration for [`EmptyEnv`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmptyConfig {
    /// Grid side length in cells (including perimeter walls).
    pub size: usize,
    /// Maximum number of steps before the episode times out.
    pub max_steps: usize,
    /// Seed for the environment's RNG. Empty is deterministic, but the
    /// RNG slot is reserved for future random-spawn variants and so that
    /// all grid envs share the same config surface.
    pub seed: u64,
}

impl EmptyConfig {
    /// Construct a new config.
    #[must_use]
    pub const fn new(size: usize, max_steps: usize, seed: u64) -> Self {
        Self {
            size,
            max_steps,
            seed,
        }
    }
}

impl Default for EmptyConfig {
    fn default() -> Self {
        let size = 8;
        Self {
            size,
            max_steps: 4 * size * size,
            seed: 0,
        }
    }
}

impl FromStr for EmptyConfig {
    type Err = String;

    /// Parse a config from a comma-separated list.
    ///
    /// Accepts positional values (`"5"`, `"5,100"`, `"5,100,42"`) and
    /// `key=value` pairs (`"size=5,max_steps=100,seed=42"`). Unknown keys
    /// and sizes below [`MIN_SIZE`] produce an error.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut cfg = Self::default();
        for (i, raw) in s.trim().split(',').map(str::trim).enumerate() {
            if raw.is_empty() {
                continue;
            }
            if let Some((key, value)) = raw.split_once('=') {
                apply_kv(&mut cfg, key.trim(), value.trim())?;
            } else {
                apply_positional(&mut cfg, i, raw)?;
            }
        }
        if cfg.size < MIN_SIZE {
            return Err(format!("size must be >= {MIN_SIZE}, got {}", cfg.size));
        }
        Ok(cfg)
    }
}

fn apply_kv(cfg: &mut EmptyConfig, key: &str, value: &str) -> Result<(), String> {
    match key {
        "size" => cfg.size = value.parse().map_err(|e| format!("size: {e}"))?,
        "max_steps" => cfg.max_steps = value.parse().map_err(|e| format!("max_steps: {e}"))?,
        "seed" => cfg.seed = value.parse().map_err(|e| format!("seed: {e}"))?,
        other => return Err(format!("unknown key `{other}`")),
    }
    Ok(())
}

fn apply_positional(cfg: &mut EmptyConfig, index: usize, value: &str) -> Result<(), String> {
    match index {
        0 => cfg.size = value.parse().map_err(|e| format!("size: {e}"))?,
        1 => cfg.max_steps = value.parse().map_err(|e| format!("max_steps: {e}"))?,
        2 => cfg.seed = value.parse().map_err(|e| format!("seed: {e}"))?,
        _ => return Err(format!("unexpected positional value `{value}`")),
    }
    Ok(())
}

/// Minigrid's `Empty` environment: reach the green goal tile.
#[derive(Debug)]
pub struct EmptyEnv {
    state: GridState,
    config: EmptyConfig,
    steps: usize,
    render: bool,
    /// RNG slot held for parity with random-spawn variants.
    _rng: StdRng,
}

impl EmptyEnv {
    /// Construct an [`EmptyEnv`] from an explicit configuration.
    #[must_use]
    pub fn with_config(config: EmptyConfig, render: bool) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        let (grid, agent) = Self::build(&config);
        Self {
            state: GridState::new(grid, agent),
            config,
            steps: 0,
            render,
            _rng: rng,
        }
    }

    /// Borrow the active configuration.
    #[must_use]
    pub const fn config(&self) -> &EmptyConfig {
        &self.config
    }

    /// Current step count within the episode.
    #[must_use]
    pub const fn steps(&self) -> usize {
        self.steps
    }

    /// Borrow the full grid + agent state (useful in tests).
    #[must_use]
    pub const fn state(&self) -> &GridState {
        &self.state
    }

    fn build(config: &EmptyConfig) -> (Grid, AgentState) {
        let mut grid = Grid::new(config.size, config.size);
        grid.draw_walls();
        #[allow(clippy::cast_possible_wrap)]
        let gx = (config.size - 2) as i32;
        #[allow(clippy::cast_possible_wrap)]
        let gy = (config.size - 2) as i32;
        grid.set(gx, gy, Entity::Goal);
        let agent = AgentState::new(1, 1, Direction::East);
        (grid, agent)
    }

    fn snapshot(&self, reward: f32, done: bool) -> SnapshotBase<3, GridObservation, ScalarReward> {
        if self.render {
            // Render is a debug side effect; return the string so callers can
            // capture it if they wish, or drop it when invoked internally.
            let _ = super::core::render::render_ascii(&self.state.grid, &self.state.agent);
        }
        if done {
            SnapshotBase::terminated(self.state.observe(), ScalarReward::new(reward))
        } else {
            SnapshotBase::running(self.state.observe(), ScalarReward::new(reward))
        }
    }
}

impl Display for EmptyEnv {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "EmptyEnv(size={}, step={}/{})",
            self.config.size, self.steps, self.config.max_steps
        )
    }
}

impl Environment<3, 3, 1> for EmptyEnv {
    type StateType = GridState;
    type ObservationType = GridObservation;
    type ActionType = GridAction;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<3, GridObservation, ScalarReward>;

    fn new(render: bool) -> Self {
        Self::with_config(EmptyConfig::default(), render)
    }

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        let (grid, agent) = Self::build(&self.config);
        self.state = GridState::new(grid, agent);
        self.steps = 0;
        self._rng = StdRng::seed_from_u64(self.config.seed);
        Ok(self.snapshot(0.0, false))
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
        Ok(self.snapshot(reward, done))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlevo_core::action::DiscreteAction;
    use rlevo_core::base::Observation;
    use rlevo_core::environment::Snapshot;

    #[test]
    fn default_config_is_8x8_with_budget() {
        let cfg = EmptyConfig::default();
        assert_eq!(cfg.size, 8);
        assert_eq!(cfg.max_steps, 4 * 8 * 8);
        assert_eq!(cfg.seed, 0);
    }

    #[test]
    fn fromstr_positional_size_only() {
        let cfg: EmptyConfig = "5".parse().unwrap();
        assert_eq!(cfg.size, 5);
        assert_eq!(cfg.max_steps, EmptyConfig::default().max_steps);
    }

    #[test]
    fn fromstr_all_positional() {
        let cfg: EmptyConfig = "6,50,7".parse().unwrap();
        assert_eq!(cfg.size, 6);
        assert_eq!(cfg.max_steps, 50);
        assert_eq!(cfg.seed, 7);
    }

    #[test]
    fn fromstr_key_value() {
        let cfg: EmptyConfig = "size=6,max_steps=100,seed=7".parse().unwrap();
        assert_eq!(cfg.size, 6);
        assert_eq!(cfg.max_steps, 100);
        assert_eq!(cfg.seed, 7);
    }

    #[test]
    fn fromstr_rejects_small_size() {
        let err = "2".parse::<EmptyConfig>().unwrap_err();
        assert!(err.contains("size must be"));
    }

    #[test]
    fn fromstr_rejects_unknown_key() {
        let err = "wat=5".parse::<EmptyConfig>().unwrap_err();
        assert!(err.contains("unknown key"));
    }

    #[test]
    fn new_places_goal_and_agent() {
        let env = EmptyEnv::with_config(EmptyConfig::new(5, 100, 0), false);
        let grid = &env.state().grid;
        assert_eq!(grid.get(3, 3), Entity::Goal);
        assert_eq!(env.state().agent.x, 1);
        assert_eq!(env.state().agent.y, 1);
        assert_eq!(env.state().agent.direction, Direction::East);
    }

    #[test]
    fn reset_is_deterministic_for_same_seed() {
        let cfg = EmptyConfig::new(5, 100, 42);
        let mut a = EmptyEnv::with_config(cfg, false);
        let mut b = EmptyEnv::with_config(cfg, false);
        let snap_a = a.reset().unwrap();
        let snap_b = b.reset().unwrap();
        assert_eq!(snap_a.observation(), snap_b.observation());
        assert!(!snap_a.is_done());
    }

    #[test]
    fn observation_shape_is_view_sized() {
        assert_eq!(<GridObservation as Observation<3>>::shape(), [7, 7, 3]);
    }

    #[test]
    fn forward_into_wall_bumps_and_holds_position() {
        let cfg = EmptyConfig::new(5, 100, 0);
        let mut env = EmptyEnv::with_config(cfg, false);
        env.reset().unwrap();
        // Turn to face north; wall lies at (1, 0).
        env.step(GridAction::TurnLeft).unwrap();
        let _ = env.step(GridAction::Forward).unwrap();
        assert_eq!(env.state().agent.x, 1);
        assert_eq!(env.state().agent.y, 1);
    }

    #[test]
    fn optimal_rollout_reaches_goal_with_positive_reward() {
        let cfg = EmptyConfig::new(5, 100, 0);
        let mut env = EmptyEnv::with_config(cfg, false);
        env.reset().unwrap();

        // Agent at (1,1) facing East. Goal at (3,3).
        // Forward → (2,1); Forward → (3,1); TurnRight → facing South;
        // Forward → (3,2); Forward → (3,3).
        let script = [
            GridAction::Forward,
            GridAction::Forward,
            GridAction::TurnRight,
            GridAction::Forward,
            GridAction::Forward,
        ];

        let mut last = None;
        for action in script {
            last = Some(env.step(action).unwrap());
        }
        let snap = last.expect("at least one step");
        assert!(snap.is_done(), "reaching the goal should terminate");
        let reward: f32 = (*snap.reward()).into();
        assert!(reward > 0.0, "goal reward must be positive, got {reward}");
        // 5 steps out of 100 → 1 - 0.9 * 0.05 = 0.955.
        assert!((reward - 0.955).abs() < 1e-4, "reward was {reward}");
    }

    #[test]
    fn timeout_terminates_with_zero_reward() {
        let cfg = EmptyConfig::new(5, 3, 0);
        let mut env = EmptyEnv::with_config(cfg, false);
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
        let cfg = EmptyConfig::new(5, 100, 0);
        let mut env = EmptyEnv::with_config(cfg, false);
        env.reset().unwrap();
        for _ in 0..3 {
            env.step(GridAction::TurnLeft).unwrap();
        }
        assert_eq!(env.steps(), 3);
        env.reset().unwrap();
        assert_eq!(env.steps(), 0);
    }

    #[test]
    fn random_policy_never_errors() {
        // Sanity check that `step` is total under all 7 actions.
        let cfg = EmptyConfig::new(5, 50, 0);
        let mut env = EmptyEnv::with_config(cfg, false);
        env.reset().unwrap();
        for i in 0..50 {
            let action = GridAction::from_index(i % GridAction::ACTION_COUNT);
            let snap = env.step(action).unwrap();
            if snap.is_done() {
                break;
            }
        }
    }

    #[test]
    fn display_contains_step_budget() {
        let env = EmptyEnv::with_config(EmptyConfig::new(5, 50, 0), false);
        let s = env.to_string();
        assert!(s.contains("EmptyEnv"));
        assert!(s.contains("50"));
    }
}
