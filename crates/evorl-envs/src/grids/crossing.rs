//! `Crossing`: navigate across `K` obstacle strips with one gap each.
//!
//! Ports Farama Minigrid's [`CrossingEnv`]. The room is sliced by `K`
//! horizontal strips — either [`Lava`] or [`Wall`] depending on
//! [`CrossingKind`] — with a single gap per strip. A gap column is
//! shared across all strips so there is always a straight vertical
//! corridor connecting the start and goal.
//!
//! [`CrossingEnv`]: https://minigrid.farama.org/environments/minigrid/CrossingEnv/
//! [`Lava`]: super::core::entity::Entity::Lava
//! [`Wall`]: super::core::entity::Entity::Wall

use super::core::{
    action::GridAction,
    agent::AgentState,
    build_snapshot,
    direction::Direction,
    dynamics::{apply_action, StepOutcome},
    entity::Entity,
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

/// Whether the strips are lava (terminal hazard) or walls (bumpable blockers).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CrossingKind {
    /// Terminal hazard strips; stepping in ends the episode.
    #[default]
    Lava,
    /// Impassable walls; stepping in bumps but the episode continues.
    Wall,
}

impl CrossingKind {
    /// Entity used for the blocking strip cells.
    #[must_use]
    pub const fn entity(self) -> Entity {
        match self {
            Self::Lava => Entity::Lava,
            Self::Wall => Entity::Wall,
        }
    }
}

impl FromStr for CrossingKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "lava" => Ok(Self::Lava),
            "wall" => Ok(Self::Wall),
            other => Err(format!("unknown kind `{other}`")),
        }
    }
}

/// Minimum supported side length; we need at least one interior row per strip
/// plus free space for the agent and goal.
const MIN_SIZE: usize = 7;

/// Configuration for [`CrossingEnv`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrossingConfig {
    pub size: usize,
    pub max_steps: usize,
    pub seed: u64,
    pub kind: CrossingKind,
}

impl CrossingConfig {
    #[must_use]
    pub const fn new(size: usize, max_steps: usize, seed: u64, kind: CrossingKind) -> Self {
        Self {
            size,
            max_steps,
            seed,
            kind,
        }
    }
}

impl Default for CrossingConfig {
    fn default() -> Self {
        let size = 7;
        Self {
            size,
            max_steps: 4 * size * size,
            seed: 0,
            kind: CrossingKind::Lava,
        }
    }
}

impl FromStr for CrossingConfig {
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
                    "kind" => cfg.kind = value.trim().parse()?,
                    other => return Err(format!("unknown key `{other}`")),
                }
            } else {
                match idx {
                    0 => cfg.size = raw.parse().map_err(|e| format!("size: {e}"))?,
                    1 => cfg.max_steps = raw.parse().map_err(|e| format!("max_steps: {e}"))?,
                    2 => cfg.seed = raw.parse().map_err(|e| format!("seed: {e}"))?,
                    3 => cfg.kind = raw.parse()?,
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

/// Minigrid's `Crossing` environment.
#[derive(Debug)]
pub struct CrossingEnv {
    state: GridState,
    config: CrossingConfig,
    steps: usize,
    render: bool,
    _rng: StdRng,
}

impl CrossingEnv {
    #[must_use]
    pub fn with_config(config: CrossingConfig, render: bool) -> Self {
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
    pub const fn config(&self) -> &CrossingConfig {
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

    /// Column where every strip has its open gap.
    #[must_use]
    pub fn gap_col(&self) -> i32 {
        #[allow(clippy::cast_possible_wrap)]
        let col = (self.config.size / 2) as i32;
        col
    }

    /// Rows on which the obstacle strips are placed.
    #[must_use]
    pub fn strip_rows(&self) -> Vec<i32> {
        #[allow(clippy::cast_possible_wrap)]
        let size = self.config.size as i32;
        let mid = size / 2;
        vec![mid - 1, mid + 1]
    }

    fn build(config: &CrossingConfig) -> GridState {
        let mut grid = Grid::new(config.size, config.size);
        grid.draw_walls();
        let blocker = config.kind.entity();
        #[allow(clippy::cast_possible_wrap)]
        let size = config.size as i32;
        let mid = size / 2;
        let strip_rows = [mid - 1, mid + 1];
        for &row in &strip_rows {
            for x in 1..size - 1 {
                if x != mid {
                    grid.set(x, row, blocker);
                }
            }
        }
        let goal_xy = size - 2;
        grid.set(goal_xy, goal_xy, Entity::Goal);
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

impl Display for CrossingEnv {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CrossingEnv(size={}, kind={:?}, step={}/{})",
            self.config.size, self.config.kind, self.steps, self.config.max_steps
        )
    }
}

impl Environment<3, 3, 1> for CrossingEnv {
    type StateType = GridState;
    type ObservationType = super::core::GridObservation;
    type ActionType = GridAction;
    type RewardType = ScalarReward;
    type SnapshotType = GridSnapshot;

    fn new(render: bool) -> Self {
        Self::with_config(CrossingConfig::default(), render)
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

    fn default_env(kind: CrossingKind) -> CrossingEnv {
        CrossingEnv::with_config(CrossingConfig::new(7, 196, 0, kind), false)
    }

    /// Optimal rollout that works for both lava and wall variants.
    fn optimal_script() -> [GridAction; 10] {
        [
            GridAction::Forward,   // (2,1)
            GridAction::Forward,   // (3,1)
            GridAction::TurnRight, // face south
            GridAction::Forward,   // (3,2) gap
            GridAction::Forward,   // (3,3)
            GridAction::Forward,   // (3,4) gap
            GridAction::Forward,   // (3,5)
            GridAction::TurnLeft,  // face east
            GridAction::Forward,   // (4,5)
            GridAction::Forward,   // (5,5) goal
        ]
    }

    #[test]
    fn default_config_is_lava() {
        let cfg = CrossingConfig::default();
        assert_eq!(cfg.kind, CrossingKind::Lava);
        assert_eq!(cfg.size, 7);
    }

    #[test]
    fn fromstr_rejects_small_size() {
        assert!("5".parse::<CrossingConfig>().is_err());
    }

    #[test]
    fn fromstr_kind_parses() {
        let cfg: CrossingConfig = "kind=wall".parse().unwrap();
        assert_eq!(cfg.kind, CrossingKind::Wall);
    }

    #[test]
    fn build_places_lava_strips_with_gap() {
        let env = default_env(CrossingKind::Lava);
        // Strip rows for size 7 with mid = 3 → rows 2 and 4.
        for x in 1..=5 {
            if x == 3 {
                assert_eq!(env.state().grid.get(x, 2), Entity::Empty);
                assert_eq!(env.state().grid.get(x, 4), Entity::Empty);
            } else {
                assert_eq!(env.state().grid.get(x, 2), Entity::Lava);
                assert_eq!(env.state().grid.get(x, 4), Entity::Lava);
            }
        }
        assert_eq!(env.state().grid.get(5, 5), Entity::Goal);
    }

    #[test]
    fn wall_variant_uses_walls_instead_of_lava() {
        let env = default_env(CrossingKind::Wall);
        assert_eq!(env.state().grid.get(1, 2), Entity::Wall);
        assert_eq!(env.state().grid.get(1, 4), Entity::Wall);
    }

    #[test]
    fn optimal_rollout_solves_lava_variant() {
        let mut env = default_env(CrossingKind::Lava);
        env.reset().unwrap();
        let mut last = None;
        for a in optimal_script() {
            last = Some(env.step(a).unwrap());
        }
        let snap = last.unwrap();
        assert!(snap.is_done());
        let reward: f32 = (*snap.reward()).into();
        assert!(reward > 0.9, "reward was {reward}");
    }

    #[test]
    fn optimal_rollout_solves_wall_variant() {
        let mut env = default_env(CrossingKind::Wall);
        env.reset().unwrap();
        let mut last = None;
        for a in optimal_script() {
            last = Some(env.step(a).unwrap());
        }
        let snap = last.unwrap();
        assert!(snap.is_done());
        let reward: f32 = (*snap.reward()).into();
        assert!(reward > 0.9);
    }

    #[test]
    fn stepping_onto_lava_strip_ends_episode() {
        let mut env = default_env(CrossingKind::Lava);
        env.reset().unwrap();
        // (1,1) E → TurnRight → S → Forward → (1,2) = Lava
        env.step(GridAction::TurnRight).unwrap();
        let snap = env.step(GridAction::Forward).unwrap();
        assert!(snap.is_done());
        let reward: f32 = (*snap.reward()).into();
        assert_eq!(reward, 0.0);
    }

    #[test]
    fn walking_into_wall_strip_only_bumps() {
        let mut env = default_env(CrossingKind::Wall);
        env.reset().unwrap();
        env.step(GridAction::TurnRight).unwrap();
        let snap = env.step(GridAction::Forward).unwrap();
        assert!(!snap.is_done(), "walls should not terminate the episode");
        assert_eq!(env.state().agent.y, 1, "agent should not have moved");
    }

    #[test]
    fn reset_is_deterministic() {
        let cfg = CrossingConfig::new(7, 100, 5, CrossingKind::Wall);
        let mut a = CrossingEnv::with_config(cfg, false);
        let mut b = CrossingEnv::with_config(cfg, false);
        let sa = a.reset().unwrap();
        let sb = b.reset().unwrap();
        assert_eq!(sa.observation(), sb.observation());
    }

    #[test]
    fn gap_col_and_strip_rows_match_config() {
        let env = default_env(CrossingKind::Lava);
        assert_eq!(env.gap_col(), 3);
        assert_eq!(env.strip_rows(), vec![2, 4]);
    }
}
