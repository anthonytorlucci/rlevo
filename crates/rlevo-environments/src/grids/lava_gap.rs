//! `LavaGap`: cross a vertical lava column through a single gap.
//!
//! Ports Farama Minigrid's [`LavaGapEnv`]. A vertical strip of [`Lava`]
//! bisects the room with one empty cell forming the crossing. Stepping
//! into lava terminates the episode with reward `0.0`; reaching the goal
//! pays [`success_reward`].
//!
//! ## Layout (5 × 5 default)
//!
//! ```text
//! # # # # #
//! # A ~ . #    ~ = Lava
//! # . . . #    A = agent, start (1, 1) facing East
//! # . ~ G #    G = Goal (3, 3)
//! # # # # #    # = wall
//! ```
//!
//! The lava strip occupies `x = size / 2`; the single gap sits at
//! `y = size / 2`. The goal is always at `(size-2, size-2)`.
//!
//! | Observation | 7 × 7 egocentric grid encoded as `[type, color, state]` per cell  |
//! |-------------|---------------------------------------------------------------------|
//! | Action      | `TurnLeft`, `TurnRight`, `Forward`                                  |
//! | Reward      | `success_reward(steps, max_steps)` on goal; `0.0` on lava / timeout |
//!
//! # Examples
//!
//! ```no_run
//! use rlevo_envs::grids::lava_gap::{LavaGapConfig, LavaGapEnv};
//! use rlevo_core::environment::Environment;
//!
//! let cfg = LavaGapConfig::new(5, 100, 0);
//! let mut env = LavaGapEnv::with_config(cfg, false);
//! let snap = env.reset().unwrap();
//! println!("lava col: {}, gap row: {}", env.lava_col(), env.gap_row());
//! ```
//!
//! [`LavaGapEnv`]: https://minigrid.farama.org/environments/minigrid/LavaGapEnv/
//! [`Lava`]: super::core::entity::Entity::Lava

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

/// Minimum grid side length; the lava column needs at least one interior
/// cell on each side.
const MIN_SIZE: usize = 5;

/// Configuration for [`LavaGapEnv`].
///
/// # Examples
///
/// ```no_run
/// use rlevo_envs::grids::lava_gap::LavaGapConfig;
///
/// let cfg = LavaGapConfig::new(7, 200, 42);
/// assert_eq!(cfg.size, 7);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LavaGapConfig {
    /// Grid side length in cells (width = height = `size`); must be ≥ `MIN_SIZE` (5).
    ///
    /// The lava column sits at `x = size / 2` and the gap at `y = size / 2`.
    pub size: usize,
    /// Maximum steps before the episode times out with reward `0.0`.
    pub max_steps: usize,
    /// RNG seed; reserved for future stochastic placement variants.
    pub seed: u64,
}

impl LavaGapConfig {
    /// Creates a [`LavaGapConfig`] with the given parameters.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rlevo_envs::grids::lava_gap::LavaGapConfig;
    ///
    /// let cfg = LavaGapConfig::new(5, 100, 0);
    /// assert_eq!(cfg.max_steps, 100);
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

impl Default for LavaGapConfig {
    fn default() -> Self {
        let size = 5;
        Self {
            size,
            max_steps: 4 * size * size,
            seed: 0,
        }
    }
}

impl FromStr for LavaGapConfig {
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

/// Minigrid's `LavaGap` environment.
///
/// A vertical lava strip bisects the room; the agent must navigate
/// through the single gap to reach the goal in the opposite corner.
/// Stepping into lava ends the episode immediately with reward `0.0`.
///
/// Implements [`Environment<3, 3, 1>`] with [`GridState`] /
/// [`GridObservation`] / [`GridAction`] / [`ScalarReward`].
///
/// # Examples
///
/// ```no_run
/// use rlevo_envs::grids::lava_gap::LavaGapEnv;
/// use rlevo_core::environment::Environment;
///
/// let mut env = LavaGapEnv::new(false);
/// let snap = env.reset().unwrap();
/// println!("lava col: {}, gap row: {}", env.lava_col(), env.gap_row());
/// ```
#[derive(Debug)]
pub struct LavaGapEnv {
    state: GridState,
    config: LavaGapConfig,
    steps: usize,
    render: bool,
    _rng: StdRng,
}

impl LavaGapEnv {
    /// Constructs a [`LavaGapEnv`] from an explicit configuration.
    #[must_use]
    pub fn with_config(config: LavaGapConfig, render: bool) -> Self {
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
    pub const fn config(&self) -> &LavaGapConfig {
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

    /// Column index where the lava strip sits.
    #[must_use]
    pub fn lava_col(&self) -> i32 {
        #[allow(clippy::cast_possible_wrap)]
        let col = (self.config.size / 2) as i32;
        col
    }

    /// Row index of the single gap in the lava strip.
    #[must_use]
    pub fn gap_row(&self) -> i32 {
        #[allow(clippy::cast_possible_wrap)]
        let row = (self.config.size / 2) as i32;
        row
    }

    /// Renders the current grid state as an ASCII string.
    #[must_use]
    pub fn ascii(&self) -> String {
        render_ascii(&self.state.grid, &self.state.agent)
    }

    fn build(config: &LavaGapConfig) -> GridState {
        let mut grid = Grid::new(config.size, config.size);
        grid.draw_walls();

        #[allow(clippy::cast_possible_wrap)]
        let lava_col = (config.size / 2) as i32;
        #[allow(clippy::cast_possible_wrap)]
        let gap_row = (config.size / 2) as i32;
        #[allow(clippy::cast_possible_wrap)]
        let height = (config.size - 1) as i32;
        for y in 1..height {
            if y != gap_row {
                grid.set(lava_col, y, Entity::Lava);
            }
        }

        #[allow(clippy::cast_possible_wrap)]
        let gx = (config.size - 2) as i32;
        #[allow(clippy::cast_possible_wrap)]
        let gy = (config.size - 2) as i32;
        grid.set(gx, gy, Entity::Goal);

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

impl Display for LavaGapEnv {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LavaGapEnv(size={}, step={}/{})",
            self.config.size, self.steps, self.config.max_steps
        )
    }
}

impl Environment<3, 3, 1> for LavaGapEnv {
    type StateType = GridState;
    type ObservationType = super::core::GridObservation;
    type ActionType = GridAction;
    type RewardType = ScalarReward;
    type SnapshotType = GridSnapshot;

    fn new(render: bool) -> Self {
        Self::with_config(LavaGapConfig::default(), render)
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

    fn env_5x5() -> LavaGapEnv {
        LavaGapEnv::with_config(LavaGapConfig::new(5, 100, 0), false)
    }

    #[test]
    fn default_config_values() {
        let cfg = LavaGapConfig::default();
        assert_eq!(cfg.size, 5);
        assert_eq!(cfg.max_steps, 4 * 5 * 5);
    }

    #[test]
    fn fromstr_keyvalue() {
        let cfg: LavaGapConfig = "size=6,max_steps=40".parse().unwrap();
        assert_eq!(cfg.size, 6);
        assert_eq!(cfg.max_steps, 40);
    }

    #[test]
    fn fromstr_rejects_small_size() {
        assert!("3".parse::<LavaGapConfig>().is_err());
    }

    #[test]
    fn build_places_lava_strip_with_gap() {
        let env = env_5x5();
        assert_eq!(env.state().grid.get(2, 1), Entity::Lava);
        assert_eq!(env.state().grid.get(2, 3), Entity::Lava);
        assert_eq!(env.state().grid.get(2, 2), Entity::Empty);
        assert_eq!(env.state().grid.get(3, 3), Entity::Goal);
    }

    #[test]
    fn reset_is_deterministic() {
        let cfg = LavaGapConfig::new(5, 100, 11);
        let mut a = LavaGapEnv::with_config(cfg, false);
        let mut b = LavaGapEnv::with_config(cfg, false);
        let sa = a.reset().unwrap();
        let sb = b.reset().unwrap();
        assert_eq!(sa.observation(), sb.observation());
    }

    #[test]
    fn stepping_into_lava_terminates_with_zero_reward() {
        let mut env = env_5x5();
        env.reset().unwrap();
        // Agent at (1,1) facing East. Directly forward is (2,1) = lava.
        let snap = env.step(GridAction::Forward).unwrap();
        assert!(snap.is_done());
        let reward: f32 = (*snap.reward()).into();
        assert_eq!(reward, 0.0);
    }

    #[test]
    fn optimal_rollout_crosses_gap_and_reaches_goal() {
        let mut env = env_5x5();
        env.reset().unwrap();
        // Navigate south to the gap row, cross east, then south to the goal.
        let script = [
            GridAction::TurnRight, // face south
            GridAction::Forward,   // (1,2)
            GridAction::TurnLeft,  // face east
            GridAction::Forward,   // (2,2) through gap
            GridAction::Forward,   // (3,2)
            GridAction::TurnRight, // face south
            GridAction::Forward,   // (3,3) goal
        ];
        let mut last = None;
        for a in script {
            last = Some(env.step(a).unwrap());
        }
        let snap = last.unwrap();
        assert!(snap.is_done());
        let reward: f32 = (*snap.reward()).into();
        assert!(reward > 0.0, "reward was {reward}");
    }

    #[test]
    fn timeout_ends_with_zero_reward() {
        let cfg = LavaGapConfig::new(5, 3, 0);
        let mut env = LavaGapEnv::with_config(cfg, false);
        env.reset().unwrap();
        env.step(GridAction::TurnLeft).unwrap();
        env.step(GridAction::TurnLeft).unwrap();
        let snap = env.step(GridAction::TurnLeft).unwrap();
        assert!(snap.is_done());
        let reward: f32 = (*snap.reward()).into();
        assert_eq!(reward, 0.0);
    }

    #[test]
    fn lava_col_and_gap_row_match_config() {
        let env = env_5x5();
        assert_eq!(env.lava_col(), 2);
        assert_eq!(env.gap_row(), 2);
    }

    #[test]
    fn reset_clears_step_counter() {
        let mut env = env_5x5();
        env.reset().unwrap();
        env.step(GridAction::TurnLeft).unwrap();
        env.step(GridAction::TurnLeft).unwrap();
        assert_eq!(env.steps(), 2);
        env.reset().unwrap();
        assert_eq!(env.steps(), 0);
    }
}
