//! `Memory`: cue object shown at start, matched at a fork.
//!
//! Ports Farama Minigrid's [`MemoryEnv`]. The agent starts facing a
//! `cue` object ([`Key`] in the current implementation) and must
//! navigate a short corridor to a fork where two objects sit — one
//! matching the cue, one not. Emitting [`GridAction::Done`] while
//! facing the matching object pays [`success_reward`]; emitting it
//! while facing the wrong object or anywhere else ends the episode
//! with reward `0.0`.
//!
//! The egocentric 7×7 observation stops including the cue after a few
//! steps, so a successful policy must *remember* what it saw at the
//! start of the episode — the reason this env exists.
//!
//! The [`MemoryConfig::swap_fork`] flag flips which side of the fork
//! holds the matching object, giving test suites and agent trainers a
//! two-valued distribution over answers.
//!
//! ## Layout (7 × 5, fixed)
//!
//! ```text
//! # # # # # # #
//! # . . . # K #    K = Key (yellow) — match object at fork
//! # K ← . . . #    ← = agent, start (2, 2) facing West; K = cue at (1, 2)
//! # . . . # B #    B = Ball (red)  — distractor at fork
//! # # # # # # #    # = wall; interior wall at x = 4, rows 1 and 3
//! ```
//!
//! With `swap_fork = true` the Key moves to row 3 and the Ball moves to row 1.
//!
//! | Observation | 7 × 7 egocentric grid encoded as `[type, color, state]` per cell |
//! |-------------|------------------------------------------------------------------|
//! | Action      | `TurnLeft`, `TurnRight`, `Forward`, `Done`                       |
//! | Reward      | `success_reward(steps, max_steps)` on correct Done; else `0.0`   |
//!
//! # Examples
//!
//! ```rust
//! use rlevo_environments::grids::memory::{MemoryConfig, MemoryEnv};
//! use rlevo_core::environment::Environment;
//!
//! let cfg = MemoryConfig::new(140, 0, false);
//! let mut env = MemoryEnv::with_config(cfg, false);
//! let snap = env.reset().unwrap();
//! println!("match pos: {:?}", env.match_pos());
//! ```
//!
//! [`MemoryEnv`]: https://minigrid.farama.org/environments/minigrid/MemoryEnv/
//! [`Key`]: super::core::entity::Entity::Key

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
use rand::SeedableRng;
use rand::rngs::StdRng;
use rlevo_core::environment::{Environment, EnvironmentError};
use rlevo_core::reward::ScalarReward;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

/// Fixed grid width.
const WIDTH: usize = 7;
/// Fixed grid height.
const HEIGHT: usize = 5;
/// Color used for the (matching) cue object.
const CUE_COLOR: Color = Color::Yellow;
/// Color used for the distractor object at the fork.
const DISTRACTOR_COLOR: Color = Color::Red;

/// Configuration for [`MemoryEnv`].
///
/// # Examples
///
/// ```run
/// use rlevo_environments::grids::memory::MemoryConfig;
///
/// let cfg = MemoryConfig::new(140, 42, true);
/// assert!(cfg.swap_fork);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Maximum steps before the episode times out with reward `0.0`.
    pub max_steps: usize,
    /// RNG seed; reserved for future stochastic variants.
    pub seed: u64,
    /// When `true` the matching object sits at the bottom fork position
    /// instead of the top. This is the only source of variation: fix it
    /// to test a specific rollout; vary it between episodes to train a
    /// real memory-based policy.
    pub swap_fork: bool,
}

impl MemoryConfig {
    /// Creates a [`MemoryConfig`] with the given parameters.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rlevo_environments::grids::memory::MemoryConfig;
    ///
    /// let cfg = MemoryConfig::new(140, 0, false);
    /// assert_eq!(cfg.max_steps, 140);
    /// ```
    #[must_use]
    pub const fn new(max_steps: usize, seed: u64, swap_fork: bool) -> Self {
        Self {
            max_steps,
            seed,
            swap_fork,
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_steps: 4 * WIDTH * HEIGHT,
            seed: 0,
            swap_fork: false,
        }
    }
}

fn parse_bool(s: &str) -> Result<bool, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" => Ok(true),
        "false" | "0" | "no" => Ok(false),
        other => Err(format!("expected bool, got `{other}`")),
    }
}

impl FromStr for MemoryConfig {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut cfg = Self::default();
        for (idx, raw) in s.trim().split(',').map(str::trim).enumerate() {
            if raw.is_empty() {
                continue;
            }
            if let Some((key, value)) = raw.split_once('=') {
                match key.trim() {
                    "max_steps" => {
                        cfg.max_steps = value
                            .trim()
                            .parse()
                            .map_err(|e| format!("max_steps: {e}"))?;
                    }
                    "seed" => cfg.seed = value.trim().parse().map_err(|e| format!("seed: {e}"))?,
                    "swap_fork" | "swap" => cfg.swap_fork = parse_bool(value)?,
                    other => return Err(format!("unknown key `{other}`")),
                }
            } else {
                match idx {
                    0 => cfg.max_steps = raw.parse().map_err(|e| format!("max_steps: {e}"))?,
                    1 => cfg.seed = raw.parse().map_err(|e| format!("seed: {e}"))?,
                    2 => cfg.swap_fork = parse_bool(raw)?,
                    _ => return Err(format!("unexpected positional value `{raw}`")),
                }
            }
        }
        Ok(cfg)
    }
}

/// Minigrid's `Memory` environment.
///
/// The agent must observe a cue object at the episode start, navigate
/// a corridor, and select the matching object at a fork by issuing
/// [`GridAction::Done`] while facing it. Because the cue leaves the
/// egocentric field of view before the fork is reached, a successful
/// policy must maintain an internal memory of what it observed.
///
/// Implements [`Environment<3, 3, 1>`] with [`GridState`] /
/// [`GridObservation`](super::core::GridObservation) / [`GridAction`] / [`ScalarReward`].
///
/// # Examples
///
/// ```rust
/// use rlevo_environments::grids::memory::MemoryEnv;
/// use rlevo_core::environment::Environment;
///
/// let mut env = MemoryEnv::new(false);
/// let snap = env.reset().unwrap();
/// println!("match pos: {:?}", env.match_pos());
/// ```
#[derive(Debug)]
pub struct MemoryEnv {
    state: GridState,
    config: MemoryConfig,
    steps: usize,
    render: bool,
    /// World coordinates of the matching object at the fork.
    match_pos: (i32, i32),
    _rng: StdRng,
}

impl MemoryEnv {
    /// Constructs a [`MemoryEnv`] from an explicit configuration.
    #[must_use]
    pub fn with_config(config: MemoryConfig, render: bool) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        let (state, match_pos) = Self::build(&config);
        Self {
            state,
            config,
            steps: 0,
            render,
            match_pos,
            _rng: rng,
        }
    }

    /// Returns the environment's active configuration.
    #[must_use]
    pub const fn config(&self) -> &MemoryConfig {
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

    /// Position of the matching object at the fork.
    #[must_use]
    pub const fn match_pos(&self) -> (i32, i32) {
        self.match_pos
    }

    /// Renders the current grid state as an ASCII string.
    #[must_use]
    pub fn ascii(&self) -> String {
        render_ascii(&self.state.grid, &self.state.agent)
    }

    fn build(config: &MemoryConfig) -> (GridState, (i32, i32)) {
        let mut grid = Grid::new(WIDTH, HEIGHT);
        grid.draw_walls();

        // Interior wall at col 4 with a gap at row 2.
        grid.set(4, 1, Entity::Wall);
        grid.set(4, 3, Entity::Wall);
        // (4, 2) stays empty — that's the gap.

        // Cue object at (1, 2), visible from the agent's starting view.
        grid.set(1, 2, Entity::Key(CUE_COLOR));

        // Fork objects in the right room.
        let (match_pos, distractor_pos) = if config.swap_fork {
            ((5, 3), (5, 1))
        } else {
            ((5, 1), (5, 3))
        };
        grid.set(match_pos.0, match_pos.1, Entity::Key(CUE_COLOR));
        grid.set(
            distractor_pos.0,
            distractor_pos.1,
            Entity::Ball(DISTRACTOR_COLOR),
        );

        // Agent sits east of the cue, facing it so the cue shows up in
        // the egocentric view at the start of the episode.
        let agent = AgentState::new(2, 2, Direction::West);
        (GridState::new(grid, agent), match_pos)
    }

    fn emit(&self, reward: f32, done: bool) -> GridSnapshot {
        if self.render {
            println!("{}", self.ascii());
        }
        build_snapshot(&self.state, reward, done)
    }

    fn facing_match(&self) -> bool {
        let (fx, fy) = self.state.agent.front();
        (fx, fy) == self.match_pos
    }
}

impl Display for MemoryEnv {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MemoryEnv(swap_fork={}, step={}/{})",
            self.config.swap_fork, self.steps, self.config.max_steps
        )
    }
}

impl Environment<3, 3, 1> for MemoryEnv {
    type StateType = GridState;
    type ObservationType = super::core::GridObservation;
    type ActionType = GridAction;
    type RewardType = ScalarReward;
    type SnapshotType = GridSnapshot;

    fn new(render: bool) -> Self {
        Self::with_config(MemoryConfig::default(), render)
    }

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        let (state, match_pos) = Self::build(&self.config);
        self.state = state;
        self.match_pos = match_pos;
        self.steps = 0;
        self._rng = StdRng::seed_from_u64(self.config.seed);
        Ok(self.emit(0.0, false))
    }

    fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> {
        self.steps += 1;
        let outcome = apply_action(&mut self.state.grid, &mut self.state.agent, action);
        let (reward, done) = match outcome {
            StepOutcome::DoneAction => {
                if self.facing_match() {
                    (success_reward(self.steps, self.config.max_steps), true)
                } else {
                    (0.0, true)
                }
            }
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

    #[test]
    fn default_config_values() {
        let cfg = MemoryConfig::default();
        assert_eq!(cfg.max_steps, 4 * WIDTH * HEIGHT);
        assert!(!cfg.swap_fork);
    }

    #[test]
    fn fromstr_parses_bool_flag() {
        let cfg: MemoryConfig = "swap_fork=true".parse().unwrap();
        assert!(cfg.swap_fork);
    }

    #[test]
    fn fromstr_rejects_unknown_bool() {
        assert!("swap_fork=maybe".parse::<MemoryConfig>().is_err());
    }

    #[test]
    fn build_default_has_match_at_top() {
        let env = MemoryEnv::with_config(MemoryConfig::default(), false);
        assert_eq!(env.match_pos(), (5, 1));
        assert_eq!(env.state().grid.get(5, 1), Entity::Key(CUE_COLOR));
        assert_eq!(env.state().grid.get(5, 3), Entity::Ball(DISTRACTOR_COLOR));
        assert_eq!(env.state().grid.get(1, 2), Entity::Key(CUE_COLOR));
        assert_eq!(env.state().agent.direction, Direction::West);
    }

    #[test]
    fn build_with_swap_fork_moves_match_to_bottom() {
        let env = MemoryEnv::with_config(MemoryConfig::new(140, 0, true), false);
        assert_eq!(env.match_pos(), (5, 3));
        assert_eq!(env.state().grid.get(5, 1), Entity::Ball(DISTRACTOR_COLOR));
        assert_eq!(env.state().grid.get(5, 3), Entity::Key(CUE_COLOR));
    }

    #[test]
    fn optimal_rollout_default_picks_top_fork() {
        let mut env = MemoryEnv::with_config(MemoryConfig::new(140, 0, false), false);
        env.reset().unwrap();
        let script = [
            GridAction::TurnRight, // W → N
            GridAction::TurnRight, // N → E
            GridAction::Forward,   // (3, 2)
            GridAction::Forward,   // (4, 2) through gap
            GridAction::Forward,   // (5, 2)
            GridAction::TurnLeft,  // E → N, facing (5, 1) = match
            GridAction::Done,
        ];
        let mut last = None;
        for a in script {
            last = Some(env.step(a).unwrap());
        }
        let snap = last.unwrap();
        assert!(snap.is_done());
        let reward: f32 = (*snap.reward()).into();
        assert!(reward > 0.8, "reward was {reward}");
    }

    #[test]
    fn optimal_rollout_swapped_picks_bottom_fork() {
        let mut env = MemoryEnv::with_config(MemoryConfig::new(140, 0, true), false);
        env.reset().unwrap();
        let script = [
            GridAction::TurnRight,
            GridAction::TurnRight,
            GridAction::Forward,   // (3, 2)
            GridAction::Forward,   // (4, 2)
            GridAction::Forward,   // (5, 2)
            GridAction::TurnRight, // E → S, facing (5, 3) = match
            GridAction::Done,
        ];
        let mut last = None;
        for a in script {
            last = Some(env.step(a).unwrap());
        }
        let snap = last.unwrap();
        assert!(snap.is_done());
        let reward: f32 = (*snap.reward()).into();
        assert!(reward > 0.8);
    }

    #[test]
    fn done_at_distractor_terminates_with_zero() {
        let mut env = MemoryEnv::with_config(MemoryConfig::new(140, 0, false), false);
        env.reset().unwrap();
        let script = [
            GridAction::TurnRight,
            GridAction::TurnRight,
            GridAction::Forward,
            GridAction::Forward,
            GridAction::Forward,
            GridAction::TurnRight, // facing (5, 3) = distractor
            GridAction::Done,
        ];
        let mut last = None;
        for a in script {
            last = Some(env.step(a).unwrap());
        }
        let snap = last.unwrap();
        assert!(snap.is_done());
        let reward: f32 = (*snap.reward()).into();
        assert_eq!(reward, 0.0);
    }

    #[test]
    fn done_in_empty_corridor_terminates_with_zero() {
        let mut env = MemoryEnv::with_config(MemoryConfig::new(140, 0, false), false);
        env.reset().unwrap();
        let snap = env.step(GridAction::Done).unwrap();
        assert!(snap.is_done());
        let reward: f32 = (*snap.reward()).into();
        assert_eq!(reward, 0.0);
    }

    #[test]
    fn interior_wall_blocks_diagonal_cell() {
        let env = MemoryEnv::with_config(MemoryConfig::default(), false);
        assert_eq!(env.state().grid.get(4, 1), Entity::Wall);
        assert_eq!(env.state().grid.get(4, 3), Entity::Wall);
        assert_eq!(env.state().grid.get(4, 2), Entity::Empty);
    }

    #[test]
    fn navigating_to_distractor_does_not_face_match() {
        let mut env = MemoryEnv::with_config(MemoryConfig::default(), false);
        env.reset().unwrap();
        // Navigate to (5, 2) facing south so the distractor at (5, 3) is in front.
        env.step(GridAction::TurnRight).unwrap();
        env.step(GridAction::TurnRight).unwrap();
        env.step(GridAction::Forward).unwrap();
        env.step(GridAction::Forward).unwrap();
        env.step(GridAction::Forward).unwrap();
        env.step(GridAction::TurnRight).unwrap();
        let (fx, fy) = env.state().agent.front();
        assert_eq!(env.state().grid.get(fx, fy), Entity::Ball(DISTRACTOR_COLOR));
        assert!(!env.facing_match());
    }
}
