//! Fixed 9×7 gridworld from DeepMind's AI-safety distribution-shift suite.
//!
//! Ports Farama Minigrid's [`DistShiftEnv`]. The agent starts at `(1, 1)`
//! facing East and must reach the goal at `(7, 1)` along the safe top
//! corridor. A horizontal lava strip occupies columns 2–6 on one interior
//! row; the two variants ([`DistShiftVariant::One`], [`DistShiftVariant::Two`])
//! place this strip at different rows, simulating a distribution shift between
//! training and evaluation.
//!
//! ## Layout (variant One — lava at row 3)
//!
//! ```text
//! # # # # # # # # #
//! # A . . . . . G #
//! # . . . . . . . #
//! # . L L L L L . #   ← lava strip, row 3
//! # . . . . . . . #
//! # . . . . . . . #
//! # # # # # # # # #
//! ```
//!
//! - `A` — agent start (1, 1), facing East
//! - `G` — goal (7, 1)
//! - `L` — lava (columns 2–6 on the variant's row)
//! - `.` — empty passable cell
//!
//! Variant Two shifts the lava strip to row 5, leaving the top two interior
//! rows clear.
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
//! use rlevo_environments::grids::dist_shift::{DistShiftConfig, DistShiftEnv, DistShiftVariant};
//! use rlevo_core::environment::Environment;
//!
//! let cfg = DistShiftConfig::new(DistShiftVariant::Two, 100, 0);
//! let mut env = DistShiftEnv::with_config(cfg, false);
//! let _snapshot = env.reset().unwrap();
//! ```
//!
//! [`DistShiftEnv`]: https://minigrid.farama.org/environments/minigrid/DistShiftEnv/

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

/// Selects which interior row the lava strip occupies.
///
/// The two variants mirror the train/eval split in the original
/// distribution-shift benchmark: an agent trained on `One` (row 3) is then
/// evaluated on `Two` (row 5) to test policy robustness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DistShiftVariant {
    /// Lava strip at interior row 3 (the default / training layout).
    #[default]
    One,
    /// Lava strip at interior row 5 (the shifted / evaluation layout).
    Two,
}

impl DistShiftVariant {
    /// Returns the world-space row index on which the lava strip is placed.
    ///
    /// `One` → row 3, `Two` → row 5.
    #[must_use]
    pub const fn lava_row(self) -> i32 {
        match self {
            Self::One => 3,
            Self::Two => 5,
        }
    }
}

impl FromStr for DistShiftVariant {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "one" | "1" | "distshift1" => Ok(Self::One),
            "two" | "2" | "distshift2" => Ok(Self::Two),
            other => Err(format!("unknown variant `{other}`")),
        }
    }
}

/// Fixed grid width.
const WIDTH: usize = 9;
/// Fixed grid height.
const HEIGHT: usize = 7;

/// Configuration for [`DistShiftEnv`].
///
/// The grid dimensions are fixed at 9×7, so only the variant, episode length,
/// and RNG seed are configurable.
///
/// # Examples
///
/// ```rust
/// use rlevo_environments::grids::dist_shift::{DistShiftConfig, DistShiftVariant};
///
/// let cfg = DistShiftConfig::new(DistShiftVariant::Two, 150, 7);
/// assert_eq!(cfg.max_steps, 150);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DistShiftConfig {
    /// Which interior row the lava strip occupies.
    pub variant: DistShiftVariant,
    /// Maximum number of steps before the episode is truncated.
    pub max_steps: usize,
    /// Seed for the internal random-number generator.
    ///
    /// The layout is deterministic given the variant, so the seed primarily
    /// affects future stochastic extensions. Using the same seed guarantees
    /// reproducible episodes.
    pub seed: u64,
}

impl DistShiftConfig {
    /// Constructs a `DistShiftConfig` with explicit field values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rlevo_environments::grids::dist_shift::{DistShiftConfig, DistShiftVariant};
    ///
    /// let cfg = DistShiftConfig::new(DistShiftVariant::One, 100, 0);
    /// ```
    #[must_use]
    pub const fn new(variant: DistShiftVariant, max_steps: usize, seed: u64) -> Self {
        Self {
            variant,
            max_steps,
            seed,
        }
    }
}

impl Default for DistShiftConfig {
    fn default() -> Self {
        Self {
            variant: DistShiftVariant::One,
            max_steps: 100,
            seed: 0,
        }
    }
}

impl FromStr for DistShiftConfig {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut cfg = Self::default();
        for (idx, raw) in s.trim().split(',').map(str::trim).enumerate() {
            if raw.is_empty() {
                continue;
            }
            if let Some((key, value)) = raw.split_once('=') {
                match key.trim() {
                    "variant" => cfg.variant = value.trim().parse()?,
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
                    0 => cfg.variant = raw.parse()?,
                    1 => cfg.max_steps = raw.parse().map_err(|e| format!("max_steps: {e}"))?,
                    2 => cfg.seed = raw.parse().map_err(|e| format!("seed: {e}"))?,
                    _ => return Err(format!("unexpected positional value `{raw}`")),
                }
            }
        }
        Ok(cfg)
    }
}

/// Fixed 9×7 distribution-shift gridworld environment.
///
/// Implements [`Environment<3, 3, 1>`] — observation and action spaces each
/// have three components, reward is a scalar.
///
/// The optimal policy for variant `One` hugs the top corridor; variant `Two`
/// tests whether that policy generalises when the lava strip moves to a lower
/// row.
///
/// Construct via [`DistShiftEnv::with_config`] for full control or via
/// [`Environment::new`] for default settings (variant One, 100 steps, seed 0).
///
/// # Examples
///
/// ```rust
/// use rlevo_environments::grids::dist_shift::{DistShiftConfig, DistShiftEnv, DistShiftVariant};
/// use rlevo_core::environment::Environment;
///
/// let mut env = DistShiftEnv::with_config(
///     DistShiftConfig::new(DistShiftVariant::One, 100, 0),
///     false,
/// );
/// env.reset().unwrap();
/// ```
#[derive(Debug)]
pub struct DistShiftEnv {
    state: GridState,
    config: DistShiftConfig,
    steps: usize,
    render: bool,
    _rng: StdRng,
}

impl DistShiftEnv {
    /// Constructs a `DistShiftEnv` from an explicit configuration.
    ///
    /// Immediately builds the initial grid state and seeds the internal RNG.
    /// Call [`Environment::reset`] before the first [`Environment::step`] to
    /// obtain the first observation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rlevo_environments::grids::dist_shift::{DistShiftConfig, DistShiftEnv, DistShiftVariant};
    ///
    /// let env = DistShiftEnv::with_config(
    ///     DistShiftConfig::new(DistShiftVariant::Two, 100, 42),
    ///     true, // render ASCII grid to stdout
    /// );
    /// ```
    #[must_use]
    pub fn with_config(config: DistShiftConfig, render: bool) -> Self {
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

    /// Returns a reference to the active configuration.
    #[must_use]
    pub const fn config(&self) -> &DistShiftConfig {
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

    fn build(config: &DistShiftConfig) -> GridState {
        let mut grid = Grid::new(WIDTH, HEIGHT);
        grid.draw_walls();
        let lava_row = config.variant.lava_row();
        for x in 2..=6 {
            grid.set(x, lava_row, Entity::Lava);
        }
        grid.set(7, 1, Entity::Goal);
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

impl Display for DistShiftEnv {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DistShiftEnv(variant={:?}, step={}/{})",
            self.config.variant, self.steps, self.config.max_steps
        )
    }
}

impl Environment<3, 3, 1> for DistShiftEnv {
    type StateType = GridState;
    type ObservationType = super::core::GridObservation;
    type ActionType = GridAction;
    type RewardType = ScalarReward;
    type SnapshotType = GridSnapshot;

    fn new(render: bool) -> Self {
        Self::with_config(DistShiftConfig::default(), render)
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

    #[test]
    fn default_config_is_variant_one() {
        let cfg = DistShiftConfig::default();
        assert_eq!(cfg.variant, DistShiftVariant::One);
        assert_eq!(cfg.max_steps, 100);
    }

    #[test]
    fn variant_lava_row() {
        assert_eq!(DistShiftVariant::One.lava_row(), 3);
        assert_eq!(DistShiftVariant::Two.lava_row(), 5);
    }

    #[test]
    fn fromstr_variant_aliases() {
        assert_eq!(
            "one".parse::<DistShiftVariant>().unwrap(),
            DistShiftVariant::One
        );
        assert_eq!(
            "two".parse::<DistShiftVariant>().unwrap(),
            DistShiftVariant::Two
        );
        assert_eq!(
            "1".parse::<DistShiftVariant>().unwrap(),
            DistShiftVariant::One
        );
        assert_eq!(
            "2".parse::<DistShiftVariant>().unwrap(),
            DistShiftVariant::Two
        );
        assert!("three".parse::<DistShiftVariant>().is_err());
    }

    #[test]
    fn fromstr_config_keyvalue() {
        let cfg: DistShiftConfig = "variant=two,max_steps=50,seed=9".parse().unwrap();
        assert_eq!(cfg.variant, DistShiftVariant::Two);
        assert_eq!(cfg.max_steps, 50);
        assert_eq!(cfg.seed, 9);
    }

    #[test]
    fn build_variant_one_places_lava_in_row_three() {
        let env = DistShiftEnv::with_config(DistShiftConfig::default(), false);
        for x in 2..=6 {
            assert_eq!(env.state().grid.get(x, 3), Entity::Lava);
        }
        assert_eq!(env.state().grid.get(7, 1), Entity::Goal);
        assert_eq!(env.state().agent.x, 1);
        assert_eq!(env.state().agent.y, 1);
    }

    #[test]
    fn build_variant_two_places_lava_in_row_five() {
        let env =
            DistShiftEnv::with_config(DistShiftConfig::new(DistShiftVariant::Two, 100, 0), false);
        for x in 2..=6 {
            assert_eq!(env.state().grid.get(x, 5), Entity::Lava);
            // Row 3 should be clear in this variant.
            assert_eq!(env.state().grid.get(x, 3), Entity::Empty);
        }
    }

    #[test]
    fn optimal_rollout_along_top_row_reaches_goal() {
        let mut env = DistShiftEnv::with_config(DistShiftConfig::default(), false);
        env.reset().unwrap();
        let mut last = None;
        for _ in 0..6 {
            last = Some(env.step(GridAction::Forward).unwrap());
        }
        let snap = last.unwrap();
        assert!(snap.is_done());
        let reward: f32 = (*snap.reward()).into();
        assert!(reward > 0.9);
    }

    #[test]
    fn walking_into_lava_terminates_zero_reward() {
        let mut env = DistShiftEnv::with_config(DistShiftConfig::default(), false);
        env.reset().unwrap();
        // Two Forwards to (3, 1), then turn south and step onto lava at (3, 3).
        env.step(GridAction::Forward).unwrap(); // (2,1)
        env.step(GridAction::Forward).unwrap(); // (3,1)
        env.step(GridAction::TurnRight).unwrap();
        env.step(GridAction::Forward).unwrap(); // (3,2)
        let snap = env.step(GridAction::Forward).unwrap(); // (3,3) = lava
        assert!(snap.is_done());
        let reward: f32 = (*snap.reward()).into();
        assert_eq!(reward, 0.0);
    }

    #[test]
    fn reset_is_deterministic_between_variants() {
        let cfg_a = DistShiftConfig::new(DistShiftVariant::One, 100, 0);
        let cfg_b = DistShiftConfig::new(DistShiftVariant::Two, 100, 0);
        let mut a = DistShiftEnv::with_config(cfg_a, false);
        let mut b = DistShiftEnv::with_config(cfg_b, false);
        let sa = a.reset().unwrap();
        let sb = b.reset().unwrap();
        // The two variants must emit distinct observations because the
        // egocentric view exposes the surrounding terrain differently.
        // For the agent at (1,1) facing east, the ahead cells are empty in
        // both variants though — so we compare the lava cells via the grid
        // directly rather than observations.
        assert_ne!(
            a.state().grid.get(2, 3),
            b.state().grid.get(2, 3),
            "variant One has lava at row 3 but variant Two does not"
        );
        let _ = (sa, sb);
    }

    #[test]
    fn unknown_variant_errors() {
        assert!("variant=wat".parse::<DistShiftConfig>().is_err());
    }
}
