//! Navigate an empty walled room to reach the goal tile.
//!
//! Ports Farama Minigrid's [`EmptyEnv`]. The grid is an `N×N` room with a
//! wall perimeter, a single [`Goal`] tile at `(size - 2, size - 2)`, and the
//! agent starting at `(1, 1)` facing East. Stepping onto the goal terminates
//! the episode and pays [`success_reward`]; exceeding `max_steps` terminates
//! with reward `0.0`.
//!
//! This is the simplest grid environment — no obstacles, no items, no doors.
//! It serves as a baseline for testing navigation policies and as a reference
//! for implementing more complex environments.
//!
//! ## Layout (default `size = 8`)
//!
//! ```text
//! # # # # # # # #
//! # A . . . . . #   A = agent (1, 1), facing East
//! # . . . . . . #
//! # . . . . . . #
//! # . . . . . . #
//! # . . . . . . #
//! # . . . . . G #   G = goal (6, 6)
//! # # # # # # # #
//! ```
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
//! use rlevo_environments::grids::empty::{EmptyConfig, EmptyEnv};
//! use rlevo_core::environment::{ConstructableEnv, Environment};
//!
//! let cfg = EmptyConfig::new(8, 256, 0);
//! let mut env = EmptyEnv::with_config(cfg, false).expect("valid config");
//! let _snapshot = env.reset().unwrap();
//! ```
//!
//! [`EmptyEnv`]: https://minigrid.farama.org/environments/minigrid/EmptyEnv/
//! [`Goal`]: super::core::entity::Entity::Goal

use super::core::{
    Visibility,
    action::GridAction,
    agent::AgentState,
    direction::Direction,
    dynamics::{StepOutcome, apply_action},
    entity::Entity,
    grid::Grid,
    observation::GridObservation,
    observe_grid,
    reward::success_reward,
    state::GridState,
};
use crate::episode::EpisodeGuard;
use rlevo_core::config::{self, ConfigError, Validate};
use rlevo_core::environment::{
    ConstructableEnv, Environment, EnvironmentError, Sensor, Snapshot, SnapshotBase,
};
use rlevo_core::reward::ScalarReward;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

/// Minimum grid side length; smaller grids can't host both an agent
/// start cell and a distinct goal.
const MIN_SIZE: usize = 4;

/// Configuration for [`EmptyEnv`].
///
/// # Examples
///
/// ```rust
/// use rlevo_environments::grids::empty::EmptyConfig;
///
/// let cfg = EmptyConfig::new(8, 256, 0);
/// assert_eq!(cfg.size, 8);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmptyConfig {
    /// Grid side length in cells (including perimeter walls); must be ≥
    /// `MIN_SIZE` (4).
    pub size: usize,
    /// Maximum number of steps before the episode times out.
    pub max_steps: usize,
    /// Accepted but unused: this environment makes no random draws.
    ///
    /// The layout is fixed — walls, the goal at `(size - 2, size - 2)`, and the
    /// agent at `(1, 1)` facing East — matching upstream `MiniGrid-Empty-*`,
    /// whose `_gen_grid` samples nothing. Changing this value therefore cannot
    /// change any observation, reward, or transition. It exists so every grid
    /// env presents the same config surface to schedulers and sweep scripts.
    ///
    /// A random-spawn Empty would be a **distinct env type**, mirroring
    /// upstream's separately registered `MiniGrid-Empty-Random-*`, not a mode of
    /// this one selected by a seed.
    pub seed: u64,
}

impl EmptyConfig {
    /// Constructs an `EmptyConfig` with explicit field values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rlevo_environments::grids::empty::EmptyConfig;
    ///
    /// let cfg = EmptyConfig::new(8, 256, 0);
    /// assert_eq!(cfg.size, 8);
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

impl Validate for EmptyConfig {
    /// Rejects any `size` below `MIN_SIZE` (4) and a zero `max_steps`.
    ///
    /// The `size` guard lives **here**, not only in [`FromStr`]: `EmptyConfig`
    /// derives `Deserialize`, so a config loaded from a file is user-supplied
    /// runtime data that never passes through `from_str` (rules.md §4 — "if an
    /// invalid value can arrive via `Deserialize`, it must be an `Err`").
    /// The layout builder subtracts 2 from `size` to place the goal, which
    /// underflows for `size < 2`.
    ///
    /// `at_least` subsumes the previous `nonzero` check — `MIN_SIZE >= 1`, so a
    /// zero `size` is still rejected, now as
    /// [`ConstraintKind::TooSmall`](rlevo_core::config::ConstraintKind::TooSmall).
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "EmptyConfig";
        config::at_least(C, "size", self.size, MIN_SIZE)?;
        config::nonzero(C, "max_steps", self.max_steps)?;
        Ok(())
    }
}

impl FromStr for EmptyConfig {
    type Err = String;

    /// Parse a config from a comma-separated list.
    ///
    /// Accepts positional values (`"5"`, `"5,100"`, `"5,100,42"`) and
    /// `key=value` pairs (`"size=5,max_steps=100,seed=42"`).
    ///
    /// # Errors
    ///
    /// Returns the offending key/value, or the [`Validate`] rejection — the same
    /// guard [`EmptyEnv::with_config`] applies, so this parser cannot admit a
    /// config that construction would refuse.
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
        cfg.validate()
            .map_err(|e| format!("{e} (got size={})", cfg.size))?;
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

/// Simplest grid environment: navigate an empty room to the goal.
///
/// Implements [`Environment<3, 3, 1>`] — observation and action spaces each
/// have three components, reward is a scalar.
///
/// No obstacles, keys, or doors — useful as a baseline for verifying that a
/// navigation policy can solve a trivial task before tackling harder
/// environments.
///
/// Construct via [`EmptyEnv::with_config`] for full control or via
/// [`ConstructableEnv::new`] for default settings (size 8, seed 0).
///
/// # Examples
///
/// ```no_run
/// use rlevo_environments::grids::empty::{EmptyConfig, EmptyEnv};
/// use rlevo_core::environment::{ConstructableEnv, Environment};
///
/// let mut env = EmptyEnv::with_config(EmptyConfig::new(5, 100, 0), false).expect("valid config");
/// env.reset().unwrap();
/// ```
#[derive(Debug)]
pub struct EmptyEnv {
    state: GridState,
    config: EmptyConfig,
    steps: usize,
    render: bool,
    /// Rejects a `step()` taken after the episode ended. Reaching the goal does
    /// not consume the [`Goal`] tile or freeze the agent, and `done` is
    /// recomputed from the *current* outcome each call, so without this guard a
    /// post-terminal `step()` emitted a fresh **`Running`** snapshot — silently
    /// resurrecting a finished episode — while `steps` kept advancing past the
    /// true episode length. Worse, the agent stands *on* the goal: stepping off
    /// and back on re-triggers `StepOutcome::ReachedGoal` and pays
    /// [`success_reward`] a second time. On a 5×5 grid with `max_steps = 100`
    /// the optimal rollout terminates at step 5 with `0.955`, then six more
    /// steps re-pay `0.901` — an episode return of `1.856` for one goal, and
    /// unbounded under repetition. The inflated `steps` also deflates every
    /// later payout, since `success_reward` divides by the step count.
    guard: EpisodeGuard,
}

impl EmptyEnv {
    /// Emission-model visibility policy: does this environment's agent see
    /// through opaque cells?
    ///
    /// The rlevo spelling of canonical Minigrid's `see_through_walls`
    /// constructor argument. Read only by this environment's [`Sensor`] impl,
    /// and an inherent const rather than a config field because it is part of
    /// the task definition, not a knob a caller tunes.
    ///
    /// [`Visibility::SeeThrough`], because canonical `minigrid/envs/empty.py`
    /// passes `see_through_walls=True` explicitly. That is an **opt-out**:
    /// `MiniGridEnv.__init__` defaults the flag to `False`, so an env is
    /// occluded unless it says otherwise, and `EmptyEnv` says otherwise. See
    /// ADR 0063 (`docs/adr/0063-grid-visibility-occlusion.md`) for the whole
    /// twelve-environment table.
    const VISIBILITY: Visibility = Visibility::SeeThrough;

    /// Constructs an `EmptyEnv` from an explicit configuration.
    ///
    /// Immediately builds the initial grid state. Call [`Environment::reset`]
    /// before the first [`Environment::step`] to obtain the first observation.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `config` fails [`Validate`]: a `size` below
    /// `MIN_SIZE` (4) or a zero `max_steps`. This is the construction chokepoint
    /// (rules.md §4), so it also rejects a config that arrived by `Deserialize`
    /// or struct-update syntax without passing through [`FromStr`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rlevo_environments::grids::empty::{EmptyConfig, EmptyEnv};
    ///
    /// let env = EmptyEnv::with_config(
    ///     EmptyConfig::new(8, 256, 0),
    ///     true, // render ASCII grid to stdout
    /// )
    /// .expect("valid config");
    /// ```
    /// This is the only constructor that builds an `EmptyEnv` value —
    /// [`ConstructableEnv::new`] delegates here — so it is also the single
    /// place the [`EpisodeGuard`] is initialized.
    pub fn with_config(config: EmptyConfig, render: bool) -> Result<Self, ConfigError> {
        config.validate()?;
        let (grid, agent) = Self::build(&config);
        Ok(Self {
            state: GridState::new(grid, agent),
            config,
            steps: 0,
            render,
            guard: EpisodeGuard::new(),
        })
    }

    /// Returns a reference to the active configuration.
    #[must_use]
    pub const fn config(&self) -> &EmptyConfig {
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

    fn snapshot(
        &self,
        observation: GridObservation,
        reward: f32,
        done: bool,
    ) -> SnapshotBase<3, GridObservation, ScalarReward> {
        if self.render {
            // Render is a debug side effect; return the string so callers can
            // capture it if they wish, or drop it when invoked internally.
            let _ = super::core::render::render_ascii(&self.state.grid, &self.state.agent);
        }
        if done {
            SnapshotBase::terminated(observation, ScalarReward::new(reward))
        } else {
            SnapshotBase::running(observation, ScalarReward::new(reward))
        }
    }
}

impl crate::render::AsciiRenderable for EmptyEnv {
    fn render_ascii(&self) -> String {
        super::core::render::render_ascii(&self.state.grid, &self.state.agent)
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        super::core::render::render_styled(&self.state.grid, &self.state.agent)
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

impl ConstructableEnv for EmptyEnv {
    fn new(render: bool) -> Self {
        Self::with_config(EmptyConfig::default(), render).expect("default config must validate")
    }
}

impl Sensor<3, 1, 3> for EmptyEnv {
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

impl Environment<3, 3, 1> for EmptyEnv {
    type StateType = GridState;
    type ObservationType = GridObservation;
    type ActionType = GridAction;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<3, GridObservation, ScalarReward>;

    /// Rebuilds the fixed layout, clears the step counter, and re-opens the
    /// episode guard so a terminated environment becomes steppable again.
    ///
    /// # Errors
    ///
    /// Currently infallible; always returns `Ok`.
    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.guard.reset();
        let (grid, agent) = Self::build(&self.config);
        self.state = GridState::new(grid, agent);
        self.steps = 0;
        let observation = self.observe_reset(&self.state);
        Ok(self.snapshot(observation, 0.0, false))
    }

    /// Applies `action`, then maps the resulting [`StepOutcome`] to reward and
    /// termination.
    ///
    /// # Errors
    ///
    /// Returns [`EnvironmentError::StepAfterEpisodeEnd`] if the episode has
    /// already ended; call [`reset`](Environment::reset) first.
    fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> {
        // Guard first: before `steps` advances and before `apply_action` touches
        // the grid or the agent. A rejected call must leave the environment
        // bit-identical — and this env draws no randomness, but the ordering is
        // the same one ADR 0029 requires of the envs that do, so a rejected step
        // can never shift a seed stream.
        self.guard.check()?;

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

        // Single exit: one snapshot is built, and the guard is fed that
        // snapshot's own status, so the two cannot drift apart.
        let snapshot = self.snapshot(observation, reward, done);
        self.guard.record(snapshot.status());
        Ok(snapshot)
    }
}

impl rlevo_core::render::payload::GridPayloadSource for EmptyEnv {
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
    use crate::episode::assert_rejects_post_terminal_step;
    use rlevo_core::action::DiscreteAction;
    use rlevo_core::base::Observation;
    use rlevo_core::config::ConstraintKind;
    use rlevo_core::environment::EpisodeStatus;

    #[test]
    fn default_config_validates() {
        assert!(EmptyConfig::default().validate().is_ok());
    }

    #[test]
    fn rejects_zero_size() {
        let bad = EmptyConfig {
            size: 0,
            ..Default::default()
        };
        assert!(EmptyEnv::with_config(bad, false).is_err());
    }

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
        // `from_str` delegates to `Validate`, so the text is the `ConfigError`
        // rendering plus the parsed size. Substrings only: the structured
        // assertion lives in `with_config_rejects_size_below_min`.
        let err = "2".parse::<EmptyConfig>().unwrap_err();
        assert!(err.contains("EmptyConfig.size"), "was {err}");
        assert!(err.contains("at least 4"), "was {err}");
    }

    /// Issue #106: `MIN_SIZE` was enforced only in [`FromStr`], so a config
    /// built by `Deserialize` or struct-update syntax reached `build`, where
    /// `size - 2` underflowed and panicked. The guard now lives in
    /// [`Validate`], which `with_config` runs (ADR 0026 chokepoint).
    #[test]
    fn with_config_rejects_size_below_min() {
        let bad = EmptyConfig {
            size: MIN_SIZE - 1,
            ..Default::default()
        };
        let err = EmptyEnv::with_config(bad, false).unwrap_err();
        assert_eq!(err.config, "EmptyConfig");
        assert_eq!(err.field, "size");
        assert_eq!(
            err.kind,
            ConstraintKind::TooSmall {
                min: MIN_SIZE as u64,
                got: (MIN_SIZE - 1) as u64,
            }
        );
    }

    #[test]
    fn fromstr_rejects_unknown_key() {
        let err = "wat=5".parse::<EmptyConfig>().unwrap_err();
        assert!(err.contains("unknown key"));
    }

    #[test]
    fn new_places_goal_and_agent() {
        let env = EmptyEnv::with_config(EmptyConfig::new(5, 100, 0), false).expect("valid config");
        let grid = &env.state().grid;
        assert_eq!(grid.get(3, 3), Entity::Goal);
        assert_eq!(env.state().agent.x, 1);
        assert_eq!(env.state().agent.y, 1);
        assert_eq!(env.state().agent.direction, Direction::East);
    }

    #[test]
    fn reset_is_deterministic_for_same_seed() {
        let cfg = EmptyConfig::new(5, 100, 42);
        let mut a = EmptyEnv::with_config(cfg, false).expect("valid config");
        let mut b = EmptyEnv::with_config(cfg, false).expect("valid config");
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
        let mut env = EmptyEnv::with_config(cfg, false).expect("valid config");
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
        let mut env = EmptyEnv::with_config(cfg, false).expect("valid config");
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
        let mut env = EmptyEnv::with_config(cfg, false).expect("valid config");
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
        let mut env = EmptyEnv::with_config(cfg, false).expect("valid config");
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
        let mut env = EmptyEnv::with_config(cfg, false).expect("valid config");
        env.reset().unwrap();
        for i in 0..50 {
            let action = GridAction::from_index(i % GridAction::ACTION_COUNT);
            let snap = env.step(action).unwrap();
            if snap.is_done() {
                break;
            }
        }
    }

    // ── post-terminal step guard (ADR 0044) ───────────────────────────────────

    /// Drives a fresh 5×5 episode to the goal with real `step()` calls.
    ///
    /// The termination is a **goal arrival**, deliberately not a step-limit
    /// cutoff: `grids::core::build_snapshot` currently reports a timeout as
    /// `Terminated` (issue #1028), and these tests must not bake that in.
    fn drive_to_goal(env: &mut EmptyEnv) -> SnapshotBase<3, GridObservation, ScalarReward> {
        env.reset().expect("reset must succeed");
        // Agent at (1,1) facing East, goal at (3,3).
        let script = [
            GridAction::Forward,
            GridAction::Forward,
            GridAction::TurnRight,
            GridAction::Forward,
            GridAction::Forward,
        ];
        let mut last = None;
        for action in script {
            last = Some(env.step(action).expect("scripted step must succeed"));
        }
        last.expect("the script is non-empty")
    }

    fn goal_env() -> EmptyEnv {
        EmptyEnv::with_config(EmptyConfig::new(5, 100, 0), false).expect("valid config")
    }

    #[test]
    /// `EmptyEnv` satisfies the shared post-terminal conformance check: once the
    /// agent has reached the goal, a further legal `step()` fails with
    /// `StepAfterEpisodeEnd` carrying the status that ended the episode.
    fn rejects_post_terminal_step() {
        let mut env = goal_env();
        assert_rejects_post_terminal_step(&mut env, drive_to_goal, GridAction::TurnLeft);
    }

    #[test]
    /// Regression for the concrete defect the guard prevents. Reaching the goal
    /// neither consumes the `Goal` tile nor freezes the agent, so an unguarded
    /// post-terminal `step()` advanced `steps`, moved the agent, and emitted a
    /// fresh `Running` snapshot — and walking off the goal and back on re-paid
    /// `success_reward` (measured: `0.955` then another `0.901` on this 5×5).
    /// A rejected step must mutate nothing at all.
    fn post_terminal_step_does_not_mutate_state() {
        let mut env = goal_env();
        let terminal = drive_to_goal(&mut env);
        assert!(terminal.is_done(), "reaching the goal must end the episode");
        let ended = terminal.status();

        let steps_at_end = env.steps();
        let pos_at_end = (env.state().agent.x, env.state().agent.y);
        let dir_at_end = env.state().agent.direction;

        let err = env
            .step(GridAction::TurnLeft)
            .expect_err("a step after the goal must return Err, not another snapshot");
        match err {
            EnvironmentError::StepAfterEpisodeEnd { status } => assert_eq!(
                status, ended,
                "the error must carry the status that ended the episode"
            ),
            other => panic!("expected StepAfterEpisodeEnd, got {other:?}"),
        }

        assert_eq!(
            env.steps(),
            steps_at_end,
            "a rejected step must not advance the step counter"
        );
        assert_eq!(
            (env.state().agent.x, env.state().agent.y),
            pos_at_end,
            "a rejected step must not move the agent"
        );
        assert_eq!(
            env.state().agent.direction,
            dir_at_end,
            "a rejected step must not turn the agent"
        );
        assert_eq!(
            env.guard.status(),
            ended,
            "a rejected step must not reopen the episode"
        );
    }

    #[test]
    /// `reset()` re-opens a finished episode, so a latched guard cannot strand
    /// the environment for the rest of the run.
    fn reset_reopens_terminated_episode() {
        let mut env = goal_env();
        drive_to_goal(&mut env);
        assert!(
            env.step(GridAction::TurnLeft).is_err(),
            "the episode has ended; a step must be rejected before reset()"
        );

        let first = env.reset().expect("reset must succeed after termination");
        assert!(!first.is_done(), "a fresh episode must not start done");
        assert_eq!(
            env.guard.status(),
            EpisodeStatus::Running,
            "reset() must return the guard to Running"
        );

        let snap = env
            .step(GridAction::TurnLeft)
            .expect("reset() must re-open the environment for a new episode");
        assert!(
            !snap.is_done(),
            "the first step of a fresh episode must not be done"
        );
    }

    #[test]
    fn display_contains_step_budget() {
        let env = EmptyEnv::with_config(EmptyConfig::new(5, 50, 0), false).expect("valid config");
        let s = env.to_string();
        assert!(s.contains("EmptyEnv"));
        assert!(s.contains("50"));
    }
}
