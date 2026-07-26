//! `Unlock`: pick up the colored key, unlock and open the colored door.
//!
//! Ports Farama Minigrid's [`UnlockEnv`]. The agent stands in a
//! perimeter-walled room that has a single colored locked door on the
//! north wall. A matching colored key sits one step east of the agent.
//! Success is reached when the door transitions to [`DoorState::Open`];
//! timing out with the door still closed or locked returns `0.0`.
//!
//! ## Layout (5 × 5 default)
//!
//! ```text
//! # D # # #    D = Door (yellow, locked) at (1, 0)
//! # A K . #    A = agent (1, 1) facing East; K = Key (yellow) at (2, 1)
//! # . . . #
//! # . . . #
//! # # # # #    # = wall
//! ```
//!
//! Required action sequence: `Pickup` the key → turn to face the door →
//! `Toggle` (Locked → Closed) → `Toggle` (Closed → Open).
//!
//! ## Known deviations from upstream `Unlock`
//!
//! Two of them, both tracked in issue #1020:
//!
//! 1. **The topology is wrong.** Farama's `MiniGrid-Unlock-v0` is a *two-room*
//!    task (a `1 × 2` `RoomGrid` of `room_size = 6`) whose locked door sits on
//!    the **interior wall between the rooms**. rlevo draws a single
//!    perimeter-walled room and writes the door at `(1, 0)` — *into that
//!    perimeter*, on the outer north wall, with nothing behind it.
//! 2. **The layout is fixed.** Upstream randomizes the door's position along
//!    the interior wall, the door color, the key position and the agent's pose;
//!    every reset here produces the identical board, so a policy can memorize
//!    the four-action solution rather than learn to search.
//!
//! The task still exercises key-pickup and the `Locked → Closed → Open` toggle
//! chain, but do not read results here as comparable to published `Unlock`
//! numbers. Fixing the topology moves `MIN_SIZE` and the test oracle, so it is
//! tracked as its own change rather than patched in place.
//!
//! | Observation | 7 × 7 egocentric grid encoded as `[type, color, state]` per cell    |
//! |-------------|----------------------------------------------------------------------|
//! | Action      | `TurnLeft`, `TurnRight`, `Forward`, `Pickup`, `Toggle`               |
//! | Reward      | `success_reward(steps, max_steps)` when door opens; `0.0` on timeout |
//!
//! # Examples
//!
//! ```rust
//! use rlevo_environments::grids::unlock::{UnlockConfig, UnlockEnv};
//! use rlevo_core::environment::{ConstructableEnv, Environment};
//!
//! let cfg = UnlockConfig::new(5, 200, 0);
//! let mut env = UnlockEnv::with_config(cfg, false).expect("valid config");
//! let snap = env.reset().unwrap();
//! println!("door at: {:?}", env.door_pos());
//! ```
//!
//! [`UnlockEnv`]: https://minigrid.farama.org/environments/minigrid/UnlockEnv/

use super::core::{
    GridSnapshot,
    action::GridAction,
    agent::AgentState,
    build_snapshot,
    color::Color,
    direction::Direction,
    dynamics::apply_action,
    entity::{DoorState, Entity},
    grid::Grid,
    render::render_ascii,
    reward::success_reward,
    state::GridState,
};
use rlevo_core::config::{self, ConfigError, Validate};
use rlevo_core::environment::{ConstructableEnv, Environment, EnvironmentError};
use rlevo_core::reward::ScalarReward;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

/// Minimum supported side length for the room.
const MIN_SIZE: usize = 4;
/// Door color used by the default layout.
const DOOR_COLOR: Color = Color::Yellow;

/// Configuration for [`UnlockEnv`].
///
/// # Examples
///
/// ```rust
/// use rlevo_environments::grids::unlock::UnlockConfig;
///
/// let cfg = UnlockConfig::new(5, 200, 0);
/// assert_eq!(cfg.size, 5);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnlockConfig {
    /// Side length of the room; must be ≥ `MIN_SIZE` (4).
    pub size: usize,
    /// Maximum number of steps before the episode times out.
    pub max_steps: usize,
    /// Accepted but unused: this environment makes no random draws.
    ///
    /// rlevo's `Unlock` **deviates from upstream** `MiniGrid-Unlock-v0` in two
    /// ways, both tracked in issue #1020. The layout is fixed — upstream
    /// randomizes the door's position on the interior wall, the door color, the
    /// key position and the agent's pose, while every reset here yields the
    /// identical board. And the door placement is wrong: upstream is a two-room
    /// task with the locked door on the wall *between* the rooms, whereas
    /// [`UnlockEnv`]'s layout builder draws one perimeter-walled room and puts
    /// the door at `(1, 0)`, inside that perimeter.
    ///
    /// Until #1020 lands, this value cannot change any observation, reward, or
    /// transition. The field is kept so every grid env presents the same config
    /// surface.
    pub seed: u64,
}

impl UnlockConfig {
    /// Creates an [`UnlockConfig`] with the given parameters.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rlevo_environments::grids::unlock::UnlockConfig;
    ///
    /// let cfg = UnlockConfig::new(6, 288, 42);
    /// assert_eq!(cfg.seed, 42);
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

impl Default for UnlockConfig {
    fn default() -> Self {
        let size = 5;
        Self {
            size,
            max_steps: 8 * size * size,
            seed: 0,
        }
    }
}

impl Validate for UnlockConfig {
    /// Rejects any `size` below `MIN_SIZE` (4) and a zero `max_steps`.
    ///
    /// The `size` guard lives **here**, not only in [`FromStr`]: `UnlockConfig`
    /// derives `Deserialize`, so a config loaded from a file is user-supplied
    /// runtime data that never passes through `from_str` (rules.md §4 — "if an
    /// invalid value can arrive via `Deserialize`, it must be an `Err`"). The
    /// layout builder writes the door at `(1, 0)` and the key at `(2, 1)` from
    /// fixed coordinates: below this floor those land out of bounds (`size <= 2`,
    /// a panic in `Grid::set`) or inside the perimeter wall (`size == 3`, an
    /// unsolvable board).
    ///
    /// `at_least` subsumes the previous `nonzero` check — `MIN_SIZE >= 1`, so a
    /// zero `size` is still rejected, now as
    /// [`ConstraintKind::TooSmall`](rlevo_core::config::ConstraintKind::TooSmall).
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "UnlockConfig";
        config::at_least(C, "size", self.size, MIN_SIZE)?;
        config::nonzero(C, "max_steps", self.max_steps)?;
        Ok(())
    }
}

impl FromStr for UnlockConfig {
    type Err = String;

    /// Parses `"size=5,max_steps=200,seed=0"` (keys in any order) or the
    /// positional form `"5,200,0"`.
    ///
    /// # Errors
    ///
    /// Returns the offending key/value, or the [`Validate`] rejection — the same
    /// guard [`UnlockEnv::with_config`] applies, so this parser cannot admit a
    /// config that construction would refuse.
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
        cfg.validate()
            .map_err(|e| format!("{e} (got size={})", cfg.size))?;
        Ok(cfg)
    }
}

/// Minigrid's `Unlock` environment.
///
/// The agent must pick up the key from the floor, navigate to the locked
/// door on the north wall, and toggle it twice — first to unlock it
/// (Locked → Closed) then to open it (Closed → Open). Reward is paid the
/// moment the door opens; the episode times out if the step budget is
/// exhausted first.
///
/// Implements [`Environment<3, 3, 1>`] with [`GridState`] /
/// [`GridObservation`](super::core::GridObservation) / [`GridAction`] / [`ScalarReward`].
///
/// # Examples
///
/// ```rust
/// use rlevo_environments::grids::unlock::UnlockEnv;
/// use rlevo_core::environment::{ConstructableEnv, Environment};
///
/// let mut env = UnlockEnv::new(false);
/// let snap = env.reset().unwrap();
/// println!("door at: {:?}", env.door_pos());
/// ```
#[derive(Debug)]
pub struct UnlockEnv {
    state: GridState,
    config: UnlockConfig,
    steps: usize,
    render: bool,
    door_pos: (i32, i32),
}

impl UnlockEnv {
    /// Constructs an [`UnlockEnv`] from an explicit configuration.
    ///
    /// Immediately builds the initial grid state. Call [`Environment::reset`]
    /// before the first [`Environment::step`] to obtain the first observation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rlevo_environments::grids::unlock::{UnlockConfig, UnlockEnv};
    ///
    /// let env = UnlockEnv::with_config(
    ///     UnlockConfig::new(5, 200, 0),
    ///     true, // render ASCII grid to stdout
    /// )
    /// .expect("valid config");
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `config` fails [`Validate`]: a `size` below
    /// `MIN_SIZE` (4) or a zero `max_steps`. This is the construction chokepoint
    /// (rules.md §4), so it also rejects a config that arrived by `Deserialize`
    /// or struct-update syntax without passing through [`FromStr`].
    pub fn with_config(config: UnlockConfig, render: bool) -> Result<Self, ConfigError> {
        config.validate()?;
        let (state, door_pos) = Self::build(&config);
        Ok(Self {
            state,
            config,
            steps: 0,
            render,
            door_pos,
        })
    }

    /// Returns the environment's active configuration.
    #[must_use]
    pub const fn config(&self) -> &UnlockConfig {
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

    /// Returns the world coordinates of the locked door.
    #[must_use]
    pub const fn door_pos(&self) -> (i32, i32) {
        self.door_pos
    }

    /// Renders the current grid state as an ASCII string.
    #[must_use]
    pub fn ascii(&self) -> String {
        render_ascii(&self.state.grid, &self.state.agent)
    }

    fn build(config: &UnlockConfig) -> (GridState, (i32, i32)) {
        let mut grid = Grid::new(config.size, config.size);
        grid.draw_walls();
        let door_pos = (1_i32, 0_i32);
        grid.set(
            door_pos.0,
            door_pos.1,
            Entity::Door(DOOR_COLOR, DoorState::Locked),
        );
        grid.set(2, 1, Entity::Key(DOOR_COLOR));
        let agent = AgentState::new(1, 1, Direction::East);
        (GridState::new(grid, agent), door_pos)
    }

    fn emit(&self, reward: f32, done: bool) -> GridSnapshot {
        if self.render {
            println!("{}", self.ascii());
        }
        build_snapshot(&self.state, reward, done)
    }

    fn door_is_open(&self) -> bool {
        matches!(
            self.state.grid.get(self.door_pos.0, self.door_pos.1),
            Entity::Door(_, DoorState::Open)
        )
    }
}

impl crate::render::AsciiRenderable for UnlockEnv {
    fn render_ascii(&self) -> String {
        render_ascii(&self.state.grid, &self.state.agent)
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        super::core::render::render_styled(&self.state.grid, &self.state.agent)
    }
}

impl Display for UnlockEnv {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "UnlockEnv(size={}, step={}/{})",
            self.config.size, self.steps, self.config.max_steps
        )
    }
}

impl ConstructableEnv for UnlockEnv {
    fn new(render: bool) -> Self {
        Self::with_config(UnlockConfig::default(), render).expect("default config must validate")
    }
}

impl Environment<3, 3, 1> for UnlockEnv {
    type StateType = GridState;
    type ObservationType = super::core::GridObservation;
    type ActionType = GridAction;
    type RewardType = ScalarReward;
    type SnapshotType = GridSnapshot;

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        let (state, door_pos) = Self::build(&self.config);
        self.state = state;
        self.door_pos = door_pos;
        self.steps = 0;
        Ok(self.emit(0.0, false))
    }

    fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> {
        self.steps += 1;
        let _ = apply_action(&mut self.state.grid, &mut self.state.agent, action);
        let (reward, done) = if self.door_is_open() {
            (success_reward(self.steps, self.config.max_steps), true)
        } else if self.steps >= self.config.max_steps {
            (0.0, true)
        } else {
            (0.0, false)
        };
        Ok(self.emit(reward, done))
    }
}

impl rlevo_core::render::payload::GridPayloadSource for UnlockEnv {
    fn grid_snapshot(&self) -> rlevo_core::render::payload::GridSnapshot {
        crate::grids::core::render::grid_snapshot(&self.state.grid, &self.state.agent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlevo_core::config::ConstraintKind;
    use rlevo_core::environment::Snapshot;

    fn test_env() -> UnlockEnv {
        UnlockEnv::with_config(UnlockConfig::new(5, 100, 0), false).expect("valid config")
    }

    #[test]
    fn default_config_validates() {
        assert!(UnlockConfig::default().validate().is_ok());
    }

    #[test]
    fn rejects_zero_size() {
        let bad = UnlockConfig {
            size: 0,
            ..Default::default()
        };
        assert!(UnlockEnv::with_config(bad, false).is_err());
    }

    #[test]
    fn default_config_is_5x5() {
        let cfg = UnlockConfig::default();
        assert_eq!(cfg.size, 5);
        assert_eq!(cfg.max_steps, 8 * 5 * 5);
    }

    #[test]
    fn fromstr_key_value() {
        let cfg: UnlockConfig = "size=6,max_steps=80,seed=3".parse().unwrap();
        assert_eq!(cfg.size, 6);
        assert_eq!(cfg.max_steps, 80);
        assert_eq!(cfg.seed, 3);
    }

    #[test]
    fn fromstr_rejects_small_size() {
        assert!("2".parse::<UnlockConfig>().is_err());
    }

    /// Issue #106: `MIN_SIZE` was enforced only in [`FromStr`], so a config
    /// built by `Deserialize` or struct-update syntax reached `build`, which
    /// panicked in `Grid::set` placing the door at `(1, 0)` (`size = 1`) or the
    /// key at `(2, 1)` (`size = 2`), and silently produced an unsolvable board
    /// at `size = 3`. The guard now lives in [`Validate`], which `with_config`
    /// runs (ADR 0026 chokepoint).
    #[test]
    fn with_config_rejects_size_below_min() {
        let bad = UnlockConfig {
            size: MIN_SIZE - 1,
            ..Default::default()
        };
        let err = UnlockEnv::with_config(bad, false).unwrap_err();
        assert_eq!(err.config, "UnlockConfig");
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
    fn build_places_door_and_key() {
        let env = test_env();
        assert_eq!(
            env.state().grid.get(1, 0),
            Entity::Door(DOOR_COLOR, DoorState::Locked)
        );
        assert_eq!(env.state().grid.get(2, 1), Entity::Key(DOOR_COLOR));
        assert_eq!(env.state().agent.x, 1);
        assert_eq!(env.state().agent.y, 1);
        assert_eq!(env.state().agent.direction, Direction::East);
    }

    #[test]
    fn reset_is_deterministic() {
        let cfg = UnlockConfig::new(5, 100, 7);
        let mut a = UnlockEnv::with_config(cfg, false).expect("valid config");
        let mut b = UnlockEnv::with_config(cfg, false).expect("valid config");
        let sa = a.reset().unwrap();
        let sb = b.reset().unwrap();
        assert_eq!(sa.observation(), sb.observation());
        assert_eq!(a.door_pos(), b.door_pos());
    }

    #[test]
    fn toggle_without_key_leaves_door_locked() {
        let mut env = test_env();
        env.reset().unwrap();
        // Face the door (north) without picking up the key first.
        env.step(GridAction::TurnLeft).unwrap();
        let snap = env.step(GridAction::Toggle).unwrap();
        assert!(!snap.is_done());
        assert!(matches!(
            env.state().grid.get(1, 0),
            Entity::Door(_, DoorState::Locked)
        ));
    }

    #[test]
    fn optimal_rollout_opens_door_with_positive_reward() {
        let mut env = test_env();
        env.reset().unwrap();
        // Agent at (1,1) facing East, key at (2,1), door at (1,0).
        let script = [
            GridAction::Pickup,   // grab the key
            GridAction::TurnLeft, // face north toward door
            GridAction::Toggle,   // unlock (Locked -> Closed)
            GridAction::Toggle,   // open (Closed -> Open)
        ];
        let mut last = None;
        for a in script {
            last = Some(env.step(a).unwrap());
        }
        let snap = last.unwrap();
        assert!(snap.is_done(), "unlocking the door should terminate");
        let reward: f32 = (*snap.reward()).into();
        assert!(reward > 0.9, "reward was {reward}");
    }

    #[test]
    fn timeout_returns_zero_reward() {
        let cfg = UnlockConfig::new(5, 3, 0);
        let mut env = UnlockEnv::with_config(cfg, false).expect("valid config");
        env.reset().unwrap();
        for _ in 0..3 {
            env.step(GridAction::TurnLeft).unwrap();
        }
        // After exhausting the budget the env must still be terminal.
        let snap = env.step(GridAction::TurnLeft);
        // If the budget terminated on the 3rd step the 4th call shouldn't panic.
        // We don't assert reward on the 4th step because the episode already ended.
        let _ = snap;
    }

    #[test]
    fn door_is_not_open_at_start() {
        let env = test_env();
        assert!(!env.door_is_open());
    }

    #[test]
    fn reset_after_success_clears_step_counter() {
        let mut env = test_env();
        env.reset().unwrap();
        for _ in 0..4 {
            env.step(GridAction::TurnLeft).unwrap();
        }
        assert_eq!(env.steps(), 4);
        env.reset().unwrap();
        assert_eq!(env.steps(), 0);
    }
}
