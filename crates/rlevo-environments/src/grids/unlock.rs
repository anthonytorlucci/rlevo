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
//! Two of them, both stemming from the same layout builder shortcut — the
//! deviation is a topology change (two rooms sharing an interior wall, not a
//! single perimeter room), so fixing it moves `MIN_SIZE` and the solvability
//! oracle rather than a coordinate:
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
    GridSnapshot, Visibility,
    action::GridAction,
    agent::AgentState,
    build_snapshot,
    color::Color,
    direction::Direction,
    dynamics::apply_action,
    entity::{DoorState, Entity},
    grid::Grid,
    observation::GridObservation,
    observe_grid,
    render::render_ascii,
    reward::success_reward,
    state::GridState,
};
use crate::episode::EpisodeGuard;
use rlevo_core::config::{self, ConfigError, Validate};
use rlevo_core::environment::{ConstructableEnv, Environment, EnvironmentError, Sensor, Snapshot};
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
    /// ways. The layout is fixed — upstream randomizes the door's position on
    /// the interior wall, the door color, the key position and the agent's
    /// pose, while every reset here yields the identical board. And the door
    /// placement is wrong: upstream is a two-room task with the locked door on
    /// the wall *between* the rooms, whereas [`UnlockEnv`]'s layout builder
    /// draws one perimeter-walled room and puts the door at `(1, 0)`, inside
    /// that perimeter — the row `draw_walls()` just filled, with no second
    /// room behind it.
    ///
    /// Until the layout builder is reworked to place two rooms with an
    /// interior dividing wall, this value cannot change any observation,
    /// reward, or transition. The field is kept so every grid env presents the
    /// same config surface.
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
    /// Rejects a `step()` taken after the door opened (or the step budget ran
    /// out).
    ///
    /// Termination here is recomputed from the live grid every call —
    /// `door_is_open()` reads `self.state.grid` — so nothing latched the episode
    /// as over. Without this guard, a post-terminal step re-read an open door
    /// and paid `success_reward(self.steps, max_steps)` **again**, once per
    /// call, while `self.steps` kept advancing past the true episode length and
    /// decayed each successive payout; past `max_steps` the formula goes
    /// negative (`success_reward` deliberately does not clamp, "no env should
    /// call it past termination"), so the same finished episode could keep
    /// billing the agent.
    ///
    /// Worse, the door is re-toggleable: `dynamics::toggle` maps
    /// `Open -> Closed` unconditionally and `Closed -> Open` without needing the
    /// key. So after the winning `Toggle`, an unguarded agent could shut the
    /// door and re-open it, collecting the unlock reward a second, third, and
    /// n-th time from a single unlock — an unbounded return farmed out of one
    /// solved episode.
    guard: EpisodeGuard,
}

impl UnlockEnv {
    /// Emission-model visibility policy: does this environment's agent see
    /// through opaque cells?
    ///
    /// The rlevo spelling of canonical Minigrid's `see_through_walls`
    /// constructor argument. Read only by this environment's [`Sensor`] impl,
    /// and an inherent const rather than a config field because it is part of
    /// the task definition, not a knob a caller tunes.
    ///
    /// [`Visibility::Occluded`], because canonical `Unlock` derives from
    /// `RoomGrid`, and `minigrid/core/roomgrid.py`'s `RoomGrid.__init__` passes
    /// `see_through_walls=False` up to `MiniGridEnv` for every room-based env.
    /// See ADR 0063 (`docs/adr/0063-grid-visibility-occlusion.md`) for the
    /// whole twelve-environment table.
    const VISIBILITY: Visibility = Visibility::Occluded;

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
            guard: EpisodeGuard::new(),
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

    fn emit(&self, observation: GridObservation, reward: f32, done: bool) -> GridSnapshot {
        if self.render {
            println!("{}", self.ascii());
        }
        build_snapshot(observation, reward, done)
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

impl Sensor<3, 1, 3> for UnlockEnv {
    type Action = GridAction;
    type State = GridState;
    type Observation = GridObservation;

    /// Emission model `$O(a, s')$`. The observation is a function of the resulting
    /// `next_state` alone, so this forwards to the same projection as
    /// [`observe_reset`](Self::observe_reset).
    fn observe(&self, _action: &GridAction, next_state: &GridState) -> GridObservation {
        observe_grid(next_state, Self::VISIBILITY)
    }

    fn observe_reset(&self, state: &GridState) -> GridObservation {
        observe_grid(state, Self::VISIBILITY)
    }
}

impl Environment<3, 3, 1> for UnlockEnv {
    type StateType = GridState;
    type ObservationType = super::core::GridObservation;
    type ActionType = GridAction;
    type RewardType = ScalarReward;
    type SnapshotType = GridSnapshot;

    /// Rebuilds the fixed board and re-opens the episode.
    ///
    /// # Errors
    ///
    /// Currently infallible; always returns `Ok`.
    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.guard.reset();
        let (state, door_pos) = Self::build(&self.config);
        self.state = state;
        self.door_pos = door_pos;
        self.steps = 0;
        let observation = self.observe_reset(&self.state);
        Ok(self.emit(observation, 0.0, false))
    }

    /// Applies `action`, then ends the episode if the door is now open or the
    /// step budget is exhausted.
    ///
    /// # Errors
    ///
    /// Returns [`EnvironmentError::StepAfterEpisodeEnd`] if the episode has
    /// already ended; call [`reset`](Environment::reset) first.
    fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> {
        // Guard first: before `steps` advances and before `apply_action` touches
        // the grid, so a rejected call leaves the board, the step counter, and
        // the door state exactly as the terminal snapshot described them. Making
        // termination re-derivable from a mutated grid is precisely how the
        // unlock reward became farmable (ADR 0044); this env draws no
        // randomness, but the same ordering keeps a future draw off a rejected
        // path (ADR 0029).
        self.guard.check()?;

        self.steps += 1;
        let _ = apply_action(&mut self.state.grid, &mut self.state.agent, action);
        let (reward, done) = if self.door_is_open() {
            (success_reward(self.steps, self.config.max_steps), true)
        } else if self.steps >= self.config.max_steps {
            (0.0, true)
        } else {
            (0.0, false)
        };
        let observation = self.observe(&action, &self.state);
        // Single exit: one snapshot is built, and the guard is fed that
        // snapshot's own status rather than a re-derived literal.
        let snapshot = self.emit(observation, reward, done);
        self.guard.record(snapshot.status());
        Ok(snapshot)
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
    // Test-only: the `Visibility` dispatch point, the raw window extractor, and
    // the byte a masked cell carries — all read by
    // `test_unlock_occlusion_is_carried_entirely_by_the_locked_door`.
    use super::super::core::grid::egocentric_view;
    use super::super::core::{UNSEEN_TYPE, VIEW_SIZE, mask_view};
    // `Snapshot` arrives through `super::*`: `step()` reads `status()` off the
    // snapshot it emits, so the trait is now in scope crate-side.
    use rlevo_core::config::ConstraintKind;

    /// The four facings, for the occlusion sweep.
    const DIRECTIONS: [Direction; 4] = [
        Direction::North,
        Direction::East,
        Direction::South,
        Direction::West,
    ];

    /// Window cells `(row, col)` immediately behind the door at `(1, 0)`, as
    /// seen from the start cell facing North.
    ///
    /// A shut door's shadow covers far more of the window than these three (the
    /// perimeter it sits in accounts for the rest); these are the three whose
    /// visibility the *door's own state* controls — the ones an open door lights
    /// and a `Locked` or `Closed` one does not.
    const BEHIND_THE_DOOR: [(usize, usize); 3] = [(4, 2), (4, 3), (4, 4)];

    fn test_env() -> UnlockEnv {
        UnlockEnv::with_config(UnlockConfig::new(5, 100, 0), false).expect("valid config")
    }

    /// A same-shaped board of transparent cells, used as an **in-grid mask**.
    ///
    /// [`Grid::get`] reads out of bounds as [`Entity::Wall`] (matching canonical
    /// `Grid.slice`), so the window `egocentric_view` cuts from this board is
    /// [`Entity::Floor`] exactly where that window cell lies inside the world and
    /// [`Entity::Wall`] exactly where it does not. That is the view geometry read
    /// off the real extractor, rather than re-derived by a test-local copy of
    /// `rotate_view_offset` that could drift away from it.
    fn all_floor_like(grid: &Grid) -> Grid {
        let (w, h) = (grid.width(), grid.height());
        let mut probe = Grid::new(w, h);
        let (w, h) = (
            i32::try_from(w).expect("test grids fit in i32"),
            i32::try_from(h).expect("test grids fit in i32"),
        );
        for y in 0..h {
            for x in 0..w {
                probe.set(x, y, Entity::Floor);
            }
        }
        probe
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

    /// The `MIN_SIZE` validation-bypass fix: it was enforced only in [`FromStr`], so a config
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

    /// The finding this environment's occlusion story starts from: **nothing
    /// inside the room is ever hidden**, at any size, cell or facing.
    ///
    /// That is a consequence of a *known deviation*, not of the shadow cast.
    /// rlevo's `Unlock` draws **one** perimeter-walled room and writes the door
    /// into the outer north wall — see the module docs' "Known deviations"
    /// section; upstream `MiniGrid-Unlock-v0` is a two-room `RoomGrid` whose
    /// locked door sits on the wall *between* the rooms, with a whole room
    /// behind it to hide. Here the interior is a plain rectangle of
    /// transparent cells and `process_vis` is a forward flood fill, so every
    /// in-room cell — the key included — is lit from every pose. Once the
    /// layout builder places the door on an interior wall between two rooms,
    /// this test should be replaced by one that names the far room.
    ///
    /// Deliberately **not** a substitute for
    /// `test_unlock_occlusion_is_carried_entirely_by_the_locked_door`: this
    /// property is vacuously true under [`Visibility::SeeThrough`], so it cannot
    /// detect the emission model being switched off.
    #[test]
    fn test_unlock_occlusion_never_hides_an_in_room_cell() {
        for size in [4usize, 5, 8, 11] {
            let mut env = UnlockEnv::with_config(UnlockConfig::new(size, 100, 0), false)
                .expect("valid config");
            env.reset().expect("reset");
            let grid = env.state().grid.clone();
            let in_grid = all_floor_like(&grid);
            #[allow(clippy::cast_possible_wrap)]
            let side = size as i32;
            for y in 0..side {
                for x in 0..side {
                    if !grid.get(x, y).is_passable() {
                        continue;
                    }
                    for dir in DIRECTIONS {
                        let agent = AgentState::new(x, y, dir);
                        let masked = mask_view(&grid, &agent, UnlockEnv::VISIBILITY);
                        let inside = egocentric_view(&in_grid, &agent);
                        for r in 0..VIEW_SIZE {
                            for c in 0..VIEW_SIZE {
                                assert!(
                                    masked[r][c].is_some() || inside[r][c] == Entity::Wall,
                                    "size {size}: ({x}, {y}) facing {dir:?} masks the in-room \
                                     window cell ({r}, {c}) — the single room has no interior \
                                     occluder, so this is new geometry, not a shadow"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// The one thing occlusion *does* carry here: the door's own opacity, at the
    /// pose the whole task turns on.
    ///
    /// Standing at the start cell facing the door, the agent cannot see past it
    /// while it is `Locked`, still cannot while it is `Closed`, and can the
    /// moment it is `Open` — canonical `Door.see_behind == is_open`, driven
    /// through the real `Pickup → Toggle → Toggle` chain rather than by
    /// hand-editing a grid. Exactly three window cells change hands, and they are
    /// named.
    ///
    /// What lies behind that door is the outside of the world rather than a
    /// second room, for the reason
    /// `test_unlock_occlusion_never_hides_an_in_room_cell` documents: the door
    /// sits in the perimeter wall rather than an interior wall between two
    /// rooms. This is therefore the strongest honest assertion available for
    /// this environment, and it is still a real one: flip
    /// [`UnlockEnv::VISIBILITY`] to [`Visibility::SeeThrough`] and it fails.
    #[test]
    fn test_unlock_occlusion_is_carried_entirely_by_the_locked_door() {
        let mut env = test_env();
        env.reset().expect("reset");
        let picked = env.step(GridAction::Pickup).expect("take the key");
        // The `Pickup` genuinely lands: the agent starts at (1, 1) facing East
        // with the key at (2, 1), so `dynamics::pickup` finds a pickable entity
        // directly in front. Pin the *colour* byte the carried key reports in
        // the agent's own view cell — `dynamics.rs:126` unlocks only when
        // `agent.carrying == Some(Entity::Key(color))` for the door's colour, so
        // channel 1 here is the byte the door gate compares against. Channel 2
        // is a literal `0`: carryables inherit `WorldObj.encode` and only
        // `Door` writes a state byte.
        assert_eq!(
            picked.observation().view[VIEW_SIZE - 1][VIEW_SIZE / 2],
            [Entity::Key(DOOR_COLOR).type_u8(), DOOR_COLOR.to_u8(), 0],
            "the carried key — colour included — must reach the agent's own view cell"
        );

        let locked = env.step(GridAction::TurnLeft).expect("face the door");

        let agent = env.state().agent;
        assert_eq!(
            (agent.x, agent.y, agent.direction),
            (1, 1, Direction::North),
            "the agent must be at the start cell facing the door"
        );
        assert_eq!(
            env.state().grid.get(1, 0),
            Entity::Door(DOOR_COLOR, DoorState::Locked),
            "the door must still be locked at this point of the chain"
        );

        let masked = mask_view(&env.state().grid, &agent, UnlockEnv::VISIBILITY);
        let occluded = *locked.observation();
        let see_through = observe_grid(env.state(), Visibility::SeeThrough);

        assert_eq!(
            masked[5][3],
            Some(Entity::Door(DOOR_COLOR, DoorState::Locked)),
            "the door one cell ahead is itself visible — the agent sees what blocks it"
        );
        // The three cells the door itself governs: unseen while it is shut.
        for (row, col) in BEHIND_THE_DOOR {
            assert_eq!(
                masked[row][col], None,
                "({row}, {col}) is behind the locked door and must be unseen"
            );
            assert_eq!(
                occluded.view[row][col],
                [UNSEEN_TYPE, 0, 0],
                "({row}, {col}) must encode as the unseen triple"
            );
            assert_eq!(
                see_through.view[row][col][0],
                Entity::Wall.type_u8(),
                "the old see-through model reported straight through the locked door"
            );
        }

        // Differ *exactly* on the hidden set, in both directions.
        let hidden_locked = masked.iter().flatten().filter(|c| c.is_none()).count();
        for (r, c) in (0..VIEW_SIZE).flat_map(|r| (0..VIEW_SIZE).map(move |c| (r, c))) {
            if masked[r][c].is_none() {
                assert_eq!(
                    occluded.view[r][c],
                    [UNSEEN_TYPE, 0, 0],
                    "hidden cell ({r}, {c}) must encode as the unseen triple"
                );
                assert_ne!(
                    occluded.view[r][c], see_through.view[r][c],
                    "hidden cell ({r}, {c}) must not encode the same as the seen one"
                );
            } else {
                assert_eq!(
                    occluded.view[r][c], see_through.view[r][c],
                    "visible cell ({r}, {c}) must be unaffected by occlusion"
                );
            }
        }

        // Unlocking is not opening: a `Closed` door is still opaque.
        env.step(GridAction::Toggle).expect("unlock");
        assert_eq!(
            env.state().grid.get(1, 0),
            Entity::Door(DOOR_COLOR, DoorState::Closed),
            "the first toggle unlocks without opening"
        );
        let closed = mask_view(&env.state().grid, &agent, UnlockEnv::VISIBILITY);
        assert_eq!(
            closed.iter().flatten().filter(|c| c.is_none()).count(),
            hidden_locked,
            "a Closed door occludes exactly as a Locked one does — see_behind is is_open"
        );

        // Opening it hands back precisely the three cells behind it.
        let opened = env.step(GridAction::Toggle).expect("open");
        assert!(opened.is_done(), "opening the door ends the episode");
        let after = mask_view(&env.state().grid, &env.state().agent, UnlockEnv::VISIBILITY);
        for (row, col) in BEHIND_THE_DOOR {
            assert_eq!(
                after[row][col],
                Some(Entity::Wall),
                "opening the door reveals ({row}, {col}) behind it"
            );
        }
        assert_eq!(
            after.iter().flatten().filter(|c| c.is_none()).count(),
            hidden_locked - BEHIND_THE_DOOR.len(),
            "exactly the three cells the open door lights may change hands"
        );

        assert_eq!(
            UnlockEnv::VISIBILITY,
            Visibility::Occluded,
            "Unlock derives from RoomGrid, which passes see_through_walls=False"
        );
    }

    /// The carried-item visibility fix, end to end: [`AgentState::carrying`]
    /// must reach the emitted observation.
    ///
    /// Before the fix, `mask_view` returned the masked window untouched and the
    /// agent's own cell reported the *world* tile it stands on, so picking the
    /// key up changed nothing the policy could see — the one bit the whole task
    /// turns on was invisible. `grid::stamp_carried` now writes
    /// `view[VIEW_SIZE - 1][VIEW_SIZE / 2]` after the visibility match, matching
    /// canonical `MiniGridEnv.gen_obs_grid`.
    ///
    /// The reproducer is the observation *delta* across a single `Pickup`, and
    /// it must move **exactly two** cells:
    ///
    /// * the agent's own cell `(6, 3)` — `Empty` → `Key(DOOR_COLOR)`, the stamp
    ///   under test;
    /// * the key's cell `(5, 3)` — `Key(DOOR_COLOR)` → `Empty`, because
    ///   `dynamics::pickup` clears the key out of the world grid, and that cell
    ///   (one step ahead of an agent at `(1, 1)` facing East) lies inside the
    ///   view window.
    ///
    /// Nothing else in the window may move: `Pickup` neither turns nor walks the
    /// agent, so the geometry is fixed and every other cell is byte-identical.
    #[test]
    fn test_unlock_carried_key_reaches_the_observation() {
        /// The agent's own cell: where the carried item is stamped.
        const AGENT_CELL: (usize, usize) = (VIEW_SIZE - 1, VIEW_SIZE / 2);
        /// The key's world cell (2, 1), seen one step ahead from the start pose.
        const KEY_CELL: (usize, usize) = (VIEW_SIZE - 2, VIEW_SIZE / 2);

        let mut env = test_env();
        let before = *env.reset().expect("reset").observation();
        assert_eq!(
            env.state().agent.carrying,
            None,
            "the agent starts empty-handed"
        );

        let after = *env
            .step(GridAction::Pickup)
            .expect("take the key")
            .observation();
        assert_eq!(
            env.state().agent.carrying,
            Some(Entity::Key(DOOR_COLOR)),
            "the Pickup must actually land, or this test proves nothing"
        );

        let moved: Vec<(usize, usize)> = (0..VIEW_SIZE)
            .flat_map(|r| (0..VIEW_SIZE).map(move |c| (r, c)))
            .filter(|&(r, c)| before.view[r][c] != after.view[r][c])
            .collect();
        assert_eq!(
            moved,
            vec![KEY_CELL, AGENT_CELL],
            "a Pickup must change exactly the key's cell and the agent's own cell"
        );

        // The agent's cell before the Pickup: `Some(Entity::Empty)`, never the
        // unseen triple — `stamp_carried` spells the empty hand as `Empty`
        // precisely so it cannot collide with an occluded cell.
        assert_eq!(
            before.view[AGENT_CELL.0][AGENT_CELL.1],
            [Entity::Empty.type_u8(), 0, 0],
            "an empty-handed agent reports Empty in its own cell, not the unseen triple"
        );
        // And after: the key, with its colour, and a literal `0` state byte —
        // carryables inherit `WorldObj.encode`; only `Door` has a state byte.
        assert_eq!(
            after.view[AGENT_CELL.0][AGENT_CELL.1],
            [Entity::Key(DOOR_COLOR).type_u8(), DOOR_COLOR.to_u8(), 0],
            "the carried key must be encoded into the agent's own cell"
        );
        // The other half of the delta: the key left the world grid.
        assert_eq!(
            before.view[KEY_CELL.0][KEY_CELL.1],
            [Entity::Key(DOOR_COLOR).type_u8(), DOOR_COLOR.to_u8(), 0],
            "the key sits one cell ahead of the start pose"
        );
        assert_eq!(
            after.view[KEY_CELL.0][KEY_CELL.1],
            [Entity::Empty.type_u8(), 0, 0],
            "picking the key up clears it from the world grid"
        );
    }

    // ── post-terminal step guard (ADR 0044) ───────────────────────────────────

    /// Drives a fresh episode to the winning terminal snapshot the honest way:
    /// through real `step()` calls, ending by **unlocking and opening the door**
    /// rather than by exhausting `max_steps`.
    ///
    /// The step-limit ending is deliberately avoided: `step()` OR-s the
    /// genuine terminal condition (door open) with the `steps >= max_steps`
    /// cutoff into one collapsed `done: bool`, and `build_snapshot` maps that
    /// bool straight to `Terminated`/`running` with no `Truncated` arm — so a
    /// timeout reaches the same terminal status as a real win, biasing the
    /// value target for any bootstrapping agent. A timeout-driven conformance
    /// test would bake in that wrong status. Opening the door is a genuine
    /// termination.
    fn drive_to_open_door(env: &mut UnlockEnv) -> GridSnapshot {
        env.reset().expect("reset must succeed");
        let script = [
            GridAction::Pickup,   // grab the key
            GridAction::TurnLeft, // face north toward the door
            GridAction::Toggle,   // unlock (Locked -> Closed)
            GridAction::Toggle,   // open   (Closed -> Open)
        ];
        let mut last = None;
        for a in script {
            last = Some(env.step(a).expect("the winning script must be steppable"));
        }
        last.expect("the script is non-empty")
    }

    /// Conformance with the shared post-terminal contract: once the door is
    /// open, a further legal `step()` fails with `StepAfterEpisodeEnd` carrying
    /// the status that ended the episode.
    #[test]
    fn test_unlock_rejects_post_terminal_step() {
        let mut env = test_env();
        crate::episode::assert_rejects_post_terminal_step(
            &mut env,
            drive_to_open_door,
            GridAction::Toggle,
        );
    }

    /// Regression for the concrete defect the guard prevents.
    ///
    /// Termination is recomputed from the live grid (`door_is_open()`), so an
    /// unguarded post-terminal step advanced `self.steps`, re-ran
    /// `apply_action`, and paid `success_reward` again on the still-open door —
    /// and a replayed `Toggle` shut the door (`Open -> Closed`, no key needed),
    /// so the very next `Toggle` re-opened it and billed the unlock reward a
    /// second time. A rejected step must mutate nothing: not the counter, not
    /// the agent's pose, not the door.
    #[test]
    fn test_unlock_post_terminal_step_cannot_re_pay_the_unlock_reward() {
        use rlevo_core::environment::EpisodeStatus;

        let mut env = test_env();
        let terminal = drive_to_open_door(&mut env);
        assert!(terminal.is_done(), "opening the door must terminate");
        let paid: f32 = (*terminal.reward()).into();
        assert!(paid > 0.0, "the unlock reward was {paid}");

        let steps_at_end = env.steps();
        let agent_at_end = env.state().agent;
        let door_at_end = env.state().grid.get(env.door_pos().0, env.door_pos().1);
        assert_eq!(
            door_at_end,
            Entity::Door(DOOR_COLOR, DoorState::Open),
            "the episode ended with the door open"
        );

        let err = env
            .step(GridAction::Toggle)
            .expect_err("a step after the door opened must return Err, not another payout");
        match err {
            EnvironmentError::StepAfterEpisodeEnd { status } => assert_eq!(
                status,
                terminal.status(),
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
            (
                env.state().agent.x,
                env.state().agent.y,
                env.state().agent.direction
            ),
            (agent_at_end.x, agent_at_end.y, agent_at_end.direction),
            "a rejected step must not move or turn the agent"
        );
        assert_eq!(
            env.state().grid.get(env.door_pos().0, env.door_pos().1),
            door_at_end,
            "a rejected Toggle must not shut the door and re-arm another unlock reward"
        );
        assert_eq!(
            env.guard.status(),
            EpisodeStatus::Terminated,
            "a rejected step must not reopen the episode"
        );
    }

    /// `reset()` re-opens a finished episode, so a latched guard cannot strand
    /// the environment for the rest of the run.
    #[test]
    fn test_unlock_reset_reopens_terminated_episode() {
        use rlevo_core::environment::EpisodeStatus;

        let mut env = test_env();
        drive_to_open_door(&mut env);
        assert!(
            env.step(GridAction::Toggle).is_err(),
            "the episode has ended; a step must be rejected before reset()"
        );

        env.reset().expect("reset must succeed after termination");
        assert_eq!(
            env.guard.status(),
            EpisodeStatus::Running,
            "reset() must return the guard to Running"
        );
        assert_eq!(
            env.state().grid.get(1, 0),
            Entity::Door(DOOR_COLOR, DoorState::Locked),
            "reset() must re-lock the door for the new episode"
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
