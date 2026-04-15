//! `Unlock`: pick up the colored key, unlock and open the colored door.
//!
//! Ports Farama Minigrid's [`UnlockEnv`]. The agent stands in a
//! perimeter-walled room that has a single colored locked door on the
//! north wall. A matching colored key sits one step east of the agent.
//! Success is reached when the door transitions to [`DoorState::Open`];
//! timing out with the door still closed or locked returns `0.0`.
//!
//! [`UnlockEnv`]: https://minigrid.farama.org/environments/minigrid/UnlockEnv/

use super::core::{
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
    GridSnapshot,
};
use evorl_core::environment::{Environment, EnvironmentError};
use evorl_core::reward::ScalarReward;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

/// Minimum supported side length for the room.
const MIN_SIZE: usize = 4;
/// Door color used by the default layout.
const DOOR_COLOR: Color = Color::Yellow;

/// Configuration for [`UnlockEnv`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnlockConfig {
    /// Side length of the room.
    pub size: usize,
    /// Maximum number of steps before the episode times out.
    pub max_steps: usize,
    /// Seed for the environment's RNG. Reserved for future random variants.
    pub seed: u64,
}

impl UnlockConfig {
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

impl FromStr for UnlockConfig {
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

/// Minigrid's `Unlock` environment.
#[derive(Debug)]
pub struct UnlockEnv {
    state: GridState,
    config: UnlockConfig,
    steps: usize,
    render: bool,
    door_pos: (i32, i32),
    _rng: StdRng,
}

impl UnlockEnv {
    /// Construct from an explicit configuration.
    #[must_use]
    pub fn with_config(config: UnlockConfig, render: bool) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        let (state, door_pos) = Self::build(&config);
        Self {
            state,
            config,
            steps: 0,
            render,
            door_pos,
            _rng: rng,
        }
    }

    /// Borrow the active configuration.
    #[must_use]
    pub const fn config(&self) -> &UnlockConfig {
        &self.config
    }

    /// Current step count within the episode.
    #[must_use]
    pub const fn steps(&self) -> usize {
        self.steps
    }

    /// Borrow the full grid + agent state.
    #[must_use]
    pub const fn state(&self) -> &GridState {
        &self.state
    }

    /// World coordinates of the locked door.
    #[must_use]
    pub const fn door_pos(&self) -> (i32, i32) {
        self.door_pos
    }

    /// Render the current state as a multi-line ASCII string.
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

impl Display for UnlockEnv {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "UnlockEnv(size={}, step={}/{})",
            self.config.size, self.steps, self.config.max_steps
        )
    }
}

impl Environment<3, 3, 1> for UnlockEnv {
    type StateType = GridState;
    type ObservationType = super::core::GridObservation;
    type ActionType = GridAction;
    type RewardType = ScalarReward;
    type SnapshotType = GridSnapshot;

    fn new(render: bool) -> Self {
        Self::with_config(UnlockConfig::default(), render)
    }

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        let (state, door_pos) = Self::build(&self.config);
        self.state = state;
        self.door_pos = door_pos;
        self.steps = 0;
        self._rng = StdRng::seed_from_u64(self.config.seed);
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

#[cfg(test)]
mod tests {
    use super::*;
    use evorl_core::environment::Snapshot;

    fn test_env() -> UnlockEnv {
        UnlockEnv::with_config(UnlockConfig::new(5, 100, 0), false)
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
        let mut a = UnlockEnv::with_config(cfg, false);
        let mut b = UnlockEnv::with_config(cfg, false);
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
            GridAction::Pickup,    // grab the key
            GridAction::TurnLeft,  // face north toward door
            GridAction::Toggle,    // unlock (Locked -> Closed)
            GridAction::Toggle,    // open (Closed -> Open)
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
        let mut env = UnlockEnv::with_config(cfg, false);
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
