//! `UnlockPickup`: unlock a door and pick up the colored box.
//!
//! Ports Farama Minigrid's [`UnlockPickupEnv`]. Two rooms are separated
//! by an interior wall with a single locked door. A matching key sits in
//! the starting room; a target colored box sits in the far room. Success
//! is reached when the agent is carrying the target box.
//!
//! [`UnlockPickupEnv`]: https://minigrid.farama.org/environments/minigrid/UnlockPickupEnv/

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
use rand::SeedableRng;
use rand::rngs::StdRng;
use rlevo_core::environment::{Environment, EnvironmentError};
use rlevo_core::reward::ScalarReward;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

/// Minimum side length; with the interior wall we need at least three
/// interior columns per side.
const MIN_SIZE: usize = 7;
const DOOR_COLOR: Color = Color::Yellow;
const BOX_COLOR: Color = Color::Purple;

/// Configuration for [`UnlockPickupEnv`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnlockPickupConfig {
    pub size: usize,
    pub max_steps: usize,
    pub seed: u64,
}

impl UnlockPickupConfig {
    #[must_use]
    pub const fn new(size: usize, max_steps: usize, seed: u64) -> Self {
        Self {
            size,
            max_steps,
            seed,
        }
    }
}

impl Default for UnlockPickupConfig {
    fn default() -> Self {
        let size = 7;
        Self {
            size,
            max_steps: 4 * size * size,
            seed: 0,
        }
    }
}

impl FromStr for UnlockPickupConfig {
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

/// Minigrid's `UnlockPickup` environment.
#[derive(Debug)]
pub struct UnlockPickupEnv {
    state: GridState,
    config: UnlockPickupConfig,
    steps: usize,
    render: bool,
    target: Entity,
    _rng: StdRng,
}

impl UnlockPickupEnv {
    #[must_use]
    pub fn with_config(config: UnlockPickupConfig, render: bool) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        let state = Self::build(&config);
        Self {
            state,
            config,
            steps: 0,
            render,
            target: Entity::Box(BOX_COLOR),
            _rng: rng,
        }
    }

    #[must_use]
    pub const fn config(&self) -> &UnlockPickupConfig {
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

    /// Entity the agent must be holding to solve the task.
    #[must_use]
    pub const fn target(&self) -> Entity {
        self.target
    }

    #[must_use]
    pub fn ascii(&self) -> String {
        render_ascii(&self.state.grid, &self.state.agent)
    }

    fn build(config: &UnlockPickupConfig) -> GridState {
        let mut grid = Grid::new(config.size, config.size);
        grid.draw_walls();

        #[allow(clippy::cast_possible_wrap)]
        let size = config.size as i32;
        #[allow(clippy::cast_possible_wrap)]
        let split_col = (config.size / 2) as i32;

        // Interior vertical wall with a locked door.
        for y in 1..size - 1 {
            grid.set(split_col, y, Entity::Wall);
        }
        grid.set(
            split_col,
            split_col,
            Entity::Door(DOOR_COLOR, DoorState::Locked),
        );

        // Key in the left room; box one cell east of the door in the right room.
        grid.set(1, 1, Entity::Key(DOOR_COLOR));
        grid.set(split_col + 1, split_col, Entity::Box(BOX_COLOR));

        let agent = AgentState::new(1, 2, Direction::North);
        GridState::new(grid, agent)
    }

    fn emit(&self, reward: f32, done: bool) -> GridSnapshot {
        if self.render {
            println!("{}", self.ascii());
        }
        build_snapshot(&self.state, reward, done)
    }

    fn has_target(&self) -> bool {
        self.state.agent.carrying == Some(self.target)
    }
}

impl Display for UnlockPickupEnv {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "UnlockPickupEnv(size={}, step={}/{})",
            self.config.size, self.steps, self.config.max_steps
        )
    }
}

impl Environment<3, 3, 1> for UnlockPickupEnv {
    type StateType = GridState;
    type ObservationType = super::core::GridObservation;
    type ActionType = GridAction;
    type RewardType = ScalarReward;
    type SnapshotType = GridSnapshot;

    fn new(render: bool) -> Self {
        Self::with_config(UnlockPickupConfig::default(), render)
    }

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.state = Self::build(&self.config);
        self.steps = 0;
        self._rng = StdRng::seed_from_u64(self.config.seed);
        Ok(self.emit(0.0, false))
    }

    fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> {
        self.steps += 1;
        let _ = apply_action(&mut self.state.grid, &mut self.state.agent, action);
        let (reward, done) = if self.has_target() {
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
    use rlevo_core::environment::Snapshot;

    fn env_7x7() -> UnlockPickupEnv {
        UnlockPickupEnv::with_config(UnlockPickupConfig::new(7, 196, 0), false)
    }

    #[test]
    fn default_config_is_7x7() {
        let cfg = UnlockPickupConfig::default();
        assert_eq!(cfg.size, 7);
        assert_eq!(cfg.max_steps, 4 * 7 * 7);
    }

    #[test]
    fn fromstr_rejects_small_size() {
        assert!("5".parse::<UnlockPickupConfig>().is_err());
    }

    #[test]
    fn fromstr_parses_all_fields() {
        let cfg: UnlockPickupConfig = "size=8,max_steps=200,seed=4".parse().unwrap();
        assert_eq!(cfg.size, 8);
        assert_eq!(cfg.max_steps, 200);
        assert_eq!(cfg.seed, 4);
    }

    #[test]
    fn build_places_door_key_and_box() {
        let env = env_7x7();
        let grid = &env.state().grid;
        assert_eq!(grid.get(3, 3), Entity::Door(DOOR_COLOR, DoorState::Locked));
        assert_eq!(grid.get(1, 1), Entity::Key(DOOR_COLOR));
        assert_eq!(grid.get(4, 3), Entity::Box(BOX_COLOR));
        assert_eq!(env.state().agent.x, 1);
        assert_eq!(env.state().agent.y, 2);
        assert_eq!(env.state().agent.direction, Direction::North);
    }

    #[test]
    fn reset_is_deterministic() {
        let cfg = UnlockPickupConfig::new(7, 196, 42);
        let mut a = UnlockPickupEnv::with_config(cfg, false);
        let mut b = UnlockPickupEnv::with_config(cfg, false);
        let sa = a.reset().unwrap();
        let sb = b.reset().unwrap();
        assert_eq!(sa.observation(), sb.observation());
    }

    #[test]
    fn optimal_rollout_unlocks_and_grabs_box() {
        let mut env = env_7x7();
        env.reset().unwrap();
        // Script described in module comments: grab key, navigate to the
        // door, unlock + open, drop key in the left room before entering,
        // then cross and pick up the box.
        let script = [
            GridAction::Pickup,    //  1. grab yellow key at (1,1)
            GridAction::TurnRight, //  2. face east
            GridAction::Forward,   //  3. → (2,2)
            GridAction::TurnRight, //  4. face south
            GridAction::Forward,   //  5. → (2,3)
            GridAction::TurnLeft,  //  6. face east
            GridAction::Toggle,    //  7. unlock door at (3,3)
            GridAction::Toggle,    //  8. open door
            GridAction::TurnRight, //  9. face south
            GridAction::Drop,      // 10. drop key at (2,4)
            GridAction::TurnLeft,  // 11. face east
            GridAction::Forward,   // 12. → (3,3) through door
            GridAction::Forward,   // 13. bump box at (4,3)
            GridAction::Pickup,    // 14. grab box
        ];
        let mut last = None;
        for a in script {
            last = Some(env.step(a).unwrap());
        }
        let snap = last.unwrap();
        assert!(snap.is_done());
        assert_eq!(env.state().agent.carrying, Some(Entity::Box(BOX_COLOR)));
        let reward: f32 = (*snap.reward()).into();
        assert!(reward > 0.9, "reward was {reward}");
    }

    #[test]
    fn picking_up_wrong_object_does_not_terminate() {
        let mut env = env_7x7();
        env.reset().unwrap();
        // Grab the key (not the target). The env should not terminate.
        let snap = env.step(GridAction::Pickup).unwrap();
        assert!(!snap.is_done());
        assert_eq!(env.state().agent.carrying, Some(Entity::Key(DOOR_COLOR)));
    }

    #[test]
    fn timeout_terminates_with_zero_reward() {
        let cfg = UnlockPickupConfig::new(7, 3, 0);
        let mut env = UnlockPickupEnv::with_config(cfg, false);
        env.reset().unwrap();
        env.step(GridAction::TurnLeft).unwrap();
        env.step(GridAction::TurnLeft).unwrap();
        let snap = env.step(GridAction::TurnLeft).unwrap();
        assert!(snap.is_done());
        let reward: f32 = (*snap.reward()).into();
        assert_eq!(reward, 0.0);
    }

    #[test]
    fn target_is_box_of_declared_color() {
        let env = env_7x7();
        assert_eq!(env.target(), Entity::Box(BOX_COLOR));
    }

    #[test]
    fn reset_clears_episode_state() {
        let mut env = env_7x7();
        env.reset().unwrap();
        env.step(GridAction::Pickup).unwrap();
        assert_eq!(env.state().agent.carrying, Some(Entity::Key(DOOR_COLOR)));
        env.reset().unwrap();
        assert_eq!(env.state().agent.carrying, None);
        assert_eq!(env.steps(), 0);
    }
}
