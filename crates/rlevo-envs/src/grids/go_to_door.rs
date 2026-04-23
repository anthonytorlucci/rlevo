//! `GoToDoor`: mission-conditioned navigation to a colored door.
//!
//! Ports Farama Minigrid's [`GoToDoorEnv`]. Four colored doors are
//! embedded in the four perimeter walls of an otherwise empty room. The
//! per-episode [`Mission`] specifies a target color; the agent solves
//! the task by issuing [`GridAction::Done`] while facing a door of that
//! color. Doing so anywhere else (or facing a door of a different color)
//! terminates the episode with reward `0.0`.
//!
//! This is the first grid environment with a mission field — the target
//! color is stored on the env and exposed via [`GoToDoorEnv::mission`]
//! so callers can build instruction-conditioned policies.
//!
//! [`GoToDoorEnv`]: https://minigrid.farama.org/environments/minigrid/GoToDoorEnv/

use super::core::{
    GridSnapshot,
    action::GridAction,
    agent::AgentState,
    build_snapshot,
    color::Color,
    direction::Direction,
    dynamics::{StepOutcome, apply_action},
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

/// Minimum side length; we need at least one interior cell and one
/// central slot on each perimeter wall for a door.
const MIN_SIZE: usize = 5;

/// Instruction the agent must fulfil in a given episode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Mission {
    /// Color of the door the agent must face when emitting [`GridAction::Done`].
    pub target_color: Color,
}

impl Mission {
    /// Construct a new mission targeting the given color.
    #[must_use]
    pub const fn new(target_color: Color) -> Self {
        Self { target_color }
    }

    /// Short plain-text rendering of the mission.
    #[must_use]
    pub fn describe(&self) -> String {
        format!("go to the {:?} door", self.target_color)
    }
}

/// Configuration for [`GoToDoorEnv`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoToDoorConfig {
    pub size: usize,
    pub max_steps: usize,
    pub seed: u64,
    pub target_color: Color,
}

impl GoToDoorConfig {
    #[must_use]
    pub const fn new(size: usize, max_steps: usize, seed: u64, target_color: Color) -> Self {
        Self {
            size,
            max_steps,
            seed,
            target_color,
        }
    }
}

impl Default for GoToDoorConfig {
    fn default() -> Self {
        let size = 6;
        Self {
            size,
            max_steps: 4 * size * size,
            seed: 0,
            target_color: Color::Red,
        }
    }
}

fn parse_color(s: &str) -> Result<Color, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "red" => Ok(Color::Red),
        "green" => Ok(Color::Green),
        "blue" => Ok(Color::Blue),
        "purple" => Ok(Color::Purple),
        "yellow" => Ok(Color::Yellow),
        "grey" | "gray" => Ok(Color::Grey),
        other => Err(format!("unknown color `{other}`")),
    }
}

impl FromStr for GoToDoorConfig {
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
                    "target_color" | "color" => cfg.target_color = parse_color(value)?,
                    other => return Err(format!("unknown key `{other}`")),
                }
            } else {
                match idx {
                    0 => cfg.size = raw.parse().map_err(|e| format!("size: {e}"))?,
                    1 => cfg.max_steps = raw.parse().map_err(|e| format!("max_steps: {e}"))?,
                    2 => cfg.seed = raw.parse().map_err(|e| format!("seed: {e}"))?,
                    3 => cfg.target_color = parse_color(raw)?,
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

/// Minigrid's `GoToDoor` environment.
#[derive(Debug)]
pub struct GoToDoorEnv {
    state: GridState,
    config: GoToDoorConfig,
    steps: usize,
    render: bool,
    mission: Mission,
    _rng: StdRng,
}

impl GoToDoorEnv {
    #[must_use]
    pub fn with_config(config: GoToDoorConfig, render: bool) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        let state = Self::build(&config);
        let mission = Mission::new(config.target_color);
        Self {
            state,
            config,
            steps: 0,
            render,
            mission,
            _rng: rng,
        }
    }

    #[must_use]
    pub const fn config(&self) -> &GoToDoorConfig {
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
    pub const fn mission(&self) -> &Mission {
        &self.mission
    }

    #[must_use]
    pub fn ascii(&self) -> String {
        render_ascii(&self.state.grid, &self.state.agent)
    }

    fn build(config: &GoToDoorConfig) -> GridState {
        let mut grid = Grid::new(config.size, config.size);
        grid.draw_walls();
        #[allow(clippy::cast_possible_wrap)]
        let size = config.size as i32;
        // Place one colored door on each perimeter wall at its midpoint.
        let mid_x = size / 2;
        let mid_y = size / 2;
        let closed = DoorState::Closed;
        grid.set(mid_x, 0, Entity::Door(Color::Red, closed)); // North wall
        grid.set(size - 1, mid_y, Entity::Door(Color::Green, closed)); // East wall
        grid.set(mid_x, size - 1, Entity::Door(Color::Blue, closed)); // South wall
        grid.set(0, mid_y, Entity::Door(Color::Yellow, closed)); // West wall
        let agent_pos = (mid_x - 1).max(1);
        let agent = AgentState::new(agent_pos, agent_pos, Direction::East);
        GridState::new(grid, agent)
    }

    fn emit(&self, reward: f32, done: bool) -> GridSnapshot {
        if self.render {
            println!("{}", self.ascii());
        }
        build_snapshot(&self.state, reward, done)
    }

    /// Color of the door currently in front of the agent, if any.
    fn door_in_front_color(&self) -> Option<Color> {
        let (fx, fy) = self.state.agent.front();
        match self.state.grid.get(fx, fy) {
            Entity::Door(color, _) => Some(color),
            _ => None,
        }
    }
}

impl Display for GoToDoorEnv {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GoToDoorEnv(target={:?}, step={}/{})",
            self.config.target_color, self.steps, self.config.max_steps
        )
    }
}

impl Environment<3, 3, 1> for GoToDoorEnv {
    type StateType = GridState;
    type ObservationType = super::core::GridObservation;
    type ActionType = GridAction;
    type RewardType = ScalarReward;
    type SnapshotType = GridSnapshot;

    fn new(render: bool) -> Self {
        Self::with_config(GoToDoorConfig::default(), render)
    }

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.state = Self::build(&self.config);
        self.mission = Mission::new(self.config.target_color);
        self.steps = 0;
        self._rng = StdRng::seed_from_u64(self.config.seed);
        Ok(self.emit(0.0, false))
    }

    fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> {
        self.steps += 1;
        let outcome = apply_action(&mut self.state.grid, &mut self.state.agent, action);
        let (reward, done) = match outcome {
            StepOutcome::DoneAction => {
                if self.door_in_front_color() == Some(self.mission.target_color) {
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

    fn env_6x6(target: Color) -> GoToDoorEnv {
        GoToDoorEnv::with_config(GoToDoorConfig::new(6, 100, 0, target), false)
    }

    #[test]
    fn default_config_targets_red() {
        let cfg = GoToDoorConfig::default();
        assert_eq!(cfg.size, 6);
        assert_eq!(cfg.target_color, Color::Red);
    }

    #[test]
    fn fromstr_keyvalue_parses_color() {
        let cfg: GoToDoorConfig = "target_color=blue".parse().unwrap();
        assert_eq!(cfg.target_color, Color::Blue);
    }

    #[test]
    fn fromstr_rejects_unknown_color() {
        assert!("color=cyan".parse::<GoToDoorConfig>().is_err());
    }

    #[test]
    fn fromstr_rejects_small_size() {
        assert!("3".parse::<GoToDoorConfig>().is_err());
    }

    #[test]
    fn build_places_four_colored_doors() {
        let env = env_6x6(Color::Red);
        assert!(matches!(
            env.state().grid.get(3, 0),
            Entity::Door(Color::Red, DoorState::Closed)
        ));
        assert!(matches!(
            env.state().grid.get(5, 3),
            Entity::Door(Color::Green, DoorState::Closed)
        ));
        assert!(matches!(
            env.state().grid.get(3, 5),
            Entity::Door(Color::Blue, DoorState::Closed)
        ));
        assert!(matches!(
            env.state().grid.get(0, 3),
            Entity::Door(Color::Yellow, DoorState::Closed)
        ));
    }

    #[test]
    fn mission_matches_config_target() {
        let env = env_6x6(Color::Green);
        assert_eq!(env.mission().target_color, Color::Green);
        assert!(env.mission().describe().contains("Green"));
    }

    #[test]
    fn optimal_rollout_red_door_succeeds() {
        let mut env = env_6x6(Color::Red);
        env.reset().unwrap();
        // Agent at (2, 2) facing East. Red door at (3, 0).
        let script = [
            GridAction::Forward,  // → (3, 2)
            GridAction::TurnLeft, // face north
            GridAction::Forward,  // → (3, 1)
            GridAction::Done,     // facing Red door at (3, 0)
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
    fn optimal_rollout_green_door_succeeds() {
        let mut env = env_6x6(Color::Green);
        env.reset().unwrap();
        // Green door at (5, 3). Agent at (2, 2) facing East.
        let script = [
            GridAction::Forward,   // → (3, 2)
            GridAction::Forward,   // → (4, 2)
            GridAction::TurnRight, // face south
            GridAction::Forward,   // → (4, 3)
            GridAction::TurnLeft,  // face east
            GridAction::Done,      // facing Green door at (5, 3)
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
    fn done_at_wrong_door_terminates_with_zero() {
        let mut env = env_6x6(Color::Red);
        env.reset().unwrap();
        // Drive to the Green door and issue Done — wrong color.
        let script = [
            GridAction::Forward,
            GridAction::Forward,
            GridAction::TurnRight,
            GridAction::Forward,
            GridAction::TurnLeft,
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
    fn closed_doors_are_impassable() {
        let mut env = env_6x6(Color::Red);
        env.reset().unwrap();
        // Walk into the red door: Forward, TurnLeft, Forward x 2 should bump.
        env.step(GridAction::Forward).unwrap();
        env.step(GridAction::TurnLeft).unwrap();
        env.step(GridAction::Forward).unwrap(); // (3, 1)
        let snap = env.step(GridAction::Forward).unwrap();
        assert!(!snap.is_done());
        assert_eq!(
            env.state().agent.y,
            1,
            "should have bumped into closed door"
        );
    }

    #[test]
    fn reset_is_deterministic() {
        let cfg = GoToDoorConfig::new(6, 100, 9, Color::Yellow);
        let mut a = GoToDoorEnv::with_config(cfg, false);
        let mut b = GoToDoorEnv::with_config(cfg, false);
        let sa = a.reset().unwrap();
        let sb = b.reset().unwrap();
        assert_eq!(sa.observation(), sb.observation());
        assert_eq!(a.mission(), b.mission());
    }
}
