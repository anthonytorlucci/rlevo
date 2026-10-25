//! Your environment.
//!
//! `MyEnv` works as it stands, so the crate runs from its first `cargo run`:
//! a corridor of `LENGTH` cells. An episode starts on a random cell, each
//! step moves one cell left or right, and reaching the right-hand end pays
//! 1 and ends the episode. Replace the corridor with your task, and keep the
//! shape:
//!
//! - an observation type, with `Observation` and `HostRow`;
//! - an action type, with `Action` and, for a choice among a fixed number
//!   of actions, `DiscreteAction`;
//! - the environment, with `Environment`: `reset` starts an episode and
//!   `step` advances it.
//!
//! # Endings
//!
//! A step ends an episode in one of two ways. `Snapshot::terminated` means
//! the task ended there, as it does at the goal: there is nothing after it.
//! `Snapshot::truncated` means the episode was cut while the task went on.
//! A learner bootstraps through the second and not the first, so the
//! difference matters. This corridor has no horizon of its own; `main` cuts
//! its episodes with the shipped `TimeLimit`, which truncates.
//!
//! A `step` after the episode has ended is an error and changes nothing.
//! `EpisodeGuard` holds that rule: `check` it first, `record` every status.
//!
//! # Randomness
//!
//! The environment owns its generator, seeded once from the seed the harness
//! gives each trial, so a seed reproduces every episode of a run.

use rlevo::core::episode::EpisodeGuard;
use rlevo::core::rand::SeedableRng;
use rlevo::core::rand::rngs::StdRng;
use rlevo::prelude::{
    Action, DiscreteAction, Environment, EnvironmentError, HostRow, Observation, RngExt,
    ScalarReward, Snapshot,
};

/// Cells in the corridor. The goal is the last.
pub const LENGTH: usize = 10;
/// Reward for reaching the goal.
const GOAL_REWARD: f32 = 1.0;
/// Reward for every other step: a small cost, so a shorter path is better.
const STEP_REWARD: f32 = -0.01;

/// What the agent sees: the cell it stands on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cell {
    pub index: usize,
}

impl Observation<1> for Cell {
    fn shape() -> [usize; 1] {
        [1]
    }
}

/// The observation as the numbers a learner reads: the cell's index scaled
/// to `[0, 1]`. A shipped agent that builds a tensor from an observation,
/// such as `DqnAgent`, needs this.
impl HostRow<1> for Cell {
    fn write_host_row(&self, buf: &mut Vec<f32>) {
        #[allow(clippy::cast_precision_loss)]
        buf.push(self.index as f32 / (LENGTH - 1) as f32);
    }
}

/// One step left or right.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Move {
    Left,
    Right,
}

impl Action<1> for Move {
    fn shape() -> [usize; 1] {
        [Self::ACTION_COUNT]
    }

    fn is_valid(&self) -> bool {
        true
    }
}

impl DiscreteAction<1> for Move {
    const ACTION_COUNT: usize = 2;

    fn from_index(index: usize) -> Self {
        match index {
            0 => Self::Left,
            1 => Self::Right,
            _ => panic!("a move's index is 0 or 1, not {index}"),
        }
    }

    fn to_index(&self) -> usize {
        match self {
            Self::Left => 0,
            Self::Right => 1,
        }
    }
}

/// The corridor.
#[derive(Debug)]
pub struct MyEnv {
    cell: usize,
    rng: StdRng,
    guard: EpisodeGuard,
}

impl MyEnv {
    /// A corridor whose start cells are drawn from `seed`.
    pub fn with_seed(seed: u64) -> Self {
        Self {
            cell: 0,
            rng: StdRng::seed_from_u64(seed),
            guard: EpisodeGuard::new(),
        }
    }

    fn observe(&self) -> Cell {
        Cell { index: self.cell }
    }
}

impl Environment<1, 1> for MyEnv {
    type ObservationType = Cell;
    type ActionType = Move;

    /// Starts an episode on a random cell short of the goal.
    fn reset(&mut self) -> Result<Snapshot<Cell>, EnvironmentError> {
        self.cell = self.rng.random_range(0..LENGTH - 1);
        self.guard.reset();
        Ok(Snapshot::running(self.observe(), ScalarReward::new(0.0)))
    }

    fn step(&mut self, action: Move) -> Result<Snapshot<Cell>, EnvironmentError> {
        // Before anything changes: a step after the end is refused whole.
        self.guard.check()?;
        self.cell = match action {
            Move::Left => self.cell.saturating_sub(1),
            Move::Right => self.cell + 1,
        };
        let snapshot = if self.cell == LENGTH - 1 {
            Snapshot::terminated(self.observe(), ScalarReward::new(GOAL_REWARD))
        } else {
            Snapshot::running(self.observe(), ScalarReward::new(STEP_REWARD))
        };
        self.guard.record(snapshot.status);
        Ok(snapshot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_goal_terminates_the_episode_and_a_later_step_is_refused() {
        let mut env = MyEnv::with_seed(7);
        let mut snapshot = env.reset().expect("reset");
        let mut steps = 0;
        while !snapshot.status.is_done() {
            snapshot = env.step(Move::Right).expect("a step before the end");
            steps += 1;
            assert!(steps < LENGTH, "walking right reaches the goal");
        }
        assert!(snapshot.status.is_terminated());
        assert_eq!(snapshot.observation, Cell { index: LENGTH - 1 });

        assert!(env.step(Move::Left).is_err());
        assert_eq!(env.observe(), Cell { index: LENGTH - 1 });
    }

    #[test]
    fn a_seed_repeats_its_start_cells() {
        let starts = |seed| {
            let mut env = MyEnv::with_seed(seed);
            (0..20)
                .map(|_| env.reset().expect("reset").observation)
                .collect::<Vec<_>>()
        };
        assert_eq!(starts(3), starts(3));
    }
}
