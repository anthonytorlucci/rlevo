//! Toy-text tabular RL environments.
//!
//! Faithful Rust implementations of the canonical Sutton & Barto tabular testbeds:
//! [`blackjack`], [`taxi`], [`cliff_walking`], and [`frozen_lake`].
//! All environments are discrete MDPs with no physics.

pub mod blackjack;
pub mod cliff_walking;
pub mod frozen_lake;
pub mod taxi;

/// Error returned when a custom [`frozen_lake::FrozenMapSpec::Custom`] map fails validation,
/// or when random map generation exhausts its retry budget.
#[derive(Debug)]
pub enum MapError {
    RowLengthMismatch { row: usize, got: usize, expected: usize },
    WrongStartCount(usize),
    NoGoal(usize),
    GoalUnreachable,
    InvalidTile { row: usize, col: usize, ch: char },
    MaxRetriesExceeded,
}

impl std::fmt::Display for MapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MapError::RowLengthMismatch { row, got, expected } => {
                write!(f, "row {row} has length {got}, expected {expected}")
            }
            MapError::WrongStartCount(n) => {
                write!(f, "map must contain exactly one 'S' tile, found {n}")
            }
            MapError::NoGoal(n) => {
                write!(f, "map must contain at least one 'G' tile, found {n}")
            }
            MapError::GoalUnreachable => write!(f, "goal is unreachable from start under BFS"),
            MapError::InvalidTile { row, col, ch } => {
                write!(f, "invalid tile {ch:?} at ({row}, {col})")
            }
            MapError::MaxRetriesExceeded => {
                write!(f, "failed to generate solvable random map within retry limit")
            }
        }
    }
}

impl std::error::Error for MapError {}
