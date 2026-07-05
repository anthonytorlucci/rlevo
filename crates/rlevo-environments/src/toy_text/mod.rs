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
#[derive(Debug, thiserror::Error)]
pub enum MapError {
    /// A row has a different number of tiles than the first row.
    #[error("row {row} has length {got}, expected {expected}")]
    RowLengthMismatch { row: usize, got: usize, expected: usize },
    /// The map does not contain exactly one `'S'` start tile.
    #[error("map must contain exactly one 'S' tile, found {0}")]
    WrongStartCount(usize),
    /// The map contains no `'G'` goal tiles.
    #[error("map must contain at least one 'G' tile, found {0}")]
    NoGoal(usize),
    /// BFS from start cannot reach any goal tile.
    #[error("goal is unreachable from start under BFS")]
    GoalUnreachable,
    /// A tile character is not one of `'S'`, `'F'`, `'H'`, `'G'`.
    #[error("invalid tile {ch:?} at ({row}, {col})")]
    InvalidTile { row: usize, col: usize, ch: char },
    /// Random map generation failed to produce a solvable map within the retry budget.
    #[error("failed to generate solvable random map within retry limit")]
    MaxRetriesExceeded,
    /// The supplied configuration failed [`Validate`](rlevo_core::config::Validate).
    #[error(transparent)]
    InvalidConfig(#[from] rlevo_core::config::ConfigError),
}
