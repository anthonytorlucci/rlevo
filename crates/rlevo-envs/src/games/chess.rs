//! Chess environment using the AlphaZero action/observation encoding.
//!
//! - [`board`]: Board state representation and legal-move generation
//! - [`moves`]: AlphaZero action-plane encoding (4672-action space)
//! - [`observations`]: 119-plane observation tensor layout (8×8×119)
//! - [`environment`]: `Environment` trait implementation (work in progress)

pub mod board;
pub mod environment;
pub mod moves;
pub mod observations;
