//! BipedalWalker environment using Rapier2D physics.

pub mod action;
pub mod config;
pub mod env;
pub mod observation;
pub mod state;
pub mod terrain;

pub use action::BipedalWalkerAction;
pub use config::{BipedalTerrain, BipedalWalkerConfig, BipedalWalkerConfigBuilder};
pub use env::BipedalWalker;
pub use observation::BipedalWalkerObservation;
pub use state::BipedalWalkerState;
pub use terrain::{FlatTerrain, HardcoreTerrain, RoughTerrain, TerrainGenerator};
