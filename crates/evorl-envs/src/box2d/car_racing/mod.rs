//! CarRacing environment using Rapier2D physics with pixel observations.

pub mod action;
pub mod config;
pub mod env;
pub mod observation;
pub mod rasterizer;
pub mod state;
pub mod track;

pub use action::CarRacingAction;
pub use config::{CarRacingConfig, CarRacingConfigBuilder};
pub use env::CarRacing;
pub use observation::CarRacingObservation;
pub use rasterizer::Rasterizer;
pub use state::CarRacingState;
pub use track::Track;
