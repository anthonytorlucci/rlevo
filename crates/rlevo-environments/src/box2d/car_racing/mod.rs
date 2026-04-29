//! CarRacing — top-down 2D car racing environment with pixel observations.
//!
//! Implements the CarRacing-v3 Gymnasium environment using Rapier2D physics and
//! a software rasterizer. The agent receives a 96×96 RGB pixel observation and
//! controls steering, gas, and brake inputs to complete a closed-loop track.
//!
//! ## Spaces
//!
//! | | Type | Shape | Range |
//! |---|---|---|---|
//! | Action | Continuous | 3 | `steer ∈ [-1,1]`, `gas ∈ [0,1]`, `brake ∈ [0,1]` |
//! | Observation | Pixel | 96 × 96 × 3 | `u8` per channel |
//!
//! ## Reward structure
//!
//! - `+tile_reward` for each new track tile visited (total ≈ 1000 for a full lap).
//! - `frame_penalty` (default −0.1) applied every step.
//!
//! ## Termination conditions
//!
//! - Car visits ≥ 95% of track tiles → `Terminated` (lap complete).
//! - Step count reaches `config.max_steps` (default 1000) → `Truncated`.
//!
//! ## Track generation
//!
//! Each `reset()` call builds a unique closed-loop track via Catmull-Rom
//! interpolation over randomly perturbed control points. Pass a fixed `seed`
//! in [`CarRacingConfig`] for reproducible episodes.
//!
//! ## Quick start
//!
//! ```no_run
//! # use rlevo_envs::box2d::car_racing::{CarRacing, CarRacingAction, CarRacingConfig};
//! # use rlevo_core::environment::{Environment, Snapshot};
//! let mut env = CarRacing::with_config(CarRacingConfig::default());
//! let _snap = env.reset().unwrap();
//! let snap = env.step(CarRacingAction { steer: 0.0, gas: 0.5, brake: 0.0 }).unwrap();
//! println!("reward: {:?}", snap.reward());
//! ```

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
