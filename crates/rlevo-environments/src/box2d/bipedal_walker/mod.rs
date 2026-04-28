//! BipedalWalker — 2D bipedal locomotion environment.
//!
//! Implements the BipedalWalker-v3 Gymnasium environment using Rapier2D physics.
//! The agent controls hip and knee motor velocity targets for a two-legged robot
//! walking forward over procedurally generated terrain.
//!
//! ## Spaces
//!
//! | | Type | Dim | Range |
//! |---|---|---|---|
//! | Action | Continuous | 4 | `[-1, 1]` per component |
//! | Observation | Continuous | 24 | hull kinematics + joint angles/speeds + lidar |
//!
//! ## Terrain variants
//!
//! | [`BipedalTerrain`] | Description |
//! |---|---|
//! | `Flat` | Flat horizontal surface — good for initial training |
//! | `Rough` | Randomly varying height — intermediate difficulty |
//! | `Hardcore` | Stumps, pits, and stair-like obstacles — hardest |
//!
//! ## Termination conditions
//!
//! - Hull contacts the ground → `Terminated` (−100 penalty applied).
//! - Cumulative reward < −100 → `Terminated`.
//! - Step count reaches `config.max_steps` (default 1600) → `Truncated`.
//!
//! ## Quick start
//!
//! ```no_run
//! # use rlevo_envs::box2d::bipedal_walker::{
//! #     BipedalWalker, BipedalWalkerAction, BipedalWalkerConfig, BipedalTerrain,
//! # };
//! # use rlevo_core::environment::{Environment, Snapshot};
//! let cfg = BipedalWalkerConfig::builder()
//!     .terrain(BipedalTerrain::Rough)
//!     .seed(42)
//!     .build();
//! let mut env = BipedalWalker::with_config(cfg);
//! let _snap = env.reset().unwrap();
//! let snap = env.step(BipedalWalkerAction([0.1, -0.1, 0.1, -0.1])).unwrap();
//! println!("reward: {:?}", snap.reward());
//! ```

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
