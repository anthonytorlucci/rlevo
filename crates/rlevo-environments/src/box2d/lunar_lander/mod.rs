//! LunarLander — 2D spacecraft landing environment (discrete and continuous variants).
//!
//! Implements the LunarLander-v3 Gymnasium environment using Rapier2D physics.
//! The agent fires engines to safely land on a helipad at the origin; reward is
//! potential-based, shaped by distance, velocity, and tilt relative to target.
//!
//! ## Variants
//!
//! | Type | Action space |
//! |---|---|
//! | [`LunarLanderDiscrete`] | 4-way discrete: no-op, left, main, right engine |
//! | [`LunarLanderContinuous`] | 2-dim continuous: throttle `∈ [-1,1]`, lateral `∈ [-1,1]` |
//!
//! ## Wind model
//!
//! Configurable via [`WindMode`] (design decision D2):
//! - `Off` — no wind (default)
//! - `Constant { force }` — steady lateral force applied every step
//! - `Stochastic { seed, max_force }` — random force drawn each step
//!
//! ## Termination conditions
//!
//! - Crash (lander touches ground without both legs) → `Terminated` (−100).
//! - Out of bounds → `Terminated` (−100).
//! - Soft landing (both legs down, low velocity and angle) → `Terminated` (+100).
//! - Step count reaches `config.max_steps` (default 1000) → `Truncated`.
//!
//! ## Quick start
//!
//! ```no_run
//! # use rlevo_envs::box2d::lunar_lander::{
//! #     LunarLanderDiscrete, LunarLanderDiscreteAction, LunarLanderConfig,
//! # };
//! # use rlevo_core::environment::{Environment, Snapshot};
//! let mut env = LunarLanderDiscrete::with_config(LunarLanderConfig::default());
//! let _snap = env.reset().unwrap();
//! let snap = env.step(LunarLanderDiscreteAction::MainEngine).unwrap();
//! println!("reward: {:?}", snap.reward());
//! ```

pub mod action_continuous;
pub mod action_discrete;
pub mod config;
pub mod env;
pub mod observation;
pub mod snapshot;
pub mod state;

pub use action_continuous::LunarLanderContinuousAction;
pub use action_discrete::LunarLanderDiscreteAction;
pub use config::{LunarLanderConfig, LunarLanderConfigBuilder, WindMode};
pub use env::{LunarLanderContinuous, LunarLanderDiscrete};
pub use observation::LunarLanderObservation;
pub use snapshot::{LunarLanderSnapshot, METADATA_KEY_SHAPING};
pub use state::LunarLanderState;
