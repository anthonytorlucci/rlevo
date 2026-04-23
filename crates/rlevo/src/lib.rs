//! Evolutionary Deep Reinforcement Learning built on the [Burn](https://github.com/tracel-ai/burn) framework.
//!
//! # Modules
//!
//! - [`core`] — foundational traits: `Environment`, `State`, `Action`, `Reward`, replay buffers
//! - [`envs`] — benchmark environments: classic control, gridworlds, Box2D physics, locomotion
//! - [`rl`] — deep RL algorithms: DQN, C51, QR-DQN, PPO, PPG, DDPG, TD3, SAC
//! - [`evolution`] — evolutionary algorithms: GA, ES, EP, DE, CGP with GPU kernels
//! - [`hybrid`] — combined evolutionary + RL strategies
//! - [`utils`] — shared math utilities
//!
//! # Quick Start
//!
//! ```toml
//! [dependencies]
//! rlevo = "0.1"
//! ```
//!
//! ```rust,no_run
//! use rlevo::core::environment::Environment;
//! use rlevo::envs::classic::cart_pole::CartPoleEnv;
//! ```

pub use rlevo_core as core;
pub use rlevo_envs as envs;
pub use rlevo_evolution as evolution;
pub use rlevo_hybrid as hybrid;
pub use rlevo_rl as rl;
pub use rlevo_utils as utils;
