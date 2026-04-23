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
//! burn-evorl = "0.1"
//! ```
//!
//! ```rust,no_run
//! use burn_evorl::core::environment::Environment;
//! use burn_evorl::envs::classic::cart_pole::CartPoleEnv;
//! ```

pub use evorl_core as core;
pub use evorl_envs as envs;
pub use evorl_evolution as evolution;
pub use evorl_hybrid as hybrid;
pub use evorl_rl as rl;
pub use evorl_utils as utils;
