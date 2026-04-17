//! RL environments for the burn-evorl framework.
//!
//! This crate provides a collection of standard benchmark environments
//! for training and evaluating reinforcement learning agents, including
//! classic control problems, game environments, and custom benchmarks.
//!
//! # Supported Environments
//!
//! ## Classic Control
//! - **CartPole**: Balance a pole on a moving cart
//! - **MountainCar**: Drive a car up a mountain with limited power
//!
//! ## Games
//! - **Chess**: Full game tree search in chess
//! - **Connect Four**: Two-player game solving
//!
//! # Getting Started
//!
//! ```rust
//! use evorl_core::environment::{Environment, Snapshot};
//! use evorl_envs::grids::core::GridAction;
//! use evorl_envs::grids::{EmptyConfig, EmptyEnv};
//!
//! let mut env = EmptyEnv::with_config(EmptyConfig::default(), false);
//! env.reset().expect("reset");
//! let snapshot = env.step(GridAction::Forward).expect("step");
//! assert!(!snapshot.is_done());
//! ```
//!
//! # Module Organization
//!
//! - [`classic`]: Classic control problems
//! - [`games`]: Board games and game trees
//! - [`benchmarks`]: Standard benchmarks
//!
//! # Design Principles
//!
//! All environments implement a common interface:
//! - `reset()` → initial state
//! - `step(action)` → (next_state, reward, done)
//! - Deterministic or stochastic based on configuration

pub mod benchmarks {
    pub mod ackley;
    pub mod rastrigin;
    pub mod sphere;
}
pub mod box2d;
pub mod classic;
pub mod games {
    pub mod chess;
    pub mod connect_four;
}
pub mod grids;
pub mod render;
pub mod toy_text;
pub mod wrappers;
