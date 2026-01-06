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
//! use evorl_envs::classic::CartPole;
//! use evorl_core::environment::Environment;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create environment
//!     let mut env = CartPole::new()?;
//!
//!     // Reset to initial state
//!     let state = env.reset()?;
//!
//!     // Take an action
//!     let (next_state, reward, done) = env.step(0)?;
//!
//!     Ok(())
//! }
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

// todo! pub mod benchmarks;
pub mod classic {
    pub mod cartpole;
    // todo! pub mod mountain_car;
}
pub mod games {
    pub mod chess;
    pub mod connect_four;
}
