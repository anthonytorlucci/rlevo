//! RL environments for the burn-evorl framework.
//!
//! This crate provides a collection of standard benchmark environments
//! for training and evaluating reinforcement learning agents, including
//! classic control problems, gridworlds, tabular MDPs, physics-based
//! continuous-control tasks, and optimisation benchmark functions.
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
//! - [`classic`]: Classic control problems (CartPole, MountainCar, Pendulum, Acrobot, TenArmedBandit)
//! - [`benchmarks`]: Optimization benchmark functions (Sphere, Ackley, Rastrigin)
//! - [`grids`]: Gridworld environments inspired by Farama Minigrid
//! - [`toy_text`]: Tabular RL environments (Blackjack, Taxi, CliffWalking, FrozenLake)
//! - [`wrappers`]: Environment wrappers (TimeLimit)
//! - [`render`]: ASCII and null renderers
//! - [`box2d`]: Box2D-style physics environments (BipedalWalker, LunarLander, CarRacing)
//! - [`locomotion`]: MuJoCo-inspired locomotion environments via Rapier3D
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
/// Board-game environments — **stub, planned for v0.2**.
///
/// The submodules compile but do not yet implement the [`evorl_core::environment::Environment`]
/// trait. They are hidden from the rendered docs until the `Environment` impls land.
/// Internal dead-code and doc-lint warnings are suppressed here because the
/// contained code is scaffolding for the v0.2 implementation.
#[doc(hidden)]
#[allow(dead_code, clippy::doc_lazy_continuation, clippy::needless_range_loop)]
pub mod games {
    pub mod chess;
    pub mod connect_four;
}
pub mod grids;
pub mod locomotion;
pub mod render;
pub mod toy_text;
pub mod wrappers;
