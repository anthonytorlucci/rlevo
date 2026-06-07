//! RL environments for the rlevo framework.
//!
//! This crate provides a collection of standard benchmark environments
//! for training and evaluating reinforcement learning agents, including
//! classic control problems, gridworlds, tabular MDPs, physics-based
//! continuous-control tasks, and optimisation benchmark functions.
//!
//! # Module Organization
//!
//! - [`classic`]: Classic control problems (CartPole, MountainCar, Pendulum, Acrobot, TenArmedBandit)
//! - [`landscapes`]: Optimization fitness landscapes (Sphere, Ackley, Rastrigin)
//! - [`grids`]: Gridworld environments inspired by Farama Minigrid
//! - [`toy_text`]: Tabular RL environments (Blackjack, Taxi, CliffWalking, FrozenLake)
//! - [`wrappers`]: Environment wrappers (TimeLimit)
//! - [`render`]: Re-exports of the render surface defined in `rlevo-core`
//! - [`box2d`]: Box2D-style physics environments (BipedalWalker, LunarLander, CarRacing)
//! - [`locomotion`]: MuJoCo-inspired locomotion environments via Rapier3D
//!
//! # Design Principles
//!
//! All environments implement the [`rlevo_core::environment::Environment`] trait.
//! Every environment exposes two fallible operations:
//!
//! - [`reset`](rlevo_core::environment::Environment::reset) — returns an initial
//!   [`SnapshotBase`](rlevo_core::environment::SnapshotBase) with
//!   [`EpisodeStatus::Running`](rlevo_core::environment::EpisodeStatus::Running).
//! - [`step`](rlevo_core::environment::Environment::step) — applies an action and
//!   returns the next snapshot; status transitions to `Terminated` or `Truncated`
//!   when the episode ends.
//!
//! Environments may be deterministic or stochastic depending on their configuration.

pub mod landscapes {
    pub mod ackley;
    pub mod rastrigin;
    pub mod render;
    pub mod sphere;
}
#[cfg(feature = "bench")]
pub mod bench;
pub mod box2d;
pub mod classic;
/// Board-game environments — **stub, planned for v0.2**.
///
/// The submodules compile but do not yet implement the [`rlevo_core::environment::Environment`]
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
