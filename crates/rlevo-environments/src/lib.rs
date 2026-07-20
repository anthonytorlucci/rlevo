//! RL environments for the rlevo framework.
//!
//! This crate provides a collection of standard benchmark environments
//! for training and evaluating reinforcement learning agents, including
//! classic control problems, gridworlds, tabular MDPs, physics-based
//! continuous-control tasks, and optimisation benchmark functions.
//!
//! # Module Organization
//!
//! - [`classic`]: Classic control problems (`CartPole`, `MountainCar`, Pendulum, Acrobot, `TenArmedBandit`)
//! - [`episode`]: [`EpisodeGuard`](episode::EpisodeGuard) — the shared post-terminal
//!   `step()` guard consumed by `toy_text` and [`wrappers::time_limit`]
//! - [`landscapes`]: Optimization fitness landscapes (Sphere, Ackley, Rastrigin,
//!   plus the scalable n-D, classical 2-D, and stress-test benchmark suites)
//! - [`grids`]: Gridworld environments inspired by Farama Minigrid
//! - [`pixel_grid`]: Synthetic pixel-over-grid env — a rank-1 latent observed as
//!   a rank-3 RGB image (first real [`rlevo_core::state::Observable`] consumer)
//! - [`toy_text`]: Tabular RL environments (Blackjack, Taxi, `CliffWalking`, `FrozenLake`)
//! - [`wrappers`]: Environment wrappers (`TimeLimit`)
//! - [`render`]: Re-exports of the render surface defined in `rlevo-core`
//! - [`box2d`]: Box2D-style physics environments (`BipedalWalker`, `LunarLander`, `CarRacing`)
//! - [`locomotion`]: MuJoCo-inspired locomotion environments via `Rapier3D`
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

// The numeric-cast family is suppressed crate-wide pending the per-site audit
// tracked in #396. Enabling `[workspace.lints]` here surfaced 194 cast warnings
// in production physics, rasterizer, and grid-indexing paths. Each needs a real
// range argument at the site (is this `f32` provably non-negative before `as
// usize`?) rather than a mechanical rewrite, so they are burned down module by
// module — narrow this allow as modules are cleared, do not widen it.
//
// Every *other* lint in the workspace table is enforced on this crate today.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

pub mod landscapes {
    pub mod ackley;
    pub mod rastrigin;
    pub mod render;
    pub mod sphere;

    // Tier 1 — scalable n-D landscapes.
    pub mod concatenated_trap;
    pub mod griewank;
    pub mod michalewicz;
    pub mod penalized1;
    pub mod rosenbrock;
    pub mod schwefel;

    // Tier 2 — classical 2-D landscapes.
    pub mod branin;
    pub mod bukin6;
    pub mod cross_in_tray;
    pub mod easom;
    pub mod goldstein_price;
    pub mod himmelblau;
    pub mod six_hump_camel;

    // Tier 3 — stress-test landscapes.
    pub mod alpine1;
    pub mod deb1;
    pub mod eggholder;
    pub mod lunacek_bi_rastrigin;
    pub mod needle_eye;
    pub mod rosenbrock_flat;
    pub mod trefethen;
}
#[cfg(feature = "bench")]
pub mod bench;
pub mod box2d;
pub mod classic;
/// Cardinal direction primitive (4-way heading + rotation) shared across
/// environments. Originally lived under [`grids::core`]; lifted to the crate
/// root so non-grid environments (e.g. the Santa Fe ant) can reuse it without
/// depending on the Minigrid framework. `grids::core` re-exports it for
/// backward compatibility.
pub mod direction;
pub mod episode;
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
pub mod pixel_grid;
pub mod render;
/// Shared helpers for decoding tensors back into discrete environment types
/// (e.g. `NaN`-safe argmax for action `from_tensor`).
pub(crate) mod tensor_decode;
pub mod toy_text;
pub mod wrappers;
