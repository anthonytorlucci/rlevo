//! Hybrid evolutionary + reinforcement-learning strategies for `rlevo`.
//!
//! This crate marks the boundary between pure evolution and anything that
//! runs an environment: strictly evolutionary code lives in
//! `rlevo-evolution`, which never names an
//! [`Environment`](rlevo_core::environment::Environment), while anything that
//! couples a strategy to an `Environment` rollout lives here.
//!
//! It depends on `rlevo-core` and `rlevo-evolution`. Nothing here uses
//! `rlevo-reinforcement-learning` yet, so the crate does not depend on it;
//! the first strategy that trains a gradient learner inside the evolutionary
//! loop brings that dependency with it.
//!
//! # Surface area
//!
//! - [`rollout_fitness`] — [`RolloutFitness`], a
//!   [`BatchFitnessFn`](rlevo_evolution::fitness::BatchFitnessFn) that scores a
//!   population of flat policy parameters by running episodes against an
//!   [`Environment`](rlevo_core::environment::Environment).
//!   Hand it to an [`EvolutionaryLoop`](rlevo_evolution::EvolutionaryLoop)
//!   with any flat-genome strategy to evolve a policy's weights.
//! - [`policy`] — [`StatefulPolicy`], the rollout contract by which a policy
//!   module carries per-episode state across a [`RolloutFitness`] rollout
//!   (`Hidden = ()` for a memoryless one).

pub mod policy;
pub mod rollout_fitness;

pub use policy::StatefulPolicy;
pub use rollout_fitness::RolloutFitness;
