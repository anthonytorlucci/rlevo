//! Hybrid evolutionary + reinforcement-learning strategies for `rlevo`.
//!
//! This crate marks the dependency boundary between pure evolution and
//! reinforcement learning: strictly evolutionary code lives in
//! `rlevo-evolution`, which never depends on `rlevo-core`, while anything that
//! couples a strategy to an [`Environment`](rlevo_core::environment::Environment)
//! rollout lives here, where that dependency is permitted.
//!
//! # Surface area
//!
//! - [`rollout_fitness`] — [`RolloutFitness`], a
//!   [`BatchFitnessFn`](rlevo_evolution::fitness::BatchFitnessFn) that scores a
//!   population of flat policy parameters by running episodes against an
//!   [`Environment`](rlevo_core::environment::Environment).
//! - [`policy_neuroevolution`] — [`PolicyNeuroevolution`], which pairs a
//!   [`WeightOnly`](rlevo_evolution::WeightOnly) strategy with a
//!   [`RolloutFitness`] inside an
//!   [`EvolutionaryHarness`](rlevo_evolution::EvolutionaryHarness).

pub mod policy_neuroevolution;
pub mod rollout_fitness;

pub use policy_neuroevolution::PolicyNeuroevolution;
pub use rollout_fitness::RolloutFitness;
