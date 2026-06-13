//! Hybrid evolutionary + reinforcement-learning strategies for `rlevo`.
//!
//! Per the umbrella boundary policy (advanced-evo-algos spec §5.1), strictly
//! evolutionary code lives in `rlevo-evolution`; anything that couples to an
//! `Environment` rollout lives here.
//!
//! # Surface area (phase 3d1)
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
