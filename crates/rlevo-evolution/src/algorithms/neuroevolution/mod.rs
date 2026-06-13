//! Neuroevolution strategies — evolving the parameters of a Burn `Module`.
//!
//! Phase 3d1 ships **weight-only** neuroevolution: a fixed-topology network is
//! declared once and a phase-1 [`Strategy`](crate::strategy::Strategy)
//! (GA / ES / DE / …) evolves its flattened weights. The
//! [`WeightOnly`](weight_only::WeightOnly) wrapper composes any such strategy
//! with any [`Module`](burn::module::Module); reshaping between the flat
//! genome and the network happens at the fitness boundary
//! ([`ModuleEvalFn`](crate::module_eval_fn::ModuleEvalFn)), never inside the
//! strategy.
//!
//! Phase 3d2 adds **bounded architecture NAS**
//! ([`arch_nas`]): a custom [`ArchNasStrategy`](arch_nas::ArchNasStrategy)
//! harness evolves *which* fixed-topology `Module` variant — from an enumerated
//! menu — wins a task, alongside each variant's weights, reusing the same
//! per-variant `ModuleReshaper` infrastructure. Full topology evolution (NEAT)
//! and a batched/vectorized forward pass remain deferred to issue #41.

pub mod arch_nas;
pub mod weight_only;

pub use arch_nas::{
    ArchNasBuilder, ArchNasFitnessFn, ArchNasStrategy, NasBuilderConfig, NasGenome, NasParams,
    NasState, VariantEvaluator,
};
pub use weight_only::WeightOnly;
