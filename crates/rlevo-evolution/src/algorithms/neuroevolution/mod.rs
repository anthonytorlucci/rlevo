//! Neuroevolution strategies — evolving the parameters of a Burn `Module`.
//!
//! `rlevo` ships three neuroevolution approaches, distinguished by how much
//! network structure is fixed before the search begins.
//!
//! **Weight-only** ([`weight_only`]) fixes the architecture: a network is
//! declared once and a flat-genome [`Strategy`](crate::strategy::Strategy)
//! (GA / ES / DE / …) evolves its flattened weights. The
//! [`WeightOnly`](weight_only::WeightOnly) wrapper composes any such strategy
//! with any [`Module`](burn::module::Module); reshaping between the flat genome
//! and the network happens at the fitness boundary
//! ([`ModuleEvalFn`](crate::module_eval_fn::ModuleEvalFn)), never inside the
//! strategy.
//!
//! **Bounded architecture NAS** ([`arch_nas`]) co-evolves the choice of
//! architecture: a custom [`ArchNasStrategy`](arch_nas::ArchNasStrategy) harness
//! searches *which* fixed-topology `Module` variant — from an enumerated menu —
//! wins a task, alongside each variant's weights, reusing the same per-variant
//! `ModuleReshaper` infrastructure.
//!
//! **NEAT** ([`neat`]) evolves the topology itself: the
//! [`NeatStrategy`](neat::NeatStrategy) custom harness grows network structure
//! and weights open-endedly from a minimal seed via innovation-aligned crossover
//! and speciation, scoring graph genomes through the
//! [`GraphFitnessFn`](neat::GraphFitnessFn) seam. The graph data model lives in
//! [`crate::neuroevolution`]. Genomes are scored either one at a time — the
//! interpreted, host-side reference path — or a whole population at once through
//! the device-batched
//! [`BatchPhenotypeEvaluator`](crate::neuroevolution::phenotype::BatchPhenotypeEvaluator).

pub mod arch_nas;
pub mod neat;
pub mod weight_only;

pub use arch_nas::{
    ArchNasBuilder, ArchNasFitnessFn, ArchNasStrategy, NasBuilderConfig, NasGenome, NasParams,
    NasState, VariantEvaluator,
};
pub use neat::{BatchGraphFitness, GraphFitnessFn, NeatParams, NeatState, NeatStrategy};
pub use weight_only::WeightOnly;
