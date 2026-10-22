//! Neuroevolution strategies — evolving the parameters of a Burn `Module`.
//!
//! `rlevo` ships three neuroevolution approaches, distinguished by how much
//! network structure is fixed before the search begins.
//!
//! **Weight-only** fixes the architecture and needs no strategy of its own: a
//! network is declared once, and any flat-genome
//! [`Strategy`](crate::strategy::Strategy) (GA / ES / DE / …) evolves its
//! flattened weights unchanged, with the genome width read from a
//! [`ModuleReshaper`](crate::param_reshaper::ModuleReshaper). Reshaping between
//! the flat genome and the [`Module`](burn::module::Module) happens at the
//! fitness boundary ([`ModuleEvalFn`](crate::module_eval_fn::ModuleEvalFn)),
//! never inside the strategy.
//!
//! **Bounded architecture NAS** ([`arch_nas`]) co-evolves the choice of
//! architecture: a standalone [`ArchNasStrategy`]
//! searches *which* fixed-topology `Module` variant — from an enumerated menu —
//! wins a task, alongside each variant's weights, reusing the same per-variant
//! `ModuleReshaper` infrastructure.
//!
//! **NEAT** ([`neat`]) evolves the topology itself: the
//! standalone [`NeatStrategy`] grows network structure
//! and weights open-endedly from a minimal seed via innovation-aligned crossover
//! and speciation, scoring graph genomes through the
//! [`GraphFitnessFn`] seam. The graph data model lives in
//! [`crate::neuroevolution`]. Genomes are scored either one at a time — the
//! interpreted, host-side reference path — or a whole population at once through
//! the device-batched
//! [`DensePaddedEvaluator`](crate::neuroevolution::phenotype::DensePaddedEvaluator).

pub mod arch_nas;
pub mod neat;

pub use arch_nas::{
    ArchNasBuilder, ArchNasFitnessFn, ArchNasStrategy, NasBuilderConfig, NasGenome, NasParams,
    NasState, VariantEvaluator,
};
pub use neat::{BatchGraphFitness, GraphFitnessFn, NeatParams, NeatState, NeatStrategy};
