//! Tensor-native classical evolutionary algorithms for `rlevo`.
//!
//! This crate ships the classical EA families — Genetic Algorithm (GA),
//! Evolution Strategy (ES), Evolutionary Programming (EP), Differential
//! Evolution (DE), and Cartesian Genetic Programming (CGP) — on top of the
//! Burn tensor abstraction, with GPU acceleration via custom `CubeCL` kernels
//! on hot paths.
//!
//! # Surface area
//!
//! - [`strategy`] — the central [`Strategy`] trait and
//!   the [`EvolutionaryHarness`] adapter that
//!   wraps any strategy into `rlevo-core::evaluation::BenchEnv`.
//! - [`genome`] — zero-sized marker types (`Real`, `Binary`, `Integer`,
//!   `Tree`, `Permutation`) that parameterize the operator set.
//! - [`population`] — [`Population<B, K>`](population::Population), a thin
//!   wrapper around `Tensor<B, 2>` carrying shape metadata.
//! - [`fitness`] — [`FitnessFn`](fitness::FitnessFn) /
//!   [`BatchFitnessFn`](fitness::BatchFitnessFn), the
//!   [`FromFitnessEvaluable`](fitness::FromFitnessEvaluable) adapter for
//!   `rlevo-core::fitness::FitnessEvaluable`, and the
//!   [`FromLandscape`](fitness::FromLandscape) adapter for landscapes that
//!   carry their own `evaluate` method.
//! - [`observer`] — [`PopulationObserver`] /
//!   [`PopulationSnapshot`] /
//!   [`SharedPopulationObserver`]:
//!   structured per-generation callback for recorders that need more than
//!   the scalar `tracing` events (full fitness vector, best-individual
//!   index, lineage).
//! - [`rng`] — deterministic seed streams (splitmix64) for reproducibility.
//! - [`shaping`] — fitness shaping transforms (centered rank, z-score).
//! - [`ops`] — selection, crossover, mutation, and replacement operators.
//! - [`local_search`] — host-side, gradient-free refinement
//!   ([`LocalSearch`] and the four reference
//!   searchers) for memetic algorithms.
//! - [`probability_model`] — the [`ProbabilityModel`] trait shared by the
//!   estimation-of-distribution (EDA) strategies.
//! - [`coevolution`] — competitive / cooperative co-evolution
//!   ([`CompetitiveCoEA`], [`CooperativeCoEA`]), the [`CoupledFitness`] trait,
//!   the [`HallOfFameFitness`] cycling mitigation, and the
//!   [`CoEvolutionaryHarness`] `BenchEnv` adapter.
//! - [`algorithms`] — concrete strategies.

pub mod algorithms;
pub mod coevolution;
pub mod fitness;
pub mod genome;
pub mod local_search;
pub mod module_eval_fn;
pub mod neuroevolution;
pub mod observer;
pub mod ops;
pub mod param_reshaper;
pub mod population;
pub mod probability_model;
pub mod rng;
pub mod shaping;
pub mod strategy;

pub use algorithms::eda::{
    BayesianNetwork, BayesianNetworkParams, CompactGenetic, CompactGeneticParams, DependencyChain,
    DependencyChainParams, EdaParams, EdaState, EdaStrategy, UnivariateBernoulli,
    UnivariateBernoulliParams, UnivariateGaussian, UnivariateGaussianParams,
};
pub use algorithms::memetic::{CoveragePolicy, MemeticWrapper, WritebackPolicy};
pub use algorithms::neuroevolution::{
    ArchNasBuilder, ArchNasFitnessFn, ArchNasStrategy, BatchGraphFitness, GraphFitnessFn,
    NasBuilderConfig, NasGenome, NasParams, NasState, NeatParams, NeatState, NeatStrategy,
    VariantEvaluator, WeightOnly,
};
pub use neuroevolution::{
    ActivationFn, BatchPhenotypeEvaluator, ConnectionGene, DensePaddedEvaluator, InnovationId,
    InnovationRegistry, InterpretedBuilder, InterpretedPhenotype, NodeGene, NodeId, NodeKind,
    NodeSplit, Phenotype, PhenotypeBuilder, Species, SpeciesId, TopologyGenome,
    compatibility_distance,
};
pub use coevolution::{
    CoEAMetrics, CoEAState, CoEvolutionaryAlgorithm, CoEvolutionaryHarness, CompetitiveCoEA,
    CompetitiveCoEAParams, CooperativeCoEA, CooperativeCoEAParams, CooperativeState, CoupledFitness,
    HallOfFame, HallOfFameFitness, RepresentativePolicy,
};
pub use module_eval_fn::ModuleEvalFn;
pub use param_reshaper::{ModuleReshaper, ParamReshaper};
pub use probability_model::ProbabilityModel;
pub use local_search::{
    HillClimbing, LocalSearch, NelderMead, RandomRestart, SimulatedAnnealing,
};
pub use observer::{PopulationObserver, PopulationSnapshot, SharedPopulationObserver};
pub use strategy::{EvolutionaryHarness, Strategy, StrategyMetrics};
