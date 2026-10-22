//! Tensor-native evolutionary algorithms for `rlevo`.
//!
//! This crate ships the classical EA families — Genetic Algorithm (GA),
//! Evolution Strategy (ES, with CMA-ES and CMSA-ES), Evolutionary
//! Programming (EP), Differential Evolution (DE), and Cartesian and Gene
//! Expression Programming (CGP, GEP) — alongside swarm metaheuristics,
//! estimation-of-distribution algorithms (EDAs), memetic algorithms,
//! neuroevolution and co-evolution, on top of the Burn tensor abstraction.
//! Every operator is composed from Burn tensor primitives, so a strategy
//! runs on whichever backend the caller selects.
//!
//! # Surface area
//!
//! - [`strategy`] — the central [`Strategy`] trait, and
//!   [`EvolutionaryLoop`], which holds a run's state and drives a strategy
//!   with `reset`, `step` and `is_done`.
//! - [`probe`] — how the evaluation harness drives a loop, through
//!   `rlevo-core::evaluation::GenerationProbe`, and how that is built from
//!   the three calls above.
//! - [`function_set`] — the shared [`FunctionSet`] opcode contract (and the
//!   default [`ArithmeticFunctionSet`]) used by both Cartesian GP and Gene
//!   Expression Programming.
//! - [`fitness`] — [`BatchFitnessFn`](fitness::BatchFitnessFn), and the
//!   [`FromLandscape`](fitness::FromLandscape) adapter for landscapes that
//!   carry their own `evaluate` method.
//! - [`observer`] — [`PopulationObserver`] /
//!   [`PopulationSnapshot`] /
//!   [`SharedPopulationObserver`]:
//!   structured per-generation callback for recorders that need more than
//!   the scalar `tracing` events (full fitness vector, best-individual
//!   index, lineage).
//! - [`rng`] — deterministic seed streams (splitmix64) for reproducibility.
//! - [`ops`] — selection, crossover, mutation, and replacement operators.
//! - [`local_search`] — host-side, gradient-free refinement
//!   ([`LocalSearch`] and the four reference
//!   searchers) for memetic algorithms.
//! - [`probability_model`] — the [`ProbabilityModel`] trait shared by the
//!   estimation-of-distribution (EDA) strategies.
//! - [`coevolution`] — competitive / cooperative co-evolution
//!   ([`CompetitiveCoEA`], [`CooperativeCoEA`]), the [`CoupledFitness`] trait,
//!   the [`HallOfFameFitness`] cycling mitigation, and
//!   [`CoEvolutionaryLoop`], which drives one as [`EvolutionaryLoop`] drives
//!   a strategy.
//! - [`algorithms`] — concrete strategies.

pub mod algorithms;
pub mod coevolution;
pub mod fitness;
pub mod function_set;
pub mod local_search;
pub mod module_eval_fn;
pub mod neuroevolution;
pub mod observer;
pub mod ops;
pub mod param_reshaper;
pub mod probability_model;
pub mod probe;
pub mod rng;
pub(crate) mod sampling;
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
pub use coevolution::{
    CoEAMetrics, CoEAState, CoEvolutionaryAlgorithm, CoEvolutionaryLoop, CompetitiveCoEA,
    CompetitiveCoEAParams, CooperativeCoEA, CooperativeCoEAParams, CooperativeState,
    CoupledFitness, HallOfFame, HallOfFameFitness, RepresentativePolicy,
};
pub use function_set::{ArithmeticFunctionSet, FunctionSet, Symbol};
pub use local_search::{HillClimbing, LocalSearch, NelderMead, RandomRestart, SimulatedAnnealing};
pub use module_eval_fn::ModuleEvalFn;
pub use neuroevolution::{
    ActivationFn, BatchPhenotypeEvaluator, ConnectionGene, DensePaddedEvaluator, InnovationId,
    InnovationRegistry, InterpretedBuilder, InterpretedPhenotype, NodeGene, NodeId, NodeKind,
    NodeSplit, Phenotype, PhenotypeBuilder, Species, SpeciesId, TopologyGenome,
    compatibility_distance,
};
pub use observer::{PopulationObserver, PopulationSnapshot, SharedPopulationObserver};
pub use param_reshaper::{ModuleReshaper, ParamReshaper};
pub use probability_model::ProbabilityModel;
pub use rlevo_core::evaluation::GenerationProbe;
pub use strategy::{EvolutionaryLoop, Strategy, StrategyMetrics};
