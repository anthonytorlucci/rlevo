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
//! - [`algorithms`] — concrete strategies.

pub mod algorithms;
pub mod fitness;
pub mod genome;
pub mod observer;
pub mod ops;
pub mod population;
pub mod rng;
pub mod shaping;
pub mod strategy;

pub use observer::{PopulationObserver, PopulationSnapshot, SharedPopulationObserver};
pub use strategy::{EvolutionaryHarness, Strategy, StrategyMetrics};
