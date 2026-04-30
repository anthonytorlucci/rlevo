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
//! - [`strategy`] — the central [`Strategy`](strategy::Strategy) trait and
//!   the [`EvolutionaryHarness`](strategy::EvolutionaryHarness) adapter that
//!   wraps any strategy into `rlevo-benchmarks::BenchEnv`.
//! - [`genome`] — zero-sized marker types (`Real`, `Binary`, `Integer`,
//!   `Tree`) that parameterize the operator set.
//! - [`population`] — [`Population<B, K>`](population::Population), a thin
//!   wrapper around `Tensor<B, 2>` carrying shape metadata.
//! - [`fitness`] — [`FitnessFn`](fitness::FitnessFn) /
//!   [`BatchFitnessFn`](fitness::BatchFitnessFn) and the
//!   [`FromFitnessEvaluable`](fitness::FromFitnessEvaluable) adapter for
//!   `rlevo-benchmarks::FitnessEvaluable`.
//! - [`rng`] — deterministic seed streams (splitmix64) for reproducibility.
//! - [`shaping`] — fitness shaping transforms (centered rank, z-score).
//! - [`ops`] — selection, crossover, mutation, and replacement operators.
//! - [`algorithms`] — concrete strategies.

pub mod algorithms;
pub mod fitness;
pub mod genome;
pub mod ops;
pub mod population;
pub mod rng;
pub mod shaping;
pub mod strategy;

pub use strategy::{EvolutionaryHarness, Strategy, StrategyMetrics};
