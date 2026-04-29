//! Reproducible benchmarking harness for `rlevo`.
//!
//! The trait surface (`BenchEnv`, `BenchError`, `BenchStep`,
//! `BenchableAgent`, `FitnessEvaluable`, `Landscape`, `SeedStream`) lives
//! in `rlevo-core` (per ADR 0004). This crate provides the runner that
//! drives those traits to produce reports.

pub mod checkpoint;
pub mod evaluator;
pub mod metrics;
pub mod report;
pub mod reporter;
#[doc(hidden)]
pub mod storage;
pub mod suite;

/// Backward-compatible alias for the relocated environment trait surface.
pub mod env {
    pub use rlevo_core::evaluation::{BenchEnv, BenchError, BenchStep};
}

/// Backward-compatible alias for the relocated agent / fitness traits.
pub mod agent {
    pub use rlevo_core::fitness::{BenchableAgent, FitnessEvaluable, Landscape};
}

/// Backward-compatible alias for the relocated seed-derivation utility.
pub mod seed {
    pub use rlevo_core::util::seed::SeedStream;
}

pub use rlevo_core::evaluation::{BenchEnv, BenchError, BenchStep};
