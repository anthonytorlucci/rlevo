//! Reproducible benchmarking harness for `rlevo`.
//!
//! The shared trait surface (`BenchEnv`, `BenchError`, `BenchStep`,
//! `BenchableAgent`, `FitnessEvaluable`, `Landscape`, `SeedStream`) lives in
//! `rlevo-core`. This crate provides everything needed to drive those traits
//! and produce structured reports:
//!
//! - [`evaluator`] — trial-level parallel runner (rayon) with checkpoint resume.
//! - [`suite`] — grouping and factory type for environments under test.
//! - [`metrics`] — core statistics (return, length, throughput), EA metrics, and
//!   RL metric name constants.
//! - [`reporter`] — event-sink trait plus `logging`, `json`, and `tui` sinks.
//! - [`report`] — in-memory report types written by the evaluator.
//! - [`checkpoint`] — atomic partial-report persistence for crash recovery.
//! - `record` *(feature `record`)* — per-episode binary recording surface.
//! - `tui` *(feature `tui`)* — live `ratatui` terminal dashboard.
//! - `env_wrappers` *(feature `tui`)* — composable env wrappers that emit
//!   per-episode returns to the metrics-only live TUI (ADR-0013).

pub mod checkpoint;
#[cfg(feature = "tui")]
pub mod env_wrappers;
pub mod metrics_registry;
pub mod evaluator;
pub mod metrics;
#[cfg(feature = "record")]
pub mod record;
pub mod report;
pub mod reporter;
#[doc(hidden)]
pub mod storage;
pub mod suite;
#[cfg(feature = "tui")]
pub mod tui;

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
