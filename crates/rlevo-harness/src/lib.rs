//! Reproducible evaluation harness for `rlevo`.
//!
//! The umbrella re-exports this crate as `rlevo::harness`. Through 0.5.0 it
//! was named `rlevo-benchmarks` (ADR 0088).
//!
//! The shared trait surface (`Environment`, `Agent`, `GenerationProbe`,
//! `Landscape`, `SeedStream`) lives in
//! `rlevo-core`. This crate provides everything needed to drive those traits
//! and produce structured reports:
//!
//! - [`evaluator`] — trial-level parallel runner (rayon) with checkpoint resume.
//! - [`suite`] — grouping and factory type for environments under test.
//! - [`metrics`] — core statistics (return, length, throughput), EA metrics, and
//!   RL metric name constants.
//! - [`metrics_registry`] — shared registry for metric name constants used
//!   across evaluator and reporter.
//! - [`reporter`] — event-sink trait plus `logging` and `json` sinks.
//! - [`report`] — in-memory report types written by the evaluator.
//! - [`checkpoint`] — atomic partial-report persistence for crash recovery.
//! - `record` *(feature `record`)* — per-episode binary recording surface.

pub mod checkpoint;
pub mod evaluator;
pub mod metrics;
pub mod metrics_registry;
#[cfg(feature = "record")]
pub mod record;
pub mod report;
pub mod reporter;
#[doc(hidden)]
pub mod storage;
pub mod suite;

/// Flat crate-root re-export of the drive seam for self-paced work.
///
/// Lets callers write `rlevo_harness::GenerationProbe` without navigating
/// to `rlevo_core::evaluation`. Episodic work needs no such re-export: it is
/// driven through `rlevo_core::environment::Environment` directly.
pub use rlevo_core::evaluation::GenerationProbe;
