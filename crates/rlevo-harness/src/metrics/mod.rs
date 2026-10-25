//! Metric primitives and aggregators.
//!
//! [`Metric`] lives in `rlevo_core::fitness`, so an
//! [`Agent`](rlevo_core::agent::Agent) or a
//! [`GenerationProbe`](rlevo_core::evaluation::GenerationProbe) reports
//! through its `emit_metrics` without depending on this crate. It is
//! re-exported here for ergonomic access alongside the harness-specific
//! aggregators in [`core`] and [`ea`].

pub mod core;
pub mod ea;

pub use rlevo_core::fitness::Metric;
