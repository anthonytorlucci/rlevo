//! Metric primitives and aggregators.
//!
//! [`Metric`] and [`MetricsProvider`] live in `rlevo_core::fitness`
//! (shared trait surface) so the [`BenchableAgent`](rlevo_core::fitness::BenchableAgent)
//! trait stays self-contained. They are re-exported here for ergonomic
//! access alongside the harness-specific aggregators in [`core`], [`ea`],
//! and [`rl`].

pub mod core;
pub mod ea;
pub mod rl;

pub use rlevo_core::fitness::{Metric, MetricsProvider};
