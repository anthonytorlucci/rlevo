//! Metric primitives and aggregators.
//!
//! [`Metric`] and [`MetricsProvider`] live in `rlevo_core::fitness`
//! (shared trait surface), so an [`Agent`](rlevo_core::agent::Agent) reports
//! through them without depending on this crate. They are re-exported here
//! for ergonomic access alongside the harness-specific aggregators in
//! [`core`] and [`ea`].

pub mod core;
pub mod ea;

pub use rlevo_core::fitness::{Metric, MetricsProvider};
