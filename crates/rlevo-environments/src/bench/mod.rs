//! Glue between [`rlevo_core::environment::Environment`] and the
//! [`rlevo_benchmarks`] harness.
//!
//! Enabled by the `bench` cargo feature. Disabled by default so the base
//! envs dep cone (rapier, nalgebra) is not bundled with the harness's
//! (rayon, tracing, `serde_json`) for users who only want one of the two.
//!
//! # Contents
//!
//! - [`suites`] — preset [`Suite`] factories for the canonical envs in
//!   this crate, ready to feed [`Evaluator::run_suite`].
//!
//! There is no adapter type: [`Evaluator::run_suite`] binds directly on
//! [`Environment`], so envs are registered as themselves (ADR 0076).
//!
//! The [`Landscape`](rlevo_core::fitness::Landscape) impls are **not** here.
//! They name nothing from the harness, so gating them behind `bench` would
//! have made them reachable only by compiling it; they live unconditionally in
//! [`crate::landscapes::fitness`].
//!
//! [`Environment`]: rlevo_core::environment::Environment
//! [`Suite`]: rlevo_benchmarks::suite::Suite
//! [`Evaluator::run_suite`]: rlevo_benchmarks::evaluator::Evaluator::run_suite

/// [`RecordedEnvFamily`](rlevo_benchmarks::record::RecordedEnvFamily) impls
/// for the built-in environments (feature `record`).
#[cfg(feature = "record")]
pub mod family;
pub mod suites;
