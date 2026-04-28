//! Glue between [`rlevo_core::environment::Environment`] and the
//! [`rlevo_benchmarks`] harness.
//!
//! Enabled by the `bench` cargo feature. Disabled by default so the base
//! envs dep cone (rapier, nalgebra) is not bundled with the harness's
//! (rayon, tracing, serde_json) for users who only want one of the two.
//!
//! # Contents
//!
//! - [`adapter::BenchAdapter`] — wraps any [`Environment`] whose reward is
//!   [`ScalarReward`] into a [`BenchEnv`].
//! - [`suites`] — preset [`Suite`] factories for the canonical envs in
//!   this crate, ready to feed [`Evaluator::run_suite`].
//! - [`landscape`] — [`Landscape`](rlevo_core::fitness::Landscape) impls
//!   for the numerical landscapes in [`crate::landscapes`], for use with
//!   `rlevo-evolution`'s `FromLandscape` adapter.
//!
//! [`Environment`]: rlevo_core::environment::Environment
//! [`ScalarReward`]: rlevo_core::reward::ScalarReward
//! [`BenchEnv`]: rlevo_core::evaluation::BenchEnv
//! [`Suite`]: rlevo_benchmarks::suite::Suite
//! [`Evaluator::run_suite`]: rlevo_benchmarks::evaluator::Evaluator::run_suite

pub mod adapter;
pub mod landscape;
pub mod suites;

pub use adapter::BenchAdapter;
