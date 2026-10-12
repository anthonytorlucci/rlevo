//! Glue between the built-in environments and the [`rlevo_benchmarks`]
//! harness's recording tier.
//!
//! Enabled by the `record` cargo feature, the one feature under which this
//! crate depends on the harness. It is off by default, so a build that only
//! wants environments does not link the harness.
//!
//! # Contents
//!
//! - [`family`] — the [`RecordedEnvFamily`] impl that ties each built-in
//!   environment to the family its recordings are drawn as.
//!
//! Running an environment through the harness needs nothing from here:
//! [`Evaluator::run_suite`] binds directly on [`Environment`], so an
//! environment is registered as itself with [`Suite::with_env`] (ADR 0076).
//!
//! [`Environment`]: rlevo_core::environment::Environment
//! [`RecordedEnvFamily`]: rlevo_benchmarks::record::RecordedEnvFamily
//! [`Suite::with_env`]: rlevo_benchmarks::suite::Suite::with_env
//! [`Evaluator::run_suite`]: rlevo_benchmarks::evaluator::Evaluator::run_suite

pub mod family;
