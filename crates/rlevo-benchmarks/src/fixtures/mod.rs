//! Preset harness wiring for the built-in [`rlevo_environments`] types.
//!
//! Enabled by the `fixtures` cargo feature, off by default: this is the one
//! module in the crate that names concrete environments, and a consumer
//! bringing their own environment should not compile `rapier` to run the
//! harness.
//!
//! # Contents
//!
//! - [`suites`] — preset [`Suite`] factories for the canonical envs, ready to
//!   feed [`Evaluator::run_suite`].
//! - `family` *(feature `record`)* — `RecordedEnvFamily` impls tying each
//!   built-in env to the recording family its frames decode as. Deliberately
//!   unlinked: this overview compiles whenever `fixtures` is on, but the
//!   module and the trait both need `record`, so an intra-doc link here is
//!   unresolvable in the default configuration.
//!
//! There is no adapter type: [`Evaluator::run_suite`] binds directly on
//! [`Environment`], so envs are registered as themselves (ADR 0076).
//!
//! # Why this lives here and not in `rlevo-environments`
//!
//! Everything in this module names a `rlevo-benchmarks` item —
//! [`EvaluatorConfig`](crate::evaluator::EvaluatorConfig), [`Suite`], and
//! `EnvFamily` (`record`-gated, hence unlinked) — and nothing else. Hosting
//! it on the environments side meant `rlevo-environments` carried an optional
//! dependency on the harness, which is the wrong direction: the harness
//! consumes environments, environments do not consume the harness. The orphan
//! rule permits the impls here because this crate owns the traits (ADR 0080,
//! superseding the module-placement half of ADR 0001).
//!
//! # The `fixtures-box2d` / `fixtures-locomotion` gates
//!
//! The env-family impls for the physics environments are gated on
//! **`rlevo-benchmarks`'s own** passthrough features, not on
//! `rlevo-environments/box2d` directly — a crate cannot `cfg` on another
//! crate's feature. Enable `rlevo-environments/box2d` without
//! `rlevo-benchmarks/fixtures-box2d` and the walker still exists but has no
//! `RecordedEnvFamily` impl. The `rlevo` umbrella keeps the pair in lockstep
//! (`rlevo/box2d` turns both on); a consumer wiring `rlevo-benchmarks`
//! directly has to enable both.
//!
//! [`Environment`]: rlevo_core::environment::Environment
//! [`Suite`]: crate::suite::Suite
//! [`Evaluator::run_suite`]: crate::evaluator::Evaluator::run_suite

/// [`RecordedEnvFamily`](crate::record::RecordedEnvFamily) impls for the
/// built-in environments (feature `record`).
#[cfg(feature = "record")]
pub mod family;
pub mod suites;
