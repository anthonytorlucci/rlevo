//! Shared fixtures and helpers for `rlevo` integration tests.
//!
//! Integration tests under `crates/rlevo/tests/` each compile into a separate
//! binary, so any code they share must live in a library. Before this crate
//! existed, the per-algorithm test files (`ddpg_integration.rs`,
//! `td3_integration.rs`, `sac_integration.rs`, ...) each carried a verbatim
//! copy of the same ~140-line synthetic environment, the same Burn `Flex`
//! determinism preamble, and the same finite/threshold assertions. This crate
//! holds the algorithm-agnostic parts of that boilerplate so each test file is
//! left with only its agent configuration and its genuinely algorithm-specific
//! checks.
//!
//! The crate is **dev-only** — it is consumed exclusively through
//! `[dev-dependencies]` and is never published (`publish = false`). It depends
//! only on `rlevo-core`, so it imposes no constraints on the production
//! algorithm crates.
//!
//! # Modules
//!
//! - [`env`] — the synthetic [`LinearEnv`](env::LinearEnv) 1-D continuous
//!   tracking task shared by every continuous-control algorithm test.
//! - [`flex`] — the [`Autodiff<Flex>`](flex::FlexAutodiff) backend alias plus
//!   the [`flex_guard`](flex::flex_guard) / [`seeded_device`](flex::seeded_device)
//!   determinism helpers.
//! - [`baseline`] — uniform-random-policy rollouts
//!   ([`random_return`](baseline::random_return)) that measure the random
//!   baseline a learning test must beat, instead of hard-coding it.
//! - [`assert`] — reward-finiteness, baseline, and reproducibility assertions
//!   that standardise the per-algorithm acceptance checks.

pub mod assert;
pub mod baseline;
pub mod env;
pub mod flex;
mod macros;

/// Result of one standardised training run, consumed by the suite macros
/// ([`rl_learning_test!`] / [`rl_reproducibility_test!`]).
///
/// A test file provides a single `run(seed, total) -> TrainOutcome` function —
/// the only algorithm-specific glue — and the macros generate the `#[test]`
/// scaffolding around it. `avg_score` feeds the learning / bit-equal checks;
/// `rewards` (the per-episode reward column) feeds the sequence-equal check.
#[derive(Debug, Clone)]
pub struct TrainOutcome {
    /// The agent's moving-average score at the end of the run.
    pub avg_score: f32,
    /// Per-episode rewards from the agent's recent history.
    pub rewards: Vec<f32>,
}
