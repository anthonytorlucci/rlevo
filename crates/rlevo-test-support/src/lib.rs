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
//! only on `rlevo-core` and `rlevo-environments`, neither of them an algorithm
//! crate, so it imposes no constraints on the production algorithm crates.
//!
//! # Modules
//!
//! - [`mod@env`] — the synthetic [`LinearEnv`](env::LinearEnv) 1-D continuous
//!   tracking task shared by every continuous-control algorithm test.
//! - [`flex`] — the [`flex_guard`](flex::flex_guard) /
//!   [`seeded_device`](flex::seeded_device) determinism helpers for the `Flex`
//!   device.
//! - [`baseline`] — uniform-random-policy rollouts
//!   ([`random_return`](baseline::random_return)) that measure the random
//!   baseline a learning test must beat, instead of hard-coding it.
//! - [`mod@assert`] — reward-finiteness, baseline, and reproducibility assertions
//!   that standardise the per-algorithm acceptance checks.
//! - [`capture`] — [`FieldCapture`](capture::FieldCapture), for tests that
//!   assert on the `tracing` events an agent emits rather than on the values
//!   it returns.

pub mod assert;
pub mod baseline;
pub mod capture;
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
    /// The mean return of the run's last [`TrainOutcome::WINDOW`] episodes.
    pub avg_score: f32,
    /// The returns of those episodes, oldest first.
    pub rewards: Vec<f32>,
}

impl TrainOutcome {
    /// How many of the run's last episodes the outcome averages.
    pub const WINDOW: usize = 100;

    /// The outcome of a run whose completed episodes returned `returns`, in
    /// order: the harness's `train` hands back one summary per episode, and a
    /// test passes their `return_value`s. With no episode, `avg_score` is `0.0`.
    #[must_use]
    // Episode counts are bounded by a test's step budget, far below f32's
    // 2^24 exact-integer limit, and a return narrows to the f32 the agents
    // sum their rewards in.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn from_returns(returns: impl IntoIterator<Item = f64>) -> Self {
        let all: Vec<f32> = returns.into_iter().map(|r| r as f32).collect();
        let recent = all[all.len().saturating_sub(Self::WINDOW)..].to_vec();
        let avg_score = if recent.is_empty() {
            0.0
        } else {
            recent.iter().sum::<f32>() / recent.len() as f32
        };
        Self {
            avg_score,
            rewards: recent,
        }
    }
}
