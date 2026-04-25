//! Preset [`Suite`] factories for the canonical environments in this crate.
//!
//! Each factory returns a single-env [`Suite`] keyed on a deterministic
//! per-trial seed. Stitching multiple suites of different env types is a
//! v0.1 limitation — [`Suite<E>`] is monomorphic, so a heterogeneous
//! "all of classic control" suite needs a `Box<dyn BenchEnv<…>>` design
//! (deferred; tracked in the follow-up to ADR-0001).
//!
//! [`Suite`]: rlevo_benchmarks::suite::Suite
//! [`Suite<E>`]: rlevo_benchmarks::suite::Suite

use rlevo_benchmarks::evaluator::EvaluatorConfig;
use rlevo_benchmarks::suite::Suite;

use crate::bench::adapter::BenchAdapter;
use crate::classic::{CartPole, CartPoleConfig, Pendulum, PendulumConfig, TenArmedBandit};

/// Single-env suite running [`TenArmedBandit`] on the harness.
///
/// Uses [`TenArmedBandit::with_seed`] so the per-trial seed routes into the
/// arm-mean RNG.
#[must_use]
pub fn ten_armed_bandit_suite(cfg: EvaluatorConfig) -> Suite<BenchAdapter<TenArmedBandit, 1, 1, 1>> {
    Suite::new("ten-armed-bandit", cfg).with_env("ten-armed-bandit-default", |seed| {
        BenchAdapter::new(TenArmedBandit::with_seed(seed))
    })
}

/// Single-env suite running [`CartPole`] (Gymnasium `CartPole-v1`).
#[must_use]
pub fn cartpole_suite(cfg: EvaluatorConfig) -> Suite<BenchAdapter<CartPole, 1, 1, 1>> {
    Suite::new("cartpole", cfg).with_env("cartpole-default", |seed| {
        BenchAdapter::new(CartPole::with_config(CartPoleConfig {
            seed,
            ..CartPoleConfig::default()
        }))
    })
}

/// Single-env suite running [`Pendulum`] (continuous-action swing-up).
#[must_use]
pub fn pendulum_suite(cfg: EvaluatorConfig) -> Suite<BenchAdapter<Pendulum, 1, 1, 1>> {
    Suite::new("pendulum", cfg).with_env("pendulum-default", |seed| {
        BenchAdapter::new(Pendulum::with_config(PendulumConfig {
            seed,
            ..PendulumConfig::default()
        }))
    })
}
