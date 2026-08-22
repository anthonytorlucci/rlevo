//! Preset [`Suite`] factories for the canonical environments in this crate.
//!
//! Each factory returns a single-env [`Suite`] keyed on a deterministic
//! per-trial seed. Envs are registered as themselves — [`Evaluator::run_suite`]
//! binds on [`Environment`] directly, so there is no adapter to wrap them in
//! (ADR 0076). Stitching multiple suites of different env types is not
//! supported — [`Suite<E>`] is monomorphic.
//!
//! A heterogeneous "all of classic control" suite is deferred, and boxing is
//! **not** the missing piece: `BenchEnv` erases the const-generic ranks but
//! preserves `Observation`/`Action`, and those are exactly what differ across
//! these envs (`CartPoleObservation` vs. `PendulumObservation`, …), so one
//! `Box<dyn BenchEnv<…>>` can only hold envs already sharing an obs/action
//! pair. The real prerequisite is an obs/action normalization, which does not
//! exist. See ADR 0075.
//!
//! [`Suite`]: crate::suite::Suite
//! [`Suite<E>`]: crate::suite::Suite
//! [`Environment`]: rlevo_core::environment::Environment
//! [`Evaluator::run_suite`]: crate::evaluator::Evaluator::run_suite

use rlevo_environments::classic::{
    CartPole, CartPoleConfig, Pendulum, PendulumConfig, TenArmedBandit,
};

use crate::evaluator::EvaluatorConfig;
use crate::suite::Suite;

/// Single-env suite running [`TenArmedBandit`] on the harness.
///
/// Uses [`TenArmedBandit::with_seed`] so the per-trial seed routes into the
/// arm-mean RNG.
#[must_use]
pub fn ten_armed_bandit_suite(cfg: EvaluatorConfig) -> Suite<TenArmedBandit> {
    Suite::new("ten-armed-bandit", cfg)
        .with_env("ten-armed-bandit-default", TenArmedBandit::with_seed)
}

/// Single-env suite running [`CartPole`] (Gymnasium `CartPole-v1`).
///
/// # Panics
///
/// Panics if the hard-coded suite configuration fails validation. The values
/// are literals in this function, so a panic indicates a bug here rather than
/// bad caller input.
#[must_use]
pub fn cartpole_suite(cfg: EvaluatorConfig) -> Suite<CartPole> {
    Suite::new("cartpole", cfg).with_env("cartpole-default", |seed| {
        CartPole::with_config(CartPoleConfig {
            seed,
            ..CartPoleConfig::default()
        })
        .expect("valid config")
    })
}

/// Single-env suite running [`Pendulum`] (continuous-action swing-up).
///
/// # Panics
///
/// Panics if the hard-coded suite configuration fails validation. The values
/// are literals in this function, so a panic indicates a bug here rather than
/// bad caller input.
#[must_use]
pub fn pendulum_suite(cfg: EvaluatorConfig) -> Suite<Pendulum> {
    Suite::new("pendulum", cfg).with_env("pendulum-default", |seed| {
        Pendulum::with_config(PendulumConfig {
            seed,
            ..PendulumConfig::default()
        })
        .expect("valid config")
    })
}
