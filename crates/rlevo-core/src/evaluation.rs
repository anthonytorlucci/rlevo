//! Object-safe environment interface consumed by external evaluators.
//!
//! [`BenchEnv`] is intentionally narrower than [`Environment`] so consumers
//! (benchmarking harnesses, evolutionary outer loops) do not have to thread
//! const-generic dimensions through their signatures. Adapters that wrap
//! concrete [`Environment`] impls live in `rlevo-environments` (behind the
//! `bench` feature).
//!
//! `reset` and `step` return `Result<_, BenchError>` so adapters preserve
//! upstream recoverable errors ([`EnvironmentError`]) without escalating
//! them to panics. Consuming harnesses still wrap callers in `catch_unwind`
//! to capture genuine programming-bug panics separately.
//!
//! [`Environment`]: crate::environment::Environment
//! [`EnvironmentError`]: crate::environment::EnvironmentError

use crate::environment::EnvironmentError;

#[derive(Debug, Clone)]
pub struct BenchStep<Obs> {
    pub observation: Obs,
    pub reward: f64,
    pub done: bool,
}

/// Recoverable error reported by a [`BenchEnv`] impl.
///
/// Wraps [`EnvironmentError`] so adapters preserve the typed upstream
/// error rather than collapsing it to a string.
#[derive(Debug, thiserror::Error)]
pub enum BenchError {
    #[error("environment reset failed: {0}")]
    Reset(#[source] EnvironmentError),
    #[error("environment step failed: {0}")]
    Step(#[source] EnvironmentError),
}

pub trait BenchEnv {
    type Observation;
    type Action;

    fn reset(&mut self) -> Result<Self::Observation, BenchError>;
    fn step(
        &mut self,
        action: Self::Action,
    ) -> Result<BenchStep<Self::Observation>, BenchError>;
}
