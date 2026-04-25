//! Minimal environment interface consumed by the evaluator.
//!
//! `BenchEnv` is intentionally narrower than `rlevo_core::Environment` so
//! the harness does not have to thread const-generic dimensions through its
//! signatures. Adapters that wrap concrete `Environment` impls live in
//! `rlevo-envs` (behind the `bench` feature).
//!
//! `reset` and `step` return `Result<_, BenchError>` so adapters can preserve
//! upstream recoverable errors (e.g. `rlevo_core::EnvironmentError`) without
//! escalating them to panics. The harness's `catch_unwind` boundary remains
//! in place to capture genuine programming-bug panics separately.

#[derive(Debug, Clone)]
pub struct BenchStep<Obs> {
    pub observation: Obs,
    pub reward: f64,
    pub done: bool,
}

/// Recoverable error reported by a `BenchEnv` impl.
///
/// Stringly-typed by design: `rlevo-benchmarks` does not depend on
/// `rlevo-core`, so a typed bridge from `rlevo_core::EnvironmentError`
/// would invert the dependency direction. Adapters convert via `Display`.
#[derive(Debug, Clone, thiserror::Error)]
pub enum BenchError {
    #[error("environment reset failed: {0}")]
    Reset(String),
    #[error("environment step failed: {0}")]
    Step(String),
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
