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

/// A single environment step as seen by an external evaluator.
///
/// Returned by [`BenchEnv::step`]. The observation type is generic so
/// adapters can expose whatever concrete type the underlying environment
/// produces without further erasure.
#[derive(Debug, Clone)]
pub struct BenchStep<Obs> {
    /// The observation the agent receives after the action was applied.
    pub observation: Obs,
    /// Scalar reward signal for the transition.
    pub reward: f64,
    /// Whether the episode has ended (terminal or truncated).
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

/// Object-safe environment interface consumed by external evaluators.
///
/// `BenchEnv` strips the const-generic dimensionality of [`Environment`] so
/// benchmarking harnesses and evolutionary outer loops can work with a plain
/// trait object (`dyn BenchEnv`) rather than threading dimension parameters
/// through their own type signatures.
///
/// Concrete adapters that bridge a typed [`Environment`] to `BenchEnv` live
/// in `rlevo-environments` behind the `bench` feature.
///
/// # Errors
///
/// Both [`reset`] and [`step`] return [`BenchError`], which wraps the
/// upstream [`EnvironmentError`] variants so callers can distinguish
/// recoverable environment failures from programming bugs caught by
/// `catch_unwind`.
///
/// [`Environment`]: crate::environment::Environment
/// [`EnvironmentError`]: crate::environment::EnvironmentError
/// [`reset`]: BenchEnv::reset
/// [`step`]: BenchEnv::step
pub trait BenchEnv {
    /// The observation type the environment produces on each step.
    type Observation;
    /// The action type the environment accepts on each step.
    type Action;

    /// Reset the environment to an initial state and return the first observation.
    ///
    /// # Errors
    ///
    /// Returns [`BenchError::Reset`] if the underlying environment's reset
    /// operation fails.
    fn reset(&mut self) -> Result<Self::Observation, BenchError>;

    /// Apply `action` and advance the environment by one step.
    ///
    /// Returns a [`BenchStep`] containing the next observation, the scalar
    /// reward, and a `done` flag indicating episode termination.
    ///
    /// # Errors
    ///
    /// Returns [`BenchError::Step`] if the underlying environment's step
    /// operation fails.
    fn step(&mut self, action: Self::Action) -> Result<BenchStep<Self::Observation>, BenchError>;
}
