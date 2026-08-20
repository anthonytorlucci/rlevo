//! Minimal drive interface consumed by external evaluators.
//!
//! [`BenchEnv`] carries no const-generic dimensions, so consumers
//! (benchmarking harnesses, evolutionary outer loops) do not have to thread
//! them through their signatures. Adapters that wrap concrete
//! [`Environment`] impls live in `rlevo-environments` (behind the `bench`
//! feature).
//!
//! Its real job is spanning **two disjoint implementor families**: typed
//! environments (via `BenchAdapter`) and evolutionary generation-steppers
//! (`EvolutionaryHarness`, `CoEvolutionaryHarness`), which are not
//! [`Environment`]s at all. See ADR 0075.
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

/// Minimal drive interface consumed by external evaluators.
///
/// `BenchEnv` strips the const-generic **ranks** of [`Environment`]
/// (`R`/`SR`/`AR`) so harnesses and evolutionary outer loops need not thread
/// dimension parameters through their own signatures.
///
/// # What this trait does *not* buy
///
/// The trait is object-safe — `Box<dyn BenchEnv<Observation = O, Action = A>>`
/// is legal once both associated types are named — but that erasure is on the
/// rank axis, not the modality axis. `Observation` and `Action` survive
/// erasure, and those are precisely what differ between environments
/// (`CartPoleObservation` vs. `PendulumObservation`, and so on). A single
/// `dyn BenchEnv` can therefore hold only envs sharing one obs/action pair,
/// so this trait does **not** enable a heterogeneous "all of classic control"
/// suite; that needs an obs/action normalization, which does not exist. No
/// `dyn BenchEnv` is constructed anywhere in the workspace today.
///
/// See ADR 0075, which supersedes the object-safety rationale this trait was
/// originally justified by.
///
/// # Implementors
///
/// Two disjoint families, which is the one thing this trait does that nothing
/// else in `rlevo-core` does:
///
/// - typed [`Environment`]s, via `BenchAdapter` in `rlevo-environments`
///   (behind the `bench` feature);
/// - evolutionary generation-steppers (`EvolutionaryHarness`,
///   `CoEvolutionaryHarness` in `rlevo-evolution`), which step generations
///   rather than transitions and are not [`Environment`]s.
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

/// Something that runs a fixed budget of self-paced units of work.
///
/// The second drive seam in this module, alongside [`BenchEnv`]. Where
/// `BenchEnv` models an episodic agent/environment interaction — observation
/// in, action out, reward and a `done` flag back — `GenerationProbe` models a
/// loop that simply advances itself: no observation, no action, no episode
/// axis. An evolutionary generation loop is the motivating case, but nothing
/// here names a genome or a population.
///
/// # Contract
///
/// - [`begin`](Self::begin) resets to a fresh initial state and re-seeds
///   deterministically. Two `begin`-to-exhaustion runs of the same probe MUST
///   produce identical metric sequences.
/// - [`advance`](Self::advance) runs exactly one generation and returns its
///   metrics, or `None` once the generation budget is exhausted. It MUST check
///   the budget before stepping, so calling it past exhaustion is a cheap
///   no-op rather than a panic or an over-run generation.
/// - `advance` before `begin` is a caller error; implementors may panic.
///
/// # Why `Option` rather than a `done` flag
///
/// The budget lives in the probe, which already owns it. A separate `done`
/// boolean invites a second, unenforced copy of the same number in the
/// driver's configuration — which is exactly what `rlevo-benchmarks`'
/// `EvaluatorConfig::max_steps` is today, hand-synced against the harness's
/// own `max_generations` at every call site. `Option` makes the exhausted
/// state unrepresentable as anything else.
pub trait GenerationProbe {
    /// Typed per-unit metrics this probe reports.
    type Metrics;

    /// Reset to a fresh initial state, re-seeding deterministically.
    fn begin(&mut self);

    /// Run one unit, or return `None` if the budget is exhausted.
    fn advance(&mut self) -> Option<Self::Metrics>;
}
