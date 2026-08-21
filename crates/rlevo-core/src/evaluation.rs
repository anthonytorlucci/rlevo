//! Drive seams consumed by external evaluators.
//!
//! [`GenerationProbe`] is the seam a benchmarking harness uses to run work that
//! paces itself — an evolutionary generation loop, most concretely — rather
//! than an episodic agent/environment interaction. Episodic work is driven
//! through [`Environment`] and [`Snapshot`] directly; the evaluator binds on
//! them and infers their rank parameters from the suite it is given (ADR 0076).
//!
//! This module previously also held `BenchEnv`, `BenchStep`, and `BenchError`,
//! a rank-erasing environment interface hoisted here by ADR 0004. They were
//! removed by ADR 0077 once nothing implemented them: the erasure they existed
//! for was never exercised, and it was on the wrong axis to deliver the
//! heterogeneous suite it was justified by (ADR 0075).
//!
//! [`Environment`]: crate::environment::Environment
//! [`Snapshot`]: crate::environment::Snapshot

/// Something that runs a fixed budget of self-paced units of work.
///
/// Where an [`Environment`] models an episodic interaction — observation in,
/// action out, reward and an episode status back — `GenerationProbe` models a
/// loop that simply advances itself: no observation, no action, no episode
/// axis. An evolutionary generation loop is the motivating case, but nothing
/// here names a genome or a population.
///
/// [`Environment`]: crate::environment::Environment
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
