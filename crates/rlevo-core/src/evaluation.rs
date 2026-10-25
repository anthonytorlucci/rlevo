//! Drive seams consumed by external evaluators.
//!
//! [`GenerationProbe`] is the seam an evaluation harness uses to run work that
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

use crate::fitness::Metric;

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
///   produce identical metric sequences, to the extent the arithmetic under
///   the probe is itself deterministic (a GPU backend may not promise it).
/// - [`advance`](Self::advance) runs exactly one generation and returns its
///   metrics, or `None` once the generation budget is exhausted. It MUST check
///   the budget before stepping, so calling it past exhaustion is a cheap
///   no-op rather than a panic or an over-run generation.
/// - `advance` before `begin` is a caller error; implementors may panic.
/// - [`emit_metrics`](Self::emit_metrics) describes the run so far, since
///   `begin`, under the rule [`Metric`] states, and changes nothing. Before
///   the first generation it reports none.
///
/// # Why `Option` rather than a `done` flag
///
/// The budget lives in the probe, which already owns it. A separate `done`
/// boolean invites a second, unenforced copy of the same number in the
/// driver's configuration — which is exactly what `rlevo-harness`'
/// `EvaluatorConfig::max_steps` is today, hand-synced against the loop's
/// own `max_generations` at every call site. `Option` makes the exhausted
/// state unrepresentable as anything else.
///
/// An implementor may still answer the question itself. The evolutionary
/// loops in `rlevo-evolution` have an inherent `is_done()` beside their
/// `step()`, for a caller who drives one by hand, and their `advance` is that
/// check followed by one step. What this trait avoids is a flag the *driver*
/// must keep in step with the probe: the consumer needs no copy of the
/// budget to know when to stop.
///
/// The evaluator still holds `max_steps`, as a backstop against a probe that
/// never returns `None`. Set it at least as large as the probe's budget. When
/// the backstop is reached, the evaluator calls `advance` once more to learn
/// whether the probe was finished, and warns if it was not. The contract
/// above is what makes that call safe: on a finished probe it does nothing.
/// On an unfinished one it runs one generation, whose metrics are dropped:
/// the evaluator reads `emit_metrics` before that call.
///
/// # Who calls this, and who implements it
///
/// An evaluator calls it, not a person. A user who wants one run drives the
/// loop's own methods; a user who wants many seeded trials and a report gives
/// an evaluator a loop per trial, and the evaluator calls `begin`, `advance`
/// and `emit_metrics` for them.
///
/// Implement it for a generation loop of your own that an evaluator should
/// run. `rlevo-evolution`'s `probe` module has the recipe and two worked
/// impls: `begin` is the loop's reset, `advance` is its budget check, then
/// one step, then that step's metrics, and `emit_metrics` turns the latest
/// generation's metrics into [`Metric`]s.
pub trait GenerationProbe {
    /// Typed per-unit metrics this probe reports.
    type Metrics;

    /// Reset to a fresh initial state, re-seeding deterministically.
    fn begin(&mut self);

    /// Run one unit, or return `None` if the budget is exhausted.
    fn advance(&mut self) -> Option<Self::Metrics>;

    /// Method-specific metrics about the run so far, since
    /// [`begin`](Self::begin), as [`Metric`] states: the same name and the
    /// same rule as an agent's
    /// [`Agent::emit_metrics`](crate::agent::Agent::emit_metrics).
    ///
    /// The evaluation harness reads them once, after the last generation it
    /// counts. The default reports none.
    fn emit_metrics(&self) -> Vec<Metric> {
        Vec::new()
    }
}
