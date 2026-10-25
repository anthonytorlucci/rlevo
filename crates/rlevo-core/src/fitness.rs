//! Fitness evaluation, and the metric data type.
//!
//! - [`Landscape`] — a self-evaluating numerical landscape, used when the
//!   "environment" is a pure fitness function (Sphere, Rastrigin, Ackley).
//!   Consumed by `rlevo-evolution`'s `FromLandscape` adapter.
//! - [`Metric`] — method-specific signal about a trial, read from what the
//!   trial drives: an agent through
//!   [`Agent::emit_metrics`](crate::agent::Agent::emit_metrics), a
//!   generation loop through
//!   [`GenerationProbe::emit_metrics`](crate::evaluation::GenerationProbe::emit_metrics).
//!   It lives in this crate so neither producer depends on the harness that
//!   reads it.

use crate::objective::ObjectiveSense;

/// Method-specific signal about a trial, read from the agent or the
/// generation loop the trial drives.
///
/// Each variant carries a `name` that identifies the metric (e.g.
/// `"q_loss"`, `"policy_entropy"`). As far as this crate is concerned the name
/// is a free-form string: nothing here validates, normalises, or rewrites it,
/// and no naming scheme is imposed on implementors.
///
/// A consumer that collects these metrics may reserve namespaces of its own —
/// typically to protect measurements it made itself from being replaced by an
/// emitted one of the same name. Such a consumer may record an emitted metric
/// under a name other than the one given here. Which names a consumer reserves,
/// and what it does with a collision, is governed by that consumer's own
/// documentation; this crate neither defines nor constrains it.
///
/// # What a read covers
///
/// A producer's `emit_metrics` describes the trial so far, from its start to
/// the read:
///
/// - a [`Counter`](Self::Counter) is the total since the trial began;
/// - a [`Scalar`](Self::Scalar) is the value at the read: a running best, a
///   mean so far, the latest generation's best;
/// - a [`Histogram`](Self::Histogram) holds every sample since the trial
///   began.
///
/// A trial begins when its agent is built, or when its generation loop's
/// `begin` resets it. A read changes nothing, so it can be repeated: each read
/// is complete on its own, the reader has one value per read, and the latest
/// is the current one. A reader never adds two reads together. The evaluation
/// harness reads once, after the last episode or generation it counts; a
/// reader that logs a time series reads at each boundary and logs every value
/// at its point on the timeline.
#[derive(Debug, Clone)]
pub enum Metric {
    /// A single floating-point measurement (loss, reward, step count as f64).
    Scalar { name: String, value: f64 },
    /// A distribution of values (per-step returns, priority weights), every
    /// sample since the trial began. Consumers may summarise via
    /// mean/variance.
    Histogram { name: String, values: Vec<f64> },
    /// A tally of discrete occurrences (environment steps, gradient updates),
    /// the total since the trial began. The counted quantity only ever grows
    /// — a count is never negative and never decreases — so `count` is a
    /// number of events, not a level that can fall.
    Counter { name: String, count: u64 },
}

/// Self-evaluating numerical fitness landscape.
///
/// Implementors carry both the parameters of the landscape (dimension,
/// constants) and the scalar `f(x)` evaluation: the landscape *is* the
/// fitness function, as Sphere, Ackley and Rastrigin are. A test or an
/// example that scores a candidate with its own formula implements this
/// trait on a type of its own.
///
/// `rlevo-evolution`'s `FromLandscape` adapter wraps any `Landscape` into
/// a `BatchFitnessFn<Tensor<2>>`, evaluating each population row on
/// the host.
pub trait Landscape: Send + Sync {
    /// Evaluates the landscape at point `x` and returns its scalar cost in the
    /// landscape's **natural** value space (no hand-negation).
    ///
    /// A landscape is a cost surface by definition: the bundled landscapes
    /// (Sphere, Ackley, Rastrigin) are zero at their global optimum and
    /// positive elsewhere. The maximise-native engine reconciles this via
    /// [`sense`](Landscape::sense) at a single chokepoint — you never negate
    /// the value here.
    #[must_use]
    fn evaluate(&self, x: &[f64]) -> f64;

    /// The optimisation direction of this landscape.
    ///
    /// A landscape is a cost surface, so this defaults to
    /// [`ObjectiveSense::Minimize`] — the bundled landscapes need no
    /// per-type override. The `FromLandscape` adapter forwards this to the
    /// evolutionary loop, which negates a `Minimize` objective into canonical (maximise)
    /// space and reports `best_fitness` back in this natural sense.
    fn sense(&self) -> ObjectiveSense {
        ObjectiveSense::Minimize
    }
}
