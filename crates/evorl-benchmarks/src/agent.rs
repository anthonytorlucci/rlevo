//! Inference-only agent interfaces for the benchmark harness.
//!
//! Two traits:
//!
//! - [`BenchableAgent`] — frozen policies (RL-style) that consume an
//!   observation and produce an action. RL training/replay state must be
//!   stripped in a `Frozen*Policy` adapter before benchmarking.
//! - [`FitnessEvaluable`] — optimizer-on-landscape path, used when the
//!   "environment" is a pure fitness function (Rastrigin, Ackley, MPB).

use rand::Rng;

use crate::metrics::Metric;

/// Minimal inference interface required by the `Evaluator`.
///
/// Implementors must be deterministic given a fixed RNG state and must not
/// mutate learnable parameters. Internal RNG state (e.g. for stochastic
/// policies) may be mutated, which is why `act` takes `&mut self`.
///
/// The `rng` argument is owned by the harness so reproducibility is
/// guaranteed regardless of the agent's internal RNG discipline.
pub trait BenchableAgent<Obs, Act> {
    fn act(&mut self, obs: &Obs, rng: &mut dyn Rng) -> Act;

    /// Optional hook to emit method-specific metrics at trial end.
    fn emit_metrics(&self) -> Vec<Metric> {
        Vec::new()
    }
}

/// Evaluates an optimizer against a fitness landscape.
///
/// Used when the benchmark IS the fitness function (e.g. Rastrigin) rather
/// than a stateful `Environment`. The `Evaluator::run_optimizer_trial` path
/// consumes this trait.
pub trait FitnessEvaluable {
    type Individual;
    type Landscape;

    fn evaluate(&self, individual: &Self::Individual, landscape: &Self::Landscape) -> f64;
}
