//! Inference-only agent interfaces and fitness-evaluation traits.
//!
//! Three traits plus the [`Metric`] data type:
//!
//! - [`BenchableAgent`] — frozen policies (RL-style) that consume an
//!   observation and produce an action. RL training/replay state must be
//!   stripped in a `Frozen*Policy` adapter before benchmarking.
//! - [`FitnessEvaluable`] — optimizer-on-landscape path, used when the
//!   "environment" is a pure fitness function (Rastrigin, Ackley, MPB).
//! - [`Landscape`] — self-evaluating numerical landscape; collapses the
//!   evaluator/landscape split when the landscape *is* the fitness
//!   function. Consumed by `rlevo-evolution`'s `FromLandscape` adapter.
//!
//! [`Metric`] and [`MetricsProvider`] live here because [`BenchableAgent`]
//! returns `Vec<Metric>` from its optional `emit_metrics` hook; co-locating
//! them keeps the trait self-contained without a dep on the harness crate.

use rand::Rng;

/// Method-specific signal emitted by an agent or aggregator at trial
/// boundaries.
#[derive(Debug, Clone)]
pub enum Metric {
    Scalar { name: String, value: f64 },
    Histogram { name: String, values: Vec<f64> },
    Counter { name: String, count: u64 },
}

/// Trait implemented by agents (and internal collectors) that can report
/// method-specific metrics at trial boundaries.
pub trait MetricsProvider {
    fn emit(&self) -> Vec<Metric>;
}

/// Minimal inference interface required by an external evaluator.
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
/// in `rlevo-benchmarks` consumes this trait.
pub trait FitnessEvaluable {
    type Individual;
    type Landscape;

    fn evaluate(&self, individual: &Self::Individual, landscape: &Self::Landscape) -> f64;
}

/// Self-evaluating numerical fitness landscape.
///
/// Implementors carry both the parameters of the landscape (dimension,
/// constants) and the scalar `f(x)` evaluation. Use this when the
/// landscape *is* the fitness function — Sphere, Ackley, Rastrigin — so
/// callers do not need to define a separate evaluator alongside a marker
/// landscape type as `FitnessEvaluable` requires.
///
/// `rlevo-evolution`'s `FromLandscape` adapter wraps any `Landscape` into
/// a `BatchFitnessFn<B, Tensor<B, 2>>`, mirroring the row-by-row host
/// evaluation that `FromFitnessEvaluable` performs.
pub trait Landscape: Send + Sync {
    /// Evaluates the landscape at point `x` and returns scalar fitness.
    fn evaluate(&self, x: &[f64]) -> f64;
}
