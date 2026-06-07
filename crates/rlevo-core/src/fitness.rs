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
///
/// Each variant carries a `name` that identifies the metric (e.g.
/// `"q_loss"`, `"policy_entropy"`). Names are free-form strings; the harness
/// records them verbatim without normalisation.
#[derive(Debug, Clone)]
pub enum Metric {
    /// A single floating-point measurement (loss, reward, step count as f64).
    Scalar { name: String, value: f64 },
    /// A distribution of values collected over a trial (per-step returns,
    /// priority weights). Consumers may summarise via mean/variance.
    Histogram { name: String, values: Vec<f64> },
    /// A monotonically increasing integer count (environment steps, gradient
    /// updates) that the harness may accumulate across trials.
    Counter { name: String, count: u64 },
}

/// Trait implemented by agents (and internal collectors) that can report
/// method-specific metrics at trial boundaries.
pub trait MetricsProvider {
    /// Returns all metrics accumulated since the last call (or since
    /// construction). Implementations should drain internal accumulators so
    /// that repeated calls do not double-count.
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
    /// Selects an action given the current observation.
    ///
    /// `rng` is supplied by the harness and must be used for any randomness
    /// the policy requires (e.g. epsilon-greedy exploration, stochastic
    /// sampling). Do not source randomness from a privately held RNG; doing
    /// so breaks the harness's reproducibility guarantees.
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
    /// The optimizer's candidate solution type (e.g. a parameter vector or a
    /// genome).
    type Individual;

    /// The fitness landscape the individual is evaluated against.  May be a
    /// marker type when the evaluator itself encodes the landscape (e.g.
    /// `RastriginEvaluator` + `RastriginLandscape`).
    type Landscape;

    /// Returns the scalar fitness of `individual` on `landscape`.
    ///
    /// Higher values must mean better fitness; callers may negate internally
    /// if the landscape is formulated as a minimisation problem.
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
