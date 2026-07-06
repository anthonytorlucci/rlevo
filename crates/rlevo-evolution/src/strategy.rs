//! Central [`Strategy`] trait and the [`EvolutionaryHarness`] adapter.
//!
//! # The ask / tell contract
//!
//! A [`Strategy`] exposes three methods that together drive one
//! generation:
//!
//! 1. [`init`](Strategy::init) — build the initial state (sampling the
//!    population, initializing σ, generation counter, etc).
//! 2. [`ask`](Strategy::ask) — propose the next population as a genome
//!    container.
//! 3. [`tell`](Strategy::tell) — consume that population together with
//!    its fitness and produce the next state plus a metrics snapshot.
//!
//! All three methods take the RNG explicitly so the harness owns all
//! stochasticity; strategies carry *no* internal PRNG state.
//!
//! # Fitness convention
//!
//! The engine is **maximise-native**: the fitness tensor passed to
//! [`tell`](Strategy::tell) is a **canonical** value where *higher is
//! better*, and strategies maximise it directly. The
//! [`StrategyMetrics::best_fitness`] field is the largest value observed in
//! a generation; [`StrategyMetrics::best_fitness_ever`] is a rolling
//! maximum. Strategies are **sense-unaware** — they never see an
//! [`ObjectiveSense`](rlevo_core::objective::ObjectiveSense). Cost
//! objectives (e.g. the benchmark landscapes) are negated into canonical
//! space at exactly one chokepoint, [`EvolutionaryHarness`], which also
//! maps metrics back to the objective's declared sense for reporting.
//!
//! # The harness adapter
//!
//! [`EvolutionaryHarness`] glues a strategy to any
//! [`BatchFitnessFn`] and implements
//! [`BenchEnv`], so the benchmark
//! evaluator drives it just like an RL environment.

use std::fmt::Debug;
use std::marker::PhantomData;

use burn::tensor::{Tensor, backend::Backend};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use rlevo_core::config::{ConfigError, Validate};
use rlevo_core::evaluation::{BenchEnv, BenchError, BenchStep};
use rlevo_core::objective::ObjectiveSense;

use crate::fitness::BatchFitnessFn;
use crate::observer::{PopulationSnapshot, SharedPopulationObserver};

/// Central evolutionary-strategy abstraction.
///
/// The trait is intentionally pure — [`ask`](Self::ask) and
/// [`tell`](Self::tell) return a new `State` rather than mutating
/// through `&mut self`. That keeps strategies free of interior
/// mutability (so many instances can run in parallel without locks) and
/// makes [`Clone`]-based checkpointing straightforward.
///
/// # Example
///
/// The example below uses [`GeneticAlgorithm`] as a concrete strategy and
/// drives one ask/tell cycle by hand. Concrete strategies expose their state
/// fields directly; generic code over `S: Strategy<B>` must access state
/// only through [`Strategy::best`] and the tuple returns of `ask`/`tell`.
///
/// ```no_run
/// use burn::backend::Flex;
/// use burn::tensor::TensorData;
/// use rlevo_evolution::Strategy;
/// use rlevo_evolution::algorithms::ga::{GaConfig, GeneticAlgorithm};
/// use rand::{rngs::StdRng, SeedableRng};
///
/// let device = Default::default();
/// let strategy = GeneticAlgorithm::<Flex>::new();
/// let params = GaConfig::default_for(64, 10);
/// let mut rng = StdRng::seed_from_u64(0);
/// let state = strategy.init(&params, &mut rng, &device);
/// // state.population is a GaState field; dims() is (pop_size, genome_dim).
/// assert_eq!(state.population.dims(), [64, 10]);
/// ```
///
/// [`GeneticAlgorithm`]: crate::algorithms::ga::GeneticAlgorithm
///
/// # Type Parameters
///
/// - `B`: Burn backend.
///
/// # Associated Types
///
/// - `Params`: Static configuration for a run (population size, σ, F,
///   CR, …). Adaptive algorithms mutate their adaptive quantities inside
///   `State`, not `Params`.
/// - `State`: Generation-to-generation state (current population, σ,
///   best-so-far, RNG-free sub-statistics). Must be clonable so the
///   harness can snapshot before a risky step if needed.
/// - `Genome`: Genome container produced by `ask` and consumed by
///   `tell`. Typically a `Tensor<B, 2>` for real-valued strategies or a
///   `Tensor<B, 2, Int>` for binary/integer kinds.
pub trait Strategy<B: Backend>: Send + Sync {
    /// Static parameters for a run.
    type Params: Clone + Debug + Send + Sync;

    /// Generation-to-generation state.
    type State: Clone + Debug + Send;

    /// Genome container produced by [`ask`](Self::ask).
    type Genome: Clone + Send;

    /// Build the initial state.
    ///
    /// Samples the initial population, primes adaptive quantities, and
    /// sets the generation counter to zero.
    fn init(
        &self,
        params: &Self::Params,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Self::State;

    /// Propose the next population.
    ///
    /// Takes the current `state` and returns the genome to evaluate
    /// together with an updated state. The returned state typically
    /// carries pre-computed bookkeeping (e.g. the parent indices a
    /// tournament-based GA sampled) so [`tell`](Self::tell) can reuse
    /// them without re-sampling.
    fn ask(
        &self,
        params: &Self::Params,
        state: &Self::State,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> (Self::Genome, Self::State);

    /// Consume fitness values and produce the next state.
    ///
    /// `fitness` has shape `(pop_size,)` on the same device as the
    /// population. Strategies pull it to host only if they need to —
    /// e.g. for tournament index lookups.
    fn tell(
        &self,
        params: &Self::Params,
        population: Self::Genome,
        fitness: Tensor<B, 1>,
        state: Self::State,
        rng: &mut dyn Rng,
    ) -> (Self::State, StrategyMetrics);

    /// Best-so-far accessor.
    ///
    /// Returns `None` before the first [`tell`](Self::tell) call.
    /// The tuple is `(genome, fitness)` where `fitness` is the **canonical**
    /// (maximise-convention) scalar — the largest value seen across all
    /// completed generations. The harness maps it back to the objective's
    /// declared sense before surfacing it to callers.
    fn best(&self, state: &Self::State) -> Option<(Self::Genome, f32)>;
}

/// Per-generation summary reported by [`Strategy::tell`].
///
/// All statistics refer to the generation that just finished evaluating.
/// These values are in **canonical (maximise) space**: *higher is better*.
/// Strategies are sense-unaware, so the metrics they emit are always
/// canonical. [`EvolutionaryHarness::latest_metrics`] maps them back to the
/// objective's declared sense before surfacing them to callers, so a
/// `Minimize` landscape reads as its natural cost (Sphere → 0).
///
/// When printed in a benchmark showcase (e.g. `ackley_showcase`), the
/// two most informative fields are:
///
/// - **`best_fitness_ever`** — the best (canonical: largest) fitness seen
///   across *all* generations so far. This is a rolling maximum that tells
///   you how close the best individual ever found came to the optimum.
/// - **`mean_fitness`** — the arithmetic mean of the current generation's
///   per-individual fitness vector. This tells you the average quality of
///   the population in that generation.
///
/// A large gap between `best_fitness_ever` and `mean_fitness` in the final
/// generation usually indicates premature convergence: a few elite
/// individuals found a good basin while the rest of the population is still
/// scattered. A small gap suggests the whole population has settled near the
/// same optimum.
#[derive(Debug, Clone)]
pub struct StrategyMetrics {
    /// Zero-based generation index.
    generation: usize,
    /// Number of individuals evaluated in this generation.
    population_size: usize,
    /// Best (canonical: largest) fitness observed in this generation.
    best_fitness: f32,
    /// Mean fitness across this generation's population.
    ///
    /// This is the arithmetic mean of the per-individual fitness vector
    /// for the generation that just finished. In a showcase table printed
    /// after a run, this value reflects the *final* generation's average
    /// quality. See the struct-level docs for how to interpret the gap
    /// between this field and [`Self::best_fitness_ever`].
    mean_fitness: f32,
    /// Worst (canonical: smallest) fitness observed in this generation.
    worst_fitness: f32,
    /// Best fitness seen across *all* generations to date.
    ///
    /// This is a rolling maximum (`previous_best.max(current_generation_best)`)
    /// in canonical space. When mapped back to the objective's sense and
    /// printed in a benchmark showcase, it represents the best solution
    /// quality found during the entire run. For landscapes whose global
    /// optimum is known (e.g. Ackley → 0), the harness-reported value tells
    /// you how close the algorithm got to the theoretical optimum.
    best_fitness_ever: f32,
}

impl StrategyMetrics {
    /// Computes population statistics from a host-side fitness slice.
    ///
    /// Each value is passed through the crate's NaN-hygiene primitive
    /// [`sanitize_fitness`](crate::fitness::sanitize_fitness) before folding, so
    /// a `NaN` is treated as `f32::NEG_INFINITY` (the worst value under the
    /// maximise convention, ADR 0023) *consistently* across every statistic. A
    /// generation containing any `NaN` therefore yields `worst_fitness = −∞` and
    /// `mean_fitness = −∞` while `best_fitness` remains the largest finite value —
    /// degenerate but well-defined, rather than the old silent asymmetry where
    /// `best`/`worst` ignored the `NaN` (comparisons against `NaN` are false) yet
    /// `mean` propagated it.
    ///
    /// # Panics
    ///
    /// Panics if `fitnesses` is empty. Callers hold a non-empty population by
    /// construction — `pop_size` is validated non-zero at the harness
    /// constructor (ADR 0026).
    #[must_use]
    pub fn from_host_fitness(generation: usize, fitnesses: &[f32], best_fitness_ever: f32) -> Self {
        assert!(!fitnesses.is_empty(), "fitness slice must be non-empty");
        let population_size = fitnesses.len();
        // Canonical (maximise) space: best is the largest value, worst the
        // smallest, and best-ever a rolling maximum. Each value is sanitized
        // (NaN → −∞) up front so all three statistics agree on the crate-wide
        // NaN convention instead of best/worst silently dropping NaN while the
        // sum propagates it.
        let (mut best, mut worst, mut sum) = (f32::NEG_INFINITY, f32::INFINITY, 0.0_f32);
        for &f in fitnesses {
            let f = crate::fitness::sanitize_fitness(f);
            if f > best {
                best = f;
            }
            if f < worst {
                worst = f;
            }
            sum += f;
        }
        #[allow(clippy::cast_precision_loss)]
        let mean = sum / population_size as f32;
        Self {
            generation,
            population_size,
            best_fitness: best,
            mean_fitness: mean,
            worst_fitness: worst,
            best_fitness_ever: best_fitness_ever.max(best),
        }
    }

    /// Zero-based generation index.
    #[must_use]
    pub fn generation(&self) -> usize {
        self.generation
    }

    /// Number of individuals evaluated in this generation.
    #[must_use]
    pub fn population_size(&self) -> usize {
        self.population_size
    }

    /// Best (canonical: largest) fitness observed in this generation.
    #[must_use]
    pub fn best_fitness(&self) -> f32 {
        self.best_fitness
    }

    /// Mean fitness across this generation's population.
    #[must_use]
    pub fn mean_fitness(&self) -> f32 {
        self.mean_fitness
    }

    /// Worst (canonical: smallest) fitness observed in this generation.
    #[must_use]
    pub fn worst_fitness(&self) -> f32 {
        self.worst_fitness
    }

    /// Best (canonical: largest) fitness seen across *all* generations to date.
    #[must_use]
    pub fn best_fitness_ever(&self) -> f32 {
        self.best_fitness_ever
    }
}

/// Builds a per-generation [`PopulationSnapshot`] from a host-side fitness
/// vector, or `None` when the vector is empty.
///
/// `fitnesses` is in **natural (user-sense)** space: the best individual is the
/// smallest value for a [`Minimize`](ObjectiveSense::Minimize) objective and the
/// largest for [`Maximize`](ObjectiveSense::Maximize). Returning `None` on an
/// empty vector guards against emitting an out-of-range `best_index` (the fold
/// would otherwise default to `0`, indexing into a zero-length slice).
fn build_population_snapshot(
    generation: u32,
    fitnesses: Vec<f32>,
    sense: ObjectiveSense,
) -> Option<PopulationSnapshot> {
    if fitnesses.is_empty() {
        return None;
    }
    let best_index = fitnesses
        .iter()
        .enumerate()
        .reduce(|best, cur| {
            let better = match sense {
                ObjectiveSense::Minimize => cur.1 < best.1,
                ObjectiveSense::Maximize => cur.1 > best.1,
            };
            if better { cur } else { best }
        })
        .map_or(0, |(i, _)| u32::try_from(i).unwrap_or(0));
    Some(PopulationSnapshot {
        generation,
        fitnesses,
        diversity: None,
        best_index,
        best_genome_digest: None,
        parents_of_best: Vec::new(),
    })
}

/// Wraps a [`Strategy`] into a [`BenchEnv`] so the benchmark harness can
/// drive it.
///
/// # Example
///
/// ```no_run
/// use burn::backend::Flex;
/// use rlevo_core::fitness::FitnessEvaluable;
/// use rlevo_core::evaluation::BenchEnv;
/// use rlevo_evolution::algorithms::ga::{GaConfig, GeneticAlgorithm};
/// use rlevo_evolution::fitness::FromFitnessEvaluable;
/// use rlevo_evolution::strategy::EvolutionaryHarness;
///
/// struct Sphere;
/// struct SphereFit;
/// impl FitnessEvaluable for SphereFit {
///     type Individual = Vec<f64>;
///     type Landscape = Sphere;
///     fn evaluate(&self, x: &Self::Individual, _: &Self::Landscape) -> f64 {
///         x.iter().map(|v| v * v).sum()
///     }
/// }
///
/// let device = Default::default();
/// let mut harness = EvolutionaryHarness::<Flex, _, _>::new(
///     GeneticAlgorithm::<Flex>::new(),
///     GaConfig::default_for(32, 5),
///     FromFitnessEvaluable::new(SphereFit, Sphere),
///     0, device, 100,
/// ).expect("valid params");
/// harness.reset();
/// while !harness.step(()).done {}
/// ```
///
/// Each [`step`](BenchEnv::step) runs one generation (ask → evaluate →
/// tell). The harness is the sole canonicaliser: it reads the fitness fn's
/// [`ObjectiveSense`](rlevo_core::objective::ObjectiveSense), negates a
/// `Minimize` objective into the engine's maximise space before `tell`, and
/// maps the metrics back to the declared sense for reporting. The reward
/// returned is the **canonical** `best_fitness_ever` directly (already
/// higher-is-better — no negation), so the per-episode cumulative return
/// (Σ step rewards) integrates the optimization trajectory. The harness only
/// exposes episode-level returns to reporters, so the "best at end" signal
/// would otherwise be lost.
///
/// # Determinism and parallel execution
///
/// Burn backends seed their tensor RNG through process-global state —
/// the `flex` backend uses a `Mutex<Option<FlexRng>>`, the
/// `wgpu` backend a per-device seeded stream. When multiple harness
/// instances run in parallel threads (e.g.
/// `Evaluator::run_suite` with the default rayon pool), their
/// interleaved `B::seed(...) → Tensor::random(...)` call pairs race on
/// that shared state and destroy bit-reproducibility across runs.
///
/// For deterministic reproduction, pass
/// `EvaluatorConfig::num_threads = Some(1)` or run one harness per
/// process. The `tests/determinism.rs` and `tests/rastrigin_run_suite.rs`
/// integration tests both enforce serial execution for this reason.
pub struct EvolutionaryHarness<B, S, F>
where
    B: Backend,
    S: Strategy<B>,
    F: BatchFitnessFn<B, S::Genome>,
{
    strategy: S,
    params: S::Params,
    fitness_fn: F,
    state: Option<S::State>,
    rng: StdRng,
    base_seed: u64,
    device: B::Device,
    generation: usize,
    max_generations: usize,
    latest_metrics: Option<StrategyMetrics>,
    observer: Option<SharedPopulationObserver>,
    _backend: PhantomData<B>,
}

impl<B, S, F> Debug for EvolutionaryHarness<B, S, F>
where
    B: Backend,
    S: Strategy<B>,
    F: BatchFitnessFn<B, S::Genome>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EvolutionaryHarness")
            .field("base_seed", &self.base_seed)
            .field("generation", &self.generation)
            .field("max_generations", &self.max_generations)
            .field("latest_metrics", &self.latest_metrics)
            .finish_non_exhaustive()
    }
}

impl<B, S, F> EvolutionaryHarness<B, S, F>
where
    B: Backend,
    S: Strategy<B>,
    F: BatchFitnessFn<B, S::Genome>,
{
    /// Build a new harness from its parts.
    ///
    /// The caller-supplied `params` are validated up front — this is the
    /// harness consumption chokepoint (ADR 0026), so an invalid configuration
    /// is rejected here rather than surfacing as a panic deep inside a
    /// strategy's tensor code.
    ///
    /// The harness is lazily initialized — the first [`reset`](BenchEnv::reset)
    /// call materializes the initial state on the supplied device.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] when `params` fails [`Validate::validate`],
    /// naming the offending field and violated invariant.
    pub fn new(
        strategy: S,
        params: S::Params,
        fitness_fn: F,
        seed: u64,
        device: B::Device,
        max_generations: usize,
    ) -> Result<Self, ConfigError>
    where
        S::Params: Validate,
    {
        params.validate()?;
        Ok(Self {
            strategy,
            params,
            fitness_fn,
            state: None,
            rng: StdRng::seed_from_u64(seed),
            base_seed: seed,
            device,
            generation: 0,
            max_generations,
            latest_metrics: None,
            observer: None,
            _backend: PhantomData,
        })
    }

    /// Attach a per-generation [`PopulationObserver`].
    ///
    /// The observer is called once per [`step`](Self::step) call, after the
    /// canonical `tracing::info!("evolution generation", …)` event. It
    /// receives a [`PopulationSnapshot`]
    /// carrying the full per-individual fitness vector for the completed
    /// generation. The intended consumer is a benchmark-tier recording sink
    /// that persists population-level data alongside the scalar metric stream.
    ///
    /// Attaching an observer adds one device→host transfer of the fitness
    /// tensor per generation; runs without an observer pay nothing.
    ///
    /// [`PopulationObserver`]: crate::observer::PopulationObserver
    #[must_use]
    pub fn with_observer(mut self, observer: SharedPopulationObserver) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Snapshot of the most recent generation's metrics, if any.
    #[must_use]
    pub fn latest_metrics(&self) -> Option<&StrategyMetrics> {
        self.latest_metrics.as_ref()
    }

    /// Generation counter — number of completed `tell` calls.
    #[must_use]
    pub fn generation(&self) -> usize {
        self.generation
    }

    /// Borrow the current strategy state if it exists.
    #[must_use]
    pub fn state(&self) -> Option<&S::State> {
        self.state.as_ref()
    }

    /// Forward to [`Strategy::best`] when a state exists.
    ///
    /// The strategy tracks the best genome in **canonical (maximise)** space;
    /// the returned fitness is mapped back to the objective's declared sense so
    /// a `Minimize` landscape reads as its natural cost.
    pub fn best(&self) -> Option<(S::Genome, f32)> {
        let sense = self.fitness_fn.sense();
        self.state
            .as_ref()
            .and_then(|s| self.strategy.best(s))
            .map(|(genome, canonical)| (genome, sense.from_canonical(canonical)))
    }

    /// Reset to a fresh initial state.
    ///
    /// Inherent shape (infallible): `EvolutionaryHarness` cannot legitimately
    /// fail to reset — it is a deterministic optimization driver. The
    /// [`BenchEnv`] trait impl wraps this in `Ok(())` so the harness is
    /// callable both directly (this method) and via the [`BenchEnv`] surface
    /// when fed to `Evaluator::run_suite`.
    pub fn reset(&mut self) {
        self.rng = StdRng::seed_from_u64(self.base_seed);
        self.generation = 0;
        self.latest_metrics = None;
        self.state = Some(
            self.strategy
                .init(&self.params, &mut self.rng, &self.device),
        );
    }

    /// Run one ask → evaluate → tell generation.
    ///
    /// Inherent shape (infallible). The [`BenchEnv`] trait impl wraps this
    /// in `Ok(...)`. See [`Self::reset`] for the rationale.
    ///
    /// # Panics
    ///
    /// Panics if [`reset`](Self::reset) has not been called first.
    pub fn step(&mut self, _action: ()) -> BenchStep<()> {
        let state = self
            .state
            .take()
            .expect("EvolutionaryHarness::reset must be called before step");
        let (population, state) =
            self.strategy
                .ask(&self.params, &state, &mut self.rng, &self.device);
        // The fitness function reports NATURAL values; the harness is the sole
        // canonicaliser. `sense` is the single source of truth (read off the
        // fitness fn, so the ctor and the adapter can never disagree).
        let sense = self.fitness_fn.sense();
        let fitness_natural = self.fitness_fn.evaluate_batch(&population, &self.device);
        // Mirror the NATURAL fitness tensor to host only if someone's actually
        // listening — the device→host transfer is the expensive part. The
        // observer records natural (user-sense) per-individual fitness.
        let snapshot_fitness: Option<Vec<f32>> = self.observer.as_ref().map(|_| {
            fitness_natural
                .clone()
                .into_data()
                .into_vec::<f32>()
                .unwrap_or_default()
        });
        // Canonicalise into the engine's maximise-native space: a `Minimize`
        // objective is negated (one device op), a `Maximize` one passes through.
        let fitness_canon = match sense {
            ObjectiveSense::Maximize => fitness_natural,
            ObjectiveSense::Minimize => fitness_natural.neg(),
        };
        let (new_state, metrics_canon) =
            self.strategy
                .tell(&self.params, population, fitness_canon, state, &mut self.rng);
        self.state = Some(new_state);
        self.generation += 1;
        // The reward is the canonical `best_fitness_ever` directly — canonical
        // space is already higher-is-better, so the old `-best_fitness_ever`
        // negation is gone. It stays monotone non-decreasing over a run, so the
        // cumulative return (Σ step reward) integrates the optimization
        // trajectory under the best-so-far curve. The benchmark harness reads
        // per-episode `return_value`, not per-step rewards, so a pure "last
        // best" signal would be lost.
        let reward = f64::from(metrics_canon.best_fitness_ever);
        // Map the canonical metrics back into the objective's declared sense so
        // every surfaced value (tracing, `latest_metrics`, records) reads in
        // user space — a `Minimize` landscape's `best_fitness` is its natural
        // cost (Sphere → 0).
        let metrics = StrategyMetrics {
            generation: metrics_canon.generation,
            population_size: metrics_canon.population_size,
            best_fitness: sense.from_canonical(metrics_canon.best_fitness),
            mean_fitness: sense.from_canonical(metrics_canon.mean_fitness),
            worst_fitness: sense.from_canonical(metrics_canon.worst_fitness),
            best_fitness_ever: sense.from_canonical(metrics_canon.best_fitness_ever),
        };
        // Structured per-generation event. Picked up by the
        // canonical-metric registry in
        // `rlevo-benchmarks::tui::log_layer::CANONICAL_METRICS` so the
        // live TUI's fitness sparkline lights up without coupling this
        // crate to the dashboard. Field names match the registry
        // verbatim; renaming any of them requires a paired update on
        // the benchmarks side.
        tracing::info!(
            generation = metrics.generation,
            population_size = metrics.population_size,
            best_fitness = f64::from(metrics.best_fitness),
            mean_fitness = f64::from(metrics.mean_fitness),
            worst_fitness = f64::from(metrics.worst_fitness),
            best_fitness_ever = f64::from(metrics.best_fitness_ever),
            "evolution generation",
        );
        if let (Some(observer), Some(fitnesses)) = (self.observer.as_ref(), snapshot_fitness) {
            let generation = u32::try_from(metrics.generation).unwrap_or(u32::MAX);
            match build_population_snapshot(generation, fitnesses, sense) {
                Some(snapshot) => {
                    // Isolate the observer: a panicking third-party sink drops
                    // this snapshot but must not abort an otherwise-healthy
                    // optimization run. `SharedPopulationObserver` is backed by a
                    // `parking_lot::Mutex` (no poisoning), so the guard drops
                    // during unwind and the next generation re-locks cleanly.
                    let dispatched = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        observer.lock().on_population(snapshot);
                    }));
                    if dispatched.is_err() {
                        tracing::warn!(
                            generation,
                            "population observer panicked; dropping snapshot and continuing",
                        );
                    }
                }
                None => {
                    // An empty fitness vector means the device→host transfer at
                    // `snapshot_fitness` yielded nothing (a masked conversion
                    // failure). Surface it rather than emitting an out-of-range
                    // `best_index` into a zero-length vector.
                    tracing::warn!(
                        generation,
                        "empty population fitness vector; skipping observer snapshot \
                         (device→host transfer likely failed)",
                    );
                }
            }
        }
        self.latest_metrics = Some(metrics);
        let done = self.generation >= self.max_generations;
        BenchStep {
            observation: (),
            reward,
            done,
        }
    }
}

impl<B, S, F> BenchEnv for EvolutionaryHarness<B, S, F>
where
    B: Backend,
    S: Strategy<B>,
    F: BatchFitnessFn<B, S::Genome>,
{
    type Observation = ();
    type Action = ();

    fn reset(&mut self) -> Result<Self::Observation, BenchError> {
        EvolutionaryHarness::<B, S, F>::reset(self);
        Ok(())
    }

    fn step(&mut self, action: Self::Action) -> Result<BenchStep<Self::Observation>, BenchError> {
        Ok(EvolutionaryHarness::<B, S, F>::step(self, action))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;
    use burn::tensor::TensorData;
    type TestBackend = Flex;

    /// Trivial strategy for unit-testing the harness plumbing: it
    /// ignores `ask`/`tell` semantics and always reports the same best
    /// fitness. Nothing here exercises real evolutionary dynamics.
    #[derive(Debug, Clone, Copy)]
    struct Constant;

    #[derive(Debug, Clone)]
    struct Params {
        pop_size: usize,
        dim: usize,
    }

    impl Validate for Params {
        fn validate(&self) -> Result<(), ConfigError> {
            rlevo_core::config::nonzero("Params", "pop_size", self.pop_size)?;
            rlevo_core::config::nonzero("Params", "dim", self.dim)?;
            Ok(())
        }
    }

    #[derive(Debug, Clone)]
    struct State {
        generation: usize,
        best: f32,
    }

    impl Strategy<TestBackend> for Constant {
        type Params = Params;
        type State = State;
        type Genome = Tensor<TestBackend, 2>;

        fn init(
            &self,
            params: &Params,
            _: &mut dyn Rng,
            device: &<TestBackend as burn::tensor::backend::BackendTypes>::Device,
        ) -> State {
            let _ = device;
            let _ = params;
            State {
                generation: 0,
                best: f32::NEG_INFINITY,
            }
        }

        fn ask(
            &self,
            params: &Params,
            state: &State,
            _: &mut dyn Rng,
            device: &<TestBackend as burn::tensor::backend::BackendTypes>::Device,
        ) -> (Tensor<TestBackend, 2>, State) {
            let data = TensorData::new(
                vec![0.0f32; params.pop_size * params.dim],
                [params.pop_size, params.dim],
            );
            let pop = Tensor::<TestBackend, 2>::from_data(data, device);
            (pop, state.clone())
        }

        fn tell(
            &self,
            _: &Params,
            _: Tensor<TestBackend, 2>,
            fitness: Tensor<TestBackend, 1>,
            mut state: State,
            _: &mut dyn Rng,
        ) -> (State, StrategyMetrics) {
            let values = fitness.into_data().into_vec::<f32>().unwrap();
            state.generation += 1;
            let metrics = StrategyMetrics::from_host_fitness(state.generation, &values, state.best);
            state.best = metrics.best_fitness_ever();
            (state, metrics)
        }

        fn best(&self, _state: &State) -> Option<(Tensor<TestBackend, 2>, f32)> {
            None
        }
    }

    /// Constant fitness = 42 regardless of input.
    struct FortyTwo;
    impl<B: Backend> BatchFitnessFn<B, Tensor<B, 2>> for FortyTwo {
        fn evaluate_batch(
            &mut self,
            population: &Tensor<B, 2>,
            device: &<B as burn::tensor::backend::BackendTypes>::Device,
        ) -> Tensor<B, 1> {
            let n = population.dims()[0];
            let data = TensorData::new(vec![42.0f32; n], [n]);
            Tensor::<B, 1>::from_data(data, device)
        }

        fn sense(&self) -> ObjectiveSense {
            // Treated as a cost so the harness reports natural 42 and reward
            // stays the canonical −42 the existing assertions expect.
            ObjectiveSense::Minimize
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn harness_runs_one_generation() {
        let device = Default::default();
        let strategy = Constant;
        let params = Params {
            pop_size: 4,
            dim: 3,
        };
        let mut harness =
            EvolutionaryHarness::<TestBackend, _, _>::new(strategy, params, FortyTwo, 1, device, 5).expect("valid params");
        harness.reset();
        let step = harness.step(());
        assert_eq!(step.reward, -42.0);
        assert!(!step.done);
        assert_eq!(harness.generation(), 1);
        let m = harness.latest_metrics().unwrap();
        assert_eq!(m.generation, 1);
        assert_eq!(m.population_size, 4);
        approx::assert_relative_eq!(m.best_fitness, 42.0, epsilon = 1e-6);
    }

    #[test]
    fn harness_reports_done_after_budget() {
        let device = Default::default();
        let mut harness = EvolutionaryHarness::<TestBackend, _, _>::new(
            Constant,
            Params {
                pop_size: 2,
                dim: 2,
            },
            FortyTwo,
            1,
            device,
            2,
        ).expect("valid params");
        harness.reset();
        assert!(!harness.step(()).done);
        assert!(harness.step(()).done);
    }

    #[test]
    fn from_host_fitness_computes_stats() {
        let m = StrategyMetrics::from_host_fitness(5, &[3.0, 1.0, 5.0, 2.0], 4.0);
        // Read through the public accessors (fields are private).
        assert_eq!(m.generation(), 5);
        assert_eq!(m.population_size(), 4);
        // Canonical maximise: best is the largest, worst the smallest.
        approx::assert_relative_eq!(m.best_fitness(), 5.0, epsilon = 1e-6);
        approx::assert_relative_eq!(m.worst_fitness(), 1.0, epsilon = 1e-6);
        approx::assert_relative_eq!(m.mean_fitness(), 2.75, epsilon = 1e-6);
        // best_fitness_ever = max(prior=4.0, current=5.0)
        approx::assert_relative_eq!(m.best_fitness_ever(), 5.0, epsilon = 1e-6);
    }

    #[test]
    fn from_host_fitness_sanitizes_nan() {
        // A NaN is sanitized to −∞ (worst under maximise) *consistently*: it
        // never becomes best, and it drags worst and mean to −∞ rather than the
        // old asymmetry where best/worst ignored it but mean turned NaN.
        let m = StrategyMetrics::from_host_fitness(0, &[1.0, f32::NAN, 3.0, 2.0], 0.0);
        approx::assert_relative_eq!(m.best_fitness(), 3.0, epsilon = 1e-6);
        assert!(m.worst_fitness().is_infinite() && m.worst_fitness().is_sign_negative());
        assert!(m.mean_fitness().is_infinite() && m.mean_fitness().is_sign_negative());
        approx::assert_relative_eq!(m.best_fitness_ever(), 3.0, epsilon = 1e-6);
    }

    #[test]
    fn build_population_snapshot_empty_returns_none() {
        assert!(build_population_snapshot(0, Vec::new(), ObjectiveSense::Minimize).is_none());
    }

    #[test]
    fn build_population_snapshot_picks_best_for_sense() {
        // Values: [0.3, 0.1, 0.9]. Minimize → best is the smallest (index 1);
        // Maximize → best is the largest (index 2).
        let min = build_population_snapshot(7, vec![0.3, 0.1, 0.9], ObjectiveSense::Minimize)
            .expect("non-empty");
        assert_eq!(min.best_index, 1);
        assert_eq!(min.generation, 7);
        let max = build_population_snapshot(7, vec![0.3, 0.1, 0.9], ObjectiveSense::Maximize)
            .expect("non-empty");
        assert_eq!(max.best_index, 2);
    }

    /// Per-individual fitness = `1.0 / (i + 1)` so the best (smallest)
    /// is always at index `pop_size - 1` — a deterministic shape the
    /// observer test can pin against.
    struct RankedFitness;
    impl<B: Backend> BatchFitnessFn<B, Tensor<B, 2>> for RankedFitness {
        fn evaluate_batch(
            &mut self,
            population: &Tensor<B, 2>,
            device: &<B as burn::tensor::backend::BackendTypes>::Device,
        ) -> Tensor<B, 1> {
            let n = population.dims()[0];
            #[allow(clippy::cast_precision_loss)]
            let values: Vec<f32> = (0..n).map(|i| 1.0 / (i as f32 + 1.0)).collect();
            let data = TensorData::new(values, [n]);
            Tensor::<B, 1>::from_data(data, device)
        }

        fn sense(&self) -> ObjectiveSense {
            // Cost: the best (smallest) is the last index, which the observer
            // test pins via the sense-aware `best_index`.
            ObjectiveSense::Minimize
        }
    }

    #[derive(Debug, Default)]
    struct CountingObserver {
        snapshots: Vec<PopulationSnapshot>,
    }

    impl crate::observer::PopulationObserver for CountingObserver {
        fn on_population(&mut self, snapshot: PopulationSnapshot) {
            self.snapshots.push(snapshot);
        }
    }

    #[test]
    fn harness_fires_observer_per_generation() {
        use std::sync::Arc;

        use parking_lot::Mutex;
        let device = Default::default();
        let observer = Arc::new(Mutex::new(CountingObserver::default()));
        let mut harness = EvolutionaryHarness::<TestBackend, _, _>::new(
            Constant,
            Params {
                pop_size: 5,
                dim: 2,
            },
            RankedFitness,
            1,
            device,
            3,
        ).expect("valid params")
        .with_observer(observer.clone() as SharedPopulationObserver);
        harness.reset();
        for _ in 0..3 {
            harness.step(());
        }
        let guard = observer.lock();
        assert_eq!(guard.snapshots.len(), 3);
        // pop_size = 5, ranked fitness = [1/1, 1/2, 1/3, 1/4, 1/5]; best
        // (smallest) is the last element.
        assert_eq!(guard.snapshots[0].fitnesses.len(), 5);
        assert_eq!(guard.snapshots[0].best_index, 4);
        assert_eq!(guard.snapshots[2].generation, 3);
        // M8.1 leaves these fields empty / None — see observer.rs docs.
        assert!(guard.snapshots[0].diversity.is_none());
        assert!(guard.snapshots[0].best_genome_digest.is_none());
        assert!(guard.snapshots[0].parents_of_best.is_empty());
    }

    /// Observer whose callback always panics — used to prove the harness
    /// isolates a misbehaving sink instead of aborting the run.
    #[derive(Debug, Default)]
    struct PanicObserver;

    impl crate::observer::PopulationObserver for PanicObserver {
        fn on_population(&mut self, _snapshot: PopulationSnapshot) {
            panic!("observer intentionally panics");
        }
    }

    #[test]
    fn harness_survives_panicking_observer() {
        use std::sync::Arc;

        use parking_lot::Mutex;
        let device = Default::default();
        let observer = Arc::new(Mutex::new(PanicObserver));
        let mut harness = EvolutionaryHarness::<TestBackend, _, _>::new(
            Constant,
            Params {
                pop_size: 4,
                dim: 2,
            },
            RankedFitness,
            1,
            device,
            2,
        )
        .expect("valid params")
        .with_observer(observer.clone() as SharedPopulationObserver);
        harness.reset();
        // Each step's observer dispatch panics; the harness must swallow it and
        // keep advancing generations to completion.
        assert!(!harness.step(()).done);
        assert!(harness.step(()).done);
        assert_eq!(harness.generation(), 2);
    }

    #[test]
    fn harness_without_observer_skips_host_transfer() {
        // Smoke: no observer attached → step() still works, no panic,
        // no transfer cost. Observability is verified above; here we
        // just want the no-observer path to remain functional.
        let device = Default::default();
        let mut harness = EvolutionaryHarness::<TestBackend, _, _>::new(
            Constant,
            Params {
                pop_size: 3,
                dim: 1,
            },
            RankedFitness,
            1,
            device,
            1,
        ).expect("valid params");
        harness.reset();
        let step = harness.step(());
        assert!(step.done);
    }
}
