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
//! The fitness tensor passed to [`tell`](Strategy::tell) is the raw
//! objective value. Strategies in this crate minimize it: the
//! [`StrategyMetrics::best_fitness`] field is the smallest value observed
//! so far, and the harness reports `reward = -best_fitness` so the
//! benchmark harness's "higher = better" convention still holds.
//!
//! # The harness adapter
//!
//! [`EvolutionaryHarness`] glues a strategy to any
//! [`BatchFitnessFn`](crate::fitness::BatchFitnessFn) and implements
//! [`BenchEnv`](rlevo_core::evaluation::BenchEnv), so the benchmark
//! evaluator drives it just like an RL environment.

use std::fmt::Debug;
use std::marker::PhantomData;

use burn::tensor::{Tensor, backend::Backend};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use rlevo_core::evaluation::{BenchEnv, BenchError, BenchStep};

use crate::fitness::BatchFitnessFn;

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
/// ```no_run
/// use burn::backend::NdArray;
/// use rlevo_evolution::algorithms::ga::{GaConfig, GeneticAlgorithm};
/// use rlevo_evolution::Strategy;
/// use rand::{rngs::StdRng, SeedableRng};
///
/// let device = Default::default();
/// let strategy = GeneticAlgorithm::<NdArray>::new();
/// let params = GaConfig::default_for(64, 10);
/// let mut rng = StdRng::seed_from_u64(0);
/// let state = strategy.init(&params, &mut rng, &device);
/// assert_eq!(state.population.shape().dims, vec![64, 10]);
/// ```
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
    fn init(&self, params: &Self::Params, rng: &mut dyn Rng, device: &B::Device) -> Self::State;

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
        device: &B::Device,
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
    fn best(&self, state: &Self::State) -> Option<(Self::Genome, f32)>;
}

/// Per-generation summary reported by [`Strategy::tell`].
///
/// All statistics refer to the generation that just finished evaluating.
/// Fitness values follow the minimization convention: lower is better.
#[derive(Debug, Clone)]
pub struct StrategyMetrics {
    /// Zero-based generation index.
    pub generation: usize,
    /// Number of individuals evaluated in this generation.
    pub population_size: usize,
    /// Smallest fitness observed in this generation.
    pub best_fitness: f32,
    /// Mean fitness across this generation's population.
    pub mean_fitness: f32,
    /// Largest fitness observed in this generation.
    pub worst_fitness: f32,
    /// Best fitness seen across *all* generations to date.
    pub best_fitness_ever: f32,
}

impl StrategyMetrics {
    /// Computes population statistics from a host-side fitness slice.
    ///
    /// # Panics
    ///
    /// Panics if `fitnesses` is empty.
    #[must_use]
    pub fn from_host_fitness(generation: usize, fitnesses: &[f32], best_fitness_ever: f32) -> Self {
        assert!(!fitnesses.is_empty(), "fitness slice must be non-empty");
        let population_size = fitnesses.len();
        let (mut best, mut worst, mut sum) = (f32::INFINITY, f32::NEG_INFINITY, 0.0_f32);
        for &f in fitnesses {
            if f < best {
                best = f;
            }
            if f > worst {
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
            best_fitness_ever: best_fitness_ever.min(best),
        }
    }
}

/// Wraps a [`Strategy`] into a [`BenchEnv`] so the benchmark harness can
/// drive it.
///
/// # Example
///
/// ```no_run
/// use burn::backend::NdArray;
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
/// let mut harness = EvolutionaryHarness::<NdArray, _, _>::new(
///     GeneticAlgorithm::<NdArray>::new(),
///     GaConfig::default_for(32, 5),
///     FromFitnessEvaluable::new(SphereFit, Sphere),
///     0, device, 100,
/// );
/// harness.reset();
/// while !harness.step(()).done {}
/// ```
///
/// Each [`step`](BenchEnv::step) runs one generation (ask → evaluate →
/// tell). The reward returned to the harness is `-best_fitness_ever` so
/// the harness's "higher = better" convention matches the strategy's
/// minimization direction, and so the per-episode cumulative return
/// (Σ step rewards) integrates the optimization trajectory —
/// `return_value / num_steps` bounds the final `best_fitness_ever` from
/// above. The harness only exposes episode-level returns to reporters,
/// so the "best at end" signal would otherwise be lost.
///
/// # Determinism and parallel execution
///
/// Burn backends seed their tensor RNG through process-global state —
/// the `ndarray` backend uses a `Mutex<Option<NdArrayRng>>`, the
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
    /// The harness is lazily initialized — the first [`reset`](BenchEnv::reset)
    /// call materializes the initial state on the supplied device.
    pub fn new(
        strategy: S,
        params: S::Params,
        fitness_fn: F,
        seed: u64,
        device: B::Device,
        max_generations: usize,
    ) -> Self {
        Self {
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
            _backend: PhantomData,
        }
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
    pub fn best(&self) -> Option<(S::Genome, f32)> {
        self.state.as_ref().and_then(|s| self.strategy.best(s))
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
        let fitness = self.fitness_fn.evaluate_batch(&population, &self.device);
        let (new_state, metrics) =
            self.strategy
                .tell(&self.params, population, fitness, state, &mut self.rng);
        self.state = Some(new_state);
        self.generation += 1;
        // Emit `-best_fitness_ever` so the reward is monotone
        // non-decreasing over a run and the cumulative return (Σ step
        // reward) integrates the optimization trajectory under the
        // best-so-far curve. The benchmark harness reads per-episode
        // `return_value` not per-step rewards, so a pure "last best"
        // signal would be lost.
        let reward = -f64::from(metrics.best_fitness_ever);
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

    fn step(
        &mut self,
        action: Self::Action,
    ) -> Result<BenchStep<Self::Observation>, BenchError> {
        Ok(EvolutionaryHarness::<B, S, F>::step(self, action))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;
    type TestBackend = NdArray;

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
            device: &<TestBackend as Backend>::Device,
        ) -> State {
            let _ = device;
            let _ = params;
            State {
                generation: 0,
                best: f32::INFINITY,
            }
        }

        fn ask(
            &self,
            params: &Params,
            state: &State,
            _: &mut dyn Rng,
            device: &<TestBackend as Backend>::Device,
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
            state.best = metrics.best_fitness_ever;
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
            device: &B::Device,
        ) -> Tensor<B, 1> {
            let n = population.shape().dims[0];
            let data = TensorData::new(vec![42.0f32; n], [n]);
            Tensor::<B, 1>::from_data(data, device)
        }
    }

    #[test]
    fn harness_runs_one_generation() {
        let device = Default::default();
        let strategy = Constant;
        let params = Params {
            pop_size: 4,
            dim: 3,
        };
        let mut harness =
            EvolutionaryHarness::<TestBackend, _, _>::new(strategy, params, FortyTwo, 1, device, 5);
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
        );
        harness.reset();
        assert!(!harness.step(()).done);
        assert!(harness.step(()).done);
    }

    #[test]
    fn from_host_fitness_computes_stats() {
        let m = StrategyMetrics::from_host_fitness(5, &[3.0, 1.0, 5.0, 2.0], 4.0);
        assert_eq!(m.generation, 5);
        assert_eq!(m.population_size, 4);
        approx::assert_relative_eq!(m.best_fitness, 1.0, epsilon = 1e-6);
        approx::assert_relative_eq!(m.worst_fitness, 5.0, epsilon = 1e-6);
        approx::assert_relative_eq!(m.mean_fitness, 2.75, epsilon = 1e-6);
        // best_fitness_ever = min(prior=4.0, current=1.0)
        approx::assert_relative_eq!(m.best_fitness_ever, 1.0, epsilon = 1e-6);
    }
}
