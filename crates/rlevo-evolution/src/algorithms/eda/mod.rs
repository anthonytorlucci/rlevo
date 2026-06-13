//! Estimation-of-distribution algorithms (EDAs).
//!
//! An EDA replaces the crossover and mutation operators of a classical GA
//! with an explicit probabilistic model of the promising region of search
//! space. Each generation runs a `fit` → `sample` loop:
//!
//! 1. Evaluate the current population (done externally by the harness).
//! 2. Truncation-select the best fraction of the population.
//! 3. [`fit`](crate::ProbabilityModel::fit) the model to those survivors.
//! 4. [`sample`](crate::ProbabilityModel::sample) a fresh population.
//!
//! The model is supplied as a [`ProbabilityModel`]; the generic
//! [`EdaStrategy`] driver is model-agnostic. Five reference models ship here:
//!
//! - [`UnivariateGaussian`] — UMDA, a per-dimension Gaussian (unweighted MLE,
//!   `÷k` variance, `min_variance` floor; fitness is accepted but ignored).
//! - [`UnivariateBernoulli`] — PBIL, a per-bit probability vector (no
//!   classic probability-mutation step; fitness is used only to identify
//!   the best/worst individual).
//! - [`CompactGenetic`] — cGA, a virtual-population probability vector
//!   (winner/loser come from the truncation-selected subset, not a fresh
//!   pairwise draw as in classic cGA).
//! - [`DependencyChain`] — a continuous-Gaussian MIMIC chain capturing
//!   pairwise dependencies (fitness accepted but ignored).
//! - [`BayesianNetwork`] — BOA, a BIC-scored Bayesian network (bounded-in-degree
//!   DAG over binary genes; non-incremental, unweighted fit; Pelikan, Goldberg
//!   & Cantú-Paz 1999).
//!
//! The three binary models ([`UnivariateBernoulli`], [`CompactGenetic`], and
//! [`BayesianNetwork`]) emit raw `{0, 1}` genes; the [`EdaParams::bounds`] clamp
//! is therefore a no-op for them.
//!
//! # References
//!
//! - Mühlenbein & Paaß (1996), *From recombination of genes to the
//!   estimation of distributions I. Binary parameters*.
//! - Baluja (1994), *Population-based incremental learning: a method for
//!   integrating genetic search based function optimization and competitive
//!   learning*.
//! - Harik, Lobo & Goldberg (1999), *The compact genetic algorithm*.
//! - De Bonet, Isbell & Viola (1997), *MIMIC: Finding optima by estimating
//!   probability densities*.
//! - Pelikan, Goldberg & Cantú-Paz (1999), *BOA: The Bayesian optimization
//!   algorithm*.

pub mod bayesian_network;
pub mod compact_genetic;
pub mod dependency_chain;
pub mod univariate_bernoulli;
pub mod univariate_gaussian;

pub use bayesian_network::{BayesianNetwork, BayesianNetworkParams, BayesianNetworkState};
pub use compact_genetic::{CompactGenetic, CompactGeneticParams, CompactGeneticState};
pub use dependency_chain::{DependencyChain, DependencyChainParams, DependencyChainState};
pub use univariate_bernoulli::{
    UnivariateBernoulli, UnivariateBernoulliParams, UnivariateBernoulliState,
};
pub use univariate_gaussian::{UnivariateGaussian, UnivariateGaussianParams, UnivariateGaussianState};

use std::fmt::Debug;
use std::marker::PhantomData;

use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use rand::Rng;

use crate::probability_model::ProbabilityModel;
use crate::rng::{SeedPurpose, seed_stream};
use crate::strategy::{Strategy, StrategyMetrics};

/// Static configuration for an [`EdaStrategy`] run.
///
/// All fields are public so callers can use struct-literal construction or
/// update syntax. The only runtime contract enforced at call time is that
/// `selection_ratio` lies strictly in `(0, 1)` (checked by a
/// `debug_assert!` in [`EdaStrategy::init`]).
#[derive(Debug, Clone)]
pub struct EdaParams<MP> {
    /// Number of individuals sampled per generation.
    pub pop_size: usize,
    /// Fraction of the population kept by truncation selection; must lie
    /// strictly in `(0, 1)`. The effective `k` is
    /// `ceil(selection_ratio · pop_size)` clamped to `[2, pop_size]`.
    pub selection_ratio: f32,
    /// Optional inclusive `[lo, hi]` clamp applied to each gene after
    /// sampling. A no-op for binary models ([`UnivariateBernoulli`],
    /// [`CompactGenetic`]) whose genes are already `{0, 1}`.
    pub bounds: Option<(f32, f32)>,
    /// Model-specific parameters (includes `genome_dim`).
    pub model: MP,
}

/// Generation-to-generation state carried by [`EdaStrategy`].
///
/// The driver updates this in `tell` and passes it back through `ask` unchanged.
/// Callers should treat it as an opaque blob; the public fields are exposed to
/// enable checkpointing and observability, not mutation.
#[derive(Debug, Clone)]
pub struct EdaState<B: Backend, MS> {
    /// Current fitted model state (updated once per [`EdaStrategy::tell`] call).
    pub model_state: MS,
    /// Best genome ever observed, shape `(genome_dim,)`. `None` before the
    /// first [`EdaStrategy::tell`] call.
    pub best_genome: Option<Tensor<B, 1>>,
    /// Smallest (best) fitness ever observed across all generations
    /// (minimization convention). Starts at `f32::INFINITY`.
    pub best_fitness_ever: f32,
    /// Number of completed generations (incremented by each `tell` call).
    pub generation: usize,
}

/// Generic estimation-of-distribution strategy.
///
/// Drives the `fit` → `sample` loop of any [`ProbabilityModel`]. The strategy
/// itself is stateless (the model is held by value; all mutable generation
/// state lives in the returned [`EdaState`]), so it slots straight into
/// [`EvolutionaryHarness`](crate::strategy::EvolutionaryHarness).
///
/// Type parameter `M` must implement [`ProbabilityModel<B>`]; in practice
/// this is one of [`UnivariateGaussian`], [`UnivariateBernoulli`],
/// [`CompactGenetic`], [`DependencyChain`], or [`BayesianNetwork`].
///
/// # Example
///
/// ```no_run
/// use burn::backend::Flex;
/// use rlevo_evolution::algorithms::eda::{EdaParams, EdaStrategy, UnivariateGaussian};
/// use rlevo_evolution::algorithms::eda::univariate_gaussian::UnivariateGaussianParams;
///
/// let strategy = EdaStrategy::<Flex, _>::new(UnivariateGaussian);
/// let params = EdaParams {
///     pop_size: 32,
///     selection_ratio: 0.5,
///     bounds: Some((-5.12, 5.12)),
///     model: UnivariateGaussianParams::default_for(8),
/// };
/// let _ = (strategy, params);
/// ```
pub struct EdaStrategy<B: Backend, M> {
    model: M,
    _backend: PhantomData<fn() -> B>,
}

impl<B: Backend, M: Debug> Debug for EdaStrategy<B, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EdaStrategy")
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl<B: Backend, M> EdaStrategy<B, M> {
    /// Build a new EDA strategy around `model`.
    #[must_use]
    pub fn new(model: M) -> Self {
        Self {
            model,
            _backend: PhantomData,
        }
    }
}

impl<B: Backend, M: ProbabilityModel<B>> Strategy<B> for EdaStrategy<B, M> {
    type Params = EdaParams<M::Params>;
    type State = EdaState<B, M::State>;
    type Genome = Tensor<B, 2>;

    /// Build the initial state.
    ///
    /// Fits the model's prior from `params.model` (passing `prev = None` and
    /// empty `0 × 0` / length-`0` population and fitness tensors, per the
    /// [`ProbabilityModel`] invariants). The `rng` is unused — the prior is
    /// deterministic. The best-so-far trackers start empty.
    ///
    /// # Panics
    ///
    /// Debug-asserts that `params.selection_ratio` lies strictly in `(0, 1)`.
    /// A ratio of `0.0` would select no parents and a ratio of `1.0` would
    /// defeat truncation selection entirely.
    fn init(
        &self,
        params: &Self::Params,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Self::State {
        debug_assert!(
            params.selection_ratio > 0.0 && params.selection_ratio < 1.0,
            "selection_ratio must lie strictly in (0, 1), got {}",
            params.selection_ratio
        );
        let _ = rng;
        let model_state = self.model.fit(
            &params.model,
            None,
            Tensor::empty([0, 0], device),
            Tensor::empty([0], device),
            device,
        );
        EdaState {
            model_state,
            best_genome: None,
            best_fitness_ever: f32::INFINITY,
            generation: 0,
        }
    }

    /// Sample the next candidate population from the current model.
    ///
    /// Draws exactly one `u64` from `rng` to seed a per-generation
    /// [`SeedPurpose::EdaSampling`] stream, samples `params.pop_size`
    /// individuals from the model through that host stream, and applies the
    /// optional `params.bounds` clamp. The state is returned unchanged.
    fn ask(
        &self,
        params: &Self::Params,
        state: &Self::State,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> (Self::Genome, Self::State) {
        let draw = rng.next_u64();
        let mut stream = seed_stream(draw, state.generation as u64, SeedPurpose::EdaSampling);
        let mut pop = self
            .model
            .sample(&state.model_state, params.pop_size, &mut stream, device);
        if let Some((lo, hi)) = params.bounds {
            pop = pop.clamp(lo, hi);
        }
        (pop, state.clone())
    }

    /// Consume the population's fitness and refit the model.
    ///
    /// Pulls fitness to host, sanitizes `NaN` → `+inf` via the crate's
    /// `sanitize_fitness` helper, updates the best-so-far tracker,
    /// truncation-selects the best `k` rows (ascending fitness order, with
    /// `k = ceil(selection_ratio · pop_size)` clamped to `[2, pop_size]`),
    /// and refits the model to them (passing `prev = Some(model_state)`).
    ///
    /// The `fitness` tensor is forwarded to [`ProbabilityModel::fit`]; models
    /// that weight or rank their selected individuals can use it. The five
    /// built-in models ([`UnivariateGaussian`], [`UnivariateBernoulli`],
    /// [`CompactGenetic`], [`DependencyChain`], [`BayesianNetwork`]) all perform
    /// an unweighted fit and ignore it.
    fn tell(
        &self,
        params: &Self::Params,
        population: Self::Genome,
        fitness: Tensor<B, 1>,
        mut state: Self::State,
        _rng: &mut dyn Rng,
    ) -> (Self::State, StrategyMetrics) {
        let raw = fitness.into_data().into_vec::<f32>().unwrap_or_default();
        let sanitized: Vec<f32> = raw
            .iter()
            .map(|&f| crate::local_search::sanitize_fitness(f))
            .collect();
        let n = sanitized.len();
        let device = population.device();

        // Argmin (ties → lowest index) for the best-so-far update.
        let mut best_idx = 0_usize;
        let mut best_f = f32::INFINITY;
        for (i, &f) in sanitized.iter().enumerate() {
            if f.total_cmp(&best_f) == std::cmp::Ordering::Less {
                best_f = f;
                best_idx = i;
            }
        }
        if best_f < state.best_fitness_ever {
            // usize → i64 for the Burn Int index tensor; population indices are
            // far below i64::MAX so the cast never wraps.
            #[allow(clippy::cast_possible_wrap)]
            let idx = Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![best_idx as i64], [1]),
                &device,
            );
            let row = population.clone().select(0, idx).squeeze_dim::<1>(0);
            state.best_genome = Some(row);
        }

        // Truncation selection: keep the best `k` rows in ascending fitness
        // order so the model sees a deterministic, best-first population.
        #[allow(clippy::cast_precision_loss)]
        let target = (params.selection_ratio * params.pop_size as f32).ceil();
        // ceil of a finite non-negative product → small usize; never truncates
        // meaningfully for any realistic population size.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let k = (target as usize).max(2).min(n.max(1));
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            sanitized[a]
                .total_cmp(&sanitized[b])
                .then(a.cmp(&b))
        });
        order.truncate(k);

        // usize → i64 index tensor (see best-so-far note above).
        #[allow(clippy::cast_possible_wrap)]
        let idx_vec: Vec<i64> = order.iter().map(|&i| i as i64).collect();
        let idx = Tensor::<B, 1, Int>::from_data(TensorData::new(idx_vec, [k]), &device);
        let selected = population.clone().select(0, idx);
        let selected_fitness_host: Vec<f32> = order.iter().map(|&i| sanitized[i]).collect();
        let selected_fitness =
            Tensor::<B, 1>::from_data(TensorData::new(selected_fitness_host, [k]), &device);

        let model_state = self.model.fit(
            &params.model,
            Some(&state.model_state),
            selected,
            selected_fitness,
            &device,
        );
        state.model_state = model_state;
        state.generation += 1;
        let metrics =
            StrategyMetrics::from_host_fitness(state.generation, &sanitized, state.best_fitness_ever);
        state.best_fitness_ever = metrics.best_fitness_ever;
        (state, metrics)
    }

    /// Return the best-so-far genome (shape `(1, D)`) and its fitness.
    ///
    /// Returns `None` before the first [`tell`](Self::tell) call. The genome is
    /// stored internally as a `(D,)` vector and unsqueezed to `(1, D)` here to
    /// match the `Genome = Tensor<B, 2>` container.
    fn best(&self, state: &Self::State) -> Option<(Self::Genome, f32)> {
        state
            .best_genome
            .as_ref()
            .map(|g| (g.clone().unsqueeze::<2>(), state.best_fitness_ever))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    type TestBackend = Flex;

    fn make_pop(rows: &[f32], n: usize, d: usize) -> Tensor<TestBackend, 2> {
        let device = Default::default();
        Tensor::<TestBackend, 2>::from_data(TensorData::new(rows.to_vec(), [n, d]), &device)
    }

    fn make_fitness(values: &[f32]) -> Tensor<TestBackend, 1> {
        let device = Default::default();
        let n = values.len();
        Tensor::<TestBackend, 1>::from_data(TensorData::new(values.to_vec(), [n]), &device)
    }

    fn params(pop_size: usize, ratio: f32, dim: usize) -> EdaParams<UnivariateGaussianParams> {
        EdaParams {
            pop_size,
            selection_ratio: ratio,
            bounds: None,
            model: UnivariateGaussianParams::default_for(dim),
        }
    }

    #[test]
    fn best_is_none_before_tell_some_after() {
        let device = Default::default();
        let strategy = EdaStrategy::<TestBackend, _>::new(UnivariateGaussian);
        let p = params(4, 0.5, 2);
        let mut rng = StdRng::seed_from_u64(0);
        let state = strategy.init(&p, &mut rng, &device);
        assert!(strategy.best(&state).is_none());

        let pop = make_pop(&[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0], 4, 2);
        let fitness = make_fitness(&[5.0, 1.0, 9.0, 7.0]);
        let (state, _m) = strategy.tell(&p, pop, fitness, state, &mut rng);
        let best = strategy.best(&state);
        assert!(best.is_some());
        let (genome, f) = best.unwrap();
        assert_eq!(genome.dims(), [1, 2]);
        approx::assert_relative_eq!(f, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn k_is_clamped_to_at_least_two_and_at_most_n() {
        let device = Default::default();
        let strategy = EdaStrategy::<TestBackend, _>::new(UnivariateGaussian);
        let mut rng = StdRng::seed_from_u64(0);

        // ratio 0.1 of 5 = ceil(0.5) = 1, clamped up to 2.
        let p = params(5, 0.1, 1);
        let state = strategy.init(&p, &mut rng, &device);
        let pop = make_pop(&[5.0, 4.0, 3.0, 2.0, 1.0], 5, 1);
        let fitness = make_fitness(&[5.0, 4.0, 3.0, 2.0, 1.0]);
        // Refit must not panic with k clamped to 2.
        let (state, _m) = strategy.tell(&p, pop, fitness, state, &mut rng);
        assert_eq!(state.generation, 1);

        // ratio 0.99 of 3 = ceil(2.97) = 3, clamped to n = 3.
        let p2 = params(3, 0.99, 1);
        let state2 = strategy.init(&p2, &mut rng, &device);
        let pop2 = make_pop(&[1.0, 2.0, 3.0], 3, 1);
        let fitness2 = make_fitness(&[1.0, 2.0, 3.0]);
        let (state2, _m) = strategy.tell(&p2, pop2, fitness2, state2, &mut rng);
        assert_eq!(state2.generation, 1);
    }

    #[test]
    fn tie_breaking_prefers_lowest_index() {
        let device = Default::default();
        let strategy = EdaStrategy::<TestBackend, _>::new(UnivariateGaussian);
        let mut rng = StdRng::seed_from_u64(0);
        let p = params(4, 0.5, 1);
        let state = strategy.init(&p, &mut rng, &device);
        // Two individuals tie for best fitness (1.0) at indices 1 and 3; the
        // best-so-far must be index 1 (genome value 10.0), not 30.0.
        let pop = make_pop(&[0.0, 10.0, 20.0, 30.0], 4, 1);
        let fitness = make_fitness(&[5.0, 1.0, 5.0, 1.0]);
        let (state, _m) = strategy.tell(&p, pop, fitness, state, &mut rng);
        let (genome, _f) = strategy.best(&state).unwrap();
        let v = genome.into_data().into_vec::<f32>().unwrap();
        approx::assert_relative_eq!(v[0], 10.0, epsilon = 1e-6);
    }

    #[test]
    fn nan_fitness_never_becomes_best_and_does_not_break_ordering() {
        let device = Default::default();
        let strategy = EdaStrategy::<TestBackend, _>::new(UnivariateGaussian);
        let mut rng = StdRng::seed_from_u64(0);
        let p = params(4, 0.5, 1);
        let state = strategy.init(&p, &mut rng, &device);
        // index 0 is NaN (→ +inf); the finite best (2.0) is at index 2.
        let pop = make_pop(&[0.0, 1.0, 2.0, 3.0], 4, 1);
        let fitness = make_fitness(&[f32::NAN, 9.0, 2.0, 7.0]);
        let (state, m) = strategy.tell(&p, pop, fitness, state, &mut rng);
        let (genome, f) = strategy.best(&state).unwrap();
        let v = genome.into_data().into_vec::<f32>().unwrap();
        approx::assert_relative_eq!(v[0], 2.0, epsilon = 1e-6);
        approx::assert_relative_eq!(f, 2.0, epsilon = 1e-6);
        assert!(m.best_fitness.is_finite());
    }

    #[test]
    fn bounds_clamp_is_applied_in_ask() {
        let device = Default::default();
        let strategy = EdaStrategy::<TestBackend, _>::new(UnivariateGaussian);
        let mut rng = StdRng::seed_from_u64(7);
        // Wide init std so unclamped draws routinely exceed the bounds.
        let p = EdaParams {
            pop_size: 64,
            selection_ratio: 0.5,
            bounds: Some((-0.5, 0.5)),
            model: UnivariateGaussianParams {
                genome_dim: 3,
                init_mean: 0.0,
                init_std: 5.0,
                min_variance: 1e-6,
            },
        };
        let state = strategy.init(&p, &mut rng, &device);
        let (pop, _s) = strategy.ask(&p, &state, &mut rng, &device);
        let values = pop.into_data().into_vec::<f32>().unwrap();
        for v in values {
            assert!((-0.5..=0.5).contains(&v), "value {v} escaped the clamp");
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "selection_ratio")]
    fn selection_ratio_zero_panics() {
        let device = Default::default();
        let strategy = EdaStrategy::<TestBackend, _>::new(UnivariateGaussian);
        let mut rng = StdRng::seed_from_u64(0);
        let p = params(4, 0.0, 1);
        let _ = strategy.init(&p, &mut rng, &device);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "selection_ratio")]
    fn selection_ratio_one_panics() {
        let device = Default::default();
        let strategy = EdaStrategy::<TestBackend, _>::new(UnivariateGaussian);
        let mut rng = StdRng::seed_from_u64(0);
        let p = params(4, 1.0, 1);
        let _ = strategy.init(&p, &mut rng, &device);
    }
}
