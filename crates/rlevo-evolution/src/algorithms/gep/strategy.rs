//! The [`GepStrategy`] evolutionary engine and a symbolic-regression fitness.

use std::marker::PhantomData;

use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use rand::{Rng, RngExt};
use rayon::prelude::*;
use rlevo_core::config::{self, ConfigError};

use crate::fitness::BatchFitnessFn;
use crate::function_set::{FunctionSet, Symbol};
use crate::rng::{SeedPurpose, seed_stream};
use crate::strategy::{Strategy, StrategyMetrics};

use super::alphabet::Alphabet;
use super::config::GepConfig;
use super::decode::{GenotypePhenotypeMap, GepDecoder};
use super::operators::{
    is_transposition, one_point_crossover, point_mutation, ris_transposition, two_point_crossover,
};

/// Generation state for [`GepStrategy`].
///
/// Fields are private so the host-side fitness cache cannot fall out of sync
/// with the population from outside this module; build one with
/// [`try_new`](GepState::try_new) and read it through the accessors.
#[derive(Debug, Clone)]
pub struct GepState<B: Backend> {
    /// Current population, shape `(pop_size, genome_len)`, `i32` symbol ids.
    population: Tensor<B, 2, Int>,
    /// Host-side fitness cache for the current population, in canonical
    /// (maximise) space — *higher is better*. Empty until the first
    /// [`Strategy::tell`].
    fitnesses: Vec<f32>,
    /// Best-so-far genome, shape `(1, genome_len)`.
    best_genome: Option<Tensor<B, 2, Int>>,
    /// Best-so-far fitness.
    best_fitness: f32,
    /// Generation counter.
    generation: usize,
}

impl<B: Backend> GepState<B> {
    /// Assembles a GEP state, checking the fitness cache matches the
    /// population.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `population` has zero rows or if
    /// `fitnesses` is non-empty with a length other than `pop_size`. The
    /// bootstrap state (empty `fitnesses`) is accepted.
    pub fn try_new(
        population: Tensor<B, 2, Int>,
        fitnesses: Vec<f32>,
        best_genome: Option<Tensor<B, 2, Int>>,
        best_fitness: f32,
        generation: usize,
    ) -> Result<Self, ConfigError> {
        let pop = population.dims()[0];
        config::nonzero("GepState", "pop_size", pop)?;
        if !fitnesses.is_empty() && fitnesses.len() != pop {
            return Err(ConfigError {
                config: "GepState",
                field: "fitnesses",
                kind: rlevo_core::config::ConstraintKind::Custom(
                    "fitness cache length must equal pop_size",
                ),
            });
        }
        Ok(Self {
            population,
            fitnesses,
            best_genome,
            best_fitness,
            generation,
        })
    }

    /// Current population, shape `(pop_size, genome_len)`.
    #[must_use]
    pub fn population(&self) -> &Tensor<B, 2, Int> {
        &self.population
    }

    /// Host-side fitness cache (empty at bootstrap, else `pop_size` long).
    #[must_use]
    pub fn fitnesses(&self) -> &[f32] {
        &self.fitnesses
    }

    /// Best-so-far genome (shape `(1, genome_len)`), or `None` before the
    /// first `tell`.
    #[must_use]
    pub fn best_genome(&self) -> Option<&Tensor<B, 2, Int>> {
        self.best_genome.as_ref()
    }

    /// Best-so-far (canonical, maximise) fitness.
    #[must_use]
    pub fn best_fitness(&self) -> f32 {
        self.best_fitness
    }

    /// Generation counter.
    #[must_use]
    pub fn generation(&self) -> usize {
        self.generation
    }
}

/// Gene Expression Programming as a generational [`Strategy`].
///
/// The genome is a `Tensor<B, 2, Int>` of shape `(pop_size, head_len +
/// tail_len)`. Selection is roulette-wheel (fitness-proportionate) with
/// elitism, following Ferreira (2001): the best-so-far individual is copied
/// unchanged into every new generation, while the rest are produced by
/// roulette selection followed by crossover, transposition, and locus-class
/// point mutation. All randomness is host-side via
/// [`seed_stream`]; each operator draws from its own
/// [`SeedPurpose`] stream.
///
/// Decoding and phenotype evaluation are **not** part of the strategy — they
/// live in the [`BatchFitnessFn`] (see [`GepSymRegression`]), keeping the
/// ask/tell loop free of program interpretation.
#[derive(Debug, Clone)]
pub struct GepStrategy<B: Backend, F: FunctionSet> {
    alphabet: Alphabet<F>,
    _backend: PhantomData<fn() -> B>,
}

impl<B: Backend, F: FunctionSet> GepStrategy<B, F> {
    /// Builds a strategy over the given symbol alphabet.
    #[must_use]
    pub fn new(alphabet: Alphabet<F>) -> Self {
        Self {
            alphabet,
            _backend: PhantomData,
        }
    }

    /// The alphabet this strategy samples and decodes against.
    #[must_use]
    pub fn alphabet(&self) -> &Alphabet<F> {
        &self.alphabet
    }

    /// Samples one fresh valid chromosome: head loci any symbol, tail loci
    /// terminals only.
    fn sample_chromosome(&self, cfg: &GepConfig, rng: &mut dyn Rng) -> Vec<Symbol> {
        let mut g = Vec::with_capacity(cfg.genome_len());
        for _ in 0..cfg.head_len {
            g.push(self.alphabet.sample_head_symbol(rng));
        }
        for _ in 0..cfg.tail_len {
            g.push(self.alphabet.sample_tail_symbol(rng));
        }
        g
    }
}

/// Pulls an integer population tensor to host as per-row symbol chromosomes.
fn tensor_to_rows<B: Backend>(pop: &Tensor<B, 2, Int>, genome_len: usize) -> Vec<Vec<Symbol>> {
    let flat: Vec<i32> = pop
        .clone()
        .into_data()
        .into_vec::<i32>()
        // Panics if the population tensor cannot be read as `i32` on the host.
        .expect("genome tensor must be readable as i32");
    flat.chunks(genome_len)
        .map(|row| row.iter().map(|&v| Symbol::from_raw(v)).collect())
        .collect()
}

/// Uploads per-row symbol chromosomes as an integer population tensor.
fn rows_to_tensor<B: Backend>(
    rows: &[Vec<Symbol>],
    genome_len: usize,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<B, 2, Int> {
    let pop_size = rows.len();
    let mut flat: Vec<i32> = Vec::with_capacity(pop_size * genome_len);
    for row in rows {
        flat.extend(row.iter().map(|s| s.value()));
    }
    Tensor::<B, 2, Int>::from_data(TensorData::new(flat, [pop_size, genome_len]), device)
}

/// Roulette-wheel selection of `k` parent indices from canonical (maximise)
/// fitness.
///
/// Fitness is canonical — *higher is better*. Each individual's selection
/// weight is its fitness shifted so the worst finite individual sits at a
/// small floor `ε`, i.e. `weight = (f − min_finite) + ε`, so a higher
/// fitness yields a larger weight. This is sense-agnostic: the harness has
/// already mapped a cost objective into canonical space, so no `mse`
/// appears here. Non-finite fitness contributes zero weight. If the total
/// weight is non-positive (e.g. every individual diverged), selection falls
/// back to uniform sampling.
fn roulette_select(fitnesses: &[f32], k: usize, rng: &mut dyn Rng) -> Vec<usize> {
    const EPS: f32 = 1e-6;
    let n = fitnesses.len();
    let min_finite = fitnesses
        .iter()
        .copied()
        .filter(|f| f.is_finite())
        .fold(f32::INFINITY, f32::min);
    let weights: Vec<f32> = fitnesses
        .iter()
        .map(|&f| {
            if f.is_finite() && min_finite.is_finite() {
                (f - min_finite).max(0.0) + EPS
            } else {
                0.0
            }
        })
        .collect();
    let total: f32 = weights.iter().sum();

    let mut out = Vec::with_capacity(k);
    if total <= 0.0 || !total.is_finite() {
        for _ in 0..k {
            out.push(rng.random_range(0..n));
        }
        return out;
    }
    for _ in 0..k {
        let mut r = rng.random::<f32>() * total;
        let mut chosen = n - 1;
        for (i, &w) in weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                chosen = i;
                break;
            }
        }
        out.push(chosen);
    }
    out
}

fn update_best<B: Backend>(state: &mut GepState<B>, pop: &Tensor<B, 2, Int>, fitness: &[f32]) {
    if fitness.is_empty() {
        return;
    }
    let mut best_idx = 0usize;
    let mut best_f = fitness[0];
    for (i, &f) in fitness.iter().enumerate().skip(1) {
        if f > best_f {
            best_f = f;
            best_idx = i;
        }
    }
    if best_f > state.best_fitness {
        let device = pop.device();
        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
        let idx =
            Tensor::<B, 1, Int>::from_data(TensorData::new(vec![best_idx as i32], [1]), &device);
        state.best_genome = Some(pop.clone().select(0, idx));
        state.best_fitness = best_f;
    }
}

impl<B: Backend, F: FunctionSet> Strategy<B> for GepStrategy<B, F>
where
    B::Device: Clone,
{
    type Params = GepConfig;
    type State = GepState<B>;
    type Genome = Tensor<B, 2, Int>;

    /// Samples the initial population: each chromosome's head loci hold any
    /// symbol and its tail loci hold terminals, so every individual is valid by
    /// construction.
    fn init(
        &self,
        params: &GepConfig,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> GepState<B> {
        debug_assert_eq!(
            self.alphabet.n_vars, params.n_vars,
            "GepStrategy: alphabet/config variable counts must agree"
        );
        let genome_len = params.genome_len();
        let mut stream = seed_stream(rng.next_u64(), 0, SeedPurpose::Init);
        let rows: Vec<Vec<Symbol>> = (0..params.pop_size)
            .map(|_| self.sample_chromosome(params, &mut stream))
            .collect();
        let population = rows_to_tensor::<B>(&rows, genome_len, device);
        GepState {
            population,
            fitnesses: Vec::new(),
            best_genome: None,
            best_fitness: f32::NEG_INFINITY,
            generation: 0,
        }
    }

    /// Returns the population to evaluate this generation.
    ///
    /// On the first call (no fitness cached yet) the initial population is
    /// returned unchanged. On subsequent calls a new generation is bred from
    /// the current one: roulette selection, then one-/two-point crossover,
    /// IS/RIS transposition, and locus-class point mutation, with the
    /// best-so-far individual copied into row 0 (elitism). Each operator draws
    /// from its own [`SeedPurpose`] stream off a single base seed.
    fn ask(
        &self,
        params: &GepConfig,
        state: &GepState<B>,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> (Tensor<B, 2, Int>, GepState<B>) {
        // First call: evaluate the initial population as-is.
        if state.fitnesses.is_empty() {
            return (state.population.clone(), state.clone());
        }

        let genome_len = params.genome_len();
        let head_len = params.head_len;
        let pop_size = params.pop_size;
        let parents = tensor_to_rows::<B>(&state.population, genome_len);

        // One base seed; four independent operator streams off it.
        let base = rng.next_u64();
        let generation = state.generation as u64;
        let mut sel_rng = seed_stream(base, generation, SeedPurpose::Selection);
        let mut xover_rng = seed_stream(base, generation, SeedPurpose::Crossover);
        let mut trans_rng = seed_stream(base, generation, SeedPurpose::Transposition);
        let mut mut_rng = seed_stream(base, generation, SeedPurpose::Mutation);

        // Roulette selection -> offspring seeds.
        let chosen = roulette_select(&state.fitnesses, pop_size, &mut sel_rng);
        let mut offspring: Vec<Vec<Symbol>> =
            chosen.into_iter().map(|i| parents[i].clone()).collect();

        // Crossover over consecutive pairs.
        for pair in offspring.chunks_mut(2) {
            if pair.len() < 2 {
                break;
            }
            let (left, right) = pair.split_at_mut(1);
            if xover_rng.random::<f32>() < params.crossover_1p_rate.get() {
                one_point_crossover(&mut left[0], &mut right[0], &mut xover_rng);
            }
            if xover_rng.random::<f32>() < params.crossover_2p_rate.get() {
                two_point_crossover(&mut left[0], &mut right[0], &mut xover_rng);
            }
        }

        // Transposition + point mutation, per individual.
        for child in &mut offspring {
            if trans_rng.random::<f32>() < params.is_transpose_rate.get() {
                is_transposition(child, head_len, &mut trans_rng);
            }
            if trans_rng.random::<f32>() < params.ris_transpose_rate.get() {
                ris_transposition(child, head_len, &self.alphabet, &mut trans_rng);
            }
            point_mutation(
                child,
                head_len,
                &self.alphabet,
                params.mutation_rate.get(),
                &mut mut_rng,
            );
        }

        // Elitism: copy the best-so-far genome into row 0 unchanged.
        if let Some(best) = &state.best_genome {
            let best_rows = tensor_to_rows::<B>(best, genome_len);
            if let Some(elite) = best_rows.into_iter().next() {
                offspring[0] = elite;
            }
        }

        let population = rows_to_tensor::<B>(&offspring, genome_len, device);
        (population, state.clone())
    }

    /// Caches the evaluated population and its fitness, then updates the
    /// best-so-far record. The returned population becomes the current
    /// generation that the next [`ask`](Strategy::ask) breeds from.
    fn tell(
        &self,
        _params: &GepConfig,
        population: Tensor<B, 2, Int>,
        fitness: Tensor<B, 1>,
        mut state: GepState<B>,
        _rng: &mut dyn Rng,
    ) -> (GepState<B>, StrategyMetrics) {
        let fitness_host = fitness
            .into_data()
            .into_vec::<f32>()
            .expect("fitness tensor must be readable as f32");

        update_best(&mut state, &population, &fitness_host);
        state.population = population;
        state.generation += 1;

        let metrics =
            StrategyMetrics::from_host_fitness(state.generation, &fitness_host, state.best_fitness);
        state.best_fitness = metrics.best_fitness_ever();
        // Cache this generation's fitness for the next `ask`'s roulette draw.
        state.fitnesses = fitness_host;
        (state, metrics)
    }

    fn best(&self, state: &GepState<B>) -> Option<(Tensor<B, 2, Int>, f32)> {
        state
            .best_genome
            .as_ref()
            .map(|g| (g.clone(), state.best_fitness))
    }
}

/// A symbolic-regression [`BatchFitnessFn`] for GEP populations.
///
/// Holds the target dataset (input rows and expected outputs) and the alphabet
/// to decode against. [`evaluate_batch`](BatchFitnessFn::evaluate_batch) pulls
/// the population to host and, in parallel across rows (`rayon`), decodes each
/// chromosome to an [`ExpressionTree`](super::ExpressionTree) and scores it by
/// mean squared error. Decoding is deterministic, so the row-parallel order
/// does not affect results.
#[derive(Debug, Clone)]
pub struct GepSymRegression<F: FunctionSet> {
    alphabet: Alphabet<F>,
    genome_len: usize,
    inputs: Vec<Vec<f32>>,
    targets: Vec<f32>,
}

impl<F: FunctionSet> GepSymRegression<F> {
    /// Builds the fitness function from an alphabet, the genome length, the
    /// input rows, and the matching expected outputs.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `inputs` is empty — an empty dataset would make MSE `0.0 / 0.0 = NaN`
    ///   fitness (`n_points == 0`), silently poisoning the population.
    /// - `inputs` and `targets` differ in length.
    /// - any input row does not have exactly `alphabet.n_vars` entries — a short
    ///   row is otherwise silently zero-padded by
    ///   [`ExpressionTree::eval`](super::ExpressionTree::eval)
    ///   (`inputs.get(i).unwrap_or(0.0)`), so the regression would quietly fit
    ///   the wrong target.
    #[must_use]
    pub fn new(
        alphabet: Alphabet<F>,
        genome_len: usize,
        inputs: Vec<Vec<f32>>,
        targets: Vec<f32>,
    ) -> Self {
        assert!(
            !inputs.is_empty(),
            "GepSymRegression: dataset must be non-empty (empty inputs give NaN fitness)"
        );
        assert_eq!(
            inputs.len(),
            targets.len(),
            "GepSymRegression: inputs and targets must have equal length"
        );
        let n_vars = alphabet.n_vars;
        assert!(
            inputs.iter().all(|row| row.len() == n_vars),
            "GepSymRegression: every input row must have exactly n_vars = {n_vars} entries"
        );
        Self {
            alphabet,
            genome_len,
            inputs,
            targets,
        }
    }
}

impl<B: Backend, F: FunctionSet> BatchFitnessFn<B, Tensor<B, 2, Int>> for GepSymRegression<F> {
    fn evaluate_batch(
        &mut self,
        population: &Tensor<B, 2, Int>,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Tensor<B, 1> {
        let rows = tensor_to_rows::<B>(population, self.genome_len);
        let pop_size = rows.len();
        #[allow(clippy::cast_precision_loss)]
        let n_points = self.targets.len() as f32;

        let fitness: Vec<f32> = rows
            .par_iter()
            .map(|genome| {
                let tree = GepDecoder.decode(&self.alphabet, genome);
                let mut sse = 0.0f32;
                for (input, &target) in self.inputs.iter().zip(self.targets.iter()) {
                    let pred = tree.eval(&self.alphabet, input);
                    let err = pred - target;
                    sse += err * err;
                }
                sse / n_points
            })
            .collect();

        Tensor::<B, 1>::from_data(TensorData::new(fitness, [pop_size]), device)
    }

    fn sense(&self) -> rlevo_core::objective::ObjectiveSense {
        // Mean squared error is a cost — lower is better.
        rlevo_core::objective::ObjectiveSense::Minimize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::function_set::ArithmeticFunctionSet;
    use crate::strategy::EvolutionaryHarness;
    use burn::backend::Flex;

    type TestBackend = Flex;

    /// All-equal fitness gives every individual the same weight: draws are valid
    /// indices and every index is reachable.
    #[test]
    fn roulette_all_equal_fitness_stays_in_range() {
        let mut rng = seed_stream(31, 0, SeedPurpose::Selection);
        let fits: Vec<f32> = vec![1.0; 5];
        let picks: Vec<usize> = roulette_select(&fits, 10, &mut rng);
        assert_eq!(picks.len(), 10);
        assert!(picks.iter().all(|&i| i < 5));
    }

    /// All-`NaN` fitness zeroes every weight, so selection falls back to
    /// *uniform* sampling — not a degenerate always-index-0 draw. Over a large
    /// deterministic sample each of the 4 candidates must appear roughly N/4
    /// times (loose band), which a silent "always pick 0" fallback would fail.
    #[test]
    fn roulette_all_nan_falls_back_to_uniform() {
        const N: usize = 4000;
        let mut rng = seed_stream(32, 0, SeedPurpose::Selection);
        let fits: Vec<f32> = vec![f32::NAN; 4];
        let picks: Vec<usize> = roulette_select(&fits, N, &mut rng);
        assert_eq!(picks.len(), N);
        assert!(picks.iter().all(|&i| i < 4));

        let mut counts = [0usize; 4];
        for &i in &picks {
            counts[i] += 1;
        }
        // Uniform expectation is N/4 = 1000; allow a generous ±40% band so the
        // fixed-seed draw cannot flake while still rejecting a collapsed
        // distribution (e.g. every draw landing on one index).
        let expected: usize = N / 4;
        let low = expected - expected * 2 / 5;
        let high = expected + expected * 2 / 5;
        for (idx, &c) in counts.iter().enumerate() {
            assert!(
                (low..=high).contains(&c),
                "index {idx} drawn {c} times, outside uniform band [{low}, {high}]"
            );
        }
    }

    /// A single-individual population can only ever pick index 0.
    #[test]
    fn roulette_single_element_always_picks_zero() {
        let mut rng = seed_stream(33, 0, SeedPurpose::Selection);
        let picks: Vec<usize> = roulette_select(&[3.5], 6, &mut rng);
        assert_eq!(picks, vec![0usize; 6]);
    }

    /// Requesting zero parents from an empty fitness slice yields no picks (and
    /// never samples an empty range).
    #[test]
    fn roulette_empty_with_zero_k_is_empty() {
        let mut rng = seed_stream(34, 0, SeedPurpose::Selection);
        let picks: Vec<usize> = roulette_select(&[], 0, &mut rng);
        assert!(picks.is_empty());
    }

    /// With all-negative (canonical) fitness the shift `f − min_finite` still
    /// favours the least-negative individual; it is picked more often than the
    /// most-negative one, and every draw is in range.
    #[test]
    fn roulette_negative_fitness_favours_highest() {
        let mut rng = seed_stream(35, 0, SeedPurpose::Selection);
        // Index 2 (−1.0) is the largest; index 0 (−100.0) the smallest.
        let fits: Vec<f32> = vec![-100.0, -50.0, -1.0, -75.0];
        let picks: Vec<usize> = roulette_select(&fits, 2000, &mut rng);
        assert!(picks.iter().all(|&i| i < 4));
        let count_best: usize = picks.iter().filter(|&&i| i == 2).count();
        let count_worst: usize = picks.iter().filter(|&&i| i == 0).count();
        assert!(
            count_best > count_worst,
            "highest fitness ({count_best}) should be picked more than lowest ({count_worst})"
        );
    }

    #[test]
    fn try_new_checks_fitness_length() {
        let device = Default::default();
        let pop = Tensor::<TestBackend, 2, Int>::zeros([3, 4], &device);
        assert!(GepState::try_new(pop.clone(), vec![], None, f32::MIN, 0).is_ok());
        assert!(GepState::try_new(pop.clone(), vec![1.0; 3], None, 1.0, 1).is_ok());
        assert!(GepState::try_new(pop, vec![1.0; 2], None, 1.0, 1).is_err());
    }

    fn alphabet(n_vars: usize) -> Alphabet<ArithmeticFunctionSet> {
        Alphabet::new(ArithmeticFunctionSet, n_vars, vec![])
    }

    /// Runs GEP on a dataset and returns the best MSE found.
    fn run_gep(
        n_vars: usize,
        inputs: Vec<Vec<f32>>,
        targets: Vec<f32>,
        seed: u64,
        max_gens: usize,
    ) -> f32 {
        let device = Default::default();
        let cfg = GepConfig::new(7, 2, n_vars, 100).unwrap();
        let genome_len = cfg.genome_len();
        let strategy = GepStrategy::<TestBackend, _>::new(alphabet(n_vars));
        let fitness = GepSymRegression::new(alphabet(n_vars), genome_len, inputs, targets);
        let mut harness = EvolutionaryHarness::<TestBackend, _, _>::new(
            strategy, cfg, fitness, seed, device, max_gens,
        )
        .expect("valid params");
        harness.reset();
        loop {
            if harness.step(()).done {
                break;
            }
        }
        harness.latest_metrics().unwrap().best_fitness_ever()
    }

    /// Converges on `f(x) = x² + x + 1` over 20 points in [-1, 1].
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn converges_on_quadratic() {
        let xs: Vec<f32> = (0..20).map(|i| -1.0 + 2.0 * (i as f32) / 19.0).collect();
        let inputs: Vec<Vec<f32>> = xs.iter().map(|&x| vec![x]).collect();
        let targets: Vec<f32> = xs.iter().map(|&x| x * x + x + 1.0).collect();
        let best = run_gep(1, inputs, targets, 11, 500);
        assert!(best <= 0.01, "expected MSE <= 0.01, got {best}");
    }

    /// Converges on `f(x) = sin(x) · x` over 20 points in [-3, 3].
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn converges_on_sin_times_x() {
        let xs: Vec<f32> = (0..20).map(|i| -3.0 + 6.0 * (i as f32) / 19.0).collect();
        let inputs: Vec<Vec<f32>> = xs.iter().map(|&x| vec![x]).collect();
        let targets: Vec<f32> = xs.iter().map(|&x| x.sin() * x).collect();
        let best = run_gep(1, inputs, targets, 7, 500);
        assert!(best <= 0.01, "expected MSE <= 0.01, got {best}");
    }

    /// An empty dataset is rejected at construction (would give NaN fitness).
    #[test]
    #[should_panic(expected = "dataset must be non-empty")]
    fn test_gep_sym_regression_rejects_empty_dataset() {
        let _ = GepSymRegression::new(alphabet(1), 15, Vec::new(), Vec::new());
    }

    /// A row whose width differs from `n_vars` is rejected (would be silently
    /// zero-padded and fit the wrong target).
    #[test]
    #[should_panic(expected = "every input row must have exactly n_vars")]
    fn test_gep_sym_regression_rejects_mismatched_row_width() {
        // n_vars = 2 but the single row has only one entry.
        let _ = GepSymRegression::new(alphabet(2), 15, vec![vec![1.0]], vec![0.0]);
    }

    /// A well-formed tiny dataset yields finite fitness for the zero genome.
    #[test]
    fn test_gep_sym_regression_valid_dataset_is_finite() {
        let device = Default::default();
        let cfg = GepConfig::new(7, 2, 1, 4).unwrap();
        let genome_len = cfg.genome_len();
        let mut fitness = GepSymRegression::new(
            alphabet(1),
            genome_len,
            vec![vec![0.5], vec![-0.5]],
            vec![0.25, 0.25],
        );
        // A valid chromosome: every locus is variable `x` (id 8), which decodes
        // to the single-node tree `x`. An all-zeros genome would be all `add`
        // with no terminal tail — a malformed chromosome that trips the separate
        // decode out-of-bounds path (decode.rs §1.1), unrelated to this test.
        let pop = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![8i32; 4 * genome_len], [4, genome_len]),
            &device,
        );
        let scores: Vec<f32> = fitness
            .evaluate_batch(&pop, &device)
            .into_data()
            .into_vec()
            .expect("fitness host-read of a tensor this test just built");
        assert!(
            scores.iter().all(|s| s.is_finite()),
            "all fitness values must be finite, got {scores:?}"
        );
    }

    /// Converges on `f(x, y) = x² + y²` over a 5×5 grid in [-2, 2]² (`n_vars` = 2).
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn converges_on_sum_of_squares() {
        let coords: Vec<f32> = (0..5).map(|i| -2.0 + 4.0 * (i as f32) / 4.0).collect();
        let mut inputs = Vec::new();
        let mut targets = Vec::new();
        for &x in &coords {
            for &y in &coords {
                inputs.push(vec![x, y]);
                targets.push(x * x + y * y);
            }
        }
        let best = run_gep(2, inputs, targets, 5, 500);
        assert!(best <= 0.01, "expected MSE <= 0.01, got {best}");
    }
}
