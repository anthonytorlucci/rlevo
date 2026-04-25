//! Binary-coded Genetic Algorithm.
//!
//! Operates on a `Tensor<B, 2, Int>` population where each gene is
//! restricted to `{0, 1}`. Selection and elitism reuse the host-side
//! fitness indexing helpers shared with the real-coded GA; crossover
//! and mutation use the binary-specific operators in
//! [`crate::ops::crossover::binary_uniform_crossover`] and
//! [`crate::ops::mutation::bit_flip_mutation`].
//!
//! # Test landscape
//!
//! The unit test uses OneMax phrased as minimization
//! (`cost = D − count_ones(x)`) so the harness's "lower is better"
//! convention applies uniformly across strategies.

use std::marker::PhantomData;

use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use rand::Rng;

use crate::ops::crossover::binary_uniform_crossover;
use crate::ops::mutation::bit_flip_mutation;
use crate::ops::selection::{tournament_indices_host, truncation_indices_host};
use crate::rng::{SeedPurpose, seed_stream};
use crate::strategy::{Strategy, StrategyMetrics};

/// Static configuration for a [`BinaryGeneticAlgorithm`] run.
#[derive(Debug, Clone)]
pub struct BinaryGaConfig {
    /// Population size.
    pub pop_size: usize,
    /// Genome dimensionality.
    pub genome_dim: usize,
    /// Probability of bit flip per gene.
    pub mutation_rate: f32,
    /// Probability of taking parent A's bit in uniform crossover.
    pub crossover_p: f32,
    /// Tournament size for parent selection.
    pub tournament_size: usize,
    /// Number of elites carried over to the next generation.
    pub elitism_k: usize,
}

impl BinaryGaConfig {
    /// Sensible defaults for small-scale binary optimization.
    #[must_use]
    pub fn default_for(pop_size: usize, genome_dim: usize) -> Self {
        Self {
            pop_size,
            genome_dim,
            mutation_rate: 1.0 / genome_dim as f32,
            crossover_p: 0.5,
            tournament_size: 2,
            elitism_k: 1,
        }
    }
}

/// State for [`BinaryGeneticAlgorithm`].
#[derive(Debug, Clone)]
pub struct BinaryGaState<B: Backend> {
    /// Current population, shape `(pop_size, D)`.
    pub population: Tensor<B, 2, Int>,
    /// Host-side fitness cache.
    pub fitness: Vec<f32>,
    /// Best-so-far genome.
    pub best_genome: Option<Tensor<B, 2, Int>>,
    /// Best-so-far fitness.
    pub best_fitness: f32,
    /// Generation counter.
    pub generation: usize,
}

/// Binary-coded canonical Genetic Algorithm.
///
/// # Example
///
/// ```no_run
/// use burn::backend::NdArray;
/// use evorl_evolution::algorithms::ga_binary::{BinaryGaConfig, BinaryGeneticAlgorithm};
///
/// let strategy = BinaryGeneticAlgorithm::<NdArray>::new();
/// let params = BinaryGaConfig::default_for(32, 16);
/// let _ = (strategy, params);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct BinaryGeneticAlgorithm<B: Backend> {
    _backend: PhantomData<fn() -> B>,
}

impl<B: Backend> BinaryGeneticAlgorithm<B> {
    /// Builds a new (stateless) strategy object.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _backend: PhantomData,
        }
    }

    fn sample_initial_population(
        params: &BinaryGaConfig,
        rng: &mut dyn Rng,
        device: &B::Device,
    ) -> Tensor<B, 2, Int> {
        B::seed(device, rng.next_u64());
        let u = Tensor::<B, 2>::random(
            [params.pop_size, params.genome_dim],
            burn::tensor::Distribution::Uniform(0.0, 1.0),
            device,
        );
        u.lower_elem(0.5).int()
    }
}

impl<B: Backend> Strategy<B> for BinaryGeneticAlgorithm<B>
where
    B::Device: Clone,
{
    type Params = BinaryGaConfig;
    type State = BinaryGaState<B>;
    type Genome = Tensor<B, 2, Int>;

    fn init(
        &self,
        params: &BinaryGaConfig,
        rng: &mut dyn Rng,
        device: &B::Device,
    ) -> BinaryGaState<B> {
        BinaryGaState {
            population: Self::sample_initial_population(params, rng, device),
            fitness: Vec::new(),
            best_genome: None,
            best_fitness: f32::INFINITY,
            generation: 0,
        }
    }

    fn ask(
        &self,
        params: &BinaryGaConfig,
        state: &BinaryGaState<B>,
        rng: &mut dyn Rng,
        device: &B::Device,
    ) -> (Tensor<B, 2, Int>, BinaryGaState<B>) {
        if state.fitness.is_empty() {
            return (state.population.clone(), state.clone());
        }

        let mut selection_rng = seed_stream(
            rng.next_u64(),
            state.generation as u64,
            SeedPurpose::Selection,
        );
        let mut crossover_rng = seed_stream(
            rng.next_u64(),
            state.generation as u64,
            SeedPurpose::Crossover,
        );
        let mut mutation_rng = seed_stream(
            rng.next_u64(),
            state.generation as u64,
            SeedPurpose::Mutation,
        );

        let idx_a = tournament_indices_host(
            &state.fitness,
            params.tournament_size,
            params.pop_size,
            &mut selection_rng,
        );
        let idx_b = tournament_indices_host(
            &state.fitness,
            params.tournament_size,
            params.pop_size,
            &mut selection_rng,
        );
        let parents_a = state.population.clone().select(
            0,
            Tensor::<B, 1, Int>::from_data(TensorData::new(idx_a, [params.pop_size]), device),
        );
        let parents_b = state.population.clone().select(
            0,
            Tensor::<B, 1, Int>::from_data(TensorData::new(idx_b, [params.pop_size]), device),
        );

        B::seed(device, crossover_rng.next_u64());
        let offspring = binary_uniform_crossover(parents_a, parents_b, params.crossover_p, device);

        B::seed(device, mutation_rng.next_u64());
        let offspring = bit_flip_mutation(offspring, params.mutation_rate, device);

        (offspring, state.clone())
    }

    fn tell(
        &self,
        params: &BinaryGaConfig,
        offspring: Tensor<B, 2, Int>,
        fitness: Tensor<B, 1>,
        mut state: BinaryGaState<B>,
        _rng: &mut dyn Rng,
    ) -> (BinaryGaState<B>, StrategyMetrics) {
        let fitness_host = fitness.into_data().into_vec::<f32>().unwrap_or_default();
        let device = offspring.device();

        // First `tell`: initial population just evaluated.
        if state.fitness.is_empty() {
            state.fitness = fitness_host.clone();
            state.generation += 1;
            update_best(&mut state, &offspring, &fitness_host);
            let m = StrategyMetrics::from_host_fitness(
                state.generation,
                &fitness_host,
                state.best_fitness,
            );
            state.best_fitness = m.best_fitness_ever;
            state.population = offspring;
            return (state, m);
        }

        // Elitist replacement on (pop, fitness) × (offspring, fitness).
        let pop_size = params.pop_size;
        let k = params.elitism_k.min(pop_size);

        let elite_idx = truncation_indices_host(&state.fitness, k);
        let elites = state.population.clone().select(
            0,
            Tensor::<B, 1, Int>::from_data(TensorData::new(elite_idx.clone(), [k]), &device),
        );
        let n_off_keep = pop_size - k;
        let off_keep_idx = truncation_indices_host(&fitness_host, n_off_keep);
        let kept_off = offspring.clone().select(
            0,
            Tensor::<B, 1, Int>::from_data(
                TensorData::new(off_keep_idx.clone(), [n_off_keep]),
                &device,
            ),
        );
        let next_pop = Tensor::cat(vec![elites, kept_off], 0);
        let mut next_fit = Vec::with_capacity(pop_size);
        for i in elite_idx {
            #[allow(clippy::cast_sign_loss)]
            next_fit.push(state.fitness[i as usize]);
        }
        for i in off_keep_idx {
            #[allow(clippy::cast_sign_loss)]
            next_fit.push(fitness_host[i as usize]);
        }

        update_best(&mut state, &next_pop, &next_fit);
        state.population = next_pop;
        state.fitness = next_fit.clone();
        state.generation += 1;
        let m = StrategyMetrics::from_host_fitness(state.generation, &next_fit, state.best_fitness);
        state.best_fitness = m.best_fitness_ever;
        (state, m)
    }

    fn best(&self, state: &BinaryGaState<B>) -> Option<(Tensor<B, 2, Int>, f32)> {
        state
            .best_genome
            .as_ref()
            .map(|g| (g.clone(), state.best_fitness))
    }
}

fn update_best<B: Backend>(state: &mut BinaryGaState<B>, pop: &Tensor<B, 2, Int>, fitness: &[f32]) {
    if fitness.is_empty() {
        return;
    }
    let mut best_idx = 0usize;
    let mut best_f = fitness[0];
    for (i, &f) in fitness.iter().enumerate().skip(1) {
        if f < best_f {
            best_f = f;
            best_idx = i;
        }
    }
    if best_f < state.best_fitness {
        let device = pop.device();
        #[allow(clippy::cast_possible_wrap)]
        let idx =
            Tensor::<B, 1, Int>::from_data(TensorData::new(vec![best_idx as i64], [1]), &device);
        state.best_genome = Some(pop.clone().select(0, idx));
        state.best_fitness = best_f;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::BatchFitnessFn;
    use crate::strategy::EvolutionaryHarness;
    use burn::backend::NdArray;
    type TestBackend = NdArray;

    /// OneMax phrased as minimization: `cost = D − count_ones`.
    struct OneMaxCost {
        dim: usize,
    }

    impl<B: Backend> BatchFitnessFn<B, Tensor<B, 2, Int>> for OneMaxCost {
        fn evaluate_batch(
            &mut self,
            population: &Tensor<B, 2, Int>,
            device: &B::Device,
        ) -> Tensor<B, 1> {
            let dims = population.shape().dims;
            let pop_size = dims[0];
            let data = population
                .clone()
                .into_data()
                .into_vec::<i64>()
                .unwrap_or_default();
            let mut fitness = Vec::with_capacity(pop_size);
            for row in 0..pop_size {
                let mut ones = 0_u32;
                for col in 0..self.dim {
                    if data[row * self.dim + col] != 0 {
                        ones += 1;
                    }
                }
                #[allow(clippy::cast_precision_loss)]
                let cost = (self.dim as f32) - (ones as f32);
                fitness.push(cost);
            }
            Tensor::<B, 1>::from_data(TensorData::new(fitness, [pop_size]), device)
        }
    }

    #[test]
    fn binary_ga_solves_onemax() {
        let device = Default::default();
        let dim = 16;
        let params = BinaryGaConfig::default_for(32, dim);
        let mut harness = EvolutionaryHarness::<TestBackend, _, _>::new(
            BinaryGeneticAlgorithm::<TestBackend>::new(),
            params,
            OneMaxCost { dim },
            7,
            device,
            200,
        );
        harness.reset();
        loop {
            if harness.step(()).done {
                break;
            }
        }
        let best = harness.latest_metrics().unwrap().best_fitness_ever;
        // OneMax optimum: cost == 0 (all ones).
        approx::assert_relative_eq!(best, 0.0, epsilon = 1e-6);
    }
}
