//! Real-valued Genetic Algorithm.
//!
//! A canonical textbook GA over `Tensor<B, 2>` populations:
//!
//! 1. Evaluate the current population (done externally by the harness).
//! 2. Select parents via [`crate::ops::selection::tournament_select`].
//! 3. Recombine via [`crate::ops::crossover::blx_alpha`] or
//!    [`crate::ops::crossover::uniform_crossover`].
//! 4. Mutate via [`crate::ops::mutation::gaussian_mutation`].
//! 5. Replace via [`crate::ops::replacement::elitist`] or
//!    [`crate::ops::replacement::generational`].
//!
//! Operator variants are enum-selected via the [`GaConfig`] to avoid a
//! generic explosion; custom operator mixtures can still be built
//! bottom-up against the [`Strategy`] trait directly.
//!
//! # References
//!
//! - Goldberg (1989), *Genetic Algorithms in Search, Optimization, and
//!   Machine Learning*.
//! - Deb & Agrawal (1995), *Simulated binary crossover for continuous
//!   search space*.

use std::marker::PhantomData;

use burn::tensor::{Tensor, TensorData, backend::Backend};
use rand::Rng;

use crate::ops::{
    crossover::{blx_alpha, uniform_crossover},
    mutation::gaussian_mutation,
    replacement::{elitist, generational},
    selection::tournament_select,
};
use crate::rng::{SeedPurpose, seed_stream};
use crate::strategy::{Strategy, StrategyMetrics};

/// Selection algorithm choice.
#[derive(Debug, Clone, Copy)]
pub enum GaSelection {
    /// k-tournament selection.
    Tournament { size: usize },
}

/// Crossover algorithm choice.
#[derive(Debug, Clone, Copy)]
pub enum GaCrossover {
    /// BLX-α real-valued crossover.
    BlxAlpha { alpha: f32 },
    /// Uniform swap crossover with per-gene probability `p`.
    Uniform { p: f32 },
}

/// Replacement algorithm choice.
#[derive(Debug, Clone, Copy)]
pub enum GaReplacement {
    /// Offspring replace the entire parent population.
    Generational,
    /// `elitism_k` best parents persist; the rest come from offspring.
    Elitist { elitism_k: usize },
}

/// Static configuration for a [`GeneticAlgorithm`] run.
#[derive(Debug, Clone)]
pub struct GaConfig {
    /// Number of individuals per generation.
    pub pop_size: usize,
    /// Genome dimensionality.
    pub genome_dim: usize,
    /// Lower / upper bound on initial samples and clamping.
    pub bounds: (f32, f32),
    /// σ for isotropic Gaussian mutation.
    pub mutation_sigma: f32,
    /// Selection operator.
    pub selection: GaSelection,
    /// Crossover operator.
    pub crossover: GaCrossover,
    /// Replacement policy.
    pub replacement: GaReplacement,
}

impl GaConfig {
    /// Sensible defaults for small-scale continuous optimization.
    #[must_use]
    pub fn default_for(pop_size: usize, genome_dim: usize) -> Self {
        Self {
            pop_size,
            genome_dim,
            bounds: (-5.12, 5.12),
            mutation_sigma: 0.3,
            selection: GaSelection::Tournament { size: 2 },
            crossover: GaCrossover::BlxAlpha { alpha: 0.5 },
            replacement: GaReplacement::Elitist { elitism_k: 1 },
        }
    }
}

/// Generation-to-generation state carried by [`GeneticAlgorithm`].
#[derive(Debug, Clone)]
pub struct GaState<B: Backend> {
    /// Current population, shape `(pop_size, genome_dim)`.
    pub population: Tensor<B, 2>,
    /// Cached fitness for the current population. Empty on init until
    /// the first `tell` call overwrites it.
    pub fitness: Vec<f32>,
    /// Best-so-far genome, shape `(1, genome_dim)`.
    pub best_genome: Option<Tensor<B, 2>>,
    /// Best-so-far fitness.
    pub best_fitness: f32,
    /// Completed-generation counter.
    pub generation: usize,
}

/// Real-valued canonical Genetic Algorithm.
///
/// # Example
///
/// ```no_run
/// use burn::backend::NdArray;
/// use evorl_evolution::algorithms::ga::{
///     GaConfig, GaCrossover, GaReplacement, GaSelection, GeneticAlgorithm,
/// };
///
/// let strategy = GeneticAlgorithm::<NdArray>::new();
/// let params = GaConfig {
///     pop_size: 64,
///     genome_dim: 10,
///     bounds: (-5.12, 5.12),
///     mutation_sigma: 0.3,
///     selection: GaSelection::Tournament { size: 2 },
///     crossover: GaCrossover::BlxAlpha { alpha: 0.5 },
///     replacement: GaReplacement::Elitist { elitism_k: 2 },
/// };
/// let _ = (strategy, params);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct GeneticAlgorithm<B: Backend> {
    _backend: PhantomData<fn() -> B>,
}

impl<B: Backend> GeneticAlgorithm<B> {
    /// Build a new (stateless) strategy object.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _backend: PhantomData,
        }
    }

    fn sample_initial_population(
        params: &GaConfig,
        rng: &mut dyn Rng,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let (lo, hi) = params.bounds;
        let range = f64::from(hi - lo);
        let lo_f = f64::from(lo);
        B::seed(device, rng.next_u64());
        let u = Tensor::<B, 2>::random(
            [params.pop_size, params.genome_dim],
            burn::tensor::Distribution::Uniform(lo_f, lo_f + range),
            device,
        );
        u
    }
}

impl<B: Backend> Strategy<B> for GeneticAlgorithm<B>
where
    B::Device: Clone,
{
    type Params = GaConfig;
    type State = GaState<B>;
    type Genome = Tensor<B, 2>;

    fn init(&self, params: &GaConfig, rng: &mut dyn Rng, device: &B::Device) -> GaState<B> {
        let population = Self::sample_initial_population(params, rng, device);
        GaState {
            population,
            fitness: Vec::new(),
            best_genome: None,
            best_fitness: f32::INFINITY,
            generation: 0,
        }
    }

    fn ask(
        &self,
        params: &GaConfig,
        state: &GaState<B>,
        rng: &mut dyn Rng,
        device: &B::Device,
    ) -> (Tensor<B, 2>, GaState<B>) {
        // On the first call, state.fitness is empty; the harness has not
        // evaluated anyone yet. Return the initial population unchanged.
        if state.fitness.is_empty() {
            return (state.population.clone(), state.clone());
        }

        let GaConfig {
            pop_size,
            mutation_sigma,
            selection,
            crossover,
            ..
        } = params;

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
        let mut selection_rng = seed_stream(
            rng.next_u64(),
            state.generation as u64,
            SeedPurpose::Selection,
        );

        // 1. Select two tournaments' worth of parents.
        let parents_a = match selection {
            GaSelection::Tournament { size } => tournament_select(
                &state.population,
                &state.fitness,
                *size,
                *pop_size,
                &mut selection_rng,
                device,
            ),
        };
        let parents_b = match selection {
            GaSelection::Tournament { size } => tournament_select(
                &state.population,
                &state.fitness,
                *size,
                *pop_size,
                &mut selection_rng,
                device,
            ),
        };

        // 2. Recombine.
        B::seed(device, crossover_rng.next_u64());
        let offspring = match crossover {
            GaCrossover::BlxAlpha { alpha } => blx_alpha(parents_a, parents_b, *alpha, device),
            GaCrossover::Uniform { p } => uniform_crossover(parents_a, parents_b, *p, device),
        };

        // 3. Mutate.
        B::seed(device, mutation_rng.next_u64());
        let offspring = gaussian_mutation(offspring, *mutation_sigma, device);

        // 4. Clamp to bounds.
        let (lo, hi) = params.bounds;
        let offspring = offspring.clamp(lo, hi);

        (offspring, state.clone())
    }

    fn tell(
        &self,
        params: &GaConfig,
        population: Tensor<B, 2>,
        fitness: Tensor<B, 1>,
        mut state: GaState<B>,
        _rng: &mut dyn Rng,
    ) -> (GaState<B>, StrategyMetrics) {
        let fitness_host = fitness.into_data().into_vec::<f32>().unwrap_or_default();

        // First `tell` after `init`: cache fitness for the seed population.
        if state.fitness.is_empty() {
            state.fitness = fitness_host.clone();
            state.generation += 1;
            update_best(&mut state, &population, &fitness_host);
            let m = StrategyMetrics::from_host_fitness(
                state.generation,
                &fitness_host,
                state.best_fitness,
            );
            state.best_fitness = m.best_fitness_ever;
            return (state, m);
        }

        let device = state.population.device();
        let (next_pop, next_fitness) = match params.replacement {
            GaReplacement::Generational => generational::<B>(
                state.population.clone(),
                &state.fitness,
                population.clone(),
                fitness_host.clone(),
            ),
            GaReplacement::Elitist { elitism_k } => elitist::<B>(
                state.population.clone(),
                &state.fitness,
                population.clone(),
                fitness_host.clone(),
                elitism_k,
                &device,
            ),
        };

        update_best(&mut state, &next_pop, &next_fitness);
        state.population = next_pop;
        state.fitness = next_fitness.clone();
        state.generation += 1;
        let m =
            StrategyMetrics::from_host_fitness(state.generation, &next_fitness, state.best_fitness);
        state.best_fitness = m.best_fitness_ever;
        (state, m)
    }

    fn best(&self, state: &GaState<B>) -> Option<(Tensor<B, 2>, f32)> {
        state
            .best_genome
            .as_ref()
            .map(|g| (g.clone(), state.best_fitness))
    }
}

fn update_best<B: Backend>(state: &mut GaState<B>, pop: &Tensor<B, 2>, fitness: &[f32]) {
    if fitness.is_empty() {
        return;
    }
    let mut best_idx = 0_usize;
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
        let idx = Tensor::<B, 1, burn::tensor::Int>::from_data(
            TensorData::new(vec![best_idx as i64], [1]),
            &device,
        );
        state.best_genome = Some(pop.clone().select(0, idx));
        state.best_fitness = best_f;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::FromFitnessEvaluable;
    use crate::strategy::EvolutionaryHarness;
    use burn::backend::NdArray;
    use rlevo_core::fitness::FitnessEvaluable;

    type TestBackend = NdArray;

    struct Sphere;
    struct SphereFit;
    impl FitnessEvaluable for SphereFit {
        type Individual = Vec<f64>;
        type Landscape = Sphere;
        fn evaluate(&self, x: &Self::Individual, _: &Self::Landscape) -> f64 {
            x.iter().map(|v| v * v).sum()
        }
    }

    #[test]
    fn ga_converges_on_sphere_d2() {
        let device = Default::default();
        let strategy = GeneticAlgorithm::<TestBackend>::new();
        let params = GaConfig {
            pop_size: 64,
            genome_dim: 2,
            bounds: (-5.0, 5.0),
            mutation_sigma: 0.2,
            selection: GaSelection::Tournament { size: 2 },
            crossover: GaCrossover::BlxAlpha { alpha: 0.5 },
            replacement: GaReplacement::Elitist { elitism_k: 1 },
        };
        let fitness_fn = FromFitnessEvaluable::new(SphereFit, Sphere);

        let mut harness = EvolutionaryHarness::<TestBackend, _, _>::new(
            strategy, params, fitness_fn, 42, device, 200,
        );
        harness.reset();
        loop {
            let step = harness.step(());
            if step.done {
                break;
            }
        }
        let m = harness.latest_metrics().unwrap();
        assert!(
            m.best_fitness_ever < 1e-2,
            "expected Sphere-D2 convergence, got best_fitness_ever={}",
            m.best_fitness_ever
        );
    }
}
