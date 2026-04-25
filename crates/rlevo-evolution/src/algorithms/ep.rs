//! Evolutionary Programming (Fogel-style).
//!
//! Classical EP differs from ES in the details:
//!
//! - **No crossover**. Each parent produces exactly one offspring by
//!   Gaussian mutation.
//! - **Self-adaptive σ**. Each individual carries its own σ, updated
//!   by the log-normal rule `σ' = σ · exp(τ · N(0, 1))`. Shared with ES
//!   but applied before every mutation call, not only at survivor time.
//! - **q-tournament survivor selection** on the `(μ + μ)` pool. Each
//!   individual plays `q` random opponents; the μ individuals with the
//!   highest win-counts survive. This diverges from truncation
//!   selection — EP gives weaker individuals a stochastic chance to
//!   survive.
//!
//! # Reference
//!
//! - Fogel (1994), *An introduction to simulated evolutionary
//!   optimization*.

use std::marker::PhantomData;

use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use rand::Rng;

use crate::ops::mutation::gaussian_mutation_per_row;
use crate::rng::{SeedPurpose, seed_stream};
use crate::strategy::{Strategy, StrategyMetrics};

/// Static configuration for an [`EvolutionaryProgramming`] run.
#[derive(Debug, Clone)]
pub struct EpConfig {
    /// Parent population size (offspring population is also μ — EP is
    /// strictly `μ + μ`).
    pub mu: usize,
    /// Genome dimensionality.
    pub genome_dim: usize,
    /// Search-space bounds (initialization and clamping).
    pub bounds: (f32, f32),
    /// Initial σ for every individual.
    pub initial_sigma: f32,
    /// Learning rate for the log-normal σ update. Default is
    /// `1 / sqrt(2 · sqrt(D))`.
    pub tau: f32,
    /// Number of opponents per tournament round (q-tournament).
    pub tournament_q: usize,
}

impl EpConfig {
    /// Default configuration for a given dimensionality.
    #[must_use]
    pub fn default_for(mu: usize, genome_dim: usize) -> Self {
        #[allow(clippy::cast_precision_loss)]
        let d = genome_dim as f32;
        let tau = 1.0 / (2.0 * d.sqrt()).sqrt();
        Self {
            mu,
            genome_dim,
            bounds: (-5.12, 5.12),
            initial_sigma: 1.0,
            tau,
            tournament_q: 10,
        }
    }
}

/// Generation-to-generation state for [`EvolutionaryProgramming`].
#[derive(Debug, Clone)]
pub struct EpState<B: Backend> {
    /// Parents, shape `(μ, D)`.
    pub parents: Tensor<B, 2>,
    /// Per-parent σ, shape `(μ,)`.
    pub sigmas: Tensor<B, 1>,
    /// Parent fitnesses, host-side cache.
    pub parent_fitness: Vec<f32>,
    /// Best-so-far genome, shape `(1, D)`.
    pub best_genome: Option<Tensor<B, 2>>,
    /// Best-so-far fitness.
    pub best_fitness: f32,
    /// Generation counter.
    pub generation: usize,
}

/// Classical Fogel EP.
///
/// # Example
///
/// ```no_run
/// use burn::backend::NdArray;
/// use evorl_evolution::algorithms::ep::{EpConfig, EvolutionaryProgramming};
///
/// let strategy = EvolutionaryProgramming::<NdArray>::new();
/// let params = EpConfig::default_for(30, 10);
/// let _ = (strategy, params);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct EvolutionaryProgramming<B: Backend> {
    _backend: PhantomData<fn() -> B>,
}

impl<B: Backend> EvolutionaryProgramming<B> {
    /// Builds a new (stateless) strategy object.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> Strategy<B> for EvolutionaryProgramming<B>
where
    B::Device: Clone,
{
    type Params = EpConfig;
    type State = EpState<B>;
    type Genome = Tensor<B, 2>;

    fn init(&self, params: &EpConfig, rng: &mut dyn Rng, device: &B::Device) -> EpState<B> {
        let (lo, hi) = params.bounds;
        B::seed(device, rng.next_u64());
        let parents = Tensor::<B, 2>::random(
            [params.mu, params.genome_dim],
            burn::tensor::Distribution::Uniform(f64::from(lo), f64::from(hi)),
            device,
        );
        let sigmas = Tensor::<B, 1>::from_data(
            TensorData::new(vec![params.initial_sigma; params.mu], [params.mu]),
            device,
        );
        EpState {
            parents,
            sigmas,
            parent_fitness: Vec::new(),
            best_genome: None,
            best_fitness: f32::INFINITY,
            generation: 0,
        }
    }

    fn ask(
        &self,
        params: &EpConfig,
        state: &EpState<B>,
        rng: &mut dyn Rng,
        device: &B::Device,
    ) -> (Tensor<B, 2>, EpState<B>) {
        // First call: evaluate the initial parents.
        if state.parent_fitness.is_empty() {
            return (state.parents.clone(), state.clone());
        }

        let mu = params.mu;
        let mut sigma_rng =
            seed_stream(rng.next_u64(), state.generation as u64, SeedPurpose::Other);
        let mut mutation_rng = seed_stream(
            rng.next_u64(),
            state.generation as u64,
            SeedPurpose::Mutation,
        );

        // Log-normal σ update for every parent.
        B::seed(device, sigma_rng.next_u64());
        let noise =
            Tensor::<B, 1>::random([mu], burn::tensor::Distribution::Normal(0.0, 1.0), device);
        let offspring_sigmas = state.sigmas.clone() * noise.mul_scalar(params.tau).exp();

        // Mutate each parent exactly once using its own σ.
        B::seed(device, mutation_rng.next_u64());
        let offspring =
            gaussian_mutation_per_row(state.parents.clone(), offspring_sigmas.clone(), device);
        let (lo, hi) = params.bounds;
        let offspring = offspring.clamp(lo, hi);

        // Stash offspring σ onto state via concatenation (parent_σ || offspring_σ).
        let mut state = state.clone();
        state.sigmas = Tensor::cat(vec![state.sigmas.clone(), offspring_sigmas], 0);
        (offspring, state)
    }

    fn tell(
        &self,
        params: &EpConfig,
        offspring: Tensor<B, 2>,
        fitness: Tensor<B, 1>,
        mut state: EpState<B>,
        rng: &mut dyn Rng,
    ) -> (EpState<B>, StrategyMetrics) {
        let fitness_host = fitness.into_data().into_vec::<f32>().unwrap_or_default();
        let device = offspring.device();

        // First `tell`: evaluated the initial parents.
        if state.parent_fitness.is_empty() {
            state.parent_fitness = fitness_host.clone();
            state.generation += 1;
            update_best(&mut state, &offspring, &fitness_host);
            let m = StrategyMetrics::from_host_fitness(
                state.generation,
                &fitness_host,
                state.best_fitness,
            );
            state.best_fitness = m.best_fitness_ever;
            state.parents = offspring;
            state.sigmas = Tensor::<B, 1>::from_data(
                TensorData::new(vec![params.initial_sigma; params.mu], [params.mu]),
                &device,
            );
            return (state, m);
        }

        let mu = params.mu;
        // Build the (μ + μ) pool.
        let combined_pop = Tensor::cat(vec![state.parents.clone(), offspring.clone()], 0);
        let combined_fit: Vec<f32> = state
            .parent_fitness
            .iter()
            .chain(fitness_host.iter())
            .copied()
            .collect();
        let combined_sigmas = state.sigmas.clone(); // already (μ + μ) thanks to `ask`.

        // q-tournament: for each of the 2μ members, sample q opponents
        // and count wins (lower fitness beats higher). The μ highest-
        // win members survive.
        let mut selection_rng = seed_stream(
            rng.next_u64(),
            state.generation as u64,
            SeedPurpose::Selection,
        );
        let n = combined_fit.len();
        let mut win_counts: Vec<u32> = vec![0; n];
        for (i, &my_fit) in combined_fit.iter().enumerate() {
            for _ in 0..params.tournament_q {
                use rand::RngExt;
                let opp = selection_rng.random_range(0..n);
                if my_fit < combined_fit[opp] {
                    win_counts[i] += 1;
                }
            }
        }

        // Sort by (win_count desc, fitness asc) and pick top μ.
        let mut indexed: Vec<usize> = (0..n).collect();
        indexed.sort_by(|&a, &b| {
            win_counts[b].cmp(&win_counts[a]).then_with(|| {
                combined_fit[a]
                    .partial_cmp(&combined_fit[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });
        indexed.truncate(mu);
        #[allow(clippy::cast_possible_wrap)]
        let survivor_idx: Vec<i64> = indexed.iter().map(|&i| i as i64).collect();

        let idx_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::new(survivor_idx.clone(), [mu]), &device);
        let next_parents = combined_pop.select(0, idx_tensor.clone());
        let next_sigmas = combined_sigmas.select(0, idx_tensor);
        let next_fitness: Vec<f32> = survivor_idx
            .iter()
            .map(|&i| {
                #[allow(clippy::cast_sign_loss)]
                combined_fit[i as usize]
            })
            .collect();

        state.parents = next_parents;
        state.sigmas = next_sigmas;
        state.parent_fitness = next_fitness;
        state.generation += 1;
        update_best(&mut state, &offspring, &fitness_host);
        let m =
            StrategyMetrics::from_host_fitness(state.generation, &fitness_host, state.best_fitness);
        state.best_fitness = m.best_fitness_ever;
        (state, m)
    }

    fn best(&self, state: &EpState<B>) -> Option<(Tensor<B, 2>, f32)> {
        state
            .best_genome
            .as_ref()
            .map(|g| (g.clone(), state.best_fitness))
    }
}

fn update_best<B: Backend>(state: &mut EpState<B>, pop: &Tensor<B, 2>, fitness: &[f32]) {
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
    use crate::fitness::FromFitnessEvaluable;
    use crate::strategy::EvolutionaryHarness;
    use burn::backend::NdArray;
    use rlevo_benchmarks::agent::FitnessEvaluable;
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
    fn ep_converges_on_sphere_d2() {
        let device = Default::default();
        let params = EpConfig::default_for(10, 2);
        let fitness_fn = FromFitnessEvaluable::new(SphereFit, Sphere);
        let mut harness = EvolutionaryHarness::<TestBackend, _, _>::new(
            EvolutionaryProgramming::<TestBackend>::new(),
            params,
            fitness_fn,
            3,
            device,
            300,
        );
        harness.reset();
        loop {
            if harness.step(()).done {
                break;
            }
        }
        let best = harness.latest_metrics().unwrap().best_fitness_ever;
        assert!(best < 1e-2, "EP Sphere-D2 best={best}");
    }

    #[test]
    fn ep_converges_on_sphere_d10() {
        let device = Default::default();
        let params = EpConfig::default_for(20, 10);
        let fitness_fn = FromFitnessEvaluable::new(SphereFit, Sphere);
        let mut harness = EvolutionaryHarness::<TestBackend, _, _>::new(
            EvolutionaryProgramming::<TestBackend>::new(),
            params,
            fitness_fn,
            5,
            device,
            2000,
        );
        harness.reset();
        loop {
            if harness.step(()).done {
                break;
            }
        }
        let best = harness.latest_metrics().unwrap().best_fitness_ever;
        assert!(best < 1e-4, "EP Sphere-D10 best={best}");
    }
}
