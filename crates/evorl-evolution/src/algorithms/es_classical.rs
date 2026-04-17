//! Classical Evolution Strategies.
//!
//! Four canonical variants parameterized on a single [`EsConfig`]:
//!
//! - `(1+1)` — a single parent, a single offspring, 1/5th success-rule
//!   σ adaptation.
//! - `(1+λ)` — a single parent, λ offspring per generation; the best
//!   offspring replaces the parent iff its fitness improves. Also used
//!   by Cartesian GP (landing in M3).
//! - `(μ,λ)` — μ parents, λ offspring; parents are discarded each
//!   generation.
//! - `(μ+λ)` — μ parents, λ offspring; survivors are the μ best of the
//!   combined pool.
//!
//! σ adaptation is by log-normal self-adaptation in the multi-parent
//! variants; `(1+1)` uses Rechenberg's 1/5th success rule.
//!
//! # References
//!
//! - Beyer & Schwefel (2002), *Evolution strategies: A comprehensive
//!   introduction*.
//! - Rechenberg (1973), *Evolutionsstrategie*.

use std::marker::PhantomData;

use burn::tensor::{backend::Backend, Tensor, TensorData};
use rand::Rng;

use crate::ops::mutation::gaussian_mutation_per_row;
use crate::ops::replacement::{mu_comma_lambda, mu_plus_lambda};
use crate::rng::{seed_stream, SeedPurpose};
use crate::strategy::{Strategy, StrategyMetrics};

/// Which selection scheme the ES uses.
#[derive(Debug, Clone, Copy)]
pub enum EsKind {
    /// `(1+1)` with 1/5-rule σ adaptation.
    OnePlusOne,
    /// `(1+λ)` with shared σ across offspring.
    OnePlusLambda { lambda: usize },
    /// `(μ,λ)` with log-normal per-individual σ adaptation.
    MuCommaLambda { mu: usize, lambda: usize },
    /// `(μ+λ)` with log-normal per-individual σ adaptation.
    MuPlusLambda { mu: usize, lambda: usize },
}

impl EsKind {
    #[must_use]
    pub fn population_size(&self) -> usize {
        match self {
            EsKind::OnePlusOne => 1,
            EsKind::OnePlusLambda { lambda }
            | EsKind::MuCommaLambda { lambda, .. }
            | EsKind::MuPlusLambda { lambda, .. } => *lambda,
        }
    }
}

/// Static configuration for an [`EvolutionStrategy`] run.
#[derive(Debug, Clone)]
pub struct EsConfig {
    /// Variant to run.
    pub kind: EsKind,
    /// Genome dimensionality.
    pub genome_dim: usize,
    /// Search-space bounds; used for initialization and clamping.
    pub bounds: (f32, f32),
    /// Initial σ (log-normal self-adaptation modifies it in state).
    pub initial_sigma: f32,
    /// Learning-rate scale for log-normal σ update. Standard default is
    /// `1.0 / sqrt(2 * sqrt(D))`.
    pub tau: f32,
}

impl EsConfig {
    /// Default configuration for a given ES variant and dimensionality.
    #[must_use]
    pub fn default_for(kind: EsKind, genome_dim: usize) -> Self {
        #[allow(clippy::cast_precision_loss)]
        let d = genome_dim as f32;
        let tau = 1.0 / (2.0 * d.sqrt()).sqrt();
        Self {
            kind,
            genome_dim,
            bounds: (-5.12, 5.12),
            initial_sigma: 1.0,
            tau,
        }
    }
}

/// Generation state for [`EvolutionStrategy`].
#[derive(Debug, Clone)]
pub struct EsState<B: Backend> {
    /// Parent population. `(μ, D)` for μ-parent variants; `(1, D)` for
    /// (1+1) and (1+λ).
    pub parents: Tensor<B, 2>,
    /// Per-parent σ values. `(μ,)` shape for log-normal adaptation;
    /// `(1,)` shape for (1+1)/(1+λ) with shared σ.
    pub sigmas: Tensor<B, 1>,
    /// Parent fitnesses.
    pub parent_fitness: Vec<f32>,
    /// Best-so-far genome, shape `(1, D)`.
    pub best_genome: Option<Tensor<B, 2>>,
    /// Best-so-far fitness.
    pub best_fitness: f32,
    /// Completed-generation counter.
    pub generation: usize,
    /// (1+1) only: running success-rate counter for the 1/5th rule.
    pub successes_in_window: u32,
    /// (1+1) only: window length observed so far.
    pub window_len: u32,
}

/// Classical Evolution Strategy.
///
/// # Example
///
/// ```no_run
/// use burn::backend::NdArray;
/// use evorl_evolution::algorithms::es_classical::{EsConfig, EsKind, EvolutionStrategy};
///
/// let strategy = EvolutionStrategy::<NdArray>::new();
/// let params = EsConfig::default_for(EsKind::MuPlusLambda { mu: 5, lambda: 20 }, 10);
/// let _ = (strategy, params);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct EvolutionStrategy<B: Backend> {
    _backend: PhantomData<fn() -> B>,
}

impl<B: Backend> EvolutionStrategy<B> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            _backend: PhantomData,
        }
    }

    fn mu(kind: EsKind) -> usize {
        match kind {
            EsKind::OnePlusOne | EsKind::OnePlusLambda { .. } => 1,
            EsKind::MuCommaLambda { mu, .. } | EsKind::MuPlusLambda { mu, .. } => mu,
        }
    }

    fn sample_initial_parents(
        params: &EsConfig,
        rng: &mut dyn Rng,
        device: &B::Device,
    ) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let mu = Self::mu(params.kind);
        let (lo, hi) = params.bounds;
        B::seed(device, rng.next_u64());
        let parents = Tensor::<B, 2>::random(
            [mu, params.genome_dim],
            burn::tensor::Distribution::Uniform(f64::from(lo), f64::from(hi)),
            device,
        );
        let sigmas = Tensor::<B, 1>::from_data(
            TensorData::new(vec![params.initial_sigma; mu], [mu]),
            device,
        );
        (parents, sigmas)
    }
}

impl<B: Backend> Strategy<B> for EvolutionStrategy<B>
where
    B::Device: Clone,
{
    type Params = EsConfig;
    type State = EsState<B>;
    type Genome = Tensor<B, 2>;

    fn init(
        &self,
        params: &EsConfig,
        rng: &mut dyn Rng,
        device: &B::Device,
    ) -> EsState<B> {
        let (parents, sigmas) = Self::sample_initial_parents(params, rng, device);
        EsState {
            parents,
            sigmas,
            parent_fitness: Vec::new(),
            best_genome: None,
            best_fitness: f32::INFINITY,
            generation: 0,
            successes_in_window: 0,
            window_len: 0,
        }
    }

    fn ask(
        &self,
        params: &EsConfig,
        state: &EsState<B>,
        rng: &mut dyn Rng,
        device: &B::Device,
    ) -> (Tensor<B, 2>, EsState<B>) {
        // First call: evaluate the initial parents as the "offspring"
        // so fitness is populated in the subsequent `tell`.
        if state.parent_fitness.is_empty() {
            return (state.parents.clone(), state.clone());
        }

        let lambda = params.kind.population_size();
        let mu = Self::mu(params.kind);

        let mut mutation_rng =
            seed_stream(rng.next_u64(), state.generation as u64, SeedPurpose::Mutation);
        let mut sigma_rng =
            seed_stream(rng.next_u64(), state.generation as u64, SeedPurpose::Other);

        // Build an offspring population of size λ by sampling a parent
        // index per offspring and mutating. Uniform random parent
        // selection — no fitness pressure applied at this stage in
        // classical ES; survivor selection provides the pressure.
        let mut parent_indices: Vec<i64> = Vec::with_capacity(lambda);
        {
            use rand::RngExt;
            for _ in 0..lambda {
                #[allow(clippy::cast_possible_wrap)]
                parent_indices.push(sigma_rng.random_range(0..mu) as i64);
            }
        }
        let idx_tensor = Tensor::<B, 1, burn::tensor::Int>::from_data(
            TensorData::new(parent_indices.clone(), [lambda]),
            device,
        );
        let duplicated_parents = state.parents.clone().select(0, idx_tensor.clone());
        let duplicated_sigmas = state.sigmas.clone().select(0, idx_tensor);

        // Apply log-normal σ adaptation (multi-parent case) or keep σ
        // shared (1+1 / 1+λ). Log-normal: σ' = σ * exp(τ · N(0,1)).
        let is_one_plus = matches!(
            params.kind,
            EsKind::OnePlusOne | EsKind::OnePlusLambda { .. }
        );
        let offspring_sigmas = if is_one_plus {
            duplicated_sigmas
        } else {
            B::seed(device, sigma_rng.next_u64());
            let noise =
                Tensor::<B, 1>::random([lambda], burn::tensor::Distribution::Normal(0.0, 1.0), device);
            duplicated_sigmas * noise.mul_scalar(params.tau).exp()
        };

        // Mutate parents by the per-offspring σ.
        B::seed(device, mutation_rng.next_u64());
        let mutated = gaussian_mutation_per_row(duplicated_parents, offspring_sigmas.clone(), device);

        // Clamp to bounds.
        let (lo, hi) = params.bounds;
        let mutated = mutated.clamp(lo, hi);

        let mut state = state.clone();
        // Stash the offspring σ values into state so `tell` can promote
        // the survivors' σ alongside their genomes. We use `parents`
        // here to sneak the offspring σ through; a proper refactor
        // would carry a dedicated "pending offspring σ" field, but
        // storing it on `state.sigmas` keeps the State small (see
        // strategy.rs invariant about pure ask/tell).
        // To avoid ambiguity, we actually overwrite sigmas only in
        // `tell`. We therefore stash the offspring sigmas in a
        // scratchpad via cloning into `state.sigmas` is NOT correct.
        // Instead, we leverage the fact that `ask` returns `state` and
        // `tell` re-derives the σ from the variant-specific rule.
        // Pragmatically we let `tell` receive the offspring_sigmas via
        // state.sigmas *only* when the caller keeps it synchronized.
        // To keep this correct, we attach offspring σ to `state` via
        // the `sigmas` field: survivors in mu_plus_lambda pull indices
        // against a union, so we store the concatenation of parent +
        // offspring σ.
        let combined_sigmas = Tensor::cat(vec![state.sigmas.clone(), offspring_sigmas], 0);
        state.sigmas = combined_sigmas;
        (mutated, state)
    }

    fn tell(
        &self,
        params: &EsConfig,
        offspring: Tensor<B, 2>,
        fitness: Tensor<B, 1>,
        mut state: EsState<B>,
        _rng: &mut dyn Rng,
    ) -> (EsState<B>, StrategyMetrics) {
        let fitness_host = fitness.into_data().into_vec::<f32>().unwrap_or_default();

        // First `tell` after `init`: offspring here is actually the
        // initial parent population evaluated.
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
            // Restore parent-count σ vector.
            let mu = Self::mu(params.kind);
            let device = state.parents.device();
            state.sigmas = Tensor::<B, 1>::from_data(
                TensorData::new(vec![params.initial_sigma; mu], [mu]),
                &device,
            );
            return (state, m);
        }

        let device = offspring.device();
        let mu = Self::mu(params.kind);
        // state.sigmas currently holds parent σ concatenated with
        // offspring σ, per `ask`'s scratchpad trick.
        let lambda = params.kind.population_size();
        let parent_sigmas = state.sigmas.clone().slice([0..mu]);
        let offspring_sigmas = state.sigmas.clone().slice([mu..(mu + lambda)]);

        match params.kind {
            EsKind::OnePlusOne => {
                // One parent, one offspring. Fitness[0] is the offspring.
                let parent_fit = state.parent_fitness[0];
                let offspring_fit = fitness_host[0];
                let success = offspring_fit < parent_fit;
                state.window_len += 1;
                if success {
                    state.successes_in_window += 1;
                    state.parents = offspring.clone();
                    state.parent_fitness = vec![offspring_fit];
                }
                // Rechenberg 1/5-rule every 10 · D generations.
                #[allow(clippy::cast_precision_loss)]
                let window = 10_u32.saturating_mul(params.genome_dim as u32).max(1);
                if state.window_len >= window {
                    #[allow(clippy::cast_precision_loss)]
                    let rate = state.successes_in_window as f32 / state.window_len as f32;
                    let current_sigma = state.sigmas.clone().into_data().into_vec::<f32>().unwrap()[0];
                    let new_sigma = if rate > 0.2 {
                        current_sigma * 1.22
                    } else if rate < 0.2 {
                        current_sigma / 1.22
                    } else {
                        current_sigma
                    };
                    state.sigmas = Tensor::<B, 1>::from_data(
                        TensorData::new(vec![new_sigma], [1]),
                        &device,
                    );
                    state.successes_in_window = 0;
                    state.window_len = 0;
                } else {
                    state.sigmas = parent_sigmas;
                }
            }
            EsKind::OnePlusLambda { .. } => {
                // Best of (parent, offspring pool).
                let best_off_idx = argmin(&fitness_host);
                let best_off_fit = fitness_host[best_off_idx];
                if best_off_fit < state.parent_fitness[0] {
                    state.parents = offspring
                        .clone()
                        .slice([best_off_idx..best_off_idx + 1]);
                    state.parent_fitness = vec![best_off_fit];
                }
                state.sigmas = parent_sigmas;
            }
            EsKind::MuCommaLambda { mu, .. } => {
                let (survivors, survivor_f) =
                    mu_comma_lambda::<B>(offspring.clone(), &fitness_host, mu, &device);
                // Gather survivor σs matching the same indices.
                let survivor_idx = crate::ops::selection::truncation_indices_host(
                    &fitness_host,
                    mu,
                );
                let survivor_sigmas = offspring_sigmas.select(
                    0,
                    Tensor::<B, 1, burn::tensor::Int>::from_data(
                        TensorData::new(survivor_idx, [mu]),
                        &device,
                    ),
                );
                state.parents = survivors;
                state.parent_fitness = survivor_f;
                state.sigmas = survivor_sigmas;
            }
            EsKind::MuPlusLambda { mu, .. } => {
                let (survivors, survivor_f) = mu_plus_lambda::<B>(
                    state.parents.clone(),
                    &state.parent_fitness,
                    offspring.clone(),
                    &fitness_host,
                    mu,
                    &device,
                );
                // Survivor σ via truncation_indices_host on the combined fitness.
                let combined_f: Vec<f32> = state
                    .parent_fitness
                    .iter()
                    .chain(fitness_host.iter())
                    .copied()
                    .collect();
                let survivor_idx = crate::ops::selection::truncation_indices_host(&combined_f, mu);
                let combined_sigmas = Tensor::cat(vec![parent_sigmas, offspring_sigmas], 0);
                let survivor_sigmas = combined_sigmas.select(
                    0,
                    Tensor::<B, 1, burn::tensor::Int>::from_data(
                        TensorData::new(survivor_idx, [mu]),
                        &device,
                    ),
                );
                state.parents = survivors;
                state.parent_fitness = survivor_f;
                state.sigmas = survivor_sigmas;
            }
        }

        state.generation += 1;
        update_best(&mut state, &offspring, &fitness_host);
        let m = StrategyMetrics::from_host_fitness(
            state.generation,
            &fitness_host,
            state.best_fitness,
        );
        state.best_fitness = m.best_fitness_ever;
        (state, m)
    }

    fn best(&self, state: &EsState<B>) -> Option<(Tensor<B, 2>, f32)> {
        state
            .best_genome
            .as_ref()
            .map(|g| (g.clone(), state.best_fitness))
    }
}

fn argmin(xs: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best = f32::INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v < best {
            best = v;
            best_idx = i;
        }
    }
    best_idx
}

fn update_best<B: Backend>(state: &mut EsState<B>, pop: &Tensor<B, 2>, fitness: &[f32]) {
    if fitness.is_empty() {
        return;
    }
    let best_idx = argmin(fitness);
    let best_f = fitness[best_idx];
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
    use evorl_benchmarks::agent::FitnessEvaluable;
    use evorl_benchmarks::env::BenchEnv;
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

    fn run_es(kind: EsKind, dim: usize, generations: usize, seed: u64) -> f32 {
        let device = Default::default();
        let strategy = EvolutionStrategy::<TestBackend>::new();
        let params = EsConfig::default_for(kind, dim);
        let fitness_fn = FromFitnessEvaluable::new(SphereFit, Sphere);
        let mut harness = EvolutionaryHarness::<TestBackend, _, _>::new(
            strategy, params, fitness_fn, seed, device, generations,
        );
        harness.reset();
        loop {
            let step = harness.step(());
            if step.done {
                break;
            }
        }
        harness.latest_metrics().unwrap().best_fitness_ever
    }

    #[test]
    fn one_plus_lambda_converges_on_sphere_d2() {
        let best = run_es(EsKind::OnePlusLambda { lambda: 8 }, 2, 200, 7);
        assert!(best < 1e-2, "OnePlusLambda best={best}");
    }

    #[test]
    fn one_plus_one_converges_on_sphere_d2() {
        let best = run_es(EsKind::OnePlusOne, 2, 500, 11);
        assert!(best < 1e-2, "OnePlusOne best={best}");
    }

    #[test]
    fn mu_plus_lambda_converges_on_sphere_d2() {
        let best = run_es(EsKind::MuPlusLambda { mu: 3, lambda: 8 }, 2, 200, 7);
        assert!(best < 1e-2, "MuPlusLambda best={best}");
    }

    #[test]
    fn mu_comma_lambda_converges_on_sphere_d2() {
        let best = run_es(EsKind::MuCommaLambda { mu: 3, lambda: 8 }, 2, 200, 7);
        assert!(best < 1e-1, "MuCommaLambda best={best}");
    }

    #[test]
    fn mu_plus_lambda_converges_on_sphere_d10() {
        // Spec §12.2: each family must converge on Sphere (D=10) to
        // best_fitness < 1e-6 within budget on ndarray. We allow a
        // generous budget because the classical ES is slower than
        // CMA-ES; the goal is to verify convergence direction, not to
        // optimize hyperparameters.
        let best = run_es(EsKind::MuPlusLambda { mu: 5, lambda: 20 }, 10, 1500, 42);
        assert!(best < 1e-6, "MuPlusLambda D10 best={best}");
    }
}
