//! Ant Colony Optimization for continuous domains (ACO_R).
//!
//! Socha & Dorigo's 2008 extension of ACO to real-valued search spaces.
//! The colony maintains an **archive** of the `k` best solutions seen
//! so far; per generation it samples `m` offspring by drawing each
//! dimension from a Gaussian kernel centred on an archive solution
//! selected by rank-weighted roulette:
//!
//! 1. Compute per-archive, per-dim `σ_{l,d} = ξ · mean_e |x_{e,d} − x_{l,d}|`.
//! 2. For every offspring `i` and every dimension `d`:
//!    - Sample an archive index `l ∼ Categorical(w)` where
//!      `w_l ∝ exp(−(rank_l − 1)² / (2·q²·k²))`.
//!    - Sample `x_{i,d} ∼ N(x_{l,d}, σ_{l,d})`.
//! 3. Evaluate offspring, merge with archive, keep top `k`.
//!
//! # References
//!
//! - Socha & Dorigo (2008), *Ant colony optimization for continuous domains*.

use std::f32::consts::PI;
use std::marker::PhantomData;

use burn::tensor::{Distribution, Int, Tensor, TensorData, backend::Backend};
use rand::Rng;
use rand_distr::{Distribution as RandDistDist, Normal};

use crate::rng::{SeedPurpose, seed_stream};
use crate::strategy::{Strategy, StrategyMetrics};

/// Static configuration for [`AntColonyReal`].
#[derive(Debug, Clone)]
pub struct AcoRConfig {
    /// Archive size (number of "best" solutions kept). Socha & Dorigo's
    /// recommended default is `k = 50`.
    pub archive_size: usize,
    /// Offspring per generation (`m` in the paper).
    pub m: usize,
    /// Genome dimensionality.
    pub genome_dim: usize,
    /// Search-space bounds.
    pub bounds: (f32, f32),
    /// Exploration scale (`ξ`). Higher → wider sampling. Canonical 0.85.
    pub xi: f32,
    /// Rank-weight decay (`q`). Smaller → stronger bias toward top of
    /// the archive. Canonical `q = 0.01` (sharp) up to `q ≈ 0.5` (flat).
    pub q: f32,
}

impl AcoRConfig {
    /// Default configuration for a given archive size, offspring count, and dimensionality.
    #[must_use]
    pub fn default_for(archive_size: usize, m: usize, genome_dim: usize) -> Self {
        Self {
            archive_size,
            m,
            genome_dim,
            bounds: (-5.12, 5.12),
            xi: 0.85,
            q: 0.1,
        }
    }

    /// Steady-state offspring count per generation (`m`). Note that
    /// the very first generation evaluates the full initial archive
    /// (`archive_size` rows) instead — only generations ≥ 1 score `m`.
    #[must_use]
    pub fn steady_state_pop_size(&self) -> usize {
        self.m
    }
}

/// Generation state for [`AntColonyReal`].
#[derive(Debug, Clone)]
pub struct AcoRState<B: Backend> {
    /// Archive of `k` best solutions, shape `(k, D)`.
    pub archive: Tensor<B, 2>,
    /// Host-side archive fitness, sorted ascending (best first) after
    /// the first `tell`.
    pub archive_fitness: Vec<f32>,
    /// Cached archive weights (recomputed only when `q` or `k` change).
    pub weights: Vec<f32>,
    /// Best-so-far genome.
    pub best_genome: Option<Tensor<B, 2>>,
    /// Best-so-far fitness.
    pub best_fitness: f32,
    /// Generation counter.
    pub generation: usize,
}

/// Ant Colony Optimization (continuous domains).
///
/// # Panics
///
/// [`Strategy::init`] panics if `params.archive_size < 2` (the σ
/// computation needs at least two archive solutions to take a pairwise
/// distance) or if `params.m < 1` (no offspring to draw).
///
/// # Example
///
/// ```no_run
/// use burn::backend::NdArray;
/// use rlevo_evolution::algorithms::metaheuristic::aco_r::{AcoRConfig, AntColonyReal};
///
/// let strategy = AntColonyReal::<NdArray>::new();
/// let params = AcoRConfig::default_for(50, 10, 10);
/// let _ = (strategy, params);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct AntColonyReal<B: Backend> {
    _backend: PhantomData<fn() -> B>,
}

impl<B: Backend> AntColonyReal<B> {
    /// Builds a new (stateless) strategy object.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _backend: PhantomData,
        }
    }

    /// Compute rank-based archive weights `w_l ∝ exp(−(l−1)² / (2·q²·k²))`.
    fn compute_weights(archive_size: usize, q: f32) -> Vec<f32> {
        #[allow(clippy::cast_precision_loss)]
        let k = archive_size as f32;
        let denom = 2.0 * q * q * k * k;
        let scale = 1.0 / (q * k * (2.0 * PI).sqrt());
        let mut w: Vec<f32> = (0..archive_size)
            .map(|l| {
                #[allow(clippy::cast_precision_loss)]
                let rank = l as f32;
                scale * (-(rank * rank) / denom).exp()
            })
            .collect();
        let total: f32 = w.iter().sum();
        for v in &mut w {
            *v /= total;
        }
        w
    }
}

impl<B: Backend> Strategy<B> for AntColonyReal<B>
where
    B::Device: Clone,
{
    type Params = AcoRConfig;
    type State = AcoRState<B>;
    type Genome = Tensor<B, 2>;

    fn init(&self, params: &AcoRConfig, rng: &mut dyn Rng, device: &B::Device) -> AcoRState<B> {
        assert!(params.archive_size >= 2, "ACO_R requires archive_size >= 2");
        assert!(params.m >= 1, "ACO_R requires m >= 1");
        let (lo, hi) = params.bounds;
        B::seed(device, rng.next_u64());
        let archive = Tensor::<B, 2>::random(
            [params.archive_size, params.genome_dim],
            Distribution::Uniform(f64::from(lo), f64::from(hi)),
            device,
        );
        AcoRState {
            archive,
            archive_fitness: Vec::new(),
            weights: Self::compute_weights(params.archive_size, params.q),
            best_genome: None,
            best_fitness: f32::INFINITY,
            generation: 0,
        }
    }

    fn ask(
        &self,
        params: &AcoRConfig,
        state: &AcoRState<B>,
        rng: &mut dyn Rng,
        device: &B::Device,
    ) -> (Tensor<B, 2>, AcoRState<B>) {
        // First call: evaluate the initial archive.
        if state.archive_fitness.is_empty() {
            return (state.archive.clone(), state.clone());
        }

        let k = params.archive_size;
        let m = params.m;
        let d = params.genome_dim;

        // σ[l, j] = ξ · (1/(k-1)) · Σ_e |archive[e, j] - archive[l, j]|
        // Computed on-device by expanding archive along axis 0 to (k, k, d),
        // taking |a - b|, reducing along axis 0 (the "e" axis).
        let archive_l = state.archive.clone().unsqueeze_dim::<3>(0); // (1, k, d)
        let archive_e = state.archive.clone().unsqueeze_dim::<3>(1); // (k, 1, d)
        let diffs = (archive_l.expand([k, k, d]) - archive_e.expand([k, k, d])).abs();
        #[allow(clippy::cast_precision_loss)]
        let inv = params.xi / ((k - 1).max(1) as f32);
        let sigma = diffs.sum_dim(0).squeeze::<2>().mul_scalar(inv); // (k, d)

        // Weighted index sampling (host-side) — `m · d` independent draws.
        let mut stream = seed_stream(
            rng.next_u64(),
            state.generation as u64,
            SeedPurpose::Selection,
        );
        let mut mean_rows = vec![0f32; m * d];
        let mut sigma_rows = vec![0f32; m * d];

        // Gather host-side slices for indexing.
        let archive_host = state.archive.clone().into_data().into_vec::<f32>().unwrap();
        let sigma_host = sigma.into_data().into_vec::<f32>().unwrap();
        let cdf: Vec<f32> = {
            let mut acc = 0.0;
            let mut v = Vec::with_capacity(k);
            for &w in &state.weights {
                acc += w;
                v.push(acc);
            }
            v
        };
        let pick = |u: f32| -> usize { cdf.iter().position(|&c| u <= c).unwrap_or(k - 1) };

        for i in 0..m {
            for j in 0..d {
                use rand::RngExt;
                let u: f32 = stream.random::<f32>();
                let l = pick(u);
                mean_rows[i * d + j] = archive_host[l * d + j];
                sigma_rows[i * d + j] = sigma_host[l * d + j].max(1e-12);
            }
        }

        // Sample N(mean, sigma) host-side. Using rand_distr keeps the
        // draw on the same splitmix stream already threaded through
        // `stream` above.
        let mut offspring = vec![0f32; m * d];
        let mut sample_rng = seed_stream(
            rng.next_u64(),
            state.generation as u64,
            SeedPurpose::Mutation,
        );
        for (idx, out) in offspring.iter_mut().enumerate() {
            let normal = Normal::new(mean_rows[idx], sigma_rows[idx]).expect("sigma > 0");
            *out = normal.sample(&mut sample_rng);
        }
        let (lo, hi) = params.bounds;
        for v in &mut offspring {
            *v = v.clamp(lo, hi);
        }
        let new_pop = Tensor::<B, 2>::from_data(TensorData::new(offspring, [m, d]), device);

        (new_pop, state.clone())
    }

    fn tell(
        &self,
        params: &AcoRConfig,
        population: Tensor<B, 2>,
        fitness: Tensor<B, 1>,
        mut state: AcoRState<B>,
        _rng: &mut dyn Rng,
    ) -> (AcoRState<B>, StrategyMetrics) {
        let fitness_host = fitness.into_data().into_vec::<f32>().unwrap_or_default();
        let device = population.device();
        let k = params.archive_size;

        // First tell: the population being scored IS the initial archive.
        if state.archive_fitness.is_empty() {
            // Sort archive by fitness.
            let mut idx: Vec<usize> = (0..fitness_host.len()).collect();
            idx.sort_by(|&a, &b| fitness_host[a].partial_cmp(&fitness_host[b]).unwrap());
            #[allow(clippy::cast_possible_wrap)]
            let sorted_idx = Tensor::<B, 1, Int>::from_data(
                TensorData::new(idx.iter().map(|&i| i as i64).collect::<Vec<_>>(), [k]),
                &device,
            );
            state.archive = population.clone().select(0, sorted_idx);
            state.archive_fitness = idx.iter().map(|&i| fitness_host[i]).collect();
            state.best_fitness = state.archive_fitness[0];
            let first_idx =
                Tensor::<B, 1, Int>::from_data(TensorData::new(vec![0_i64], [1]), &device);
            state.best_genome = Some(state.archive.clone().select(0, first_idx));
            state.generation += 1;
            let m = StrategyMetrics::from_host_fitness(
                state.generation,
                &fitness_host,
                state.best_fitness,
            );
            state.best_fitness = m.best_fitness_ever;
            return (state, m);
        }

        // Steady state: merge archive + offspring, keep top-k.
        let combined = Tensor::cat(vec![state.archive.clone(), population.clone()], 0);
        let mut combined_f: Vec<f32> = state.archive_fitness.clone();
        combined_f.extend_from_slice(&fitness_host);
        let mut idx: Vec<usize> = (0..combined_f.len()).collect();
        idx.sort_by(|&a, &b| combined_f[a].partial_cmp(&combined_f[b]).unwrap());
        idx.truncate(k);
        #[allow(clippy::cast_possible_wrap)]
        let top_idx = Tensor::<B, 1, Int>::from_data(
            TensorData::new(idx.iter().map(|&i| i as i64).collect::<Vec<_>>(), [k]),
            &device,
        );
        state.archive = combined.select(0, top_idx);
        state.archive_fitness = idx.iter().map(|&i| combined_f[i]).collect();

        if state.archive_fitness[0] < state.best_fitness {
            state.best_fitness = state.archive_fitness[0];
            let first_idx =
                Tensor::<B, 1, Int>::from_data(TensorData::new(vec![0_i64], [1]), &device);
            state.best_genome = Some(state.archive.clone().select(0, first_idx));
        }

        state.generation += 1;
        let m =
            StrategyMetrics::from_host_fitness(state.generation, &fitness_host, state.best_fitness);
        state.best_fitness = m.best_fitness_ever;
        (state, m)
    }

    fn best(&self, state: &AcoRState<B>) -> Option<(Tensor<B, 2>, f32)> {
        state
            .best_genome
            .as_ref()
            .map(|g| (g.clone(), state.best_fitness))
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
    fn weights_sum_to_one() {
        let w = AntColonyReal::<TestBackend>::compute_weights(10, 0.1);
        let total: f32 = w.iter().sum();
        approx::assert_relative_eq!(total, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn aco_r_converges_on_sphere_d10() {
        let device = Default::default();
        let strategy = AntColonyReal::<TestBackend>::new();
        let params = AcoRConfig::default_for(30, 15, 10);
        let fitness_fn = FromFitnessEvaluable::new(SphereFit, Sphere);
        let mut harness = EvolutionaryHarness::<TestBackend, _, _>::new(
            strategy, params, fitness_fn, 17, device, 400,
        );
        harness.reset();
        while !harness.step(()).done {}
        let best = harness.latest_metrics().unwrap().best_fitness_ever;
        assert!(best < 1e-3, "ACO_R D10 best={best}");
    }
}
