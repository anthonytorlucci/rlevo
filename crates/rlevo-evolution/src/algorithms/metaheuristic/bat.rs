//! Bat Algorithm.
//!
//! Each bat carries a position, a velocity, a frequency `f`, a loudness
//! `A`, and a pulse rate `r`. Per generation:
//!
//! 1. Sample `f_i = f_min + (f_max − f_min)·β`, `β ∈ U[0, 1]`.
//! 2. Update velocity: `v_i ← v_i + (x_i − x_best)·f_i`.
//! 3. Propose candidate: `x'_i = x_i + v_i`. If `rand > r_i`, override
//!    with a local walk `x'_i = x_best + ε · mean(A)`, `ε ∈ U[−1, 1]`.
//! 4. `tell` accepts the candidate iff
//!    `rand < A_i` **and** `f(x'_i) ≤ f(x_i)`. On acceptance:
//!    `A_i *= α` (decay loudness), `r_i = r_{i,0}·(1 − exp(−γ·t))`
//!    (grow pulse rate).
//!
//! # Candor
//!
//! Legacy comparator. The velocity/position update is structurally a
//! PSO variant toward the global best; the probabilistic acceptance
//! adds simulated-annealing-style noise. Camacho Villalón et al. (2020)
//! discuss the lack of search mechanisms not already present in
//! simpler algorithms. Ship it for API coverage; prefer CMA-ES or
//! LSHADE when available.
//!
//! # References
//!
//! - Yang (2010), *A New Metaheuristic Bat-Inspired Algorithm*.

use std::marker::PhantomData;

use burn::tensor::{Distribution, Int, Tensor, TensorData, backend::Backend};
use rand::Rng;
use rand::RngExt;

use crate::rng::{SeedPurpose, seed_stream};
use crate::strategy::{Strategy, StrategyMetrics};

/// Static configuration for [`BatAlgorithm`].
#[derive(Debug, Clone)]
pub struct BatConfig {
    /// Number of bats.
    pub pop_size: usize,
    /// Genome dimensionality.
    pub genome_dim: usize,
    /// Search-space bounds.
    pub bounds: (f32, f32),
    /// Minimum frequency.
    pub f_min: f32,
    /// Maximum frequency.
    pub f_max: f32,
    /// Initial loudness.
    pub a0: f32,
    /// Initial pulse rate.
    pub r0: f32,
    /// Loudness decay factor (0 < α ≤ 1). Canonical `α = 0.9`.
    pub alpha: f32,
    /// Pulse-rate growth factor (γ > 0). Canonical `γ = 0.9`.
    pub gamma: f32,
}

impl BatConfig {
    /// Default configuration for a given population size and genome dimensionality.
    #[must_use]
    pub fn default_for(pop_size: usize, genome_dim: usize) -> Self {
        Self {
            pop_size,
            genome_dim,
            bounds: (-5.12, 5.12),
            f_min: 0.0,
            f_max: 2.0,
            a0: 1.0,
            r0: 0.5,
            alpha: 0.9,
            gamma: 0.9,
        }
    }
}

/// Generation state for [`BatAlgorithm`].
#[derive(Debug, Clone)]
pub struct BatState<B: Backend> {
    /// Current positions, shape `(pop_size, D)`.
    pub positions: Tensor<B, 2>,
    /// Current velocities, shape `(pop_size, D)`.
    pub velocities: Tensor<B, 2>,
    /// Per-bat loudness.
    pub loudness: Vec<f32>,
    /// Per-bat pulse rate.
    pub pulse_rate: Vec<f32>,
    /// Host-side fitness cache for the current positions.
    pub fitness: Vec<f32>,
    /// Best-so-far genome.
    pub best_genome: Option<Tensor<B, 2>>,
    /// Best-so-far fitness.
    pub best_fitness: f32,
    /// Generation counter.
    pub generation: usize,
    /// Per-generation "accept this candidate?" decisions recorded in
    /// `ask` so `tell` can gate the loudness/pulse updates consistently
    /// with the RNG draws.
    pub pending_accept: Vec<bool>,
}

/// Bat Algorithm strategy.
///
/// # Example
///
/// ```no_run
/// use burn::backend::NdArray;
/// use rlevo_evolution::algorithms::metaheuristic::bat::{BatAlgorithm, BatConfig};
///
/// let strategy = BatAlgorithm::<NdArray>::new();
/// let params = BatConfig::default_for(32, 10);
/// let _ = (strategy, params);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct BatAlgorithm<B: Backend> {
    _backend: PhantomData<fn() -> B>,
}

impl<B: Backend> BatAlgorithm<B> {
    /// Builds a new (stateless) strategy object.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> Strategy<B> for BatAlgorithm<B>
where
    B::Device: Clone,
{
    type Params = BatConfig;
    type State = BatState<B>;
    type Genome = Tensor<B, 2>;

    fn init(&self, params: &BatConfig, rng: &mut dyn Rng, device: &B::Device) -> BatState<B> {
        let (lo, hi) = params.bounds;
        B::seed(device, rng.next_u64());
        let positions = Tensor::<B, 2>::random(
            [params.pop_size, params.genome_dim],
            Distribution::Uniform(f64::from(lo), f64::from(hi)),
            device,
        );
        let velocities = Tensor::<B, 2>::zeros([params.pop_size, params.genome_dim], device);
        BatState {
            positions,
            velocities,
            loudness: vec![params.a0; params.pop_size],
            pulse_rate: vec![params.r0; params.pop_size],
            fitness: Vec::new(),
            best_genome: None,
            best_fitness: f32::INFINITY,
            generation: 0,
            pending_accept: Vec::new(),
        }
    }

    fn ask(
        &self,
        params: &BatConfig,
        state: &BatState<B>,
        rng: &mut dyn Rng,
        device: &B::Device,
    ) -> (Tensor<B, 2>, BatState<B>) {
        if state.fitness.is_empty() {
            // Evaluate the initial colony first; the velocity update is
            // only defined once a best exists.
            return (state.positions.clone(), state.clone());
        }

        let pop = params.pop_size;
        let genome_dim = params.genome_dim;
        let (lo, hi) = params.bounds;

        // Host-side sampling for β, pulse check, acceptance draw, and
        // local-walk ε. Keeping these on host preserves bit-parity
        // across backends (Mantegna / wgpu normal RNG has documented
        // fp drift under FMA reordering; BA draws are mostly uniform).
        let mut stream = seed_stream(rng.next_u64(), state.generation as u64, SeedPurpose::Other);

        let mut betas = Vec::with_capacity(pop);
        let mut use_local = Vec::with_capacity(pop);
        let mut accept_draw = Vec::with_capacity(pop);
        let mut epsilon_rows = Vec::with_capacity(pop * genome_dim);
        for i in 0..pop {
            betas.push(stream.random::<f32>());
            use_local.push(stream.random::<f32>() > state.pulse_rate[i]);
            accept_draw.push(stream.random::<f32>());
            for _ in 0..genome_dim {
                epsilon_rows.push(2.0 * stream.random::<f32>() - 1.0);
            }
        }

        // Mean loudness across the colony — used by the local-walk
        // step to scale its ε perturbation.
        let mean_loudness: f32 = {
            let s: f32 = state.loudness.iter().sum();
            #[allow(clippy::cast_precision_loss)]
            {
                s / pop as f32
            }
        };

        let best = state
            .best_genome
            .as_ref()
            .expect("best populated after first tell")
            .clone()
            .expand([pop, genome_dim]);

        // Frequency: f_i = f_min + (f_max - f_min) · β_i  → shape (pop, 1) → (pop, D).
        let f_vec: Vec<f32> = betas
            .iter()
            .map(|b| params.f_min + (params.f_max - params.f_min) * b)
            .collect();
        let f_mat = Tensor::<B, 1>::from_data(TensorData::new(f_vec, [pop]), device)
            .unsqueeze_dim::<2>(1)
            .expand([pop, genome_dim]);

        let new_velocities =
            state.velocities.clone() + (state.positions.clone() - best.clone()).mul(f_mat);
        let global_move = state.positions.clone() + new_velocities.clone();
        // Local walk: x_best + ε · mean(A).
        let eps =
            Tensor::<B, 2>::from_data(TensorData::new(epsilon_rows, [pop, genome_dim]), device);
        let local_move = best + eps.mul_scalar(mean_loudness);

        #[allow(clippy::cast_possible_wrap)]
        let mask = Tensor::<B, 1, Int>::from_data(
            TensorData::new(
                use_local.iter().map(|&b| i64::from(b)).collect::<Vec<_>>(),
                [pop],
            ),
            device,
        )
        .equal_elem(1)
        .unsqueeze_dim::<2>(1)
        .expand([pop, genome_dim]);
        let candidates = global_move.mask_where(mask, local_move).clamp(lo, hi);

        // Defer acceptance decisions to tell; record the random draws.
        let mut next = state.clone();
        next.velocities = new_velocities;
        next.pending_accept = accept_draw
            .iter()
            .zip(state.loudness.iter())
            .map(|(&draw, &a)| draw < a)
            .collect();
        (candidates, next)
    }

    fn tell(
        &self,
        params: &BatConfig,
        candidates: Tensor<B, 2>,
        fitness: Tensor<B, 1>,
        mut state: BatState<B>,
        _rng: &mut dyn Rng,
    ) -> (BatState<B>, StrategyMetrics) {
        let fitness_host = fitness.into_data().into_vec::<f32>().unwrap_or_default();
        let device = candidates.device();
        let pop = params.pop_size;
        let genome_dim = params.genome_dim;

        if state.fitness.is_empty() {
            state.fitness.clone_from(&fitness_host);
            let best_idx = argmin(&fitness_host);
            state.best_fitness = fitness_host[best_idx];
            #[allow(clippy::cast_possible_wrap)]
            let idx = Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![best_idx as i64], [1]),
                &device,
            );
            state.best_genome = Some(candidates.clone().select(0, idx));
            state.positions = candidates;
            state.generation += 1;
            let m = StrategyMetrics::from_host_fitness(
                state.generation,
                &fitness_host,
                state.best_fitness,
            );
            state.best_fitness = m.best_fitness_ever;
            return (state, m);
        }

        // Acceptance: accept candidate `i` iff `pending_accept[i]` AND
        // candidate's fitness is no worse than current.
        #[allow(clippy::cast_possible_wrap)]
        let mut rs: Vec<i64> = (0..pop).map(|i| i as i64).collect();
        let mut new_fitness = state.fitness.clone();
        #[allow(clippy::cast_precision_loss)]
        let t = state.generation as f32;
        for i in 0..pop {
            let accept_gate = state.pending_accept.get(i).copied().unwrap_or(false);
            let improves = fitness_host[i] <= state.fitness[i];
            if accept_gate && improves {
                #[allow(clippy::cast_possible_wrap)]
                {
                    rs[i] = (pop + i) as i64;
                }
                new_fitness[i] = fitness_host[i];
                state.loudness[i] *= params.alpha;
                state.pulse_rate[i] = params.r0 * (1.0 - (-params.gamma * t).exp());
            }
        }
        let stacked = Tensor::cat(vec![state.positions.clone(), candidates], 0);
        let idx = Tensor::<B, 1, Int>::from_data(TensorData::new(rs, [pop]), &device);
        state.positions = stacked.select(0, idx);
        state.fitness = new_fitness;

        // Refresh global best.
        let best_idx = argmin(&state.fitness);
        if state.fitness[best_idx] < state.best_fitness {
            state.best_fitness = state.fitness[best_idx];
            #[allow(clippy::cast_possible_wrap)]
            let idx = Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![best_idx as i64], [1]),
                &device,
            );
            state.best_genome = Some(state.positions.clone().select(0, idx));
        }

        state.generation += 1;
        let m =
            StrategyMetrics::from_host_fitness(state.generation, &fitness_host, state.best_fitness);
        state.best_fitness = m.best_fitness_ever;
        let _ = genome_dim;
        (state, m)
    }

    fn best(&self, state: &BatState<B>) -> Option<(Tensor<B, 2>, f32)> {
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
    fn bat_converges_on_sphere_d10() {
        // Bat is a "legacy comparator" per the module-level candor
        // note. We require strong reduction from the random baseline,
        // not machine precision — the probabilistic acceptance gate
        // (A_i decay) throttles late-stage progress. Threshold 0.1 on
        // Sphere-D10 in 800 generations is still orders of magnitude
        // below the uniform-random baseline (≈ 87).
        let device = Default::default();
        let strategy = BatAlgorithm::<TestBackend>::new();
        let params = BatConfig::default_for(40, 10);
        let fitness_fn = FromFitnessEvaluable::new(SphereFit, Sphere);
        let mut harness = EvolutionaryHarness::<TestBackend, _, _>::new(
            strategy, params, fitness_fn, 23, device, 800,
        );
        harness.reset();
        while !harness.step(()).done {}
        let best = harness.latest_metrics().unwrap().best_fitness_ever;
        assert!(best < 0.1, "Bat D10 best={best}");
    }
}
