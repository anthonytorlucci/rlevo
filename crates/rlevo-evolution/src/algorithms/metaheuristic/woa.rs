//! Whale Optimization Algorithm.
//!
//! Each whale chooses per generation between two behaviours, driven by a
//! uniform random `p ∈ U[0, 1]`:
//!
//! - `p < 0.5` — **encircle / search**:
//!     - `|A| < 1`: exploit the current best (`X ← X_best − A·|C·X_best − X|`),
//!     - `|A| ≥ 1`: explore by pulling toward a random other whale
//!       (`X ← X_rand − A·|C·X_rand − X|`).
//! - `p ≥ 0.5` — **spiral bubble-net**:
//!   `X ← |X_best − X|·exp(b·l)·cos(2π·l) + X_best`, `l ∈ U[−1, 1]`,
//!   `b = 1` (canonical).
//!
//! `A = 2a·r − a`, `C = 2r`, with `a` linearly decreased from 2 to 0
//! over the budget.
//!
//! The branches are realized as two boolean masks and three tensor
//! candidates — no divergent kernel paths.
//!
//! # Candor
//!
//! Legacy comparator. The spiral bubble-net and encircle-best operators
//! compose to a motion pattern that is equivalent in expectation to a
//! weighted PSO update toward the current best (Camacho Villalón et al.
//! 2020 review the structural similarities). Ship it for API coverage;
//! prefer CMA-ES or LSHADE when available.
//!
//! # References
//!
//! - Mirjalili & Lewis (2016), *The Whale Optimization Algorithm*.

use std::f32::consts::PI;
use std::marker::PhantomData;

use burn::tensor::{Distribution, Int, Tensor, TensorData, backend::Backend};
use rand::Rng;
use rand::RngExt;

use crate::rng::{SeedPurpose, seed_stream};
use crate::strategy::{Strategy, StrategyMetrics};

/// Static configuration for [`WhaleOptimization`].
#[derive(Debug, Clone)]
pub struct WoaConfig {
    /// Number of whales.
    pub pop_size: usize,
    /// Genome dimensionality.
    pub genome_dim: usize,
    /// Search-space bounds.
    pub bounds: (f32, f32),
    /// Budget pacing `a = 2·(1 − t/max_generations)`.
    pub max_generations: usize,
    /// Spiral shape constant (Mirjalili's canonical `b = 1`).
    pub b: f32,
}

impl WoaConfig {
    /// Default configuration for a given population size and genome dimensionality.
    #[must_use]
    pub fn default_for(pop_size: usize, genome_dim: usize) -> Self {
        Self {
            pop_size,
            genome_dim,
            bounds: (-5.12, 5.12),
            max_generations: 500,
            b: 1.0,
        }
    }
}

/// Generation state for [`WhaleOptimization`].
#[derive(Debug, Clone)]
pub struct WoaState<B: Backend> {
    /// Current positions, shape `(pop_size, D)`.
    pub positions: Tensor<B, 2>,
    /// Host-side fitness cache.
    pub fitness: Vec<f32>,
    /// Best-so-far genome.
    pub best_genome: Option<Tensor<B, 2>>,
    /// Best-so-far fitness.
    pub best_fitness: f32,
    /// Generation counter.
    pub generation: usize,
}

/// Whale Optimization Algorithm strategy.
///
/// # Example
///
/// ```no_run
/// use burn::backend::NdArray;
/// use rlevo_evolution::algorithms::metaheuristic::woa::{WhaleOptimization, WoaConfig};
///
/// let strategy = WhaleOptimization::<NdArray>::new();
/// let params = WoaConfig::default_for(32, 10);
/// let _ = (strategy, params);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct WhaleOptimization<B: Backend> {
    _backend: PhantomData<fn() -> B>,
}

impl<B: Backend> WhaleOptimization<B> {
    /// Builds a new (stateless) strategy object.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> Strategy<B> for WhaleOptimization<B>
where
    B::Device: Clone,
{
    type Params = WoaConfig;
    type State = WoaState<B>;
    type Genome = Tensor<B, 2>;

    fn init(&self, params: &WoaConfig, rng: &mut dyn Rng, device: &B::Device) -> WoaState<B> {
        let (lo, hi) = params.bounds;
        B::seed(device, rng.next_u64());
        let positions = Tensor::<B, 2>::random(
            [params.pop_size, params.genome_dim],
            Distribution::Uniform(f64::from(lo), f64::from(hi)),
            device,
        );
        WoaState {
            positions,
            fitness: Vec::new(),
            best_genome: None,
            best_fitness: f32::INFINITY,
            generation: 0,
        }
    }

    fn ask(
        &self,
        params: &WoaConfig,
        state: &WoaState<B>,
        rng: &mut dyn Rng,
        device: &B::Device,
    ) -> (Tensor<B, 2>, WoaState<B>) {
        // First call: evaluate initial whales so `tell` can record fitness.
        if state.fitness.is_empty() {
            return (state.positions.clone(), state.clone());
        }

        let pop_size = params.pop_size;
        let genome_dim = params.genome_dim;

        // Linear schedule for a.
        #[allow(clippy::cast_precision_loss)]
        let t = state.generation as f32;
        #[allow(clippy::cast_precision_loss)]
        let max_t = params.max_generations.max(1) as f32;
        let a = 2.0 * (1.0 - (t / max_t).min(1.0));

        // Per-whale scalars: A, C, p, l. Sample on host via the scope
        // splitmix stream so the seed contract is fully reproducible.
        let mut stream = seed_stream(rng.next_u64(), state.generation as u64, SeedPurpose::Other);
        let mut rand_idx: Vec<i64> = Vec::with_capacity(pop_size);
        let mut a_scalar: Vec<f32> = Vec::with_capacity(pop_size);
        let mut c_scalar: Vec<f32> = Vec::with_capacity(pop_size);
        let mut p_scalar: Vec<f32> = Vec::with_capacity(pop_size);
        let mut l_scalar: Vec<f32> = Vec::with_capacity(pop_size);
        let mut abs_a_lt_one: Vec<i64> = Vec::with_capacity(pop_size);
        let mut p_lt_half: Vec<i64> = Vec::with_capacity(pop_size);
        for i in 0..pop_size {
            let r_a: f32 = stream.random::<f32>();
            let r_c: f32 = stream.random::<f32>();
            let p: f32 = stream.random::<f32>();
            let l: f32 = 2.0 * stream.random::<f32>() - 1.0;
            let a_val = 2.0 * a * r_a - a;
            let c_val = 2.0 * r_c;
            a_scalar.push(a_val);
            c_scalar.push(c_val);
            p_scalar.push(p);
            l_scalar.push(l);
            abs_a_lt_one.push(i64::from(a_val.abs() < 1.0));
            p_lt_half.push(i64::from(p < 0.5));
            // Pick a different index for the "search" branch.
            let mut r = stream.random_range(0..pop_size);
            if r == i {
                r = (r + 1) % pop_size;
            }
            #[allow(clippy::cast_possible_wrap)]
            rand_idx.push(r as i64);
        }

        let a_row = Tensor::<B, 1>::from_data(TensorData::new(a_scalar, [pop_size]), device)
            .unsqueeze_dim::<2>(1)
            .expand([pop_size, genome_dim]);
        let c_row = Tensor::<B, 1>::from_data(TensorData::new(c_scalar, [pop_size]), device)
            .unsqueeze_dim::<2>(1)
            .expand([pop_size, genome_dim]);
        let l_vec = Tensor::<B, 1>::from_data(TensorData::new(l_scalar, [pop_size]), device);
        let rand_idx_t =
            Tensor::<B, 1, Int>::from_data(TensorData::new(rand_idx, [pop_size]), device);
        let x_rand = state.positions.clone().select(0, rand_idx_t);

        let x_best = state
            .best_genome
            .as_ref()
            .expect("best_genome populated after the first tell")
            .clone()
            .expand([pop_size, genome_dim]);

        // Encircle toward X_best:  X_best − A · |C · X_best − X|
        let enc_best = x_best.clone()
            - a_row
                .clone()
                .mul((c_row.clone().mul(x_best.clone()) - state.positions.clone()).abs());
        // Search toward X_rand:    X_rand − A · |C · X_rand − X|
        let enc_rand =
            x_rand.clone() - a_row.mul((c_row.mul(x_rand) - state.positions.clone()).abs());
        // Spiral toward X_best:    |X_best − X| · exp(b·l) · cos(2π·l) + X_best
        let dist = (x_best.clone() - state.positions.clone()).abs();
        let factor = l_vec
            .clone()
            .mul_scalar(params.b)
            .exp()
            .mul(l_vec.mul_scalar(2.0 * PI).cos());
        let factor_mat = factor.unsqueeze_dim::<2>(1).expand([pop_size, genome_dim]);
        let spiral = dist.mul(factor_mat) + x_best;

        // Compose: p < 0.5 ? (|A|<1 ? enc_best : enc_rand) : spiral.
        let m_abs_a_lt_one =
            Tensor::<B, 1, Int>::from_data(TensorData::new(abs_a_lt_one, [pop_size]), device)
                .equal_elem(1)
                .unsqueeze_dim::<2>(1)
                .expand([pop_size, genome_dim]);
        let m_p_lt_half =
            Tensor::<B, 1, Int>::from_data(TensorData::new(p_lt_half, [pop_size]), device)
                .equal_elem(1)
                .unsqueeze_dim::<2>(1)
                .expand([pop_size, genome_dim]);

        let encircle = enc_rand.mask_where(m_abs_a_lt_one, enc_best);
        let new_positions = spiral.mask_where(m_p_lt_half, encircle);

        let (lo, hi) = params.bounds;
        let new_positions = new_positions.clamp(lo, hi);

        let mut next = state.clone();
        next.positions = new_positions.clone();
        (new_positions, next)
    }

    fn tell(
        &self,
        _params: &WoaConfig,
        population: Tensor<B, 2>,
        fitness: Tensor<B, 1>,
        mut state: WoaState<B>,
        _rng: &mut dyn Rng,
    ) -> (WoaState<B>, StrategyMetrics) {
        let fitness_host = fitness.into_data().into_vec::<f32>().unwrap_or_default();
        state.fitness = fitness_host.clone();
        state.positions = population.clone();
        let best_idx = argmin(&fitness_host);
        if fitness_host[best_idx] < state.best_fitness {
            state.best_fitness = fitness_host[best_idx];
            let device = population.device();
            #[allow(clippy::cast_possible_wrap)]
            let idx = Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![best_idx as i64], [1]),
                &device,
            );
            state.best_genome = Some(population.select(0, idx));
        }
        state.generation += 1;
        let m =
            StrategyMetrics::from_host_fitness(state.generation, &fitness_host, state.best_fitness);
        state.best_fitness = m.best_fitness_ever;
        (state, m)
    }

    fn best(&self, state: &WoaState<B>) -> Option<(Tensor<B, 2>, f32)> {
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
    fn woa_converges_on_sphere_d10() {
        // WOA as "legacy comparator" per module-level candor note. We
        // verify convergence direction — reaching within 1e-4 of the
        // optimum on Sphere-D10 in 600 generations confirms the
        // spiral/encircle composition functions correctly.
        let device = Default::default();
        let strategy = WhaleOptimization::<TestBackend>::new();
        let params = WoaConfig::default_for(32, 10);
        let fitness_fn = FromFitnessEvaluable::new(SphereFit, Sphere);
        let mut harness = EvolutionaryHarness::<TestBackend, _, _>::new(
            strategy, params, fitness_fn, 5, device, 600,
        );
        harness.reset();
        while !harness.step(()).done {}
        let best = harness.latest_metrics().unwrap().best_fitness_ever;
        assert!(best < 1e-4, "WOA D10 best={best}");
    }
}
