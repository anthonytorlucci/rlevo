//! Salp Swarm Algorithm.
//!
//! The population splits into a *leader* half and a *follower* chain:
//!
//! - Leaders (indices `0 .. N/2`) update toward the food source `F`
//!   (the best-so-far position), modulated by a time-decayed coefficient
//!   `c1 = 2·exp(−(4·t / T)²)`:
//!
//!   `X_i ← F ± c1 · ((ub − lb)·c2 + lb)` with `c2, c3 ∈ U[0, 1]`,
//!   where the sign follows `c3 ≥ 0.5 → +` and `c3 < 0.5 → −`.
//!
//! - Followers (indices `N/2 .. N`) track the chain:
//!
//!   `X_i ← (X_i + X_{i−1}) / 2`.
//!
//! The follower rule is realized as a parallel stencil — the shifted
//! copy is `concat([last_leader, followers[..−1]])` — rather than a
//! host-side Python-style loop. That keeps the update GPU-friendly and
//! deterministic.
//!
//! # Candor
//!
//! Legacy comparator. The leader update is a thinly-veiled biased
//! random walk toward the best; the follower chain is a stencil average
//! of old positions. Neither mechanism introduces a novel search
//! dynamic not already present in PSO or DE. Ship it for API coverage;
//! prefer CMA-ES or LSHADE when available.
//!
//! # References
//!
//! - Mirjalili, Gandomi, Mirjalili, Saremi, Faris & Mirjalili (2017),
//!   *Salp Swarm Algorithm*.

use std::marker::PhantomData;

use burn::tensor::{Distribution, Int, Tensor, TensorData, backend::Backend};
use rand::Rng;
use rand::RngExt;

use crate::rng::{SeedPurpose, seed_stream};
use crate::strategy::{Strategy, StrategyMetrics};

/// Static configuration for [`SalpSwarm`].
#[derive(Debug, Clone)]
pub struct SalpConfig {
    /// Swarm size; must be `≥ 2` so there is at least one leader and
    /// one follower.
    pub pop_size: usize,
    /// Genome dimensionality.
    pub genome_dim: usize,
    /// Search-space bounds. The leader update pulls from a scaled
    /// uniform over this range.
    pub bounds: (f32, f32),
    /// Budget pacing the `c1` decay.
    pub max_generations: usize,
}

impl SalpConfig {
    /// Default configuration for a given population size and genome dimensionality.
    #[must_use]
    pub fn default_for(pop_size: usize, genome_dim: usize) -> Self {
        Self {
            pop_size,
            genome_dim,
            bounds: (-5.12, 5.12),
            max_generations: 500,
        }
    }
}

/// Generation state for [`SalpSwarm`].
#[derive(Debug, Clone)]
pub struct SalpState<B: Backend> {
    /// Current positions, shape `(pop_size, D)`.
    pub positions: Tensor<B, 2>,
    /// Host-side fitness cache.
    pub fitness: Vec<f32>,
    /// Food-source (best) genome.
    pub best_genome: Option<Tensor<B, 2>>,
    /// Food-source fitness.
    pub best_fitness: f32,
    /// Generation counter.
    pub generation: usize,
}

/// Salp Swarm Algorithm strategy.
///
/// # Panics
///
/// [`Strategy::init`] panics if `params.pop_size < 2`, since the
/// leader/follower split requires at least one of each.
///
/// # Example
///
/// ```no_run
/// use burn::backend::NdArray;
/// use rlevo_evolution::algorithms::metaheuristic::salp::{SalpConfig, SalpSwarm};
///
/// let strategy = SalpSwarm::<NdArray>::new();
/// let params = SalpConfig::default_for(32, 10);
/// let _ = (strategy, params);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct SalpSwarm<B: Backend> {
    _backend: PhantomData<fn() -> B>,
}

impl<B: Backend> SalpSwarm<B> {
    /// Builds a new (stateless) strategy object.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> Strategy<B> for SalpSwarm<B>
where
    B::Device: Clone,
{
    type Params = SalpConfig;
    type State = SalpState<B>;
    type Genome = Tensor<B, 2>;

    fn init(&self, params: &SalpConfig, rng: &mut dyn Rng, device: &B::Device) -> SalpState<B> {
        assert!(params.pop_size >= 2, "SSA requires pop_size >= 2");
        let (lo, hi) = params.bounds;
        B::seed(device, rng.next_u64());
        let positions = Tensor::<B, 2>::random(
            [params.pop_size, params.genome_dim],
            Distribution::Uniform(f64::from(lo), f64::from(hi)),
            device,
        );
        SalpState {
            positions,
            fitness: Vec::new(),
            best_genome: None,
            best_fitness: f32::INFINITY,
            generation: 0,
        }
    }

    fn ask(
        &self,
        params: &SalpConfig,
        state: &SalpState<B>,
        rng: &mut dyn Rng,
        device: &B::Device,
    ) -> (Tensor<B, 2>, SalpState<B>) {
        if state.fitness.is_empty() {
            return (state.positions.clone(), state.clone());
        }

        let pop_size = params.pop_size;
        let genome_dim = params.genome_dim;
        let n_leaders = pop_size / 2;
        let (lo, hi) = params.bounds;

        // c1 decay: 2 * exp(-(4t/T)^2)
        #[allow(clippy::cast_precision_loss)]
        let t = state.generation as f32;
        #[allow(clippy::cast_precision_loss)]
        let max_t = params.max_generations.max(1) as f32;
        let frac = (4.0 * t / max_t).min(4.0);
        let c1 = 2.0 * (-(frac * frac)).exp();

        // Host-side sampling for c2, c3 so the leader step is
        // backend-agnostic and reproducible under the splitmix contract.
        let mut stream = seed_stream(rng.next_u64(), state.generation as u64, SeedPurpose::Other);
        let mut leader_delta: Vec<f32> = Vec::with_capacity(n_leaders * genome_dim);
        for _ in 0..n_leaders {
            for _ in 0..genome_dim {
                let c2: f32 = stream.random::<f32>();
                let c3: f32 = stream.random::<f32>();
                let scaled = (hi - lo) * c2 + lo;
                let sign = if c3 >= 0.5 { 1.0 } else { -1.0 };
                leader_delta.push(sign * c1 * scaled);
            }
        }

        let best = state
            .best_genome
            .as_ref()
            .expect("best_genome populated after first tell")
            .clone()
            .expand([n_leaders, genome_dim]);
        let delta = Tensor::<B, 2>::from_data(
            TensorData::new(leader_delta, [n_leaders, genome_dim]),
            device,
        );
        let new_leaders = (best + delta).clamp(lo, hi);

        // Follower stencil: X_i ← (X_i + X_{i-1}) / 2.
        let followers = state
            .positions
            .clone()
            .slice([n_leaders..pop_size, 0..genome_dim]);
        // Shifted copy: follower i reads the position of the salp
        // directly ahead, which is the last leader for the first
        // follower and `followers[i-1]` otherwise. We build the shifted
        // stack by gathering indices `[n_leaders-1, n_leaders, …, pop_size-2]`
        // from the *updated* leader block + the old followers slice.
        let joined = Tensor::cat(vec![new_leaders.clone(), followers.clone()], 0); // (pop_size, D) — leaders are already the new ones.
        #[allow(clippy::cast_possible_wrap)]
        let shift_idx: Vec<i64> = (0..(pop_size - n_leaders))
            .map(|k| (n_leaders + k - 1) as i64)
            .collect();
        let idx = Tensor::<B, 1, Int>::from_data(
            TensorData::new(shift_idx, [pop_size - n_leaders]),
            device,
        );
        let previous = joined.clone().select(0, idx);
        let new_followers = (followers + previous).mul_scalar(0.5).clamp(lo, hi);

        let new_positions = Tensor::cat(vec![new_leaders, new_followers], 0);
        let mut next = state.clone();
        next.positions = new_positions.clone();
        (new_positions, next)
    }

    fn tell(
        &self,
        _params: &SalpConfig,
        population: Tensor<B, 2>,
        fitness: Tensor<B, 1>,
        mut state: SalpState<B>,
        _rng: &mut dyn Rng,
    ) -> (SalpState<B>, StrategyMetrics) {
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

    fn best(&self, state: &SalpState<B>) -> Option<(Tensor<B, 2>, f32)> {
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
    fn ssa_converges_on_sphere_d10() {
        // SSA reduces strongly but, per module-level candor, is not
        // expected to reach machine precision. A 1e-2 target in 600
        // generations is the strong-reduction guarantee mirroring the
        // DE premature-convergence test pattern.
        let device = Default::default();
        let strategy = SalpSwarm::<TestBackend>::new();
        let params = SalpConfig::default_for(40, 10);
        let fitness_fn = FromFitnessEvaluable::new(SphereFit, Sphere);
        let mut harness = EvolutionaryHarness::<TestBackend, _, _>::new(
            strategy, params, fitness_fn, 3, device, 600,
        );
        harness.reset();
        while !harness.step(()).done {}
        let best = harness.latest_metrics().unwrap().best_fitness_ever;
        assert!(best < 1e-2, "SSA D10 best={best}");
    }
}
