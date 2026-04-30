//! Cuckoo Search via Lévy flights.
//!
//! Each generation every nest proposes a new egg by taking a
//! Lévy-stable step from its current position:
//!
//! - `u ∼ N(0, σ_u²)`, `v ∼ N(0, 1)`,
//! - `step = u / |v|^(1/β)`,
//! - `x'_i = x_i + α · step`,
//!
//! where `σ_u = (Γ(1+β)·sin(π·β/2) / (Γ((1+β)/2)·β·2^((β−1)/2)))^(1/β)`
//! (Mantegna's algorithm, β ≈ 1.5).
//!
//! `tell` greedy-accepts each new egg against its own slot, then
//! abandons the `p_a · N` worst nests and reinitializes them from the
//! search bounds. Abandoned slots carry sentinel `+∞` fitness so the
//! next generation's Lévy proposal always lands.
//!
//! # Numerical parity caveat
//!
//! The fractional power `|v|^(1/β)` is FMA-reorder-sensitive — wgpu
//! reductions can drift ~`1e-3` relative from ndarray on the same seed.
//! The backend-parity test relaxes tolerance for CS accordingly.
//!
//! # References
//!
//! - Yang & Deb (2009), *Cuckoo Search via Lévy Flights*.
//! - Mantegna (1994), *Fast, accurate algorithm for numerical simulation
//!   of Lévy stable stochastic processes*.

use std::f32::consts::PI;
use std::marker::PhantomData;

use burn::tensor::{Distribution, Int, Tensor, TensorData, backend::Backend};
use rand::Rng;
use rand_distr::{Distribution as RandDistDist, Normal};

use crate::rng::{SeedPurpose, seed_stream};
use crate::strategy::{Strategy, StrategyMetrics};

/// Static configuration for [`CuckooSearch`].
#[derive(Debug, Clone)]
pub struct CuckooConfig {
    /// Nest count.
    pub pop_size: usize,
    /// Genome dimensionality.
    pub genome_dim: usize,
    /// Search-space bounds.
    pub bounds: (f32, f32),
    /// Step size scale (`α` in the paper). Canonical `α = 0.01`
    /// multiplied by the search-space width; strategy users should
    /// tune relative to their domain.
    pub alpha: f32,
    /// Lévy index (`β`). Must be in `(0, 2)`; canonical 1.5.
    pub beta: f32,
    /// Nest abandonment probability (`p_a`). Canonical 0.25.
    pub p_a: f32,
}

impl CuckooConfig {
    /// Default configuration for a given population size and genome dimensionality.
    #[must_use]
    pub fn default_for(pop_size: usize, genome_dim: usize) -> Self {
        Self {
            pop_size,
            genome_dim,
            bounds: (-5.12, 5.12),
            alpha: 0.05,
            beta: 1.5,
            p_a: 0.25,
        }
    }
}

/// Generation state for [`CuckooSearch`].
#[derive(Debug, Clone)]
pub struct CuckooState<B: Backend> {
    /// Current nests, shape `(pop_size, D)`.
    pub nests: Tensor<B, 2>,
    /// Host-side fitness cache; `+∞` for abandoned slots.
    pub fitness: Vec<f32>,
    /// Best-so-far genome.
    pub best_genome: Option<Tensor<B, 2>>,
    /// Best-so-far fitness.
    pub best_fitness: f32,
    /// Generation counter.
    pub generation: usize,
}

/// Cuckoo Search strategy.
///
/// # Example
///
/// ```no_run
/// use burn::backend::NdArray;
/// use rlevo_evolution::algorithms::metaheuristic::cuckoo::{CuckooConfig, CuckooSearch};
///
/// let strategy = CuckooSearch::<NdArray>::new();
/// let params = CuckooConfig::default_for(30, 10);
/// let _ = (strategy, params);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct CuckooSearch<B: Backend> {
    _backend: PhantomData<fn() -> B>,
}

impl<B: Backend> CuckooSearch<B> {
    /// Builds a new (stateless) strategy object.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _backend: PhantomData,
        }
    }

    /// Mantegna's `σ_u` for the `u ∼ N(0, σ_u²)` draw.
    fn mantegna_sigma_u(beta: f32) -> f32 {
        // Γ(1 + β) · sin(π·β/2)  /  ( Γ((1+β)/2) · β · 2^((β-1)/2) ) ) ^ (1/β)
        let num = gamma(1.0 + beta) * ((PI * beta) / 2.0).sin();
        let den = gamma(f32::midpoint(1.0, beta)) * beta * 2f32.powf((beta - 1.0) / 2.0);
        (num / den).powf(1.0 / beta)
    }
}

/// Lanczos approximation for `Γ(z)` on positive reals. Only used
/// host-side in [`CuckooSearch`] and [`super::bat`] for small constants.
#[allow(clippy::many_single_char_names)]
fn gamma(z: f32) -> f32 {
    // 5-term Lanczos coefficients (g = 7). Enough for `z ∈ [0.5, 5]`
    // which covers the Lévy-flight parameter range.
    let g = 7.0_f32;
    let p: [f32; 9] = [
        0.999_999_999_999_809_93,
        676.520_4,
        -1_259.139_2,
        771.323_4,
        -176.615_04,
        12.507_343,
        -0.138_571_1,
        9.984_369e-6,
        1.505_632_7e-7,
    ];
    if z < 0.5 {
        return PI / ((PI * z).sin() * gamma(1.0 - z));
    }
    let z = z - 1.0;
    let mut x = p[0];
    for (i, &coef) in p.iter().enumerate().skip(1) {
        #[allow(clippy::cast_precision_loss)]
        let i_f32 = i as f32;
        x += coef / (z + i_f32);
    }
    let t = z + g + 0.5;
    (2.0 * PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * x
}

impl<B: Backend> Strategy<B> for CuckooSearch<B>
where
    B::Device: Clone,
{
    type Params = CuckooConfig;
    type State = CuckooState<B>;
    type Genome = Tensor<B, 2>;

    fn init(&self, params: &CuckooConfig, rng: &mut dyn Rng, device: &B::Device) -> CuckooState<B> {
        let (lo, hi) = params.bounds;
        B::seed(device, rng.next_u64());
        let nests = Tensor::<B, 2>::random(
            [params.pop_size, params.genome_dim],
            Distribution::Uniform(f64::from(lo), f64::from(hi)),
            device,
        );
        CuckooState {
            nests,
            fitness: Vec::new(),
            best_genome: None,
            best_fitness: f32::INFINITY,
            generation: 0,
        }
    }

    fn ask(
        &self,
        params: &CuckooConfig,
        state: &CuckooState<B>,
        rng: &mut dyn Rng,
        device: &B::Device,
    ) -> (Tensor<B, 2>, CuckooState<B>) {
        if state.fitness.is_empty() {
            return (state.nests.clone(), state.clone());
        }

        let pop = params.pop_size;
        let d = params.genome_dim;
        let sigma_u = Self::mantegna_sigma_u(params.beta);

        let mut stream = seed_stream(
            rng.next_u64(),
            state.generation as u64,
            SeedPurpose::Mutation,
        );
        let normal_u = Normal::new(0.0_f32, sigma_u).expect("σ_u > 0");
        let normal_v = Normal::new(0.0_f32, 1.0_f32).unwrap();
        let mut step = vec![0f32; pop * d];
        for v in &mut step {
            let u: f32 = normal_u.sample(&mut stream);
            let w: f32 = normal_v.sample(&mut stream);
            *v = u / w.abs().powf(1.0 / params.beta);
        }
        let step_tensor = Tensor::<B, 2>::from_data(TensorData::new(step, [pop, d]), device);

        let (lo, hi) = params.bounds;
        let new_nests = (state.nests.clone() + step_tensor.mul_scalar(params.alpha)).clamp(lo, hi);

        let mut next = state.clone();
        next.nests.clone_from(&new_nests);
        (new_nests, next)
    }

    fn tell(
        &self,
        params: &CuckooConfig,
        population: Tensor<B, 2>,
        fitness: Tensor<B, 1>,
        mut state: CuckooState<B>,
        rng: &mut dyn Rng,
    ) -> (CuckooState<B>, StrategyMetrics) {
        let fitness_host = fitness.into_data().into_vec::<f32>().unwrap_or_default();
        let device = population.device();
        let pop = params.pop_size;
        let d = params.genome_dim;

        if state.fitness.is_empty() {
            state.fitness.clone_from(&fitness_host);
            let best_idx = argmin(&fitness_host);
            state.best_fitness = fitness_host[best_idx];
            #[allow(clippy::cast_possible_wrap)]
            let idx = Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![best_idx as i64], [1]),
                &device,
            );
            state.best_genome = Some(population.clone().select(0, idx));
            state.nests = population;
            state.generation += 1;
            let m = StrategyMetrics::from_host_fitness(
                state.generation,
                &fitness_host,
                state.best_fitness,
            );
            state.best_fitness = m.best_fitness_ever;
            return (state, m);
        }

        // Greedy accept per slot.
        #[allow(clippy::cast_possible_wrap)]
        let mut rs: Vec<i64> = (0..pop).map(|i| i as i64).collect();
        let mut new_fitness = state.fitness.clone();
        for i in 0..pop {
            if fitness_host[i] <= state.fitness[i] {
                #[allow(clippy::cast_possible_wrap)]
                {
                    rs[i] = (pop + i) as i64;
                }
                new_fitness[i] = fitness_host[i];
            }
        }
        let stacked = Tensor::cat(vec![state.nests.clone(), population.clone()], 0);
        let idx = Tensor::<B, 1, Int>::from_data(TensorData::new(rs, [pop]), &device);
        state.nests = stacked.select(0, idx);
        state.fitness = new_fitness;

        // Abandon worst `p_a · pop` nests — reinit with uniform sample;
        // mark fitness +∞ so next ask's Lévy proposal always lands.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)]
        let n_abandon = (params.p_a * pop as f32) as usize;
        if n_abandon > 0 {
            let mut rank: Vec<usize> = (0..pop).collect();
            rank.sort_by(|&a, &b| state.fitness[b].partial_cmp(&state.fitness[a]).unwrap());
            let worst: Vec<usize> = rank.into_iter().take(n_abandon).collect();
            let (lo, hi) = params.bounds;
            B::seed(&device, rng.next_u64());
            let fresh = Tensor::<B, 2>::random(
                [n_abandon, d],
                Distribution::Uniform(f64::from(lo), f64::from(hi)),
                &device,
            );
            #[allow(clippy::cast_possible_wrap)]
            let mut rs2: Vec<i64> = (0..pop).map(|i| i as i64).collect();
            for (k, &slot) in worst.iter().enumerate() {
                #[allow(clippy::cast_possible_wrap)]
                {
                    rs2[slot] = (pop + k) as i64;
                }
                state.fitness[slot] = f32::INFINITY;
            }
            let stacked2 = Tensor::cat(vec![state.nests.clone(), fresh], 0);
            let idx2 = Tensor::<B, 1, Int>::from_data(TensorData::new(rs2, [pop]), &device);
            state.nests = stacked2.select(0, idx2);
        }

        // Best-so-far from finite-fitness slots.
        let best_idx = argmin(&state.fitness);
        if state.fitness[best_idx].is_finite() && state.fitness[best_idx] < state.best_fitness {
            state.best_fitness = state.fitness[best_idx];
            #[allow(clippy::cast_possible_wrap)]
            let idx = Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![best_idx as i64], [1]),
                &device,
            );
            state.best_genome = Some(state.nests.clone().select(0, idx));
        }

        state.generation += 1;
        let m =
            StrategyMetrics::from_host_fitness(state.generation, &fitness_host, state.best_fitness);
        state.best_fitness = m.best_fitness_ever;
        (state, m)
    }

    fn best(&self, state: &CuckooState<B>) -> Option<(Tensor<B, 2>, f32)> {
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
    fn gamma_matches_known_values() {
        // Γ(1) = 1, Γ(2) = 1, Γ(5) = 24, Γ(0.5) = √π.
        approx::assert_relative_eq!(gamma(1.0), 1.0, epsilon = 1e-4);
        approx::assert_relative_eq!(gamma(2.0), 1.0, epsilon = 1e-4);
        approx::assert_relative_eq!(gamma(5.0), 24.0, epsilon = 1e-3);
        approx::assert_relative_eq!(gamma(0.5), PI.sqrt(), epsilon = 1e-3);
    }

    #[test]
    fn mantegna_sigma_u_is_finite() {
        let s = CuckooSearch::<TestBackend>::mantegna_sigma_u(1.5);
        assert!(s.is_finite() && s > 0.0);
    }

    #[test]
    fn cuckoo_reduces_on_sphere_d10() {
        // Pure-Lévy CS has no gradient-biased update — it's a biased
        // random walk with abandonment. The Lévy flights are the
        // interesting part; otherwise CS is a thin wrapper around
        // random walk + abandonment, so convergence to machine
        // precision is not expected within reasonable budgets on
        // Sphere-D10. Threshold 20.0 in 800 generations is still a ~4×
        // reduction from the uniform-random baseline (≈ 87) — it
        // verifies the Lévy machinery composes correctly.
        let device = Default::default();
        let strategy = CuckooSearch::<TestBackend>::new();
        let mut params = CuckooConfig::default_for(30, 10);
        params.alpha = 0.2;
        let fitness_fn = FromFitnessEvaluable::new(SphereFit, Sphere);
        let mut harness = EvolutionaryHarness::<TestBackend, _, _>::new(
            strategy, params, fitness_fn, 19, device, 800,
        );
        harness.reset();
        while !harness.step(()).done {}
        let best = harness.latest_metrics().unwrap().best_fitness_ever;
        assert!(best < 20.0, "Cuckoo D10 best={best}");
    }
}
