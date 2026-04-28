//! Firefly Algorithm.
//!
//! Each firefly `i` moves toward every **brighter** firefly `j`, with
//! attractiveness decaying exponentially in the squared distance:
//!
//! - `β(r_ij) = β₀ · exp(−γ · r_ij²)` where `r_ij = ‖x_i − x_j‖`,
//! - `Δx_i = Σ_{j : f(x_j) < f(x_i)} β(r_ij) · (x_j − x_i) + α · (U[−0.5, 0.5])`,
//! - `x_i ← x_i + Δx_i`.
//!
//! The attraction sum is canonically `O(N²)`; a naïve tensor
//! implementation materializes an `(N, N, D)` pairwise-difference
//! tensor and therefore blows out memory at `N > 128`. This module
//! enforces that hard cap when the `custom-kernels` feature is off. A
//! future fused CubeCL kernel
//! ([`super::kernels::pairwise_attract_cube`]) is designed to stream
//! over the neighbour axis and keep memory at `O(ND)`, removing the
//! cap; until that kernel lands, the pure-tensor path runs even when
//! the feature is enabled.
//!
//! # References
//!
//! - Yang (2008), *Nature-Inspired Metaheuristic Algorithms*.

use std::marker::PhantomData;

use burn::tensor::{Distribution, Int, Tensor, TensorData, backend::Backend};
use rand::Rng;

use crate::rng::{SeedPurpose, seed_stream};
use crate::strategy::{Strategy, StrategyMetrics};

/// Hard cap for the pure-tensor `O(N²D)` Firefly path. Exceeding this
/// without the fused CubeCL kernel would allocate a cubic tensor on
/// device; the kernel path removes the cap.
pub const FIREFLY_PURE_TENSOR_CAP: usize = 128;

/// Static configuration for [`FireflyAlgorithm`].
#[derive(Debug, Clone)]
pub struct FireflyConfig {
    /// Number of fireflies.
    pub pop_size: usize,
    /// Genome dimensionality.
    pub genome_dim: usize,
    /// Search-space bounds.
    pub bounds: (f32, f32),
    /// Base attractiveness `β₀`. Canonical 1.0.
    pub beta0: f32,
    /// Light-absorption coefficient `γ`. Canonical 1.0; controls the
    /// range over which fireflies can see each other.
    pub gamma: f32,
    /// Noise scale for the random walk term. Canonical 0.2.
    pub alpha: f32,
}

impl FireflyConfig {
    /// Default configuration. `γ` is scaled by the search-space extent
    /// so the exponential decay lands in a useful regime — Yang's
    /// canonical `γ = 1` assumes `[0, 1]` normalization; for the usual
    /// `[−5.12, 5.12]` domain, `γ ≈ 1 / L²` keeps attractiveness
    /// non-vanishing across pairs.
    #[must_use]
    pub fn default_for(pop_size: usize, genome_dim: usize) -> Self {
        Self {
            pop_size,
            genome_dim,
            bounds: (-5.12, 5.12),
            beta0: 1.0,
            gamma: 0.01,
            alpha: 0.2,
        }
    }
}

/// Generation state for [`FireflyAlgorithm`].
#[derive(Debug, Clone)]
pub struct FireflyState<B: Backend> {
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

/// Firefly Algorithm strategy.
///
/// # Panics
///
/// [`Strategy::init`] enforces a `pop_size <= FIREFLY_PURE_TENSOR_CAP`
/// (= 128) cap when the `custom-kernels` feature is **off**, since the
/// pure-tensor path materializes an `(N, N, D)` pairwise tensor.
/// With the feature on the same cap is enforced via `debug_assert!`,
/// because the fused kernel
/// [`super::kernels::pairwise_attract_cube`] is still designed-only and
/// the strategy keeps using the pure-tensor path in the meantime.
///
/// # Example
///
/// ```no_run
/// use burn::backend::NdArray;
/// use rlevo_evolution::algorithms::metaheuristic::firefly::{FireflyAlgorithm, FireflyConfig};
///
/// let strategy = FireflyAlgorithm::<NdArray>::new();
/// let params = FireflyConfig::default_for(32, 10);
/// let _ = (strategy, params);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct FireflyAlgorithm<B: Backend> {
    _backend: PhantomData<fn() -> B>,
}

impl<B: Backend> FireflyAlgorithm<B> {
    /// Builds a new (stateless) strategy object.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _backend: PhantomData,
        }
    }

    /// Pure-tensor `O(N²D)` attraction kernel — always available, even
    /// without the `custom-kernels` feature. The fused CubeCL kernel
    /// designed in [`super::kernels::pairwise_attract_cube`] slots in at
    /// this call site once it lands.
    fn pure_tensor_attract(
        positions: &Tensor<B, 2>,
        fitness: &[f32],
        beta0: f32,
        gamma: f32,
        alpha: f32,
        device: &B::Device,
        noise_seed: u64,
    ) -> Tensor<B, 2> {
        let pop = fitness.len();
        let shape = positions.shape().dims;
        let d = shape[1];

        // Pairwise squared distances via (x·x^T + ||x||² - 2x·x^T).
        // Cheaper memory than the (N, N, D) difference tensor, but we
        // still need the (N, N, D) tensor for the displacement `x_j -
        // x_i`. Cap enforced at module level.
        let xi = positions.clone().unsqueeze_dim::<3>(1); // (N, 1, D)
        let xj = positions.clone().unsqueeze_dim::<3>(0); // (1, N, D)
        let diff = xj.expand([pop, pop, d]) - xi.expand([pop, pop, d]); // (N, N, D)
        let r2 = diff.clone().powi_scalar(2).sum_dim(2).squeeze::<2>(); // (N, N)
        let beta = r2.mul_scalar(-gamma).exp().mul_scalar(beta0); // (N, N)

        // Brightness mask: bright[i, j] = 1 iff fitness[j] < fitness[i].
        let mut bright = vec![0i64; pop * pop];
        for i in 0..pop {
            for j in 0..pop {
                if fitness[j] < fitness[i] {
                    bright[i * pop + j] = 1;
                }
            }
        }
        let bright_mask =
            Tensor::<B, 2, Int>::from_data(TensorData::new(bright, [pop, pop]), device)
                .equal_elem(1);
        // Zero-out non-bright pairs in β then multiply diff.
        let zero = Tensor::<B, 2>::zeros([pop, pop], device);
        let beta_m = beta.mask_where(bright_mask.bool_not(), zero);
        let weight = beta_m.unsqueeze_dim::<3>(2).expand([pop, pop, d]); // (N, N, D)
        let weighted = diff.mul(weight); // (N, N, D)
        let attr_sum = weighted.sum_dim(1).squeeze::<2>(); // (N, D)

        // Noise: α · (U[0,1] - 0.5).
        B::seed(device, noise_seed);
        let noise = Tensor::<B, 2>::random([pop, d], Distribution::Uniform(-0.5, 0.5), device);
        attr_sum + noise.mul_scalar(alpha)
    }
}

impl<B: Backend> Strategy<B> for FireflyAlgorithm<B>
where
    B::Device: Clone,
{
    type Params = FireflyConfig;
    type State = FireflyState<B>;
    type Genome = Tensor<B, 2>;

    fn init(
        &self,
        params: &FireflyConfig,
        rng: &mut dyn Rng,
        device: &B::Device,
    ) -> FireflyState<B> {
        #[cfg(not(feature = "custom-kernels"))]
        assert!(
            params.pop_size <= FIREFLY_PURE_TENSOR_CAP,
            "Firefly without `custom-kernels` feature caps pop_size at {} to keep the O(N²D) \
             pairwise tensor bounded; enable `custom-kernels` for larger swarms",
            FIREFLY_PURE_TENSOR_CAP
        );
        // Even with the kernel feature active, the fused pairwise-attract
        // kernel is currently a design placeholder and the pure-tensor
        // path is still in use. A debug assert surfaces the limitation in
        // tests without blocking downstream users who have wired in their
        // own kernel.
        #[cfg(feature = "custom-kernels")]
        debug_assert!(
            params.pop_size <= FIREFLY_PURE_TENSOR_CAP,
            "Firefly pop_size > {} requires the fused pairwise-attract kernel; \
             the placeholder kernel module still runs the pure-tensor path",
            FIREFLY_PURE_TENSOR_CAP
        );
        let (lo, hi) = params.bounds;
        B::seed(device, rng.next_u64());
        let positions = Tensor::<B, 2>::random(
            [params.pop_size, params.genome_dim],
            Distribution::Uniform(f64::from(lo), f64::from(hi)),
            device,
        );
        FireflyState {
            positions,
            fitness: Vec::new(),
            best_genome: None,
            best_fitness: f32::INFINITY,
            generation: 0,
        }
    }

    fn ask(
        &self,
        params: &FireflyConfig,
        state: &FireflyState<B>,
        rng: &mut dyn Rng,
        device: &B::Device,
    ) -> (Tensor<B, 2>, FireflyState<B>) {
        if state.fitness.is_empty() {
            return (state.positions.clone(), state.clone());
        }

        let seed = seed_stream(
            rng.next_u64(),
            state.generation as u64,
            SeedPurpose::Mutation,
        )
        .next_u64();
        let delta = Self::pure_tensor_attract(
            &state.positions,
            &state.fitness,
            params.beta0,
            params.gamma,
            params.alpha,
            device,
            seed,
        );
        let (lo, hi) = params.bounds;
        let new_positions = (state.positions.clone() + delta).clamp(lo, hi);

        let mut next = state.clone();
        next.positions = new_positions.clone();
        (new_positions, next)
    }

    fn tell(
        &self,
        _params: &FireflyConfig,
        population: Tensor<B, 2>,
        fitness: Tensor<B, 1>,
        mut state: FireflyState<B>,
        _rng: &mut dyn Rng,
    ) -> (FireflyState<B>, StrategyMetrics) {
        let fitness_host = fitness.into_data().into_vec::<f32>().unwrap_or_default();
        let device = population.device();
        state.fitness = fitness_host.clone();
        state.positions = population.clone();

        let best_idx = argmin(&fitness_host);
        if fitness_host[best_idx] < state.best_fitness {
            state.best_fitness = fitness_host[best_idx];
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

    fn best(&self, state: &FireflyState<B>) -> Option<(Tensor<B, 2>, f32)> {
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
    fn firefly_converges_on_sphere_d10() {
        // Firefly's attraction sum is O(N²D); we use 24 fireflies to
        // keep the test fast while still exercising the pairwise
        // kernel path.
        let device = Default::default();
        let strategy = FireflyAlgorithm::<TestBackend>::new();
        let params = FireflyConfig::default_for(24, 10);
        let fitness_fn = FromFitnessEvaluable::new(SphereFit, Sphere);
        let mut harness = EvolutionaryHarness::<TestBackend, _, _>::new(
            strategy, params, fitness_fn, 29, device, 500,
        );
        harness.reset();
        while !harness.step(()).done {}
        let best = harness.latest_metrics().unwrap().best_fitness_ever;
        assert!(best < 1.0, "Firefly D10 best={best}");
    }
}
