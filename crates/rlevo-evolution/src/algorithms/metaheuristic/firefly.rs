//! Firefly Algorithm.
//!
//! Each firefly `i` moves toward every **brighter** firefly `j`, with
//! attractiveness decaying exponentially in the squared distance:
//!
//! - `β(r_ij) = β₀ · exp(−γ · r_ij²)` where `r_ij = ‖x_i − x_j‖`,
//! - `Δx_i = Σ_{j : f(x_j) > f(x_i)} β(r_ij) · (x_j − x_i) + α · (U[−0.5, 0.5])`,
//! - `x_i ← x_i + Δx_i`.
//!
//! The attraction sum is canonically `O(N²)`; a naïve tensor
//! implementation materializes an `(N, N, D)` pairwise-difference
//! tensor and therefore blows out memory at `N > 128`. This module
//! enforces that hard cap when the `custom-kernels` feature is off. A
//! future fused `CubeCL` kernel
//! ([`super::kernels::pairwise_attract_cube`]) is designed to stream
//! over the neighbour axis and keep memory at `O(ND)`, removing the
//! cap; until that kernel lands, the pure-tensor path runs even when
//! the feature is enabled.
//!
//! # References
//!
//! - Yang (2008), *Nature-Inspired Metaheuristic Algorithms*.

use std::marker::PhantomData;

use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use rand::Rng;
use rand::RngExt;
use rand::SeedableRng;

use rlevo_core::bounds::Bounds;
use rlevo_core::config::{self, ConfigError, Validate};

use super::len_matches_pop;
use crate::ops::selection::argmax_host;
use crate::rng::{SeedPurpose, seed_stream};
use crate::strategy::{Strategy, StrategyMetrics};

/// Hard cap for the pure-tensor `O(N²D)` Firefly path. Exceeding this
/// without the fused `CubeCL` kernel would allocate a cubic tensor on
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
    pub bounds: Bounds,
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
        let (lo, hi): (f32, f32) = (-5.12, 5.12);
        let length: f32 = hi - lo;
        // γ ≈ 1/L², Yang's canonical regime scaled to the domain extent.
        let gamma: f32 = 1.0 / (length * length);
        Self {
            pop_size,
            genome_dim,
            bounds: Bounds::new(lo, hi),
            beta0: 1.0,
            gamma,
            alpha: 0.2,
        }
    }
}

impl Validate for FireflyConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "FireflyConfig";
        config::at_least(C, "pop_size", self.pop_size, 1)?;
        // Without the fused kernel the pure-tensor path materialises an
        // (N, N, D) tensor, so cap the swarm at FIREFLY_PURE_TENSOR_CAP.
        #[cfg(not(feature = "custom-kernels"))]
        if self.pop_size > FIREFLY_PURE_TENSOR_CAP {
            return Err(ConfigError {
                config: C,
                field: "pop_size",
                kind: rlevo_core::config::ConstraintKind::Custom(
                    "pop_size exceeds the pure-tensor cap (128); enable `custom-kernels`",
                ),
            });
        }
        config::nonzero(C, "genome_dim", self.genome_dim)?;
        config::in_range(C, "beta0", 0.0, f64::INFINITY, f64::from(self.beta0))?;
        config::positive(C, "gamma", f64::from(self.gamma))?;
        config::in_range(C, "alpha", 0.0, f64::INFINITY, f64::from(self.alpha))?;
        Ok(())
    }
}

/// Generation state for [`FireflyAlgorithm`].
#[derive(Debug, Clone)]
pub struct FireflyState<B: Backend> {
    /// Current positions, shape `(pop_size, D)`.
    positions: Tensor<B, 2>,
    /// Host-side fitness cache.
    fitness: Vec<f32>,
    /// Best-so-far genome.
    best_genome: Option<Tensor<B, 2>>,
    /// Best-so-far fitness.
    best_fitness: f32,
    /// Generation counter.
    generation: usize,
}

impl<B: Backend> FireflyState<B> {
    /// Assembles a firefly state, checking the fitness cache matches `pop`.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `positions` has zero rows or if `fitness`
    /// is non-empty with a length other than `pop_size`.
    pub fn try_new(
        positions: Tensor<B, 2>,
        fitness: Vec<f32>,
        best_genome: Option<Tensor<B, 2>>,
        best_fitness: f32,
        generation: usize,
    ) -> Result<Self, ConfigError> {
        let pop = positions.dims()[0];
        config::nonzero("FireflyState", "pop_size", pop)?;
        len_matches_pop("FireflyState", "fitness", pop, fitness.len())?;
        Ok(Self {
            positions,
            fitness,
            best_genome,
            best_fitness,
            generation,
        })
    }

    /// Current positions, shape `(pop_size, D)`.
    #[must_use]
    pub fn positions(&self) -> &Tensor<B, 2> {
        &self.positions
    }

    /// Host-side fitness cache (empty at bootstrap, else `pop_size` long).
    #[must_use]
    pub fn fitness(&self) -> &[f32] {
        &self.fitness
    }

    /// Best-so-far genome, or `None` before the first `tell`.
    #[must_use]
    pub fn best_genome(&self) -> Option<&Tensor<B, 2>> {
        self.best_genome.as_ref()
    }

    /// Best-so-far (canonical, maximise) fitness.
    #[must_use]
    pub fn best_fitness(&self) -> f32 {
        self.best_fitness
    }

    /// Generation counter.
    #[must_use]
    pub fn generation(&self) -> usize {
        self.generation
    }
}

/// Firefly Algorithm strategy.
///
/// When the `custom-kernels` feature is **off**, [`FireflyConfig`] enforces a
/// `pop_size <= FIREFLY_PURE_TENSOR_CAP` (= 128) cap through
/// [`Validate::validate`] at the harness chokepoint, since the pure-tensor path
/// materializes an `(N, N, D)` pairwise tensor. With the feature on the same
/// cap is surfaced via a `debug_assert!` in [`Strategy::init`], because the
/// fused kernel [`super::kernels::pairwise_attract_cube`] is still designed-only
/// and the strategy keeps using the pure-tensor path in the meantime.
///
/// # Example
///
/// ```no_run
/// use burn::backend::Flex;
/// use rlevo_evolution::algorithms::metaheuristic::firefly::{FireflyAlgorithm, FireflyConfig};
///
/// let strategy = FireflyAlgorithm::<Flex>::new();
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
    /// without the `custom-kernels` feature. The fused `CubeCL` kernel
    /// designed in [`super::kernels::pairwise_attract_cube`] slots in at
    /// this call site once it lands.
    fn pure_tensor_attract(
        positions: &Tensor<B, 2>,
        fitness: &[f32],
        beta0: f32,
        gamma: f32,
        alpha: f32,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
        noise_seed: u64,
    ) -> Tensor<B, 2> {
        let pop = fitness.len();
        let shape = positions.dims();
        let d = shape[1];

        // Pairwise squared distances via (x·x^T + ||x||² - 2x·x^T).
        // Cheaper memory than the (N, N, D) difference tensor, but we
        // still need the (N, N, D) tensor for the displacement `x_j -
        // x_i`. Cap enforced at module level.
        let xi = positions.clone().unsqueeze_dim::<3>(1); // (N, 1, D)
        let xj = positions.clone().unsqueeze_dim::<3>(0); // (1, N, D)
        let diff = xj.expand([pop, pop, d]) - xi.expand([pop, pop, d]); // (N, N, D)
        let r2 = diff.clone().powi_scalar(2).sum_dim(2).squeeze_dim::<2>(2); // (N, N)
        let beta = r2.mul_scalar(-gamma).exp().mul_scalar(beta0); // (N, N)

        // Brightness mask: bright[i, j] = 1 iff fitness[j] > fitness[i].
        let mut bright = vec![0i64; pop * pop];
        for i in 0..pop {
            for j in 0..pop {
                if fitness[j] > fitness[i] {
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
        let attr_sum = weighted.sum_dim(1).squeeze_dim::<2>(1); // (N, D)

        // Noise: α · (U[0,1] - 0.5). Host-sample from the supplied seed so
        // the draw is reproducible across thread schedules rather than
        // racing the process-wide Flex RNG.
        let mut noise_rng = rand::rngs::StdRng::seed_from_u64(noise_seed);
        let mut noise_rows = Vec::with_capacity(pop * d);
        for _ in 0..pop * d {
            noise_rows.push(noise_rng.random::<f32>() - 0.5);
        }
        let noise = Tensor::<B, 2>::from_data(TensorData::new(noise_rows, [pop, d]), device);
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

    /// Build the initial swarm by host-sampling `pop_size` positions
    /// uniformly in `[bounds.lo, bounds.hi]`.
    ///
    /// Positions are drawn from a deterministic [`seed_stream`] so
    /// initialisation is bit-stable regardless of core count or test
    /// ordering; the process-wide Flex RNG is never touched.
    ///
    /// The `pop_size <= FIREFLY_PURE_TENSOR_CAP` cap (without `custom-kernels`)
    /// is enforced by [`FireflyConfig`]'s [`Validate`] impl at the harness
    /// chokepoint.
    fn init(
        &self,
        params: &FireflyConfig,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> FireflyState<B> {
        debug_assert!(
            params.validate().is_ok(),
            "invalid FireflyConfig reached init: {params:?}"
        );
        // Even with the kernel feature active, the fused pairwise-attract
        // kernel is currently a design placeholder and the pure-tensor
        // path is still in use. A debug assert surfaces the limitation in
        // tests without blocking downstream users who have wired in their
        // own kernel.
        #[cfg(feature = "custom-kernels")]
        debug_assert!(
            params.pop_size <= FIREFLY_PURE_TENSOR_CAP,
            "Firefly pop_size > {FIREFLY_PURE_TENSOR_CAP} requires the fused pairwise-attract kernel; \
             the placeholder kernel module still runs the pure-tensor path"
        );
        let (lo, hi): (f32, f32) = params.bounds.into();
        // Host-sample the initial swarm from a deterministic `seed_stream`
        // rather than the process-wide Flex RNG (`B::seed` + `Tensor::random`),
        // whose draws interleave with sibling tests under the parallel runner
        // and are not reproducible across thread schedules.
        let pop = params.pop_size;
        let genome_dim = params.genome_dim;
        let mut stream = seed_stream(rng.next_u64(), 0, SeedPurpose::Init);
        let mut position_rows = Vec::with_capacity(pop * genome_dim);
        for _ in 0..pop * genome_dim {
            position_rows.push(lo + (hi - lo) * stream.random::<f32>());
        }
        let positions =
            Tensor::<B, 2>::from_data(TensorData::new(position_rows, [pop, genome_dim]), device);
        FireflyState {
            positions,
            fitness: Vec::new(),
            best_genome: None,
            best_fitness: f32::NEG_INFINITY,
            generation: 0,
        }
    }

    /// Propose the next swarm positions.
    ///
    /// On the first call (`state.fitness` is empty) returns the initial
    /// positions unchanged so the caller can evaluate generation zero.
    /// On subsequent calls, computes the pairwise attractiveness update
    /// via `pure_tensor_attract` and clips positions to
    /// `params.bounds`. The noise seed is derived from the host RNG
    /// through [`seed_stream`], keeping draws reproducible.
    fn ask(
        &self,
        params: &FireflyConfig,
        state: &FireflyState<B>,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
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
        let (lo, hi): (f32, f32) = params.bounds.into();
        let new_positions = (state.positions.clone() + delta).clamp(lo, hi);

        let mut next = state.clone();
        next.positions.clone_from(&new_positions);
        (new_positions, next)
    }

    /// Ingest fitness values, update the swarm, and advance the generation counter.
    ///
    /// Pulls `fitness` to host, updates `state.positions` and
    /// `state.fitness`, then refreshes the best-so-far genome if the
    /// current generation contains a new maximum.  Returns the updated
    /// state and a [`StrategyMetrics`] snapshot for the completed
    /// generation.
    fn tell(
        &self,
        _params: &FireflyConfig,
        population: Tensor<B, 2>,
        fitness: Tensor<B, 1>,
        mut state: FireflyState<B>,
        _rng: &mut dyn Rng,
    ) -> (FireflyState<B>, StrategyMetrics) {
        let fitness_host = fitness
            .into_data()
            .into_vec::<f32>()
            .expect("fitness tensor must be readable as f32");
        let device = population.device();
        state.fitness.clone_from(&fitness_host);
        state.positions.clone_from(&population);

        let best_idx = argmax_host(&fitness_host);
        if fitness_host[best_idx] > state.best_fitness {
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
        state.best_fitness = m.best_fitness_ever();
        (state, m)
    }

    /// Returns the best-so-far `(genome, fitness)` pair, or `None` before
    /// the first [`tell`](Strategy::tell) call.
    fn best(&self, state: &FireflyState<B>) -> Option<(Tensor<B, 2>, f32)> {
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
    use burn::backend::Flex;
    use rand::rngs::StdRng;
    use rlevo_core::fitness::FitnessEvaluable;

    type TestBackend = Flex;

    #[test]
    fn try_new_checks_fitness_length() {
        let device = Default::default();
        let pos = Tensor::<TestBackend, 2>::zeros([3, 2], &device);
        assert!(FireflyState::try_new(pos.clone(), vec![1.0; 3], None, 1.0, 1).is_ok());
        assert!(FireflyState::try_new(pos.clone(), vec![], None, f32::MIN, 0).is_ok());
        assert!(FireflyState::try_new(pos, vec![1.0; 2], None, 1.0, 1).is_err());
        let empty = Tensor::<TestBackend, 2>::zeros([0, 2], &device);
        assert!(FireflyState::try_new(empty, vec![], None, 1.0, 0).is_err());
    }

    #[test]
    fn default_config_validates() {
        assert!(FireflyConfig::default_for(32, 10).validate().is_ok());
    }

    #[test]
    fn default_gamma_matches_inverse_length_squared() {
        let cfg = FireflyConfig::default_for(32, 10);
        let (lo, hi): (f32, f32) = cfg.bounds.into();
        let length: f32 = hi - lo;
        let expected: f32 = 1.0 / (length * length);
        approx::assert_relative_eq!(cfg.gamma, expected);
    }

    #[test]
    fn rejects_zero_gamma() {
        let mut cfg = FireflyConfig::default_for(32, 10);
        cfg.gamma = 0.0;
        assert_eq!(cfg.validate().unwrap_err().field, "gamma");
    }

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
        )
        .expect("valid params");
        harness.reset();
        while !harness.step(()).done {}
        let best = harness.latest_metrics().unwrap().best_fitness_ever();
        assert!(best < 1.0, "Firefly D10 best={best}");
    }

    /// Fitness fn: row 0 → `NaN`, the rest finite. `Maximize` so natural ==
    /// canonical, exercising the ADR-0034 harness sanitize with no `neg()`.
    struct PartialNanFitness;
    impl<B: Backend> crate::fitness::BatchFitnessFn<B, Tensor<B, 2>> for PartialNanFitness {
        fn evaluate_batch(
            &mut self,
            population: &Tensor<B, 2>,
            device: &<B as burn::tensor::backend::BackendTypes>::Device,
        ) -> Tensor<B, 1> {
            let n = population.dims()[0];
            #[allow(clippy::cast_precision_loss)]
            let mut vals: Vec<f32> = (0..n).map(|i| -(i as f32)).collect();
            vals[0] = f32::NAN;
            Tensor::<B, 1>::from_data(TensorData::new(vals, [n]), device)
        }
        fn sense(&self) -> rlevo_core::objective::ObjectiveSense {
            rlevo_core::objective::ObjectiveSense::Maximize
        }
    }

    // Gap (a): the validator rejects a zero swarm and negative kernel scalars.
    // `gamma` is `positive` (0 and negatives rejected — 0 is the existing case);
    // `beta0` and `alpha` are `[0, ∞)` (0 allowed, negatives rejected).
    #[test]
    fn rejects_invalid_configs() {
        let mut cfg = FireflyConfig::default_for(0, 10);
        assert_eq!(cfg.validate().unwrap_err().field, "pop_size");

        cfg = FireflyConfig::default_for(32, 10);
        cfg.gamma = -1.0;
        assert_eq!(cfg.validate().unwrap_err().field, "gamma");

        cfg = FireflyConfig::default_for(32, 10);
        cfg.beta0 = -1.0;
        assert_eq!(cfg.validate().unwrap_err().field, "beta0");

        cfg = FireflyConfig::default_for(32, 10);
        cfg.alpha = -1.0;
        assert_eq!(cfg.validate().unwrap_err().field, "alpha");
    }

    // Gap (a) cont.: an inverted range is unrepresentable — `Bounds::new` panics
    // before a `FireflyConfig` can carry `(5, −5)`.
    #[test]
    #[should_panic(expected = "invalid range")]
    fn inverted_bounds_are_unrepresentable() {
        let _ = FireflyConfig {
            bounds: Bounds::new(5.0, -5.0),
            ..FireflyConfig::default_for(32, 10)
        };
    }

    // Gap (g): `pop_size > 128` on the pure-tensor path panics in `init` in a
    // debug (test) build — either via the `Validate` `debug_assert!` (feature
    // off) or the explicit cap `debug_assert!` (feature on). Either way the
    // oversized swarm never materializes its cubic pairwise tensor.
    #[test]
    // The panic message differs by feature (`Validate` debug_assert vs. the cap
    // debug_assert), so no single `expected` substring covers both builds.
    #[allow(clippy::should_panic_without_expect)]
    #[should_panic]
    fn pop_size_over_cap_panics_in_init() {
        let device = Default::default();
        let strategy = FireflyAlgorithm::<TestBackend>::new();
        let params = FireflyConfig::default_for(FIREFLY_PURE_TENSOR_CAP + 1, 4);
        let mut rng = StdRng::seed_from_u64(0);
        let _ = strategy.init(&params, &mut rng, &device);
    }

    // Gap (b): the pure-tensor attraction kernel. Firefly 0 is dimmer, so it is
    // pulled toward the brighter firefly 1 (positive displacement); firefly 1 has
    // no brighter neighbour, so its displacement is zero. With `gamma = 0` the
    // attractiveness is exactly `beta0`, and `alpha = 0` removes noise, making
    // the displacement exact. The output shape is `(pop, d)`.
    #[test]
    fn pure_tensor_attract_pulls_toward_brighter() {
        let device = Default::default();
        // 2-D positions (row-major): firefly 0 at (0, 0), firefly 1 at (1, 0).
        let positions = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![0.0_f32, 0.0, 1.0, 0.0], [2, 2]),
            &device,
        );
        // firefly 1 is brighter (higher canonical fitness).
        let fitness = [0.0_f32, 1.0];
        let delta = FireflyAlgorithm::<TestBackend>::pure_tensor_attract(
            &positions, &fitness, 1.0, // beta0
            0.0, // gamma → attractiveness == beta0
            0.0, // alpha → no noise
            &device, 0,
        );
        assert_eq!(delta.dims(), [2, 2], "displacement is (pop, d)");
        let d = delta
            .into_data()
            .into_vec::<f32>()
            .expect("delta readable as f32");
        // Firefly 0 moves +1·(x_1 − x_0) = (+1, 0) toward the brighter one.
        approx::assert_relative_eq!(d[0], 1.0, epsilon = 1e-6);
        approx::assert_relative_eq!(d[1], 0.0, epsilon = 1e-6);
        // Brightest firefly has no brighter neighbour → no attraction.
        approx::assert_relative_eq!(d[2], 0.0, epsilon = 1e-6);
        approx::assert_relative_eq!(d[3], 0.0, epsilon = 1e-6);
    }

    // Gap (b') / issue #233: the `genome_dim = 1` (D == 1) pure-tensor path. The
    // reductions `sum_dim(2).squeeze_dim::<2>(2)` and `sum_dim(1).squeeze_dim::<2>(1)`
    // now strip only the reduced axis, so the trailing size-1 genome axis survives
    // and the kernel no longer panics in burn's `Squeeze` rank-check. Firefly 0 is
    // dimmer, so it is pulled toward the brighter firefly 1 along the single axis.
    #[test]
    fn pure_tensor_attract_d1_pulls_toward_brighter() {
        let device = Default::default();
        // 1-D positions: firefly 0 at [0.0], firefly 1 at [1.0].
        let positions = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(vec![0.0_f32, 1.0], [2, 1]),
            &device,
        );
        // firefly 1 is brighter (higher canonical fitness).
        let fitness = [0.0_f32, 1.0];
        let delta = FireflyAlgorithm::<TestBackend>::pure_tensor_attract(
            &positions, &fitness, 1.0, // beta0
            0.0, // gamma → attractiveness == beta0
            0.0, // alpha → no noise
            &device, 0,
        );
        assert_eq!(delta.dims(), [2, 1], "displacement is (pop, d)");
        let d = delta
            .into_data()
            .into_vec::<f32>()
            .expect("delta readable as f32");
        // Firefly 0 moves +1·(x_1 − x_0) = +1 toward the brighter one.
        approx::assert_relative_eq!(d[0], 1.0, epsilon = 1e-6);
        // Brightest firefly has no brighter neighbour → no attraction.
        approx::assert_relative_eq!(d[1], 0.0, epsilon = 1e-6);
    }

    // Gap (c): `argmax_host` edge cases. Empty slice panics; an all-`NaN` slice
    // (nothing exceeds the `−∞` seed) falls back to index 0; a single element is
    // trivially the max.
    #[test]
    #[should_panic(expected = "must be non-empty")]
    fn argmax_host_empty_panics() {
        let _ = argmax_host(&[]);
    }

    #[test]
    fn argmax_host_all_nan_and_single() {
        assert_eq!(argmax_host(&[f32::NAN, f32::NAN, f32::NAN]), 0);
        assert_eq!(argmax_host(&[7.0]), 0);
    }

    // Gap (d): the bootstrap `ask` (empty `fitness`) returns positions verbatim.
    #[test]
    #[allow(clippy::float_cmp)] // byte-identical: `ask` clones `state.positions`
    fn first_ask_returns_positions_unchanged() {
        let device = Default::default();
        let strategy = FireflyAlgorithm::<TestBackend>::new();
        let params = FireflyConfig::default_for(8, 4);
        let mut rng = StdRng::seed_from_u64(1);
        let state = strategy.init(&params, &mut rng, &device);
        let (genome, next) = strategy.ask(&params, &state, &mut rng, &device);
        let before = state
            .positions()
            .clone()
            .into_data()
            .into_vec::<f32>()
            .expect("positions readable as f32");
        let after = genome
            .into_data()
            .into_vec::<f32>()
            .expect("genome readable as f32");
        assert_eq!(before, after);
        assert!(next.fitness().is_empty());
    }

    // Gap (e): the best-so-far accessor is `None` until a `tell` records one.
    #[test]
    fn best_is_none_before_first_tell() {
        let device = Default::default();
        let strategy = FireflyAlgorithm::<TestBackend>::new();
        let params = FireflyConfig::default_for(8, 4);
        let mut rng = StdRng::seed_from_u64(2);
        let state = strategy.init(&params, &mut rng, &device);
        assert!(strategy.best(&state).is_none());
    }

    // Gap (f): every proposed position is clamped into `bounds` after `ask`,
    // across 32 seeds.
    #[test]
    fn proposed_positions_within_bounds() {
        let device = Default::default();
        let strategy = FireflyAlgorithm::<TestBackend>::new();
        let params = FireflyConfig::default_for(10, 4);
        let (lo, hi): (f32, f32) = params.bounds.into();
        // A steady-state swarm with a non-empty fitness cache so `ask` runs the
        // attraction update rather than the bootstrap early return.
        let mut rng = StdRng::seed_from_u64(0);
        let base = strategy.init(&params, &mut rng, &device);
        #[allow(clippy::cast_precision_loss)]
        let fitness: Vec<f32> = (0..params.pop_size).map(|i| -(i as f32)).collect();
        // Firefly's `ask` reads positions + fitness only, never `best_genome`.
        let state = FireflyState::try_new(base.positions().clone(), fitness, None, 0.0, 1)
            .expect("valid steady state");
        for seed in 0..32 {
            let mut rng = StdRng::seed_from_u64(seed);
            let (pos, _next) = strategy.ask(&params, &state, &mut rng, &device);
            let vals = pos
                .into_data()
                .into_vec::<f32>()
                .expect("positions readable as f32");
            for &v in &vals {
                assert!(
                    v >= lo && v <= hi,
                    "position {v} out of bounds [{lo}, {hi}] (seed {seed})"
                );
            }
        }
    }

    // Gap: a partly-`NaN` objective is neutralized by the harness sanitize
    // chokepoint (ADR 0034).
    #[test]
    fn nan_fitness_survives_harness() {
        let device = Default::default();
        let strategy = FireflyAlgorithm::<TestBackend>::new();
        let params = FireflyConfig::default_for(8, 3);
        let mut harness = EvolutionaryHarness::<TestBackend, _, _>::new(
            strategy,
            params,
            PartialNanFitness,
            4,
            device,
            4,
        )
        .expect("valid params");
        harness.reset();
        while !harness.step(()).done {}
        let m = harness.latest_metrics().unwrap();
        assert!(
            m.best_fitness_ever().is_finite(),
            "best_fitness_ever not finite: {}",
            m.best_fitness_ever()
        );
        assert!(m.broken_count() > 0, "expected a broken (NaN) member");
    }

    // Gap (b') cont. / issue #233: a full `genome_dim = 1` run drives the harness
    // to completion without panicking and records a finite best.
    #[test]
    fn boundary_genome_dim_one_runs() {
        let device = Default::default();
        let strategy = FireflyAlgorithm::<TestBackend>::new();
        let params = FireflyConfig::default_for(8, 1);
        let fitness_fn = FromFitnessEvaluable::new(SphereFit, Sphere);
        let mut harness = EvolutionaryHarness::<TestBackend, _, _>::new(
            strategy, params, fitness_fn, 6, device, 6,
        )
        .expect("valid params");
        harness.reset();
        while !harness.step(()).done {}
        assert!(
            harness
                .latest_metrics()
                .unwrap()
                .best_fitness_ever()
                .is_finite(),
            "non-finite best for genome_dim = 1"
        );
    }
}
