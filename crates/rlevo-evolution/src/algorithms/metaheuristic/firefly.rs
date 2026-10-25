//! Firefly Algorithm.
//!
//! Each firefly `i` moves toward every **brighter** firefly `j`, with
//! attractiveness decaying exponentially in the squared distance:
//!
//! - `$\beta(r_{ij}) = \beta_0 \cdot \exp(-\gamma r_{ij}^2)$` where
//!   `$r_{ij} = \lVert x_i - x_j \rVert$`,
//! - `$\Delta x_i = \sum_{j : f(x_j) > f(x_i)} \beta(r_{ij}) \cdot (x_j - x_i) + \alpha \cdot (U[-0.5, 0.5])$`,
//! - `$x_i \leftarrow x_i + \Delta x_i$`.
//!
//! The attraction sum is canonically `$O(N^2)$`, and this implementation
//! materializes an `(N, N, D)` pairwise-difference tensor, so its device
//! memory grows with the square of the swarm. [`FireflyConfig`] therefore
//! rejects a swarm larger than [`FIREFLY_PURE_TENSOR_CAP`].
//!
//! # Candor
//!
//! Legacy comparator. Camacho Villalón, Dorigo & Stützle (2020, 2023)
//! show that the pairwise attraction reduces to a PSO-style attractor
//! under a new metaphor; it has no novel search mechanism. Ship it for
//! API coverage; prefer [CMA-ES](crate::algorithms::cma_es) for serious
//! work.
//!
//! # References
//!
//! - Yang (2008), *Nature-Inspired Metaheuristic Algorithms*.
//! - Camacho Villalón, Dorigo & Stützle (2020), *Grey Wolf, Firefly and
//!   Bat Algorithms: Three Widespread Algorithms that Do Not Contain
//!   Any Novelty*.
//! - Camacho Villalón, Dorigo & Stützle (2023), *Exposing the grey wolf,
//!   moth-flame, whale, firefly, bat, and antlion algorithms: six misleading
//!   optimization techniques inspired by bestial metaphors*, International
//!   Transactions in Operational Research 30:2945-2971.

use burn::tensor::{Device, Int, Tensor, TensorData};
use rand::Rng;
use rand::RngExt;
use rand::SeedableRng;

use rlevo_core::bounds::Bounds;
use rlevo_core::config::{self, ConfigError, Validate};

use super::len_matches_pop;
use crate::ops::selection::argmax_host;
use crate::rng::{SeedPurpose, seed_stream};
use crate::strategy::{Strategy, StrategyMetrics};

/// Largest swarm [`FireflyConfig`] accepts. The attraction step allocates an
/// `(N, N, D)` tensor on the device, so its memory grows as `$O(N^2 D)$`.
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
    /// Base attractiveness `$\beta_0$`. Canonical 1.0.
    pub beta0: f32,
    /// Light-absorption coefficient `$\gamma$`; controls the range over which
    /// fireflies can see each other. It is used as given, not rescaled to
    /// `bounds`. Yang's canonical `$\gamma = 1$` assumes a domain normalized to
    /// `$[0, 1]$`; [`default_for`](Self::default_for) sets `$\gamma = 1 / L^2$`
    /// for its domain of extent `L`.
    pub gamma: f32,
    /// Noise scale for the random walk term. Canonical 0.2.
    pub alpha: f32,
}

impl FireflyConfig {
    /// Default configuration. `$\gamma$` is set from the extent `L` of the
    /// default `$[-5.12, 5.12]$` domain so the exponential decay lands in a
    /// useful regime — Yang's canonical `$\gamma = 1$` assumes `$[0, 1]$`
    /// normalization, and `$\gamma = 1 / L^2$` keeps attractiveness
    /// non-vanishing across pairs. To search other `bounds`, set `gamma` to
    /// match: it is not rescaled.
    #[must_use]
    pub fn default_for(pop_size: usize, genome_dim: usize) -> Self {
        let (lo, hi): (f32, f32) = (-5.12, 5.12);
        let length: f32 = hi - lo;
        // `$\gamma \approx 1/L^2$`, Yang's canonical regime scaled to the domain extent.
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
        // The attraction step materialises an (N, N, D) tensor, so cap the
        // swarm at FIREFLY_PURE_TENSOR_CAP.
        config::at_most(C, "pop_size", self.pop_size, FIREFLY_PURE_TENSOR_CAP)?;
        config::nonzero(C, "genome_dim", self.genome_dim)?;
        config::in_range(C, "beta0", 0.0, f64::INFINITY, f64::from(self.beta0))?;
        config::positive(C, "gamma", f64::from(self.gamma))?;
        config::in_range(C, "alpha", 0.0, f64::INFINITY, f64::from(self.alpha))?;
        Ok(())
    }
}

/// Generation state for [`FireflyAlgorithm`].
#[derive(Debug, Clone)]
pub struct FireflyState {
    /// Current positions, shape `(pop_size, D)`.
    positions: Tensor<2>,
    /// Host-side fitness cache.
    fitness: Vec<f32>,
    /// Best-so-far genome.
    best_genome: Option<Tensor<2>>,
    /// Best-so-far fitness.
    best_fitness: f32,
    /// Generation counter.
    generation: usize,
}

impl FireflyState {
    /// Assembles a firefly state, checking the fitness cache matches `pop`.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `positions` has zero rows or if `fitness`
    /// is non-empty with a length other than `pop_size`.
    pub fn try_new(
        positions: Tensor<2>,
        fitness: Vec<f32>,
        best_genome: Option<Tensor<2>>,
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
    pub fn positions(&self) -> &Tensor<2> {
        &self.positions
    }

    /// Host-side fitness cache (empty at bootstrap, else `pop_size` long).
    #[must_use]
    pub fn fitness(&self) -> &[f32] {
        &self.fitness
    }

    /// Best-so-far genome, or `None` before the first `tell`.
    #[must_use]
    pub fn best_genome(&self) -> Option<&Tensor<2>> {
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
/// [`FireflyConfig`] enforces `pop_size <= FIREFLY_PURE_TENSOR_CAP` (= 128)
/// through [`Validate::validate`] at the loop chokepoint, since the
/// attraction step materializes an `(N, N, D)` pairwise tensor.
///
/// # Example
///
/// ```no_run
/// use rlevo_evolution::algorithms::metaheuristic::firefly::{FireflyAlgorithm, FireflyConfig};
///
/// let strategy = FireflyAlgorithm::new();
/// let params = FireflyConfig::default_for(32, 10);
/// let _ = (strategy, params);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct FireflyAlgorithm {}

impl FireflyAlgorithm {
    /// Builds a new (stateless) strategy object.
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }

    /// The `$O(N^2 D)$` attraction step, composed from Burn tensor primitives
    /// so it runs on every backend.
    fn pure_tensor_attract(
        positions: &Tensor<2>,
        fitness: &[f32],
        beta0: f32,
        gamma: f32,
        alpha: f32,
        device: &Device,
        noise_seed: u64,
    ) -> Tensor<2> {
        let pop = fitness.len();
        let shape = positions.dims();
        let d = shape[1];

        // Pairwise squared distances via `$(x x^\top + \lVert x\rVert^2 - 2 x x^\top)$`. Cheaper
        // memory than the (N, N, D) difference tensor, but we still need the (N, N, D) tensor for
        // the displacement `x_j - x_i`. Cap enforced at module level.
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
            Tensor::<2, Int>::from_data(TensorData::new(bright, [pop, pop]), device).equal_elem(1);
        // Zero-out non-bright pairs in `$\beta$` then multiply diff.
        let zero = Tensor::<2>::zeros([pop, pop], device);
        let beta_m = beta.mask_where(bright_mask.bool_not(), zero);
        let weight = beta_m.unsqueeze_dim::<3>(2).expand([pop, pop, d]); // (N, N, D)
        let weighted = diff.mul(weight); // (N, N, D)
        let attr_sum = weighted.sum_dim(1).squeeze_dim::<2>(1); // (N, D)

        // Noise: `$\alpha \cdot (U[0,1] - 0.5)$`. Host-sample from the supplied seed so the draw is
        // reproducible across thread schedules rather than racing the process-wide Flex RNG.
        let mut noise_rng = rand::rngs::StdRng::seed_from_u64(noise_seed);
        let mut noise_rows = Vec::with_capacity(pop * d);
        for _ in 0..pop * d {
            noise_rows.push(noise_rng.random::<f32>() - 0.5);
        }
        let noise = Tensor::<2>::from_data(TensorData::new(noise_rows, [pop, d]), device);
        attr_sum + noise.mul_scalar(alpha)
    }
}

impl Strategy for FireflyAlgorithm {
    type Params = FireflyConfig;
    type State = FireflyState;
    type Genome = Tensor<2>;

    /// Build the initial swarm by host-sampling `pop_size` positions
    /// uniformly in `[bounds.lo, bounds.hi]`.
    ///
    /// Positions are drawn from a deterministic [`seed_stream`] so
    /// initialisation is bit-stable regardless of core count or test
    /// ordering; the process-wide Flex RNG is never touched.
    ///
    /// The `pop_size <= FIREFLY_PURE_TENSOR_CAP` cap is enforced by
    /// [`FireflyConfig`]'s [`Validate`] impl at the loop chokepoint.
    fn init(&self, params: &FireflyConfig, rng: &mut dyn Rng, device: &Device) -> FireflyState {
        debug_assert!(
            params.validate().is_ok(),
            "invalid FireflyConfig reached init: {params:?}"
        );
        let (lo, hi): (f32, f32) = params.bounds.into();
        // Host-sample the initial swarm from a deterministic `seed_stream`
        // rather than the process-wide Flex RNG (`Device::seed` + `Tensor::random`),
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
            Tensor::<2>::from_data(TensorData::new(position_rows, [pop, genome_dim]), device);
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
        state: &FireflyState,
        rng: &mut dyn Rng,
        device: &Device,
    ) -> (Tensor<2>, FireflyState) {
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
        population: Tensor<2>,
        fitness: Tensor<1>,
        mut state: FireflyState,
        _rng: &mut dyn Rng,
    ) -> (FireflyState, StrategyMetrics) {
        // Sanitize at the pull (NaN → `$-\infty$`, `$+\infty$` → f32::MAX). This is the per-site
        // correctness floor for a caller driving `ask`/`tell` directly instead of
        // `EvolutionaryLoop::step`, which already sanitizes (the ADR 0034 decision-3 bypass
        // hole): otherwise a raw NaN lands in the public `state.fitness` cache and silently
        // disables the `fitness[j] > fitness[i]` brightness ordering in the next `ask`.
        // `sanitize_fitness` is idempotent, so on the loop path this is a provable no-op — not
        // redundant, load-bearing.
        let fitness_host: Vec<f32> = fitness
            .into_data()
            .try_into_vec::<f32>()
            .expect("fitness tensor must be readable as f32")
            .into_iter()
            .map(crate::fitness::sanitize_fitness)
            .collect();
        let device = population.device();
        state.fitness.clone_from(&fitness_host);
        state.positions.clone_from(&population);

        let best_idx = argmax_host(&fitness_host);
        if fitness_host[best_idx] > state.best_fitness {
            state.best_fitness = fitness_host[best_idx];
            #[allow(clippy::cast_possible_wrap)]
            let idx =
                Tensor::<1, Int>::from_data(TensorData::new(vec![best_idx as i64], [1]), &device);
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
    fn best(&self, state: &FireflyState) -> Option<(Tensor<2>, f32)> {
        state
            .best_genome
            .as_ref()
            .map(|g| (g.clone(), state.best_fitness))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::FromLandscape;
    use crate::strategy::EvolutionaryLoop;
    use rand::rngs::StdRng;
    use rlevo_core::config::ConstraintKind;
    use rlevo_core::fitness::Landscape;

    #[test]
    fn try_new_checks_fitness_length() {
        let device = Device::flex();
        let pos = Tensor::<2>::zeros([3, 2], &device);
        assert!(FireflyState::try_new(pos.clone(), vec![1.0; 3], None, 1.0, 1).is_ok());
        assert!(FireflyState::try_new(pos.clone(), vec![], None, f32::MIN, 0).is_ok());
        assert!(FireflyState::try_new(pos, vec![1.0; 2], None, 1.0, 1).is_err());
        let empty = Tensor::<2>::zeros([0, 2], &device);
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
    impl Landscape for Sphere {
        fn evaluate(&self, x: &[f64]) -> f64 {
            x.iter().map(|v| v * v).sum()
        }
    }

    #[test]
    fn firefly_converges_on_sphere_d10() {
        // Firefly's attraction sum is `$O(N^2 D)$`; we use 24 fireflies to keep the test fast while
        // still exercising the pairwise kernel path.
        let device = Device::flex();
        let strategy = FireflyAlgorithm::new();
        let params = FireflyConfig::default_for(24, 10);
        let fitness_fn = FromLandscape::new(Sphere);
        let mut evo_loop = EvolutionaryLoop::new(strategy, params, fitness_fn, 29, device, 500)
            .expect("valid params");
        evo_loop.reset();
        while !evo_loop.is_done() {
            evo_loop.step();
        }
        let best = evo_loop.latest_metrics().unwrap().best_fitness_ever();
        assert!(best < 1.0, "Firefly D10 best={best}");
    }

    /// Fitness fn: row 0 → `NaN`, the rest finite. `Maximize` so natural ==
    /// canonical, exercising the ADR-0034 loop sanitize with no `neg()`.
    struct PartialNanFitness;
    impl crate::fitness::BatchFitnessFn<Tensor<2>> for PartialNanFitness {
        fn evaluate_batch(&mut self, population: &Tensor<2>, device: &Device) -> Tensor<1> {
            let n = population.dims()[0];
            #[allow(clippy::cast_precision_loss)]
            let mut vals: Vec<f32> = (0..n).map(|i| -(i as f32)).collect();
            vals[0] = f32::NAN;
            Tensor::<1>::from_data(TensorData::new(vals, [n]), device)
        }
        fn sense(&self) -> rlevo_core::objective::ObjectiveSense {
            rlevo_core::objective::ObjectiveSense::Maximize
        }
    }

    // Gap (a): the validator rejects a zero swarm and negative kernel scalars. `gamma` is
    // `positive` (0 and negatives rejected — 0 is the existing case); `beta0` and `alpha` are
    // `$[0, \infty)$` (0 allowed, negatives rejected).
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

    // Gap (a) cont.: an inverted range is unrepresentable — `Bounds::new` panics before a
    // `FireflyConfig` can carry `$(5, -5)$`.
    #[test]
    #[should_panic(expected = "invalid range")]
    fn inverted_bounds_are_unrepresentable() {
        let _ = FireflyConfig {
            bounds: Bounds::new(5.0, -5.0),
            ..FireflyConfig::default_for(32, 10)
        };
    }

    // Gap (g): the cap is exact. `validate` accepts a swarm of exactly
    // `FIREFLY_PURE_TENSOR_CAP` and rejects one more, naming the field and
    // both numbers. This is the only guard a release build has.
    #[test]
    fn validate_caps_pop_size_at_the_pure_tensor_cap() {
        assert!(
            FireflyConfig::default_for(FIREFLY_PURE_TENSOR_CAP, 4)
                .validate()
                .is_ok()
        );
        let err = FireflyConfig::default_for(FIREFLY_PURE_TENSOR_CAP + 1, 4)
            .validate()
            .unwrap_err();
        assert_eq!(err.field, "pop_size");
        assert_eq!(
            err.kind,
            ConstraintKind::TooLarge {
                max: FIREFLY_PURE_TENSOR_CAP as u64,
                got: FIREFLY_PURE_TENSOR_CAP as u64 + 1,
            }
        );
    }

    // An oversized swarm that skips `validate` still never materializes its
    // cubic pairwise tensor in a debug (test) build: `init` asserts the config.
    #[test]
    #[should_panic(expected = "invalid FireflyConfig reached init")]
    fn pop_size_over_cap_panics_in_init() {
        let device = Device::flex();
        let strategy = FireflyAlgorithm::new();
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
        let device = Device::flex();
        // 2-D positions (row-major): firefly 0 at (0, 0), firefly 1 at (1, 0).
        let positions = Tensor::<2>::from_data(
            TensorData::new(vec![0.0_f32, 0.0, 1.0, 0.0], [2, 2]),
            &device,
        );
        // firefly 1 is brighter (higher canonical fitness).
        let fitness = [0.0_f32, 1.0];
        let delta = FireflyAlgorithm::pure_tensor_attract(
            &positions, &fitness, 1.0, // beta0
            0.0, // gamma → attractiveness == beta0
            0.0, // alpha → no noise
            &device, 0,
        );
        assert_eq!(delta.dims(), [2, 2], "displacement is (pop, d)");
        let d = delta
            .into_data()
            .try_into_vec::<f32>()
            .expect("delta readable as f32");
        // Firefly 0 moves `$+1 \cdot (x_1 - x_0) = (+1, 0)$` toward the brighter one.
        approx::assert_relative_eq!(d[0], 1.0, epsilon = 1e-6);
        approx::assert_relative_eq!(d[1], 0.0, epsilon = 1e-6);
        // Brightest firefly has no brighter neighbour → no attraction.
        approx::assert_relative_eq!(d[2], 0.0, epsilon = 1e-6);
        approx::assert_relative_eq!(d[3], 0.0, epsilon = 1e-6);
    }

    // Covers the `genome_dim = 1` (D == 1) pure-tensor path. The
    // reductions `sum_dim(2).squeeze_dim::<2>(2)` and `sum_dim(1).squeeze_dim::<2>(1)`
    // now strip only the reduced axis, so the trailing size-1 genome axis survives
    // and the kernel no longer panics in burn's `Squeeze` rank-check. Firefly 0 is
    // dimmer, so it is pulled toward the brighter firefly 1 along the single axis.
    #[test]
    fn pure_tensor_attract_d1_pulls_toward_brighter() {
        let device = Device::flex();
        // 1-D positions: firefly 0 at [0.0], firefly 1 at [1.0].
        let positions =
            Tensor::<2>::from_data(TensorData::new(vec![0.0_f32, 1.0], [2, 1]), &device);
        // firefly 1 is brighter (higher canonical fitness).
        let fitness = [0.0_f32, 1.0];
        let delta = FireflyAlgorithm::pure_tensor_attract(
            &positions, &fitness, 1.0, // beta0
            0.0, // gamma → attractiveness == beta0
            0.0, // alpha → no noise
            &device, 0,
        );
        assert_eq!(delta.dims(), [2, 1], "displacement is (pop, d)");
        let d = delta
            .into_data()
            .try_into_vec::<f32>()
            .expect("delta readable as f32");
        // Firefly 0 moves `$+1 \cdot (x_1 - x_0) = +1$` toward the brighter one.
        approx::assert_relative_eq!(d[0], 1.0, epsilon = 1e-6);
        // Brightest firefly has no brighter neighbour → no attraction.
        approx::assert_relative_eq!(d[1], 0.0, epsilon = 1e-6);
    }

    // Gap (c): `argmax_host` edge cases. Empty slice panics; an all-`NaN` slice (nothing exceeds
    // the `$-\infty$` seed) falls back to index 0; a single element is trivially the max.
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
        let device = Device::flex();
        let strategy = FireflyAlgorithm::new();
        let params = FireflyConfig::default_for(8, 4);
        let mut rng = StdRng::seed_from_u64(1);
        let state = strategy.init(&params, &mut rng, &device);
        let (genome, next) = strategy.ask(&params, &state, &mut rng, &device);
        let before = state
            .positions()
            .clone()
            .into_data()
            .try_into_vec::<f32>()
            .expect("positions readable as f32");
        let after = genome
            .into_data()
            .try_into_vec::<f32>()
            .expect("genome readable as f32");
        assert_eq!(before, after);
        assert!(
            next.fitness().is_empty(),
            "next.fitness() must be empty, got {:?}",
            next.fitness()
        );
    }

    // Gap (e): the best-so-far accessor is `None` until a `tell` records one.
    #[test]
    fn best_is_none_before_first_tell() {
        let device = Device::flex();
        let strategy = FireflyAlgorithm::new();
        let params = FireflyConfig::default_for(8, 4);
        let mut rng = StdRng::seed_from_u64(2);
        let state = strategy.init(&params, &mut rng, &device);
        assert!(strategy.best(&state).is_none());
    }

    // Gap (f): every proposed position is clamped into `bounds` after `ask`,
    // across 32 seeds.
    #[test]
    fn proposed_positions_within_bounds() {
        let device = Device::flex();
        let strategy = FireflyAlgorithm::new();
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
                .try_into_vec::<f32>()
                .expect("positions readable as f32");
            for &v in &vals {
                assert!(
                    v >= lo && v <= hi,
                    "position {v} out of bounds [{lo}, {hi}] (seed {seed})"
                );
            }
        }
    }

    // Gap: a partly-`NaN` objective is neutralized by the loop sanitize
    // chokepoint (ADR 0034).
    #[test]
    fn nan_fitness_survives_loop() {
        let device = Device::flex();
        let strategy = FireflyAlgorithm::new();
        let params = FireflyConfig::default_for(8, 3);
        let mut evo_loop = EvolutionaryLoop::new(strategy, params, PartialNanFitness, 4, device, 4)
            .expect("valid params");
        evo_loop.reset();
        while !evo_loop.is_done() {
            evo_loop.step();
        }
        let m = evo_loop.latest_metrics().unwrap();
        assert!(
            m.best_fitness_ever().is_finite(),
            "best_fitness_ever not finite: {}",
            m.best_fitness_ever()
        );
        assert!(m.broken_count() > 0, "expected a broken (NaN) member");
    }

    /// Cache hygiene (not a freeze): `tell` overwrites `state.fitness`
    /// wholesale every generation, so a `NaN` can never latch here. What this
    /// pins is the **store** — driving `init → ask → tell` directly, the
    /// ADR-0034 decision-3 bypass of `EvolutionaryLoop::step`, must not
    /// leave a raw `NaN` in the public fitness cache. `ask` feeds that cache
    /// to the brightness ordering (`fitness[j] > fitness[i]`), where a `NaN`
    /// would silently disable one firefly's attraction for a generation.
    #[test]
    fn nan_fitness_does_not_latch_without_loop() {
        let device = Device::flex();
        let strategy = FireflyAlgorithm::new();
        let params = FireflyConfig::default_for(6, 3);
        let mut rng = StdRng::seed_from_u64(7);
        let state = strategy.init(&params, &mut rng, &device);
        let (pop, state) = strategy.ask(&params, &state, &mut rng, &device);
        // Gen-0 fitness supplied by the test, never by a landscape: [NaN, finite…].
        #[allow(clippy::cast_precision_loss)]
        let mut vals: Vec<f32> = (0..params.pop_size).map(|i| -(i as f32) - 1.0).collect();
        vals[0] = f32::NAN;
        let fitness = Tensor::<1>::from_data(TensorData::new(vals, [params.pop_size]), &device);
        let (state, _m) = strategy.tell(&params, pop, fitness, state, &mut rng);
        assert!(
            state.fitness().iter().all(|f| !f.is_nan()),
            "raw NaN reached the public fitness cache: {:?}",
            state.fitness()
        );
        // Pin the *value*, not just "not NaN": under the canonical maximise convention (ADR 0023 /
        // ADR 0034) `$-\infty$` is the worst representable fitness, and that is precisely what
        // makes a sanitized member unable to win a champion scan. Any other finite substitute (e.g.
        // `0.0`) clears `is_nan` yet would rank firefly 0 *above* every finite -1/-2/… row and make
        // the NaN-scoring firefly the brightest — the leader poisoning this regression exists to
        // catch.
        assert!(
            state.fitness()[0].is_infinite() && state.fitness()[0].is_sign_negative(),
            "sanitized NaN must land as -inf in the firefly-0 fitness cache: {:?}",
            state.fitness()
        );
    }

    // Companion boundary test: a full `genome_dim = 1` run drives the loop
    // to completion without panicking and records a finite best.
    #[test]
    fn boundary_genome_dim_one_runs() {
        let device = Device::flex();
        let strategy = FireflyAlgorithm::new();
        let params = FireflyConfig::default_for(8, 1);
        let fitness_fn = FromLandscape::new(Sphere);
        let mut evo_loop = EvolutionaryLoop::new(strategy, params, fitness_fn, 6, device, 6)
            .expect("valid params");
        evo_loop.reset();
        while !evo_loop.is_done() {
            evo_loop.step();
        }
        assert!(
            evo_loop
                .latest_metrics()
                .unwrap()
                .best_fitness_ever()
                .is_finite(),
            "non-finite best for genome_dim = 1"
        );
    }
}
