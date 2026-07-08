//! Ant Colony Optimization for continuous domains (`ACO_R`).
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

use std::marker::PhantomData;

use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use rand::Rng;
use rand::RngExt;

use rlevo_core::bounds::Bounds;
use rlevo_core::config::{self, ConfigError, Validate};

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
    pub bounds: Bounds,
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
            bounds: Bounds::new(-5.12, 5.12),
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

impl Validate for AcoRConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "AcoRConfig";
        config::at_least(C, "archive_size", self.archive_size, 2)?;
        config::at_least(C, "m", self.m, 1)?;
        config::nonzero(C, "genome_dim", self.genome_dim)?;
        config::positive(C, "xi", f64::from(self.xi))?;
        config::positive(C, "q", f64::from(self.q))?;
        Ok(())
    }
}

/// Generation state for [`AntColonyReal`].
#[derive(Debug, Clone)]
pub struct AcoRState<B: Backend> {
    /// Archive of `k` best solutions, shape `(k, D)`.
    pub archive: Tensor<B, 2>,
    /// Host-side archive fitness, sorted descending (best, i.e. highest,
    /// first) after the first `tell`.
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
/// The `archive_size >= 2` (the σ computation needs at least two archive
/// solutions to take a pairwise distance) and `m >= 1` invariants are enforced
/// by [`Validate::validate`] at the harness chokepoint.
///
/// # Example
///
/// ```no_run
/// use burn::backend::Flex;
/// use rlevo_evolution::algorithms::metaheuristic::aco_r::{AcoRConfig, AntColonyReal};
///
/// let strategy = AntColonyReal::<Flex>::new();
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
        // Drop the Gaussian-PDF normalisation constant: it cancels exactly under
        // the sum-to-one renormalisation below, and for tiny q·k it overflows to
        // +inf, producing inf/inf = NaN. `l` is 0-indexed here; equivalent to the
        // paper's 1-indexed (l−1)² after normalisation.
        let mut w: Vec<f32> = (0..archive_size)
            .map(|l| {
                #[allow(clippy::cast_precision_loss)]
                let rank = l as f32;
                (-(rank * rank) / denom).exp()
            })
            .collect();
        let total: f32 = w.iter().sum();
        if !total.is_finite() || total == 0.0 {
            // Degenerate q (e.g. q == 0 / underflow): fall back to uniform so the
            // CDF sampler in `ask` never sees all-zero / NaN weights.
            w.fill(1.0 / k);
        } else {
            for v in &mut w {
                *v /= total;
            }
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

    /// Initialises the archive by host-sampling `archive_size × genome_dim`
    /// values uniformly from `params.bounds`.
    ///
    /// All random draws go through [`seed_stream`] derived from `rng` rather
    /// than `B::seed` + `Tensor::random`; this keeps draws reproducible across
    /// thread schedules when multiple tests or harnesses share the same
    /// process-wide Burn RNG state.
    fn init(
        &self,
        params: &AcoRConfig,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> AcoRState<B> {
        debug_assert!(
            params.validate().is_ok(),
            "invalid AcoRConfig reached init: {params:?}"
        );
        let (lo, hi): (f32, f32) = params.bounds.into();
        // Host-sample the initial archive from a deterministic `seed_stream`
        // rather than the process-wide Flex RNG (`B::seed` + `Tensor::random`),
        // whose draws interleave with sibling tests under the parallel runner
        // and are not reproducible across thread schedules.
        let rows = params.archive_size;
        let genome_dim = params.genome_dim;
        let mut stream = seed_stream(rng.next_u64(), 0, SeedPurpose::Init);
        let mut archive_rows = Vec::with_capacity(rows * genome_dim);
        for _ in 0..rows * genome_dim {
            archive_rows.push(lo + (hi - lo) * stream.random::<f32>());
        }
        let archive =
            Tensor::<B, 2>::from_data(TensorData::new(archive_rows, [rows, genome_dim]), device);
        AcoRState {
            archive,
            archive_fitness: Vec::new(),
            weights: Self::compute_weights(params.archive_size, params.q),
            best_genome: None,
            best_fitness: f32::NEG_INFINITY,
            generation: 0,
        }
    }

    /// Proposes the next population.
    ///
    /// **First call** (`state.archive_fitness.is_empty()`): returns the
    /// initial archive as-is so the harness scores it before any generation
    /// update occurs.
    ///
    /// **Subsequent calls**: draws `m` offspring by, for each dimension `d`
    /// of each offspring `i`:
    ///
    /// 1. Selecting an archive index `l` by CDF-weighted roulette from
    ///    `state.weights` (host-side, via `seed_stream`).
    /// 2. Computing `σ_{l,d} = ξ · mean_e |archive_{e,d} − archive_{l,d}|`
    ///    on-device.
    /// 3. Sampling `x_{i,d} ~ N(archive_{l,d}, σ_{l,d})` host-side via
    ///    `rand_distr::Normal` and a second `seed_stream`.
    ///
    /// Offspring are clamped to `params.bounds` before upload to the device.
    /// The returned state is the same object passed in (no mutation occurs in
    /// `ask`; all state changes are deferred to [`tell`](Self::tell)).
    #[allow(clippy::many_single_char_names)]
    fn ask(
        &self,
        params: &AcoRConfig,
        state: &AcoRState<B>,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
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
        let archive_host = state
            .archive
            .clone()
            .into_data()
            .into_vec::<f32>()
            .expect("archive tensor must be readable as f32");
        let sigma_host = sigma
            .into_data()
            .into_vec::<f32>()
            .expect("sigma tensor must be readable as f32");
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
            // A non-finite σ falls back to the archive mean rather than
            // panicking (σ is already floored to 1e-12 above, so this is a
            // belt-and-braces guard). A NaN mean would pass through unchanged;
            // the clamp below bounds finite draws but does NOT launder a NaN —
            // that is neutralized by the ADR-0034 fitness-hygiene chokepoint
            // downstream.
            *out =
                crate::sampling::normal_or_mean(mean_rows[idx], sigma_rows[idx], &mut sample_rng);
        }
        let (lo, hi): (f32, f32) = params.bounds.into();
        for v in &mut offspring {
            *v = v.clamp(lo, hi);
        }
        let new_pop = Tensor::<B, 2>::from_data(TensorData::new(offspring, [m, d]), device);

        (new_pop, state.clone())
    }

    /// Merges `population` with the current archive and updates state.
    ///
    /// **First tell** (`state.archive_fitness.is_empty()`): `population` is
    /// the initial archive returned verbatim by the first `ask`; this call
    /// sorts it by fitness and records `best_genome` and `best_fitness`.
    ///
    /// **Steady-state tell**: concatenates `state.archive` (shape `(k, D)`)
    /// with `population` (shape `(m, D)`), sorts the combined `k + m` rows by
    /// fitness, and retains only the top `k`. Updates `best_genome` if the new
    /// archive leader improves on `state.best_fitness`.
    ///
    /// Returns the updated state and a [`StrategyMetrics`] snapshot built from
    /// `fitness` (the offspring scores, not the full archive).
    fn tell(
        &self,
        params: &AcoRConfig,
        population: Tensor<B, 2>,
        fitness: Tensor<B, 1>,
        mut state: AcoRState<B>,
        _rng: &mut dyn Rng,
    ) -> (AcoRState<B>, StrategyMetrics) {
        let fitness_host = fitness
            .into_data()
            .into_vec::<f32>()
            .expect("fitness tensor must be readable as f32");
        let device = population.device();
        let k = params.archive_size;

        // First tell: the population being scored IS the initial archive.
        if state.archive_fitness.is_empty() {
            // Sort archive by fitness, best (highest) first.
            let mut idx: Vec<usize> = (0..fitness_host.len()).collect();
            // Sanitize NaN → −inf (worst) so it can never rank as best; descending.
            let sane: Vec<f32> = fitness_host
                .iter()
                .map(|&f| crate::fitness::sanitize_fitness(f))
                .collect();
            idx.sort_by(|&a, &b| sane[b].total_cmp(&sane[a]));
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
            state.best_fitness = m.best_fitness_ever();
            return (state, m);
        }

        // Steady state: merge archive + offspring, keep top-k.
        let combined = Tensor::cat(vec![state.archive.clone(), population.clone()], 0);
        let mut combined_f: Vec<f32> = state.archive_fitness.clone();
        combined_f.extend_from_slice(&fitness_host);
        let mut idx: Vec<usize> = (0..combined_f.len()).collect();
        // Sanitize NaN → −inf (worst) so it can never rank as best; descending.
        let sane: Vec<f32> = combined_f
            .iter()
            .map(|&f| crate::fitness::sanitize_fitness(f))
            .collect();
        idx.sort_by(|&a, &b| sane[b].total_cmp(&sane[a]));
        idx.truncate(k);
        #[allow(clippy::cast_possible_wrap)]
        let top_idx = Tensor::<B, 1, Int>::from_data(
            TensorData::new(idx.iter().map(|&i| i as i64).collect::<Vec<_>>(), [k]),
            &device,
        );
        state.archive = combined.select(0, top_idx);
        state.archive_fitness = idx.iter().map(|&i| combined_f[i]).collect();

        if state.archive_fitness[0] > state.best_fitness {
            state.best_fitness = state.archive_fitness[0];
            let first_idx =
                Tensor::<B, 1, Int>::from_data(TensorData::new(vec![0_i64], [1]), &device);
            state.best_genome = Some(state.archive.clone().select(0, first_idx));
        }

        state.generation += 1;
        let m =
            StrategyMetrics::from_host_fitness(state.generation, &fitness_host, state.best_fitness);
        state.best_fitness = m.best_fitness_ever();
        (state, m)
    }

    /// Returns the best genome seen across all generations, or `None` before
    /// the first [`tell`](Self::tell) call.
    ///
    /// The returned tensor has shape `(1, genome_dim)`.
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
    use burn::backend::Flex;
    use burn::backend::flex::FlexDevice;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rlevo_core::fitness::FitnessEvaluable;

    #[test]
    fn default_config_validates() {
        assert!(AcoRConfig::default_for(50, 30, 10).validate().is_ok());
    }

    #[test]
    fn rejects_archive_below_two() {
        let mut cfg = AcoRConfig::default_for(50, 30, 10);
        cfg.archive_size = 1;
        assert_eq!(cfg.validate().unwrap_err().field, "archive_size");
    }

    type TestBackend = Flex;

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
    fn weights_normal_case_monotone_non_increasing() {
        let w: Vec<f32> = AntColonyReal::<TestBackend>::compute_weights(10, 0.1);
        let total: f32 = w.iter().sum();
        approx::assert_relative_eq!(total, 1.0, epsilon = 1e-5);
        // Rank 0 (best) must be the heaviest; weights decay with rank.
        for pair in w.windows(2) {
            assert!(
                pair[0] >= pair[1],
                "weights must be non-increasing: {} < {}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn weights_degenerate_q_fall_back_to_uniform() {
        for q in [1e-30_f32, 0.0_f32] {
            let w: Vec<f32> = AntColonyReal::<TestBackend>::compute_weights(10, q);
            assert!(
                w.iter().all(|v| v.is_finite() && *v >= 0.0),
                "degenerate q={q} produced non-finite / negative weights: {w:?}"
            );
            let total: f32 = w.iter().sum();
            approx::assert_relative_eq!(total, 1.0, epsilon = 1e-5);
        }
    }

    // --- NaN-safe sampling coverage through the real `ask` call site ---
    //
    // These exercise the `sampling::normal_or_mean` guard end-to-end: a
    // non-finite value is injected into the archive that `ask` reads, and we
    // assert `ask` neither panics nor returns garbage where the guard should
    // recover. Directly injectable because every `AcoRState` field is `pub`.

    /// Builds a 3-row archive whose weights force the roulette in `ask` to
    /// always select archive row 0 (`weights = [1, 0, 0]` ⇒ CDF `[1, 1, 1]` ⇒
    /// `pick(u) == 0` for every `u ∈ [0, 1)`), so the sampled mean is
    /// deterministically `archive[0, :]`.
    fn state_forcing_row_zero(
        archive_vals: Vec<f32>,
        device: FlexDevice,
    ) -> AcoRState<TestBackend> {
        let archive: Tensor<TestBackend, 2> =
            Tensor::from_data(TensorData::new(archive_vals, [3, 2]), &device);
        AcoRState {
            archive,
            // Non-empty so `ask` takes the sampling path, not the first-call
            // early return.
            archive_fitness: vec![3.0, 0.0, -3.0],
            weights: vec![1.0, 0.0, 0.0],
            best_genome: None,
            best_fitness: f32::NEG_INFINITY,
            generation: 1,
        }
    }

    #[test]
    fn ask_recovers_from_infinite_sigma_via_mean_fallback() {
        // Row 1, column 0 holds +∞. The on-device σ for column 0 becomes
        // Σ_e |archive[e,0] − archive[0,0]| ⊇ |∞ − 1| = ∞, and ∞.max(1e-12) == ∞
        // survives the floor — so `normal_or_mean(mean=1.0, std=∞)` hits the
        // `Err` fallback and returns the finite mean instead of panicking.
        let device: FlexDevice = Default::default();
        let strategy: AntColonyReal<TestBackend> = AntColonyReal::new();
        let params: AcoRConfig = AcoRConfig::default_for(3, 4, 2);
        let state: AcoRState<TestBackend> =
            state_forcing_row_zero(vec![1.0, 2.0, f32::INFINITY, 0.5, -1.0, -2.0], device);

        let mut rng: StdRng = StdRng::seed_from_u64(21);
        // Must not panic.
        let (pop, _next): (Tensor<TestBackend, 2>, AcoRState<TestBackend>) =
            strategy.ask(&params, &state, &mut rng, &device);
        let vals: Vec<f32> = pop
            .into_data()
            .into_vec::<f32>()
            .expect("offspring tensor must be readable as f32");

        // Every offspring is finite; column 0 recovers exactly to the finite
        // mean archive[0,0] = 1.0 (the fallback draw, clamped inside bounds).
        assert_eq!(vals.len(), 4 * 2);
        for (idx, &v) in vals.iter().enumerate() {
            assert!(v.is_finite(), "offspring[{idx}] = {v} is not finite");
        }
        for i in 0..4 {
            approx::assert_relative_eq!(vals[i * 2], 1.0_f32, epsilon = 1e-6);
        }
    }

    #[test]
    fn ask_passes_nan_mean_through_for_downstream_hygiene() {
        // Row 0, column 0 holds NaN and is the always-selected mean. σ for
        // column 0 is NaN, but NaN.max(1e-12) == 1e-12 (the floor launders the
        // NaN σ), so `normal_or_mean(mean=NaN, std=1e-12)` takes the `Ok` path
        // and the NaN *mean* propagates: `ask` intentionally does NOT launder it
        // (that is the ADR-0034 fitness-hygiene chokepoint's job downstream).
        let device: FlexDevice = Default::default();
        let strategy: AntColonyReal<TestBackend> = AntColonyReal::new();
        let params: AcoRConfig = AcoRConfig::default_for(3, 4, 2);
        let state: AcoRState<TestBackend> =
            state_forcing_row_zero(vec![f32::NAN, 2.0, 1.0, 0.5, -1.0, -2.0], device);

        let mut rng: StdRng = StdRng::seed_from_u64(23);
        // Must not panic despite the NaN in the read path.
        let (pop, _next): (Tensor<TestBackend, 2>, AcoRState<TestBackend>) =
            strategy.ask(&params, &state, &mut rng, &device);
        let vals: Vec<f32> = pop
            .into_data()
            .into_vec::<f32>()
            .expect("offspring tensor must be readable as f32");

        assert_eq!(vals.len(), 4 * 2);
        for i in 0..4 {
            // Column 0: NaN mean passes through unchanged.
            assert!(
                vals[i * 2].is_nan(),
                "expected NaN passthrough at column 0, got {}",
                vals[i * 2]
            );
            // Column 1: finite mean/σ still yield a finite draw.
            assert!(
                vals[i * 2 + 1].is_finite(),
                "column 1 offspring should stay finite, got {}",
                vals[i * 2 + 1]
            );
        }
    }

    #[test]
    fn aco_r_converges_on_sphere_d10() {
        let device = Default::default();
        let strategy = AntColonyReal::<TestBackend>::new();
        let params = AcoRConfig::default_for(30, 15, 10);
        let fitness_fn = FromFitnessEvaluable::new(SphereFit, Sphere);
        let mut harness = EvolutionaryHarness::<TestBackend, _, _>::new(
            strategy, params, fitness_fn, 17, device, 400,
        )
        .expect("valid params");
        harness.reset();
        while !harness.step(()).done {}
        let best = harness.latest_metrics().unwrap().best_fitness_ever();
        assert!(best < 1e-3, "ACO_R D10 best={best}");
    }
}
