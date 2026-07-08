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
//!    `rand < A_i` **and** `f(x'_i) ≥ f(x_i)`. On acceptance:
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

use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use rand::Rng;
use rand::RngExt;

use rlevo_core::bounds::Bounds;
use rlevo_core::config::{self, ConfigError, ConstraintKind, Validate};

use super::len_matches_pop;
use crate::ops::selection::argmax_host;
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
    pub bounds: Bounds,
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
            bounds: Bounds::new(-5.12, 5.12),
            f_min: 0.0,
            f_max: 2.0,
            a0: 1.0,
            r0: 0.5,
            alpha: 0.9,
            gamma: 0.9,
        }
    }
}

impl Validate for BatConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "BatConfig";
        config::at_least(C, "pop_size", self.pop_size, 1)?;
        config::nonzero(C, "genome_dim", self.genome_dim)?;
        if self.f_min > self.f_max {
            return Err(ConfigError {
                config: C,
                field: "f_min",
                kind: ConstraintKind::Custom("f_min must not exceed f_max"),
            });
        }
        config::in_range(C, "a0", 0.0, f64::INFINITY, f64::from(self.a0))?;
        config::in_range(C, "r0", 0.0, 1.0, f64::from(self.r0))?;
        // α ∈ (0, 1]: strictly positive and at most one.
        config::positive(C, "alpha", f64::from(self.alpha))?;
        config::in_range(C, "alpha", 0.0, 1.0, f64::from(self.alpha))?;
        config::positive(C, "gamma", f64::from(self.gamma))?;
        Ok(())
    }
}

/// Generation state for [`BatAlgorithm`].
#[derive(Debug, Clone)]
pub struct BatState<B: Backend> {
    /// Current positions, shape `(pop_size, D)`.
    positions: Tensor<B, 2>,
    /// Current velocities, shape `(pop_size, D)`.
    velocities: Tensor<B, 2>,
    /// Per-bat loudness.
    loudness: Vec<f32>,
    /// Per-bat pulse rate.
    pulse_rate: Vec<f32>,
    /// Host-side fitness cache for the current positions.
    fitness: Vec<f32>,
    /// Best-so-far genome.
    best_genome: Option<Tensor<B, 2>>,
    /// Best-so-far fitness.
    best_fitness: f32,
    /// Generation counter.
    generation: usize,
    /// Per-generation "accept this candidate?" decisions recorded in
    /// `ask` so `tell` can gate the loudness/pulse updates consistently
    /// with the RNG draws.
    pending_accept: Vec<bool>,
}

impl<B: Backend> BatState<B> {
    /// Assembles a bat-swarm state, checking the per-bat caches match `pop`.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `positions` has zero rows or if any of
    /// `loudness` / `pulse_rate` / `fitness` / `pending_accept` is non-empty
    /// with a length other than `pop_size`.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        positions: Tensor<B, 2>,
        velocities: Tensor<B, 2>,
        loudness: Vec<f32>,
        pulse_rate: Vec<f32>,
        fitness: Vec<f32>,
        best_genome: Option<Tensor<B, 2>>,
        best_fitness: f32,
        generation: usize,
        pending_accept: Vec<bool>,
    ) -> Result<Self, ConfigError> {
        let pop = positions.dims()[0];
        config::nonzero("BatState", "pop_size", pop)?;
        len_matches_pop("BatState", "loudness", pop, loudness.len())?;
        len_matches_pop("BatState", "pulse_rate", pop, pulse_rate.len())?;
        len_matches_pop("BatState", "fitness", pop, fitness.len())?;
        len_matches_pop("BatState", "pending_accept", pop, pending_accept.len())?;
        Ok(Self {
            positions,
            velocities,
            loudness,
            pulse_rate,
            fitness,
            best_genome,
            best_fitness,
            generation,
            pending_accept,
        })
    }

    /// Current positions, shape `(pop_size, D)`.
    #[must_use]
    pub fn positions(&self) -> &Tensor<B, 2> {
        &self.positions
    }

    /// Current velocities, shape `(pop_size, D)`.
    #[must_use]
    pub fn velocities(&self) -> &Tensor<B, 2> {
        &self.velocities
    }

    /// Per-bat loudness, `pop_size` long.
    #[must_use]
    pub fn loudness(&self) -> &[f32] {
        &self.loudness
    }

    /// Per-bat pulse rate, `pop_size` long.
    #[must_use]
    pub fn pulse_rate(&self) -> &[f32] {
        &self.pulse_rate
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

    /// Per-candidate accept decisions recorded by `ask` (empty at bootstrap,
    /// else `pop_size` long).
    #[must_use]
    pub fn pending_accept(&self) -> &[bool] {
        &self.pending_accept
    }
}

/// Bat Algorithm strategy.
///
/// # Example
///
/// ```no_run
/// use burn::backend::Flex;
/// use rlevo_evolution::algorithms::metaheuristic::bat::{BatAlgorithm, BatConfig};
///
/// let strategy = BatAlgorithm::<Flex>::new();
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

    /// Build the initial colony by host-sampling `pop_size` positions
    /// uniformly in `[bounds.lo, bounds.hi]`.
    ///
    /// Velocities are zeroed, loudness is set to `params.a0`, pulse rate
    /// to `params.r0`, and `fitness` left empty so that the first
    /// [`ask`](Strategy::ask) → [`tell`](Strategy::tell) pair initialises
    /// those fields before any acceptance logic runs.  Positions are drawn
    /// from a deterministic [`seed_stream`]; the process-wide Flex RNG is
    /// never touched.
    fn init(
        &self,
        params: &BatConfig,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> BatState<B> {
        debug_assert!(
            params.validate().is_ok(),
            "invalid BatConfig reached init: {params:?}"
        );
        let (lo, hi): (f32, f32) = params.bounds.into();
        // Sample initial positions on the host from a deterministic
        // `seed_stream`, mirroring `ask`/`tell`. The Flex backend's
        // `Tensor::random` draws from a process-wide RNG mutex; under the
        // parallel test runner those draws interleave with sibling tests,
        // so `B::seed` + `Tensor::random` is NOT reproducible across
        // thread schedules. Host sampling keeps initialisation bit-stable
        // regardless of core count or test ordering.
        let pop = params.pop_size;
        let genome_dim = params.genome_dim;
        let mut stream = seed_stream(rng.next_u64(), 0, SeedPurpose::Init);
        let mut position_rows = Vec::with_capacity(pop * genome_dim);
        for _ in 0..pop * genome_dim {
            position_rows.push(lo + (hi - lo) * stream.random::<f32>());
        }
        let positions =
            Tensor::<B, 2>::from_data(TensorData::new(position_rows, [pop, genome_dim]), device);
        let velocities = Tensor::<B, 2>::zeros([params.pop_size, params.genome_dim], device);
        BatState {
            positions,
            velocities,
            loudness: vec![params.a0; params.pop_size],
            pulse_rate: vec![params.r0; params.pop_size],
            fitness: Vec::new(),
            best_genome: None,
            best_fitness: f32::NEG_INFINITY,
            generation: 0,
            pending_accept: Vec::new(),
        }
    }

    /// Propose candidate positions for the current generation.
    ///
    /// On the first call (`state.fitness` is empty) returns the initial
    /// positions unchanged so the caller can evaluate generation zero.
    ///
    /// On subsequent calls the update proceeds in three host/device steps:
    ///
    /// 1. **Frequency** — sample `f_i = f_min + (f_max − f_min)·β_i`,
    ///    `β_i ∈ U[0,1]`.
    /// 2. **Global move** — `v_i ← v_i + (x_i − x_best)·f_i`,
    ///    `x'_i = x_i + v_i`.
    /// 3. **Local walk** (when `rand > r_i`) — override with
    ///    `x'_i = x_best + ε·mean(A)`, `ε ∈ U[−1,1]`.
    ///
    /// All random draws are host-sampled through [`seed_stream`] for
    /// bit-stable reproduction across thread schedules.  The
    /// per-bat acceptance decisions (`pending_accept`) are recorded in the
    /// returned state and consumed by [`tell`](Strategy::tell).
    ///
    /// # Panics
    ///
    /// Panics if called when `state.best_genome` is `None` after the first
    /// generation has been evaluated (i.e. if `state.fitness` is non-empty
    /// but `state.best_genome` was not set by a preceding `tell` call).
    fn ask(
        &self,
        params: &BatConfig,
        state: &BatState<B>,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> (Tensor<B, 2>, BatState<B>) {
        if state.fitness.is_empty() {
            // Evaluate the initial colony first; the velocity update is
            // only defined once a best exists.
            return (state.positions.clone(), state.clone());
        }

        let pop = params.pop_size;
        let genome_dim = params.genome_dim;
        let (lo, hi): (f32, f32) = params.bounds.into();

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

        // Clamp velocity to the search extent to prevent unbounded ±∞/NaN
        // drift when a bat is pinned against a bound (parity with PSO's v_max).
        let span = (hi - lo).abs();
        let new_velocities = (state.velocities.clone()
            + (state.positions.clone() - best.clone()).mul(f_mat))
        .clamp(-span, span);
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

    /// Ingest candidate fitness values, apply the acceptance gate, and
    /// advance the generation counter.
    ///
    /// On the first call (generation zero bootstrap) all candidates are
    /// unconditionally accepted and loudness/pulse-rate updates are
    /// skipped.
    ///
    /// On subsequent calls candidate `i` replaces position `i` iff
    /// `pending_accept[i]` (drawn in [`ask`](Strategy::ask)) **and**
    /// `fitness[i] ≥ state.fitness[i]`.  On acceptance, loudness decays
    /// (`A_i *= α`) and pulse rate grows
    /// (`r_i = r₀·(1 − exp(−γ·t))`).
    fn tell(
        &self,
        params: &BatConfig,
        candidates: Tensor<B, 2>,
        fitness: Tensor<B, 1>,
        mut state: BatState<B>,
        _rng: &mut dyn Rng,
    ) -> (BatState<B>, StrategyMetrics) {
        let fitness_host = fitness
            .into_data()
            .into_vec::<f32>()
            .expect("fitness tensor must be readable as f32");
        let device = candidates.device();
        let pop = params.pop_size;
        let genome_dim = params.genome_dim;

        if state.fitness.is_empty() {
            state.fitness.clone_from(&fitness_host);
            let best_idx = argmax_host(&fitness_host);
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
            state.best_fitness = m.best_fitness_ever();
            return (state, m);
        }

        // Acceptance: accept candidate `i` iff `pending_accept[i]` AND
        // candidate's fitness is no worse than current (higher is better).
        #[allow(clippy::cast_possible_wrap)]
        let mut rs: Vec<i64> = (0..pop).map(|i| i as i64).collect();
        let mut new_fitness = state.fitness.clone();
        #[allow(clippy::cast_precision_loss)]
        let t = state.generation as f32;
        for i in 0..pop {
            let accept_gate = state.pending_accept.get(i).copied().unwrap_or(false);
            let improves = fitness_host[i] >= state.fitness[i];
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
        let best_idx = argmax_host(&state.fitness);
        if state.fitness[best_idx] > state.best_fitness {
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
        state.best_fitness = m.best_fitness_ever();
        let _ = genome_dim;
        (state, m)
    }

    /// Returns the best-so-far `(genome, fitness)` pair, or `None` before
    /// the first [`tell`](Strategy::tell) call.
    fn best(&self, state: &BatState<B>) -> Option<(Tensor<B, 2>, f32)> {
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
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rlevo_core::fitness::FitnessEvaluable;

    type TestBackend = Flex;

    #[test]
    fn try_new_checks_cache_lengths() {
        let device = Default::default();
        let pos = Tensor::<TestBackend, 2>::zeros([3, 2], &device);
        let vel = Tensor::<TestBackend, 2>::zeros([3, 2], &device);
        assert!(
            BatState::try_new(
                pos.clone(),
                vel.clone(),
                vec![1.0; 3],
                vec![0.5; 3],
                vec![1.0; 3],
                None,
                1.0,
                1,
                vec![false; 3],
            )
            .is_ok()
        );
        // loudness length 2 ≠ pop 3.
        assert!(
            BatState::try_new(
                pos,
                vel,
                vec![1.0; 2],
                vec![0.5; 3],
                vec![1.0; 3],
                None,
                1.0,
                1,
                vec![false; 3],
            )
            .is_err()
        );
    }

    #[test]
    fn default_config_validates() {
        assert!(BatConfig::default_for(30, 10).validate().is_ok());
    }

    #[test]
    fn rejects_inverted_frequency_range() {
        let mut cfg = BatConfig::default_for(30, 10);
        cfg.f_min = 3.0;
        cfg.f_max = 1.0;
        assert_eq!(cfg.validate().unwrap_err().field, "f_min");
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
        )
        .expect("valid params");
        harness.reset();
        while !harness.step(()).done {}
        let best = harness.latest_metrics().unwrap().best_fitness_ever();
        assert!(best < 0.1, "Bat D10 best={best}");
    }

    #[test]
    fn velocities_stay_finite_and_bounded_under_pinning() {
        // Regression for #156 (Bat §1.1). The velocity update
        // `v ← v + (x − x_best)·f` is repulsive for a bat sitting away
        // from `x_best`, and `ask` rewrites `state.velocities` every
        // generation regardless of `tell`'s acceptance gate. A bat whose
        // position stays fixed while `x_best` sits elsewhere therefore
        // accrues a near-constant increment each generation, so unclamped
        // velocities drift linearly to ±∞ and then to NaN (via `inf − inf`).
        // The ±span clamp (parity with PSO's `v_max`) must keep every
        // velocity finite and within the search extent no matter how long
        // the swarm runs.
        let device = Default::default();
        let strategy = BatAlgorithm::<TestBackend>::new();
        let params = BatConfig::default_for(20, 4);
        let (lo, hi): (f32, f32) = params.bounds.into();
        let span = (hi - lo).abs();
        let fitness_fn = FromFitnessEvaluable::new(SphereFit, Sphere);
        let mut harness = EvolutionaryHarness::<TestBackend, _, _>::new(
            strategy, params, fitness_fn, 7, device, 100,
        )
        .expect("valid params");
        harness.reset();
        while !harness.step(()).done {}
        let velocities: Vec<f32> = harness
            .state()
            .expect("state populated after stepping")
            .velocities()
            .clone()
            .into_data()
            .into_vec::<f32>()
            .expect("velocities readable as f32");
        for v in velocities {
            assert!(v.is_finite(), "velocity not finite: {v}");
            assert!(
                v.abs() <= span + 1e-3,
                "velocity {v} exceeds search span {span}"
            );
        }
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

    /// Builds a steady-state (generation ≥ 1) bat swarm at the origin with a
    /// non-empty fitness cache and a populated `best_genome`, so `ask` takes the
    /// velocity-update path rather than the bootstrap early return.
    fn steady_state(
        pop: usize,
        d: usize,
        device: burn::backend::flex::FlexDevice,
    ) -> BatState<TestBackend> {
        let positions = Tensor::<TestBackend, 2>::zeros([pop, d], &device);
        let velocities = Tensor::<TestBackend, 2>::zeros([pop, d], &device);
        let best = Tensor::<TestBackend, 2>::zeros([1, d], &device);
        BatState::try_new(
            positions,
            velocities,
            vec![1.0; pop],
            vec![0.5; pop],
            vec![0.0; pop],
            Some(best),
            0.0,
            1,
            vec![false; pop],
        )
        .expect("valid steady state")
    }

    // Gap (e): the best-so-far accessor is `None` until a `tell` records one.
    #[test]
    fn best_is_none_before_first_tell() {
        let device = Default::default();
        let strategy = BatAlgorithm::<TestBackend>::new();
        let params = BatConfig::default_for(8, 4);
        let mut rng = StdRng::seed_from_u64(1);
        let state = strategy.init(&params, &mut rng, &device);
        assert!(strategy.best(&state).is_none());
    }

    // Gap (a): a lone bat (`pop_size = 1`) and a single-dimension genome are the
    // degenerate extremes. Both must run a full harness loop without panicking.
    #[test]
    fn degenerate_dims_run() {
        for (pop, d) in [(1usize, 4usize), (6, 1)] {
            let device = Default::default();
            let strategy = BatAlgorithm::<TestBackend>::new();
            let params = BatConfig::default_for(pop, d);
            let fitness_fn = FromFitnessEvaluable::new(SphereFit, Sphere);
            let mut harness = EvolutionaryHarness::<TestBackend, _, _>::new(
                strategy, params, fitness_fn, 3, device, 8,
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
                "non-finite best for (pop={pop}, d={d})"
            );
        }
    }

    // Gap (b): every proposed candidate is clamped into `bounds` after `ask`,
    // across 32 seeds.
    #[test]
    fn proposed_positions_within_bounds() {
        let device = Default::default();
        let strategy = BatAlgorithm::<TestBackend>::new();
        let params = BatConfig::default_for(10, 4);
        let (lo, hi): (f32, f32) = params.bounds.into();
        let state = steady_state(10, 4, device);
        for seed in 0..32 {
            let mut rng = StdRng::seed_from_u64(seed);
            let (cand, _next) = strategy.ask(&params, &state, &mut rng, &device);
            let vals = cand
                .into_data()
                .into_vec::<f32>()
                .expect("candidates readable as f32");
            for &v in &vals {
                assert!(
                    v >= lo && v <= hi,
                    "candidate {v} out of bounds [{lo}, {hi}] (seed {seed})"
                );
            }
        }
    }

    // Gap (c): the BA-specific loudness/pulse update math and the acceptance
    // gate. Bee 0 has `pending_accept = true` and an improving candidate, so it
    // accepts: loudness decays by α and pulse rate jumps to
    // `r₀·(1 − exp(−γ·t))`, and its position becomes the candidate. Bee 1 has
    // `pending_accept = false`, so despite an improving candidate it is
    // rejected: loudness, pulse rate, and position are all untouched.
    #[test]
    fn loudness_decay_pulse_growth_and_acceptance_gate() {
        let device = Default::default();
        let strategy = BatAlgorithm::<TestBackend>::new();
        let params = BatConfig::default_for(2, 1); // α = 0.9, γ = 0.9, r₀ = 0.5
        let generation: usize = 3;
        let state = BatState::try_new(
            Tensor::<TestBackend, 2>::zeros([2, 1], &device),
            Tensor::<TestBackend, 2>::zeros([2, 1], &device),
            vec![1.0, 1.0], // loudness = a0
            vec![0.5, 0.5], // pulse = r0
            vec![0.0, 0.0], // current fitness
            Some(Tensor::<TestBackend, 2>::zeros([1, 1], &device)),
            0.0,
            generation,
            vec![true, false], // bee 0 accepts, bee 1 rejects
        )
        .expect("valid state");
        let candidates = Tensor::<TestBackend, 2>::full([2, 1], 0.1, &device);
        // Both candidates improve (1.0 ≥ 0.0), isolating the acceptance gate.
        let fit =
            Tensor::<TestBackend, 1>::from_data(TensorData::new(vec![1.0_f32, 1.0], [2]), &device);
        let mut rng = StdRng::seed_from_u64(0);
        let (next, _m) = strategy.tell(&params, candidates, fit, state, &mut rng);

        // Bee 0 accepted → loudness *= α; bee 1 rejected → unchanged.
        approx::assert_relative_eq!(next.loudness()[0], 0.9, epsilon = 1e-6);
        approx::assert_relative_eq!(next.loudness()[1], 1.0, epsilon = 1e-6);

        // Bee 0 accepted → pulse = r0·(1 − exp(−γ·t)); bee 1 rejected → stays r0.
        #[allow(clippy::cast_precision_loss)]
        let expected_pulse = 0.5 * (1.0 - (-0.9_f32 * generation as f32).exp());
        approx::assert_relative_eq!(next.pulse_rate()[0], expected_pulse, epsilon = 1e-6);
        approx::assert_relative_eq!(next.pulse_rate()[1], 0.5, epsilon = 1e-6);

        // Position: bee 0 takes the candidate (0.1), bee 1 keeps the origin.
        let pos = next
            .positions()
            .clone()
            .into_data()
            .into_vec::<f32>()
            .expect("positions readable as f32");
        approx::assert_relative_eq!(pos[0], 0.1, epsilon = 1e-6);
        approx::assert_relative_eq!(pos[1], 0.0, epsilon = 1e-6);
    }

    // Gap (d): a partly-`NaN` objective is neutralized by the harness sanitize
    // chokepoint (ADR 0034).
    #[test]
    fn nan_fitness_survives_harness() {
        let device = Default::default();
        let strategy = BatAlgorithm::<TestBackend>::new();
        let params = BatConfig::default_for(8, 3);
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
}
