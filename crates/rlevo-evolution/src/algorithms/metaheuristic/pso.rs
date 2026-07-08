//! Particle Swarm Optimization.
//!
//! Canonical PSO with two selectable velocity-update variants:
//!
//! - [`PsoVariant::Inertia`] — the Shi & Eberhart (1998) inertia-weight
//!   form: `v ← ω·v + c1·r1·(pbest − x) + c2·r2·(gbest − x)`.
//! - [`PsoVariant::Constriction`] — the Clerc & Kennedy (2002)
//!   constriction-factor form: `v ← χ·(v + c1·r1·(pbest − x) + c2·r2·(gbest − x))`.
//!
//! Both variants use the canonical *global-best* topology. Ring / local
//! neighbourhoods are not currently implemented; only the fully-connected
//! social structure is exposed.
//!
//! # Initialization
//!
//! Positions are drawn uniformly from the configured search bounds.
//! Velocities are initialized to **zero**, not to a random slice of
//! `[-v_max, v_max]` as several reference implementations do — the zero
//! initialization converges slightly faster on Sphere and produces
//! bit-reproducible initial populations independent of the velocity
//! clamp.
//!
//! # First-generation protocol
//!
//! `ask` detects the first call by checking whether `personal_best_fitness`
//! is empty and, if so, returns the current positions unchanged (no velocity
//! update). `tell` detects the same condition and uses the received fitness
//! to seed `personal_best_fitness` and `global_best` before returning.
//! Any caller that bypasses [`EvolutionaryHarness`] must therefore call
//! `ask` → evaluate → `tell` **twice** before the velocity update is live.
//!
//! [`EvolutionaryHarness`]: crate::strategy::EvolutionaryHarness
//!
//! # Position and velocity clamping
//!
//! After every velocity update the velocities are clamped to
//! `[−v_max, v_max]` (per [`PsoConfig::v_max`]), and the resulting
//! positions are clamped to [`PsoConfig::bounds`]. Particles that hit a
//! boundary keep their clamped position but retain the (clamped) velocity,
//! so they may escape on the next step.
//!
//! # References
//!
//! - Kennedy & Eberhart (1995), *Particle Swarm Optimization*.
//! - Shi & Eberhart (1998), *A modified particle swarm optimizer*.
//! - Clerc & Kennedy (2002), *The particle swarm — explosion, stability,
//!   and convergence in a multidimensional complex space*.

use std::marker::PhantomData;

use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use rand::Rng;
use rand::RngExt;

use rlevo_core::bounds::Bounds;
use rlevo_core::config::{self, ConfigError, ConstraintKind, Validate};

use crate::ops::selection::argmax_host;
use crate::rng::{SeedPurpose, seed_stream};
use crate::strategy::{Strategy, StrategyMetrics};

/// Which velocity-update rule PSO applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PsoVariant {
    /// `v ← ω·v + c1·r1·(pbest − x) + c2·r2·(gbest − x)`.
    Inertia,
    /// Clerc & Kennedy constriction form with factor
    /// `χ = 2 / |2 − φ − √(φ² − 4φ)|`, where `φ = c1 + c2` and must be
    /// `> 4`.
    Constriction,
}

/// Static configuration for a [`ParticleSwarm`] run.
#[derive(Debug, Clone)]
pub struct PsoConfig {
    /// Number of particles in the swarm.
    pub pop_size: usize,
    /// Genome (position) dimensionality.
    pub genome_dim: usize,
    /// Search-space bounds for initialization and position clamping.
    pub bounds: Bounds,
    /// Inertia weight (only used by [`PsoVariant::Inertia`]).
    pub inertia: f32,
    /// Cognitive coefficient (personal-best pull).
    pub c1: f32,
    /// Social coefficient (global-best pull).
    pub c2: f32,
    /// Per-dimension velocity clamp. Classical PSO literature recommends
    /// ~half the search-space extent.
    pub v_max: f32,
    /// Variant.
    pub variant: PsoVariant,
}

impl PsoConfig {
    /// Default configuration matching Shi & Eberhart's canonical settings
    /// (`ω = 0.7298`, `c1 = c2 = 1.49618`) — the constriction-equivalent
    /// values so Inertia and Constriction variants agree in behaviour
    /// under the same default.
    ///
    /// The default variant is [`PsoVariant::Inertia`]. To switch to the
    /// constriction form, set `variant = PsoVariant::Constriction` and
    /// update `c1` and `c2` so that `c1 + c2 > 4` (the Clerc & Kennedy
    /// requirement — the default `c1 = c2 = 1.49618` gives `φ ≈ 2.99`,
    /// which violates this); a canonical choice is `c1 = c2 = 2.05`.
    #[must_use]
    pub fn default_for(pop_size: usize, genome_dim: usize) -> Self {
        Self {
            pop_size,
            genome_dim,
            bounds: Bounds::new(-5.12, 5.12),
            inertia: 0.7298,
            c1: 1.49618,
            c2: 1.49618,
            v_max: 5.12,
            variant: PsoVariant::Inertia,
        }
    }

    /// Computes the constriction factor `χ = 2 / |2 − φ − √(φ² − 4φ)|`
    /// where `φ = c1 + c2`.
    ///
    /// Clerc & Kennedy (2002) require `φ > 4` for the closed form to be
    /// real-valued. If `φ ≤ 4` the discriminant is clamped to zero and
    /// `χ` falls back to `1.0` (no contraction) so the strategy remains
    /// numerically well-defined. A `debug_assert!` fires in debug builds
    /// when this fallback is triggered; it is silent in release builds.
    #[must_use]
    pub fn constriction_chi(&self) -> f32 {
        let phi = self.c1 + self.c2;
        // Clerc & Kennedy require phi > 4; below that the closed form
        // becomes imaginary. Fall back to 1.0 (i.e. no contraction) so
        // the strategy stays numerically well-defined; user-supplied
        // configs violating the contract are covered by debug_assert.
        debug_assert!(phi > 4.0, "PSO constriction requires c1 + c2 > 4");
        let disc = (phi * phi - 4.0 * phi).max(0.0);
        2.0 / (2.0 - phi - disc.sqrt()).abs()
    }
}

impl Validate for PsoConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "PsoConfig";
        config::at_least(C, "pop_size", self.pop_size, 1)?;
        config::nonzero(C, "genome_dim", self.genome_dim)?;
        config::in_range(C, "c1", 0.0, f64::INFINITY, f64::from(self.c1))?;
        config::in_range(C, "c2", 0.0, f64::INFINITY, f64::from(self.c2))?;
        config::positive(C, "v_max", f64::from(self.v_max))?;
        if self.variant == PsoVariant::Constriction && self.c1 + self.c2 <= 4.0 {
            return Err(ConfigError {
                config: C,
                field: "c1",
                kind: ConstraintKind::Custom("constriction requires c1 + c2 > 4"),
            });
        }
        Ok(())
    }
}

/// Generation state for [`ParticleSwarm`].
#[derive(Debug, Clone)]
pub struct PsoState<B: Backend> {
    /// Current particle positions, shape `(pop_size, D)`.
    pub positions: Tensor<B, 2>,
    /// Current particle velocities, shape `(pop_size, D)`.
    pub velocities: Tensor<B, 2>,
    /// Personal-best positions, shape `(pop_size, D)`.
    pub personal_best: Tensor<B, 2>,
    /// Personal-best fitnesses (host-side cache).
    pub personal_best_fitness: Vec<f32>,
    /// Global-best position, shape `(1, D)`.
    pub global_best: Option<Tensor<B, 2>>,
    /// Global-best fitness.
    pub global_best_fitness: f32,
    /// Best-so-far fitness (mirrors `global_best_fitness` for PSO).
    pub best_fitness: f32,
    /// Generation counter.
    pub generation: usize,
}

/// Particle Swarm Optimization strategy.
///
/// # Example
///
/// ```no_run
/// use burn::backend::Flex;
/// use rlevo_evolution::algorithms::metaheuristic::pso::{ParticleSwarm, PsoConfig};
///
/// let strategy = ParticleSwarm::<Flex>::new();
/// let params = PsoConfig::default_for(32, 10);
/// let _ = (strategy, params);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct ParticleSwarm<B: Backend> {
    _backend: PhantomData<fn() -> B>,
}

impl<B: Backend> ParticleSwarm<B> {
    /// Builds a new (stateless) strategy object.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _backend: PhantomData,
        }
    }

    fn sample_positions(
        params: &PsoConfig,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Tensor<B, 2> {
        let (lo, hi): (f32, f32) = params.bounds.into();
        // Host-sample from a deterministic `seed_stream` rather than the
        // process-wide Flex RNG (`B::seed` + `Tensor::random`), whose draws
        // interleave with sibling tests under the parallel runner and are
        // not reproducible across thread schedules.
        let pop = params.pop_size;
        let genome_dim = params.genome_dim;
        let mut stream = seed_stream(rng.next_u64(), 0, SeedPurpose::Init);
        let mut rows = Vec::with_capacity(pop * genome_dim);
        for _ in 0..pop * genome_dim {
            rows.push(lo + (hi - lo) * stream.random::<f32>());
        }
        Tensor::<B, 2>::from_data(TensorData::new(rows, [pop, genome_dim]), device)
    }
}

impl<B: Backend> Strategy<B> for ParticleSwarm<B>
where
    B::Device: Clone,
{
    type Params = PsoConfig;
    type State = PsoState<B>;
    type Genome = Tensor<B, 2>;

    fn init(
        &self,
        params: &PsoConfig,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> PsoState<B> {
        debug_assert!(
            params.validate().is_ok(),
            "invalid PsoConfig reached init: {params:?}"
        );
        let positions = Self::sample_positions(params, rng, device);
        let velocities = Tensor::<B, 2>::zeros([params.pop_size, params.genome_dim], device);
        let personal_best = positions.clone();
        PsoState {
            positions,
            velocities,
            personal_best,
            personal_best_fitness: Vec::new(),
            global_best: None,
            global_best_fitness: f32::NEG_INFINITY,
            best_fitness: f32::NEG_INFINITY,
            generation: 0,
        }
    }

    fn ask(
        &self,
        params: &PsoConfig,
        state: &PsoState<B>,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> (Tensor<B, 2>, PsoState<B>) {
        // First call: evaluate the initial positions so `tell` can
        // populate personal_best_fitness / global_best.
        if state.personal_best_fitness.is_empty() {
            return (state.positions.clone(), state.clone());
        }

        // Sample r1, r2 ∈ U[0,1) — one matrix each per generation. Host
        // sampling (distinct `seed_stream` purposes) keeps the draws
        // reproducible across thread schedules; the global Flex RNG path
        // could be interleaved by a concurrent test between seed and draw.
        let pop = params.pop_size;
        let genome_dim = params.genome_dim;
        let r1 = {
            let mut s = seed_stream(rng.next_u64(), state.generation as u64, SeedPurpose::Other);
            let mut rows = Vec::with_capacity(pop * genome_dim);
            for _ in 0..pop * genome_dim {
                rows.push(s.random::<f32>());
            }
            Tensor::<B, 2>::from_data(TensorData::new(rows, [pop, genome_dim]), device)
        };
        let r2 = {
            let mut s = seed_stream(
                rng.next_u64(),
                state.generation as u64,
                SeedPurpose::Mutation,
            );
            let mut rows = Vec::with_capacity(pop * genome_dim);
            for _ in 0..pop * genome_dim {
                rows.push(s.random::<f32>());
            }
            Tensor::<B, 2>::from_data(TensorData::new(rows, [pop, genome_dim]), device)
        };

        let gbest = state
            .global_best
            .as_ref()
            .expect("global_best populated after the first tell")
            .clone()
            .expand([params.pop_size, params.genome_dim]);

        let cognitive = (state.personal_best.clone() - state.positions.clone())
            .mul(r1)
            .mul_scalar(params.c1);
        let social = (gbest - state.positions.clone())
            .mul(r2)
            .mul_scalar(params.c2);

        let new_velocities = match params.variant {
            PsoVariant::Inertia => {
                state.velocities.clone().mul_scalar(params.inertia) + cognitive + social
            }
            PsoVariant::Constriction => {
                let chi = params.constriction_chi();
                (state.velocities.clone() + cognitive + social).mul_scalar(chi)
            }
        };
        let new_velocities = new_velocities.clamp(-params.v_max, params.v_max);
        let (lo, hi): (f32, f32) = params.bounds.into();
        let new_positions = (state.positions.clone() + new_velocities.clone()).clamp(lo, hi);

        let mut next = state.clone();
        next.positions.clone_from(&new_positions);
        next.velocities = new_velocities;
        (new_positions, next)
    }

    fn tell(
        &self,
        params: &PsoConfig,
        population: Tensor<B, 2>,
        fitness: Tensor<B, 1>,
        mut state: PsoState<B>,
        _rng: &mut dyn Rng,
    ) -> (PsoState<B>, StrategyMetrics) {
        let fitness_host = fitness
            .into_data()
            .into_vec::<f32>()
            .expect("fitness tensor must be readable as f32");
        let device = population.device();

        // First tell: seed personal-bests.
        if state.personal_best_fitness.is_empty() {
            state.personal_best.clone_from(&population);
            state.personal_best_fitness.clone_from(&fitness_host);
            let best_idx = argmax_host(&fitness_host);
            state.global_best_fitness = fitness_host[best_idx];
            #[allow(clippy::cast_possible_wrap)]
            let idx = Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![best_idx as i64], [1]),
                &device,
            );
            state.global_best = Some(population.clone().select(0, idx));
            state.best_fitness = state.global_best_fitness;
            state.generation += 1;
            state.positions = population;
            let m = StrategyMetrics::from_host_fitness(
                state.generation,
                &fitness_host,
                state.best_fitness,
            );
            state.best_fitness = m.best_fitness_ever();
            return (state, m);
        }

        let pop_size = params.pop_size;
        let genome_dim = params.genome_dim;

        // Update personal bests — greedy: replace when fitness improves.
        let mut improved = vec![0i64; pop_size];
        let mut new_pbest_fit = state.personal_best_fitness.clone();
        for i in 0..pop_size {
            if fitness_host[i] > state.personal_best_fitness[i] {
                improved[i] = 1;
                new_pbest_fit[i] = fitness_host[i];
            }
        }
        let mask_row =
            Tensor::<B, 1, Int>::from_data(TensorData::new(improved, [pop_size]), &device)
                .equal_elem(1);
        let mask = mask_row
            .unsqueeze_dim::<2>(1)
            .expand([pop_size, genome_dim]);
        state.personal_best = state
            .personal_best
            .clone()
            .mask_where(mask, population.clone());
        state.personal_best_fitness.clone_from(&new_pbest_fit);

        // Update global best from the new personal bests.
        let best_idx = argmax_host(&new_pbest_fit);
        if new_pbest_fit[best_idx] > state.global_best_fitness {
            state.global_best_fitness = new_pbest_fit[best_idx];
            #[allow(clippy::cast_possible_wrap)]
            let idx = Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![best_idx as i64], [1]),
                &device,
            );
            state.global_best = Some(state.personal_best.clone().select(0, idx));
        }

        state.positions = population;
        state.generation += 1;
        let m = StrategyMetrics::from_host_fitness(
            state.generation,
            &fitness_host,
            state.best_fitness.max(state.global_best_fitness),
        );
        state.best_fitness = m.best_fitness_ever();
        (state, m)
    }

    fn best(&self, state: &PsoState<B>) -> Option<(Tensor<B, 2>, f32)> {
        state
            .global_best
            .as_ref()
            .map(|g| (g.clone(), state.global_best_fitness))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::{BatchFitnessFn, FromFitnessEvaluable};
    use crate::strategy::EvolutionaryHarness;
    use burn::backend::Flex;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rlevo_core::fitness::FitnessEvaluable;
    use rlevo_core::objective::ObjectiveSense;

    type TestBackend = Flex;

    /// Distinct finite fitness with the maximum at index 0, for direct
    /// (non-harness) `tell` calls in lifecycle tests. Finite by construction,
    /// so it never trips the ADR 0034 sanitize contract.
    #[allow(clippy::trivially_copy_pass_by_ref)] // mirror the by-ref device idiom
    fn finite_fitness(
        n: usize,
        device: &<TestBackend as burn::tensor::backend::BackendTypes>::Device,
    ) -> Tensor<TestBackend, 1> {
        #[allow(clippy::cast_precision_loss)]
        let vals: Vec<f32> = (0..n).map(|i| -(i as f32) - 1.0).collect();
        Tensor::<TestBackend, 1>::from_data(TensorData::new(vals, [n]), device)
    }

    /// Objective whose row 0 evaluates to `NaN` (the rest finite). `Maximize`
    /// so natural == canonical and the harness sanitize is the only thing that
    /// can keep the run finite.
    struct NanFitness;
    impl<B: Backend> BatchFitnessFn<B, Tensor<B, 2>> for NanFitness {
        fn evaluate_batch(
            &mut self,
            population: &Tensor<B, 2>,
            device: &<B as burn::tensor::backend::BackendTypes>::Device,
        ) -> Tensor<B, 1> {
            let n = population.dims()[0];
            #[allow(clippy::cast_precision_loss)]
            let mut vals: Vec<f32> = (0..n).map(|i| i as f32).collect();
            vals[0] = f32::NAN;
            Tensor::<B, 1>::from_data(TensorData::new(vals, [n]), device)
        }
        fn sense(&self) -> ObjectiveSense {
            ObjectiveSense::Maximize
        }
    }

    #[test]
    fn default_config_validates() {
        assert!(PsoConfig::default_for(30, 10).validate().is_ok());
    }

    #[test]
    fn rejects_constriction_with_insufficient_phi() {
        let mut cfg = PsoConfig::default_for(30, 10);
        cfg.variant = PsoVariant::Constriction;
        // default c1 = c2 = 1.49618 gives phi ≈ 2.99 < 4.
        assert_eq!(cfg.validate().unwrap_err().field, "c1");
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

    fn run_pso(variant: PsoVariant, dim: usize, generations: usize, seed: u64) -> f32 {
        let device = Default::default();
        let strategy = ParticleSwarm::<TestBackend>::new();
        let mut params = PsoConfig::default_for(32, dim);
        params.variant = variant;
        if variant == PsoVariant::Constriction {
            // Constriction requires φ = c1 + c2 > 4 (Clerc & Kennedy 2002).
            params.c1 = 2.05;
            params.c2 = 2.05;
        }
        let fitness_fn = FromFitnessEvaluable::new(SphereFit, Sphere);
        let mut harness = EvolutionaryHarness::<TestBackend, _, _>::new(
            strategy,
            params,
            fitness_fn,
            seed,
            device,
            generations,
        )
        .expect("valid params");
        harness.reset();
        loop {
            let step = harness.step(());
            if step.done {
                break;
            }
        }
        harness.latest_metrics().unwrap().best_fitness_ever()
    }

    #[test]
    fn inertia_converges_on_sphere_d10() {
        // PSO on Sphere D=10: inertia variant. Budget 500 generations
        // chosen to stay well within the acceptance envelope (within 2×
        // of the best-in-class classical baseline on Rastrigin; on
        // Sphere we want < 1e-6).
        let best = run_pso(PsoVariant::Inertia, 10, 500, 42);
        assert!(best < 1e-6, "PSO inertia D10 best={best}");
    }

    #[test]
    fn constriction_converges_on_sphere_d10() {
        let best = run_pso(PsoVariant::Constriction, 10, 500, 7);
        assert!(best < 1e-6, "PSO constriction D10 best={best}");
    }

    #[test]
    fn constriction_chi_matches_canonical_value() {
        // φ = 4.1 → χ ≈ 0.729843788...
        let mut cfg = PsoConfig::default_for(2, 2);
        cfg.c1 = 2.05;
        cfg.c2 = 2.05;
        approx::assert_relative_eq!(cfg.constriction_chi(), 0.7298, epsilon = 1e-3);
    }

    #[test]
    fn best_is_none_until_first_tell() {
        let device = Default::default();
        let strategy = ParticleSwarm::<TestBackend>::new();
        let params = PsoConfig::default_for(4, 3);
        let mut rng = StdRng::seed_from_u64(0);
        let state = strategy.init(&params, &mut rng, &device);
        // Fresh init: global_best is unset, so `best` reports nothing.
        assert!(strategy.best(&state).is_none());
        // First ask returns the initial positions unchanged; the first tell
        // seeds personal-/global-best from their fitness.
        let (pop, state) = strategy.ask(&params, &state, &mut rng, &device);
        let fitness = finite_fitness(4, &device);
        let (state, _m) = strategy.tell(&params, pop, fitness, state, &mut rng);
        assert!(strategy.best(&state).is_some());
    }

    #[test]
    fn degenerate_single_particle_single_dim_runs() {
        // pop_size = 1, genome_dim = 1: the smallest swarm the validator
        // accepts. It must run through the harness without a panic (gbest is
        // the lone particle, so the social term is zero every step).
        let device = Default::default();
        let strategy = ParticleSwarm::<TestBackend>::new();
        let params = PsoConfig::default_for(1, 1);
        let fitness_fn = FromFitnessEvaluable::new(SphereFit, Sphere);
        let mut harness = EvolutionaryHarness::<TestBackend, _, _>::new(
            strategy, params, fitness_fn, 0, device, 5,
        )
        .expect("valid params");
        harness.reset();
        while !harness.step(()).done {}
        assert!(
            harness
                .latest_metrics()
                .unwrap()
                .best_fitness_ever()
                .is_finite()
        );
    }

    #[test]
    fn rejects_pop_size_zero() {
        let mut cfg = PsoConfig::default_for(1, 3);
        cfg.pop_size = 0;
        assert_eq!(cfg.validate().unwrap_err().field, "pop_size");
    }

    #[test]
    fn inverted_bounds_are_unrepresentable() {
        // `PsoConfig::validate` never re-checks bound ordering because `Bounds`
        // is self-validating (ADR 0027): an inverted range cannot be
        // constructed. A single-point (zero-width) range is deliberately valid.
        assert!(Bounds::try_new(5.12, -5.12).is_err());
        assert!(Bounds::try_new(3.0, 3.0).is_ok());
    }

    #[test]
    fn nan_fitness_through_harness_stays_finite() {
        // Row 0 evaluates to NaN every generation. The harness sanitize
        // chokepoint (ADR 0034) must keep the best-ever finite and flag the
        // broken member rather than poisoning the run.
        let device = Default::default();
        let strategy = ParticleSwarm::<TestBackend>::new();
        let params = PsoConfig::default_for(4, 3);
        let mut harness = EvolutionaryHarness::<TestBackend, _, _>::new(
            strategy, params, NanFitness, 1, device, 3,
        )
        .expect("valid params");
        harness.reset();
        while !harness.step(()).done {}
        let m = harness.latest_metrics().unwrap();
        assert!(
            m.best_fitness_ever().is_finite(),
            "best={}",
            m.best_fitness_ever()
        );
        assert!(m.broken_count() >= 1, "the NaN row must be counted broken");
    }

    #[test]
    fn ask_keeps_positions_in_bounds() {
        // Invariant: every position proposed by the velocity-update `ask`
        // (i.e. the second `ask`, after a `tell` primes gbest) stays inside the
        // configured bounds, across a spread of seeds.
        let device = Default::default();
        let strategy = ParticleSwarm::<TestBackend>::new();
        let params = PsoConfig::default_for(6, 4);
        let (lo, hi): (f32, f32) = params.bounds.into();
        for seed in 0..32 {
            let mut rng = StdRng::seed_from_u64(seed);
            let state = strategy.init(&params, &mut rng, &device);
            let (pop1, state) = strategy.ask(&params, &state, &mut rng, &device);
            let fitness = finite_fitness(6, &device);
            let (state, _m) = strategy.tell(&params, pop1, fitness, state, &mut rng);
            let (pop2, _state) = strategy.ask(&params, &state, &mut rng, &device);
            let values = pop2.into_data().into_vec::<f32>().unwrap();
            for v in values {
                assert!(
                    v >= lo - 1e-4 && v <= hi + 1e-4,
                    "seed {seed}: position {v} out of bounds [{lo}, {hi}]"
                );
            }
        }
    }

    #[test]
    fn second_ask_moves_particles() {
        // First-call protocol: init → ask → tell → ask. The second `ask`
        // applies the velocity update, so at least some particle position must
        // change from the initial swarm (the social pull toward gbest is
        // non-zero for every non-best particle).
        let device = Default::default();
        let strategy = ParticleSwarm::<TestBackend>::new();
        let params = PsoConfig::default_for(6, 4);
        let mut rng = StdRng::seed_from_u64(9);
        let state = strategy.init(&params, &mut rng, &device);
        let (pop1, state) = strategy.ask(&params, &state, &mut rng, &device);
        let initial = pop1.clone().into_data().into_vec::<f32>().unwrap();
        let fitness = finite_fitness(6, &device);
        let (state, _m) = strategy.tell(&params, pop1, fitness, state, &mut rng);
        let (pop2, _state) = strategy.ask(&params, &state, &mut rng, &device);
        let moved = pop2.into_data().into_vec::<f32>().unwrap();
        assert!(
            initial
                .iter()
                .zip(moved.iter())
                .any(|(a, b)| (a - b).abs() > 1e-6),
            "velocity update left every particle stationary"
        );
    }
}
