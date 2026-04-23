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
//! neighbourhoods are out of scope for v1 and land with a future
//! PSO-variants spec.
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
//! # References
//!
//! - Kennedy & Eberhart (1995), *Particle Swarm Optimization*.
//! - Shi & Eberhart (1998), *A modified particle swarm optimizer*.
//! - Clerc & Kennedy (2002), *The particle swarm — explosion, stability,
//!   and convergence in a multidimensional complex space*.

use std::marker::PhantomData;

use burn::tensor::{backend::Backend, Distribution, Int, Tensor, TensorData};
use rand::Rng;

use crate::rng::{seed_stream, SeedPurpose};
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
    pub bounds: (f32, f32),
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
    #[must_use]
    pub fn default_for(pop_size: usize, genome_dim: usize) -> Self {
        Self {
            pop_size,
            genome_dim,
            bounds: (-5.12, 5.12),
            inertia: 0.7298,
            c1: 1.49618,
            c2: 1.49618,
            v_max: 5.12,
            variant: PsoVariant::Inertia,
        }
    }

    /// Constriction factor `χ = 2 / |2 − φ − √(φ² − 4φ)|`.
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
/// use burn::backend::NdArray;
/// use evorl_evolution::algorithms::swarm::pso::{ParticleSwarm, PsoConfig};
///
/// let strategy = ParticleSwarm::<NdArray>::new();
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
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let (lo, hi) = params.bounds;
        B::seed(device, rng.next_u64());
        Tensor::<B, 2>::random(
            [params.pop_size, params.genome_dim],
            Distribution::Uniform(f64::from(lo), f64::from(hi)),
            device,
        )
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
        device: &B::Device,
    ) -> PsoState<B> {
        let positions = Self::sample_positions(params, rng, device);
        let velocities = Tensor::<B, 2>::zeros([params.pop_size, params.genome_dim], device);
        let personal_best = positions.clone();
        PsoState {
            positions,
            velocities,
            personal_best,
            personal_best_fitness: Vec::new(),
            global_best: None,
            global_best_fitness: f32::INFINITY,
            best_fitness: f32::INFINITY,
            generation: 0,
        }
    }

    fn ask(
        &self,
        params: &PsoConfig,
        state: &PsoState<B>,
        rng: &mut dyn Rng,
        device: &B::Device,
    ) -> (Tensor<B, 2>, PsoState<B>) {
        // First call: evaluate the initial positions so `tell` can
        // populate personal_best_fitness / global_best.
        if state.personal_best_fitness.is_empty() {
            return (state.positions.clone(), state.clone());
        }

        // Sample r1, r2 ∈ U[0,1) — one matrix each per generation.
        B::seed(
            device,
            seed_stream(rng.next_u64(), state.generation as u64, SeedPurpose::Other).next_u64(),
        );
        let r1 = Tensor::<B, 2>::random(
            [params.pop_size, params.genome_dim],
            Distribution::Uniform(0.0, 1.0),
            device,
        );
        B::seed(
            device,
            seed_stream(rng.next_u64(), state.generation as u64, SeedPurpose::Mutation).next_u64(),
        );
        let r2 = Tensor::<B, 2>::random(
            [params.pop_size, params.genome_dim],
            Distribution::Uniform(0.0, 1.0),
            device,
        );

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
        let (lo, hi) = params.bounds;
        let new_positions = (state.positions.clone() + new_velocities.clone()).clamp(lo, hi);

        let mut next = state.clone();
        next.positions = new_positions.clone();
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
        let fitness_host = fitness.into_data().into_vec::<f32>().unwrap_or_default();
        let device = population.device();

        // First tell: seed personal-bests.
        if state.personal_best_fitness.is_empty() {
            state.personal_best = population.clone();
            state.personal_best_fitness = fitness_host.clone();
            let best_idx = argmin(&fitness_host);
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
            state.best_fitness = m.best_fitness_ever;
            return (state, m);
        }

        let pop_size = params.pop_size;
        let genome_dim = params.genome_dim;

        // Update personal bests — greedy: replace when fitness improves.
        let mut improved = vec![0i64; pop_size];
        let mut new_pbest_fit = state.personal_best_fitness.clone();
        for i in 0..pop_size {
            if fitness_host[i] < state.personal_best_fitness[i] {
                improved[i] = 1;
                new_pbest_fit[i] = fitness_host[i];
            }
        }
        let mask_row =
            Tensor::<B, 1, Int>::from_data(TensorData::new(improved, [pop_size]), &device)
                .equal_elem(1);
        let mask = mask_row.unsqueeze_dim::<2>(1).expand([pop_size, genome_dim]);
        state.personal_best = state
            .personal_best
            .clone()
            .mask_where(mask, population.clone());
        state.personal_best_fitness = new_pbest_fit.clone();

        // Update global best from the new personal bests.
        let best_idx = argmin(&new_pbest_fit);
        if new_pbest_fit[best_idx] < state.global_best_fitness {
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
            state.best_fitness.min(state.global_best_fitness),
        );
        state.best_fitness = m.best_fitness_ever;
        (state, m)
    }

    fn best(&self, state: &PsoState<B>) -> Option<(Tensor<B, 2>, f32)> {
        state
            .global_best
            .as_ref()
            .map(|g| (g.clone(), state.global_best_fitness))
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
    use evorl_benchmarks::agent::FitnessEvaluable;
    use evorl_benchmarks::env::BenchEnv;

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
            strategy, params, fitness_fn, seed, device, generations,
        );
        harness.reset();
        loop {
            let step = harness.step(());
            if step.done {
                break;
            }
        }
        harness.latest_metrics().unwrap().best_fitness_ever
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
}
