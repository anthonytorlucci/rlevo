//! Evolutionary Programming (Fogel-style).
//!
//! Classical EP differs from ES in the details:
//!
//! - **No crossover**. Each parent produces exactly one offspring by
//!   Gaussian mutation.
//! - **Self-adaptive σ**. Each individual carries its own σ, updated
//!   by the log-normal rule `σ' = σ · exp(τ · N(0, 1))`. This is the
//!   same mechanism and ordering as the multi-parent ES variants: σ is
//!   perturbed first, and the updated σ' drives that individual's gene
//!   mutation. Survivor σ are inherited, not reset.
//! - **q-tournament survivor selection** on the `(μ + μ)` pool. Each
//!   individual plays `q` random opponents; the μ individuals with the
//!   highest win-counts survive. This diverges from truncation
//!   selection — EP gives weaker individuals a stochastic chance to
//!   survive.
//!
//! # Reference
//!
//! - Fogel (1994), *An introduction to simulated evolutionary
//!   optimization*.

use std::marker::PhantomData;

use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use rand::Rng;
use rand::RngExt;
use rand_distr::{Distribution as _, Normal};

use rlevo_core::bounds::Bounds;
use rlevo_core::config::{self, ConfigError, ConstraintKind, Validate};

use crate::ops::mutation::gaussian_mutation_per_row;
use crate::rng::{SeedPurpose, seed_stream};
use crate::strategy::{Strategy, StrategyMetrics};

/// Static configuration for an [`EvolutionaryProgramming`] run.
#[derive(Debug, Clone)]
pub struct EpConfig {
    /// Parent population size (offspring population is also μ — EP is
    /// strictly `μ + μ`).
    pub mu: usize,
    /// Genome dimensionality.
    pub genome_dim: usize,
    /// Search-space bounds (initialization and clamping).
    pub bounds: Bounds,
    /// Initial σ for every individual.
    pub initial_sigma: f32,
    /// Learning rate for the log-normal σ update. Default is
    /// `1 / sqrt(2 · sqrt(D))`.
    pub tau: f32,
    /// Number of opponents per tournament round (q-tournament).
    pub tournament_q: usize,
}

impl EpConfig {
    /// Default configuration for a given dimensionality.
    ///
    /// Sets `initial_sigma = 1.0`, `tournament_q = 10`, and derives
    /// `tau = 1.0 / sqrt(2.0 · sqrt(D))` — the standard EP learning-rate
    /// recommendation from Fogel (1994). Bounds are `(-5.12, 5.12)`.
    #[must_use]
    pub fn default_for(mu: usize, genome_dim: usize) -> Self {
        #[allow(clippy::cast_precision_loss)]
        let d = genome_dim as f32;
        let tau = 1.0 / (2.0 * d.sqrt()).sqrt();
        Self {
            mu,
            genome_dim,
            bounds: Bounds::new(-5.12, 5.12),
            initial_sigma: 1.0,
            tau,
            tournament_q: 10,
        }
    }
}

impl Validate for EpConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "EpConfig";
        config::at_least(C, "mu", self.mu, 1)?;
        config::nonzero(C, "genome_dim", self.genome_dim)?;
        config::positive(C, "initial_sigma", f64::from(self.initial_sigma))?;
        config::positive(C, "tau", f64::from(self.tau))?;
        config::at_least(C, "tournament_q", self.tournament_q, 1)?;
        if self.tournament_q > 2 * self.mu {
            return Err(ConfigError {
                config: C,
                field: "tournament_q",
                kind: ConstraintKind::Custom("tournament_q must not exceed 2 * mu"),
            });
        }
        Ok(())
    }
}

/// Generation-to-generation state for [`EvolutionaryProgramming`].
///
/// The two-phase ask/tell handshake uses `parent_fitness.is_empty()` as
/// a sentinel: on the very first [`Strategy::ask`] call the initial
/// parents are returned unchanged; on the very first [`Strategy::tell`]
/// call `parent_fitness` is populated and
/// `best_genome`/`best_fitness` are initialized. Subsequent
/// ask/tell cycles produce, evaluate, and select from the `(μ + μ)` pool.
///
/// During `ask`, `sigmas` is temporarily expanded to length `2μ` (parent
/// σ concatenated with offspring σ) so `tell` can apply q-tournament
/// selection over the combined pool without re-deriving σ values. After
/// `tell` completes, `sigmas` is back to length `μ`.
#[derive(Debug, Clone)]
pub struct EpState<B: Backend> {
    /// Current parents, shape `(μ, D)`.
    pub parents: Tensor<B, 2>,
    /// Per-individual step-size σ, shape `(μ,)` between generations and
    /// `(2μ,)` transiently inside an ask/tell cycle (parent σ ‖ offspring σ).
    pub sigmas: Tensor<B, 1>,
    /// Host-side fitness cache for the current parents.
    ///
    /// Empty before the first [`Strategy::tell`] call; length `μ`
    /// thereafter. The `is_empty()` check distinguishes the initial
    /// evaluation phase from subsequent tournament-selection generations.
    pub parent_fitness: Vec<f32>,
    /// Best-so-far genome, shape `(1, D)`.
    ///
    /// `None` before the first [`Strategy::tell`] call.
    pub best_genome: Option<Tensor<B, 2>>,
    /// Best-so-far fitness across all completed generations.
    ///
    /// `f32::NEG_INFINITY` before the first [`Strategy::tell`] call
    /// (the worst value under the maximise convention).
    pub best_fitness: f32,
    /// Number of completed `tell` calls (zero-based generation index + 1).
    pub generation: usize,
}

/// Classical Fogel EP.
///
/// # Example
///
/// ```no_run
/// use burn::backend::Flex;
/// use rlevo_evolution::algorithms::ep::{EpConfig, EvolutionaryProgramming};
///
/// let strategy = EvolutionaryProgramming::<Flex>::new();
/// let params = EpConfig::default_for(30, 10);
/// let _ = (strategy, params);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct EvolutionaryProgramming<B: Backend> {
    _backend: PhantomData<fn() -> B>,
}

impl<B: Backend> EvolutionaryProgramming<B> {
    /// Builds a new (stateless) strategy object.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> Strategy<B> for EvolutionaryProgramming<B>
where
    B::Device: Clone,
{
    type Params = EpConfig;
    type State = EpState<B>;
    type Genome = Tensor<B, 2>;

    /// Samples the initial parent population uniformly within
    /// `params.bounds`, initializes per-parent σ to
    /// `params.initial_sigma`, and returns an [`EpState`] with an empty
    /// fitness cache.
    ///
    /// Initial sampling goes through [`seed_stream`] rather than
    /// `B::seed + Tensor::random` to keep results reproducible across
    /// parallel test threads.
    fn init(&self, params: &EpConfig, rng: &mut dyn Rng, device: &<B as burn::tensor::backend::BackendTypes>::Device) -> EpState<B> {
        debug_assert!(params.validate().is_ok(), "invalid EpConfig reached init: {params:?}");
        let (lo, hi): (f32, f32) = params.bounds.into();
        // Host-sample the initial parents from a deterministic `seed_stream`
        // rather than the process-wide Flex RNG (`B::seed` + `Tensor::random`),
        // whose draws interleave with sibling tests under the parallel runner
        // and are not reproducible across thread schedules.
        let mu = params.mu;
        let genome_dim = params.genome_dim;
        let mut stream = seed_stream(rng.next_u64(), 0, SeedPurpose::Init);
        let mut parent_rows = Vec::with_capacity(mu * genome_dim);
        for _ in 0..mu * genome_dim {
            parent_rows.push(lo + (hi - lo) * stream.random::<f32>());
        }
        let parents =
            Tensor::<B, 2>::from_data(TensorData::new(parent_rows, [mu, genome_dim]), device);
        let sigmas = Tensor::<B, 1>::from_data(
            TensorData::new(vec![params.initial_sigma; params.mu], [params.mu]),
            device,
        );
        EpState {
            parents,
            sigmas,
            parent_fitness: Vec::new(),
            best_genome: None,
            best_fitness: f32::NEG_INFINITY,
            generation: 0,
        }
    }

    /// Proposes the offspring population for this generation.
    ///
    /// **First call (fitness cache empty):** returns the initial parents
    /// unchanged so the caller can evaluate them before any mutation step.
    ///
    /// **Subsequent calls:**
    ///
    /// 1. Applies the log-normal σ update to each parent:
    ///    `σ'_i = σ_i · exp(τ · N(0, 1))`, host-sampled via
    ///    [`seed_stream`] with [`SeedPurpose::Other`].
    /// 2. Mutates each parent by its updated σ using
    ///    [`gaussian_mutation_per_row`], host-sampled via [`seed_stream`]
    ///    with [`SeedPurpose::Mutation`].
    /// 3. Clamps offspring to `params.bounds`.
    /// 4. Appends the offspring σ values to `state.sigmas`, making it
    ///    length `2μ` so [`Strategy::tell`] can select over the combined
    ///    pool without re-deriving them.
    ///
    /// Returns the offspring tensor and the updated state.
    fn ask(
        &self,
        params: &EpConfig,
        state: &EpState<B>,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> (Tensor<B, 2>, EpState<B>) {
        // First call: evaluate the initial parents.
        if state.parent_fitness.is_empty() {
            return (state.parents.clone(), state.clone());
        }

        let mu = params.mu;
        let mut sigma_rng =
            seed_stream(rng.next_u64(), state.generation as u64, SeedPurpose::Other);
        let mut mutation_rng = seed_stream(
            rng.next_u64(),
            state.generation as u64,
            SeedPurpose::Mutation,
        );

        // Log-normal σ update for every parent. Host-sample the N(0,1)
        // noise from the deterministic `sigma_rng` so it is reproducible
        // across thread schedules.
        let normal = Normal::new(0.0f32, 1.0).expect("unit normal is well-defined");
        let mut noise_rows = Vec::with_capacity(mu);
        for _ in 0..mu {
            noise_rows.push(normal.sample(&mut sigma_rng));
        }
        let noise = Tensor::<B, 1>::from_data(TensorData::new(noise_rows, [mu]), device);
        let offspring_sigmas = state.sigmas.clone() * noise.mul_scalar(params.tau).exp();

        // Mutate each parent exactly once using its own σ, drawing from the
        // host `mutation_rng`.
        let offspring = gaussian_mutation_per_row(
            state.parents.clone(),
            offspring_sigmas.clone(),
            &mut mutation_rng,
            device,
        );
        let (lo, hi): (f32, f32) = params.bounds.into();
        let offspring = offspring.clamp(lo, hi);

        // Stash offspring σ onto state via concatenation (parent_σ || offspring_σ).
        let mut state = state.clone();
        state.sigmas = Tensor::cat(vec![state.sigmas.clone(), offspring_sigmas], 0);
        (offspring, state)
    }

    /// Consumes the evaluated offspring and advances the state.
    ///
    /// **First call (fitness cache empty):** stores the initial parent
    /// fitness, initializes `best_genome`/`best_fitness`, resets σ to
    /// `params.initial_sigma`, and increments the generation counter.
    ///
    /// **Subsequent calls:**
    ///
    /// 1. Builds the `(μ + μ)` combined pool of parents and offspring
    ///    (and their `2μ` σ values from [`Strategy::ask`]).
    /// 2. Runs q-tournament selection: each of the `2μ` members plays
    ///    `params.tournament_q` random opponents; the member wins a bout
    ///    if its fitness is strictly higher. The μ members with the most
    ///    wins survive; ties are broken by fitness (higher wins).
    ///    Tournament indices are host-sampled via [`seed_stream`] with
    ///    [`SeedPurpose::Selection`].
    /// 3. Updates `best_genome`/`best_fitness` from the offspring
    ///    fitness if improved.
    ///
    /// Returns the updated [`EpState`] and a [`StrategyMetrics`] snapshot
    /// covering the current offspring generation's fitness distribution.
    fn tell(
        &self,
        params: &EpConfig,
        offspring: Tensor<B, 2>,
        fitness: Tensor<B, 1>,
        mut state: EpState<B>,
        rng: &mut dyn Rng,
    ) -> (EpState<B>, StrategyMetrics) {
        let fitness_host = fitness.into_data().into_vec::<f32>().unwrap_or_default();
        let device = offspring.device();

        // First `tell`: evaluated the initial parents.
        if state.parent_fitness.is_empty() {
            state.parent_fitness.clone_from(&fitness_host);
            state.generation += 1;
            update_best(&mut state, &offspring, &fitness_host);
            let m = StrategyMetrics::from_host_fitness(
                state.generation,
                &fitness_host,
                state.best_fitness,
            );
            state.best_fitness = m.best_fitness_ever();
            state.parents = offspring;
            state.sigmas = Tensor::<B, 1>::from_data(
                TensorData::new(vec![params.initial_sigma; params.mu], [params.mu]),
                &device,
            );
            return (state, m);
        }

        let mu = params.mu;
        // Build the (μ + μ) pool.
        let combined_pop = Tensor::cat(vec![state.parents.clone(), offspring.clone()], 0);
        let combined_fit: Vec<f32> = state
            .parent_fitness
            .iter()
            .chain(fitness_host.iter())
            .copied()
            .collect();
        let combined_sigmas = state.sigmas.clone(); // already (μ + μ) thanks to `ask`.

        // q-tournament: for each of the 2μ members, sample q opponents
        // and count wins (higher fitness beats lower). The μ highest-
        // win members survive.
        let mut selection_rng = seed_stream(
            rng.next_u64(),
            state.generation as u64,
            SeedPurpose::Selection,
        );
        let n = combined_fit.len();
        let mut win_counts: Vec<u32> = vec![0; n];
        for (i, &my_fit) in combined_fit.iter().enumerate() {
            for _ in 0..params.tournament_q {
                let opp = selection_rng.random_range(0..n);
                if my_fit > combined_fit[opp] {
                    win_counts[i] += 1;
                }
            }
        }

        // Sort by (win_count desc, fitness desc) and pick top μ. Sanitize the
        // fitness tiebreak (NaN → −inf, worst) so a NaN can never rank as best.
        let mut indexed: Vec<usize> = (0..n).collect();
        let sane: Vec<f32> = combined_fit
            .iter()
            .map(|&f| crate::fitness::sanitize_fitness(f))
            .collect();
        indexed.sort_by(|&a, &b| {
            win_counts[b]
                .cmp(&win_counts[a])
                .then_with(|| sane[b].total_cmp(&sane[a]))
        });
        indexed.truncate(mu);
        #[allow(clippy::cast_possible_wrap)]
        let survivor_idx: Vec<i64> = indexed.iter().map(|&i| i as i64).collect();

        let idx_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::new(survivor_idx.clone(), [mu]), &device);
        let next_parents = combined_pop.select(0, idx_tensor.clone());
        let next_sigmas = combined_sigmas.select(0, idx_tensor);
        let next_fitness: Vec<f32> = survivor_idx
            .iter()
            .map(|&i| {
                #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                combined_fit[i as usize]
            })
            .collect();

        state.parents = next_parents;
        state.sigmas = next_sigmas;
        state.parent_fitness = next_fitness;
        state.generation += 1;
        update_best(&mut state, &offspring, &fitness_host);
        let m =
            StrategyMetrics::from_host_fitness(state.generation, &fitness_host, state.best_fitness);
        state.best_fitness = m.best_fitness_ever();
        (state, m)
    }

    /// Returns the best-so-far genome and its canonical (maximise) fitness.
    ///
    /// Returns `None` before the first [`Strategy::tell`] call, when
    /// `EpState::best_genome` is still `None`.
    fn best(&self, state: &EpState<B>) -> Option<(Tensor<B, 2>, f32)> {
        state
            .best_genome
            .as_ref()
            .map(|g| (g.clone(), state.best_fitness))
    }
}

fn update_best<B: Backend>(state: &mut EpState<B>, pop: &Tensor<B, 2>, fitness: &[f32]) {
    if fitness.is_empty() {
        return;
    }
    let mut best_idx = 0usize;
    let mut best_f = fitness[0];
    for (i, &f) in fitness.iter().enumerate().skip(1) {
        if f > best_f {
            best_f = f;
            best_idx = i;
        }
    }
    if best_f > state.best_fitness {
        let device = pop.device();
        #[allow(clippy::cast_possible_wrap)]
        let idx =
            Tensor::<B, 1, Int>::from_data(TensorData::new(vec![best_idx as i64], [1]), &device);
        state.best_genome = Some(pop.clone().select(0, idx));
        state.best_fitness = best_f;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::FromFitnessEvaluable;
    use crate::strategy::EvolutionaryHarness;
    use burn::backend::Flex;
    use rlevo_core::fitness::FitnessEvaluable;
    type TestBackend = Flex;

    #[test]
    fn default_config_validates() {
        assert!(EpConfig::default_for(30, 10).validate().is_ok());
    }

    #[test]
    fn rejects_tournament_q_above_two_mu() {
        let mut cfg = EpConfig::default_for(5, 10);
        cfg.tournament_q = 11;
        assert_eq!(cfg.validate().unwrap_err().field, "tournament_q");
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
    fn ep_converges_on_sphere_d2() {
        let device = Default::default();
        let params = EpConfig::default_for(10, 2);
        let fitness_fn = FromFitnessEvaluable::new(SphereFit, Sphere);
        let mut harness = EvolutionaryHarness::<TestBackend, _, _>::new(
            EvolutionaryProgramming::<TestBackend>::new(),
            params,
            fitness_fn,
            3,
            device,
            300,
        ).expect("valid params");
        harness.reset();
        loop {
            if harness.step(()).done {
                break;
            }
        }
        let best = harness.latest_metrics().unwrap().best_fitness_ever();
        assert!(best < 1e-2, "EP Sphere-D2 best={best}");
    }

    #[test]
    fn ep_converges_on_sphere_d10() {
        let device = Default::default();
        let params = EpConfig::default_for(20, 10);
        let fitness_fn = FromFitnessEvaluable::new(SphereFit, Sphere);
        let mut harness = EvolutionaryHarness::<TestBackend, _, _>::new(
            EvolutionaryProgramming::<TestBackend>::new(),
            params,
            fitness_fn,
            5,
            device,
            2000,
        ).expect("valid params");
        harness.reset();
        loop {
            if harness.step(()).done {
                break;
            }
        }
        let best = harness.latest_metrics().unwrap().best_fitness_ever();
        assert!(best < 1e-4, "EP Sphere-D10 best={best}");
    }
}
