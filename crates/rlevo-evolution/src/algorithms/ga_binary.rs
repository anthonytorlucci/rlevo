//! Binary-coded Genetic Algorithm.
//!
//! A canonical GA over `Tensor<B, 2, Int>` populations where every gene is
//! restricted to `{0, 1}`:
//!
//! 1. Evaluate the current population (done externally by the harness).
//! 2. Select two independent sets of parents via
//!    [`crate::ops::selection::tournament_indices_host`] (k-tournament).
//! 3. Recombine via [`crate::ops::crossover::binary_uniform_crossover`]
//!    (per-gene coin flip with probability `crossover_p`).
//! 4. Mutate via [`crate::ops::mutation::bit_flip_mutation`]
//!    (per-gene flip with probability `mutation_rate`).
//! 5. Replace via fixed elitist policy: the `elitism_k` best parents
//!    survive; the remaining `pop_size − elitism_k` slots are filled by
//!    the best offspring.
//!
//! Unlike [`crate::algorithms::ga::GeneticAlgorithm`], there is no
//! enum-selectable replacement policy — only elitist replacement is
//! supported. Extend [`BinaryGaConfig`] and the `tell` impl if a
//! generational variant is needed.
//!
//! All random draws go through [`crate::rng::seed_stream`] — never
//! `B::seed` + `Tensor::random` — so per-run results are reproducible
//! across thread schedules.
//!
//! # Fitness convention
//!
//! Fitness is canonical (higher is better), matching all other strategies in
//! this crate. A maximisation benchmark like `OneMax` (`count_ones(genome)`)
//! plugs in directly; a cost objective is reconciled into canonical space by
//! the harness/adapter chokepoint rather than hand-negated here.
//!
//! # References
//!
//! - Holland (1975), *Adaptation in Natural and Artificial Systems*.
//! - Goldberg (1989), *Genetic Algorithms in Search, Optimization, and
//!   Machine Learning*.

use std::marker::PhantomData;

use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use rand::Rng;
use rand::RngExt;

use crate::ops::crossover::binary_uniform_crossover;
use crate::ops::mutation::bit_flip_mutation;
use crate::ops::selection::{tournament_indices_host, truncation_indices_host};
use crate::rng::{SeedPurpose, seed_stream};
use crate::strategy::{Strategy, StrategyMetrics};
use rlevo_core::config::{self, ConfigError, ConstraintKind, Validate};
use rlevo_core::probability::Probability;

/// Static configuration for a [`BinaryGeneticAlgorithm`] run.
#[derive(Debug, Clone)]
pub struct BinaryGaConfig {
    /// Population size.
    pub pop_size: usize,
    /// Genome dimensionality.
    pub genome_dim: usize,
    /// Probability of bit flip per gene. Valid by construction (`[0, 1]`).
    pub mutation_rate: Probability,
    /// Probability of taking parent A's bit in uniform crossover. Valid by
    /// construction (`[0, 1]`).
    pub crossover_p: Probability,
    /// Tournament size for parent selection.
    pub tournament_size: usize,
    /// Number of elites carried over to the next generation.
    pub elitism_k: usize,
}

impl BinaryGaConfig {
    /// Sensible defaults for small-scale binary optimization.
    ///
    /// Mutation rate defaults to `1 / D` (the standard "one expected
    /// flip per genome" rule from the binary-GA literature).
    ///
    /// # Panics
    ///
    /// Panics when `genome_dim == 0`: the `1 / D` mutation rate becomes
    /// non-finite (`1.0 / 0.0 == +∞`), which [`Probability::new`] rejects as
    /// outside `[0, 1]`. This is the sanctioned builder-time panic on an
    /// unusable config (ADR 0026); pass a non-zero `genome_dim`.
    #[must_use]
    pub fn default_for(pop_size: usize, genome_dim: usize) -> Self {
        Self {
            pop_size,
            genome_dim,
            #[allow(clippy::cast_precision_loss)]
            mutation_rate: Probability::new(1.0 / genome_dim as f32),
            crossover_p: Probability::new(0.5),
            tournament_size: 2,
            elitism_k: 1,
        }
    }
}

impl Validate for BinaryGaConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "BinaryGaConfig";
        config::at_least(C, "pop_size", self.pop_size, 1)?;
        config::nonzero(C, "genome_dim", self.genome_dim)?;
        // `mutation_rate` / `crossover_p` are `Probability`: valid by
        // construction (`[0, 1]`, NaN/Inf rejected), so no `in_range` check
        // here — see ADR 0031.
        config::at_least(C, "tournament_size", self.tournament_size, 1)?;
        if self.tournament_size > self.pop_size {
            return Err(ConfigError {
                config: C,
                field: "tournament_size",
                kind: ConstraintKind::Custom("tournament_size must not exceed pop_size"),
            });
        }
        if self.elitism_k > self.pop_size {
            return Err(ConfigError {
                config: C,
                field: "elitism_k",
                kind: ConstraintKind::Custom("elitism_k must not exceed pop_size"),
            });
        }
        Ok(())
    }
}

/// State for [`BinaryGeneticAlgorithm`].
#[derive(Debug, Clone)]
pub struct BinaryGaState<B: Backend> {
    /// Current population, shape `(pop_size, D)`.
    pub population: Tensor<B, 2, Int>,
    /// Host-side fitness cache for the current population. Empty on
    /// init until the first `tell` call populates it.
    pub fitness: Vec<f32>,
    /// Best-so-far genome.
    pub best_genome: Option<Tensor<B, 2, Int>>,
    /// Best-so-far fitness.
    pub best_fitness: f32,
    /// Generation counter.
    pub generation: usize,
}

/// Binary-coded canonical Genetic Algorithm.
///
/// # Example
///
/// ```no_run
/// use burn::backend::Flex;
/// use rlevo_evolution::algorithms::ga_binary::{BinaryGaConfig, BinaryGeneticAlgorithm};
///
/// let strategy = BinaryGeneticAlgorithm::<Flex>::new();
/// let params = BinaryGaConfig::default_for(32, 16);
/// let _ = (strategy, params);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct BinaryGeneticAlgorithm<B: Backend> {
    _backend: PhantomData<fn() -> B>,
}

impl<B: Backend> BinaryGeneticAlgorithm<B> {
    /// Builds a new (stateless) strategy object.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _backend: PhantomData,
        }
    }

    fn sample_initial_population(
        params: &BinaryGaConfig,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Tensor<B, 2, Int> {
        // Host-sample U[0,1) from a deterministic `seed_stream` rather than
        // the process-wide Flex RNG (`B::seed` + `Tensor::random`), whose
        // draws interleave with sibling tests under the parallel runner and
        // are not reproducible across thread schedules.
        let pop = params.pop_size;
        let genome_dim = params.genome_dim;
        let mut stream = seed_stream(rng.next_u64(), 0, SeedPurpose::Init);
        let mut rows = Vec::with_capacity(pop * genome_dim);
        for _ in 0..pop * genome_dim {
            rows.push(stream.random::<f32>());
        }
        let u = Tensor::<B, 2>::from_data(TensorData::new(rows, [pop, genome_dim]), device);
        u.lower_elem(0.5).int()
    }
}

impl<B: Backend> Strategy<B> for BinaryGeneticAlgorithm<B>
where
    B::Device: Clone,
{
    type Params = BinaryGaConfig;
    type State = BinaryGaState<B>;
    type Genome = Tensor<B, 2, Int>;

    /// Build the initial state.
    ///
    /// Samples an `(pop_size, D)` binary population uniformly at random
    /// (each gene independently `Bernoulli(0.5)`) using a host RNG derived
    /// from `rng`. Sets `fitness` to empty and `best_fitness` to
    /// `f32::NEG_INFINITY` (the worst value under the maximise convention);
    /// the first [`tell`](Self::tell) call populates both.
    fn init(
        &self,
        params: &BinaryGaConfig,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> BinaryGaState<B> {
        debug_assert!(
            params.validate().is_ok(),
            "invalid BinaryGaConfig reached init: {params:?}"
        );
        BinaryGaState {
            population: Self::sample_initial_population(params, rng, device),
            fitness: Vec::new(),
            best_genome: None,
            best_fitness: f32::NEG_INFINITY,
            generation: 0,
        }
    }

    /// Propose the next offspring population.
    ///
    /// On the very first call (before any [`tell`](Self::tell)), `state.fitness`
    /// is empty — the harness has not evaluated the seed population yet. In
    /// that case the unchanged seed population is returned so the harness can
    /// evaluate and pass it back to `tell`.
    ///
    /// On subsequent calls the method runs one full selection → crossover →
    /// mutation pipeline, deriving three independent host sub-streams from
    /// `rng` via [`crate::rng::seed_stream`]:
    ///
    /// - `SeedPurpose::Selection` — two independent tournament draws
    ///   (parents A and parents B);
    /// - `SeedPurpose::Crossover` — per-gene coin flip
    ///   (probability `crossover_p`);
    /// - `SeedPurpose::Mutation` — per-gene bit-flip
    ///   (probability `mutation_rate`).
    ///
    /// # Panics
    ///
    /// Propagates the programming-error panics of the delegated host operators
    /// once `state.fitness` is non-empty:
    /// [`tournament_indices_host`]
    /// panics if the fitness slice is empty or `tournament_size == 0`, and the
    /// crossover / mutation operators assume shape-consistent inputs.
    /// `validate()` rules
    /// these out for a well-formed config, but `BinaryGaState` / `BinaryGaConfig`
    /// fields are public and can be mutated into an invalid state after
    /// validation.
    fn ask(
        &self,
        params: &BinaryGaConfig,
        state: &BinaryGaState<B>,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> (Tensor<B, 2, Int>, BinaryGaState<B>) {
        if state.fitness.is_empty() {
            return (state.population.clone(), state.clone());
        }

        let mut selection_rng = seed_stream(
            rng.next_u64(),
            state.generation as u64,
            SeedPurpose::Selection,
        );
        let mut crossover_rng = seed_stream(
            rng.next_u64(),
            state.generation as u64,
            SeedPurpose::Crossover,
        );
        let mut mutation_rng = seed_stream(
            rng.next_u64(),
            state.generation as u64,
            SeedPurpose::Mutation,
        );

        let idx_a = tournament_indices_host(
            &state.fitness,
            params.tournament_size,
            params.pop_size,
            &mut selection_rng,
        );
        let idx_b = tournament_indices_host(
            &state.fitness,
            params.tournament_size,
            params.pop_size,
            &mut selection_rng,
        );
        let parents_a = state.population.clone().select(
            0,
            Tensor::<B, 1, Int>::from_data(TensorData::new(idx_a, [params.pop_size]), device),
        );
        let parents_b = state.population.clone().select(
            0,
            Tensor::<B, 1, Int>::from_data(TensorData::new(idx_b, [params.pop_size]), device),
        );

        let offspring = binary_uniform_crossover(
            parents_a,
            parents_b,
            params.crossover_p,
            &mut crossover_rng,
            device,
        );

        let offspring =
            bit_flip_mutation(offspring, params.mutation_rate, &mut mutation_rng, device);

        (offspring, state.clone())
    }

    /// Consume offspring fitness and produce the next generation's state.
    ///
    /// The first call (when `state.fitness` is empty) caches the seed
    /// population's fitness and increments the generation counter; no
    /// replacement is performed.
    ///
    /// On subsequent calls the method performs elitist replacement: the
    /// `elitism_k` highest-fitness parents survive directly, and the remaining
    /// `pop_size − elitism_k` slots are filled with the best offspring.
    /// Both selections use [`crate::ops::selection::truncation_indices_host`].
    ///
    /// `fitness` must have shape `(pop_size,)` in the canonical maximise
    /// convention — higher is better. The harness canonicalises a `Minimize`
    /// objective (negation) before this call, so the strategy stays
    /// sense-unaware per ADR 0023.
    ///
    /// # Panics
    ///
    /// Propagates the programming-error panics of the delegated host path on the
    /// elitist-replacement branch:
    /// [`truncation_indices_host`]
    /// and the bare `state.fitness[i]` indexing assume a `pop_size`-consistent
    /// fitness buffer. `validate()` rules this out for a well-formed config, but
    /// `BinaryGaState` / `BinaryGaConfig` fields are public and can be mutated
    /// into an inconsistent state after validation. Also panics if `fitness`
    /// cannot be read back as `f32`.
    fn tell(
        &self,
        params: &BinaryGaConfig,
        offspring: Tensor<B, 2, Int>,
        fitness: Tensor<B, 1>,
        mut state: BinaryGaState<B>,
        _rng: &mut dyn Rng,
    ) -> (BinaryGaState<B>, StrategyMetrics) {
        let fitness_host = fitness
            .into_data()
            .into_vec::<f32>()
            .expect("fitness tensor must be readable as f32");
        let device = offspring.device();

        // First `tell`: initial population just evaluated.
        if state.fitness.is_empty() {
            state.fitness.clone_from(&fitness_host);
            state.generation += 1;
            update_best(&mut state, &offspring, &fitness_host);
            let m = StrategyMetrics::from_host_fitness(
                state.generation,
                &fitness_host,
                state.best_fitness,
            );
            state.best_fitness = m.best_fitness_ever();
            state.population = offspring;
            return (state, m);
        }

        // Elitist replacement on (pop, fitness) × (offspring, fitness).
        let pop_size = params.pop_size;
        let k = params.elitism_k.min(pop_size);

        let elite_idx = truncation_indices_host(&state.fitness, k);
        let elites = state.population.clone().select(
            0,
            Tensor::<B, 1, Int>::from_data(TensorData::new(elite_idx.clone(), [k]), &device),
        );
        let n_off_keep = pop_size - k;
        let off_keep_idx = truncation_indices_host(&fitness_host, n_off_keep);
        let kept_off = offspring.clone().select(
            0,
            Tensor::<B, 1, Int>::from_data(
                TensorData::new(off_keep_idx.clone(), [n_off_keep]),
                &device,
            ),
        );
        let next_pop = Tensor::cat(vec![elites, kept_off], 0);
        let mut next_fit = Vec::with_capacity(pop_size);
        for i in elite_idx {
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            next_fit.push(state.fitness[i as usize]);
        }
        for i in off_keep_idx {
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            next_fit.push(fitness_host[i as usize]);
        }

        update_best(&mut state, &next_pop, &next_fit);
        state.population = next_pop;
        state.fitness.clone_from(&next_fit);
        state.generation += 1;
        let m = StrategyMetrics::from_host_fitness(state.generation, &next_fit, state.best_fitness);
        state.best_fitness = m.best_fitness_ever();
        (state, m)
    }

    /// Return the best-so-far genome and its fitness.
    ///
    /// Returns `None` before the first [`tell`](Self::tell) call.
    /// The fitness value uses the canonical maximise convention (higher is better).
    fn best(&self, state: &BinaryGaState<B>) -> Option<(Tensor<B, 2, Int>, f32)> {
        state
            .best_genome
            .as_ref()
            .map(|g| (g.clone(), state.best_fitness))
    }
}

fn update_best<B: Backend>(state: &mut BinaryGaState<B>, pop: &Tensor<B, 2, Int>, fitness: &[f32]) {
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
    use crate::fitness::BatchFitnessFn;
    use crate::strategy::EvolutionaryHarness;
    use burn::backend::Flex;
    type TestBackend = Flex;

    #[test]
    fn default_config_validates() {
        assert!(BinaryGaConfig::default_for(32, 16).validate().is_ok());
    }

    #[test]
    fn rejects_elitism_larger_than_pop() {
        let mut cfg = BinaryGaConfig::default_for(8, 16);
        cfg.elitism_k = 16;
        assert_eq!(cfg.validate().unwrap_err().field, "elitism_k");
    }

    /// `OneMax` as a native maximisation: `fitness = count_ones`, optimum at
    /// `D` (all ones).
    struct OneMax {
        dim: usize,
    }

    impl<B: Backend> BatchFitnessFn<B, Tensor<B, 2, Int>> for OneMax {
        fn evaluate_batch(
            &mut self,
            population: &Tensor<B, 2, Int>,
            device: &<B as burn::tensor::backend::BackendTypes>::Device,
        ) -> Tensor<B, 1> {
            let dims = population.dims();
            let pop_size = dims[0];
            let data = population
                .clone()
                .into_data()
                .into_vec::<i32>()
                .expect("genome host-read of a tensor this test just built");
            let mut fitness = Vec::with_capacity(pop_size);
            for row in 0..pop_size {
                let mut ones = 0_u32;
                for col in 0..self.dim {
                    if data[row * self.dim + col] != 0 {
                        ones += 1;
                    }
                }
                #[allow(clippy::cast_precision_loss)]
                fitness.push(ones as f32);
            }
            Tensor::<B, 1>::from_data(TensorData::new(fitness, [pop_size]), device)
        }

        fn sense(&self) -> rlevo_core::objective::ObjectiveSense {
            rlevo_core::objective::ObjectiveSense::Maximize
        }
    }

    #[test]
    fn binary_ga_solves_onemax() {
        let device = Default::default();
        let dim = 16;
        let params = BinaryGaConfig::default_for(32, dim);
        let mut harness = EvolutionaryHarness::<TestBackend, _, _>::new(
            BinaryGeneticAlgorithm::<TestBackend>::new(),
            params,
            OneMax { dim },
            7,
            device,
            200,
        )
        .expect("valid params");
        harness.reset();
        loop {
            if harness.step(()).done {
                break;
            }
        }
        let best = harness.latest_metrics().unwrap().best_fitness_ever();
        // OneMax optimum: all ones → fitness == D.
        #[allow(clippy::cast_precision_loss)]
        let optimum = dim as f32;
        approx::assert_relative_eq!(best, optimum, epsilon = 1e-6);
    }

    /// Runs one `ask` pipeline over a *uniform* population (every parent is the
    /// same constant bit) and returns the flat offspring bits. With every
    /// parent identical, uniform crossover is a no-op (both parents agree at
    /// every gene), so the only source of variation is bit-flip mutation — the
    /// offspring is therefore a pure function of `mutation_rate`, RNG-invariant.
    fn ask_over_uniform_population(rate: f32, bit: i32) -> Vec<i32> {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let device = Default::default();
        let (pop, dim) = (4usize, 8usize);
        let mut params = BinaryGaConfig::default_for(pop, dim);
        params.mutation_rate = Probability::new(rate);
        let strategy = BinaryGeneticAlgorithm::<TestBackend>::new();

        let population = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![bit; pop * dim], [pop, dim]),
            &device,
        );
        // Non-empty `fitness` drives the real selection→crossover→mutation path
        // (an empty cache would short-circuit to returning the seed population).
        let state = BinaryGaState {
            population,
            fitness: vec![1.0; pop],
            best_genome: None,
            best_fitness: f32::NEG_INFINITY,
            generation: 1,
        };
        let mut rng = StdRng::seed_from_u64(0);
        let (offspring, _) = strategy.ask(&params, &state, &mut rng, &device);
        offspring
            .into_data()
            .into_vec::<i32>()
            .expect("offspring host-read of a tensor this test just built")
    }

    /// `mutation_rate == 0.0` performs no bit flips: over a uniform all-zero
    /// population (where crossover is a no-op) the offspring must be all zeros,
    /// i.e. a subset of the parents' bit values.
    #[test]
    fn mutation_rate_zero_flips_no_bits() {
        let bits = ask_over_uniform_population(0.0, 0);
        assert!(
            bits.iter().all(|&b| b == 0),
            "rate 0.0 must leave every offspring bit at the parent value, got {bits:?}"
        );
    }

    /// `mutation_rate == 1.0` flips every bit deterministically: an all-zero
    /// uniform population becomes all ones after mutation.
    #[test]
    fn mutation_rate_one_flips_every_bit() {
        let bits = ask_over_uniform_population(1.0, 0);
        assert!(
            bits.iter().all(|&b| b == 1),
            "rate 1.0 must flip every 0→1, got {bits:?}"
        );
    }

    /// Elitism boundary `elitism_k == pop_size - 1`: all but one slot are the
    /// top parents, leaving exactly one offspring slot. The `pop_size − 1`
    /// fittest parents survive (in descending-fitness order) followed by the
    /// single best offspring, and the champion tracks the best offspring.
    #[test]
    fn elitism_k_equals_pop_minus_one_keeps_top_parents_and_one_offspring() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let device = Default::default();
        let (pop, dim) = (4usize, 4usize);
        let mut params = BinaryGaConfig::default_for(pop, dim);
        params.elitism_k = pop - 1; // boundary: one offspring slot.
        let strategy = BinaryGeneticAlgorithm::<TestBackend>::new();

        // Four distinct parents with strictly increasing fitness.
        let parent_pop = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![0i32; pop * dim], [pop, dim]),
            &device,
        );
        let state = BinaryGaState {
            population: parent_pop,
            fitness: vec![1.0, 2.0, 3.0, 4.0],
            best_genome: None,
            best_fitness: 4.0,
            generation: 1,
        };

        // Offspring: only row 0 beats every parent; the rest are worst.
        let offspring = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1i32; pop * dim], [pop, dim]),
            &device,
        );
        let off_fitness = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(vec![10.0f32, 0.0, 0.0, 0.0], [pop]),
            &device,
        );

        let mut rng = StdRng::seed_from_u64(0);
        let (next, m) = strategy.tell(&params, offspring, off_fitness, state, &mut rng);

        // Top `pop-1` parents (fitness 4,3,2 in descending order) then the best
        // offspring (fitness 10).
        assert_eq!(next.population.dims(), [pop, dim]);
        assert_eq!(next.fitness, vec![4.0, 3.0, 2.0, 10.0]);
        approx::assert_relative_eq!(m.best_fitness_ever(), 10.0, epsilon = 1e-6);
    }
}
