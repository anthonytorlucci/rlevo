//! Drive loop adapting a [`CoEvolutionaryAlgorithm`] to `BenchEnv`.
//!
//! [`CoEvolutionaryHarness`] is to [`CoEvolutionaryAlgorithm`] what
//! [`EvolutionaryHarness`](crate::strategy::EvolutionaryHarness) is to
//! [`Strategy`](crate::strategy::Strategy): it owns the joint state, the RNG,
//! and the generation budget, and exposes the run to the `rlevo-benchmarks`
//! evaluator through `rlevo-core::evaluation::BenchEnv` with no benchmark-side
//! changes. One [`BenchEnv::step`] drives one simultaneous-update generation.

use std::fmt::Debug;
use std::marker::PhantomData;

use burn::tensor::backend::Backend;
use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_core::config::{ConfigError, Validate};
use rlevo_core::evaluation::{BenchEnv, BenchError, BenchStep};

use super::CoEvolutionaryAlgorithm;

/// Per-generation summary for a co-evolutionary run.
///
/// The [`CoEAMetrics`] analogue of
/// [`StrategyMetrics`](crate::strategy::StrategyMetrics), but tracking both
/// populations separately so a benchmark report can plot per-population
/// dynamics. Fitness follows the canonical maximise convention (higher is better).
#[derive(Debug, Clone)]
pub struct CoEAMetrics {
    /// Number of completed simultaneous-update generations.
    pub generation: u64,
    /// Best (highest) fitness population A has seen so far.
    pub best_fitness_a: f32,
    /// Best (highest) fitness population B has seen so far.
    pub best_fitness_b: f32,
    /// Mean fitness of population A in this generation.
    pub mean_fitness_a: f32,
    /// Mean fitness of population B in this generation.
    pub mean_fitness_b: f32,
    /// Hall-of-fame archive size for population A (`0` if no archive).
    pub hof_size_a: usize,
    /// Hall-of-fame archive size for population B (`0` if no archive).
    pub hof_size_b: usize,
}

/// Wraps a [`CoEvolutionaryAlgorithm`] into a `BenchEnv`.
///
/// Like [`EvolutionaryHarness`](crate::strategy::EvolutionaryHarness), the
/// harness is lazily initialized: [`reset`](BenchEnv::reset) materializes the
/// joint state on the configured device, and each
/// [`step`](BenchEnv::step) runs one generation. The reward exposed to the
/// benchmark harness is `min(best_a, best_b)` (canonical maximise, no
/// negation): the weaker population — the lower canonical fitness — is the
/// binding constraint, and a higher binding value is better.
///
/// Per-generation metrics are emitted through `tracing` with structured
/// per-population fields. (A dual-population [`PopulationObserver`] channel —
/// the single-population
/// [`PopulationSnapshot`](crate::observer::PopulationSnapshot) cannot carry
/// both populations — is deferred to a follow-up.)
///
/// # Determinism
///
/// Determinism follows the same backend-RNG caveats documented on
/// [`EvolutionaryHarness`](crate::strategy::EvolutionaryHarness): run one
/// harness per process, or pin `EvaluatorConfig::num_threads = Some(1)`, for
/// bit-reproducible runs.
///
/// [`PopulationObserver`]: crate::observer::PopulationObserver
pub struct CoEvolutionaryHarness<B, C>
where
    B: Backend,
    C: CoEvolutionaryAlgorithm<B>,
{
    algorithm: C,
    params: C::Params,
    state: Option<C::State>,
    rng: StdRng,
    base_seed: u64,
    device: B::Device,
    generation: usize,
    max_generations: usize,
    latest_metrics: Option<CoEAMetrics>,
    _backend: PhantomData<B>,
}

impl<B, C> Debug for CoEvolutionaryHarness<B, C>
where
    B: Backend,
    C: CoEvolutionaryAlgorithm<B>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoEvolutionaryHarness")
            .field("base_seed", &self.base_seed)
            .field("generation", &self.generation)
            .field("max_generations", &self.max_generations)
            .field("latest_metrics", &self.latest_metrics)
            .finish_non_exhaustive()
    }
}

impl<B, C> CoEvolutionaryHarness<B, C>
where
    B: Backend,
    C: CoEvolutionaryAlgorithm<B>,
{
    /// Build a new harness from an algorithm, its params, a seed, a device,
    /// and a generation budget.
    ///
    /// The caller-supplied `params` are validated up front — this is the
    /// co-evolutionary harness consumption chokepoint (ADR 0026).
    ///
    /// The harness is lazily initialized — the first [`reset`](Self::reset)
    /// call materializes the joint state on `device`.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] when `params` fails [`Validate::validate`],
    /// naming the offending field and violated invariant.
    pub fn new(
        algorithm: C,
        params: C::Params,
        seed: u64,
        device: B::Device,
        max_generations: usize,
    ) -> Result<Self, ConfigError>
    where
        C::Params: Validate,
    {
        params.validate()?;
        Ok(Self {
            algorithm,
            params,
            state: None,
            rng: StdRng::seed_from_u64(seed),
            base_seed: seed,
            device,
            generation: 0,
            max_generations,
            latest_metrics: None,
            _backend: PhantomData,
        })
    }

    /// The most recent generation's metrics, if any.
    #[must_use]
    pub fn latest_metrics(&self) -> Option<&CoEAMetrics> {
        self.latest_metrics.as_ref()
    }

    /// Number of completed generations.
    #[must_use]
    pub fn generation(&self) -> usize {
        self.generation
    }

    /// Reset to a fresh joint state, re-seeding the RNG.
    ///
    /// Infallible; the [`BenchEnv`] impl wraps this in `Ok(())`.
    pub fn reset(&mut self) {
        self.rng = StdRng::seed_from_u64(self.base_seed);
        self.generation = 0;
        self.latest_metrics = None;
        self.state = Some(
            self.algorithm
                .init(&self.params, &mut self.rng, &self.device),
        );
    }

    /// Run one simultaneous-update generation.
    ///
    /// Infallible; the [`BenchEnv`] impl wraps the result in `Ok(...)`.
    ///
    /// # Panics
    ///
    /// Panics if [`reset`](Self::reset) has not been called first.
    pub fn step(&mut self, _action: ()) -> BenchStep<()> {
        let state = self
            .state
            .take()
            .expect("CoEvolutionaryHarness::reset must be called before step");
        let (new_state, metrics) =
            self.algorithm
                .step(&self.params, state, &mut self.rng, &self.device);
        self.state = Some(new_state);
        self.generation += 1;

        // Canonical maximise: the weaker population (lower canonical fitness)
        // is the binding constraint, and a higher binding value is better — so
        // the reward is the binding value directly, with no negation.
        //
        // Fitness hygiene (ADR 0034): `best_fitness_{a,b}` are sourced from the
        // per-population `tell` metrics over the fitness the coupled-fitness
        // chokepoint already sanitized (competitive/cooperative `step`), so each
        // is finite-or-`−∞`, never `NaN`. `f32::min` therefore yields a
        // finite-or-`−∞` binding value — the reward is never `NaN`. (Even a
        // stray `NaN` operand would be dropped by IEEE `f32::min`, which returns
        // the non-`NaN` argument; the chokepoint makes that path unreachable.)
        let binding = metrics.best_fitness_a.min(metrics.best_fitness_b);
        let reward = f64::from(binding);

        tracing::info!(
            generation = metrics.generation,
            best_fitness_a = f64::from(metrics.best_fitness_a),
            best_fitness_b = f64::from(metrics.best_fitness_b),
            mean_fitness_a = f64::from(metrics.mean_fitness_a),
            mean_fitness_b = f64::from(metrics.mean_fitness_b),
            hof_size_a = metrics.hof_size_a,
            hof_size_b = metrics.hof_size_b,
            "coevolution generation",
        );

        self.latest_metrics = Some(metrics);
        let done = self.generation >= self.max_generations;
        BenchStep {
            observation: (),
            reward,
            done,
        }
    }
}

impl<B, C> BenchEnv for CoEvolutionaryHarness<B, C>
where
    B: Backend,
    C: CoEvolutionaryAlgorithm<B>,
{
    type Observation = ();
    type Action = ();

    fn reset(&mut self) -> Result<Self::Observation, BenchError> {
        CoEvolutionaryHarness::<B, C>::reset(self);
        Ok(())
    }

    fn step(&mut self, action: Self::Action) -> Result<BenchStep<Self::Observation>, BenchError> {
        Ok(CoEvolutionaryHarness::<B, C>::step(self, action))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;
    use burn::tensor::{Tensor, TensorData};

    use rlevo_core::bounds::Bounds;
    use rlevo_core::probability::Probability;
    use rlevo_core::rate::NonNegativeRate;

    use crate::algorithms::ga::{
        GaConfig, GaCrossover, GaReplacement, GaSelection, GeneticAlgorithm,
    };
    use crate::coevolution::{CompetitiveCoEA, CompetitiveCoEAParams, CoupledFitness};

    type TB = Flex;

    const POP: usize = 4;
    const DIM: usize = 2;

    fn ga_config() -> GaConfig {
        GaConfig {
            pop_size: POP,
            genome_dim: DIM,
            bounds: Bounds::new(0.0, 1.0),
            mutation_sigma: NonNegativeRate::new(0.1),
            selection: GaSelection::Tournament { size: 2 },
            crossover: GaCrossover::Uniform { p: Probability::new(0.5) },
            replacement: GaReplacement::Elitist { elitism_k: 1 },
        }
    }

    /// Row 0 is `NaN`, the rest a finite ramp — for both populations.
    struct PoisonRow0Nan;

    impl CoupledFitness<TB> for PoisonRow0Nan {
        fn evaluate_coupled(&self, populations: &[Tensor<TB, 2>]) -> Vec<Tensor<TB, 1>> {
            populations
                .iter()
                .map(|p| {
                    let n = p.dims()[0];
                    let device = p.device();
                    #[allow(clippy::cast_precision_loss)]
                    let v: Vec<f32> = (0..n)
                        .map(|i| if i == 0 { f32::NAN } else { i as f32 })
                        .collect();
                    Tensor::<TB, 1>::from_data(TensorData::new(v, [n]), &device)
                })
                .collect()
        }
    }

    /// A `NaN` fitness from a [`CoupledFitness`] impl cannot make the harness
    /// reward `NaN`: the coupled-fitness chokepoint sanitizes before `best_a`/
    /// `best_b` are computed, so `min(best_a, best_b)` is finite-or-`−∞`.
    /// Regression for issue #134 (harness §1.1) / ADR 0034.
    #[test]
    fn harness_reward_is_never_nan_with_nan_fitness() {
        let device = Default::default();
        let algo = CompetitiveCoEA::new(
            GeneticAlgorithm::<TB>::new(),
            GeneticAlgorithm::<TB>::new(),
            PoisonRow0Nan,
        );
        let params: CompetitiveCoEAParams<GaConfig, GaConfig> = CompetitiveCoEAParams {
            params_a: ga_config(),
            params_b: ga_config(),
        };
        let mut harness =
            CoEvolutionaryHarness::<TB, _>::new(algo, params, 7, device, 3).expect("valid params");
        harness.reset();
        let step = harness.step(());

        assert!(!step.reward.is_nan(), "harness reward must never be NaN");
        // The finite ramp maximum (POP - 1) binds both populations, so the
        // reward is that finite value — the NaN row was sanitized, not crowned.
        assert!(
            step.reward.is_finite(),
            "reward should be the finite binding value, got {}",
            step.reward
        );
        #[allow(clippy::cast_precision_loss)]
        let expected = f64::from((POP - 1) as f32);
        approx::assert_relative_eq!(step.reward, expected, epsilon = 1e-6);
    }
}
