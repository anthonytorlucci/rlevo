//! The generation loop for a [`CoEvolutionaryAlgorithm`].
//!
//! [`CoEvolutionaryLoop`] is to [`CoEvolutionaryAlgorithm`] what
//! [`EvolutionaryLoop`](crate::strategy::EvolutionaryLoop) is to
//! [`Strategy`](crate::strategy::Strategy): it owns the joint state, the RNG,
//! and the generation budget. It is driven the same way, with
//! [`reset`](CoEvolutionaryLoop::reset), [`step`](CoEvolutionaryLoop::step)
//! and [`is_done`](CoEvolutionaryLoop::is_done), and it reaches the evaluation
//! harness the same way, through `rlevo-core::evaluation::GenerationProbe`.
//! One `step`, or one
//! [`advance`](rlevo_core::evaluation::GenerationProbe::advance), is one
//! simultaneous-update generation.

use std::fmt::Debug;
use std::marker::PhantomData;

use burn::tensor::backend::Backend;
use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_core::config::{ConfigError, Validate};

use super::CoEvolutionaryAlgorithm;

/// Per-generation summary for a co-evolutionary run.
///
/// The [`CoEAMetrics`] analogue of
/// [`StrategyMetrics`](crate::strategy::StrategyMetrics), but tracking both
/// populations separately so a benchmark report can plot per-population
/// dynamics. The four `best_fitness_*` / `mean_fitness_*` display fields are
/// reported in the objective's **natural** declared sense (parity with
/// single-population `StrategyMetrics`); the separate `binding_fitness` field
/// stays canonical (engine-space) (ADR 0023).
#[derive(Debug, Clone)]
pub struct CoEAMetrics {
    /// Number of completed simultaneous-update generations.
    pub generation: u64,
    /// Best fitness population A has seen so far, in the objective's **natural**
    /// declared sense (a `Minimize` cost reads as its natural cost).
    pub best_fitness_a: f32,
    /// Best fitness population B has seen so far, in the objective's **natural**
    /// declared sense (a `Minimize` cost reads as its natural cost).
    pub best_fitness_b: f32,
    /// Mean fitness of population A this generation, in the objective's
    /// **natural** declared sense.
    pub mean_fitness_a: f32,
    /// Mean fitness of population B this generation, in the objective's
    /// **natural** declared sense.
    pub mean_fitness_b: f32,
    /// Canonical (engine-space, maximise) binding fitness `min(best_a, best_b)`
    /// — the weaker population binds. Engine-space, NOT mapped to the
    /// objective's natural sense, so a higher value is always better. All
    /// other fitness fields are in the objective's natural sense.
    pub binding_fitness: f32,
    /// Hall-of-fame archive size for population A (`0` if no archive).
    pub hof_size_a: usize,
    /// Hall-of-fame archive size for population B (`0` if no archive).
    pub hof_size_b: usize,
}

/// Runs a [`CoEvolutionaryAlgorithm`] one generation per call, owning the
/// joint state, the seed and the generation budget of one run.
///
/// It is driven exactly as
/// [`EvolutionaryLoop`](crate::strategy::EvolutionaryLoop) is, and for the
/// same reasons, which that type's documentation gives:
///
/// ```text
/// coevo_loop.reset();
/// while !coevo_loop.is_done() {
///     coevo_loop.step();
/// }
/// ```
///
/// [`reset`](Self::reset) builds the joint state on the configured device.
/// [`step`](Self::step) runs one generation and returns nothing.
/// [`is_done`](Self::is_done) says whether the generation budget is spent,
/// and the loop that calls `step` is what stops there.
///
/// The single number that summarises a generation is the
/// **canonical** `binding_fitness = min(best_a, best_b)` (canonical maximise,
/// no negation): the weaker population — the lower canonical fitness — is the
/// binding constraint, and a higher binding value is better.
/// The per-population `best_fitness_{a,b}` / `mean_fitness_{a,b}` in
/// [`CoEAMetrics`] are reported in the objective's **natural** sense (ADR 0023);
/// only `binding_fitness` stays canonical.
///
/// Per-generation metrics are emitted through `tracing` with structured
/// per-population fields. (A dual-population [`PopulationObserver`] channel —
/// the single-population
/// [`PopulationSnapshot`](crate::observer::PopulationSnapshot) cannot carry
/// both populations — is deferred to a follow-up.)
///
/// # Determinism
///
/// As for [`EvolutionaryLoop`](crate::strategy::EvolutionaryLoop): a run is
/// a function of its seed, because every draw comes from the loop's host RNG
/// and none from the backend's process-global generator. The parallel case
/// is measured and pinned for the single-population loop only.
///
/// [`PopulationObserver`]: crate::observer::PopulationObserver
pub struct CoEvolutionaryLoop<B, C>
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

impl<B, C> Debug for CoEvolutionaryLoop<B, C>
where
    B: Backend,
    C: CoEvolutionaryAlgorithm<B>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoEvolutionaryLoop")
            .field("base_seed", &self.base_seed)
            .field("generation", &self.generation)
            .field("max_generations", &self.max_generations)
            .field("latest_metrics", &self.latest_metrics)
            .finish_non_exhaustive()
    }
}

impl<B, C> CoEvolutionaryLoop<B, C>
where
    B: Backend,
    C: CoEvolutionaryAlgorithm<B>,
{
    /// Build a new loop from an algorithm, its params, a seed, a device,
    /// and a generation budget.
    ///
    /// The caller-supplied `params` are validated up front — this is the
    /// co-evolutionary loop consumption chokepoint (ADR 0026).
    ///
    /// The loop is lazily initialized — the first [`reset`](Self::reset)
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

    /// The configured generation budget.
    ///
    /// The loop owns this number. A driver should ask
    /// [`is_done`](Self::is_done) rather than keep a second copy of the budget
    /// and count against it.
    #[must_use]
    pub const fn max_generations(&self) -> usize {
        self.max_generations
    }

    /// Whether the generation budget is spent.
    ///
    /// `true` once [`generation`](Self::generation) has reached
    /// [`max_generations`](Self::max_generations). A driving loop checks it
    /// before each [`step`](Self::step), so a budget of zero runs no
    /// generation. It reports the budget and nothing else, and it does not
    /// stop `step`, which runs a generation whenever it is called.
    ///
    /// [`GenerationProbe::advance`](rlevo_core::evaluation::GenerationProbe::advance)
    /// is this check followed by one `step`.
    #[must_use]
    pub const fn is_done(&self) -> bool {
        self.generation >= self.max_generations
    }

    /// Start a run: re-seed the RNG, zero the generation count and build a
    /// fresh joint state.
    ///
    /// Call it once before the first [`step`](Self::step); calling it again
    /// replays the run from the same seed. Infallible;
    /// [`GenerationProbe::begin`](rlevo_core::evaluation::GenerationProbe::begin)
    /// forwards to it.
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
    /// Returns nothing. Ask [`is_done`](Self::is_done) whether the budget is
    /// spent, and read the generation's summary from
    /// [`latest_metrics`](Self::latest_metrics).
    ///
    /// `step` does not consult the budget. Called when `is_done` is already
    /// `true`, it runs one more generation, for the reasons given on
    /// [`EvolutionaryLoop`](crate::strategy::EvolutionaryLoop) (ADR 0085).
    ///
    /// # Panics
    ///
    /// Panics if [`reset`](Self::reset) has not been called first.
    pub fn step(&mut self) {
        let state = self
            .state
            .take()
            .expect("CoEvolutionaryLoop::reset must be called before step");
        let (new_state, metrics) =
            self.algorithm
                .step(&self.params, state, &mut self.rng, &self.device);
        self.state = Some(new_state);
        self.generation += 1;

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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;
    use burn::tensor::{Tensor, TensorData};

    use rlevo_core::bounds::Bounds;
    use rlevo_core::objective::ObjectiveSense;
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
            crossover: GaCrossover::Uniform {
                p: Probability::new(0.5),
            },
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
        fn sense(&self) -> ObjectiveSense {
            ObjectiveSense::Maximize
        }
    }

    /// A `NaN` fitness from a [`CoupledFitness`] impl cannot make `binding_fitness` `NaN`: the
    /// coupled-fitness chokepoint sanitizes before `best_a`/ `best_b` are computed, so
    /// `min(best_a, best_b)` is finite-or-`$-\infty$`. Pins this guarantee per ADR 0034's
    /// chokepoint convention.
    #[test]
    fn binding_fitness_is_never_nan_with_nan_fitness() {
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
        let mut evo_loop =
            CoEvolutionaryLoop::<TB, _>::new(algo, params, 7, device, 3).expect("valid params");
        evo_loop.reset();
        evo_loop.step();

        let binding = evo_loop
            .latest_metrics()
            .expect("metrics after a step")
            .binding_fitness;
        assert!(!binding.is_nan(), "binding fitness must never be NaN");
        // The finite ramp maximum (POP - 1) binds both populations, so the
        // binding fitness is that finite value — the NaN row was sanitized,
        // not crowned.
        assert!(
            binding.is_finite(),
            "binding fitness should be the finite ramp maximum, got {binding}"
        );
        #[allow(clippy::cast_precision_loss)]
        let expected = (POP - 1) as f32;
        approx::assert_relative_eq!(binding, expected, epsilon = 1e-6);
    }

    /// Row-wise cost `i + 1` declaring [`ObjectiveSense::Minimize`]: row 0 is best (cost `1.0`),
    /// canonicalising to `$-1.0$` (the maximum).
    struct RowCostMin;

    impl CoupledFitness<TB> for RowCostMin {
        fn evaluate_coupled(&self, populations: &[Tensor<TB, 2>]) -> Vec<Tensor<TB, 1>> {
            populations
                .iter()
                .map(|p| {
                    let n = p.dims()[0];
                    let device = p.device();
                    #[allow(clippy::cast_precision_loss)]
                    let v: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();
                    Tensor::<TB, 1>::from_data(TensorData::new(v, [n]), &device)
                })
                .collect()
        }
        fn sense(&self) -> ObjectiveSense {
            ObjectiveSense::Minimize
        }
    }

    /// For a `Minimize` objective `binding_fitness` is CANONICAL (`min` of the canonical bests),
    /// not the natural cost. Row 0's natural cost `1.0` canonicalises to `$-1.0$`, so the binding
    /// value is `$-1.0$`, while the natural `best_fitness_a` reads `1.0`.
    #[test]
    fn minimize_binding_fitness_is_canonical() {
        let device = Default::default();
        let algo = CompetitiveCoEA::new(
            GeneticAlgorithm::<TB>::new(),
            GeneticAlgorithm::<TB>::new(),
            RowCostMin,
        );
        let params: CompetitiveCoEAParams<GaConfig, GaConfig> = CompetitiveCoEAParams {
            params_a: ga_config(),
            params_b: ga_config(),
        };
        let mut evo_loop =
            CoEvolutionaryLoop::<TB, _>::new(algo, params, 7, device, 3).expect("valid params");
        evo_loop.reset();
        evo_loop.step();

        let m = evo_loop.latest_metrics().expect("metrics after a step");
        // Canonical binding = `$\min(-1, -1) = -1$`.
        approx::assert_relative_eq!(m.binding_fitness, -1.0, epsilon = 1e-6);
        // Natural best reads back as the low cost 1.0.
        approx::assert_relative_eq!(m.best_fitness_a, 1.0, epsilon = 1e-6);
    }

    /// `is_done` is asked before a step, so for a budget of zero it is `true`
    /// straight after `reset` and a driving loop runs no generation. The same
    /// contract as `EvolutionaryLoop`, pinned here because the two loops are
    /// written separately.
    #[test]
    fn zero_budget_is_done_before_any_step() {
        let device = Default::default();
        let algo = CompetitiveCoEA::new(
            GeneticAlgorithm::<TB>::new(),
            GeneticAlgorithm::<TB>::new(),
            RowCostMin,
        );
        let params: CompetitiveCoEAParams<GaConfig, GaConfig> = CompetitiveCoEAParams {
            params_a: ga_config(),
            params_b: ga_config(),
        };
        let mut evo_loop =
            CoEvolutionaryLoop::<TB, _>::new(algo, params, 7, device, 0).expect("valid params");
        evo_loop.reset();
        assert!(evo_loop.is_done());
        assert!(evo_loop.latest_metrics().is_none());
    }

    /// `step` does not consult the budget: past it, a call still runs a
    /// generation, and `is_done` stays `true`.
    #[test]
    fn step_past_the_budget_runs_another_generation() {
        let device = Default::default();
        let algo = CompetitiveCoEA::new(
            GeneticAlgorithm::<TB>::new(),
            GeneticAlgorithm::<TB>::new(),
            RowCostMin,
        );
        let params: CompetitiveCoEAParams<GaConfig, GaConfig> = CompetitiveCoEAParams {
            params_a: ga_config(),
            params_b: ga_config(),
        };
        let mut evo_loop =
            CoEvolutionaryLoop::<TB, _>::new(algo, params, 7, device, 1).expect("valid params");
        evo_loop.reset();
        assert!(!evo_loop.is_done(), "a fresh run with a budget is not done");
        evo_loop.step();
        assert!(evo_loop.is_done());
        evo_loop.step();
        assert_eq!(evo_loop.generation(), 2, "the second call ran a generation");
        assert!(evo_loop.is_done());
    }
}
