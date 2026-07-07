//! Competitive co-evolution — predator vs. prey (Hillis 1990).
//!
//! Two populations are adversaries: each is scored by how well it performs
//! against the other, driving an arms race. [`CompetitiveCoEA`] runs both
//! under simultaneous updates — `ask` both, evaluate the pair with a single
//! [`CoupledFitness`], `tell` both. The fitness function carries the
//! adversarial relationship; wrapping it in
//! [`HallOfFameFitness`](super::HallOfFameFitness) adds cycling mitigation
//! without any change to this algorithm.

use std::fmt::Debug;
use std::marker::PhantomData;

use burn::tensor::{Tensor, backend::Backend};
use rand::Rng;

use rlevo_core::config::{ConfigError, Validate};

use crate::fitness::sanitize_fitness_tensor;
use crate::strategy::Strategy;

use super::fitness::CoupledFitness;
use super::harness::CoEAMetrics;
use super::{CoEAState, CoEvolutionaryAlgorithm};

/// Static parameters for [`CompetitiveCoEA`]: one params bundle per inner
/// strategy.
#[derive(Debug, Clone)]
pub struct CompetitiveCoEAParams<PA, PB> {
    /// Params for population A's inner strategy.
    pub params_a: PA,
    /// Params for population B's inner strategy.
    pub params_b: PB,
}

/// Validation delegates to nothing: [`CompetitiveCoEAParams`] carries only the
/// two inner-strategy params, each already validated at that strategy's own
/// harness chokepoint. No `PA: Validate` / `PB: Validate` bound is imposed, so
/// the params stay usable with unit-typed inner params in tests.
impl<PA, PB> Validate for CompetitiveCoEAParams<PA, PB> {
    fn validate(&self) -> Result<(), ConfigError> {
        Ok(())
    }
}

/// Competitive co-evolutionary algorithm over two adversarial populations.
///
/// Generic over the backend `B`, the two inner strategies `SA`/`SB` (each
/// producing `Tensor<B, 2>` genomes), and the [`CoupledFitness`] `F` that
/// scores them against each other. Implements
/// [`CoEvolutionaryAlgorithm`] so it can be driven by
/// [`CoEvolutionaryHarness`](super::CoEvolutionaryHarness).
pub struct CompetitiveCoEA<B, SA, SB, F>
where
    B: Backend,
    SA: Strategy<B, Genome = Tensor<B, 2>>,
    SB: Strategy<B, Genome = Tensor<B, 2>>,
    F: CoupledFitness<B>,
{
    strategy_a: SA,
    strategy_b: SB,
    fitness: F,
    _backend: PhantomData<fn() -> B>,
}

impl<B, SA, SB, F> Debug for CompetitiveCoEA<B, SA, SB, F>
where
    B: Backend,
    SA: Strategy<B, Genome = Tensor<B, 2>>,
    SB: Strategy<B, Genome = Tensor<B, 2>>,
    F: CoupledFitness<B>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompetitiveCoEA").finish_non_exhaustive()
    }
}

impl<B, SA, SB, F> CompetitiveCoEA<B, SA, SB, F>
where
    B: Backend,
    SA: Strategy<B, Genome = Tensor<B, 2>>,
    SB: Strategy<B, Genome = Tensor<B, 2>>,
    F: CoupledFitness<B>,
{
    /// Build a competitive co-evolution from two inner strategies and a
    /// coupled fitness.
    pub fn new(strategy_a: SA, strategy_b: SB, fitness: F) -> Self {
        Self {
            strategy_a,
            strategy_b,
            fitness,
            _backend: PhantomData,
        }
    }

    /// Project the joint state into public [`CoEAMetrics`].
    ///
    /// The `best`/`mean` fields are copied from `state`, which `step` sources
    /// from the per-population `tell` metrics computed over the **sanitized**
    /// fitness (see the chokepoint in [`step`](Self::step) and ADR 0034). No
    /// non-finite value can reach here as a `NaN`: means are averaged over the
    /// finite members only (`StrategyMetrics::from_host_fitness`), so a broken
    /// individual cannot blank a mean.
    fn snapshot(&self, state: &CoEAState<SA::State, SB::State>) -> CoEAMetrics {
        let sizes = self.fitness.archive_sizes();
        CoEAMetrics {
            generation: state.generation,
            best_fitness_a: state.best_a,
            best_fitness_b: state.best_b,
            mean_fitness_a: state.mean_a,
            mean_fitness_b: state.mean_b,
            hof_size_a: sizes.first().copied().unwrap_or(0),
            hof_size_b: sizes.get(1).copied().unwrap_or(0),
        }
    }
}

impl<B, SA, SB, F> CoEvolutionaryAlgorithm<B> for CompetitiveCoEA<B, SA, SB, F>
where
    B: Backend,
    SA: Strategy<B, Genome = Tensor<B, 2>>,
    SB: Strategy<B, Genome = Tensor<B, 2>>,
    F: CoupledFitness<B>,
{
    type Params = CompetitiveCoEAParams<SA::Params, SB::Params>;
    type State = CoEAState<SA::State, SB::State>;

    fn init(
        &self,
        params: &Self::Params,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Self::State {
        let state_a = self.strategy_a.init(&params.params_a, rng, device);
        let state_b = self.strategy_b.init(&params.params_b, rng, device);
        CoEAState::new(state_a, state_b)
    }

    fn step(
        &self,
        params: &Self::Params,
        mut state: Self::State,
        rng: &mut dyn Rng,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> (Self::State, CoEAMetrics) {
        // Both populations propose simultaneously.
        let (pop_a, asked_a) = self.strategy_a.ask(&params.params_a, &state.state_a, rng, device);
        let (pop_b, asked_b) = self.strategy_b.ask(&params.params_b, &state.state_b, rng, device);

        // Single coupled evaluation scores each against the other.
        let fits = self
            .fitness
            .evaluate_coupled(&[pop_a.clone(), pop_b.clone()]);
        debug_assert_eq!(fits.len(), 2, "competitive co-evolution is bi-population");

        // Fitness-hygiene chokepoint for the coupled-fitness path (ADR 0034):
        // `evaluate_coupled` may return non-finite fitness, and — unlike the
        // single-population `EvolutionaryHarness` — nothing above this point
        // sanitizes. Clean each vector *once* here (`NaN → −∞` worst,
        // `+∞ → f32::MAX`), so the per-population `tell`, the `snapshot`
        // best/mean written into `CoEAState`, and any `HallOfFameFitness`
        // downstream all see finite-or-`−∞` fitness. A raw positive `NaN` would
        // otherwise `total_cmp` as the maximum and be crowned champion.
        let fit_a = sanitize_fitness_tensor(fits[0].clone());
        let fit_b = sanitize_fitness_tensor(fits[1].clone());

        // Both populations consume their relative fitness.
        let (next_a, metrics_a) = self
            .strategy_a
            .tell(&params.params_a, pop_a, fit_a, asked_a, rng);
        let (next_b, metrics_b) = self
            .strategy_b
            .tell(&params.params_b, pop_b, fit_b, asked_b, rng);

        state.state_a = next_a;
        state.state_b = next_b;
        state.generation += 1;
        state.best_a = metrics_a.best_fitness_ever();
        state.best_b = metrics_b.best_fitness_ever();
        state.mean_a = metrics_a.mean_fitness();
        state.mean_b = metrics_b.mean_fitness();

        let metrics = self.snapshot(&state);
        (state, metrics)
    }

    fn metrics(&self, state: &Self::State) -> CoEAMetrics {
        self.snapshot(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;
    use burn::tensor::TensorData;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use rlevo_core::bounds::Bounds;
    use rlevo_core::probability::Probability;
    use rlevo_core::rate::NonNegativeRate;

    use crate::algorithms::ga::{
        GaConfig, GaCrossover, GaReplacement, GaSelection, GeneticAlgorithm,
    };

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

    /// Coupled fitness that poisons row 0 of every population with a
    /// non-finite value and fills the rest with a finite ramp `1, 2, …`.
    /// `poison` selects `NaN` or `+∞`; either must be sanitized at the
    /// chokepoint so the finite ramp maximum (`POP - 1`) is the champion.
    struct PoisonRow0 {
        poison: f32,
    }

    impl CoupledFitness<TB> for PoisonRow0 {
        fn evaluate_coupled(&self, populations: &[Tensor<TB, 2>]) -> Vec<Tensor<TB, 1>> {
            populations
                .iter()
                .map(|p| {
                    let n = p.dims()[0];
                    let device = p.device();
                    #[allow(clippy::cast_precision_loss)]
                    let v: Vec<f32> = (0..n)
                        .map(|i| if i == 0 { self.poison } else { i as f32 })
                        .collect();
                    Tensor::<TB, 1>::from_data(TensorData::new(v, [n]), &device)
                })
                .collect()
        }
    }

    /// A single all-`NaN` coupled fitness — the degenerate whole-population
    /// break — must sanitize to `−∞`, never `NaN`.
    struct AllNan;

    impl CoupledFitness<TB> for AllNan {
        fn evaluate_coupled(&self, populations: &[Tensor<TB, 2>]) -> Vec<Tensor<TB, 1>> {
            populations
                .iter()
                .map(|p| {
                    let n = p.dims()[0];
                    let device = p.device();
                    Tensor::<TB, 1>::from_data(TensorData::new(vec![f32::NAN; n], [n]), &device)
                })
                .collect()
        }
    }

    fn run_one_step<F: CoupledFitness<TB>>(fitness: F) -> CoEAMetrics {
        let device = Default::default();
        let algo = CompetitiveCoEA::new(
            GeneticAlgorithm::<TB>::new(),
            GeneticAlgorithm::<TB>::new(),
            fitness,
        );
        let params: CompetitiveCoEAParams<GaConfig, GaConfig> = CompetitiveCoEAParams {
            params_a: ga_config(),
            params_b: ga_config(),
        };
        let mut rng = StdRng::seed_from_u64(7);
        let state = algo.init(&params, &mut rng, &device);
        let (_next, metrics) = algo.step(&params, state, &mut rng, &device);
        metrics
    }

    /// A `NaN` fitness from a [`CoupledFitness`] impl is sanitized to `−∞` at
    /// the chokepoint, so it can neither become the champion nor blank a mean:
    /// the finite ramp maximum (`POP - 1`) is `best`, and the mean is finite
    /// (averaged over the finite members only). Regression for issue #134.
    #[test]
    fn nan_row_is_not_crowned_and_mean_stays_finite() {
        let m = run_one_step(PoisonRow0 { poison: f32::NAN });
        #[allow(clippy::cast_precision_loss)]
        let expected_best = (POP - 1) as f32;
        approx::assert_relative_eq!(m.best_fitness_a, expected_best, epsilon = 1e-6);
        approx::assert_relative_eq!(m.best_fitness_b, expected_best, epsilon = 1e-6);
        assert!(
            m.mean_fitness_a.is_finite(),
            "mean_fitness_a must stay finite when a NaN individual is present, got {}",
            m.mean_fitness_a
        );
        assert!(
            m.mean_fitness_b.is_finite(),
            "mean_fitness_b must stay finite when a NaN individual is present, got {}",
            m.mean_fitness_b
        );
    }

    /// A `+∞` fitness is clamped to `f32::MAX` (still the top rank, but finite),
    /// so it cannot blow the population mean up to `+∞`. Regression for the
    /// ADR 0034 `+∞` rule on the coevolution path.
    #[test]
    fn pos_inf_fitness_is_clamped_finite_in_metrics() {
        let m = run_one_step(PoisonRow0 { poison: f32::INFINITY });
        approx::assert_relative_eq!(m.best_fitness_a, f32::MAX);
        assert!(
            m.mean_fitness_a.is_finite(),
            "a +∞ individual must not push the mean to +∞, got {}",
            m.mean_fitness_a
        );
    }

    /// An all-`NaN` population sanitizes to `−∞` everywhere: `best`/`mean` are
    /// the well-defined `−∞` sentinel (degenerate but flagged), never `NaN`.
    #[test]
    fn all_nan_population_yields_neg_inf_never_nan() {
        let m = run_one_step(AllNan);
        assert!(!m.best_fitness_a.is_nan(), "best must never be NaN");
        assert!(!m.mean_fitness_a.is_nan(), "mean must never be NaN");
        assert!(
            m.best_fitness_a.is_infinite() && m.best_fitness_a.is_sign_negative(),
            "all-broken population best is the −∞ sentinel, got {}",
            m.best_fitness_a
        );
    }
}
