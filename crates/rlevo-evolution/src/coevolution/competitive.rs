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
        let fit_a = fits[0].clone();
        let fit_b = fits[1].clone();

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
