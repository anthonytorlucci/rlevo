//! The control: a strategy that never moves.
//!
//! Its `ask` returns the initial population unchanged, so its best fitness
//! after any number of generations is its first. It takes the
//! hill-climber's parameters and its `init`, so from one seed the two start
//! from the same point, and any difference between them is the search.

use rand::Rng;
use rlevo::burn::tensor::{Device, Tensor};
use rlevo::evo::strategy::{Strategy, StrategyMetrics};

use crate::hill_climber::{HillClimber, HillClimberParams, HillClimberState, row};

/// The strategy that proposes its starting point every generation.
#[derive(Debug, Clone, Copy, Default)]
pub struct Null;

impl Strategy for Null {
    type Params = HillClimberParams;
    type State = HillClimberState;
    type Genome = Tensor<2>;

    fn init(&self, params: &Self::Params, rng: &mut dyn Rng, device: &Device) -> Self::State {
        HillClimber.init(params, rng, device)
    }

    fn ask(
        &self,
        _params: &Self::Params,
        state: &Self::State,
        _rng: &mut dyn Rng,
        device: &Device,
    ) -> (Self::Genome, Self::State) {
        (row(&state.parent, device), state.clone())
    }

    fn tell(
        &self,
        params: &Self::Params,
        population: Self::Genome,
        fitness: Tensor<1>,
        state: Self::State,
        rng: &mut dyn Rng,
    ) -> (Self::State, StrategyMetrics) {
        // The proposal is always the parent, so the hill-climber's
        // bookkeeping keeps the parent where it is.
        HillClimber.tell(params, population, fitness, state, rng)
    }

    fn best(&self, state: &Self::State) -> Option<(Self::Genome, f32)> {
        HillClimber.best(state)
    }
}
