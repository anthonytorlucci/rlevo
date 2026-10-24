//! The submission: a `(1+1)` hill-climber, written against the `Strategy`
//! seam from outside.
//!
//! One parent. The first generation scores the parent itself; every later
//! one proposes the parent plus a uniform step in each coordinate, and keeps
//! the proposal if it scores no worse. The strategy holds nothing between
//! calls: `init`, `ask` and `tell` pass the state along, and every draw comes
//! from the RNG the loop hands in, so a run is a function of its seed.

use rlevo::burn::tensor::{Device, Tensor, TensorData};
use rlevo::core::config::{ConfigError, Validate, nonzero, ordered, positive};
use rlevo::evo::fitness::sanitize_fitness;
use rlevo::evo::strategy::{Strategy, StrategyMetrics};
use rlevo::prelude::{Rng, RngExt};

/// What a run is configured with.
#[derive(Debug, Clone)]
pub struct HillClimberParams {
    /// The number of coordinates a point has.
    pub dim: usize,
    /// The lowest value a coordinate may take.
    pub low: f32,
    /// The highest value a coordinate may take.
    pub high: f32,
    /// The largest change a proposal makes to one coordinate.
    pub step: f32,
}

impl Validate for HillClimberParams {
    fn validate(&self) -> Result<(), ConfigError> {
        const NAME: &str = "HillClimberParams";
        nonzero(NAME, "dim", self.dim)?;
        ordered(NAME, "low", f64::from(self.low), f64::from(self.high))?;
        positive(NAME, "step", f64::from(self.step))
    }
}

/// What one generation hands the next.
#[derive(Debug, Clone)]
pub struct HillClimberState {
    /// The point proposals are made from.
    pub parent: Vec<f32>,
    /// The parent's fitness, in the engine's direction (higher is better);
    /// `None` until the parent has been scored.
    pub parent_fitness: Option<f32>,
    /// The best point scored so far, and its fitness.
    pub best: Option<(Tensor<2>, f32)>,
    /// Generations completed.
    pub generation: usize,
}

/// The hill-climber. It has no fields: everything a run needs is in its
/// parameters and its state.
#[derive(Debug, Clone, Copy, Default)]
pub struct HillClimber;

/// One point as a one-row population.
pub fn row(point: &[f32], device: &Device) -> Tensor<2> {
    Tensor::from_data(TensorData::new(point.to_vec(), [1, point.len()]), device)
}

fn to_host(tensor: Tensor<2>) -> Vec<f32> {
    tensor
        .into_data()
        .try_into_vec::<f32>()
        .expect("a population built from f32 data reads back as f32")
}

impl Strategy for HillClimber {
    type Params = HillClimberParams;
    type State = HillClimberState;
    type Genome = Tensor<2>;

    fn init(&self, params: &Self::Params, rng: &mut dyn Rng, _device: &Device) -> Self::State {
        let parent = (0..params.dim)
            .map(|_| rng.random_range(params.low..=params.high))
            .collect();
        HillClimberState {
            parent,
            parent_fitness: None,
            best: None,
            generation: 0,
        }
    }

    fn ask(
        &self,
        params: &Self::Params,
        state: &Self::State,
        rng: &mut dyn Rng,
        device: &Device,
    ) -> (Self::Genome, Self::State) {
        let proposal: Vec<f32> = if state.parent_fitness.is_none() {
            // Nothing has been scored yet, so the parent itself goes first.
            state.parent.clone()
        } else {
            state
                .parent
                .iter()
                .map(|&x| {
                    (x + rng.random_range(-params.step..=params.step))
                        .clamp(params.low, params.high)
                })
                .collect()
        };
        (row(&proposal, device), state.clone())
    }

    fn tell(
        &self,
        _params: &Self::Params,
        population: Self::Genome,
        fitness: Tensor<1>,
        mut state: Self::State,
        _rng: &mut dyn Rng,
    ) -> (Self::State, StrategyMetrics) {
        // The loop hands over fitness already sanitised, but `tell` can be
        // called without it, so it is sanitised again where it reaches the
        // host, as `Strategy::tell`'s documentation asks.
        let scores: Vec<f32> = fitness
            .into_data()
            .try_into_vec::<f32>()
            .expect("fitness reads back as f32")
            .into_iter()
            .map(sanitize_fitness)
            .collect();
        let score = scores[0];
        let best_before = state.best.as_ref().map_or(f32::NEG_INFINITY, |(_, f)| *f);
        if state.parent_fitness.is_none_or(|parent| score >= parent) {
            state.parent = to_host(population.clone());
            state.parent_fitness = Some(score);
        }
        if score > best_before {
            state.best = Some((population, score));
        }
        state.generation += 1;
        let metrics = StrategyMetrics::from_host_fitness(state.generation, &scores, best_before);
        (state, metrics)
    }

    fn best(&self, state: &Self::State) -> Option<(Self::Genome, f32)> {
        state.best.clone()
    }
}
