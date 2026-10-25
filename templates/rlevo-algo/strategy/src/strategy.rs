//! Your strategy.
//!
//! `MyStrategy` works as it stands, so the crate runs from its first `cargo
//! run`: random search. Each generation proposes `population` points drawn
//! uniformly inside the bounds, and the state keeps the best point scored
//! so far. Replace `ask` and `tell` with your method, and keep the shape:
//!
//! - `Params`, the run's fixed settings, with `Validate`:
//!   `EvolutionaryLoop::new` refuses parameters that fail it;
//! - `State`, everything that changes from one generation to the next. The
//!   strategy itself holds nothing: `init`, `ask` and `tell` pass the state
//!   along;
//! - `Genome`, the population: here a tensor with one row per point.
//!
//! # Fitness
//!
//! Inside a strategy, higher is better. `EvolutionaryLoop` negates a cost it
//! is asked to minimise, and cleans the fitness it hands to `tell`. `tell`
//! cleans it again where it reaches the host, with `sanitize_fitness`,
//! because `tell` can be called without the loop, and a `NaN` must never be
//! crowned the best.
//!
//! # Randomness
//!
//! Every draw comes from the `rng` the loop passes, so a seed reproduces a
//! run.

use rlevo::burn::tensor::{Device, Tensor, TensorData};
use rlevo::core::config::{ConfigError, Validate, nonzero, ordered};
use rlevo::evo::fitness::sanitize_fitness;
use rlevo::evo::strategy::{Strategy, StrategyMetrics};
use rlevo::prelude::{Rng, RngExt};

/// A run's fixed settings.
#[derive(Debug, Clone)]
pub struct MyParams {
    /// Coordinates in a point.
    pub dim: usize,
    /// The lowest value a coordinate takes.
    pub low: f32,
    /// The highest value a coordinate takes.
    pub high: f32,
    /// Points proposed each generation.
    pub population: usize,
}

impl Validate for MyParams {
    fn validate(&self) -> Result<(), ConfigError> {
        const NAME: &str = "MyParams";
        nonzero(NAME, "dim", self.dim)?;
        nonzero(NAME, "population", self.population)?;
        ordered(NAME, "low", f64::from(self.low), f64::from(self.high))
    }
}

/// What one generation hands the next.
#[derive(Debug, Clone)]
pub struct MyState {
    /// The best point scored so far, as a one-row tensor, and its fitness.
    pub best: Option<(Tensor<2>, f32)>,
    /// Generations completed.
    pub generation: usize,
}

/// Random search.
#[derive(Debug, Clone, Copy, Default)]
pub struct MyStrategy;

impl Strategy for MyStrategy {
    type Params = MyParams;
    type State = MyState;
    type Genome = Tensor<2>;

    fn init(&self, _params: &MyParams, _rng: &mut dyn Rng, _device: &Device) -> MyState {
        MyState {
            best: None,
            generation: 0,
        }
    }

    /// `population` points, uniform inside the bounds.
    fn ask(
        &self,
        params: &MyParams,
        state: &MyState,
        rng: &mut dyn Rng,
        device: &Device,
    ) -> (Tensor<2>, MyState) {
        let points: Vec<f32> = (0..params.population * params.dim)
            .map(|_| rng.random_range(params.low..=params.high))
            .collect();
        let population = Tensor::from_data(
            TensorData::new(points, [params.population, params.dim]),
            device,
        );
        (population, state.clone())
    }

    /// Keeps the generation's best point if it beats every earlier one.
    fn tell(
        &self,
        params: &MyParams,
        population: Tensor<2>,
        fitness: Tensor<1>,
        mut state: MyState,
        _rng: &mut dyn Rng,
    ) -> (MyState, StrategyMetrics) {
        let scores: Vec<f32> = fitness
            .into_data()
            .try_into_vec::<f32>()
            .expect("fitness reads back as f32")
            .into_iter()
            .map(sanitize_fitness)
            .collect();
        let best_before = state.best.as_ref().map_or(f32::NEG_INFINITY, |(_, f)| *f);
        let mut index = 0;
        for (i, score) in scores.iter().enumerate() {
            if *score > scores[index] {
                index = i;
            }
        }
        if scores[index] > best_before {
            let point = population.slice([index..index + 1, 0..params.dim]);
            state.best = Some((point, scores[index]));
        }
        state.generation += 1;
        let metrics = StrategyMetrics::from_host_fitness(state.generation, &scores, best_before);
        (state, metrics)
    }

    fn best(&self, state: &MyState) -> Option<(Tensor<2>, f32)> {
        state.best.clone()
    }
}

#[cfg(test)]
mod tests {
    use rlevo::core::rand::SeedableRng;
    use rlevo::core::rand::rngs::StdRng;

    use super::*;

    fn params() -> MyParams {
        MyParams {
            dim: 2,
            low: -1.0,
            high: 1.0,
            population: 3,
        }
    }

    #[test]
    fn a_nan_fitness_is_never_the_best() {
        let device = Device::flex();
        let mut rng = StdRng::seed_from_u64(0);
        let state = MyStrategy.init(&params(), &mut rng, &device);
        let (population, state) = MyStrategy.ask(&params(), &state, &mut rng, &device);
        let fitness = Tensor::from_data(TensorData::new(vec![f32::NAN, 1.0, 0.5], [3]), &device);
        let (state, _) = MyStrategy.tell(&params(), population, fitness, state, &mut rng);
        let (_, best) = MyStrategy.best(&state).expect("a scored generation");
        assert_eq!(best.to_bits(), 1.0_f32.to_bits());
    }

    #[test]
    fn bounds_in_the_wrong_order_are_refused() {
        let backwards = MyParams {
            low: 1.0,
            high: -1.0,
            ..params()
        };
        assert!(backwards.validate().is_err());
    }
}
