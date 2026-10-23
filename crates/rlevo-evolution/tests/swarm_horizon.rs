//! The schedule horizon of GWO and the Salp swarm is a required argument,
//! and this is why.
//!
//! Both pace a coefficient over a number of generations the caller names:
//! `max_generations` on the config. A strategy cannot read the loop's budget,
//! so the two are separate numbers, and a horizon that is far longer than
//! the budget leaves the swarm exploring when the run ends. The constructors
//! once filled the horizon in with 500, which was wrong for every loop that
//! did not run 500 generations (ADR 0087).
//!
//! The test runs each strategy twice on the same seeds and the same budget:
//! once with the horizon equal to the budget, once with it far longer. The
//! objective is a Sphere whose optimum sits away from the origin, because
//! GWO is drawn to the origin whatever its schedule and would pass on an
//! unshifted one for that reason alone.

use burn::tensor::{Device, Tensor, TensorData};

use rlevo_core::config::Validate;
use rlevo_core::objective::ObjectiveSense;

use rlevo_evolution::algorithms::metaheuristic::gwo::{GreyWolfOptimizer, GwoConfig};
use rlevo_evolution::algorithms::metaheuristic::salp::{SalpConfig, SalpSwarm};
use rlevo_evolution::fitness::BatchFitnessFn;
use rlevo_evolution::strategy::{EvolutionaryLoop, Strategy};

const POP: usize = 32;
const DIM: usize = 10;
/// The loop's generation budget in every run below.
const BUDGET: usize = 120;
/// A horizon far longer than the budget. It is the value the constructors
/// used to fill in.
const LONG_HORIZON: usize = 500;
/// Where the optimum sits in every dimension, inside the default bounds of
/// `[-5.12, 5.12]`.
const SHIFT: f32 = 1.7;

/// `sum((x - SHIFT)^2)`, a cost.
struct ShiftedSphere;

impl BatchFitnessFn<Tensor<2>> for ShiftedSphere {
    fn evaluate_batch(&mut self, population: &Tensor<2>, device: &Device) -> Tensor<1> {
        let genes = population
            .clone()
            .into_data()
            .try_into_vec::<f32>()
            .expect("the population tensor holds f32");
        let cost: Vec<f32> = genes
            .chunks(DIM)
            .map(|row| row.iter().map(|x| (x - SHIFT).powi(2)).sum())
            .collect();
        let n = cost.len();
        Tensor::from_data(TensorData::new(cost, [n]), device)
    }

    fn sense(&self) -> ObjectiveSense {
        ObjectiveSense::Minimize
    }
}

/// Best cost after `BUDGET` generations.
fn best_cost<S>(strategy: S, params: S::Params, seed: u64) -> f32
where
    S: Strategy<Genome = Tensor<2>>,
    S::Params: Validate,
{
    let mut evo_loop = EvolutionaryLoop::new(
        strategy,
        params,
        ShiftedSphere,
        seed,
        Device::flex(),
        BUDGET,
    )
    .expect("valid params");
    evo_loop.reset();
    while !evo_loop.is_done() {
        evo_loop.step();
    }
    evo_loop
        .latest_metrics()
        .expect("a loop that has stepped has metrics")
        .best_fitness_ever()
}

// Measured over twelve seeds on `Flex`. GWO: at most 6.9e-4 with the horizon
// at the budget, at least 0.36 with it at 500. The Salp swarm: at most
// 6.1e-6, and at least 0.69. Each bar below leaves an order of magnitude or
// more on its side.

#[test]
fn gwo_converges_when_its_horizon_is_the_budget() {
    for seed in 0..3 {
        let matched = best_cost(
            GreyWolfOptimizer::new(),
            GwoConfig::default_for(POP, DIM, BUDGET),
            seed,
        );
        let long = best_cost(
            GreyWolfOptimizer::new(),
            GwoConfig::default_for(POP, DIM, LONG_HORIZON),
            seed,
        );

        assert!(
            matched < 1e-2,
            "seed {seed}: with the horizon at the budget the pack converges; best cost {matched}"
        );
        assert!(
            long > 1e-1,
            "seed {seed}: with the horizon at {LONG_HORIZON} the pack is still \
             exploring after {BUDGET} generations; best cost {long}"
        );
    }
}

#[test]
fn the_salp_swarm_converges_when_its_horizon_is_the_budget() {
    for seed in 0..3 {
        let matched = best_cost(
            SalpSwarm::new(),
            SalpConfig::default_for(POP, DIM, BUDGET),
            seed,
        );
        let long = best_cost(
            SalpSwarm::new(),
            SalpConfig::default_for(POP, DIM, LONG_HORIZON),
            seed,
        );

        assert!(
            matched < 1e-3,
            "seed {seed}: with the horizon at the budget the swarm converges; best cost {matched}"
        );
        assert!(
            long > 1e-1,
            "seed {seed}: with the horizon at {LONG_HORIZON} the leaders are \
             still thrown wide after {BUDGET} generations; best cost {long}"
        );
    }
}
