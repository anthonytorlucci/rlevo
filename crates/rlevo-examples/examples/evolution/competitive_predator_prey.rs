//! Competitive co-evolution: a predator–prey arms race on a separable quadratic.
//!
//! Two populations of points in `R^d` co-evolve under [`CompetitiveCoEA`]. The
//! **predator** (population A) minimizes its squared distance to the prey
//! (it wants to close in); the **prey** (population B) minimizes its closeness
//! to the predator (it wants to escape). The two objectives are coupled and
//! opposed — a textbook arms race. Each generation prints the best fitness of
//! each population (lower is better for both, under their own objective).
//!
//! # Running
//!
//! ```text
//! cargo run -p rlevo-examples --example competitive_predator_prey
//! ```

#![allow(clippy::cast_precision_loss)]

use burn::backend::Flex;
use burn::tensor::{Tensor, TensorData};
use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_core::bounds::Bounds;
use rlevo_core::rate::NonNegativeRate;
use rlevo_evolution::algorithms::ga::{
    GaConfig, GaCrossover, GaReplacement, GaSelection, GeneticAlgorithm,
};
use rlevo_evolution::coevolution::{
    CoEvolutionaryAlgorithm, CompetitiveCoEA, CompetitiveCoEAParams, CoupledFitness,
};

type B = Flex;

const DIM: usize = 2;
const POP: usize = 48;
const GENS: usize = 60;

fn rows(pop: &Tensor<B, 2>) -> Vec<Vec<f32>> {
    let dims = pop.dims();
    let (n, d) = (dims[0], dims[1]);
    let flat = pop.clone().into_data().into_vec::<f32>().unwrap();
    (0..n).map(|i| flat[i * d..i * d + d].to_vec()).collect()
}

fn sqdist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Predator (pop 0) minimizes mean squared distance to prey; prey (pop 1)
/// minimizes mean closeness `exp(-dist^2)` to the predator.
///
/// The co-evolution engine is maximise-native with no `ObjectiveSense`
/// chokepoint, so both objectives are returned in **canonical** form (their
/// natural costs negated, higher = better); the driver negates back for display.
struct PredatorPrey;

impl CoupledFitness<B> for PredatorPrey {
    fn evaluate_coupled(&self, populations: &[Tensor<B, 2>]) -> Vec<Tensor<B, 1>> {
        debug_assert_eq!(populations.len(), 2);
        let device = populations[0].device();
        let a = rows(&populations[0]);
        let b = rows(&populations[1]);

        let predator: Vec<f32> = a
            .iter()
            .map(|ai| -(b.iter().map(|bj| sqdist(ai, bj)).sum::<f32>() / b.len().max(1) as f32))
            .collect();
        let prey: Vec<f32> = b
            .iter()
            .map(|bj| {
                -(a.iter().map(|ai| (-sqdist(bj, ai)).exp()).sum::<f32>() / a.len().max(1) as f32)
            })
            .collect();

        vec![
            Tensor::<B, 1>::from_data(TensorData::new(predator.clone(), [predator.len()]), &device),
            Tensor::<B, 1>::from_data(TensorData::new(prey.clone(), [prey.len()]), &device),
        ]
    }
}

fn ga_config() -> GaConfig {
    GaConfig {
        pop_size: POP,
        genome_dim: DIM,
        bounds: Bounds::new(-5.0, 5.0),
        mutation_sigma: NonNegativeRate::new(0.2),
        selection: GaSelection::Tournament { size: 3 },
        crossover: GaCrossover::BlxAlpha {
            alpha: NonNegativeRate::new(0.5),
        },
        replacement: GaReplacement::Elitist { elitism_k: 1 },
    }
}

fn main() {
    let device = Default::default();
    let algo = CompetitiveCoEA::new(
        GeneticAlgorithm::<B>::new(),
        GeneticAlgorithm::<B>::new(),
        PredatorPrey,
    );
    let params = CompetitiveCoEAParams {
        params_a: ga_config(),
        params_b: ga_config(),
    };
    let mut rng = StdRng::seed_from_u64(7);
    let mut state = algo.init(&params, &mut rng, &device);

    println!("Predator–prey arms race on a separable quadratic (R^{DIM})");
    println!("(lower is better for each population's own objective)\n");
    println!(" gen | predator best (mean dist^2) | prey best (mean closeness)");
    println!("-----+-----------------------------+---------------------------");
    for g in 0..GENS {
        let (next, metrics) = algo.step(&params, state, &mut rng, &device);
        state = next;
        if g % 5 == 0 || g == GENS - 1 {
            // Metrics are canonical (negated cost); negate to show natural cost.
            println!(
                " {:>3} | {:>27.4} | {:>25.4}",
                metrics.generation, -metrics.best_fitness_a, -metrics.best_fitness_b
            );
        }
    }
}
