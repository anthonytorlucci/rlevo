//! Chapter 1 — Optimising a function.
//!
//! Demonstrates the three primitives that every evolutionary run in `rlevo`
//! uses: a `Landscape` (the problem), a `Strategy` (the searcher), and the
//! ask/tell loop driven by `EvolutionaryHarness`.
//!
//! Run with:
//!
//! ```text
//! cargo run -p rlevo-examples --example ch01_sphere_ga
//! ```

// ANCHOR: harness
use burn::backend::Flex;
use rlevo_environments::landscapes::sphere::Sphere;
use rlevo_evolution::algorithms::ga::{
    GaCrossover, GaReplacement, GaSelection, GaConfig, GeneticAlgorithm,
};
use rlevo_core::objective::ObjectiveSense;
use rlevo_evolution::fitness::FromLandscape;
use rlevo_evolution::strategy::EvolutionaryHarness;

type B = Flex;

const DIM: usize = 8;
const POP_SIZE: usize = 64;
const GENS: usize = 200;
const SEED: u64 = 42;
const PRINT_EVERY: usize = 25;

fn main() {
    let device: <B as burn::tensor::backend::BackendTypes>::Device = Default::default();

    let strategy = GeneticAlgorithm::<B>::new();
    let config = GaConfig {
        pop_size: POP_SIZE,
        genome_dim: DIM,
        bounds: (-5.12, 5.12),
        mutation_sigma: 0.3,
        selection: GaSelection::Tournament { size: 3 },
        crossover: GaCrossover::BlxAlpha { alpha: 0.5 },
        replacement: GaReplacement::Elitist { elitism_k: 2 },
    };
    let fitness_fn = FromLandscape::with_sense(Sphere::new(DIM), ObjectiveSense::Minimize);

    let mut harness =
        EvolutionaryHarness::<B, _, _>::new(strategy, config, fitness_fn, SEED, device, GENS);

    harness.reset();

    loop {
        let step = harness.step(());
        if let Some(m) = harness.latest_metrics()
            && (m.generation % PRINT_EVERY == 0 || step.done)
        {
            println!("gen {:>3}   best = {:.2e}", m.generation, m.best_fitness_ever);
        }
        if step.done {
            break;
        }
    }
}
// ANCHOR_END: harness

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_ga_converges() {
        let device: <B as burn::tensor::backend::BackendTypes>::Device = Default::default();

        let mut harness = EvolutionaryHarness::<B, _, _>::new(
            GeneticAlgorithm::<B>::new(),
            GaConfig {
                pop_size: POP_SIZE,
                genome_dim: DIM,
                bounds: (-5.12, 5.12),
                mutation_sigma: 0.3,
                selection: GaSelection::Tournament { size: 3 },
                crossover: GaCrossover::BlxAlpha { alpha: 0.5 },
                replacement: GaReplacement::Elitist { elitism_k: 2 },
            },
            FromLandscape::with_sense(Sphere::new(DIM), ObjectiveSense::Minimize),
            SEED,
            device,
            GENS,
        );

        harness.reset();
        loop {
            if harness.step(()).done {
                break;
            }
        }

        let best = harness.latest_metrics().unwrap().best_fitness_ever;
        assert!(
            best < 1.0,
            "GA on Sphere-D{DIM} should reach best < 1.0 after {GENS} gens; got {best}"
        );
    }
}
