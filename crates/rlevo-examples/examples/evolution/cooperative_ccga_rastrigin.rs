//! Cooperative co-evolution: CCGA on a separable Rastrigin function.
//!
//! The 6-dimensional Rastrigin problem is decomposed across two populations
//! of three dimensions each (Potter & De Jong 1994). Neither population holds
//! a complete solution: [`CooperativeCoEA`] completes each sub-genome with the
//! *best representative* of the other population, evaluates the assembled
//! 6-D candidate with a stateless row-wise Rastrigin [`CoupledFitness`], and
//! feeds the result back to each sub-strategy. Both sub-populations should
//! drive their assembled candidate's Rastrigin value toward the optimum (0).
//!
//! # Running
//!
//! ```text
//! cargo run -p rlevo-examples --example cooperative_ccga_rastrigin
//! ```

use std::f32::consts::PI;

use burn::backend::Flex;
use burn::tensor::{Tensor, TensorData};
use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_evolution::algorithms::ga::{
    GaConfig, GaCrossover, GaReplacement, GaSelection, GeneticAlgorithm,
};
use rlevo_evolution::coevolution::{
    CoEvolutionaryAlgorithm, CooperativeCoEA, CooperativeCoEAParams, CoupledFitness,
    RepresentativePolicy,
};

type B = Flex;

const TOTAL: usize = 6;
const HALF: usize = TOTAL / 2;
const POP: usize = 64;
const GENS: usize = 120;

/// Rastrigin value of one genome row: `10·n + Σ (x_i² − 10·cos(2π x_i))`,
/// minimized at the all-zeros vector (value 0).
fn rastrigin(x: &[f32]) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    let base = 10.0 * x.len() as f32;
    base + x
        .iter()
        .map(|&xi| xi * xi - 10.0 * (2.0 * PI * xi).cos())
        .sum::<f32>()
}

/// Stateless row-wise Rastrigin fitness. [`CooperativeCoEA`] passes
/// already-assembled full-dimensional candidates, so this never sees the
/// dimension split or representatives — it just scores each row.
///
/// The co-evolution engine is maximise-native and has no `ObjectiveSense`
/// chokepoint, so this returns the **canonical** fitness `−rastrigin` (higher
/// is better) directly; the driver negates back to the natural cost for display.
struct RastriginCoupled;

impl CoupledFitness<B> for RastriginCoupled {
    fn evaluate_coupled(&self, populations: &[Tensor<B, 2>]) -> Vec<Tensor<B, 1>> {
        debug_assert_eq!(populations.len(), 2);
        populations
            .iter()
            .map(|pop| {
                let dims = pop.dims();
                let (n, d) = (dims[0], dims[1]);
                let flat = pop.clone().into_data().into_vec::<f32>().unwrap();
                let values: Vec<f32> =
                    (0..n).map(|i| -rastrigin(&flat[i * d..i * d + d])).collect();
                Tensor::<B, 1>::from_data(TensorData::new(values, [n]), &pop.device())
            })
            .collect()
    }
}

fn ga_config() -> GaConfig {
    GaConfig {
        pop_size: POP,
        genome_dim: HALF,
        bounds: (-5.12, 5.12),
        mutation_sigma: 0.3,
        selection: GaSelection::Tournament { size: 3 },
        crossover: GaCrossover::BlxAlpha { alpha: 0.5 },
        replacement: GaReplacement::Elitist { elitism_k: 2 },
    }
}

fn main() {
    let device = Default::default();
    let algo = CooperativeCoEA::new(
        GeneticAlgorithm::<B>::new(),
        GeneticAlgorithm::<B>::new(),
        RastriginCoupled,
    );
    let params = CooperativeCoEAParams::new(
        ga_config(),
        ga_config(),
        vec![0, 1, 2],
        TOTAL,
        RepresentativePolicy::Best,
        TOTAL * POP,
    )
    .expect("valid cooperative params");
    let mut rng = StdRng::seed_from_u64(7);
    let mut state = algo.init(&params, &mut rng, &device);

    println!("CCGA on separable Rastrigin-{TOTAL} (two {HALF}-D populations, Best representative)");
    println!("(Rastrigin optimum = 0)\n");
    println!(" gen | best_a    | best_b");
    println!("-----+-----------+-----------");
    for g in 0..GENS {
        let (next, metrics) = algo.step(&params, state, &mut rng, &device);
        state = next;
        if g % 10 == 0 || g == GENS - 1 {
            // Metrics are canonical (−rastrigin); negate to show natural cost.
            println!(
                " {:>3} | {:>9.5} | {:>9.5}",
                metrics.generation, -metrics.best_fitness_a, -metrics.best_fitness_b
            );
        }
    }

    let metrics = algo.metrics(&state);
    // Canonical best is the larger value; the natural assembled cost is its
    // negation — the lower (better) of the two populations' costs.
    let best = -metrics.best_fitness_a.max(metrics.best_fitness_b);
    println!("\nfinal assembled best Rastrigin = {best:.5} (optimum 0.0)");
}
