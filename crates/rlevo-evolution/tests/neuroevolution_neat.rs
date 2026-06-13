//! Interpreted NEAT, end-to-end on XOR (spec §6 AC5/AC6).
//!
//! Drives the [`NeatStrategy`] custom harness through the manual generational
//! loop (`init → loop{ ask → GraphFitnessFn::evaluate → tell }`) on the `Flex`
//! backend and asserts:
//!
//! - **AC5** — fitness `4 − Σ(out − target)²` reaches `≥ 3.9` within `≤ 300`
//!   generations at `pop ≈ 150`.
//! - **AC6** — species count `> 1` for `> 50%` of the generations run.
//!
//! XOR is not linearly separable, so a solution requires an add-node mutation —
//! which also drives the structural divergence that creates species, so the two
//! criteria are coupled by construction.

use burn::backend::Flex;
use burn::tensor::{Tensor, TensorData};
use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_evolution::neuroevolution::phenotype::{InterpretedBuilder, PhenotypeBuilder};
use rlevo_evolution::neuroevolution::topology::TopologyGenome;
use rlevo_evolution::{GraphFitnessFn, NeatParams, NeatStrategy};

type B = Flex;

/// XOR fitness: `4 − Σ(out − target)²` over the four binary input rows
/// (maximization; the optimum is `4.0`, a perfect solver scores `≥ 3.9`).
struct XorFitness {
    inputs: Tensor<B, 2>,
    targets: [f32; 4],
}

impl XorFitness {
    // `device` is borrowed to mirror the production `&B::Device` convention; the
    // Flex test device is zero-sized, hence the targeted allow (cf. arch_nas).
    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn new(device: &<B as burn::tensor::backend::BackendTypes>::Device) -> Self {
        let inputs = Tensor::<B, 2>::from_data(
            TensorData::new(vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], [4, 2]),
            device,
        );
        Self {
            inputs,
            targets: [0.0, 1.0, 1.0, 0.0],
        }
    }
}

impl GraphFitnessFn<B> for XorFitness {
    fn evaluate(
        &self,
        population: &[TopologyGenome],
        builder: &dyn PhenotypeBuilder<B>,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Vec<f32> {
        population
            .iter()
            .map(|genome| {
                let phenotype = builder.build(genome, device);
                let out = phenotype.forward(self.inputs.clone());
                let values = out.into_data().into_vec::<f32>().unwrap();
                let sse: f32 = values
                    .iter()
                    .zip(self.targets.iter())
                    .map(|(o, t)| (o - t) * (o - t))
                    .sum();
                4.0 - sse
            })
            .collect()
    }
}

/// Tuned NEAT parameters for XOR at `pop = 150`. Structural-mutation rates are
/// raised from the R1 §6 canonical defaults (R1 notes these are the knobs worth
/// tuning) so a hidden node — and the species divergence it creates — appears
/// early and reliably within the budget.
fn xor_params(pop: usize) -> NeatParams {
    let mut params = NeatParams::default_for(pop, 2, 1);
    params.p_add_node = 0.15;
    params.p_add_connection = 0.3;
    params.p_toggle_enable = 0.03;
    params.compat_threshold = 2.5;
    params.weight_perturb_std = 0.8;
    params
}

#[test]
fn test_neat_solves_xor_with_speciation() {
    const MAX_GENERATIONS: usize = 300;
    const SOLVE_THRESHOLD: f32 = 3.9;

    let device = Default::default();
    let params = xor_params(150);
    let strat = NeatStrategy::<B>::new();
    let builder = InterpretedBuilder;
    let fitness_fn = XorFitness::new(&device);

    let mut rng = StdRng::seed_from_u64(42);
    let mut state = strat.init(&params, &mut rng, &device);

    let mut total_gens = 0usize;
    let mut multi_species_gens = 0usize;
    let mut solved = false;

    for _ in 0..MAX_GENERATIONS {
        let (population, next) = strat.ask(&params, &state, &mut rng);
        let fitness = fitness_fn.evaluate(&population, &builder, &device);
        state = strat.tell(&params, population, fitness, next, &mut rng);

        total_gens += 1;
        if state.species.len() > 1 {
            multi_species_gens += 1;
        }
        if state.best_fitness >= SOLVE_THRESHOLD {
            solved = true;
        }
        // Stop once both criteria hold, so AC6 is robust to a fast solve.
        if solved && multi_species_gens * 2 > total_gens {
            break;
        }
    }

    assert!(
        solved,
        "AC5: XOR not solved within {MAX_GENERATIONS} generations; best fitness {} < {SOLVE_THRESHOLD}",
        state.best_fitness
    );
    assert!(
        multi_species_gens * 2 > total_gens,
        "AC6: species > 1 for only {multi_species_gens}/{total_gens} generations (need > 50%)"
    );

    // The recovered champion really solves XOR through the phenotype.
    let (best_genome, best_fitness) = strat.best(&state).expect("a best genome exists");
    let recomputed = fitness_fn.evaluate(std::slice::from_ref(best_genome), &builder, &device);
    approx::assert_relative_eq!(recomputed[0], best_fitness, epsilon = 1e-4);
    assert!(recomputed[0] >= SOLVE_THRESHOLD, "champion re-evaluates as a solver");
}

#[test]
fn test_neat_run_is_reproducible_under_fixed_seed() {
    // End-to-end determinism via the public API: two identical seeded runs reach
    // the same best fitness and the same species count after a fixed budget.
    let device = Default::default();
    let params = xor_params(64);
    let strat = NeatStrategy::<B>::new();
    let builder = InterpretedBuilder;
    let fitness_fn = XorFitness::new(&device);

    let run = || {
        let mut rng = StdRng::seed_from_u64(7);
        let mut state = strat.init(&params, &mut rng, &device);
        for _ in 0..30 {
            let (population, next) = strat.ask(&params, &state, &mut rng);
            let fitness = fitness_fn.evaluate(&population, &builder, &device);
            state = strat.tell(&params, population, fitness, next, &mut rng);
        }
        (state.best_fitness, state.species.len(), state.generation)
    };

    let first = run();
    let second = run();
    approx::assert_relative_eq!(first.0, second.0, epsilon = 1e-6);
    assert_eq!(first.1, second.1, "species count is reproducible");
    assert_eq!(first.2, second.2, "generation count is reproducible");
}
