//! Interpreted NEAT, end-to-end on XOR.
//!
//! Drives the [`NeatStrategy`] custom harness through the manual generational
//! loop (`init → loop{ ask → GraphFitnessFn::evaluate → tell }`) on the `Flex`
//! backend and asserts:
//!
//! - **Solve** — fitness `4 − Σ(out − target)²` reaches `≥ 3.9` within `≤ 300`
//!   generations at `pop ≈ 150`.
//! - **Speciation** — species count `> 1` for `> 50%` of the generations run.
//!
//! XOR is not linearly separable, so a solution requires an add-node mutation —
//! which also drives the structural divergence that creates species, so the two
//! criteria are coupled by construction.

use burn::backend::Flex;
use burn::tensor::{Tensor, TensorData};
use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_evolution::neuroevolution::phenotype::{
    BatchPhenotypeEvaluator, DensePaddedEvaluator, InterpretedBuilder, InterpretedPhenotype,
    Phenotype, PhenotypeBuilder,
};
use rlevo_evolution::neuroevolution::topology::{
    ActivationFn, ConnectionGene, InnovationId, NodeGene, NodeId, NodeKind, TopologyGenome,
};
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
/// raised from the canonical NEAT defaults (Stanley & Miikkulainen, 2002 — the
/// structural-mutation rates and compatibility threshold are the knobs worth
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
        // Stop once both criteria hold, so the speciation check is robust to a
        // fast solve (a short run could otherwise miss the >50% threshold).
        if solved && multi_species_gens * 2 > total_gens {
            break;
        }
    }

    assert!(
        solved,
        "solve: XOR not solved within {MAX_GENERATIONS} generations; best fitness {} < {SOLVE_THRESHOLD}",
        state.best_fitness
    );
    assert!(
        multi_species_gens * 2 > total_gens,
        "speciation: species > 1 for only {multi_species_gens}/{total_gens} generations (need > 50%)"
    );

    // The recovered champion really solves XOR through the phenotype.
    let (best_genome, best_fitness) = strat.best(&state).expect("a best genome exists");
    let recomputed = fitness_fn.evaluate(std::slice::from_ref(best_genome), &builder, &device);
    approx::assert_relative_eq!(recomputed[0], best_fitness, epsilon = 1e-4);
    assert!(
        recomputed[0] >= SOLVE_THRESHOLD,
        "champion re-evaluates as a solver"
    );
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

// ---------------------------------------------------------------------------
// Tensorized / dense-padded parity
// ---------------------------------------------------------------------------

/// A small fixed parity population: every genome has 2 inputs (ids 0, 1) and 1
/// output (id 2) — the constant I/O the batched evaluator requires — while the
/// hidden structure, depth, activations, and a disabled edge vary across genomes
/// so the dense path is exercised on all four [`ActivationFn`] variants, padding
/// (node counts 3..=5, so `N = 5`), a two-hidden-layer chain, and edge disabling.
fn parity_population() -> Vec<TopologyGenome> {
    let input = |id: u64| NodeGene {
        id: NodeId::new(id),
        kind: NodeKind::Input,
        activation: ActivationFn::Linear,
        bias: 0.0,
    };

    // G1 — sigmoid output, no hidden.
    let g1 = TopologyGenome::new(
        vec![
            input(0),
            input(1),
            NodeGene {
                id: NodeId::new(2),
                kind: NodeKind::Output,
                activation: ActivationFn::Sigmoid,
                bias: 0.2,
            },
        ],
        vec![
            ConnectionGene {
                innovation: InnovationId::new(0),
                source: NodeId::new(0),
                target: NodeId::new(2),
                weight: 0.7,
                enabled: true,
            },
            ConnectionGene {
                innovation: InnovationId::new(1),
                source: NodeId::new(1),
                target: NodeId::new(2),
                weight: -0.4,
                enabled: true,
            },
        ],
    );

    // G2 — tanh hidden → linear output, plus a DISABLED direct edge 0→2.
    let g2 = TopologyGenome::new(
        vec![
            input(0),
            input(1),
            NodeGene {
                id: NodeId::new(2),
                kind: NodeKind::Output,
                activation: ActivationFn::Linear,
                bias: 0.0,
            },
            NodeGene {
                id: NodeId::new(3),
                kind: NodeKind::Hidden,
                activation: ActivationFn::Tanh,
                bias: 0.1,
            },
        ],
        vec![
            ConnectionGene {
                innovation: InnovationId::new(0),
                source: NodeId::new(0),
                target: NodeId::new(2),
                weight: 5.0,
                enabled: false,
            },
            ConnectionGene {
                innovation: InnovationId::new(1),
                source: NodeId::new(0),
                target: NodeId::new(3),
                weight: 0.9,
                enabled: true,
            },
            ConnectionGene {
                innovation: InnovationId::new(2),
                source: NodeId::new(1),
                target: NodeId::new(3),
                weight: -1.2,
                enabled: true,
            },
            ConnectionGene {
                innovation: InnovationId::new(3),
                source: NodeId::new(3),
                target: NodeId::new(2),
                weight: 1.5,
                enabled: true,
            },
        ],
    );

    // G3 — two hidden layers: relu (3) → sigmoid (4) → linear output (longest
    // path 3 edges, so it needs the multi-iteration propagation to settle).
    let g3 = TopologyGenome::new(
        vec![
            input(0),
            input(1),
            NodeGene {
                id: NodeId::new(2),
                kind: NodeKind::Output,
                activation: ActivationFn::Linear,
                bias: -0.3,
            },
            NodeGene {
                id: NodeId::new(3),
                kind: NodeKind::Hidden,
                activation: ActivationFn::Relu,
                bias: 0.0,
            },
            NodeGene {
                id: NodeId::new(4),
                kind: NodeKind::Hidden,
                activation: ActivationFn::Sigmoid,
                bias: 0.05,
            },
        ],
        vec![
            ConnectionGene {
                innovation: InnovationId::new(0),
                source: NodeId::new(0),
                target: NodeId::new(3),
                weight: 1.1,
                enabled: true,
            },
            ConnectionGene {
                innovation: InnovationId::new(1),
                source: NodeId::new(1),
                target: NodeId::new(3),
                weight: 0.6,
                enabled: true,
            },
            ConnectionGene {
                innovation: InnovationId::new(2),
                source: NodeId::new(3),
                target: NodeId::new(4),
                weight: -0.8,
                enabled: true,
            },
            ConnectionGene {
                innovation: InnovationId::new(3),
                source: NodeId::new(4),
                target: NodeId::new(2),
                weight: 2.0,
                enabled: true,
            },
        ],
    );

    // G4 — linear output, no hidden, non-zero output bias.
    let g4 = TopologyGenome::new(
        vec![
            input(0),
            input(1),
            NodeGene {
                id: NodeId::new(2),
                kind: NodeKind::Output,
                activation: ActivationFn::Linear,
                bias: 0.5,
            },
        ],
        vec![
            ConnectionGene {
                innovation: InnovationId::new(0),
                source: NodeId::new(0),
                target: NodeId::new(2),
                weight: 1.0,
                enabled: true,
            },
            ConnectionGene {
                innovation: InnovationId::new(1),
                source: NodeId::new(1),
                target: NodeId::new(2),
                weight: 1.0,
                enabled: true,
            },
        ],
    );

    vec![g1, g2, g3, g4]
}

/// The four binary XOR input rows as a `[4, 2]` observation batch.
// `device` is borrowed to mirror the production `&B::Device` convention; the
// Flex test device is zero-sized, hence the targeted allow (cf. `XorFitness`).
#[allow(clippy::trivially_copy_pass_by_ref)]
fn parity_obs(device: &<B as burn::tensor::backend::BackendTypes>::Device) -> Tensor<B, 2> {
    Tensor::<B, 2>::from_data(
        TensorData::new(vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], [4, 2]),
        device,
    )
}

/// Ground truth: run the interpreted phenotype per genome and concatenate the
/// flattened `[batch, num_outputs]` outputs in population order (matching the
/// dense evaluator's `[pop, batch, action]` row-major layout).
fn interpreted_stacked(genomes: &[TopologyGenome], obs: &Tensor<B, 2>) -> Vec<f32> {
    let mut out: Vec<f32> = Vec::new();
    for genome in genomes {
        let pheno = InterpretedPhenotype::<B>::new(genome);
        let row = pheno
            .forward(obs.clone())
            .into_data()
            .into_vec::<f32>()
            .unwrap();
        out.extend(row);
    }
    out
}

/// The dense-padded batched forward pass reproduces the interpreted
/// per-genome output within float epsilon, across all four activations, a
/// two-hidden-layer genome, a disabled edge, and padding.
#[test]
fn test_dense_padded_matches_interpreted_population() {
    let device = Default::default();
    let genomes = parity_population();
    let obs = parity_obs(&device);

    let expected = interpreted_stacked(&genomes, &obs);
    let got = DensePaddedEvaluator::default()
        .evaluate_population(&genomes, obs, &device)
        .into_data()
        .into_vec::<f32>()
        .unwrap();

    assert_eq!(
        got.len(),
        expected.len(),
        "one scalar per (genome, row, output)"
    );
    for (g, e) in got.iter().zip(expected.iter()) {
        approx::assert_relative_eq!(*g, *e, epsilon = 1e-4);
    }
}

/// A single-genome population evaluates identically to the interpreted path.
#[test]
fn test_dense_padded_matches_interpreted_single_genome() {
    let device = Default::default();
    let genomes = vec![parity_population().remove(2)]; // the deep two-hidden genome
    let obs = parity_obs(&device);

    let expected = interpreted_stacked(&genomes, &obs);
    let got = DensePaddedEvaluator::default()
        .evaluate_population(&genomes, obs, &device)
        .into_data()
        .into_vec::<f32>()
        .unwrap();

    for (g, e) in got.iter().zip(expected.iter()) {
        approx::assert_relative_eq!(*g, *e, epsilon = 1e-4);
    }
}

/// The node-budget ceiling binds exactly at the largest genome's node count.
#[test]
fn test_dense_padded_cap_boundary_is_inclusive() {
    let device = Default::default();
    let genomes = parity_population(); // largest genome has 5 nodes
    let obs = parity_obs(&device);
    // cap == max node count must succeed (inclusive bound).
    let got = DensePaddedEvaluator::new(5)
        .evaluate_population(&genomes, obs, &device)
        .into_data()
        .into_vec::<f32>()
        .unwrap();
    let expected = interpreted_stacked(&genomes, &parity_obs(&device));
    for (g, e) in got.iter().zip(expected.iter()) {
        approx::assert_relative_eq!(*g, *e, epsilon = 1e-4);
    }
}

/// A population that overflows the node-budget ceiling panics rather than
/// silently allocating an oversized weight tensor.
#[test]
#[should_panic(expected = "exceeding max_nodes_cap")]
fn test_dense_padded_panics_over_cap() {
    let device = Default::default();
    let genomes = parity_population(); // largest genome has 5 nodes
    let obs = parity_obs(&device);
    let _ = DensePaddedEvaluator::new(4).evaluate_population(&genomes, obs, &device);
}

/// The `BatchGraphFitness` adapter drives `NeatStrategy` to the same XOR fitness
/// the interpreted `GraphFitnessFn` produces, confirming the two evaluation
/// seams are interchangeable behind the harness.
#[test]
fn test_batch_graph_fitness_matches_interpreted_fitness() {
    use rlevo_evolution::BatchGraphFitness;

    let device = Default::default();
    let genomes = parity_population();
    let obs = parity_obs(&device);
    let targets = [0.0f32, 1.0, 1.0, 0.0];

    // Interpreted per-genome fitness: 4 − Σ(out − target)².
    let builder = InterpretedBuilder;
    let interpreted_fitness = XorFitness::new(&device).evaluate(&genomes, &builder, &device);

    // The same scoring, but reduced over the batched evaluator's output slab.
    let batched = BatchGraphFitness::new(
        DensePaddedEvaluator::default(),
        obs,
        move |slab: &[f32]| {
            let sse: f32 = slab
                .iter()
                .zip(targets.iter())
                .map(|(o, t)| (o - t) * (o - t))
                .sum();
            4.0 - sse
        },
    );
    let batched_fitness = batched.evaluate(&genomes, &builder, &device);

    assert_eq!(batched_fitness.len(), interpreted_fitness.len());
    for (b, i) in batched_fitness.iter().zip(interpreted_fitness.iter()) {
        approx::assert_relative_eq!(*b, *i, epsilon = 1e-4);
    }
}
