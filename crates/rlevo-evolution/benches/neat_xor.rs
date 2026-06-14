//! NEAT generation + population-evaluation micro-benchmarks on the `Flex`
//! backend (spec §6 AC7 / 3d3 §6 AC5).
//!
//! Two groups:
//!
//! - `neat_xor` — one full NEAT generation (`ask` → evaluate → `tell`) at a
//!   **fixed** `pop_size = 64`, comparing the interpreted per-genome evaluation
//!   (`generation_pop64`, the ≈ 590 µs reference) against the dense-padded
//!   batched evaluation (`generation_batched_pop64`). At XOR scale (`N ≈ 5–10`)
//!   the two run at **par** (~580 µs vs ~590 µs): the workload is too small for
//!   batching to pay off, and the per-generation cost is dominated by host-side
//!   reproduction, not evaluation. This run measures, it does not threshold.
//! - `neat_eval_scale` — the forward-pass hotspot only (the part that is
//!   tensorized), interpreted loop vs. dense-batched, on a synthetic wide
//!   feedforward population at `P = 256` and growing hidden width. The batched
//!   path wins and the margin grows with width (measured ~1.7× at `h = 10`,
//!   ~1.9× at `h = 100` on Flex/CPU) — the asymptotic tensorization benefit. The
//!   win hinges on the **depth-bounded** iteration count (these genomes are
//!   shallow); a naive static `N − 1` bound erases it and then some.
//!
//! Run with `cargo bench -p rlevo-evolution --bench neat_xor`.

use std::hint::black_box;

use burn::backend::Flex;
use burn::tensor::{Tensor, TensorData};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use rlevo_evolution::neuroevolution::phenotype::{
    BatchPhenotypeEvaluator, DensePaddedEvaluator, InterpretedBuilder, PhenotypeBuilder,
};
use rlevo_evolution::neuroevolution::topology::{
    ActivationFn, ConnectionGene, NodeGene, NodeKind, TopologyGenome,
};
use rlevo_evolution::{BatchGraphFitness, GraphFitnessFn, NeatParams, NeatStrategy};

type B = Flex;

/// XOR fitness `4 − Σ(out − target)²` over the four binary input rows.
struct XorFitness {
    inputs: Tensor<B, 2>,
    targets: [f32; 4],
}

impl XorFitness {
    // `device` is borrowed to mirror the production `&B::Device` convention; the
    // Flex device is zero-sized, hence the targeted allow (cf. arch_nas).
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

fn xor_params() -> NeatParams {
    let mut params = NeatParams::default_for(64, 2, 1);
    params.p_add_node = 0.15;
    params.p_add_connection = 0.3;
    params.compat_threshold = 2.5;
    params
}

fn neat_xor_generation(c: &mut Criterion) {
    let device = Default::default();
    let params = xor_params();
    let strat = NeatStrategy::<B>::new();
    let builder = InterpretedBuilder;
    let fitness_fn = XorFitness::new(&device);
    // Same XOR objective, but scored over the dense-batched evaluator's output.
    let batched_fitness = BatchGraphFitness::new(
        DensePaddedEvaluator::default(),
        Tensor::<B, 2>::from_data(
            TensorData::new(vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], [4, 2]),
            &device,
        ),
        |slab: &[f32]| {
            let targets = [0.0f32, 1.0, 1.0, 0.0];
            let sse: f32 = slab
                .iter()
                .zip(targets.iter())
                .map(|(o, t)| (o - t) * (o - t))
                .sum();
            4.0 - sse
        },
    );

    // Warm the state to a representative mid-run topology (some hidden nodes,
    // several species) so the measured generation reflects real per-generation
    // cost rather than the trivial minimal-seed first step.
    let mut rng = StdRng::seed_from_u64(0);
    let mut state = strat.init(&params, &mut rng, &device);
    for _ in 0..15 {
        let (population, next) = strat.ask(&params, &state, &mut rng);
        let fitness = fitness_fn.evaluate(&population, &builder, &device);
        state = strat.tell(&params, population, fitness, next, &mut rng);
    }
    let warm_state = state;

    let mut group = c.benchmark_group("neat_xor");
    group.sample_size(10);
    group.bench_function("generation_pop64", |b| {
        b.iter_batched(
            || (warm_state.clone(), StdRng::seed_from_u64(123)),
            |(state, mut rng)| {
                let (population, next) = strat.ask(&params, &state, &mut rng);
                let fitness = fitness_fn.evaluate(&population, &builder, &device);
                let next = strat.tell(&params, population, fitness, next, &mut rng);
                black_box(next.best_fitness)
            },
            BatchSize::SmallInput,
        );
    });
    group.bench_function("generation_batched_pop64", |b| {
        b.iter_batched(
            || (warm_state.clone(), StdRng::seed_from_u64(123)),
            |(state, mut rng)| {
                let (population, next) = strat.ask(&params, &state, &mut rng);
                let fitness = batched_fitness.evaluate(&population, &builder, &device);
                let next = strat.tell(&params, population, fitness, next, &mut rng);
                black_box(next.best_fitness)
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// A wide single-hidden-layer feedforward genome: `2` inputs → `num_hidden`
/// sigmoid hidden nodes → `1` linear output, fully connected, random weights.
fn wide_genome(num_hidden: usize, rng: &mut StdRng) -> TopologyGenome {
    let mut nodes = vec![
        NodeGene { id: 0, kind: NodeKind::Input, activation: ActivationFn::Linear, bias: 0.0 },
        NodeGene { id: 1, kind: NodeKind::Input, activation: ActivationFn::Linear, bias: 0.0 },
        NodeGene { id: 2, kind: NodeKind::Output, activation: ActivationFn::Linear, bias: 0.0 },
    ];
    let mut conns = Vec::new();
    let mut innovation = 0u64;
    let weight = |rng: &mut StdRng| rng.random::<f32>() - 0.5;
    for h in 0..num_hidden {
        let id = 3 + h as u64;
        nodes.push(NodeGene { id, kind: NodeKind::Hidden, activation: ActivationFn::Sigmoid, bias: 0.0 });
        for source in [0u64, 1] {
            conns.push(ConnectionGene { innovation, source, target: id, weight: weight(rng), enabled: true });
            innovation += 1;
        }
        conns.push(ConnectionGene { innovation, source: id, target: 2, weight: weight(rng), enabled: true });
        innovation += 1;
    }
    TopologyGenome::new(nodes, conns)
}

/// Forward-pass-only comparison (the tensorized hotspot): interpreted per-genome
/// loop vs. one dense-batched pass, at `P = 256` and growing hidden width.
fn neat_eval_scale(c: &mut Criterion) {
    let device = Default::default();
    let builder = InterpretedBuilder;
    let evaluator = DensePaddedEvaluator::default();
    // A 16-row observation batch gives the device a non-trivial amount of work.
    let obs_rows: Vec<f32> = [0.0f32, 0.5, 0.25].iter().cycle().take(32).copied().collect();
    let obs = Tensor::<B, 2>::from_data(TensorData::new(obs_rows, [16, 2]), &device);

    let mut group = c.benchmark_group("neat_eval_scale");
    group.sample_size(10);
    for &num_hidden in &[10usize, 50, 100] {
        let mut rng = StdRng::seed_from_u64(7);
        let pop: Vec<TopologyGenome> = (0..256).map(|_| wide_genome(num_hidden, &mut rng)).collect();

        group.bench_function(format!("interpreted_p256_h{num_hidden}"), |b| {
            b.iter(|| {
                let mut acc = 0.0f32;
                for genome in &pop {
                    let net = PhenotypeBuilder::<B>::build(&builder, genome, &device);
                    let out = net.forward(obs.clone());
                    acc += out.into_data().into_vec::<f32>().unwrap().iter().sum::<f32>();
                }
                black_box(acc)
            });
        });
        group.bench_function(format!("batched_p256_h{num_hidden}"), |b| {
            b.iter(|| {
                let out = evaluator.evaluate_population(&pop, obs.clone(), &device);
                black_box(out.into_data().into_vec::<f32>().unwrap().iter().sum::<f32>())
            });
        });
    }
    group.finish();
}

criterion_group!(benches, neat_xor_generation, neat_eval_scale);
criterion_main!(benches);
