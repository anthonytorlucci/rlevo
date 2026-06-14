//! Interpreted-NEAT baseline micro-benchmark on XOR (spec §6 AC7).
//!
//! Measures the wall-clock cost of one full NEAT generation (`ask` → evaluate →
//! `tell`) on the `Flex` backend at a **fixed** `pop_size = 64` (this run does
//! not need to solve XOR — it is a measurement, not a threshold). This is the
//! reference point the tensorized / GPU-batched path (#41) measures its speedup
//! against.
//!
//! Run with `cargo bench -p rlevo-evolution --bench neat_xor`.

use std::hint::black_box;

use burn::backend::Flex;
use burn::tensor::{Tensor, TensorData};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_evolution::neuroevolution::phenotype::{InterpretedBuilder, PhenotypeBuilder};
use rlevo_evolution::neuroevolution::topology::TopologyGenome;
use rlevo_evolution::{GraphFitnessFn, NeatParams, NeatStrategy};

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
    group.finish();
}

criterion_group!(benches, neat_xor_generation);
criterion_main!(benches);
