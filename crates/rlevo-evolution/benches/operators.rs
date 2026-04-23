//! Baseline micro-benchmarks for hot-path operators on the ndarray
//! backend.
//!
//! The numbers these benches produce are the reference point for the
//! custom CubeCL kernel work scoped in `ops/kernels/mod.rs`. Kernels
//! should strictly beat the pure-tensor baseline at `pop_size ≥ 256`.
//!
//! Run with `cargo bench -p evorl-evolution`. Pass
//! `--save-baseline pre-kernel` before landing kernels and
//! `--baseline pre-kernel` afterwards to get a side-by-side report.

use burn::backend::NdArray;
use burn::tensor::{Distribution, Tensor, backend::Backend as _};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::SeedableRng;
use rand::rngs::StdRng;

use rlevo_evolution::algorithms::de::{DeConfig, DeVariant, DifferentialEvolution};
use rlevo_evolution::fitness::BatchFitnessFn;
use rlevo_evolution::ops::selection::tournament_select;
use rlevo_evolution::strategy::{EvolutionaryHarness, Strategy};

use rlevo_benchmarks::env::BenchEnv;

type B = NdArray;

struct ZeroFitness;
impl BatchFitnessFn<B, Tensor<B, 2>> for ZeroFitness {
    fn evaluate_batch(
        &mut self,
        population: &Tensor<B, 2>,
        device: &<B as burn::tensor::backend::Backend>::Device,
    ) -> Tensor<B, 1> {
        let n = population.shape().dims[0];
        Tensor::<B, 1>::zeros([n], device)
    }
}

fn bench_tournament(c: &mut Criterion) {
    let mut group = c.benchmark_group("tournament_select");
    let device = Default::default();
    B::seed(&device, 7);
    for &pop_size in &[64_usize, 256, 1024] {
        let dim = 10;
        let fitness: Vec<f32> = (0..pop_size).map(|i| (i as f32) * 0.1).collect();
        let population =
            Tensor::<B, 2>::random([pop_size, dim], Distribution::Uniform(-1.0, 1.0), &device);
        let mut rng = StdRng::seed_from_u64(1);
        group.bench_with_input(BenchmarkId::from_parameter(pop_size), &pop_size, |b, &n| {
            b.iter(|| tournament_select::<B>(&population, &fitness, 2, n, &mut rng, &device));
        });
    }
    group.finish();
}

fn bench_de_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("de_one_generation");
    group.sample_size(10);
    for &pop_size in &[64_usize, 256, 1024] {
        let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
        B::seed(&device, 42);
        let mut params = DeConfig::default_for(pop_size, 10);
        params.variant = DeVariant::Rand1Bin;
        group.bench_with_input(BenchmarkId::from_parameter(pop_size), &pop_size, |b, _n| {
            b.iter_batched(
                || {
                    // Fresh harness per iteration so the bench
                    // measures a full `ask → evaluate → tell`
                    // cycle on a warm population (no one-time init).
                    let strategy = DifferentialEvolution::<B>::new();
                    let mut harness = EvolutionaryHarness::<B, _, _>::new(
                        strategy,
                        params.clone(),
                        ZeroFitness,
                        11,
                        device.clone(),
                        1_000,
                    );
                    harness.reset();
                    // Warm up: run one generation so init costs are
                    // outside the measurement window.
                    let _ = harness.step(());
                    harness
                },
                |mut harness| {
                    let _ = harness.step(());
                    harness
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_tournament, bench_de_generation);
criterion_main!(benches);

// Ensure the `Strategy` trait lookup remains monomorphizable from this
// crate-bench.
#[allow(dead_code)]
fn _static_asserts<B: burn::tensor::backend::Backend>()
where
    DifferentialEvolution<B>: Strategy<B>,
    B::Device: Clone,
{
}
