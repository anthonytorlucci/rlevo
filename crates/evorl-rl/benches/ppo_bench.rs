//! Criterion micro-benchmarks for the PPO building blocks that matter for
//! per-iteration throughput: GAE, advantage normalisation, and the
//! clipped surrogate objective.

use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, black_box};
use evorl_rl::algorithms::ppo::losses::{clipped_surrogate, normalize_advantages};
use evorl_rl::algorithms::ppo::rollout::compute_gae;

type B = NdArray;

fn synthetic_trajectory(n: usize) -> (Vec<f32>, Vec<f32>, Vec<bool>, Vec<bool>) {
    let mut rewards = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);
    let terminated = vec![false; n];
    let truncated = vec![false; n];
    for i in 0..n {
        rewards.push((i as f32 % 7.0) * 0.3 - 1.0);
        values.push(((i as f32) * 0.05).sin() * 0.5);
    }
    (rewards, values, terminated, truncated)
}

fn bench_compute_gae(c: &mut Criterion) {
    let mut group = c.benchmark_group("ppo_compute_gae");
    for &n in &[128_usize, 512, 2048] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let (r, v, t, tr) = synthetic_trajectory(n);
            b.iter(|| {
                let (advs, rets) = compute_gae(
                    black_box(&r),
                    black_box(&v),
                    black_box(&t),
                    black_box(&tr),
                    black_box(0.0),
                    black_box(false),
                    0.99,
                    0.95,
                );
                black_box((advs, rets));
            });
        });
    }
    group.finish();
}

fn bench_advantage_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("ppo_advantage_norm");
    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
    for &n in &[64_usize, 256, 1024] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 5.0).collect();
            let t: Tensor<B, 1> = Tensor::from_data(TensorData::new(data, vec![n]), &device);
            b.iter(|| {
                let out = normalize_advantages(black_box(t.clone()));
                black_box(out);
            });
        });
    }
    group.finish();
}

fn bench_clipped_surrogate(c: &mut Criterion) {
    let mut group = c.benchmark_group("ppo_clipped_surrogate");
    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
    for &n in &[64_usize, 256, 1024] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let new_lp: Tensor<B, 1> = Tensor::from_data(
                TensorData::new(vec![0.0_f32; n], vec![n]),
                &device,
            );
            let old_lp: Tensor<B, 1> = Tensor::from_data(
                TensorData::new(vec![0.05_f32; n], vec![n]),
                &device,
            );
            let advs: Tensor<B, 1> = Tensor::from_data(
                TensorData::new(vec![1.0_f32; n], vec![n]),
                &device,
            );
            b.iter(|| {
                let out = clipped_surrogate(
                    black_box(new_lp.clone()),
                    black_box(old_lp.clone()),
                    black_box(advs.clone()),
                    0.2,
                );
                black_box(out);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_compute_gae, bench_advantage_norm, bench_clipped_surrogate);
criterion_main!(benches);
