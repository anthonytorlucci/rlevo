//! Criterion micro-benchmarks for the PPG-specific building blocks.
//!
//! PPG inherits all of PPO's hot paths (GAE, clipped surrogate, advantage
//! normalisation) so those are covered by `ppo_bench`. What's left:
//!
//! - `policy_kl_categorical` — the distillation KL kernel called once per
//!   aux-phase minibatch.
//! - `aux_value_loss_path` — the aux-phase value-target MSE (same kernel as
//!   PPO's unclipped value loss, benchmarked here at aux-phase-shaped sizes).

use burn::backend::NdArray;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use evorl_rl::algorithms::ppg::losses::policy_kl_categorical;
use evorl_rl::algorithms::ppo::losses::unclipped_value_loss;

type B = NdArray;

fn bench_policy_kl_categorical(c: &mut Criterion) {
    let mut group = c.benchmark_group("ppg_policy_kl_categorical");
    let device: <B as Backend>::Device = Default::default();
    for &(batch, num_actions) in &[(64_usize, 2_usize), (256, 2), (1024, 2), (256, 18)] {
        let id = format!("batch{batch}_actions{num_actions}");
        group.bench_with_input(
            BenchmarkId::from_parameter(&id),
            &(batch, num_actions),
            |b, &(n, a)| {
                let old: Tensor<B, 2> =
                    Tensor::from_data(TensorData::new(vec![0.1_f32; n * a], vec![n, a]), &device);
                let new: Tensor<B, 2> =
                    Tensor::from_data(TensorData::new(vec![0.0_f32; n * a], vec![n, a]), &device);
                b.iter(|| {
                    let out = policy_kl_categorical(black_box(old.clone()), black_box(new.clone()));
                    black_box(out);
                });
            },
        );
    }
    group.finish();
}

fn bench_aux_value_loss(c: &mut Criterion) {
    let mut group = c.benchmark_group("ppg_aux_value_loss_mse");
    let device: <B as Backend>::Device = Default::default();
    for &n in &[256_usize, 1024, 4096] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let v: Tensor<B, 1> =
                Tensor::from_data(TensorData::new(vec![0.0_f32; n], vec![n]), &device);
            let r: Tensor<B, 1> =
                Tensor::from_data(TensorData::new(vec![1.0_f32; n], vec![n]), &device);
            b.iter(|| {
                let out = unclipped_value_loss(black_box(v.clone()), black_box(r.clone()));
                black_box(out);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_policy_kl_categorical, bench_aux_value_loss);
criterion_main!(benches);
