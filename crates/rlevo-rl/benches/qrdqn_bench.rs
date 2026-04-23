//! Micro benchmarks for the QR-DQN quantile Huber loss.
//!
//! Runs the loss at a spread of `(num_quantiles, batch_size)` settings so a
//! regression that turns the `(B, N, N)` broadcast into a `(B, B, N, N)` or
//! otherwise super-linear blow-up is easy to spot.

use burn::backend::NdArray;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

use criterion::{Criterion, criterion_group, criterion_main};

use rlevo_rl::algorithms::qrdqn::quantile_loss::quantile_huber_loss;

type Be = NdArray;

fn make_taus(n: usize, device: &<Be as Backend>::Device) -> Tensor<Be, 1> {
    let data: Vec<f32> = (0..n).map(|i| (i as f32 + 0.5) / n as f32).collect();
    Tensor::from_data(TensorData::new(data, vec![n]), device)
}

fn make_quantile_batch(batch: usize, n: usize, device: &<Be as Backend>::Device) -> Tensor<Be, 2> {
    // Deterministic but non-trivial payload: a saw-tooth across the row.
    let data: Vec<f32> = (0..batch * n)
        .map(|i| ((i % n) as f32) * 0.1 - ((i / n) as f32) * 0.01)
        .collect();
    Tensor::from_data(TensorData::new(data, vec![batch, n]), device)
}

fn bench_quantile_huber_loss(c: &mut Criterion) {
    let device: <Be as Backend>::Device = Default::default();

    for &num_quantiles in &[51_usize, 101, 200] {
        for &batch in &[32_usize, 128] {
            let taus = make_taus(num_quantiles, &device);
            let pred = make_quantile_batch(batch, num_quantiles, &device);
            let target = make_quantile_batch(batch, num_quantiles, &device);

            let label = format!("quantile_huber_loss/quantiles={num_quantiles}/batch={batch}");
            c.bench_function(&label, |b| {
                b.iter(|| {
                    let out = quantile_huber_loss(pred.clone(), target.clone(), taus.clone(), 1.0);
                    let _ = out.into_data();
                });
            });
        }
    }
}

criterion_group!(micro, bench_quantile_huber_loss);
criterion_main!(micro);
