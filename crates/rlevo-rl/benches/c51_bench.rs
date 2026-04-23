//! Micro benchmarks for the C51 categorical projection operator.
//!
//! Runs the projection at a spread of `(num_atoms, batch_size)` settings so
//! a regression that turns the linear scatter-add into a quadratic broadcast
//! is easy to spot in the output.

use burn::backend::NdArray;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

use criterion::{Criterion, criterion_group, criterion_main};

use evorl_rl::algorithms::c51::projection::project_distribution;

type Be = NdArray;

fn make_support(
    v_min: f32,
    v_max: f32,
    n: usize,
    device: &<Be as Backend>::Device,
) -> Tensor<Be, 1> {
    let delta = (v_max - v_min) / (n as f32 - 1.0);
    let data: Vec<f32> = (0..n).map(|i| v_min + (i as f32) * delta).collect();
    Tensor::from_data(TensorData::new(data, vec![n]), device)
}

fn bench_projection(c: &mut Criterion) {
    let device: <Be as Backend>::Device = Default::default();
    let v_min = -10.0_f32;
    let v_max = 10.0_f32;

    for &num_atoms in &[21_usize, 51, 101] {
        for &batch in &[32_usize, 128] {
            let support = make_support(v_min, v_max, num_atoms, &device);

            // Fill next_probs with a valid distribution per row.
            let row: Vec<f32> = vec![1.0_f32 / num_atoms as f32; num_atoms];
            let data: Vec<f32> = (0..batch).flat_map(|_| row.clone()).collect();
            let next_probs: Tensor<Be, 2> =
                Tensor::from_data(TensorData::new(data, vec![batch, num_atoms]), &device);

            let rewards_data: Vec<f32> = (0..batch).map(|i| (i as f32) * 0.01 - 0.5).collect();
            let rewards: Tensor<Be, 1> =
                Tensor::from_data(TensorData::new(rewards_data, vec![batch]), &device);

            let dones_data: Vec<f32> = (0..batch)
                .map(|i| if i % 5 == 0 { 1.0 } else { 0.0 })
                .collect();
            let dones: Tensor<Be, 1> =
                Tensor::from_data(TensorData::new(dones_data, vec![batch]), &device);

            let label = format!("project_distribution/atoms={num_atoms}/batch={batch}");
            c.bench_function(&label, |b| {
                b.iter(|| {
                    let out = project_distribution(
                        next_probs.clone(),
                        rewards.clone(),
                        dones.clone(),
                        support.clone(),
                        0.99,
                        v_min,
                        v_max,
                        num_atoms,
                    );
                    // Touch the result so the optimizer doesn't elide it.
                    let _ = out.into_data();
                });
            });
        }
    }
}

criterion_group!(micro, bench_projection);
criterion_main!(micro);
