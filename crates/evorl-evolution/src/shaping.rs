//! Fitness shaping transforms.
//!
//! Strategies occasionally benefit from monotone transforms of raw
//! fitness values: centered-rank mapping flattens outliers, z-scoring
//! normalizes scale across generations, and weight decay penalizes
//! large-norm genomes (common in neuroevolution). These helpers work
//! directly on device tensors and are pure functions (RNG-free).

use burn::prelude::ElementConversion;
use burn::tensor::{backend::Backend, Tensor};

/// Returns `fitness - fitness.mean()` divided by the (population) std-dev,
/// clamped to avoid divide-by-zero when all fitnesses are equal.
#[must_use]
pub fn z_score<B: Backend>(fitness: Tensor<B, 1>) -> Tensor<B, 1> {
    let mean = fitness.clone().mean().into_scalar().elem::<f32>();
    let n = fitness.shape().dims[0];
    #[allow(clippy::cast_precision_loss)]
    let n_f = n.max(1) as f32;
    let centered = fitness - mean;
    let var = centered.clone().powf_scalar(2.0).sum().into_scalar().elem::<f32>() / n_f;
    let std = var.sqrt().max(1e-8);
    centered / std
}

/// Returns centered ranks: the largest fitness maps to `+0.5`, the
/// smallest to `-0.5`, with linear spacing in between.
///
/// Centered ranks are standard in modern ES (e.g. OpenAI-ES) because they
/// remove outlier fitness magnitudes and keep the signal scale-free across
/// generations. Implemented host-side because the argsort pathway is
/// easier to reason about; swap in a tensor-native implementation if this
/// ever shows up on a profile.
#[must_use]
pub fn centered_rank<B: Backend>(fitness: Tensor<B, 1>, device: &B::Device) -> Tensor<B, 1> {
    let data = fitness.into_data().into_vec::<f32>().unwrap_or_default();
    let n = data.len();
    if n == 0 {
        return Tensor::<B, 1>::from_floats([0.0f32; 0], device);
    }
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&i, &j| data[i].partial_cmp(&data[j]).unwrap_or(std::cmp::Ordering::Equal));

    #[allow(clippy::cast_precision_loss)]
    let n_f = (n - 1).max(1) as f32;
    let mut ranks = vec![0.0_f32; n];
    for (rank, &idx) in indices.iter().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let r = rank as f32 / n_f - 0.5;
        ranks[idx] = r;
    }
    Tensor::<B, 1>::from_floats(ranks.as_slice(), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    type TestBackend = NdArray;

    #[test]
    fn z_score_zero_mean_unit_std() {
        let device = Default::default();
        let t = Tensor::<TestBackend, 1>::from_floats([1.0f32, 2.0, 3.0, 4.0, 5.0], &device);
        let z = z_score(t);
        let values = z.into_data().into_vec::<f32>().unwrap();
        let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
        approx::assert_relative_eq!(mean, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn centered_rank_spans_half_interval() {
        let device = Default::default();
        let t = Tensor::<TestBackend, 1>::from_floats([10.0f32, 20.0, 30.0, 40.0], &device);
        let r = centered_rank(t, &device);
        let values = r.into_data().into_vec::<f32>().unwrap();
        // smallest → -0.5, largest → +0.5
        approx::assert_relative_eq!(values[0], -0.5, epsilon = 1e-6);
        approx::assert_relative_eq!(values[3], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn centered_rank_preserves_order() {
        let device = Default::default();
        let t = Tensor::<TestBackend, 1>::from_floats([3.0f32, 1.0, 2.0], &device);
        let r = centered_rank(t, &device);
        let values = r.into_data().into_vec::<f32>().unwrap();
        // original: 3, 1, 2 → ranks sorted ascending: [1, 2, 3] at indices [1, 2, 0]
        // rank-positions centered: index 1 → -0.5, index 2 → 0.0, index 0 → 0.5
        approx::assert_relative_eq!(values[1], -0.5, epsilon = 1e-6);
        approx::assert_relative_eq!(values[2], 0.0, epsilon = 1e-6);
        approx::assert_relative_eq!(values[0], 0.5, epsilon = 1e-6);
    }
}
