//! Categorical cross-entropy loss for C51.
//!
//! Given the target distribution (see
//! [`crate::algorithms::c51::projection::project_distribution`]) and the
//! policy network's **log-probabilities** for the taken action, this returns
//! the scalar cross-entropy `−Σ_i target_i · log p_i`, meaned across the
//! batch.
//!
//! Kept separate from the agent struct so it can be reused from benchmarks
//! and unit-tested independently.

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

/// Mean cross-entropy between `target_probs` and `predicted_log_probs`.
///
/// Both tensors have shape `(batch, num_atoms)`.
///
/// `predicted_log_probs` must already be log-probabilities along the atom
/// axis (typically produced via `tensor.log_softmax(atom_axis)`); no
/// normalisation is done internally.
pub fn categorical_cross_entropy<B: Backend>(
    target_probs: Tensor<B, 2>,
    predicted_log_probs: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let per_sample: Tensor<B, 1> = (target_probs * predicted_log_probs)
        .sum_dim(1)
        .squeeze_dim::<1>(1)
        .neg();
    per_sample.mean()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;
    use burn::tensor::activation;

    type B = NdArray;

    fn tensor_2d(data: Vec<f32>, rows: usize, cols: usize) -> Tensor<B, 2> {
        let device = <B as Backend>::Device::default();
        Tensor::from_data(TensorData::new(data, vec![rows, cols]), &device)
    }

    #[test]
    fn cross_entropy_zero_on_identity() {
        // CE(p, log p) when the distribution is one-hot is exactly 0 (log 1).
        let target = tensor_2d(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 2, 3);
        let logits = tensor_2d(vec![10.0, -10.0, -10.0, -10.0, 10.0, -10.0], 2, 3);
        let log_p = activation::log_softmax(logits, 1);
        let loss = categorical_cross_entropy(target, log_p)
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("f32 loss");
        assert!(loss[0].abs() < 1e-4, "expected ~0 loss, got {}", loss[0]);
    }

    #[test]
    fn cross_entropy_nonnegative_on_uniform() {
        // For any valid probability distributions, CE ≥ 0.
        let n = 4;
        let batch = 3;
        let target = tensor_2d(vec![1.0 / n as f32; batch * n], batch, n);
        let logits = tensor_2d(vec![0.5, -0.2, 1.1, -0.7, 0.0, 0.9, -1.3, 0.4, -0.5, 0.2, 0.8, -0.1], batch, n);
        let log_p = activation::log_softmax(logits, 1);
        let loss = categorical_cross_entropy(target, log_p)
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("f32 loss")[0];
        assert!(loss >= 0.0, "cross-entropy must be non-negative, got {loss}");
        assert!(loss.is_finite());
    }
}
