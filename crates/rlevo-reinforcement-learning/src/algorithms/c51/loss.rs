//! Categorical cross-entropy loss for C51.
//!
//! Given the target distribution (see
//! [`crate::algorithms::c51::projection::project_distribution`]) and the
//! policy network's **log-probabilities** for the taken action, this returns
//! the **per-sample** cross-entropy `−Σ_i target_i · log p_i`, one value per
//! batch element. Reduction is the caller's job.
//!
//! Leaving the batch axis unreduced is what lets a caller multiply by a
//! per-sample importance-sampling weight *before* reducing (ADR 0050 §14);
//! at `w ≡ 1` the caller's `.mean()` is bit-identical to reducing here.
//!
//! Kept separate from the agent struct so it can be reused from benchmarks
//! and unit-tested independently.

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

/// Per-sample cross-entropy between `target_probs` and `predicted_log_probs`.
///
/// # Shapes
/// - `target_probs`:        `(batch, num_atoms)`
/// - `predicted_log_probs`: `(batch, num_atoms)`
/// - **returns**:           `(batch,)`
///
/// # Returns
///
/// An **unreduced** `Tensor<B, 1>` of shape `[batch]` — element `n` is
/// `−Σ_i target_probs[n, i] · predicted_log_probs[n, i]`. This is *not* a
/// scalar: a caller that wants the usual training loss must apply its own
/// reduction (`.mean()`), and a caller that omits it will either hit a shape
/// error or silently backpropagate a summed-over-batch gradient.
///
/// `predicted_log_probs` must already be log-probabilities along the atom
/// axis (typically produced via `tensor.log_softmax(atom_axis)`); no
/// normalisation is done internally.
pub fn categorical_cross_entropy_per_sample<B: Backend>(
    target_probs: Tensor<B, 2>,
    predicted_log_probs: Tensor<B, 2>,
) -> Tensor<B, 1> {
    (target_probs * predicted_log_probs)
        .sum_dim(1)
        .squeeze_dim::<1>(1)
        .neg()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;
    use burn::tensor::TensorData;
    use burn::tensor::activation;

    type B = Flex;

    fn tensor_2d(data: Vec<f32>, rows: usize, cols: usize) -> Tensor<B, 2> {
        let device = <B as burn::tensor::backend::BackendTypes>::Device::default();
        Tensor::from_data(TensorData::new(data, vec![rows, cols]), &device)
    }

    #[test]
    fn cross_entropy_zero_on_identity() {
        // CE(p, log p) when the distribution is one-hot is exactly 0 (log 1).
        let target = tensor_2d(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 2, 3);
        let logits = tensor_2d(vec![10.0, -10.0, -10.0, -10.0, 10.0, -10.0], 2, 3);
        let log_p = activation::log_softmax(logits, 1);
        let loss = categorical_cross_entropy_per_sample(target, log_p)
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("f32 loss");
        assert_eq!(
            loss.len(),
            2,
            "loss must be unreduced, one value per sample"
        );
        for (n, l) in loss.iter().enumerate() {
            assert!(l.abs() < 1e-4, "expected ~0 loss at sample {n}, got {l}");
        }
    }

    #[test]
    fn cross_entropy_is_unreduced_over_batch() {
        // The return must carry the batch axis: a 4-row input yields 4 values,
        // and their mean must equal the pre-ADR-0050 reduced-in-callee value.
        let batch = 4;
        let target = tensor_2d(
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.0],
            batch,
            3,
        );
        let logits = tensor_2d(
            vec![
                0.3, -0.1, 0.7, -0.4, 1.2, 0.0, 0.9, 0.2, -0.6, 0.1, 0.5, -0.2,
            ],
            batch,
            3,
        );
        let log_p = activation::log_softmax(logits, 1);
        let per_sample = categorical_cross_entropy_per_sample(target, log_p);
        assert_eq!(
            per_sample.dims(),
            [batch],
            "per-sample loss must have shape [batch]"
        );

        let values = per_sample
            .clone()
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("f32 loss");
        #[allow(clippy::cast_precision_loss)]
        let manual_mean = values.iter().sum::<f32>() / batch as f32;
        let reduced = per_sample
            .mean()
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("f32 loss")[0];
        assert!(
            (manual_mean - reduced).abs() < 1e-6,
            "caller-side mean {reduced} must match the manual mean {manual_mean}"
        );
    }

    #[test]
    fn cross_entropy_nonnegative_on_uniform() {
        // For any valid probability distributions, CE ≥ 0.
        let n = 4;
        let batch = 3;
        let target = tensor_2d(vec![1.0 / n as f32; batch * n], batch, n);
        let logits = tensor_2d(
            vec![
                0.5, -0.2, 1.1, -0.7, 0.0, 0.9, -1.3, 0.4, -0.5, 0.2, 0.8, -0.1,
            ],
            batch,
            n,
        );
        let log_p = activation::log_softmax(logits, 1);
        let losses = categorical_cross_entropy_per_sample(target, log_p)
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("f32 loss");
        assert_eq!(losses.len(), batch, "one loss value per batch element");
        for loss in losses {
            assert!(
                loss >= 0.0,
                "cross-entropy must be non-negative, got {loss}"
            );
            assert!(loss.is_finite());
        }
    }
}
