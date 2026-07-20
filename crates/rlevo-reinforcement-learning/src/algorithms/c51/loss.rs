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

/// Clamp floor for `log(target)` so that an exact-zero atom contributes
/// `0 · log(TINY) = 0` to the entropy term rather than `0 · (−∞) = NaN`.
///
/// A projected C51 target routinely has exact-zero atoms outside the two bins a
/// backed-up value lands in, so this guard is load-bearing, not defensive.
const KL_LOG_FLOOR: f32 = 1e-30;

/// Per-sample KL divergence `D_KL(target ‖ pred)` — C51's **priority** signal
/// for prioritized replay (Rainbow), which is deliberately *not* the
/// cross-entropy the gradient uses.
///
/// # Why KL and not the cross-entropy
///
/// `categorical_cross_entropy_per_sample` returns `CE = −Σ_i t_i · log p_i`,
/// which is what C51 minimises and therefore what the gradient sees. Rainbow
/// prioritizes transitions by the **KL loss** instead ("since this is what the
/// algorithm is minimizing", verbatim), and
///
/// ```text
/// D_KL(t ‖ p) = Σ_i t_i · log(t_i / p_i) = CE − H(t),   H(t) = −Σ_i t_i · log t_i
/// ```
///
/// The two differ by the **target entropy** `H(t)`. `H(t)` is constant with
/// respect to the network parameters θ, so it vanishes from the gradient — the
/// loss can stay CE with no change to training. But `H(t)` **varies from sample
/// to sample**, so using CE as the replay priority ranks transitions
/// differently from KL. This function subtracts `H(t)` explicitly, computing
/// `CE + Σ_i t_i · log t_i`, so the priority is the KL Rainbow specifies rather
/// than the CE that happens to be lying around.
///
/// # Shapes
/// - `target_probs`:        `(batch, num_atoms)`
/// - `predicted_log_probs`: `(batch, num_atoms)`
/// - **returns**:           `(batch,)`
///
/// The result is the non-negative KL per batch element. `predicted_log_probs`
/// must already be log-probabilities along the atom axis.
pub fn categorical_kl_per_sample<B: Backend>(
    target_probs: Tensor<B, 2>,
    predicted_log_probs: Tensor<B, 2>,
) -> Tensor<B, 1> {
    // CE = −Σ t·log p.
    let ce = categorical_cross_entropy_per_sample(target_probs.clone(), predicted_log_probs);
    // Σ t·log t  (= −H(t), ≤ 0). The clamp floors log(0) so a zero atom's
    // `t·log t` is `0·log(TINY) = 0` rather than `0·(−∞) = NaN`.
    let neg_entropy = (target_probs.clone() * target_probs.clamp_min(KL_LOG_FLOOR).log())
        .sum_dim(1)
        .squeeze_dim::<1>(1);
    // KL = CE − H(t) = CE + Σ t·log t.
    ce + neg_entropy
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

    /// Reads a `[batch]` loss tensor to a host `Vec<f32>`.
    fn to_vec(t: Tensor<B, 1>) -> Vec<f32> {
        t.into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("f32 loss")
    }

    /// Builds `predicted_log_probs` from explicit probabilities by taking their
    /// natural log — the exact log-probs the KL formula expects.
    fn log_of(probs: Vec<f32>, rows: usize, cols: usize) -> Tensor<B, 2> {
        tensor_2d(probs, rows, cols).log()
    }

    #[test]
    fn kl_matches_hand_computed_values() {
        // Sample A: t = [0.5, 0.5], p = [0.6, 0.4].
        //   CE_A = −(0.5·ln0.6 + 0.5·ln0.4) = 0.71355818
        //   H_A  = ln 2 = 0.69314718
        //   KL_A = CE_A − H_A = 0.02041100
        // Sample B: t = [0.9, 0.1], p = [0.7, 0.3].
        //   CE_B = −(0.9·ln0.7 + 0.1·ln0.3) = 0.44140473
        //   H_B  = −(0.9·ln0.9 + 0.1·ln0.1) = 0.32508297
        //   KL_B = CE_B − H_B = 0.11632176
        let target = tensor_2d(vec![0.5, 0.5, 0.9, 0.1], 2, 2);
        let log_p = log_of(vec![0.6, 0.4, 0.7, 0.3], 2, 2);
        let kl = to_vec(categorical_kl_per_sample(target, log_p));
        assert!(
            (kl[0] - 0.020_411_0).abs() < 1e-4,
            "KL_A must equal the hand-computed 0.0204110, got {}",
            kl[0]
        );
        assert!(
            (kl[1] - 0.116_321_76).abs() < 1e-4,
            "KL_B must equal the hand-computed 0.1163218, got {}",
            kl[1]
        );
    }

    /// The load-bearing test: CE and KL must **rank the two transitions in
    /// opposite order**, so a future "simplification" back to CE-as-priority
    /// changes replay ranking and fails here.
    ///
    /// With the samples above, `CE_A (0.714) > CE_B (0.441)` but
    /// `KL_A (0.020) < KL_B (0.116)`: cross-entropy would prioritize A, the KL
    /// Rainbow specifies prioritizes B. The inversion is driven entirely by the
    /// per-sample target entropy `H(t)`, which CE ignores.
    #[test]
    fn kl_and_ce_rank_transitions_in_opposite_order() {
        let target = tensor_2d(vec![0.5, 0.5, 0.9, 0.1], 2, 2);
        let log_p = log_of(vec![0.6, 0.4, 0.7, 0.3], 2, 2);
        let ce = to_vec(categorical_cross_entropy_per_sample(
            target.clone(),
            log_p.clone(),
        ));
        let kl = to_vec(categorical_kl_per_sample(target, log_p));
        assert!(
            ce[0] > ce[1],
            "cross-entropy ranks sample A above B: {} vs {}",
            ce[0],
            ce[1]
        );
        assert!(
            kl[0] < kl[1],
            "KL ranks sample B above A — the opposite order: {} vs {}",
            kl[0],
            kl[1]
        );
    }

    #[test]
    fn kl_equals_ce_for_a_one_hot_target() {
        // A one-hot target has H(t) = 0, so KL = CE exactly — and the exact-zero
        // atom must contribute 0, not NaN, to the entropy term.
        let target = tensor_2d(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 2, 3);
        let log_p = log_of(vec![0.8, 0.1, 0.1, 0.2, 0.5, 0.3], 2, 3);
        let ce = to_vec(categorical_cross_entropy_per_sample(
            target.clone(),
            log_p.clone(),
        ));
        let kl = to_vec(categorical_kl_per_sample(target, log_p));
        for (n, (k, c)) in kl.iter().zip(&ce).enumerate() {
            assert!(
                k.is_finite(),
                "KL must be finite despite zero atoms (sample {n})"
            );
            assert!(
                (k - c).abs() < 1e-5,
                "one-hot target ⇒ H(t) = 0 ⇒ KL == CE (sample {n}): {k} vs {c}"
            );
        }
    }

    #[test]
    fn kl_is_nonnegative_and_zero_on_identity() {
        // D_KL(t ‖ t) = 0, and KL ≥ 0 for any distributions.
        let target = tensor_2d(vec![0.2, 0.3, 0.5, 0.6, 0.1, 0.3], 2, 3);
        let identical = to_vec(categorical_kl_per_sample(
            target.clone(),
            target.clone().log(),
        ));
        for (n, k) in identical.iter().enumerate() {
            assert!(k.abs() < 1e-5, "KL(t ‖ t) must be ~0 (sample {n}), got {k}");
        }
        let log_p = log_of(vec![0.3, 0.3, 0.4, 0.2, 0.5, 0.3], 2, 3);
        let kl = to_vec(categorical_kl_per_sample(target, log_p));
        for (n, k) in kl.iter().enumerate() {
            assert!(*k >= -1e-6, "KL must be non-negative (sample {n}), got {k}");
        }
    }

    #[test]
    // Test fixture data: the loop counter and element count are bounded by small
    // constants declared in this test, far below f32's 2^24 exact-integer limit,
    // so every generated value is represented exactly.
    #[allow(clippy::cast_precision_loss)]
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
