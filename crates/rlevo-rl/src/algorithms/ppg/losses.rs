//! Auxiliary-phase losses specific to PPG.
//!
//! The auxiliary-value regression target and the main value loss are shared
//! with PPO's [`unclipped_value_loss`](crate::algorithms::ppo::losses::unclipped_value_loss),
//! so only the categorical KL-distillation loss needs a new implementation.
//!
//! # References
//! - Cobbe et al. (2021), *Phasic Policy Gradient*, ICML.

use burn::tensor::Tensor;
use burn::tensor::activation::log_softmax;
use burn::tensor::backend::Backend;

/// Mean per-batch categorical KL divergence `KL(π_old ‖ π_new)`.
///
/// Both inputs are raw logits of shape `(batch, num_actions)`; the function
/// softmaxes internally. Returns a scalar tensor (shape `()`) suitable for
/// adding to other losses.
///
/// `KL = Σ softmax(old) · (log_softmax(old) − log_softmax(new))`, averaged
/// over the batch axis.
pub fn policy_kl_categorical<B: Backend>(
    old_logits: Tensor<B, 2>,
    new_logits: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let old_lp = log_softmax(old_logits, 1);
    let new_lp = log_softmax(new_logits, 1);
    let old_p = old_lp.clone().exp();
    // Σ_a p_old · (log p_old − log p_new), per row.
    let per_row = (old_p * (old_lp - new_lp)).sum_dim(1).squeeze_dim::<1>(1);
    per_row.mean()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::{ElementConversion, TensorData};

    type Be = NdArray;

    fn t2(data: &[f32], rows: usize, cols: usize) -> Tensor<Be, 2> {
        let device: <Be as Backend>::Device = Default::default();
        Tensor::<Be, 2>::from_data(TensorData::new(data.to_vec(), vec![rows, cols]), &device)
    }

    #[test]
    fn policy_kl_zero_when_logits_identical() {
        let l = t2(&[0.1_f32, 0.2, -0.3, 0.5, -0.1, 0.0], 2, 3);
        let kl = policy_kl_categorical(l.clone(), l)
            .into_scalar()
            .elem::<f32>();
        assert!(kl.abs() < 1e-6, "expected 0, got {kl}");
    }

    #[test]
    fn policy_kl_nonnegative_for_shifted_logits() {
        let old = t2(&[0.0_f32, 1.0, -1.0], 1, 3);
        let new = t2(&[0.5_f32, 0.5, 0.5], 1, 3);
        let kl = policy_kl_categorical(old, new).into_scalar().elem::<f32>();
        assert!(kl > 0.0, "expected > 0, got {kl}");
    }

    #[test]
    fn policy_kl_zero_when_logits_differ_by_constant() {
        // Softmax is shift-invariant, so KL should be 0.
        let old = t2(&[0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6], 2, 3);
        let mut shifted = vec![];
        for x in [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6].iter() {
            shifted.push(x + 5.0);
        }
        let new = t2(&shifted, 2, 3);
        let kl = policy_kl_categorical(old, new).into_scalar().elem::<f32>();
        assert!(kl.abs() < 1e-5, "expected ≈0, got {kl}");
    }
}
