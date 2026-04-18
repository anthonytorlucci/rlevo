//! Quantile Huber loss for QR-DQN (Dabney et al. 2018, Eq. 10).
//!
//! The asymmetric κ-Huber loss weighted by `|τ − 𝟙{u < 0}|` is the
//! distributional counterpart of the Huber TD loss used by DQN. Given the
//! network's predicted quantile values for the taken action together with
//! the target quantile values (produced by the Bellman backup on the
//! bootstrap action's quantile vector), this returns the scalar loss
//! aggregated as: mean over the target axis, sum over the predicted
//! axis, mean over the batch.
//!
//! Kept separate from the agent struct so the math can be reused from
//! benchmarks and unit-tested independently.

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

/// Elementwise Huber loss with threshold `kappa`.
///
/// `L_κ(u) = 0.5 · u²` when `|u| ≤ κ`, `κ · (|u| − 0.5 · κ)` otherwise.
///
/// Differentiable at the seam — the classic Huber smoothing used in QR-DQN
/// to avoid a discontinuous derivative at `u = 0`.
pub fn huber<B: Backend, const D: usize>(u: Tensor<B, D>, kappa: f32) -> Tensor<B, D> {
    let abs_u = u.clone().abs();
    // Small-region mask: 1.0 where `|u| <= kappa`, 0.0 otherwise.
    let small = abs_u.clone().lower_equal_elem(kappa).float();
    let quad = u.powi_scalar(2).mul_scalar(0.5);
    let linear = abs_u.sub_scalar(0.5 * kappa).mul_scalar(kappa);
    small.clone() * quad + small.neg().add_scalar(1.0) * linear
}

/// Quantile Huber loss between `pred_quantiles` and `target_quantiles`.
///
/// # Shapes
/// - `pred_quantiles`:   `(batch, num_quantiles)` — policy network's quantile
///   estimates for the taken action.
/// - `target_quantiles`: `(batch, num_quantiles)` — Bellman-backed-up target
///   quantile vector (for the bootstrap action).
/// - `taus`:             `(num_quantiles,)` — midpoint quantile levels
///   `τ_i = (i + 0.5) / N`.
/// - `kappa`: Huber threshold.
///
/// # Aggregation
/// `u_ij = target_j − pred_i`, then `ρ_κ_{τ_i}(u_ij) = |τ_i − 𝟙{u_ij < 0}| ·
/// L_κ(u_ij) / κ`. Aggregated as `mean_j → sum_i → mean_batch` following
/// Dabney Eq. 10.
pub fn quantile_huber_loss<B: Backend>(
    pred_quantiles: Tensor<B, 2>,
    target_quantiles: Tensor<B, 2>,
    taus: Tensor<B, 1>,
    kappa: f32,
) -> Tensor<B, 1> {
    // Broadcast to (B, N_pred, N_target).
    let pred_3d: Tensor<B, 3> = pred_quantiles.unsqueeze_dim::<3>(2); // (B, N, 1)
    let target_3d: Tensor<B, 3> = target_quantiles.unsqueeze_dim::<3>(1); // (B, 1, N)
    let u = target_3d - pred_3d; // (B, N_pred, N_target)

    // `|τ_i − 𝟙{u_ij < 0}|` — the Bool → float conversion detaches from
    // the autodiff graph, so the gradient flows only through `u`.
    let taus_3d: Tensor<B, 3> = taus.unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(2); // (1, N, 1)
    let neg_mask = u.clone().lower_elem(0.0).float(); // (B, N_pred, N_target)
    let weight = (taus_3d - neg_mask).abs();

    let huber_u = huber::<B, 3>(u, kappa);
    let weighted = (weight * huber_u).div_scalar(kappa);

    // mean over target axis, sum over pred axis, mean over batch.
    let per_pred: Tensor<B, 2> = weighted.mean_dim(2).squeeze_dim::<2>(2);
    let per_sample: Tensor<B, 1> = per_pred.sum_dim(1).squeeze_dim::<1>(1);
    per_sample.mean()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;

    type B = NdArray;

    fn tensor_1d(data: Vec<f32>) -> Tensor<B, 1> {
        let device = <B as Backend>::Device::default();
        let n = data.len();
        Tensor::from_data(TensorData::new(data, vec![n]), &device)
    }

    fn tensor_2d(data: Vec<f32>, rows: usize, cols: usize) -> Tensor<B, 2> {
        let device = <B as Backend>::Device::default();
        Tensor::from_data(TensorData::new(data, vec![rows, cols]), &device)
    }

    fn scalar(t: Tensor<B, 1>) -> f32 {
        t.into_data().convert::<f32>().into_vec::<f32>().unwrap()[0]
    }

    #[test]
    fn huber_zero_on_zero_input() {
        let u = tensor_2d(vec![0.0; 6], 2, 3);
        let h = huber::<B, 2>(u, 1.0);
        let v: Vec<f32> = h.into_data().convert::<f32>().into_vec::<f32>().unwrap();
        for x in v {
            assert!(x.abs() < 1e-6);
        }
    }

    #[test]
    fn huber_quadratic_below_kappa() {
        // |u| < kappa ⇒ 0.5 u²
        let u = tensor_2d(vec![0.2_f32, -0.5, 0.75, -0.9], 2, 2);
        let h = huber::<B, 2>(u, 1.0);
        let v: Vec<f32> = h.into_data().convert::<f32>().into_vec::<f32>().unwrap();
        let expected = [0.02_f32, 0.125, 0.28125, 0.405];
        for (got, want) in v.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-5, "got {got}, want {want}");
        }
    }

    #[test]
    fn huber_linear_above_kappa() {
        // |u| > kappa ⇒ kappa*(|u| − 0.5*kappa) with kappa=1.0 ⇒ |u| − 0.5
        let u = tensor_2d(vec![2.0_f32, -3.0, 5.0, -1.5], 2, 2);
        let h = huber::<B, 2>(u, 1.0);
        let v: Vec<f32> = h.into_data().convert::<f32>().into_vec::<f32>().unwrap();
        let expected = [1.5_f32, 2.5, 4.5, 1.0];
        for (got, want) in v.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-5, "got {got}, want {want}");
        }
    }

    #[test]
    fn loss_zero_on_constant_distribution() {
        // If every predicted and target quantile equals the same constant,
        // `u_ij = target_j − pred_i = 0` for all (i, j), so the loss is 0.
        // Note: pred == target alone is NOT sufficient — off-diagonal terms
        // `u_ij = target_j − pred_i` remain non-zero whenever the quantile
        // values differ across `i`.
        let pred = tensor_2d(vec![1.0_f32; 6], 2, 3);
        let target = pred.clone();
        let taus = tensor_1d(vec![1.0 / 6.0, 0.5, 5.0 / 6.0]);
        let loss = scalar(quantile_huber_loss(pred, target, taus, 1.0));
        assert!(loss.abs() < 1e-6, "loss should be ~0, got {loss}");
    }

    #[test]
    fn loss_nonnegative_on_random_inputs() {
        // Any finite pred/target should yield non-negative loss.
        let pred = tensor_2d(vec![0.3_f32, -0.5, 1.1, 0.4, -0.2, 0.9], 2, 3);
        let target = tensor_2d(vec![0.1_f32, 0.2, 0.8, -0.3, 0.5, 1.0], 2, 3);
        let taus = tensor_1d(vec![1.0 / 6.0, 0.5, 5.0 / 6.0]);
        let loss = scalar(quantile_huber_loss(pred, target, taus, 1.0));
        assert!(loss >= 0.0, "loss must be non-negative, got {loss}");
        assert!(loss.is_finite());
    }

    #[test]
    fn symmetric_at_median_tau_matches_mean_huber() {
        // With N=1 and τ=0.5, |τ − 𝟙{u<0}| is always 0.5, so the quantile
        // Huber loss reduces to 0.5 · L_κ(u) / κ. Verify against a
        // hand-computed scalar reference: pred=1.0, target=0.5 ⇒ u=−0.5,
        // L_1(−0.5)=0.125, weight=0.5 ⇒ loss = 0.5·0.125/1.0 = 0.0625.
        let pred = tensor_2d(vec![1.0_f32], 1, 1);
        let target = tensor_2d(vec![0.5_f32], 1, 1);
        let taus = tensor_1d(vec![0.5_f32]);
        let loss = scalar(quantile_huber_loss(pred, target, taus, 1.0));
        assert!((loss - 0.0625).abs() < 1e-6, "got {loss}, want 0.0625");
    }
}
