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

/// Number of predicted-quantile rows processed per loop iteration.
///
/// The full pairwise tensor is `(B, N_pred, N_target)`. At N=200, B=128 that
/// is ~20 MB of f32 — too large for L2/L3 cache, causing repeated cache misses
/// across the ~10 element-wise passes over the data. Chunking keeps the working
/// set at `B × CHUNK × N_target × 4` bytes: with CHUNK=32 and the largest
/// benchmark config (B=128, N=200) that is 3.2 MB, which fits comfortably in
/// Apple-Silicon L2. The math is identical; only peak allocation changes.
const QUANTILE_CHUNK_SIZE: usize = 32;

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
///
/// The pred axis is processed in chunks of [`QUANTILE_CHUNK_SIZE`] to avoid
/// materialising the full `(B, N, N)` tensor at once.
pub fn quantile_huber_loss<B: Backend>(
    pred_quantiles: Tensor<B, 2>,
    target_quantiles: Tensor<B, 2>,
    taus: Tensor<B, 1>,
    kappa: f32,
) -> Tensor<B, 1> {
    let [batch, n_pred] = pred_quantiles.dims();

    // Accumulate `Σ_i mean_j ρ(u_ij)` for each sample, one chunk of pred
    // quantiles at a time. Each chunk materialises (B, chunk, N_target) rather
    // than the full (B, N_pred, N_target) block.
    let mut per_sample_acc: Option<Tensor<B, 1>> = None;
    let mut chunk_start = 0;

    while chunk_start < n_pred {
        let chunk_end = (chunk_start + QUANTILE_CHUNK_SIZE).min(n_pred);

        let pred_chunk = pred_quantiles
            .clone()
            .slice([0..batch, chunk_start..chunk_end]); // (B, chunk)
        let chunk_range = chunk_start..chunk_end;
        let taus_chunk = taus.clone().slice([chunk_range]); // (chunk,)

        // Broadcast to (B, chunk, N_target).
        let pred_3d: Tensor<B, 3> = pred_chunk.unsqueeze_dim::<3>(2); // (B, chunk, 1)
        let target_3d: Tensor<B, 3> = target_quantiles.clone().unsqueeze_dim::<3>(1); // (B, 1, N_target)
        let u = target_3d - pred_3d; // (B, chunk, N_target)

        // `|τ_i − 𝟙{u_ij < 0}|` — Bool → float detaches from autodiff graph.
        let taus_3d: Tensor<B, 3> = taus_chunk
            .unsqueeze_dim::<2>(0) // (1, chunk)
            .unsqueeze_dim::<3>(2); // (1, chunk, 1)
        let neg_mask = u.clone().lower_elem(0.0).float(); // (B, chunk, N_target)
        let weight = (taus_3d - neg_mask).abs();

        let huber_u = huber::<B, 3>(u, kappa);
        let weighted = (weight * huber_u).div_scalar(kappa);

        // mean over target axis (dim 2) → (B, chunk); sum over chunk (dim 1) → (B,).
        let per_pred: Tensor<B, 2> = weighted.mean_dim(2).squeeze_dim::<2>(2);
        let chunk_sum: Tensor<B, 1> = per_pred.sum_dim(1).squeeze_dim::<1>(1);

        per_sample_acc = Some(match per_sample_acc {
            None => chunk_sum,
            Some(acc) => acc + chunk_sum,
        });

        chunk_start = chunk_end;
    }

    per_sample_acc
        .expect("quantile_huber_loss: pred_quantiles must have at least one quantile")
        .mean()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;
    use burn::tensor::TensorData;

    type B = Flex;

    fn tensor_1d(data: Vec<f32>) -> Tensor<B, 1> {
        let device = <B as burn::tensor::backend::BackendTypes>::Device::default();
        let n = data.len();
        Tensor::from_data(TensorData::new(data, vec![n]), &device)
    }

    fn tensor_2d(data: Vec<f32>, rows: usize, cols: usize) -> Tensor<B, 2> {
        let device = <B as burn::tensor::backend::BackendTypes>::Device::default();
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
