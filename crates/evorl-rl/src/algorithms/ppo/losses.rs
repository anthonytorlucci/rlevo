//! Stateless loss functions and diagnostics used by the PPO update step.
//!
//! Every function takes tensors in, returns either a loss tensor (for the
//! three loss variants) or a scalar diagnostic (for `approx_kl`,
//! `clip_fraction`). Nothing here depends on `PpoAgent`; each is independently
//! unit-testable.
//!
//! # References
//! - Schulman et al. (2017), *Proximal Policy Optimization Algorithms*.
//! - Huang et al. (2022), *The 37 Implementation Details of PPO*.

use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, Tensor};

/// The clipped surrogate policy objective.
///
/// `ratio = exp(new_log_probs − old_log_probs)`. The loss returned is
/// `-mean(min(ratio·A, clip(ratio, 1−ε, 1+ε)·A))` — negative because
/// downstream code minimises. Shapes: all three inputs are `(batch,)`.
pub fn clipped_surrogate<B: Backend>(
    new_log_probs: Tensor<B, 1>,
    old_log_probs: Tensor<B, 1>,
    advantages: Tensor<B, 1>,
    clip_coef: f32,
) -> Tensor<B, 1> {
    let log_ratio = new_log_probs - old_log_probs;
    let ratio = log_ratio.exp();
    let surrogate1 = ratio.clone() * advantages.clone();
    let surrogate2 = ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef) * advantages;
    // elementwise min, then -mean
    let pointwise_min = min_elem(surrogate1, surrogate2);
    pointwise_min.mean().neg()
}

/// Clipped value-function loss.
///
/// `v_loss_clipped = max((clip(v, v_old ± ε) − R)², (v − R)²)`, mean × 0.5.
/// Follows CleanRL's clipped-value variant (an ablation-selected detail in
/// Huang et al. §5).
pub fn clipped_value_loss<B: Backend>(
    new_values: Tensor<B, 1>,
    old_values: Tensor<B, 1>,
    returns: Tensor<B, 1>,
    clip_coef: f32,
) -> Tensor<B, 1> {
    let v_unclipped = new_values.clone() - returns.clone();
    let v_unclipped_sq = v_unclipped.clone() * v_unclipped;

    let delta = new_values - old_values.clone();
    let clipped = old_values + delta.clamp(-clip_coef, clip_coef);
    let v_clipped = clipped - returns;
    let v_clipped_sq = v_clipped.clone() * v_clipped;

    let pointwise_max = max_elem(v_unclipped_sq, v_clipped_sq);
    pointwise_max.mean().mul_scalar(0.5)
}

/// Unclipped value-function loss `0.5 · mean((v − R)²)`.
pub fn unclipped_value_loss<B: Backend>(
    new_values: Tensor<B, 1>,
    returns: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let delta = new_values - returns;
    let sq = delta.clone() * delta;
    sq.mean().mul_scalar(0.5)
}

/// Standardise advantages to zero mean and unit variance batch-wise.
///
/// Uses an `epsilon = 1e-8` denominator floor. Consumes and returns the
/// tensor so the caller doesn't need an extra clone.
pub fn normalize_advantages<B: Backend>(advantages: Tensor<B, 1>) -> Tensor<B, 1> {
    let mean = advantages.clone().mean();
    let centered = advantages - mean;
    let var = (centered.clone() * centered.clone()).mean();
    let std = var.sqrt();
    centered / (std + 1e-8)
}

/// Schulman's "k3" approximate KL divergence: `mean((r − 1) − log r)`.
///
/// Always non-negative (in expectation); used as a sanity-check diagnostic
/// and, when `target_kl` is configured, as an early-stop trigger.
pub fn approx_kl<B: Backend>(new_log_probs: Tensor<B, 1>, old_log_probs: Tensor<B, 1>) -> f32 {
    let log_ratio = new_log_probs - old_log_probs;
    let ratio = log_ratio.clone().exp();
    let one_minus_ratio = ratio.neg() + 1.0; // (1 − r)
    let kl = one_minus_ratio.neg() - log_ratio; // (r − 1) − log r
    kl.mean().into_scalar().elem::<f32>()
}

/// Fraction of batch entries whose importance ratio was clipped.
pub fn clip_fraction<B: Backend>(
    new_log_probs: Tensor<B, 1>,
    old_log_probs: Tensor<B, 1>,
    clip_coef: f32,
) -> f32 {
    let ratio = (new_log_probs - old_log_probs).exp();
    let batch = ratio.dims()[0] as f32;
    let lo = ratio.clone().lower_elem(1.0 - clip_coef).float();
    let hi = ratio.greater_elem(1.0 + clip_coef).float();
    let clipped = lo + hi;
    let total = clipped.sum().into_scalar().elem::<f32>();
    total / batch.max(1.0)
}

fn min_elem<B: Backend>(a: Tensor<B, 1>, b: Tensor<B, 1>) -> Tensor<B, 1> {
    let mask = a.clone().lower_equal(b.clone());
    a.mask_where(mask.bool_not(), b)
}

fn max_elem<B: Backend>(a: Tensor<B, 1>, b: Tensor<B, 1>) -> Tensor<B, 1> {
    let mask = a.clone().greater_equal(b.clone());
    a.mask_where(mask.bool_not(), b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;

    type B = NdArray;

    fn t1(data: &[f32]) -> Tensor<B, 1> {
        let device = Default::default();
        Tensor::<B, 1>::from_data(TensorData::new(data.to_vec(), vec![data.len()]), &device)
    }

    #[test]
    fn ppo_clipped_obj_hand_rolled() {
        // ratio = [1.5, 0.5], advantages = [1.0, -1.0], clip = 0.2
        // new_lp − old_lp = ln(ratio) ⇒ use old_lp=0, new_lp=ln(r).
        let new_lp = t1(&[1.5_f32.ln(), 0.5_f32.ln()]);
        let old_lp = t1(&[0.0, 0.0]);
        let advs = t1(&[1.0, -1.0]);
        let loss = clipped_surrogate(new_lp, old_lp, advs, 0.2);
        let v = loss.into_scalar();
        // surrogate1 = [1.5, -0.5], surrogate2 = [clamp(1.5,[.8,1.2])=1.2, clamp(0.5,[.8,1.2])=0.8]·[-1] = [-0.8]
        // surrogate1 paired with surrogate2:
        //   row 0: min(1.5, 1.2) = 1.2
        //   row 1: min(-0.5, -0.8) = -0.8
        // mean = (1.2 + (-0.8)) / 2 = 0.2
        // loss = -0.2
        assert!((v + 0.2).abs() < 1e-5, "expected -0.2, got {v}");
    }

    #[test]
    fn advantage_norm_has_unit_var() {
        let advs = t1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let norm = normalize_advantages(advs);
        let data = norm.into_data();
        let v: Vec<f32> = data.as_slice::<f32>().expect("float").to_vec();
        let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
        let var: f32 = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32;
        assert!(mean.abs() < 1e-5, "mean not zero: {mean}");
        assert!((var.sqrt() - 1.0).abs() < 1e-4, "std not 1: {}", var.sqrt());
    }

    #[test]
    fn unclipped_value_loss_is_half_mse() {
        let v = t1(&[0.0, 2.0]);
        let r = t1(&[1.0, 0.0]);
        // (v - r) = [-1, 2] → sq = [1, 4] → mean = 2.5 → × 0.5 = 1.25
        let loss = unclipped_value_loss(v, r);
        assert!((loss.into_scalar() - 1.25).abs() < 1e-5);
    }

    #[test]
    fn clipped_value_loss_limits_update() {
        // old_v = [0], new_v = [10], returns = [0], clip = 0.5
        // unclipped sq = 100
        // clipped_v = old + clamp(new - old, ±0.5) = 0 + 0.5 = 0.5, sq = 0.25
        // max(100, 0.25) = 100 → 0.5 * 100 = 50
        // (clipping picks the LARGER of the two squared errors, per CleanRL)
        let new_v = t1(&[10.0]);
        let old_v = t1(&[0.0]);
        let r = t1(&[0.0]);
        let loss = clipped_value_loss(new_v, old_v, r, 0.5);
        assert!((loss.into_scalar() - 50.0).abs() < 1e-4);
    }

    #[test]
    fn approx_kl_nonnegative_for_shifted_distribution() {
        let new_lp = t1(&[0.6_f32.ln(), 0.4_f32.ln()]);
        let old_lp = t1(&[0.5_f32.ln(), 0.5_f32.ln()]);
        let kl = approx_kl(new_lp, old_lp);
        assert!(kl > 0.0, "expected non-negative kl, got {kl}");
    }

    #[test]
    fn approx_kl_zero_when_identical() {
        let lp = t1(&[0.1_f32.ln(), 0.9_f32.ln()]);
        let kl = approx_kl(lp.clone(), lp);
        assert!(kl.abs() < 1e-6, "expected 0 kl, got {kl}");
    }

    #[test]
    fn clip_fraction_counts_both_sides() {
        // ratios [0.5, 1.0, 1.5], clip 0.2 → below 0.8 and above 1.2 → 2/3
        let new_lp = t1(&[0.5_f32.ln(), 1.0_f32.ln(), 1.5_f32.ln()]);
        let old_lp = t1(&[0.0, 0.0, 0.0]);
        let frac = clip_fraction(new_lp, old_lp, 0.2);
        assert!((frac - 2.0 / 3.0).abs() < 1e-5, "expected 2/3, got {frac}");
    }
}
