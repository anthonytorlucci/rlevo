//! Target-policy smoothing noise — the second of TD3's three deltas over
//! DDPG.
//!
//! Fujimoto et al. 2018 argue that deterministic policies overfit sharp
//! peaks in the Q-landscape; smoothing the target action with clipped
//! Gaussian noise makes the critic fit a locally averaged Q and suppresses
//! the error this overfit otherwise produces. The noise is applied only
//! inside the target computation (not at env-interaction time) which is why
//! this helper lives in `td3`, not in [`super::super::ddpg::exploration`].
//!
//! The update rule is
//!
//! ```text
//! a' = clip(target_actor(s') + clip(N(0, σ²), -c, c), low, high)
//! ```
//!
//! where `σ = policy_noise` and `c = noise_clip`.

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

/// Applies target-policy smoothing to a batch of target-actor outputs.
///
/// Adds independent `N(0, policy_noise²)` noise to each element of
/// `target_action`, clips the noise to `[-noise_clip, +noise_clip]`, and
/// clips the summed result to `[low, high]`. The operation runs on the
/// target's (non-autodiff) backend; no gradients are produced.
///
/// # Panics
///
/// Panics if `policy_noise < 0`, `noise_clip < 0`, or `low > high`.
pub fn smoothed_target_action<BI, R, const DAB: usize>(
    target_action: Tensor<BI, DAB>,
    policy_noise: f32,
    noise_clip: f32,
    low: f32,
    high: f32,
    rng: &mut R,
) -> Tensor<BI, DAB>
where
    BI: Backend,
    R: Rng + ?Sized,
{
    assert!(
        policy_noise.is_finite() && policy_noise >= 0.0,
        "policy_noise must be finite and non-negative, got {policy_noise}"
    );
    assert!(
        noise_clip.is_finite() && noise_clip >= 0.0,
        "noise_clip must be finite and non-negative, got {noise_clip}"
    );
    assert!(
        low <= high,
        "low must be <= high, got low={low} high={high}"
    );

    let device = target_action.device();
    let dims: [usize; DAB] = target_action.dims();
    let numel: usize = dims.iter().product();

    let normal = StandardNormal;
    let noise_vec: Vec<f32> = (0..numel)
        .map(|_| {
            let eps: f64 = normal.sample(rng);
            let scaled = policy_noise * (eps as f32);
            scaled.clamp(-noise_clip, noise_clip)
        })
        .collect();
    let noise_tensor: Tensor<BI, DAB> =
        Tensor::from_data(TensorData::new(noise_vec, dims.to_vec()), &device);

    (target_action + noise_tensor).clamp(low, high)
}

#[cfg(test)]
mod tests {
    use super::*;

    use burn::backend::NdArray;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    type B = NdArray;

    #[test]
    fn zero_policy_noise_reduces_to_action_clip() {
        let device = Default::default();
        let action: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(vec![0.3_f32, -5.0], vec![1, 2]), &device);
        let mut rng = StdRng::seed_from_u64(42);
        let out = smoothed_target_action::<B, _, 2>(
            action, 0.0, 0.5, -1.0, 1.0, &mut rng,
        );
        let data = out.into_data().convert::<f32>();
        let slice = data.as_slice::<f32>().unwrap();
        // `0.3` passes through; `-5.0` clips to `-1.0`.
        assert!((slice[0] - 0.3).abs() < 1e-6, "got {}", slice[0]);
        assert!((slice[1] + 1.0).abs() < 1e-6, "got {}", slice[1]);
    }

    #[test]
    fn smoothing_noise_clipped_symmetrically() {
        // Huge σ → raw noise frequently exceeds `noise_clip`; the helper
        // must always keep it inside `[-noise_clip, +noise_clip]`. We
        // choose action = 0 and a wide `[low, high]` so the outer clip is
        // inactive and the output is the clipped noise itself.
        let device = Default::default();
        let noise_clip: f32 = 0.5;
        let low = -100.0_f32;
        let high = 100.0_f32;
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..256 {
            let action: Tensor<B, 2> = Tensor::zeros([4, 2], &device);
            let out = smoothed_target_action::<B, _, 2>(
                action,
                100.0,
                noise_clip,
                low,
                high,
                &mut rng,
            );
            let data = out.into_data().convert::<f32>();
            for v in data.as_slice::<f32>().unwrap() {
                assert!(
                    (*v) >= -noise_clip - 1e-6 && (*v) <= noise_clip + 1e-6,
                    "value {v} outside [-{noise_clip}, +{noise_clip}]"
                );
            }
        }
    }

    #[test]
    fn final_action_clipped_to_bounds() {
        // Push the mean outside the bounds; the outer clip should force
        // every output into `[low, high]`.
        let device = Default::default();
        let low = -1.0_f32;
        let high = 1.0_f32;
        let mut rng = StdRng::seed_from_u64(0);
        let action: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(vec![10.0_f32; 6], vec![3, 2]), &device);
        let out = smoothed_target_action::<B, _, 2>(
            action, 0.1, 0.1, low, high, &mut rng,
        );
        let data = out.into_data().convert::<f32>();
        for v in data.as_slice::<f32>().unwrap() {
            assert!((*v - high).abs() < 1e-6, "expected clip to {high}, got {v}");
        }
    }

    #[test]
    #[should_panic(expected = "policy_noise must be finite and non-negative")]
    fn rejects_negative_policy_noise() {
        let device = Default::default();
        let action: Tensor<B, 2> = Tensor::zeros([1, 1], &device);
        let mut rng = StdRng::seed_from_u64(0);
        let _ = smoothed_target_action::<B, _, 2>(
            action, -0.1, 0.5, -1.0, 1.0, &mut rng,
        );
    }

    #[test]
    #[should_panic(expected = "noise_clip must be finite and non-negative")]
    fn rejects_negative_noise_clip() {
        let device = Default::default();
        let action: Tensor<B, 2> = Tensor::zeros([1, 1], &device);
        let mut rng = StdRng::seed_from_u64(0);
        let _ = smoothed_target_action::<B, _, 2>(
            action, 0.2, -0.1, -1.0, 1.0, &mut rng,
        );
    }
}
