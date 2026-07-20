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
//! where `σ = policy_noise` and `c = noise_clip`. `c` is a scalar; `low` and
//! `high` are the per-component `Box` bound vectors (ADR 0053 §6).

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::algorithms::shared::clip_to_action_bounds;

/// Applies target-policy smoothing to a batch of target-actor outputs.
///
/// Adds independent `N(0, policy_noise²)` noise to each element of
/// `target_action`, clips the noise to the **scalar** range
/// `[-noise_clip, +noise_clip]`, and clips the summed result **per component**
/// against `low` / `high`. The operation runs on the target's (non-autodiff)
/// backend; no gradients are produced.
///
/// # Arguments
///
/// - `noise_clip` — the symmetric magnitude limit `c` on the smoothing noise.
///   This is a scalar in Eq. 14 and stays one here: it bounds how far the
///   target may be perturbed, which is a property of the smoothing, not of the
///   action space.
/// - `low` / `high` — the `Box(low, high)` bound tensors, shaped
///   `[1, ..action_shape]` so they broadcast across the batch. Per-component,
///   because Eq. 14's outer clip is against the bound *vectors*: an asymmetric
///   space (e.g. `CarRacing`'s `Box([-1,0,0], [1,1,1])`) has no single scalar
///   that expresses it, and a scalar collapse would pass impossible actions —
///   negative gas, negative brake — to the critic. Burn's `clamp` is
///   scalar-only, so the clip goes through `max_pair` / `min_pair`.
///
/// # Panics
///
/// Panics if `policy_noise < 0` or `noise_clip < 0`. The `low <= high` ordering
/// is a [`BoundedAction`](rlevo_core::action::BoundedAction) invariant checked
/// at agent construction, not here.
// `rand`'s standard-normal sampler yields f64; the tensor being filled is f32.
// Narrowing to the tensor's own dtype is the intent, and the sample is finite
// by construction.
#[allow(clippy::cast_possible_truncation)]
pub fn smoothed_target_action<BI, R, const DAB: usize>(
    target_action: Tensor<BI, DAB>,
    policy_noise: f32,
    noise_clip: f32,
    low: Tensor<BI, DAB>,
    high: Tensor<BI, DAB>,
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

    clip_to_action_bounds(target_action + noise_tensor, low, high)
}

#[cfg(test)]
mod tests {
    use super::*;

    use burn::backend::Flex;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    type B = Flex;

    /// Builds a `[1, C]` bound tensor from the per-component limits.
    fn bounds(values: &[f32]) -> Tensor<B, 2> {
        let device = Default::default();
        Tensor::from_data(
            TensorData::new(values.to_vec(), vec![1, values.len()]),
            &device,
        )
    }

    /// Uniform bounds repeated across `c` components — the symmetric case the
    /// old scalar signature covered.
    fn uniform_bounds(low: f32, high: f32, c: usize) -> (Tensor<B, 2>, Tensor<B, 2>) {
        (bounds(&vec![low; c]), bounds(&vec![high; c]))
    }

    #[test]
    fn zero_policy_noise_reduces_to_action_clip() {
        let device = Default::default();
        let action: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(vec![0.3_f32, -5.0], vec![1, 2]), &device);
        let mut rng = StdRng::seed_from_u64(42);
        let (low, high) = uniform_bounds(-1.0, 1.0, 2);
        let out = smoothed_target_action::<B, _, 2>(action, 0.0, 0.5, low, high, &mut rng);
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
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..256 {
            let action: Tensor<B, 2> = Tensor::zeros([4, 2], &device);
            let (low, high) = uniform_bounds(-100.0, 100.0, 2);
            let out =
                smoothed_target_action::<B, _, 2>(action, 100.0, noise_clip, low, high, &mut rng);
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
        let high_value = 1.0_f32;
        let mut rng = StdRng::seed_from_u64(0);
        let action: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(vec![10.0_f32; 6], vec![3, 2]), &device);
        let (low, high) = uniform_bounds(-1.0, high_value, 2);
        let out = smoothed_target_action::<B, _, 2>(action, 0.1, 0.1, low, high, &mut rng);
        let data = out.into_data().convert::<f32>();
        for v in data.as_slice::<f32>().unwrap() {
            assert!(
                (*v - high_value).abs() < 1e-6,
                "expected clip to {high_value}, got {v}"
            );
        }
    }

    #[test]
    fn asymmetric_bounds_clip_each_component_independently() {
        // Regression witness for ADR 0053 §6. CarRacing's action space is
        // `Box([-1,0,0], [1,1,1])`: steering is signed, gas and brake are not.
        // A scalar clip on `low[0]`/`high[0]` = `-1`/`1` — the pre-fix
        // behaviour — leaves the -1 in components 1 and 2 untouched, handing
        // the critic a negative gas and a negative brake: actions the
        // environment cannot execute. Only a per-component clip floors them
        // at 0.
        let device = Default::default();
        let low = bounds(&[-1.0, 0.0, 0.0]);
        let high = bounds(&[1.0, 1.0, 1.0]);
        let mut rng = StdRng::seed_from_u64(0);
        // Two rows, so the `[1, 3]` bounds must broadcast over the batch.
        let action: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(vec![-5.0_f32, -5.0, -5.0, 5.0, 5.0, 5.0], vec![2, 3]),
            &device,
        );
        // Zero policy noise isolates the outer clip from the smoothing.
        let out = smoothed_target_action::<B, _, 2>(action, 0.0, 0.5, low, high, &mut rng);
        let data = out.into_data().convert::<f32>();
        let slice = data.as_slice::<f32>().unwrap();

        assert!(
            (slice[0] + 1.0).abs() < 1e-6,
            "steering must clip to its own low of -1, got {}",
            slice[0]
        );
        for (i, name) in [(1_usize, "gas"), (2, "brake")] {
            assert!(
                slice[i] >= -1e-6,
                "{name} has a lower bound of 0 and must never go negative, got {}",
                slice[i]
            );
        }
        for (i, expected) in [(3_usize, 1.0_f32), (4, 1.0), (5, 1.0)] {
            assert!(
                (slice[i] - expected).abs() < 1e-6,
                "component {i} must clip to its upper bound {expected}, got {}",
                slice[i]
            );
        }
    }

    #[test]
    #[should_panic(expected = "policy_noise must be finite and non-negative")]
    fn rejects_negative_policy_noise() {
        let device = Default::default();
        let action: Tensor<B, 2> = Tensor::zeros([1, 1], &device);
        let mut rng = StdRng::seed_from_u64(0);
        let (low, high) = uniform_bounds(-1.0, 1.0, 1);
        let _ = smoothed_target_action::<B, _, 2>(action, -0.1, 0.5, low, high, &mut rng);
    }

    #[test]
    #[should_panic(expected = "noise_clip must be finite and non-negative")]
    fn rejects_negative_noise_clip() {
        let device = Default::default();
        let action: Tensor<B, 2> = Tensor::zeros([1, 1], &device);
        let mut rng = StdRng::seed_from_u64(0);
        let (low, high) = uniform_bounds(-1.0, 1.0, 1);
        let _ = smoothed_target_action::<B, _, 2>(action, 0.2, -0.1, low, high, &mut rng);
    }
}
