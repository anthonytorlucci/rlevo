//! Gaussian exploration noise for DDPG.
//!
//! CleanRL's `ddpg_continuous_action.py` defaults to additive Gaussian noise
//! rather than the Ornstein–Uhlenbeck process of the original paper; it is
//! simpler to reason about, stateless, and empirically no worse on Gym
//! continuous-control tasks. [`GaussianNoise`] is a thin newtype over the
//! standard deviation that knows how to add noise to an action row and clip
//! the result back into the valid action range.

use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

/// Per-component Gaussian exploration noise with a fixed standard deviation.
#[derive(Clone, Copy, Debug)]
pub struct GaussianNoise {
    sigma: f32,
}

impl GaussianNoise {
    /// Creates a new noise sampler with standard deviation `sigma`.
    ///
    /// `sigma = 0.0` disables exploration and makes [`apply`](Self::apply) a
    /// pure clip (useful for evaluation rollouts).
    #[must_use]
    pub fn new(sigma: f32) -> Self {
        assert!(sigma.is_finite(), "sigma must be finite");
        assert!(sigma >= 0.0, "sigma must be non-negative, got {sigma}");
        Self { sigma }
    }

    /// The current standard deviation.
    #[must_use]
    pub fn sigma(&self) -> f32 {
        self.sigma
    }

    /// Adds independent `N(0, sigma²)` noise to each component of `mean` and
    /// clips the result into `[low[i], high[i]]`.
    ///
    /// # Panics
    ///
    /// Panics if `mean`, `low`, and `high` do not all have the same length.
    pub fn apply<R: Rng + ?Sized>(
        &self,
        mean: &[f32],
        low: &[f32],
        high: &[f32],
        rng: &mut R,
    ) -> Vec<f32> {
        assert_eq!(mean.len(), low.len(), "mean/low length mismatch");
        assert_eq!(mean.len(), high.len(), "mean/high length mismatch");
        let normal = StandardNormal;
        mean.iter()
            .zip(low.iter())
            .zip(high.iter())
            .map(|((&m, &lo), &hi)| {
                let eps: f64 = normal.sample(rng);
                let noisy = m + self.sigma * (eps as f32);
                noisy.clamp(lo, hi)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn gaussian_noise_zero_sigma_is_identity_then_clip() {
        let noise = GaussianNoise::new(0.0);
        let mut rng = StdRng::seed_from_u64(42);
        let out = noise.apply(&[0.3, -0.7], &[-1.0, -1.0], &[1.0, 1.0], &mut rng);
        assert_eq!(out, vec![0.3, -0.7]);
    }

    #[test]
    fn gaussian_noise_respects_clip_bounds() {
        // Huge sigma → noise will frequently exceed bounds; every emitted
        // component must still land inside [low, high].
        let noise = GaussianNoise::new(100.0);
        let mut rng = StdRng::seed_from_u64(7);
        let mean = vec![0.0_f32; 4];
        let low = vec![-2.0, -1.0, -0.5, -3.0];
        let high = vec![2.0, 1.0, 0.5, 3.0];
        for _ in 0..1_000 {
            let out = noise.apply(&mean, &low, &high, &mut rng);
            for (i, v) in out.iter().enumerate() {
                assert!(*v >= low[i] && *v <= high[i], "bound {i}: value {v}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "sigma must be non-negative")]
    fn gaussian_noise_rejects_negative_sigma() {
        let _ = GaussianNoise::new(-0.1);
    }

    #[test]
    #[should_panic(expected = "mean/low length mismatch")]
    fn gaussian_noise_rejects_length_mismatch() {
        let noise = GaussianNoise::new(0.1);
        let mut rng = StdRng::seed_from_u64(0);
        let _ = noise.apply(&[0.0, 1.0], &[-1.0], &[1.0, 1.0], &mut rng);
    }
}
