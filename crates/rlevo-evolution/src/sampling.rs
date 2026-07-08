//! NaN-safe Gaussian sampling helpers shared across evolution operators.
//!
//! These draw from a caller-owned host RNG (typically produced by
//! [`crate::rng::seed_stream`]); they never touch backend-global RNG state,
//! preserving the crate's reproducibility convention.

use rand::Rng;
use rand_distr::{Distribution as _, Normal, StandardNormal};

/// Draws one `N(0, 1)` sample from a caller-owned host `rng`.
///
/// Uses [`rand_distr::StandardNormal`], which has no fallible constructor — the
/// unit normal is always well-defined, so there is no `expect`/`unwrap` and no
/// panic path. The sample is drawn as `f64` and narrowed to `f32` to preserve
/// tail precision, matching the RL crate's `standard_normal_tensor` convention.
#[allow(
    clippy::cast_possible_truncation,
    reason = "narrowing f64 -> f32 is the intended tail-precision convention"
)]
pub(crate) fn standard_normal<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    let x: f64 = StandardNormal.sample(rng);
    x as f32
}

/// Draws one `N(mean, std)` sample, falling back to `mean` on a degenerate σ.
///
/// Guards against [`Normal::new`] failure by returning `mean` — the degenerate,
/// zero-perturbation draw — instead of panicking. In `rand_distr` 0.6,
/// [`Normal::new`] fails only when `std` is **non-finite** (`NaN` or `±∞`); a
/// negative-but-finite `std` is accepted (it mirrors the distribution but still
/// yields finite samples) and `std = 0.0` is accepted (every draw is exactly
/// `mean`). A `NaN` `mean` passes through unchanged and is *not* laundered here;
/// it is neutralized downstream by the ADR-0034 fitness-hygiene chokepoint.
///
/// An `rng` draw is consumed only on the `Ok` path: constructing the [`Normal`]
/// distribution does not touch the `rng`, so the fallback leaves the stream
/// position unchanged. This matches the previous
/// `Normal::new(...).expect(...); .sample(rng)` behavior exactly on the success
/// path.
pub(crate) fn normal_or_mean<R: Rng + ?Sized>(mean: f32, std: f32, rng: &mut R) -> f32 {
    match Normal::new(mean, std) {
        Ok(d) => d.sample(rng),
        Err(_) => mean,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn standard_normal_is_finite() {
        let mut rng: StdRng = StdRng::seed_from_u64(7);
        for _ in 0..1_000 {
            let x: f32 = standard_normal(&mut rng);
            assert!(x.is_finite(), "standard_normal produced non-finite {x}");
        }
    }

    #[test]
    fn standard_normal_same_seed_is_deterministic() {
        let mut a: StdRng = StdRng::seed_from_u64(42);
        let mut b: StdRng = StdRng::seed_from_u64(42);
        for _ in 0..64 {
            let xa: f32 = standard_normal(&mut a);
            let xb: f32 = standard_normal(&mut b);
            assert_eq!(xa.to_bits(), xb.to_bits());
        }
    }

    #[test]
    fn standard_normal_different_seeds_differ() {
        let mut a: StdRng = StdRng::seed_from_u64(1);
        let mut b: StdRng = StdRng::seed_from_u64(2);
        let seq_a: Vec<u32> = (0..16).map(|_| standard_normal(&mut a).to_bits()).collect();
        let seq_b: Vec<u32> = (0..16).map(|_| standard_normal(&mut b).to_bits()).collect();
        assert_ne!(seq_a, seq_b);
    }

    #[test]
    fn normal_or_mean_falls_back_on_nonfinite_std() {
        // `rand_distr::Normal::new` fails only for a non-finite `std`, so those
        // are the cases that trigger the `mean` fallback.
        let mean: f32 = 3.5;
        for std in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            let mut rng: StdRng = StdRng::seed_from_u64(11);
            let out: f32 = normal_or_mean(mean, std, &mut rng);
            assert_eq!(
                out.to_bits(),
                mean.to_bits(),
                "expected fallback to mean for std = {std}"
            );
        }
    }

    #[test]
    fn normal_or_mean_zero_std_yields_exact_mean() {
        // A zero variance is a valid (degenerate) distribution: every draw is
        // exactly `mean`. It goes through the `Ok` path and consumes an rng draw.
        let mean: f32 = -2.25;
        let mut rng: StdRng = StdRng::seed_from_u64(17);
        let out: f32 = normal_or_mean(mean, 0.0, &mut rng);
        assert_eq!(out.to_bits(), mean.to_bits());
    }

    #[test]
    fn normal_or_mean_samples_finite_for_finite_std() {
        // Valid positive σ, plus a negative-but-finite σ (accepted by
        // `rand_distr`, mirrors the distribution): both must yield finite draws.
        let mut rng: StdRng = StdRng::seed_from_u64(13);
        for _ in 0..1_000 {
            let x: f32 = normal_or_mean(0.0, 1.0, &mut rng);
            let y: f32 = normal_or_mean(0.0, -1.0, &mut rng);
            assert!(x.is_finite(), "normal_or_mean produced non-finite {x}");
            assert!(y.is_finite(), "normal_or_mean produced non-finite {y}");
        }
    }
}
