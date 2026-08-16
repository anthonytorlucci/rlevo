//! Fitness shaping transforms.
//!
//! Strategies occasionally benefit from monotone transforms of raw
//! fitness values: centered-rank mapping flattens outliers, z-scoring
//! normalizes scale across generations, and weight decay penalizes
//! large-norm genomes (common in neuroevolution). These helpers work
//! directly on device tensors and are pure functions (RNG-free).

use burn::prelude::ElementConversion;
use burn::tensor::{Tensor, backend::Backend};

/// Error returned by fallible fitness-shaping transforms.
#[derive(Debug, thiserror::Error)]
pub enum ShapingError {
    /// The input tensor's element data could not be read as `f32` — e.g. an
    /// integer-typed backend tensor was passed to a transform that ranks host
    /// values. This is a backend/dtype mismatch, surfaced as a recoverable error
    /// rather than a panic.
    #[error("shaping transform requires f32 tensor data")]
    NonFloatData,
}

/// Std-dev floor, in **raw fitness units**, applied so a degenerate (all-equal)
/// population maps to zeros instead of dividing by zero.
const STD_FLOOR: f32 = 1e-8;

/// Returns `fitness - fitness.mean()` divided by the (population) std-dev,
/// clamped to avoid divide-by-zero when all fitnesses are equal.
///
/// The std-dev floor is `1e-8` in raw fitness units; degenerate populations
/// (all-equal fitness) therefore map to a vector of zeros rather than producing
/// NaNs.
///
/// # Saturated fitness
///
/// ADR 0034 maps a raw `+∞` fitness to `f32::MAX`, which is *finite* and so
/// flows into this reduction. Computing the variance directly in `f32` overflows
/// **the individual squared term** for such a member — `(f32::MAX − mean)² = +∞`
/// — at `N = 1` saturated member, before any accumulation happens. That drives
/// `std` to `+∞` and collapses *every* output element to `±0.0`: a finite,
/// plausible, silent zero gradient for the ES update this transform feeds.
///
/// ADR 0069 §Decision 1's remedy — accumulate in `f64` — is not reachable
/// through a Burn tensor, whose accumulator width is `B::FloatElem` and fixed by
/// the backend. The equivalent guarantee is obtained here by **bounding the
/// terms instead of widening the accumulator**: the population is divided by its
/// own max-abs magnitude before the reduction, so every scaled value lies in
/// `[−1, 1]`, the mean in `[−1, 1]`, and every squared centred term in `[0, 4]`.
/// For a **finite** population neither the mean nor the variance can then
/// overflow at any size that fits in memory — on this backend or on a narrower
/// one. The bound is conditional: a non-finite max-abs falls back to a scale of
/// `1.0` and the guarantee lapses, which is exactly the `−∞` case described
/// under *Non-finite inputs* below. z-scoring is invariant
/// to a positive rescale, so the mathematical result is unchanged (the extra
/// division perturbs ordinary inputs by at most a few ULP), and the `1e-8` floor
/// is divided by the same scale so it keeps its raw-fitness meaning.
///
/// `fitness::sanitized_mean` / `sanitized_sum` (ADR 0069 §Decision 2) are
/// deliberately *not* used here: they reduce a host `f32` iterator, and reaching
/// for them would mean pulling the whole population off the device — the
/// round-trip `sanitize_fitness_tensor` exists to avoid (`rules.md` §3).
///
/// # Non-finite inputs
///
/// A `−∞` member is legal here — it is ADR 0034's worst-value sentinel and
/// passes sanitization through unchanged. `z_score` does **not** handle it and
/// never has: the mean is `−∞`, so every finite member centres to `+∞` and the
/// `−∞` member itself centres to `NaN`. That behaviour is unchanged by the
/// saturation fix above (a non-finite max-abs falls back to a scale of `1.0`,
/// reproducing the original arithmetic exactly) and is pinned by
/// `z_score_negative_infinity_member_is_unchanged_by_the_saturation_fix`.
/// Deciding what a `−∞` member *should* shape to is a separate question.
///
/// # Examples
///
/// ```
/// use burn::backend::Flex;
/// use burn::tensor::Tensor;
/// use rlevo_evolution::shaping::z_score;
///
/// let device = Default::default();
/// // Five fitness values: mean 3.0, all distinct.
/// let t = Tensor::<Flex, 1>::from_floats([1.0f32, 2.0, 3.0, 4.0, 5.0], &device);
/// let z = z_score(t);
/// let values = z.into_data().into_vec::<f32>().expect("shaped tensor must be readable as f32");
/// // After z-scoring the mean of the output is 0 (within floating-point tolerance).
/// let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
/// assert!(mean.abs() < 1e-5);
/// ```
#[must_use]
pub fn z_score<B: Backend>(fitness: Tensor<B, 1>) -> Tensor<B, 1> {
    let n = fitness.dims()[0];
    // A max reduction cannot overflow, so this is safe to take on raw fitness
    // (unlike the sum below). See the `# Saturated fitness` section above.
    let scale_raw = fitness.clone().max_abs().into_scalar().elem::<f32>();
    // Non-finite max ⇒ the input carries a raw `±∞`; zero max ⇒ an all-zero
    // population; an empty population has no max at all. A scale of `1.0`
    // reproduces the unscaled arithmetic bit-for-bit, so none of those three
    // cases changes behaviour relative to the pre-ADR-0069 formula.
    let scale = if scale_raw.is_finite() && scale_raw > 0.0 {
        scale_raw
    } else {
        1.0
    };

    let scaled = fitness / scale;
    let mean = scaled.clone().mean().into_scalar().elem::<f32>();
    #[allow(clippy::cast_precision_loss)]
    let n_f = n.max(1) as f32;
    let centered = scaled - mean;
    let var = centered
        .clone()
        .powf_scalar(2.0)
        .sum()
        .into_scalar()
        .elem::<f32>()
        / n_f;
    // `centered` is in scaled units, so the raw-unit floor is scaled too. The
    // second clamp covers the one case where that quotient underflows to zero
    // (`STD_FLOOR / f32::MAX`, i.e. an entirely saturated population): `var` is
    // then also zero, and dividing 0 by a positive number gives the documented
    // all-equal answer instead of `0 / 0 = NaN`.
    let floor = (STD_FLOOR / scale).max(f32::MIN_POSITIVE);
    let std = var.sqrt().max(floor);
    centered / std
}

/// Returns centered ranks: the largest fitness maps to `+0.5`, the
/// smallest to `-0.5`, with linear spacing in between.
///
/// Under the crate's maximise convention (canonical: higher is better)
/// this assigns the **best** (highest-fitness) individual the highest
/// utility `+0.5` and the worst the lowest `-0.5`, which is the sign a
/// gradient-style ES update expects — no negation at the call site.
///
/// Centered ranks are standard in modern ES (e.g. OpenAI-ES) because they
/// remove outlier fitness magnitudes and keep the signal scale-free across
/// generations. Implemented host-side because the argsort pathway is
/// easier to reason about; swap in a tensor-native implementation if this
/// ever shows up on a profile.
///
/// An empty input returns an empty tensor.
///
/// # Examples
///
/// ```
/// use burn::backend::Flex;
/// use burn::tensor::Tensor;
/// use rlevo_evolution::shaping::centered_rank;
///
/// let device = Default::default();
/// let t = Tensor::<Flex, 1>::from_floats([10.0f32, 20.0, 30.0, 40.0], &device);
/// let r = centered_rank(t, &device).unwrap();
/// let values = r.into_data().into_vec::<f32>().expect("shaped tensor must be readable as f32");
/// // Smallest value maps to -0.5, largest to +0.5.
/// assert!((values[0] - (-0.5)).abs() < 1e-6);
/// assert!((values[3] - 0.5).abs() < 1e-6);
/// ```
///
/// # Errors
///
/// Returns [`ShapingError::NonFloatData`] if the tensor's element data cannot be
/// read as `f32` — for example, when using a backend that stores integer-typed
/// tensors and `into_vec::<f32>()` fails.
pub fn centered_rank<B: Backend>(
    fitness: Tensor<B, 1>,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Result<Tensor<B, 1>, ShapingError> {
    let raw = fitness
        .into_data()
        .into_vec::<f32>()
        .map_err(|_| ShapingError::NonFloatData)?;
    let n = raw.len();
    if n == 0 {
        return Ok(Tensor::<B, 1>::from_floats([0.0f32; 0], device));
    }
    // Sanitize NaN → −inf (worst under maximise) so a NaN fitness ranks lowest
    // rather than corrupting the ascending order.
    let data: Vec<f32> = raw
        .iter()
        .map(|&f| crate::fitness::sanitize_fitness(f))
        .collect();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&i, &j| data[i].total_cmp(&data[j]));

    #[allow(clippy::cast_precision_loss)]
    let n_f = (n - 1).max(1) as f32;
    let mut ranks = vec![0.0_f32; n];
    for (rank, &idx) in indices.iter().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let r = rank as f32 / n_f - 0.5;
        ranks[idx] = r;
    }
    Ok(Tensor::<B, 1>::from_floats(ranks.as_slice(), device))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;
    type TestBackend = Flex;

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn z_score_zero_mean_unit_std() {
        let device = Default::default();
        let t = Tensor::<TestBackend, 1>::from_floats([1.0f32, 2.0, 3.0, 4.0, 5.0], &device);
        let z = z_score(t);
        let values = z
            .into_data()
            .into_vec::<f32>()
            .expect("shaped tensor host-read of a tensor this test just built");
        let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
        approx::assert_relative_eq!(mean, 0.0, epsilon = 1e-5);
    }

    /// Reads a shaped tensor back to host `f32` for assertion.
    fn host(t: Tensor<TestBackend, 1>) -> Vec<f32> {
        t.into_data()
            .into_vec::<f32>()
            .expect("shaped tensor host-read of a tensor this test just built")
    }

    /// Control (ADR 0069 §Context): nine zeros and one merely-large-but-safe
    /// top member. The z-score of `[0; 9] ++ [M]` is `[-1/3; 9] ++ [3.0]` for
    /// **any** `M > 0`, because z-scoring is scale-invariant. This test pins the
    /// ordinary-input behaviour that the saturated cases below must match.
    #[test]
    fn z_score_control_large_finite_top_member() {
        let device = Default::default();
        let mut raw = [0.0f32; 10];
        raw[9] = 1e18;
        let values = host(z_score(Tensor::<TestBackend, 1>::from_floats(raw, &device)));
        for (i, v) in values.iter().take(9).enumerate() {
            approx::assert_relative_eq!(*v, -1.0 / 3.0, epsilon = 1e-5);
            assert!(v.is_finite(), "control element {i} must be finite");
        }
        approx::assert_relative_eq!(values[9], 3.0, epsilon = 1e-5);
    }

    /// A **single** `f32::MAX` member (ADR 0034's sanitized `+∞`) must shape
    /// identically to the control above — `powf_scalar(2.0)` overflows that one
    /// member's squared term to `+∞` before any accumulation, so the `f32`
    /// formula collapses the whole vector to `±0.0` at `N = 1` saturated member.
    /// A silent zero gradient, not a `NaN`. (ADR 0069 §Decision 4.)
    #[test]
    fn z_score_single_saturated_member_matches_control() {
        let device = Default::default();
        let mut raw = [0.0f32; 10];
        raw[9] = f32::MAX;
        let values = host(z_score(Tensor::<TestBackend, 1>::from_floats(raw, &device)));
        for v in values.iter().take(9) {
            approx::assert_relative_eq!(*v, -1.0 / 3.0, epsilon = 1e-5);
        }
        approx::assert_relative_eq!(values[9], 3.0, epsilon = 1e-5);
    }

    /// An entirely saturated population is *degenerate*, not pathological: every
    /// member is equal, so the documented all-equal behaviour (a vector of zeros
    /// via the std-dev floor) is the correct answer. The `f32` formula instead
    /// blows the mean to `+∞` at `N = 2` and yields `NaN`.
    #[test]
    fn z_score_all_saturated_is_degenerate_zeros() {
        let device = Default::default();
        let values = host(z_score(Tensor::<TestBackend, 1>::from_floats(
            [f32::MAX; 4],
            &device,
        )));
        for (i, v) in values.iter().enumerate() {
            assert!(
                v.is_finite(),
                "all-saturated element {i} must be finite, got {v}"
            );
            approx::assert_relative_eq!(*v, 0.0, epsilon = 1e-6);
        }
    }

    /// A mix of saturated and ordinary members. Two `f32::MAX` and two `1.0`
    /// scale to `[1, 1, ~0, ~0]`, whose z-score is exactly `[1, 1, -1, -1]`.
    #[test]
    fn z_score_mixed_saturated_and_finite() {
        let device = Default::default();
        let values = host(z_score(Tensor::<TestBackend, 1>::from_floats(
            [f32::MAX, f32::MAX, 1.0, 1.0],
            &device,
        )));
        approx::assert_relative_eq!(values[0], 1.0, epsilon = 1e-5);
        approx::assert_relative_eq!(values[1], 1.0, epsilon = 1e-5);
        approx::assert_relative_eq!(values[2], -1.0, epsilon = 1e-5);
        approx::assert_relative_eq!(values[3], -1.0, epsilon = 1e-5);
    }

    /// Mirror image of the case above, with the saturation on the negative side.
    /// The scale must be the max **absolute** magnitude: a signed `max()` here
    /// would be `-1.0`, which is not a usable divisor and would send the whole
    /// vector back through the unscaled (overflowing) path.
    #[test]
    fn z_score_negatively_saturated_uses_max_abs_scale() {
        let device = Default::default();
        let values = host(z_score(Tensor::<TestBackend, 1>::from_floats(
            [-f32::MAX, -f32::MAX, -1.0, -1.0],
            &device,
        )));
        approx::assert_relative_eq!(values[0], -1.0, epsilon = 1e-5);
        approx::assert_relative_eq!(values[1], -1.0, epsilon = 1e-5);
        approx::assert_relative_eq!(values[2], 1.0, epsilon = 1e-5);
        approx::assert_relative_eq!(values[3], 1.0, epsilon = 1e-5);
    }

    /// An all-zero population has a max-abs of `0.0`, which must not be used as
    /// a divisor. It is the documented all-equal case and shapes to zeros.
    #[test]
    fn z_score_all_zero_population_is_zeros() {
        let device = Default::default();
        let values = host(z_score(Tensor::<TestBackend, 1>::from_floats(
            [0.0f32; 3],
            &device,
        )));
        for (i, v) in values.iter().enumerate() {
            assert!(
                v.is_finite(),
                "all-zero element {i} must be finite, got {v}"
            );
            approx::assert_relative_eq!(*v, 0.0, epsilon = 1e-6);
        }
    }

    /// Saturation must not perturb an ordinary population: rescaling by the
    /// max-abs magnitude leaves the mathematical result identical, so the
    /// existing example keeps its exact z-scores.
    #[test]
    fn z_score_ordinary_input_unchanged() {
        let device = Default::default();
        let values = host(z_score(Tensor::<TestBackend, 1>::from_floats(
            [1.0f32, 2.0, 3.0, 4.0, 5.0],
            &device,
        )));
        // mean 3, population std sqrt(2) → ±2/√2, ±1/√2, 0
        let expected = [
            -std::f32::consts::SQRT_2,
            -std::f32::consts::FRAC_1_SQRT_2,
            0.0,
            std::f32::consts::FRAC_1_SQRT_2,
            std::f32::consts::SQRT_2,
        ];
        for (got, want) in values.iter().zip(expected) {
            approx::assert_relative_eq!(*got, want, epsilon = 1e-5);
        }
    }

    /// **Pin, not a fix.** `−∞` is a legal input here (ADR 0034's worst-value
    /// sentinel passes through sanitization), and `z_score` has never handled it:
    /// the mean is `−∞`, so every finite member centres to `+∞` and the `−∞`
    /// member itself centres to `NaN`. This test records that pre-existing
    /// behaviour verbatim so the saturation fix is visibly *not* changing it —
    /// deciding what a `−∞` member should shape to is a separate, still-open
    /// policy question (`rules.md` §12). When it's settled, this test flips
    /// from pinning `[+∞, +∞, NaN]` to asserting the chosen semantics.
    #[test]
    fn z_score_negative_infinity_member_is_unchanged_by_the_saturation_fix() {
        let device = Default::default();
        let values = host(z_score(Tensor::<TestBackend, 1>::from_floats(
            [f32::MAX, 1.0, f32::NEG_INFINITY],
            &device,
        )));
        for (i, v) in values.iter().take(2).enumerate() {
            assert!(
                v.is_infinite() && v.is_sign_positive(),
                "finite member {i} centres to +inf against a -inf mean, got {v}"
            );
        }
        assert!(
            values[2].is_nan(),
            "the -inf member centres to NaN (-inf - -inf), got {}",
            values[2]
        );
    }

    /// An empty population has no max-abs magnitude; the scale fallback must
    /// keep the added `max_abs` reduction from turning that into a panic or a
    /// `NaN`-shaped result. (Behaviour verified identical pre- and post-fix.)
    #[test]
    fn z_score_empty_input_is_empty() {
        let device = Default::default();
        let z = z_score(Tensor::<TestBackend, 1>::from_floats([0.0f32; 0], &device));
        assert_eq!(z.dims()[0], 0, "empty input shapes to an empty output");
    }

    #[test]
    fn centered_rank_spans_half_interval() {
        let device = Default::default();
        let t = Tensor::<TestBackend, 1>::from_floats([10.0f32, 20.0, 30.0, 40.0], &device);
        let r = centered_rank(t, &device).unwrap();
        let values = r
            .into_data()
            .into_vec::<f32>()
            .expect("shaped tensor host-read of a tensor this test just built");
        // smallest → -0.5, largest → +0.5
        approx::assert_relative_eq!(values[0], -0.5, epsilon = 1e-6);
        approx::assert_relative_eq!(values[3], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn centered_rank_preserves_order() {
        let device = Default::default();
        let t = Tensor::<TestBackend, 1>::from_floats([3.0f32, 1.0, 2.0], &device);
        let r = centered_rank(t, &device).unwrap();
        let values = r
            .into_data()
            .into_vec::<f32>()
            .expect("shaped tensor host-read of a tensor this test just built");
        // original: 3, 1, 2 → ranks sorted ascending: [1, 2, 3] at indices [1, 2, 0]
        // rank-positions centered: index 1 → -0.5, index 2 → 0.0, index 0 → 0.5
        approx::assert_relative_eq!(values[1], -0.5, epsilon = 1e-6);
        approx::assert_relative_eq!(values[2], 0.0, epsilon = 1e-6);
        approx::assert_relative_eq!(values[0], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn centered_rank_empty_is_ok() {
        let device = Default::default();
        let t = Tensor::<TestBackend, 1>::from_floats([0.0f32; 0], &device);
        let r = centered_rank(t, &device).expect("empty input is not an error");
        assert_eq!(r.dims()[0], 0);
    }

    // ------------------------------------------------------------------
    // ADR 0069 §Decision 5 — behavioural rescale-invariance property.
    // ------------------------------------------------------------------

    use proptest::collection::vec as prop_vec;
    // Explicit list rather than the glob prelude, matching the crate's other
    // proptest modules.
    use proptest::prelude::{ProptestConfig, prop_assert, prop_assume, proptest};

    /// Population standard deviation of `xs`, in `f64`. This is the σ the bound
    /// on `c` is stated against; it is computed on the host in double precision
    /// precisely so it is independent of the `f32` device reduction under test.
    fn population_std(xs: &[f32]) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let n = xs.len() as f64;
        let mean = xs.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
        (xs.iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / n)
            .sqrt()
    }

    proptest! {
        // Each case runs three `z_score` calls on a `Flex` tensor of at most 16
        // elements. Device-touching but not backend-*heavy* (no strategy step),
        // so this sits between ADR 0036 §5's two tiers; `max_shrink_iters` is
        // capped per that section's backend-heavy guidance.
        #![proptest_config(ProptestConfig {
            cases: 32,
            max_shrink_iters: 256,
            ..ProptestConfig::default()
        })]

        /// **ADR 0069 §Decision 5, property 3.** `z_score` is invariant to a
        /// positive affine rescale `x ↦ c·(x + d)` of its input — **bounded in
        /// `c`**, for the reason derived below.
        ///
        /// This is a **device** reduction, so §Decision 4 applies rather than
        /// §Decision 1: the accumulator width belongs to `B::FloatElem`, and the
        /// guarantee comes from bounding the terms — dividing by the population's
        /// max-abs magnitude before centering. The property is what checks that
        /// bounding step, and it needs real `Tensor` ops to do so.
        ///
        /// # The bound on `c`
        ///
        /// Derived from `z_score`'s own arithmetic. Writing `σ` for the
        /// population std of the *raw* input and `M` for its max-abs magnitude:
        ///
        /// - `scale = M`, so `scaled = x / M` and the std of `scaled` is `σ / M`.
        /// - Under `x ↦ c·(x + d)` the scale becomes `M' = c·max|x + d|` and the
        ///   std of `scaled` becomes `c·σ / M'` — the shift cancels in the
        ///   centering and the `c` cancels against `M'`, so the *scaled* std is
        ///   invariant. That is what makes the transform a no-op.
        /// - The floor is not invariant: `floor = STD_FLOOR / M'`. It fires when
        ///   `floor > c·σ / M'`, i.e. when **`c·σ < STD_FLOOR = 1e-8`**.
        ///
        /// So the admissible range is
        ///
        /// > `STD_FLOOR / σ  ≤  c  ≤  f32::MAX / max|x|`
        ///
        /// — read as: the transformed population's spread must stay at or above
        /// the `1e-8` **raw-fitness-unit** floor, and the transformed values must
        /// stay representable. Below the lower bound `z_score` deliberately
        /// returns the degenerate all-zeros answer, so a property asserting
        /// unconditional invariance would fail on *correct* code; above the upper
        /// bound the input is no longer a rescaled population but a vector of
        /// `±∞`. The test takes `c` as a power of two in `[2^k_min, 2^k_max]`
        /// with a factor-2 margin on the lower end.
        ///
        /// The second clamp, `.max(f32::MIN_POSITIVE)`, is **`c`-independent**:
        /// it fires when `σ / M < f32::MIN_POSITIVE`, a property of the shape of
        /// `x` alone. Small-integer inputs keep `σ / M ≥ ~1e-3`, far clear of it.
        ///
        /// # Why the offset is inside the scale
        ///
        /// The transform is written `c·(x + d)`, not `c·x + d`, and the
        /// difference is not cosmetic — the second form has a **bound of its
        /// own**, which ADR 0069 §Decision 5 does not mention and which this
        /// property discovered while being written. An offset applied *after* the
        /// scale annihilates the population's spread in `f32` once
        /// `c·σ ≲ f32::EPSILON·|d|`: at `c = 2^-24`, `d = 1`, `x = [0, 1]`, both
        /// members round to exactly `1.0`, and `z_score` correctly returns the
        /// degenerate all-zeros answer for a population that really has become
        /// degenerate. That is a representability limit of the *input*, not a
        /// property of `z_score`, so folding the offset into the scale removes it
        /// rather than papering over it with a tolerance — and for integer `x`,
        /// integer `d` and a power-of-two `c`, `c·(x + d)` is computed with **no
        /// rounding at all**.
        ///
        /// `2^k_max` — where the members sit at the very top of the `f32` range
        /// and the unbounded formula's squared centered term overflows at `N = 1`
        /// — is asserted on **every** case, not only when `headroom` shrinks to
        /// zero. That is the point of the property.
        #[test]
        fn prop_z_score_is_invariant_to_bounded_positive_affine_rescale(
            xs in prop_vec(-100i32..=100, 2..=16),
            shift in -100i32..=100,
            headroom in 0u32..=160,
        ) {
            // A degenerate (all-equal) population is the case the floor exists
            // for; it has no positive `c` window at all (σ = 0), so it is out of
            // the property's domain by construction rather than by tolerance.
            prop_assume!(xs.iter().any(|&v| v != xs[0]));

            #[allow(clippy::cast_precision_loss)]
            let base: Vec<f32> = xs.iter().map(|&v| v as f32).collect();
            #[allow(clippy::cast_precision_loss)]
            let d = shift as f32;
            // Integers in `[-100, 100]` plus a shift in `[-100, 100]` are exact
            // in `f32`, and a power-of-two `c` scales them without error, so
            // `c·(x + d)` carries no rounding whatsoever. See "Why the offset is
            // inside the scale" above.
            let affine = |c: f32| -> Vec<f32> { base.iter().map(|&v| c * (v + d)).collect() };

            let max_abs = base.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
            let sigma = population_std(&base);

            #[allow(clippy::cast_possible_truncation)]
            let mut k_max = (f64::from(f32::MAX) / f64::from(max_abs)).log2().floor() as i32;
            // The closed form ignores `d`; step down until the transform really
            // is representable, so the property never strays past its own bound.
            while affine(2.0_f32.powi(k_max)).iter().any(|v| !v.is_finite()) {
                k_max -= 1;
            }
            // `c·σ ≥ 2·STD_FLOOR` — the derived lower bound with a factor-2 margin.
            #[allow(clippy::cast_possible_truncation)]
            let k_min = (2.0 * f64::from(STD_FLOOR) / sigma).log2().ceil() as i32;
            // Non-empty by construction for this generator (`k_min ≈ -24`,
            // `k_max ≈ 120`); asserted so an inverted window reads as a failed
            // property rather than as a `clamp` panic.
            prop_assert!(k_min <= k_max, "empty admissible window [2^{k_min}, 2^{k_max}]");
            #[allow(clippy::cast_possible_wrap)]
            let k = (k_max - headroom as i32).clamp(k_min, k_max);

            let device = Default::default();
            let z_base = host(z_score(Tensor::<TestBackend, 1>::from_floats(
                base.as_slice(),
                &device,
            )));
            prop_assert!(
                z_base.iter().all(|v| v.is_finite()),
                "baseline z-score must be finite, got {z_base:?}"
            );

            for exponent in [k_max, k] {
                let c = 2.0_f32.powi(exponent);
                prop_assert!(
                    f64::from(c) * sigma >= f64::from(STD_FLOOR),
                    "c = 2^{exponent} is below the std-floor bound; the property \
                     would be asserting something false of correct code"
                );
                let z = host(z_score(Tensor::<TestBackend, 1>::from_floats(
                    affine(c).as_slice(),
                    &device,
                )));
                for (i, (got, want)) in z.iter().zip(&z_base).enumerate() {
                    // Bit-exact whenever `d == 0` (a power-of-two rescale cancels
                    // exactly against the max-abs divisor); the tolerance covers
                    // only the centering cancellation the shift introduces, whose
                    // worst case here is `(max|x + d| / σ)·f32::EPSILON ≈ 1e-4`.
                    prop_assert!(
                        (got - want).abs() <= 5e-3,
                        "element {i} moved from {want} to {got} under \
                         x -> 2^{exponent}·(x + {d})"
                    );
                }
            }
        }
    }
}
