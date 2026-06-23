//! Standardised acceptance assertions for the algorithm integration tests.
//!
//! These cover the checks every algorithm test converges on: a learned policy
//! beats a *measured* random baseline ([`assert_improves_over_random`]) or
//! reaches an absolute floor ([`assert_reaches`]), and a seeded run is
//! reproducible either as a single scalar ([`assert_reproducible_bits`]) or as
//! a full reward sequence ([`assert_reproducible_seq`]).

/// Asserts a trained agent's score is finite and beats a *measured* random
/// baseline by at least `margin`.
///
/// `random` is produced by an actual uniform-random-policy rollout over the same
/// environment (see [`crate::baseline::random_return`]), so the bar tracks the
/// environment rather than a hand-copied magic number. Use this for the
/// `improves_over_random` learning checks; use [`assert_reaches`] when the
/// target is an absolute floor instead.
///
/// # Panics
///
/// Panics if either score is non-finite, or if `trained` does not exceed
/// `random + margin`.
pub fn assert_improves_over_random(trained: f32, random: f32, margin: f32) {
    assert!(trained.is_finite(), "trained avg must be finite, got {trained}");
    assert!(
        random.is_finite(),
        "random baseline must be finite, got {random}"
    );
    assert!(
        trained > random + margin,
        "expected trained avg > random baseline {random:.3} + margin {margin:.3}, got {trained:.3}"
    );
}

/// Asserts a moving-average reward is finite and reaches `threshold` (`>=`).
///
/// The `>=` counterpart to [`assert_beats_baseline`], used by the discrete
/// "reaches N" convergence checks (e.g. `CartPole` avg ≥ 100) where the target is
/// an absolute floor rather than a margin over a random baseline.
///
/// # Panics
///
/// Panics if `avg` is non-finite or below `threshold`.
pub fn assert_reaches(avg: f32, threshold: f32) {
    assert!(avg.is_finite(), "avg reward must be finite, got {avg}");
    assert!(
        avg >= threshold,
        "expected avg reward >= {threshold}, got {avg:.2}"
    );
}

/// Asserts every value in `values` is finite, labelling failures with `label`
/// and the offending index.
///
/// Used by the smoke tests to catch a metric column (rewards, a loss, a
/// `q_mean`, a quantile spread) silently NaN-ing out over a short run.
///
/// # Panics
///
/// Panics on the first non-finite value.
pub fn assert_all_finite(label: &str, values: &[f32]) {
    for (i, v) in values.iter().enumerate() {
        assert!(v.is_finite(), "non-finite {label} at index {i}: {v}");
    }
}

/// Asserts two scalar metrics from seeded back-to-back runs are bit-identical.
///
/// Compares the raw IEEE-754 bit patterns so the check is exact rather than
/// approximate — the whole point of a reproducibility test.
///
/// # Panics
///
/// Panics if the two values differ in any bit.
pub fn assert_reproducible_bits(a: f32, b: f32) {
    assert_eq!(a.to_bits(), b.to_bits(), "run not reproducible: {a} vs {b}");
}

/// Asserts two reward sequences from seeded back-to-back runs match exactly,
/// element for element.
///
/// # Panics
///
/// Panics if the sequences differ in length or at any episode.
#[allow(clippy::float_cmp)]
pub fn assert_reproducible_seq(a: &[f32], b: &[f32]) {
    assert_eq!(
        a.len(),
        b.len(),
        "history length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(x, y, "divergence at episode {i}: {x} vs {y}");
    }
}
