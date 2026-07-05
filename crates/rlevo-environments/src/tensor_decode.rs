//! Shared helpers for decoding tensors back into discrete environment types.
//!
//! Action `from_tensor` implementations decode a one-hot (or logit) vector into
//! a discrete action index by taking the argmax. Doing so with
//! `partial_cmp(..).unwrap()` panics the moment any element is `NaN` (e.g. a
//! diverged policy emitting non-finite logits). [`argmax`] instead uses a
//! `NaN`-safe fold, matching the crate-wide `total_cmp`/`NaN`-safe comparison
//! convention (see `docs/rules.md`).

/// Returns the index of the largest element in `values`, `NaN`-safely.
///
/// Ties resolve to the **lowest** index. `NaN` elements never win: `NaN > x` is
/// `false`, so the running best is only displaced by a strictly greater finite
/// value. An all-`NaN` (or empty) slice therefore returns `0`, matching the
/// historical `.map(..).unwrap_or(0)` fallback of the classic action decoders.
///
/// This is the discrete-action analog of the crate's `total_cmp` sort
/// convention: it never panics on non-finite input.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(argmax(&[0.1, 0.9, 0.3]), 1);
/// assert_eq!(argmax(&[f32::NAN, f32::NAN]), 0); // no panic
/// assert_eq!(argmax(&[1.0, 1.0]), 0); // lowest index on ties
/// ```
pub(crate) fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |best, (i, &x)| {
            if x > best.1 { (i, x) } else { best }
        })
        .0
}

#[cfg(test)]
mod tests {
    use super::argmax;

    #[test]
    fn argmax_picks_largest() {
        assert_eq!(argmax(&[0.1, 0.9, 0.3]), 1);
        assert_eq!(argmax(&[5.0, -2.0, 1.0]), 0);
    }

    #[test]
    fn argmax_ties_resolve_to_lowest_index() {
        assert_eq!(argmax(&[1.0, 1.0, 1.0]), 0);
    }

    #[test]
    fn argmax_is_nan_safe() {
        assert_eq!(argmax(&[f32::NAN, f32::NAN, f32::NAN]), 0);
        // A finite value still beats surrounding NaNs.
        assert_eq!(argmax(&[f32::NAN, 2.0, f32::NAN]), 1);
    }

    #[test]
    fn argmax_empty_returns_zero() {
        assert_eq!(argmax(&[]), 0);
    }
}
