//! Validated closed range: the [`Bounds`] newtype and [`BoundsError`].
//!
//! A `(f32, f32)` bounds tuple is a standing hazard: [`f32::clamp`] panics when
//! `min > max` (or on a `NaN` bound), and the alternative `x.max(lo).min(hi)`
//! idiom silently collapses every value to `hi` when `lo > hi` — a wrong result
//! instead of a loud one. Both failure modes require an *invalid* range to
//! exist in the first place.
//!
//! [`Bounds`] removes that possibility. It is an inclusive range `[lo, hi]` that
//! cannot be constructed inverted (`lo > hi`) or `NaN`, so every call site that
//! holds a `Bounds` is clamp-safe by construction — the invariant travels with
//! the value rather than being re-checked at each boundary. A degenerate
//! single-point range (`lo == hi`) is deliberately **allowed**: clamping to a
//! constant is well-defined, and every search-space consumer samples with
//! `lo + (hi - lo) * r`, so a zero-width range is safe.
//!
//! An infinite endpoint is **permitted** — it expresses a one-sided range such
//! as a `[0.7, ∞)` "healthy above this height" check, and [`f32::clamp`] is
//! well-defined with an infinite bound (it panics only on `min > max` or `NaN`).
//! The whole invariant is therefore exactly `lo <= hi`: a `NaN` endpoint makes
//! that comparison `false` and is rejected, an infinite one does not.
//!
//! This complements the [`config`](crate::config) validation convention (ADR
//! 0026): a config field of type `Bounds` is self-validating, so its
//! [`Validate`](crate::config::Validate) impl no longer repeats a
//! `config::ordered(…, "bounds", …)` check for that field. `Bounds` narrows one
//! field's invariant into the type system; it does not discharge a config's
//! other cross-field checks. See ADR 0027.
//!
//! # Examples
//!
//! ```
//! use rlevo_core::bounds::Bounds;
//!
//! // `new` is for compile-time-known endpoints and panics on a bad literal.
//! let b = Bounds::new(-1.0, 1.0);
//! assert_eq!(b.clamp(2.5), 1.0);
//! assert_eq!(b.clamp(-9.0), -1.0);
//! assert_eq!(b.clamp(0.25), 0.25);
//!
//! // `try_new` is for runtime / user-supplied endpoints.
//! assert!(Bounds::try_new(1.0, -1.0).is_err()); // inverted
//! assert!(Bounds::try_new(0.0, f32::NAN).is_err()); // NaN
//! assert!(Bounds::try_new(3.0, 3.0).is_ok()); // single point is fine
//! ```

use serde::{Deserialize, Serialize};

/// An inclusive range `[lo, hi]` over `f32`, valid by construction.
///
/// A `Bounds` can never hold an inverted (`lo > hi`) or `NaN` endpoint: every
/// constructor rejects those (the invariant is exactly `lo <= hi`, which a `NaN`
/// fails). As a result [`clamp`](Self::clamp) is total and panic-free, and the
/// silent `lo > hi` collapse of `x.max(lo).min(hi)` is unrepresentable wherever
/// a `Bounds` is held. A degenerate single-point range (`lo == hi`) and a
/// one-sided infinite range (e.g. `[0.7, ∞)`) are both permitted.
///
/// Construct with [`new`](Self::new) for literals (panics on an invalid pair) or
/// [`try_new`](Self::try_new) for runtime data (returns [`BoundsError`]).
/// `Deserialize` routes through [`try_new`] via [`TryFrom`], so a range loaded
/// from a file cannot deserialize into an invalid `Bounds`.
///
/// [`try_new`]: Self::try_new
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "(f32, f32)", into = "(f32, f32)")]
pub struct Bounds {
    lo: f32,
    hi: f32,
}

impl Bounds {
    /// Builds a range from compile-time-known endpoints, panicking on an
    /// invalid pair.
    ///
    /// This is the constructor for literals and `Default`s — the bad value is
    /// right at the call site, mirroring the documented builder-setter panic
    /// exception of ADR 0026. Prefer [`try_new`](Self::try_new) for any value
    /// derived from runtime or user-supplied data.
    ///
    /// # Panics
    ///
    /// Panics when `lo > hi` or either endpoint is `NaN`.
    #[must_use]
    pub const fn new(lo: f32, hi: f32) -> Self {
        assert!(
            lo <= hi,
            "Bounds::new: invalid range (lo > hi or NaN endpoint)"
        );
        Self { lo, hi }
    }

    /// Builds a range from runtime / user-supplied endpoints.
    ///
    /// # Errors
    ///
    /// Returns [`BoundsError`] when `lo > hi` or either endpoint is `NaN`.
    pub fn try_new(lo: f32, hi: f32) -> Result<Self, BoundsError> {
        if lo <= hi {
            Ok(Self { lo, hi })
        } else {
            Err(BoundsError { lo, hi })
        }
    }

    /// The inclusive lower endpoint.
    #[must_use]
    pub const fn lo(&self) -> f32 {
        self.lo
    }

    /// The inclusive upper endpoint.
    #[must_use]
    pub const fn hi(&self) -> f32 {
        self.hi
    }

    /// The width of the range, `hi - lo` (always `>= 0`).
    #[must_use]
    pub const fn span(&self) -> f32 {
        self.hi - self.lo
    }

    /// Clamps `x` into the range. Total and panic-free: the endpoints are
    /// guaranteed finite with `lo <= hi`, so [`f32::clamp`] never panics here.
    #[must_use]
    pub fn clamp(&self, x: f32) -> f32 {
        x.clamp(self.lo, self.hi)
    }

    /// Clamps every element of `xs` into the range in place.
    pub fn clamp_slice(&self, xs: &mut [f32]) {
        for x in xs.iter_mut() {
            *x = self.clamp(*x);
        }
    }
}

impl TryFrom<(f32, f32)> for Bounds {
    type Error = BoundsError;

    fn try_from((lo, hi): (f32, f32)) -> Result<Self, Self::Error> {
        Self::try_new(lo, hi)
    }
}

impl From<Bounds> for (f32, f32) {
    fn from(b: Bounds) -> Self {
        (b.lo, b.hi)
    }
}

/// The single way constructing a [`Bounds`] can fail.
///
/// Allocation-free and `Copy`, carrying the offending endpoints. Returned by
/// [`Bounds::try_new`] / [`Bounds::try_from`] (and thus by `Deserialize`). A
/// dedicated error rather than [`ConfigError`](crate::config::ConfigError)
/// because construction has no config/field name to report — a config that
/// builds a `Bounds` from its own scalar fields wraps this as needed.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[error("invalid bounds: lo {lo} must not exceed hi {hi} (and neither may be NaN)")]
pub struct BoundsError {
    /// The lower endpoint that was supplied.
    pub lo: f32,
    /// The upper endpoint that was supplied.
    pub hi: f32,
}

#[cfg(test)]
mod tests {
    // These tests assert exact round-trip of values that are stored and read
    // back without arithmetic, so bit-exact equality is the property under
    // test; an approximate comparison would weaken them.
    #![allow(clippy::float_cmp)]
    use super::{Bounds, BoundsError};

    #[test]
    fn new_accepts_ordered_and_single_point() {
        assert_eq!(Bounds::new(-5.12, 5.12).lo(), -5.12);
        assert_eq!(Bounds::new(-5.12, 5.12).hi(), 5.12);
        let point = Bounds::new(3.0, 3.0);
        assert_eq!(point.lo(), point.hi());
        assert_eq!(point.span(), 0.0);
    }

    #[test]
    #[should_panic(expected = "invalid range")]
    fn new_panics_on_inverted() {
        let _ = Bounds::new(1.0, -1.0);
    }

    #[test]
    #[should_panic(expected = "invalid range")]
    fn new_panics_on_nan() {
        let _ = Bounds::new(0.0, f32::NAN);
    }

    #[test]
    fn try_new_accepts_valid_including_one_sided_infinite() {
        assert!(Bounds::try_new(-1.0, 1.0).is_ok());
        assert!(Bounds::try_new(2.0, 2.0).is_ok()); // single point
        // A one-sided infinite range is a valid healthy-interval bound.
        assert!(Bounds::try_new(0.7, f32::INFINITY).is_ok());
        assert!(Bounds::try_new(f32::NEG_INFINITY, 0.0).is_ok());
    }

    #[test]
    fn try_new_rejects_inverted_and_nan() {
        assert_eq!(
            Bounds::try_new(1.0, -1.0),
            Err(BoundsError { lo: 1.0, hi: -1.0 })
        );
        assert!(Bounds::try_new(f32::NAN, 1.0).is_err());
        assert!(Bounds::try_new(0.0, f32::NAN).is_err());
        assert!(Bounds::try_new(f32::NAN, f32::NAN).is_err());
    }

    #[test]
    fn span_is_nonnegative_width() {
        assert_eq!(Bounds::new(-2.0, 3.0).span(), 5.0);
        assert_eq!(Bounds::new(4.0, 4.0).span(), 0.0);
    }

    #[test]
    fn clamp_below_within_and_above() {
        let b = Bounds::new(-1.0, 1.0);
        assert_eq!(b.clamp(-9.0), -1.0);
        assert_eq!(b.clamp(0.25), 0.25);
        assert_eq!(b.clamp(2.5), 1.0);
    }

    #[test]
    fn clamp_on_single_point_pins_to_constant() {
        let b = Bounds::new(3.0, 3.0);
        assert_eq!(b.clamp(0.0), 3.0);
        assert_eq!(b.clamp(9.0), 3.0);
    }

    #[test]
    fn clamp_slice_clamps_in_place() {
        let b = Bounds::new(0.0, 10.0);
        let mut xs = [-5.0, 3.0, 42.0, 10.0];
        b.clamp_slice(&mut xs);
        assert_eq!(xs, [0.0, 3.0, 10.0, 10.0]);
    }

    #[test]
    fn tuple_round_trip_via_try_from_and_into() {
        // This is exactly the path `#[serde(try_from, into)]` exercises, so a
        // deserialized range is validated and cannot be inverted.
        let b = Bounds::try_from((-2.0, 6.0)).unwrap();
        let back: (f32, f32) = b.into();
        assert_eq!(back, (-2.0, 6.0));
        assert!(Bounds::try_from((6.0, -2.0)).is_err());
    }

    #[test]
    fn error_display_names_both_endpoints() {
        let s = BoundsError { lo: 5.0, hi: 1.0 }.to_string();
        assert!(s.contains('5'));
        assert!(s.contains('1'));
        assert!(s.contains("must not exceed"));
    }
}
