//! Validated non-negative rate: the [`NonNegativeRate`] newtype and
//! [`NonNegativeRateError`].
//!
//! Some evolutionary-operator parameters are non-negative magnitudes that are
//! *not* probabilities — they have no upper bound of one. BLX-α's expansion
//! factor `α` (conventionally `0.5`, but legitimately larger) widens the
//! crossover sampling box, and Gaussian mutation's step size `σ` scales the
//! injected noise. Carried as bare `f32`s, a `NaN` or `Inf` value poisons the
//! whole offspring tensor: `diff * NaN` and `noise * Inf` propagate silently
//! into every gene with no error.
//!
//! [`NonNegativeRate`] removes that possibility. It is a **finite**, non-negative
//! `f32` — the invariant is `is_finite() && r >= 0.0`, which rejects `NaN`,
//! `±∞`, and negatives while permitting any finite magnitude (including `0.0`,
//! which is a well-defined "no expansion / no mutation" rate). The invariant
//! travels with the value, so an operator that takes a `NonNegativeRate` cannot
//! be handed a poisoning scalar.
//!
//! This complements the [`config`](crate::config) validation convention (ADR
//! 0026) exactly as [`Bounds`](crate::bounds::Bounds) does (ADR 0027): a config
//! field of type `NonNegativeRate` is self-validating, so its
//! [`Validate`](crate::config::Validate) impl no longer repeats a
//! `config::in_range(…, 0.0, ∞, …)` check for that field. See ADR 0031.
//!
//! For a rate that must additionally be bounded above by one (a Bernoulli
//! probability) see [`Probability`](crate::probability::Probability).
//!
//! # Examples
//!
//! ```
//! use rlevo_core::rate::NonNegativeRate;
//!
//! // `new` is for compile-time-known rates and panics on a bad literal.
//! let sigma = NonNegativeRate::new(0.3);
//! assert_eq!(sigma.get(), 0.3);
//!
//! // Unlike a probability, values above one are fine.
//! assert!(NonNegativeRate::try_new(2.5).is_ok());
//! assert!(NonNegativeRate::try_new(0.0).is_ok()); // zero is a valid rate
//! assert!(NonNegativeRate::try_new(-0.1).is_err()); // negative
//! assert!(NonNegativeRate::try_new(f32::NAN).is_err()); // NaN
//! assert!(NonNegativeRate::try_new(f32::INFINITY).is_err()); // infinite
//! ```

use serde::{Deserialize, Serialize};

/// A finite, non-negative `f32` rate, valid by construction.
///
/// A `NonNegativeRate` can never hold a `NaN`, an infinity, or a negative: the
/// invariant is `is_finite() && r >= 0.0`. Unlike
/// [`Probability`](crate::probability::Probability) it has **no** upper bound —
/// it is the right type for an unbounded magnitude such as BLX-α's expansion
/// factor or Gaussian mutation's σ, where the hazard is a `NaN`/`Inf` poisoning
/// the offspring tensor rather than an out-of-`[0,1]` saturation.
///
/// Construct with [`new`](Self::new) for literals (panics on an invalid value)
/// or [`try_new`](Self::try_new) for runtime data (returns
/// [`NonNegativeRateError`]). `Deserialize` routes through [`try_new`] via
/// [`TryFrom`], so a rate loaded from a file cannot deserialize into a
/// non-finite or negative `NonNegativeRate`.
///
/// [`try_new`]: Self::try_new
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "f32", into = "f32")]
pub struct NonNegativeRate(f32);

impl NonNegativeRate {
    /// Builds a rate from a compile-time-known value, panicking on an invalid
    /// one.
    ///
    /// This is the constructor for literals and `Default`s — the bad value is
    /// right at the call site, mirroring the documented builder-setter panic
    /// exception of ADR 0026. Prefer [`try_new`](Self::try_new) for any value
    /// derived from runtime or user-supplied data.
    ///
    /// # Panics
    ///
    /// Panics when `r` is negative, `NaN`, or infinite.
    #[must_use]
    pub const fn new(r: f32) -> Self {
        assert!(r.is_finite() && r >= 0.0, "NonNegativeRate::new: value must be finite and >= 0 (NaN/Inf/negative rejected)");
        Self(r)
    }

    /// Builds a rate from a runtime / user-supplied value.
    ///
    /// # Errors
    ///
    /// Returns [`NonNegativeRateError`] when `r` is negative, `NaN`, or
    /// infinite.
    pub fn try_new(r: f32) -> Result<Self, NonNegativeRateError> {
        if r.is_finite() && r >= 0.0 {
            Ok(Self(r))
        } else {
            Err(NonNegativeRateError { got: r })
        }
    }

    /// The wrapped rate, guaranteed finite and `>= 0`.
    #[must_use]
    pub const fn get(self) -> f32 {
        self.0
    }
}

impl TryFrom<f32> for NonNegativeRate {
    type Error = NonNegativeRateError;

    fn try_from(r: f32) -> Result<Self, Self::Error> {
        Self::try_new(r)
    }
}

impl From<NonNegativeRate> for f32 {
    fn from(r: NonNegativeRate) -> Self {
        r.0
    }
}

/// The single way constructing a [`NonNegativeRate`] can fail.
///
/// Allocation-free and `Copy`, carrying the offending value. Returned by
/// [`NonNegativeRate::try_new`] / [`NonNegativeRate::try_from`] (and thus by
/// `Deserialize`). A dedicated error rather than
/// [`ConfigError`](crate::config::ConfigError) because construction has no
/// config/field name to report — a config that builds a `NonNegativeRate` from
/// its own scalar field wraps this as needed (ADR 0027 §5).
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[error("invalid rate: {got} must be finite and >= 0 (NaN, infinite, and negative are rejected)")]
pub struct NonNegativeRateError {
    /// The value that was supplied.
    pub got: f32,
}

#[cfg(test)]
mod tests {
    use super::{NonNegativeRate, NonNegativeRateError};

    #[test]
    fn new_accepts_zero_and_large_finite() {
        assert_eq!(NonNegativeRate::new(0.0).get(), 0.0);
        assert_eq!(NonNegativeRate::new(0.5).get(), 0.5);
        assert_eq!(NonNegativeRate::new(42.0).get(), 42.0); // above one is fine
    }

    #[test]
    #[should_panic(expected = "finite and >= 0")]
    fn new_panics_on_negative() {
        let _ = NonNegativeRate::new(-0.1);
    }

    #[test]
    #[should_panic(expected = "finite and >= 0")]
    fn new_panics_on_nan() {
        let _ = NonNegativeRate::new(f32::NAN);
    }

    #[test]
    #[should_panic(expected = "finite and >= 0")]
    fn new_panics_on_infinite() {
        let _ = NonNegativeRate::new(f32::INFINITY);
    }

    #[test]
    fn try_new_accepts_valid_including_large() {
        assert!(NonNegativeRate::try_new(0.0).is_ok());
        assert!(NonNegativeRate::try_new(0.3).is_ok());
        assert!(NonNegativeRate::try_new(100.0).is_ok());
    }

    #[test]
    fn try_new_rejects_negative_nan_and_inf() {
        assert_eq!(NonNegativeRate::try_new(-1.0), Err(NonNegativeRateError { got: -1.0 }));
        assert!(NonNegativeRate::try_new(f32::NAN).is_err());
        assert!(NonNegativeRate::try_new(f32::INFINITY).is_err());
        assert!(NonNegativeRate::try_new(f32::NEG_INFINITY).is_err());
    }

    #[test]
    fn round_trip_via_try_from_and_into() {
        // This is exactly the path `#[serde(try_from, into)]` exercises, so a
        // deserialized rate is validated and cannot be non-finite/negative.
        let r = NonNegativeRate::try_from(0.3).unwrap();
        let back: f32 = r.into();
        assert_eq!(back, 0.3);
        assert!(NonNegativeRate::try_from(f32::INFINITY).is_err());
    }

    #[test]
    fn error_display_names_the_value() {
        let s = NonNegativeRateError { got: -2.0 }.to_string();
        assert!(s.contains("-2"));
        assert!(s.contains("finite"));
    }
}
