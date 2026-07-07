//! Validated unit-interval probability: the [`Probability`] newtype and
//! [`ProbabilityError`].
//!
//! A mutation/crossover **rate** carried as a bare `f32` is a standing hazard.
//! Every rate is consumed as a Bernoulli threshold — `u.lower_elem(p)`,
//! `rng < p`, `rng >= rate` — and a `NaN` rate silently degenerates the
//! operator: `x < NaN` is `false` for all `x`, so the mask goes all-false and
//! the operator becomes a no-op or a full clone of one parent. Worse, the
//! Cartesian-GP host path compares with the *opposite* sense (`rng >= rate`),
//! so a `NaN` there mutates **every** gene — the same bad input, the opposite
//! wrong answer. An out-of-range `p > 1` / `p < 0` saturates just as silently.
//!
//! [`Probability`] removes that possibility. It is a value in the closed unit
//! interval `[0, 1]` that cannot be constructed `NaN`, `Inf`, negative, or
//! above one: the whole invariant is `0.0 <= p <= 1.0`, which a `NaN`/`Inf`
//! fails. Every operator that takes a `Probability` is therefore well-behaved
//! by construction — the invariant travels with the value rather than being
//! re-checked (or, today, silently skipped) at each boundary.
//!
//! This complements the [`config`](crate::config) validation convention (ADR
//! 0026), exactly as [`Bounds`](crate::bounds::Bounds) does (ADR 0027): a config
//! field of type `Probability` is self-validating, so its
//! [`Validate`](crate::config::Validate) impl no longer repeats a
//! `config::in_range(…, 0.0, 1.0, …)` check for that field. See ADR 0031.
//!
//! For a non-negative but *unbounded* rate (a step size or expansion factor
//! such as BLX-α or Gaussian σ) see [`NonNegativeRate`](crate::rate::NonNegativeRate).
//!
//! # Examples
//!
//! ```
//! use rlevo_core::probability::Probability;
//!
//! // `new` is for compile-time-known rates and panics on a bad literal.
//! let p = Probability::new(0.5);
//! assert_eq!(p.get(), 0.5);
//!
//! // `try_new` is for runtime / user-supplied rates.
//! assert!(Probability::try_new(1.0).is_ok()); // endpoints are inclusive
//! assert!(Probability::try_new(0.0).is_ok());
//! assert!(Probability::try_new(1.5).is_err()); // above one
//! assert!(Probability::try_new(-0.1).is_err()); // below zero
//! assert!(Probability::try_new(f32::NAN).is_err()); // NaN
//! ```

use serde::{Deserialize, Serialize};

/// A probability in the closed unit interval `[0, 1]`, valid by construction.
///
/// A `Probability` can never hold a `NaN`, an infinity, a negative, or a value
/// above one: every constructor enforces `0.0 <= p <= 1.0` (a `NaN`/`Inf` fails
/// the comparison). As a result the silent all-false-mask degeneracy of
/// `u.lower_elem(p)` / `rng < p` on a bad rate is unrepresentable wherever a
/// `Probability` is held.
///
/// Construct with [`new`](Self::new) for literals (panics on an invalid value)
/// or [`try_new`](Self::try_new) for runtime data (returns [`ProbabilityError`]).
/// `Deserialize` routes through [`try_new`] via [`TryFrom`], so a rate loaded
/// from a file cannot deserialize into an out-of-range `Probability`.
///
/// [`try_new`]: Self::try_new
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "f32", into = "f32")]
pub struct Probability(f32);

impl Probability {
    /// Builds a probability from a compile-time-known value, panicking on an
    /// invalid one.
    ///
    /// This is the constructor for literals and `Default`s — the bad value is
    /// right at the call site, mirroring the documented builder-setter panic
    /// exception of ADR 0026. Prefer [`try_new`](Self::try_new) for any value
    /// derived from runtime or user-supplied data.
    ///
    /// # Panics
    ///
    /// Panics when `p` is outside `[0, 1]` (including `NaN` or infinite).
    #[must_use]
    pub const fn new(p: f32) -> Self {
        assert!(
            p >= 0.0 && p <= 1.0,
            "Probability::new: value outside [0, 1] (or NaN)"
        );
        Self(p)
    }

    /// Builds a probability from a runtime / user-supplied value.
    ///
    /// # Errors
    ///
    /// Returns [`ProbabilityError`] when `p` is outside `[0, 1]` (including
    /// `NaN` or infinite).
    pub fn try_new(p: f32) -> Result<Self, ProbabilityError> {
        // `(0.0..=1.0).contains(&p)` is false for NaN and out-of-range values.
        if (0.0..=1.0).contains(&p) {
            Ok(Self(p))
        } else {
            Err(ProbabilityError { got: p })
        }
    }

    /// The wrapped probability, guaranteed to lie in `[0, 1]`.
    #[must_use]
    pub const fn get(self) -> f32 {
        self.0
    }
}

impl TryFrom<f32> for Probability {
    type Error = ProbabilityError;

    fn try_from(p: f32) -> Result<Self, Self::Error> {
        Self::try_new(p)
    }
}

impl From<Probability> for f32 {
    fn from(p: Probability) -> Self {
        p.0
    }
}

/// The single way constructing a [`Probability`] can fail.
///
/// Allocation-free and `Copy`, carrying the offending value. Returned by
/// [`Probability::try_new`] / [`Probability::try_from`] (and thus by
/// `Deserialize`). A dedicated error rather than
/// [`ConfigError`](crate::config::ConfigError) because construction has no
/// config/field name to report — a config that builds a `Probability` from its
/// own scalar field wraps this as needed (ADR 0027 §5).
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[error(
    "invalid probability: {got} is outside the closed unit interval [0, 1] (and must not be NaN)"
)]
pub struct ProbabilityError {
    /// The value that was supplied.
    pub got: f32,
}

#[cfg(test)]
mod tests {
    use super::{Probability, ProbabilityError};

    #[test]
    fn new_accepts_interval_including_endpoints() {
        assert_eq!(Probability::new(0.0).get(), 0.0);
        assert_eq!(Probability::new(1.0).get(), 1.0);
        assert_eq!(Probability::new(0.25).get(), 0.25);
    }

    #[test]
    #[should_panic(expected = "outside [0, 1]")]
    fn new_panics_above_one() {
        let _ = Probability::new(1.5);
    }

    #[test]
    #[should_panic(expected = "outside [0, 1]")]
    fn new_panics_below_zero() {
        let _ = Probability::new(-0.1);
    }

    #[test]
    #[should_panic(expected = "outside [0, 1]")]
    fn new_panics_on_nan() {
        let _ = Probability::new(f32::NAN);
    }

    #[test]
    fn try_new_accepts_valid() {
        assert!(Probability::try_new(0.0).is_ok());
        assert!(Probability::try_new(1.0).is_ok());
        assert!(Probability::try_new(0.5).is_ok());
    }

    #[test]
    fn try_new_rejects_out_of_range_nan_and_inf() {
        assert_eq!(
            Probability::try_new(1.5),
            Err(ProbabilityError { got: 1.5 })
        );
        assert!(Probability::try_new(-0.1).is_err());
        assert!(Probability::try_new(f32::NAN).is_err());
        assert!(Probability::try_new(f32::INFINITY).is_err());
        assert!(Probability::try_new(f32::NEG_INFINITY).is_err());
    }

    #[test]
    fn round_trip_via_try_from_and_into() {
        // This is exactly the path `#[serde(try_from, into)]` exercises, so a
        // deserialized rate is validated and cannot be out of range.
        let p = Probability::try_from(0.7).unwrap();
        let back: f32 = p.into();
        assert_eq!(back, 0.7);
        assert!(Probability::try_from(2.0).is_err());
    }

    #[test]
    fn error_display_names_the_value() {
        let s = ProbabilityError { got: 1.5 }.to_string();
        assert!(s.contains("1.5"));
        assert!(s.contains("[0, 1]"));
    }
}
