//! Validated importance-sampling exponent: the [`ImportanceExponent`] newtype
//! and [`ImportanceExponentError`].
//!
//! # The defect this closes
//!
//! [`PrioritizedReplay::sample`](super::PrioritizedReplay::sample) raises a
//! probability ratio to β. Before this newtype the parameter was a bare `f32`
//! guarded only by a `debug_assert!`, so in a **release** build a non-finite or
//! out-of-range β propagated as follows:
//!
//! 1. `(min_mass / m).powf(beta)` yields `NaN` (β = `NaN`) or `±∞`/`0` (β
//!    unbounded).
//! 2. The `NaN` lands in the batch's importance weights.
//! 3. The agent uploads those weights as a tensor and multiplies the per-sample
//!    loss by them, so the reduced loss is `NaN`.
//! 4. `backward()` propagates `NaN` into every parameter the optimizer touches
//!    next, permanently. No panic, no error, no log line — the same silent
//!    poisoning shape as the `NaN`-priority defect the [`Priority`] newtype
//!    closes (see the [`priority`](super::priority) module docs).
//!
//! # Why the caller's schedule cannot be the validation site
//!
//! ADR 0050 §11 put the β schedule on the agent config and accepted, as a cost,
//! that "a caller can pass a nonsense β" with `Validate` on the config as the
//! mitigation. ADR 0051 **withdraws** that as unsound, and this type is the
//! replacement.
//!
//! A config holds schedule *endpoints* — `beta_start`, `beta_end`,
//! `beta_anneal_steps`. What reaches `powf` is the *evaluated* interpolation:
//!
//! ```text
//! beta_start + (beta_end - beta_start) * limit(step as f32 / anneal_steps as f32)
//! ```
//!
//! With `anneal_steps == 0` the fraction is `0.0 / 0.0` = `NaN` at `step == 0`
//! and `+∞` thereafter — while **every endpoint is individually valid**. Whether
//! that `NaN` survives `limit` then turns on an incidental IEEE-754 detail:
//! `f32::min` returns the non-`NaN` operand and silently launders it into
//! `1.0`, whereas `f32::clamp` and an ordinary `if x > 1.0` comparison both
//! propagate it. So a refactor between three spellings that look interchangeable
//! decides whether the buffer receives a `NaN`.
//!
//! That is the argument, and it is stronger than "the config forgot a check":
//! endpoint validation cannot establish a property of the *evaluated* value at
//! all, and the one thing standing between a zero-length schedule and poisoned
//! gradients is a method-choice nobody documented. The invariant has to travel
//! with the *value*, which is what a newtype does.
//!
//! # Examples
//!
//! ```
//! use rlevo_reinforcement_learning::replay::ImportanceExponent;
//!
//! // `new` is for compile-time-known literals and panics on a bad one.
//! assert_eq!(ImportanceExponent::new(0.4).get(), 0.4);
//!
//! // The annealed endpoint is a named constant, not a magic literal.
//! assert_eq!(ImportanceExponent::ONE.get(), 1.0);
//!
//! // `try_new` is for runtime values — including an evaluated schedule.
//! assert!(ImportanceExponent::try_new(0.0).is_ok()); // no correction
//! assert!(ImportanceExponent::try_new(1.0).is_ok()); // full correction
//! assert!(ImportanceExponent::try_new(-0.1).is_err());
//! assert!(ImportanceExponent::try_new(1.1).is_err());
//! assert!(ImportanceExponent::try_new(f32::NAN).is_err()); // the poisoning defect
//! ```
//!
//! [`Priority`]: super::Priority

use serde::{Deserialize, Serialize};

/// Schaul et al. (2016) §3.4's importance-sampling exponent β: finite and
/// within `[0, 1]` by construction.
///
/// An `ImportanceExponent` can never hold a `NaN`, an infinity, or a value
/// outside `[0, 1]`. Every constructor enforces
/// `b.is_finite() && (0.0..=1.0).contains(&b)`. As a result the `NaN`-weights →
/// `NaN`-loss → poisoned-gradients chain described in the
/// [module documentation](self) is unrepresentable wherever an
/// `ImportanceExponent` is held.
///
/// `0.0` applies no importance correction; `1.0` ([`ONE`](Self::ONE)) applies
/// the full correction and is the endpoint of Schaul Table 3's `β₀ = 0.4 → 1.0`
/// annealing schedule.
///
/// Construct with [`new`](Self::new) for literals (panics on an invalid value)
/// or [`try_new`](Self::try_new) for runtime data — including a schedule
/// evaluated against a step counter, which is precisely the value that can
/// arrive as `NaN`. `Deserialize` routes through [`try_new`] via [`TryFrom`], so
/// a β loaded from a file cannot deserialize into an invalid one.
///
/// [`try_new`]: Self::try_new
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "f32", into = "f32")]
pub struct ImportanceExponent(f32);

impl ImportanceExponent {
    /// Full importance-sampling correction — Schaul Table 3's annealed
    /// endpoint, and the value a strategy that emits no weights is handed.
    ///
    /// Named rather than spelled `1.0` at each call site so that "the schedule
    /// has finished annealing" and "this strategy ignores β anyway" read as the
    /// same documented intent everywhere they appear.
    pub const ONE: Self = Self(1.0);

    /// Builds an exponent from a compile-time-known value, panicking on an
    /// invalid one.
    ///
    /// This is the constructor for literals and `Default`s — the bad value is
    /// right at the call site, mirroring the documented builder-setter panic
    /// exception of ADR 0026. Prefer [`try_new`](Self::try_new) for any value
    /// derived from runtime data, which includes an annealing schedule
    /// evaluated against a step counter.
    ///
    /// # Panics
    ///
    /// Panics when `b` is not finite or lies outside `[0, 1]`.
    #[must_use]
    pub fn new(b: f32) -> Self {
        assert!(
            b.is_finite() && (0.0..=1.0).contains(&b),
            "ImportanceExponent::new: value must be finite and within [0, 1]"
        );
        Self(b)
    }

    /// Builds an exponent from a runtime value.
    ///
    /// # Errors
    ///
    /// Returns [`ImportanceExponentError`] when `b` is not finite (`NaN`, `±∞`)
    /// or lies outside `[0, 1]`.
    pub fn try_new(b: f32) -> Result<Self, ImportanceExponentError> {
        if b.is_finite() && (0.0..=1.0).contains(&b) {
            Ok(Self(b))
        } else {
            Err(ImportanceExponentError { got: b })
        }
    }

    /// The wrapped exponent, guaranteed finite and within `[0, 1]`.
    #[must_use]
    pub const fn get(self) -> f32 {
        self.0
    }
}

impl TryFrom<f32> for ImportanceExponent {
    type Error = ImportanceExponentError;

    fn try_from(b: f32) -> Result<Self, Self::Error> {
        Self::try_new(b)
    }
}

impl From<ImportanceExponent> for f32 {
    fn from(b: ImportanceExponent) -> Self {
        b.0
    }
}

/// The single way constructing an [`ImportanceExponent`] can fail.
///
/// Allocation-free and `Copy`, carrying the offending value. A dedicated error
/// rather than [`ConfigError`](rlevo_core::config::ConfigError) because
/// construction has no config/field name to report — an agent config that
/// evaluates its own β schedule wraps this as needed (ADR 0027 §5).
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[error("invalid importance exponent: {got} must be finite and within [0, 1]")]
pub struct ImportanceExponentError {
    /// The value that was supplied.
    pub got: f32,
}

#[cfg(test)]
mod tests {
    use super::{ImportanceExponent, ImportanceExponentError};

    #[test]
    fn test_importance_exponent_try_new_accepts_the_closed_unit_interval() {
        for b in [0.0, 1e-6, 0.4, 0.5, 1.0] {
            assert!(
                ImportanceExponent::try_new(b).is_ok(),
                "beta = {b} lies in [0, 1] and must be accepted"
            );
        }
    }

    /// The regression test for the defect named in the module docs: `NaN`,
    /// `±∞`, and out-of-range values must all be rejected *at construction*, so
    /// they can never reach `powf` and become `NaN` importance weights.
    #[test]
    fn test_importance_exponent_try_new_rejects_nan_infinite_and_out_of_range() {
        let rejected = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -1e-6,
            -1.0,
            1.000_001,
            2.0,
        ];
        for b in rejected {
            let err = ImportanceExponent::try_new(b).unwrap_err();
            assert!(
                b.is_nan() || err == ImportanceExponentError { got: b },
                "rejected exponent {b} must report itself in the error"
            );
        }
    }

    /// `-0.0` is finite and compares equal to `0.0`, so it is inside the range
    /// and `powf` treats it exactly as `0.0` does. Unlike `Priority`, where a
    /// zero starves a transition, a zero β is a *meaningful* setting (no
    /// importance correction), so this is an accept, not a reject.
    #[test]
    fn test_importance_exponent_accepts_negative_zero_as_zero() {
        let b = ImportanceExponent::try_new(-0.0).expect("-0.0 == 0.0 is in range");
        assert_eq!(b.get(), 0.0, "-0.0 is the no-correction setting");
    }

    #[test]
    #[should_panic(expected = "finite and within [0, 1]")]
    fn test_importance_exponent_new_panics_on_nan() {
        let _ = ImportanceExponent::new(f32::NAN);
    }

    #[test]
    #[should_panic(expected = "finite and within [0, 1]")]
    fn test_importance_exponent_new_panics_above_one() {
        let _ = ImportanceExponent::new(1.5);
    }

    #[test]
    fn test_importance_exponent_one_is_the_annealed_endpoint() {
        assert_eq!(
            ImportanceExponent::ONE.get(),
            1.0,
            "ONE must be Schaul Table 3's annealed endpoint"
        );
        assert_eq!(
            ImportanceExponent::ONE,
            ImportanceExponent::new(1.0),
            "the constant and the literal constructor must agree"
        );
    }

    /// The unsoundness ADR 0051 §3 records, spelled as executable code.
    ///
    /// Both schedule *endpoints* are individually valid, yet with
    /// `anneal_steps == 0` the progress fraction is `0.0 / 0.0` = `NaN` at
    /// `step == 0`. Whether that `NaN` reaches `powf` then depends on an
    /// incidental IEEE-754 detail of the limiter spelling — and *that* is the
    /// point: the soundness of the evaluated β is not a property the config can
    /// see, so endpoint validation cannot establish it.
    #[test]
    fn test_importance_exponent_rejects_an_evaluated_zero_anneal_schedule() {
        let (beta_start, beta_end, anneal_steps) = (0.4_f32, 1.0_f32, 0_u32);
        assert!(
            ImportanceExponent::try_new(beta_start).is_ok()
                && ImportanceExponent::try_new(beta_end).is_ok(),
            "both endpoints are individually valid"
        );

        let raw = 0.0_f32 / anneal_steps as f32;
        assert!(raw.is_nan(), "a zero-length schedule divides 0 by 0");

        // `f32::min` returns the non-NaN operand, silently laundering the NaN
        // into a plausible-looking 1.0. This spelling happens to be safe.
        let laundered = beta_start + (beta_end - beta_start) * raw.min(1.0);
        assert_eq!(
            laundered, beta_end,
            "`.min(1.0)` launders the NaN; the bug is masked, not absent"
        );

        // `clamp` and an ordinary comparison both propagate it. A refactor
        // between these three spellings silently decides whether the buffer
        // receives a NaN — which is why the invariant belongs on the value.
        for progress in [raw.clamp(0.0, 1.0), if raw > 1.0 { 1.0 } else { raw }] {
            let evaluated = beta_start + (beta_end - beta_start) * progress;
            assert!(
                evaluated.is_nan(),
                "a NaN-propagating limiter yields a NaN beta, got {evaluated}"
            );
            assert!(
                ImportanceExponent::try_new(evaluated).is_err(),
                "the evaluated NaN must be rejected at the construction site"
            );
        }
    }

    /// `Deserialize` is `#[serde(try_from = "f32")]`, so it routes through
    /// [`ImportanceExponent::try_new`]. Exercising [`TryFrom`] directly proves
    /// the same gate without pulling a format crate into `dev-dependencies`.
    #[test]
    fn test_importance_exponent_try_from_f32_is_the_deserialize_gate() {
        let good = ImportanceExponent::try_from(0.4).expect("0.4 is a valid exponent");
        assert_eq!(good.get(), 0.4, "TryFrom must preserve the value");
        assert_eq!(f32::from(good), 0.4, "Into<f32> must round-trip");
        for bad in [-1.0, 1.5, f32::NAN, f32::INFINITY] {
            assert!(
                ImportanceExponent::try_from(bad).is_err(),
                "Deserialize must reject {bad} by routing through try_new"
            );
        }
    }
}
