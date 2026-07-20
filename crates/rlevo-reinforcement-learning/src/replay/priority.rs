//! Validated sampling priority: the [`Priority`] newtype and [`PriorityError`].
//!
//! # The defect this closes
//!
//! The pre-ADR-0050 `memory.rs` stored caller-supplied priorities unvalidated
//! (`memory.rs:234`). A single `NaN` priority — reachable in production off a
//! diverging network, not theoretical — propagated as follows:
//!
//! 1. `p.powf(alpha)` at `memory.rs:312` yields `NaN`.
//! 2. `sum_priorities` becomes `NaN`, so every sampling probability is `NaN`.
//! 3. The inverse-CDF comparison `random_val < cumulative` is `false` for all
//!    `random_val`, because every comparison against `NaN` is `false`.
//! 4. `selected_pos` therefore never advances and pins at `0` **forever** — the
//!    buffer silently degenerates into "always replay the oldest transition",
//!    with no panic, no error, and no log line.
//!
//! A negative priority is the same class of bug with a different shape: it
//! subtracts from the running total, so the inverse-CDF scan skips a contiguous
//! run of transitions.
//!
//! [`Priority`] makes both unrepresentable. It is `finite && > 0` by
//! construction, mirroring [`Bounds`](rlevo_core::bounds::Bounds) /
//! [`Probability`](rlevo_core::probability::Probability) /
//! [`NonNegativeRate`](rlevo_core::rate::NonNegativeRate) (ADR 0027 / 0031), so
//! the invariant travels with the value instead of being re-checked — or, as
//! before, silently skipped — at each boundary. See ADR 0050 §10.
//!
//! # Why strictly positive, not merely non-negative
//!
//! Schaul et al. (2016) §3.3 define the proportional priority as
//! `p_i = |δ_i| + ε`, where ε is "a small positive constant that prevents the
//! edge-case of transitions not being revisited once their error is zero". With
//! ε > 0 the priority is *never* zero, so `> 0` is the formulation's own
//! invariant rather than an extra restriction. Admitting `0` would readmit the
//! starvation edge case ε exists to prevent, and — at `α = 0`, where every
//! positive priority maps to `1.0` — would make a zero-priority transition the
//! sole exception to "α = 0 is uniform".
//!
//! # Examples
//!
//! ```
//! use rlevo_reinforcement_learning::replay::Priority;
//!
//! // `new` is for compile-time-known literals and panics on a bad one.
//! assert_eq!(Priority::new(1.0).get(), 1.0);
//!
//! // `try_new` is for runtime values — including TD errors off a network.
//! assert!(Priority::try_new(1e-6).is_ok());
//! assert!(Priority::try_new(0.0).is_err()); // zero starves
//! assert!(Priority::try_new(-1.0).is_err()); // negative breaks the CDF scan
//! assert!(Priority::try_new(f32::NAN).is_err()); // the pinned-at-zero defect
//! assert!(Priority::try_new(f32::INFINITY).is_err()); // poisons the total
//!
//! // `from_td_error` applies Schaul's `p_i = |delta_i| + epsilon`.
//! let p = Priority::from_td_error(-0.25, 1e-6).unwrap();
//! assert!((p.get() - 0.250_001).abs() < 1e-6);
//!
//! // A converged transition keeps a positive priority, so it is still revisited.
//! assert_eq!(Priority::from_td_error(0.0, 1e-6).unwrap().get(), 1e-6);
//!
//! // A `NaN` TD error is rejected at the boundary, not stored.
//! assert!(Priority::from_td_error(f32::NAN, 1e-6).is_err());
//! ```

use serde::{Deserialize, Serialize};

/// A replay sampling priority: finite and strictly positive by construction.
///
/// A `Priority` can never hold a `NaN`, an infinity, a zero, or a negative
/// value. Every constructor enforces `p.is_finite() && p > 0.0`. As a result the
/// silent "`selected_pos` pinned at 0" degeneracy described in the
/// [module documentation](self) is unrepresentable wherever a `Priority` is
/// held.
///
/// Construct with [`new`](Self::new) for literals (panics on an invalid value),
/// [`try_new`](Self::try_new) for runtime data, or
/// [`from_td_error`](Self::from_td_error) to apply Schaul's `p_i = |δ_i| + ε`.
/// `Deserialize` routes through [`try_new`] via [`TryFrom`], so a priority
/// loaded from a file cannot deserialize into an invalid one.
///
/// [`try_new`]: Self::try_new
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "f32", into = "f32")]
pub struct Priority(f32);

impl Priority {
    /// Builds a priority from a compile-time-known value, panicking on an
    /// invalid one.
    ///
    /// This is the constructor for literals and `Default`s — the bad value is
    /// right at the call site, mirroring the documented builder-setter panic
    /// exception of ADR 0026. Prefer [`try_new`](Self::try_new) for any value
    /// derived from runtime data, and [`from_td_error`](Self::from_td_error)
    /// for anything derived from a network output.
    ///
    /// # Panics
    ///
    /// Panics when `p` is not finite or not strictly positive.
    #[must_use]
    pub fn new(p: f32) -> Self {
        assert!(
            p.is_finite() && p > 0.0,
            "Priority::new: value must be finite and strictly positive"
        );
        Self(p)
    }

    /// Builds a priority from a runtime value.
    ///
    /// # Errors
    ///
    /// Returns [`PriorityError`] when `p` is not finite (`NaN`, `±∞`) or not
    /// strictly positive.
    pub fn try_new(p: f32) -> Result<Self, PriorityError> {
        if p.is_finite() && p > 0.0 {
            Ok(Self(p))
        } else {
            Err(PriorityError { got: p })
        }
    }

    /// Applies Schaul et al. (2016) §3.3's proportional priority
    /// `p_i = |δ_i| + ε`.
    ///
    /// This is the constructor agents use: `td_error` is a per-sample residual
    /// read back off a network, so it is precisely the value that can arrive as
    /// `NaN` when training diverges. Rejecting it here is what keeps the
    /// divergence a reported error rather than a silently degenerate buffer.
    ///
    /// `epsilon` is the buffer's configured
    /// [`priority_epsilon`](super::PrioritizedReplayConfig::priority_epsilon);
    /// prefer
    /// [`PrioritizedReplay::priority_from_td_error`](super::PrioritizedReplay::priority_from_td_error)
    /// over calling this directly, so the buffer's own ε is applied rather than
    /// a second copy that can drift.
    ///
    /// # Errors
    ///
    /// Returns [`PriorityError`] when `|td_error| + epsilon` is not finite or
    /// not strictly positive — which covers a `NaN` or infinite `td_error`, and
    /// a non-positive or non-finite `epsilon`.
    pub fn from_td_error(td_error: f32, epsilon: f32) -> Result<Self, PriorityError> {
        Self::try_new(td_error.abs() + epsilon)
    }

    /// The wrapped priority, guaranteed finite and strictly positive.
    #[must_use]
    pub const fn get(self) -> f32 {
        self.0
    }

    /// The larger of two priorities.
    ///
    /// Both operands are finite by construction, so there is no
    /// `partial_cmp`-on-`NaN` hazard here — the `total_cmp` rule of
    /// `rules.md` §3 exists for values that *can* be `NaN`. Used to maintain
    /// the running maximum of Schaul's Algorithm 1 line 6.
    #[must_use]
    pub fn max(self, other: Self) -> Self {
        if other.0 > self.0 { other } else { self }
    }
}

impl TryFrom<f32> for Priority {
    type Error = PriorityError;

    fn try_from(p: f32) -> Result<Self, Self::Error> {
        Self::try_new(p)
    }
}

impl From<Priority> for f32 {
    fn from(p: Priority) -> Self {
        p.0
    }
}

/// The single way constructing a [`Priority`] can fail.
///
/// Allocation-free and `Copy`, carrying the offending value. A dedicated error
/// rather than [`ConfigError`](rlevo_core::config::ConfigError) because
/// construction has no config/field name to report — a config that builds a
/// `Priority` from its own scalar field wraps this as needed (ADR 0027 §5).
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[error("invalid priority: {got} must be finite and strictly positive")]
pub struct PriorityError {
    /// The value that was supplied.
    pub got: f32,
}

#[cfg(test)]
mod tests {
    // Exact comparison is intentional throughout this test module: the values are
    // config literals read back unchanged, or a computed result whose bit-exactness
    // is itself the property under test (that an anneal lands exactly on its
    // endpoint, that `-0.0` is accepted as the no-correction setting). A tolerance
    // would let a real regression pass. Reviewed as a class, not site-by-site.
    #![allow(clippy::float_cmp)]
    use super::{Priority, PriorityError};

    #[test]
    fn test_priority_try_new_accepts_finite_positive() {
        for p in [f32::MIN_POSITIVE, 1e-6, 0.5, 1.0, 1e6, f32::MAX] {
            assert!(
                Priority::try_new(p).is_ok(),
                "finite positive {p} must be an accepted priority"
            );
        }
    }

    /// The regression test for the defect named in the module docs: `NaN`,
    /// `±∞`, `0`, and negatives must all be rejected *at construction*, so they
    /// can never reach `powf` or the prefix scan.
    #[test]
    fn test_priority_try_new_rejects_nan_infinite_zero_and_negative() {
        let rejected = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            -1e-9,
            -1.0,
            f32::MIN,
        ];
        for p in rejected {
            let err = Priority::try_new(p).unwrap_err();
            assert!(
                p.is_nan() || err == PriorityError { got: p },
                "rejected priority {p} must report itself in the error"
            );
        }
    }

    #[test]
    #[should_panic(expected = "finite and strictly positive")]
    fn test_priority_new_panics_on_nan() {
        let _ = Priority::new(f32::NAN);
    }

    #[test]
    #[should_panic(expected = "finite and strictly positive")]
    fn test_priority_new_panics_on_zero() {
        let _ = Priority::new(0.0);
    }

    #[test]
    fn test_priority_from_td_error_applies_epsilon_floor() {
        // A converged transition (delta == 0) keeps a strictly positive
        // priority — Schaul's stated purpose for epsilon.
        let converged = Priority::from_td_error(0.0, 1e-6).expect("epsilon floor keeps p > 0");
        assert!(
            converged.get() > 0.0,
            "a zero-TD-error transition must stay revisitable, got {}",
            converged.get()
        );
        assert!(
            (converged.get() - 1e-6).abs() < 1e-12,
            "p_i for delta == 0 must equal epsilon"
        );
    }

    #[test]
    fn test_priority_from_td_error_uses_absolute_value() {
        let pos = Priority::from_td_error(0.75, 1e-6).expect("positive residual");
        let neg = Priority::from_td_error(-0.75, 1e-6).expect("negative residual");
        assert!(
            (pos.get() - neg.get()).abs() < 1e-9,
            "p_i = |delta_i| + eps must be sign-agnostic: {} vs {}",
            pos.get(),
            neg.get()
        );
    }

    #[test]
    fn test_priority_from_td_error_rejects_diverged_network_output() {
        for delta in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                Priority::from_td_error(delta, 1e-6).is_err(),
                "a diverged TD error ({delta}) must not become a stored priority"
            );
        }
    }

    #[test]
    fn test_priority_max_tracks_running_maximum() {
        let a = Priority::new(0.25);
        let b = Priority::new(4.0);
        assert_eq!(a.max(b), b, "max must return the larger operand");
        assert_eq!(b.max(a), b, "max must be order-independent");
        assert_eq!(a.max(a), a, "max must be idempotent");
    }

    /// `Deserialize` is `#[serde(try_from = "f32")]`, so it routes through
    /// [`Priority::try_new`]. Exercising [`TryFrom`] directly proves the same
    /// gate without pulling a format crate into `dev-dependencies`.
    #[test]
    fn test_priority_try_from_f32_is_the_deserialize_gate() {
        let good = Priority::try_from(0.5).expect("0.5 is a valid priority");
        assert_eq!(good.get(), 0.5, "TryFrom must preserve the value");
        assert_eq!(f32::from(good), 0.5, "Into<f32> must round-trip");
        for bad in [-1.0, 0.0, f32::NAN, f32::INFINITY] {
            assert!(
                Priority::try_from(bad).is_err(),
                "Deserialize must reject {bad} by routing through try_new"
            );
        }
    }
}
