//! Configuration validation: the [`Validate`] trait and [`ConfigError`].
//!
//! Across the workspace, dozens of `*Config` and hyperparameter-bearing structs
//! accept values that are only checked — if at all — deep inside training,
//! physics, or genetic-operator code. A `v_min == v_max`, a `pop_size == 0`, a
//! `tau` outside `[0, 1]`, or an inverted `action_clip` surfaces as a confusing
//! panic (or, worse, a silently wrong result) far from the line that supplied
//! it. This module is the single shared convention that turns those into a
//! clear, field-named rejection at construction.
//!
//! # The contract
//!
//! A config implements [`Validate`], checking its invariants and returning the
//! **first** violation as a [`ConfigError`] (fail-fast). The *construction
//! chokepoint* that consumes a caller-supplied config — an environment's
//! `with_config`, an agent's `new`, a builder's `build`, a harness's `new` —
//! calls [`Validate`]'s `validate` before the value can reach the algorithm, and
//! propagates the error as `Result<_, ConfigError>`.
//!
//! `Default` construction stays infallible: a library default must itself be
//! valid, which every implementor guarantees with a
//! `assert!(Config::default().validate().is_ok())` unit test. That is what lets
//! the infallible `Default` / `ConstructableEnv::new` paths keep returning
//! `Self`.
//!
//! # Panic vs. `Result` (rules.md §4 / ADR 0026)
//!
//! Validating an **assembled config as a whole** returns `Result` — never a
//! panic — because many configs derive `Deserialize`, so a config loaded from a
//! file is user-supplied runtime data. A single documented **builder setter**
//! (`with_capacity(0)`, `with_alpha(x ∉ [0, 1])`) may still panic: the panic
//! points at the offending call site, and it is an additive fail-fast
//! convenience, not a substitute for `validate`.
//!
//! # Example
//!
//! ```
//! use rlevo_core::config::{self, ConfigError, Validate};
//!
//! struct AtomsConfig {
//!     num_atoms: usize,
//!     v_min: f32,
//!     v_max: f32,
//! }
//!
//! impl Validate for AtomsConfig {
//!     fn validate(&self) -> Result<(), ConfigError> {
//!         const C: &str = "AtomsConfig";
//!         config::at_least(C, "num_atoms", self.num_atoms, 2)?;
//!         config::distinct(C, "v_max", f64::from(self.v_min), f64::from(self.v_max))?;
//!         config::ordered(C, "v_max", f64::from(self.v_min), f64::from(self.v_max))?;
//!         Ok(())
//!     }
//! }
//!
//! let good = AtomsConfig { num_atoms: 51, v_min: -10.0, v_max: 10.0 };
//! assert!(good.validate().is_ok());
//!
//! let bad = AtomsConfig { num_atoms: 51, v_min: 5.0, v_max: 5.0 };
//! let err = bad.validate().unwrap_err();
//! assert_eq!(err.field, "v_max");
//! ```

/// A configuration (or hyperparameter-bearing) type that can check its own
/// invariants before it is used to construct anything.
///
/// Implement this on every public `*Config` and on builders that carry tunable
/// hyperparameters. Keep the check cheap and purely local — inspect the fields,
/// never run the algorithm. Return the **first** violated invariant; callers
/// that want every violation can call `validate` after fixing each one.
///
/// The [`config` module helpers](self) (`positive`, `in_range`, `ordered`,
/// `distinct`, `nonzero`, `at_least`) keep an implementation to one line per
/// invariant. See the [module documentation](self) for the construction-time
/// contract and the panic-vs-`Result` rule.
pub trait Validate {
    /// Returns `Ok(())` when every invariant holds, or the first
    /// [`ConfigError`] otherwise.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] naming the offending field and the violated
    /// [`ConstraintKind`] as soon as any invariant fails.
    fn validate(&self) -> Result<(), ConfigError>;
}

/// A single violated configuration invariant.
///
/// Names the config type, the offending field, and the specific
/// [`ConstraintKind`]. The type is allocation-free — every field is `Copy` or
/// `&'static str` — so validation is cheap and [`ConfigError`] is `PartialEq`
/// for straightforward assertions in tests.
///
/// ```
/// use rlevo_core::config::{ConfigError, ConstraintKind};
///
/// let err = ConfigError {
///     config: "GaConfig",
///     field: "pop_size",
///     kind: ConstraintKind::Zero,
/// };
/// assert!(err.to_string().contains("GaConfig.pop_size"));
/// ```
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{config}.{field}: {kind}")]
pub struct ConfigError {
    /// The config type that failed validation, e.g. `"C51TrainingConfig"`.
    pub config: &'static str,
    /// The offending field, e.g. `"v_max"`.
    pub field: &'static str,
    /// The specific invariant that was violated.
    pub kind: ConstraintKind,
}

/// The closed set of configuration-invariant violations.
///
/// Structured rather than string-based (per `rules.md` §4). The `Custom`
/// variant carries a `&'static str`, never an owned `String`, so
/// [`ConfigError`] allocates nothing.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ConstraintKind {
    /// Value must lie in the closed interval `[lo, hi]`.
    #[error("value {got} is out of range [{lo}, {hi}]")]
    OutOfRange {
        /// Inclusive lower bound.
        lo: f64,
        /// Inclusive upper bound.
        hi: f64,
        /// The offending value.
        got: f64,
    },
    /// Value must be strictly positive (`> 0`).
    #[error("value {got} must be strictly positive")]
    NotPositive {
        /// The offending value.
        got: f64,
    },
    /// A `(low, high)` pair must satisfy `low < high`.
    #[error("{low} must be strictly less than {high}")]
    NotOrdered {
        /// The lower value that was not strictly below `high`.
        low: f64,
        /// The upper value.
        high: f64,
    },
    /// Two values that must differ are equal (e.g. `v_min == v_max`).
    #[error("value {value} must differ from its pair")]
    DegenerateInterval {
        /// The value shared by both sides of the degenerate interval.
        value: f64,
    },
    /// An integer count / size / capacity must be at least `min`.
    #[error("count {got} must be at least {min}")]
    TooSmall {
        /// The required minimum.
        min: u64,
        /// The offending count.
        got: u64,
    },
    /// A count / size / capacity must be non-zero.
    #[error("count/size must be non-zero")]
    Zero,
    /// A one-off invariant with a static explanation.
    #[error("{0}")]
    Custom(&'static str),
}

/// Rejects a value that is not strictly positive (`got > 0`).
///
/// # Errors
///
/// Returns [`ConstraintKind::NotPositive`] when `got <= 0` (or is `NaN`).
pub fn positive(config: &'static str, field: &'static str, got: f64) -> Result<(), ConfigError> {
    if got > 0.0 {
        Ok(())
    } else {
        Err(ConfigError { config, field, kind: ConstraintKind::NotPositive { got } })
    }
}

/// Rejects a value outside the closed interval `[lo, hi]`.
///
/// # Errors
///
/// Returns [`ConstraintKind::OutOfRange`] when `got < lo`, `got > hi`, or `got`
/// is `NaN`.
pub fn in_range(
    config: &'static str,
    field: &'static str,
    lo: f64,
    hi: f64,
    got: f64,
) -> Result<(), ConfigError> {
    if got >= lo && got <= hi {
        Ok(())
    } else {
        Err(ConfigError { config, field, kind: ConstraintKind::OutOfRange { lo, hi, got } })
    }
}

/// Rejects a `(low, high)` pair that is not strictly ordered (`low < high`).
///
/// # Errors
///
/// Returns [`ConstraintKind::NotOrdered`] when `low >= high` (or either is
/// `NaN`).
pub fn ordered(
    config: &'static str,
    field: &'static str,
    low: f64,
    high: f64,
) -> Result<(), ConfigError> {
    if low < high {
        Ok(())
    } else {
        Err(ConfigError { config, field, kind: ConstraintKind::NotOrdered { low, high } })
    }
}

/// Rejects two values that must differ but are equal.
///
/// # Errors
///
/// Returns [`ConstraintKind::DegenerateInterval`] when `a == b`.
pub fn distinct(
    config: &'static str,
    field: &'static str,
    a: f64,
    b: f64,
) -> Result<(), ConfigError> {
    if (a - b).abs() > 0.0 {
        Ok(())
    } else {
        Err(ConfigError { config, field, kind: ConstraintKind::DegenerateInterval { value: a } })
    }
}

/// Rejects a zero count / size / capacity.
///
/// # Errors
///
/// Returns [`ConstraintKind::Zero`] when `n == 0`.
pub fn nonzero(config: &'static str, field: &'static str, n: usize) -> Result<(), ConfigError> {
    if n == 0 {
        Err(ConfigError { config, field, kind: ConstraintKind::Zero })
    } else {
        Ok(())
    }
}

/// Rejects an integer count below `min`.
///
/// # Errors
///
/// Returns [`ConstraintKind::TooSmall`] when `got < min`.
pub fn at_least(
    config: &'static str,
    field: &'static str,
    got: usize,
    min: usize,
) -> Result<(), ConfigError> {
    if got >= min {
        Ok(())
    } else {
        Err(ConfigError {
            config,
            field,
            kind: ConstraintKind::TooSmall { min: min as u64, got: got as u64 },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        at_least, distinct, in_range, nonzero, ordered, positive, ConfigError, ConstraintKind,
    };

    const C: &str = "TestConfig";

    #[test]
    fn positive_accepts_and_rejects() {
        assert!(positive(C, "dt", 0.5).is_ok());
        let err = positive(C, "dt", 0.0).unwrap_err();
        assert_eq!(err.field, "dt");
        assert_eq!(err.kind, ConstraintKind::NotPositive { got: 0.0 });
        assert!(positive(C, "dt", -1.0).is_err());
    }

    #[test]
    fn positive_rejects_nan() {
        assert!(positive(C, "dt", f64::NAN).is_err());
    }

    #[test]
    fn in_range_boundaries_are_inclusive() {
        assert!(in_range(C, "gamma", 0.0, 1.0, 0.0).is_ok());
        assert!(in_range(C, "gamma", 0.0, 1.0, 1.0).is_ok());
        assert!(in_range(C, "gamma", 0.0, 1.0, 0.5).is_ok());
        let err = in_range(C, "gamma", 0.0, 1.0, 1.5).unwrap_err();
        assert_eq!(err.kind, ConstraintKind::OutOfRange { lo: 0.0, hi: 1.0, got: 1.5 });
        assert!(in_range(C, "gamma", 0.0, 1.0, f64::NAN).is_err());
    }

    #[test]
    fn ordered_requires_strict() {
        assert!(ordered(C, "clip", -1.0, 1.0).is_ok());
        assert!(ordered(C, "clip", 1.0, 1.0).is_err());
        let err = ordered(C, "clip", 2.0, 1.0).unwrap_err();
        assert_eq!(err.kind, ConstraintKind::NotOrdered { low: 2.0, high: 1.0 });
    }

    #[test]
    fn distinct_rejects_equal() {
        assert!(distinct(C, "v_max", -10.0, 10.0).is_ok());
        let err = distinct(C, "v_max", 5.0, 5.0).unwrap_err();
        assert_eq!(err.kind, ConstraintKind::DegenerateInterval { value: 5.0 });
    }

    #[test]
    fn nonzero_and_at_least() {
        assert!(nonzero(C, "max_steps", 1).is_ok());
        assert_eq!(nonzero(C, "max_steps", 0).unwrap_err().kind, ConstraintKind::Zero);
        assert!(at_least(C, "pop_size", 2, 2).is_ok());
        assert_eq!(
            at_least(C, "pop_size", 1, 2).unwrap_err().kind,
            ConstraintKind::TooSmall { min: 2, got: 1 }
        );
    }

    #[test]
    fn error_display_format() {
        let err = ConfigError {
            config: "C51TrainingConfig",
            field: "v_max",
            kind: ConstraintKind::DegenerateInterval { value: 0.0 },
        };
        let s = err.to_string();
        assert!(s.contains("C51TrainingConfig.v_max"));
        assert!(s.contains("must differ"));
    }
}
