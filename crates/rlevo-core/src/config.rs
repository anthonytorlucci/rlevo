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
//! ## A config *value* must be finite; a config *bound* may be `±∞`
//!
//! Every float the helpers below check as a **value** — `positive`'s `got`,
//! `in_range`'s `got`, `ordered`'s `(low, high)`, `distinct`'s `(a, b)` — is a
//! config's own field, and a field is never legitimately `NaN` or `±∞`. Those
//! are rejected as [`ConstraintKind::NotFinite`], and the finiteness check runs
//! *first*, so `NotFinite` wins over `NotPositive` / `OutOfRange` /
//! `NotOrdered` / `DegenerateInterval`. A non-finite value is a
//! wrong-kind-of-number problem, and naming it as one is a better diagnosis
//! than reporting the downstream comparison it also happens to fail.
//!
//! The **bounds** of [`in_range`] are the opposite case: `lo` and `hi` are the
//! *schema*, not the data, and `hi = f64::INFINITY` is the intended spelling of
//! "unbounded above". `in_range(C, f, 0.0, f64::INFINITY, x)` therefore means
//! "`x` must be finite and non-negative" and is accepted. `lo`/`hi` are not
//! checked for finiteness, deliberately.
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

    /// Returns *every* violated invariant rather than just the first.
    ///
    /// The default implementation defers to [`validate`](Self::validate),
    /// yielding at most one error wrapped in a `Vec`. Override it for configs
    /// with several **independent** derived fields — CMA-ES recombination
    /// weights and covariance learning rates being the motivating case — where
    /// surfacing all violations at once spares the caller a fix-recheck-repeat
    /// cycle. Accumulate the checks with [`Violations`], and keep [`validate`](Self::validate)
    /// consistent by deriving it from the first collected error.
    ///
    /// # Errors
    ///
    /// Returns a non-empty `Vec<ConfigError>` — one entry per violated
    /// invariant — if any invariant fails.
    fn validate_all(&self) -> Result<(), Vec<ConfigError>> {
        self.validate().map_err(|e| vec![e])
    }
}

/// Accumulator for [`Validate::validate_all`] implementations.
///
/// Feed each check into [`check`](Self::check): a failing [`ConfigError`] is
/// recorded and evaluation continues, so one pass collects *every* violation.
/// [`into_result`](Self::into_result) then yields `Ok(())` when nothing failed
/// or the full list otherwise.
///
/// ```
/// use rlevo_core::config::{self, Violations};
///
/// let mut v = Violations::new();
/// v.check(config::positive("Demo", "a", -1.0)); // fails
/// v.check(config::nonzero("Demo", "b", 3)); // ok
/// v.check(config::in_range("Demo", "c", 0.0, 1.0, 2.0)); // fails
/// let errs = v.into_result().unwrap_err();
/// assert_eq!(errs.len(), 2);
/// assert_eq!(errs[0].field, "a");
/// assert_eq!(errs[1].field, "c");
/// ```
#[derive(Debug, Default)]
pub struct Violations {
    errors: Vec<ConfigError>,
}

impl Violations {
    /// Creates an empty accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records `check` when it failed; a passing check is a no-op. Either way
    /// evaluation continues, so callers list all their invariants back to back.
    pub fn check(&mut self, check: Result<(), ConfigError>) {
        if let Err(e) = check {
            self.errors.push(e);
        }
    }

    /// Consumes the accumulator: `Ok(())` if nothing failed, otherwise every
    /// collected violation.
    ///
    /// # Errors
    ///
    /// Returns the non-empty `Vec<ConfigError>` of all recorded violations.
    pub fn into_result(self) -> Result<(), Vec<ConfigError>> {
        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(self.errors)
        }
    }
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
///
/// `#[non_exhaustive]`: a new violation kind (as [`NotFinite`] itself was) must
/// not break a downstream `match`. Match with a trailing `_` arm.
///
/// [`NotFinite`]: ConstraintKind::NotFinite
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ConstraintKind {
    /// Value must be finite: `NaN` and `±∞` are never valid config *values*.
    ///
    /// Checked before every other float constraint, so a non-finite value is
    /// reported as what it is rather than as the range / ordering / positivity
    /// comparison it also fails.
    ///
    /// A *bound* may still be infinite — see [`in_range`]'s `lo`/`hi`.
    #[error("value {got} must be finite")]
    NotFinite {
        /// The offending value.
        got: f64,
    },
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

/// Rejects a value that is not finite, or not strictly positive (`got > 0`).
///
/// `+∞` is *not* a positive config value: `f64::INFINITY > 0.0` is true, but a
/// hyperparameter of `+∞` is a mistake, not an intent. It is rejected as
/// [`ConstraintKind::NotFinite`], as is `−∞` and `NaN`.
///
/// # Errors
///
/// Returns [`ConstraintKind::NotFinite`] when `got` is `NaN` or `±∞` — that
/// check runs first — and [`ConstraintKind::NotPositive`] when `got` is finite
/// but `<= 0`.
pub fn positive(config: &'static str, field: &'static str, got: f64) -> Result<(), ConfigError> {
    if !got.is_finite() {
        Err(ConfigError {
            config,
            field,
            kind: ConstraintKind::NotFinite { got },
        })
    } else if got > 0.0 {
        Ok(())
    } else {
        Err(ConfigError {
            config,
            field,
            kind: ConstraintKind::NotPositive { got },
        })
    }
}

/// Rejects a value outside the closed interval `[lo, hi]`.
///
/// # The value must be finite; the bounds need not be
///
/// `got` is a config's own field, so `NaN` and `±∞` are rejected outright as
/// [`ConstraintKind::NotFinite`] rather than as an out-of-range comparison.
/// `lo` and `hi` are the *schema* and are deliberately **not** checked:
/// `hi = f64::INFINITY` is the intended spelling of "unbounded above", so
///
/// ```
/// # use rlevo_core::config;
/// assert!(config::in_range("C", "x", 0.0, f64::INFINITY, 3.0).is_ok());
/// assert!(config::in_range("C", "x", 0.0, f64::INFINITY, f64::INFINITY).is_err());
/// ```
///
/// is the blessed way to say "non-negative, unbounded above".
///
/// # Errors
///
/// Returns [`ConstraintKind::NotFinite`] when `got` is `NaN` or `±∞` — that
/// check runs first — and [`ConstraintKind::OutOfRange`] when `got` is finite
/// but `< lo` or `> hi`.
pub fn in_range(
    config: &'static str,
    field: &'static str,
    lo: f64,
    hi: f64,
    got: f64,
) -> Result<(), ConfigError> {
    if !got.is_finite() {
        Err(ConfigError {
            config,
            field,
            kind: ConstraintKind::NotFinite { got },
        })
    } else if got >= lo && got <= hi {
        Ok(())
    } else {
        Err(ConfigError {
            config,
            field,
            kind: ConstraintKind::OutOfRange { lo, hi, got },
        })
    }
}

/// Rejects a `(low, high)` pair that is not strictly ordered (`low < high`).
///
/// Both endpoints are config *values* here — not bounds — so both must be
/// finite. `ordered(C, f, -∞, ∞)` is rejected, unlike [`in_range`]'s `lo`/`hi`.
///
/// # Errors
///
/// Returns [`ConstraintKind::NotFinite`] when either endpoint is `NaN` or `±∞`
/// (`low` is checked first), and [`ConstraintKind::NotOrdered`] when both are
/// finite but `low >= high`.
///
/// Note the pre-existing limitation that this helper names a **single** field
/// for a two-value check, so the returned `NotFinite` cannot say *which*
/// endpoint by field name; its `got` payload carries the offending value.
pub fn ordered(
    config: &'static str,
    field: &'static str,
    low: f64,
    high: f64,
) -> Result<(), ConfigError> {
    if !low.is_finite() {
        Err(ConfigError {
            config,
            field,
            kind: ConstraintKind::NotFinite { got: low },
        })
    } else if !high.is_finite() {
        Err(ConfigError {
            config,
            field,
            kind: ConstraintKind::NotFinite { got: high },
        })
    } else if low < high {
        Ok(())
    } else {
        Err(ConfigError {
            config,
            field,
            kind: ConstraintKind::NotOrdered { low, high },
        })
    }
}

/// Rejects two values that must differ but are equal.
///
/// Both are config *values*, so both must be finite. Without that guard
/// `distinct(C, f, f64::NAN, 1.0)` rejected — correctly — but reported
/// `DegenerateInterval`, which is a *wrong diagnosis*: it claims two values
/// that plainly differ are equal.
///
/// `-0.0` and `0.0` are equal under IEEE-754 and so are **not** distinct:
/// `distinct(C, f, -0.0, 0.0)` is rejected as `DegenerateInterval`, not as
/// `NotFinite` — both are ordinary finite values. As with any degenerate pair,
/// the reported `value` is `a` alone, so the error names only one of the two.
///
/// # Errors
///
/// Returns [`ConstraintKind::NotFinite`] when either value is `NaN` or `±∞`
/// (`a` is checked first), and [`ConstraintKind::DegenerateInterval`] when both
/// are finite but equal.
///
/// As with [`ordered`], one field name covers a two-value check, so
/// `NotFinite`'s `got` payload — not the field — identifies the offender.
pub fn distinct(
    config: &'static str,
    field: &'static str,
    a: f64,
    b: f64,
) -> Result<(), ConfigError> {
    if !a.is_finite() {
        Err(ConfigError {
            config,
            field,
            kind: ConstraintKind::NotFinite { got: a },
        })
    } else if !b.is_finite() {
        Err(ConfigError {
            config,
            field,
            kind: ConstraintKind::NotFinite { got: b },
        })
    // `(a - b).abs() > 0.0`, not `a != b`: the workspace warns on every clippy
    // category and `clippy::float_cmp` would fire on the latter. The two agree
    // exactly once both operands are known finite — for finite IEEE-754 inputs
    // `a - b` is nonzero iff `a != b`, gradual underflow included — so the
    // guards above make this spelling correct rather than merely accidental.
    // Do not "simplify" it back into a lint.
    } else if (a - b).abs() > 0.0 {
        Ok(())
    } else {
        Err(ConfigError {
            config,
            field,
            kind: ConstraintKind::DegenerateInterval { value: a },
        })
    }
}

/// Rejects a zero count / size / capacity.
///
/// # Errors
///
/// Returns [`ConstraintKind::Zero`] when `n == 0`.
pub fn nonzero(config: &'static str, field: &'static str, n: usize) -> Result<(), ConfigError> {
    if n == 0 {
        Err(ConfigError {
            config,
            field,
            kind: ConstraintKind::Zero,
        })
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
            kind: ConstraintKind::TooSmall {
                min: min as u64,
                got: got as u64,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ConfigError, ConstraintKind, Validate, Violations, at_least, distinct, in_range, nonzero,
        ordered, positive,
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

    /// The headline case of issue #353: `f64::INFINITY > 0.0` is `true`, so
    /// `positive` used to hand `+∞` back as a perfectly good hyperparameter at
    /// every one of its call sites. A config *value* is never legitimately
    /// infinite, so both infinities are now `NotFinite` — and the carried `got`
    /// is the offending value verbatim, so the message names what arrived.
    #[test]
    fn positive_rejects_infinity() {
        for got in [f64::INFINITY, f64::NEG_INFINITY] {
            let err = positive(C, "dt", got).unwrap_err();
            assert_eq!(err.config, C);
            assert_eq!(err.field, "dt");
            assert_eq!(
                err.kind,
                ConstraintKind::NotFinite { got },
                "{got} must be rejected as NotFinite, carrying the value verbatim"
            );
        }
    }

    /// `NaN` fails `got > 0.0` too, so it was already rejected — but as
    /// `NotPositive`, which reads as a claim about sign that `NaN` does not
    /// have. Pin the kind, not merely `is_err()`.
    #[test]
    fn positive_rejects_nan_as_not_finite() {
        let err = positive(C, "dt", f64::NAN).unwrap_err();
        assert_eq!(err.field, "dt");
        match err.kind {
            ConstraintKind::NotFinite { got } => assert!(
                got.is_nan(),
                "NotFinite must carry the offending NaN verbatim, got {got}"
            ),
            other => panic!("NaN must be rejected as NotFinite, got {other:?}"),
        }
    }

    #[test]
    fn in_range_boundaries_are_inclusive() {
        assert!(in_range(C, "gamma", 0.0, 1.0, 0.0).is_ok());
        assert!(in_range(C, "gamma", 0.0, 1.0, 1.0).is_ok());
        assert!(in_range(C, "gamma", 0.0, 1.0, 0.5).is_ok());
        let err = in_range(C, "gamma", 0.0, 1.0, 1.5).unwrap_err();
        assert_eq!(
            err.kind,
            ConstraintKind::OutOfRange {
                lo: 0.0,
                hi: 1.0,
                got: 1.5
            }
        );
        assert!(in_range(C, "gamma", 0.0, 1.0, f64::NAN).is_err());
    }

    /// `NaN` is now caught by the explicit `!got.is_finite()` guard rather than
    /// by falling out of `got >= lo && got <= hi`, so the reported kind is
    /// `NotFinite`. This test still pins the **kind**, not merely `is_err()`,
    /// and for the same paranoid reason as before: the finiteness guard is one
    /// line that a refactor can drop, and if it goes, `NaN` falls back onto the
    /// comparison chain — where the superficially equivalent
    /// `!(got < lo || got > hi)` would silently start *accepting* it. Only a
    /// kind-level assertion distinguishes "rejected as non-finite" from
    /// "rejected for some other reason", and therefore notices the day the
    /// guard is gone.
    #[test]
    fn in_range_rejects_nan_as_not_finite() {
        let err = in_range(C, "tau", 0.0, 1.0, f64::NAN).unwrap_err();
        assert_eq!(err.config, C, "NaN rejection must name the config");
        assert_eq!(err.field, "tau", "NaN rejection must name the field");
        match err.kind {
            ConstraintKind::NotFinite { got } => assert!(
                got.is_nan(),
                "NotFinite must carry the offending NaN verbatim, got {got}"
            ),
            other => panic!("NaN must be rejected as NotFinite, got {other:?}"),
        }
    }

    /// The other half of the finiteness guard. `−∞` would also fall out of a
    /// finite `[lo, hi]` on comparison alone, but `+∞` would *not* if the upper
    /// bound were `f64::INFINITY` — which is exactly the idiom at 37 call
    /// sites. So the kind assertion here is load-bearing: it pins that an
    /// infinite **value** is rejected by the guard, independently of whatever
    /// the bounds happen to be.
    #[test]
    fn in_range_rejects_infinities_as_not_finite() {
        for got in [f64::INFINITY, f64::NEG_INFINITY] {
            for (lo, hi) in [(0.0, 1.0), (0.0, f64::INFINITY)] {
                let err = in_range(C, "tau", lo, hi, got).unwrap_err();
                assert_eq!(
                    err.kind,
                    ConstraintKind::NotFinite { got },
                    "{got} in [{lo}, {hi}] must be rejected as NotFinite, carrying \
                     the value verbatim"
                );
            }
        }
    }

    /// The regression guard for every `in_range(.., f64::INFINITY, ..)` call
    /// site in the workspace (37 of them). Issue #353 tightened the rule on
    /// config *values*; it must not have touched the rule on config *bounds*,
    /// where `hi = f64::INFINITY` is the intended spelling of "unbounded
    /// above". If this test ever fails, the finiteness guard has leaked from
    /// `got` onto `lo`/`hi` and a third of the workspace's configs reject their
    /// own defaults.
    #[test]
    fn in_range_accepts_finite_value_with_infinite_upper_bound() {
        assert!(in_range(C, "x", 0.0, f64::INFINITY, 3.0).is_ok());
        assert!(in_range(C, "x", 0.0, f64::INFINITY, 0.0).is_ok());
        assert!(in_range(C, "x", 0.0, f64::INFINITY, f64::MAX).is_ok());
        assert!(in_range(C, "x", f64::NEG_INFINITY, f64::INFINITY, -1e300).is_ok());
        // The bound stays a bound: a finite value below it is still rejected.
        assert!(in_range(C, "x", 0.0, f64::INFINITY, -1.0).is_err());
    }

    #[test]
    fn ordered_requires_strict() {
        assert!(ordered(C, "clip", -1.0, 1.0).is_ok());
        assert!(ordered(C, "clip", 1.0, 1.0).is_err());
        let err = ordered(C, "clip", 2.0, 1.0).unwrap_err();
        assert_eq!(
            err.kind,
            ConstraintKind::NotOrdered {
                low: 2.0,
                high: 1.0
            }
        );
    }

    /// Both endpoints are config *values*, not bounds, so `ordered` refuses an
    /// infinite one — including `(−∞, ∞)`, which satisfies `low < high` and was
    /// therefore accepted outright. `(∞, ∞)` previously reported `NotOrdered`,
    /// a true-but-secondary diagnosis; it is a finiteness failure first.
    #[test]
    fn ordered_rejects_infinite_endpoints() {
        for (low, high) in [
            (f64::NEG_INFINITY, f64::INFINITY),
            (1.0, f64::INFINITY),
            (f64::NEG_INFINITY, 1.0),
            (f64::INFINITY, f64::INFINITY),
        ] {
            let err = ordered(C, "clip", low, high).unwrap_err();
            assert_eq!(err.field, "clip");
            // `low` is checked first, so it is the reported offender whenever
            // it is itself non-finite.
            let expected = if low.is_finite() { high } else { low };
            assert_eq!(
                err.kind,
                ConstraintKind::NotFinite { got: expected },
                "({low}, {high}) must be rejected as NotFinite, not NotOrdered"
            );
        }
    }

    #[test]
    fn distinct_rejects_equal() {
        assert!(distinct(C, "v_max", -10.0, 10.0).is_ok());
        let err = distinct(C, "v_max", 5.0, 5.0).unwrap_err();
        assert_eq!(err.kind, ConstraintKind::DegenerateInterval { value: 5.0 });
    }

    /// Closes the wrong-diagnosis bug: `distinct(NaN, 1.0)` rejected, but as
    /// `DegenerateInterval` — an error message asserting that two plainly
    /// different values are equal. `(∞, ∞)` is the converse trap: genuinely
    /// equal, but the actionable complaint is the infinity, not the degeneracy.
    /// None of these may report `DegenerateInterval`.
    #[test]
    fn distinct_rejects_non_finite() {
        for (a, b) in [
            (f64::NAN, 1.0),
            (1.0, f64::NAN),
            (f64::INFINITY, f64::INFINITY),
        ] {
            let err = distinct(C, "v_max", a, b).unwrap_err();
            assert_eq!(err.field, "v_max");
            // `a` is checked first, so it is the reported offender unless it is
            // itself finite.
            let expected = if a.is_finite() { b } else { a };
            match err.kind {
                ConstraintKind::NotFinite { got } => assert!(
                    got.to_bits() == expected.to_bits() || (got.is_nan() && expected.is_nan()),
                    "NotFinite must carry the offending value verbatim: \
                     ({a}, {b}) reported {got}, expected {expected}"
                ),
                other => panic!("({a}, {b}) must be rejected as NotFinite, got {other:?}"),
            }
        }
    }

    /// The boundary marker for the finiteness guard. `-0.0` is the one input
    /// that *looks* non-finite-adjacent — a signed zero, the neighbour of every
    /// underflow — but is an ordinary finite value and must still take the old
    /// path. IEEE-754 has `-0.0 == 0.0`, so the pair is degenerate, and the
    /// asserted kind is load-bearing: `DegenerateInterval`, never `NotFinite`.
    /// It also pins that only one of the two values is reported, `a` verbatim.
    #[test]
    fn distinct_treats_signed_zeros_as_equal() {
        for (a, b) in [(-0.0_f64, 0.0_f64), (0.0, -0.0), (-0.0, -0.0)] {
            let err = distinct(C, "v_max", a, b).unwrap_err();
            assert_eq!(err.field, "v_max");
            match err.kind {
                ConstraintKind::DegenerateInterval { value } => assert_eq!(
                    value.to_bits(),
                    a.to_bits(),
                    "DegenerateInterval must report `a` verbatim, sign of zero \
                     included: ({a}, {b}) reported {value}"
                ),
                other => panic!(
                    "({a}, {b}) are equal finite values and must be rejected as \
                     DegenerateInterval, got {other:?}"
                ),
            }
        }
    }

    #[test]
    fn nonzero_and_at_least() {
        assert!(nonzero(C, "max_steps", 1).is_ok());
        assert_eq!(
            nonzero(C, "max_steps", 0).unwrap_err().kind,
            ConstraintKind::Zero
        );
        assert!(at_least(C, "pop_size", 2, 2).is_ok());
        assert_eq!(
            at_least(C, "pop_size", 1, 2).unwrap_err().kind,
            ConstraintKind::TooSmall { min: 2, got: 1 }
        );
    }

    #[test]
    fn violations_accumulate_every_failure() {
        let mut v = Violations::new();
        v.check(positive(C, "a", -1.0));
        v.check(nonzero(C, "b", 4));
        v.check(in_range(C, "c", 0.0, 1.0, 2.0));
        let errs = v.into_result().unwrap_err();
        assert_eq!(errs.len(), 2);
        assert_eq!(errs[0].field, "a");
        assert_eq!(errs[1].field, "c");
    }

    #[test]
    fn violations_all_ok_is_ok() {
        let mut v = Violations::new();
        v.check(positive(C, "a", 1.0));
        v.check(nonzero(C, "b", 4));
        assert!(v.into_result().is_ok());
    }

    #[test]
    fn validate_all_default_wraps_first_error() {
        struct One;
        impl Validate for One {
            fn validate(&self) -> Result<(), ConfigError> {
                positive(C, "x", 0.0)
            }
        }
        let errs = One.validate_all().unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].field, "x");
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
