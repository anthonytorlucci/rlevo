//! Target-network update rule: the [`PolyakTau`] newtype, the [`TargetUpdate`]
//! config type, and [`TargetUpdateError`].
//!
//! Every off-policy algorithm in this crate (DQN, C51, QR-DQN, DDPG, TD3, SAC)
//! keeps a lagged copy of one or more networks and periodically moves it toward
//! the live network. Historically each agent spelled that rule out with two
//! independent scalars — a `tau: f32` and a `target_update_frequency: usize` —
//! and the two meant *different things in different agents*: the value-based
//! agents treated the frequency as a **hard-copy** cadence and ignored τ, while
//! the actor-critic agents treated it as a **Polyak** cadence. Same field name,
//! two mechanisms.
//!
//! [`TargetUpdate`] collapses that into **one** mechanism with two knobs:
//!
//! - the **cadence** [`every`](TargetUpdate::every) decides *when* an update
//!   fires, and
//! - the **coefficient** [`tau`](TargetUpdate::tau) decides *how far* the target
//!   moves when it does — `target ← (1 − τ)·target + τ·active`.
//!
//! A hard copy is not a second mechanism: it is the degenerate case `τ = 1.0`,
//! where the blend reduces to `target ← active`. [`TargetUpdate::hard`] is a
//! constructor for that point, not a variant. See ADR 0058 for the decision to
//! unify the two, and ADR 0059 for the cadence's unit.
//!
//! # Why no enum
//!
//! The obvious alternative is an enum with `Hard { every }` and
//! `Polyak { tau, every }` arms. It is the wrong shape here, because
//! `Hard { every: n }` and `Polyak { tau: 1.0, every: n }` would denote the
//! *same* state: two spellings of one behaviour, which every downstream `match`
//! must then either handle twice or accidentally handle once. That
//! representable-but-equivalent redundancy is exactly the defect this type
//! exists to remove — replacing two overloaded scalars with two overloaded
//! variants would move the ambiguity rather than delete it. A struct with a
//! validated τ has a single representation per behaviour, and the
//! [`hard`](TargetUpdate::hard) constructor recovers all of the discoverability
//! a variant would have offered.
//!
//! # Why `PolyakTau` and not [`Probability`]
//!
//! [`Probability`]'s invariant is the *closed* unit interval `0 <= p <= 1`, so
//! it admits `0.0`. A τ of `0.0` is a permanently frozen target — the update
//! fires on schedule and moves nothing, a silent no-op that looks like a
//! configured update. [`PolyakTau`] excludes zero: its invariant is the
//! half-open interval `0 < τ <= 1`. A caller who genuinely wants no target
//! tracking omits the target network, rather than configuring one that never
//! moves.
//!
//! # Why this lives in the RL crate
//!
//! Target networks are a gradient-based-RL construct with exactly one consuming
//! crate, and the function that applies the blend
//! ([`polyak_update`](crate::utils::polyak_update)) already lives in
//! [`crate::utils`]. ADR 0031 placed [`Bounds`](rlevo_core::bounds::Bounds) in
//! `rlevo-core` because two crates consumed it; that rationale does not apply
//! here, so this type stays next to its only consumer.
//!
//! # Examples
//!
//! ```
//! use rlevo_reinforcement_learning::target::TargetUpdate;
//!
//! // SAC-style: a small Polyak step on every gradient update.
//! let soft = TargetUpdate::polyak(0.005, 1);
//! assert!(!soft.is_hard());
//! assert_eq!(soft.every(), 1);
//!
//! // DQN-style: a full weight copy every 10 000 gradient updates.
//! let hard = TargetUpdate::hard(10_000);
//! assert!(hard.is_hard());
//! assert_eq!(hard.tau(), 1.0);
//!
//! // The cadence gate: `fires_at` yields the τ to apply, or `None`.
//! assert_eq!(soft.fires_at(7), Some(f64::from(0.005_f32)));
//! assert_eq!(hard.fires_at(7), None);
//! assert_eq!(hard.fires_at(10_000), Some(1.0));
//! ```
//!
//! Scale that `10_000` to your run before copying it: it is the Atari-derived
//! figure, and in gradient updates it is about 40 000 environment steps at
//! DQN's default `train_frequency: 4`. A classic-control run of a few tens of
//! thousands of env steps wants a much smaller cadence; which values ship by
//! default is under review in issue #337.
//!
//! [`Probability`]: rlevo_core::probability::Probability

use std::num::NonZeroUsize;

/// A Polyak (soft-update) coefficient τ in the half-open interval `(0, 1]`,
/// valid by construction.
///
/// τ is the interpolation weight of `target ← (1 − τ)·target + τ·active`. A
/// `PolyakTau` can never hold a `NaN`, an infinity, a negative, a zero, or a
/// value above one: every constructor enforces `0.0 < τ <= 1.0`, which a
/// `NaN`/`Inf` fails. Both excluded endpoints matter:
///
/// - `τ = 0.0` would be a frozen target — the update fires on schedule and
///   moves nothing. That is why this is not a
///   [`Probability`](rlevo_core::probability::Probability), whose invariant is
///   the closed interval `[0, 1]`.
/// - `τ > 1.0` would overshoot the live network, extrapolating past it rather
///   than interpolating toward it.
///
/// `τ = 1.0` is deliberately *included*: it is the hard copy `target ← active`.
///
/// Construct with [`new`](Self::new) for literals (panics on an invalid value)
/// or [`try_new`](Self::try_new) for runtime data (returns
/// [`TargetUpdateError`]).
///
/// # Examples
///
/// ```
/// use rlevo_reinforcement_learning::target::PolyakTau;
///
/// let tau = PolyakTau::new(0.005);
/// assert_eq!(tau.get(), 0.005);
///
/// assert!(PolyakTau::try_new(1.0).is_ok()); // the hard-copy endpoint
/// assert!(PolyakTau::try_new(0.0).is_err()); // a frozen target
/// assert!(PolyakTau::try_new(1.5).is_err()); // overshoots
/// assert!(PolyakTau::try_new(f32::NAN).is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolyakTau(f32);

impl PolyakTau {
    /// Builds a τ from a compile-time-known value, panicking on an invalid one.
    ///
    /// This is the constructor for literals and `Default`s — the bad value is
    /// right at the call site, mirroring the documented builder-setter panic
    /// exception of ADR 0026. Prefer [`try_new`](Self::try_new) for any value
    /// derived from runtime or user-supplied data.
    ///
    /// # Panics
    ///
    /// Panics when `tau` is outside `(0, 1]` — that is, when it is zero,
    /// negative, above one, `NaN`, or infinite.
    #[must_use]
    pub const fn new(tau: f32) -> Self {
        assert!(
            tau > 0.0 && tau <= 1.0,
            "PolyakTau::new: value must lie in (0, 1] (zero, negative, above-one, NaN, and infinite are rejected)"
        );
        Self(tau)
    }

    /// Builds a τ from a runtime / user-supplied value.
    ///
    /// # Errors
    ///
    /// Returns [`TargetUpdateError::Tau`] when `tau` is outside `(0, 1]` —
    /// zero, negative, above one, `NaN`, or infinite. Finiteness needs no
    /// separate check: `NaN` fails both comparisons, `+∞` fails `<= 1.0`, and
    /// `−∞` fails `> 0.0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rlevo_reinforcement_learning::target::{PolyakTau, TargetUpdateError};
    ///
    /// assert_eq!(
    ///     PolyakTau::try_new(0.0),
    ///     Err(TargetUpdateError::Tau { got: 0.0 })
    /// );
    /// ```
    pub fn try_new(tau: f32) -> Result<Self, TargetUpdateError> {
        if tau > 0.0 && tau <= 1.0 {
            Ok(Self(tau))
        } else {
            Err(TargetUpdateError::Tau { got: tau })
        }
    }

    /// The wrapped coefficient, guaranteed to lie in `(0, 1]`.
    #[must_use]
    pub const fn get(self) -> f32 {
        self.0
    }
}

/// One target-network update rule: a cadence and a Polyak coefficient.
///
/// The rule is read as "every `every` gradient updates, move the target toward
/// the live network by τ": `target ← (1 − τ)·target + τ·active`. There is a
/// single mechanism — a hard copy is the degenerate `τ = 1.0`, reachable via
/// [`hard`](Self::hard), not a second variant. See the
/// [module docs](self#why-no-enum) for why this is a struct rather than an
/// enum.
///
/// The cadence counts **gradient/optimizer updates, not environment steps**;
/// see [`every`](Self::every).
///
/// Construct with [`polyak`](Self::polyak) or [`hard`](Self::hard) for literals
/// and `Default` impls (both panic on an invalid value), or with
/// [`try_polyak`](Self::try_polyak) for runtime data. Read the rule back with
/// [`fires_at`](Self::fires_at), which combines the gate and the coefficient.
///
/// # Examples
///
/// ```
/// use rlevo_reinforcement_learning::target::TargetUpdate;
///
/// // A soft update on every gradient step (SAC/DDPG/TD3 convention).
/// let soft = TargetUpdate::polyak(0.005, 1);
/// assert_eq!(soft.every(), 1);
/// assert!(!soft.is_hard());
///
/// // A hard copy every 10 000 gradient updates (DQN/C51/QR-DQN convention).
/// let hard = TargetUpdate::hard(10_000);
/// assert_eq!(hard.tau(), 1.0);
/// assert!(hard.is_hard());
/// ```
///
/// The `10_000` there is the Atari-derived figure — about 40 000 environment
/// steps at DQN's default `train_frequency: 4` — so classic-control-scale runs
/// want a much smaller cadence (issue #337).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TargetUpdate {
    /// How far the target moves when an update fires.
    tau: PolyakTau,
    /// How many gradient updates elapse between fires; never zero.
    every: NonZeroUsize,
}

impl TargetUpdate {
    /// A full weight copy (`τ = 1.0`) every `every` gradient updates.
    ///
    /// This is the value-based (DQN / C51 / QR-DQN) convention. It is a
    /// convenience over [`polyak(1.0, every)`](Self::polyak), not a distinct
    /// mechanism: the two are the same value, and
    /// [`is_hard`](Self::is_hard) reports `true` for both.
    ///
    /// Size `every` against your own run, because it counts **gradient
    /// updates**: the familiar `10_000` is Atari-derived and lands near 40 000
    /// environment steps under DQN's default `train_frequency: 4`, so a
    /// classic-control run of a few tens of thousands of env steps would fire
    /// it only once or twice. Which figures ship by default is under review in
    /// issue #337.
    ///
    /// # Panics
    ///
    /// Panics when `every` is zero — a cadence of zero would never fire, which
    /// is a frozen target rather than a configured update.
    ///
    /// # Examples
    ///
    /// ```
    /// use rlevo_reinforcement_learning::target::TargetUpdate;
    ///
    /// let rule = TargetUpdate::hard(10_000);
    /// assert_eq!(rule, TargetUpdate::polyak(1.0, 10_000));
    /// ```
    #[must_use]
    pub const fn hard(every: usize) -> Self {
        Self::polyak(1.0, every)
    }

    /// A soft update of size `tau` every `every` gradient updates, panicking on
    /// an invalid argument.
    ///
    /// This is the constructor for literals and `Default`s — the bad value is
    /// right at the call site, mirroring the documented builder-setter panic
    /// exception of ADR 0026. Prefer [`try_polyak`](Self::try_polyak) for any
    /// value derived from runtime or user-supplied data.
    ///
    /// # Panics
    ///
    /// Panics when `tau` is outside `(0, 1]` (zero, negative, above one, `NaN`,
    /// or infinite), or when `every` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use rlevo_reinforcement_learning::target::TargetUpdate;
    ///
    /// let rule = TargetUpdate::polyak(0.005, 1);
    /// assert_eq!(rule.every(), 1);
    /// ```
    #[must_use]
    pub const fn polyak(tau: f32, every: usize) -> Self {
        // `match` rather than `let … else` keeps the body plainly const-legal.
        match NonZeroUsize::new(every) {
            Some(every) => Self {
                tau: PolyakTau::new(tau),
                every,
            },
            None => {
                panic!("TargetUpdate::polyak: `every` must be >= 1 (a cadence of 0 never fires)")
            }
        }
    }

    /// A soft update of size `tau` every `every` gradient updates, built from
    /// runtime / user-supplied values.
    ///
    /// Not a `const fn`: it mirrors the `try_new` convention of
    /// [`Probability`](rlevo_core::probability::Probability) and
    /// [`NonNegativeRate`](rlevo_core::rate::NonNegativeRate), which are also
    /// non-`const` on the fallible path.
    ///
    /// # Errors
    ///
    /// Returns [`TargetUpdateError::Tau`] when `tau` is outside `(0, 1]`, or
    /// [`TargetUpdateError::ZeroEvery`] when `every` is zero. τ is validated
    /// first.
    ///
    /// # Examples
    ///
    /// ```
    /// use rlevo_reinforcement_learning::target::{TargetUpdate, TargetUpdateError};
    ///
    /// assert!(TargetUpdate::try_polyak(0.005, 1).is_ok());
    /// assert_eq!(
    ///     TargetUpdate::try_polyak(0.005, 0),
    ///     Err(TargetUpdateError::ZeroEvery)
    /// );
    /// ```
    pub fn try_polyak(tau: f32, every: usize) -> Result<Self, TargetUpdateError> {
        let tau = PolyakTau::try_new(tau)?;
        let every = NonZeroUsize::new(every).ok_or(TargetUpdateError::ZeroEvery)?;
        Ok(Self { tau, every })
    }

    /// The Polyak coefficient, widened to `f64`.
    ///
    /// `f64` because that is what every `soft_update` model-trait method takes;
    /// returning it pre-widened keeps the cast out of each agent's learn step.
    /// Use [`PolyakTau::get`] on a bare [`PolyakTau`] when the `f32` is wanted.
    // `f64::from` is the idiomatic widening, but it is not a `const fn`, so the
    // `as` cast is the only way to keep this accessor usable in const context.
    // The conversion is exact — every `f32` is representable in `f64` — and
    // `clippy::cast_lossless` does not fire here precisely because it has no
    // const-compatible replacement to suggest, so no `allow` is carried.
    #[must_use]
    pub const fn tau(self) -> f64 {
        self.tau.get() as f64
    }

    /// The cadence: the number of **gradient (optimizer) updates** between
    /// fires, never zero.
    ///
    /// The unit is deliberate and is the point of this type (ADR 0059). The
    /// counter a caller feeds to [`fires_at`](Self::fires_at) must be its
    /// gradient-update counter — advanced once per learn step that *attempts*
    /// an optimizer step, unconditionally, including on a non-finite-loss skip
    /// — and **not** its environment-step counter. The two differ by the train
    /// frequency and by any warm-up period, so reading a cadence in the wrong
    /// unit silently rescales every target update in the run.
    #[must_use]
    pub const fn every(self) -> usize {
        self.every.get()
    }

    /// `true` when `τ == 1.0`, i.e. when a fired update is a full weight copy.
    ///
    /// This is a query over the single representation, not a tag: `hard(n)` and
    /// `polyak(1.0, n)` are the same value and both report `true`.
    // Exact equality is the intended predicate: τ is stored verbatim from the
    // constructor and never arithmetically derived, so `1.0` is a bit-exact
    // sentinel rather than the result of a computation.
    #[allow(clippy::float_cmp)]
    #[must_use]
    pub const fn is_hard(self) -> bool {
        self.tau.get() == 1.0
    }

    /// `Some(τ)` when an update fires at gradient-update index `updates`, else
    /// `None`.
    ///
    /// `updates` is a **gradient/optimizer-update** count, not an
    /// environment-step count (see [`every`](Self::every)). Callers pass their
    /// *post-increment* counter — every in-tree agent does
    /// `self.critic_updates += 1;` immediately before the gate, so the first
    /// call observes `1`.
    ///
    /// The predicate is `updates.is_multiple_of(every)`, matching the hand-rolled
    /// gate it replaces exactly. Consequently `every = 1` fires on every call,
    /// `every = 2` fires at 2, 4, 6, …, and — because zero is a multiple of
    /// every positive integer — index `0` also reports a fire. A post-increment
    /// counter never presents `0`, so that case is unreachable through the
    /// documented calling convention; it is stated here so the predicate is not
    /// mistaken for a "has at least one update happened" check.
    ///
    /// # Examples
    ///
    /// ```
    /// use rlevo_reinforcement_learning::target::TargetUpdate;
    ///
    /// let rule = TargetUpdate::hard(2);
    /// assert_eq!(rule.fires_at(1), None);
    /// assert_eq!(rule.fires_at(2), Some(1.0));
    /// assert_eq!(rule.fires_at(3), None);
    /// assert_eq!(rule.fires_at(4), Some(1.0));
    /// ```
    #[must_use]
    pub const fn fires_at(self, updates: usize) -> Option<f64> {
        if updates.is_multiple_of(self.every.get()) {
            Some(self.tau())
        } else {
            None
        }
    }
}

/// The ways constructing a [`PolyakTau`] or a [`TargetUpdate`] can fail.
///
/// Allocation-free and `Copy`, carrying the offending value where there is one.
/// Returned by [`PolyakTau::try_new`] and [`TargetUpdate::try_polyak`]. A
/// dedicated error rather than
/// [`ConfigError`](rlevo_core::config::ConfigError) because construction has no
/// config/field name to report — a config that builds a [`TargetUpdate`] from
/// its own scalar fields wraps this as needed (ADR 0027 §5).
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum TargetUpdateError {
    /// The Polyak coefficient was outside the half-open interval `(0, 1]`.
    #[error(
        "invalid Polyak coefficient: {got} must lie in (0, 1] (zero, negative, above-one, NaN, and infinite are rejected)"
    )]
    Tau {
        /// The value that was supplied.
        got: f32,
    },
    /// The cadence was zero, which would never fire.
    #[error("invalid target-update cadence: `every` must be >= 1 (a cadence of 0 never fires)")]
    ZeroEvery,
}

#[cfg(test)]
mod tests {
    use super::{PolyakTau, TargetUpdate, TargetUpdateError};

    use approx::assert_abs_diff_eq;

    /// τ is stored and read back without arithmetic, so the only slack these
    /// assertions need is the `f32`→`f64` widening, which is exact.
    const EPS: f64 = 1e-12;

    // -----------------------------------------------------------------------
    // PolyakTau construction
    // -----------------------------------------------------------------------

    #[test]
    fn try_new_accepts_open_lower_closed_upper_interval() {
        assert!(PolyakTau::try_new(1.0).is_ok()); // hard-copy endpoint
        assert!(PolyakTau::try_new(0.005).is_ok()); // the SAC/DDPG default
        assert!(PolyakTau::try_new(f32::MIN_POSITIVE).is_ok()); // just above zero
    }

    #[test]
    fn try_new_rejects_zero_negative_zero_nan_inf_and_above_one() {
        assert_eq!(
            PolyakTau::try_new(0.0),
            Err(TargetUpdateError::Tau { got: 0.0 })
        );
        // `-0.0 > 0.0` is false, so negative zero is rejected like zero.
        assert!(PolyakTau::try_new(-0.0).is_err());
        assert!(PolyakTau::try_new(-0.5).is_err());
        assert!(PolyakTau::try_new(f32::NAN).is_err());
        assert!(PolyakTau::try_new(f32::INFINITY).is_err());
        assert!(PolyakTau::try_new(f32::NEG_INFINITY).is_err());
        // The smallest `f32` strictly above one must not slip through.
        assert!(PolyakTau::try_new(1.000_000_1).is_err());
    }

    #[test]
    fn new_round_trips_the_constructed_value() {
        assert_abs_diff_eq!(
            f64::from(PolyakTau::new(0.005).get()),
            f64::from(0.005_f32),
            epsilon = EPS
        );
        assert_abs_diff_eq!(f64::from(PolyakTau::new(1.0).get()), 1.0, epsilon = EPS);
    }

    #[test]
    #[should_panic(expected = "must lie in (0, 1]")]
    fn new_panics_on_zero() {
        let _ = PolyakTau::new(0.0);
    }

    #[test]
    #[should_panic(expected = "must lie in (0, 1]")]
    fn new_panics_above_one() {
        let _ = PolyakTau::new(1.000_000_1);
    }

    #[test]
    #[should_panic(expected = "must lie in (0, 1]")]
    fn new_panics_on_nan() {
        let _ = PolyakTau::new(f32::NAN);
    }

    #[test]
    #[should_panic(expected = "must lie in (0, 1]")]
    fn new_panics_on_infinity() {
        let _ = PolyakTau::new(f32::INFINITY);
    }

    // -----------------------------------------------------------------------
    // TargetUpdate construction
    // -----------------------------------------------------------------------

    #[test]
    fn try_polyak_rejects_zero_cadence() {
        assert_eq!(
            TargetUpdate::try_polyak(0.005, 0),
            Err(TargetUpdateError::ZeroEvery)
        );
    }

    #[test]
    fn try_polyak_validates_tau_before_cadence() {
        // Both arguments are invalid; τ is checked first, so the τ error wins.
        assert_eq!(
            TargetUpdate::try_polyak(0.0, 0),
            Err(TargetUpdateError::Tau { got: 0.0 })
        );
    }

    #[test]
    #[should_panic(expected = "`every` must be >= 1")]
    fn polyak_panics_on_zero_cadence() {
        let _ = TargetUpdate::polyak(0.005, 0);
    }

    #[test]
    #[should_panic(expected = "`every` must be >= 1")]
    fn hard_panics_on_zero_cadence() {
        let _ = TargetUpdate::hard(0);
    }

    #[test]
    fn hard_equals_try_polyak_at_tau_one() {
        for every in [1_usize, 2, 3, 10_000] {
            assert_eq!(
                TargetUpdate::hard(every),
                TargetUpdate::try_polyak(1.0, every).expect("τ = 1.0 and every >= 1 are valid"),
                "hard({every}) must be the same value as polyak(1.0, {every})"
            );
        }
    }

    #[test]
    fn tau_round_trips_the_constructed_value() {
        assert_abs_diff_eq!(
            TargetUpdate::polyak(0.005, 1).tau(),
            f64::from(0.005_f32),
            epsilon = EPS
        );
        assert_abs_diff_eq!(TargetUpdate::hard(10_000).tau(), 1.0, epsilon = EPS);
        assert_eq!(TargetUpdate::polyak(0.25, 4).every(), 4);
    }

    // -----------------------------------------------------------------------
    // is_hard
    // -----------------------------------------------------------------------

    #[test]
    fn is_hard_is_true_only_at_tau_one() {
        assert!(TargetUpdate::hard(1).is_hard());
        assert!(TargetUpdate::polyak(1.0, 7).is_hard());
        for tau in [0.005_f32, 0.25, 0.5, 0.999_999] {
            assert!(
                !TargetUpdate::polyak(tau, 1).is_hard(),
                "τ = {tau} is a soft update"
            );
        }
    }

    // -----------------------------------------------------------------------
    // fires_at cadence table
    //
    // This table pins the gate's phase. An off-by-one here silently rescales
    // every agent's target-update cadence, so the expectations are written out
    // literally rather than computed.
    // -----------------------------------------------------------------------

    /// Whether an update fires at indices `0..=12`, per cadence.
    ///
    /// Index `0` fires because zero is a multiple of every positive integer;
    /// callers pass a post-increment counter, so index `0` never occurs in
    /// practice. See [`TargetUpdate::fires_at`].
    #[test]
    fn fires_at_cadence_table() {
        #[rustfmt::skip]
        let table: [(usize, [bool; 13]); 4] = [
            //        idx: 0     1      2     3      4     5      6     7      8     9      10    11     12
            (1,     [true, true,  true, true,  true, true,  true, true,  true, true,  true, true,  true]),
            (2,     [true, false, true, false, true, false, true, false, true, false, true, false, true]),
            (3,     [true, false, false, true, false, false, true, false, false, true, false, false, true]),
            (10_000, [true, false, false, false, false, false, false, false, false, false, false, false, false]),
        ];

        for (every, expected) in table {
            let rule = TargetUpdate::polyak(0.005, every);
            for (updates, &fires) in expected.iter().enumerate() {
                let got = rule.fires_at(updates);
                assert_eq!(
                    got.is_some(),
                    fires,
                    "every = {every}: fires_at({updates}) should be {}",
                    if fires { "Some" } else { "None" }
                );
                if let Some(tau) = got {
                    assert_abs_diff_eq!(tau, f64::from(0.005_f32), epsilon = EPS);
                }
            }
        }
    }

    #[test]
    fn fires_at_reaches_a_large_cadence() {
        let rule = TargetUpdate::hard(10_000);
        assert_eq!(rule.fires_at(9_999), None);
        assert_eq!(rule.fires_at(10_000), Some(1.0));
        assert_eq!(rule.fires_at(10_001), None);
        assert_eq!(rule.fires_at(20_000), Some(1.0));
    }

    // -----------------------------------------------------------------------
    // Errors
    // -----------------------------------------------------------------------

    #[test]
    fn error_display_names_the_problem() {
        let tau = TargetUpdateError::Tau { got: 1.5 }.to_string();
        assert!(tau.contains("1.5"));
        assert!(tau.contains("(0, 1]"));

        let cadence = TargetUpdateError::ZeroEvery.to_string();
        assert!(cadence.contains("`every`"));
    }
}
