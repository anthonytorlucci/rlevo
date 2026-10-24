//! Concrete reward types for reinforcement learning environments.
//!
//! This module provides [`ScalarReward`], a lightweight wrapper around an
//! `f32`: the reward every environment emits per step, carried on its
//! [`Snapshot`](crate::environment::Snapshot). It is a type, not a trait
//! (ADR 0090); a vector reward would return as a new type beside it, and
//! named reward components already travel in
//! [`SnapshotMetadata`](crate::environment::SnapshotMetadata).
//!
//! ```
//! use rlevo_core::reward::ScalarReward;
//!
//! let total = ScalarReward::new(1.5) + ScalarReward::new(-0.5);
//! assert_eq!(f32::from(total), 1.0);
//! assert_eq!(ScalarReward::default(), ScalarReward::new(0.0));
//! ```

use serde::{Deserialize, Serialize};
use std::ops::Add;

/// Scalar reward wrapping an `f32`.
///
/// The inner value is public so environments can construct a reward with the
/// tuple-struct form (`ScalarReward(0.5)`) where it aids brevity. Prefer
/// [`ScalarReward::new`] from external crates for readability.
///
/// The default is zero, the additive identity.
///
/// # Finiteness
///
/// An environment **should** emit finite values. The framework reads every
/// reward as an `f32` at its ingestion points, so a `NaN` or an infinity
/// arriving there is an environment bug — not user input that some layer is
/// expected to sanitise.
///
/// `rlevo-core` enforces none of this. Neither the tuple form nor
/// [`ScalarReward::new`] validates the value (ADR 0065 closed that
/// question). What follows is what the framework *actually does* when the
/// obligation is broken, and it is three different things at three places:
///
/// - **Dropped at replay ingestion.** The reinforcement-learning crate's reward
///   guard (`FiniteRewardGuard`) refuses to store a transition whose reward is
///   non-finite, counts every such drop, and logs a warning on an escalating
///   schedule. The transition never reaches the buffer, so learning is not
///   poisoned. (ADR 0065.)
/// - **Transited, deliberately, by the reported score.** Episode return
///   accumulates the raw value, so an affected episode reports a non-finite
///   return, and every mean over it transits it rather than sanitising it.
///   This is intentional: a non-finite return is a true statement about a run
///   whose environment emitted a non-finite reward, and hiding it would make the
///   bug harder to find, not rarer. (ADR 0065 and ADR 0070.)
/// - **Counted and reported by the evaluation harness.** The evaluator
///   accumulates raw (matching the above), logs a warning, and records a
///   `return/non_finite_steps` counter on the trial report. (ADR 0078.)
///
/// ## Both signs corrupt a report, and neither announces itself
///
/// This is not a `NaN`-only hazard. For trial returns `[1, 5, X, 20, 100]` with a
/// success threshold of `10`, the harness's own aggregates come out as:
///
/// | `X`     | mean | median | std   | min | max   | success rate |
/// | ------- | ---- | ------ | ----- | --- | ----- | ------------ |
/// | finite  | 27.0 | 9.0    | 41.42 | 1   | 100   | 0.40         |
/// | `NaN`   | NaN  | 20.0   | NaN   | 1   | 100   | 0.40         |
/// | `+inf`  | inf  | 20.0   | NaN   | 1   | inf   | 0.60         |
///
/// `NaN` is scored as a *failure* the agent may not have earned; `+inf` is scored
/// as a *success* it certainly did not, and poisons `max` besides. In both cases
/// the median silently shifts (9.0 to 20.0) while min stays finite-looking, so a
/// report can present a poisoned aggregate next to plausible order statistics
/// with no single tell. Treat a non-finite reward as a defect to fix upstream in
/// the environment, not as a condition to detect downstream.
///
/// The enforcement that does exist lives at the points where the reward has
/// already been erased to `f32` — by then the type that produced it is gone, and
/// with it any chance of a useful diagnostic. This section is the obligation;
/// meeting it is the implementor's job.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct ScalarReward(pub f32);

impl ScalarReward {
    /// Construct a new reward from a scalar value.
    ///
    /// # Examples
    ///
    /// ```
    /// use rlevo_core::reward::ScalarReward;
    ///
    /// let reward = ScalarReward::new(-1.0);
    /// assert_eq!(f32::from(reward), -1.0_f32);
    /// ```
    #[must_use]
    pub const fn new(value: f32) -> Self {
        Self(value)
    }

    /// Unwrap the inner scalar value.
    ///
    /// # Examples
    ///
    /// ```
    /// use rlevo_core::reward::ScalarReward;
    ///
    /// let reward = ScalarReward::new(3.14);
    /// assert!((reward.value() - 3.14_f32).abs() < 1e-6);
    /// ```
    #[must_use]
    pub const fn value(self) -> f32 {
        self.0
    }
}

impl Add for ScalarReward {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl From<ScalarReward> for f32 {
    fn from(reward: ScalarReward) -> Self {
        reward.0
    }
}

impl From<f32> for ScalarReward {
    fn from(value: f32) -> Self {
        Self(value)
    }
}

#[cfg(test)]
mod tests {
    // These tests assert exact round-trip of values that are stored and read
    // back without arithmetic, so bit-exact equality is the property under
    // test; an approximate comparison would weaken them.
    #![allow(clippy::float_cmp)]
    use super::*;

    #[test]
    fn zero_is_additive_identity() {
        let zero = ScalarReward::default();
        let reward = ScalarReward::new(42.5);
        assert_eq!(zero + reward, reward);
        assert_eq!(reward + zero, reward);
    }

    #[test]
    fn addition() {
        let total = ScalarReward::new(10.0) + ScalarReward::new(25.5);
        assert_eq!(total, ScalarReward::new(35.5));
    }

    #[test]
    fn into_f32() {
        let reward: f32 = ScalarReward::new(4.25).into();
        assert!((reward - 4.25).abs() < 1e-6);
    }

    #[test]
    fn from_f32() {
        let reward: ScalarReward = 1.5_f32.into();
        assert_eq!(reward, ScalarReward::new(1.5));
    }

    #[test]
    fn value_accessor_matches_inner() {
        let reward = ScalarReward::new(-1.5);
        assert_eq!(reward.value(), -1.5);
    }
}
