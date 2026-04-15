//! Concrete reward types for reinforcement learning environments.
//!
//! This module provides [`ScalarReward`], a lightweight wrapper around an
//! `f32` that satisfies the [`Reward`] trait. Most environments that emit a
//! single numeric reward per step should use this type; custom aggregation
//! schemes (distributional, vector-valued, etc.) can implement [`Reward`]
//! directly.
//!
//! ```
//! use evorl_core::base::Reward;
//! use evorl_core::reward::ScalarReward;
//!
//! let total = ScalarReward::new(1.5) + ScalarReward::new(-0.5);
//! assert_eq!(f32::from(total), 1.0);
//! assert_eq!(ScalarReward::zero(), ScalarReward::new(0.0));
//! ```

use crate::base::Reward;
use serde::{Deserialize, Serialize};
use std::ops::Add;

/// Scalar reward wrapping an `f32`.
///
/// The inner value is public so environments can construct a reward with the
/// tuple-struct form (`ScalarReward(0.5)`) where it aids brevity. Prefer
/// [`ScalarReward::new`] from external crates for readability.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct ScalarReward(pub f32);

impl ScalarReward {
    /// Construct a new reward from a scalar value.
    #[must_use]
    pub const fn new(value: f32) -> Self {
        Self(value)
    }

    /// Unwrap the inner scalar value.
    #[must_use]
    pub const fn value(self) -> f32 {
        self.0
    }
}

impl Reward for ScalarReward {
    fn zero() -> Self {
        Self(0.0)
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
    use super::*;

    #[test]
    fn zero_is_additive_identity() {
        let zero = ScalarReward::zero();
        let r = ScalarReward::new(42.5);
        assert_eq!(zero + r, r);
        assert_eq!(r + zero, r);
    }

    #[test]
    fn addition() {
        let total = ScalarReward::new(10.0) + ScalarReward::new(25.5);
        assert_eq!(total, ScalarReward::new(35.5));
    }

    #[test]
    fn into_f32() {
        let f: f32 = ScalarReward::new(4.25).into();
        assert!((f - 4.25).abs() < 1e-6);
    }

    #[test]
    fn from_f32() {
        let r: ScalarReward = 1.5_f32.into();
        assert_eq!(r, ScalarReward::new(1.5));
    }

    #[test]
    fn value_accessor_matches_inner() {
        let r = ScalarReward::new(-1.5);
        assert_eq!(r.value(), -1.5);
    }
}
