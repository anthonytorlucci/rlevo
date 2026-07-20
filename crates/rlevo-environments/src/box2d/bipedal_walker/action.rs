//! Action type for the `BipedalWalker` environment.
//!
//! [`BipedalWalkerAction`] wraps four motor velocity targets — one per joint —
//! each clamped to `[-1, 1]`. The targets are scaled by the per-joint speed
//! constants (`speed_hip`, `speed_knee`) inside `apply_motors` before being
//! passed to the `Rapier2D` impulse-joint motor.

use rlevo_core::action::{BoundedAction, ContinuousAction};
use rlevo_core::base::Action;
use serde::{Deserialize, Serialize};

/// 4-dimensional continuous action for `BipedalWalker`.
///
/// Components (all in `[-1, 1]`):
/// * `[0]` hip1 motor target (positive = forward)
/// * `[1]` knee1 motor target (positive = extend)
/// * `[2]` hip2 motor target
/// * `[3]` knee2 motor target
///
/// `BipedalWalker::step` returns `Err(InvalidAction)` if any component is
/// outside `[-1, 1]` or is non-finite.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BipedalWalkerAction(pub [f32; 4]);

impl BipedalWalkerAction {
    /// Valid range for all motor targets.
    pub const BOUND: f32 = 1.0;

    /// Returns `true` if all components are finite and in `[-1, 1]`.
    fn all_valid(v: &[f32; 4]) -> bool {
        v.iter().all(|x| x.is_finite() && x.abs() <= Self::BOUND)
    }
}

impl Action<1> for BipedalWalkerAction {
    fn shape() -> [usize; 1] {
        [4]
    }

    fn is_valid(&self) -> bool {
        Self::all_valid(&self.0)
    }
}

impl ContinuousAction<1> for BipedalWalkerAction {
    const COMPONENTS: usize = 4;

    /// Returns the four motor targets as a contiguous `f32` slice.
    fn as_slice(&self) -> &[f32] {
        &self.0
    }

    /// Returns a new action with every component clamped to `[min, max]`.
    fn clip(&self, min: f32, max: f32) -> Self {
        Self(self.0.map(|v| v.clamp(min, max)))
    }

    /// Constructs an action from exactly `COMPONENTS` motor targets.
    ///
    /// # Panics
    ///
    /// Panics if `values.len() != Self::COMPONENTS`.
    fn from_slice(values: &[f32]) -> Self {
        assert_eq!(
            values.len(),
            Self::COMPONENTS,
            "BipedalWalkerAction needs exactly {} components, got {}",
            Self::COMPONENTS,
            values.len(),
        );
        let mut arr = [0.0f32; 4];
        arr.copy_from_slice(values);
        Self(arr)
    }
}

/// Per-component bounds `[-1; 4] .. [1; 4]`.
///
/// Source: Gymnasium `BipedalWalker-v3` declares
/// `Box(-1.0, 1.0, (4,), float32)`, and this crate's
/// [`all_valid`](BipedalWalkerAction::all_valid) rejects any component with
/// `abs() > BOUND` where `BOUND == 1.0`. Spec and in-repo dynamics agree.
///
/// The bound is on the **pre-gear** motor target: `apply_motors` scales each
/// component by `speed_hip` / `speed_knee` afterwards, so the torque limits
/// live in the environment config, not here.
impl BoundedAction<1> for BipedalWalkerAction {
    fn low() -> &'static [f32] {
        &[-1.0, -1.0, -1.0, -1.0]
    }

    fn high() -> &'static [f32] {
        &[1.0, 1.0, 1.0, 1.0]
    }
}

#[cfg(test)]
mod tests {
    // Clipping and slice round-trips move values without arithmetic, so the
    // asserted results are bit-exact by construction; a tolerance would let a
    // genuinely wrong clip pass.
    #![allow(clippy::float_cmp)]

    use super::*;

    #[test]
    fn test_shape() {
        assert_eq!(BipedalWalkerAction::shape(), [4]);
    }

    #[test]
    fn test_bounds_match_components_and_is_valid() {
        assert_eq!(
            BipedalWalkerAction::low().len(),
            BipedalWalkerAction::COMPONENTS
        );
        assert_eq!(
            BipedalWalkerAction::high().len(),
            BipedalWalkerAction::COMPONENTS
        );
        for i in 0..BipedalWalkerAction::COMPONENTS {
            assert!(BipedalWalkerAction::low()[i] < BipedalWalkerAction::high()[i]);
            assert!(
                (BipedalWalkerAction::low()[i] + BipedalWalkerAction::BOUND).abs() < f32::EPSILON
            );
            assert!(
                (BipedalWalkerAction::high()[i] - BipedalWalkerAction::BOUND).abs() < f32::EPSILON
            );
        }
        assert!(BipedalWalkerAction::from_slice(BipedalWalkerAction::low()).is_valid());
        assert!(BipedalWalkerAction::from_slice(BipedalWalkerAction::high()).is_valid());
    }

    #[test]
    fn test_valid_action() {
        let a = BipedalWalkerAction([0.5, -0.3, 1.0, -1.0]);
        assert!(a.is_valid());
    }

    #[test]
    fn test_invalid_out_of_range() {
        let a = BipedalWalkerAction([2.0, 0.0, 0.0, 0.0]);
        assert!(!a.is_valid());
    }

    #[test]
    fn test_invalid_nan() {
        let a = BipedalWalkerAction([f32::NAN, 0.0, 0.0, 0.0]);
        assert!(!a.is_valid());
    }

    #[test]
    fn test_clip() {
        let a = BipedalWalkerAction([2.0, -3.0, 0.5, 0.0]);
        let clipped = a.clip(-1.0, 1.0);
        assert_eq!(clipped.0, [1.0, -1.0, 0.5, 0.0]);
    }

    #[test]
    fn test_from_slice() {
        let a = BipedalWalkerAction::from_slice(&[0.1, 0.2, 0.3, 0.4]);
        assert_eq!(a.0, [0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    #[should_panic(expected = "needs exactly 4 components, got 5")]
    fn test_from_slice_rejects_an_over_long_slice() {
        // `ContinuousAction::from_slice` accepts *exactly* `COMPONENTS` values
        // (docs/rules.md §3); the previous `>= 4` check silently truncated.
        let _ = BipedalWalkerAction::from_slice(&[0.1, 0.2, 0.3, 0.4, 0.5]);
    }

    #[test]
    #[should_panic(expected = "needs exactly 4 components, got 3")]
    fn test_from_slice_rejects_a_short_slice() {
        let _ = BipedalWalkerAction::from_slice(&[0.1, 0.2, 0.3]);
    }

    #[test]
    fn test_random_is_valid() {
        // Regression for #100: the corrected default `random()` samples
        // `COMPONENTS` values (4), so `from_slice` no longer panics; symmetric
        // `[-1, 1)` sampling keeps every draw valid.
        for _ in 0..100 {
            let a = BipedalWalkerAction::random();
            assert!(a.is_valid(), "random action must be valid: {a:?}");
        }
    }

    #[test]
    fn test_components_matches_as_slice() {
        let a = BipedalWalkerAction([0.0; 4]);
        assert_eq!(BipedalWalkerAction::COMPONENTS, a.as_slice().len());
    }
}
