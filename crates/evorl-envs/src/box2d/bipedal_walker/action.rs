//! Action type for BipedalWalker.

use evorl_core::action::ContinuousAction;
use evorl_core::base::Action;
use serde::{Deserialize, Serialize};

/// 4-dimensional continuous action for BipedalWalker.
///
/// Components (all in `[-1, 1]`):
/// * `[0]` hip1 motor target (positive = forward)
/// * `[1]` knee1 motor target (positive = extend)
/// * `[2]` hip2 motor target
/// * `[3]` knee2 motor target
///
/// Design decision D5: step() returns `Err(InvalidAction)` if any component
/// is outside `[-1, 1]` or is non-finite.
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
    fn as_slice(&self) -> &[f32] {
        &self.0
    }

    fn clip(&self, min: f32, max: f32) -> Self {
        Self(self.0.map(|v| v.clamp(min, max)))
    }

    fn from_slice(values: &[f32]) -> Self {
        assert!(values.len() >= 4, "BipedalWalkerAction needs 4 values");
        let mut arr = [0.0f32; 4];
        arr.copy_from_slice(&values[..4]);
        Self(arr)
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape() {
        assert_eq!(BipedalWalkerAction::shape(), [4]);
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
}
