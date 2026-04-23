//! Continuous action type for LunarLander (design decision D1).

use rlevo_core::action::ContinuousAction;
use rlevo_core::base::Action;
use serde::{Deserialize, Serialize};

/// 2-dimensional continuous action for [`super::LunarLanderContinuous`] (D1).
///
/// Components (both in `[-1, 1]`):
/// * `[0]` main engine throttle: `−1..0` = off, `0..1` = firing
/// * `[1]` lateral engine: negative = left, positive = right
///
/// Design decision D5: `step()` returns `Err(InvalidAction)` if any component
/// is outside `[-1, 1]` or non-finite.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LunarLanderContinuousAction(pub [f32; 2]);

impl LunarLanderContinuousAction {
    /// Valid symmetric bound for all components.
    pub const BOUND: f32 = 1.0;
}

impl Action<1> for LunarLanderContinuousAction {
    fn shape() -> [usize; 1] {
        [2]
    }

    fn is_valid(&self) -> bool {
        self.0
            .iter()
            .all(|x| x.is_finite() && x.abs() <= Self::BOUND)
    }
}

impl ContinuousAction<1> for LunarLanderContinuousAction {
    fn as_slice(&self) -> &[f32] {
        &self.0
    }

    fn clip(&self, min: f32, max: f32) -> Self {
        Self(self.0.map(|v| v.clamp(min, max)))
    }

    fn from_slice(values: &[f32]) -> Self {
        assert!(
            values.len() >= 2,
            "LunarLanderContinuousAction needs 2 values"
        );
        Self([values[0], values[1]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape() {
        assert_eq!(LunarLanderContinuousAction::shape(), [2]);
    }

    #[test]
    fn test_valid_range() {
        assert!(LunarLanderContinuousAction([0.5, -0.3]).is_valid());
        assert!(LunarLanderContinuousAction([-1.0, 1.0]).is_valid());
    }

    #[test]
    fn test_invalid_out_of_range() {
        assert!(!LunarLanderContinuousAction([1.5, 0.0]).is_valid());
        assert!(!LunarLanderContinuousAction([0.0, -1.1]).is_valid());
    }

    #[test]
    fn test_clip() {
        let a = LunarLanderContinuousAction([2.0, -2.0]);
        let c = a.clip(-1.0, 1.0);
        assert_eq!(c.0, [1.0, -1.0]);
    }
}
