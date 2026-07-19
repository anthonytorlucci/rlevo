//! Continuous action type for LunarLander (design decision D1).

use rlevo_core::action::{BoundedAction, ContinuousAction};
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
    const COMPONENTS: usize = 2;

    fn as_slice(&self) -> &[f32] {
        &self.0
    }

    fn clip(&self, min: f32, max: f32) -> Self {
        Self(self.0.map(|v| v.clamp(min, max)))
    }

    /// Construct from a slice, taking the first two elements.
    ///
    /// # Panics
    ///
    /// Panics if `values.len() < 2`.
    fn from_slice(values: &[f32]) -> Self {
        assert!(
            values.len() >= 2,
            "LunarLanderContinuousAction needs 2 values"
        );
        Self([values[0], values[1]])
    }
}

/// Per-component bounds `[-1, -1] .. [1, 1]`.
///
/// Source: Gymnasium `LunarLander-v3` with `continuous=True` declares
/// `Box(-1, +1, (2,), float32)`, and this crate's
/// [`is_valid`](Action::is_valid) rejects any component with
/// `abs() > BOUND` where `BOUND == 1.0`. Spec and in-repo dynamics agree.
///
/// The main engine's `−1..0` dead band is a property of how the environment
/// *interprets* component 0, not of the action space: `-1` is a legal
/// (engine-off) command, so the lower bound is `-1`, not `0`. This is why the
/// space is symmetric despite one component behaving one-sidedly — unlike
/// CarRacing, whose gas and brake are genuinely floored at `0`.
impl BoundedAction<1> for LunarLanderContinuousAction {
    fn low() -> &'static [f32] {
        &[-1.0, -1.0]
    }

    fn high() -> &'static [f32] {
        &[1.0, 1.0]
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
    fn test_bounds_match_components_and_is_valid() {
        let low = LunarLanderContinuousAction::low();
        let high = LunarLanderContinuousAction::high();
        assert_eq!(low.len(), LunarLanderContinuousAction::COMPONENTS);
        assert_eq!(high.len(), LunarLanderContinuousAction::COMPONENTS);
        for i in 0..LunarLanderContinuousAction::COMPONENTS {
            assert!(low[i] < high[i]);
        }
        // `-1` on the main engine is "off", a legal command — the lower bound
        // is not 0.
        assert!(LunarLanderContinuousAction::from_slice(low).is_valid());
        assert!(LunarLanderContinuousAction::from_slice(high).is_valid());
        assert!(!LunarLanderContinuousAction([-1.1, 0.0]).is_valid());
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

    #[test]
    fn test_random_is_valid() {
        // Regression for #100: the corrected default `random()` samples
        // `COMPONENTS` values (2) rather than `RANK` (1), so `from_slice` no
        // longer panics; symmetric `[-1, 1)` sampling keeps every draw valid.
        for _ in 0..100 {
            let a = LunarLanderContinuousAction::random();
            assert!(a.is_valid(), "random action must be valid: {a:?}");
        }
    }

    #[test]
    fn test_components_matches_as_slice() {
        let a = LunarLanderContinuousAction([0.0; 2]);
        assert_eq!(LunarLanderContinuousAction::COMPONENTS, a.as_slice().len());
    }
}
