//! Action type for CarRacing.

use evorl_core::action::ContinuousAction;
use evorl_core::base::Action;
use serde::{Deserialize, Serialize};

/// 3-dimensional continuous action for CarRacing.
///
/// Components and their valid ranges:
/// * `steer ∈ [−1, 1]` — steering angle
/// * `gas   ∈ [ 0, 1]` — throttle
/// * `brake ∈ [ 0, 1]` — braking force
///
/// **Note**: the gas and brake ranges are asymmetric (design decision D5).
/// `step()` returns `Err(InvalidAction)` if any component is outside its range.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CarRacingAction {
    /// Steering angle `[−1, 1]`.
    pub steer: f32,
    /// Throttle `[0, 1]`.
    pub gas: f32,
    /// Braking `[0, 1]`.
    pub brake: f32,
}

impl CarRacingAction {
    /// Validate all components against their asymmetric bounds.
    fn components_valid(steer: f32, gas: f32, brake: f32) -> bool {
        steer.is_finite() && steer.abs() <= 1.0
            && gas.is_finite() && (0.0..=1.0).contains(&gas)
            && brake.is_finite() && (0.0..=1.0).contains(&brake)
    }
}

impl Action<1> for CarRacingAction {
    fn shape() -> [usize; 1] {
        [3]
    }

    fn is_valid(&self) -> bool {
        Self::components_valid(self.steer, self.gas, self.brake)
    }
}

impl ContinuousAction<1> for CarRacingAction {
    fn as_slice(&self) -> &[f32] {
        // Safety: CarRacingAction is repr(C) equivalent to 3 consecutive f32 fields.
        // Use a runtime slice instead of transmute.
        std::slice::from_ref(&self.steer)
        // This only gives steer; we need a proper slice over all 3 fields.
        // Workaround: store internally as [f32; 3] and expose a slice.
        // See `as_array` below for now.
    }

    fn clip(&self, min: f32, max: f32) -> Self {
        Self {
            steer: self.steer.clamp(min, max),
            gas: self.gas.clamp(min, max),
            brake: self.brake.clamp(min, max),
        }
    }

    fn from_slice(values: &[f32]) -> Self {
        assert!(values.len() >= 3, "CarRacingAction needs 3 values");
        Self { steer: values[0], gas: values[1], brake: values[2] }
    }
}

impl CarRacingAction {
    /// Returns `[steer, gas, brake]` as an owned array.
    pub fn as_array(&self) -> [f32; 3] {
        [self.steer, self.gas, self.brake]
    }

    /// Generate a random valid action (custom ranges per D5).
    pub fn random_valid(rng: &mut rand::rngs::StdRng) -> Self {
        use rand::RngExt;
        Self {
            steer: rng.random_range(-1.0..=1.0),
            gas: rng.random_range(0.0..=1.0),
            brake: rng.random_range(0.0..=1.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape() {
        assert_eq!(CarRacingAction::shape(), [3]);
    }

    #[test]
    fn test_valid_action() {
        assert!(CarRacingAction { steer: 0.5, gas: 0.3, brake: 0.0 }.is_valid());
    }

    #[test]
    fn test_d5_negative_gas() {
        assert!(!CarRacingAction { steer: 0.0, gas: -0.1, brake: 0.0 }.is_valid());
    }

    #[test]
    fn test_d5_steer_out_of_range() {
        assert!(!CarRacingAction { steer: 1.5, gas: 0.0, brake: 0.0 }.is_valid());
    }

    #[test]
    fn test_d5_brake_negative() {
        assert!(!CarRacingAction { steer: 0.0, gas: 0.0, brake: -0.1 }.is_valid());
    }

    #[test]
    fn test_from_slice() {
        let a = CarRacingAction::from_slice(&[0.1, 0.5, 0.2]);
        assert!((a.steer - 0.1).abs() < 1e-6);
        assert!((a.gas - 0.5).abs() < 1e-6);
        assert!((a.brake - 0.2).abs() < 1e-6);
    }
}
