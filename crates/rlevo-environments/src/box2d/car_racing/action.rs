//! Action type for CarRacing.

use rlevo_core::action::ContinuousAction;
use rlevo_core::base::Action;
use serde::{Deserialize, Serialize};

/// 3-dimensional continuous action for CarRacing.
///
/// Components and their valid ranges:
/// * `steer ∈ [−1, 1]` — steering angle
/// * `gas   ∈ [ 0, 1]` — throttle
/// * `brake ∈ [ 0, 1]` — braking force
///
/// The three components are stored contiguously as `[steer, gas, brake]` so
/// [`as_slice`](ContinuousAction::as_slice) can expose all of them at once.
///
/// **Note**: the gas and brake ranges are asymmetric (design decision D5).
/// `step()` returns `Err(InvalidAction)` if any component is outside its range.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CarRacingAction {
    /// Contiguous `[steer, gas, brake]` backing store.
    components: [f32; 3],
}

impl CarRacingAction {
    /// Constructs an action from its three components `[steer, gas, brake]`.
    ///
    /// No validation is performed here; use [`Action::is_valid`] (or let
    /// `step()` reject the action) to check the asymmetric bounds.
    #[must_use]
    pub fn new(steer: f32, gas: f32, brake: f32) -> Self {
        Self {
            components: [steer, gas, brake],
        }
    }

    /// Returns the steering component `[−1, 1]`.
    #[must_use]
    pub fn steer(&self) -> f32 {
        self.components[0]
    }

    /// Returns the throttle component `[0, 1]`.
    #[must_use]
    pub fn gas(&self) -> f32 {
        self.components[1]
    }

    /// Returns the braking component `[0, 1]`.
    #[must_use]
    pub fn brake(&self) -> f32 {
        self.components[2]
    }

    /// Returns `[steer, gas, brake]` as an owned array.
    #[must_use]
    pub fn as_array(&self) -> [f32; 3] {
        self.components
    }

    /// Validate all components against their asymmetric bounds.
    fn components_valid(components: &[f32; 3]) -> bool {
        let [steer, gas, brake] = *components;
        steer.is_finite()
            && steer.abs() <= 1.0
            && gas.is_finite()
            && (0.0..=1.0).contains(&gas)
            && brake.is_finite()
            && (0.0..=1.0).contains(&brake)
    }

    /// Generate a random valid action sampled uniformly within the asymmetric
    /// bounds: `steer ∈ [−1, 1]`, `gas ∈ [0, 1]`, `brake ∈ [0, 1]`.
    ///
    /// Generic over the RNG so it can back both the deterministic seeded path
    /// and the [`ContinuousAction::random`] override (ADR 0038).
    pub fn random_valid<R: rand::Rng + ?Sized>(rng: &mut R) -> Self {
        use rand::RngExt;
        Self::new(
            rng.random_range(-1.0..=1.0),
            rng.random_range(0.0..=1.0),
            rng.random_range(0.0..=1.0),
        )
    }
}

impl Action<1> for CarRacingAction {
    fn shape() -> [usize; 1] {
        [3]
    }

    fn is_valid(&self) -> bool {
        Self::components_valid(&self.components)
    }
}

impl ContinuousAction<1> for CarRacingAction {
    const COMPONENTS: usize = 3;

    /// Returns all three components as a contiguous `[steer, gas, brake]` slice.
    fn as_slice(&self) -> &[f32] {
        &self.components
    }

    /// Clamps all three components to `[min, max]`.
    ///
    /// Note that clamping `gas` or `brake` with a negative `min` will produce a
    /// value that fails `is_valid`. Prefer [`CarRacingAction::as_array`] and manual
    /// per-component clamping when asymmetric bounds matter.
    fn clip(&self, min: f32, max: f32) -> Self {
        Self {
            components: self.components.map(|v| v.clamp(min, max)),
        }
    }

    /// Construct from a slice of at least 3 values `[steer, gas, brake]`.
    ///
    /// # Panics
    ///
    /// Panics if `values.len() < 3`.
    fn from_slice(values: &[f32]) -> Self {
        assert!(values.len() >= 3, "CarRacingAction needs 3 values");
        Self::new(values[0], values[1], values[2])
    }

    /// Samples a uniformly-random **valid** action within CarRacing's asymmetric
    /// bounds. Overrides the trait default, whose symmetric `[-1, 1)` range would
    /// sample negative gas/brake and fail [`Action::is_valid`] (ADR 0038).
    fn random() -> Self {
        Self::random_valid(&mut rand::rng())
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
        assert!(CarRacingAction::new(0.5, 0.3, 0.0).is_valid());
    }

    #[test]
    fn test_d5_negative_gas() {
        assert!(!CarRacingAction::new(0.0, -0.1, 0.0).is_valid());
    }

    #[test]
    fn test_d5_steer_out_of_range() {
        assert!(!CarRacingAction::new(1.5, 0.0, 0.0).is_valid());
    }

    #[test]
    fn test_d5_brake_negative() {
        assert!(!CarRacingAction::new(0.0, 0.0, -0.1).is_valid());
    }

    #[test]
    fn test_from_slice() {
        let a = CarRacingAction::from_slice(&[0.1, 0.5, 0.2]);
        assert!((a.steer() - 0.1).abs() < 1e-6);
        assert!((a.gas() - 0.5).abs() < 1e-6);
        assert!((a.brake() - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_as_slice_exposes_all_three_components() {
        // Regression for #100: `as_slice` previously returned only `steer`
        // (a length-1 slice), silently dropping gas and brake.
        let a = CarRacingAction::new(-0.5, 0.5, 0.25);
        assert_eq!(a.as_slice(), &[-0.5, 0.5, 0.25]);
    }

    #[test]
    fn test_random_is_valid() {
        // The overridden `random()` respects the asymmetric bounds, so every
        // draw must satisfy `is_valid` (the trait default's `[-1, 1)` range
        // would sample negative gas/brake and fail here).
        for _ in 0..100 {
            let a = CarRacingAction::random();
            assert!(a.is_valid(), "random action must be valid: {a:?}");
        }
    }

    #[test]
    fn test_components_matches_as_slice() {
        let a = CarRacingAction::new(0.0, 0.0, 0.0);
        assert_eq!(CarRacingAction::COMPONENTS, a.as_slice().len());
    }
}
