//! Action type for CarRacing.

use rlevo_core::action::{BoundedAction, ContinuousAction};
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

    /// Construct from a slice of exactly `COMPONENTS` values
    /// `[steer, gas, brake]`.
    ///
    /// # Panics
    ///
    /// Panics if `values.len() != Self::COMPONENTS`.
    fn from_slice(values: &[f32]) -> Self {
        assert_eq!(
            values.len(),
            Self::COMPONENTS,
            "CarRacingAction needs exactly {} components, got {}",
            Self::COMPONENTS,
            values.len(),
        );
        Self::new(values[0], values[1], values[2])
    }

    /// Samples a uniformly-random **valid** action within CarRacing's asymmetric
    /// bounds. Overrides the trait default, whose symmetric `[-1, 1)` range would
    /// sample negative gas/brake and fail [`Action::is_valid`] (ADR 0038).
    fn random() -> Self {
        Self::random_valid(&mut rand::rng())
    }
}

/// Per-component bounds `[-1, 0, 0] .. [1, 1, 1]`.
///
/// Source: Gymnasium `CarRacing-v3` declares
/// `Box([-1, 0, 0], [1, 1, 1], (3,), float32)`, and this crate's own
/// [`components_valid`](CarRacingAction::components_valid) enforces exactly the
/// same ranges (`steer.abs() <= 1`, `(0.0..=1.0).contains(gas)`,
/// `(0.0..=1.0).contains(brake)`), as does
/// [`random_valid`](CarRacingAction::random_valid). Spec and in-repo dynamics
/// agree; nothing here is inferred.
///
/// This is the **only** action in the workspace whose components disagree on
/// their bounds, which makes it the regression witness for the per-component
/// target clip in DDPG/TD3 (ADR 0053 §6/§7): under a scalar `low[0]`/`high[0]`
/// collapse, gas and brake would inherit steering's `-1` floor.
impl BoundedAction<1> for CarRacingAction {
    fn low() -> &'static [f32] {
        &[-1.0, 0.0, 0.0]
    }

    fn high() -> &'static [f32] {
        &[1.0, 1.0, 1.0]
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
    fn test_bounds_are_per_component_and_asymmetric() {
        assert_eq!(CarRacingAction::low(), &[-1.0, 0.0, 0.0]);
        assert_eq!(CarRacingAction::high(), &[1.0, 1.0, 1.0]);
        assert_eq!(CarRacingAction::low().len(), CarRacingAction::COMPONENTS);
        assert_eq!(CarRacingAction::high().len(), CarRacingAction::COMPONENTS);
        for i in 0..CarRacingAction::COMPONENTS {
            assert!(CarRacingAction::low()[i] < CarRacingAction::high()[i]);
        }
        // The asymmetry is load-bearing, not incidental: it is what lets this
        // type witness a scalar `low[0]` collapse.
        assert!(
            CarRacingAction::low()[1] > CarRacingAction::low()[0],
            "gas must not inherit steering's lower bound"
        );
    }

    #[test]
    fn test_bounds_agree_with_is_valid() {
        // The bounds and the validity predicate are two statements of the same
        // contract; a corner action must be valid, and one step outside any
        // bound must not be.
        let low = CarRacingAction::low();
        let high = CarRacingAction::high();
        assert!(CarRacingAction::from_slice(low).is_valid());
        assert!(CarRacingAction::from_slice(high).is_valid());
        for i in 0..CarRacingAction::COMPONENTS {
            let mut below = low.to_vec();
            below[i] -= 0.1;
            assert!(
                !CarRacingAction::from_slice(&below).is_valid(),
                "component {i} below its lower bound must be invalid"
            );
            let mut above = high.to_vec();
            above[i] += 0.1;
            assert!(
                !CarRacingAction::from_slice(&above).is_valid(),
                "component {i} above its upper bound must be invalid"
            );
        }
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
    #[should_panic(expected = "needs exactly 3 components, got 4")]
    fn test_from_slice_rejects_an_over_long_slice() {
        // `ContinuousAction::from_slice` accepts *exactly* `COMPONENTS` values
        // (docs/rules.md §3). A `>=` check would silently truncate the extra
        // value, hiding a caller that disagreed with this action's width.
        let _ = CarRacingAction::from_slice(&[0.1, 0.5, 0.2, 0.9]);
    }

    #[test]
    #[should_panic(expected = "needs exactly 3 components, got 2")]
    fn test_from_slice_rejects_a_short_slice() {
        let _ = CarRacingAction::from_slice(&[0.1, 0.5]);
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
