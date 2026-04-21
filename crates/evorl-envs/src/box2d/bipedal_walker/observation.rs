//! Observation type for BipedalWalker.

use evorl_core::base::Observation;
use serde::{Deserialize, Serialize};

/// 24-dimensional observation for BipedalWalker.
///
/// Layout:
/// * `[0]`  hull angle (rad)
/// * `[1]`  hull angular velocity (rad/s)
/// * `[2]`  horizontal velocity (clipped to `[-1, 1]`)
/// * `[3]`  vertical velocity (clipped to `[-1, 1]`)
/// * `[4]`  hip1 joint angle
/// * `[5]`  hip1 joint speed
/// * `[6]`  knee1 joint angle
/// * `[7]`  knee1 joint speed
/// * `[8]`  leg1 ground contact (0 or 1)
/// * `[9]`  hip2 joint angle
/// * `[10]` hip2 joint speed
/// * `[11]` knee2 joint angle
/// * `[12]` knee2 joint speed
/// * `[13]` leg2 ground contact (0 or 1)
/// * `[14..23]` 10 lidar ray distances (normalised to `[0, 1]`)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BipedalWalkerObservation {
    /// Raw 24-float observation vector.
    pub values: [f32; 24],
}

impl BipedalWalkerObservation {
    /// Construct from a raw array.
    pub fn new(values: [f32; 24]) -> Self {
        Self { values }
    }

    /// Returns true if all values are finite.
    pub fn is_finite(&self) -> bool {
        self.values.iter().all(|v| v.is_finite())
    }
}

impl Default for BipedalWalkerObservation {
    fn default() -> Self {
        Self { values: [0.0; 24] }
    }
}

impl Observation<1> for BipedalWalkerObservation {
    fn shape() -> [usize; 1] {
        [24]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape() {
        assert_eq!(BipedalWalkerObservation::shape(), [24]);
    }

    #[test]
    fn test_default_is_finite() {
        assert!(BipedalWalkerObservation::default().is_finite());
    }
}
