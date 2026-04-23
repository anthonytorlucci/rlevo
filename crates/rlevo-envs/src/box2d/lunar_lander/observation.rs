//! Observation type for LunarLander.

use rlevo_core::base::Observation;
use serde::{Deserialize, Serialize};

/// 8-dimensional observation for LunarLander.
///
/// Layout:
/// * `[0]` x position (normalised)
/// * `[1]` y position (normalised)
/// * `[2]` x velocity
/// * `[3]` y velocity
/// * `[4]` angle (rad)
/// * `[5]` angular velocity (rad/s)
/// * `[6]` leg 1 contact (0 or 1)
/// * `[7]` leg 2 contact (0 or 1)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LunarLanderObservation {
    /// Raw 8-float observation vector.
    pub values: [f32; 8],
}

impl LunarLanderObservation {
    /// Construct from a raw array.
    pub fn new(values: [f32; 8]) -> Self {
        Self { values }
    }

    /// Accessors for named fields.
    pub fn x(&self) -> f32 {
        self.values[0]
    }
    pub fn y(&self) -> f32 {
        self.values[1]
    }
    pub fn vx(&self) -> f32 {
        self.values[2]
    }
    pub fn vy(&self) -> f32 {
        self.values[3]
    }
    pub fn angle(&self) -> f32 {
        self.values[4]
    }
    pub fn angular_vel(&self) -> f32 {
        self.values[5]
    }
    pub fn leg1_contact(&self) -> f32 {
        self.values[6]
    }
    pub fn leg2_contact(&self) -> f32 {
        self.values[7]
    }

    /// Returns `true` if all values are finite.
    pub fn is_finite(&self) -> bool {
        self.values.iter().all(|v| v.is_finite())
    }
}

impl Default for LunarLanderObservation {
    fn default() -> Self {
        Self { values: [0.0; 8] }
    }
}

impl Observation<1> for LunarLanderObservation {
    fn shape() -> [usize; 1] {
        [8]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape() {
        assert_eq!(LunarLanderObservation::shape(), [8]);
    }

    #[test]
    fn test_default_is_finite() {
        assert!(LunarLanderObservation::default().is_finite());
    }
}
