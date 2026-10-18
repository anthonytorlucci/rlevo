//! Observation type for `LunarLander`.

use rlevo_core::base::{HostRow, Observation};
use serde::{Deserialize, Serialize};

/// 8-dimensional observation for `LunarLander`.
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
    #[must_use]
    pub fn new(values: [f32; 8]) -> Self {
        Self { values }
    }

    /// Returns the normalised x position relative to the helipad centre.
    #[must_use]
    pub fn x(&self) -> f32 {
        self.values[0]
    }

    /// Returns the normalised y position relative to the helipad height.
    #[must_use]
    pub fn y(&self) -> f32 {
        self.values[1]
    }

    /// Returns the normalised x velocity.
    #[must_use]
    pub fn vx(&self) -> f32 {
        self.values[2]
    }

    /// Returns the normalised y velocity.
    #[must_use]
    pub fn vy(&self) -> f32 {
        self.values[3]
    }

    /// Returns the hull rotation angle in radians.
    #[must_use]
    pub fn angle(&self) -> f32 {
        self.values[4]
    }

    /// Returns the hull angular velocity in rad/s (scaled).
    #[must_use]
    pub fn angular_vel(&self) -> f32 {
        self.values[5]
    }

    /// Returns 1.0 if the left leg is in ground contact, 0.0 otherwise.
    #[must_use]
    pub fn leg1_contact(&self) -> f32 {
        self.values[6]
    }

    /// Returns 1.0 if the right leg is in ground contact, 0.0 otherwise.
    #[must_use]
    pub fn leg2_contact(&self) -> f32 {
        self.values[7]
    }

    /// Returns `true` if all values are finite.
    #[must_use]
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

impl HostRow<1> for LunarLanderObservation {
    fn write_host_row(&self, buf: &mut Vec<f32>) {
        buf.extend_from_slice(&self.values);
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

    /// The row is the 8 stored values in order, leg-contact flags at 6 and 7
    /// included.
    #[test]
    fn host_row_is_the_stored_values() {
        let values = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 1.0, 0.0];
        let obs = LunarLanderObservation::new(values);
        let mut row = Vec::new();
        obs.write_host_row(&mut row);

        assert_eq!(
            row.len(),
            LunarLanderObservation::shape().iter().product::<usize>()
        );
        assert_eq!(row, values);
    }
}
