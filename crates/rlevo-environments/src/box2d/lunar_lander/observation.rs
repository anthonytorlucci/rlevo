//! Observation type for LunarLander.

use rlevo_core::base::{Observation, TensorConversionError, TensorConvertible};
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

    /// Returns the normalised x position relative to the helipad centre.
    pub fn x(&self) -> f32 {
        self.values[0]
    }

    /// Returns the normalised y position relative to the helipad height.
    pub fn y(&self) -> f32 {
        self.values[1]
    }

    /// Returns the normalised x velocity.
    pub fn vx(&self) -> f32 {
        self.values[2]
    }

    /// Returns the normalised y velocity.
    pub fn vy(&self) -> f32 {
        self.values[3]
    }

    /// Returns the hull rotation angle in radians.
    pub fn angle(&self) -> f32 {
        self.values[4]
    }

    /// Returns the hull angular velocity in rad/s (scaled).
    pub fn angular_vel(&self) -> f32 {
        self.values[5]
    }

    /// Returns 1.0 if the left leg is in ground contact, 0.0 otherwise.
    pub fn leg1_contact(&self) -> f32 {
        self.values[6]
    }

    /// Returns 1.0 if the right leg is in ground contact, 0.0 otherwise.
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

impl<B: burn::tensor::backend::Backend> TensorConvertible<1, B> for LunarLanderObservation {
    fn row_shape() -> [usize; 1] {
        [8]
    }

    fn write_host_row(&self, buf: &mut Vec<f32>) {
        buf.extend_from_slice(&self.values);
    }

    fn from_tensor(tensor: burn::tensor::Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let dims = tensor.dims();
        if dims.as_slice() != [8] {
            return Err(TensorConversionError {
                message: format!("expected shape [8], got {dims:?}"),
            });
        }
        let v = tensor
            .into_data()
            .into_vec::<f32>()
            .map_err(|e| TensorConversionError {
                message: e.to_string(),
            })?;
        let mut values = [0.0_f32; 8];
        values.copy_from_slice(&v);
        Ok(Self { values })
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

    #[test]
    fn round_trips_through_tensor() {
        use burn::backend::Flex;
        type TestBackend = Flex;
        let device = Default::default();

        let obs = LunarLanderObservation::new([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 1.0, 0.0]);
        let tensor =
            <LunarLanderObservation as TensorConvertible<1, TestBackend>>::to_tensor(&obs, &device);
        let round_tripped =
            <LunarLanderObservation as TensorConvertible<1, TestBackend>>::from_tensor(tensor)
                .unwrap();

        assert_eq!(round_tripped, obs);
    }

    #[test]
    fn from_tensor_rejects_wrong_shape() {
        use burn::backend::Flex;
        type TestBackend = Flex;
        let device = Default::default();

        let tensor = burn::tensor::Tensor::<TestBackend, 1>::from_floats([0.0, 1.0, 2.0], &device);
        let err =
            <LunarLanderObservation as TensorConvertible<1, TestBackend>>::from_tensor(tensor)
                .unwrap_err();
        assert!(err.message.contains("expected shape [8]"));
    }
}
