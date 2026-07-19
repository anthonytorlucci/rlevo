//! Observation type for the BipedalWalker environment.
//!
//! [`BipedalWalkerObservation`] is a 24-element `f32` vector produced after
//! every `reset()` and `step()`. The first 14 elements capture hull and joint
//! kinematics; elements `[14..24]` contain 10 lidar range readings swept from
//! −90° to +90° relative to the hull, normalised to `[0, 1]` by
//! `lidar_range`.

use rlevo_core::base::{HostRow, Observation, TensorConversionError, TensorConvertible};
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
/// * `[14..24]` 10 lidar ray distances (normalised to `[0, 1]`)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BipedalWalkerObservation {
    /// Raw 24-float observation vector.
    pub values: [f32; 24],
}

impl BipedalWalkerObservation {
    /// Construct an observation from a pre-filled 24-element array.
    pub fn new(values: [f32; 24]) -> Self {
        Self { values }
    }

    /// Returns `true` if every element is finite (not NaN or infinite).
    ///
    /// Used by tests and callers to assert a physics step did not diverge into
    /// NaN / infinite observations.
    pub fn is_finite(&self) -> bool {
        self.values.iter().all(|v| v.is_finite())
    }
}

impl Default for BipedalWalkerObservation {
    /// Returns a zero-filled observation, used as a neutral placeholder value.
    fn default() -> Self {
        Self { values: [0.0; 24] }
    }
}

impl Observation<1> for BipedalWalkerObservation {
    fn shape() -> [usize; 1] {
        [24]
    }
}

impl HostRow<1> for BipedalWalkerObservation {
    fn row_shape() -> [usize; 1] {
        [24]
    }

    fn write_host_row(&self, buf: &mut Vec<f32>) {
        buf.extend_from_slice(&self.values);
    }
}

impl<B: burn::tensor::backend::Backend> TensorConvertible<1, B> for BipedalWalkerObservation {
    fn from_tensor(tensor: burn::tensor::Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let dims = tensor.dims();
        if dims.as_slice() != [24] {
            return Err(TensorConversionError {
                message: format!("expected shape [24], got {dims:?}"),
            });
        }
        let v = tensor
            .into_data()
            .into_vec::<f32>()
            .map_err(|e| TensorConversionError {
                message: e.to_string(),
            })?;
        let mut values = [0.0_f32; 24];
        values.copy_from_slice(&v);
        Ok(Self { values })
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

    #[test]
    fn round_trips_through_tensor() {
        use burn::backend::Flex;
        type TestBackend = Flex;
        let device = Default::default();

        let obs = BipedalWalkerObservation::new([
            0.01, -0.02, 0.03, -0.04, 0.05, -0.06, 0.07, -0.08, 0.09, -0.10, 0.11, -0.12, 0.13,
            -0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24,
        ]);
        let tensor = <BipedalWalkerObservation as TensorConvertible<1, TestBackend>>::to_tensor(
            &obs, &device,
        );
        let round_tripped =
            <BipedalWalkerObservation as TensorConvertible<1, TestBackend>>::from_tensor(tensor)
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
            <BipedalWalkerObservation as TensorConvertible<1, TestBackend>>::from_tensor(tensor)
                .unwrap_err();
        assert!(err.message.contains("expected shape [24]"));
    }
}
