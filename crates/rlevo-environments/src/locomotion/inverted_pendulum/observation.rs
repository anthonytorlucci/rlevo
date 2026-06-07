//! Observation type for [`super::InvertedPendulum`].

use burn::prelude::{Backend, Tensor};
use rlevo_core::base::{Observation, TensorConversionError, TensorConvertible};
use serde::{Deserialize, Serialize};

/// 4-dim observation: `[cart_x, pole_angle, cart_vx, pole_angvel_y]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct InvertedPendulumObservation(pub [f32; 4]);

impl InvertedPendulumObservation {
    /// Horizontal position of the cart along the world-x axis, in metres.
    /// Observation index 0.
    #[must_use]
    pub const fn cart_position(&self) -> f32 {
        self.0[0]
    }

    /// Pole angle relative to the upright vertical, in radians, wrapped to
    /// `(-π, π]`. Positive values tilt the pole toward +x. Observation index 1.
    #[must_use]
    pub const fn pole_angle(&self) -> f32 {
        self.0[1]
    }

    /// Linear velocity of the cart along the world-x axis, in m/s.
    /// Observation index 2.
    #[must_use]
    pub const fn cart_velocity(&self) -> f32 {
        self.0[2]
    }

    /// Angular velocity of the pole about the world-y axis, in rad/s.
    /// Observation index 3.
    #[must_use]
    pub const fn pole_angular_velocity(&self) -> f32 {
        self.0[3]
    }

    /// Returns `true` if all four observation components are finite (not NaN
    /// or infinite). Used by `InvertedPendulumState::is_valid`
    /// as a validity guard.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.0.iter().all(|v| v.is_finite())
    }
}

impl Default for InvertedPendulumObservation {
    fn default() -> Self {
        Self([0.0; 4])
    }
}

impl Observation<1> for InvertedPendulumObservation {
    fn shape() -> [usize; 1] {
        [4]
    }
}

impl<B: Backend> TensorConvertible<1, B> for InvertedPendulumObservation {
    fn to_tensor(&self, device: &<B as burn::tensor::backend::BackendTypes>::Device) -> Tensor<B, 1> {
        Tensor::from_floats(self.0, device)
    }

    fn from_tensor(tensor: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let data = tensor.into_data();
        let slice = data.as_slice::<f32>().map_err(|e| TensorConversionError {
            message: format!("expected f32 observation tensor: {e:?}"),
        })?;
        if slice.len() != 4 {
            return Err(TensorConversionError {
                message: format!("expected 4 observation elements, got {}", slice.len()),
            });
        }
        Ok(Self([slice[0], slice[1], slice[2], slice[3]]))
    }
}
