//! Observation type for [`super::InvertedDoublePendulum`].

use burn::prelude::{Backend, Tensor};
use rlevo_core::base::{Observation, TensorConversionError, TensorConvertible};
use serde::{Deserialize, Serialize};

/// 9-dim observation: `[cart_x, sin θ₁, sin θ₂, cos θ₁, cos θ₂, cart_vx,
/// θ̇₁, θ̇₂, constraint_force_x]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct InvertedDoublePendulumObservation(pub [f32; 9]);

impl InvertedDoublePendulumObservation {
    #[must_use]
    pub const fn cart_position(&self) -> f32 {
        self.0[0]
    }
    #[must_use]
    pub const fn sin_theta1(&self) -> f32 {
        self.0[1]
    }
    #[must_use]
    pub const fn sin_theta2(&self) -> f32 {
        self.0[2]
    }
    #[must_use]
    pub const fn cos_theta1(&self) -> f32 {
        self.0[3]
    }
    #[must_use]
    pub const fn cos_theta2(&self) -> f32 {
        self.0[4]
    }
    #[must_use]
    pub const fn cart_velocity(&self) -> f32 {
        self.0[5]
    }
    #[must_use]
    pub const fn theta1_dot(&self) -> f32 {
        self.0[6]
    }
    #[must_use]
    pub const fn theta2_dot(&self) -> f32 {
        self.0[7]
    }
    #[must_use]
    pub const fn constraint_force_x(&self) -> f32 {
        self.0[8]
    }

    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.0.iter().all(|v| v.is_finite())
    }
}

impl Default for InvertedDoublePendulumObservation {
    fn default() -> Self {
        Self([0.0; 9])
    }
}

impl Observation<1> for InvertedDoublePendulumObservation {
    fn shape() -> [usize; 1] {
        [9]
    }
}

impl<B: Backend> TensorConvertible<1, B> for InvertedDoublePendulumObservation {
    fn to_tensor(&self, device: &B::Device) -> Tensor<B, 1> {
        Tensor::from_floats(self.0, device)
    }

    fn from_tensor(tensor: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let data = tensor.into_data();
        let slice = data.as_slice::<f32>().map_err(|e| TensorConversionError {
            message: format!("expected f32 observation tensor: {e:?}"),
        })?;
        if slice.len() != 9 {
            return Err(TensorConversionError {
                message: format!("expected 9 observation elements, got {}", slice.len()),
            });
        }
        let mut out = [0.0f32; 9];
        out.copy_from_slice(slice);
        Ok(Self(out))
    }
}
