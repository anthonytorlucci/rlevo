//! Observation type for [`super::Reacher`].

use burn::prelude::{Backend, Tensor};
use rlevo_core::base::{Observation, TensorConversionError, TensorConvertible};
use serde::{Deserialize, Serialize};

/// 10-dim observation. Layout:
/// `[cos θ₁, cos θ₂, sin θ₁, sin θ₂, target_x, target_y, θ̇₁, θ̇₂,
///   (finger − target)_x, (finger − target)_y]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReacherObservation(pub [f32; 10]);

impl ReacherObservation {
    #[must_use]
    pub const fn theta1_cos(&self) -> f32 {
        self.0[0]
    }
    #[must_use]
    pub const fn theta2_cos(&self) -> f32 {
        self.0[1]
    }
    #[must_use]
    pub const fn theta1_sin(&self) -> f32 {
        self.0[2]
    }
    #[must_use]
    pub const fn theta2_sin(&self) -> f32 {
        self.0[3]
    }
    #[must_use]
    pub const fn target_xy(&self) -> [f32; 2] {
        [self.0[4], self.0[5]]
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
    pub const fn finger_minus_target_xy(&self) -> [f32; 2] {
        [self.0[8], self.0[9]]
    }

    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.0.iter().all(|v| v.is_finite())
    }
}

impl Default for ReacherObservation {
    fn default() -> Self {
        Self([0.0; 10])
    }
}

impl Observation<1> for ReacherObservation {
    fn shape() -> [usize; 1] {
        [10]
    }
}

impl<B: Backend> TensorConvertible<1, B> for ReacherObservation {
    fn to_tensor(&self, device: &B::Device) -> Tensor<B, 1> {
        Tensor::from_floats(self.0, device)
    }

    fn from_tensor(tensor: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let data = tensor.into_data();
        let slice = data.as_slice::<f32>().map_err(|e| TensorConversionError {
            message: format!("expected f32 observation tensor: {e:?}"),
        })?;
        if slice.len() != 10 {
            return Err(TensorConversionError {
                message: format!("expected 10 observation elements, got {}", slice.len()),
            });
        }
        let mut arr = [0.0f32; 10];
        arr.copy_from_slice(slice);
        Ok(Self(arr))
    }
}
