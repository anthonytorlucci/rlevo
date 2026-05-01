//! Action type for [`super::Swimmer`].

use burn::prelude::{Backend, Tensor};
use rlevo_core::action::ContinuousAction;
use rlevo_core::base::{Action, TensorConversionError, TensorConvertible};
use serde::{Deserialize, Serialize};

/// 2D continuous action — `[joint1, joint2]` torque targets in pre-gear
/// units. Bounds: `[-1.0, 1.0]` per element.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SwimmerAction(pub [f32; 2]);

impl SwimmerAction {
    #[must_use]
    pub const fn new(joint1: f32, joint2: f32) -> Self {
        Self([joint1, joint2])
    }
}

impl Action<1> for SwimmerAction {
    fn shape() -> [usize; 1] {
        [2]
    }

    fn is_valid(&self) -> bool {
        self.0.iter().all(|v| v.is_finite() && v.abs() <= 1.0)
    }
}

impl ContinuousAction<1> for SwimmerAction {
    fn as_slice(&self) -> &[f32] {
        &self.0
    }

    fn clip(&self, min: f32, max: f32) -> Self {
        Self([self.0[0].clamp(min, max), self.0[1].clamp(min, max)])
    }

    fn from_slice(values: &[f32]) -> Self {
        Self([values[0], values[1]])
    }

    fn random() -> Self {
        Self([
            rand::random::<f32>() * 2.0 - 1.0,
            rand::random::<f32>() * 2.0 - 1.0,
        ])
    }
}

impl<B: Backend> TensorConvertible<1, B> for SwimmerAction {
    fn to_tensor(&self, device: &B::Device) -> Tensor<B, 1> {
        Tensor::from_floats(self.0, device)
    }

    fn from_tensor(tensor: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let data = tensor.into_data();
        let slice = data.as_slice::<f32>().map_err(|e| TensorConversionError {
            message: format!("expected f32 action tensor: {e:?}"),
        })?;
        if slice.len() != 2 {
            return Err(TensorConversionError {
                message: format!("expected 2 action elements, got {}", slice.len()),
            });
        }
        Ok(Self([slice[0], slice[1]]))
    }
}
