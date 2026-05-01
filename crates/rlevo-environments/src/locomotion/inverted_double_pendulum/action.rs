//! Action type for [`super::InvertedDoublePendulum`].

use burn::prelude::{Backend, Tensor};
use rlevo_core::action::ContinuousAction;
use rlevo_core::base::{Action, TensorConversionError, TensorConvertible};
use serde::{Deserialize, Serialize};

/// 1D continuous action — horizontal force target on the cart, in pre-gear
/// units. Bounds: `[-1.0, 1.0]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct InvertedDoublePendulumAction(pub [f32; 1]);

impl InvertedDoublePendulumAction {
    #[must_use]
    pub const fn new(force: f32) -> Self {
        Self([force])
    }
}

impl Action<1> for InvertedDoublePendulumAction {
    fn shape() -> [usize; 1] {
        [1]
    }

    fn is_valid(&self) -> bool {
        self.0[0].is_finite() && self.0[0].abs() <= 1.0
    }
}

impl ContinuousAction<1> for InvertedDoublePendulumAction {
    fn as_slice(&self) -> &[f32] {
        &self.0
    }

    fn clip(&self, min: f32, max: f32) -> Self {
        Self([self.0[0].clamp(min, max)])
    }

    fn from_slice(values: &[f32]) -> Self {
        Self([values[0]])
    }

    fn random() -> Self {
        Self([rand::random::<f32>() * 2.0 - 1.0])
    }
}

impl<B: Backend> TensorConvertible<1, B> for InvertedDoublePendulumAction {
    fn to_tensor(&self, device: &B::Device) -> Tensor<B, 1> {
        Tensor::from_floats(self.0, device)
    }

    fn from_tensor(tensor: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let data = tensor.into_data();
        let slice = data.as_slice::<f32>().map_err(|e| TensorConversionError {
            message: format!("expected f32 action tensor: {e:?}"),
        })?;
        if slice.len() != 1 {
            return Err(TensorConversionError {
                message: format!("expected 1 action element, got {}", slice.len()),
            });
        }
        Ok(Self([slice[0]]))
    }
}
