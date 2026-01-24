use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use std::error::Error;

#[derive(Debug, Clone, PartialEq)]
pub struct TensorConversionError {
    pub message: String,
}

impl std::fmt::Display for TensorConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Invalid tensor conversion: {}", self.message)
    }
}

impl Error for TensorConversionError {}

pub trait TensorConvertible<const D: usize, B: Backend> {
    fn to_tensor(&self, device: &B::Device) -> Tensor<B, D>;
    // todo! fn from_tensor(tensor: &Tensor<B, D>) -> Result<Self, TensorConversionError>;
}
