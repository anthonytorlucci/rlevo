use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use std::fmt::Debug;

pub trait DrlModel<B: Backend, const R: usize> {
    /// Input type (e.g., state tensor).
    type Input: Into<Tensor<B, R>> + Debug + Clone;

    /// Output type (e.g., Q-values, policy logits).
    type Output: From<Tensor<B, R>> + Debug + Clone;

    /// Forward pass: compute output from input.
    fn forward(&self, input: Self::Input) -> Self::Output;

    /// Update the model parameters from a loss tensor.
    fn update(&mut self, loss: Tensor<B, 1>) -> Result<(), DrlModelError>;
}

// Custom error type for models:
#[derive(Debug)]
pub enum DrlModelError {
    UpdateError {
        message: String,
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
    BackendError(String),
    // ... other variants
}

impl std::fmt::Display for DrlModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UpdateError { message, .. } => write!(f, "Failed to update model: {}", message),
            Self::BackendError(msg) => write!(f, "Backend error occurred: {}", msg),
            // ... other variants
        }
    }
}

impl std::error::Error for DrlModelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::UpdateError {
                source: Some(s), ..
            } => Some(s.as_ref()),
            _ => None,
        }
    }
}
