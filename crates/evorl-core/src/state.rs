use burn::tensor::backend::Backend;
use burn::tensor::Int;
use burn::tensor::Tensor;
use std::fmt::Debug;
use std::hash::Hash;

/// Core trait for all state types. Minimal, framework-agnostic foundation.
pub trait State: Debug + Clone + PartialEq + Eq + Hash {
    /// Validate that this state is legal in the environment
    fn is_valid(&self) -> bool {
        true
    }

    /// Get the total number of elements in this state
    /// Useful for determining buffer sizes and model input dimensions
    fn numel(&self) -> usize;

    /// Get the logical shape/dimensions of this state
    /// Returns a flat representation (e.g., `[seq_len, features]`)
    fn shape(&self) -> Vec<usize>;
}

/// Trait for states that can be represented as contiguous numeric arrays
/// Useful for neural network inputs and efficient computation
pub trait FlattenedState: State {
    /// Flatten state into a f32 vector for neural network input
    /// Order must be consistent for a given state type
    fn flatten(&self) -> Vec<f32>;

    /// Reconstruct state from flattened vector
    /// Returns Err if vector length doesn't match expected size
    fn from_flattened(data: Vec<f32>) -> Result<Self, StateError>;
}

/// Trait for sequential/temporal states
/// For states that contain historical observations or time-series data
pub trait TemporalState: State {
    /// Get the length of the sequence (e.g., number of historical steps)
    fn sequence_length(&self) -> usize;

    /// Get the most recent element(s) in the sequence
    /// Useful for incremental updates without full reconstruction
    fn latest(&self) -> &[f32];

    /// Create a new state by appending a new observation and dropping the oldest
    /// Returns Err if new observation has incompatible shape
    fn push_pop(&self, new_observation: &[f32]) -> Result<Self, StateError>;
}

/// Error types for state operations
#[derive(Debug, Clone, PartialEq)]
pub enum StateError {
    InvalidShape {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    InvalidData(String),
    InvalidSize {
        expected: usize,
        got: usize,
    },
}

impl std::fmt::Display for StateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StateError::InvalidShape { expected, got } => {
                write!(f, "Invalid shape: expected {:?}, got {:?}", expected, got)
            }
            StateError::InvalidData(msg) => write!(f, "Invalid data: {}", msg),
            StateError::InvalidSize { expected, got } => {
                write!(f, "Invalid size: expected {}, got {}", expected, got)
            }
        }
    }
}

impl std::error::Error for StateError {}

/// Framework-Specific Conversion Trait
/// Trait for converting states to framework tensors
/// Keeps core state logic framework-agnostic
pub trait StateTensorConvertible<const R: usize> {
    const R1: usize = R + 1; // dimensions including batch_size
    fn to_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, R>;

    /// Optional: convert back from tensor
    /// Useful for verifying round-trip conversion
    fn from_tensor<B: Backend>(tensor: &Tensor<B, R>) -> Result<Self, StateError>
    where
        Self: Sized;
}

/// For states that can also be integer tensors
pub trait StateIntTensorConvertible<const R: usize> {
    const R1: usize = R + 1; // dimensions including batch_size
    fn to_int_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, R, Int>;
}
