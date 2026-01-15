use crate::base::TensorConvertible;
use crate::state::State;
use std::fmt::Debug;

// todo! Snapshot?

// todo! Environment

/// Error type for environment operations.
///
/// `EnvironmentError` captures failures that can occur during environment
/// initialization, reset, or stepping. It provides detailed error messages
/// and supports error chaining via the standard `Error` trait.
///
/// # Variants
///
/// * `InvalidAction` - The provided action is not valid in the current state
/// * `RenderFailed` - Rendering/display operation failed
/// * `IoError` - An I/O operation failed (wrapped std::io::Error)
#[derive(Debug)]
pub enum EnvironmentError {
    /// An invalid or out-of-bounds action was provided.
    InvalidAction(String),
    /// Rendering or display failed.
    RenderFailed(String),
    /// An I/O operation failed (wraps std::io::Error).
    IoError(std::io::Error),
}

impl std::error::Error for EnvironmentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            EnvironmentError::IoError(io_err) => Some(io_err),
            _ => None,
        }
    }
}

impl std::fmt::Display for EnvironmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EnvironmentError::InvalidAction(action_error) => {
                write!(f, "Invalid action: {}", action_error)
            }
            EnvironmentError::RenderFailed(render_error) => {
                write!(f, "Render failed: {}", render_error)
            }
            EnvironmentError::IoError(io_err) => {
                write!(f, "IO operation failed: {}", io_err)
            }
        }
    }
}

impl From<std::io::Error> for EnvironmentError {
    fn from(error: std::io::Error) -> Self {
        EnvironmentError::IoError(error)
    }
}
