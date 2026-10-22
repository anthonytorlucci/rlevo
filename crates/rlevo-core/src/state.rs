//! [`StateError`]: why a value is not a legal state of an environment.

/// Error type for state validation failures.
///
/// Returned by a constructor or conversion when a value's contents do not
/// make a legal state.
///
/// # Examples
///
/// ```
/// use rlevo_core::state::StateError;
///
/// let err = StateError::InvalidData("NaN in position field".into());
/// assert!(err.to_string().contains("NaN in position field"));
/// ```
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum StateError {
    /// Data contents violate invariants.
    #[error("Invalid data: {0}")]
    InvalidData(String),
}
