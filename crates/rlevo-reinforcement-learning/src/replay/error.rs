//! Error type for replay-buffer operations.
//!
//! [`ReplayBufferError`] is the single error domain for everything under
//! [`crate::replay`]. It is re-exported from [`crate::memory`] so the pre-ADR
//! 0050 import path keeps compiling until `memory.rs` is deleted.

/// Errors that can occur during replay buffer operations.
///
/// Every fallible method on a [`ReplayStrategy`](super::ReplayStrategy)
/// returns this type. [`UniformReplay`](super::UniformReplay) can only ever
/// produce [`InsufficientData`](ReplayBufferError::InsufficientData); the other
/// variants exist for strategies and batch-assembly paths that touch tensors.
///
/// # Examples
///
/// ```
/// use rlevo_reinforcement_learning::replay::ReplayBufferError;
///
/// let err = ReplayBufferError::InsufficientData {
///     requested: 64,
///     available: 8,
/// };
/// assert_eq!(
///     err.to_string(),
///     "Insufficient data: requested 64, available 8"
/// );
/// ```
#[derive(Debug, thiserror::Error)]
pub enum ReplayBufferError {
    /// A general batch-assembly failure.
    ///
    /// Carries a human-readable description of what went wrong during
    /// tensor stacking or batch construction.
    #[error("Batch error: {0}")]
    BatchError(String),
    /// The buffer holds fewer experiences than the requested batch size.
    ///
    /// Returned by [`ReplayStrategy::sample`](super::ReplayStrategy::sample)
    /// when `batch_size > self.len()`. The caller should either reduce the
    /// batch size or wait until more transitions have been collected. Every
    /// in-crate agent gates its `learn_step` on a `can_learn()` predicate that
    /// makes this variant unreachable in practice.
    #[error("Insufficient data: requested {requested}, available {available}")]
    InsufficientData {
        /// Number of transitions the caller asked for.
        requested: usize,
        /// Number of transitions actually stored.
        available: usize,
    },
    /// A domain type could not be converted to or from a tensor.
    ///
    /// Wraps errors surfaced by
    /// [`TensorConvertible`](rlevo_core::base::TensorConvertible)
    /// implementations during observation, action, or reward tensor
    /// conversion.
    #[error("Tensor conversion error: {0}")]
    TensorConversionError(String),
}
