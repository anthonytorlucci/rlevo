//! Error type for replay-buffer operations.
//!
//! [`ReplayBufferError`] is the single error domain for everything under
//! [`crate::replay`]. [`ReplayStrategy::sample`](crate::replay::ReplayStrategy::sample)
//! is the only method on the seam that returns it — `push`, `len`, `is_empty`,
//! `get`, `get_mut`, and `iter` are infallible by signature. It superseded the
//! error type of the deleted `memory` module (ADR 0050 §8).

/// Errors that can occur during replay buffer operations.
///
/// The seam has one fallible method,
/// [`ReplayStrategy::sample`](super::ReplayStrategy::sample), and one way for it
/// to fail, so this enum carries one variant. Both shipped strategies —
/// [`UniformReplay`](super::UniformReplay) and
/// [`PrioritizedReplay`](super::PrioritizedReplay) — can only ever produce
/// [`InsufficientData`](ReplayBufferError::InsufficientData).
///
/// The enum is `#[non_exhaustive]`: downstream `match` expressions must carry a
/// wildcard arm, so a future variant is not a breaking change.
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
///
/// # Why there is no tensor or batch variant
///
/// This enum previously carried `TensorConversionError(String)` and
/// `BatchError(String)`. Neither was ever constructed — not in this module, and
/// not in the pre-ADR-0050 `memory.rs` it replaced — and neither can be. ADR
/// 0050 §3 puts tensors outside the seam entirely: `ReplayStrategy` "never sees
/// a `Tensor`, a `Backend`, or a device", and staging stays in the agent, the
/// only place that knows whether an action becomes an `Int` index tensor or a
/// float vector tensor. `sample` is the sole place a strategy could return an
/// error at all, so no conforming implementation has a tensor conversion or a
/// batch assembly available to fail at.
///
/// Carrying the failure as a typed payload —
/// `#[from] rlevo_core::base::TensorConversionError` instead of a `String` — was
/// considered and rejected. It would hand out-of-crate implementors `?`
/// ergonomics for exactly the boundary ADR 0050 §3 forbids, advertising the
/// crossing as a supported thing to do.
///
/// This is the same reasoning that keeps a bad-β variant out of the enum:
/// `sample`'s β is an [`ImportanceExponent`](super::ImportanceExponent), valid
/// by construction, and `UniformReplay` ignores β entirely, so per ADR 0051 §2
/// "a fallible signature would force it to carry an error variant it can never
/// produce" (restated on
/// [`ReplayStrategy::sample`](super::ReplayStrategy::sample)). It is also
/// consistent with how the module already handles the one *real*
/// batch-assembly failure it has: [`SampledBatch::weighted`](super::SampledBatch::weighted)
/// treats `weights.len() != ids.len()` as a deliberate panic — "a programming
/// error in the strategy" — rather than an `Err`. A `BatchError` variant would
/// contradict a decision this module has already made explicitly.
///
/// Because the enum is `#[non_exhaustive]`, adding a variant later is not a
/// breaking change; a disk-backed or distributed strategy whose `sample`
/// performs IO is the shape that would plausibly need one. The bar for adding it
/// is an actual consumer that constructs it, not an anticipated one — an
/// unconstructible variant is what this section exists to prevent recurring.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ReplayBufferError {
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
}
