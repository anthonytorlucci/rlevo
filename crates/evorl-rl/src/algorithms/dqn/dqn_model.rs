use burn::module::{AutodiffModule, Module};
use burn::tensor::backend::AutodiffBackend;

/// Bridges a trainable DQN model and its non-differentiable target counterpart.
///
/// `DqnModel` extends [`AutodiffModule`] so any implementor can transition to
/// its `InnerModule` (the non-differentiable copy used as the target network).
/// This is the standard Burn pattern for managing trainable vs. frozen network
/// copies in off-policy algorithms.
pub trait DqnModel<B: AutodiffBackend>: AutodiffModule<B> {
    /// Updates the target network via Polyak averaging.
    ///
    /// Applies `target ← (1 − τ) · target + τ · active` element-wise to all
    /// parameters and returns the updated target network.
    fn soft_update(active: &Self, target: Self::InnerModule, tau: f64) -> Self::InnerModule;
}
