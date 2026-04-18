//! The [`DqnModel`] trait implemented by every network used with [`DqnAgent`].
//!
//! [`DqnAgent`]: crate::algorithms::dqn::dqn_agent::DqnAgent

use burn::module::AutodiffModule;
use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;

/// Contract implemented by any network usable as a DQN policy or target.
///
/// Implementors provide:
/// - [`forward`](Self::forward) — autodiff forward pass used when training
///   the policy network. Produces a `(batch, n_actions)` tensor of
///   Q-values.
/// - [`forward_inner`](Self::forward_inner) — the same computation against
///   the inner non-autodiff module used as the target network.
/// - [`soft_update`](Self::soft_update) — Polyak averaging of the target
///   network: `target ← (1 − τ) · target + τ · active`.
///
/// The `DB` const generic is the observation tensor rank *including* the
/// leading batch dimension (e.g. `DB = 2` for vector observations of shape
/// `[batch, features]`; `DB = 4` for image observations of shape
/// `[batch, channels, height, width]`).
pub trait DqnModel<B: AutodiffBackend, const DB: usize>: AutodiffModule<B> {
    /// Autodiff forward pass: computes Q-values for a batch of observations.
    fn forward(&self, observations: Tensor<B, DB>) -> Tensor<B, 2>;

    /// Forward pass against the inner (non-autodiff) target module.
    fn forward_inner(
        inner: &Self::InnerModule,
        observations: Tensor<B::InnerBackend, DB>,
    ) -> Tensor<B::InnerBackend, 2>;

    /// Updates the target network via Polyak averaging.
    ///
    /// Applies `target ← (1 − τ) · target + τ · active` element-wise to all
    /// parameters and returns the updated target network.
    fn soft_update(active: &Self, target: Self::InnerModule, tau: f64) -> Self::InnerModule;
}
