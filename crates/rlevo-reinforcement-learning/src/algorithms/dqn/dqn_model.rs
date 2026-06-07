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
    ///
    /// Runs the same computation as [`forward`](Self::forward) but on the
    /// frozen target network, which lives on `B::InnerBackend`. No autodiff
    /// graph is constructed, making this suitable for computing bootstrap
    /// targets inside [`DqnAgent::learn_step`] and for inference.
    ///
    /// [`DqnAgent::learn_step`]: crate::algorithms::dqn::dqn_agent::DqnAgent::learn_step
    fn forward_inner(
        inner: &Self::InnerModule,
        observations: Tensor<B::InnerBackend, DB>,
    ) -> Tensor<B::InnerBackend, 2>;

    /// Updates the target network via Polyak averaging.
    ///
    /// Applies `target ← (1 − τ) · target + τ · active` element-wise to every
    /// parameter tensor and returns the updated target network. `target` is
    /// consumed and replaced; `active` is borrowed read-only.
    ///
    /// Typical `τ` values are in the range `[0.001, 0.01]`. Setting `τ = 1.0`
    /// is equivalent to a hard copy of the policy into the target. When
    /// [`DqnTrainingConfig::tau`] is `0.0`, the caller should use
    /// [`DqnAgent::sync_target`] for periodic hard syncs instead.
    ///
    /// [`DqnTrainingConfig::tau`]: crate::algorithms::dqn::dqn_config::DqnTrainingConfig::tau
    /// [`DqnAgent::sync_target`]: crate::algorithms::dqn::dqn_agent::DqnAgent::sync_target
    fn soft_update(active: &Self, target: Self::InnerModule, tau: f64) -> Self::InnerModule;
}
