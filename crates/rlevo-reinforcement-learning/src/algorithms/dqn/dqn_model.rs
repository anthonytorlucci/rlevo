//! The [`DqnModel`] trait implemented by every network used with [`DqnAgent`].
//!
//! [`DqnAgent`]: crate::algorithms::dqn::dqn_agent::DqnAgent

use burn::module::AutodiffModule;
use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;

use crate::utils::PolyakError;

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
/// The `BOR` const generic is the observation tensor rank *including* the
/// leading batch dimension (e.g. `BOR = 2` for vector observations of shape
/// `[batch_size, features]`; `BOR = 4` for image observations of shape
/// `[batch_size, channels, height, width]`).
pub trait DqnModel<B: AutodiffBackend, const BOR: usize>: AutodiffModule<B> {
    /// Autodiff forward pass: computes Q-values for a batch of observations.
    fn forward(&self, observations: Tensor<B, BOR>) -> Tensor<B, 2>;

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
        observations: Tensor<B::InnerBackend, BOR>,
    ) -> Tensor<B::InnerBackend, 2>;

    /// Updates the target network via Polyak averaging.
    ///
    /// Applies `target ← (1 − τ) · target + τ · active` element-wise to every
    /// parameter tensor and returns the updated target network. `target` is
    /// consumed and replaced; `active` is borrowed read-only.
    ///
    /// Typical `τ` values are in the range `[0.001, 0.01]`; `τ = 1.0` is a hard
    /// copy of the policy into the target, by degeneracy of the same formula.
    /// Callers do not choose τ per call — [`DqnTrainingConfig::target_update`]
    /// owns both the coefficient and the cadence, and
    /// [`TargetUpdate::fires_at`] hands the applicable τ to this method.
    ///
    /// [`DqnTrainingConfig::target_update`]: crate::algorithms::dqn::dqn_config::DqnTrainingConfig::target_update
    /// [`TargetUpdate::fires_at`]: crate::target::TargetUpdate::fires_at
    ///
    /// # Errors
    ///
    /// Propagates [`PolyakError`] if `active` and `target` have mismatched
    /// [`ParamId`](burn::module::ParamId) topologies — see
    /// [`polyak_update`](crate::utils::polyak_update).
    fn soft_update(
        active: &Self,
        target: Self::InnerModule,
        tau: f64,
    ) -> Result<Self::InnerModule, PolyakError>;
}
