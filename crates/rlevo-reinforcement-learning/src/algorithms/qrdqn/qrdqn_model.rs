//! The [`QrDqnModel`] trait implemented by every network used with
//! [`QrDqnAgent`](crate::algorithms::qrdqn::qrdqn_agent::QrDqnAgent).
//!
//! Parallel to [`C51Model`](crate::algorithms::c51::c51_model::C51Model), but the forward pass
//! returns a rank-3 tensor of **raw quantile values** shaped `(batch, num_actions, num_quantiles)`
//! rather than atom logits. No activation is applied at the head — quantile values are scalar
//! estimates of the return distribution's quantile function at the implicit midpoints
//! `$\tau_i = (i + 0.5)/N$`.

use burn::module::Module;
use burn::tensor::Tensor;

/// Contract implemented by any network usable as a QR-DQN policy or target.
///
/// One method, [`forward`](Self::forward): quantile values for a batch of
/// observations, a `(batch, num_actions, num_quantiles)` tensor (no activation
/// at the head). The agent runs it on the policy network, which trains, and on
/// the target network, a [`valid`](Module::valid) snapshot of the policy that
/// does not. The target's update is
/// [`polyak_update`](crate::utils::polyak_update), the same for every network,
/// so it is not part of this trait.
///
/// The `BOR` const generic is the observation tensor rank *including* the
/// leading batch dimension (e.g. `BOR = 2` for vector observations of shape
/// `[batch, features]`).
pub trait QrDqnModel<const BOR: usize>: Module {
    /// Quantile values of shape `(batch_size, num_actions, num_quantiles)`.
    fn forward(&self, observations: Tensor<BOR>) -> Tensor<3>;
}
