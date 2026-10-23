//! The [`DqnModel`] trait implemented by every network used with [`DqnAgent`].
//!
//! [`DqnAgent`]: crate::algorithms::dqn::dqn_agent::DqnAgent

use burn::module::Module;
use burn::tensor::Tensor;

/// Contract implemented by any network usable as a DQN policy or target.
///
/// One method, [`forward`](Self::forward): Q-values for a batch of
/// observations, a `(batch, n_actions)` tensor. The agent runs it on the policy
/// network, which trains, and on the target network, a
/// [`valid`](Module::valid) snapshot of the policy that does not. The target's
/// update is [`polyak_update`](crate::utils::polyak_update), the same for
/// every network, so it is not part of this trait.
///
/// The `BOR` const generic is the observation tensor rank *including* the
/// leading batch dimension (e.g. `BOR = 2` for vector observations of shape
/// `[batch_size, features]`; `BOR = 4` for image observations of shape
/// `[batch_size, channels, height, width]`).
pub trait DqnModel<const BOR: usize>: Module {
    /// Computes Q-values for a batch of observations.
    fn forward(&self, observations: Tensor<BOR>) -> Tensor<2>;
}
