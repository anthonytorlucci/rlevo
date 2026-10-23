//! The [`C51Model`] trait implemented by every network used with
//! [`C51Agent`](crate::algorithms::c51::c51_agent::C51Agent).
//!
//! Parallel to [`DqnModel`](crate::algorithms::dqn::dqn_model::DqnModel), but
//! the forward pass returns a rank-3 tensor of **unnormalised logits** shaped
//! `(batch, num_actions, num_atoms)` rather than a rank-2 tensor of
//! Q-values. Agents convert logits → probabilities via `softmax` along the
//! atom axis and compute expected Q-values as `$\sum_i z_i \cdot p_i$`.

use burn::module::Module;
use burn::tensor::Tensor;

/// Contract implemented by any network usable as a C51 policy or target.
///
/// One method, [`forward`](Self::forward): atom logits for a batch of
/// observations, a `(batch, num_actions, num_atoms)` tensor. The agent runs it
/// on the policy network, which trains, and on the target network, a
/// [`valid`](Module::valid) snapshot of the policy that does not. The target's
/// update is [`polyak_update`](crate::utils::polyak_update), the same for every
/// network, so it is not part of this trait.
///
/// The `BS` const generic is the observation tensor rank *including* the
/// leading batch dimension (e.g. `BS = 2` for vector observations of shape
/// `[batch_size, features]`).
pub trait C51Model<const BS: usize>: Module {
    /// Logits of shape `(batch, num_actions, num_atoms)`.
    fn forward(&self, observations: Tensor<BS>) -> Tensor<3>;
}
