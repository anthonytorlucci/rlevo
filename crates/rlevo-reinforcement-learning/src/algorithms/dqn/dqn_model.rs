//! The [`DqnModel`] trait implemented by every network used with [`DqnAgent`],
//! [`C51Agent`] or [`QrDqnAgent`].
//!
//! [`DqnAgent`]: crate::algorithms::dqn::dqn_agent::DqnAgent
//! [`C51Agent`]: crate::algorithms::c51::c51_agent::C51Agent
//! [`QrDqnAgent`]: crate::algorithms::qrdqn::qrdqn_agent::QrDqnAgent

use burn::module::Module;
use burn::tensor::Tensor;

/// Contract implemented by any network usable as a DQN, C51 or QR-DQN
/// policy or target.
///
/// One method, [`forward`](Self::forward), from a batch of observations to a
/// value per action. The output rank `OUT` says which:
///
/// - **`OUT = 2`, the default:** Q-values, a `(batch, num_actions)` tensor.
///   [`DqnAgent`] takes this, so `DqnModel<BOR>` means a Q-network.
/// - **`OUT = 3`:** a return distribution per action, a `(batch,
///   num_actions, num_atoms)` tensor. [`C51Agent`] reads it as
///   **unnormalised logits** over a fixed atom support: it takes a `softmax`
///   along the atom axis and the expected Q-value `$\sum_i z_i \cdot p_i$`.
///   [`QrDqnAgent`] reads it as **raw quantile values** at the midpoints
///   `$\tau_i = (i + 0.5)/N$`, so a network applies no activation at its
///   head. The network is the same for both; the agent decides what the
///   numbers mean.
///
/// The agent runs `forward` on the policy network, which trains, and on the
/// target network, a [`valid`](Module::valid) snapshot of the policy that
/// does not. The target's update is
/// [`polyak_update`](crate::utils::polyak_update), the same for every
/// network, so it is not part of this trait.
///
/// The `BOR` const generic is the observation tensor rank *including* the
/// leading batch dimension (e.g. `BOR = 2` for vector observations of shape
/// `[batch_size, features]`; `BOR = 4` for image observations of shape
/// `[batch_size, channels, height, width]`).
///
/// [`DqnAgent`]: crate::algorithms::dqn::dqn_agent::DqnAgent
/// [`C51Agent`]: crate::algorithms::c51::c51_agent::C51Agent
/// [`QrDqnAgent`]: crate::algorithms::qrdqn::qrdqn_agent::QrDqnAgent
pub trait DqnModel<const BOR: usize, const OUT: usize = 2>: Module {
    /// A value per action for a batch of observations: Q-values when
    /// `OUT = 2`, a distribution per action when `OUT = 3`.
    fn forward(&self, observations: Tensor<BOR>) -> Tensor<OUT>;
}
