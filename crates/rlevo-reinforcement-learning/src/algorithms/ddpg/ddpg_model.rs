//! Model traits implemented by every actor and critic used with [`DdpgAgent`].
//!
//! DDPG pairs a deterministic actor — mapping observations to continuous
//! actions — with a Q-critic that takes an `(obs, action)` pair and returns a
//! scalar value. Both networks have a Polyak-averaged target twin, a
//! [`valid`](Module::valid) snapshot updated by
//! [`polyak_update`](crate::utils::polyak_update). These traits mirror
//! [`crate::algorithms::dqn::dqn_model::DqnModel`]: one [`forward`] each, run
//! on the training network and on its target alike.
//!
//! [`DdpgAgent`]: super::ddpg_agent::DdpgAgent
//! [`forward`]: DeterministicPolicy::forward

use burn::module::Module;
use burn::tensor::Tensor;

/// Contract implemented by any network usable as a DDPG actor or target
/// actor.
///
/// The actor maps a batch of observations of rank `BOR` (including the batch
/// dimension) to a batch of continuous actions of rank `BAR`. Implementors
/// are expected to bake any tanh-squash + action-scale/bias into
/// [`forward`](Self::forward) so that the emitted tensor is already in the
/// env-visible action range.
///
/// The agent runs [`forward`](Self::forward) on the training actor, where the
/// autodiff graph is active, and on the frozen target actor during target-Q
/// computation, where it is not.
pub trait DeterministicPolicy<const BOR: usize, const BAR: usize>: Module {
    /// `(batch, ...obs)` → `(batch, ...action)`.
    fn forward(&self, obs: Tensor<BOR>) -> Tensor<BAR>;
}

/// Contract implemented by any network usable as a DDPG continuous Q-critic
/// or target critic.
///
/// The critic scores an `(observation, action)` pair and returns a scalar
/// value per batch row (shape `(batch,)`). Implementors typically concatenate
/// the observation and action along the feature axis before running an MLP,
/// but the trait does not prescribe the internal layout.
///
/// The agent runs [`forward`](Self::forward) on the training critic during
/// the critic and actor gradient steps, where the autodiff graph is active,
/// and on the target critic for target-Q computation, where no gradients
/// flow.
pub trait ContinuousQ<const BOR: usize, const BAR: usize>: Module {
    /// A `(batch,)` Q-value tensor.
    fn forward(&self, obs: Tensor<BOR>, act: Tensor<BAR>) -> Tensor<1>;
}
