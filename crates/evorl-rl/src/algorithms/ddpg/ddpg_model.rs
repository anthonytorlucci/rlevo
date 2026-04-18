//! Model traits implemented by every actor and critic used with [`DdpgAgent`].
//!
//! DDPG pairs a deterministic actor — mapping observations to continuous
//! actions — with a Q-critic that takes an `(obs, action)` pair and returns a
//! scalar value. Both networks have a Polyak-averaged target twin. These
//! traits mirror [`crate::algorithms::dqn::dqn_model::DqnModel`]: implementors
//! provide an autodiff [`forward`], an inner [`forward_inner`] against the
//! target module, and a [`soft_update`] hook so the agent need not know how
//! each concrete module is parameterised.
//!
//! [`DdpgAgent`]: super::ddpg_agent::DdpgAgent
//! [`forward`]: DeterministicPolicy::forward
//! [`forward_inner`]: DeterministicPolicy::forward_inner
//! [`soft_update`]: DeterministicPolicy::soft_update

use burn::module::AutodiffModule;
use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;

/// Contract implemented by any network usable as a DDPG actor or target
/// actor.
///
/// The actor maps a batch of observations of rank `DB` (including the batch
/// dimension) to a batch of continuous actions of rank `DAB`. Implementors
/// are expected to bake any tanh-squash + action-scale/bias into
/// [`forward`](Self::forward) so that the emitted tensor is already in the
/// env-visible action range.
pub trait DeterministicPolicy<B: AutodiffBackend, const DB: usize, const DAB: usize>:
    AutodiffModule<B>
{
    /// Autodiff forward pass: `(batch, ...obs)` → `(batch, ...action)`.
    fn forward(&self, obs: Tensor<B, DB>) -> Tensor<B, DAB>;

    /// Forward pass against the inner (non-autodiff) target actor.
    fn forward_inner(
        inner: &Self::InnerModule,
        obs: Tensor<B::InnerBackend, DB>,
    ) -> Tensor<B::InnerBackend, DAB>;

    /// Polyak-averages the target toward the active network.
    ///
    /// Returns `(1 − τ) · target + τ · active` element-wise for every
    /// parameter.
    fn soft_update(active: &Self, target: Self::InnerModule, tau: f64)
    -> Self::InnerModule;
}

/// Contract implemented by any network usable as a DDPG continuous Q-critic
/// or target critic.
///
/// The critic scores an `(observation, action)` pair and returns a scalar
/// value per batch row (shape `(batch,)`). Implementors typically concatenate
/// the observation and action along the feature axis before running an MLP,
/// but the trait does not prescribe the internal layout.
pub trait ContinuousQ<B: AutodiffBackend, const DB: usize, const DAB: usize>:
    AutodiffModule<B>
{
    /// Autodiff forward pass producing a `(batch,)` Q-value tensor.
    fn forward(&self, obs: Tensor<B, DB>, act: Tensor<B, DAB>) -> Tensor<B, 1>;

    /// Forward pass against the inner (non-autodiff) target critic.
    fn forward_inner(
        inner: &Self::InnerModule,
        obs: Tensor<B::InnerBackend, DB>,
        act: Tensor<B::InnerBackend, DAB>,
    ) -> Tensor<B::InnerBackend, 1>;

    /// Polyak-averages the target toward the active network.
    fn soft_update(active: &Self, target: Self::InnerModule, tau: f64)
    -> Self::InnerModule;
}
