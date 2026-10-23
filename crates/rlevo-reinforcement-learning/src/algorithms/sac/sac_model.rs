//! Model traits implemented by every actor and critic used with
//! [`SacAgent`](super::sac_agent::SacAgent).
//!
//! SAC pairs a **stochastic** squashed-Gaussian actor with **two** continuous
//! Q-critics. The critics reuse the DDPG/TD3 [`ContinuousQ`] contract
//! verbatim (same signatures, same Polyak-averaged target twins). The actor
//! trait is SAC-specific: instead of emitting a single deterministic action,
//! it samples `$a \sim \pi(\cdot|s)$` via the reparameterization trick and returns the
//! paired `$\log \pi(a|s)$` so the agent can score the entropy term used in both
//! the Bellman backup (`$y = r + \gamma(1-d)(\min Q - \alpha \log \pi)$`) and the actor loss
//! (`$L_\pi = \alpha \log \pi - \min Q$`).
//!
//! There is **no target actor**: SAC's stochastic policy plus
//! `min`-of-twin-Q already handles critic overestimation, so target bootstraps
//! go through the live actor. Callers must pre-sample the reparameterization
//! noise `$\epsilon \sim \mathcal{N}(0, I)$` on CPU and pass it to [`forward_sample`] so the
//! stochasticity stays reproducible under a seeded `rand::Rng`.
//!
//! [`forward_sample`]: SquashedGaussianPolicy::forward_sample

use burn::module::Module;
use burn::tensor::Tensor;

pub use crate::algorithms::ddpg::ddpg_model::ContinuousQ;

/// Paired squashed-action sample and per-row log-probability.
#[derive(Debug, Clone)]
pub struct SampleOutput<const BAR: usize> {
    /// Post-squash action tensor of shape `(batch, ...action)` already in the
    /// env's action range.
    pub action: Tensor<BAR>,
    /// Summed log-probability of `action` under the current policy, including
    /// the tanh Jacobian correction. Shape `(batch,)`.
    pub log_prob: Tensor<1>,
}

/// Contract implemented by any network usable as a SAC actor.
///
/// The actor maps a batch of observations of rank `BOR` (batch size included)
/// to a batch of reparameterized squashed-Gaussian samples of rank `BAR`,
/// along with their log-probabilities under the current policy. The squashing
/// (`tanh` + optional scale/bias) and the associated Jacobian correction must
/// be applied by the implementor so the returned `action` is already in the
/// env-visible range and the returned `log_prob` is the true log-density of
/// that squashed sample.
pub trait SquashedGaussianPolicy<const BOR: usize, const BAR: usize>: Module {
    /// Number of continuous action dimensions emitted per row. Needed by the
    /// agent to size the `$\epsilon$` noise tensor without another trait bound.
    fn action_dim(&self) -> usize;

    /// Given a pre-drawn standard-normal noise `eps` of shape
    /// `(batch, ...action)`, returns the squashed action and its
    /// log-probability (shape `(batch,)`) under the current policy. The agent
    /// runs it on the training actor and, for the Bellman target against
    /// `next_obs`, on a [`valid`](Module::valid) snapshot of it.
    fn forward_sample(&self, obs: Tensor<BOR>, eps: Tensor<BAR>) -> SampleOutput<BAR>;

    /// Deterministic evaluation action (policy mean squashed) — used by
    /// `SacAgent::act(training=false)`. Returns shape `(batch, ...action)`
    /// already in the env's action range.
    fn deterministic_action(&self, obs: Tensor<BOR>) -> Tensor<BAR>;
}
