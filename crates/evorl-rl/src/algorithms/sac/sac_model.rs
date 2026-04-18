//! Model traits implemented by every actor and critic used with
//! [`SacAgent`](super::sac_agent::SacAgent).
//!
//! SAC pairs a **stochastic** squashed-Gaussian actor with **two** continuous
//! Q-critics. The critics reuse the DDPG/TD3 [`ContinuousQ`] contract
//! verbatim (same signatures, same Polyak-averaged target twins). The actor
//! trait is SAC-specific: instead of emitting a single deterministic action,
//! it samples `a ~ π(·|s)` via the reparameterization trick and returns the
//! paired `log π(a|s)` so the agent can score the entropy term used in both
//! the Bellman backup (`y = r + γ(1−d)(min Q − α log π)`) and the actor loss
//! (`L_π = α log π − min Q`).
//!
//! There is **no target actor**: SAC's stochastic policy plus
//! `min`-of-twin-Q already handles critic overestimation, so target bootstraps
//! go through the live actor. Callers must pre-sample the reparameterization
//! noise `ε ~ N(0, I)` on CPU and pass it to [`forward_sample`] /
//! [`forward_sample_inner`] so the stochasticity stays reproducible under a
//! seeded `rand::Rng`.
//!
//! [`forward_sample`]: SquashedGaussianPolicy::forward_sample
//! [`forward_sample_inner`]: SquashedGaussianPolicy::forward_sample_inner

use burn::module::AutodiffModule;
use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;

pub use crate::algorithms::ddpg::ddpg_model::ContinuousQ;

/// Paired squashed-action sample and per-row log-probability.
#[derive(Debug, Clone)]
pub struct SampleOutput<B: burn::tensor::backend::Backend, const DAB: usize> {
    /// Post-squash action tensor of shape `(batch, ...action)` already in the
    /// env's action range.
    pub action: Tensor<B, DAB>,
    /// Summed log-probability of `action` under the current policy, including
    /// the tanh Jacobian correction. Shape `(batch,)`.
    pub log_prob: Tensor<B, 1>,
}

/// Contract implemented by any network usable as a SAC actor.
///
/// The actor maps a batch of observations of rank `DB` (batch dim included)
/// to a batch of reparameterized squashed-Gaussian samples of rank `DAB`,
/// along with their log-probabilities under the current policy. The squashing
/// (`tanh` + optional scale/bias) and the associated Jacobian correction must
/// be applied by the implementor so the returned `action` is already in the
/// env-visible range and the returned `log_prob` is the true log-density of
/// that squashed sample.
pub trait SquashedGaussianPolicy<B: AutodiffBackend, const DB: usize, const DAB: usize>:
    AutodiffModule<B>
{
    /// Number of continuous action dimensions emitted per row. Needed by the
    /// agent to size the `ε` noise tensor without another trait bound.
    fn action_dim(&self) -> usize;

    /// Autodiff forward sample: given a pre-drawn standard-normal noise
    /// `eps` of shape `(batch, ...action)`, returns the squashed action and
    /// its log-probability (shape `(batch,)`) under the current policy.
    fn forward_sample(
        &self,
        obs: Tensor<B, DB>,
        eps: Tensor<B, DAB>,
    ) -> SampleOutput<B, DAB>;

    /// No-autodiff counterpart used when computing the Bellman target against
    /// `next_obs`. Produces the same squashed sample + log-prob on the inner
    /// backend.
    fn forward_sample_inner(
        inner: &Self::InnerModule,
        obs: Tensor<B::InnerBackend, DB>,
        eps: Tensor<B::InnerBackend, DAB>,
    ) -> SampleOutput<B::InnerBackend, DAB>;

    /// Deterministic evaluation action (policy mean squashed) — used by
    /// `SacAgent::act(training=false)`. Returns shape `(batch, ...action)`
    /// already in the env's action range.
    fn deterministic_action(&self, obs: Tensor<B, DB>) -> Tensor<B, DAB>;
}
