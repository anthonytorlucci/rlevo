//! The [`PpoPolicy`] trait implemented by every policy network used with
//! [`PpoAgent`](crate::algorithms::ppo::ppo_agent::PpoAgent).
//!
//! Modelled on [`C51Model`](crate::algorithms::c51::c51_model::C51Model): the
//! trait is deliberately scoped to this module rather than being promoted to a
//! crate-level "stochastic policy" trait. Each existing algorithm in
//! `rlevo-reinforcement-learning` owns its model trait (`DqnModel`, `C51Model`, `QrDqnModel`);
//! cross-algo abstraction waits until SAC/PPG drive a concrete second consumer.
//!
//! # Associated `ActionTensor`
//!
//! Discrete categorical policies produce a `Tensor<B, 2, Int>` of shape
//! `(batch, 1)`; continuous tanh-squashed-Gaussian policies produce a
//! `Tensor<B, 2>` of shape `(batch, action_dim)`. The element-type difference
//! is surfaced via the associated `ActionTensor` type so one generic
//! `PpoAgent<B, P, V, ...>` serves both.
//!
//! # Serialisation to/from the rollout buffer
//!
//! The rollout buffer stores action rows as plain `Vec<f32>` (indices
//! cast-to-f32 for discrete, raw components for continuous). Each policy
//! implementation is responsible for round-tripping between its
//! [`ActionTensor`](PpoPolicy::ActionTensor) and flat f32 rows — see
//! [`PpoPolicy::action_row_from_tensor`] and
//! [`PpoPolicy::action_tensor_from_flat`].

use burn::module::AutodiffModule;
use burn::tensor::Tensor;
use burn::tensor::backend::{AutodiffBackend, Backend};
use rand::Rng;

/// Result of sampling from a policy: action tensor plus its log-probability
/// and entropy under the current policy.
#[derive(Debug, Clone)]
pub struct PolicyOutput<B: Backend, T> {
    /// Sampled action batch. Shape and dtype are policy-specific.
    pub action: T,
    /// Log-probability of each sampled action under the policy that produced
    /// it, shape `(batch,)`.
    pub log_prob: Tensor<B, 1>,
    /// Per-row entropy of the policy distribution, shape `(batch,)`.
    pub entropy: Tensor<B, 1>,
}

/// Log-probability and entropy of GIVEN actions under the current policy.
#[derive(Debug, Clone)]
pub struct LogProbEntropy<B: Backend> {
    /// Log-probability of each row's action under the current policy.
    pub log_prob: Tensor<B, 1>,
    /// Per-row entropy.
    pub entropy: Tensor<B, 1>,
}

/// Contract implemented by any network usable as a PPO policy.
///
/// The `DB` const generic is the batched observation tensor rank (usually
/// `2` for vector observations of shape `[batch, features]`).
pub trait PpoPolicy<B: AutodiffBackend, const DB: usize>: AutodiffModule<B> {
    /// The tensor type produced by [`sample_with_logprob`](Self::sample_with_logprob)
    /// and consumed by [`evaluate`](Self::evaluate).
    type ActionTensor: Clone;

    /// Number of f32 components used to represent one action row in the
    /// rollout buffer. `1` for categorical (the index), `action_dim` for
    /// continuous.
    fn action_dim(&self) -> usize;

    /// Samples actions for each row of `obs`, returning action, log-prob, and
    /// entropy. Implementations must thread `rng` explicitly rather than
    /// relying on thread-local state.
    fn sample_with_logprob(
        &self,
        obs: Tensor<B, DB>,
        rng: &mut dyn Rng,
    ) -> PolicyOutput<B, Self::ActionTensor>;

    /// Evaluates log-probability and entropy of `actions` under the current
    /// policy. Used by the PPO update loop to compute the new log-probs for
    /// importance weighting.
    fn evaluate(&self, obs: Tensor<B, DB>, actions: Self::ActionTensor) -> LogProbEntropy<B>;

    /// Extracts one action row from the sampled tensor as `action_dim()`
    /// f32s in the **buffer representation**. For the tanh-squashed
    /// Gaussian this is the pre-squash Gaussian sample `z`; for the
    /// categorical this is the index cast to f32. The buffer stores this
    /// representation so that [`evaluate`](Self::evaluate) can be called on
    /// exactly the value the old policy log-prob was computed against.
    fn action_row_from_tensor(action: &Self::ActionTensor, row: usize) -> Vec<f32>;

    /// Rebuilds a batched action tensor from row-major f32 data of length
    /// `n_rows · action_dim()`. Called at minibatch materialisation time.
    /// Must be the inverse of [`action_row_from_tensor`](Self::action_row_from_tensor).
    fn action_tensor_from_flat(
        flat: &[f32],
        n_rows: usize,
        device: &B::Device,
    ) -> Self::ActionTensor;

    /// Transforms one action row from the buffer representation (the
    /// pre-squash `z` for Gaussian, the index for Categorical) into the
    /// representation the environment expects. Default: identity.
    ///
    /// Gaussian heads override this to apply `scale · tanh(z)`.
    fn raw_to_env_row(&self, raw_row: &[f32]) -> Vec<f32> {
        raw_row.to_vec()
    }
}
