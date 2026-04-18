//! Auxiliary-head contract for Phasic Policy Gradient policy networks.
//!
//! In addition to the standard
//! [`PpoPolicy`](crate::algorithms::ppo::ppo_policy::PpoPolicy) surface (policy-phase sampling
//! and log-prob evaluation), PPG requires two extra forward passes during the
//! auxiliary phase:
//!
//! 1. `aux_value(obs) → V_aux(s)` — an auxiliary value head that shares the
//!    policy network's trunk, trained against the same target returns as the
//!    main value network.
//! 2. `logits(obs) → logits(s)` — access to the raw pre-softmax logits so the
//!    auxiliary phase can compute `KL(π_old ‖ π_new)` for distillation.
//!
//! # Scope
//!
//! v1 of PPG is discrete-only: we expose logits directly and compute the KL
//! via `Σ p_old · (log p_old − log p_new)`. A continuous analogue (exposing
//! Gaussian distribution parameters) is a follow-up once a Gaussian PPG head
//! drives a second concrete consumer — matching the "cross-algo abstraction
//! waits until SAC/PPG drive a concrete second consumer" pattern noted in
//! [`ppo_policy`](crate::algorithms::ppo::ppo_policy).

use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;

/// Auxiliary-phase contract for PPG policy networks.
///
/// Implemented by any policy network used with
/// [`PpgAgent`](crate::algorithms::ppg::ppg_agent::PpgAgent). The network is
/// expected to also implement
/// [`PpoPolicy`](crate::algorithms::ppo::ppo_policy::PpoPolicy) so the policy
/// phase runs unchanged.
pub trait PpgAuxValueHead<B: AutodiffBackend, const DB: usize>: burn::module::AutodiffModule<B> {
    /// Auxiliary value prediction, shape `(batch,)`.
    ///
    /// Trained in the auxiliary phase against the same target returns as the
    /// main value network. Unused during the policy phase.
    fn aux_value(&self, obs: Tensor<B, DB>) -> Tensor<B, 1>;

    /// Raw policy logits, shape `(batch, num_actions)`.
    ///
    /// Exposed so the auxiliary phase can compute
    /// `KL(π_old ‖ π_new) = Σ softmax(old_logits) · (log_softmax(old_logits) − log_softmax(new_logits))`
    /// for distillation. Returning logits (rather than pre-computed
    /// log-probs) keeps the KL kernel decoupled from how the network
    /// internally parameterises its distribution.
    fn logits(&self, obs: Tensor<B, DB>) -> Tensor<B, 2>;
}
