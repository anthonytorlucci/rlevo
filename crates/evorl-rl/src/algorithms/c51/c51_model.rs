//! The [`C51Model`] trait implemented by every network used with
//! [`C51Agent`](crate::algorithms::c51::c51_agent::C51Agent).
//!
//! Parallel to [`DqnModel`](crate::algorithms::dqn::dqn_model::DqnModel), but
//! the forward pass returns a rank-3 tensor of **unnormalised logits** shaped
//! `(batch, num_actions, num_atoms)` rather than a rank-2 tensor of
//! Q-values. Agents convert logits → probabilities via `softmax` along the
//! atom axis and compute expected Q-values as `Σ_i z_i · p_i`.

use burn::module::AutodiffModule;
use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;

/// Contract implemented by any network usable as a C51 policy or target.
///
/// Implementors provide:
/// - [`forward`](Self::forward) — autodiff forward pass used when training
///   the policy network. Produces a `(batch, num_actions, num_atoms)` tensor
///   of atom logits.
/// - [`forward_inner`](Self::forward_inner) — the same computation against
///   the inner non-autodiff module used as the frozen target network.
/// - [`soft_update`](Self::soft_update) — Polyak averaging of the target
///   network: `target ← (1 − τ) · target + τ · active`.
///
/// The `DB` const generic is the observation tensor rank *including* the
/// leading batch dimension (e.g. `DB = 2` for vector observations of shape
/// `[batch, features]`).
pub trait C51Model<B: AutodiffBackend, const DB: usize>: AutodiffModule<B> {
    /// Autodiff forward pass: logits of shape
    /// `(batch, num_actions, num_atoms)`.
    fn forward(&self, observations: Tensor<B, DB>) -> Tensor<B, 3>;

    /// Forward pass against the inner (non-autodiff) target module.
    fn forward_inner(
        inner: &Self::InnerModule,
        observations: Tensor<B::InnerBackend, DB>,
    ) -> Tensor<B::InnerBackend, 3>;

    /// Polyak-averages `active` into `target` with coefficient `tau` and
    /// returns the updated target network.
    fn soft_update(active: &Self, target: Self::InnerModule, tau: f64) -> Self::InnerModule;
}
