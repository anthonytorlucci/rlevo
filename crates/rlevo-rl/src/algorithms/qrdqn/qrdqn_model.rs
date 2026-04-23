//! The [`QrDqnModel`] trait implemented by every network used with
//! [`QrDqnAgent`](crate::algorithms::qrdqn::qrdqn_agent::QrDqnAgent).
//!
//! Parallel to [`C51Model`](crate::algorithms::c51::c51_model::C51Model), but
//! the forward pass returns a rank-3 tensor of **raw quantile values**
//! shaped `(batch, num_actions, num_quantiles)` rather than atom logits. No
//! activation is applied at the head — quantile values are scalar
//! estimates of the return distribution's quantile function at the implicit
//! midpoints `τ_i = (i + 0.5) / N`.

use burn::module::AutodiffModule;
use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;

/// Contract implemented by any network usable as a QR-DQN policy or target.
///
/// Implementors provide:
/// - [`forward`](Self::forward) — autodiff forward pass used when training
///   the policy network. Produces a `(batch, num_actions, num_quantiles)`
///   tensor of quantile values (no activation at the head).
/// - [`forward_inner`](Self::forward_inner) — the same computation against
///   the inner non-autodiff module used as the frozen target network.
/// - [`soft_update`](Self::soft_update) — Polyak averaging of the target
///   network: `target ← (1 − τ) · target + τ · active`.
///
/// The `DB` const generic is the observation tensor rank *including* the
/// leading batch dimension (e.g. `DB = 2` for vector observations of shape
/// `[batch, features]`).
pub trait QrDqnModel<B: AutodiffBackend, const DB: usize>: AutodiffModule<B> {
    /// Autodiff forward pass: quantile values of shape
    /// `(batch, num_actions, num_quantiles)`.
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
