//! The [`PpoValue`] trait implemented by every value network used with
//! [`PpoAgent`](crate::algorithms::ppo::ppo_agent::PpoAgent).
//!
//! Parallels [`PpoPolicy`](crate::algorithms::ppo::ppo_policy::PpoPolicy).
//! Value networks output a scalar per batch row.

use burn::module::AutodiffModule;
use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;

/// Contract implemented by any network usable as a PPO value function.
///
/// The `DB` const generic is the batched observation tensor rank (usually
/// `2` for vector observations of shape `[batch, features]`). The output is
/// shape `(batch,)`.
pub trait PpoValue<B: AutodiffBackend, const DB: usize>: AutodiffModule<B> {
    /// Forward pass: predicts `V(s)` per batch row. Return shape: `(batch,)`.
    fn forward(&self, obs: Tensor<B, DB>) -> Tensor<B, 1>;
}
