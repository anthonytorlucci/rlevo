//! The [`PpoValue`] trait implemented by every value network used with
//! [`PpoAgent`](crate::algorithms::ppo::ppo_agent::PpoAgent).
//!
//! Parallels [`PpoPolicy`](crate::algorithms::ppo::ppo_policy::PpoPolicy).
//! Value networks output one scalar `V(s)` per batch row (return shape
//! `(batch_size,)`). There is no built-in value-head struct: callers supply their
//! own [`AutodiffModule`] implementation. A
//! typical implementation is a two-hidden-layer MLP whose final layer produces
//! a single linear output, mirroring the structure of the built-in policy heads.

use burn::module::AutodiffModule;
use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;

/// Contract implemented by any network usable as a PPO value function.
///
/// The `BOR` const generic is the batched observation tensor rank (usually
/// `2` for vector observations of shape `[batch, features]`). The output is
/// shape `(batch_size,)`.
pub trait PpoValue<B: AutodiffBackend, const BOR: usize>: AutodiffModule<B> {
    /// Forward pass: predicts `V(s)` per batch row. Return shape: `(batch_size,)`.
    fn forward(&self, obs: Tensor<B, BOR>) -> Tensor<B, 1>;
}
