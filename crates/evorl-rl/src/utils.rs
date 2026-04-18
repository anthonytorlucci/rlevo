//! Shared utility functions for reinforcement learning.
//!
//! Provides stateless helper functions used across multiple RL algorithms,
//! such as Bellman target computation.

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

/// Computes Bellman backup target Q-values for a mini-batch.
///
/// Applies the standard one-step TD target:
/// `target = reward + γ · max_next_Q · (1 − done)`.
/// The `dones` mask zeros out the bootstrap term for terminal transitions.
pub fn compute_target_q_values<B: Backend>(
    rewards: Tensor<B, 1>,
    next_q_max: Tensor<B, 1>,
    dones: Tensor<B, 1>,
    gamma: f32,
) -> Tensor<B, 1> {
    rewards.clone() + gamma * next_q_max * (1.0 - dones)
}
