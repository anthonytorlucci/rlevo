use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

// --- Utility functions for reinforcement learning

pub fn compute_target_q_values<B: Backend>(
    rewards: Tensor<B, 1>,
    next_q_max: Tensor<B, 1>,
    dones: Tensor<B, 1>,
    gamma: f32,
) -> Tensor<B, 1> {
    // target = reward + gamma * max_next_q * (1 - done)
    rewards.clone() + gamma * next_q_max * (1.0 - dones)
}
