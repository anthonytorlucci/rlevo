//! Shared utility functions for reinforcement learning.
//!
//! Provides stateless helper functions used across multiple RL algorithms,
//! such as Bellman target computation and Polyak averaging.

use std::collections::HashMap;
use std::marker::PhantomData;

use burn::module::{Module, ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::tensor::{Tensor, TensorData};
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

// ---------------------------------------------------------------------------
// Polyak averaging
// ---------------------------------------------------------------------------

struct ParamCollector<B: Backend> {
    tensors: HashMap<ParamId, TensorData>,
    _marker: PhantomData<B>,
}

impl<B: Backend> ModuleVisitor<B> for ParamCollector<B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        self.tensors.insert(param.id, param.val().to_data());
    }
}

struct PolyakMapper<B: Backend> {
    active: HashMap<ParamId, TensorData>,
    tau: f32,
    _marker: PhantomData<B>,
}

impl<B: Backend> ModuleMapper<B> for PolyakMapper<B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let id = param.id;
        let active = self
            .active
            .remove(&id)
            .expect("param not collected from active network");
        let tau = self.tau;
        param.map(move |target_tensor| {
            let device = target_tensor.device();
            let active_tensor = Tensor::<B, D>::from_data(active, &device);
            target_tensor.mul_scalar(1.0 - tau) + active_tensor.mul_scalar(tau)
        })
    }
}

/// Polyak-averages `active` into `target`: `target ← (1 − τ)·target + τ·active`.
///
/// Used by every off-policy algorithm that maintains a target network (DQN,
/// C51, QR-DQN, DDPG, TD3, SAC). Pass `tau = 1.0` for a hard copy.
pub fn polyak_update<B: Backend, M: Module<B>>(active: &M, target: M, tau: f32) -> M {
    let mut collector = ParamCollector::<B> {
        tensors: HashMap::new(),
        _marker: PhantomData,
    };
    active.visit(&mut collector);
    let mut mapper = PolyakMapper::<B> {
        active: collector.tensors,
        tau,
        _marker: PhantomData,
    };
    target.map(&mut mapper)
}
