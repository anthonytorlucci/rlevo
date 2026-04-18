//! Deep Q-Network agent: actor, trainer, and replay buffer management.
//!
//! The [`DqnAgent`] struct owns the policy network, frozen target network,
//! optimizer, a uniform replay buffer, and the ε-greedy exploration schedule.
//! Call [`DqnAgent::act`] to sample actions, [`DqnAgent::remember`] to push
//! transitions, and [`DqnAgent::learn_step`] to perform one gradient update.
//! The end-to-end training loop is assembled in
//! [`crate::algorithms::dqn::train`].

use std::collections::VecDeque;
use std::marker::PhantomData;

use burn::nn::loss::{HuberLossConfig, Reduction};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{ElementConversion, Int, Tensor, TensorData};
use rand::{Rng, RngExt};

use evorl_core::action::DiscreteAction;
use evorl_core::base::{Observation, TensorConvertible};
use evorl_core::memory::ReplayBufferError;
use evorl_core::metrics::{AgentStats, PerformanceRecord};

use crate::algorithms::dqn::dqn_config::DqnTrainingConfig;
use crate::algorithms::dqn::dqn_model::DqnModel;
use crate::algorithms::dqn::exploration::EpsilonGreedy;
use crate::utils::compute_target_q_values;

/// Error variants returned by [`DqnAgent`] operations.
#[derive(Debug, thiserror::Error)]
pub enum DqnAgentError {
    /// A tensor-to-action or action-to-tensor conversion failed.
    #[error("Tensor conversion failed: {0}")]
    TensorConversionFailed(String),
    /// The sampled or requested action is outside the valid action space.
    #[error("Invalid action: {0}")]
    InvalidAction(String),
    /// A replay buffer operation failed.
    #[error(transparent)]
    Buffer(#[from] ReplayBufferError),
    /// An I/O error occurred while saving or loading model weights.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Per-episode statistics emitted by the DQN training loop.
///
/// Implements [`PerformanceRecord`] so it can be accumulated by
/// [`AgentStats`]; `score` returns the episode reward and `duration`
/// returns the step count.
#[derive(Debug, Clone, Copy)]
pub struct DqnMetrics {
    /// Total reward collected during the episode.
    pub reward: f32,
    /// Number of environment steps taken.
    pub steps: usize,
    /// Most recent policy (Q-network) loss value.
    pub policy_loss: f32,
    /// Mirror of `policy_loss` kept for parity with actor-critic algorithms.
    pub value_loss: f32,
    /// Exploration rate at the end of the episode.
    pub epsilon: f32,
    /// Mean predicted Q-value across the most recent learn step.
    pub q_mean: f32,
}

impl PerformanceRecord for DqnMetrics {
    fn score(&self) -> f32 {
        self.reward
    }

    fn duration(&self) -> usize {
        self.steps
    }
}

#[derive(Debug, Clone)]
struct Transition<O: Clone> {
    obs: O,
    action_idx: usize,
    reward: f32,
    next_obs: O,
    done: bool,
}

/// Summary values returned by a single [`DqnAgent::learn_step`].
#[derive(Debug, Clone, Copy)]
pub struct LearnOutcome {
    /// Huber loss between predicted and target Q-values.
    pub loss: f32,
    /// Mean predicted Q-value across the batch, for diagnostics.
    pub q_mean: f32,
}

/// Deep Q-Network agent.
///
/// # Const generics
///
/// - `DO` — rank of a single observation tensor (e.g. `1` for vector
///   observations of shape `[features]`).
/// - `DB` — rank of a batched observation tensor (= `DO + 1`, e.g. `2` for
///   `[batch, features]`). Rust cannot express `DO + 1` in generic position
///   on stable, so both are supplied by the caller.
pub struct DqnAgent<B, M, O, A, const DO: usize, const DB: usize>
where
    B: AutodiffBackend,
    M: DqnModel<B, DB>,
    O: Observation<DO> + TensorConvertible<DO, B> + TensorConvertible<DO, B::InnerBackend>,
    A: DiscreteAction<1>,
{
    policy_net: Option<M>,
    target_net: M::InnerModule,
    optimizer: OptimizerAdaptor<Adam, M, B>,
    buffer: VecDeque<Transition<O>>,
    exploration: EpsilonGreedy,
    config: DqnTrainingConfig,
    device: B::Device,
    step: usize,
    stats: AgentStats<DqnMetrics>,
    _action: PhantomData<A>,
}

impl<B, M, O, A, const DO: usize, const DB: usize> std::fmt::Debug
    for DqnAgent<B, M, O, A, DO, DB>
where
    B: AutodiffBackend,
    M: DqnModel<B, DB>,
    O: Observation<DO> + TensorConvertible<DO, B> + TensorConvertible<DO, B::InnerBackend>,
    A: DiscreteAction<1>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DqnAgent")
            .field("step", &self.step)
            .field("buffer_len", &self.buffer.len())
            .field("epsilon", &self.exploration.value())
            .field("config", &self.config)
            .finish()
    }
}

impl<B, M, O, A, const DO: usize, const DB: usize> DqnAgent<B, M, O, A, DO, DB>
where
    B: AutodiffBackend,
    M: DqnModel<B, DB>,
    O: Observation<DO> + TensorConvertible<DO, B> + TensorConvertible<DO, B::InnerBackend>,
    A: DiscreteAction<1>,
{
    /// Constructs a new agent from a pre-built policy network and training config.
    pub fn new(policy_net: M, config: DqnTrainingConfig, device: B::Device) -> Self {
        let target_net = policy_net.valid();
        let adam = config.optimizer.clone();
        let optimizer = match &config.clip_grad {
            Some(clip) => adam.with_grad_clipping(Some(clip.clone())).init::<B, M>(),
            None => adam.init::<B, M>(),
        };
        let exploration = EpsilonGreedy::from_config(&config);
        let stats = AgentStats::<DqnMetrics>::new(100);
        Self {
            policy_net: Some(policy_net),
            target_net,
            optimizer,
            buffer: VecDeque::with_capacity(config.replay_buffer_capacity),
            exploration,
            config,
            device,
            step: 0,
            stats,
            _action: PhantomData,
        }
    }

    /// Current exploration rate (ε).
    pub fn epsilon(&self) -> f64 {
        self.exploration.value()
    }

    /// Current agent statistics.
    pub fn stats(&self) -> &AgentStats<DqnMetrics> {
        &self.stats
    }

    /// Records one completed episode into the running statistics.
    pub fn record_episode(&mut self, metrics: DqnMetrics) {
        self.stats.record(metrics);
    }

    /// Number of transitions currently stored.
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Global step count.
    pub fn step(&self) -> usize {
        self.step
    }

    fn policy(&self) -> &M {
        self.policy_net
            .as_ref()
            .expect("policy_net is always populated except transiently during learn_step")
    }

    /// ε-greedy action selection.
    ///
    /// With probability `ε` returns a uniformly random discrete action;
    /// otherwise runs the policy network on `obs` and returns the argmax.
    pub fn act<R: Rng + ?Sized>(&self, obs: &O, rng: &mut R) -> A {
        if self.exploration.should_explore(rng) {
            A::from_index(rng.random_range(0..A::ACTION_COUNT))
        } else {
            let obs_t: Tensor<B, DO> = obs.to_tensor(&self.device);
            let batched: Tensor<B, DB> = obs_t.unsqueeze::<DB>();
            let q_values: Tensor<B, 2> = self.policy().forward(batched);
            let idx = q_values.argmax(1).into_scalar();
            A::from_index(idx.elem::<i64>() as usize)
        }
    }

    /// Appends a transition to the replay buffer, evicting the oldest entry
    /// when the buffer is at capacity.
    pub fn remember(&mut self, obs: O, action: &A, reward: f32, next_obs: O, done: bool) {
        if self.buffer.len() >= self.config.replay_buffer_capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(Transition {
            obs,
            action_idx: action.to_index(),
            reward,
            next_obs,
            done,
        });
    }

    /// Decays ε by one step.
    pub fn decay_exploration(&mut self) {
        self.exploration.decay();
    }

    /// Advances the global step counter. Called once per env step.
    pub fn on_env_step(&mut self) {
        self.step += 1;
    }

    /// Returns `true` when the agent has enough transitions to run a learn step.
    pub fn can_learn(&self) -> bool {
        self.buffer.len() >= self.config.batch_size && self.step >= self.config.learning_starts
    }

    /// Returns `true` when the agent's internal clock matches
    /// `config.train_frequency`.
    pub fn should_train(&self) -> bool {
        self.config.train_frequency > 0 && self.step % self.config.train_frequency == 0
    }

    /// Hard-syncs the target network with the policy network when
    /// `self.step` is a multiple of `config.target_update_frequency`.
    pub fn sync_target(&mut self) {
        if self.config.target_update_frequency == 0 {
            return;
        }
        if self.step > 0 && self.step % self.config.target_update_frequency == 0 {
            self.target_net = self.policy().valid();
        }
    }

    /// Runs one learning step: samples a batch uniformly, computes the Huber
    /// loss against the Bellman target, and updates the policy network.
    ///
    /// Returns `None` if the agent does not yet have enough transitions to
    /// form a batch.
    pub fn learn_step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Option<LearnOutcome> {
        if !self.can_learn() {
            return None;
        }
        let batch_size = self.config.batch_size;
        let indices: Vec<usize> = (0..batch_size)
            .map(|_| rng.random_range(0..self.buffer.len()))
            .collect();

        let obs_shape = O::shape();
        let numel_per_obs: usize = obs_shape.iter().product();

        let mut obs_flat: Vec<f32> = Vec::with_capacity(batch_size * numel_per_obs);
        let mut next_flat: Vec<f32> = Vec::with_capacity(batch_size * numel_per_obs);
        let mut action_idxs: Vec<i64> = Vec::with_capacity(batch_size);
        let mut rewards: Vec<f32> = Vec::with_capacity(batch_size);
        let mut dones: Vec<f32> = Vec::with_capacity(batch_size);

        for idx in &indices {
            let t = &self.buffer[*idx];
            let obs_tensor: Tensor<B::InnerBackend, DO> = t.obs.to_tensor(&self.device);
            let next_tensor: Tensor<B::InnerBackend, DO> = t.next_obs.to_tensor(&self.device);
            let obs_data = obs_tensor.into_data().convert::<f32>();
            let next_data = next_tensor.into_data().convert::<f32>();
            obs_flat.extend_from_slice(obs_data.as_slice::<f32>().expect("float data"));
            next_flat.extend_from_slice(next_data.as_slice::<f32>().expect("float data"));
            action_idxs.push(t.action_idx as i64);
            rewards.push(t.reward);
            dones.push(if t.done { 1.0 } else { 0.0 });
        }

        let mut batched_shape: Vec<usize> = Vec::with_capacity(DB);
        batched_shape.push(batch_size);
        batched_shape.extend_from_slice(&obs_shape);

        let device = self.device.clone();
        let obs_tensor: Tensor<B, DB> =
            Tensor::from_data(TensorData::new(obs_flat, batched_shape.clone()), &device);
        let next_tensor_inner: Tensor<B::InnerBackend, DB> =
            Tensor::from_data(TensorData::new(next_flat, batched_shape), &device);

        let action_tensor_1: Tensor<B, 1, Int> = Tensor::from_data(
            TensorData::new(action_idxs, vec![batch_size]),
            &device,
        );
        let action_tensor: Tensor<B, 2, Int> = action_tensor_1.unsqueeze_dim::<2>(1);

        let rewards_inner: Tensor<B::InnerBackend, 1> =
            Tensor::from_data(TensorData::new(rewards, vec![batch_size]), &device);
        let dones_inner: Tensor<B::InnerBackend, 1> =
            Tensor::from_data(TensorData::new(dones, vec![batch_size]), &device);

        // --- Forward ---
        let policy = self.policy_net.take().expect("policy_net already taken");
        let q_all: Tensor<B, 2> = policy.forward(obs_tensor);
        let q_mean = q_all.clone().mean().into_scalar().elem::<f32>();
        let q_pred: Tensor<B, 2> = q_all.gather(1, action_tensor);
        let q_pred_flat: Tensor<B, 1> = q_pred.squeeze_dim::<1>(1);

        // --- Target ---
        let next_q_target_inner: Tensor<B::InnerBackend, 2> =
            M::forward_inner(&self.target_net, next_tensor_inner.clone());
        let next_q_max_inner: Tensor<B::InnerBackend, 1> = if self.config.double_q {
            let next_q_policy_inner: Tensor<B::InnerBackend, 2> =
                M::forward_inner(&policy.valid(), next_tensor_inner);
            let next_actions: Tensor<B::InnerBackend, 2, Int> = next_q_policy_inner.argmax(1);
            next_q_target_inner.gather(1, next_actions).squeeze_dim::<1>(1)
        } else {
            next_q_target_inner.max_dim(1).squeeze_dim::<1>(1)
        };

        let target_inner: Tensor<B::InnerBackend, 1> = compute_target_q_values(
            rewards_inner,
            next_q_max_inner,
            dones_inner,
            self.config.gamma as f32,
        );
        let target: Tensor<B, 1> =
            Tensor::from_data(target_inner.into_data(), &device);

        let loss_tensor = HuberLossConfig::new(1.0)
            .init()
            .forward(q_pred_flat, target, Reduction::Mean);
        let loss_value = loss_tensor.clone().into_scalar().elem::<f32>();

        let grads = loss_tensor.backward();
        let grads = GradientsParams::from_grads(grads, &policy);
        let updated = self.optimizer.step(self.config.learning_rate, policy, grads);
        self.policy_net = Some(updated);

        // Soft update when tau > 0; otherwise rely on hard sync_target().
        if self.config.tau > 0.0 {
            let fresh_valid = self.policy().valid();
            let target = std::mem::replace(&mut self.target_net, fresh_valid);
            self.target_net = M::soft_update(self.policy(), target, self.config.tau);
        }

        Some(LearnOutcome {
            loss: loss_value,
            q_mean,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_performance_record_returns_reward_and_steps() {
        let m = DqnMetrics {
            reward: 42.0,
            steps: 7,
            policy_loss: 0.5,
            value_loss: 0.5,
            epsilon: 0.1,
            q_mean: 1.0,
        };
        assert_eq!(m.score(), 42.0);
        assert_eq!(m.duration(), 7);
    }

    #[test]
    fn error_display_uses_thiserror_messages() {
        let err = DqnAgentError::InvalidAction("bad index".into());
        assert_eq!(err.to_string(), "Invalid action: bad index");
    }
}
