//! Categorical DQN (C51) agent: actor, trainer, and replay buffer.
//!
//! The [`C51Agent`] struct owns the policy network (returning atom logits),
//! a frozen target network, an Adam optimizer, a uniform replay buffer, and
//! the ε-greedy exploration schedule shared with DQN. Action selection uses
//! the expected Q-value `Σ_i z_i · softmax(logits_i)`; the learning step
//! projects the bootstrap distribution onto the fixed atom support (see
//! [`crate::algorithms::c51::projection`]) and minimises the categorical
//! cross-entropy against the policy's log-probabilities.

use std::collections::VecDeque;
use std::marker::PhantomData;

use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, GradientsParams, Optimizer};
use burn::tensor::activation;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{ElementConversion, Int, Tensor, TensorData};
use rand::{Rng, RngExt};

use rlevo_core::action::DiscreteAction;
use rlevo_core::base::{Observation, TensorConvertible};
use crate::memory::ReplayBufferError;
use crate::metrics::{AgentStats, PerformanceRecord};

use crate::algorithms::c51::c51_config::C51TrainingConfig;
use crate::algorithms::c51::c51_model::C51Model;
use crate::algorithms::c51::loss::categorical_cross_entropy;
use crate::algorithms::c51::projection::project_distribution;
use crate::algorithms::dqn::exploration::EpsilonGreedy;

/// Error variants returned by [`C51Agent`] operations.
#[derive(Debug, thiserror::Error)]
pub enum C51AgentError {
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

/// Per-episode statistics emitted by the C51 training loop.
#[derive(Debug, Clone, Copy)]
pub struct C51Metrics {
    /// Total reward collected during the episode.
    pub reward: f32,
    /// Number of environment steps taken.
    pub steps: usize,
    /// Most recent categorical cross-entropy loss.
    pub policy_loss: f32,
    /// Exploration rate at the end of the episode.
    pub epsilon: f32,
    /// Mean expected Q-value `E[Z]` across the most recent learn step.
    pub q_mean: f32,
    /// Mean entropy of the predicted return distribution at the taken
    /// action (natural-log units). Low values flag distribution collapse.
    pub entropy: f32,
}

impl PerformanceRecord for C51Metrics {
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

/// Summary values returned by a single [`C51Agent::learn_step`].
#[derive(Debug, Clone, Copy)]
pub struct LearnOutcome {
    /// Categorical cross-entropy loss.
    pub loss: f32,
    /// Mean expected Q-value `E[Z]` across the batch, for diagnostics.
    pub q_mean: f32,
    /// Mean entropy of the predicted return distribution.
    pub entropy: f32,
}

/// Categorical DQN (C51) agent.
///
/// # Const generics
///
/// - `DO` — rank of a single observation tensor (e.g. `1` for vector
///   observations of shape `[features]`).
/// - `DB` — rank of a batched observation tensor (= `DO + 1`).
pub struct C51Agent<B, M, O, A, const DO: usize, const DB: usize>
where
    B: AutodiffBackend,
    M: C51Model<B, DB>,
    O: Observation<DO> + TensorConvertible<DO, B> + TensorConvertible<DO, B::InnerBackend>,
    A: DiscreteAction<1>,
{
    policy_net: Option<M>,
    target_net: M::InnerModule,
    optimizer: OptimizerAdaptor<Adam, M, B>,
    buffer: VecDeque<Transition<O>>,
    exploration: EpsilonGreedy,
    config: C51TrainingConfig,
    device: B::Device,
    step: usize,
    stats: AgentStats<C51Metrics>,
    _action: PhantomData<A>,
}

impl<B, M, O, A, const DO: usize, const DB: usize> std::fmt::Debug for C51Agent<B, M, O, A, DO, DB>
where
    B: AutodiffBackend,
    M: C51Model<B, DB>,
    O: Observation<DO> + TensorConvertible<DO, B> + TensorConvertible<DO, B::InnerBackend>,
    A: DiscreteAction<1>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("C51Agent")
            .field("step", &self.step)
            .field("buffer_len", &self.buffer.len())
            .field("epsilon", &self.exploration.value())
            .field("config", &self.config)
            .finish()
    }
}

impl<B, M, O, A, const DO: usize, const DB: usize> C51Agent<B, M, O, A, DO, DB>
where
    B: AutodiffBackend,
    M: C51Model<B, DB>,
    O: Observation<DO> + TensorConvertible<DO, B> + TensorConvertible<DO, B::InnerBackend>,
    A: DiscreteAction<1>,
{
    /// Constructs a new agent from a pre-built policy network and config.
    pub fn new(policy_net: M, config: C51TrainingConfig, device: B::Device) -> Self {
        let target_net = policy_net.valid();
        let adam = config.optimizer.clone();
        let optimizer = match &config.clip_grad {
            Some(clip) => adam.with_grad_clipping(Some(clip.clone())).init::<B, M>(),
            None => adam.init::<B, M>(),
        };
        // ε-greedy is reused verbatim from DQN — construct it from the
        // shared epsilon fields.
        let exploration = EpsilonGreedy::new(
            config.epsilon_start,
            config.epsilon_end,
            config.epsilon_decay,
        );
        let stats = AgentStats::<C51Metrics>::new(100);
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
    pub fn stats(&self) -> &AgentStats<C51Metrics> {
        &self.stats
    }

    /// Records one completed episode into the running statistics.
    pub fn record_episode(&mut self, metrics: C51Metrics) {
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
            .expect("policy_net not restored — earlier panic in learn_step?")
    }

    /// Builds a fresh support tensor `[z_0, z_1, …, z_{N-1}]` on the
    /// requested backend and device. Small (N=51 default), so we prefer
    /// allocating on demand over storing a cached copy per backend flavour.
    fn build_support<BK: burn::tensor::backend::Backend>(
        &self,
        device: &BK::Device,
    ) -> Tensor<BK, 1> {
        let n = self.config.num_atoms;
        let delta = self.config.delta_z();
        let data: Vec<f32> = (0..n)
            .map(|i| self.config.v_min + (i as f32) * delta)
            .collect();
        Tensor::from_data(TensorData::new(data, vec![n]), device)
    }

    /// ε-greedy action selection using the expected value of the predicted
    /// return distribution.
    pub fn act<R: Rng + ?Sized>(&self, obs: &O, rng: &mut R) -> A {
        if self.exploration.should_explore(rng) {
            return A::from_index(rng.random_range(0..A::ACTION_COUNT));
        }
        let obs_t: Tensor<B, DO> = obs.to_tensor(&self.device);
        let batched: Tensor<B, DB> = obs_t.unsqueeze::<DB>();
        let logits: Tensor<B, 3> = self.policy().forward(batched); // (1, A, N)
        let probs: Tensor<B, 3> = activation::softmax(logits, 2);
        let support: Tensor<B, 1> = self.build_support::<B>(&self.device);
        let support_3d: Tensor<B, 3> = support.unsqueeze::<3>(); // (1, 1, N)
        let q: Tensor<B, 2> = (probs * support_3d).sum_dim(2).squeeze_dim::<2>(2); // (1, A)
        let idx = q.argmax(1).into_scalar();
        A::from_index(idx.elem::<i64>() as usize)
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
        self.config.train_frequency > 0 && self.step.is_multiple_of(self.config.train_frequency)
    }

    /// Hard-syncs the target network with the policy network when
    /// `self.step` is a multiple of `config.target_update_frequency`.
    pub fn sync_target(&mut self) {
        if self.config.target_update_frequency == 0 {
            return;
        }
        if self.step > 0
            && self
                .step
                .is_multiple_of(self.config.target_update_frequency)
        {
            self.target_net = self.policy().valid();
        }
    }

    /// Runs one learning step: samples a batch uniformly, projects the
    /// target distribution, and minimises the categorical cross-entropy
    /// between it and the policy's log-probabilities.
    pub fn learn_step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Option<LearnOutcome> {
        if !self.can_learn() {
            return None;
        }
        let batch_size = self.config.batch_size;
        let num_atoms = self.config.num_atoms;

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

        let action_tensor_1: Tensor<B, 1, Int> =
            Tensor::from_data(TensorData::new(action_idxs, vec![batch_size]), &device);

        let rewards_inner: Tensor<B::InnerBackend, 1> =
            Tensor::from_data(TensorData::new(rewards, vec![batch_size]), &device);
        let dones_inner: Tensor<B::InnerBackend, 1> =
            Tensor::from_data(TensorData::new(dones, vec![batch_size]), &device);

        // --- Target (inner backend, no autodiff) ---
        let support_inner: Tensor<B::InnerBackend, 1> =
            self.build_support::<B::InnerBackend>(&device);
        let next_logits_inner: Tensor<B::InnerBackend, 3> =
            M::forward_inner(&self.target_net, next_tensor_inner);
        let next_probs_all: Tensor<B::InnerBackend, 3> = activation::softmax(next_logits_inner, 2);

        let support_3d_inner: Tensor<B::InnerBackend, 3> = support_inner.clone().unsqueeze::<3>();
        let q_next: Tensor<B::InnerBackend, 2> = (next_probs_all.clone() * support_3d_inner)
            .sum_dim(2)
            .squeeze_dim::<2>(2); // (B, A)
        let a_star_2: Tensor<B::InnerBackend, 2, Int> = q_next.argmax(1); // (B, 1)
        let a_star_3: Tensor<B::InnerBackend, 3, Int> =
            a_star_2.unsqueeze_dim::<3>(2).repeat_dim(2, num_atoms); // (B, 1, N)
        let next_probs_chosen: Tensor<B::InnerBackend, 2> =
            next_probs_all.gather(1, a_star_3).squeeze_dim::<2>(1); // (B, N)

        let target_inner: Tensor<B::InnerBackend, 2> = project_distribution(
            next_probs_chosen,
            rewards_inner,
            dones_inner,
            support_inner,
            self.config.gamma as f32,
            self.config.v_min,
            self.config.v_max,
            num_atoms,
        );

        // --- Policy forward (autodiff) ---
        let policy = self
            .policy_net
            .take()
            .expect("policy_net not restored — earlier panic in learn_step?");
        let logits: Tensor<B, 3> = policy.forward(obs_tensor); // (B, A, N)

        let probs_all: Tensor<B, 3> = activation::softmax(logits.clone(), 2);
        let support_auto: Tensor<B, 1> = self.build_support::<B>(&device);
        let support_3d_auto: Tensor<B, 3> = support_auto.unsqueeze::<3>();
        let q_all: Tensor<B, 2> = (probs_all.clone() * support_3d_auto)
            .sum_dim(2)
            .squeeze_dim::<2>(2); // (B, A)
        let q_mean = q_all.mean().into_scalar().elem::<f32>();

        // Gather the atom logits for the taken action → (B, N).
        let action_idx_3d: Tensor<B, 3, Int> = action_tensor_1
            .unsqueeze_dim::<2>(1)
            .unsqueeze_dim::<3>(2)
            .repeat_dim(2, num_atoms); // (B, 1, N)
        let pred_logits: Tensor<B, 2> = logits.gather(1, action_idx_3d).squeeze_dim::<2>(1);
        let pred_log_p: Tensor<B, 2> = activation::log_softmax(pred_logits, 1);

        // Diagnostic: mean entropy of the predicted distribution at the
        // taken action. Uses `probs * log_probs` computed with the same
        // `log_softmax` output for numerical stability.
        let pred_probs: Tensor<B, 2> = pred_log_p.clone().exp();
        let entropy: Tensor<B, 1> = (pred_probs * pred_log_p.clone())
            .sum_dim(1)
            .squeeze_dim::<1>(1)
            .neg()
            .mean();
        let entropy_value = entropy.into_scalar().elem::<f32>();

        // Upload the target distribution onto the autodiff backend and
        // compute the cross-entropy loss.
        let target_autodiff: Tensor<B, 2> = Tensor::from_data(target_inner.into_data(), &device);
        let loss_tensor = categorical_cross_entropy(target_autodiff, pred_log_p);
        let loss_value = loss_tensor.clone().into_scalar().elem::<f32>();

        let grads = loss_tensor.backward();
        let grads = GradientsParams::from_grads(grads, &policy);
        let updated = self
            .optimizer
            .step(self.config.learning_rate, policy, grads);
        self.policy_net = Some(updated);

        if self.config.tau > 0.0 {
            let fresh_valid = self.policy().valid();
            let target = std::mem::replace(&mut self.target_net, fresh_valid);
            self.target_net = M::soft_update(self.policy(), target, self.config.tau);
        }

        Some(LearnOutcome {
            loss: loss_value,
            q_mean,
            entropy: entropy_value,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_performance_record_returns_reward_and_steps() {
        let m = C51Metrics {
            reward: 42.0,
            steps: 7,
            policy_loss: 0.5,
            epsilon: 0.1,
            q_mean: 1.0,
            entropy: 2.5,
        };
        assert_eq!(m.score(), 42.0);
        assert_eq!(m.duration(), 7);
    }

    #[test]
    fn error_display_uses_thiserror_messages() {
        let err = C51AgentError::InvalidAction("bad index".into());
        assert_eq!(err.to_string(), "Invalid action: bad index");
    }
}
