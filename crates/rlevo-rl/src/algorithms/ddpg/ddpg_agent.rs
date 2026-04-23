//! Deep Deterministic Policy Gradient agent: actor, critic, and replay buffer
//! management.
//!
//! [`DdpgAgent`] owns a deterministic actor, a Q-critic, their Polyak-averaged
//! target twins, two independent Adam optimizers, a uniform FIFO replay
//! buffer, and a [`GaussianNoise`] exploration module. Call
//! [`DdpgAgent::act`] to sample actions, [`DdpgAgent::remember`] to push
//! transitions, and [`DdpgAgent::learn_step`] to run one gradient update.
//! Drive the full loop with [`crate::algorithms::ddpg::train`].

use std::collections::VecDeque;
use std::marker::PhantomData;

use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{ElementConversion, Tensor, TensorData};
use rand::Rng;
use rand::RngExt;

use rlevo_core::action::BoundedAction;
use rlevo_core::base::{Observation, TensorConvertible};
use rlevo_core::memory::ReplayBufferError;
use rlevo_core::metrics::{AgentStats, PerformanceRecord};

use crate::algorithms::ddpg::ddpg_config::DdpgTrainingConfig;
use crate::algorithms::ddpg::ddpg_model::{ContinuousQ, DeterministicPolicy};
use crate::algorithms::ddpg::exploration::GaussianNoise;
use crate::utils::compute_target_q_values;

/// Error variants returned by [`DdpgAgent`] operations.
#[derive(Debug, thiserror::Error)]
pub enum DdpgAgentError {
    /// A tensor-to-action or action-to-tensor conversion failed.
    #[error("Tensor conversion failed: {0}")]
    TensorConversionFailed(String),
    /// The sampled or requested action is outside the valid action space.
    #[error("Invalid action: {0}")]
    InvalidAction(String),
    /// A replay-buffer operation failed.
    #[error(transparent)]
    Buffer(#[from] ReplayBufferError),
    /// An I/O error occurred while saving or loading model weights.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Per-episode statistics emitted by the DDPG training loop.
///
/// Implements [`PerformanceRecord`] so it can be accumulated by
/// [`AgentStats`]; `score` is the episode reward and `duration` is the step
/// count.
#[derive(Debug, Clone, Copy)]
pub struct DdpgMetrics {
    /// Total reward collected during the episode.
    pub reward: f32,
    /// Number of environment steps taken.
    pub steps: usize,
    /// Most recent critic (Q-network) loss.
    pub critic_loss: f32,
    /// Most recent actor loss. `0.0` until the first policy update fires.
    pub actor_loss: f32,
    /// Mean predicted Q-value across the most recent learn step.
    pub q_mean: f32,
}

impl PerformanceRecord for DdpgMetrics {
    fn score(&self) -> f32 {
        self.reward
    }

    fn duration(&self) -> usize {
        self.steps
    }
}

/// Summary values returned by a single [`DdpgAgent::learn_step`].
#[derive(Debug, Clone, Copy)]
pub struct LearnOutcome {
    /// Mean-squared Bellman error on this batch.
    pub critic_loss: f32,
    /// Actor loss, or `None` on critic-only iterations.
    pub actor_loss: Option<f32>,
    /// Mean predicted Q-value across the batch.
    pub q_mean: f32,
}

#[derive(Debug, Clone)]
struct Transition<O: Clone> {
    obs: O,
    action: Vec<f32>,
    reward: f32,
    next_obs: O,
    done: bool,
}

/// Deep Deterministic Policy Gradient agent.
///
/// # Const generics
///
/// - `DO` — rank of a single observation tensor (`1` for vector observations
///   of shape `[features]`).
/// - `DB` — rank of a batched observation tensor (= `DO + 1`). Rust cannot
///   express `DO + 1` in generic position on stable, so both are supplied.
/// - `DA` — rank of a single action tensor (`1` for vector actions of shape
///   `[action_dim]`).
/// - `DAB` — rank of a batched action tensor (= `DA + 1`).
pub struct DdpgAgent<
    B,
    Actor,
    Critic,
    O,
    A,
    const DO: usize,
    const DB: usize,
    const DA: usize,
    const DAB: usize,
> where
    B: AutodiffBackend,
    Actor: DeterministicPolicy<B, DB, DAB>,
    Critic: ContinuousQ<B, DB, DAB>,
    O: Observation<DO> + TensorConvertible<DO, B> + TensorConvertible<DO, B::InnerBackend>,
    A: BoundedAction<DA>,
{
    actor: Option<Actor>,
    target_actor: Actor::InnerModule,
    critic: Option<Critic>,
    target_critic: Critic::InnerModule,
    actor_opt: OptimizerAdaptor<Adam, Actor, B>,
    critic_opt: OptimizerAdaptor<Adam, Critic, B>,
    buffer: VecDeque<Transition<O>>,
    exploration: GaussianNoise,
    low: [f32; DA],
    high: [f32; DA],
    config: DdpgTrainingConfig,
    device: B::Device,
    step: usize,
    critic_updates: usize,
    stats: AgentStats<DdpgMetrics>,
    last_actor_loss: f32,
    _action: PhantomData<A>,
}

impl<B, Actor, Critic, O, A, const DO: usize, const DB: usize, const DA: usize, const DAB: usize>
    std::fmt::Debug for DdpgAgent<B, Actor, Critic, O, A, DO, DB, DA, DAB>
where
    B: AutodiffBackend,
    Actor: DeterministicPolicy<B, DB, DAB>,
    Critic: ContinuousQ<B, DB, DAB>,
    O: Observation<DO> + TensorConvertible<DO, B> + TensorConvertible<DO, B::InnerBackend>,
    A: BoundedAction<DA>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DdpgAgent")
            .field("step", &self.step)
            .field("critic_updates", &self.critic_updates)
            .field("buffer_len", &self.buffer.len())
            .field("sigma", &self.exploration.sigma())
            .field("low", &self.low)
            .field("high", &self.high)
            .field("config", &self.config)
            .finish()
    }
}

impl<B, Actor, Critic, O, A, const DO: usize, const DB: usize, const DA: usize, const DAB: usize>
    DdpgAgent<B, Actor, Critic, O, A, DO, DB, DA, DAB>
where
    B: AutodiffBackend,
    Actor: DeterministicPolicy<B, DB, DAB>,
    Critic: ContinuousQ<B, DB, DAB>,
    O: Observation<DO> + TensorConvertible<DO, B> + TensorConvertible<DO, B::InnerBackend>,
    A: BoundedAction<DA>,
{
    /// Constructs a new agent from pre-built actor and critic networks.
    pub fn new(
        actor: Actor,
        critic: Critic,
        config: DdpgTrainingConfig,
        device: B::Device,
    ) -> Self {
        let target_actor = actor.valid();
        let target_critic = critic.valid();
        let adam = config.optimizer.clone();
        let (actor_opt, critic_opt) = match &config.clip_grad {
            Some(clip) => (
                adam.clone()
                    .with_grad_clipping(Some(clip.clone()))
                    .init::<B, Actor>(),
                adam.clone()
                    .with_grad_clipping(Some(clip.clone()))
                    .init::<B, Critic>(),
            ),
            None => (
                adam.clone().init::<B, Actor>(),
                adam.clone().init::<B, Critic>(),
            ),
        };
        let exploration = GaussianNoise::new(config.exploration_noise);
        let stats = AgentStats::<DdpgMetrics>::new(100);
        Self {
            actor: Some(actor),
            target_actor,
            critic: Some(critic),
            target_critic,
            actor_opt,
            critic_opt,
            buffer: VecDeque::with_capacity(config.buffer_capacity),
            exploration,
            low: A::low(),
            high: A::high(),
            config,
            device,
            step: 0,
            critic_updates: 0,
            stats,
            last_actor_loss: 0.0,
            _action: PhantomData,
        }
    }

    /// Current agent statistics.
    pub fn stats(&self) -> &AgentStats<DdpgMetrics> {
        &self.stats
    }

    /// Records one completed episode into the running statistics.
    pub fn record_episode(&mut self, metrics: DdpgMetrics) {
        self.stats.record(metrics);
    }

    /// Number of transitions currently stored.
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Global env-step count.
    pub fn step(&self) -> usize {
        self.step
    }

    fn actor_ref(&self) -> &Actor {
        self.actor
            .as_ref()
            .expect("actor not restored — earlier panic in learn_step?")
    }

    fn critic_ref(&self) -> &Critic {
        self.critic
            .as_ref()
            .expect("critic not restored — earlier panic in learn_step?")
    }

    /// Samples an action for the current observation.
    ///
    /// Before `learning_starts` steps, draws a uniform random action on
    /// `[low, high]`. Afterwards runs the actor, adds Gaussian noise, and
    /// clips to the action bounds. The unnoised policy mean is emitted when
    /// `training == false`.
    pub fn act<R: Rng + ?Sized>(&self, obs: &O, training: bool, rng: &mut R) -> A {
        if training && self.step < self.config.learning_starts {
            let sample: Vec<f32> = (0..A::DIM)
                .map(|i| rng.random_range(self.low[i]..=self.high[i]))
                .collect();
            return A::from_slice(&sample);
        }

        let obs_t: Tensor<B, DO> = obs.to_tensor(&self.device);
        let batched: Tensor<B, DB> = obs_t.unsqueeze::<DB>();
        let raw: Tensor<B, DAB> = self.actor_ref().forward(batched);
        let data = raw.into_data().convert::<f32>();
        let slice = data.as_slice::<f32>().expect("actor output is f32");
        let mean: Vec<f32> = slice.iter().take(A::DIM).copied().collect();
        let out = if training {
            self.exploration.apply(&mean, &self.low, &self.high, rng)
        } else {
            mean.iter()
                .enumerate()
                .map(|(i, v)| v.clamp(self.low[i], self.high[i]))
                .collect()
        };
        A::from_slice(&out)
    }

    /// Appends a transition to the replay buffer, evicting the oldest entry
    /// when the buffer is at capacity.
    pub fn remember(&mut self, obs: O, action: &A, reward: f32, next_obs: O, done: bool) {
        if self.buffer.len() >= self.config.buffer_capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(Transition {
            obs,
            action: action.as_slice().to_vec(),
            reward,
            next_obs,
            done,
        });
    }

    /// Advances the global env-step counter. Called once per env step.
    pub fn on_env_step(&mut self) {
        self.step += 1;
    }

    /// Returns `true` once the warm-up period has elapsed and the buffer has
    /// enough transitions to draw a batch.
    pub fn can_learn(&self) -> bool {
        self.buffer.len() >= self.config.batch_size && self.step >= self.config.learning_starts
    }

    /// Runs one learning step: a critic update and (every
    /// `policy_frequency`-th critic update) an actor + Polyak update.
    ///
    /// Returns `None` if the agent is still in warm-up.
    pub fn learn_step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Option<LearnOutcome> {
        if !self.can_learn() {
            return None;
        }
        let batch_size = self.config.batch_size;
        let device = self.device.clone();

        // --- Sample batch ---
        let indices: Vec<usize> = (0..batch_size)
            .map(|_| rng.random_range(0..self.buffer.len()))
            .collect();

        let obs_shape = O::shape();
        let obs_numel: usize = obs_shape.iter().product();
        let action_shape = A::shape();
        let action_numel: usize = action_shape.iter().product();

        let mut obs_flat: Vec<f32> = Vec::with_capacity(batch_size * obs_numel);
        let mut next_flat: Vec<f32> = Vec::with_capacity(batch_size * obs_numel);
        let mut action_flat: Vec<f32> = Vec::with_capacity(batch_size * action_numel);
        let mut rewards: Vec<f32> = Vec::with_capacity(batch_size);
        let mut dones: Vec<f32> = Vec::with_capacity(batch_size);

        for idx in &indices {
            let t = &self.buffer[*idx];
            let obs_tensor: Tensor<B::InnerBackend, DO> = t.obs.to_tensor(&device);
            let next_tensor: Tensor<B::InnerBackend, DO> = t.next_obs.to_tensor(&device);
            let obs_data = obs_tensor.into_data().convert::<f32>();
            let next_data = next_tensor.into_data().convert::<f32>();
            obs_flat.extend_from_slice(obs_data.as_slice::<f32>().expect("float data"));
            next_flat.extend_from_slice(next_data.as_slice::<f32>().expect("float data"));
            action_flat.extend_from_slice(&t.action);
            rewards.push(t.reward);
            dones.push(if t.done { 1.0 } else { 0.0 });
        }

        let mut batched_obs_shape: Vec<usize> = Vec::with_capacity(DB);
        batched_obs_shape.push(batch_size);
        batched_obs_shape.extend_from_slice(&obs_shape);
        let mut batched_action_shape: Vec<usize> = Vec::with_capacity(DAB);
        batched_action_shape.push(batch_size);
        batched_action_shape.extend_from_slice(&action_shape);

        let obs_t: Tensor<B, DB> = Tensor::from_data(
            TensorData::new(obs_flat, batched_obs_shape.clone()),
            &device,
        );
        let next_t_inner: Tensor<B::InnerBackend, DB> =
            Tensor::from_data(TensorData::new(next_flat, batched_obs_shape), &device);
        let action_t: Tensor<B, DAB> =
            Tensor::from_data(TensorData::new(action_flat, batched_action_shape), &device);

        let rewards_inner: Tensor<B::InnerBackend, 1> =
            Tensor::from_data(TensorData::new(rewards, vec![batch_size]), &device);
        let dones_inner: Tensor<B::InnerBackend, 1> =
            Tensor::from_data(TensorData::new(dones, vec![batch_size]), &device);

        // --- Target computation (no autodiff) ---
        // CleanRL uses low[0]/high[0] as a scalar clip on the target action;
        // adopt the same convention (documented in BoundedAction).
        let low_scalar = self.low[0];
        let high_scalar = self.high[0];
        let next_actions: Tensor<B::InnerBackend, DAB> =
            Actor::forward_inner(&self.target_actor, next_t_inner.clone())
                .clamp(low_scalar, high_scalar);
        let next_q: Tensor<B::InnerBackend, 1> =
            Critic::forward_inner(&self.target_critic, next_t_inner, next_actions);
        let target_inner: Tensor<B::InnerBackend, 1> =
            compute_target_q_values(rewards_inner, next_q, dones_inner, self.config.gamma);
        let target: Tensor<B, 1> = Tensor::from_data(target_inner.into_data(), &device);

        // --- Critic update ---
        let critic = self
            .critic
            .take()
            .expect("critic not restored — earlier panic in learn_step?");
        let q_pred: Tensor<B, 1> = critic.forward(obs_t.clone(), action_t);
        let q_mean = q_pred.clone().mean().into_scalar().elem::<f32>();
        let td_error = q_pred - target;
        let critic_loss_tensor = td_error.powi_scalar(2).mean();
        let critic_loss = critic_loss_tensor.clone().into_scalar().elem::<f32>();

        let grads = critic_loss_tensor.backward();
        let grads = GradientsParams::from_grads(grads, &critic);
        let critic = self.critic_opt.step(self.config.critic_lr, critic, grads);
        self.critic = Some(critic);
        self.critic_updates += 1;

        // --- Actor + Polyak update (every policy_frequency-th critic step) ---
        let mut actor_loss_opt: Option<f32> = None;
        if self
            .critic_updates
            .is_multiple_of(self.config.policy_frequency)
        {
            let actor = self
                .actor
                .take()
                .expect("actor not restored — earlier panic in learn_step?");
            let predicted_actions: Tensor<B, DAB> = actor.forward(obs_t.clone());
            let q_actor: Tensor<B, 1> = self.critic_ref().forward(obs_t, predicted_actions);
            let actor_loss_tensor = q_actor.mean().neg();
            let actor_loss_value = actor_loss_tensor.clone().into_scalar().elem::<f32>();

            let grads = actor_loss_tensor.backward();
            let actor_grads = GradientsParams::from_grads(grads, &actor);
            let actor = self
                .actor_opt
                .step(self.config.actor_lr, actor, actor_grads);

            let tau = self.config.tau as f64;
            let fresh_target_actor = actor.valid();
            let target_actor = std::mem::replace(&mut self.target_actor, fresh_target_actor);
            self.target_actor = Actor::soft_update(&actor, target_actor, tau);

            let fresh_target_critic = self.critic_ref().valid();
            let target_critic = std::mem::replace(&mut self.target_critic, fresh_target_critic);
            self.target_critic = Critic::soft_update(self.critic_ref(), target_critic, tau);

            self.actor = Some(actor);
            self.last_actor_loss = actor_loss_value;
            actor_loss_opt = Some(actor_loss_value);
        }

        Some(LearnOutcome {
            critic_loss,
            actor_loss: actor_loss_opt,
            q_mean,
        })
    }

    /// Most recent actor loss (persists between policy updates).
    pub fn last_actor_loss(&self) -> f32 {
        self.last_actor_loss
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_performance_record_returns_reward_and_steps() {
        let m = DdpgMetrics {
            reward: 3.5,
            steps: 42,
            critic_loss: 0.1,
            actor_loss: -0.2,
            q_mean: 1.0,
        };
        assert_eq!(m.score(), 3.5);
        assert_eq!(m.duration(), 42);
    }

    #[test]
    fn error_display_uses_thiserror_messages() {
        let err = DdpgAgentError::InvalidAction("bad slice".into());
        assert_eq!(err.to_string(), "Invalid action: bad slice");
    }
}
