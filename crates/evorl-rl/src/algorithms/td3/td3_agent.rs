//! Twin Delayed DDPG (TD3) agent: actor, twin critics, and replay buffer
//! management.
//!
//! [`Td3Agent`] pairs a single deterministic actor with **two** Q-critics
//! (each with its own Polyak-averaged target), a uniform FIFO replay buffer,
//! and a Gaussian exploration module shared with DDPG. The three deltas
//! relative to [`crate::algorithms::ddpg::ddpg_agent::DdpgAgent`] are
//! (1) twin critics with a `min`-of-targets Bellman backup,
//! (2) Gaussian target-policy smoothing via
//! [`super::target_smoothing::smoothed_target_action`], and
//! (3) delayed actor + Polyak updates every `policy_frequency`-th critic
//! step.
//!
//! Drive the full loop with [`super::train::train`].

use std::collections::VecDeque;
use std::marker::PhantomData;

use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{ElementConversion, Tensor, TensorData};
use rand::Rng;
use rand::RngExt;

use evorl_core::action::BoundedAction;
use evorl_core::base::{Observation, TensorConvertible};
use evorl_core::memory::ReplayBufferError;
use evorl_core::metrics::{AgentStats, PerformanceRecord};

use crate::algorithms::ddpg::exploration::GaussianNoise;
use crate::algorithms::td3::target_smoothing::smoothed_target_action;
use crate::algorithms::td3::td3_config::Td3TrainingConfig;
use crate::algorithms::td3::td3_model::{ContinuousQ, DeterministicPolicy};
use crate::utils::compute_target_q_values;

/// Error variants returned by [`Td3Agent`] operations.
#[derive(Debug, thiserror::Error)]
pub enum Td3AgentError {
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

/// Per-episode statistics emitted by the TD3 training loop.
///
/// Mirrors [`crate::algorithms::ddpg::ddpg_agent::DdpgMetrics`] so dashboards
/// and A/B tooling can handle DDPG and TD3 runs interchangeably. `critic_loss`
/// is the sum across the two critics; `q_mean` is the mean of the
/// element-wise `min(q1, q2)` used by the learn step.
#[derive(Debug, Clone, Copy)]
pub struct Td3Metrics {
    /// Total reward collected during the episode.
    pub reward: f32,
    /// Number of environment steps taken.
    pub steps: usize,
    /// Sum of the two critics' mean-squared Bellman errors on the most
    /// recent learn step.
    pub critic_loss: f32,
    /// Most recent actor loss. `0.0` until the first policy update fires.
    pub actor_loss: f32,
    /// Mean of `min(q1, q2)` across the most recent learn step.
    pub q_mean: f32,
}

impl PerformanceRecord for Td3Metrics {
    fn score(&self) -> f32 {
        self.reward
    }

    fn duration(&self) -> usize {
        self.steps
    }
}

/// Summary values returned by a single [`Td3Agent::learn_step`].
#[derive(Debug, Clone, Copy)]
pub struct LearnOutcome {
    /// Sum of the two critics' mean-squared Bellman errors on this batch.
    pub critic_loss: f32,
    /// Actor loss, or `None` on critic-only iterations (delayed-update
    /// skips).
    pub actor_loss: Option<f32>,
    /// Mean of `min(q1, q2)` across the batch.
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

/// Computes the twin-critic TD target:
/// `y = r + γ · (1 − done) · min(next_q1, next_q2)`.
///
/// Exposed at crate visibility so the unit tests can cover the core TD3
/// invariant without standing up a full agent.
pub(crate) fn compute_twin_critic_target<BI: Backend>(
    rewards: Tensor<BI, 1>,
    next_q1: Tensor<BI, 1>,
    next_q2: Tensor<BI, 1>,
    dones: Tensor<BI, 1>,
    gamma: f32,
) -> Tensor<BI, 1> {
    let min_q = next_q1.min_pair(next_q2);
    compute_target_q_values(rewards, min_q, dones, gamma)
}

/// Twin Delayed DDPG (TD3) agent.
///
/// # Const generics
///
/// Same layout as [`DdpgAgent`](crate::algorithms::ddpg::ddpg_agent::DdpgAgent):
/// - `DO` — rank of a single observation tensor.
/// - `DB` — rank of a batched observation tensor (= `DO + 1`).
/// - `DA` — rank of a single action tensor.
/// - `DAB` — rank of a batched action tensor (= `DA + 1`).
pub struct Td3Agent<
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
    O: Observation<DO>
        + TensorConvertible<DO, B>
        + TensorConvertible<DO, B::InnerBackend>,
    A: BoundedAction<DA>,
{
    actor: Option<Actor>,
    target_actor: Actor::InnerModule,
    critic_1: Option<Critic>,
    critic_2: Option<Critic>,
    target_critic_1: Critic::InnerModule,
    target_critic_2: Critic::InnerModule,
    actor_opt: OptimizerAdaptor<Adam, Actor, B>,
    critic_1_opt: OptimizerAdaptor<Adam, Critic, B>,
    critic_2_opt: OptimizerAdaptor<Adam, Critic, B>,
    buffer: VecDeque<Transition<O>>,
    exploration: GaussianNoise,
    low: [f32; DA],
    high: [f32; DA],
    config: Td3TrainingConfig,
    device: B::Device,
    step: usize,
    critic_updates: usize,
    stats: AgentStats<Td3Metrics>,
    last_actor_loss: f32,
    _action: PhantomData<A>,
}

impl<
    B,
    Actor,
    Critic,
    O,
    A,
    const DO: usize,
    const DB: usize,
    const DA: usize,
    const DAB: usize,
> std::fmt::Debug for Td3Agent<B, Actor, Critic, O, A, DO, DB, DA, DAB>
where
    B: AutodiffBackend,
    Actor: DeterministicPolicy<B, DB, DAB>,
    Critic: ContinuousQ<B, DB, DAB>,
    O: Observation<DO>
        + TensorConvertible<DO, B>
        + TensorConvertible<DO, B::InnerBackend>,
    A: BoundedAction<DA>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Td3Agent")
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

impl<
    B,
    Actor,
    Critic,
    O,
    A,
    const DO: usize,
    const DB: usize,
    const DA: usize,
    const DAB: usize,
> Td3Agent<B, Actor, Critic, O, A, DO, DB, DA, DAB>
where
    B: AutodiffBackend,
    Actor: DeterministicPolicy<B, DB, DAB>,
    Critic: ContinuousQ<B, DB, DAB>,
    O: Observation<DO>
        + TensorConvertible<DO, B>
        + TensorConvertible<DO, B::InnerBackend>,
    A: BoundedAction<DA>,
{
    /// Constructs a new agent from pre-built actor and two independent
    /// critic networks.
    ///
    /// The caller is expected to initialise `critic_1` and `critic_2` with
    /// different random seeds — Fujimoto et al. rely on independent initial
    /// errors so the `min` target actually suppresses overestimation.
    pub fn new(
        actor: Actor,
        critic_1: Critic,
        critic_2: Critic,
        config: Td3TrainingConfig,
        device: B::Device,
    ) -> Self {
        let target_actor = actor.valid();
        let target_critic_1 = critic_1.valid();
        let target_critic_2 = critic_2.valid();
        let adam = config.optimizer.clone();
        let (actor_opt, critic_1_opt, critic_2_opt) = match &config.clip_grad {
            Some(clip) => (
                adam.clone()
                    .with_grad_clipping(Some(clip.clone()))
                    .init::<B, Actor>(),
                adam.clone()
                    .with_grad_clipping(Some(clip.clone()))
                    .init::<B, Critic>(),
                adam.clone()
                    .with_grad_clipping(Some(clip.clone()))
                    .init::<B, Critic>(),
            ),
            None => (
                adam.clone().init::<B, Actor>(),
                adam.clone().init::<B, Critic>(),
                adam.clone().init::<B, Critic>(),
            ),
        };
        let exploration = GaussianNoise::new(config.exploration_noise);
        let stats = AgentStats::<Td3Metrics>::new(100);
        Self {
            actor: Some(actor),
            target_actor,
            critic_1: Some(critic_1),
            critic_2: Some(critic_2),
            target_critic_1,
            target_critic_2,
            actor_opt,
            critic_1_opt,
            critic_2_opt,
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
    pub fn stats(&self) -> &AgentStats<Td3Metrics> {
        &self.stats
    }

    /// Records one completed episode into the running statistics.
    pub fn record_episode(&mut self, metrics: Td3Metrics) {
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

    /// Number of critic updates applied so far. Exposed for tests that need
    /// to verify the delayed-update schedule.
    pub fn critic_updates(&self) -> usize {
        self.critic_updates
    }

    fn actor_ref(&self) -> &Actor {
        self.actor
            .as_ref()
            .expect("actor is populated except transiently during learn_step")
    }

    fn critic_1_ref(&self) -> &Critic {
        self.critic_1
            .as_ref()
            .expect("critic_1 is populated except transiently during learn_step")
    }

    fn critic_2_ref(&self) -> &Critic {
        self.critic_2
            .as_ref()
            .expect("critic_2 is populated except transiently during learn_step")
    }

    /// Samples an action for the current observation.
    ///
    /// Before `learning_starts` steps, draws a uniform random action on
    /// `[low, high]`. Afterwards runs the actor, adds Gaussian exploration
    /// noise, and clips to the action bounds. The unnoised policy mean is
    /// emitted when `training == false`.
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
        self.buffer.len() >= self.config.batch_size
            && self.step >= self.config.learning_starts
    }

    /// Runs one learning step.
    ///
    /// 1. Samples a batch from the replay buffer.
    /// 2. Builds the twin-critic TD target:
    ///    `y = r + γ(1−d)·min(target_q1(s', ã), target_q2(s', ã))`
    ///    where `ã = clip(target_actor(s') + clip(N(0, σ²), -c, c),
    ///    low, high)` (target-policy smoothing).
    /// 3. Runs an independent backward + optimizer step for each critic.
    /// 4. Every `policy_frequency`-th critic step, runs an actor update
    ///    against `critic_1` and Polyak-averages all three targets.
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
        let next_t_inner: Tensor<B::InnerBackend, DB> = Tensor::from_data(
            TensorData::new(next_flat, batched_obs_shape),
            &device,
        );
        let action_t: Tensor<B, DAB> = Tensor::from_data(
            TensorData::new(action_flat, batched_action_shape),
            &device,
        );

        let rewards_inner: Tensor<B::InnerBackend, 1> =
            Tensor::from_data(TensorData::new(rewards, vec![batch_size]), &device);
        let dones_inner: Tensor<B::InnerBackend, 1> =
            Tensor::from_data(TensorData::new(dones, vec![batch_size]), &device);

        // --- Target computation (no autodiff) ---
        // Convention matches DDPG and CleanRL: clip to `low[0]`/`high[0]`.
        let low_scalar = self.low[0];
        let high_scalar = self.high[0];

        let raw_next_action: Tensor<B::InnerBackend, DAB> =
            Actor::forward_inner(&self.target_actor, next_t_inner.clone());
        let next_actions = smoothed_target_action::<B::InnerBackend, R, DAB>(
            raw_next_action,
            self.config.policy_noise,
            self.config.noise_clip,
            low_scalar,
            high_scalar,
            rng,
        );

        let next_q1: Tensor<B::InnerBackend, 1> = Critic::forward_inner(
            &self.target_critic_1,
            next_t_inner.clone(),
            next_actions.clone(),
        );
        let next_q2: Tensor<B::InnerBackend, 1> = Critic::forward_inner(
            &self.target_critic_2,
            next_t_inner,
            next_actions,
        );
        let target_inner: Tensor<B::InnerBackend, 1> = compute_twin_critic_target(
            rewards_inner,
            next_q1,
            next_q2,
            dones_inner,
            self.config.gamma,
        );
        let target: Tensor<B, 1> =
            Tensor::from_data(target_inner.into_data(), &device);

        // --- Critic updates: two independent backward passes ---
        let critic_1 = self.critic_1.take().expect("critic_1 already taken");
        let critic_2 = self.critic_2.take().expect("critic_2 already taken");

        let q1_pred: Tensor<B, 1> = critic_1.forward(obs_t.clone(), action_t.clone());
        let q2_pred: Tensor<B, 1> = critic_2.forward(obs_t.clone(), action_t);

        let q_mean = q1_pred
            .clone()
            .min_pair(q2_pred.clone())
            .mean()
            .into_scalar()
            .elem::<f32>();

        let loss_1_tensor = (q1_pred - target.clone()).powi_scalar(2).mean();
        let loss_2_tensor = (q2_pred - target).powi_scalar(2).mean();
        let loss_1 = loss_1_tensor.clone().into_scalar().elem::<f32>();
        let loss_2 = loss_2_tensor.clone().into_scalar().elem::<f32>();
        let critic_loss = loss_1 + loss_2;

        let grads_1 = loss_1_tensor.backward();
        let grads_1_params = GradientsParams::from_grads(grads_1, &critic_1);
        let critic_1 = self
            .critic_1_opt
            .step(self.config.critic_lr, critic_1, grads_1_params);

        let grads_2 = loss_2_tensor.backward();
        let grads_2_params = GradientsParams::from_grads(grads_2, &critic_2);
        let critic_2 = self
            .critic_2_opt
            .step(self.config.critic_lr, critic_2, grads_2_params);

        self.critic_1 = Some(critic_1);
        self.critic_2 = Some(critic_2);
        self.critic_updates += 1;

        // --- Actor + Polyak update (every policy_frequency-th critic step) ---
        let mut actor_loss_opt: Option<f32> = None;
        if self.critic_updates.is_multiple_of(self.config.policy_frequency) {
            let actor = self.actor.take().expect("actor already taken");
            let predicted_actions: Tensor<B, DAB> = actor.forward(obs_t.clone());
            let q_actor: Tensor<B, 1> = self
                .critic_1_ref()
                .forward(obs_t, predicted_actions);
            let actor_loss_tensor = q_actor.mean().neg();
            let actor_loss_value =
                actor_loss_tensor.clone().into_scalar().elem::<f32>();

            let grads = actor_loss_tensor.backward();
            let actor_grads = GradientsParams::from_grads(grads, &actor);
            let actor = self
                .actor_opt
                .step(self.config.actor_lr, actor, actor_grads);

            let tau = self.config.tau as f64;

            let fresh_target_actor = actor.valid();
            let target_actor = std::mem::replace(&mut self.target_actor, fresh_target_actor);
            self.target_actor = Actor::soft_update(&actor, target_actor, tau);

            let fresh_target_critic_1 = self.critic_1_ref().valid();
            let target_critic_1 =
                std::mem::replace(&mut self.target_critic_1, fresh_target_critic_1);
            self.target_critic_1 =
                Critic::soft_update(self.critic_1_ref(), target_critic_1, tau);

            let fresh_target_critic_2 = self.critic_2_ref().valid();
            let target_critic_2 =
                std::mem::replace(&mut self.target_critic_2, fresh_target_critic_2);
            self.target_critic_2 =
                Critic::soft_update(self.critic_2_ref(), target_critic_2, tau);

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

    use burn::backend::NdArray;
    use burn::tensor::TensorData;

    type BI = NdArray;

    #[test]
    fn metrics_performance_record_returns_reward_and_steps() {
        let m = Td3Metrics {
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
        let err = Td3AgentError::InvalidAction("bad slice".into());
        assert_eq!(err.to_string(), "Invalid action: bad slice");
    }

    #[test]
    fn td3_target_is_min_of_twin_critics() {
        // `next_q1 = [2.0, 1.0, 5.0]` and `next_q2 = [3.0, 0.5, 4.0]` →
        // element-wise min is `[2.0, 0.5, 4.0]`. With γ = 0.9, non-terminal
        // rewards `[0.1, 0.2, 0.3]` and dones `[0, 0, 1]`, the target is
        // `[0.1 + 0.9*2.0, 0.2 + 0.9*0.5, 0.3 + 0.9*4.0*0]`
        // `= [1.9, 0.65, 0.3]`.
        let device = Default::default();
        let rewards = Tensor::<BI, 1>::from_data(
            TensorData::new(vec![0.1_f32, 0.2, 0.3], vec![3]),
            &device,
        );
        let next_q1 = Tensor::<BI, 1>::from_data(
            TensorData::new(vec![2.0_f32, 1.0, 5.0], vec![3]),
            &device,
        );
        let next_q2 = Tensor::<BI, 1>::from_data(
            TensorData::new(vec![3.0_f32, 0.5, 4.0], vec![3]),
            &device,
        );
        let dones = Tensor::<BI, 1>::from_data(
            TensorData::new(vec![0.0_f32, 0.0, 1.0], vec![3]),
            &device,
        );

        let target = compute_twin_critic_target(rewards, next_q1, next_q2, dones, 0.9);
        let data = target.into_data().convert::<f32>();
        let slice = data.as_slice::<f32>().unwrap();
        assert!((slice[0] - 1.9).abs() < 1e-6, "row 0: {}", slice[0]);
        assert!((slice[1] - 0.65).abs() < 1e-6, "row 1: {}", slice[1]);
        assert!((slice[2] - 0.3).abs() < 1e-6, "row 2: {}", slice[2]);
    }
}
