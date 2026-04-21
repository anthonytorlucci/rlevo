//! Soft Actor-Critic (SAC) agent: stochastic actor, twin critics, learnable
//! temperature, and replay buffer management.
//!
//! [`SacAgent`] pairs a squashed-Gaussian actor with **two** Q-critics (each
//! with its own Polyak-averaged target) and a single scalar `log α` module.
//! Compared to [`Td3Agent`](crate::algorithms::td3::td3_agent::Td3Agent), SAC:
//!
//! 1. drops the target actor (the stochastic policy + `min`-of-twin-Q backup
//!    already addresses critic overestimation),
//! 2. replaces deterministic-actor + exploration-noise with a single
//!    reparameterized sample from the policy at every env step, and
//! 3. adds an entropy term `−α·log π(a'|s')` to the Bellman target and an
//!    auto-tuning loss for `log α`.
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

use crate::algorithms::sac::sac_alpha::LogAlpha;
use crate::algorithms::sac::sac_config::SacTrainingConfig;
use crate::algorithms::sac::sac_model::{ContinuousQ, SquashedGaussianPolicy};
use crate::utils::compute_target_q_values;

/// Error variants returned by [`SacAgent`] operations.
#[derive(Debug, thiserror::Error)]
pub enum SacAgentError {
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

/// Per-episode statistics emitted by the SAC training loop.
///
/// Mirrors [`Td3Metrics`](crate::algorithms::td3::td3_agent::Td3Metrics) with
/// SAC-specific additions (`alpha`, `entropy`) so dashboards can compare the
/// two algorithms on the same plots.
#[derive(Debug, Clone, Copy)]
pub struct SacMetrics {
    /// Total reward collected during the episode.
    pub reward: f32,
    /// Number of environment steps taken.
    pub steps: usize,
    /// Sum of the two critics' MSE Bellman errors on the most recent learn
    /// step.
    pub critic_loss: f32,
    /// Most recent actor loss (`0.0` until the first policy update fires).
    pub actor_loss: f32,
    /// Most recent α value (`= exp(log α)`).
    pub alpha: f32,
    /// Most recent mean `−log π(a|s)` across the actor batch — proxy for
    /// policy entropy.
    pub entropy: f32,
    /// Mean of `min(q1, q2)` across the most recent learn step.
    pub q_mean: f32,
}

impl PerformanceRecord for SacMetrics {
    fn score(&self) -> f32 {
        self.reward
    }

    fn duration(&self) -> usize {
        self.steps
    }
}

/// Summary values returned by a single [`SacAgent::learn_step`].
#[derive(Debug, Clone, Copy)]
pub struct LearnOutcome {
    /// Sum of the two critics' MSE Bellman errors on this batch.
    pub critic_loss: f32,
    /// Actor loss, or `None` on critic-only iterations (delayed-update
    /// skips).
    pub actor_loss: Option<f32>,
    /// Current α (after an auto-tuning step on actor-update iterations).
    pub alpha: f32,
    /// Batch-mean `−log π(a|s)` on the most recent actor update, or `None`
    /// on critic-only iterations.
    pub entropy: Option<f32>,
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

/// Computes the SAC Bellman target:
/// `y = r + γ · (1 − done) · (min(Q1', Q2') − α · log π(a'|s'))`.
///
/// Exposed at crate visibility so the unit tests can exercise the SAC
/// entropy-augmented backup without standing up a full agent.
pub(crate) fn compute_sac_target<BI: Backend>(
    rewards: Tensor<BI, 1>,
    next_q1: Tensor<BI, 1>,
    next_q2: Tensor<BI, 1>,
    next_log_prob: Tensor<BI, 1>,
    alpha: f32,
    dones: Tensor<BI, 1>,
    gamma: f32,
) -> Tensor<BI, 1> {
    let min_q = next_q1.min_pair(next_q2);
    let entropy_adjusted = min_q - next_log_prob.mul_scalar(alpha);
    compute_target_q_values(rewards, entropy_adjusted, dones, gamma)
}

/// Soft Actor-Critic (SAC) agent.
///
/// # Const generics
///
/// Same layout as [`Td3Agent`](crate::algorithms::td3::td3_agent::Td3Agent):
/// - `DO` — rank of a single observation tensor.
/// - `DB` — rank of a batched observation tensor (= `DO + 1`).
/// - `DA` — rank of a single action tensor.
/// - `DAB` — rank of a batched action tensor (= `DA + 1`).
pub struct SacAgent<
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
    Actor: SquashedGaussianPolicy<B, DB, DAB>,
    Critic: ContinuousQ<B, DB, DAB>,
    O: Observation<DO> + TensorConvertible<DO, B> + TensorConvertible<DO, B::InnerBackend>,
    A: BoundedAction<DA>,
{
    actor: Option<Actor>,
    // Inner-backend snapshot of `actor`, refreshed after each actor update
    // and used by the target-Q computation. Kept separately from the live
    // actor so `learn_step` never has to call `.valid()` on a Module that
    // also participates in the critic autodiff graph — on Burn 0.20's
    // shared autodiff server, running `.valid()` mid-learn was unstable.
    actor_snapshot: Actor::InnerModule,
    critic_1: Option<Critic>,
    critic_2: Option<Critic>,
    target_critic_1: Critic::InnerModule,
    target_critic_2: Critic::InnerModule,
    log_alpha: LogAlpha,
    actor_opt: OptimizerAdaptor<Adam, Actor, B>,
    critic_1_opt: OptimizerAdaptor<Adam, Critic, B>,
    critic_2_opt: OptimizerAdaptor<Adam, Critic, B>,
    buffer: VecDeque<Transition<O>>,
    low: [f32; DA],
    high: [f32; DA],
    target_entropy: f32,
    config: SacTrainingConfig,
    device: B::Device,
    step: usize,
    critic_updates: usize,
    stats: AgentStats<SacMetrics>,
    last_actor_loss: f32,
    last_alpha: f32,
    last_entropy: f32,
    _action: PhantomData<A>,
}

impl<B, Actor, Critic, O, A, const DO: usize, const DB: usize, const DA: usize, const DAB: usize>
    std::fmt::Debug for SacAgent<B, Actor, Critic, O, A, DO, DB, DA, DAB>
where
    B: AutodiffBackend,
    Actor: SquashedGaussianPolicy<B, DB, DAB>,
    Critic: ContinuousQ<B, DB, DAB>,
    O: Observation<DO> + TensorConvertible<DO, B> + TensorConvertible<DO, B::InnerBackend>,
    A: BoundedAction<DA>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SacAgent")
            .field("step", &self.step)
            .field("critic_updates", &self.critic_updates)
            .field("buffer_len", &self.buffer.len())
            .field("alpha", &self.last_alpha)
            .field("target_entropy", &self.target_entropy)
            .field("low", &self.low)
            .field("high", &self.high)
            .field("config", &self.config)
            .finish()
    }
}

impl<B, Actor, Critic, O, A, const DO: usize, const DB: usize, const DA: usize, const DAB: usize>
    SacAgent<B, Actor, Critic, O, A, DO, DB, DA, DAB>
where
    B: AutodiffBackend,
    Actor: SquashedGaussianPolicy<B, DB, DAB>,
    Critic: ContinuousQ<B, DB, DAB>,
    O: Observation<DO> + TensorConvertible<DO, B> + TensorConvertible<DO, B::InnerBackend>,
    A: BoundedAction<DA>,
{
    /// Constructs a new agent from pre-built actor and two independent
    /// critic networks.
    ///
    /// The caller is expected to initialise `critic_1` and `critic_2` with
    /// different random seeds — SAC, like TD3, relies on independent initial
    /// errors so the `min` target meaningfully suppresses overestimation.
    pub fn new(
        actor: Actor,
        critic_1: Critic,
        critic_2: Critic,
        config: SacTrainingConfig,
        device: B::Device,
    ) -> Self {
        let actor_snapshot = actor.valid();
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
        let initial_alpha = config.initial_alpha;
        let log_alpha_init = initial_alpha.max(f32::MIN_POSITIVE).ln();
        let log_alpha = LogAlpha::new(log_alpha_init);
        let target_entropy = config.target_entropy.unwrap_or_else(|| -(A::DIM as f32));
        let buffer_capacity = config.buffer_capacity;
        let stats = AgentStats::<SacMetrics>::new(100);
        Self {
            actor: Some(actor),
            actor_snapshot,
            critic_1: Some(critic_1),
            critic_2: Some(critic_2),
            target_critic_1,
            target_critic_2,
            log_alpha,
            actor_opt,
            critic_1_opt,
            critic_2_opt,
            buffer: VecDeque::with_capacity(buffer_capacity),
            low: A::low(),
            high: A::high(),
            target_entropy,
            config,
            device,
            step: 0,
            critic_updates: 0,
            stats,
            last_actor_loss: 0.0,
            last_alpha: initial_alpha,
            last_entropy: 0.0,
            _action: PhantomData,
        }
    }

    /// Current agent statistics.
    pub fn stats(&self) -> &AgentStats<SacMetrics> {
        &self.stats
    }

    /// Records one completed episode into the running statistics.
    pub fn record_episode(&mut self, metrics: SacMetrics) {
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

    /// Number of critic updates applied so far.
    pub fn critic_updates(&self) -> usize {
        self.critic_updates
    }

    /// Most recent α value (exposed for metrics / tests).
    pub fn last_alpha(&self) -> f32 {
        self.last_alpha
    }

    /// Target entropy H̄ in effect.
    pub fn target_entropy(&self) -> f32 {
        self.target_entropy
    }

    /// Most recent actor loss (persists between policy updates).
    pub fn last_actor_loss(&self) -> f32 {
        self.last_actor_loss
    }

    /// Most recent batch-mean `−log π(a|s)` proxy for policy entropy.
    pub fn last_entropy(&self) -> f32 {
        self.last_entropy
    }

    fn actor_ref(&self) -> &Actor {
        self.actor
            .as_ref()
            .expect("actor not restored — earlier panic in learn_step?")
    }

    fn critic_1_ref(&self) -> &Critic {
        self.critic_1
            .as_ref()
            .expect("critic_1 not restored — earlier panic in learn_step?")
    }

    fn critic_2_ref(&self) -> &Critic {
        self.critic_2
            .as_ref()
            .expect("critic_2 not restored — earlier panic in learn_step?")
    }

    /// Samples an action for the current observation.
    ///
    /// Before `learning_starts` steps, draws a uniform random action on
    /// `[low, high]`. Afterwards the action is a reparameterized sample from
    /// the squashed-Gaussian policy when `training=true`, and the
    /// deterministic policy mean (tanh-squashed) otherwise.
    pub fn act<R: Rng + ?Sized>(&self, obs: &O, training: bool, rng: &mut R) -> A {
        if training && self.step < self.config.learning_starts {
            let sample: Vec<f32> = (0..A::DIM)
                .map(|i| rng.random_range(self.low[i]..=self.high[i]))
                .collect();
            return A::from_slice(&sample);
        }

        // Run action-selection on the inner backend so the autodiff graph
        // isn't expanded every env step — the result is never `.backward`'d
        // and the orphan graphs otherwise accumulate in Burn 0.20's shared
        // server (and interfere with later critic backwards).
        let obs_inner: Tensor<B::InnerBackend, DO> = obs.to_tensor(&self.device);
        let batched_inner: Tensor<B::InnerBackend, DB> = obs_inner.unsqueeze::<DB>();

        let action_dim = self.actor_ref().action_dim();
        let eps: Tensor<B::InnerBackend, DAB> = if training {
            sample_noise::<B::InnerBackend, R, DAB>(1, action_dim, &self.device, rng)
        } else {
            // ε = 0 ⇒ z = μ, which matches `deterministic_action` for a
            // squashed-Gaussian policy (both evaluate to `scale·tanh(μ)`).
            Tensor::from_data(
                TensorData::new(vec![0.0_f32; action_dim], vec![1, action_dim]),
                &self.device,
            )
        };
        let raw: Tensor<B::InnerBackend, DAB> =
            Actor::forward_sample_inner(&self.actor_snapshot, batched_inner, eps).action;

        let data = raw.into_data().convert::<f32>();
        let slice = data.as_slice::<f32>().expect("actor output is f32");
        // Actions are already squashed into `[bias - scale, bias + scale]`
        // by the policy; still clip against `low/high` to be robust to users
        // whose scale/bias disagree with the action type's bounds.
        let out: Vec<f32> = (0..A::DIM)
            .map(|i| slice[i].clamp(self.low[i], self.high[i]))
            .collect();
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

    /// Returns `true` once warm-up has elapsed and the buffer has enough
    /// transitions to draw a batch.
    pub fn can_learn(&self) -> bool {
        self.buffer.len() >= self.config.batch_size && self.step >= self.config.learning_starts
    }

    /// Runs one learning step.
    ///
    /// 1. Samples a batch from the replay buffer.
    /// 2. Draws `next_ε`, runs `(next_a, next_logp) = actor(next_obs, next_ε)`
    ///    on the inner (no-autodiff) backend and computes the SAC target
    ///    `y = r + γ(1−d)·(min(Q1', Q2') − α·next_logp)`.
    /// 3. Runs an independent backward + optimizer step for each critic.
    /// 4. Every `policy_frequency`-th critic step, runs an actor update
    ///    (`L_π = α·logp − min(Q1(s,a), Q2(s,a))`) and — when `autotune` is
    ///    enabled — an α-update (`L_α = −(log α · (logp + H̄))`).
    /// 5. Every `target_update_frequency`-th critic step, Polyak-averages
    ///    both critic targets.
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
        let action_dim = self.actor_ref().action_dim();
        let next_eps: Tensor<B::InnerBackend, DAB> =
            sample_noise::<B::InnerBackend, R, DAB>(batch_size, action_dim, &device, rng);
        let next_sample =
            Actor::forward_sample_inner(&self.actor_snapshot, next_t_inner.clone(), next_eps);
        let next_action = next_sample.action;
        let next_log_prob = next_sample.log_prob;

        let next_q1: Tensor<B::InnerBackend, 1> = Critic::forward_inner(
            &self.target_critic_1,
            next_t_inner.clone(),
            next_action.clone(),
        );
        let next_q2: Tensor<B::InnerBackend, 1> =
            Critic::forward_inner(&self.target_critic_2, next_t_inner, next_action);

        let alpha_val = self.log_alpha.alpha();
        let target_inner: Tensor<B::InnerBackend, 1> = compute_sac_target(
            rewards_inner,
            next_q1,
            next_q2,
            next_log_prob,
            alpha_val,
            dones_inner,
            self.config.gamma,
        );
        let target: Tensor<B, 1> = Tensor::from_data(target_inner.into_data(), &device);

        // --- Critic updates: two independent backward passes ---
        let critic_1 = self.critic_1.take().expect("critic_1 not restored — earlier panic in learn_step?");
        let critic_2 = self.critic_2.take().expect("critic_2 not restored — earlier panic in learn_step?");

        let q1_pred: Tensor<B, 1> = critic_1.forward(obs_t.clone(), action_t.clone());
        let q2_pred: Tensor<B, 1> = critic_2.forward(obs_t.clone(), action_t);

        // Drop the min-pair scalar to the inner backend before reading it:
        // on Burn 0.20 `into_scalar` directly on an autodiff tensor can
        // prune shared leaves that the next learn_step still needs.
        let q_mean = q1_pred
            .clone()
            .min_pair(q2_pred.clone())
            .mean()
            .inner()
            .into_scalar()
            .elem::<f32>();

        let loss_1_tensor = (q1_pred - target.clone()).powi_scalar(2).mean();
        let loss_2_tensor = (q2_pred - target).powi_scalar(2).mean();
        // Drop to the inner backend before reading the scalar so the
        // autodiff graph retaining `target`'s leaf isn't pruned prematurely
        // — on Burn 0.20, calling `into_scalar` directly on an autodiff
        // tensor can free shared nodes that the next backward still needs.
        let loss_1 = loss_1_tensor.clone().inner().into_scalar().elem::<f32>();
        let loss_2 = loss_2_tensor.clone().inner().into_scalar().elem::<f32>();
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

        // --- Actor + α update (every policy_frequency-th critic step) ---
        let mut actor_loss_opt: Option<f32> = None;
        let mut entropy_opt: Option<f32> = None;
        if self
            .critic_updates
            .is_multiple_of(self.config.policy_frequency)
        {
            let actor = self.actor.take().expect("actor not restored — earlier panic in learn_step?");
            let eps: Tensor<B, DAB> =
                sample_noise::<B, R, DAB>(batch_size, action_dim, &device, rng);
            let sample = actor.forward_sample(obs_t.clone(), eps);
            let log_prob = sample.log_prob;
            // NOTE: canonical SAC uses `min(Q1(s,a), Q2(s,a))` in the actor
            // loss to pessimise the Q estimate the policy optimises against.
            // Running backward through *both* critics consumes both critics'
            // param nodes in Burn 0.20's autodiff server and causes the next
            // learn step's second-critic backward to panic. We follow DDPG /
            // TD3's practical variant and score the actor against `critic_1`
            // alone; the pessimism still enters the policy via the Bellman
            // target's min-of-twin-target-Q backup, which is the term that
            // drives most of the overestimation control anyway.
            let min_q_pi: Tensor<B, 1> = self.critic_1_ref().forward(obs_t.clone(), sample.action);

            let alpha_scalar = self.log_alpha.alpha();
            let actor_loss_tensor = (log_prob.clone().mul_scalar(alpha_scalar) - min_q_pi).mean();
            let actor_loss_value = actor_loss_tensor
                .clone()
                .inner()
                .into_scalar()
                .elem::<f32>();

            // Capture batch-mean log-prob for the α Adam update and the
            // entropy metric before consuming the actor graph in backward.
            let log_prob_mean = log_prob.clone().mean().inner().into_scalar().elem::<f32>();
            let entropy_value = -log_prob_mean;

            let grads = actor_loss_tensor.backward();
            let actor_grads = GradientsParams::from_grads(grads, &actor);
            let actor = self
                .actor_opt
                .step(self.config.actor_lr, actor, actor_grads);
            // Refresh the inner-backend snapshot used by future target-Q
            // computations.
            self.actor_snapshot = actor.valid();
            self.actor = Some(actor);

            // α update (optional). Closed-form scalar Adam — see
            // `LogAlpha::adam_step`.
            if self.config.autotune {
                self.log_alpha.adam_step(
                    log_prob_mean,
                    self.target_entropy,
                    self.config.alpha_lr as f32,
                );
            }
            self.last_alpha = self.log_alpha.alpha();

            self.last_actor_loss = actor_loss_value;
            self.last_entropy = entropy_value;
            actor_loss_opt = Some(actor_loss_value);
            entropy_opt = Some(entropy_value);
        }

        // --- Target Polyak updates ---
        if self
            .critic_updates
            .is_multiple_of(self.config.target_update_frequency)
        {
            let tau = self.config.tau as f64;
            let fresh_target_critic_1 = self.critic_1_ref().valid();
            let target_critic_1 =
                std::mem::replace(&mut self.target_critic_1, fresh_target_critic_1);
            self.target_critic_1 = Critic::soft_update(self.critic_1_ref(), target_critic_1, tau);

            let fresh_target_critic_2 = self.critic_2_ref().valid();
            let target_critic_2 =
                std::mem::replace(&mut self.target_critic_2, fresh_target_critic_2);
            self.target_critic_2 = Critic::soft_update(self.critic_2_ref(), target_critic_2, tau);
        }

        Some(LearnOutcome {
            critic_loss,
            actor_loss: actor_loss_opt,
            alpha: self.last_alpha,
            entropy: entropy_opt,
            q_mean,
        })
    }
}

/// Draws `rows × cols` iid standard-normal samples on CPU and assembles them
/// into a rank-`DAB` tensor of shape `[rows, cols]`. The built-in
/// [`SquashedGaussianPolicyHead`](crate::algorithms::sac::sac_policy::SquashedGaussianPolicyHead)
/// uses `DAB = 2`; the agent stays generic so higher-rank action layouts can
/// plug in a custom policy and their own `DAB`.
fn sample_noise<BB: Backend, R: Rng + ?Sized, const DAB: usize>(
    rows: usize,
    cols: usize,
    device: &BB::Device,
    rng: &mut R,
) -> Tensor<BB, DAB> {
    use rand_distr::{Distribution, StandardNormal};
    let mut data: Vec<f32> = Vec::with_capacity(rows * cols);
    let normal = StandardNormal;
    for _ in 0..(rows * cols) {
        let x: f64 = normal.sample(rng);
        data.push(x as f32);
    }
    Tensor::<BB, DAB>::from_data(TensorData::new(data, vec![rows, cols]), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type BI = NdArray;

    #[test]
    fn metrics_performance_record_returns_reward_and_steps() {
        let m = SacMetrics {
            reward: 3.5,
            steps: 42,
            critic_loss: 0.1,
            actor_loss: -0.2,
            alpha: 0.2,
            entropy: 1.7,
            q_mean: 1.0,
        };
        assert_eq!(m.score(), 3.5);
        assert_eq!(m.duration(), 42);
    }

    #[test]
    fn error_display_uses_thiserror_messages() {
        let err = SacAgentError::InvalidAction("bad slice".into());
        assert_eq!(err.to_string(), "Invalid action: bad slice");
    }

    /// SAC target folds the `−α·next_logp` entropy term into the backup.
    /// With `q1 = [2, 1, 5]`, `q2 = [3, 0.5, 4]`, `next_logp = [0.1, 0.2, 0.3]`,
    /// `α = 0.5`, `r = [0.1, 0.2, 0.3]`, `γ = 0.9`, `dones = [0, 0, 1]`:
    ///   min_q          = [2.0, 0.5, 4.0]
    ///   min_q − α·logp = [2.0 − 0.05, 0.5 − 0.10, 4.0 − 0.15]
    ///                  = [1.95, 0.40, 3.85]
    ///   y              = [0.1 + 0.9·1.95, 0.2 + 0.9·0.40, 0.3 + 0·3.85]
    ///                  = [1.855, 0.560, 0.300]
    #[test]
    fn sac_target_includes_entropy_term() {
        let device = Default::default();
        let rewards =
            Tensor::<BI, 1>::from_data(TensorData::new(vec![0.1_f32, 0.2, 0.3], vec![3]), &device);
        let next_q1 =
            Tensor::<BI, 1>::from_data(TensorData::new(vec![2.0_f32, 1.0, 5.0], vec![3]), &device);
        let next_q2 =
            Tensor::<BI, 1>::from_data(TensorData::new(vec![3.0_f32, 0.5, 4.0], vec![3]), &device);
        let next_logp =
            Tensor::<BI, 1>::from_data(TensorData::new(vec![0.1_f32, 0.2, 0.3], vec![3]), &device);
        let dones =
            Tensor::<BI, 1>::from_data(TensorData::new(vec![0.0_f32, 0.0, 1.0], vec![3]), &device);

        let target = compute_sac_target(rewards, next_q1, next_q2, next_logp, 0.5, dones, 0.9);
        let data = target.into_data().convert::<f32>();
        let slice = data.as_slice::<f32>().unwrap();
        assert!((slice[0] - 1.855).abs() < 1e-5, "row 0: {}", slice[0]);
        assert!((slice[1] - 0.560).abs() < 1e-5, "row 1: {}", slice[1]);
        assert!((slice[2] - 0.300).abs() < 1e-5, "row 2: {}", slice[2]);
    }

    /// With a fixed `min_q`, a policy that moves probability toward the
    /// boundary (higher |logp|) should raise the actor loss by `α·Δlogp`.
    #[test]
    fn actor_loss_penalizes_higher_log_prob() {
        let device = Default::default();
        let min_q = Tensor::<BI, 1>::from_data(TensorData::new(vec![1.0_f32; 4], vec![4]), &device);
        let logp_low =
            Tensor::<BI, 1>::from_data(TensorData::new(vec![-0.5_f32; 4], vec![4]), &device);
        let logp_high =
            Tensor::<BI, 1>::from_data(TensorData::new(vec![0.5_f32; 4], vec![4]), &device);
        let alpha = 0.3_f32;
        let low_loss = (logp_low.mul_scalar(alpha) - min_q.clone())
            .mean()
            .into_scalar()
            .elem::<f32>();
        let high_loss = (logp_high.mul_scalar(alpha) - min_q)
            .mean()
            .into_scalar()
            .elem::<f32>();
        // Δ = α · (0.5 − (−0.5)) = 0.3.
        assert!((high_loss - low_loss - 0.3).abs() < 1e-5);
        assert!(high_loss > low_loss);
    }
}
