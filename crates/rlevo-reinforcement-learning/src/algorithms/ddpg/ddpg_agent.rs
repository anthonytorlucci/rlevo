//! Deep Deterministic Policy Gradient agent: actor, critic, and replay buffer
//! management.
//!
//! [`DdpgAgent`] owns a deterministic actor, a Q-critic, their Polyak-averaged
//! target twins, two independent Adam optimizers, a uniform FIFO replay
//! buffer, and a [`GaussianNoise`] exploration module. Call
//! [`DdpgAgent::act`] to sample actions, [`DdpgAgent::remember`] to push
//! transitions, and [`DdpgAgent::learn_step`] to run one gradient update.
//! Drive the full loop with [`crate::algorithms::ddpg::train`].

use std::marker::PhantomData;

use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, GradientsParams};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{ElementConversion, Tensor, TensorData};
use rand::Rng;
use rand::RngExt;

use crate::metrics::{AgentStats, PerformanceRecord};
use crate::replay::{ContinuousTransition, ReplayBufferError, ReplayStrategy, UniformReplay};
use rlevo_core::action::BoundedAction;
use rlevo_core::base::{Observation, TensorConvertible};
use rlevo_core::config::Validate;

use crate::algorithms::ddpg::ddpg_config::DdpgTrainingConfig;
use crate::algorithms::ddpg::ddpg_model::{ContinuousQ, DeterministicPolicy};
use crate::algorithms::ddpg::exploration::GaussianNoise;
use crate::algorithms::shared::{
    Slot, UNIFORM_REPLAY_BETA, action_bound_tensors, assert_bounds_match_components,
    clip_to_action_bounds,
};
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

/// Deep Deterministic Policy Gradient agent.
///
/// Owns a deterministic actor, a continuous Q-critic, their Polyak-averaged
/// target twins, two independent Adam optimizers, a uniform FIFO replay
/// buffer, and a [`GaussianNoise`] exploration module.
///
/// The typical call sequence per environment step is:
///
/// 1. [`act`](Self::act) — choose an action (random during warm-up, noisy
///    policy thereafter).
/// 2. [`remember`](Self::remember) — store the resulting transition.
/// 3. [`on_env_step`](Self::on_env_step) — advance the global step counter.
/// 4. [`learn_step`](Self::learn_step) — run one gradient update (no-op
///    during warm-up).
///
/// Drive the complete loop with [`crate::algorithms::ddpg::train::train`].
///
/// # Network ownership
///
/// The actor and critic live in a [`Slot`], which holds each network across the
/// Burn optimizer step that consumes it by value. Every fallible operation —
/// the forward pass, the loss, `backward`, and `GradientsParams::from_grads` —
/// runs on a borrow, so a panic there leaves the agent intact and usable. The
/// networks are out of their slots only for the duration of the
/// [`Optimizer::step`](burn::optim::Optimizer::step) call itself; a panic in
/// that one window is terminal for this agent and requires rebuilding it. See
/// the [`shared`](crate::algorithms::shared) module docs for why that residual
/// window is irreducible.
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
    actor: Slot<Actor>,
    target_actor: Actor::InnerModule,
    critic: Slot<Critic>,
    target_critic: Critic::InnerModule,
    actor_opt: OptimizerAdaptor<Adam, Actor, B>,
    critic_opt: OptimizerAdaptor<Adam, Critic, B>,
    buffer: UniformReplay<ContinuousTransition<O>>,
    exploration: GaussianNoise,
    low: &'static [f32],
    high: &'static [f32],
    /// `[1, ..action_shape]` per-component bounds for the target-action clip,
    /// built once at construction — see [`action_bound_tensors`].
    low_t: Tensor<B::InnerBackend, DAB>,
    high_t: Tensor<B::InnerBackend, DAB>,
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
            .finish_non_exhaustive()
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
    ///
    /// Target networks are initialised as clones of the supplied modules via
    /// [`AutodiffModule::valid`](burn::module::AutodiffModule::valid). Optimizers are built from
    /// [`DdpgTrainingConfig::optimizer`]; if
    /// [`DdpgTrainingConfig::clip_grad`] is `Some`, the same clipping
    /// configuration is applied to both the actor and critic optimizers.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`](rlevo_core::config::ConfigError) if `config`
    /// fails [`DdpgTrainingConfig::validate`](rlevo_core::config::Validate::validate).
    ///
    /// # Panics
    ///
    /// Panics if `A`'s [`BoundedAction`] impl violates its length contract —
    /// i.e. if `A::low().len()` or `A::high().len()` differs from
    /// `A::COMPONENTS`. `&'static [f32]` cannot carry that guarantee in the
    /// type system (ADR 0053), so it is checked once here rather than being
    /// discovered as an out-of-bounds index mid-episode.
    pub fn new(
        actor: Actor,
        critic: Critic,
        config: DdpgTrainingConfig,
        device: B::Device,
    ) -> Result<Self, rlevo_core::config::ConfigError> {
        config.validate()?;
        assert_bounds_match_components::<DA, A>();
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
        let (low_t, high_t) = action_bound_tensors::<B::InnerBackend, A, DA, DAB>(&device);
        Ok(Self {
            actor: Slot::new(actor),
            target_actor,
            critic: Slot::new(critic),
            target_critic,
            actor_opt,
            critic_opt,
            buffer: UniformReplay::new(config.replay_buffer_capacity),
            exploration,
            low: A::low(),
            high: A::high(),
            low_t,
            high_t,
            config,
            device,
            step: 0,
            critic_updates: 0,
            stats,
            last_actor_loss: 0.0,
            _action: PhantomData,
        })
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

    /// Samples an action for the current observation.
    ///
    /// Before `learning_starts` steps, draws a uniform random action on
    /// `[low, high]`. Afterwards runs the actor, adds Gaussian noise, and
    /// clips to the action bounds. The unnoised policy mean is emitted when
    /// `training == false`.
    ///
    /// # Panics
    ///
    /// Panics if the actor slot is poisoned — see
    /// [`Slot::get`](crate::algorithms::shared::Slot::get). This requires a
    /// prior panic *inside* the actor's optimizer step; a panic anywhere else
    /// in [`learn_step`](Self::learn_step) leaves this method working.
    pub fn act<R: Rng + ?Sized>(&self, obs: &O, training: bool, rng: &mut R) -> A {
        if training && self.step < self.config.learning_starts {
            let sample: Vec<f32> = (0..A::COMPONENTS)
                .map(|i| rng.random_range(self.low[i]..=self.high[i]))
                .collect();
            return A::from_slice(&sample);
        }

        let obs_t: Tensor<B, DO> = obs.to_tensor(&self.device);
        let batched: Tensor<B, DB> = obs_t.unsqueeze::<DB>();
        let raw: Tensor<B, DAB> = self.actor.get().forward(batched);
        let data = raw.into_data().convert::<f32>();
        let slice = data.as_slice::<f32>().expect("actor output is f32");
        let mean: Vec<f32> = slice.iter().take(A::COMPONENTS).copied().collect();
        let out = if training {
            self.exploration.apply(&mean, self.low, self.high, rng)
        } else {
            mean.iter()
                .enumerate()
                .map(|(i, v)| v.clamp(self.low[i], self.high[i]))
                .collect()
        };
        A::from_slice(&out)
    }

    /// Snapshots the actor onto the inner (non-autodiff) backend for repeated
    /// greedy inference.
    ///
    /// Returns a frozen inference handle for use with
    /// [`act_with`](Self::act_with). Snapshot once after training, then reuse
    /// across many steps — the snapshot goes stale if the actor is updated
    /// again.
    ///
    /// # Panics
    ///
    /// Panics if the actor slot is poisoned — see
    /// [`Slot::get`](crate::algorithms::shared::Slot::get).
    pub fn inference_net(&self) -> Actor::InnerModule {
        self.actor.get().valid()
    }

    /// Deterministic action against a pre-snapshotted inner actor.
    ///
    /// Equivalent to `act(obs, false, _)` (the unnoised, bound-clipped policy
    /// mean) but runs on the non-autodiff backend via
    /// [`inference_net`](Self::inference_net), avoiding the per-call autodiff
    /// graph construction that [`act`](Self::act) incurs.
    /// # Panics
    ///
    /// Panics if the actor's output tensor is not `f32`, or if it yields fewer
    /// than `A::COMPONENTS` values. Both indicate the supplied `net` does not
    /// match the action type this agent was built for.
    pub fn act_with(&self, net: &Actor::InnerModule, obs: &O) -> A {
        let obs_t: Tensor<B::InnerBackend, DO> = obs.to_tensor(&self.device);
        let batched: Tensor<B::InnerBackend, DB> = obs_t.unsqueeze::<DB>();
        let raw: Tensor<B::InnerBackend, DAB> = Actor::forward_inner(net, batched);
        let data = raw.into_data().convert::<f32>();
        let slice = data.as_slice::<f32>().expect("actor output is f32");
        let out: Vec<f32> = (0..A::COMPONENTS)
            .map(|i| slice[i].clamp(self.low[i], self.high[i]))
            .collect();
        A::from_slice(&out)
    }

    /// Appends a transition to the replay buffer, evicting the oldest entry
    /// when the buffer is at capacity.
    ///
    /// # Arguments
    ///
    /// - `terminated` — pass [`Snapshot::is_terminated`], **not**
    ///   [`Snapshot::is_done`]. Only a true environmental termination may zero
    ///   the Bellman bootstrap; on a truncation (time-limit cutoff) `next_obs`
    ///   is a genuine continuation state whose value must still be
    ///   bootstrapped. See [`Transition::terminated`].
    ///
    /// [`Transition::terminated`]: crate::replay::Transition::terminated
    /// [`Snapshot::is_terminated`]: rlevo_core::environment::Snapshot::is_terminated
    /// [`Snapshot::is_done`]: rlevo_core::environment::Snapshot::is_done
    pub fn remember(&mut self, obs: O, action: &A, reward: f32, next_obs: O, terminated: bool) {
        self.buffer.push(ContinuousTransition {
            obs,
            action: action.as_slice().to_vec(),
            reward,
            next_obs,
            terminated,
        });
    }

    /// Test-only view of the Bellman bootstrap masks currently held in the
    /// replay buffer, oldest first.
    ///
    /// Exists so the `train`-loop tests can assert that a *truncated* step was
    /// recorded with `terminated = false` — the invariant that separates a
    /// time-limit cutoff from a real MDP termination.
    #[cfg(test)]
    pub(crate) fn replay_terminated_flags(&self) -> Vec<bool> {
        self.buffer.iter().map(|t| t.terminated).collect()
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
    /// Returns `None` if the agent is still in warm-up (see
    /// [`can_learn`](Self::can_learn)).
    ///
    /// The critic is updated every call via mean-squared Bellman error. The
    /// actor is updated with the deterministic policy gradient (negative mean
    /// Q-value over the batch). Both target networks are Polyak-averaged
    /// toward the active networks with rate τ
    /// ([`DdpgTrainingConfig::tau`]) after every actor update.
    ///
    /// # Panics
    ///
    /// Panics if the actor or critic slot is poisoned, i.e. a previous
    /// [`Optimizer::step`](burn::optim::Optimizer::step) unwound and lost the
    /// network. Every other operation here — sampling, the forward passes, the
    /// losses, `backward`, and gradient reduction — runs on a borrow, so a
    /// panic in any of them leaves the agent fully usable; only a panic inside
    /// the optimizer step itself is terminal. See
    /// [`Slot`](crate::algorithms::shared::Slot).
    pub fn learn_step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Option<LearnOutcome> {
        if !self.can_learn() {
            return None;
        }
        let batch_size = self.config.batch_size;
        let device = self.device.clone();

        // --- Sample batch ---
        // `can_learn()` above already established `buffer.len() >= batch_size`,
        // so the only variant `sample` can return here is unreachable.
        let batch = self
            .buffer
            .sample(batch_size, UNIFORM_REPLAY_BETA, rng)
            .ok()?;

        let obs_shape = O::shape();
        let obs_numel: usize = obs_shape.iter().product();
        let action_shape = A::shape();
        let action_numel: usize = action_shape.iter().product();

        let mut obs_flat: Vec<f32> = Vec::with_capacity(batch_size * obs_numel);
        let mut next_flat: Vec<f32> = Vec::with_capacity(batch_size * obs_numel);
        let mut action_flat: Vec<f32> = Vec::with_capacity(batch_size * action_numel);
        let mut rewards: Vec<f32> = Vec::with_capacity(batch_size);
        let mut terminated: Vec<f32> = Vec::with_capacity(batch_size);

        for &id in batch.ids() {
            let t = self.buffer.get(id).expect("a freshly sampled id is live");
            // Stage host-side: `to_tensor` would upload each row only to read it
            // straight back -- one wgpu sync point per row, no op in between.
            t.obs.write_host_row(&mut obs_flat);
            t.next_obs.write_host_row(&mut next_flat);
            action_flat.extend_from_slice(&t.action);
            rewards.push(t.reward);
            terminated.push(if t.terminated { 1.0 } else { 0.0 });
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
        let terminated_inner: Tensor<B::InnerBackend, 1> =
            Tensor::from_data(TensorData::new(terminated, vec![batch_size]), &device);

        // --- Target computation (no autodiff) ---
        // Clip the target action per component, against the `Box(low, high)`
        // vectors — TD3 Eq. 14 (Fujimoto et al. 2018, arXiv:1802.09477), whose
        // clip DDPG's target path shares. A scalar `low[0]`/`high[0]` collapse
        // is only equivalent when every component shares a bound; for an
        // asymmetric space such as CarRacing's `Box([-1,0,0], [1,1,1])` it would
        // admit negative gas and brake into the target — precisely the "values
        // of impossible actions" the clip exists to suppress. Burn's `clamp` is
        // scalar-only, so this goes through `max_pair`/`min_pair` against the
        // `[1, ..action_shape]` bound tensors, broadcast over the batch.
        let next_actions: Tensor<B::InnerBackend, DAB> = clip_to_action_bounds(
            Actor::forward_inner(&self.target_actor, next_t_inner.clone()),
            self.low_t.clone(),
            self.high_t.clone(),
        );
        let next_q: Tensor<B::InnerBackend, 1> =
            Critic::forward_inner(&self.target_critic, next_t_inner, next_actions);
        let target_inner: Tensor<B::InnerBackend, 1> =
            compute_target_q_values(rewards_inner, next_q, terminated_inner, self.config.gamma);
        let target: Tensor<B, 1> = Tensor::from_data(target_inner.into_data(), &device);

        // --- Critic update ---
        // Everything up to `step_with` runs on a borrow of the slot, so a panic
        // in the forward/loss/backward region cannot poison the agent.
        let q_pred: Tensor<B, 1> = self.critic.get().forward(obs_t.clone(), action_t);
        let q_mean = q_pred.clone().mean().into_scalar().elem::<f32>();
        let td_error = q_pred - target;
        let critic_loss_tensor = td_error.powi_scalar(2).mean();
        let critic_loss = critic_loss_tensor.clone().into_scalar().elem::<f32>();

        let grads = critic_loss_tensor.backward();
        let grads = GradientsParams::from_grads(grads, self.critic.get());
        self.critic
            .step_with(&mut self.critic_opt, self.config.critic_lr, grads);
        self.critic_updates += 1;

        // --- Actor + Polyak update (every policy_frequency-th critic step) ---
        let mut actor_loss_opt: Option<f32> = None;
        if self
            .critic_updates
            .is_multiple_of(self.config.policy_frequency)
        {
            let predicted_actions: Tensor<B, DAB> = self.actor.get().forward(obs_t.clone());
            let q_actor: Tensor<B, 1> = self.critic.get().forward(obs_t, predicted_actions);
            let actor_loss_tensor = q_actor.mean().neg();
            let actor_loss_value = actor_loss_tensor.clone().into_scalar().elem::<f32>();

            let grads = actor_loss_tensor.backward();
            let actor_grads = GradientsParams::from_grads(grads, self.actor.get());
            self.actor
                .step_with(&mut self.actor_opt, self.config.actor_lr, actor_grads);

            // Clone rather than move out: each target field stays intact if
            // `soft_update` panics, so a failure can't silently hard-sync the
            // target onto its live network.
            let tau = f64::from(self.config.tau);
            self.target_actor =
                Actor::soft_update(self.actor.get(), self.target_actor.clone(), tau);
            self.target_critic =
                Critic::soft_update(self.critic.get(), self.target_critic.clone(), tau);

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
    // Exact comparison is intentional throughout this test module: the values are
    // config literals read back unchanged, or a computed result whose bit-exactness
    // is itself the property under test (that an anneal lands exactly on its
    // endpoint, that `-0.0` is accepted as the no-correction setting). A tolerance
    // would let a real regression pass. Reviewed as a class, not site-by-site.
    #![allow(clippy::float_cmp)]
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
