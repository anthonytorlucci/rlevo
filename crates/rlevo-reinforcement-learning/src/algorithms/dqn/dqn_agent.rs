//! Deep Q-Network agent: actor, trainer, and replay buffer management.
//!
//! The [`DqnAgent`] struct owns the policy network, frozen target network,
//! optimizer, a uniform replay buffer, and the ε-greedy exploration schedule.
//! Call [`DqnAgent::act`] to sample actions, [`DqnAgent::remember`] to push
//! transitions, and [`DqnAgent::learn_step`] to perform one gradient update.
//! The end-to-end training loop is assembled in
//! [`crate::algorithms::dqn::train`].

use std::marker::PhantomData;

use burn::nn::loss::HuberLossConfig;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, GradientsParams};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{ElementConversion, Int, Tensor, TensorData};
use rand::{Rng, RngExt};

use crate::metrics::{AgentStats, PerformanceRecord};
use crate::replay::{DiscreteTransition, ReplayBufferError, ReplayKind, ReplayStrategy};
use rlevo_core::action::DiscreteAction;
use rlevo_core::base::{Observation, TensorConvertible};
use rlevo_core::config::Validate;

use crate::algorithms::dqn::dqn_config::DqnTrainingConfig;
use crate::algorithms::dqn::dqn_model::DqnModel;
use crate::algorithms::dqn::exploration::EpsilonGreedy;
use crate::algorithms::shared::{FiniteLossGuard, Slot, UNIFORM_REPLAY_BETA, reduce_weighted_loss};
use crate::utils::{PolyakError, compute_target_q_values};

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
    /// The target soft-update failed because the policy and target networks
    /// have mismatched parameter topologies.
    #[error(transparent)]
    Polyak(#[from] PolyakError),
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
    /// Most recent TD (Q-network) loss value.
    ///
    /// DQN has a single TD loss — unlike actor-critic algorithms there is no
    /// separate policy/value pair, so only this field is reported.
    pub policy_loss: f32,
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
/// `DqnAgent` owns the full DQN training state: policy network, frozen target
/// network, Adam optimizer, uniform replay buffer, and the ε-greedy
/// exploration schedule. It is the primary entry point for the collect-learn
/// cycle; the end-to-end training loop is assembled by
/// [`crate::algorithms::dqn::train::train`].
///
/// # Const generics
///
/// - `DO` — rank of a *single* observation tensor (e.g. `1` for a flat vector
///   of shape `[features]`, `3` for an image of shape `[channels, H, W]`).
/// - `DB` — rank of a *batched* observation tensor (`= DO + 1`; e.g. `2` for
///   `[batch, features]`). Rust's const-generic system cannot express `DO + 1`
///   in generic position on stable, so the caller supplies both.
///
/// # Field notes
///
/// - `policy_net` is held in a [`Slot`], the newtype that owns a network across
///   Burn's by-value [`Optimizer::step`](burn::optim::Optimizer::step). Every
///   read goes through `Slot::get`, and the module leaves the field only for the
///   duration of the `step` call itself inside `Slot::step_with` — the forward
///   pass, loss, and `backward` all run on a borrow, so a panic in any of them
///   leaves the agent intact. The one exception is a panic *inside* `step`,
///   which poisons the slot permanently; that window is irreducible and is
///   documented on [`Slot`].
/// - `target_net` lives on `B::InnerBackend` (the non-autodiff backend) so
///   that computing bootstrap targets never builds an autodiff graph.
/// - `step` counts **environment** steps and drives `learning_starts` /
///   `train_frequency`; `gradient_updates` counts **optimizer** steps and
///   drives the target-update cadence. The two units are deliberately distinct
///   (ADR 0059) — see [`gradient_updates`](Self::gradient_updates).
pub struct DqnAgent<B, M, O, A, const DO: usize, const DB: usize>
where
    B: AutodiffBackend,
    M: DqnModel<B, DB>,
    O: Observation<DO> + TensorConvertible<DO, B> + TensorConvertible<DO, B::InnerBackend>,
    A: DiscreteAction<1>,
{
    policy_net: Slot<M>,
    target_net: M::InnerModule,
    optimizer: OptimizerAdaptor<Adam, M, B>,
    buffer: ReplayKind<DiscreteTransition<O>>,
    exploration: EpsilonGreedy,
    config: DqnTrainingConfig,
    device: B::Device,
    step: usize,
    /// Gradient (optimizer) updates attempted so far — the unit
    /// `config.target_update`'s cadence counts (ADR 0059). Advanced
    /// unconditionally, including on a non-finite-loss skip.
    gradient_updates: usize,
    stats: AgentStats<DqnMetrics>,
    /// Non-finite-loss guard for the TD-loss site (ADR 0056, #318). One
    /// per-run `warn!` latch; the skip it drives fires every occurrence.
    loss_guard: FiniteLossGuard,
    _action: PhantomData<A>,
}

impl<B, M, O, A, const DO: usize, const DB: usize> std::fmt::Debug for DqnAgent<B, M, O, A, DO, DB>
where
    B: AutodiffBackend,
    M: DqnModel<B, DB>,
    O: Observation<DO> + TensorConvertible<DO, B> + TensorConvertible<DO, B::InnerBackend>,
    A: DiscreteAction<1>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DqnAgent")
            .field("step", &self.step)
            .field("gradient_updates", &self.gradient_updates)
            .field("buffer_len", &self.buffer.len())
            .field("epsilon", &self.exploration.value())
            .field("config", &self.config)
            .finish_non_exhaustive()
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
    ///
    /// The target network is initialized as a frozen copy of `policy_net` via
    /// [`AutodiffModule::valid`](burn::module::AutodiffModule::valid). Gradient
    /// clipping is applied to the Adam optimizer when
    /// [`DqnTrainingConfig::clip_grad`] is `Some`. The replay buffer is
    /// pre-allocated to `config.replay_buffer_capacity` entries, and the
    /// running-average statistics window is fixed at 100 episodes.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`](rlevo_core::config::ConfigError) if `config`
    /// fails [`DqnTrainingConfig::validate`](rlevo_core::config::Validate::validate).
    pub fn new(
        policy_net: M,
        config: DqnTrainingConfig,
        device: B::Device,
    ) -> Result<Self, rlevo_core::config::ConfigError> {
        config.validate()?;
        let target_net = policy_net.valid();
        let adam = config.optimizer.clone();
        let optimizer = match &config.clip_grad {
            Some(clip) => adam.with_grad_clipping(Some(clip.clone())).init::<B, M>(),
            None => adam.init::<B, M>(),
        };
        let exploration = EpsilonGreedy::from_config(&config);
        let stats = AgentStats::<DqnMetrics>::new(100);
        let buffer = match &config.prioritized_replay {
            None => ReplayKind::uniform(config.replay_buffer_capacity),
            Some(per) => ReplayKind::prioritized(per.buffer_config(config.replay_buffer_capacity))?,
        };
        Ok(Self {
            policy_net: Slot::new(policy_net),
            target_net,
            optimizer,
            buffer,
            exploration,
            config,
            device,
            step: 0,
            gradient_updates: 0,
            stats,
            loss_guard: FiniteLossGuard::new("dqn/loss"),
            _action: PhantomData,
        })
    }

    /// Current exploration rate (ε).
    pub fn epsilon(&self) -> f64 {
        self.exploration.value()
    }

    /// Configured optimiser learning rate (DQN uses a fixed rate, no annealing).
    pub fn learning_rate(&self) -> f64 {
        self.config.learning_rate
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

    /// Global **environment** step count.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Number of gradient (optimizer) updates attempted so far.
    ///
    /// This is the counter [`DqnTrainingConfig::target_update`]'s cadence is
    /// read against (ADR 0059), and it is *not*
    /// [`step`](Self::step) — the two differ by `train_frequency` and by the
    /// `learning_starts` warm-up. It advances once per [`learn_step`] that gets
    /// as far as computing a loss, **including** one the non-finite-loss guard
    /// then skips (ADR 0056 §3): counting only applied updates would make the
    /// target cadence a function of run health, stretching it exactly when a
    /// run is diverging.
    ///
    /// [`DqnTrainingConfig::target_update`]: crate::algorithms::dqn::dqn_config::DqnTrainingConfig::target_update
    /// [`learn_step`]: Self::learn_step
    pub fn gradient_updates(&self) -> usize {
        self.gradient_updates
    }

    /// Read-only view of the target network.
    ///
    /// The observation seam for the target-update rule: with it, a caller — or
    /// a test — can check *that* a target update fired on the expected gradient
    /// update and moved the weights by the expected τ. Issue #182's
    /// double-update defect survived its own test suite precisely because no
    /// such seam existed, so every assertion had to be made through Q-values,
    /// which are a lossy function of the weights.
    ///
    /// `pub`, and a shared borrow rather than a clone: `M::InnerModule` is the
    /// caller's own network type, so this hands back nothing the caller did not
    /// supply, and `&` cannot perturb agent state. Compare
    /// [`inference_net`](Self::inference_net), which already returns an owned
    /// snapshot of the *policy* side.
    pub fn target_net(&self) -> &M::InnerModule {
        &self.target_net
    }

    fn policy(&self) -> &M {
        self.policy_net.get()
    }

    /// ε-greedy action selection.
    ///
    /// With probability `ε` returns a uniformly random discrete action;
    /// otherwise runs the policy network on `obs` and returns the argmax.
    pub fn act<R: Rng + ?Sized>(&self, obs: &O, rng: &mut R) -> A {
        if self.exploration.should_explore(rng) {
            A::from_index(rng.random_range(0..A::ACTION_COUNT))
        } else {
            self.act_greedy(obs)
        }
    }

    /// Greedy (deterministic) action selection — the argmax over Q-values.
    ///
    /// Unlike [`act`](Self::act) this never explores, so it is the policy to
    /// use for evaluation: it reflects what the network has learned without the
    /// ε-greedy exploration noise that floors at `epsilon_end`.
    // Action indices only. `argmax` yields a non-negative index below
    // `A::ACTION_COUNT`, so the i64 -> usize narrowing can neither wrap nor lose a
    // sign; where an index round-trips through f32 it stays far below the 2^24
    // exact-integer limit. `from_index` bounds-checks on the way back.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn act_greedy(&self, obs: &O) -> A {
        let obs_t: Tensor<B, DO> = obs.to_tensor(&self.device);
        let batched: Tensor<B, DB> = obs_t.unsqueeze::<DB>();
        let q_values: Tensor<B, 2> = self.policy().forward(batched);
        let idx = q_values.argmax(1).into_scalar();
        A::from_index(idx.elem::<i64>() as usize)
    }

    /// Snapshots the policy network onto the inner (non-autodiff) backend.
    ///
    /// Returns a frozen inference handle for use with
    /// [`act_greedy_with`](Self::act_greedy_with). Action selection never needs
    /// gradients, so running it on the inner backend avoids the per-call
    /// autodiff graph construction that [`act_greedy`](Self::act_greedy)
    /// incurs. Snapshot once after training, then reuse across many steps —
    /// the snapshot goes stale if the policy is updated again.
    pub fn inference_net(&self) -> M::InnerModule {
        self.policy().valid()
    }

    /// Greedy action selection against a pre-snapshotted inner network.
    ///
    /// Equivalent to [`act_greedy`](Self::act_greedy) but runs on the
    /// non-autodiff backend via [`inference_net`](Self::inference_net), which
    /// is dramatically cheaper for repeated single-observation inference.
    // Action indices only. `argmax` yields a non-negative index below
    // `A::ACTION_COUNT`, so the i64 -> usize narrowing can neither wrap nor lose a
    // sign; where an index round-trips through f32 it stays far below the 2^24
    // exact-integer limit. `from_index` bounds-checks on the way back.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn act_greedy_with(&self, net: &M::InnerModule, obs: &O) -> A {
        let obs_t: Tensor<B::InnerBackend, DO> = obs.to_tensor(&self.device);
        let batched: Tensor<B::InnerBackend, DB> = obs_t.unsqueeze::<DB>();
        let q_values: Tensor<B::InnerBackend, 2> = M::forward_inner(net, batched);
        let idx = q_values.argmax(1).into_scalar();
        A::from_index(idx.elem::<i64>() as usize)
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
    ///
    /// [`Snapshot::is_terminated`]: rlevo_core::environment::Snapshot::is_terminated
    /// [`Snapshot::is_done`]: rlevo_core::environment::Snapshot::is_done
    pub fn remember(&mut self, obs: O, action: &A, reward: f32, next_obs: O, terminated: bool) {
        self.buffer.push(DiscreteTransition {
            obs,
            action: action.to_index(),
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

    /// Runs one learning step: samples a batch uniformly, computes the Huber
    /// loss against the Bellman target, and updates the policy network.
    ///
    /// The target network is updated here and nowhere else: once the loss is
    /// computed, [`gradient_updates`](Self::gradient_updates) advances and
    /// [`TargetUpdate::fires_at`] decides whether this update moves the target
    /// (ADR 0059).
    ///
    /// Returns `None` if the agent does not yet have enough transitions to
    /// form a batch, or if the computed loss is non-finite (NaN/±Inf): in that
    /// case the backward pass, optimizer step, target update, and PER writeback
    /// are all skipped and a one-shot `warn!` fires (ADR 0056, #318), so the
    /// caller keeps its last healthy reported metrics rather than folding a
    /// NaN into them. The gradient-update counter advances even then, so the
    /// target cadence does not drift on a diverging run.
    ///
    /// [`TargetUpdate::fires_at`]: crate::target::TargetUpdate::fires_at
    ///
    /// # Panics
    ///
    /// Panics if the replay buffer hands back an id that is no longer live.
    /// Sampling and lookup run under the same `&mut self`, so this can only
    /// fire if a `ReplayStrategy` implementation violates the contract that a
    /// freshly sampled id resolves.
    ///
    /// # Errors
    ///
    /// Returns [`DqnAgentError::Polyak`] if the target soft-update finds a
    /// parameter-topology mismatch between the policy and target networks (see
    /// [`polyak_update`](crate::utils::polyak_update)). Every in-tree target is
    /// cloned from its policy, so this cannot occur for agents built normally.
    // Config knobs are stored as f64 for ergonomics; every tensor in this crate is
    // f32. This is the intended narrowing point, and the values are hyperparameters
    // (rates, discounts, epsilons) where f32 has far more precision than the
    // schedules that produce them.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::too_many_lines
    )]
    pub fn learn_step<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
    ) -> Result<Option<LearnOutcome>, DqnAgentError> {
        if !self.can_learn() {
            return Ok(None);
        }
        let batch_size = self.config.batch_size;
        // β is only consulted by prioritized replay; uniform ignores it. When
        // PER is enabled, evaluate the annealing schedule at the current step.
        let beta = self
            .config
            .prioritized_replay
            .as_ref()
            .map_or(UNIFORM_REPLAY_BETA, |per| per.beta(self.step));
        // `can_learn()` above already established `buffer.len() >= batch_size`,
        // so the only variant `sample` can return here is unreachable; treat it
        // as a skipped step for safety.
        let Ok(batch) = self.buffer.sample(batch_size, beta, rng) else {
            return Ok(None);
        };

        let obs_shape = O::shape();
        let numel_per_obs: usize = obs_shape.iter().product();

        let mut obs_flat: Vec<f32> = Vec::with_capacity(batch_size * numel_per_obs);
        let mut next_flat: Vec<f32> = Vec::with_capacity(batch_size * numel_per_obs);
        let mut action_idxs: Vec<i64> = Vec::with_capacity(batch_size);
        let mut rewards: Vec<f32> = Vec::with_capacity(batch_size);
        let mut terminated: Vec<f32> = Vec::with_capacity(batch_size);

        for &id in batch.ids() {
            let t = self.buffer.get(id).expect("a freshly sampled id is live");
            // Stage host-side: `to_tensor` would upload each row only to read it
            // straight back -- one wgpu sync point per row, no op in between.
            t.obs.write_host_row(&mut obs_flat);
            t.next_obs.write_host_row(&mut next_flat);
            action_idxs.push(t.action as i64);
            rewards.push(t.reward);
            terminated.push(if t.terminated { 1.0 } else { 0.0 });
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
        let action_tensor: Tensor<B, 2, Int> = action_tensor_1.unsqueeze_dim::<2>(1);

        let rewards_inner: Tensor<B::InnerBackend, 1> =
            Tensor::from_data(TensorData::new(rewards, vec![batch_size]), &device);
        let terminated_inner: Tensor<B::InnerBackend, 1> =
            Tensor::from_data(TensorData::new(terminated, vec![batch_size]), &device);

        // --- Forward ---
        //
        // Everything from here to `step_with` runs against a borrow of the
        // network, so a panic in the forward pass, the target computation, the
        // loss, or `backward` leaves `policy_net` populated and the agent usable.
        let q_all: Tensor<B, 2> = self.policy().forward(obs_tensor);
        let q_mean = q_all.clone().mean().into_scalar().elem::<f32>();
        let q_pred: Tensor<B, 2> = q_all.gather(1, action_tensor);
        let q_pred_flat: Tensor<B, 1> = q_pred.squeeze_dim::<1>(1);

        // --- Target ---
        let next_q_target_inner: Tensor<B::InnerBackend, 2> =
            M::forward_inner(&self.target_net, next_tensor_inner.clone());
        let next_q_max_inner: Tensor<B::InnerBackend, 1> = if self.config.double_q {
            let next_q_policy_inner: Tensor<B::InnerBackend, 2> =
                M::forward_inner(&self.policy().valid(), next_tensor_inner);
            let next_actions: Tensor<B::InnerBackend, 2, Int> = next_q_policy_inner.argmax(1);
            next_q_target_inner
                .gather(1, next_actions)
                .squeeze_dim::<1>(1)
        } else {
            next_q_target_inner.max_dim(1).squeeze_dim::<1>(1)
        };

        let target_inner: Tensor<B::InnerBackend, 1> = compute_target_q_values(
            rewards_inner,
            next_q_max_inner,
            terminated_inner,
            self.config.gamma as f32,
        );
        let target: Tensor<B, 1> = Tensor::from_data(target_inner.into_data(), &device);

        // Per-sample `[batch]` Huber residual, reduced here rather than inside
        // `forward`, so an importance-sampling weight scales each sample before
        // the mean (ADR 0050 §14). At `w ≡ 1` (uniform replay) this is
        // bit-identical to `forward(.., Reduction::Mean)`, which burn-nn 0.21.0
        // implements as literally `forward_no_reduction(..).mean()`
        // (`loss/huber.rs:92-94`).
        let per_sample_loss = HuberLossConfig::new(1.0)
            .init()
            .forward_no_reduction(q_pred_flat.clone(), target.clone());
        let loss_tensor = reduce_weighted_loss(per_sample_loss, &batch, &device);
        let loss_value = loss_tensor.clone().into_scalar().elem::<f32>();

        // An optimizer step is now attempted, so the cadence counter advances —
        // unconditionally, BEFORE the non-finite-loss guard below (ADR 0059 §4,
        // matching SAC/DDPG/TD3). Counting only *applied* updates would make
        // the target-update rhythm a function of run health: a diverging run
        // emitting non-finite losses would advance the cadence more slowly
        // exactly when stability matters most.
        self.gradient_updates += 1;

        // #318 / ADR 0056: `loss_value` is already host-resident, so the
        // finiteness check costs no extra sync. A non-finite loss skips
        // `backward()`, the optimizer step, the target soft-update, and the PER
        // writeback (Burn would otherwise fold NaN into the weights silently),
        // and returns `None` — the train loop keeps its last healthy reported
        // loss/q-mean rather than advancing them with a NaN, and never counts
        // this as an applied update. The `warn!` is the surfacing mechanism.
        if !self.loss_guard.check(loss_value) {
            return Ok(None);
        }

        let grads = loss_tensor.backward();
        // `from_grads` takes `&M` and returns an owned, lifetime-free value, so
        // NLL ends the borrow here — the only window in which the module is out
        // of the slot is the `Optimizer::step` call inside `step_with`.
        let grads = GradientsParams::from_grads(grads, self.policy());
        self.policy_net
            .step_with(&mut self.optimizer, self.config.learning_rate, grads);

        // One target-update mechanism, gated on gradient updates (ADR 0058 /
        // 0059). `fires_at` yields the τ to apply on this update, or `None`.
        // A hard copy is the degenerate τ = 1.0, not a separate path.
        if let Some(tau) = self.config.target_update.fires_at(self.gradient_updates) {
            // Clone rather than move out: `soft_update` consumes `target` by
            // value, so on `Err` the `?` returns before this reassignment and
            // the field keeps its prior weights — no silent hard-sync (the
            // invariant now holds via early return, and equally for a panic).
            self.target_net = M::soft_update(self.policy(), self.target_net.clone(), tau)?;
        }

        // PER priority writeback (Schaul Alg. 1 lines 11-12): the DQN priority
        // signal is the per-sample TD error `δ = q_pred − target`; the buffer
        // applies `p = |δ| + ε`. A no-op for uniform replay, so gate on the
        // strategy to avoid an unnecessary host read. The writeback never enters
        // the target computation — `δ` is read here only after the gradient step.
        if self.buffer.is_prioritized() {
            let td = q_pred_flat - target;
            let td_host: Vec<f32> = td
                .into_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("finite TD errors read to host");
            if let Err(err) = self
                .buffer
                .update_priorities_from_td_errors(batch.ids(), &td_host)
            {
                tracing::warn!(
                    ?err,
                    "skipping PER priority writeback: non-finite TD error (diverging network)"
                );
            }
        }

        Ok(Some(LearnOutcome {
            loss: loss_value,
            q_mean,
        }))
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

    use burn::backend::{Autodiff, Flex};
    use burn::module::{AutodiffModule, Module, ModuleMapper, Param};
    use burn::nn::{Linear, LinearConfig};
    use burn::tensor::backend::Backend;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rlevo_core::base::Action;
    use rlevo_core::base::{HostRow, TensorConversionError};
    use serde::{Deserialize, Serialize};

    use crate::algorithms::dqn::dqn_config::DqnTrainingConfigBuilder;
    use crate::replay::PrioritizedReplaySettings;
    use crate::target::TargetUpdate;
    use crate::utils::polyak_update;

    type TestBackend = Autodiff<Flex>;
    type TestInner = <TestBackend as AutodiffBackend>::InnerBackend;
    type TestAgent = DqnAgent<TestBackend, TestNet<TestBackend>, TestObs, TestAction, 1, 2>;

    /// Minimal two-in/two-out linear Q-network used by the target-sync tests.
    ///
    /// Weights are set to a caller-chosen constant so "policy differs from
    /// target" is provable by inspection rather than by luck of the seed.
    #[derive(Module, Debug)]
    struct TestNet<B: Backend> {
        linear: Linear<B>,
    }

    impl<B: Backend> TestNet<B> {
        /// Builds a bias-free 2x2 linear layer whose every weight equals `value`.
        fn constant(
            device: &<B as burn::tensor::backend::BackendTypes>::Device,
            value: f32,
        ) -> Self {
            let linear: Linear<B> = LinearConfig::new(2, 2).with_bias(false).init(device);
            let weight: Tensor<B, 2> =
                Tensor::from_data(TensorData::new(vec![value; 4], vec![2, 2]), device);
            Self {
                linear: Linear {
                    weight: Param::from_tensor(weight),
                    ..linear
                },
            }
        }
    }

    /// Reads a network's weight tensor back to the host for exact comparison.
    fn weights<B: Backend>(net: &TestNet<B>) -> Vec<f32> {
        net.linear
            .weight
            .val()
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("weight tensor is float data")
    }

    impl<B: AutodiffBackend> DqnModel<B, 2> for TestNet<B> {
        fn forward(&self, observations: Tensor<B, 2>) -> Tensor<B, 2> {
            self.linear.forward(observations)
        }

        fn forward_inner(
            inner: &Self::InnerModule,
            observations: Tensor<B::InnerBackend, 2>,
        ) -> Tensor<B::InnerBackend, 2> {
            inner.linear.forward(observations)
        }

        #[allow(clippy::cast_possible_truncation)]
        fn soft_update(
            active: &Self,
            target: Self::InnerModule,
            tau: f64,
        ) -> Result<Self::InnerModule, PolyakError> {
            polyak_update::<B::InnerBackend, TestNet<B::InnerBackend>>(
                &active.valid(),
                target,
                tau as f32,
            )
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct TestObs([f32; 2]);

    impl Observation<1> for TestObs {
        fn shape() -> [usize; 1] {
            [2]
        }
    }

    impl HostRow<1> for TestObs {
        fn row_shape() -> [usize; 1] {
            [2]
        }

        fn write_host_row(&self, buf: &mut Vec<f32>) {
            buf.extend_from_slice(&self.0);
        }
    }

    impl<B: Backend> TensorConvertible<1, B> for TestObs {
        fn from_tensor(tensor: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
            let data = tensor.into_data().convert::<f32>();
            let v = data.as_slice::<f32>().map_err(|_| TensorConversionError {
                message: "non-float tensor".into(),
            })?;
            Ok(Self([v[0], v[1]]))
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct TestAction(usize);

    impl Action<1> for TestAction {
        fn shape() -> [usize; 1] {
            [2]
        }

        fn is_valid(&self) -> bool {
            self.0 < 2
        }
    }

    impl DiscreteAction<1> for TestAction {
        const ACTION_COUNT: usize = 2;

        fn from_index(index: usize) -> Self {
            assert!(index < Self::ACTION_COUNT, "action index out of range");
            Self(index)
        }

        fn to_index(&self) -> usize {
            self.0
        }
    }

    /// Builds a prioritized-replay agent primed with four transitions and
    /// `learning_starts = 0`, so a single `learn_step` runs immediately.
    // Test fixture data: the loop counter and element count are bounded by small
    // constants declared in this test, far below f32's 2^24 exact-integer limit,
    // so every generated value is represented exactly.
    #[allow(clippy::cast_precision_loss)]
    fn primed_prioritized_agent() -> TestAgent {
        let device = <TestInner as burn::tensor::backend::BackendTypes>::Device::default();
        let config = DqnTrainingConfigBuilder::new()
            .batch_size(2)
            .learning_starts(0)
            .learning_rate(0.05)
            .prioritized_replay(PrioritizedReplaySettings {
                beta_anneal_steps: 100,
                ..PrioritizedReplaySettings::default()
            })
            .build()
            .expect("valid prioritized config");
        let policy: TestNet<TestBackend> = TestNet::constant(&device, 0.5);
        let mut agent = TestAgent::new(policy, config, device).expect("valid config");
        for i in 0..4usize {
            let x = i as f32;
            agent.remember(
                TestObs([x, -x]),
                &TestAction(i % 2),
                1.0,
                TestObs([x + 1.0, -x]),
                false,
            );
        }
        agent
    }

    #[test]
    fn test_dqn_defaults_to_uniform_replay() {
        let device = <TestInner as burn::tensor::backend::BackendTypes>::Device::default();
        let agent = TestAgent::new(
            TestNet::constant(&device, 0.5),
            DqnTrainingConfig::default(),
            device,
        )
        .expect("default config is valid");
        assert!(
            !agent.buffer.is_prioritized(),
            "the default config must keep uniform replay (PER is opt-in)"
        );
    }

    #[test]
    fn test_dqn_prioritized_opt_in_selects_prioritized_replay() {
        let agent = primed_prioritized_agent();
        assert!(
            agent.buffer.is_prioritized(),
            "a Some(prioritized_replay) config must select the prioritized buffer"
        );
    }

    /// The feedback edge: after a learn step, the sampled transitions' priorities
    /// have been rewritten from the initial running-max seed, so the total
    /// sampling mass moves. A uniform buffer's `learn_step` cannot do this.
    #[test]
    fn test_dqn_priority_writeback_runs_after_learn_step() {
        let mut agent = primed_prioritized_agent();
        let before = agent
            .buffer
            .as_prioritized()
            .expect("prioritized")
            .total_priority();
        let mut rng = StdRng::seed_from_u64(7);
        agent
            .learn_step(&mut rng)
            .expect("no polyak error")
            .expect("a primed agent with learning_starts = 0 learns");
        let after = agent
            .buffer
            .as_prioritized()
            .expect("prioritized")
            .total_priority();
        assert!(
            (after - before).abs() > 1e-9,
            "the |δ| priority writeback must change the sampling mass: {before} -> {after}"
        );
    }

    #[test]
    fn metrics_performance_record_returns_reward_and_steps() {
        let m = DqnMetrics {
            reward: 42.0,
            steps: 7,
            policy_loss: 0.5,
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

    // -------- non-finite-loss guard (ADR 0056, #318) --------

    /// Replaces every float parameter of a module with `NaN`, simulating a
    /// policy network that has diverged to non-finite weights — the realistic
    /// source of a non-finite TD loss. Applied to a *clone* so the live net's
    /// `ParamId`s are preserved and the target pairing stays valid.
    struct NanInjector;

    impl<B: Backend> ModuleMapper<B> for NanInjector {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
            let (id, tensor, mapper) = param.consume();
            Param::from_mapped_value(id, tensor.mul_scalar(f32::NAN), mapper)
        }
    }

    /// Builds a uniform-replay agent primed with four transitions and
    /// `learning_starts = 0`, so each `learn_step` runs immediately. `rule` is
    /// the target-update rule under test; the target network is `policy.valid()`
    /// (built by `DqnAgent::new`), so its `ParamId`s match and `soft_update`
    /// can pair them.
    // Test fixture data: the loop counter and element count are bounded by small
    // constants declared in this test, far below f32's 2^24 exact-integer limit.
    #[allow(clippy::cast_precision_loss)]
    fn primed_uniform_agent_with(rule: TargetUpdate) -> TestAgent {
        let device = <TestInner as burn::tensor::backend::BackendTypes>::Device::default();
        let config = DqnTrainingConfigBuilder::new()
            .batch_size(2)
            .learning_starts(0)
            .learning_rate(0.05)
            .target_update(rule)
            .build()
            .expect("valid config");
        let policy: TestNet<TestBackend> = TestNet::constant(&device, 0.5);
        let mut agent = TestAgent::new(policy, config, device).expect("valid config");
        for i in 0..4usize {
            let x = i as f32;
            agent.remember(
                TestObs([x, -x]),
                &TestAction(i % 2),
                1.0,
                TestObs([x + 1.0, -x]),
                false,
            );
        }
        agent
    }

    /// The default-configured fixture: `polyak(0.005, 1)`, i.e. a soft update
    /// on every gradient step, so a skipped step is provable by the target
    /// network staying untouched.
    fn primed_uniform_agent() -> TestAgent {
        let agent = primed_uniform_agent_with(DqnTrainingConfig::default().target_update);
        assert_eq!(
            agent.config.target_update,
            TargetUpdate::polyak(0.005, 1),
            "the soft target update must be live on every gradient update"
        );
        agent
    }

    /// A non-finite TD loss must skip the whole learn step: `backward`, the
    /// optimizer step, and the soft target sync (ADR 0056, #318). Diverging the
    /// policy net to NaN forces a NaN loss; the guard must fire, `learn_step`
    /// must return `None`, the target must stay untouched and finite, and the
    /// agent must remain usable.
    #[test]
    fn dqn_nonfinite_loss_skips_step_and_warns() {
        let mut agent = primed_uniform_agent();
        // The target net is the healthy sibling: a skipped step leaves it intact.
        let target_before = weights(&agent.target_net);

        // Poison a *clone* of the live policy so its `ParamId`s are preserved.
        let poisoned = agent.policy().clone().map(&mut NanInjector);
        agent.policy_net = Slot::new(poisoned);

        let mut rng = StdRng::seed_from_u64(0);
        let outcome = agent.learn_step(&mut rng).expect("no polyak error");

        assert!(
            outcome.is_none(),
            "a non-finite TD loss must skip the step and return None"
        );
        assert!(
            agent.loss_guard.warning_fired(),
            "the non-finite TD loss must fire the guard"
        );
        assert_eq!(
            weights(&agent.target_net),
            target_before,
            "the soft target update must be skipped, leaving the target untouched"
        );
        assert!(
            weights(&agent.target_net).iter().all(|w| w.is_finite()),
            "the target net must stay finite — the policy NaN must not reach it"
        );
        // The agent is still usable: action selection returns a valid action.
        let action = agent.act(&TestObs([0.2, -0.2]), &mut rng);
        assert!(
            action.is_valid(),
            "act must still return a valid action after a skipped step"
        );
    }

    /// Agent-level guard for ADR 0057 / issue #341: when the target soft-update
    /// fails inside `learn_step` because the target network carries independent
    /// `ParamId`s, the agent must (a) surface the failure as
    /// `DqnAgentError::Polyak(PolyakError::MissingActive(_))` and (b) leave its
    /// `target_net` field byte-identical — no silent hard-sync onto the live
    /// network. The unit tests in `utils.rs` prove `polyak_update` itself
    /// returns `Err` without mutating; this proves the agent propagates that
    /// `Err` through `?` *before* the `self.target_net = …` reassignment.
    #[test]
    fn dqn_soft_update_err_leaves_target_untouched() {
        let mut agent = primed_uniform_agent();
        assert_eq!(
            agent.config.target_update.every(),
            1,
            "the target update must fire on this learn step for this path to run"
        );

        // Inject a target built independently of the policy: `TestNet::constant`
        // mints fresh `ParamId`s, so its weight id has no counterpart in the
        // policy net and `polyak_update` must reject it — a *topology* mismatch,
        // not merely a value divergence.
        let device = <TestInner as burn::tensor::backend::BackendTypes>::Device::default();
        let foreign_target: TestNet<TestInner> = TestNet::constant(&device, 2.0);
        agent.target_net = foreign_target;

        // Byte-for-byte snapshot of the target field before the failed step.
        let target_before = weights(&agent.target_net);

        let mut rng = StdRng::seed_from_u64(0);
        let err = agent
            .learn_step(&mut rng)
            .expect_err("an independently-minted target must fail the soft update");

        assert!(
            matches!(err, DqnAgentError::Polyak(PolyakError::MissingActive(_))),
            "a foreign target param must surface as \
             DqnAgentError::Polyak(PolyakError::MissingActive(_)); got {err:?}"
        );
        assert_eq!(
            weights(&agent.target_net),
            target_before,
            "a failed soft update must leave the target field untouched — no silent hard-sync"
        );
    }

    // -------- target-update cadence (ADR 0058 / 0059) --------
    //
    // These four replace the pair of `sync_target` tests that pinned issue
    // #182's two-mechanism gate (`sync_target_is_noop_when_tau_positive` /
    // `..._hard_copies_when_tau_zero`). `sync_target` is gone: the cadence gate
    // now lives inside `learn_step`, so the same properties are asserted
    // against gradient updates instead of env steps, and `TargetUpdate::hard(n)`
    // expresses what `tau = 0.0, target_update_frequency = n` used to.
    //
    // All four assert on parameter tensors, via `target_net()`, not on
    // Q-values: a greedy action is a lossy argmax of the weights and would
    // agree under both correct and defective behaviour — the reason the #182
    // defect survived its first test suite.

    /// The behaviour-preserving default: at `polyak(0.005, 1)` the target moves
    /// on **every** learn step, by exactly τ toward the post-step policy.
    #[test]
    fn dqn_polyak_default_moves_target_on_every_learn_step() {
        let mut agent = primed_uniform_agent();
        let tau = 0.005_f32;
        let mut rng = StdRng::seed_from_u64(3);

        for update in 1..=3_usize {
            let target_before = weights(agent.target_net());
            agent
                .learn_step(&mut rng)
                .expect("no polyak error")
                .expect("a primed agent with learning_starts = 0 learns");
            assert_eq!(
                agent.gradient_updates(),
                update,
                "one attempted optimizer step is one gradient update"
            );

            let policy_after = weights(agent.policy());
            let target_after = weights(agent.target_net());
            for ((&before, &policy), &after) in target_before
                .iter()
                .zip(policy_after.iter())
                .zip(target_after.iter())
            {
                let expected = (1.0 - tau).mul_add(before, tau * policy);
                assert!(
                    (after - expected).abs() < 1e-6,
                    "update {update}: target must move by exactly τ toward the \
                     post-step policy: got {after}, want {expected}"
                );
            }
            assert_ne!(
                target_after, target_before,
                "update {update}: τ = 0.005 must actually move the target"
            );
        }
    }

    /// At `hard(n)` the target is frozen between firings and becomes an exact
    /// copy of the policy at one. This is the property the old
    /// `tau = 0.0` + `target_update_frequency = n` pair expressed, now counted
    /// in gradient updates rather than env steps.
    #[test]
    fn dqn_hard_cadence_holds_target_between_firings_then_copies() {
        let mut agent = primed_uniform_agent_with(TargetUpdate::hard(3));
        let mut rng = StdRng::seed_from_u64(4);
        let initial = weights(agent.target_net());

        // Updates 1 and 2 are not multiples of 3: the target must not move.
        for update in 1..=2_usize {
            agent
                .learn_step(&mut rng)
                .expect("no polyak error")
                .expect("a primed agent learns");
            assert_eq!(agent.gradient_updates(), update);
            assert_eq!(
                weights(agent.target_net()),
                initial,
                "update {update} is not a multiple of 3 — the target must be untouched"
            );
        }
        assert_ne!(
            weights(agent.policy()),
            initial,
            "precondition: the policy must have moved, or the copy below is vacuous"
        );

        // Update 3 fires: τ = 1.0 degenerates the blend to a copy.
        agent
            .learn_step(&mut rng)
            .expect("no polyak error")
            .expect("a primed agent learns");
        assert_eq!(agent.gradient_updates(), 3);
        assert_eq!(
            weights(agent.target_net()),
            weights(agent.policy()),
            "at a firing, hard(3) must copy the post-step policy exactly"
        );
    }

    /// ADR 0059 §4, the subtle one: the counter must advance even when the
    /// non-finite-loss guard skips the optimizer step, or a diverging run
    /// silently stretches the target cadence. Pinned by driving one skipped
    /// step and one healthy step through `hard(2)`: the skip consumes update 1,
    /// so the healthy step lands on update 2 and fires.
    #[test]
    fn dqn_gradient_counter_advances_through_a_nonfinite_loss_skip() {
        let mut agent = primed_uniform_agent_with(TargetUpdate::hard(2));
        let healthy_policy = agent.policy().clone();
        let target_before = weights(agent.target_net());
        let mut rng = StdRng::seed_from_u64(5);

        // Poison a *clone* of the live policy so its `ParamId`s are preserved.
        let poisoned = agent.policy().clone().map(&mut NanInjector);
        agent.policy_net = Slot::new(poisoned);
        assert!(
            agent
                .learn_step(&mut rng)
                .expect("no polyak error")
                .is_none(),
            "a non-finite loss must skip the step"
        );
        assert_eq!(
            agent.gradient_updates(),
            1,
            "the counter must advance on a skipped step (ADR 0059 §4) — gating it \
             on a successful step would let the cadence drift on a diverging run"
        );
        assert_eq!(
            weights(agent.target_net()),
            target_before,
            "update 1 is not a multiple of 2, and the step was skipped anyway"
        );

        // Restore a healthy policy; the next attempt is update 2, which fires.
        agent.policy_net = Slot::new(healthy_policy);
        agent
            .learn_step(&mut rng)
            .expect("no polyak error")
            .expect("a healthy primed agent learns");
        assert_eq!(agent.gradient_updates(), 2);
        assert_eq!(
            weights(agent.target_net()),
            weights(agent.policy()),
            "the skipped attempt still consumed update 1, so the healthy step is \
             update 2 and hard(2) fires on it"
        );
    }

    /// The counter is *not* the env-step counter: `on_env_step` must not move
    /// it, and a `learn_step` that cannot learn must not either. If these two
    /// drifted together the ADR 0059 unit change would be invisible.
    #[test]
    fn dqn_gradient_counter_is_not_the_env_step_counter() {
        let mut agent = primed_uniform_agent();
        for _ in 0..5 {
            agent.on_env_step();
        }
        assert_eq!(agent.step(), 5);
        assert_eq!(
            agent.gradient_updates(),
            0,
            "environment steps must not advance the gradient-update counter"
        );

        // Starve the buffer so `can_learn()` is false: no attempted update.
        let mut starved = primed_uniform_agent_with(TargetUpdate::polyak(0.005, 1));
        starved.config.batch_size = 1_000;
        let mut rng = StdRng::seed_from_u64(6);
        assert!(
            starved
                .learn_step(&mut rng)
                .expect("no polyak error")
                .is_none()
        );
        assert_eq!(
            starved.gradient_updates(),
            0,
            "a learn step that never reaches a loss is not an attempted update"
        );
    }
}
