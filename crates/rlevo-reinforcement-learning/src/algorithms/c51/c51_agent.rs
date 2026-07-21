//! Categorical DQN (C51) agent: actor, trainer, and replay buffer.
//!
//! The [`C51Agent`] struct owns the policy network (returning atom logits),
//! a frozen target network, an Adam optimizer, a uniform replay buffer, and
//! the ε-greedy exploration schedule shared with DQN. Action selection uses
//! the expected Q-value `Σ_i z_i · softmax(logits_i)`; the learning step
//! projects the bootstrap distribution onto the fixed atom support (see
//! [`crate::algorithms::c51::projection`]) and minimises the categorical
//! cross-entropy against the policy's log-probabilities.

use std::marker::PhantomData;

use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, GradientsParams};
use burn::tensor::activation;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{ElementConversion, Int, Tensor, TensorData};
use rand::{Rng, RngExt};

use crate::metrics::{AgentStats, PerformanceRecord};
use crate::replay::{DiscreteTransition, ReplayBufferError, ReplayKind, ReplayStrategy};
use rlevo_core::action::DiscreteAction;
use rlevo_core::base::{Observation, TensorConvertible};
use rlevo_core::config::Validate;

use crate::algorithms::c51::c51_config::C51TrainingConfig;
use crate::algorithms::c51::c51_model::C51Model;
use crate::algorithms::c51::loss::{
    categorical_cross_entropy_per_sample, categorical_kl_per_sample,
};
use crate::algorithms::c51::projection::project_distribution;
use crate::algorithms::dqn::exploration::EpsilonGreedy;
use crate::algorithms::shared::{FiniteLossGuard, Slot, UNIFORM_REPLAY_BETA, reduce_weighted_loss};

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
/// - `target_net` lives on `B::InnerBackend` (the non-autodiff backend) so that
///   computing bootstrap distributions never builds an autodiff graph.
pub struct C51Agent<B, M, O, A, const DO: usize, const DB: usize>
where
    B: AutodiffBackend,
    M: C51Model<B, DB>,
    O: Observation<DO> + TensorConvertible<DO, B> + TensorConvertible<DO, B::InnerBackend>,
    A: DiscreteAction<1>,
{
    policy_net: Slot<M>,
    target_net: M::InnerModule,
    optimizer: OptimizerAdaptor<Adam, M, B>,
    buffer: ReplayKind<DiscreteTransition<O>>,
    exploration: EpsilonGreedy,
    config: C51TrainingConfig,
    device: B::Device,
    step: usize,
    stats: AgentStats<C51Metrics>,
    /// Non-finite-loss guard for the cross-entropy loss site (ADR 0056, #318).
    /// One per-run `warn!` latch; the skip it drives fires every occurrence.
    loss_guard: FiniteLossGuard,
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
            .finish_non_exhaustive()
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
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`](rlevo_core::config::ConfigError) if `config`
    /// fails [`C51TrainingConfig::validate`](rlevo_core::config::Validate::validate)
    /// (e.g. `v_min >= v_max`).
    pub fn new(
        policy_net: M,
        config: C51TrainingConfig,
        device: B::Device,
    ) -> Result<Self, rlevo_core::config::ConfigError> {
        config.validate()?;
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
            stats,
            loss_guard: FiniteLossGuard::new("c51/loss"),
            _action: PhantomData,
        })
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
        self.policy_net.get()
    }

    /// Builds a fresh support tensor `[z_0, z_1, …, z_{N-1}]` on the
    /// requested backend and device. Small (N=51 default), so we prefer
    /// allocating on demand over storing a cached copy per backend flavour.
    // Structural size of the distributional support (atom / quantile count). The
    // configs cap these in the low hundreds, so the value is exact in f32; a
    // non-finite or zero spacing is rejected by assertion before use.
    #[allow(clippy::cast_precision_loss)]
    fn build_support<BK: burn::tensor::backend::Backend>(
        &self,
        device: &<BK as burn::tensor::backend::BackendTypes>::Device,
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
        self.act_greedy(obs)
    }

    /// Greedy (deterministic) action selection — the argmax over expected
    /// return computed from the predicted distribution.
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
        let logits: Tensor<B, 3> = self.policy().forward(batched); // (1, A, N)
        let probs: Tensor<B, 3> = activation::softmax(logits, 2);
        let support: Tensor<B, 1> = self.build_support::<B>(&self.device);
        let support_3d: Tensor<B, 3> = support.unsqueeze::<3>(); // (1, 1, N)
        let q: Tensor<B, 2> = (probs * support_3d).sum_dim(2).squeeze_dim::<2>(2); // (1, A)
        let idx = q.argmax(1).into_scalar();
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

    /// Builds the fixed atom-support vector on the inner backend, once.
    ///
    /// Pair with [`act_greedy_with`](Self::act_greedy_with) so the support is
    /// uploaded to the device a single time rather than rebuilt every step.
    pub fn inference_support(&self) -> Tensor<B::InnerBackend, 1> {
        self.build_support::<B::InnerBackend>(&self.device)
    }

    /// Greedy action selection against a pre-snapshotted inner network.
    ///
    /// Equivalent to [`act_greedy`](Self::act_greedy) but runs on the
    /// non-autodiff backend via [`inference_net`](Self::inference_net) and
    /// reuses the [`inference_support`](Self::inference_support) tensor, which
    /// is dramatically cheaper for repeated single-observation inference.
    // Action indices only. `argmax` yields a non-negative index below
    // `A::ACTION_COUNT`, so the i64 -> usize narrowing can neither wrap nor lose a
    // sign; where an index round-trips through f32 it stays far below the 2^24
    // exact-integer limit. `from_index` bounds-checks on the way back.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn act_greedy_with(
        &self,
        net: &M::InnerModule,
        support: &Tensor<B::InnerBackend, 1>,
        obs: &O,
    ) -> A {
        let obs_t: Tensor<B::InnerBackend, DO> = obs.to_tensor(&self.device);
        let batched: Tensor<B::InnerBackend, DB> = obs_t.unsqueeze::<DB>();
        let logits: Tensor<B::InnerBackend, 3> = M::forward_inner(net, batched); // (1, A, N)
        let probs: Tensor<B::InnerBackend, 3> = activation::softmax(logits, 2);
        let support_3d: Tensor<B::InnerBackend, 3> = support.clone().unsqueeze::<3>(); // (1, 1, N)
        let q: Tensor<B::InnerBackend, 2> = (probs * support_3d).sum_dim(2).squeeze_dim::<2>(2); // (1, A)
        let idx = q.argmax(1).into_scalar();
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

    /// Synchronises the target network with the current policy network.
    ///
    /// This method is **only ever the hard-sync path**: a full copy of the
    /// policy weights into the target. The soft (Polyak) path lives inside
    /// [`learn_step`](Self::learn_step), not here.
    ///
    /// The two mechanisms are mutually exclusive and selected by
    /// [`C51TrainingConfig::tau`]:
    ///
    /// - `tau > 0` — Polyak soft updates `target ← (1 − τ) · target + τ ·
    ///   policy` run per learn step inside [`learn_step`](Self::learn_step),
    ///   and this method is a **no-op**, whatever `target_update_frequency`
    ///   says. A hard copy here would erase the very lag `tau` exists to
    ///   maintain.
    /// - `tau == 0` — soft updates are disabled and this method performs the
    ///   hard sync every `target_update_frequency` steps. If
    ///   `target_update_frequency` is `0`, hard sync is disabled too; that
    ///   combination leaves the target frozen forever and is rejected by
    ///   [`C51TrainingConfig::validate`](rlevo_core::config::Validate::validate).
    ///
    /// Safe to call unconditionally once per env step — the gating lives here.
    pub fn sync_target(&mut self) {
        // `tau > 0` means the soft update in `learn_step` owns the target;
        // `target_update_frequency` is an inert fallback in that mode.
        if self.config.tau > 0.0 {
            return;
        }
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

    /// Samples a minibatch, projects the bootstrap distribution, and
    /// performs one gradient update on the policy network.
    ///
    /// The sequence of operations:
    ///
    /// 1. Sample `batch_size` transitions uniformly from the replay buffer.
    /// 2. Run the frozen target network on next-observations (inner backend,
    ///    no autodiff graph) and pick the greedy bootstrap action via
    ///    `argmax E[Z]` (the double-DQN selection step).
    /// 3. Project the bootstrap distribution through the Bellman operator
    ///    onto the fixed support via
    ///    [`project_distribution`].
    /// 4. Run the policy network on current-observations (autodiff backend).
    /// 5. Compute categorical cross-entropy between the projected target and
    ///    the policy's log-softmax probabilities at the taken action.
    /// 6. Back-propagate and apply an Adam gradient step.
    /// 7. If `tau > 0`, perform a Polyak soft update of the target network.
    ///
    /// # Returns
    ///
    /// `Some(LearnOutcome)` with loss, mean Q-value, and distribution entropy
    /// when a gradient step was taken. Returns `None` without side-effects
    /// when [`can_learn`](Self::can_learn) is false (buffer too small or
    /// step count below `learning_starts`), and also when the computed loss is
    /// non-finite (NaN/±Inf): in that case the backward pass, optimizer step,
    /// target sync, and PER writeback are all skipped and a one-shot `warn!`
    /// fires (ADR 0056, #318), so the caller keeps its last healthy reported
    /// metrics rather than folding a NaN into them.
    ///
    /// # Panics
    ///
    /// Panics if the agent was poisoned by a panic *inside* a previous
    /// optimizer step — the one window in which the network is out of its
    /// [`Slot`]. Such an agent cannot be recovered and must be rebuilt; see
    /// [`Slot`] for why. Steps 1–6 above all run against a borrow of the
    /// network, so a panic in any of them (a shape mismatch, a device error, a
    /// failed host read) leaves the agent fully usable.
    // The body is one linear pipeline — sample, forward, loss, backward,
    // optimizer step, priority writeback, metrics — with a borrow structure
    // around the module slot that the inline comments below depend on. Splitting
    // it into helpers to satisfy the line count would thread that borrow through
    // signatures without making the sequence easier to follow.
    #[allow(clippy::too_many_lines)]
    // Config knobs are stored as f64 for ergonomics; every tensor in this crate is
    // f32. This is the intended narrowing point, and the values are hyperparameters
    // (rates, discounts, epsilons) where f32 has far more precision than the
    // schedules that produce them.
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    pub fn learn_step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Option<LearnOutcome> {
        if !self.can_learn() {
            return None;
        }
        let batch_size = self.config.batch_size;
        let num_atoms = self.config.num_atoms;

        // β is only consulted by prioritized replay; uniform ignores it.
        let beta = self
            .config
            .prioritized_replay
            .as_ref()
            .map_or(UNIFORM_REPLAY_BETA, |per| per.beta(self.step));
        // `can_learn()` above already established `buffer.len() >= batch_size`,
        // so the only variant `sample` can return here is unreachable.
        let batch = self.buffer.sample(batch_size, beta, rng).ok()?;

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

        let rewards_inner: Tensor<B::InnerBackend, 1> =
            Tensor::from_data(TensorData::new(rewards, vec![batch_size]), &device);
        let terminated_inner: Tensor<B::InnerBackend, 1> =
            Tensor::from_data(TensorData::new(terminated, vec![batch_size]), &device);

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
            terminated_inner,
            support_inner,
            self.config.gamma as f32,
            self.config.v_min,
            self.config.v_max,
            num_atoms,
        );

        // --- Policy forward (autodiff) ---
        //
        // Everything from here to `step_with` runs against a borrow of the
        // network, so a panic in the forward pass, the loss, or `backward`
        // leaves `policy_net` populated and the agent usable.
        let logits: Tensor<B, 3> = self.policy().forward(obs_tensor); // (B, A, N)

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
        // Per-sample `[batch]` cross-entropy, scaled by importance weights before
        // the mean (ADR 0050 §14). The **loss** stays CE — the KL correction is
        // only the priority signal below.
        let per_sample_ce =
            categorical_cross_entropy_per_sample(target_autodiff.clone(), pred_log_p.clone());
        let loss_tensor = reduce_weighted_loss(per_sample_ce, &batch, &device);
        let loss_value = loss_tensor.clone().into_scalar().elem::<f32>();

        // #318 / ADR 0056: `loss_value` is already host-resident, so the
        // finiteness check costs no extra sync. A non-finite loss skips
        // `backward()`, the optimizer step, the target soft-update, and the PER
        // writeback (Burn would otherwise fold NaN into the weights silently),
        // and returns `None` — the train loop keeps its last healthy reported
        // metrics rather than advancing them with a NaN, and never counts this
        // as an applied update. The `warn!` is the surfacing mechanism.
        if !self.loss_guard.check(loss_value) {
            return None;
        }

        let grads = loss_tensor.backward();
        // `from_grads` takes `&M` and returns an owned, lifetime-free value, so
        // NLL ends the borrow here — the only window in which the module is out
        // of the slot is the `Optimizer::step` call inside `step_with`.
        let grads = GradientsParams::from_grads(grads, self.policy());
        self.policy_net
            .step_with(&mut self.optimizer, self.config.learning_rate, grads);

        if self.config.tau > 0.0 {
            // Clone rather than move out: the field stays intact if
            // `soft_update` panics, so a failure can't silently hard-sync
            // the target onto the policy.
            self.target_net =
                M::soft_update(self.policy(), self.target_net.clone(), self.config.tau);
        }

        // PER priority writeback (Schaul Alg. 1 lines 11-12): the C51 priority
        // signal is the **KL divergence** `D_KL(target ‖ pred)`, not the
        // cross-entropy above. They differ by the per-sample target entropy
        // `H(target)` — constant in θ (so the gradient is unchanged) but varying
        // per sample (so replay ranking differs). Rainbow prioritizes by KL "since
        // this is what the algorithm is minimizing". A no-op for uniform replay.
        if self.buffer.is_prioritized() {
            let kl = categorical_kl_per_sample(target_autodiff, pred_log_p);
            let kl_host: Vec<f32> = kl
                .into_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("finite KL priorities read to host");
            if let Err(err) = self
                .buffer
                .update_priorities_from_td_errors(batch.ids(), &kl_host)
            {
                tracing::warn!(
                    ?err,
                    "skipping PER priority writeback: non-finite KL priority (diverging network)"
                );
            }
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
    use serde::{Deserialize, Serialize};

    use crate::algorithms::c51::c51_config::C51TrainingConfigBuilder;
    use crate::replay::PrioritizedReplaySettings;
    use crate::utils::polyak_update;
    use rlevo_core::base::{Action as ActionTrait, HostRow, TensorConversionError};

    type Be = Autodiff<Flex>;

    const TEST_ACTIONS: usize = 2;
    const TEST_ATOMS: usize = 5;

    /// Two-feature observation — the smallest thing the agent will accept.
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
            let v = tensor
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .map_err(|e| TensorConversionError {
                    message: format!("host read failed: {e:?}"),
                })?;
            Ok(Self([v[0], v[1]]))
        }
    }

    /// Binary discrete action.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct TestAction(usize);

    impl ActionTrait<1> for TestAction {
        fn shape() -> [usize; 1] {
            [TEST_ACTIONS]
        }
        fn is_valid(&self) -> bool {
            self.0 < TEST_ACTIONS
        }
    }

    impl DiscreteAction<1> for TestAction {
        const ACTION_COUNT: usize = TEST_ACTIONS;
        fn from_index(index: usize) -> Self {
            assert!(index < TEST_ACTIONS, "action index out of bounds");
            Self(index)
        }
        fn to_index(&self) -> usize {
            self.0
        }
    }

    /// Minimal trainable C51 network.
    ///
    /// The backend generic **must** be named `B`: Burn's `Module` derive keys
    /// off the identifier, and any other name makes the derive treat the struct
    /// as parameter-free — the optimizer would then step nothing and every
    /// "weights moved" assertion below would pass vacuously.
    #[derive(Module, Debug)]
    struct TestNet<B: Backend> {
        linear: Linear<B>,
    }

    impl<B: Backend> TestNet<B> {
        fn init(device: &B::Device) -> Self {
            Self {
                linear: LinearConfig::new(2, TEST_ACTIONS * TEST_ATOMS).init(device),
            }
        }

        fn forward_impl(&self, observations: Tensor<B, 2>) -> Tensor<B, 3> {
            let [batch, _] = observations.dims();
            self.linear
                .forward(observations)
                .reshape([batch, TEST_ACTIONS, TEST_ATOMS])
        }
    }

    impl<B: AutodiffBackend> C51Model<B, 2> for TestNet<B> {
        fn forward(&self, observations: Tensor<B, 2>) -> Tensor<B, 3> {
            self.forward_impl(observations)
        }
        fn forward_inner(
            inner: &Self::InnerModule,
            observations: Tensor<B::InnerBackend, 2>,
        ) -> Tensor<B::InnerBackend, 3> {
            inner.forward_impl(observations)
        }
        #[allow(clippy::cast_possible_truncation)]
        fn soft_update(active: &Self, target: Self::InnerModule, tau: f64) -> Self::InnerModule {
            polyak_update::<B::InnerBackend, TestNet<B::InnerBackend>>(
                &active.valid(),
                target,
                tau as f32,
            )
        }
    }

    type TestAgent = C51Agent<Be, TestNet<Be>, TestObs, TestAction, 1, 2>;

    /// Reads the linear layer's weights back to the host, element-wise.
    ///
    /// Deliberately not a `sum()` proxy: per-element optimizer steps very
    /// nearly cancel in a sum, which would make "did the target change" a
    /// coin flip rather than a measurement.
    fn weights(net: &TestNet<Flex>) -> Vec<f32> {
        net.linear
            .weight
            .val()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("weight tensor host read")
    }

    fn policy_weights(agent: &TestAgent) -> Vec<f32> {
        weights(&agent.policy().valid())
    }

    fn target_weights(agent: &TestAgent) -> Vec<f32> {
        weights(&agent.target_net)
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "weight snapshots must have equal length");
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    /// Builds an agent whose buffer is primed and whose `learning_starts` is
    /// `0`, so a single [`C51Agent::learn_step`] can be driven directly
    /// without a training loop (and thus without coupling to ε decay or the
    /// buffer-fill schedule).
    // Test fixture data: the loop counter and element count are bounded by small
    // constants declared in this test, far below f32's 2^24 exact-integer limit,
    // so every generated value is represented exactly.
    #[allow(clippy::cast_precision_loss)]
    fn primed_agent(tau: f64, target_update_frequency: usize) -> TestAgent {
        let device = <Flex as burn::tensor::backend::BackendTypes>::Device::default();
        let config = C51TrainingConfigBuilder::new()
            .tau(tau)
            .target_update_frequency(target_update_frequency)
            .batch_size(2)
            .learning_starts(0)
            .learning_rate(0.01)
            .num_atoms(TEST_ATOMS)
            .v_min(-1.0)
            .v_max(1.0)
            .build()
            .expect("valid test config");

        let mut agent: TestAgent =
            C51Agent::new(TestNet::<Be>::init(&device), config, device).expect("agent constructs");
        for i in 0..4 {
            let x = i as f32;
            agent.remember(
                TestObs([x, -x]),
                &TestAction(i % TEST_ACTIONS),
                1.0,
                TestObs([x + 1.0, -x]),
                false,
            );
        }
        agent
    }

    /// Same as [`primed_agent`] but with prioritized replay opted in, so a
    /// single `learn_step` exercises the KL-priority writeback.
    // Test fixture data: the loop counter and element count are bounded by small
    // constants declared in this test, far below f32's 2^24 exact-integer limit,
    // so every generated value is represented exactly.
    #[allow(clippy::cast_precision_loss)]
    fn primed_prioritized_agent() -> TestAgent {
        let device = <Flex as burn::tensor::backend::BackendTypes>::Device::default();
        let config = C51TrainingConfigBuilder::new()
            .tau(0.005)
            .batch_size(2)
            .learning_starts(0)
            .learning_rate(0.01)
            .num_atoms(TEST_ATOMS)
            .v_min(-1.0)
            .v_max(1.0)
            .prioritized_replay(PrioritizedReplaySettings {
                beta_anneal_steps: 100,
                ..PrioritizedReplaySettings::default()
            })
            .build()
            .expect("valid prioritized test config");

        let mut agent: TestAgent =
            C51Agent::new(TestNet::<Be>::init(&device), config, device).expect("agent constructs");
        for i in 0..4 {
            let x = i as f32;
            agent.remember(
                TestObs([x, -x]),
                &TestAction(i % TEST_ACTIONS),
                1.0,
                TestObs([x + 1.0, -x]),
                false,
            );
        }
        agent
    }

    #[test]
    fn c51_defaults_to_uniform_replay() {
        let agent = primed_agent(0.005, 100);
        assert!(
            !agent.buffer.is_prioritized(),
            "the default config must keep uniform replay (PER is opt-in)"
        );
    }

    /// The KL-priority feedback edge: a learn step rewrites the sampled
    /// transitions' priorities off the running-max seed, moving the total mass.
    #[test]
    fn c51_priority_writeback_runs_after_learn_step() {
        let mut agent = primed_prioritized_agent();
        assert!(
            agent.buffer.is_prioritized(),
            "the opt-in config must select prioritized replay"
        );
        let before = agent
            .buffer
            .as_prioritized()
            .expect("prioritized")
            .total_priority();
        let mut rng = StdRng::seed_from_u64(7);
        agent
            .learn_step(&mut rng)
            .expect("primed agent with learning_starts = 0 learns");
        let after = agent
            .buffer
            .as_prioritized()
            .expect("prioritized")
            .total_priority();
        assert!(
            (after - before).abs() > 1e-9,
            "the KL priority writeback must change the sampling mass: {before} -> {after}"
        );
    }

    /// Regression for #182: with `tau > 0` the Polyak-lagged target must
    /// survive a `sync_target()` landing on a multiple of
    /// `target_update_frequency`. The old code hard-copied the policy here,
    /// erasing the lag every 100 steps under the *default* config.
    #[test]
    fn sync_target_is_noop_when_tau_is_positive() {
        let mut agent = primed_agent(0.005, 100);
        let mut rng = StdRng::seed_from_u64(42);
        agent
            .learn_step(&mut rng)
            .expect("learn_step runs with a primed buffer and learning_starts = 0");

        // Precondition: without divergence the assertion below is vacuous.
        let diverged = max_abs_diff(&policy_weights(&agent), &target_weights(&agent));
        assert!(
            diverged > 1e-5,
            "precondition failed: policy and target must differ before sync_target \
             (max |Δw| = {diverged:e}); the regression assertion would be vacuous"
        );

        // Land exactly on a hard-sync boundary.
        agent.step = 100;
        agent.sync_target();

        let after = max_abs_diff(&policy_weights(&agent), &target_weights(&agent));
        assert!(
            after > 1e-5,
            "tau = 0.005 must leave the target Polyak-lagged, but sync_target \
             hard-copied the policy onto it (max |Δw| = {after:e})"
        );
        assert!(
            (after - diverged).abs() < 1e-9,
            "sync_target must be a pure no-op when tau > 0: max |Δw| moved from \
             {diverged:e} to {after:e}"
        );
    }

    /// The other branch of the gate: with `tau == 0` the hard sync is the only
    /// target-update mechanism, and must still fire on a multiple of
    /// `target_update_frequency`.
    #[test]
    fn sync_target_hard_copies_when_tau_is_zero() {
        let mut agent = primed_agent(0.0, 100);
        let mut rng = StdRng::seed_from_u64(42);
        agent
            .learn_step(&mut rng)
            .expect("learn_step runs with a primed buffer and learning_starts = 0");

        let diverged = max_abs_diff(&policy_weights(&agent), &target_weights(&agent));
        assert!(
            diverged > 1e-5,
            "precondition failed: policy and target must differ before sync_target \
             (max |Δw| = {diverged:e})"
        );

        agent.step = 100;
        agent.sync_target();

        let after = max_abs_diff(&policy_weights(&agent), &target_weights(&agent));
        assert!(
            after < 1e-9,
            "tau = 0 must hard-sync the target onto the policy at a multiple of \
             target_update_frequency (max |Δw| = {after:e})"
        );
    }

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

    // -------- non-finite-loss guard (ADR 0056, #318) --------

    /// Replaces every float parameter of a module with `NaN`, simulating a
    /// policy network that has diverged to non-finite weights — the realistic
    /// source of a non-finite cross-entropy loss. Applied to a *clone* so the
    /// live net's `ParamId`s are preserved and the target pairing stays valid.
    struct NanInjector;

    impl<B: Backend> ModuleMapper<B> for NanInjector {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
            let (id, tensor, mapper) = param.consume();
            Param::from_mapped_value(id, tensor.mul_scalar(f32::NAN), mapper)
        }
    }

    /// C51 shares DQN's single-loss shape: a non-finite categorical
    /// cross-entropy loss must skip `backward`, the optimizer step, and the
    /// soft target sync (ADR 0056, #318). Diverging the policy net to NaN forces
    /// a NaN loss; the guard must fire, `learn_step` must return `None`, and the
    /// target must stay untouched and finite.
    #[test]
    fn c51_nonfinite_loss_skips_step_and_warns() {
        let mut agent = primed_agent(0.005, 100);
        // The target net is the healthy sibling: a skipped step leaves it intact.
        let target_before = target_weights(&agent);

        // Poison a *clone* of the live policy so its `ParamId`s are preserved.
        let poisoned = agent.policy().clone().map(&mut NanInjector);
        agent.policy_net = Slot::new(poisoned);

        let mut rng = StdRng::seed_from_u64(0);
        let outcome = agent.learn_step(&mut rng);

        assert!(
            outcome.is_none(),
            "a non-finite cross-entropy loss must skip the step and return None"
        );
        assert!(
            agent.loss_guard.warning_fired(),
            "the non-finite loss must fire the guard"
        );
        assert_eq!(
            target_weights(&agent),
            target_before,
            "the soft target update must be skipped, leaving the target untouched"
        );
        assert!(
            target_weights(&agent).iter().all(|w| w.is_finite()),
            "the target net must stay finite — the policy NaN must not reach it"
        );
        let action = agent.act(&TestObs([0.2, -0.2]), &mut rng);
        assert!(
            action.is_valid(),
            "act must still return a valid action after a skipped step"
        );
    }
}
