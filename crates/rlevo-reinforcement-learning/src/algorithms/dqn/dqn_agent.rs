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
use crate::replay::{DiscreteTransition, ReplayKind, ReplayStrategy, UniformReplayConfig};
use rlevo_core::action::DiscreteAction;
use rlevo_core::base::{Observation, TensorConvertible};
use rlevo_core::config::Validate;

use crate::algorithms::dqn::dqn_config::DqnTrainingConfig;
use crate::algorithms::dqn::dqn_model::DqnModel;
use crate::algorithms::dqn::exploration::EpsilonGreedy;
use crate::algorithms::shared::{
    FiniteLossGuard, FiniteObsGuard, FiniteRewardGuard, Slot, UNIFORM_REPLAY_BETA,
    reduce_weighted_loss,
};
use crate::utils::{PolyakError, compute_target_q_values};

/// Error variants returned by [`DqnAgent`] operations.
///
/// Two things can go wrong, and each has a construction site.
/// [`InvalidAction`](DqnAgentError::InvalidAction) is built in
/// [`crate::algorithms::dqn::train`], where an environment `reset`/`step`
/// rejection is converted into this domain. [`Polyak`](DqnAgentError::Polyak)
/// arrives through `?` in [`DqnAgent::learn_step`] — the agent's only fallible
/// method — when the policy and target networks have mismatched parameter
/// topologies.
///
/// # Why there is no tensor-conversion variant
///
/// This enum previously carried `TensorConversionFailed(String)`, which nothing
/// ever constructed and nothing could: the tensor host-reads on the action path
/// live in [`act`](DqnAgent::act), [`act_greedy`](DqnAgent::act_greedy) and
/// [`act_greedy_with`](DqnAgent::act_greedy_with), all of which return a bare
/// `A` rather than a `Result`. With no error channel in the signature there is
/// nowhere for the variant to be returned from, so those reads use the
/// infallible form that `docs/rules.md` §4 sanctions for a read that "cannot
/// fail by construction (e.g. a tensor the same function just built)".
///
/// Making action selection fallible is a breaking change, deferred and tracked
/// as #317. When it lands, the variant that returns must carry
/// [`rlevo_core::base::TensorConversionError`] as a `#[from]` payload, not a
/// `String`: §4 prefers structured error types over string-based ones, and
/// names `TensorConversionError` as the domain type for tensor ops.
///
/// # Why there is no buffer or I/O variant
///
/// `Buffer(#[from] ReplayBufferError)` was unreachable by design.
/// [`learn_step`](DqnAgent::learn_step) samples with
/// `let Ok(batch) = self.buffer.sample(..) else { return Ok(None) };`, and the
/// only variant `sample` can produce is
/// [`ReplayBufferError::InsufficientData`](crate::replay::ReplayBufferError::InsufficientData),
/// which means "skip this learn step", not "the step failed". `Ok(None)` is the
/// correct channel for a warm-up buffer; propagating it would misreport an
/// ordinary warm-up as an error.
///
/// `Io(#[from] std::io::Error)` anticipated checkpointing that does not exist:
/// there is no `save`, no `load`, no `Recorder` and no `std::fs` anywhere under
/// `algorithms/`, and ADR 0014 §6 defers checkpointing producer wiring to
/// Tier D.
///
/// The enum is `#[non_exhaustive]`, so adding a variant later is not a breaking
/// change — checkpointing is the shape that would plausibly need one. The bar
/// for adding it is a real construction site, not an anticipated failure mode;
/// an unconstructible variant is what this section exists to keep from
/// recurring.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DqnAgentError {
    /// The sampled or requested action is outside the valid action space.
    #[error("Invalid action: {0}")]
    InvalidAction(String),
    /// The target soft-update failed because the policy and target networks
    /// have mismatched parameter topologies.
    #[error(transparent)]
    Polyak(#[from] PolyakError),
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
    /// Non-finite-loss guard for the TD-loss site (ADR 0056, #318). DQN's only
    /// loss site, so its counter is the agent's whole
    /// [`skipped_updates`](Self::skipped_updates) (ADR 0072). Skips every
    /// occurrence; the `warn!` escalates by decades.
    loss_guard: FiniteLossGuard,
    /// Non-finite-reward guard for the `remember` ingestion site (ADR 0065,
    /// #352). Drops the transition on every occurrence; the `warn!` escalates
    /// by decades.
    reward_guard: FiniteRewardGuard,
    /// Non-finite-**observation** guard for the `remember` ingestion site (ADR
    /// 0067, #1043). Drops the transition on every occurrence — `obs` and
    /// `next_obs` are checked together, so one guard covers both rows. Distinct
    /// from `reward_guard`, and its counter is not a subset of that one: see
    /// [`dropped_observations`](Self::dropped_observations).
    obs_guard: FiniteObsGuard,
    /// Non-finite-observation guard for the three action-selection sites (ADR
    /// 0067 §Decision 4). **Detect and report only**: it counts and warns, and
    /// the action is returned unchanged. See
    /// [`degenerate_action_selections`](Self::degenerate_action_selections).
    act_obs_guard: FiniteObsGuard,
    /// Reusable host buffer backing `remember`'s finiteness check.
    ///
    /// `remember` takes `&mut self`, so the ingestion path — the hot one, once
    /// per environment step — stages its rows into an owned buffer and settles
    /// to zero allocations. The three `act` sites take `&self` and cannot share
    /// it; see the comment in [`act_greedy`](Self::act_greedy).
    obs_scratch: Vec<f32>,
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
            None => ReplayKind::uniform_from_config(UniformReplayConfig {
                capacity: config.replay_buffer_capacity,
            })?,
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
            reward_guard: FiniteRewardGuard::new("dqn/remember"),
            obs_guard: FiniteObsGuard::ingestion("dqn/remember"),
            act_obs_guard: FiniteObsGuard::act("dqn/act"),
            obs_scratch: Vec::new(),
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
    /// Because it counts attempts rather than applications, it is only half the
    /// picture on an unhealthy run: pair it with
    /// [`skipped_updates`](Self::skipped_updates), which counts the attempts
    /// that never reached the optimizer.
    ///
    /// [`DqnTrainingConfig::target_update`]: crate::algorithms::dqn::dqn_config::DqnTrainingConfig::target_update
    /// [`learn_step`]: Self::learn_step
    pub fn gradient_updates(&self) -> usize {
        self.gradient_updates
    }

    /// Number of gradient updates skipped because the TD loss was non-finite
    /// (ADR 0056, ADR 0072).
    ///
    /// DQN has exactly one loss site, so this per-site counter *is* the agent's
    /// aggregate skip count — the canonical `skipped_updates` metric the
    /// training loop emits. A non-zero value means the guard kept a `NaN`/`inf`
    /// loss out of the weights, but also that those attempts bought no learning:
    /// throughput was lost, and a persistently rising count is the signature of
    /// a diverging run (an exploding TD target, a learning rate past the stable
    /// regime), not a self-healing hiccup. The guard's `warn!` reports the same
    /// fact on a decade schedule; this accessor is its programmatic half, so a
    /// caller need not scrape log output.
    ///
    /// # Relationship to [`gradient_updates`](Self::gradient_updates)
    ///
    /// [`gradient_updates`](Self::gradient_updates) counts **attempts** and
    /// advances unconditionally, including on a skip (ADR 0059 §Decision 4);
    /// this counts the **subset of those attempts that was skipped**. So the
    /// number of updates that actually reached the optimizer is
    ///
    /// ```text
    /// applied = gradient_updates() - skipped_updates()
    /// ```
    ///
    /// and the subtraction can never underflow, because every skip is first
    /// counted as an attempt.
    ///
    /// # Relationship to the `dropped_*` counters
    ///
    /// A *drop* and a *skip* are different losses at different seams. A drop —
    /// [`dropped_transitions`](Self::dropped_transitions),
    /// [`dropped_observations`](Self::dropped_observations) — is **data that
    /// never entered the replay buffer** and can never be sampled. A skip is an
    /// **update that never reached the optimizer**, computed from data that is
    /// still in the buffer and will be sampled again. Neither counter bounds the
    /// other, and a run can show one without the other.
    #[must_use]
    pub const fn skipped_updates(&self) -> u64 {
        self.loss_guard.skipped()
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
    /// With probability `$\epsilon$` returns a uniformly random discrete action;
    /// otherwise runs the policy network on `obs` and returns the argmax.
    ///
    /// # Non-finite observations
    ///
    /// Counted and warned about, never substituted — see
    /// [`act_greedy`](Self::act_greedy) for the full reasoning and
    /// [`degenerate_action_selections`](Self::degenerate_action_selections) for
    /// the counter.
    pub fn act<R: Rng + ?Sized>(&self, obs: &O, rng: &mut R) -> A {
        if self.exploration.should_explore(rng) {
            // Guarded here rather than once at the top of the function: the
            // greedy branch below delegates to `act_greedy`, which guards
            // itself, so a top-of-function check would count that branch twice.
            // The explore branch never reaches `act_greedy`, and it still needs
            // counting — the returned action does not come from the network, but
            // the step is just as unattributable, because the *next* observation
            // it produces comes from an environment already emitting NaN.
            let mut scratch = Vec::new();
            self.act_obs_guard.report(obs.row_is_finite(&mut scratch));
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
    ///
    /// # Non-finite observations
    ///
    /// A `NaN`/`±Inf` observation is **counted and warned about; the action is
    /// returned unchanged** (ADR 0067 §Decision 4). Do not "improve" this into a
    /// substitution or a fallback action, and do not delete the check as
    /// redundant — it is the *only* thing in the system that can observe this
    /// failure, for two reasons that are both counter-intuitive:
    ///
    /// 1. On the `flex` (CPU) backend `relu` maps `NaN` to `0.0`. A ReLU-fronted
    ///    Q network fed a **fully** non-finite observation therefore emits a
    ///    finite, in-domain, `is_valid() == true` action from a bias-only Q row —
    ///    measured, the one-NaN and all-NaN Q rows are bit-identical. There is no
    ///    NaN downstream: `FiniteLossGuard` cannot fire, a Q-value check cannot
    ///    fire, an action check cannot fire.
    /// 2. Sharper, and specific to argmax-over-Q agents: on `flex`, `argmax`
    ///    over a **partly** NaN row returns the index of the *first NaN*, not the
    ///    finite max (`[1, NaN, 3, 2] -> 1`). The same row returns the correct
    ///    index on `wgpu`. So the discrete failure is CPU-specific and
    ///    invisible — and CPU is the backend CI runs.
    ///
    /// The `argmax` behaviour itself is issue #1050 and is deliberately **not**
    /// fixed here; this guard only makes it attributable.
    // Action indices only. `argmax` yields a non-negative index below
    // `A::ACTION_COUNT`, so the i64 -> usize narrowing can neither wrap nor lose a
    // sign; where an index round-trips through f32 it stays far below the 2^24
    // exact-integer limit. `from_index` bounds-checks on the way back.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn act_greedy(&self, obs: &O) -> A {
        // Function-local scratch, deliberately: `act_greedy` takes `&self` and
        // so cannot use the `obs_scratch` field `remember` uses. Sharing one
        // buffer through a `RefCell` or a `Mutex` is NOT the fix — agents must
        // stay `Sync` for the evolution layer's parallel evaluation, and a
        // `RefCell` would forfeit exactly that. The known, accepted cost is one
        // `Vec` allocation per call, and only for f32 feature-vector
        // observations (<= 24 elements anywhere in this workspace): the four
        // integer-backed image observation types override `row_is_finite` and
        // never touch `scratch`, so for them this `Vec` is never allocated into
        // (ADR 0067 §Decision 2).
        let mut scratch = Vec::new();
        self.act_obs_guard.report(obs.row_is_finite(&mut scratch));
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
    ///
    /// # Non-finite observations
    ///
    /// Counted and warned about, never substituted — see
    /// [`act_greedy`](Self::act_greedy).
    // Action indices only. `argmax` yields a non-negative index below
    // `A::ACTION_COUNT`, so the i64 -> usize narrowing can neither wrap nor lose a
    // sign; where an index round-trips through f32 it stays far below the 2^24
    // exact-integer limit. `from_index` bounds-checks on the way back.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn act_greedy_with(&self, net: &M::InnerModule, obs: &O) -> A {
        // Function-local scratch for the same `&self` / `Sync` reason as
        // `act_greedy`; same per-call allocation cost, same zero cost for the
        // integer-backed observation types.
        let mut scratch = Vec::new();
        self.act_obs_guard.report(obs.row_is_finite(&mut scratch));
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
    ///
    /// # Behavior
    ///
    /// A non-finite `reward` (`NaN` or `±Inf`) is **discarded, not stored**:
    /// the transition never enters the replay buffer and the call is otherwise
    /// a no-op. Storing it would let every minibatch that later resampled it
    /// produce a non-finite loss, which `FiniteLossGuard` then skips — silently
    /// costing gradient updates for as long as the poisoned transition stayed
    /// resident (ADR 0065, issue #352). A `tracing::warn!` fires on the 1st,
    /// 10th, 100th, … drop; use
    /// [`dropped_transitions`](Self::dropped_transitions) to detect the loss
    /// programmatically.
    ///
    /// A non-finite **observation** — a `NaN` or `±Inf` anywhere in the host row
    /// of *either* `obs` or `next_obs` — is discarded on the same terms, and
    /// counted separately by
    /// [`dropped_observations`](Self::dropped_observations) (ADR 0067, issue
    /// #1043).
    pub fn remember(&mut self, obs: O, action: &A, reward: f32, next_obs: O, terminated: bool) {
        if !self.reward_guard.admit(reward) {
            return;
        }
        // Ordering is load-bearing and is documented on both counters: the
        // reward guard runs first and returns early, so a transition that is
        // bad in both ways increments `dropped_transitions` only.
        let rows_finite = obs.row_is_finite(&mut self.obs_scratch)
            && next_obs.row_is_finite(&mut self.obs_scratch);
        if !self.obs_guard.admit(rows_finite) {
            return;
        }
        self.buffer.push(DiscreteTransition {
            obs,
            action: action.to_index(),
            reward,
            next_obs,
            terminated,
        });
    }

    /// Number of transitions [`remember`](Self::remember) discarded because
    /// their reward was non-finite.
    ///
    /// A non-zero count means those environment steps **never entered the
    /// replay buffer** and can never be sampled — the agent learned nothing
    /// from them. Watch it to detect a misbehaving environment (a division by
    /// zero, an unbounded accumulator, an exploding dynamics term in its reward
    /// function): the guard keeps the poison out of the buffer, but only the
    /// caller knows whether the resulting data loss invalidates the run. A
    /// persistently rising count also explains a buffer that never reaches
    /// `batch_size`, i.e. training that silently never starts.
    ///
    /// # Relationship to [`dropped_observations`](Self::dropped_observations)
    ///
    /// The two counters are disjoint, and **neither is the total** on its own.
    /// The reward check runs *first* in [`remember`](Self::remember) and returns
    /// early, so a transition carrying both a non-finite reward and a non-finite
    /// observation increments only this counter. Every dropped transition
    /// increments exactly one of the two, so their sum is the total — but they
    /// will legitimately disagree, and neither can be read as the other's
    /// superset (ADR 0067 §Consequences).
    #[must_use]
    pub const fn dropped_transitions(&self) -> u64 {
        self.reward_guard.dropped()
    }

    /// Number of transitions [`remember`](Self::remember) discarded because
    /// `obs` or `next_obs` carried a non-finite value.
    ///
    /// A non-zero count means those environment steps **never entered the
    /// replay buffer** and can never be sampled. Watch it to detect a
    /// misbehaving environment — a `NaN` observation is the signature of an
    /// exploding physics/dynamics term or a division by zero in the state
    /// update, which is precisely the failure mode of the rapier/box2d family.
    /// Under prioritized replay the drop is worth more than it looks: a poisoned
    /// row that *did* enter would yield a `NaN` TD error, have its priority
    /// writeback rejected, and stay pinned at the running maximum forever — so
    /// it would be resampled more often than average and never decay.
    ///
    /// Not test-gated: [`remember`](Self::remember) is public API driven from
    /// outside this crate (integration tests, benches, hand-rolled training
    /// loops), so a caller needs a programmatic way to discover that its data
    /// was dropped rather than having to scrape log output.
    ///
    /// # Relationship to [`dropped_transitions`](Self::dropped_transitions)
    ///
    /// See that accessor: the reward guard runs first and returns early, so a
    /// transition that is bad in *both* ways is counted there and **not** here.
    /// The two counters will disagree; that is by design, not a defect.
    #[must_use]
    pub fn dropped_observations(&self) -> u64 {
        self.obs_guard.count()
    }

    /// Number of action selections made from a non-finite observation.
    ///
    /// The counterpart to [`dropped_observations`](Self::dropped_observations)
    /// on the [`act`](Self::act) / [`act_greedy`](Self::act_greedy) /
    /// [`act_greedy_with`](Self::act_greedy_with) path, and **not** a drop
    /// count: per ADR 0067 §Decision 4 the action was computed, returned to the
    /// caller, and left unchanged. Substituting a plausible in-domain action
    /// would be the same class of failure this counter exists to surface.
    ///
    /// A non-zero count means every affected step is *unattributable*: on the
    /// CPU backend the network silently erased the observation and returned a
    /// finite, valid-looking action anyway (see
    /// [`act_greedy`](Self::act_greedy)). Discard the affected episode's return
    /// rather than trusting it, and fix the observation source.
    ///
    /// # Comparability across algorithms
    ///
    /// This count is comparable **within** an algorithm family across runs. It
    /// is **not** comparable between the discrete and the continuous family
    /// during the early exploration / warm-up period, because the two families
    /// place the guard on opposite sides of their random-action branch.
    ///
    /// - **Discrete** (`dqn`, `c51`, `qrdqn`) — the guard sits *inside* the
    ///   ε-explore branch, and the greedy branch delegates to `act_greedy`,
    ///   which guards itself. So **every** `act` call is counted, including the
    ///   whole early period where `epsilon_start = 1.0` means every action is
    ///   random and `obs` is read *only* to run this check.
    /// - **Continuous** (`ddpg`, `td3`, `sac`) — the guard sits *after* the
    ///   `learning_starts` warm-up early return, which draws a uniform random
    ///   action without ever reading `obs`. So while `step < learning_starts`
    ///   (and `training == true`) the count moves on **zero** calls.
    ///
    /// Both placements are correct for their own path — there is no
    /// observation read to attribute in the continuous warm-up branch — so this
    /// is a reporting caveat, not a defect. The practical consequence: a DQN
    /// run and a comparably configured SAC run over the same environment will
    /// report very different counts across their opening steps for reasons of
    /// guard placement, not environment health. Compare like with like, and
    /// note that the continuous family's `remember`-side
    /// [`dropped_observations`](Self::dropped_observations) *does* cover the
    /// warm-up steps.
    #[must_use]
    pub fn degenerate_action_selections(&self) -> u64 {
        self.act_obs_guard.count()
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

    /// Test-only view of the rewards currently held in the replay buffer,
    /// oldest first.
    ///
    /// Exists so the `train`-loop tests can assert the ADR 0065 invariant
    /// directly on buffer *contents*: no non-finite reward is ever resident,
    /// however the environment misbehaved. Mirrors
    /// [`replay_terminated_flags`](Self::replay_terminated_flags).
    #[cfg(test)]
    pub(crate) fn replay_rewards(&self) -> Vec<f32> {
        self.buffer.iter().map(|t| t.reward).collect()
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
    /// are all skipped (ADR 0056, #318) and
    /// [`skipped_updates`](Self::skipped_updates) advances, so the caller keeps
    /// its last healthy reported metrics rather than folding a NaN into them.
    /// The accompanying `warn!` fires on a decade schedule — skips 1, 10, 100,
    /// … — each line carrying the running total (ADR 0072 §1), so a run
    /// discarding 1% of its updates is distinguishable from one discarding 40%.
    /// The gradient-update counter advances even then, so the
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
    use crate::MAX_BUFFER_CAPACITY;
    use rlevo_core::config::ConstraintKind;

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
        assert_eq!(
            agent.skipped_updates(),
            1,
            "the one non-finite TD loss must be counted as exactly one skipped \
             update — not zero (guard never fired) and not more than one (a \
             single learn_step must not double-count)"
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

    /// ADR 0072: the skip counter must *count*, not latch. Three consecutive
    /// poisoned `learn_step`s must read back as exactly three skips — a `1`
    /// here would be the bool-to-counter mistranslation (a latched "it
    /// happened" flag widened to `u64`), which is invisible to any single-skip
    /// test. The attempt counter must track it one-for-one on an all-poisoned
    /// run: `applied = gradient_updates() - skipped_updates() = 0`, i.e.
    /// attempts advanced (ADR 0059 §Decision 4) and none of them was applied.
    #[test]
    fn dqn_counts_repeated_loss_skips() {
        const SKIPS: usize = 3;

        let mut agent = primed_uniform_agent();
        let target_before = weights(&agent.target_net);

        // Poison a *clone* of the live policy so its `ParamId`s are preserved.
        let poisoned = agent.policy().clone().map(&mut NanInjector);
        agent.policy_net = Slot::new(poisoned);

        let mut rng = StdRng::seed_from_u64(0);
        for i in 1..=SKIPS {
            assert!(
                agent
                    .learn_step(&mut rng)
                    .expect("no polyak error")
                    .is_none(),
                "skip #{i}: a non-finite TD loss must still skip the step — the \
                 skip must never latch off after the first occurrence"
            );
            assert_eq!(
                agent.skipped_updates(),
                i as u64,
                "after {i} poisoned learn_step(s) the counter must read {i}"
            );
        }

        assert_eq!(
            agent.skipped_updates(),
            SKIPS as u64,
            "every one of the three non-finite losses must be counted"
        );
        assert_eq!(
            agent.gradient_updates(),
            SKIPS,
            "each skipped step still counts as an attempt (ADR 0059 §Decision 4)"
        );
        assert_eq!(
            agent.gradient_updates() as u64 - agent.skipped_updates(),
            0,
            "applied = attempts − skips must be zero on an all-poisoned run"
        );
        assert_eq!(
            weights(&agent.target_net),
            target_before,
            "no optimizer step ran, so no soft target update may have run either"
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
        // The executable form of ADR 0059 §Decision 4 ∧ ADR 0072: attempts
        // advanced and *none* was applied — `applied = 1 − 1 = 0`.
        assert_eq!(
            agent.skipped_updates(),
            1,
            "the same attempt must also be counted as a skip, so that \
             gradient_updates() − skipped_updates() = 0 applied updates"
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
            agent.skipped_updates(),
            1,
            "a healthy step must not increment the skip counter: 2 attempts − 1 \
             skip = 1 applied update"
        );
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

    // ---- ADR 0065 / #352: non-finite reward is dropped at ingestion ----
    //
    // Every off-policy agent needs its OWN copy of this test. The defect had
    // six sites, not the four the issue named, precisely because C51 and
    // QR-DQN were added by copying an unguarded `remember` and no shared test
    // noticed. A per-file test is what makes agent #7's author notice.

    #[test]
    fn dqn_remember_drops_a_nonfinite_reward() {
        let device = <TestInner as burn::tensor::backend::BackendTypes>::Device::default();
        let mut agent = TestAgent::new(
            TestNet::constant(&device, 0.5),
            DqnTrainingConfig::default(),
            device,
        )
        .expect("default config is valid");

        agent.remember(
            TestObs([0.0, 0.0]),
            &TestAction(0),
            f32::NAN,
            TestObs([1.0, 0.0]),
            false,
        );
        assert_eq!(
            agent.buffer_len(),
            0,
            "a NaN-reward transition must never enter the replay buffer"
        );
        assert_eq!(
            agent.dropped_transitions(),
            1,
            "the drop must be visible to the caller through the public counter"
        );

        agent.remember(
            TestObs([1.0, 0.0]),
            &TestAction(1),
            1.0,
            TestObs([2.0, 0.0]),
            false,
        );
        assert_eq!(
            agent.buffer_len(),
            1,
            "the drop must not latch: the next finite reward is still stored"
        );
        assert_eq!(
            agent.dropped_transitions(),
            1,
            "a finite reward must not increment the drop counter"
        );
    }

    // ---- ADR 0067 / #1043: non-finite observation ----
    //
    // Per-file, for the same reason as the ADR 0065 block above: six sites, and
    // a shared test would not notice a `remember` copied without its guard.

    /// A deliberately NaN-emitting observation.
    ///
    /// `TestObs`'s `write_host_row` copies its array verbatim, so this row
    /// reaches `row_is_finite` with the `NaN` intact — the check is on the host
    /// row, *before* `to_tensor`, which is the only place the failure is
    /// visible at all.
    fn nan_obs() -> TestObs {
        TestObs([f32::NAN, 0.0])
    }

    fn obs_guard_agent() -> TestAgent {
        let device = <TestInner as burn::tensor::backend::BackendTypes>::Device::default();
        TestAgent::new(
            TestNet::constant(&device, 0.5),
            DqnTrainingConfig::default(),
            device,
        )
        .expect("default config is valid")
    }

    #[test]
    fn dqn_remember_drops_a_nonfinite_obs() {
        let mut agent = obs_guard_agent();

        agent.remember(nan_obs(), &TestAction(0), 1.0, TestObs([1.0, 0.0]), false);
        assert_eq!(
            agent.buffer_len(),
            0,
            "a NaN-observation transition must never enter the replay buffer"
        );
        assert_eq!(
            agent.dropped_observations(),
            1,
            "the drop must be visible to the caller through the public counter"
        );
        assert_eq!(
            agent.dropped_transitions(),
            0,
            "the reward was finite: the reward counter must not move"
        );

        agent.remember(TestObs([0.0, 0.0]), &TestAction(1), 1.0, nan_obs(), false);
        assert_eq!(
            agent.buffer_len(),
            0,
            "a NaN `next_obs` is checked too — it is the Bellman bootstrap input"
        );
        assert_eq!(agent.dropped_observations(), 2, "both rows are guarded");

        agent.remember(
            TestObs([1.0, 0.0]),
            &TestAction(1),
            1.0,
            TestObs([2.0, 0.0]),
            false,
        );
        assert_eq!(
            agent.buffer_len(),
            1,
            "the drop must not latch: the next finite transition is still stored"
        );
        assert_eq!(
            agent.dropped_observations(),
            2,
            "a finite observation must not increment the drop counter"
        );
    }

    /// Pins the documented ordering: the reward guard runs first and returns
    /// early, so a doubly-bad transition is counted once, on the reward side.
    /// Both accessors' rustdoc states this; this test is what keeps it true.
    #[test]
    fn dqn_remember_counts_a_doubly_bad_transition_as_a_reward_drop_only() {
        let mut agent = obs_guard_agent();

        agent.remember(nan_obs(), &TestAction(0), f32::NAN, nan_obs(), false);

        assert_eq!(agent.buffer_len(), 0, "the transition must not be stored");
        assert_eq!(
            agent.dropped_transitions(),
            1,
            "the reward guard runs first and returns early"
        );
        assert_eq!(
            agent.dropped_observations(),
            0,
            "the observation guard is never reached, so the two counters \
             legitimately disagree — the documented ordering (ADR 0067)"
        );
    }

    /// The decision under test is that the agent does **not** substitute: the
    /// counter moves and an action still comes back, at all three `act` sites.
    #[test]
    fn dqn_act_reports_a_nonfinite_obs_and_still_returns_an_action() {
        let agent = obs_guard_agent();
        let mut rng = StdRng::seed_from_u64(7);

        let action = agent.act(&nan_obs(), &mut rng);
        assert!(
            action.is_valid(),
            "ADR 0067 §Decision 4: the action is returned unchanged, not substituted"
        );
        assert_eq!(
            agent.degenerate_action_selections(),
            1,
            "`act` must count the degenerate selection on either ε-branch"
        );

        let action = agent.act_greedy(&nan_obs());
        assert!(action.is_valid(), "act_greedy must still return an action");
        assert_eq!(
            agent.degenerate_action_selections(),
            2,
            "`act_greedy` is its own site and must count"
        );

        let net = agent.inference_net();
        let action = agent.act_greedy_with(&net, &nan_obs());
        assert!(
            action.is_valid(),
            "act_greedy_with must still return an action"
        );
        assert_eq!(
            agent.degenerate_action_selections(),
            3,
            "`act_greedy_with` is its own site and must count"
        );

        agent.act_greedy(&TestObs([0.5, -0.5]));
        assert_eq!(
            agent.degenerate_action_selections(),
            3,
            "a finite observation must not increment the counter"
        );
    }

    /// An out-of-range `replay_buffer_capacity` must come back as an `Err`
    /// from `new`, never as a panic. `UniformReplay::new` *asserts* on both
    /// bounds, so before this constructor took the fallible
    /// `ReplayKind::uniform_from_config` path the only thing standing between a bad
    /// capacity and an abort was the `config.validate()?` line happening to
    /// run first. This pins the observable contract so a reordering — or a
    /// seventh agent copied from this one — cannot quietly turn it back into
    /// a panic.
    ///
    /// The config is built by struct literal rather than through the builder
    /// on purpose: `build()` runs `validate()`, so a builder physically cannot
    /// hand `new` a bad capacity, and a test that went through it would be
    /// asserting on the builder instead of on this constructor.
    #[test]
    fn new_rejects_out_of_range_replay_buffer_capacity() {
        let over = MAX_BUFFER_CAPACITY + 1;
        let cases = [
            (0usize, ConstraintKind::Zero),
            (
                over,
                ConstraintKind::TooLarge {
                    max: u64::try_from(MAX_BUFFER_CAPACITY).expect("the ceiling fits in u64"),
                    got: u64::try_from(over).expect("the ceiling plus one fits in u64"),
                },
            ),
        ];

        for (capacity, kind) in cases {
            let device = <TestInner as burn::tensor::backend::BackendTypes>::Device::default();
            let config = DqnTrainingConfig {
                replay_buffer_capacity: capacity,
                ..DqnTrainingConfig::default()
            };
            let Err(err) = TestAgent::new(TestNet::constant(&device, 0.5), config, device) else {
                panic!("capacity {capacity} must be rejected, not allocated");
            };
            assert_eq!(err.config, "DqnTrainingConfig");
            assert_eq!(err.field, "replay_buffer_capacity");
            assert_eq!(err.kind, kind);
        }
    }
}
