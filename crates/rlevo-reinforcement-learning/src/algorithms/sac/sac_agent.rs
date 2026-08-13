//! Soft Actor-Critic (SAC) agent: stochastic actor, twin critics, learnable
//! temperature, and replay buffer management.
//!
//! [`SacAgent`] pairs a squashed-Gaussian actor with **two** Q-critics (each
//! with its own Polyak-averaged target) and a single scalar `$\log \alpha$` module.
//! Compared to [`Td3Agent`](crate::algorithms::td3::td3_agent::Td3Agent), SAC:
//!
//! 1. drops the target actor (the stochastic policy + `min`-of-twin-Q backup
//!    already addresses critic overestimation),
//! 2. replaces deterministic-actor + exploration-noise with a single
//!    reparameterized sample from the policy at every env step, and
//! 3. adds an entropy term `$-\alpha \cdot \log \pi(a'|s')$` to the Bellman target and an
//!    auto-tuning loss for `$\log \alpha$`.
//!
//! Drive the full loop with [`super::train::train`].

use std::marker::PhantomData;

use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, GradientsParams};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{ElementConversion, Tensor, TensorData};
use rand::Rng;
use rand::RngExt;

use crate::metrics::{AgentStats, PerformanceRecord};
use crate::replay::{ContinuousTransition, ReplayStrategy, UniformReplay, UniformReplayConfig};
use rlevo_core::action::BoundedAction;
use rlevo_core::base::{Observation, TensorConvertible};
use rlevo_core::config::Validate;

use crate::algorithms::sac::sac_alpha::LogAlpha;
use crate::algorithms::sac::sac_config::SacTrainingConfig;
use crate::algorithms::sac::sac_model::{ContinuousQ, SquashedGaussianPolicy};
#[cfg(test)]
use crate::algorithms::shared::param_checksum;
use crate::algorithms::shared::{
    FiniteLossGuard, FiniteObsGuard, FiniteRewardGuard, Slot, UNIFORM_REPLAY_BETA,
    assert_bounds_match_components,
};
use crate::utils::{PolyakError, compute_target_q_values};

/// Error variants returned by [`SacAgent`] operations.
///
/// Two variants, and each has a real construction site in this crate.
/// [`InvalidAction`](SacAgentError::InvalidAction) is built by
/// [`train`](super::train::train) when `env.reset()` or `env.step()` rejects an
/// action the agent proposed. [`Polyak`](SacAgentError::Polyak) arrives through
/// `?` on the target soft-updates at the end of
/// [`learn_step`](SacAgent::learn_step), where a live critic and its target
/// twin can disagree on parameter topology.
///
/// The enum is `#[non_exhaustive]`: downstream `match` expressions must carry a
/// wildcard arm, so a future variant is not a breaking change.
///
/// # Why there is no tensor-conversion variant
///
/// This enum previously carried `TensorConversionFailed(String)`, which nothing
/// constructed and nothing could. The one tensor host-read on the act path is
/// the `as_slice::<f32>()` call in [`act`](SacAgent::act), and that method
/// returns a bare `A` rather than a `Result` — there is no error channel for a
/// variant to travel down. The read is an `.expect` on a named invariant, which
/// is precisely the form `docs/rules.md` §4 sanctions for a host-read that
/// "cannot fail by construction": the tensor is one the same function just
/// built from its own actor. Issue #317 tracks making that path fallible and is
/// an explicitly deferred breaking change.
///
/// When #317 lands, the variant returns as
/// `#[from]` [`rlevo_core::base::TensorConversionError`] — not as a `String`.
/// §4 prefers structured variants over string-based errors and names that type
/// as the tensor-op error domain, so re-introducing a `String` payload would
/// rebuild the discarded variant in a merely-live form.
///
/// # Why there is no buffer or I/O variant
///
/// `Buffer(#[from] ReplayBufferError)` was unreachable by design.
/// [`learn_step`](SacAgent::learn_step) samples with
/// `let Ok(batch) = self.buffer.sample(..) else { return Ok(None) };`, and the
/// only variant `sample` can produce is
/// [`ReplayBufferError::InsufficientData`](crate::replay::ReplayBufferError::InsufficientData),
/// which means "the buffer is still warming up, skip this learn step" — not
/// "this learn step failed". Propagating it would misreport an ordinary warm-up
/// as an error.
///
/// `Io(#[from] std::io::Error)` anticipated checkpointing that does not exist:
/// there is no `save`, `load`, `Recorder`, or `std::fs` use anywhere under
/// `algorithms/`, and ADR 0014 §6 defers the checkpoint write path to Tier D.
///
/// Because the enum is `#[non_exhaustive]`, adding a variant when either of
/// those lands is not a breaking change. The bar for adding one is an actual
/// construction site, not an anticipated failure mode — a declared but
/// unconstructible variant is what this section exists to keep from recurring.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SacAgentError {
    /// The sampled or requested action is outside the valid action space.
    #[error("Invalid action: {0}")]
    InvalidAction(String),
    /// A target soft-update failed because a live critic and its target twin
    /// have mismatched parameter topologies.
    #[error(transparent)]
    Polyak(#[from] PolyakError),
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
    /// Most recent α value (`$= \exp(\log \alpha)$`).
    pub alpha: f32,
    /// Most recent mean `$-\log \pi(a|s)$` across the actor batch — proxy for
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
    /// Critic-1 MSE Bellman error on this batch (`qf1_loss`).
    pub qf1_loss: f32,
    /// Critic-2 MSE Bellman error on this batch (`qf2_loss`).
    pub qf2_loss: f32,
    /// Actor loss, or `None` on critic-only iterations (delayed-update
    /// skips).
    pub actor_loss: Option<f32>,
    /// Current α (after an auto-tuning step on actor-update iterations).
    pub alpha: f32,
    /// Batch-mean `$-\log \pi(a|s)$` on the most recent actor update, or `None`
    /// on critic-only iterations.
    pub entropy: Option<f32>,
    /// Mean of `min(q1, q2)` across the batch.
    pub q_mean: f32,
}

/// Computes the SAC Bellman target:
/// `y = r + γ · (1 − terminated) · (min(Q1', Q2') − α · log π(a'|s'))`.
///
/// Exposed at crate visibility so the unit tests can exercise the SAC
/// entropy-augmented backup without standing up a full agent.
pub(crate) fn compute_sac_target<BI: Backend>(
    rewards: Tensor<BI, 1>,
    next_q1: Tensor<BI, 1>,
    next_q2: Tensor<BI, 1>,
    next_log_prob: Tensor<BI, 1>,
    alpha: f32,
    terminated: Tensor<BI, 1>,
    gamma: f32,
) -> Tensor<BI, 1> {
    let min_q = next_q1.min_pair(next_q2);
    let entropy_adjusted = min_q - next_log_prob.mul_scalar(alpha);
    compute_target_q_values(rewards, entropy_adjusted, terminated, gamma)
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
///
/// # Network ownership
///
/// The actor and both critics live in a [`Slot`], which owns each network
/// across Burn's by-value [`Optimizer::step`](burn::optim::Optimizer::step).
/// Every read — the forward pass, [`GradientsParams::from_grads`], the
/// `.valid()` snapshots — goes through [`Slot::get`], so a network is out of
/// its field only for the duration of the `step` call inside
/// [`Slot::step_with`].
///
/// The three slots are stepped independently and in sequence, so their windows
/// are disjoint: a panic inside `critic_1`'s optimizer step poisons `critic_1`
/// alone and leaves `critic_2` and the actor intact. A panic inside a `step`
/// is nonetheless terminal for the network it was stepping — see the
/// [`shared`](crate::algorithms::shared) module docs for why that residual
/// window cannot be closed.
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
    actor: Slot<Actor>,
    // Inner-backend snapshot of `actor`, refreshed after each actor update
    // and used by the target-Q computation. Kept separately from the live
    // actor so `learn_step` never has to call `.valid()` on a Module that
    // also participates in the critic autodiff graph — on Burn 0.20's
    // shared autodiff server, running `.valid()` mid-learn was unstable.
    actor_snapshot: Actor::InnerModule,
    critic_1: Slot<Critic>,
    critic_2: Slot<Critic>,
    target_critic_1: Critic::InnerModule,
    target_critic_2: Critic::InnerModule,
    log_alpha: LogAlpha,
    actor_opt: OptimizerAdaptor<Adam, Actor, B>,
    critic_1_opt: OptimizerAdaptor<Adam, Critic, B>,
    critic_2_opt: OptimizerAdaptor<Adam, Critic, B>,
    buffer: UniformReplay<ContinuousTransition<O>>,
    low: &'static [f32],
    high: &'static [f32],
    target_entropy: f32,
    config: SacTrainingConfig,
    device: B::Device,
    step: usize,
    critic_updates: usize,
    stats: AgentStats<SacMetrics>,
    last_actor_loss: f32,
    last_alpha: f32,
    last_entropy: f32,
    /// Most recent *applied* critic-1 loss — carried forward across a
    /// non-finite skip so the reported metric never folds in a NaN (#318).
    last_qf1_loss: f32,
    /// Most recent *applied* critic-2 loss (see [`Self::last_qf1_loss`]).
    last_qf2_loss: f32,
    /// Non-finite-loss guard for the critic-1 loss site (ADR 0056, #318). The
    /// **skip** fires on every occurrence (ADR 0056 §3, reaffirmed by ADR 0072
    /// §1); the **`warn!`** follows a decade schedule — at the 1st, 10th, 100th,
    /// … skip, each line carrying the running total (ADR 0072). The count itself
    /// is per-site and readable via
    /// [`skipped_critic_1_updates`](Self::skipped_critic_1_updates).
    critic_1_guard: FiniteLossGuard,
    /// Non-finite-loss guard for the critic-2 loss site. Independent counter and
    /// independent decade schedule; read it via
    /// [`skipped_critic_2_updates`](Self::skipped_critic_2_updates).
    critic_2_guard: FiniteLossGuard,
    /// Non-finite-loss guard for the actor loss site. Independent counter and
    /// independent decade schedule; read it via
    /// [`skipped_actor_updates`](Self::skipped_actor_updates).
    actor_guard: FiniteLossGuard,
    /// Non-finite-reward guard for the `remember` ingestion site (ADR 0065,
    /// #352). Drops the transition on every occurrence; the `warn!` escalates
    /// by decades.
    reward_guard: FiniteRewardGuard,
    /// Non-finite-**observation** guard for the `remember` ingestion site (ADR
    /// 0067, #1043). Drops the transition on every occurrence; the `warn!`
    /// escalates by decades. Runs *after* [`Self::reward_guard`], which returns
    /// early — so the two counters are not additive (see
    /// [`dropped_observations`](Self::dropped_observations)).
    obs_guard: FiniteObsGuard,
    /// Non-finite-observation guard for the action-selection site
    /// ([`act`](Self::act) — SAC has no `act_with`). Detect-and-report only: it
    /// counts and warns, and the action is returned unchanged (ADR 0067
    /// §Decision 4).
    act_obs_guard: FiniteObsGuard,
    /// Reusable host staging buffer for the ingestion-side row-finiteness
    /// check. `remember` takes `&mut self`, so it can own one buffer and
    /// amortize the allocation across calls; the `&self` action-selection path
    /// cannot, and pays a per-call `Vec` instead — see [`act`](Self::act).
    obs_scratch: Vec<f32>,
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
            .finish_non_exhaustive()
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
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`](rlevo_core::config::ConfigError) if `config`
    /// fails [`SacTrainingConfig::validate`](rlevo_core::config::Validate::validate).
    ///
    /// # Panics
    ///
    /// Panics if `A`'s [`BoundedAction`] impl violates its length contract —
    /// i.e. if `A::low().len()` or `A::high().len()` differs from
    /// `A::COMPONENTS`. `&'static [f32]` cannot carry that guarantee in the
    /// type system (ADR 0053), so it is checked once here rather than being
    /// discovered as an out-of-bounds index mid-episode.
    // Divisor/normalizer derived from a count -- batch size, minibatch count,
    // history length, iteration number. All are bounded by configured sizes far
    // below f32's 2^24 (f64's 2^53) exact-integer limit.
    #[allow(clippy::cast_precision_loss)]
    pub fn new(
        actor: Actor,
        critic_1: Critic,
        critic_2: Critic,
        config: SacTrainingConfig,
        device: B::Device,
    ) -> Result<Self, rlevo_core::config::ConfigError> {
        config.validate()?;
        assert_bounds_match_components::<DA, A>();
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
        let target_entropy = config
            .target_entropy
            .unwrap_or_else(|| -(A::COMPONENTS as f32));
        let stats = AgentStats::<SacMetrics>::new(100);
        // The capacity is a runtime config field, not a literal, so it takes
        // the fallible path: an out-of-range value is a `ConfigError` naming
        // `capacity`, never an allocation abort inside `VecDeque`.
        let buffer = UniformReplay::from_config(UniformReplayConfig {
            capacity: config.replay_buffer_capacity,
        })?;
        Ok(Self {
            actor: Slot::new(actor),
            actor_snapshot,
            critic_1: Slot::new(critic_1),
            critic_2: Slot::new(critic_2),
            target_critic_1,
            target_critic_2,
            log_alpha,
            actor_opt,
            critic_1_opt,
            critic_2_opt,
            buffer,
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
            last_qf1_loss: 0.0,
            last_qf2_loss: 0.0,
            critic_1_guard: FiniteLossGuard::new("sac/critic_1"),
            critic_2_guard: FiniteLossGuard::new("sac/critic_2"),
            actor_guard: FiniteLossGuard::new("sac/actor"),
            reward_guard: FiniteRewardGuard::new("sac/remember"),
            obs_guard: FiniteObsGuard::ingestion("sac/remember"),
            act_obs_guard: FiniteObsGuard::act("sac/act"),
            obs_scratch: Vec::new(),
            _action: PhantomData,
        })
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

    /// Number of critic updates **attempted** so far.
    ///
    /// Advances unconditionally, including on a step whose loss was non-finite
    /// and therefore skipped (ADR 0059 §Decision 4) — it is the cadence counter
    /// that drives the actor / α schedule, so it must not stall on a skip.
    /// [`skipped_updates`](Self::skipped_updates) and its per-site siblings
    /// count the subset that never reached an optimizer, so
    /// `applied = critic_updates() - skipped_*` for the corresponding site.
    pub fn critic_updates(&self) -> usize {
        self.critic_updates
    }

    /// Number of **critic-1** gradient updates skipped for a non-finite loss.
    ///
    /// See [`skipped_updates`](Self::skipped_updates) for the attempts-vs-skips
    /// relationship and for what this counter deliberately excludes.
    #[must_use]
    pub const fn skipped_critic_1_updates(&self) -> u64 {
        self.critic_1_guard.skipped()
    }

    /// Number of **critic-2** gradient updates skipped for a non-finite loss.
    ///
    /// Independent of [`skipped_critic_1_updates`](Self::skipped_critic_1_updates):
    /// the twins run their `backward()` + optimizer step in disjoint windows on
    /// independent graphs, so one may be skipped while the other applies (ADR
    /// 0056). The two counters will legitimately disagree.
    #[must_use]
    pub const fn skipped_critic_2_updates(&self) -> u64 {
        self.critic_2_guard.skipped()
    }

    /// Number of **actor** gradient updates skipped for a non-finite loss.
    ///
    /// # Fewer attempts than the critics, by construction
    ///
    /// The actor site sits inside the delayed-update cadence block — it is only
    /// reached every `policy_frequency`-th critic step — so actor *attempts*
    /// are a fraction of critic attempts and this counter is bounded by
    /// `critic_updates() / policy_frequency`, not by `critic_updates()`. Do not
    /// read a small value here as evidence that the actor is healthier than the
    /// critics; normalize by attempts first.
    ///
    /// # A skipped actor step also skips the snapshot refresh
    ///
    /// On a skip the inner-backend actor snapshot used by later target-Q
    /// computations is *not* refreshed either — the two are inside the same
    /// branch. That is deliberate (a snapshot of un-stepped weights would be a
    /// redundant copy), but it means a run with a persistently non-finite actor
    /// loss serves stale target actions as well as losing policy updates.
    #[must_use]
    pub const fn skipped_actor_updates(&self) -> u64 {
        self.actor_guard.skipped()
    }

    /// Total gradient updates skipped for a non-finite loss across **all three**
    /// of this agent's loss sites — critic-1, critic-2 and the actor.
    ///
    /// This is the canonical `skipped_updates` metric (ADR 0072 §2). It is the
    /// aggregate, not a per-site read, precisely so that a future fourth loss
    /// site cannot be added without appearing here: callers and the training
    /// loop consume this one accessor, and the sum is the single place that has
    /// to learn about a new guard. Saturating throughout, so the sum cannot wrap
    /// even in the physically-unreachable case of three near-`u64::MAX` terms.
    ///
    /// # Does NOT include α-update skips
    ///
    /// The temperature (α) update carries its own, separate non-finite guard on
    /// [`LogAlpha`] (issue #184, ADR 0056 §5), which was deliberately kept
    /// distinct from the shared `FiniteLossGuard` — it guards a closed-form Adam
    /// step driven by the batch-mean log-prob, not a `backward()` over a loss
    /// tensor. Its skips are **not** summed here. This accessor is therefore
    /// "all non-finite *gradient* skips", not "all non-finite events in the
    /// agent"; do not read a `0` here as proof that nothing was skipped.
    ///
    /// # Relationship to [`critic_updates`](Self::critic_updates)
    ///
    /// [`critic_updates`](Self::critic_updates) counts **attempts** and advances
    /// unconditionally, including on a skip (ADR 0059 §Decision 4); the
    /// `skipped_*` family counts the **subset of attempts that was skipped** for
    /// a non-finite loss (ADR 0056, ADR 0072). So, per site,
    ///
    /// ```text
    /// applied = attempts - skipped
    /// ```
    ///
    /// and the subtraction can never underflow, because every skip is first
    /// counted as an attempt. Note that the actor's attempt count is *not*
    /// `critic_updates()` — see
    /// [`skipped_actor_updates`](Self::skipped_actor_updates).
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
    ///
    /// [`LogAlpha`]: super::sac_alpha::LogAlpha
    #[must_use]
    pub const fn skipped_updates(&self) -> u64 {
        self.skipped_critic_1_updates()
            .saturating_add(self.skipped_critic_2_updates())
            .saturating_add(self.skipped_actor_updates())
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

    /// Most recent batch-mean `$-\log \pi(a|s)$` proxy for policy entropy.
    pub fn last_entropy(&self) -> f32 {
        self.last_entropy
    }

    /// Samples an action for the current observation.
    ///
    /// Before `learning_starts` steps, draws a uniform random action on
    /// `[low, high]`. Afterwards the action is a reparameterized sample from
    /// the squashed-Gaussian policy when `training=true`, and the
    /// deterministic policy mean (tanh-squashed) otherwise.
    ///
    /// # Behavior on a non-finite observation
    ///
    /// The observation row is checked for finiteness before it reaches the
    /// actor. A `NaN` / `±Inf` row is **counted and warned about, and the
    /// action is returned unchanged** — nothing is substituted, no fallback is
    /// returned, and the clamping is not altered (ADR 0067 §Decision 4). Read
    /// the count with
    /// [`degenerate_action_selections`](Self::degenerate_action_selections).
    ///
    /// The warm-up branch is deliberately outside the check: it never reads
    /// `obs`, so no action was selected *from* a poisoned observation there.
    /// The step is still caught at [`remember`](Self::remember).
    ///
    /// # Why this is the only place the failure is observable
    ///
    /// The continuous failure mode is **backend-split**, and both halves are
    /// bad in different ways. Measured on Pendulum/DDPG, whose actor path is
    /// the same shape as this one:
    ///
    /// - **wgpu / Metal** — a non-finite observation yields `raw_actor =
    ///   [NaN]`, and the `NaN` survives the clamp, so the returned action's
    ///   `is_valid()` is `false`.
    /// - **flex (CPU)** — `relu` rescues `NaN` to `0.0`, so the actor emits a
    ///   finite, plausible, in-bounds torque (measured `0.08646313`) from a
    ///   **fully** `NaN` observation, with `is_valid() == true`. Nothing
    ///   downstream can see it: the loss is finite, so `FiniteLossGuard`
    ///   cannot fire, and an action-validity check cannot fire either. CI has
    ///   no GPU, so CI only ever sees this half. SAC's `tanh` squash does not
    ///   change the conclusion — it maps a finite erased pre-activation to a
    ///   perfectly ordinary in-bounds action.
    ///
    /// # The clamp here is host `f32::clamp`, and must stay that way
    ///
    /// Every continuous act-path clip in this crate is [`f32::clamp`] over a
    /// `&[f32]` already read back from the actor — **not** `Tensor::clamp`.
    /// `f32::clamp(NaN, lo, hi)` propagates `NaN`, and does so identically on
    /// every backend, so ADR 0066's `clamp_preserving_nan` does **not** apply
    /// here and must not be introduced. Moving this clip into tensor space
    /// would *import* the C51 defect: wgpu's `Tensor::clamp` rescues `NaN` to
    /// the lower bound, which would turn the one backend that still reports
    /// this failure into one that produces a plausible in-domain action with
    /// `is_valid()` back to `true`.
    ///
    /// # The action type validates nothing on this path
    ///
    /// The action is built with
    /// [`ContinuousAction::from_slice`](rlevo_core::action::ContinuousAction::from_slice),
    /// which is unchecked by contract, so an action type's own `NaN` rejection
    /// is bypassed here (`PendulumAction::new` rejects `NaN`; no agent calls
    /// it). ADR 0067 records that as deliberately out of scope — do not read
    /// the returned `A` as having been validated.
    ///
    /// # Panics
    ///
    /// Panics if the actor slot is poisoned — i.e. a previous
    /// [`learn_step`](Self::learn_step) unwound *inside* the actor's optimizer
    /// step. The agent cannot recover and must be rebuilt; see
    /// [`Slot`](crate::algorithms::shared::Slot). Also panics if the actor's
    /// output tensor is not `f32`: that host-read is an `.expect` on a named
    /// invariant, the form `docs/rules.md` §4 sanctions here because `act`
    /// returns a bare action and so has no error channel to report the failure
    /// through — issue #317 tracks making the path fallible.
    pub fn act<R: Rng + ?Sized>(&self, obs: &O, training: bool, rng: &mut R) -> A {
        if training && self.step < self.config.learning_starts {
            let sample: Vec<f32> = (0..A::COMPONENTS)
                .map(|i| rng.random_range(self.low[i]..=self.high[i]))
                .collect();
            return A::from_slice(&sample);
        }

        // `&self` (agents must stay `Sync` — the evolution layer evaluates them
        // in parallel), so the staging buffer cannot be a field and must not be
        // a shared `RefCell` / `Mutex`. Cost of the deliberate alternative: one
        // `Vec<f32>` allocation per call, and only for f32 feature-vector
        // observations (≤24 elements in this workspace). The four integer-backed
        // observation types override `row_is_finite` without touching `scratch`,
        // so this `Vec` is never allocated into for them (ADR 0067 §Decision 2).
        let mut scratch: Vec<f32> = Vec::new();
        self.act_obs_guard.report(obs.row_is_finite(&mut scratch));

        // Run action-selection on the inner backend so the autodiff graph
        // isn't expanded every env step — the result is never `.backward`'d
        // and the orphan graphs otherwise accumulate in Burn 0.20's shared
        // server (and interfere with later critic backwards).
        let obs_inner: Tensor<B::InnerBackend, DO> = obs.to_tensor(&self.device);
        let batched_inner: Tensor<B::InnerBackend, DB> = obs_inner.unsqueeze::<DB>();

        let action_dim = self.actor.get().action_dim();
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
    /// A non-finite **observation** — on either `obs` or `next_obs` — is
    /// discarded the same way, with its own counter
    /// ([`dropped_observations`](Self::dropped_observations)) and its own
    /// decade-scheduled `warn!` (ADR 0067, issue #1043). The reward check runs
    /// **first** and returns early, so a transition that is bad in both ways
    /// increments only `dropped_transitions`.
    pub fn remember(&mut self, obs: O, action: &A, reward: f32, next_obs: O, terminated: bool) {
        if !self.reward_guard.admit(reward) {
            return;
        }
        // Ordering is load-bearing and documented on both accessors: the reward
        // guard above already returned, so this counter never sees a
        // both-bad transition.
        //
        // `&mut self` here, so the staging buffer is the agent-owned field and
        // the check costs no allocation after the first call. `&&`
        // short-circuits: a poisoned `obs` skips staging `next_obs` entirely.
        let rows_finite = obs.row_is_finite(&mut self.obs_scratch)
            && next_obs.row_is_finite(&mut self.obs_scratch);
        if !self.obs_guard.admit(rows_finite) {
            return;
        }
        self.buffer.push(ContinuousTransition {
            obs,
            action: action.as_slice().to_vec(),
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
    /// # Not additive with [`dropped_observations`](Self::dropped_observations)
    ///
    /// The reward check runs **first** in [`remember`](Self::remember) and
    /// returns early, so a transition carrying both a non-finite reward and a
    /// non-finite observation increments **only this** counter. The two will
    /// legitimately disagree; neither is the total number of dropped
    /// transitions on its own, and their **sum** is that total (ADR 0067
    /// §Consequences).
    #[must_use]
    pub const fn dropped_transitions(&self) -> u64 {
        self.reward_guard.dropped()
    }

    /// Number of transitions [`remember`](Self::remember) discarded because
    /// `obs` or `next_obs` carried a non-finite value (`NaN` / `±Inf`).
    ///
    /// A non-zero count means those environment steps **never entered the
    /// replay buffer** and can never be sampled. The usual source is the
    /// environment's own state update — a division by zero, an unbounded
    /// accumulator, or an exploding physics/dynamics term — which for the
    /// continuous agents is the whole rapier/box2d family. As with
    /// [`dropped_transitions`](Self::dropped_transitions), a persistently
    /// rising count explains a buffer that never reaches `batch_size`, i.e.
    /// training that silently never starts.
    ///
    /// Not test-gated: `remember` is public API driven directly from outside
    /// this crate (integration tests, benches, hand-rolled training loops), so
    /// a caller needs a programmatic way to discover that its data was dropped
    /// rather than having to scrape log output.
    ///
    /// # Not additive with [`dropped_transitions`](Self::dropped_transitions)
    ///
    /// The reward guard runs **first** and returns early, so a transition that
    /// is bad in both ways increments only `dropped_transitions` and **not**
    /// this counter. Expect the two to disagree; each drop increments exactly
    /// one of them, so their **sum** is the total number of dropped
    /// transitions and neither is that total on its own (ADR 0067
    /// §Consequences).
    #[must_use]
    pub fn dropped_observations(&self) -> u64 {
        self.obs_guard.count()
    }

    /// Number of actions [`act`](Self::act) returned from a **non-finite
    /// observation**.
    ///
    /// This is not a drop count: per ADR 0067 §Decision 4 the action was
    /// sampled, clamped, and returned to the caller unchanged. Every step it
    /// counts is unattributable — on the CPU backend the network erased the
    /// observation and returned a finite, in-bounds, `is_valid() == true`
    /// action, and no loss guard, Q-value check, or action-validity check will
    /// ever fire on it. Treat a non-zero count as invalidating the affected
    /// episode's return, and fix the observation source.
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

    /// Test-only parameter checksums of the two **target** critics, as
    /// `[target_critic_1, target_critic_2]`. SAC has no target actor.
    ///
    /// The target-network observation seam of ADR 0058: nothing else in this
    /// crate can read a target's weights, which is how the issue-#182
    /// two-schedule defect survived its tests. Paired with
    /// [`live_checksums`](Self::live_checksums) it makes a target update's
    /// *cadence* and *magnitude* both assertable — see
    /// [`param_checksum`](crate::algorithms::shared::param_checksum) for why a
    /// sum suffices under Polyak averaging.
    #[cfg(test)]
    pub(crate) fn target_checksums(&self) -> [f64; 2] {
        [
            param_checksum::<B::InnerBackend, _>(&self.target_critic_1),
            param_checksum::<B::InnerBackend, _>(&self.target_critic_2),
        ]
    }

    /// Test-only parameter checksums of the two **live** critics, in the same
    /// order as [`target_checksums`](Self::target_checksums).
    #[cfg(test)]
    pub(crate) fn live_checksums(&self) -> [f64; 2] {
        [
            param_checksum::<B, _>(self.critic_1.get()),
            param_checksum::<B, _>(self.critic_2.get()),
        ]
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
    /// 5. Every [`target_update.every()`](crate::target::TargetUpdate::every)-th
    ///    critic step, Polyak-averages both critic targets by
    ///    [`target_update.tau()`](crate::target::TargetUpdate::tau).
    ///
    /// Returns `None` if the agent is still in warm-up.
    ///
    /// Every network stays in its [`Slot`](crate::algorithms::shared::Slot) for
    /// the whole forward / loss / `backward` region and is moved out only for
    /// its own optimizer step, so a panic anywhere in that region leaves all
    /// three networks intact and the agent usable.
    ///
    /// # Panics
    ///
    /// Panics if any network slot is poisoned by an earlier unwind *inside* an
    /// optimizer step. The three slots are stepped in sequence and never held
    /// simultaneously, so such a panic poisons only the one network being
    /// stepped.
    ///
    /// # Errors
    ///
    /// Returns [`SacAgentError::Polyak`] if a target soft-update finds a
    /// parameter-topology mismatch between a live critic and its target twin
    /// (see [`polyak_update`](crate::utils::polyak_update)). Every in-tree
    /// target is cloned from its active critic, so this cannot occur for
    /// agents built normally.
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
    #[allow(clippy::cast_possible_truncation)]
    pub fn learn_step<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
    ) -> Result<Option<LearnOutcome>, SacAgentError> {
        if !self.can_learn() {
            return Ok(None);
        }
        let batch_size = self.config.batch_size;
        let device = self.device.clone();

        // --- Sample batch ---
        // `can_learn()` above already established `buffer.len() >= batch_size`,
        // so the only variant `sample` can return here is unreachable; treat it
        // as a skipped step for safety.
        let Ok(batch) = self.buffer.sample(batch_size, UNIFORM_REPLAY_BETA, rng) else {
            return Ok(None);
        };

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
        let action_dim = self.actor.get().action_dim();
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
            terminated_inner,
            self.config.gamma,
        );
        let target: Tensor<B, 1> = Tensor::from_data(target_inner.into_data(), &device);

        // --- Critic updates: two independent backward passes ---
        // Both forwards run on borrows, so neither critic leaves its slot for
        // the shared forward / loss / backward region below. The two
        // `step_with` windows are therefore disjoint: `critic_2` survives a
        // panic inside `critic_1`'s optimizer step, and vice versa.
        let q1_pred: Tensor<B, 1> = self.critic_1.get().forward(obs_t.clone(), action_t.clone());
        let q2_pred: Tensor<B, 1> = self.critic_2.get().forward(obs_t.clone(), action_t);

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

        // #318 / ADR 0056: the two critics run in DISJOINT backward+step windows
        // (independent graphs), so each site gets its own guard — a non-finite
        // loss in one critic skips only that critic's `backward()` + optimizer
        // step, while the other still updates. `loss_1`/`loss_2` are already
        // host-resident (read via `.inner()` above), so the check costs no extra
        // sync. A skipped critic's value is excluded from the reported metric:
        // `last_qf{1,2}_loss` carries its last *applied* value forward rather
        // than folding in a NaN. `critic_updates` (the actor/α cadence counter)
        // still advances unconditionally, per ADR 0056 §3.
        if self.critic_1_guard.check(loss_1) {
            let grads_1 = loss_1_tensor.backward();
            let grads_1_params = GradientsParams::from_grads(grads_1, self.critic_1.get());
            self.critic_1.step_with(
                &mut self.critic_1_opt,
                self.config.critic_lr,
                grads_1_params,
            );
            self.last_qf1_loss = loss_1;
        }

        if self.critic_2_guard.check(loss_2) {
            let grads_2 = loss_2_tensor.backward();
            let grads_2_params = GradientsParams::from_grads(grads_2, self.critic_2.get());
            self.critic_2.step_with(
                &mut self.critic_2_opt,
                self.config.critic_lr,
                grads_2_params,
            );
            self.last_qf2_loss = loss_2;
        }

        self.critic_updates += 1;

        // --- Actor + α update (every policy_frequency-th critic step) ---
        let mut actor_loss_opt: Option<f32> = None;
        let mut entropy_opt: Option<f32> = None;
        if self
            .critic_updates
            .is_multiple_of(self.config.policy_frequency)
        {
            let eps: Tensor<B, DAB> =
                sample_noise::<B, R, DAB>(batch_size, action_dim, &device, rng);
            let sample = self.actor.get().forward_sample(obs_t.clone(), eps);
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
            let min_q_pi: Tensor<B, 1> = self.critic_1.get().forward(obs_t.clone(), sample.action);

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

            // #318 / ADR 0056: guard the actor site. A non-finite actor loss
            // skips the actor `backward()` + optimizer step and the snapshot
            // refresh, and leaves `actor_loss`/`entropy` reported as `None` for
            // this iteration (mirroring the delayed-update skip) rather than
            // folding a NaN into `last_actor_loss`. `actor_loss_value` is
            // already host-resident (via `.inner()`), so the check adds no sync.
            if self.actor_guard.check(actor_loss_value) {
                let grads = actor_loss_tensor.backward();
                let actor_grads = GradientsParams::from_grads(grads, self.actor.get());
                self.actor
                    .step_with(&mut self.actor_opt, self.config.actor_lr, actor_grads);
                // Refresh the inner-backend snapshot used by future target-Q
                // computations.
                self.actor_snapshot = self.actor.get().valid();

                self.last_actor_loss = actor_loss_value;
                self.last_entropy = entropy_value;
                actor_loss_opt = Some(actor_loss_value);
                entropy_opt = Some(entropy_value);
            }

            // α update (optional). Closed-form scalar Adam with its own #184
            // non-finite guard (`LogAlpha::adam_step`), independent of the
            // actor-loss guard above: it is driven by `log_prob_mean`, not the
            // actor loss, and keeps the α cadence honest even if the actor step
            // was skipped this iteration.
            if self.config.autotune {
                self.log_alpha.adam_step(
                    log_prob_mean,
                    self.target_entropy,
                    self.config.alpha_lr as f32,
                );
            }
            self.last_alpha = self.log_alpha.alpha();
        }

        // --- Target Polyak updates ---
        // `fires_at` takes the *post-increment* `critic_updates` counter and
        // yields the τ to apply, or `None` (ADR 0058). It reproduces the former
        // `critic_updates.is_multiple_of(target_update_frequency)` gate exactly,
        // and hands back an `f64` — the type `soft_update` already took, so the
        // old `f64::from(self.config.tau)` widening is gone.
        if let Some(tau) = self.config.target_update.fires_at(self.critic_updates) {
            // Clone rather than move out: `soft_update` consumes `target` by
            // value, so on `Err` the `?` returns before the reassignment and
            // each target field keeps its prior weights — no silent hard-sync
            // onto its live critic (the invariant now holds via early return,
            // and equally for a panic).
            self.target_critic_1 =
                Critic::soft_update(self.critic_1.get(), self.target_critic_1.clone(), tau)?;
            self.target_critic_2 =
                Critic::soft_update(self.critic_2.get(), self.target_critic_2.clone(), tau)?;
        }

        Ok(Some(LearnOutcome {
            // Report the most recent *applied* critic losses, so a skipped
            // (non-finite) step carries its last healthy value forward rather
            // than poisoning the metric with a NaN (#318, ADR 0056 §3).
            critic_loss: self.last_qf1_loss + self.last_qf2_loss,
            qf1_loss: self.last_qf1_loss,
            qf2_loss: self.last_qf2_loss,
            actor_loss: actor_loss_opt,
            alpha: self.last_alpha,
            entropy: entropy_opt,
            q_mean,
        }))
    }
}

/// Draws `rows × cols` iid standard-normal samples on CPU and assembles them
/// into a rank-`DAB` tensor of shape `[rows, cols]`. The built-in
/// [`SquashedGaussianPolicyHead`](crate::algorithms::sac::sac_policy::SquashedGaussianPolicyHead)
/// uses `DAB = 2`; the agent stays generic so higher-rank action layouts can
/// plug in a custom policy and their own `DAB`.
// `rand`'s standard-normal sampler yields f64; the tensor being filled is f32.
// Narrowing to the tensor's own dtype is the intent, and the sample is finite
// by construction.
#[allow(clippy::cast_possible_truncation)]
fn sample_noise<BB: Backend, R: Rng + ?Sized, const DAB: usize>(
    rows: usize,
    cols: usize,
    device: &<BB as burn::tensor::backend::BackendTypes>::Device,
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
    // Exact comparison is intentional throughout this test module: the values are
    // config literals read back unchanged, or a computed result whose bit-exactness
    // is itself the property under test (that an anneal lands exactly on its
    // endpoint, that `-0.0` is accepted as the no-correction setting). A tolerance
    // would let a real regression pass. Reviewed as a class, not site-by-site.
    #![allow(clippy::float_cmp)]
    use super::*;
    use crate::MAX_BUFFER_CAPACITY;
    use burn::backend::Flex;
    use rlevo_core::config::ConstraintKind;

    type BI = Flex;

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
    /// `α = 0.5`, `r = [0.1, 0.2, 0.3]`, `γ = 0.9`, `terminated = [0, 0, 1]`:
    ///   `min_q`          = [2.0, 0.5, 4.0]
    ///   `min_q` − α·logp = [2.0 − 0.05, 0.5 − 0.10, 4.0 − 0.15]
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
        let terminated =
            Tensor::<BI, 1>::from_data(TensorData::new(vec![0.0_f32, 0.0, 1.0], vec![3]), &device);

        let target = compute_sac_target(rewards, next_q1, next_q2, next_logp, 0.5, terminated, 0.9);
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

    // -------- non-finite-loss guard (ADR 0056, #318) --------

    use crate::algorithms::bootstrap_mask::{
        MaskContinuousAction, MaskObservation, TinyCritic, TinySacActor,
    };
    use crate::algorithms::sac::sac_config::SacTrainingConfigBuilder;
    use burn::backend::Autodiff;
    use burn::module::{Module, ModuleMapper, Param};
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rlevo_core::action::ContinuousAction;

    type Ad = Autodiff<Flex>;
    type GuardAgent = SacAgent<
        Ad,
        TinySacActor<Ad>,
        TinyCritic<Ad>,
        MaskObservation,
        MaskContinuousAction,
        1,
        2,
        1,
        2,
    >;

    /// Replaces every float parameter of a module with `NaN`, simulating a
    /// critic that has diverged to non-finite weights — the realistic source of
    /// a single non-finite critic loss.
    struct NanInjector;

    impl<B: Backend> ModuleMapper<B> for NanInjector {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
            let (id, tensor, mapper) = param.consume();
            Param::from_mapped_value(id, tensor.mul_scalar(f32::NAN), mapper)
        }
    }

    /// Builds a `MaskObservation` from two feature values, via its
    /// `TensorConvertible` seam (its fields are private to `bootstrap_mask`).
    fn make_obs(a: f32, b: f32) -> MaskObservation {
        let device = Default::default();
        let t = Tensor::<Ad, 1>::from_data(TensorData::new(vec![a, b], vec![2]), &device);
        <MaskObservation as TensorConvertible<1, Ad>>::from_tensor(t).expect("obs from tensor")
    }

    /// Evaluates critic-2 on a fixed (obs, action) pair and reads the scalar
    /// back — a change across a learn step proves the weights were updated.
    fn critic_2_probe(agent: &GuardAgent) -> f32 {
        let device = Default::default();
        let obs =
            Tensor::<Ad, 2>::from_data(TensorData::new(vec![0.3_f32, 0.7], vec![1, 2]), &device);
        let act = Tensor::<Ad, 2>::from_data(TensorData::new(vec![0.0_f32], vec![1, 1]), &device);
        agent
            .critic_2
            .get()
            .forward(obs, act)
            .inner()
            .into_scalar()
            .elem::<f32>()
    }

    /// One diverged critic must not poison the other (ADR 0056, `.inner()`
    /// pruning path, `sac_agent.rs:611-614`).
    ///
    /// The two critics run their backward + optimizer step in disjoint windows
    /// on independent graphs, so a non-finite loss in critic-1 must skip only
    /// critic-1's update while critic-2 still learns and the agent stays finite.
    #[test]
    fn sac_one_nonfinite_critic_skips_only_that_critic() {
        let device = Default::default();
        let config = SacTrainingConfigBuilder::new()
            .batch_size(2)
            .learning_starts(0)
            .replay_buffer_capacity(64)
            .critic_lr(0.05)
            // policy_frequency = 2 keeps the actor update off this single step,
            // isolating the twin-critic disjoint-window property under test.
            .policy_frequency(2)
            .autotune(false)
            .build()
            .expect("valid config");

        let mut agent = GuardAgent::new(
            TinySacActor::<Ad>::new(&device),
            TinyCritic::<Ad>::new(&device),
            TinyCritic::<Ad>::new(&device),
            config,
            device,
        )
        .expect("valid agent");

        // Prime the buffer past `batch_size` with finite transitions.
        let action = MaskContinuousAction::from_slice(&[0.0]);
        for x in [0.0_f32, 0.1, 0.2, 0.3] {
            agent.remember(
                make_obs(x, 1.0 - x),
                &action,
                0.5,
                make_obs(x + 0.1, 0.9 - x),
                false,
            );
        }

        // Diverge critic-1 to non-finite weights so ONLY its loss is NaN; the
        // shared Bellman target and critic-2 stay finite. Poison a *clone* of
        // the live critic-1 so its `ParamId`s are preserved and the target
        // Polyak pairing stays valid.
        let poisoned = agent.critic_1.get().clone().map(&mut NanInjector);
        agent.critic_1 = Slot::new(poisoned);

        let before = critic_2_probe(&agent);
        let mut rng = StdRng::seed_from_u64(0);
        // (c) no panic: a panic here (e.g. a pruned shared-leaf backward) fails
        // the test outright.
        let outcome = agent
            .learn_step(&mut rng)
            .expect("no polyak error")
            .expect("a primed agent past warm-up learns");
        let after = critic_2_probe(&agent);

        // (a) critic-1 skipped exactly this one update; the untouched sites
        // skipped nothing. Exact counts, not `warning_fired()` booleans: a
        // `!fired` assertion was satisfiable by an already-armed latch, whereas
        // `== 0` pins that the site was never entered at all (ADR 0072).
        assert_eq!(
            agent.skipped_critic_1_updates(),
            1,
            "critic-1's one non-finite loss must count exactly one skipped update"
        );
        assert_eq!(
            agent.skipped_critic_2_updates(),
            0,
            "critic-2's loss was finite: the sibling critic must be untouched, with zero skips"
        );
        assert_eq!(
            agent.skipped_actor_updates(),
            0,
            "the actor did not update this step (policy_frequency = 2): zero actor skips"
        );
        assert_eq!(
            agent.skipped_updates(),
            1,
            "the aggregate must equal the single critic-1 skip"
        );

        // (b) critic-2 still updated.
        assert!(
            (before - after).abs() > 1e-6,
            "critic-2 must still learn while critic-1 is skipped: {before} -> {after}"
        );

        // (d) the skip kept the poison contained — no global NaN.
        assert!(after.is_finite(), "critic-2 output must stay finite");
        assert!(
            outcome.qf2_loss.is_finite(),
            "the reported critic-2 loss must be finite"
        );
        assert_eq!(
            outcome.qf1_loss, 0.0,
            "the skipped critic-1 loss must not poison the reported metric (stays at its 0.0 seed)"
        );
        assert!(
            outcome.critic_loss.is_finite(),
            "the summed critic loss must stay finite despite the skipped site"
        );
        let probe_action = agent.act(&make_obs(0.2, 0.8), false, &mut rng);
        assert!(
            probe_action.as_slice()[0].is_finite(),
            "the actor must remain finite — critic-1's NaN must not reach it"
        );
    }

    /// Builds a primed agent whose live critic-1 has been diverged to `NaN`
    /// weights, so every subsequent critic-1 loss (and, once the cadence lets
    /// the actor run, every actor loss — the actor is scored against critic-1)
    /// is non-finite while critic-2's Bellman target stays clean.
    ///
    /// Shares the exact fixture of
    /// [`sac_one_nonfinite_critic_skips_only_that_critic`]: same config shape,
    /// same four priming transitions, same `NanInjector` applied to a *clone*
    /// of the live critic so `ParamId`s — and hence the target Polyak pairing —
    /// stay valid.
    ///
    /// # Why the target update is pushed out of range
    ///
    /// The one departure from that fixture is `target_update`, held at a
    /// cadence of 100 so it never fires inside these short runs. It is
    /// load-bearing and was found empirically: with the default per-step Polyak
    /// blend, step 1 mixes the *live* critic-1's `NaN` weights into
    /// `target_critic_1`, so from step 2 the shared Bellman target — a
    /// `min(target_q1, target_q2)` — is itself `NaN` and **critic-2 starts
    /// skipping too**. That contagion is correct behaviour, but it collapses
    /// every multi-step run onto "both critics skip", destroying the unequal
    /// per-site counts these tests exist to distinguish.
    fn poisoned_critic_1_agent(policy_frequency: usize) -> GuardAgent {
        let device = Default::default();
        let config = SacTrainingConfigBuilder::new()
            .batch_size(2)
            .learning_starts(0)
            .replay_buffer_capacity(64)
            .critic_lr(0.05)
            .policy_frequency(policy_frequency)
            // See the doc comment: keeps critic-1's NaN out of the targets, so
            // critic-2's loss stays finite for the duration of the run.
            .target_update(TargetUpdate::polyak(0.005, 100))
            .autotune(false)
            .build()
            .expect("valid config");

        let mut agent = GuardAgent::new(
            TinySacActor::<Ad>::new(&device),
            TinyCritic::<Ad>::new(&device),
            TinyCritic::<Ad>::new(&device),
            config,
            device,
        )
        .expect("valid agent");

        let action = MaskContinuousAction::from_slice(&[0.0]);
        for x in [0.0_f32, 0.1, 0.2, 0.3] {
            agent.remember(
                make_obs(x, 1.0 - x),
                &action,
                0.5,
                make_obs(x + 0.1, 0.9 - x),
                false,
            );
        }

        let poisoned = agent.critic_1.get().clone().map(&mut NanInjector);
        agent.critic_1 = Slot::new(poisoned);
        agent
    }

    /// The skip counter accumulates across repeated non-finite losses — it is a
    /// running total, not a "has it ever happened" flag (ADR 0072 §2).
    ///
    /// Three poisoned steps, not one: a `1` would also be produced by a
    /// bool-to-counter mistranslation (a latch that sets the count to `1` and
    /// never advances), which is the exact defect the counter replaced.
    #[test]
    fn sac_counts_repeated_loss_skips() {
        // policy_frequency = 8 keeps the actor cadence off all three steps, so
        // the only site that can skip is critic-1 and the aggregate is
        // unambiguous.
        let mut agent = poisoned_critic_1_agent(8);
        let mut rng = StdRng::seed_from_u64(0);

        for step in 0..3 {
            agent
                .learn_step(&mut rng)
                .expect("no polyak error")
                .expect("a primed agent past warm-up learns");
            assert_eq!(
                agent.skipped_critic_1_updates(),
                u64::try_from(step).expect("small loop index") + 1,
                "critic-1's skip count must advance by exactly one per poisoned step"
            );
        }

        assert_eq!(
            agent.skipped_critic_1_updates(),
            3,
            "three consecutive non-finite critic-1 losses must count exactly three skips"
        );
        assert_eq!(
            agent.critic_updates(),
            3,
            "critic_updates counts ATTEMPTS and must advance on every skipped step \
             (ADR 0059 §Decision 4), giving applied = 3 - 3 = 0 for critic-1"
        );
        assert_eq!(
            agent.skipped_critic_2_updates(),
            0,
            "critic-2's losses stayed finite: zero skips at the sibling site"
        );
        assert_eq!(
            agent.skipped_actor_updates(),
            0,
            "policy_frequency = 8 kept the actor site unreached: zero actor skips"
        );
        assert_eq!(
            agent.skipped_updates(),
            3,
            "the aggregate must equal the three critic-1 skips and nothing else"
        );
    }

    /// The aggregate sums three *different* per-site counts — it does not read
    /// one guard three times, and it does not drop one (ADR 0072 §3).
    ///
    /// The distribution is **critic-1 = 5, critic-2 = 2, actor = 1,
    /// aggregate = 8**: three *nonzero, pairwise distinct* terms. Both
    /// properties are load-bearing, and for different defects.
    ///
    /// - **Pairwise distinct** kills duplication. Every chain that reads one
    ///   guard twice lands elsewhere: `5+5+2 = 12`, `5+5+1 = 11`,
    ///   `2+2+5 = 9`, `2+2+1 = 5`, `1+1+5 = 7`, `1+1+2 = 4`, and the
    ///   all-one-guard sums `15 / 6 / 3`. None is `8`.
    /// - **All nonzero** kills omission. A zero term is invisible to the sum, so
    ///   a chain that simply forgets that summand still totals correctly; the
    ///   earlier shape of this test (3 / **0** / 1 = 4) passed against a
    ///   `skipped_updates` with `skipped_critic_2_updates()` deleted. Here the
    ///   three single-term omissions give `5+2 = 7`, `5+1 = 6`, `2+1 = 3` —
    ///   again none is `8`. `8` is reachable only by summing each of the three
    ///   guards exactly once.
    ///
    /// # Deriving the triple from the cadence
    ///
    /// The run is five learn steps with `policy_frequency = 4`, and critic-2 is
    /// diverged *part-way through* — after step 3 — which is what makes the two
    /// critics skip by different amounts rather than in lockstep:
    ///
    /// - critic-1 is `NaN` from step 1 and its optimizer step is skipped every
    ///   time, so its weights are never overwritten: non-finite loss on all five
    ///   steps → **critic-1 = 5**.
    /// - critic-2 runs clean for steps 1-3 (its Bellman target is built from the
    ///   *target* critics and the actor snapshot, and the out-of-range Polyak
    ///   cadence documented on [`poisoned_critic_1_agent`] keeps critic-1's
    ///   `NaN` out of `target_critic_1`). It is diverged before step 4 and, like
    ///   critic-1, is never re-stepped afterwards: non-finite on steps 4 and 5 →
    ///   **critic-2 = 2**.
    /// - the actor site is reached only when `critic_updates` is a multiple of
    ///   4, i.e. on step 4 alone within a five-step run. The actor is scored
    ///   against critic-1 (see the `min_q_pi` note in `learn_step`), whose
    ///   weights are `NaN`, so that single attempt skips → **actor = 1**. This
    ///   is also the concrete case for the "actor attempts are fewer than critic
    ///   attempts by construction" note on
    ///   [`SacAgent::skipped_actor_updates`].
    ///
    /// Mid-run divergence, rather than poisoning both critics up front, is what
    /// preserves the inequality: poisoning both at step 0 would give `5 / 5 / 1`
    /// and re-open the duplication hole between the two critic terms.
    ///
    /// The sibling-isolation property (one bad critic does not drag the other
    /// down) is *not* weakened here — it is pinned by
    /// [`sac_one_nonfinite_critic_skips_only_that_critic`] and by the
    /// intermediate `critic-2 == 0` assertion below, which holds right up to the
    /// step where critic-2 is diverged deliberately.
    #[test]
    fn sac_aggregate_skip_count_sums_unequal_sites() {
        let mut agent = poisoned_critic_1_agent(4);
        let mut rng = StdRng::seed_from_u64(0);

        for _ in 0..3 {
            agent
                .learn_step(&mut rng)
                .expect("no polyak error")
                .expect("a primed agent past warm-up learns");
        }

        // Sibling isolation still holds: three steps of a `NaN` critic-1 have
        // not made critic-2 skip once.
        assert_eq!(
            agent.skipped_critic_2_updates(),
            0,
            "critic-2 must still be clean after three critic-1 skips — the \
             mid-run divergence below is the ONLY reason it starts skipping"
        );

        // Diverge critic-2 as well, via a clone so `ParamId`s (and the target
        // Polyak pairing) survive — exactly the treatment critic-1 got in
        // `poisoned_critic_1_agent`.
        let poisoned_2 = agent.critic_2.get().clone().map(&mut NanInjector);
        agent.critic_2 = Slot::new(poisoned_2);

        for _ in 0..2 {
            agent
                .learn_step(&mut rng)
                .expect("no polyak error")
                .expect("a primed agent past warm-up learns");
        }

        assert_eq!(
            agent.skipped_critic_1_updates(),
            5,
            "critic-1's loss was non-finite on all five steps: exactly five skips"
        );
        assert_eq!(
            agent.skipped_critic_2_updates(),
            2,
            "critic-2 was diverged after step 3 and never re-stepped: exactly two \
             skips, on steps 4 and 5 — a NONZERO term, so an aggregate that omits \
             this summand cannot still total 8"
        );
        assert_eq!(
            agent.skipped_actor_updates(),
            1,
            "the actor site was reached once in five steps (policy_frequency = 4, \
             on critic update 4) and its critic-1-scored loss was non-finite: \
             exactly one skip"
        );
        assert_eq!(
            agent.critic_updates(),
            5,
            "attempts advance unconditionally (ADR 0059 §Decision 4): five learn \
             steps read 5, pinning the cadence arithmetic the triple rests on"
        );
        assert_eq!(
            agent.skipped_updates(),
            8,
            "the aggregate must be 5 + 2 + 1 = 8 — a sum over three DISTINCT, \
             pairwise-unequal, NONZERO guards: no single-guard duplication and no \
             single-summand omission reaches 8"
        );
    }

    // -------- target-update cadence (ADR 0058 / 0059, #334) --------

    use crate::target::TargetUpdate;
    use approx::assert_abs_diff_eq;

    /// Slack for the Polyak identity below. Each checksum is an `f32` device
    /// reduction over a handful of parameters and each blended parameter costs
    /// two `f32` roundings, so ~2e-6 is the realistic worst case; the smallest
    /// signal any assertion here reads is `τ · gap ≈ 1e-2`, three orders of
    /// magnitude larger.
    const CHECKSUM_EPS: f64 = 1e-5;

    /// Adds a constant to every float parameter of a module.
    ///
    /// Both critic targets are built by cloning their live critic, so they
    /// start *identical* — and a Polyak blend between identical networks is a
    /// no-op that every "did the target move, and by how much?" assertion would
    /// pass vacuously. That exact vacuity is how the issue-#182 defect survived
    /// its tests; this mapper removes it by opening a known gap. It maps a
    /// *clone*, so `ParamId`s are preserved and the Polyak pairing stays valid.
    struct AddConstant(f32);

    impl<B: Backend> ModuleMapper<B> for AddConstant {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
            let (id, tensor, mapper) = param.consume();
            Param::from_mapped_value(id, tensor.add_scalar(self.0), mapper)
        }
    }

    /// A primed agent whose live critics are held a known distance from their
    /// targets, so every cadence assertion below is non-vacuous.
    fn cadence_agent(target_update: TargetUpdate) -> GuardAgent {
        let device = Default::default();
        let config = SacTrainingConfigBuilder::new()
            .batch_size(2)
            .learning_starts(0)
            .replay_buffer_capacity(64)
            .actor_lr(1e-3)
            .critic_lr(1e-3)
            .autotune(false)
            .target_update(target_update)
            .build()
            .expect("valid config");

        let mut agent = GuardAgent::new(
            TinySacActor::<Ad>::new(&device),
            TinyCritic::<Ad>::new(&device),
            TinyCritic::<Ad>::new(&device),
            config,
            device,
        )
        .expect("valid agent");

        let action = MaskContinuousAction::from_slice(&[0.0]);
        for x in [0.0_f32, 0.1, 0.2, 0.3] {
            agent.remember(
                make_obs(x, 1.0 - x),
                &action,
                0.5,
                make_obs(x + 0.1, 0.9 - x),
                false,
            );
        }

        // Nudge the *targets*, not the live critics. Perturbing a live network
        // would disturb Burn's autodiff registration across consecutive learn
        // steps; the targets are plain inner-backend modules with no graph
        // attached, and `AddConstant` preserves their `ParamId`s, so the Polyak
        // pairing holds.
        agent.target_critic_1 = agent.target_critic_1.clone().map(&mut AddConstant(0.5));
        agent.target_critic_2 = agent.target_critic_2.clone().map(&mut AddConstant(0.5));
        for (i, (live, target)) in agent
            .live_checksums()
            .iter()
            .zip(agent.target_checksums().iter())
            .enumerate()
        {
            assert!(
                (live - target).abs() > 0.5,
                "precondition: critic {i}'s live and target checksums must differ, or a \
                 Polyak no-op would satisfy every assertion below; got live {live} vs \
                 target {target}"
            );
        }
        agent
    }

    /// SAC's shipped cadence is `polyak(0.005, 1)`: both critic targets move on
    /// **every** critic update, by exactly τ of the gap.
    ///
    /// The pre-ADR-0058 gate was `critic_updates.is_multiple_of(1)`, a no-op
    /// that no test varied — so "every step" was assumed, never observed.
    #[test]
    fn sac_default_cadence_fires_on_every_critic_update() {
        let rule = TargetUpdate::polyak(0.005, 1);
        let tau = rule.tau();
        let mut agent = cadence_agent(rule);
        let mut rng = StdRng::seed_from_u64(0);

        for _ in 1..=3_usize {
            let before = agent.target_checksums();
            agent
                .learn_step(&mut rng)
                .expect("no polyak error")
                .expect("a primed agent past warm-up learns");
            let after = agent.target_checksums();
            let live = agent.live_checksums();

            for (i, ((&b, &a), &l)) in before.iter().zip(after.iter()).zip(live.iter()).enumerate()
            {
                assert_abs_diff_eq!(a, (1.0 - tau) * b + tau * l, epsilon = CHECKSUM_EPS);
                assert!(
                    (a - b).abs() > 1e-4,
                    "critic {i}'s target must actually have moved: {b} -> {a}"
                );
            }
        }
    }

    /// A cadence SAC's flat `target_update_frequency` could express but nothing
    /// in-tree ever exercised: every second critic update, silent in between.
    #[test]
    fn sac_cadence_of_two_skips_odd_critic_updates() {
        let rule = TargetUpdate::polyak(0.005, 2);
        let tau = rule.tau();
        let mut agent = cadence_agent(rule);
        let mut rng = StdRng::seed_from_u64(0);

        for critic_update in 1..=4_usize {
            let before = agent.target_checksums();
            agent
                .learn_step(&mut rng)
                .expect("no polyak error")
                .expect("a primed agent past warm-up learns");
            let after = agent.target_checksums();

            if critic_update.is_multiple_of(2) {
                let live = agent.live_checksums();
                for (i, ((&b, &a), &l)) in
                    before.iter().zip(after.iter()).zip(live.iter()).enumerate()
                {
                    assert_abs_diff_eq!(a, (1.0 - tau) * b + tau * l, epsilon = CHECKSUM_EPS);
                    assert!(
                        (a - b).abs() > 1e-4,
                        "critic {i}'s target must actually have moved: {b} -> {a}"
                    );
                }
            } else {
                assert_eq!(
                    after, before,
                    "critic update {critic_update} is not a multiple of the cadence 2, so \
                     every target weight must be untouched"
                );
            }
        }
    }

    // ---- ADR 0065 / #352: non-finite reward is dropped at ingestion ----
    //
    // Every off-policy agent needs its OWN copy of this test. The defect had
    // six sites, not the four the issue named, precisely because C51 and
    // QR-DQN were added by copying an unguarded `remember` and no shared test
    // noticed. A per-file test is what makes agent #7's author notice.

    #[test]
    fn sac_remember_drops_a_nonfinite_reward() {
        let device = Default::default();
        let config = SacTrainingConfigBuilder::new()
            .build()
            .expect("valid config");
        let mut agent = GuardAgent::new(
            TinySacActor::<Ad>::new(&device),
            TinyCritic::<Ad>::new(&device),
            TinyCritic::<Ad>::new(&device),
            config,
            device,
        )
        .expect("valid agent");
        let action = MaskContinuousAction::from_slice(&[0.0]);

        agent.remember(
            make_obs(0.0, 1.0),
            &action,
            f32::NAN,
            make_obs(0.1, 0.9),
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

        agent.remember(make_obs(0.1, 0.9), &action, 0.5, make_obs(0.2, 0.8), false);
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
    // Same per-file rule as the reward test above: each agent carries its own
    // copy rather than trusting one shared test to stand in for three call
    // sites. `MaskObservation` is the deliberately NaN-emitting observation
    // type here — its `write_host_row` copies its `[f32; 2]` payload verbatim,
    // so `make_obs(f32::NAN, _)` puts a real NaN in the staged row.

    /// A fresh agent with capacity to spare, for the observation-guard tests.
    fn obs_guard_agent() -> GuardAgent {
        let device = Default::default();
        let config = SacTrainingConfigBuilder::new()
            .replay_buffer_capacity(64)
            .build()
            .expect("valid config");
        GuardAgent::new(
            TinySacActor::<Ad>::new(&device),
            TinyCritic::<Ad>::new(&device),
            TinyCritic::<Ad>::new(&device),
            config,
            device,
        )
        .expect("valid agent")
    }

    #[test]
    fn sac_remember_drops_a_nonfinite_obs() {
        let mut agent = obs_guard_agent();
        let action = MaskContinuousAction::from_slice(&[0.0]);

        agent.remember(
            make_obs(f32::NAN, 1.0),
            &action,
            0.5,
            make_obs(0.1, 0.9),
            false,
        );
        assert_eq!(
            agent.buffer_len(),
            0,
            "a transition whose `obs` carries NaN must never enter the replay buffer"
        );
        assert_eq!(
            agent.dropped_observations(),
            1,
            "the drop must be visible through the public observation counter"
        );
        assert_eq!(
            agent.dropped_transitions(),
            0,
            "the reward was finite: the reward counter must not move"
        );

        agent.remember(make_obs(0.1, 0.9), &action, 0.5, make_obs(0.2, 0.8), false);
        assert_eq!(
            agent.buffer_len(),
            1,
            "the drop must not latch: the next finite transition is still stored"
        );
        assert_eq!(
            agent.dropped_observations(),
            1,
            "a finite observation must not increment the drop counter"
        );
    }

    #[test]
    fn sac_remember_drops_a_nonfinite_next_obs() {
        let mut agent = obs_guard_agent();
        let action = MaskContinuousAction::from_slice(&[0.0]);

        agent.remember(
            make_obs(0.0, 1.0),
            &action,
            0.5,
            make_obs(0.1, f32::INFINITY),
            false,
        );
        assert_eq!(
            agent.buffer_len(),
            0,
            "a transition whose `next_obs` carries ±Inf must never enter the replay buffer"
        );
        assert_eq!(
            agent.dropped_observations(),
            1,
            "the `next_obs` drop must be visible through the public counter"
        );

        agent.remember(make_obs(0.1, 0.9), &action, 0.5, make_obs(0.2, 0.8), false);
        assert_eq!(
            agent.buffer_len(),
            1,
            "the drop must not latch: the next finite transition is still stored"
        );
        assert_eq!(
            agent.dropped_observations(),
            1,
            "a finite `next_obs` must not increment the drop counter"
        );
    }

    /// ADR 0067 §Consequences: the reward guard runs first and returns early,
    /// so a transition that is bad in *both* ways increments only
    /// `dropped_transitions`. This test pins that ordering — it is the reason
    /// the two counters legitimately disagree, and both accessors document it.
    #[test]
    fn sac_remember_both_bad_counts_only_the_reward_drop() {
        let mut agent = obs_guard_agent();
        let action = MaskContinuousAction::from_slice(&[0.0]);

        agent.remember(
            make_obs(f32::NAN, 1.0),
            &action,
            f32::NAN,
            make_obs(f32::NAN, 0.9),
            false,
        );
        assert_eq!(agent.buffer_len(), 0, "the transition must be dropped");
        assert_eq!(
            agent.dropped_transitions(),
            1,
            "the reward guard runs first and must own this drop"
        );
        assert_eq!(
            agent.dropped_observations(),
            0,
            "the reward guard returned early: the observation guard must not \
             also count the same transition"
        );
    }

    /// ADR 0067 §Decision 4: `act` detects and reports, and **returns the
    /// action anyway**. Not substituting is the decision under test, so the
    /// assertion that an action comes back is as load-bearing as the counter.
    #[test]
    fn sac_act_counts_a_nonfinite_obs_and_still_returns_an_action() {
        let agent = obs_guard_agent();
        let mut rng = StdRng::seed_from_u64(0);

        let action = agent.act(&make_obs(f32::NAN, 1.0), false, &mut rng);
        assert_eq!(
            agent.degenerate_action_selections(),
            1,
            "a non-finite observation at `act` must be counted"
        );
        assert_eq!(
            action.as_slice().len(),
            1,
            "the action must still be returned: ADR 0067 §Decision 4 substitutes nothing"
        );

        // Non-latching, and a healthy observation is not counted.
        let _ = agent.act(&make_obs(0.2, 0.8), false, &mut rng);
        assert_eq!(
            agent.degenerate_action_selections(),
            1,
            "a finite observation must not increment the act counter"
        );
    }

    /// An out-of-range `replay_buffer_capacity` must come back as an `Err`
    /// from `new`, never as a panic. `UniformReplay::new` *asserts* on both
    /// bounds, so before this constructor took the fallible
    /// `UniformReplay::from_config` path the only thing standing between a bad
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
            let device = Default::default();
            let config = SacTrainingConfig {
                replay_buffer_capacity: capacity,
                ..SacTrainingConfig::default()
            };
            let Err(err) = GuardAgent::new(
                TinySacActor::<Ad>::new(&device),
                TinyCritic::<Ad>::new(&device),
                TinyCritic::<Ad>::new(&device),
                config,
                device,
            ) else {
                panic!("capacity {capacity} must be rejected, not allocated");
            };
            assert_eq!(err.config, "SacTrainingConfig");
            assert_eq!(err.field, "replay_buffer_capacity");
            assert_eq!(err.kind, kind);
        }
    }
}
