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
//! (3) a delayed actor update every `policy_frequency`-th critic step,
//! alongside a target Polyak update on its own, independently configured
//! cadence (ADR 0058).
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

use crate::algorithms::ddpg::exploration::GaussianNoise;
#[cfg(test)]
use crate::algorithms::shared::param_checksum;
use crate::algorithms::shared::{
    FiniteLossGuard, FiniteObsGuard, FiniteRewardGuard, Slot, UNIFORM_REPLAY_BETA,
    action_bound_tensors, assert_bounds_match_components,
};
use crate::algorithms::td3::target_smoothing::smoothed_target_action;
use crate::algorithms::td3::td3_config::Td3TrainingConfig;
use crate::algorithms::td3::td3_model::{ContinuousQ, DeterministicPolicy};
use crate::utils::{PolyakError, compute_target_q_values};

/// Error variants returned by [`Td3Agent`] operations.
///
/// Two variants, and each has a real construction site in this crate.
/// [`InvalidAction`](Td3AgentError::InvalidAction) is built by
/// [`train`](super::train::train) when `env.reset()` or `env.step()` rejects an
/// action the agent proposed. [`Polyak`](Td3AgentError::Polyak) arrives through
/// `?` on the target soft-updates at the end of
/// [`learn_step`](Td3Agent::learn_step), where a live network and its target
/// twin can disagree on parameter topology.
///
/// The enum is `#[non_exhaustive]`: downstream `match` expressions must carry a
/// wildcard arm, so a future variant is not a breaking change.
///
/// # Why there is no tensor-conversion variant
///
/// This enum previously carried `TensorConversionFailed(String)`, which nothing
/// constructed and nothing could. The tensor host-reads on the act path are the
/// `as_slice::<f32>()` calls in [`act`](Td3Agent::act) and
/// [`act_with`](Td3Agent::act_with), and both methods return a bare `A` rather
/// than a `Result` — there is no error channel for a variant to travel down. The
/// read is an `.expect` on a named invariant, which is precisely the form
/// `docs/rules.md` §4 sanctions for a host-read that "cannot fail by
/// construction": the tensor is one the same function just built from its own
/// actor. Making that path fallible is a breaking change, deferred and
/// tracked separately.
///
/// When it lands, the variant returns as
/// `#[from]` [`rlevo_core::base::TensorConversionError`] — not as a `String`.
/// §4 prefers structured variants over string-based errors and names that type
/// as the tensor-op error domain, so re-introducing a `String` payload would
/// rebuild the discarded variant in a merely-live form.
///
/// # Why there is no buffer or I/O variant
///
/// `Buffer(#[from] ReplayBufferError)` was unreachable by design.
/// [`learn_step`](Td3Agent::learn_step) samples with
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
pub enum Td3AgentError {
    /// The sampled or requested action is outside the valid action space.
    #[error("Invalid action: {0}")]
    InvalidAction(String),
    /// A target soft-update failed because a live network and its target twin
    /// have mismatched parameter topologies.
    #[error(transparent)]
    Polyak(#[from] PolyakError),
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

/// Computes the twin-critic TD target:
/// `y = r + γ · (1 − terminated) · min(next_q1, next_q2)`.
///
/// Exposed at crate visibility so the unit tests can cover the core TD3
/// invariant without standing up a full agent.
pub(crate) fn compute_twin_critic_target<BK: Backend>(
    rewards: Tensor<BK, 1>,
    next_q1: Tensor<BK, 1>,
    next_q2: Tensor<BK, 1>,
    terminated: Tensor<BK, 1>,
    gamma: f32,
) -> Tensor<BK, 1> {
    let min_q = next_q1.min_pair(next_q2);
    compute_target_q_values(rewards, min_q, terminated, gamma)
}

/// Twin Delayed DDPG (TD3) agent.
///
/// Holds the live actor, two independent critics, their three Polyak-averaged
/// target networks, three separate Adam optimizers, a FIFO replay buffer, and
/// per-episode statistics. The public surface is intentionally narrow: callers
/// interact through [`act`](Self::act), [`remember`](Self::remember),
/// [`on_env_step`](Self::on_env_step), and [`learn_step`](Self::learn_step).
/// The high-level collect-learn loop is driven by [`super::train::train`].
///
/// # Const generics
///
/// Same layout as [`DdpgAgent`](crate::algorithms::ddpg::ddpg_agent::DdpgAgent):
/// - `OR` — rank of a single observation tensor.
/// - `BOR` — rank of a batched observation tensor (= `OR + 1`).
/// - `AR` — rank of a single action tensor.
/// - `BAR` — rank of a batched action tensor (= `AR + 1`).
///
/// # Network ownership
///
/// Burn's [`Optimizer::step`](burn::optim::Optimizer::step) consumes the module
/// it updates, so each network must briefly leave the agent to be trained. Every
/// *fallible* operation — the forward pass, the loss, `backward`, and gradient
/// reduction — instead runs on a borrow, so a network is out of the agent only
/// for the duration of the `step` call itself. The actor and the two critics are
/// stepped in three disjoint windows: a panic while stepping `critic_1` leaves
/// `critic_2` and the actor fully intact and usable.
///
/// # Panics
///
/// That residual step window is irreducible: if `Optimizer::step` itself panics
/// (a shape mismatch, a device error), the network was already moved into `step`
/// and is dropped during unwinding, so there is nothing left to restore — a drop
/// guard or `catch_unwind` cannot recover it, and keeping a spare copy would
/// cost a full clone of every parameter tensor on every step. The agent is then
/// *poisoned* with respect to that one network: every later read of it —
/// [`act`](Self::act), [`inference_net`](Self::inference_net),
/// [`learn_step`](Self::learn_step) — panics with a message naming the cause and
/// the remedy, which is to rebuild the agent from a fresh network.
pub struct Td3Agent<
    B,
    Actor,
    Critic,
    O,
    A,
    const OR: usize,
    const BOR: usize,
    const AR: usize,
    const BAR: usize,
> where
    B: AutodiffBackend,
    Actor: DeterministicPolicy<B, BOR, BAR>,
    Critic: ContinuousQ<B, BOR, BAR>,
    O: Observation<OR> + TensorConvertible<OR, B> + TensorConvertible<OR, B::InnerBackend>,
    A: BoundedAction<AR>,
{
    actor: Slot<Actor>,
    target_actor: Actor::InnerModule,
    critic_1: Slot<Critic>,
    critic_2: Slot<Critic>,
    target_critic_1: Critic::InnerModule,
    target_critic_2: Critic::InnerModule,
    actor_opt: OptimizerAdaptor<Adam, Actor, B>,
    critic_1_opt: OptimizerAdaptor<Adam, Critic, B>,
    critic_2_opt: OptimizerAdaptor<Adam, Critic, B>,
    buffer: UniformReplay<ContinuousTransition<O>>,
    exploration: GaussianNoise,
    low: &'static [f32],
    high: &'static [f32],
    /// `[1, ..action_shape]` per-component bounds for the Eq. 14 target clip,
    /// built once at construction — see [`action_bound_tensors`].
    low_t: Tensor<B::InnerBackend, BAR>,
    high_t: Tensor<B::InnerBackend, BAR>,
    config: Td3TrainingConfig,
    device: B::Device,
    step: usize,
    critic_updates: usize,
    stats: AgentStats<Td3Metrics>,
    last_actor_loss: f32,
    /// Most recent *applied* critic-1 loss — carried forward across a
    /// non-finite skip so the reported metric never folds in a NaN.
    last_qf1_loss: f32,
    /// Most recent *applied* critic-2 loss (see [`Self::last_qf1_loss`]).
    last_qf2_loss: f32,
    /// Non-finite-loss guard for the critic-1 loss site (ADR 0056). The
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
    /// Non-finite-reward guard for the `remember` ingestion site (ADR 0065).
    /// Drops the transition on every occurrence; the `warn!` escalates
    /// by decades.
    reward_guard: FiniteRewardGuard,
    /// Non-finite-**observation** guard for the `remember` ingestion site (ADR
    /// 0067). Drops the transition on every occurrence; the `warn!`
    /// escalates by decades. Runs *after* [`Self::reward_guard`], which returns
    /// early — so the two counters are not additive (see
    /// [`dropped_observations`](Self::dropped_observations)).
    obs_guard: FiniteObsGuard,
    /// Non-finite-observation guard for the action-selection sites
    /// ([`act`](Self::act), [`act_with`](Self::act_with)). Detect-and-report
    /// only: it counts and warns, and the action is returned unchanged (ADR
    /// 0067 Decision 4).
    act_obs_guard: FiniteObsGuard,
    /// Reusable host staging buffer for the ingestion-side row-finiteness
    /// check. `remember` takes `&mut self`, so it can own one buffer and
    /// amortize the allocation across calls; the `&self` action-selection path
    /// cannot, and pays a per-call `Vec` instead — see [`act`](Self::act).
    obs_scratch: Vec<f32>,
    _action: PhantomData<A>,
}

impl<B, Actor, Critic, O, A, const OR: usize, const BOR: usize, const AR: usize, const BAR: usize>
    std::fmt::Debug for Td3Agent<B, Actor, Critic, O, A, OR, BOR, AR, BAR>
where
    B: AutodiffBackend,
    Actor: DeterministicPolicy<B, BOR, BAR>,
    Critic: ContinuousQ<B, BOR, BAR>,
    O: Observation<OR> + TensorConvertible<OR, B> + TensorConvertible<OR, B::InnerBackend>,
    A: BoundedAction<AR>,
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
            .finish_non_exhaustive()
    }
}

impl<B, Actor, Critic, O, A, const OR: usize, const BOR: usize, const AR: usize, const BAR: usize>
    Td3Agent<B, Actor, Critic, O, A, OR, BOR, AR, BAR>
where
    B: AutodiffBackend,
    Actor: DeterministicPolicy<B, BOR, BAR>,
    Critic: ContinuousQ<B, BOR, BAR>,
    O: Observation<OR> + TensorConvertible<OR, B> + TensorConvertible<OR, B::InnerBackend>,
    A: BoundedAction<AR>,
{
    /// Constructs a new agent from pre-built actor and two independent
    /// critic networks.
    ///
    /// The caller is expected to initialise `critic_1` and `critic_2` with
    /// different random seeds — Fujimoto et al. rely on independent initial
    /// errors so the `min` target actually suppresses overestimation.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`](rlevo_core::config::ConfigError) if `config`
    /// fails [`Td3TrainingConfig::validate`](rlevo_core::config::Validate::validate).
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
        critic_1: Critic,
        critic_2: Critic,
        config: Td3TrainingConfig,
        device: B::Device,
    ) -> Result<Self, rlevo_core::config::ConfigError> {
        config.validate()?;
        assert_bounds_match_components::<AR, A>();
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
        let (low_t, high_t) = action_bound_tensors::<B::InnerBackend, A, AR, BAR>(&device);
        // The capacity is a runtime config field, not a literal, so it takes
        // the fallible path: an out-of-range value is a `ConfigError` naming
        // `capacity`, never an allocation abort inside `VecDeque`.
        let buffer = UniformReplay::from_config(UniformReplayConfig {
            capacity: config.replay_buffer_capacity,
        })?;
        Ok(Self {
            actor: Slot::new(actor),
            target_actor,
            critic_1: Slot::new(critic_1),
            critic_2: Slot::new(critic_2),
            target_critic_1,
            target_critic_2,
            actor_opt,
            critic_1_opt,
            critic_2_opt,
            buffer,
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
            last_qf1_loss: 0.0,
            last_qf2_loss: 0.0,
            critic_1_guard: FiniteLossGuard::new("td3/critic_1"),
            critic_2_guard: FiniteLossGuard::new("td3/critic_2"),
            actor_guard: FiniteLossGuard::new("td3/actor"),
            reward_guard: FiniteRewardGuard::new("td3/remember"),
            obs_guard: FiniteObsGuard::ingestion("td3/remember"),
            act_obs_guard: FiniteObsGuard::act("td3/act"),
            obs_scratch: Vec::new(),
            _action: PhantomData,
        })
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

    /// Number of critic updates **attempted** so far. Exposed for tests that
    /// need to verify the delayed-update schedule.
    ///
    /// This is the cadence counter both the actor gate (`policy_frequency`) and
    /// the target gate ([`TargetUpdate::fires_at`]) read, so it advances
    /// **unconditionally** — including on a learn step whose critic loss was
    /// non-finite and therefore skipped (ADR 0059 §Decision 4). It is an
    /// attempt count, not an applied count.
    ///
    /// Subtract [`skipped_critic_1_updates`](Self::skipped_critic_1_updates) or
    /// [`skipped_critic_2_updates`](Self::skipped_critic_2_updates) to recover
    /// how many of those attempts actually reached that critic's optimizer:
    /// `applied = attempts − skipped`.
    ///
    /// [`TargetUpdate::fires_at`]: crate::target::TargetUpdate::fires_at
    pub fn critic_updates(&self) -> usize {
        self.critic_updates
    }

    /// Number of critic-1 gradient updates skipped because the critic-1 loss
    /// was non-finite (ADR 0056, ADR 0072).
    ///
    /// Each skip is one entered learn step whose `backward()` and optimizer
    /// step were both suppressed so a `NaN`/`±Inf` loss could not be folded
    /// into critic-1's weights. [`critic_updates`](Self::critic_updates) counts
    /// **attempts** and advances anyway, so
    /// `applied = critic_updates() − skipped_critic_1_updates()`.
    ///
    /// # Not the same event as a *drop*
    ///
    /// [`dropped_transitions`](Self::dropped_transitions) and
    /// [`dropped_observations`](Self::dropped_observations) count **data that
    /// never entered the replay buffer**. This counts an **update that never
    /// reached the optimizer**. A run can have zero drops and many skips (a
    /// diverging critic on clean data), or the reverse.
    ///
    /// # The two critics skip independently
    ///
    /// Critic-1 and critic-2 run their `backward()` + optimizer step in
    /// disjoint windows on independent graphs, each behind its own guard, so
    /// one can skip while the other applies on the very same learn step. Do not
    /// expect these two counters to agree.
    #[must_use]
    pub const fn skipped_critic_1_updates(&self) -> u64 {
        self.critic_1_guard.skipped()
    }

    /// Number of critic-2 gradient updates skipped because the critic-2 loss
    /// was non-finite (ADR 0056, ADR 0072).
    ///
    /// The critic-1 counter's twin; see
    /// [`skipped_critic_1_updates`](Self::skipped_critic_1_updates) for the
    /// attempts-versus-skips relationship, the skip-versus-drop distinction,
    /// and why the two critics' counts need not agree.
    #[must_use]
    pub const fn skipped_critic_2_updates(&self) -> u64 {
        self.critic_2_guard.skipped()
    }

    /// Number of actor gradient updates skipped because the actor loss was
    /// non-finite (ADR 0056, ADR 0072).
    ///
    /// # Not comparable with the critic counts
    ///
    /// The actor guard sits **inside** the `policy_frequency` cadence block
    /// (Fujimoto et al. 2018 delayed policy update), so the actor is
    /// only *attempted* on every `policy_frequency`-th critic update. At the
    /// shipped default of `2` the actor therefore has roughly half as many
    /// attempts as either critic, and a smaller skip count than critic-1 does
    /// **not** mean the actor is healthier. Compare each count against its own
    /// attempt base — `critic_updates() / policy_frequency` for the actor — not
    /// against another site's.
    ///
    /// The skip-versus-drop distinction is the same as for
    /// [`skipped_critic_1_updates`](Self::skipped_critic_1_updates): a *drop*
    /// is data that never entered the buffer, a *skip* is an update that never
    /// reached the optimizer.
    #[must_use]
    pub const fn skipped_actor_updates(&self) -> u64 {
        self.actor_guard.skipped()
    }

    /// Total gradient updates skipped for a non-finite loss across **all** of
    /// this agent's loss sites — critic-1, critic-2 and the actor.
    ///
    /// This is the canonical `skipped_updates` metric the training loop emits
    /// (ADR 0072). It exists as the single aggregation point on purpose: a
    /// future fourth loss site is added to this sum once, and cannot be
    /// silently omitted from the reporting path.
    ///
    /// The sum is `saturating_add`-chained (the
    /// [`AgentStats`](crate::metrics::AgentStats) counter precedent), so it
    /// pins at `u64::MAX` rather than wrapping — an unreachable total in
    /// practice, but a wrapped one would read as a *healthy* run.
    ///
    /// # It is a sum of unlike terms
    ///
    /// The three sites do not share an attempt base — the actor is only
    /// attempted every `policy_frequency`-th critic update, the critics on
    /// every one — so this total is a throughput figure ("this many updates
    /// never reached an optimizer"), not a rate. Read the per-site accessors to
    /// attribute it.
    #[must_use]
    pub const fn skipped_updates(&self) -> u64 {
        self.critic_1_guard
            .skipped()
            .saturating_add(self.critic_2_guard.skipped())
            .saturating_add(self.actor_guard.skipped())
    }

    /// Samples an action for the current observation.
    ///
    /// Before `learning_starts` steps, draws a uniform random action on
    /// `[low, high]`. Afterwards runs the actor, adds Gaussian exploration
    /// noise, and clips to the action bounds. The unnoised policy mean is
    /// emitted when `training == false`.
    ///
    /// # Behavior on a non-finite observation
    ///
    /// The observation row is checked for finiteness before it reaches the
    /// actor. A `NaN` / `±Inf` row is **counted and warned about, and the
    /// action is returned unchanged** — nothing is substituted, no fallback is
    /// returned, and the clamping is not altered (ADR 0067 Decision 4). Read
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
    ///   no GPU, so CI only ever sees this half.
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
    /// this failure into one that produces a plausible in-domain torque with
    /// `is_valid()` back to `true`. (TD3's *target-smoothing* clip is a
    /// separate, tensor-space site on the learn path — this note is about the
    /// act path only.)
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
    /// Panics if the actor slot was poisoned by a panic inside an earlier
    /// actor optimizer step (see the type-level docs); the agent must then be
    /// rebuilt. Also panics if the actor's output tensor is not `f32`: that
    /// host-read is an `.expect` on a named invariant, the form
    /// `docs/rules.md` §4 sanctions here because `act` returns a bare action and
    /// so has no error channel to report the failure through — making the path
    /// fallible is a breaking change, deferred and tracked separately.
    pub fn act(&self, obs: &O, training: bool, rng: &mut (impl Rng + ?Sized)) -> A {
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

        let obs_t: Tensor<B, OR> = obs.to_tensor(&self.device);
        let batched: Tensor<B, BOR> = obs_t.unsqueeze::<BOR>();
        let raw: Tensor<B, BAR> = self.actor.get().forward(batched);
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
    /// Panics if the actor slot was poisoned by a panic inside an earlier
    /// actor optimizer step (see the type-level docs); the agent must then be
    /// rebuilt.
    pub fn inference_net(&self) -> Actor::InnerModule {
        self.actor.get().valid()
    }

    /// Deterministic action against a pre-snapshotted inner actor.
    ///
    /// Equivalent to `act(obs, false, _)` (the unnoised, bound-clipped policy
    /// mean) but runs on the non-autodiff backend via
    /// [`inference_net`](Self::inference_net), avoiding the per-call autodiff
    /// graph construction that [`act`](Self::act) incurs.
    ///
    /// # Behavior on a non-finite observation
    ///
    /// Identical to [`act`](Self::act), and sharing its counter: the row is
    /// checked, a non-finite row is counted and warned about, and the action is
    /// returned **unchanged**. See [`act`](Self::act) for why nothing is
    /// substituted, why the backends disagree, and why the clamp below must
    /// stay a host [`f32::clamp`].
    ///
    /// # Panics
    ///
    /// Panics if the actor's output tensor is not `f32`, or if it yields fewer
    /// than `A::COMPONENTS` values. Both indicate the supplied `net` does not
    /// match the action type this agent was built for. Making this path
    /// fallible so the first of those becomes an `Err` instead is deferred
    /// and tracked separately.
    pub fn act_with(&self, net: &Actor::InnerModule, obs: &O) -> A {
        // Function-local for the same `&self` / `Sync` reason as `act`.
        let mut scratch: Vec<f32> = Vec::new();
        self.act_obs_guard.report(obs.row_is_finite(&mut scratch));

        let obs_t: Tensor<B::InnerBackend, OR> = obs.to_tensor(&self.device);
        let batched: Tensor<B::InnerBackend, BOR> = obs_t.unsqueeze::<BOR>();
        let raw: Tensor<B::InnerBackend, BAR> = Actor::forward_inner(net, batched);
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
    ///
    /// # Behavior
    ///
    /// A non-finite `reward` (`NaN` or `±Inf`) is **discarded, not stored**:
    /// the transition never enters the replay buffer and the call is otherwise
    /// a no-op. Storing it would let every minibatch that later resampled it
    /// produce a non-finite loss, which `FiniteLossGuard` then skips — silently
    /// costing gradient updates for as long as the poisoned transition stayed
    /// resident (ADR 0065). A `tracing::warn!` fires on the 1st,
    /// 10th, 100th, … drop; use
    /// [`dropped_transitions`](Self::dropped_transitions) to detect the loss
    /// programmatically.
    ///
    /// A non-finite **observation** — on either `obs` or `next_obs` — is
    /// discarded the same way, with its own counter
    /// ([`dropped_observations`](Self::dropped_observations)) and its own
    /// decade-scheduled `warn!` (ADR 0067). The reward check runs
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
    /// Consequences).
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

    /// Number of actions [`act`](Self::act) / [`act_with`](Self::act_with)
    /// returned from a **non-finite observation**.
    ///
    /// This is not a drop count: per ADR 0067 §Decision 4 the action was
    /// selected, clamped, and returned to the caller unchanged. Every step it
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

    /// Test-only parameter checksums of the three **target** networks, as
    /// `[target_actor, target_critic_1, target_critic_2]`.
    ///
    /// The target-network observation seam of ADR 0058: nothing else in this
    /// crate can read a target's weights, which is how the two-schedule
    /// defect survived its tests. Paired with
    /// [`live_checksums`](Self::live_checksums) it makes a target update's
    /// *cadence* and *magnitude* both assertable — see
    /// [`param_checksum`](crate::algorithms::shared::param_checksum) for why a
    /// sum suffices under Polyak averaging.
    #[cfg(test)]
    pub(crate) fn target_checksums(&self) -> [f64; 3] {
        [
            param_checksum::<B::InnerBackend, _>(&self.target_actor),
            param_checksum::<B::InnerBackend, _>(&self.target_critic_1),
            param_checksum::<B::InnerBackend, _>(&self.target_critic_2),
        ]
    }

    /// Test-only parameter checksums of the three **live** networks, in the
    /// same order as [`target_checksums`](Self::target_checksums).
    #[cfg(test)]
    pub(crate) fn live_checksums(&self) -> [f64; 3] {
        [
            param_checksum::<B, _>(self.actor.get()),
            param_checksum::<B, _>(self.critic_1.get()),
            param_checksum::<B, _>(self.critic_2.get()),
        ]
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

    /// Runs one learning step.
    ///
    /// 1. Samples a uniformly random batch of size `batch_size` (with
    ///    replacement) from the replay buffer.
    /// 2. Builds the twin-critic TD target on the inner (non-autodiff) backend:
    ///    `y = r + γ(1−d)·min(target_q1(s', ã), target_q2(s', ã))`
    ///    where `ã = clip(target_actor(s') + clip(N(0, σ²), −c, c), low, high)`
    ///    (target-policy smoothing; see [`super::target_smoothing`]).
    /// 3. Runs an independent backward pass and Adam step for each critic.
    /// 4. Every `policy_frequency`-th critic step, updates the actor with
    ///    deterministic policy gradient against `critic_1`.
    /// 5. Every [`target_update.every()`](crate::target::TargetUpdate::every)-th
    ///    critic step, Polyak-averages all three target networks. When both
    ///    cadences fire on the same critic update — as they do at the shipped
    ///    defaults — step 4 runs first, so no target tracks a stale actor.
    ///
    /// Returns `None` if the agent is still in warm-up
    /// ([`can_learn`](Self::can_learn) returns `false`).
    ///
    /// Every fallible stage — forward, loss, `backward`, gradient reduction —
    /// runs on a borrow of the network, so a panic anywhere in this method
    /// other than inside an optimizer step leaves all three networks intact and
    /// the agent usable.
    ///
    /// # Panics
    ///
    /// Panics if an earlier optimizer step panicked and poisoned the slot of a
    /// network this call reads. Only the network whose step unwound is lost:
    /// the three steps run in disjoint windows, so a `critic_1` step panic does
    /// not affect `critic_2` or the actor. A poisoned agent cannot be recovered
    /// and must be rebuilt — see the type-level docs.
    ///
    /// # Errors
    ///
    /// Returns [`Td3AgentError::Polyak`] if a target soft-update finds a
    /// parameter-topology mismatch between a live network and its target twin
    /// (see [`polyak_update`](crate::utils::polyak_update)). Every in-tree
    /// target is cloned from its active network, so this cannot occur for
    /// agents built normally.
    // The body is one linear pipeline — sample, forward, loss, backward,
    // optimizer step, priority writeback, metrics — with a borrow structure
    // around the module slot that the inline comments below depend on. Splitting
    // it into helpers to satisfy the line count would thread that borrow through
    // signatures without making the sequence easier to follow.
    #[allow(clippy::too_many_lines)]
    pub fn learn_step(
        &mut self,
        rng: &mut (impl Rng + ?Sized),
    ) -> Result<Option<LearnOutcome>, Td3AgentError> {
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

        let mut batched_obs_shape: Vec<usize> = Vec::with_capacity(BOR);
        batched_obs_shape.push(batch_size);
        batched_obs_shape.extend_from_slice(&obs_shape);
        let mut batched_action_shape: Vec<usize> = Vec::with_capacity(BAR);
        batched_action_shape.push(batch_size);
        batched_action_shape.extend_from_slice(&action_shape);

        let obs_t: Tensor<B, BOR> = Tensor::from_data(
            TensorData::new(obs_flat, batched_obs_shape.clone()),
            &device,
        );
        let next_t_inner: Tensor<B::InnerBackend, BOR> =
            Tensor::from_data(TensorData::new(next_flat, batched_obs_shape), &device);
        let action_t: Tensor<B, BAR> =
            Tensor::from_data(TensorData::new(action_flat, batched_action_shape), &device);

        let rewards_inner: Tensor<B::InnerBackend, 1> =
            Tensor::from_data(TensorData::new(rewards, vec![batch_size]), &device);
        let terminated_inner: Tensor<B::InnerBackend, 1> =
            Tensor::from_data(TensorData::new(terminated, vec![batch_size]), &device);

        // --- Target computation (no autodiff) ---
        // Eq. 14's outer clip is against the `Box(low, high)` **vectors**, one
        // limit per component. Collapsing them to `low[0]`/`high[0]` is only
        // equivalent when every component shares a bound; under an asymmetric
        // space such as CarRacing's `Box([-1,0,0], [1,1,1])` it would leave
        // negative gas and brake in the target action — the "values of
        // impossible actions" the clip exists to suppress. The inner
        // `noise_clip` stays scalar: it bounds the smoothing noise *magnitude*
        // symmetrically and is scalar in the paper too.
        let raw_next_action: Tensor<B::InnerBackend, BAR> =
            Actor::forward_inner(&self.target_actor, next_t_inner.clone());
        let next_actions = smoothed_target_action::<B::InnerBackend, BAR>(
            raw_next_action,
            self.config.policy_noise,
            self.config.noise_clip,
            self.low_t.clone(),
            self.high_t.clone(),
            rng,
        );

        let next_q1: Tensor<B::InnerBackend, 1> = Critic::forward_inner(
            &self.target_critic_1,
            next_t_inner.clone(),
            next_actions.clone(),
        );
        let next_q2: Tensor<B::InnerBackend, 1> =
            Critic::forward_inner(&self.target_critic_2, next_t_inner, next_actions);
        let target_inner: Tensor<B::InnerBackend, 1> = compute_twin_critic_target(
            rewards_inner,
            next_q1,
            next_q2,
            terminated_inner,
            self.config.gamma,
        );
        let target: Tensor<B, 1> = Tensor::from_data(target_inner.into_data(), &device);

        // --- Critic updates: two independent backward passes ---
        // Both critics stay in their slots for the forward / loss / backward
        // region; each is out of its slot only inside its own `step_with`. The
        // two windows are therefore disjoint — a panic stepping `critic_1`
        // leaves `critic_2` intact.
        let q1_pred: Tensor<B, 1> = self.critic_1.get().forward(obs_t.clone(), action_t.clone());
        let q2_pred: Tensor<B, 1> = self.critic_2.get().forward(obs_t.clone(), action_t);

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

        // ADR 0056: the two critics run in DISJOINT backward+step windows
        // (independent graphs), so each site gets its own guard — a non-finite
        // loss in one critic skips only that critic's `backward()` + optimizer
        // step, while the other still updates. `loss_1`/`loss_2` are already
        // host-resident, so the check costs no extra sync. A skipped critic's
        // value is excluded from the reported metric: `last_qf{1,2}_loss`
        // carries its last *applied* value forward rather than folding in a NaN.
        // `critic_updates` (the actor cadence counter) still advances
        // unconditionally, per ADR 0056 §3.
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

        // --- Actor update (every policy_frequency-th critic step) ---
        let mut actor_loss_opt: Option<f32> = None;
        if self
            .critic_updates
            .is_multiple_of(self.config.policy_frequency)
        {
            // Same shape as the critic block: the actor stays in its slot for
            // the whole fallible region, and `step_with` is its own isolated
            // window — independent of both critics'.
            let predicted_actions: Tensor<B, BAR> = self.actor.get().forward(obs_t.clone());
            let q_actor: Tensor<B, 1> = self.critic_1.get().forward(obs_t, predicted_actions);
            let actor_loss_tensor = q_actor.mean().neg();
            let actor_loss_value = actor_loss_tensor.clone().into_scalar().elem::<f32>();

            // ADR 0056: guard the actor site. A non-finite actor loss
            // skips the actor `backward()` + optimizer step and leaves
            // `actor_loss` reported as `None` this iteration (mirroring the
            // delayed-update skip) rather than folding a NaN into
            // `last_actor_loss`. `actor_loss_value` is already host-resident.
            if self.actor_guard.check(actor_loss_value) {
                let grads = actor_loss_tensor.backward();
                let actor_grads = GradientsParams::from_grads(grads, self.actor.get());
                self.actor
                    .step_with(&mut self.actor_opt, self.config.actor_lr, actor_grads);

                self.last_actor_loss = actor_loss_value;
                actor_loss_opt = Some(actor_loss_value);
            }
        }

        // --- Target Polyak updates (every target_update.every()-th critic
        // step) ---
        // This gate is deliberately *adjacent to*, not nested in, the actor
        // block above: before ADR 0058 the three Polyak calls lived inside it,
        // which made `policy_frequency` an undeclared alias for the target
        // cadence — an operator could not change Fujimoto's actor delay `d`
        // without also changing how often the targets moved. Both gates read
        // the same post-increment `critic_updates` counter, so at the shipped
        // defaults (`policy_frequency == target_update.every() == 2`) the
        // predicate is identical and the trajectory is unchanged.
        //
        // Order is load-bearing and matches the pre-ADR-0058 code exactly: the
        // actor step must precede this block, or the target actor would track a
        // stale actor for one cadence period.
        //
        // Target Polyak updates: cadence-driven (per ADR 0056 §3), so they
        // run whether or not the actor step was skipped — the healthy
        // networks keep being tracked by their targets. That invariant survives
        // the move unchanged: this block never consults the actor guard, and a
        // skipped actor step still leaves `self.actor` a valid network to blend
        // from. Clone rather than move out: `soft_update` consumes `target` by
        // value, so on `Err` the `?` returns before the reassignment and each
        // target field keeps its prior weights — no silent hard-sync onto its
        // live network (the invariant now holds via early return, and equally
        // for a panic).
        if let Some(tau) = self.config.target_update.fires_at(self.critic_updates) {
            self.target_actor =
                Actor::soft_update(self.actor.get(), self.target_actor.clone(), tau)?;
            self.target_critic_1 =
                Critic::soft_update(self.critic_1.get(), self.target_critic_1.clone(), tau)?;
            self.target_critic_2 =
                Critic::soft_update(self.critic_2.get(), self.target_critic_2.clone(), tau)?;
        }

        Ok(Some(LearnOutcome {
            // Report the most recent *applied* critic losses so a skipped
            // (non-finite) step carries its last healthy value forward rather
            // than poisoning the metric with a NaN (ADR 0056 §3).
            critic_loss: self.last_qf1_loss + self.last_qf2_loss,
            actor_loss: actor_loss_opt,
            q_mean,
        }))
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
    use crate::MAX_BUFFER_CAPACITY;
    use rlevo_core::config::ConstraintKind;

    use burn::backend::Flex;

    type TestBackend = Flex;

    #[test]
    fn test_td3agent_metrics_performance_record_returns_reward_and_steps() {
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
    fn test_td3agent_error_display_uses_thiserror_messages() {
        let err = Td3AgentError::InvalidAction("bad slice".into());
        assert_eq!(err.to_string(), "Invalid action: bad slice");
    }

    #[test]
    fn test_td3agent_td3_target_is_min_of_twin_critics() {
        // `next_q1 = [2.0, 1.0, 5.0]` and `next_q2 = [3.0, 0.5, 4.0]` →
        // element-wise min is `[2.0, 0.5, 4.0]`. With γ = 0.9, non-terminal
        // rewards `[0.1, 0.2, 0.3]` and terminated `[0, 0, 1]`, the target is
        // `[0.1 + 0.9*2.0, 0.2 + 0.9*0.5, 0.3 + 0.9*4.0*0]`
        // `= [1.9, 0.65, 0.3]`.
        let device = Default::default();
        let rewards = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(vec![0.1_f32, 0.2, 0.3], vec![3]),
            &device,
        );
        let next_q1 = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(vec![2.0_f32, 1.0, 5.0], vec![3]),
            &device,
        );
        let next_q2 = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(vec![3.0_f32, 0.5, 4.0], vec![3]),
            &device,
        );
        let terminated = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(vec![0.0_f32, 0.0, 1.0], vec![3]),
            &device,
        );

        let target = compute_twin_critic_target(rewards, next_q1, next_q2, terminated, 0.9);
        let data = target.into_data().convert::<f32>();
        let slice = data.as_slice::<f32>().unwrap();
        assert!((slice[0] - 1.9).abs() < 1e-6, "row 0: {}", slice[0]);
        assert!((slice[1] - 0.65).abs() < 1e-6, "row 1: {}", slice[1]);
        assert!((slice[2] - 0.3).abs() < 1e-6, "row 2: {}", slice[2]);
    }

    // -------- non-finite-loss guard (ADR 0056) --------

    use crate::algorithms::bootstrap_mask::{
        MaskContinuousAction, MaskObservation, TinyActor, TinyCritic,
    };
    use crate::algorithms::td3::td3_config::Td3TrainingConfigBuilder;
    use burn::backend::Autodiff;
    use burn::module::{Module, ModuleMapper, Param};
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rlevo_core::action::ContinuousAction;

    type TestAdBackend = Autodiff<Flex>;
    type GuardAgent = Td3Agent<
        TestAdBackend,
        TinyActor<TestAdBackend>,
        TinyCritic<TestAdBackend>,
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
        let t =
            Tensor::<TestAdBackend, 1>::from_data(TensorData::new(vec![a, b], vec![2]), &device);
        <MaskObservation as TensorConvertible<1, TestAdBackend>>::from_tensor(t)
            .expect("obs from tensor")
    }

    /// Evaluates critic-2 on a fixed (obs, action) pair and reads the scalar
    /// back — a change across a learn step proves the weights were updated.
    fn critic_2_probe(agent: &GuardAgent) -> f32 {
        let device = Default::default();
        let obs = Tensor::<TestAdBackend, 2>::from_data(
            TensorData::new(vec![0.3_f32, 0.7], vec![1, 2]),
            &device,
        );
        let act = Tensor::<TestAdBackend, 2>::from_data(
            TensorData::new(vec![0.0_f32], vec![1, 1]),
            &device,
        );
        agent
            .critic_2
            .get()
            .forward(obs, act)
            .inner()
            .into_scalar()
            .elem::<f32>()
    }

    /// One diverged critic must not poison the other (ADR 0056). TD3 makes
    /// the same twin-critic independence claim as SAC: the two critics run their
    /// backward + optimizer step in disjoint windows on independent graphs, so a
    /// non-finite loss in critic-1 must skip only critic-1's update while
    /// critic-2 still learns and the agent stays finite.
    #[test]
    fn test_td3agent_one_nonfinite_critic_skips_only_that_critic() {
        let device = Default::default();
        let config = Td3TrainingConfigBuilder::new()
            .batch_size(2)
            .learning_starts(0)
            .replay_buffer_capacity(64)
            .critic_lr(0.05)
            // policy_frequency = 2 keeps the actor + Polyak updates off this
            // single critic step, isolating the twin-critic disjoint-window
            // property under test.
            .policy_frequency(2)
            .build()
            .expect("valid config");

        let mut agent = GuardAgent::new(
            TinyActor::<TestAdBackend>::new(&device),
            TinyCritic::<TestAdBackend>::new(&device),
            TinyCritic::<TestAdBackend>::new(&device),
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
        // No panic: a panic here (e.g. a pruned shared-leaf backward) fails the
        // test outright.
        let outcome = agent
            .learn_step(&mut rng)
            .expect("no polyak error")
            .expect("a primed agent past warm-up learns");
        let after = critic_2_probe(&agent);

        // Exactly one site skipped, and it was critic-1's. These are exact
        // values, not `warning_fired()` booleans: a latch reads the same
        // whether one update or ten thousand were thrown away (ADR 0072 §1),
        // and `!fired` was satisfiable by an already-fired latch.
        assert_eq!(
            agent.skipped_critic_1_updates(),
            1,
            "critic-1's non-finite loss must skip exactly this one update"
        );
        assert_eq!(
            agent.skipped_critic_2_updates(),
            0,
            "critic-2's loss was finite: the sibling critic must be untouched, \
             at exactly zero skips"
        );
        assert_eq!(
            agent.skipped_actor_updates(),
            0,
            "the actor did not update this step (policy_frequency = 2), so it \
             cannot have skipped: exactly zero"
        );
        assert_eq!(
            agent.skipped_updates(),
            1,
            "the aggregate is the sum of the three sites: 1 + 0 + 0"
        );

        // critic-2 still updated.
        assert!(
            (before - after).abs() > 1e-6,
            "critic-2 must still learn while critic-1 is skipped: {before} -> {after}"
        );

        // The skip kept the poison contained — no global NaN.
        assert!(after.is_finite(), "critic-2 output must stay finite");
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

    // -------- skip COUNTERS (ADR 0072) --------

    /// A primed agent whose live critic-1 has diverged to `NaN` weights, so
    /// every subsequent learn step produces a non-finite critic-1 loss.
    ///
    /// Same fixture as [`td3_one_nonfinite_critic_skips_only_that_critic`],
    /// parameterised on the two cadences so a test can decide whether the actor
    /// gate and the Polyak gate fire during its run. Critic-1 is poisoned via a
    /// *clone* of the live network, preserving `ParamId`s so the target Polyak
    /// pairing stays valid.
    ///
    /// The poison is self-sustaining on purpose: critic-1's optimizer step is
    /// skipped every time, so its weights are never overwritten and stay `NaN`
    /// for the whole run — which is what makes a repeated-skip count testable
    /// at all.
    fn poisoned_critic_1_agent(policy_frequency: usize, target_update: TargetUpdate) -> GuardAgent {
        let device = Default::default();
        let config = Td3TrainingConfigBuilder::new()
            .batch_size(2)
            .learning_starts(0)
            .replay_buffer_capacity(64)
            .critic_lr(0.05)
            .policy_frequency(policy_frequency)
            .target_update(target_update)
            .build()
            .expect("valid config");

        let mut agent = GuardAgent::new(
            TinyActor::<TestAdBackend>::new(&device),
            TinyCritic::<TestAdBackend>::new(&device),
            TinyCritic::<TestAdBackend>::new(&device),
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
        assert_eq!(
            agent.skipped_updates(),
            0,
            "precondition: a freshly built agent must have skipped nothing, or every \
             count below would be reading someone else's skips"
        );
        agent
    }

    /// The counter counts **every** skip, not merely the first.
    ///
    /// Three consecutive poisoned learn steps must read exactly `3`. Three and
    /// not one: a `1` is what a bool-to-counter mistranslation produces — a
    /// latch that sets on the first occurrence and never moves again is
    /// indistinguishable from a correct counter at n = 1, and that latch is
    /// precisely what ADR 0072 replaced.
    #[test]
    fn test_td3agent_counts_repeated_loss_skips() {
        // Both cadences are pushed out to 8 so neither fires within the three
        // steps: the actor never attempts (so it cannot skip) and no Polyak
        // update can carry critic-1's NaN into `target_critic_1`, which would
        // poison the shared Bellman target and make critic-2 skip too. The
        // single varying site is critic-1's.
        let mut agent = poisoned_critic_1_agent(8, TargetUpdate::polyak(0.005, 8));
        let mut rng = StdRng::seed_from_u64(0);

        for _ in 0..3 {
            agent
                .learn_step(&mut rng)
                .expect("no polyak error")
                .expect("a primed agent past warm-up learns");
        }

        assert_eq!(
            agent.critic_updates(),
            3,
            "critic_updates counts ATTEMPTS and advances on a skip too (ADR 0059 \
             §Decision 4): three learn steps must read 3, not 0"
        );
        assert_eq!(
            agent.skipped_critic_1_updates(),
            3,
            "each of the three poisoned learn steps must increment the critic-1 skip \
             count: exactly 3, not a latched 1"
        );
        assert_eq!(
            agent.skipped_critic_2_updates(),
            0,
            "critic-2's loss stayed finite across all three steps: exactly 0"
        );
        assert_eq!(
            agent.skipped_actor_updates(),
            0,
            "policy_frequency = 8 means the actor was never attempted in three \
             steps, so it cannot have skipped: exactly 0"
        );
        assert_eq!(
            agent.skipped_updates(),
            3,
            "the aggregate over the three sites: 3 + 0 + 0"
        );
    }

    /// The aggregate is the sum of three **different, nonzero** numbers.
    ///
    /// The parts are critic-1 = 5, critic-2 = 2, actor = 1 — aggregate **8**.
    /// Both properties of that triple are load-bearing, against two different
    /// defects:
    ///
    /// - **Pairwise unequal** catches *duplication*: an aggregate that adds one
    ///   guard twice and another not at all still reads correctly when the terms
    ///   are equal. Here every duplicated-guard sum misses — `5+5+2 = 12`,
    ///   `5+5+1 = 11`, `2+2+5 = 9`, `2+2+1 = 5`, `1+1+5 = 7`, `1+1+2 = 4`, and
    ///   the all-one-guard sums `15 / 6 / 3`.
    /// - **All nonzero** catches *omission*: a zero term contributes nothing, so
    ///   dropping its summand leaves the total unchanged. The earlier shape of
    ///   this test (2 / **0** / 1 = 3) passed against a `skipped_updates` with
    ///   `skipped_critic_2_updates()` deleted from the chain. With no zero, the
    ///   three single-term omissions give `5+2 = 7`, `5+1 = 6`, `2+1 = 3`; `8`
    ///   is reachable only by summing each guard exactly once.
    ///
    /// How the cadence produces exactly that triple, over five learn steps with
    /// `policy_frequency = 4` and the Polyak cadence pushed out to 100:
    ///
    /// - critic-1 is `NaN`-poisoned from the start and never re-stepped (its
    ///   optimizer step is skipped every time, so the poison is self-sustaining):
    ///   it skips on **all five** steps.
    /// - critic-2 and the shared Bellman target stay finite for steps 1-3 —
    ///   critic-1's `NaN` is never blended into `target_critic_1`, so the
    ///   sibling stays clean. Critic-2 is then diverged deliberately *between*
    ///   steps 3 and 4 and, like critic-1, never re-stepped: **two** skips.
    ///   Diverging it mid-run rather than up front is what keeps the two critic
    ///   terms unequal; poisoning both at step 0 would give `5 / 5 / 1` and
    ///   re-open the duplication hole between them.
    /// - the actor sits behind `policy_frequency = 4`, so it is attempted only
    ///   on critic update 4, where its loss is `−mean(critic_1(s, π(s)))` and
    ///   therefore `NaN`: one attempt, **one** skip.
    ///
    /// This does not weaken the sibling-isolation property pinned by
    /// [`td3_one_nonfinite_critic_skips_only_that_critic`] — the intermediate
    /// assertion below re-pins `critic-2 == 0` right up to the moment critic-2 is
    /// diverged on purpose.
    #[test]
    fn test_td3agent_aggregate_skip_count_sums_unequal_sites() {
        let mut agent = poisoned_critic_1_agent(4, TargetUpdate::polyak(0.005, 100));
        let mut rng = StdRng::seed_from_u64(0);

        for _ in 0..3 {
            agent
                .learn_step(&mut rng)
                .expect("no polyak error")
                .expect("a primed agent past warm-up learns");
        }

        assert_eq!(
            agent.skipped_critic_2_updates(),
            0,
            "sibling isolation: three steps of a NaN critic-1 must not have made \
             critic-2 skip — the divergence below is the only reason it starts"
        );

        // Diverge critic-2 too, via a *clone* so `ParamId`s (and hence the
        // target Polyak pairing) stay valid — the same treatment critic-1 got
        // inside `poisoned_critic_1_agent`.
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
            "critic-1 is permanently poisoned: it must skip on all five learn steps"
        );
        assert_eq!(
            agent.skipped_critic_2_updates(),
            2,
            "critic-2 was diverged after step 3 and never re-stepped: exactly 2 \
             skips. NONZERO on purpose — a zero here would be invisible to the \
             aggregate, letting a chain that omits this summand still total 8"
        );
        assert_eq!(
            agent.skipped_actor_updates(),
            1,
            "the actor is attempted only on critic update 4 (policy_frequency = 4) \
             and its loss reads through the poisoned critic-1: exactly 1 skip from \
             exactly 1 attempt"
        );
        assert_eq!(
            agent.skipped_updates(),
            8,
            "the aggregate must be 5 + 2 + 1 = 8 over three DISTINCT per-site \
             counts — no sum that double-counts one guard, and no sum that omits \
             one, can land on 8 here"
        );
        assert_eq!(
            agent.critic_updates(),
            5,
            "attempts advance unconditionally: five learn steps read 5 even though \
             critic-1 applied none (applied = attempts − skipped = 0)"
        );
    }

    // -------- target-update cadence (ADR 0058 / 0059) --------

    use crate::target::TargetUpdate;
    use approx::assert_abs_diff_eq;

    /// Slack for the Polyak identity below. Each checksum is an `f32` device
    /// reduction over a handful of parameters and each blended parameter costs
    /// two `f32` roundings, so ~2e-6 is the realistic worst case; the smallest
    /// signal any assertion here reads is `τ · gap ≈ 7.5e-3`, three orders of
    /// magnitude larger.
    const CHECKSUM_EPS: f64 = 1e-5;

    /// Adds a constant to every float parameter of a module.
    ///
    /// All three targets are built by cloning their live network, so they start
    /// *identical* — and a Polyak blend between identical networks is a no-op
    /// that every "did the target move, and by how much?" assertion would pass
    /// vacuously. That exact vacuity is how the two-schedule defect survived
    /// its tests; this mapper removes it by opening a known gap. It maps a *clone*,
    /// so `ParamId`s are preserved and the Polyak pairing stays valid.
    struct AddConstant(f32);

    impl<B: Backend> ModuleMapper<B> for AddConstant {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
            let (id, tensor, mapper) = param.consume();
            Param::from_mapped_value(id, tensor.add_scalar(self.0), mapper)
        }
    }

    /// A primed agent whose live networks are held a known distance from their
    /// targets, so every cadence assertion below is non-vacuous.
    fn cadence_agent(policy_frequency: usize, target_update: TargetUpdate) -> GuardAgent {
        let device = Default::default();
        let config = Td3TrainingConfigBuilder::new()
            .batch_size(2)
            .learning_starts(0)
            .replay_buffer_capacity(64)
            .actor_lr(1e-3)
            .critic_lr(1e-3)
            .policy_frequency(policy_frequency)
            .target_update(target_update)
            .build()
            .expect("valid config");

        let mut agent = GuardAgent::new(
            TinyActor::<TestAdBackend>::new(&device),
            TinyCritic::<TestAdBackend>::new(&device),
            TinyCritic::<TestAdBackend>::new(&device),
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

        // Nudge the *targets*, not the live networks. Perturbing a live network
        // would disturb Burn's autodiff registration across consecutive learn
        // steps, and pushing the actor's weights up saturates its `tanh`,
        // zeroing the very actor gradient one of these tests asserts on. The
        // targets are plain inner-backend modules with no graph attached, and
        // `AddConstant` preserves their `ParamId`s, so the Polyak pairing holds.
        agent.target_actor = agent.target_actor.clone().map(&mut AddConstant(0.5));
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
                "precondition: network {i}'s live and target checksums must differ, or a \
                 Polyak no-op would satisfy every assertion below; got live {live} vs \
                 target {target}"
            );
        }
        agent
    }

    /// Asserts that a fired Polyak update landed exactly where the rule says:
    /// `target ← (1 − τ)·target + τ·active`, and that it moved at all.
    fn assert_fired(before: [f64; 3], after: [f64; 3], live: [f64; 3], tau: f64) {
        for (i, ((&b, &a), &l)) in before.iter().zip(after.iter()).zip(live.iter()).enumerate() {
            assert_abs_diff_eq!(a, (1.0 - tau) * b + tau * l, epsilon = CHECKSUM_EPS);
            assert!(
                (a - b).abs() > 1e-4,
                "network {i}'s target must actually have moved: {b} -> {a}"
            );
        }
    }

    /// TD3's shipped cadence is `polyak(0.005, 2)`: all three targets move on
    /// critic updates 2, 4, … and are **bit-identically unchanged** on 1, 3, ….
    ///
    /// This is the test that proves lifting the Polyak calls out of the
    /// `policy_frequency` block did not shift the cadence — `every = 2` is the
    /// same number the block used to be gated on, i.e. Fujimoto §5.2's `d`.
    #[test]
    fn test_td3agent_default_cadence_fires_every_second_critic_update() {
        let rule = TargetUpdate::polyak(0.005, 2);
        let tau = rule.tau();
        let mut agent = cadence_agent(2, rule);
        let mut rng = StdRng::seed_from_u64(0);

        for critic_update in 1..=4_usize {
            let before = agent.target_checksums();
            agent
                .learn_step(&mut rng)
                .expect("no polyak error")
                .expect("a primed agent past warm-up learns");
            let after = agent.target_checksums();

            if critic_update.is_multiple_of(2) {
                assert_fired(before, after, agent.live_checksums(), tau);
            } else {
                assert_eq!(
                    after, before,
                    "critic update {critic_update} is not a multiple of the cadence 2, so \
                     every target weight must be untouched"
                );
            }
        }
    }

    /// The configuration that was **unexpressible** before ADR 0058: the actor
    /// updates on every critic step while the targets still move only every
    /// second one. While the Polyak calls lived inside the `policy_frequency`
    /// block, `policy_frequency = 1` forced the targets to move every step too.
    #[test]
    fn test_td3agent_actor_cadence_and_target_cadence_are_independent() {
        let rule = TargetUpdate::polyak(0.005, 2);
        let tau = rule.tau();
        let mut agent = cadence_agent(1, rule);
        let mut rng = StdRng::seed_from_u64(0);

        // Critic update 1: the actor fires (policy_frequency = 1); the targets
        // must not (every = 2).
        let target_before = agent.target_checksums();
        let actor_before = agent.live_checksums()[0];
        let outcome = agent
            .learn_step(&mut rng)
            .expect("no polyak error")
            .expect("a primed agent past warm-up learns");
        assert!(
            outcome.actor_loss.is_some(),
            "policy_frequency = 1 must run the actor on every critic update"
        );
        assert!(
            (agent.live_checksums()[0] - actor_before).abs() > 1e-6,
            "the actor step must have moved the live actor's weights, or 'the actor \
             updated while the target did not' is vacuous"
        );
        assert_eq!(
            agent.target_checksums(),
            target_before,
            "the target cadence is 2: an actor update on critic update 1 must not drag \
             the targets with it"
        );

        // Critic update 2: now the target cadence fires.
        let before = agent.target_checksums();
        agent
            .learn_step(&mut rng)
            .expect("no polyak error")
            .expect("a primed agent past warm-up learns");
        assert_fired(
            before,
            agent.target_checksums(),
            agent.live_checksums(),
            tau,
        );
    }

    // ---- ADR 0065: non-finite reward is dropped at ingestion ----
    //
    // Every off-policy agent needs its OWN copy of this test. The defect had
    // six sites, not the four originally named, precisely because C51 and
    // QR-DQN were added by copying an unguarded `remember` and no shared test
    // noticed. A per-file test is what makes agent #7's author notice.

    #[test]
    fn test_td3agent_remember_drops_a_nonfinite_reward() {
        let device = Default::default();
        let config = Td3TrainingConfigBuilder::new()
            .build()
            .expect("valid config");
        let mut agent = GuardAgent::new(
            TinyActor::<TestAdBackend>::new(&device),
            TinyCritic::<TestAdBackend>::new(&device),
            TinyCritic::<TestAdBackend>::new(&device),
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

    // ---- ADR 0067: non-finite observation ----
    //
    // Same per-file rule as the reward test above: each agent carries its own
    // copy rather than trusting one shared test to stand in for three call
    // sites. `MaskObservation` is the deliberately NaN-emitting observation
    // type here — its `write_host_row` copies its `[f32; 2]` payload verbatim,
    // so `make_obs(f32::NAN, _)` puts a real NaN in the staged row.

    /// A fresh agent with capacity to spare, for the observation-guard tests.
    fn obs_guard_agent() -> GuardAgent {
        let device = Default::default();
        let config = Td3TrainingConfigBuilder::new()
            .replay_buffer_capacity(64)
            .build()
            .expect("valid config");
        GuardAgent::new(
            TinyActor::<TestAdBackend>::new(&device),
            TinyCritic::<TestAdBackend>::new(&device),
            TinyCritic::<TestAdBackend>::new(&device),
            config,
            device,
        )
        .expect("valid agent")
    }

    #[test]
    fn test_td3agent_remember_drops_a_nonfinite_obs() {
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
    fn test_td3agent_remember_drops_a_nonfinite_next_obs() {
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
    fn test_td3agent_remember_both_bad_counts_only_the_reward_drop() {
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
    fn test_td3agent_act_counts_a_nonfinite_obs_and_still_returns_an_action() {
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

    /// `act_with` is a **second, independent** entry point — it runs against a
    /// snapshotted inner actor and shares `act`'s counter, but not its code
    /// path. ADR 0065's copy-paste finding (two of six `remember` sites went
    /// missing behind coverage that never reached them) applies verbatim here:
    /// without this test, deleting the guard line from `act_with` is invisible.
    ///
    /// Same decision under test as [`act`]: count, warn, and **return the
    /// action anyway** (ADR 0067 §Decision 4).
    #[test]
    fn test_td3agent_act_with_counts_a_nonfinite_obs_and_still_returns_an_action() {
        let agent = obs_guard_agent();
        let net = agent.inference_net();

        let action = agent.act_with(&net, &make_obs(f32::NAN, 1.0));
        assert_eq!(
            agent.degenerate_action_selections(),
            1,
            "`act_with` is its own site and must count a NaN observation"
        );
        assert_eq!(
            action.as_slice().len(),
            1,
            "the action must still be returned: ADR 0067 §Decision 4 substitutes nothing"
        );

        let action = agent.act_with(&net, &make_obs(0.1, f32::NEG_INFINITY));
        assert_eq!(
            agent.degenerate_action_selections(),
            2,
            "the count must not latch: a second ±Inf row increments again"
        );
        assert_eq!(
            action.as_slice().len(),
            1,
            "an action still comes back on the second degenerate row"
        );

        // A healthy row through the same entry point must leave the counter put.
        let _ = agent.act_with(&net, &make_obs(0.2, 0.8));
        assert_eq!(
            agent.degenerate_action_selections(),
            2,
            "a finite observation must not increment the `act_with` counter"
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
    fn test_td3agent_new_rejects_out_of_range_replay_buffer_capacity() {
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
            let config = Td3TrainingConfig {
                replay_buffer_capacity: capacity,
                ..Td3TrainingConfig::default()
            };
            let Err(err) = GuardAgent::new(
                TinyActor::<TestAdBackend>::new(&device),
                TinyCritic::<TestAdBackend>::new(&device),
                TinyCritic::<TestAdBackend>::new(&device),
                config,
                device,
            ) else {
                panic!("capacity {capacity} must be rejected, not allocated");
            };
            assert_eq!(err.config, "Td3TrainingConfig");
            assert_eq!(err.field, "replay_buffer_capacity");
            assert_eq!(err.kind, kind);
        }
    }
}
