//! Deep Deterministic Policy Gradient agent: actor, critic, and replay buffer
//! management.
//!
//! [`DdpgAgent`] owns a deterministic actor, a Q-critic, their Polyak-averaged
//! target twins, two independent Adam optimizers, a uniform FIFO replay
//! buffer, and a `GaussianNoise` exploration module. It implements
//! [`Agent`], through which a driver steps it. [`DdpgAgent::remember`] and
//! [`DdpgAgent::learn_step`] are the externally paced path: push transitions
//! and run gradient updates on a cadence of the caller's own.

use rlevo_core::agent::{Agent, Transition};
use std::marker::PhantomData;

use burn::optim::{GradientsParams, ModuleOptimizer};
use burn::tensor::{Device, Tensor, TensorData};
use rand::Rng;
use rand::RngExt;

use crate::replay::{ContinuousTransition, ReplayStrategy, UniformReplay, UniformReplayConfig};
use rlevo_core::action::BoundedAction;
use rlevo_core::base::{Observation, TensorConvertible};
use rlevo_core::config::Validate;
use rlevo_core::environment::EpisodeStatus;

use crate::algorithms::ddpg::ddpg_config::DdpgTrainingConfig;
use crate::algorithms::ddpg::ddpg_model::{ContinuousQ, DeterministicPolicy};
use crate::algorithms::ddpg::exploration::GaussianNoise;
#[cfg(test)]
use crate::algorithms::shared::param_checksum;
use crate::algorithms::shared::{
    FiniteLossGuard, FiniteObsGuard, FiniteRewardGuard, Slot, UNIFORM_REPLAY_BETA,
    action_bound_tensors, assert_bounds_match_components, clip_to_action_bounds,
};
use crate::utils::{PolyakError, compute_target_q_values, polyak_update};

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
/// buffer, and a `GaussianNoise` exploration module.
///
/// The typical call sequence per environment step is:
///
/// 1. [`act`](Self::act) — choose an action (random during warm-up, noisy
///    policy thereafter).
/// 2. [`remember`](Self::remember) — store the resulting transition and
///    count the step.
/// 3. [`learn_step`](Self::learn_step) — run one gradient update (no-op
///    during warm-up).
///
/// A driver runs that sequence through the [`Agent`] impl.
///
/// # Network ownership
///
/// The actor and critic live in a `Slot`, which holds each network across the
/// Burn optimizer step that consumes it by value. Every fallible operation —
/// the forward pass, the loss, `backward`, and `GradientsParams::from_grads` —
/// runs on a borrow, so a panic there leaves the agent intact and usable. The
/// networks are out of their slots only for the duration of the
/// [`ModuleOptimizer::step`](burn::optim::ModuleOptimizer::step) call itself; a panic in
/// that one window is terminal for this agent and requires rebuilding it. See
/// `Slot` for why that residual window is irreducible.
///
/// # Const generics
///
/// - `OR` — rank of a single observation tensor (`1` for vector observations
///   of shape `[features]`).
/// - `BOR` — rank of a batched observation tensor (= `OR + 1`). Rust cannot
///   express `OR + 1` in generic position on stable, so both are supplied.
/// - `AR` — rank of a single action tensor (`1` for vector actions of shape
///   `[action_dim]`).
/// - `BAR` — rank of a batched action tensor (= `AR + 1`).
pub struct DdpgAgent<
    Actor,
    Critic,
    O,
    A,
    const OR: usize,
    const BOR: usize,
    const AR: usize,
    const BAR: usize,
> where
    Actor: DeterministicPolicy<BOR, BAR>,
    Critic: ContinuousQ<BOR, BAR>,
    O: Observation<OR> + TensorConvertible<OR>,
    A: BoundedAction<AR>,
{
    actor: Slot<Actor>,
    target_actor: Actor,
    critic: Slot<Critic>,
    target_critic: Critic,
    actor_opt: ModuleOptimizer,
    critic_opt: ModuleOptimizer,
    buffer: UniformReplay<ContinuousTransition<O>>,
    exploration: GaussianNoise,
    low: &'static [f32],
    high: &'static [f32],
    /// `[1, ..action_shape]` per-component bounds for the target-action clip,
    /// built once at construction on the plain device — see
    /// [`action_bound_tensors`].
    low_t: Tensor<BAR>,
    high_t: Tensor<BAR>,
    config: DdpgTrainingConfig,
    /// The training device: the caller's device with autodiff on.
    device: Device,
    /// `device` without autodiff, where the target path runs.
    plain_device: Device,
    step: usize,
    critic_updates: usize,
    last_actor_loss: f32,
    /// Most recent *applied* critic loss — carried forward across a non-finite
    /// skip so the reported metric never folds in a NaN (ADR 0056, Decision 3).
    last_critic_loss: f32,
    /// Non-finite-loss guard for the critic loss site (ADR 0056). Skips
    /// the update on every occurrence; the `warn!` escalates by decades — skips
    /// 1, 10, 100, … — each carrying the running total (ADR 0072, Decision 1),
    /// readable via [`Self::skipped_critic_updates`].
    critic_guard: FiniteLossGuard,
    /// Non-finite-loss guard for the actor loss site, counting and escalating
    /// independently of [`Self::critic_guard`]. See
    /// [`Self::skipped_actor_updates`].
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
    /// 0067, Decision 4).
    act_obs_guard: FiniteObsGuard,
    /// Reusable host staging buffer for the ingestion-side row-finiteness
    /// check. `remember` takes `&mut self`, so it can own one buffer and
    /// amortize the allocation across calls; the `&self` action-selection path
    /// cannot, and pays a per-call `Vec` instead — see [`act`](Self::act).
    obs_scratch: Vec<f32>,
    _action: PhantomData<A>,
    /// Whether the [`Agent`] impl explores and learns (`true`, the state a
    /// new agent starts in) or acts greedily and learns nothing; see
    /// [`Agent::set_training`].
    training: bool,
}

impl<Actor, Critic, O, A, const OR: usize, const BOR: usize, const AR: usize, const BAR: usize>
    std::fmt::Debug for DdpgAgent<Actor, Critic, O, A, OR, BOR, AR, BAR>
where
    Actor: DeterministicPolicy<BOR, BAR>,
    Critic: ContinuousQ<BOR, BAR>,
    O: Observation<OR> + TensorConvertible<OR>,
    A: BoundedAction<AR>,
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

impl<Actor, Critic, O, A, const OR: usize, const BOR: usize, const AR: usize, const BAR: usize>
    DdpgAgent<Actor, Critic, O, A, OR, BOR, AR, BAR>
where
    Actor: DeterministicPolicy<BOR, BAR>,
    Critic: ContinuousQ<BOR, BAR>,
    O: Observation<OR> + TensorConvertible<OR>,
    A: BoundedAction<AR>,
{
    /// Constructs a new agent from pre-built actor and critic networks.
    ///
    /// The agent trains on `device` with autodiff on, and switches both
    /// networks to training ([`Module::train`](burn::module::Module::train)), so
    /// networks built on a plain device train as well. Target networks are
    /// [`valid`](burn::module::Module::valid) snapshots of the supplied modules. Optimizers are built from
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
        device: Device,
    ) -> Result<Self, rlevo_core::config::ConfigError> {
        config.validate()?;
        assert_bounds_match_components::<AR, A>();
        let device = device.autodiff();
        let plain_device = device.clone().without_autodiff();
        let actor = actor.train();
        let critic = critic.train();
        let target_actor = actor.valid();
        let target_critic = critic.valid();
        let adam = config.optimizer.clone();
        let (actor_opt, critic_opt) = match &config.clip_grad {
            Some(clip) => (
                adam.clone().with_grad_clipping(Some(clip.clone())).init(),
                adam.clone().with_grad_clipping(Some(clip.clone())).init(),
            ),
            None => (adam.clone().init(), adam.clone().init()),
        };
        let exploration = GaussianNoise::new(config.exploration_noise);
        let (low_t, high_t) = action_bound_tensors::<A, AR, BAR>(&plain_device);
        // The capacity is a runtime config field, not a literal, so it takes
        // the fallible path: an out-of-range value is a `ConfigError` naming
        // `capacity`, never an allocation abort inside `VecDeque`.
        let buffer = UniformReplay::from_config(UniformReplayConfig {
            capacity: config.replay_buffer_capacity,
        })?;
        Ok(Self {
            training: true,
            actor: Slot::new(actor),
            target_actor,
            critic: Slot::new(critic),
            target_critic,
            actor_opt,
            critic_opt,
            buffer,
            exploration,
            low: A::low(),
            high: A::high(),
            low_t,
            high_t,
            config,
            device,
            plain_device,
            step: 0,
            critic_updates: 0,
            last_actor_loss: 0.0,
            last_critic_loss: 0.0,
            critic_guard: FiniteLossGuard::new("ddpg/critic"),
            actor_guard: FiniteLossGuard::new("ddpg/actor"),
            reward_guard: FiniteRewardGuard::new("ddpg/remember"),
            obs_guard: FiniteObsGuard::ingestion("ddpg/remember"),
            act_obs_guard: FiniteObsGuard::act("ddpg/act"),
            obs_scratch: Vec::new(),
            _action: PhantomData,
        })
    }

    /// Number of transitions currently stored.
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Global env-step count.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Number of critic updates **attempted** — the counter that drives the
    /// `policy_frequency` actor cadence and the
    /// [`target_update`](DdpgTrainingConfig::target_update) Polyak cadence.
    ///
    /// This is an *attempt* count, not an *applied* count: it advances
    /// unconditionally at the end of every learn step, including one whose
    /// critic loss was non-finite and therefore skipped (ADR 0059, Decision 4).
    /// Keeping it unconditional is what makes the two cadences immune to a
    /// transient `NaN` — a skip must not silently retime the actor and target
    /// updates. Subtract
    /// [`skipped_critic_updates`](Self::skipped_critic_updates) to recover the
    /// number that actually reached the optimizer:
    ///
    /// ```text
    /// applied = critic_updates() - skipped_critic_updates()
    /// ```
    #[must_use]
    pub const fn critic_updates(&self) -> usize {
        self.critic_updates
    }

    /// Number of **critic** updates skipped because the critic loss was
    /// non-finite (ADR 0056, ADR 0072).
    ///
    /// # Relationship to [`critic_updates`](Self::critic_updates)
    ///
    /// [`critic_updates`](Self::critic_updates) counts **attempts** and
    /// advances unconditionally, including on a skip (ADR 0059, Decision 4);
    /// this counts the **subset of those attempts that was skipped**, so
    ///
    /// ```text
    /// applied = critic_updates() - skipped_critic_updates()
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
    /// still in the buffer and will be sampled again. Neither counter bounds
    /// the other, and a run can show one without the other.
    #[must_use]
    pub const fn skipped_critic_updates(&self) -> u64 {
        self.critic_guard.skipped()
    }

    /// Number of **actor** updates skipped because the actor loss was
    /// non-finite (ADR 0056, ADR 0072).
    ///
    /// # Not comparable to [`skipped_critic_updates`](Self::skipped_critic_updates)
    ///
    /// The actor guard sits *inside* the `policy_frequency` cadence block, so
    /// the actor is only ever *attempted* on every `policy_frequency`-th critic
    /// update. Actor attempts are therefore fewer than critic attempts by
    /// construction, and this count is drawn from that smaller pool: at the
    /// shipped `policy_frequency = 2`, a run whose every loss is non-finite
    /// reports roughly half as many actor skips as critic skips. Reading the
    /// two as a like-for-like comparison — "the actor is healthier than the
    /// critic" — is a misreading of the cadence, not a signal.
    ///
    /// The same attempts-minus-skips identity holds against the actor's own
    /// attempt count, which is `critic_updates() / policy_frequency`.
    #[must_use]
    pub const fn skipped_actor_updates(&self) -> u64 {
        self.actor_guard.skipped()
    }

    /// Total gradient updates skipped for a non-finite loss, across **both**
    /// loss sites — the canonical `skipped_updates` metric this agent reports
    /// (ADR 0072, Decisions 2 and 3).
    ///
    /// The sum of [`skipped_critic_updates`](Self::skipped_critic_updates) and
    /// [`skipped_actor_updates`](Self::skipped_actor_updates), saturating at
    /// `u64::MAX` (a bound no real run approaches; the saturation exists so the
    /// metric can never wrap into a small number).
    ///
    /// # Why the aggregate is the one to report
    ///
    /// DDPG has two loss sites today and could gain a third. A reporter that
    /// reads *this* accessor rather than adding the per-site counts itself, so
    /// a new guard is picked up by extending this method alone — a site cannot
    /// be silently omitted from the reported metric by a caller that has not
    /// heard of it. Per-site attribution is what the two accessors above are
    /// for; this is the number to alert on.
    #[must_use]
    pub const fn skipped_updates(&self) -> u64 {
        self.critic_guard
            .skipped()
            .saturating_add(self.actor_guard.skipped())
    }

    /// Samples an action for the current observation.
    ///
    /// Before `learning_starts` steps, draws a uniform random action on
    /// `[low, high]`. Afterwards runs the actor, adds Gaussian noise, and
    /// clips to the action bounds. The unnoised policy mean is emitted when
    /// `training == false`.
    ///
    /// # Behavior on a non-finite observation
    ///
    /// The observation row is checked for finiteness before it reaches the actor. A `NaN` /
    /// `$\pm\infty$` row is **counted and warned about, and the action is returned unchanged** —
    /// nothing is substituted, no fallback is returned, and the clamping is not altered (ADR 0067,
    /// Decision 4). Read the count with
    /// [`degenerate_action_selections`](Self::degenerate_action_selections).
    ///
    /// The warm-up branch is deliberately outside the check: it never reads
    /// `obs`, so no action was selected *from* a poisoned observation there.
    /// The step is still caught at [`remember`](Self::remember).
    ///
    /// # Why the check stays
    ///
    /// Under Burn 0.21 the continuous failure mode was **backend-split**.
    /// Measured on Pendulum/DDPG: on wgpu/Metal a non-finite observation
    /// yielded `raw_actor = [NaN]` and an action whose `is_valid()` was
    /// `false`; on Flex `relu` rescued `NaN` to `0.0`, so the actor emitted a
    /// finite, plausible, in-bounds torque (measured `0.08646313`) from a
    /// **fully** `NaN` observation, with `is_valid() == true`, and nothing
    /// downstream could see it.
    ///
    /// Burn 0.22's `relu` propagates `NaN` on every backend measured (Flex and
    /// wgpu/Metal), so both now return an action with `is_valid() == false`.
    /// The check names the cause where that action shows only the effect, and
    /// it remains the one observer on any backend whose `relu` erases `NaN`.
    ///
    /// # The clamp here is host `f32::clamp`, and must stay that way
    ///
    /// Every continuous act-path clip in this crate is [`f32::clamp`] over a
    /// `&[f32]` already read back from the actor — **not** `Tensor::clamp`.
    /// `f32::clamp(NaN, lo, hi)` propagates `NaN`, and does so identically on
    /// every backend, so ADR 0066's `clamp_preserving_nan` does **not** apply
    /// here and must not be introduced. Moving this clip into tensor space
    /// would make it depend on each backend's `Tensor::clamp` again — the C51
    /// defect: Burn 0.21's wgpu rescued `NaN` to the lower bound, turning a
    /// reported failure into a plausible in-domain torque with `is_valid()`
    /// back to `true`.
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
    /// Panics if the actor slot is poisoned — see [`Slot::get`]. This requires
    /// a prior panic *inside* the actor's optimizer step; a panic anywhere else
    /// in [`learn_step`](Self::learn_step) leaves this method working. Also
    /// panics if the actor's output tensor is not `f32`: that host-read is an
    /// `.expect` on a named invariant, the form the Error Handling section of
    /// `docs/rules.md` sanctions here because `act` returns a bare action and so
    /// has no error channel to report the failure through — making the path
    /// fallible is a breaking change, deferred and tracked separately.
    fn select(&self, obs: &O, training: bool, rng: &mut (impl Rng + ?Sized)) -> A {
        if training && self.step < self.config.learning_starts {
            let sample: Vec<f32> = (0..A::COMPONENTS)
                .map(|i| rng.random_range(self.low[i]..=self.high[i]))
                .collect();
            return A::from_slice(&sample);
        }

        // `&self` (agents must stay `Sync` — the evolution layer evaluates them in parallel), so
        // the staging buffer cannot be a field and must not be a shared `RefCell` / `Mutex`. Cost
        // of the deliberate alternative: one `Vec<f32>` allocation per call, and only for f32
        // feature-vector observations (`$\leq 24$` elements in this workspace). The four
        // integer-backed observation types override `row_is_finite` without touching `scratch`, so
        // this `Vec` is never allocated into for them (ADR 0067, Decision 2).
        let mut scratch: Vec<f32> = Vec::new();
        self.act_obs_guard.report(obs.row_is_finite(&mut scratch));

        let obs_t: Tensor<OR> = obs.to_tensor(&self.device);
        let batched: Tensor<BOR> = obs_t.unsqueeze::<BOR>();
        let raw: Tensor<BAR> = self.actor.get().forward(batched);
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

    /// Snapshots the actor onto the plain (non-autodiff) device for repeated
    /// greedy inference.
    ///
    /// Returns a frozen inference handle for use with
    /// [`act_with`](Self::act_with). Snapshot once after training, then reuse
    /// across many steps — the snapshot goes stale if the actor is updated
    /// again.
    ///
    /// # Panics
    ///
    /// Panics if the actor slot is poisoned — see `Slot::get`.
    pub fn inference_net(&self) -> Actor {
        self.actor.get().valid()
    }

    /// Deterministic action against a pre-snapshotted inner actor.
    ///
    /// Equivalent to `act(obs, false, _)` (the unnoised, bound-clipped policy
    /// mean) but runs on the plain device via
    /// [`inference_net`](Self::inference_net), avoiding the per-call autodiff
    /// graph construction that [`act`](Self::act) incurs.
    ///
    /// # Behavior on a non-finite observation
    ///
    /// Identical to [`act`](Self::act), and sharing its counter: the row is
    /// checked, a non-finite row is counted and warned about, and the action is
    /// returned **unchanged**. See [`act`](Self::act) for why nothing is
    /// substituted, why the backends disagreed under Burn 0.21, and why the
    /// clamp below must stay a host [`f32::clamp`].
    ///
    /// # Panics
    ///
    /// Panics if the actor's output tensor is not `f32`, or if it yields fewer
    /// than `A::COMPONENTS` values. Both indicate the supplied `net` does not
    /// match the action type this agent was built for. Making this path
    /// fallible so the first of those becomes an `Err` instead is deferred
    /// and tracked separately.
    pub fn act_with(&self, net: &Actor, obs: &O) -> A {
        // Function-local for the same `&self` / `Sync` reason as `act`.
        let mut scratch: Vec<f32> = Vec::new();
        self.act_obs_guard.report(obs.row_is_finite(&mut scratch));

        let obs_t: Tensor<OR> = obs.to_tensor(&self.plain_device);
        let batched: Tensor<BOR> = obs_t.unsqueeze::<BOR>();
        let raw: Tensor<BAR> = net.forward(batched);
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
    /// Each call counts one environment step, whether the transition is
    /// stored or dropped: the counter `learning_starts` gates learning on.
    ///
    /// # Arguments
    ///
    /// - `status` — the snapshot's status, unchanged. When the transition is
    ///   sampled, only [`EpisodeStatus::Terminated`] zeroes the Bellman
    ///   bootstrap; after a truncation (a time-limit cut) `next_obs` is a
    ///   genuine continuation state whose value is still bootstrapped. See
    ///   [`Transition::status`].
    ///
    /// [`Transition::status`]: rlevo_core::agent::Transition::status
    /// [`EpisodeStatus::Terminated`]: rlevo_core::environment::EpisodeStatus::Terminated
    ///
    /// # Behavior
    ///
    /// A non-finite `reward` (`NaN` or `$\pm\infty$`) is **discarded, not stored**: the transition
    /// never enters the replay buffer and the call is otherwise a no-op. Storing it would let every
    /// minibatch that later resampled it produce a non-finite loss, which `FiniteLossGuard` then
    /// skips — silently costing gradient updates for as long as the poisoned transition stayed
    /// resident (ADR 0065). A `tracing::warn!` fires on the 1st, 10th, 100th, … drop; use
    /// [`dropped_transitions`](Self::dropped_transitions) to detect the loss programmatically.
    ///
    /// A non-finite **observation** — on either `obs` or `next_obs` — is
    /// discarded the same way, with its own counter
    /// ([`dropped_observations`](Self::dropped_observations)) and its own
    /// decade-scheduled `warn!` (ADR 0067). The reward check runs
    /// **first** and returns early, so a transition that is bad in both ways
    /// increments only `dropped_transitions`.
    pub fn remember(
        &mut self,
        obs: O,
        action: &A,
        reward: f32,
        next_obs: O,
        status: EpisodeStatus,
    ) {
        // Every step offered counts, stored or dropped: `learning_starts` and
        // the replay schedule read environment steps, not stored transitions.
        self.step += 1;
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
            status,
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
    /// transitions on its own, and their **sum** is that total (the
    /// Consequences section of ADR 0067).
    #[must_use]
    pub const fn dropped_transitions(&self) -> u64 {
        self.reward_guard.dropped()
    }

    /// Number of transitions [`remember`](Self::remember) discarded because `obs` or `next_obs`
    /// carried a non-finite value (`NaN` / `$\pm\infty$`).
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
    /// transitions and neither is that total on its own (the Consequences
    /// section of ADR 0067).
    #[must_use]
    pub fn dropped_observations(&self) -> u64 {
        self.obs_guard.count()
    }

    /// Number of actions [`act`](Self::act) / [`act_with`](Self::act_with)
    /// returned from a **non-finite observation**.
    ///
    /// This is not a drop count: per ADR 0067 Decision 4 the action was
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
    /// is not comparable between the discrete and the continuous family
    /// during the early exploration / warm-up period, because the two families
    /// place the guard on opposite sides of their random-action branch.
    ///
    /// - **Discrete** (`dqn`, `c51`, `qrdqn`) — the guard sits *inside* the
    ///   `$\epsilon$`-explore branch, and the greedy branch delegates to `act_greedy`,
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
    /// Exists so the driven tests can assert that a *truncated* step was
    /// recorded with `terminated = false` — the invariant that separates a
    /// time-limit cutoff from a real MDP termination.
    #[cfg(test)]
    pub(crate) fn replay_terminated_flags(&self) -> Vec<bool> {
        self.buffer
            .iter()
            .map(|t| t.status.is_terminated())
            .collect()
    }

    /// Test-only parameter checksums of the two **target** networks, as
    /// `[target_actor, target_critic]`.
    ///
    /// The target-network observation seam of ADR 0058: nothing else in this
    /// crate can read a target's weights, which is how the two-schedule
    /// defect survived its tests. Paired with
    /// [`live_checksums`](Self::live_checksums) it makes a target update's
    /// *cadence* and *magnitude* both assertable — see
    /// [`param_checksum`](crate::algorithms::shared::param_checksum) for why a
    /// sum suffices under Polyak averaging.
    #[cfg(test)]
    pub(crate) fn target_checksums(&self) -> [f64; 2] {
        [
            param_checksum(&self.target_actor),
            param_checksum(&self.target_critic),
        ]
    }

    /// Test-only parameter checksums of the two **live** networks, in the same
    /// order as [`target_checksums`](Self::target_checksums).
    #[cfg(test)]
    pub(crate) fn live_checksums(&self) -> [f64; 2] {
        [
            param_checksum(self.actor.get()),
            param_checksum(self.critic.get()),
        ]
    }

    /// Returns `true` once the warm-up period has elapsed and the buffer has
    /// enough transitions to draw a batch.
    fn can_learn(&self) -> bool {
        self.buffer.len() >= self.config.batch_size && self.step >= self.config.learning_starts
    }

    /// Runs one learning step: a critic update, an actor update every
    /// `policy_frequency`-th critic update, and a target Polyak update every
    /// [`target_update.every()`](crate::target::TargetUpdate::every)-th one.
    ///
    /// Returns `None` if the agent is still in warm-up (see
    /// `can_learn`).
    ///
    /// The critic is updated every call via mean-squared Bellman error. The actor is updated with
    /// the deterministic policy gradient (negative mean Q-value over the batch). Both target
    /// networks are Polyak-averaged toward the active networks with rate `$\tau$`
    /// ([`DdpgTrainingConfig::target_update`]) on their own cadence — which is independent of
    /// `policy_frequency` since ADR 0058, though the two coincide at the shipped defaults. When
    /// both fire on the same critic update the actor step runs first, so the target never tracks a
    /// stale actor.
    ///
    /// # Panics
    ///
    /// Panics if the actor or critic slot is poisoned, i.e. a previous
    /// [`ModuleOptimizer::step`](burn::optim::ModuleOptimizer::step) unwound and lost the
    /// network. Every other operation here — sampling, the forward passes, the
    /// losses, `backward`, and gradient reduction — runs on a borrow, so a
    /// panic in any of them leaves the agent fully usable; only a panic inside
    /// the optimizer step itself is terminal. See `Slot`.
    ///
    /// # Errors
    ///
    /// Returns a [`PolyakError`] if a target soft-update finds a
    /// parameter-topology mismatch between a live network and its target twin
    /// (see `polyak_update`). Every in-tree
    /// target is cloned from its active network, so this cannot occur for
    /// agents built normally.
    pub fn learn_step(
        &mut self,
        rng: &mut (impl Rng + ?Sized),
    ) -> Result<Option<LearnOutcome>, PolyakError> {
        if !self.can_learn() {
            return Ok(None);
        }
        let batch_size = self.config.batch_size;
        let device = self.device.clone();
        let plain = self.plain_device.clone();

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
            terminated.push(if t.status.is_terminated() { 1.0 } else { 0.0 });
        }

        let mut batched_obs_shape: Vec<usize> = Vec::with_capacity(BOR);
        batched_obs_shape.push(batch_size);
        batched_obs_shape.extend_from_slice(&obs_shape);
        let mut batched_action_shape: Vec<usize> = Vec::with_capacity(BAR);
        batched_action_shape.push(batch_size);
        batched_action_shape.extend_from_slice(&action_shape);

        let obs_t: Tensor<BOR> = Tensor::from_data(
            TensorData::new(obs_flat, batched_obs_shape.clone()),
            &device,
        );
        let next_t_inner: Tensor<BOR> =
            Tensor::from_data(TensorData::new(next_flat, batched_obs_shape), &plain);
        let action_t: Tensor<BAR> =
            Tensor::from_data(TensorData::new(action_flat, batched_action_shape), &device);

        let rewards_inner: Tensor<1> =
            Tensor::from_data(TensorData::new(rewards, vec![batch_size]), &plain);
        let terminated_inner: Tensor<1> =
            Tensor::from_data(TensorData::new(terminated, vec![batch_size]), &plain);

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
        let next_actions: Tensor<BAR> = clip_to_action_bounds(
            self.target_actor.forward(next_t_inner.clone()),
            self.low_t.clone(),
            self.high_t.clone(),
        );
        let next_q: Tensor<1> = self.target_critic.forward(next_t_inner, next_actions);
        let target_inner: Tensor<1> =
            compute_target_q_values(rewards_inner, next_q, terminated_inner, self.config.gamma);
        let target: Tensor<1> = target_inner.to_device(&device);

        // --- Critic update ---
        // Everything up to `step_with` runs on a borrow of the slot, so a panic
        // in the forward/loss/backward region cannot poison the agent.
        let q_pred: Tensor<1> = self.critic.get().forward(obs_t.clone(), action_t);
        let q_mean = q_pred.clone().mean().into_scalar::<f32>();
        let td_error = q_pred - target;
        let critic_loss_tensor = td_error.powi_scalar(2).mean();
        let critic_loss = critic_loss_tensor.clone().into_scalar::<f32>();

        // ADR 0056: `critic_loss` is already host-resident, so the
        // finiteness check costs no extra sync. A non-finite loss skips the
        // critic `backward()` + optimizer step (Burn would otherwise fold NaN
        // into the weights silently); the value is excluded from the reported
        // metric (`last_critic_loss` carries its last applied value forward).
        // `critic_updates` (the actor cadence counter) still advances
        // unconditionally, per ADR 0056, Decision 3.
        if self.critic_guard.check(critic_loss) {
            let grads = critic_loss_tensor.backward();
            let grads = GradientsParams::from_grads(grads, self.critic.get());
            self.critic
                .step_with(&mut self.critic_opt, self.config.critic_lr, grads);
            self.last_critic_loss = critic_loss;
        }
        self.critic_updates += 1;

        // --- Actor update (every policy_frequency-th critic step) ---
        let mut actor_loss_opt: Option<f32> = None;
        if self
            .critic_updates
            .is_multiple_of(self.config.policy_frequency)
        {
            let predicted_actions: Tensor<BAR> = self.actor.get().forward(obs_t.clone());
            let q_actor: Tensor<1> = self.critic.get().forward(obs_t, predicted_actions);
            let actor_loss_tensor = q_actor.mean().neg();
            let actor_loss_value = actor_loss_tensor.clone().into_scalar::<f32>();

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
        // block above: before ADR 0058 the Polyak calls lived inside it, which
        // made `policy_frequency` an undeclared alias for the target cadence —
        // an operator could not change the actor delay without also changing
        // how often the target moved. Both gates read the same post-increment
        // `critic_updates` counter, so at the shipped defaults
        // (`policy_frequency == target_update.every() == 2`) the predicate is
        // identical and the trajectory is unchanged.
        //
        // Order is load-bearing and matches the pre-ADR-0058 code exactly: the
        // actor step must precede this block, or the target would track a stale
        // actor for one cadence period.
        //
        // Target Polyak updates: cadence-driven (per ADR 0056, Decision 3), so they
        // run whether or not the actor step was skipped — the healthy
        // networks keep being tracked by their targets. That invariant survives
        // the move unchanged: this block never consults the actor guard, and a
        // skipped actor step still leaves `self.actor` a valid network to blend
        // from. Clone rather than move out: `polyak_update` consumes `target` by
        // value, so on `Err` the `?` returns before the reassignment and each
        // target field keeps its prior weights — no silent hard-sync onto its
        // live network (the invariant now holds via early return, and equally
        // for a panic).
        if let Some(tau) = self.config.target_update.fires_at(self.critic_updates) {
            self.target_actor =
                polyak_update(&self.actor.get().valid(), self.target_actor.clone(), tau)?;
            self.target_critic =
                polyak_update(&self.critic.get().valid(), self.target_critic.clone(), tau)?;
        }

        Ok(Some(LearnOutcome {
            // Report the most recent *applied* critic loss so a skipped
            // (non-finite) step carries its last healthy value forward rather
            // than poisoning the metric with a NaN (ADR 0056, Decision 3).
            critic_loss: self.last_critic_loss,
            actor_loss: actor_loss_opt,
            q_mean,
        }))
    }

    /// Most recent actor loss (persists between policy updates).
    pub fn last_actor_loss(&self) -> f32 {
        self.last_actor_loss
    }
}

/// Driven through [`Agent`], the agent acts with exploration and learns
/// while training, and acts on its deterministic actor otherwise.
///
/// Each step [`on_transition`](Agent::on_transition) stores the transition,
/// advances the step counter and takes a learning step, which does nothing
/// until the buffer holds `learning_starts` transitions. While evaluating it
/// does nothing.
impl<Actor, Critic, O, A, const OR: usize, const BOR: usize, const AR: usize, const BAR: usize>
    Agent<O, A> for DdpgAgent<Actor, Critic, O, A, OR, BOR, AR, BAR>
where
    Actor: DeterministicPolicy<BOR, BAR>,
    Critic: ContinuousQ<BOR, BAR>,
    O: Observation<OR> + TensorConvertible<OR>,
    A: BoundedAction<AR>,
{
    fn act(&mut self, obs: &O, rng: &mut dyn Rng) -> A {
        self.select(obs, self.training, rng)
    }

    fn on_transition(&mut self, t: Transition<&O, &A>, rng: &mut dyn Rng) {
        if !self.training {
            return;
        }
        self.remember(
            t.obs.clone(),
            t.action,
            t.reward,
            t.next_obs.clone(),
            t.status,
        );
        self.learn_step(rng)
            .expect("a target network cloned from its online network shares its parameter ids");
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
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
    use crate::algorithms::shared::{params_are_plain, params_train};
    use rlevo_core::config::ConstraintKind;

    // -------- non-finite-loss guard (ADR 0056) --------

    use crate::algorithms::bootstrap_mask::{
        MaskContinuousAction, MaskObservation, tiny_actor, tiny_critic,
    };
    use crate::algorithms::ddpg::ddpg_model::MlpActor;
    use crate::algorithms::ddpg::ddpg_model::MlpCritic;
    use burn::module::{Module, ModuleMapper, Param};
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rlevo_core::action::ContinuousAction;

    type GuardAgent =
        DdpgAgent<MlpActor, MlpCritic, MaskObservation, MaskContinuousAction, 1, 2, 1, 2>;

    /// Replaces every float parameter of a module with `NaN`, simulating a
    /// critic that has diverged to non-finite weights — the realistic source of
    /// a non-finite critic loss.
    struct NanInjector;

    impl ModuleMapper for NanInjector {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
            let (id, tensor, mapper) = param.consume();
            Param::from_mapped_value(id, tensor.mul_scalar(f32::NAN), mapper)
        }
    }

    /// Builds a `MaskObservation` from two feature values (its fields are
    /// private to `bootstrap_mask`).
    fn make_obs(a: f32, b: f32) -> MaskObservation {
        MaskObservation::new([a, b])
    }

    /// A non-finite critic loss must skip the critic `backward` + optimizer step
    /// (ADR 0056). Diverging the critic to NaN forces a NaN loss; the
    /// critic guard must fire, the skipped value must not poison the reported
    /// metric, and — with the actor + target update held off this step — the
    /// actor stays finite and untouched.
    #[test]
    fn test_ddpg_agent_nonfinite_critic_loss_skips_and_warns() {
        let device = Device::flex();
        let config = DdpgTrainingConfig {
            batch_size: 2,
            learning_starts: 0,
            replay_buffer_capacity: 64,
            critic_lr: 0.05,
            // policy_frequency = 2 keeps the actor + Polyak update off this
            // single step, so the actor and target networks are provably
            // untouched by the critic's NaN.
            policy_frequency: 2,
            ..Default::default()
        };

        let mut agent = GuardAgent::new(tiny_actor(&device), tiny_critic(&device), config, device)
            .expect("valid agent");

        let action = MaskContinuousAction::from_slice(&[0.0]);
        for x in [0.0_f32, 0.1, 0.2, 0.3] {
            agent.remember(
                make_obs(x, 1.0 - x),
                &action,
                0.5,
                make_obs(x + 0.1, 0.9 - x),
                EpisodeStatus::Running,
            );
        }

        // Diverge the critic to non-finite weights so its loss is NaN. Poison a
        // *clone* to preserve `ParamId`s and the target Polyak pairing.
        let poisoned = agent.critic.get().clone().map(&mut NanInjector);
        agent.critic = Slot::new(poisoned);

        let mut rng = StdRng::seed_from_u64(0);
        let outcome = agent
            .learn_step(&mut rng)
            .expect("no polyak error")
            .expect("a primed agent past warm-up learns");

        assert_eq!(
            agent.skipped_critic_updates(),
            1,
            "the one poisoned critic step must be counted as exactly one skip"
        );
        assert_eq!(
            agent.skipped_actor_updates(),
            0,
            "the actor block did not run this step: the sibling site must be \
             untouched, with a count of exactly zero"
        );
        assert!(
            outcome.actor_loss.is_none(),
            "the actor update was off this step, so no actor loss is reported"
        );
        assert_eq!(
            outcome.critic_loss, 0.0,
            "the skipped critic loss must not poison the metric (stays at its 0.0 seed)"
        );
        assert!(
            outcome.critic_loss.is_finite(),
            "the reported critic loss must stay finite despite the skipped site"
        );
        let probe_action = agent.select(&make_obs(0.2, 0.8), false, &mut rng);
        assert!(
            probe_action.as_slice()[0].is_finite(),
            "the actor must remain finite — the critic's NaN must not reach it"
        );
    }

    /// Builds the same primed agent as
    /// [`ddpg_nonfinite_critic_loss_skips_and_warns`] — four transitions in the
    /// buffer, `learning_starts = 0`, `batch_size = 2` — with its critic already
    /// diverged to `NaN` weights, so *every* subsequent `learn_step` produces a
    /// non-finite critic loss. `policy_frequency` is a parameter because it
    /// decides how many of those steps also attempt an actor update.
    fn poisoned_critic_agent(policy_frequency: usize) -> GuardAgent {
        let device = Device::flex();
        let config = DdpgTrainingConfig {
            batch_size: 2,
            learning_starts: 0,
            replay_buffer_capacity: 64,
            critic_lr: 0.05,
            policy_frequency,
            ..Default::default()
        };

        let mut agent = GuardAgent::new(tiny_actor(&device), tiny_critic(&device), config, device)
            .expect("valid agent");

        let action = MaskContinuousAction::from_slice(&[0.0]);
        for x in [0.0_f32, 0.1, 0.2, 0.3] {
            agent.remember(
                make_obs(x, 1.0 - x),
                &action,
                0.5,
                make_obs(x + 0.1, 0.9 - x),
                EpisodeStatus::Running,
            );
        }

        // Poison a *clone* to preserve `ParamId`s and the target Polyak pairing.
        let poisoned = agent.critic.get().clone().map(&mut NanInjector);
        agent.critic = Slot::new(poisoned);
        agent
    }

    /// The skip counter is a **counter**, not a latch: three consecutive
    /// poisoned critic steps must read exactly `3` (ADR 0072, Decision 1). A count of
    /// `1` here would mean the counter was mistranslated from the old
    /// one-shot `warning_fired` boolean.
    #[test]
    fn test_ddpg_agent_counts_repeated_loss_skips() {
        // `policy_frequency = 4` keeps the actor block off all three steps
        // (`critic_updates` reaches 1, 2, 3 — none a multiple of 4), so the
        // critic count below is attributable to the critic site alone.
        let mut agent = poisoned_critic_agent(4);
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
            "all three attempts must be counted, skip or not (ADR 0059 §Decision 4)"
        );
        assert_eq!(
            agent.skipped_critic_updates(),
            3,
            "three consecutive non-finite critic losses must count as three \
             skips, not one — the counter must not latch"
        );
        assert_eq!(
            agent.skipped_actor_updates(),
            0,
            "the actor block never ran at policy_frequency = 4, so its count \
             must be exactly zero"
        );
        assert_eq!(
            agent.skipped_updates(),
            3,
            "the aggregate over a silent actor site must equal the critic count"
        );
    }

    /// The aggregate must sum the two *distinct* guards. The parts are
    /// deliberately **unequal** (2 and 1): equal parts would also be satisfied
    /// by an aggregate that added one guard to itself.
    ///
    /// The numbers come from the cadence, which is what makes them unequal for
    /// free. At `policy_frequency = 2` over two learn steps, `critic_updates`
    /// reaches 1 then 2, so the critic site is attempted twice and the actor
    /// site only on the second step. The poisoned critic makes *both* losses
    /// non-finite — the actor loss is `-mean(critic(obs, actor(obs)))`, so a
    /// `NaN` critic poisons it too — giving 2 critic skips against 1 actor skip.
    #[test]
    fn test_ddpg_agent_skipped_updates_aggregates_unequal_sites() {
        let mut agent = poisoned_critic_agent(2);
        let mut rng = StdRng::seed_from_u64(0);

        for _ in 0..2 {
            agent
                .learn_step(&mut rng)
                .expect("no polyak error")
                .expect("a primed agent past warm-up learns");
        }

        assert_eq!(
            agent.skipped_critic_updates(),
            2,
            "the critic site is attempted every step: two steps, two skips"
        );
        assert_eq!(
            agent.skipped_actor_updates(),
            1,
            "the actor site is attempted only on the second step at \
             policy_frequency = 2: one skip"
        );
        assert_eq!(
            agent.skipped_updates(),
            3,
            "the aggregate must be critic (2) + actor (1) = 3 — not either \
             site doubled (4 or 2)"
        );
    }

    // -------- target-update cadence (ADR 0058 / 0059) --------

    use crate::target::TargetUpdate;
    use approx::assert_abs_diff_eq;

    /// Slack for the Polyak identity below. Each checksum is an `f32` device
    /// reduction over a handful of parameters and each blended parameter costs
    /// two `f32` roundings, so ~2e-6 is the realistic worst case; the smallest
    /// signal any assertion here reads is `$\tau \cdot \text{gap} \approx 7.5\times10^{-3}$`, three orders of
    /// magnitude larger.
    const CHECKSUM_EPS: f64 = 1e-5;

    /// Adds a constant to every float parameter of a module.
    ///
    /// Both targets are built by cloning their live network, so they start
    /// *identical* — and a Polyak blend between identical networks is a no-op
    /// that every "did the target move, and by how much?" assertion would pass
    /// vacuously. That exact vacuity is how the two-schedule defect survived
    /// its tests; this mapper removes it by opening a known gap. It maps a
    /// *clone*, so `ParamId`s are preserved and the Polyak pairing stays valid.
    struct AddConstant(f32);

    impl ModuleMapper for AddConstant {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
            let (id, tensor, mapper) = param.consume();
            Param::from_mapped_value(id, tensor.add_scalar(self.0), mapper)
        }
    }

    /// A primed agent whose live networks are held a known distance from their
    /// targets, so every cadence assertion below is non-vacuous.
    fn cadence_agent(policy_frequency: usize, target_update: TargetUpdate) -> GuardAgent {
        let device = Device::flex();
        let config = DdpgTrainingConfig {
            batch_size: 2,
            learning_starts: 0,
            replay_buffer_capacity: 64,
            actor_lr: 1e-3,
            critic_lr: 1e-3,
            policy_frequency,
            target_update,
            ..Default::default()
        };

        let mut agent = GuardAgent::new(tiny_actor(&device), tiny_critic(&device), config, device)
            .expect("valid agent");

        let action = MaskContinuousAction::from_slice(&[0.0]);
        for x in [0.0_f32, 0.1, 0.2, 0.3] {
            agent.remember(
                make_obs(x, 1.0 - x),
                &action,
                0.5,
                make_obs(x + 0.1, 0.9 - x),
                EpisodeStatus::Running,
            );
        }

        // Nudge the *targets*, not the live networks. Perturbing a live network
        // would disturb Burn's autodiff registration across consecutive learn
        // steps, and pushing the actor's weights up saturates its `tanh`,
        // zeroing the very actor gradient one of these tests asserts on. The
        // targets are plain-device modules with no graph attached, and
        // `AddConstant` preserves their `ParamId`s, so the Polyak pairing holds.
        agent.target_actor = agent.target_actor.clone().map(&mut AddConstant(0.5));
        agent.target_critic = agent.target_critic.clone().map(&mut AddConstant(0.5));
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
    /// `$\text{target} \leftarrow (1 - \tau) \cdot \text{target} + \tau \cdot \text{active}$`, and that it moved at all.
    fn assert_fired(before: [f64; 2], after: [f64; 2], live: [f64; 2], tau: f64) {
        for (i, ((&b, &a), &l)) in before.iter().zip(after.iter()).zip(live.iter()).enumerate() {
            assert_abs_diff_eq!(a, (1.0 - tau) * b + tau * l, epsilon = CHECKSUM_EPS);
            assert!(
                (a - b).abs() > 1e-4,
                "network {i}'s target must actually have moved: {b} -> {a}"
            );
        }
    }

    /// DDPG's shipped cadence is `polyak(0.005, 2)`: the targets move on critic
    /// updates 2, 4, … and are **bit-identically unchanged** on 1, 3, ….
    ///
    /// This is the test that proves lifting the Polyak calls out of the
    /// `policy_frequency` block did not shift the cadence — `every = 2` is the
    /// same number the block used to be gated on.
    #[test]
    fn test_ddpg_agent_default_cadence_fires_every_second_critic_update() {
        let rule = TargetUpdate::polyak(0.005, 2);
        let tau = f64::from(rule.tau());
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
    fn test_ddpg_agent_actor_cadence_and_target_cadence_are_independent() {
        let rule = TargetUpdate::polyak(0.005, 2);
        let tau = f64::from(rule.tau());
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
    fn test_ddpg_agent_remember_drops_a_nonfinite_reward() {
        let device = Device::flex();
        let config = DdpgTrainingConfig::default();
        let mut agent = GuardAgent::new(tiny_actor(&device), tiny_critic(&device), config, device)
            .expect("valid agent");
        let action = MaskContinuousAction::from_slice(&[0.0]);

        agent.remember(
            make_obs(0.0, 1.0),
            &action,
            f32::NAN,
            make_obs(0.1, 0.9),
            EpisodeStatus::Running,
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
            make_obs(0.1, 0.9),
            &action,
            0.5,
            make_obs(0.2, 0.8),
            EpisodeStatus::Running,
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

    // ---- ADR 0067: non-finite observation ----
    //
    // Same per-file rule as the reward test above: each agent carries its own
    // copy rather than trusting one shared test to stand in for three call
    // sites. `MaskObservation` is the deliberately NaN-emitting observation
    // type here — its `write_host_row` copies its `[f32; 2]` payload verbatim,
    // so `make_obs(f32::NAN, _)` puts a real NaN in the staged row.

    /// A fresh agent with capacity to spare, for the observation-guard tests.
    fn obs_guard_agent() -> GuardAgent {
        let device = Device::flex();
        let config = DdpgTrainingConfig {
            replay_buffer_capacity: 64,
            ..Default::default()
        };
        GuardAgent::new(tiny_actor(&device), tiny_critic(&device), config, device)
            .expect("valid agent")
    }

    #[test]
    fn test_ddpg_agent_remember_drops_a_nonfinite_obs() {
        let mut agent = obs_guard_agent();
        let action = MaskContinuousAction::from_slice(&[0.0]);

        agent.remember(
            make_obs(f32::NAN, 1.0),
            &action,
            0.5,
            make_obs(0.1, 0.9),
            EpisodeStatus::Running,
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

        agent.remember(
            make_obs(0.1, 0.9),
            &action,
            0.5,
            make_obs(0.2, 0.8),
            EpisodeStatus::Running,
        );
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
    fn test_ddpg_agent_remember_drops_a_nonfinite_next_obs() {
        let mut agent = obs_guard_agent();
        let action = MaskContinuousAction::from_slice(&[0.0]);

        agent.remember(
            make_obs(0.0, 1.0),
            &action,
            0.5,
            make_obs(0.1, f32::INFINITY),
            EpisodeStatus::Running,
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

        agent.remember(
            make_obs(0.1, 0.9),
            &action,
            0.5,
            make_obs(0.2, 0.8),
            EpisodeStatus::Running,
        );
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

    /// The Consequences section of ADR 0067: the reward guard runs first and returns early,
    /// so a transition that is bad in *both* ways increments only
    /// `dropped_transitions`. This test pins that ordering — it is the reason
    /// the two counters legitimately disagree, and both accessors document it.
    #[test]
    fn test_ddpg_agent_remember_both_bad_counts_only_the_reward_drop() {
        let mut agent = obs_guard_agent();
        let action = MaskContinuousAction::from_slice(&[0.0]);

        agent.remember(
            make_obs(f32::NAN, 1.0),
            &action,
            f32::NAN,
            make_obs(f32::NAN, 0.9),
            EpisodeStatus::Running,
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

    /// ADR 0067, Decision 4: `act` detects and reports, and **returns the
    /// action anyway**. Not substituting is the decision under test, so the
    /// assertion that an action comes back is as load-bearing as the counter.
    #[test]
    fn test_ddpg_agent_act_counts_a_nonfinite_obs_and_still_returns_an_action() {
        let agent = obs_guard_agent();
        let mut rng = StdRng::seed_from_u64(0);

        let action = agent.select(&make_obs(f32::NAN, 1.0), false, &mut rng);
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
        let _ = agent.select(&make_obs(0.2, 0.8), false, &mut rng);
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
    /// action anyway** (ADR 0067, Decision 4).
    #[test]
    fn test_ddpg_agent_act_with_counts_a_nonfinite_obs_and_still_returns_an_action() {
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
    #[test]
    fn test_ddpg_agent_new_rejects_out_of_range_replay_buffer_capacity() {
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
            let device = Device::flex();
            let config = DdpgTrainingConfig {
                replay_buffer_capacity: capacity,
                ..DdpgTrainingConfig::default()
            };
            let Err(err) =
                GuardAgent::new(tiny_actor(&device), tiny_critic(&device), config, device)
            else {
                panic!("capacity {capacity} must be rejected, not allocated");
            };
            assert_eq!(err.config, "DdpgTrainingConfig");
            assert_eq!(err.field, "replay_buffer_capacity");
            assert_eq!(err.kind, kind);
        }
    }

    /// The target path never tracks. Autodiff is device state in Burn 0.22, so
    /// no type keeps the target networks off the graph: each target must hold plain parameters
    /// after construction and after a learn step's Polyak update, while
    /// the actor and critic train.
    #[test]
    fn test_ddpg_agent_target_path_is_never_tracked() {
        let device = Device::flex();
        let config = DdpgTrainingConfig {
            batch_size: 2,
            learning_starts: 0,
            replay_buffer_capacity: 64,
            policy_frequency: 1,
            target_update: crate::target::TargetUpdate::polyak(0.005, 1),
            ..Default::default()
        };
        let mut agent = GuardAgent::new(tiny_actor(&device), tiny_critic(&device), config, device)
            .expect("valid agent");
        let action = MaskContinuousAction::from_slice(&[0.0]);
        for x in [0.0_f32, 0.1, 0.2, 0.3] {
            agent.remember(
                make_obs(x, 1.0 - x),
                &action,
                0.5,
                make_obs(x + 0.1, 0.9 - x),
                EpisodeStatus::Running,
            );
        }
        assert!(
            params_train(agent.actor.get()),
            "precondition: the actor trains"
        );
        assert!(
            params_train(agent.critic.get()),
            "precondition: the critic trains"
        );
        assert!(
            params_are_plain(&agent.target_actor),
            "the target actor starts plain"
        );
        assert!(
            params_are_plain(&agent.target_critic),
            "the target critic starts plain"
        );
        let before = param_checksum(&agent.target_critic);
        let mut rng = StdRng::seed_from_u64(0);
        agent.learn_step(&mut rng).expect("learn").expect("a step");
        assert!(
            (param_checksum(&agent.target_critic) - before).abs() > 0.0,
            "precondition: the Polyak update ran"
        );
        assert!(
            params_are_plain(&agent.target_actor),
            "the Polyak update leaves the target actor plain"
        );
        assert!(
            params_are_plain(&agent.target_critic),
            "the Polyak update leaves the target critic plain"
        );
        assert!(
            params_are_plain(&agent.inference_net()),
            "`inference_net` is a plain snapshot"
        );
    }
}

#[cfg(test)]
mod driven_tests {
    use super::*;

    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use crate::algorithms::bootstrap_mask::{
        ContinuousMaskEnv, MaskContinuousAction, MaskObservation, drive, tiny_critic,
    };
    use crate::algorithms::ddpg::ddpg_model::MlpActor;
    use crate::algorithms::ddpg::ddpg_model::MlpCritic;

    /// Episodes end every `PERIOD` steps; `STEPS` spans three whole episodes.
    const PERIOD: usize = 3;
    const STEPS: usize = 9;

    use crate::algorithms::bootstrap_mask::tiny_actor;

    /// Drives the agent through its `Agent` hooks against a clock env whose
    /// episodes all end with `end_status`, and returns the bootstrap masks
    /// it stored.
    ///
    /// `learning_starts` is pushed past the step budget so no gradient step
    /// ever runs — this guard is about *which predicate reaches `remember`*,
    /// not about learning.
    fn recorded_masks(end_status: EpisodeStatus) -> Vec<bool> {
        let device = Device::flex();
        let config = DdpgTrainingConfig {
            learning_starts: 1_000_000,
            replay_buffer_capacity: 64,
            ..DdpgTrainingConfig::default()
        };
        let actor = tiny_actor(&device);
        let critic = tiny_critic(&device);
        let mut agent = DdpgAgent::<
            MlpActor,
            MlpCritic,
            MaskObservation,
            MaskContinuousAction,
            1,
            2,
            1,
            2,
        >::new(actor, critic, config, device)
        .expect("default config is valid");
        let mut env = ContinuousMaskEnv::new(PERIOD, end_status);
        let mut rng = StdRng::seed_from_u64(7);
        drive(&mut env, &mut agent, STEPS, &mut rng);
        agent.replay_terminated_flags()
    }

    /// The episode boundaries this fixture produces: steps 3, 6 and 9.
    fn episode_end_steps() -> Vec<bool> {
        (1..=STEPS).map(|s| s.is_multiple_of(PERIOD)).collect()
    }

    /// A genuine environmental termination *must* zero the bootstrap.
    ///
    /// This also pins the fixture's episode cadence, which is what makes the
    /// truncation test below non-vacuous: it proves three of these nine steps
    /// really do end an episode.
    #[test]
    fn test_ddpg_on_transition_termination_sets_bootstrap_mask() {
        assert_eq!(
            recorded_masks(EpisodeStatus::Terminated),
            episode_end_steps(),
            "an environmental termination must zero the Bellman bootstrap"
        );
    }

    /// A truncation must **not** zero the bootstrap.
    ///
    /// Same env, same cadence, same `is_done()` pattern as the test above —
    /// only `is_terminated()` differs. Masking on `is_done()` here would
    /// reproduce `episode_end_steps()` and bias every Q-value downward on any
    /// time-limited env (Pardo et al., ICML 2018, Eq. 6).
    #[test]
    fn test_ddpg_on_transition_truncation_leaves_bootstrap_mask_clear() {
        let flags = recorded_masks(EpisodeStatus::Truncated);
        assert_eq!(
            flags.len(),
            STEPS,
            "every step must reach the replay buffer"
        );
        assert!(
            flags.iter().all(|f| !f),
            "truncation must not zero the bootstrap; got {flags:?}"
        );
    }
}
