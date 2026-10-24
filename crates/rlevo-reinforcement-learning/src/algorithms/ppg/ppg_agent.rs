//! Phasic Policy Gradient (PPG) agent.
//!
//! Owns the policy network (categorical logits head + auxiliary value head),
//! a separate main value network, two Adam optimizers, the PPO
//! [`RolloutBuffer`], an [`AuxRolloutBuffer`] accumulating the last
//! `n_iteration` rollouts, and the configuration. One iteration consists of:
//!
//! 1. [`PpgAgent::act`] / `PpgAgent::record_step` — collect a rollout.
//! 2. `PpgAgent::finalize_rollout` — compute GAE (shared with PPO).
//! 3. `PpgAgent::snapshot_into_aux_buffer` — snapshot observations and
//!    target returns into the auxiliary buffer.
//! 4. `PpgAgent::policy_phase_update` — PPO policy-phase update (clipped
//!    surrogate + value loss), identical to `PpoAgent::update`.
//! 5. `PpgAgent::maybe_aux_phase` — if the auxiliary buffer holds
//!    `n_iteration` rollouts, compute the `$\pi_{old}$` logits once, by running the
//!    current (post-policy-phase) policy over every buffered observation, then
//!    run `e_aux` epochs of auxiliary updates
//!    (main-value MSE + aux-value MSE + `$\beta \cdot KL(\pi_{old} \Vert \pi_{new})$`) and drain
//!    the buffer.
//!
//! The policy-phase loop is code-wise parallel to `PpoAgent::update` but
//! kept in a separate function so `PpoAgent` remains untouched and the
//! snapshot-before-clear hook can slot in cleanly.

use burn::optim::{GradientsParams, ModuleOptimizer};
use burn::tensor::{Device, Tensor, TensorData};
use rand::Rng;
use rlevo_core::action::DiscreteAction;
use rlevo_core::agent::{Agent, Transition};

use crate::algorithms::shared::{FiniteLossGuard, Slot};
use rlevo_core::base::{Observation, TensorConvertible};
use rlevo_core::config::Validate;
use rlevo_core::environment::EpisodeStatus;

use crate::algorithms::ppg::aux_buffer::AuxRolloutBuffer;
use crate::algorithms::ppg::losses::policy_kl_categorical;
use crate::algorithms::ppg::ppg_config::PpgConfig;
use crate::algorithms::ppg::ppg_policy::PpgAuxValueHead;
use crate::algorithms::ppo::losses::{
    approx_kl, clip_fraction, clipped_surrogate, clipped_value_loss, normalize_advantages,
    unclipped_value_loss,
};
use crate::algorithms::ppo::ppo_agent::{ActOutcome, PpoUpdateStats};
use crate::algorithms::ppo::ppo_config::annealed_learning_rate;
use crate::algorithms::ppo::ppo_policy::PpoPolicy;
use crate::algorithms::ppo::ppo_value::PpoValue;
use crate::algorithms::ppo::rollout::{RolloutBuffer, StepEnd};

/// Summary of one auxiliary phase (across `e_aux` epochs and its minibatches).
#[derive(Debug, Clone, Copy)]
pub struct AuxPhaseStats {
    /// Mean main-value MSE across minibatches.
    pub main_value_loss: f32,
    /// Mean aux-value MSE across minibatches.
    pub aux_value_loss: f32,
    /// Mean `$KL(\pi_{old} \Vert \pi_{new})$` distillation loss across minibatches.
    pub policy_kl: f32,
    /// Epochs actually executed.
    pub epochs_run: usize,
    /// Minibatches processed.
    pub minibatches: usize,
    /// Learning rate both optimizer steps ran at — the rate the accompanying
    /// policy phase applied, not the next tick's.
    ///
    /// Reported so callers can tell a phase that *ran but moved nothing*
    /// (`learning_rate == 0.0`) from one that moved the policy imperceptibly.
    /// The loss fields cannot make that distinction: at `lr == 0.0` every
    /// parameter is bit-exactly unchanged and [`policy_kl`](Self::policy_kl)
    /// collapses to `0.0`, which is also its value on a healthy
    /// single-minibatch phase.
    pub learning_rate: f64,
}

/// Phasic Policy Gradient agent.
///
/// Owns the policy network (`P`, which must implement both
/// [`PpoPolicy`] and
/// [`PpgAuxValueHead`]), a separate main value network (`V`), two Adam
/// optimizers, a PPO [`RolloutBuffer`], an [`AuxRolloutBuffer`], and the
/// [`PpgConfig`].
///
/// # Type parameters
///
/// - `P` — Policy network module implementing both `PpoPolicy<BOR>` and
///   `PpgAuxValueHead<BOR>`.
/// - `V` — Main value network implementing `PpoValue<BOR>`.
/// - `O` — Observation type; must be convertible to a rank-`OR` tensor.
/// - `OR` — Rank of the observation tensor (e.g. `1` for flat vectors).
/// - `BOR` — Rank of the batched observation tensor (`OR + 1`; e.g. `2` for a
///   batch of flat vectors).
///
/// # Usage
///
/// Construct with [`PpgAgent::new`], then step it through its [`Agent`] impl.
///
/// # Network ownership
///
/// The policy and value networks live in [`Slot`]s because Burn's
/// `ModuleOptimizer::step` consumes the module by value. In both the policy phase
/// (`policy_phase_update`) and the auxiliary phase
/// (`maybe_aux_phase`), every fallible operation — the
/// forward pass, the losses, `backward`, and `GradientsParams::from_grads` —
/// runs on a borrow, so a panic there leaves both networks intact and the agent
/// usable. Only a panic *inside* the optimizer step itself poisons a slot; that
/// window is irreducible and terminal for the agent (see [`Slot`]).
pub struct PpgAgent<P, V, O, const OR: usize, const BOR: usize>
where
    P: PpoPolicy<BOR> + PpgAuxValueHead<BOR>,
    V: PpoValue<BOR>,
    O: Observation<OR> + TensorConvertible<OR>,
{
    policy: Slot<P>,
    value: Slot<V>,
    policy_optim: ModuleOptimizer,
    value_optim: ModuleOptimizer,
    buffer: RolloutBuffer<O>,
    aux_buffer: AuxRolloutBuffer<O>,
    config: PpgConfig,
    /// The training device: the caller's device with autodiff on.
    device: Device,
    /// `device` without autodiff, where the greedy inference path runs.
    plain_device: Device,
    iteration: usize,
    total_iterations: usize,
    /// Learning rate the most recent [`policy_phase_update`](Self::policy_phase_update)
    /// actually applied, snapshotted *before* that update bumped `iteration`.
    ///
    /// The auxiliary phase belongs to the policy phase it follows, so it steps
    /// at this rate rather than at `current_learning_rate()` — which, read from
    /// [`maybe_aux_phase`](Self::maybe_aux_phase), has already advanced one
    /// annealing tick.
    policy_phase_lr: f64,
    step: usize,
    last_aux: Option<AuxPhaseStats>,
    /// Non-finite-loss guard for the policy-phase policy-loss site (ADR 0056).
    /// Consulted once per minibatch: the skip fires on every non-finite
    /// occurrence, the `warn!` escalates by decades (ADR 0072). Surfaced by
    /// [`skipped_policy_updates`](Self::skipped_policy_updates).
    policy_loss_guard: FiniteLossGuard,
    /// Non-finite-loss guard for the policy-phase value-loss site.
    value_loss_guard: FiniteLossGuard,
    /// Non-finite-loss guard for the auxiliary-phase main-value-loss site.
    aux_main_value_guard: FiniteLossGuard,
    /// Non-finite-loss guard for the auxiliary-phase combined loss
    /// (`$\text{aux\_v\_loss} + \beta \cdot KL$`); its input is host-*derived* from the two summands
    /// already read, adding no sync (ADR 0056, Decision 5).
    aux_total_guard: FiniteLossGuard,
    /// Whether the [`Agent`] impl explores and learns (`true`, the state a
    /// new agent starts in) or acts greedily and learns nothing; see
    /// [`Agent::set_training`].
    training: bool,
    /// The sample [`Agent::act`] drew while training, which
    /// [`Agent::on_transition`] then records with the step it produced.
    pending: Option<ActOutcome>,
    /// The next observation of the rollout's last step while that step
    /// leaves its episode `Running`, kept so the step can still be truncated
    /// if the driver ends the episode there; see `close_open_step`.
    continuation: Option<O>,
}

impl<P, V, O, const OR: usize, const BOR: usize> std::fmt::Debug for PpgAgent<P, V, O, OR, BOR>
where
    P: PpoPolicy<BOR> + PpgAuxValueHead<BOR>,
    V: PpoValue<BOR>,
    O: Observation<OR> + TensorConvertible<OR>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PpgAgent")
            .field("iteration", &self.iteration)
            .field("policy_phase_lr", &self.policy_phase_lr)
            .field("step", &self.step)
            .field("buffer_len", &self.buffer.len())
            .field("aux_slices", &self.aux_buffer.num_slices())
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<P, V, O, const OR: usize, const BOR: usize> PpgAgent<P, V, O, OR, BOR>
where
    P: PpoPolicy<BOR> + PpgAuxValueHead<BOR>,
    V: PpoValue<BOR>,
    O: Observation<OR> + TensorConvertible<OR>,
{
    /// Snapshots the policy without autodiff ([`valid`](burn::module::Module::valid)) for
    /// repeated greedy inference.
    ///
    /// Returns a frozen inference handle for use with
    /// [`act_greedy_env_row_with`](Self::act_greedy_env_row_with). Snapshot
    /// once after training, then reuse across many steps — the snapshot goes
    /// stale if the policy is updated again. Mirrors
    /// [`PpoAgent::inference_net`](crate::algorithms::ppo::ppo_agent::PpoAgent::inference_net).
    ///
    /// # Panics
    ///
    /// Panics if the policy network was lost to a panic inside an earlier
    /// optimizer step. Such an agent is poisoned and must be rebuilt — see
    /// [`Slot`].
    pub fn inference_net(&self) -> P {
        self.policy().valid()
    }

    /// Deterministic env-space action row for `obs`, evaluated against a
    /// pre-snapshotted network on the plain device — no sampling, no autodiff
    /// graph.
    ///
    /// Equivalent to `act_greedy` (the categorical mode is
    /// the argmax over logits, and `raw_to_env_row` is the identity for the
    /// discrete head) but skips per-step autodiff-graph construction, which
    /// dominates cost at batch size 1. Use this for evaluation and throughput.
    pub fn act_greedy_env_row_with(&self, net: &P, obs: &O) -> Vec<f32> {
        let obs_t: Tensor<OR> = obs.to_tensor(&self.plain_device);
        let batched: Tensor<BOR> = obs_t.unsqueeze::<BOR>();
        net.deterministic_env_row(batched)
    }
}

impl<P, V, O, const OR: usize, const BOR: usize> PpgAgent<P, V, O, OR, BOR>
where
    P: PpoPolicy<BOR> + PpgAuxValueHead<BOR>,
    V: PpoValue<BOR>,
    O: Observation<OR> + TensorConvertible<OR>,
{
    /// Construct a new agent from a pre-built policy and value network.
    ///
    /// Initialises both Adam optimizers from `config.ppo.optimizer` (with
    /// optional gradient clipping), creates an empty rollout buffer sized to
    /// `config.ppo.batch_size()`, and an empty auxiliary buffer. The
    /// `total_iterations` value is used only for learning-rate annealing.
    ///
    /// The agent trains on `device` with autodiff on, and switches both
    /// networks to training ([`Module::train`](burn::module::Module::train)), so
    /// networks built on a plain device train as well.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`](rlevo_core::config::ConfigError) if `config`
    /// fails [`PpgConfig::validate`](rlevo_core::config::Validate::validate). In
    /// particular, PPG v1 supports sequential rollout only, so
    /// `config.ppo.num_envs != 1` is rejected here.
    pub fn new(
        policy: P,
        value: V,
        config: PpgConfig,
        device: Device,
        total_iterations: usize,
    ) -> Result<Self, rlevo_core::config::ConfigError> {
        config.validate()?;
        let device = device.autodiff();
        let plain_device = device.clone().without_autodiff();
        let policy = policy.train();
        let value = value.train();
        let action_dim = policy.action_dim();
        let capacity = config.ppo.batch_size();

        let adam_policy = config.ppo.optimizer.clone();
        let policy_optim = match &config.ppo.clip_grad {
            Some(clip) => adam_policy.with_grad_clipping(Some(clip.clone())).init(),
            None => adam_policy.init(),
        };
        let adam_value = config.ppo.optimizer.clone();
        let value_optim = match &config.ppo.clip_grad {
            Some(clip) => adam_value.with_grad_clipping(Some(clip.clone())).init(),
            None => adam_value.init(),
        };

        let buffer = RolloutBuffer::new(capacity, action_dim);
        let aux_buffer = AuxRolloutBuffer::new();
        // What `current_learning_rate()` returns at `iteration == 0`: the
        // annealed rate at tick zero is the base rate, and with annealing off it
        // is the base rate throughout. Seeding the field here keeps a
        // `maybe_aux_phase` that somehow precedes any policy phase on-schedule
        // instead of stepping at an uninitialised rate.
        let policy_phase_lr = config.ppo.learning_rate;

        Ok(Self {
            training: true,

            pending: None,
            continuation: None,
            policy: Slot::new(policy),
            value: Slot::new(value),
            policy_optim,
            value_optim,
            buffer,
            aux_buffer,
            config,
            device,
            plain_device,
            iteration: 0,
            total_iterations,
            policy_phase_lr,
            step: 0,
            last_aux: None,
            policy_loss_guard: FiniteLossGuard::new("ppg/policy_loss"),
            value_loss_guard: FiniteLossGuard::new("ppg/value_loss"),
            aux_main_value_guard: FiniteLossGuard::new("ppg/aux_main_value_loss"),
            aux_total_guard: FiniteLossGuard::new("ppg/aux_total_loss"),
        })
    }

    /// Number of policy-phase updates completed so far.
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Total environment steps taken across all rollouts.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Number of transitions currently held in the rollout buffer.
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Number of rollout slices currently held in the auxiliary buffer.
    ///
    /// The auxiliary phase fires when this reaches `config.n_iteration`.
    pub fn aux_buffer_slices(&self) -> usize {
        self.aux_buffer.num_slices()
    }

    /// Total policy-phase iterations the agent was initialised to run.
    ///
    /// Used only for learning-rate annealing; has no effect when
    /// `config.ppo.anneal_lr` is `false`.
    pub fn total_iterations(&self) -> usize {
        self.total_iterations
    }

    /// Policy-phase **policy** updates skipped for a non-finite loss (ADR 0056,
    /// ADR 0072).
    ///
    /// Counts the `ppg/policy_loss` site only, and counts **minibatches**: the guard is consulted
    /// once per minibatch inside `policy_phase_update`, so one phase
    /// can contribute up to `$\text{update\_epochs} \times \text{num\_minibatches}$` to this total.
    /// It is a lifetime count over every policy phase the agent has run, never reset.
    ///
    /// # Relationship to the reported mean
    ///
    /// The policy phase's reported mean loss divides by the `policy_healthy` count of
    /// that *single* call — the per-call complement of this counter, not a
    /// duplicate of it. This accessor answers "how much of the run's step budget
    /// never reached the optimizer", which no per-call mean can express.
    ///
    #[must_use]
    pub const fn skipped_policy_updates(&self) -> u64 {
        self.policy_loss_guard.skipped()
    }

    /// Policy-phase **value** updates skipped for a non-finite loss (ADR 0056,
    /// ADR 0072).
    ///
    /// The `ppg/value_loss` site, counted per minibatch on the same terms as
    /// [`skipped_policy_updates`](Self::skipped_policy_updates). The two
    /// policy-phase sites are guarded independently — the value net can keep
    /// learning across a phase in which every policy step was skipped, and vice
    /// versa — so this is not a subset of, nor bounded by, that counter.
    #[must_use]
    pub const fn skipped_value_updates(&self) -> u64 {
        self.value_loss_guard.skipped()
    }

    /// Auxiliary-phase **main-value** updates skipped for a non-finite loss
    /// (ADR 0056, ADR 0072).
    ///
    /// The `ppg/aux_main_value_loss` site, counted per minibatch inside
    /// `maybe_aux_phase`.
    ///
    /// # Not comparable with the policy-phase counters
    ///
    /// The auxiliary phase runs on its own cadence — once every
    /// `config.n_iteration` policy phases, for `e_aux` epochs over the whole
    /// auxiliary buffer — so its attempt budget is unrelated to the policy
    /// phase's. A larger `aux_*` count than
    /// [`skipped_policy_updates`](Self::skipped_policy_updates) does **not**
    /// mean the auxiliary phase is the sicker half of the run; the two have
    /// different denominators. Compare each against its own phase's attempts.
    #[must_use]
    pub const fn skipped_aux_value_updates(&self) -> u64 {
        self.aux_main_value_guard.skipped()
    }

    /// Auxiliary-phase **combined-loss** updates skipped for a non-finite loss
    /// (ADR 0056, ADR 0072).
    ///
    /// The `ppg/aux_total_loss` site: the aux-value MSE plus `$\beta \cdot KL$`
    /// distillation term, counted per minibatch inside
    /// `maybe_aux_phase` on the same cadence caveat as
    /// [`skipped_aux_value_updates`](Self::skipped_aux_value_updates).
    ///
    /// # One guard, two reported quantities
    ///
    /// This site's guard input is host-*derived* —
    /// `total_val = aux_v_loss_val + kl_val * beta_clone`, rebuilt on the host
    /// from two scalars already read, rather than a third tensor read back (ADR
    /// 0056, Decision 5). It gates a single optimizer step that owns *both* the
    /// aux-value loss and the KL term, so on a skip **both** are excluded
    /// together. That is why [`AuxPhaseStats::aux_value_loss`] and
    /// [`AuxPhaseStats::policy_kl`] share the one `aux_total_healthy`
    /// denominator, and why four guards do not imply four independent counts
    /// over four independent quantities: three losses are individually gated,
    /// and the fourth gate covers two reported values at once.
    ///
    #[must_use]
    pub const fn skipped_aux_total_updates(&self) -> u64 {
        self.aux_total_guard.skipped()
    }

    /// Total gradient updates skipped for a non-finite loss across **all four**
    /// of PPG's loss sites — the canonical `skipped_updates` metric (ADR 0072,
    /// Decisions 2 and 3).
    ///
    /// PPG has more loss sites than any other agent in the crate, so this
    /// aggregate — not any single per-site accessor — is what its progress
    /// event logs. It exists so that adding a fifth guarded site cannot silently drop
    /// out of the reported metric: the sum is written once, here, rather than
    /// re-derived at each call site.
    ///
    /// The sum is `saturating_add`ed rather than `+`ed: a counter that pins at `u64::MAX` is a better
    /// failure mode for a long-lived agent than one that wraps in release or
    /// panics under `overflow-checks`. Reaching the ceiling is not physically
    /// attainable here, so the saturation is a convention, not a live concern.
    ///
    /// # What one unit means
    ///
    /// One skipped **minibatch** at one site — not one skipped phase call. Policy-phase sites
    /// accumulate over `$\text{update\_epochs} \times \text{num\_minibatches}$` per
    /// `policy_phase_update`; auxiliary-phase sites over
    /// `$e_{aux} \times$` (auxiliary minibatches) per firing of
    /// `maybe_aux_phase`, which is itself on a `n_iteration` cadence.
    /// Because the four terms are measured against different denominators, the aggregate is a
    /// **health signal**, not a rate: read it as "this many optimizer steps were bought and not
    /// received", and use the per-site accessors to attribute them.
    ///
    /// # `$\text{applied} = \text{attempts} - \text{skipped}$`
    ///
    /// Every guard is consulted on an attempt that was already made — the
    /// forward pass and loss ran — so a skip is always a subset of an attempt
    /// and the subtraction can never underflow. The per-call complements are the
    /// `*_healthy` locals behind the policy phase's statistics and [`AuxPhaseStats`]: they
    /// denominate that one call's means, while this counter is the lifetime
    /// total. Complements, not duplicates.
    ///
    #[must_use]
    pub const fn skipped_updates(&self) -> u64 {
        self.policy_loss_guard
            .skipped()
            .saturating_add(self.value_loss_guard.skipped())
            .saturating_add(self.aux_main_value_guard.skipped())
            .saturating_add(self.aux_total_guard.skipped())
    }

    /// The active configuration.
    pub fn config(&self) -> &PpgConfig {
        &self.config
    }

    /// Statistics from the most recent auxiliary phase, or `None` if no
    /// auxiliary phase has run yet.
    pub fn last_aux_phase(&self) -> Option<AuxPhaseStats> {
        self.last_aux
    }

    /// Borrows the policy network.
    ///
    /// # Panics
    ///
    /// Panics if the policy slot was poisoned by a panic inside a previous
    /// optimizer step — see [`Slot::get`].
    fn policy(&self) -> &P {
        self.policy.get()
    }

    /// Borrows the value network.
    ///
    /// # Panics
    ///
    /// Panics if the value slot was poisoned by a panic inside a previous
    /// optimizer step — see [`Slot::get`].
    fn value(&self) -> &V {
        self.value.get()
    }

    /// Learning rate the **next** policy phase will use, accounting for linear
    /// annealing.
    ///
    /// Returns `config.ppo.learning_rate` unchanged when `anneal_lr` is
    /// `false`; otherwise linearly decays toward zero over `total_iterations`,
    /// reaching exactly `0.0` at `iteration == total_iterations`.
    ///
    /// This is a *forward-looking* read: it is a pure function of `iteration`,
    /// which `policy_phase_update` increments on
    /// its way out. Called after an update, it therefore reports one annealing
    /// tick *past* the rate that update applied — and at the end of a run it
    /// reports `0.0`. Anything that must step at the rate of the phase that just
    /// ran reads `policy_phase_lr` instead; that is exactly what the auxiliary
    /// phase does (see `maybe_aux_phase`).
    pub fn current_learning_rate(&self) -> f64 {
        if self.config.ppo.anneal_lr {
            annealed_learning_rate(
                self.config.ppo.learning_rate,
                self.iteration,
                self.total_iterations,
            )
        } else {
            self.config.ppo.learning_rate
        }
    }

    /// Sample one action for a single observation during a training rollout.
    ///
    /// Runs forward passes through both the policy (to obtain action, log-prob,
    /// and entropy) and the main value network (to obtain the baseline value
    /// estimate). All values are pushed into the rollout buffer via
    /// [`record_step`](Self::record_step).
    ///
    /// For evaluation use [`act_greedy`](Self::act_greedy) (no exploration
    /// noise) or [`act_greedy_env_row_with`](Self::act_greedy_env_row_with)
    /// (no autodiff graph overhead).
    ///
    /// # Panics
    ///
    /// Panics if the policy or value network was lost to a panic inside an
    /// earlier optimizer step. Such an agent is poisoned and must be rebuilt —
    /// see [`Slot`].
    fn sample(&self, obs: &O, rng: &mut (impl Rng + ?Sized)) -> ActOutcome {
        let obs_t: Tensor<OR> = obs.to_tensor(&self.device);
        let batched: Tensor<BOR> = obs_t.unsqueeze::<BOR>();

        let sample = self.policy().sample_with_logprob(batched.clone(), rng);
        let raw_row = P::action_row_from_tensor(&sample.action, 0);
        let env_row = self.policy().raw_to_env_row(&raw_row);

        let log_prob = sample.log_prob.into_scalar::<f32>();

        let value_raw: Tensor<OR> = obs.to_tensor(&self.device);
        let value_t: Tensor<BOR> = value_raw.unsqueeze::<BOR>();
        let value = self.value().forward(value_t).into_scalar::<f32>();

        ActOutcome {
            raw_row,
            env_row,
            log_prob,
            value,
        }
    }

    /// Logs one policy phase, and the auxiliary phase when it ran, at `info`
    /// under the metric registry's names. The two `log_std` extrema read
    /// `None` for every PPG run, PPG being discrete-only, and are logged so a
    /// Gaussian head is observable the day one lands (ADR 0049).
    fn log_update(&self, stats: &PpoUpdateStats, aux: Option<AuxPhaseStats>) {
        tracing::info!(
            step = self.step,
            env_steps_sampled = self.step,
            iteration = self.iteration,
            policy_loss = stats.policy_loss,
            value_loss = stats.value_loss,
            entropy = stats.entropy,
            approx_kl = stats.approx_kl,
            clip_frac = stats.clip_frac,
            epochs_run = stats.epochs_run,
            min_log_std = ?stats.min_log_std,
            max_log_std = ?stats.max_log_std,
            // Cumulative over all four PPG loss sites (ADR 0072), two of which
            // are in the auxiliary phase.
            skipped_updates = self.skipped_updates(),
            learning_rate = self.current_learning_rate(),
            aux_ran = aux.is_some(),
            aux_value_loss = aux.map_or(0.0, |a| a.aux_value_loss),
            aux_policy_kl = aux.map_or(0.0, |a| a.policy_kl),
            "ppg training progress"
        );
    }

    /// Greedy (deterministic) action for one observation — the categorical mode.
    ///
    /// Returns the env-space action row for the argmax over the policy logits,
    /// with no sampling. This is the policy to use for evaluation;
    /// [`act`](Self::act) samples from the categorical and so injects
    /// exploration noise that is only appropriate during training.
    ///
    /// PPG v1 is discrete-only (see
    /// [`ppg_policy`](crate::algorithms::ppg::ppg_policy)), so the "mode" is the
    /// highest-logit action; a future Gaussian head would instead return the
    /// distribution mean.
    ///
    /// # Panics
    ///
    /// Panics if the policy network was lost to a panic inside an earlier
    /// optimizer step. Such an agent is poisoned and must be rebuilt — see
    /// [`Slot`].
    #[allow(clippy::cast_precision_loss)]
    fn act_greedy(&self, obs: &O) -> Vec<f32> {
        let obs_t: Tensor<OR> = obs.to_tensor(&self.device);
        let batched: Tensor<BOR> = obs_t.unsqueeze::<BOR>();
        let logits: Tensor<2> = PpgAuxValueHead::logits(self.policy(), batched); // (1, A)
        let idx = logits.argmax(1).into_scalar::<i64>();
        let action = P::action_tensor_from_flat(&[idx as f32], 1, &self.device);
        let raw_row = P::action_row_from_tensor(&action, 0);
        self.policy().raw_to_env_row(&raw_row)
    }

    /// Pushes one step into the rollout buffer.
    ///
    /// `next_obs` is the transition's continuation observation, used only on a
    /// [`EpisodeStatus::Truncated`] step to compute the `V(s_continuation)`
    /// bootstrap for partial-episode bootstrapping (ADR 0048). It must be the
    /// **pre-reset** observation from the snapshot `env.step` returned. See
    /// [`PpoAgent::record_step`](crate::algorithms::ppo::ppo_agent::PpoAgent::record_step).
    ///
    /// # Panics
    ///
    /// Panics if `status` is [`EpisodeStatus::Truncated`] and the value network
    /// was lost to a panic inside an earlier optimizer step — the bootstrap
    /// forward reads it. Such an agent is poisoned and must be rebuilt — see
    /// [`Slot`].
    fn record_step(
        &mut self,
        obs: O,
        action: &ActOutcome,
        reward: f32,
        next_obs: &O,
        status: EpisodeStatus,
    ) {
        let end = match status {
            EpisodeStatus::Truncated => StepEnd::Truncated {
                bootstrap_value: self.value_of(next_obs),
            },
            EpisodeStatus::Terminated => StepEnd::Terminated,
            EpisodeStatus::Running => StepEnd::Running,
        };
        self.buffer.push_step(
            obs,
            &action.raw_row,
            action.log_prob,
            action.value,
            reward,
            end,
        );
        self.step += 1;
    }

    /// One main-value-network forward on a single observation.
    fn value_of(&self, obs: &O) -> f32 {
        let t: Tensor<OR> = obs.to_tensor(&self.device);
        let batched: Tensor<BOR> = t.unsqueeze::<BOR>();
        self.value().forward(batched).into_scalar::<f32>()
    }

    /// Computes GAE advantages/returns from `last_obs`. Identical to PPO's
    /// finalize step.
    ///
    /// As in
    /// [`PpoAgent::finalize_rollout`](crate::algorithms::ppo::ppo_agent::PpoAgent::finalize_rollout),
    /// `last_obs` is consulted only when the rollout's final step left the
    /// episode `Running`; otherwise the value forward is skipped.
    ///
    /// # Panics
    ///
    /// Panics if the rollout's final step left the episode `Running` and the
    /// value network was lost to a panic inside an earlier optimizer step —
    /// the bootstrap forward reads it. Such an agent is poisoned and must be
    /// rebuilt — see [`Slot`].
    fn finalize_rollout(&mut self, last_obs: &O) {
        let last_value = if self.buffer.last_step_ended() {
            0.0
        } else {
            self.value_of(last_obs)
        };
        self.buffer.finish(
            last_value,
            self.config.ppo.gamma,
            self.config.ppo.gae_lambda,
        );
    }

    /// Snapshots the current finished rollout's observations and target
    /// returns into the auxiliary buffer.
    ///
    /// Must be called **after** [`Self::finalize_rollout`] (so returns are
    /// populated) and **before** [`Self::policy_phase_update`] (which will
    /// clear the rollout). The `$\pi_{old}$` logits for the distillation KL are
    /// **not** captured here — they are computed fresh at the start of
    /// [`Self::maybe_aux_phase`] against the post-policy-phase policy, so
    /// the distillation preserves (rather than undoes) the policy-phase's
    /// learning.
    fn snapshot_into_aux_buffer(&mut self) {
        if self.buffer.is_empty() {
            return;
        }
        let obs_vec: Vec<O> = self.buffer.obs().to_vec();
        let returns_vec: Vec<f32> = self.buffer.returns().to_vec();
        self.aux_buffer.push_slice(obs_vec, returns_vec);
    }

    /// PPO policy-phase update.
    ///
    /// Structurally mirrors
    /// [`crate::algorithms::ppo::ppo_agent::PpoAgent::update`].
    /// Clears the rollout buffer, increments the iteration counter, returns
    /// the same [`PpoUpdateStats`].
    ///
    /// The annealed learning rate is snapshotted once, *before* the increment,
    /// and retained in `policy_phase_lr` so the auxiliary phase that may follow
    /// this update steps at the same rate.
    ///
    /// # Panics
    ///
    /// Panics if the policy or value network was lost to a panic inside an
    /// earlier optimizer step. Such an agent is poisoned and must be rebuilt —
    /// see [`Slot`]. The forward passes, the losses, and `backward` all run
    /// against a borrow, so a panic in any of them leaves the agent usable;
    /// only a panic inside an optimizer step poisons a network.
    // The body is one linear pipeline — sample, forward, loss, backward,
    // optimizer step, priority writeback, metrics — with a borrow structure
    // around the module slot that the inline comments below depend on. Splitting
    // it into helpers to satisfy the line count would thread that borrow through
    // signatures without making the sequence easier to follow.
    #[allow(clippy::too_many_lines)]
    // Divisor/normalizer derived from a count -- batch size, minibatch count,
    // history length, iteration number. All are bounded by configured sizes far
    // below f32's 2^24 (f64's 2^53) exact-integer limit.
    #[allow(clippy::cast_precision_loss)]
    fn policy_phase_update(&mut self, rng: &mut (impl Rng + ?Sized)) -> PpoUpdateStats {
        let cfg = self.config.ppo.clone();
        let batch_size = self.buffer.len();
        let mb_size = (batch_size / cfg.num_minibatches.max(1)).max(1);
        // Read before `self.iteration += 1` below, so this is the rate for the
        // phase about to run. Retained for the auxiliary phase, which belongs to
        // this policy phase and must not anneal ahead of it.
        let lr = self.current_learning_rate();
        self.policy_phase_lr = lr;

        let mut policy_loss_acc = 0.0_f32;
        let mut value_loss_acc = 0.0_f32;
        let mut entropy_acc = 0.0_f32;
        let mut clip_frac_acc = 0.0_f32;
        let mut last_kl = 0.0_f32;
        let mut mb_count = 0_usize;
        let mut epochs_run = 0_usize;
        // Denominators for the two gated means: only minibatches whose loss was
        // finite and applied (ADR 0056, Decision 3). `mb_count` still denominates
        // the ungated diagnostics.
        let mut policy_healthy = 0_usize;
        let mut value_healthy = 0_usize;

        let obs_shape = O::shape();
        let numel_per_obs: usize = obs_shape.iter().product();

        'epochs: for _epoch in 0..cfg.update_epochs {
            epochs_run += 1;
            let indices = self.buffer.indices_shuffled(rng);
            let mut kl_sum = 0.0_f32;
            let mut kl_count = 0_usize;

            for chunk in indices.chunks(mb_size) {
                if chunk.is_empty() {
                    continue;
                }
                let n = chunk.len();

                let mut obs_flat: Vec<f32> = Vec::with_capacity(n * numel_per_obs);
                for &i in chunk {
                    self.buffer.obs()[i].write_host_row(&mut obs_flat);
                }
                let mut batched_shape: Vec<usize> = Vec::with_capacity(BOR);
                batched_shape.push(n);
                batched_shape.extend_from_slice(&obs_shape);
                let obs_batch: Tensor<BOR> =
                    Tensor::from_data(TensorData::new(obs_flat, batched_shape), &self.device);

                let action_flat = self.buffer.gather_action_flat(chunk);
                let actions: P::ActionTensor =
                    P::action_tensor_from_flat(&action_flat, n, &self.device);

                let old_lp: Tensor<1> =
                    self.buffer
                        .gather_f32(self.buffer.log_probs(), chunk, &self.device);
                let old_v: Tensor<1> =
                    self.buffer
                        .gather_f32(self.buffer.values(), chunk, &self.device);
                let returns: Tensor<1> =
                    self.buffer
                        .gather_f32(self.buffer.returns(), chunk, &self.device);
                let advs_raw: Tensor<1> =
                    self.buffer
                        .gather_f32(self.buffer.advantages(), chunk, &self.device);
                let advs = if cfg.normalize_advantages {
                    normalize_advantages(advs_raw)
                } else {
                    advs_raw
                };

                // Policy update. Everything up to `step_with` runs on a borrow,
                // so a panic in the forward/loss/backward region leaves the
                // slot intact.
                let eval = self.policy().evaluate(obs_batch.clone(), actions);
                let pg =
                    clipped_surrogate(eval.log_prob.clone(), old_lp.clone(), advs, cfg.clip_coef);
                let entropy_mean = eval.entropy.mean();
                let policy_loss = pg.clone() - entropy_mean.clone().mul_scalar(cfg.entropy_coef);
                let policy_loss_val = policy_loss.clone().into_scalar::<f32>();

                let kl = approx_kl(eval.log_prob.clone(), old_lp.clone());
                let cf = clip_fraction(eval.log_prob.clone(), old_lp, cfg.clip_coef);
                kl_sum += kl;
                kl_count += 1;

                // ADR 0056: `policy_loss_val` is already host-resident,
                // so the finiteness check costs no extra sync. A non-finite
                // loss skips `backward()` + the optimizer step and is excluded
                // from the reported mean.
                if self.policy_loss_guard.check(policy_loss_val) {
                    let grads = policy_loss.backward();
                    let grads_params = GradientsParams::from_grads(grads, self.policy());
                    self.policy
                        .step_with(&mut self.policy_optim, lr, grads_params);
                    policy_loss_acc += policy_loss_val;
                    policy_healthy += 1;
                }

                // Value update.
                let new_v = self.value().forward(obs_batch);
                let v_loss = if cfg.clip_value_loss {
                    clipped_value_loss(new_v, old_v, returns, cfg.clip_coef)
                } else {
                    unclipped_value_loss(new_v, returns)
                };
                let v_loss_scaled = v_loss.clone().mul_scalar(cfg.value_coef);
                let v_loss_val = v_loss.into_scalar::<f32>();
                // ADR 0056: same guard type for the value site, counted
                // separately from the policy site — its own per-site total
                // (`skipped_value_updates`) and its own decade `warn!` schedule,
                // and, like the policy site, its skip fires on every non-finite
                // minibatch rather than only the first (ADR 0056, Decision 3;
                // ADR 0072).
                if self.value_loss_guard.check(v_loss_val) {
                    let grads = v_loss_scaled.backward();
                    let grads_params = GradientsParams::from_grads(grads, self.value());
                    self.value
                        .step_with(&mut self.value_optim, lr, grads_params);
                    value_loss_acc += v_loss_val;
                    value_healthy += 1;
                }

                entropy_acc += entropy_mean.into_scalar::<f32>();
                clip_frac_acc += cf;
                mb_count += 1;
            }

            let mean_kl = if kl_count == 0 {
                0.0
            } else {
                kl_sum / kl_count as f32
            };
            last_kl = mean_kl;

            if let Some(target) = cfg.target_kl
                && mean_kl > 1.5 * target
            {
                break 'epochs;
            }
        }

        // Wired only because PPG reports through PPO's `PpoUpdateStats`; PPG has
        // no continuous head in v1, so both of these are always `None`. A future
        // Gaussian PPG head would reuse `TanhGaussianPolicyHead` and would then
        // inherit its log_std bound and its one-shot-per-bound clamp warnings
        // here — the ceiling half included, which is why `max_log_std` is wired
        // now alongside the minimum rather than left for that head to add
        // later (ADR 0049, Decision 4).
        let min_log_std = self.policy().min_log_std();
        let max_log_std = self.policy().max_log_std();

        self.buffer.clear();
        self.iteration += 1;

        // The two gated losses divide by their own healthy counts (`0.0`, not
        // NaN, when every minibatch skipped); the ungated diagnostics keep the
        // total minibatch denominator (ADR 0056, Decision 3).
        let denom = mb_count.max(1) as f32;
        PpoUpdateStats {
            policy_loss: if policy_healthy == 0 {
                0.0
            } else {
                policy_loss_acc / policy_healthy as f32
            },
            value_loss: if value_healthy == 0 {
                0.0
            } else {
                value_loss_acc / value_healthy as f32
            },
            entropy: entropy_acc / denom,
            approx_kl: last_kl,
            // PPG does not yet surface the v6 PPO diagnostics; left at their
            // neutral defaults so the shared struct stays consistent.
            old_approx_kl: 0.0,
            clip_frac: clip_frac_acc / denom,
            explained_variance: 0.0,
            epochs_run,
            min_log_std,
            max_log_std,
        }
    }

    /// Runs the auxiliary phase if the auxiliary buffer is full.
    ///
    /// When `aux_buffer.num_slices() >= n_iteration`, iterates `e_aux` epochs
    /// of auxiliary updates:
    ///
    /// - main-value MSE against target returns,
    /// - aux-value-head MSE against target returns,
    /// - `$\beta \cdot KL(\pi_{old} \Vert \pi_{new})$` distillation pulling the policy back to the
    ///   snapshot taken at the *start* of the auxiliary phase (i.e. the
    ///   policy that has just finished `n_iteration` policy-phase updates).
    ///
    /// Drains the buffer afterwards. Otherwise, no-op.
    ///
    /// # Learning rate
    ///
    /// Both optimizer steps use `policy_phase_lr` — the rate the policy phase
    /// this auxiliary phase accompanies actually applied — **not**
    /// [`current_learning_rate`](Self::current_learning_rate), which has already
    /// advanced past it. This matches `CleanRL`'s `ppg_procgen.py`, where the
    /// `# AUXILIARY PHASE` block is nested inside the phase body, shares the
    /// policy optimizer, and never rewrites its `lr`; and Algorithm 1 of Cobbe
    /// et al. (2021), where the auxiliary phase sits inside the phase rather
    /// than after a schedule tick. Reading the annealed rate here instead made
    /// every auxiliary phase run one tick early and, whenever
    /// `total_iterations % n_iteration == 0`, made the final one a bit-exact
    /// no-op at `lr == 0.0`.
    ///
    /// # Panics
    ///
    /// Panics if the auxiliary phase runs and the policy or value network was
    /// lost to a panic inside an earlier optimizer step. Such an agent is
    /// poisoned and must be rebuilt — see [`Slot`]. A call that returns `None`
    /// without running the phase reads neither network.
    // Divisor/normalizer derived from a count -- batch size, minibatch count,
    // history length, iteration number. All are bounded by configured sizes far
    // below f32's 2^24 (f64's 2^53) exact-integer limit.
    #[allow(clippy::cast_precision_loss)]
    fn maybe_aux_phase(&mut self, rng: &mut (impl Rng + ?Sized)) -> Option<AuxPhaseStats> {
        if !self.aux_buffer.is_ready(self.config.n_iteration) {
            return None;
        }
        let cfg = self.config.clone();
        // The accompanying policy phase's rate, not the next one's.
        let lr = self.policy_phase_lr;
        let total_steps = self.aux_buffer.len_steps();
        if total_steps == 0 {
            self.aux_buffer.clear();
            return None;
        }

        // Snapshot `$\pi_{old}$` once, against the current (post-policy-phase) policy. Stored flat
        // row-major so minibatch slicing is a trivial copy.
        let old_logits_flat = self.compute_old_logits_flat(total_steps);
        let num_actions = old_logits_flat.len() / total_steps;
        debug_assert_eq!(old_logits_flat.len(), total_steps * num_actions);

        let mut main_v_acc = 0.0_f32;
        let mut aux_v_acc = 0.0_f32;
        let mut kl_acc = 0.0_f32;
        let mut mb_count = 0_usize;
        let mut epochs_run = 0_usize;
        // Healthy-minibatch denominators for the two gated aux losses
        // (ADR 0056, Decision 3). The main-value site feeds `main_v_acc`; the combined
        // aux site feeds both `aux_v_acc` and `kl_acc`, so they share a count.
        let mut main_v_healthy = 0_usize;
        let mut aux_total_healthy = 0_usize;

        for _epoch in 0..cfg.e_aux {
            epochs_run += 1;
            let indices = self.aux_buffer.indices_shuffled(rng);
            let mb_size = cfg.aux_batch_size.max(1);

            for chunk in indices.chunks(mb_size) {
                if chunk.is_empty() {
                    continue;
                }
                let (obs_t, returns_t) = self
                    .aux_buffer
                    .gather_minibatch::<OR, BOR>(chunk, &self.device);

                let mut old_logits_mb: Vec<f32> = Vec::with_capacity(chunk.len() * num_actions);
                for &global in chunk {
                    let start = global * num_actions;
                    let end = start + num_actions;
                    old_logits_mb.extend_from_slice(&old_logits_flat[start..end]);
                }
                let old_logits_t: Tensor<2> = Tensor::from_data(
                    TensorData::new(old_logits_mb, vec![chunk.len(), num_actions]),
                    &self.device,
                );

                // Main value-net update. As in the policy phase, all fallible
                // work runs on a borrow ahead of `step_with`.
                let new_v = self.value().forward(obs_t.clone());
                let v_loss = unclipped_value_loss(new_v, returns_t.clone());
                let v_loss_val = v_loss.clone().into_scalar::<f32>();
                // ADR 0056: skip backward + step on a non-finite
                // main-value loss; exclude it from the reported mean.
                if self.aux_main_value_guard.check(v_loss_val) {
                    let grads = v_loss.backward();
                    let grads_params = GradientsParams::from_grads(grads, self.value());
                    self.value
                        .step_with(&mut self.value_optim, lr, grads_params);
                    main_v_acc += v_loss_val;
                    main_v_healthy += 1;
                }

                // Policy-net update: aux-value MSE + `$\beta$` `$\cdot$` KL distillation.
                let aux_v_pred = PpgAuxValueHead::aux_value(self.policy(), obs_t.clone());
                let aux_v_loss = unclipped_value_loss(aux_v_pred, returns_t);
                let new_logits = PpgAuxValueHead::logits(self.policy(), obs_t);
                let kl = policy_kl_categorical(old_logits_t, new_logits);
                let aux_v_loss_val = aux_v_loss.clone().into_scalar::<f32>();
                let kl_val = kl.clone().into_scalar::<f32>();
                let total = aux_v_loss + kl.mul_scalar(cfg.beta_clone);
                // ADR 0056: `$\text{total} = \text{aux\_v\_loss} + \beta \cdot \text{kl}$` is never
                // read host-side, so guard on the host-*derived* scalar built from its two
                // already-read summands — no extra device→host sync. Both accumulators this site
                // feeds are excluded on a skip.
                let total_val = aux_v_loss_val + kl_val * cfg.beta_clone;
                if self.aux_total_guard.check(total_val) {
                    let grads = total.backward();
                    let grads_params = GradientsParams::from_grads(grads, self.policy());
                    self.policy
                        .step_with(&mut self.policy_optim, lr, grads_params);
                    aux_v_acc += aux_v_loss_val;
                    kl_acc += kl_val;
                    aux_total_healthy += 1;
                }

                mb_count += 1;
            }
        }

        self.aux_buffer.clear();

        // Each gated loss divides by its own healthy count (`0.0`, not NaN,
        // when every minibatch skipped); `minibatches` still reports the total
        // count attempted (ADR 0056, Decision 3).
        let stats = AuxPhaseStats {
            main_value_loss: if main_v_healthy == 0 {
                0.0
            } else {
                main_v_acc / main_v_healthy as f32
            },
            // `aux_value_loss` and `policy_kl` intentionally share the `aux_total_healthy`
            // denominator: both are fed by the single aux-total guard site
            // (`$\text{aux\_v\_loss} + \beta\,\text{kl}$`), so they are applied and skipped
            // together — this is not a copy-pasted divisor.
            aux_value_loss: if aux_total_healthy == 0 {
                0.0
            } else {
                aux_v_acc / aux_total_healthy as f32
            },
            policy_kl: if aux_total_healthy == 0 {
                0.0
            } else {
                kl_acc / aux_total_healthy as f32
            },
            epochs_run,
            minibatches: mb_count,
            learning_rate: lr,
        };
        self.last_aux = Some(stats);
        Some(stats)
    }

    /// Computes `$\pi_{old}$` logits for every step in the auxiliary buffer.
    ///
    /// Returns a row-major flat `Vec<f32>` of length
    /// `total_steps * num_actions`. Batched through the policy in chunks to
    /// keep autodiff-graph memory bounded; every forward pass is immediately
    /// converted to CPU data, so the graph is discarded between chunks.
    fn compute_old_logits_flat(&self, total_steps: usize) -> Vec<f32> {
        const CHUNK: usize = 1024;
        let obs_shape = O::shape();
        let numel_per_obs: usize = obs_shape.iter().product();
        let mut out: Vec<f32> = Vec::new();

        let mut start = 0_usize;
        while start < total_steps {
            let end = (start + CHUNK).min(total_steps);
            let n = end - start;
            let mut obs_flat: Vec<f32> = Vec::with_capacity(n * numel_per_obs);
            for g in start..end {
                self.aux_buffer.obs_at(g).write_host_row(&mut obs_flat);
            }
            let mut batched_shape: Vec<usize> = Vec::with_capacity(BOR);
            batched_shape.push(n);
            batched_shape.extend_from_slice(&obs_shape);
            let obs_batch: Tensor<BOR> =
                Tensor::from_data(TensorData::new(obs_flat, batched_shape), &self.device);
            let logits = PpgAuxValueHead::logits(self.policy(), obs_batch);
            let logits_data = logits.into_data().convert::<f32>();
            out.extend_from_slice(logits_data.as_slice::<f32>().expect("f32 logits"));
            start = end;
        }
        out
    }

    /// Truncates the rollout's last step when the driver ends its episode
    /// without the environment ending it: at the driver's `max_steps` cut, or
    /// when it switches phase. That step was recorded `Running`, so GAE would
    /// otherwise bootstrap from, and carry the advantage recursion into, the
    /// next step recorded, the first of a new episode after its reset. It
    /// bootstraps from its own next observation instead (ADR 0048).
    fn close_open_step(&mut self) {
        if let Some(next_obs) = self.continuation.take() {
            let bootstrap_value = self.value_of(&next_obs);
            self.buffer.truncate_last_step(bootstrap_value);
        }
    }
}

/// Driven through [`Agent`], the agent samples from its policy while
/// training and acts on the most probable action otherwise.
///
/// Each training step [`on_transition`](Agent::on_transition) records the
/// step with the sample [`act`](Agent::act) drew. When the rollout holds
/// `num_steps` steps it finishes the rollout on the step's `next_obs`, keeps
/// it for the auxiliary phase, runs the policy phase, and runs the auxiliary
/// phase when its cadence is due. A rollout the driver leaves unfinished is
/// not used. While evaluating it does nothing.
///
/// A step after which the driver ends the episode, or switches phase,
/// without the environment ending it becomes a truncation that bootstraps
/// from its own next observation, so the advantages never run across the
/// driver's reset into the next episode.
///
/// # Panics
///
/// `on_transition` panics, while training, when no [`act`](Agent::act)
/// preceded it: there is no sample to record.
impl<V, O, A, const OR: usize> Agent<O, A>
    for PpgAgent<crate::algorithms::ppg::policies::PpgCategoricalPolicyHead, V, O, OR, 2>
where
    V: PpoValue<2>,
    O: Observation<OR> + TensorConvertible<OR>,
    A: DiscreteAction<1>,
{
    fn act(&mut self, obs: &O, rng: &mut dyn Rng) -> A {
        if self.training {
            let sample = self.sample(obs, rng);
            let action = crate::algorithms::ppo::policies::categorical::discrete_action_from_row::<
                1,
                A,
            >(&sample.env_row);
            self.pending = Some(sample);
            action
        } else {
            crate::algorithms::ppo::policies::categorical::discrete_action_from_row::<1, A>(
                &self.act_greedy(obs),
            )
        }
    }

    fn on_transition(&mut self, t: Transition<&O, &A>, rng: &mut dyn Rng) {
        if !self.training {
            return;
        }
        let sample = self
            .pending
            .take()
            .expect("a training step's act precedes its on_transition");
        self.record_step(t.obs.clone(), &sample, t.reward, t.next_obs, t.status);
        if self.buffer.len() == self.config.ppo.num_steps {
            self.finalize_rollout(t.next_obs);
            self.snapshot_into_aux_buffer();
            let stats = self.policy_phase_update(rng);
            let aux = self.maybe_aux_phase(rng);
            self.log_update(&stats, aux);
            self.continuation = None;
        } else {
            self.continuation = (!t.status.is_done()).then(|| t.next_obs.clone());
        }
    }

    fn on_episode_end(&mut self, _status: EpisodeStatus) {
        self.close_open_step();
    }

    fn set_training(&mut self, training: bool) {
        self.close_open_step();
        self.training = training;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::grad_clipping::GradientClippingConfig;
    use burn::module::{Module, ModuleMapper, ModuleVisitor, Param};
    use burn::nn::{Linear, LinearConfig};
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rlevo_core::base::HostRow;
    use rlevo_core::config::ConstraintKind;
    use serde::{Deserialize, Serialize};

    use crate::algorithms::ppg::policies::{
        PpgCategoricalPolicyHead, PpgCategoricalPolicyHeadConfig,
    };
    use crate::algorithms::ppo::ppo_config::PpoTrainingConfig;

    type TestAgent = PpgAgent<PpgCategoricalPolicyHead, TestValue, TestObs, 1, 2>;

    /// Minimal scalar main-value head, one linear layer. Not the crate's
    /// `MlpValue`: the tests below read its `head` field.
    #[derive(Module, Debug)]
    struct TestValue {
        head: Linear,
    }

    impl TestValue {
        fn init(device: &Device) -> Self {
            Self {
                head: LinearConfig::new(2, 1).init(device),
            }
        }
    }

    impl PpoValue<2> for TestValue {
        fn forward(&self, obs: Tensor<2>) -> Tensor<1> {
            self.head.forward(obs).squeeze_dim(1)
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
        fn write_host_row(&self, buf: &mut Vec<f32>) {
            buf.extend_from_slice(&self.0);
        }
    }

    /// Builds an agent whose `config.ppo.clip_grad` is exactly `clip_grad`,
    /// leaving every other knob at its default.
    fn agent_with_clip_grad(clip_grad: Option<GradientClippingConfig>) -> TestAgent {
        let device = Device::flex();
        let policy: PpgCategoricalPolicyHead = PpgCategoricalPolicyHeadConfig {
            obs_dim: 2,
            hidden: 4,
            num_actions: 2,
        }
        .try_init(&device)
        .expect("valid head config");
        let value = TestValue::init(&device);
        let config = PpgConfig {
            ppo: PpoTrainingConfig {
                clip_grad,
                ..Default::default()
            },
            ..Default::default()
        };
        TestAgent::new(policy, value, config, device, 1).expect("valid config")
    }

    /// An invalid config comes back as an `Err` from `new`, naming the config
    /// and field that rejected it: two of PPG's own fields, and one of the PPO
    /// config it wraps. `config.validate()?` is the only check a config passes
    /// on its way into the agent, so dropping its error must fail here.
    #[test]
    fn test_ppg_agent_new_rejects_invalid_config() {
        let cases = [
            (
                PpgConfig {
                    e_aux: 0,
                    ..Default::default()
                },
                ("PpgConfig", "e_aux", ConstraintKind::Zero),
            ),
            (
                PpgConfig {
                    aux_batch_size: 0,
                    ..Default::default()
                },
                ("PpgConfig", "aux_batch_size", ConstraintKind::Zero),
            ),
            (
                PpgConfig {
                    ppo: PpoTrainingConfig {
                        num_envs: 2,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                (
                    "PpoTrainingConfig",
                    "num_envs",
                    ConstraintKind::Custom("v1 supports num_envs == 1 only"),
                ),
            ),
        ];

        for (config, (name, field, kind)) in cases {
            let device = Device::flex();
            let policy: PpgCategoricalPolicyHead = PpgCategoricalPolicyHeadConfig {
                obs_dim: 2,
                hidden: 4,
                num_actions: 2,
            }
            .try_init(&device)
            .expect("valid head config");
            let value = TestValue::init(&device);
            let Err(err) = TestAgent::new(policy, value, config, device, 1) else {
                panic!("{name}.{field} must be rejected by PpgAgent::new");
            };
            assert_eq!(err.config, name);
            assert_eq!(err.field, field);
            assert_eq!(err.kind, kind);
        }
    }

    /// Regression: PPG builds its optimizers from
    /// `config.ppo.clip_grad` itself rather than delegating to `PpoAgent`, so
    /// this wiring can rot independently of the PPO copy — assert it here too.
    #[test]
    fn test_ppg_agent_clip_grad_none_leaves_both_optimizers_unclipped() {
        let agent = agent_with_clip_grad(None);
        assert!(
            !agent.policy_optim.has_gradient_clipping(),
            "clip_grad: None must leave the policy optimizer unclipped"
        );
        assert!(
            !agent.value_optim.has_gradient_clipping(),
            "clip_grad: None must leave the value optimizer unclipped"
        );
    }

    #[test]
    fn test_ppg_agent_clip_grad_some_reaches_both_optimizers() {
        let agent = agent_with_clip_grad(Some(GradientClippingConfig::Norm(0.5)));
        assert!(
            agent.policy_optim.has_gradient_clipping(),
            "clip_grad: Some(..) must configure the policy optimizer"
        );
        assert!(
            agent.value_optim.has_gradient_clipping(),
            "clip_grad: Some(..) must configure the value optimizer"
        );
    }

    // -------- non-finite-loss guard (ADR 0056) --------

    /// Replaces every float parameter of a module with `NaN`, simulating a
    /// network that has diverged to non-finite weights — the realistic source
    /// of a non-finite loss. Applied to a *clone* so the live net's `ParamId`s
    /// are preserved.
    struct NanInjector;

    impl ModuleMapper for NanInjector {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
            let (id, tensor, mapper) = param.consume();
            Param::from_mapped_value(id, tensor.mul_scalar(f32::NAN), mapper)
        }
    }

    /// Flattens every float parameter of a module into one host `Vec<f32>`, in
    /// visit order, so two snapshots taken around a phase can be diffed
    /// element-wise.
    ///
    /// Visits rather than naming fields: the policy heads keep their `Linear`
    /// layers private, and a whole-module sweep also cannot miss a parameter a
    /// future head adds.
    struct ParamSnapshot {
        values: Vec<f32>,
    }

    impl ModuleVisitor for ParamSnapshot {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            self.values.extend(
                param
                    .val()
                    .into_data()
                    .convert::<f32>()
                    .try_into_vec::<f32>()
                    .expect("f32 parameter host read"),
            );
        }
    }

    fn param_snapshot<M: Module>(module: &M) -> Vec<f32> {
        let mut snapshot = ParamSnapshot { values: Vec::new() };
        module.visit(&mut snapshot);
        snapshot.values
    }

    /// Reads the main value head's weights back to the host, element-wise, so a
    /// change across a phase proves the value net actually stepped.
    fn value_weights(v: &TestValue) -> Vec<f32> {
        v.head
            .weight
            .val()
            .into_data()
            .convert::<f32>()
            .try_into_vec::<f32>()
            .expect("f32 weight host read")
    }

    /// Collects and finalizes one four-step rollout on `agent`, leaving the
    /// buffer full and its advantages/returns populated — i.e. ready for
    /// `policy_phase_update`.
    // Test fixture data: the loop counter is bounded by a small constant, far
    // below f32's 2^24 exact-integer limit.
    #[allow(clippy::cast_precision_loss)]
    fn collect_rollout(agent: &mut TestAgent, rng: &mut StdRng) {
        let mut last = TestObs([0.4, 0.6]);
        for i in 0..4usize {
            let x = i as f32 * 0.1;
            let obs = TestObs([x, 1.0 - x]);
            let next = TestObs([x + 0.1, 0.9 - x]);
            let outcome = agent.sample(&obs, rng);
            agent.record_step(obs, &outcome, 1.0, &next, EpisodeStatus::Running);
            last = next;
        }
        agent.finalize_rollout(&last);
    }

    /// Builds an agent with a single four-step rollout collected and finalized
    /// (advantages/returns populated) with *healthy* networks. `n_iteration = 1`
    /// so a single `snapshot_into_aux_buffer` makes the auxiliary phase ready.
    fn primed_ppg_agent() -> TestAgent {
        let device = Device::flex();
        let policy: PpgCategoricalPolicyHead = PpgCategoricalPolicyHeadConfig {
            obs_dim: 2,
            hidden: 4,
            num_actions: 2,
        }
        .try_init(&device)
        .expect("valid head config");
        let value = TestValue::init(&device);
        let config = PpgConfig {
            n_iteration: 1,
            e_aux: 1,
            aux_batch_size: 4,
            ppo: PpoTrainingConfig {
                num_envs: 1,
                num_steps: 4,
                num_minibatches: 1,
                update_epochs: 1,
                anneal_lr: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut agent = TestAgent::new(policy, value, config, device, 1).expect("valid config");

        let mut rng = StdRng::seed_from_u64(0);
        collect_rollout(&mut agent, &mut rng);
        agent
    }

    /// Policy-phase site: a non-finite policy loss must skip the policy
    /// `backward` + optimizer step and be excluded from the reported mean (ADR
    /// 0056). Diverging the policy net to NaN forces a NaN clipped
    /// surrogate; the guard must fire, the reported `policy_loss` must be the
    /// `0.0` all-skipped sentinel, and the value site must still learn.
    #[test]
    #[allow(clippy::float_cmp)]
    fn test_ppg_agent_policy_phase_nonfinite_policy_loss_skips_and_warns() {
        let mut agent = primed_ppg_agent();
        let value_before = value_weights(agent.value.get());

        let poisoned = agent.policy.get().clone().map(&mut NanInjector);
        agent.policy = Slot::new(poisoned);

        let mut rng = StdRng::seed_from_u64(1);
        let stats = agent.policy_phase_update(&mut rng);

        // `primed_ppg_agent`: num_steps = 4, num_minibatches = 1 → mb_size = 4, so 1 minibatch per
        // epoch `$\times$` update_epochs = 1 → 1 guarded attempt, all of it skipped.
        assert_eq!(
            agent.skipped_policy_updates(),
            1,
            "the single NaN policy minibatch (1 minibatch × 1 epoch) must be \
             counted exactly once at the ppg/policy_loss site"
        );
        assert_eq!(
            agent.skipped_value_updates(),
            0,
            "the value net was healthy: the sibling site must be untouched, \
             pinned at exactly 0 rather than merely 'not yet warned'"
        );
        assert_eq!(
            agent.skipped_updates(),
            1,
            "the aggregate over all four sites must equal the one policy-site \
             skip: 1 + 0 + 0 + 0"
        );
        assert_eq!(
            stats.policy_loss, 0.0,
            "every policy minibatch skipped → reported 0.0, never a propagated NaN"
        );

        let value_after = value_weights(agent.value.get());
        assert!(
            value_before
                .iter()
                .zip(&value_after)
                .any(|(a, b)| (a - b).abs() > 1e-9),
            "the value net must still learn while the policy step is skipped"
        );
        assert!(
            value_after.iter().all(|w| w.is_finite()),
            "the value weights must stay finite"
        );
        assert!(
            stats.value_loss.is_finite(),
            "the reported value loss must stay finite"
        );
    }

    /// Auxiliary-phase site: the aux-total guard fires on the host-*derived*
    /// `$\text{aux\_v\_loss\_val} + \text{kl\_val} \cdot \beta$` input (ADR 0056, Decision 5).
    /// Diverging the policy net makes both the aux-value prediction and the
    /// distillation logits NaN, so the combined loss is non-finite; the guard
    /// must fire and skip only the policy step, while the independent
    /// main-value site still learns. Both `aux_value_loss` and `policy_kl`
    /// report `0.0` (they share the skipped aux-total site's healthy
    /// denominator).
    #[test]
    #[allow(clippy::float_cmp)]
    fn test_ppg_agent_aux_phase_nonfinite_total_loss_skips_and_warns() {
        let mut agent = primed_ppg_agent();
        // Snapshot the finalized rollout; `n_iteration = 1` arms the aux phase.
        agent.snapshot_into_aux_buffer();
        let value_before = value_weights(agent.value.get());

        // Poison the policy: the aux-total site (`$\text{aux\_value} + \beta\,\text{KL}$`) runs
        // through the policy, so its guard input goes NaN; the main-value site (value net) stays
        // finite.
        let poisoned = agent.policy.get().clone().map(&mut NanInjector);
        agent.policy = Slot::new(poisoned);

        let mut rng = StdRng::seed_from_u64(3);
        let stats = agent
            .maybe_aux_phase(&mut rng)
            .expect("n_iteration = 1 makes the aux phase ready after one snapshot");

        // `primed_ppg_agent`: one 4-step slice in the aux buffer, aux_batch_size = 4 → 1 minibatch
        // per epoch `$\times$` e_aux = 1 → 1 guarded attempt at each aux site.
        assert_eq!(
            agent.skipped_aux_total_updates(),
            1,
            "the single NaN aux-total minibatch (1 minibatch × 1 aux epoch) \
             must be counted exactly once at the ppg/aux_total_loss site"
        );
        assert_eq!(
            agent.skipped_aux_value_updates(),
            0,
            "the main-value site ran on the healthy value net: pinned at \
             exactly 0, not merely 'not yet warned'"
        );
        assert_eq!(
            agent.skipped_updates(),
            1,
            "the aggregate must equal the one aux-total skip: 0 + 0 + 0 + 1"
        );
        assert_eq!(
            stats.aux_value_loss, 0.0,
            "the skipped aux-total site must report 0.0 aux value loss"
        );
        assert_eq!(
            stats.policy_kl, 0.0,
            "the skipped aux-total site must report 0.0 policy KL (shared denominator)"
        );

        let value_after = value_weights(agent.value.get());
        assert!(
            value_before
                .iter()
                .zip(&value_after)
                .any(|(a, b)| (a - b).abs() > 1e-9),
            "the main value net must still learn while the aux-total site is skipped"
        );
        assert!(
            stats.main_value_loss.is_finite(),
            "the reported main-value loss must stay finite"
        );
    }

    /// Like [`primed_ppg_agent`], but with the minibatch/epoch shape under the
    /// caller's control so a test can predict the exact number of guarded
    /// attempts a phase makes.
    ///
    /// The rollout is still four steps (`collect_rollout`), so the policy phase runs
    /// `$\text{update\_epochs} \times (4/\text{num\_minibatches}\ \lceil\text{chunks}\rceil)$`
    /// minibatches and the auxiliary phase
    /// `$e_{aux} = 1 \times \lceil \text{steps}/\text{aux\_batch\_size}\rceil$`. `target_kl` is
    /// pinned to `None` so no epoch can end early and change that arithmetic.
    fn shaped_ppg_agent(
        num_minibatches: usize,
        update_epochs: usize,
        aux_batch_size: usize,
    ) -> TestAgent {
        let device = Device::flex();
        let policy: PpgCategoricalPolicyHead = PpgCategoricalPolicyHeadConfig {
            obs_dim: 2,
            hidden: 4,
            num_actions: 2,
        }
        .try_init(&device)
        .expect("valid head config");
        let value = TestValue::init(&device);
        let config = PpgConfig {
            n_iteration: 1,
            e_aux: 1,
            aux_batch_size,
            ppo: PpoTrainingConfig {
                num_envs: 1,
                num_steps: 4,
                num_minibatches,
                update_epochs,
                anneal_lr: false,
                target_kl: None,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut agent = TestAgent::new(policy, value, config, device, 1).expect("valid config");

        let mut rng = StdRng::seed_from_u64(0);
        collect_rollout(&mut agent, &mut rng);
        agent
    }

    /// The skip counter is a *counter*, not a latch (ADR 0072, Decision 2): PPG's guards
    /// are consulted once per minibatch, so a phase in which every minibatch is
    /// non-finite must record one skip per minibatch, not one per phase.
    ///
    /// This is the property `warning_fired()` could never express — a boolean
    /// latch reads identically for 1 skip and for 8.
    #[test]
    #[allow(clippy::float_cmp)]
    fn test_ppg_agent_counts_repeated_loss_skips() {
        // 4 rollout steps / num_minibatches = 4 → mb_size = 1 → 4 minibatches per epoch; `$\times$`
        // update_epochs = 2 → 8 guarded policy attempts.
        const EXPECTED_SKIPS: u64 = 8;

        let mut agent = shaped_ppg_agent(4, 2, 4);

        let poisoned = agent.policy.get().clone().map(&mut NanInjector);
        agent.policy = Slot::new(poisoned);

        let mut rng = StdRng::seed_from_u64(0);
        let stats = agent.policy_phase_update(&mut rng);

        assert_eq!(
            stats.epochs_run, 2,
            "both epochs must run (target_kl is None), pinning the 4 × 2 \
             minibatch arithmetic below"
        );
        assert_eq!(
            agent.skipped_policy_updates(),
            EXPECTED_SKIPS,
            "every one of the 4 minibatches × 2 epochs = 8 NaN policy losses \
             must be counted; a latch would report 1"
        );
        assert!(
            agent.skipped_policy_updates() > 1,
            "the counter must exceed 1 — this is the assertion that fails for a \
             one-shot latch"
        );
        assert_eq!(
            agent.skipped_value_updates(),
            0,
            "the healthy value site is pinned at exactly 0 across all 8 \
             minibatches"
        );
        assert_eq!(
            agent.skipped_updates(),
            EXPECTED_SKIPS,
            "the aggregate must equal 8 + 0 + 0 + 0"
        );
        assert_eq!(
            stats.policy_loss, 0.0,
            "all 8 policy minibatches skipped → the 0.0 all-skipped sentinel"
        );
    }

    /// The aggregate must sum **four distinct** per-site counters, each exactly
    /// once (ADR 0072). Constructed so all four terms are different *and*
    /// none is zero — 2, 6, 4, 8, aggregate **20** — because the two properties
    /// close two different holes:
    ///
    /// - **Distinct** rules out *duplication*: a chain that adds one guard twice
    ///   and another not at all still reads correctly when terms are equal. With
    ///   `{2, 6, 4, 8}` every such chain misses `20` (e.g. `2+2+4+8 = 16`,
    ///   `6+6+4+8 = 24`, `2+6+4+4 = 16`, `2+6+8+8 = 24`).
    /// - **Nonzero** rules out *omission*: a zero term is invisible to a sum, so
    ///   deleting its summand changes nothing. The earlier shape of this test
    ///   (2 / **0** / 8 / 4 = 14) passed against a `skipped_updates` with
    ///   `value_loss_guard.skipped()` removed from the chain. Here the four
    ///   single-term omissions give `18`, `14`, `16`, `12` — none is `20`.
    ///
    /// # Where each term comes from
    ///
    /// `shaped_ppg_agent(2, 1, 1)`: every policy phase runs 4 rollout steps / 2 minibatches =
    /// `mb_size` 2 → 2 minibatches `$\times$` 1 epoch = **2 guarded attempts per site per policy
    /// phase**; every aux phase runs `$\lceil \text{total\_steps}/1 \rceil = \text{total\_steps}$`
    /// minibatches `$\times$` `e_aux = 1`.
    ///
    /// The policy and value sites share those attempts, so their counts are separated by *which net
    /// is diverged* and by *how many policy phases run*: one phase with a `NaN` policy (policy =
    /// 2), then three phases with a `NaN` value net (value = 3 `$\times$` 2 = 6). Each of the
    /// latter re-collects its rollout with a **healthy** value net first — a rollout collected
    /// through a `NaN` critic would store `NaN` values, hence `NaN` advantages, and the policy site
    /// would skip too, collapsing the two terms back together.
    #[test]
    fn test_ppg_agent_skipped_updates_aggregates_unequal_per_site_counts() {
        // Policy phase #1: 2 minibatches `$\times$` 1 epoch = 2 guarded policy attempts, all NaN
        // (policy poisoned).
        const POLICY_SKIPS: u64 = 2;
        // Policy phases #2-#4: 3 phases `$\times$` 2 minibatches = 6 guarded value attempts, all
        // NaN (value poisoned, policy healthy). NONZERO on purpose — see the doc comment.
        const VALUE_PHASES: usize = 3;
        const VALUE_SKIPS: u64 = 6;
        // Aux phase #1: 1 slice `$\times$` 4 steps, aux_batch_size 1 → 4 minibatches `$\times$`
        // e_aux 1 = 4 guarded aux-total attempts, all NaN (policy poisoned).
        const AUX_TOTAL_SKIPS: u64 = 4;
        // Aux phase #2: 2 slices `$\times$` 4 steps = 8 steps, aux_batch_size 1 → 8 minibatches
        // `$\times$` e_aux 1 = 8 guarded main-value attempts, all NaN (value poisoned).
        const AUX_MAIN_SKIPS: u64 = 8;

        let mut agent = shaped_ppg_agent(2, 1, 1);
        let mut rng = StdRng::seed_from_u64(0);

        // Keep healthy copies of both nets: each phase below diverges exactly
        // one site, and restores the other from these before any rollout
        // collection so the stored values/returns stay finite.
        let healthy_policy = agent.policy.get().clone();
        let healthy_value = agent.value.get().clone();

        // One 4-step slice, snapshotted while both nets are healthy so the
        // stored returns are finite.
        agent.snapshot_into_aux_buffer();

        // --- policy phase #1, poisoned policy: policy site only ---
        agent.policy = Slot::new(healthy_policy.clone().map(&mut NanInjector));
        agent.policy_phase_update(&mut rng);

        // --- aux phase #1, still poisoned policy: aux-total site only ---
        agent
            .maybe_aux_phase(&mut rng)
            .expect("n_iteration = 1: one snapshotted slice arms the aux phase");

        assert_eq!(
            agent.skipped_value_updates(),
            0,
            "site isolation: the value net was healthy throughout phase #1, so \
             the value site is still silent — the divergence below is the only \
             reason it starts counting"
        );

        // --- policy phases #2-#4, healthy policy but poisoned value: value
        // site only. Restore BOTH nets before each rollout: the rollout's
        // stored values must be finite or the policy site would skip as well.
        agent.policy = Slot::new(healthy_policy.clone());
        for _ in 0..VALUE_PHASES {
            agent.value = Slot::new(healthy_value.clone());
            collect_rollout(&mut agent, &mut rng);
            agent.value = Slot::new(healthy_value.clone().map(&mut NanInjector));
            agent.policy_phase_update(&mut rng);
        }

        // --- aux phase #2, healthy policy but poisoned value: main-value only ---
        agent.value = Slot::new(healthy_value.clone());
        collect_rollout(&mut agent, &mut rng);
        agent.snapshot_into_aux_buffer();
        agent.snapshot_into_aux_buffer();
        agent.value = Slot::new(healthy_value.map(&mut NanInjector));
        agent
            .maybe_aux_phase(&mut rng)
            .expect("two snapshotted slices arm the aux phase again");

        assert_eq!(
            agent.skipped_policy_updates(),
            POLICY_SKIPS,
            "policy site: 2 minibatches × 1 epoch = 2 NaN policy losses, accrued \
             only in policy phase #1; phases #2-#4 ran a healthy policy against \
             finite advantages and added nothing"
        );
        assert_eq!(
            agent.skipped_value_updates(),
            VALUE_SKIPS,
            "value site: 3 policy phases × 2 minibatches = 6 NaN value losses. \
             NONZERO on purpose — a zero here would be invisible to the \
             aggregate, letting a chain that drops this summand still total 20"
        );
        assert_eq!(
            agent.skipped_aux_value_updates(),
            AUX_MAIN_SKIPS,
            "aux main-value site: 8 minibatches × 1 aux epoch = 8 NaN losses, \
             accrued only in the second aux phase"
        );
        assert_eq!(
            agent.skipped_aux_total_updates(),
            AUX_TOTAL_SKIPS,
            "aux-total site: 4 minibatches × 1 aux epoch = 4 NaN losses, \
             accrued only in the first aux phase"
        );
        assert_eq!(
            agent.skipped_updates(),
            POLICY_SKIPS + VALUE_SKIPS + AUX_MAIN_SKIPS + AUX_TOTAL_SKIPS,
            "the aggregate must be 2 + 6 + 8 + 4 = 20: each of the four \
             distinct, nonzero per-site counters summed exactly once"
        );
        assert_eq!(
            agent.skipped_updates(),
            20,
            "pinned literal, so a change to any summand's arithmetic must be \
             restated here rather than cancelling out"
        );
    }

    // -------- auxiliary-phase learning-rate alignment --------

    /// Builds an empty agent with LR annealing **on** over `total_iterations`
    /// policy phases. Mirrors [`primed_ppg_agent`]'s shapes; the rollout is left
    /// to the caller so several phases can be driven in sequence.
    fn annealing_ppg_agent(total_iterations: usize) -> TestAgent {
        let device = Device::flex();
        let policy: PpgCategoricalPolicyHead = PpgCategoricalPolicyHeadConfig {
            obs_dim: 2,
            hidden: 4,
            num_actions: 2,
        }
        .try_init(&device)
        .expect("valid head config");
        let value = TestValue::init(&device);
        let config = PpgConfig {
            n_iteration: 1,
            e_aux: 1,
            aux_batch_size: 4,
            ppo: PpoTrainingConfig {
                num_envs: 1,
                num_steps: 4,
                num_minibatches: 1,
                update_epochs: 1,
                anneal_lr: true,
                ..Default::default()
            },
            ..Default::default()
        };
        TestAgent::new(policy, value, config, device, total_iterations).expect("valid config")
    }

    /// Regression: the auxiliary phase must step at the learning
    /// rate of the policy phase it accompanies, so `policy_phase_lr` has to hold
    /// the rate read *before* `policy_phase_update` bumped `iteration` — one
    /// tick behind `current_learning_rate()`. On the final iteration that
    /// distinction is the whole defect: `current_learning_rate()` is exactly
    /// `0.0` there, which would make the closing auxiliary phase a bit-exact
    /// no-op.
    #[test]
    // Exact comparison is the property: `annealed_learning_rate` lands on
    // *exactly* 0.0 at the last tick, and `policy_phase_lr` is a copy of a
    // `current_learning_rate()` return value, not a recomputation. A tolerance
    // would let the one-tick offset this test exists to catch slip through.
    #[allow(clippy::float_cmp)]
    fn test_ppg_agent_policy_phase_lr_trails_current_learning_rate_by_one_tick() {
        const TOTAL_ITERATIONS: usize = 4;
        let mut agent = annealing_ppg_agent(TOTAL_ITERATIONS);
        let mut rng = StdRng::seed_from_u64(9);

        for completed in 0..TOTAL_ITERATIONS {
            let lr_of_this_phase = agent.current_learning_rate();
            collect_rollout(&mut agent, &mut rng);
            agent.policy_phase_update(&mut rng);

            assert_eq!(
                agent.iteration(),
                completed + 1,
                "each policy phase must advance the iteration counter exactly once"
            );
            assert_eq!(
                agent.policy_phase_lr, lr_of_this_phase,
                "policy_phase_lr must be the pre-increment rate the phase applied"
            );
            assert!(
                agent.policy_phase_lr > agent.current_learning_rate(),
                "the annealed rate must have advanced past the phase that just ran: \
                 policy_phase_lr = {}, current_learning_rate() = {}",
                agent.policy_phase_lr,
                agent.current_learning_rate()
            );
        }

        assert_eq!(
            agent.current_learning_rate(),
            0.0,
            "the anneal lands on exactly 0.0 at iteration == total_iterations"
        );
        assert!(
            agent.policy_phase_lr > 0.0,
            "the last policy phase ran at a positive rate, so the auxiliary phase \
             that accompanies it must too — got {}",
            agent.policy_phase_lr
        );
    }

    /// Behavioral companion to
    /// [`test_ppg_agent_policy_phase_lr_trails_current_learning_rate_by_one_tick`]: the
    /// aligned rate must actually *move parameters*, not merely be positive.
    ///
    /// Measures a host-side weight delta across the terminal auxiliary phase — the phase the
    /// one-tick offset above turned into a bit-exact no-op. A weight delta is
    /// `$\text{lr} \cdot \text{step}$`, linear in the learning rate and here ~1e-4 against weights
    /// of order 1e-1, so it clears `f32` resolution by several orders of magnitude.
    ///
    /// This exists because the equivalent end-to-end check in
    /// `crates/rlevo/tests/ppg_integration.rs` used to assert `AuxPhaseStats::policy_kl > 0.0`.
    /// That also catches the regression — at `lr == 0.0` the parameters are bitwise unchanged and
    /// every minibatch's KL is an exact zero — but it decides the question through an `f32` mean of
    /// log-differences between near-identical logits, measured at ~`$3.4\times$` `f32::EPSILON` on
    /// a healthy run. A backend with a different reduction order could round a *healthy* phase to
    /// zero. The integration test now asserts on `AuxPhaseStats::learning_rate`, which is exact,
    /// and the "a nonzero rate moves the policy" half lives here, where the observable has margin
    /// to spare.
    #[test]
    // `current_learning_rate()` landing on *exactly* 0.0 at the terminal
    // iteration is the precondition that makes this the regression's failing
    // configuration rather than an ordinary interior phase.
    #[allow(clippy::float_cmp)]
    fn test_ppg_agent_aux_phase_at_nonzero_lr_moves_policy_parameters() {
        const TOTAL_ITERATIONS: usize = 4;
        let mut agent = annealing_ppg_agent(TOTAL_ITERATIONS);
        let mut rng = StdRng::seed_from_u64(9);

        // `annealing_ppg_agent` sets `n_iteration = 1`, so an auxiliary phase
        // fires after every policy phase; the loop keeps the last one. Mirrors
        // the per-iteration order in `algorithms::ppg::train`.
        let mut terminal = None;
        for _ in 0..TOTAL_ITERATIONS {
            collect_rollout(&mut agent, &mut rng);
            agent.snapshot_into_aux_buffer();
            agent.policy_phase_update(&mut rng);
            let before = param_snapshot(agent.policy());
            let stats = agent.maybe_aux_phase(&mut rng);
            let after = param_snapshot(agent.policy());
            if let Some(stats) = stats {
                terminal = Some((before, after, stats));
            }
        }

        let (before, after, stats) =
            terminal.expect("with n_iteration = 1 an auxiliary phase fires every iteration");
        assert_eq!(
            agent.current_learning_rate(),
            0.0,
            "the anneal must land on exactly 0.0 at the terminal iteration, or this is not \
             the configuration the regression broke"
        );
        assert!(
            stats.learning_rate > 0.0,
            "the terminal auxiliary phase must step at the rate of the policy phase it \
             accompanies, got {}",
            stats.learning_rate
        );
        assert_eq!(
            before.len(),
            after.len(),
            "the auxiliary phase must not change the policy's parameter topology"
        );
        let max_delta = before
            .iter()
            .zip(&after)
            .map(|(b, a)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_delta > 0.0,
            "the terminal auxiliary phase left every policy parameter bit-exactly unchanged \
             (max |Δw| = {max_delta}) — that is what stepping at lr = 0.0 does"
        );
    }

    /// As PPO's: a driver that ends an episode the environment did not end,
    /// at its `max_steps` cut or by switching phase, leaves the last step
    /// recorded `Running`, and that step must become a truncation so GAE does
    /// not run on into the next episode's first step.
    #[test]
    fn test_ppg_driver_end_truncates_an_open_step() {
        use crate::algorithms::bootstrap_mask::MaskDiscreteAction as Act;

        let mut agent = agent_with_clip_grad(None);
        let mut rng = StdRng::seed_from_u64(0);
        let (obs, next) = (TestObs([0.1, -0.2]), TestObs([0.3, 0.4]));
        let step = |agent: &mut TestAgent, rng: &mut StdRng| {
            let action: Act = agent.act(&obs, rng);
            agent.on_transition(
                Transition {
                    obs: &obs,
                    action: &action,
                    reward: 1.0,
                    next_obs: &next,
                    status: EpisodeStatus::Running,
                },
                rng,
            );
        };

        step(&mut agent, &mut rng);
        assert!(
            !agent.buffer.last_step_ended(),
            "the step was recorded Running"
        );
        Agent::<TestObs, Act>::on_episode_end(&mut agent, EpisodeStatus::Truncated);
        assert!(
            agent.buffer.last_step_ended(),
            "a cut after a Running step must end the step"
        );

        step(&mut agent, &mut rng);
        Agent::<TestObs, Act>::set_training(&mut agent, false);
        assert!(
            agent.buffer.last_step_ended(),
            "leaving training after a Running step must end the step"
        );
    }
}
