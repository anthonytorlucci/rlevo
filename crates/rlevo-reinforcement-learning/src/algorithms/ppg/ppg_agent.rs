//! Phasic Policy Gradient (PPG) agent.
//!
//! Owns the policy network (categorical logits head + auxiliary value head),
//! a separate main value network, two Adam optimizers, the PPO
//! [`RolloutBuffer`], an [`AuxRolloutBuffer`] accumulating the last
//! `n_iteration` rollouts, and the configuration. One iteration consists of:
//!
//! 1. [`PpgAgent::act`] / [`PpgAgent::record_step`] — collect a rollout.
//! 2. [`PpgAgent::finalize_rollout`] — compute GAE (shared with PPO).
//! 3. [`PpgAgent::snapshot_into_aux_buffer`] — snapshot observations, target
//!    returns, and pre-aux-phase logits into the auxiliary buffer.
//! 4. [`PpgAgent::policy_phase_update`] — PPO policy-phase update (clipped
//!    surrogate + value loss), identical to [`PpoAgent::update`](crate::algorithms::ppo::ppo_agent::PpoAgent::update).
//! 5. [`PpgAgent::maybe_aux_phase`] — if the auxiliary buffer holds
//!    `n_iteration` rollouts, run `e_aux` epochs of auxiliary updates
//!    (main-value MSE + aux-value MSE + `β · KL(π_old ‖ π_new)`) and drain
//!    the buffer.
//!
//! The policy-phase loop is code-wise parallel to [`PpoAgent::update`](crate::algorithms::ppo::ppo_agent::PpoAgent::update) but
//! kept in a separate function so `PpoAgent` remains untouched and the
//! snapshot-before-clear hook can slot in cleanly.

use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, GradientsParams};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{ElementConversion, Tensor, TensorData};
use rand::Rng;

use crate::algorithms::shared::{FiniteLossGuard, Slot};
use crate::metrics::{AgentStats, PerformanceRecord};
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

/// Error variants returned by [`PpgAgent`] operations.
#[derive(Debug, thiserror::Error)]
pub enum PpgAgentError {
    /// A tensor-to-action conversion failed.
    #[error("Tensor conversion failed: {0}")]
    TensorConversionFailed(String),
    /// The config is internally inconsistent.
    #[error("Invalid config: {0}")]
    InvalidConfig(String),
    /// An I/O error occurred while saving or loading model weights.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Env-side failure surfaced through the train loop.
    #[error("Environment error: {0}")]
    Environment(String),
}

/// Per-episode statistics emitted by the PPG training loop.
#[derive(Debug, Clone, Copy)]
pub struct PpgMetrics {
    /// Undiscounted total reward for this episode.
    pub reward: f32,
    /// Number of environment steps in this episode.
    pub steps: usize,
    /// Mean policy (clipped surrogate) loss from the most recent policy-phase
    /// update preceding this episode's end.
    pub policy_loss: f32,
    /// Mean value loss from the most recent policy-phase update.
    pub value_loss: f32,
    /// Mean per-step policy entropy from the most recent policy-phase update.
    pub entropy: f32,
    /// Approximate KL divergence from the most recent policy-phase update.
    pub approx_kl: f32,
    /// Fraction of samples that hit the PPO clip boundary.
    pub clip_frac: f32,
    /// Most recent auxiliary-phase main-value loss (`0.0` if none has run yet).
    pub aux_main_value_loss: f32,
    /// Most recent auxiliary-phase aux-value-head loss.
    pub aux_value_loss: f32,
    /// Most recent auxiliary-phase distillation KL.
    pub aux_policy_kl: f32,
    /// Current learning rate (after annealing).
    pub learning_rate: f32,
}

impl PerformanceRecord for PpgMetrics {
    fn score(&self) -> f32 {
        self.reward
    }
    fn duration(&self) -> usize {
        self.steps
    }
}

/// Summary of one auxiliary phase (across `e_aux` epochs and its minibatches).
#[derive(Debug, Clone, Copy)]
pub struct AuxPhaseStats {
    /// Mean main-value MSE across minibatches.
    pub main_value_loss: f32,
    /// Mean aux-value MSE across minibatches.
    pub aux_value_loss: f32,
    /// Mean `KL(π_old ‖ π_new)` distillation loss across minibatches.
    pub policy_kl: f32,
    /// Epochs actually executed.
    pub epochs_run: usize,
    /// Minibatches processed.
    pub minibatches: usize,
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
/// - `B` — Burn autodiff backend.
/// - `P` — Policy network module implementing both `PpoPolicy<B, DB>` and
///   `PpgAuxValueHead<B, DB>`.
/// - `V` — Main value network implementing `PpoValue<B, DB>`.
/// - `O` — Observation type; must be convertible to a rank-`DO` tensor.
/// - `DO` — Rank of the observation tensor (e.g. `1` for flat vectors).
/// - `DB` — Rank of the batched observation tensor (`DO + 1`; e.g. `2` for a
///   batch of flat vectors).
///
/// # Usage
///
/// Construct with [`PpgAgent::new`], then drive the training loop via
/// [`train_discrete`](crate::algorithms::ppg::train::train_discrete) or
/// manually via the five-step sequence described in the module header.
///
/// # Network ownership
///
/// The policy and value networks live in [`Slot`]s because Burn's
/// `Optimizer::step` consumes the module by value. In both the policy phase
/// ([`policy_phase_update`](Self::policy_phase_update)) and the auxiliary phase
/// ([`maybe_aux_phase`](Self::maybe_aux_phase)), every fallible operation — the
/// forward pass, the losses, `backward`, and `GradientsParams::from_grads` —
/// runs on a borrow, so a panic there leaves both networks intact and the agent
/// usable. Only a panic *inside* the optimizer step itself poisons a slot; that
/// window is irreducible and terminal for the agent (see the
/// [`shared`](crate::algorithms::shared) module docs).
pub struct PpgAgent<B, P, V, O, const DO: usize, const DB: usize>
where
    B: AutodiffBackend,
    P: PpoPolicy<B, DB> + PpgAuxValueHead<B, DB>,
    V: PpoValue<B, DB>,
    O: Observation<DO> + TensorConvertible<DO, B>,
{
    policy: Slot<P>,
    value: Slot<V>,
    policy_optim: OptimizerAdaptor<Adam, P, B>,
    value_optim: OptimizerAdaptor<Adam, V, B>,
    buffer: RolloutBuffer<B, O>,
    aux_buffer: AuxRolloutBuffer<B, O>,
    config: PpgConfig,
    device: B::Device,
    iteration: usize,
    total_iterations: usize,
    /// Learning rate the most recent [`policy_phase_update`](Self::policy_phase_update)
    /// actually applied, snapshotted *before* that update bumped `iteration`.
    ///
    /// The auxiliary phase belongs to the policy phase it follows, so it steps
    /// at this rate rather than at `current_learning_rate()` — which, read from
    /// [`maybe_aux_phase`](Self::maybe_aux_phase), has already advanced one
    /// annealing tick (#324).
    policy_phase_lr: f64,
    step: usize,
    stats: AgentStats<PpgMetrics>,
    last_aux: Option<AuxPhaseStats>,
    /// Non-finite-loss guard for the policy-phase policy-loss site (ADR 0056,
    /// #318). One per-run `warn!` latch per site; the skip fires every time.
    policy_loss_guard: FiniteLossGuard,
    /// Non-finite-loss guard for the policy-phase value-loss site.
    value_loss_guard: FiniteLossGuard,
    /// Non-finite-loss guard for the auxiliary-phase main-value-loss site.
    aux_main_value_guard: FiniteLossGuard,
    /// Non-finite-loss guard for the auxiliary-phase combined loss
    /// (`aux_v_loss + β·KL`); its input is host-*derived* from the two summands
    /// already read, adding no sync (ADR 0056 §5).
    aux_total_guard: FiniteLossGuard,
}

impl<B, P, V, O, const DO: usize, const DB: usize> std::fmt::Debug for PpgAgent<B, P, V, O, DO, DB>
where
    B: AutodiffBackend,
    P: PpoPolicy<B, DB> + PpgAuxValueHead<B, DB>,
    V: PpoValue<B, DB>,
    O: Observation<DO> + TensorConvertible<DO, B>,
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

impl<B, P, V, O, const DO: usize, const DB: usize> PpgAgent<B, P, V, O, DO, DB>
where
    B: AutodiffBackend,
    P: PpoPolicy<B, DB> + PpgAuxValueHead<B, DB>,
    V: PpoValue<B, DB>,
    O: Observation<DO> + TensorConvertible<DO, B> + TensorConvertible<DO, B::InnerBackend>,
{
    /// Snapshots the policy onto the inner (non-autodiff) backend for repeated
    /// greedy inference.
    ///
    /// Returns a frozen inference handle for use with
    /// [`act_greedy_env_row_with`](Self::act_greedy_env_row_with). Snapshot
    /// once after training, then reuse across many steps — the snapshot goes
    /// stale if the policy is updated again. Mirrors
    /// [`PpoAgent::inference_net`](crate::algorithms::ppo::ppo_agent::PpoAgent::inference_net).
    pub fn inference_net(&self) -> P::InnerModule {
        self.policy().valid()
    }

    /// Deterministic env-space action row for `obs`, evaluated against a
    /// pre-snapshotted inner network — no sampling, no autodiff graph.
    ///
    /// Equivalent to [`act_greedy`](Self::act_greedy) (the categorical mode is
    /// the argmax over logits, and `raw_to_env_row` is the identity for the
    /// discrete head) but skips per-step autodiff-graph construction, which
    /// dominates cost at batch size 1. Use this for evaluation and throughput.
    pub fn act_greedy_env_row_with(&self, net: &P::InnerModule, obs: &O) -> Vec<f32> {
        let obs_t: Tensor<B::InnerBackend, DO> = obs.to_tensor(&self.device);
        let batched: Tensor<B::InnerBackend, DB> = obs_t.unsqueeze::<DB>();
        P::deterministic_env_row_inner(net, batched)
    }
}

impl<B, P, V, O, const DO: usize, const DB: usize> PpgAgent<B, P, V, O, DO, DB>
where
    B: AutodiffBackend,
    P: PpoPolicy<B, DB> + PpgAuxValueHead<B, DB>,
    V: PpoValue<B, DB>,
    O: Observation<DO> + TensorConvertible<DO, B>,
{
    /// Construct a new agent from a pre-built policy and value network.
    ///
    /// Initialises both Adam optimizers from `config.ppo.optimizer` (with
    /// optional gradient clipping), creates an empty rollout buffer sized to
    /// `config.ppo.batch_size()`, and an empty auxiliary buffer. The
    /// `total_iterations` value is used only for learning-rate annealing.
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
        device: B::Device,
        total_iterations: usize,
    ) -> Result<Self, rlevo_core::config::ConfigError> {
        config.validate()?;
        let action_dim = policy.action_dim();
        let capacity = config.ppo.batch_size();

        let adam_policy = config.ppo.optimizer.clone();
        let policy_optim = match &config.ppo.clip_grad {
            Some(clip) => adam_policy
                .with_grad_clipping(Some(clip.clone()))
                .init::<B, P>(),
            None => adam_policy.init::<B, P>(),
        };
        let adam_value = config.ppo.optimizer.clone();
        let value_optim = match &config.ppo.clip_grad {
            Some(clip) => adam_value
                .with_grad_clipping(Some(clip.clone()))
                .init::<B, V>(),
            None => adam_value.init::<B, V>(),
        };

        let buffer = RolloutBuffer::new(capacity, action_dim);
        let aux_buffer = AuxRolloutBuffer::new();
        let stats = AgentStats::<PpgMetrics>::new(100);
        // What `current_learning_rate()` returns at `iteration == 0`: the
        // annealed rate at tick zero is the base rate, and with annealing off it
        // is the base rate throughout. Seeding the field here keeps a
        // `maybe_aux_phase` that somehow precedes any policy phase on-schedule
        // instead of stepping at an uninitialised rate.
        let policy_phase_lr = config.ppo.learning_rate;

        Ok(Self {
            policy: Slot::new(policy),
            value: Slot::new(value),
            policy_optim,
            value_optim,
            buffer,
            aux_buffer,
            config,
            device,
            iteration: 0,
            total_iterations,
            policy_phase_lr,
            step: 0,
            stats,
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

    /// Per-episode statistics accumulated over the last 100 episodes.
    pub fn stats(&self) -> &AgentStats<PpgMetrics> {
        &self.stats
    }

    /// Records one completed episode's metrics into the rolling stats window.
    pub fn record_episode(&mut self, metrics: PpgMetrics) {
        self.stats.record(metrics);
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
    /// which [`policy_phase_update`](Self::policy_phase_update) increments on
    /// its way out. Called after an update, it therefore reports one annealing
    /// tick *past* the rate that update applied — and at the end of a run it
    /// reports `0.0`. Anything that must step at the rate of the phase that just
    /// ran reads `policy_phase_lr` instead; that is exactly what the auxiliary
    /// phase does (see [`maybe_aux_phase`](Self::maybe_aux_phase), #324).
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
    pub fn act<R: Rng + ?Sized>(&self, obs: &O, rng: &mut R) -> ActOutcome {
        let obs_t: Tensor<B, DO> = obs.to_tensor(&self.device);
        let batched: Tensor<B, DB> = obs_t.unsqueeze::<DB>();

        let sample = self.policy().sample_with_logprob(batched.clone(), rng);
        let raw_row = P::action_row_from_tensor(&sample.action, 0);
        let env_row = self.policy().raw_to_env_row(&raw_row);

        let log_prob = sample.log_prob.into_scalar().elem::<f32>();
        let entropy = sample.entropy.into_scalar().elem::<f32>();

        let value_raw: Tensor<B, DO> = obs.to_tensor(&self.device);
        let value_t: Tensor<B, DB> = value_raw.unsqueeze::<DB>();
        let value = self.value().forward(value_t).into_scalar().elem::<f32>();

        ActOutcome {
            raw_row,
            env_row,
            log_prob,
            entropy,
            value,
        }
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
    #[allow(clippy::cast_precision_loss)]
    pub fn act_greedy(&self, obs: &O) -> Vec<f32> {
        let obs_t: Tensor<B, DO> = obs.to_tensor(&self.device);
        let batched: Tensor<B, DB> = obs_t.unsqueeze::<DB>();
        let logits: Tensor<B, 2> = PpgAuxValueHead::logits(self.policy(), batched); // (1, A)
        let idx = logits.argmax(1).into_scalar().elem::<i64>();
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
    pub fn record_step(
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
        let t: Tensor<B, DO> = obs.to_tensor(&self.device);
        let batched: Tensor<B, DB> = t.unsqueeze::<DB>();
        self.value().forward(batched).into_scalar().elem::<f32>()
    }

    /// Computes GAE advantages/returns from `last_obs`. Identical to PPO's
    /// finalize step.
    ///
    /// As in
    /// [`PpoAgent::finalize_rollout`](crate::algorithms::ppo::ppo_agent::PpoAgent::finalize_rollout),
    /// `last_obs` is consulted only when the rollout's final step left the
    /// episode `Running`; otherwise the value forward is skipped.
    pub fn finalize_rollout(&mut self, last_obs: &O) {
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
    /// clear the rollout). The `π_old` logits for the distillation KL are
    /// **not** captured here — they are computed fresh at the start of
    /// [`Self::maybe_aux_phase`] against the post-policy-phase policy, so
    /// the distillation preserves (rather than undoes) the policy-phase's
    /// learning.
    pub fn snapshot_into_aux_buffer(&mut self) {
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
    /// this update steps at the same rate (#324).
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
    pub fn policy_phase_update<R: Rng + ?Sized>(&mut self, rng: &mut R) -> PpoUpdateStats {
        let cfg = self.config.ppo.clone();
        let batch_size = self.buffer.len();
        let mb_size = (batch_size / cfg.num_minibatches.max(1)).max(1);
        // Read before `self.iteration += 1` below, so this is the rate for the
        // phase about to run. Retained for the auxiliary phase, which belongs to
        // this policy phase and must not anneal ahead of it (#324).
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
        // finite and applied (#318, ADR 0056 §3). `mb_count` still denominates
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
                let mut batched_shape: Vec<usize> = Vec::with_capacity(DB);
                batched_shape.push(n);
                batched_shape.extend_from_slice(&obs_shape);
                let obs_batch: Tensor<B, DB> =
                    Tensor::from_data(TensorData::new(obs_flat, batched_shape), &self.device);

                let action_flat = self.buffer.gather_action_flat(chunk);
                let actions: P::ActionTensor =
                    P::action_tensor_from_flat(&action_flat, n, &self.device);

                let old_lp: Tensor<B, 1> =
                    self.buffer
                        .gather_f32(self.buffer.log_probs(), chunk, &self.device);
                let old_v: Tensor<B, 1> =
                    self.buffer
                        .gather_f32(self.buffer.values(), chunk, &self.device);
                let returns: Tensor<B, 1> =
                    self.buffer
                        .gather_f32(self.buffer.returns(), chunk, &self.device);
                let advs_raw: Tensor<B, 1> =
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
                let policy_loss_val = policy_loss.clone().into_scalar().elem::<f32>();

                let kl = approx_kl(eval.log_prob.clone(), old_lp.clone());
                let cf = clip_fraction(eval.log_prob.clone(), old_lp, cfg.clip_coef);
                kl_sum += kl;
                kl_count += 1;

                // #318 / ADR 0056: `policy_loss_val` is already host-resident,
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
                let v_loss_val = v_loss.into_scalar().elem::<f32>();
                // #318 / ADR 0056: same guard for the value site, latched
                // separately from the policy site.
                if self.value_loss_guard.check(v_loss_val) {
                    let grads = v_loss_scaled.backward();
                    let grads_params = GradientsParams::from_grads(grads, self.value());
                    self.value
                        .step_with(&mut self.value_optim, lr, grads_params);
                    value_loss_acc += v_loss_val;
                    value_healthy += 1;
                }

                entropy_acc += entropy_mean.into_scalar().elem::<f32>();
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
        // no continuous head in v1, so this is always `None`. A future Gaussian
        // PPG head would reuse `TanhGaussianPolicyHead` and would then inherit
        // its log_std bound and one-shot clamp warning here.
        let min_log_std = self.policy().min_log_std();

        self.buffer.clear();
        self.iteration += 1;

        // The two gated losses divide by their own healthy counts (`0.0`, not
        // NaN, when every minibatch skipped); the ungated diagnostics keep the
        // total minibatch denominator (#318, ADR 0056 §3).
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
        }
    }

    /// Runs the auxiliary phase if the auxiliary buffer is full.
    ///
    /// When `aux_buffer.num_slices() >= n_iteration`, iterates `e_aux` epochs
    /// of auxiliary updates:
    ///
    /// - main-value MSE against target returns,
    /// - aux-value-head MSE against target returns,
    /// - `β · KL(π_old ‖ π_new)` distillation pulling the policy back to the
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
    /// no-op at `lr == 0.0` (#324).
    // Divisor/normalizer derived from a count -- batch size, minibatch count,
    // history length, iteration number. All are bounded by configured sizes far
    // below f32's 2^24 (f64's 2^53) exact-integer limit.
    #[allow(clippy::cast_precision_loss)]
    pub fn maybe_aux_phase<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Option<AuxPhaseStats> {
        if !self.aux_buffer.is_ready(self.config.n_iteration) {
            return None;
        }
        let cfg = self.config.clone();
        // The accompanying policy phase's rate, not the next one's (#324).
        let lr = self.policy_phase_lr;
        let total_steps = self.aux_buffer.len_steps();
        if total_steps == 0 {
            self.aux_buffer.clear();
            return None;
        }

        // Snapshot π_old once, against the current (post-policy-phase) policy.
        // Stored flat row-major so minibatch slicing is a trivial copy.
        let old_logits_flat = self.compute_old_logits_flat(total_steps);
        let num_actions = old_logits_flat.len() / total_steps;
        debug_assert_eq!(old_logits_flat.len(), total_steps * num_actions);

        let mut main_v_acc = 0.0_f32;
        let mut aux_v_acc = 0.0_f32;
        let mut kl_acc = 0.0_f32;
        let mut mb_count = 0_usize;
        let mut epochs_run = 0_usize;
        // Healthy-minibatch denominators for the two gated aux losses (#318,
        // ADR 0056 §3). The main-value site feeds `main_v_acc`; the combined
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
                    .gather_minibatch::<DO, DB>(chunk, &self.device);

                let mut old_logits_mb: Vec<f32> = Vec::with_capacity(chunk.len() * num_actions);
                for &global in chunk {
                    let start = global * num_actions;
                    let end = start + num_actions;
                    old_logits_mb.extend_from_slice(&old_logits_flat[start..end]);
                }
                let old_logits_t: Tensor<B, 2> = Tensor::from_data(
                    TensorData::new(old_logits_mb, vec![chunk.len(), num_actions]),
                    &self.device,
                );

                // Main value-net update. As in the policy phase, all fallible
                // work runs on a borrow ahead of `step_with`.
                let new_v = self.value().forward(obs_t.clone());
                let v_loss = unclipped_value_loss(new_v, returns_t.clone());
                let v_loss_val = v_loss.clone().into_scalar().elem::<f32>();
                // #318 / ADR 0056: skip backward + step on a non-finite
                // main-value loss; exclude it from the reported mean.
                if self.aux_main_value_guard.check(v_loss_val) {
                    let grads = v_loss.backward();
                    let grads_params = GradientsParams::from_grads(grads, self.value());
                    self.value
                        .step_with(&mut self.value_optim, lr, grads_params);
                    main_v_acc += v_loss_val;
                    main_v_healthy += 1;
                }

                // Policy-net update: aux-value MSE + β · KL distillation.
                let aux_v_pred = PpgAuxValueHead::aux_value(self.policy(), obs_t.clone());
                let aux_v_loss = unclipped_value_loss(aux_v_pred, returns_t);
                let new_logits = PpgAuxValueHead::logits(self.policy(), obs_t);
                let kl = policy_kl_categorical(old_logits_t, new_logits);
                let aux_v_loss_val = aux_v_loss.clone().into_scalar().elem::<f32>();
                let kl_val = kl.clone().into_scalar().elem::<f32>();
                let total = aux_v_loss + kl.mul_scalar(cfg.beta_clone);
                // #318 / ADR 0056: `total = aux_v_loss + β·kl` is never read
                // host-side, so guard on the host-*derived* scalar built from
                // its two already-read summands — no extra device→host sync.
                // Both accumulators this site feeds are excluded on a skip.
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
        // count attempted (#318, ADR 0056 §3).
        let stats = AuxPhaseStats {
            main_value_loss: if main_v_healthy == 0 {
                0.0
            } else {
                main_v_acc / main_v_healthy as f32
            },
            // `aux_value_loss` and `policy_kl` intentionally share the
            // `aux_total_healthy` denominator: both are fed by the single
            // aux-total guard site (`aux_v_loss + β·kl`), so they are applied
            // and skipped together — this is not a copy-pasted divisor.
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
        };
        self.last_aux = Some(stats);
        Some(stats)
    }

    /// Computes `π_old` logits for every step in the auxiliary buffer.
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
            let mut batched_shape: Vec<usize> = Vec::with_capacity(DB);
            batched_shape.push(n);
            batched_shape.extend_from_slice(&obs_shape);
            let obs_batch: Tensor<B, DB> =
                Tensor::from_data(TensorData::new(obs_flat, batched_shape), &self.device);
            let logits = PpgAuxValueHead::logits(self.policy(), obs_batch);
            let logits_data = logits.into_data().convert::<f32>();
            out.extend_from_slice(logits_data.as_slice::<f32>().expect("f32 logits"));
            start = end;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, Flex};
    use burn::grad_clipping::GradientClippingConfig;
    use burn::module::{Module, ModuleMapper, Param};
    use burn::nn::{Linear, LinearConfig};
    use burn::tensor::backend::Backend;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rlevo_core::base::{HostRow, TensorConversionError};
    use serde::{Deserialize, Serialize};

    use crate::algorithms::ppg::policies::{
        PpgCategoricalPolicyHead, PpgCategoricalPolicyHeadConfig,
    };
    use crate::algorithms::ppg::ppg_config::PpgConfigBuilder;
    use crate::algorithms::ppo::ppo_config::PpoTrainingConfig;

    type TestBackend = Autodiff<Flex>;
    type TestAgent = PpgAgent<
        TestBackend,
        PpgCategoricalPolicyHead<TestBackend>,
        TestValue<TestBackend>,
        TestObs,
        1,
        2,
    >;

    /// Minimal scalar main-value head: `PpgAgent` requires a `PpoValue`
    /// implementation and the crate ships no built-in one.
    #[derive(Module, Debug)]
    struct TestValue<B: Backend> {
        head: Linear<B>,
    }

    impl<B: Backend> TestValue<B> {
        fn init(device: &<B as burn::tensor::backend::BackendTypes>::Device) -> Self {
            Self {
                head: LinearConfig::new(2, 1).init(device),
            }
        }
    }

    impl<B: AutodiffBackend> PpoValue<B, 2> for TestValue<B> {
        fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 1> {
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

    /// Builds an agent whose `config.ppo.clip_grad` is exactly `clip_grad`,
    /// leaving every other knob at its default.
    fn agent_with_clip_grad(clip_grad: Option<GradientClippingConfig>) -> TestAgent {
        let device = Default::default();
        let policy: PpgCategoricalPolicyHead<TestBackend> = PpgCategoricalPolicyHeadConfig {
            obs_dim: 2,
            hidden: 4,
            num_actions: 2,
        }
        .try_init::<TestBackend>(&device)
        .expect("valid head config");
        let value = TestValue::<TestBackend>::init(&device);
        let config = PpgConfigBuilder::new()
            .with_ppo(|p| PpoTrainingConfig { clip_grad, ..p })
            .build()
            .expect("valid config");
        TestAgent::new(policy, value, config, device, 1).expect("valid config")
    }

    /// Regression (issue #183): PPG builds its optimizers from
    /// `config.ppo.clip_grad` itself rather than delegating to `PpoAgent`, so
    /// this wiring can rot independently of the PPO copy — assert it here too.
    #[test]
    fn clip_grad_none_leaves_both_optimizers_unclipped() {
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
    fn clip_grad_some_reaches_both_optimizers() {
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

    // -------- non-finite-loss guard (ADR 0056, #318) --------

    /// Replaces every float parameter of a module with `NaN`, simulating a
    /// network that has diverged to non-finite weights — the realistic source
    /// of a non-finite loss. Applied to a *clone* so the live net's `ParamId`s
    /// are preserved.
    struct NanInjector;

    impl<B: Backend> ModuleMapper<B> for NanInjector {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
            let (id, tensor, mapper) = param.consume();
            Param::from_mapped_value(id, tensor.mul_scalar(f32::NAN), mapper)
        }
    }

    /// Reads the main value head's weights back to the host, element-wise, so a
    /// change across a phase proves the value net actually stepped.
    fn value_weights<B: Backend>(v: &TestValue<B>) -> Vec<f32> {
        v.head
            .weight
            .val()
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
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
            let outcome = agent.act(&obs, rng);
            agent.record_step(obs, &outcome, 1.0, &next, EpisodeStatus::Running);
            last = next;
        }
        agent.finalize_rollout(&last);
    }

    /// Builds an agent with a single four-step rollout collected and finalized
    /// (advantages/returns populated) with *healthy* networks. `n_iteration = 1`
    /// so a single `snapshot_into_aux_buffer` makes the auxiliary phase ready.
    fn primed_ppg_agent() -> TestAgent {
        let device = Default::default();
        let policy: PpgCategoricalPolicyHead<TestBackend> = PpgCategoricalPolicyHeadConfig {
            obs_dim: 2,
            hidden: 4,
            num_actions: 2,
        }
        .try_init::<TestBackend>(&device)
        .expect("valid head config");
        let value = TestValue::<TestBackend>::init(&device);
        let config = PpgConfigBuilder::new()
            .n_iteration(1)
            .e_aux(1)
            .aux_batch_size(4)
            .with_ppo(|p| PpoTrainingConfig {
                num_envs: 1,
                num_steps: 4,
                num_minibatches: 1,
                update_epochs: 1,
                anneal_lr: false,
                ..p
            })
            .build()
            .expect("valid config");
        let mut agent = TestAgent::new(policy, value, config, device, 1).expect("valid config");

        let mut rng = StdRng::seed_from_u64(0);
        collect_rollout(&mut agent, &mut rng);
        agent
    }

    /// Policy-phase site: a non-finite policy loss must skip the policy
    /// `backward` + optimizer step and be excluded from the reported mean (ADR
    /// 0056, #318). Diverging the policy net to NaN forces a NaN clipped
    /// surrogate; the guard must fire, the reported `policy_loss` must be the
    /// `0.0` all-skipped sentinel, and the value site must still learn.
    #[test]
    #[allow(clippy::float_cmp)]
    fn ppg_policy_phase_nonfinite_policy_loss_skips_and_warns() {
        let mut agent = primed_ppg_agent();
        let value_before = value_weights(agent.value.get());

        let poisoned = agent.policy.get().clone().map(&mut NanInjector);
        agent.policy = Slot::new(poisoned);

        let mut rng = StdRng::seed_from_u64(1);
        let stats = agent.policy_phase_update(&mut rng);

        assert!(
            agent.policy_loss_guard.warning_fired(),
            "a NaN policy loss must fire the policy-phase policy guard"
        );
        assert!(
            !agent.value_loss_guard.warning_fired(),
            "the value net was healthy: its guard must not fire"
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
    /// `aux_v_loss_val + kl_val·β` input (ADR 0056 §5). Diverging the policy net
    /// makes both the aux-value prediction and the distillation logits NaN, so
    /// the combined loss is non-finite; the guard must fire and skip only the
    /// policy step, while the independent main-value site still learns. Both
    /// `aux_value_loss` and `policy_kl` report `0.0` (they share the skipped
    /// aux-total site's healthy denominator).
    #[test]
    #[allow(clippy::float_cmp)]
    fn ppg_aux_phase_nonfinite_total_loss_skips_and_warns() {
        let mut agent = primed_ppg_agent();
        // Snapshot the finalized rollout; `n_iteration = 1` arms the aux phase.
        agent.snapshot_into_aux_buffer();
        let value_before = value_weights(agent.value.get());

        // Poison the policy: the aux-total site (aux_value + β·KL) runs through
        // the policy, so its guard input goes NaN; the main-value site (value
        // net) stays finite.
        let poisoned = agent.policy.get().clone().map(&mut NanInjector);
        agent.policy = Slot::new(poisoned);

        let mut rng = StdRng::seed_from_u64(3);
        let stats = agent
            .maybe_aux_phase(&mut rng)
            .expect("n_iteration = 1 makes the aux phase ready after one snapshot");

        assert!(
            agent.aux_total_guard.warning_fired(),
            "the NaN aux-total loss must fire the aux-total guard"
        );
        assert!(
            !agent.aux_main_value_guard.warning_fired(),
            "the main-value site was healthy: its guard must not fire"
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

    // -------- auxiliary-phase learning-rate alignment (#324) --------

    /// Builds an empty agent with LR annealing **on** over `total_iterations`
    /// policy phases. Mirrors [`primed_ppg_agent`]'s shapes; the rollout is left
    /// to the caller so several phases can be driven in sequence.
    fn annealing_ppg_agent(total_iterations: usize) -> TestAgent {
        let device = Default::default();
        let policy: PpgCategoricalPolicyHead<TestBackend> = PpgCategoricalPolicyHeadConfig {
            obs_dim: 2,
            hidden: 4,
            num_actions: 2,
        }
        .try_init::<TestBackend>(&device)
        .expect("valid head config");
        let value = TestValue::<TestBackend>::init(&device);
        let config = PpgConfigBuilder::new()
            .n_iteration(1)
            .e_aux(1)
            .aux_batch_size(4)
            .with_ppo(|p| PpoTrainingConfig {
                num_envs: 1,
                num_steps: 4,
                num_minibatches: 1,
                update_epochs: 1,
                anneal_lr: true,
                ..p
            })
            .build()
            .expect("valid config");
        TestAgent::new(policy, value, config, device, total_iterations).expect("valid config")
    }

    /// Regression (issue #324): the auxiliary phase must step at the learning
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
    fn ppg_policy_phase_lr_trails_current_learning_rate_by_one_tick() {
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
}
