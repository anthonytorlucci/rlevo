//! Proximal Policy Optimization (PPO) agent: rollout orchestration and update step.
//!
//! The agent owns the policy network, value network, a pair of Adam
//! optimizers (one per network), a CPU-resident
//! [`RolloutBuffer`], and the
//! configuration. Call [`PpoAgent::act`] to sample one action, then push the
//! same-step data into the buffer via [`PpoAgent::record_step`]. Call
//! [`PpoAgent::finalize_rollout`] at rollout end and
//! [`PpoAgent::update`] to run the PPO optimization epochs. The full loop is
//! assembled in [`crate::algorithms::ppo::train`].

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

use crate::algorithms::ppo::losses::{
    approx_kl, clip_fraction, clipped_surrogate, clipped_value_loss, explained_variance,
    normalize_advantages, old_approx_kl, unclipped_value_loss,
};
use crate::algorithms::ppo::ppo_config::{PpoTrainingConfig, annealed_learning_rate};
use crate::algorithms::ppo::ppo_policy::PpoPolicy;
use crate::algorithms::ppo::ppo_value::PpoValue;
use crate::algorithms::ppo::rollout::{RolloutBuffer, StepEnd};

/// Error variants returned by [`PpoAgent`] operations.
#[derive(Debug, thiserror::Error)]
pub enum PpoAgentError {
    /// A tensor-to-action conversion failed.
    #[error("Tensor conversion failed: {0}")]
    TensorConversionFailed(String),
    /// The config is internally inconsistent (e.g. `minibatch_size` > `batch_size`).
    #[error("Invalid config: {0}")]
    InvalidConfig(String),
    /// An I/O error occurred while saving or loading model weights.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Env-side failure surfaced through the train loop.
    #[error("Environment error: {0}")]
    Environment(String),
}

/// Per-episode statistics emitted by the PPO training loop.
///
/// Implements [`PerformanceRecord`] so it accumulates into
/// [`AgentStats`]: `score` returns the episode reward and `duration` the
/// step count.
#[derive(Debug, Clone, Copy)]
pub struct PpoMetrics {
    /// Total reward collected during the episode.
    pub reward: f32,
    /// Number of environment steps taken.
    pub steps: usize,
    /// Most recent mean clipped-surrogate policy loss.
    pub policy_loss: f32,
    /// Most recent mean value-function loss.
    pub value_loss: f32,
    /// Most recent mean policy entropy.
    pub entropy: f32,
    /// Most recent approx-KL between old and new policies.
    pub approx_kl: f32,
    /// Most recent clip fraction (share of ratios outside `[1 − ε, 1 + ε]`).
    pub clip_frac: f32,
    /// Current learning rate (after annealing).
    pub learning_rate: f32,
}

impl PerformanceRecord for PpoMetrics {
    fn score(&self) -> f32 {
        self.reward
    }
    fn duration(&self) -> usize {
        self.steps
    }
}

/// Single-step output used by the rollout loop to step the env and push into
/// the buffer.
#[derive(Debug, Clone)]
pub struct ActOutcome {
    /// Pre-squash (Gaussian) or raw-index (Categorical) action
    /// representation — the form stored in the rollout buffer.
    pub raw_row: Vec<f32>,
    /// Representation consumed by `env.step` — post-squash, scaled for
    /// Gaussian; identical to `raw_row` for Categorical.
    pub env_row: Vec<f32>,
    /// Log-prob of the sampled action under the sampling policy.
    pub log_prob: f32,
    /// Entropy of the sampling policy at this observation.
    pub entropy: f32,
    /// Value-network prediction `V(s)` used as the bootstrap.
    pub value: f32,
}

/// Summary of one PPO update (across all epochs and minibatches).
#[derive(Debug, Clone, Copy)]
pub struct PpoUpdateStats {
    /// Mean clipped-surrogate loss across minibatches.
    pub policy_loss: f32,
    /// Mean value-function loss across minibatches.
    pub value_loss: f32,
    /// Mean entropy across minibatches.
    pub entropy: f32,
    /// Last-epoch approx-KL (Schulman k3, for `target_kl` gating).
    pub approx_kl: f32,
    /// First-minibatch pre-update approx-KL (Schulman k1, `mean(−log r)`).
    ///
    /// Captured on the very first minibatch of the first epoch, before any
    /// gradient step, so it reflects the policy that generated the rollout.
    pub old_approx_kl: f32,
    /// Mean clip fraction across minibatches.
    pub clip_frac: f32,
    /// Fraction of return variance the value net explains over this rollout.
    ///
    /// `1 − Var(returns − values) / Var(returns)`; `0.0` for a degenerate
    /// (zero-variance) rollout. See [`explained_variance`].
    pub explained_variance: f32,
    /// Number of update epochs actually completed (≤ `config.update_epochs`).
    pub epochs_run: usize,
    /// Smallest clamped `log σ` across action dims after this update, or
    /// `None` for discrete (categorical) policies, which have no `log σ`.
    ///
    /// Watch this on continuous runs: a value drifting down toward the head's
    /// `log_std_min` is the policy collapsing to a deterministic action, and
    /// because PPO's `log_std` is a single state-independent parameter,
    /// *reaching* the bound freezes it permanently — see
    /// [`TanhGaussianPolicyHead`](crate::algorithms::ppo::policies::gaussian::TanhGaussianPolicyHead).
    /// The head also emits a one-shot `tracing::warn!` when that happens.
    ///
    /// Read once per update via [`PpoPolicy::min_log_std`], which costs one
    /// device→host sync.
    pub min_log_std: Option<f32>,
}

/// Proximal Policy Optimization agent.
///
/// # Const generics
///
/// - `DO` — rank of a single observation tensor.
/// - `DB` — rank of a batched observation tensor (`= DO + 1`, typically `2`).
///
/// # Network ownership
///
/// The policy and value networks live in [`Slot`]s because Burn's
/// `Optimizer::step` consumes the module by value. Every fallible operation in
/// [`update`](Self::update) — the forward pass, the losses, `backward`, and
/// `GradientsParams::from_grads` — runs on a borrow, so a panic there leaves
/// both networks intact and the agent usable. Only a panic *inside* the
/// optimizer step itself poisons a slot; that window is irreducible and
/// terminal for the agent (see the [`shared`](crate::algorithms::shared) module
/// docs).
pub struct PpoAgent<B, P, V, O, const DO: usize, const DB: usize>
where
    B: AutodiffBackend,
    P: PpoPolicy<B, DB>,
    V: PpoValue<B, DB>,
    O: Observation<DO> + TensorConvertible<DO, B>,
{
    policy: Slot<P>,
    value: Slot<V>,
    policy_optim: OptimizerAdaptor<Adam, P, B>,
    value_optim: OptimizerAdaptor<Adam, V, B>,
    buffer: RolloutBuffer<B, O>,
    config: PpoTrainingConfig,
    device: B::Device,
    iteration: usize,
    total_iterations: usize,
    step: usize,
    stats: AgentStats<PpoMetrics>,
    /// Non-finite-loss guard for the policy-loss site (ADR 0056, #318). One
    /// per-run `warn!` latch; the skip it drives fires every occurrence.
    policy_loss_guard: FiniteLossGuard,
    /// Non-finite-loss guard for the value-loss site (ADR 0056, #318),
    /// latching independently of [`Self::policy_loss_guard`].
    value_loss_guard: FiniteLossGuard,
}

impl<B, P, V, O, const DO: usize, const DB: usize> std::fmt::Debug for PpoAgent<B, P, V, O, DO, DB>
where
    B: AutodiffBackend,
    P: PpoPolicy<B, DB>,
    V: PpoValue<B, DB>,
    O: Observation<DO> + TensorConvertible<DO, B>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PpoAgent")
            .field("iteration", &self.iteration)
            .field("step", &self.step)
            .field("buffer_len", &self.buffer.len())
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

/// Greedy / inference-only methods, separated out because they additionally
/// require `O` to convert onto the inner (non-autodiff) backend. Construction
/// and training (the main `impl` below) do not need that bound.
impl<B, P, V, O, const DO: usize, const DB: usize> PpoAgent<B, P, V, O, DO, DB>
where
    B: AutodiffBackend,
    P: PpoPolicy<B, DB>,
    V: PpoValue<B, DB>,
    O: Observation<DO> + TensorConvertible<DO, B> + TensorConvertible<DO, B::InnerBackend>,
{
    /// Snapshots the policy onto the inner (non-autodiff) backend for repeated
    /// greedy inference.
    ///
    /// Returns a frozen inference handle for use with
    /// [`act_greedy_env_row_with`](Self::act_greedy_env_row_with). Snapshot
    /// once after training, then reuse across many steps — the snapshot goes
    /// stale if the policy is updated again.
    pub fn inference_net(&self) -> P::InnerModule {
        self.policy().valid()
    }

    /// Deterministic env-space action row for `obs`, evaluated against a
    /// pre-snapshotted inner network — no sampling, no autodiff graph.
    ///
    /// This is the policy to use for evaluation and throughput benchmarking;
    /// [`act`](Self::act) samples from the stochastic policy, which injects
    /// exploration noise into the returned action.
    pub fn act_greedy_env_row_with(&self, net: &P::InnerModule, obs: &O) -> Vec<f32> {
        let obs_t: Tensor<B::InnerBackend, DO> = obs.to_tensor(&self.device);
        let batched: Tensor<B::InnerBackend, DB> = obs_t.unsqueeze::<DB>();
        P::deterministic_env_row_inner(net, batched)
    }
}

impl<B, P, V, O, const DO: usize, const DB: usize> PpoAgent<B, P, V, O, DO, DB>
where
    B: AutodiffBackend,
    P: PpoPolicy<B, DB>,
    V: PpoValue<B, DB>,
    O: Observation<DO> + TensorConvertible<DO, B>,
{
    /// Construct a new agent from a pre-built policy and value network.
    ///
    /// `total_iterations` is used for linear LR annealing. If it is zero,
    /// annealing is disabled regardless of the `anneal_lr` config flag.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`](rlevo_core::config::ConfigError) if `config`
    /// fails [`PpoTrainingConfig::validate`](rlevo_core::config::Validate::validate).
    /// In particular, v1 supports sequential rollout only, so `num_envs != 1`
    /// is rejected here.
    pub fn new(
        policy: P,
        value: V,
        config: PpoTrainingConfig,
        device: B::Device,
        total_iterations: usize,
    ) -> Result<Self, rlevo_core::config::ConfigError> {
        config.validate()?;
        let action_dim = policy.action_dim();
        let capacity = config.batch_size();

        let adam_policy = config.optimizer.clone();
        let policy_optim = match &config.clip_grad {
            Some(clip) => adam_policy
                .with_grad_clipping(Some(clip.clone()))
                .init::<B, P>(),
            None => adam_policy.init::<B, P>(),
        };
        let adam_value = config.optimizer.clone();
        let value_optim = match &config.clip_grad {
            Some(clip) => adam_value
                .with_grad_clipping(Some(clip.clone()))
                .init::<B, V>(),
            None => adam_value.init::<B, V>(),
        };

        let buffer = RolloutBuffer::new(capacity, action_dim);
        let stats = AgentStats::<PpoMetrics>::new(100);

        Ok(Self {
            policy: Slot::new(policy),
            value: Slot::new(value),
            policy_optim,
            value_optim,
            buffer,
            config,
            device,
            iteration: 0,
            total_iterations,
            step: 0,
            stats,
            policy_loss_guard: FiniteLossGuard::new("ppo/policy_loss"),
            value_loss_guard: FiniteLossGuard::new("ppo/value_loss"),
        })
    }

    /// Current iteration (one per rollout+update pair).
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Current global env-step count.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Current rollout-buffer length (steps pushed this iteration).
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Target total iterations used for LR annealing (supplied at `new`).
    pub fn total_iterations(&self) -> usize {
        self.total_iterations
    }

    /// Running per-episode statistics.
    pub fn stats(&self) -> &AgentStats<PpoMetrics> {
        &self.stats
    }

    /// Records a completed episode into the running stats.
    pub fn record_episode(&mut self, metrics: PpoMetrics) {
        self.stats.record(metrics);
    }

    /// Immutable access to the current config.
    pub fn config(&self) -> &PpoTrainingConfig {
        &self.config
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

    /// Current learning rate after annealing.
    pub fn current_learning_rate(&self) -> f64 {
        if self.config.anneal_lr {
            annealed_learning_rate(
                self.config.learning_rate,
                self.iteration,
                self.total_iterations,
            )
        } else {
            self.config.learning_rate
        }
    }

    /// Sample one action for a single observation, with the policy's log-prob
    /// and entropy plus the value-network prediction at that observation.
    ///
    /// Batched rollout is not supported in v1 (`num_envs` == 1).
    pub fn act<R: Rng + ?Sized>(&self, obs: &O, rng: &mut R) -> ActOutcome {
        let obs_t: Tensor<B, DO> = obs.to_tensor(&self.device);
        let batched: Tensor<B, DB> = obs_t.unsqueeze::<DB>();

        // Policy forward (autodiff graph is dropped on tensor drop).
        let sample = self.policy().sample_with_logprob(batched.clone(), rng);
        let raw_row = P::action_row_from_tensor(&sample.action, 0);
        let env_row = self.policy().raw_to_env_row(&raw_row);

        let log_prob = sample.log_prob.into_scalar().elem::<f32>();
        let entropy = sample.entropy.into_scalar().elem::<f32>();

        // Value forward on a fresh copy of the obs tensor.
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

    /// Pushes one step into the rollout buffer. Callers typically pair this
    /// with [`PpoAgent::act`] and the env step that consumed `action.env_row`.
    ///
    /// `next_obs` is the observation the environment just produced — the
    /// continuation state of this transition. It is only *used* when `status`
    /// is [`EpisodeStatus::Truncated`], where partial-episode bootstrapping
    /// needs `V(s_continuation)` (ADR 0048); the agent runs that one extra
    /// value forward itself, so the caller can never forget to supply it and
    /// pays nothing on ordinary steps.
    ///
    /// The caller must pass the **pre-reset** observation: the one carried by
    /// the snapshot `env.step` returned, not the one a subsequent `env.reset`
    /// produces.
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

    /// One value-network forward on a single observation.
    fn value_of(&self, obs: &O) -> f32 {
        let t: Tensor<B, DO> = obs.to_tensor(&self.device);
        let batched: Tensor<B, DB> = t.unsqueeze::<DB>();
        self.value().forward(batched).into_scalar().elem::<f32>()
    }

    /// Finalises the current rollout: computes GAE advantages and bootstrap
    /// returns from `last_obs`, which is the next observation the env would
    /// have produced had the rollout continued.
    ///
    /// `last_obs` is consulted **only** when the rollout's final step left the
    /// episode `Running`. When that step ended the episode, its own stored
    /// status supplies the bootstrap and the value forward is skipped entirely
    /// — so a caller holding a stale or post-`reset` observation on that path
    /// cannot contaminate the advantages, and the forward is not wasted.
    pub fn finalize_rollout(&mut self, last_obs: &O) {
        let last_value = if self.buffer.last_step_ended() {
            0.0
        } else {
            self.value_of(last_obs)
        };
        self.buffer
            .finish(last_value, self.config.gamma, self.config.gae_lambda);
    }

    /// Runs `update_epochs × num_minibatches` gradient updates on the current
    /// rollout, applies LR annealing, then clears the buffer. Returns summary
    /// statistics used to populate [`PpoMetrics`].
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
    pub fn update<R: Rng + ?Sized>(&mut self, rng: &mut R) -> PpoUpdateStats {
        let batch_size = self.buffer.len();
        let mb_size = (batch_size / self.config.num_minibatches.max(1)).max(1);
        let lr = self.current_learning_rate();

        let mut policy_loss_acc = 0.0_f32;
        let mut value_loss_acc = 0.0_f32;
        let mut entropy_acc = 0.0_f32;
        let mut clip_frac_acc = 0.0_f32;
        let mut last_kl = 0.0_f32;
        let mut first_old_kl = 0.0_f32;
        let mut mb_count = 0_usize;
        let mut epochs_run = 0_usize;
        // Count only the minibatches whose loss was finite and therefore
        // actually applied — the denominators for the two gated means (#318,
        // ADR 0056 §3). `mb_count` still denominates the ungated diagnostics.
        let mut policy_healthy = 0_usize;
        let mut value_healthy = 0_usize;

        let obs_shape = O::shape();
        let numel_per_obs: usize = obs_shape.iter().product();

        'epochs: for _epoch in 0..self.config.update_epochs {
            epochs_run += 1;
            let indices = self.buffer.indices_shuffled(rng);
            let mut kl_sum = 0.0_f32;
            let mut kl_count = 0_usize;

            for chunk in indices.chunks(mb_size) {
                if chunk.is_empty() {
                    continue;
                }
                let n = chunk.len();

                // Stage minibatch observations host-side, then upload once
                // below: `to_tensor` here would push each row to the device
                // only to read it straight back, with no op in between.
                let mut obs_flat: Vec<f32> = Vec::with_capacity(n * numel_per_obs);
                for &i in chunk {
                    self.buffer.obs()[i].write_host_row(&mut obs_flat);
                }
                let mut batched_shape: Vec<usize> = Vec::with_capacity(DB);
                batched_shape.push(n);
                batched_shape.extend_from_slice(&obs_shape);
                let obs_batch: Tensor<B, DB> =
                    Tensor::from_data(TensorData::new(obs_flat, batched_shape), &self.device);

                // Action tensor.
                let action_flat = self.buffer.gather_action_flat(chunk);
                let actions: P::ActionTensor =
                    P::action_tensor_from_flat(&action_flat, n, &self.device);

                // Old log_probs, old_values, returns, advantages.
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
                let advs = if self.config.normalize_advantages {
                    normalize_advantages(advs_raw)
                } else {
                    advs_raw
                };

                // ----- Policy update -----
                // Everything up to `step_with` runs on a borrow, so a panic in
                // the forward/loss/backward region leaves the slot intact.
                let eval = self.policy().evaluate(obs_batch.clone(), actions);
                let pg = clipped_surrogate(
                    eval.log_prob.clone(),
                    old_lp.clone(),
                    advs,
                    self.config.clip_coef,
                );
                let entropy_mean = eval.entropy.mean();
                let policy_loss =
                    pg.clone() - entropy_mean.clone().mul_scalar(self.config.entropy_coef);
                let policy_loss_val = policy_loss.clone().into_scalar().elem::<f32>();

                // Diagnostics before backward (fresh copies).
                let kl = approx_kl(eval.log_prob.clone(), old_lp.clone());
                let cf =
                    clip_fraction(eval.log_prob.clone(), old_lp.clone(), self.config.clip_coef);
                // Capture the pre-update (k1) KL on the very first minibatch,
                // before any gradient step has perturbed the policy.
                if mb_count == 0 {
                    first_old_kl = old_approx_kl(eval.log_prob.clone(), old_lp.clone());
                }
                kl_sum += kl;
                kl_count += 1;

                // #318 / ADR 0056: `policy_loss_val` is already host-resident,
                // so the finiteness check costs no extra sync. A non-finite
                // loss skips `backward()` + the optimizer step (Burn would
                // otherwise fold NaN into the weights silently) and is excluded
                // from the reported mean.
                if self.policy_loss_guard.check(policy_loss_val) {
                    let grads = policy_loss.backward();
                    let grads_params = GradientsParams::from_grads(grads, self.policy());
                    self.policy
                        .step_with(&mut self.policy_optim, lr, grads_params);
                    policy_loss_acc += policy_loss_val;
                    policy_healthy += 1;
                }

                // ----- Value update -----
                let new_v = self.value().forward(obs_batch);
                let v_loss = if self.config.clip_value_loss {
                    clipped_value_loss(new_v, old_v, returns, self.config.clip_coef)
                } else {
                    unclipped_value_loss(new_v, returns)
                };
                let v_loss_scaled = v_loss.clone().mul_scalar(self.config.value_coef);
                let v_loss_val = v_loss.into_scalar().elem::<f32>();
                // #318 / ADR 0056: same guard for the value site, latched
                // separately so one site's failure cannot silence the other.
                if self.value_loss_guard.check(v_loss_val) {
                    let grads = v_loss_scaled.backward();
                    let grads_params = GradientsParams::from_grads(grads, self.value());
                    self.value
                        .step_with(&mut self.value_optim, lr, grads_params);
                    value_loss_acc += v_loss_val;
                    value_healthy += 1;
                }

                // Accumulate the ungated per-minibatch diagnostics.
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

            if let Some(target) = self.config.target_kl
                && mean_kl > 1.5 * target
            {
                break 'epochs;
            }
        }

        // Value-net health over the whole rollout, computed before the buffer
        // is cleared. Returns/values are the CPU-resident slices the buffer
        // already holds, so this adds no forward pass.
        let ev = explained_variance(self.buffer.returns(), self.buffer.values());

        // Policy-scale telemetry: one device→host sync per *update* (not per
        // step). This is also where the Gaussian head evaluates its one-shot
        // "the log_std clamp has bound" warning — deliberately here rather than
        // in the forward pass, which would sync on every rollout step.
        let min_log_std = self.policy().min_log_std();

        self.buffer.clear();
        self.iteration += 1;

        // Ungated diagnostics divide by the total minibatch count; the two
        // gated losses divide by their own healthy counts, reporting `0.0`
        // (not NaN) when every minibatch skipped (#318, ADR 0056 §3).
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
            old_approx_kl: first_old_kl,
            clip_frac: clip_frac_acc / denom,
            explained_variance: ev,
            epochs_run,
            min_log_std,
        }
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

    use crate::algorithms::ppo::policies::CategoricalPolicyHeadConfig;
    use crate::algorithms::ppo::policies::categorical::CategoricalPolicyHead;
    use crate::algorithms::ppo::ppo_config::PpoTrainingConfigBuilder;

    type TestBackend = Autodiff<Flex>;
    type TestAgent = PpoAgent<
        TestBackend,
        CategoricalPolicyHead<TestBackend>,
        TestValue<TestBackend>,
        TestObs,
        1,
        2,
    >;

    /// Minimal scalar value head: `PpoAgent` requires a `PpoValue`
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

    /// Builds an agent whose `clip_grad` is exactly `clip_grad`, leaving every
    /// other knob at its default.
    fn agent_with_clip_grad(clip_grad: Option<GradientClippingConfig>) -> TestAgent {
        let device = Default::default();
        let policy: CategoricalPolicyHead<TestBackend> = CategoricalPolicyHeadConfig {
            obs_dim: 2,
            hidden: 4,
            num_actions: 2,
        }
        .try_init::<TestBackend>(&device)
        .expect("valid head config");
        let value = TestValue::<TestBackend>::init(&device);
        let config = PpoTrainingConfigBuilder::new()
            .clip_grad(clip_grad)
            .build()
            .expect("valid config");
        TestAgent::new(policy, value, config, device, 1).expect("valid config")
    }

    /// Regression (issue #183): `clip_grad` must reach *both* optimizers.
    ///
    /// The sibling field `max_grad_norm` was dead `pub` state that no lint
    /// could catch — clippy's `dead_code` does not fire on unread `pub` fields
    /// of `pub` types. Only a wiring assertion like this one closes that gap,
    /// so assert the plumbing rather than Burn's `clip_by_norm` arithmetic.
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
    /// of a non-finite policy or value loss. Applied to a *clone* so the live
    /// net's `ParamId`s are preserved.
    struct NanInjector;

    impl<B: Backend> ModuleMapper<B> for NanInjector {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
            let (id, tensor, mapper) = param.consume();
            Param::from_mapped_value(id, tensor.mul_scalar(f32::NAN), mapper)
        }
    }

    /// Reads the value head's weights back to the host, element-wise, so a
    /// change across an update proves the value net actually stepped.
    fn value_weights<B: Backend>(v: &TestValue<B>) -> Vec<f32> {
        v.head
            .weight
            .val()
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("f32 weight host read")
    }

    /// Builds an agent whose one four-step rollout has been collected and
    /// finalized (advantages/returns populated) with *healthy* networks, so a
    /// subsequent `update` can be driven directly after poisoning one net.
    // Test fixture data: the loop counter is bounded by a small constant, far
    // below f32's 2^24 exact-integer limit.
    #[allow(clippy::cast_precision_loss)]
    fn primed_ppo_agent() -> TestAgent {
        let device = Default::default();
        let policy: CategoricalPolicyHead<TestBackend> = CategoricalPolicyHeadConfig {
            obs_dim: 2,
            hidden: 4,
            num_actions: 2,
        }
        .try_init::<TestBackend>(&device)
        .expect("valid head config");
        let value = TestValue::<TestBackend>::init(&device);
        let config = PpoTrainingConfigBuilder::new()
            .num_envs(1)
            .num_steps(4)
            .num_minibatches(1)
            .update_epochs(1)
            .anneal_lr(false)
            .build()
            .expect("valid config");
        let mut agent = TestAgent::new(policy, value, config, device, 1).expect("valid config");

        let mut rng = StdRng::seed_from_u64(0);
        let mut last = TestObs([0.4, 0.6]);
        for i in 0..4usize {
            let x = i as f32 * 0.1;
            let obs = TestObs([x, 1.0 - x]);
            let next = TestObs([x + 0.1, 0.9 - x]);
            let outcome = agent.act(&obs, &mut rng);
            agent.record_step(obs, &outcome, 1.0, &next, EpisodeStatus::Running);
            last = next;
        }
        agent.finalize_rollout(&last);
        agent
    }

    /// A non-finite policy loss must skip the policy `backward` + optimizer step
    /// and be excluded from the reported mean (ADR 0056, #318). Diverging the
    /// policy net to NaN forces a NaN clipped-surrogate loss; the guard must
    /// fire, the reported `policy_loss` must be the `0.0` all-skipped sentinel
    /// (never a propagated NaN), and the independent value site must still learn.
    #[test]
    #[allow(clippy::float_cmp)]
    fn ppo_nonfinite_policy_loss_skips_and_warns() {
        let mut agent = primed_ppo_agent();
        let value_before = value_weights(agent.value.get());

        // Poison a *clone* of the live policy so its `ParamId`s are preserved.
        let poisoned = agent.policy.get().clone().map(&mut NanInjector);
        agent.policy = Slot::new(poisoned);

        let mut rng = StdRng::seed_from_u64(1);
        let stats = agent.update(&mut rng);

        assert!(
            agent.policy_loss_guard.warning_fired(),
            "a NaN policy loss must fire the policy guard"
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

    /// The mirror of [`ppo_nonfinite_policy_loss_skips_and_warns`] for the value
    /// site: a NaN value loss must fire the value guard, report the `0.0`
    /// all-skipped sentinel, and leave the independent policy site to learn
    /// normally.
    #[test]
    #[allow(clippy::float_cmp)]
    fn ppo_nonfinite_value_loss_skips_and_warns() {
        let mut agent = primed_ppo_agent();

        // Poison a *clone* of the live value net so its `ParamId`s are preserved.
        let poisoned = agent.value.get().clone().map(&mut NanInjector);
        agent.value = Slot::new(poisoned);

        let mut rng = StdRng::seed_from_u64(2);
        let stats = agent.update(&mut rng);

        assert!(
            agent.value_loss_guard.warning_fired(),
            "a NaN value loss must fire the value guard"
        );
        assert!(
            !agent.policy_loss_guard.warning_fired(),
            "the policy net was healthy: its guard must not fire"
        );
        assert_eq!(
            stats.value_loss, 0.0,
            "every value minibatch skipped → reported 0.0, never a propagated NaN"
        );
        assert!(
            stats.policy_loss.is_finite(),
            "the healthy policy site must still report a finite loss"
        );
    }
}
