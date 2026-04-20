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
use burn::optim::{Adam, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{ElementConversion, Tensor, TensorData};
use rand::Rng;

use evorl_core::base::{Observation, TensorConvertible};
use evorl_core::environment::EpisodeStatus;
use evorl_core::metrics::{AgentStats, PerformanceRecord};

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
use crate::algorithms::ppo::rollout::RolloutBuffer;

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
    pub reward: f32,
    pub steps: usize,
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
    pub approx_kl: f32,
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
pub struct PpgAgent<B, P, V, O, const DO: usize, const DB: usize>
where
    B: AutodiffBackend,
    P: PpoPolicy<B, DB> + PpgAuxValueHead<B, DB>,
    V: PpoValue<B, DB>,
    O: Observation<DO> + TensorConvertible<DO, B>,
{
    policy: Option<P>,
    value: Option<V>,
    policy_optim: OptimizerAdaptor<Adam, P, B>,
    value_optim: OptimizerAdaptor<Adam, V, B>,
    buffer: RolloutBuffer<B, O>,
    aux_buffer: AuxRolloutBuffer<B, O>,
    config: PpgConfig,
    device: B::Device,
    iteration: usize,
    total_iterations: usize,
    step: usize,
    stats: AgentStats<PpgMetrics>,
    last_aux: Option<AuxPhaseStats>,
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
            .field("step", &self.step)
            .field("buffer_len", &self.buffer.len())
            .field("aux_slices", &self.aux_buffer.num_slices())
            .field("config", &self.config)
            .finish()
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
    pub fn new(
        policy: P,
        value: V,
        config: PpgConfig,
        device: B::Device,
        total_iterations: usize,
    ) -> Self {
        assert_eq!(
            config.ppo.num_envs, 1,
            "v1 supports sequential rollout only (num_envs == 1)"
        );
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

        Self {
            policy: Some(policy),
            value: Some(value),
            policy_optim,
            value_optim,
            buffer,
            aux_buffer,
            config,
            device,
            iteration: 0,
            total_iterations,
            step: 0,
            stats,
            last_aux: None,
        }
    }

    pub fn iteration(&self) -> usize {
        self.iteration
    }
    pub fn step(&self) -> usize {
        self.step
    }
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }
    pub fn aux_buffer_slices(&self) -> usize {
        self.aux_buffer.num_slices()
    }
    pub fn total_iterations(&self) -> usize {
        self.total_iterations
    }
    pub fn stats(&self) -> &AgentStats<PpgMetrics> {
        &self.stats
    }
    pub fn record_episode(&mut self, metrics: PpgMetrics) {
        self.stats.record(metrics);
    }
    pub fn config(&self) -> &PpgConfig {
        &self.config
    }
    pub fn last_aux_phase(&self) -> Option<AuxPhaseStats> {
        self.last_aux
    }

    fn policy(&self) -> &P {
        self.policy
            .as_ref()
            .expect("policy not restored — earlier panic in learn_step?")
    }

    fn value(&self) -> &V {
        self.value
            .as_ref()
            .expect("value is populated outside transient step() calls")
    }

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

    /// Sample one action for a single observation.
    pub fn act(&self, obs: &O, rng: &mut dyn Rng) -> ActOutcome {
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

    /// Pushes one step into the rollout buffer.
    pub fn record_step(&mut self, obs: O, action: &ActOutcome, reward: f32, status: EpisodeStatus) {
        self.buffer.push_step(
            obs,
            &action.raw_row,
            action.log_prob,
            action.value,
            reward,
            status,
        );
        self.step += 1;
    }

    /// Computes GAE advantages/returns from `last_obs`. Identical to PPO's
    /// finalize step.
    pub fn finalize_rollout(&mut self, last_obs: &O, last_done: bool) {
        let last_t: Tensor<B, DO> = last_obs.to_tensor(&self.device);
        let last_batched: Tensor<B, DB> = last_t.unsqueeze::<DB>();
        let last_value = self
            .value()
            .forward(last_batched)
            .into_scalar()
            .elem::<f32>();
        self.buffer.finish(
            last_value,
            last_done,
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
    pub fn policy_phase_update<R: Rng + ?Sized>(&mut self, rng: &mut R) -> PpoUpdateStats {
        let cfg = self.config.ppo.clone();
        let batch_size = self.buffer.len();
        let mb_size = (batch_size / cfg.num_minibatches.max(1)).max(1);
        let lr = self.current_learning_rate();

        let mut policy_loss_acc = 0.0_f32;
        let mut value_loss_acc = 0.0_f32;
        let mut entropy_acc = 0.0_f32;
        let mut clip_frac_acc = 0.0_f32;
        let mut last_kl = 0.0_f32;
        let mut mb_count = 0_usize;
        let mut epochs_run = 0_usize;

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
                    let t: Tensor<B, DO> = self.buffer.obs()[i].to_tensor(&self.device);
                    let data = t.into_data().convert::<f32>();
                    obs_flat.extend_from_slice(data.as_slice::<f32>().expect("f32 obs"));
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

                // Policy update.
                let policy = self.policy.take().expect("policy not restored — earlier panic in learn_step?");
                let eval = policy.evaluate(obs_batch.clone(), actions);
                let pg =
                    clipped_surrogate(eval.log_prob.clone(), old_lp.clone(), advs, cfg.clip_coef);
                let entropy_mean = eval.entropy.mean();
                let policy_loss = pg.clone() - entropy_mean.clone().mul_scalar(cfg.entropy_coef);
                let policy_loss_val = policy_loss.clone().into_scalar().elem::<f32>();

                let kl = approx_kl(eval.log_prob.clone(), old_lp.clone());
                let cf = clip_fraction(eval.log_prob.clone(), old_lp, cfg.clip_coef);
                kl_sum += kl;
                kl_count += 1;

                let grads = policy_loss.backward();
                let grads_params = GradientsParams::from_grads(grads, &policy);
                let updated_policy = self.policy_optim.step(lr, policy, grads_params);
                self.policy = Some(updated_policy);

                // Value update.
                let value = self.value.take().expect("value present");
                let new_v = value.forward(obs_batch);
                let v_loss = if cfg.clip_value_loss {
                    clipped_value_loss(new_v, old_v, returns, cfg.clip_coef)
                } else {
                    unclipped_value_loss(new_v, returns)
                };
                let v_loss_scaled = v_loss.clone().mul_scalar(cfg.value_coef);
                let v_loss_val = v_loss.into_scalar().elem::<f32>();
                let grads = v_loss_scaled.backward();
                let grads_params = GradientsParams::from_grads(grads, &value);
                let updated_value = self.value_optim.step(lr, value, grads_params);
                self.value = Some(updated_value);

                policy_loss_acc += policy_loss_val;
                value_loss_acc += v_loss_val;
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

        self.buffer.clear();
        self.iteration += 1;

        let denom = mb_count.max(1) as f32;
        PpoUpdateStats {
            policy_loss: policy_loss_acc / denom,
            value_loss: value_loss_acc / denom,
            entropy: entropy_acc / denom,
            approx_kl: last_kl,
            clip_frac: clip_frac_acc / denom,
            epochs_run,
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
    pub fn maybe_aux_phase<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Option<AuxPhaseStats> {
        if !self.aux_buffer.is_ready(self.config.n_iteration) {
            return None;
        }
        let cfg = self.config.clone();
        let lr = self.current_learning_rate();
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

                // Main value-net update.
                let value = self.value.take().expect("value present");
                let new_v = value.forward(obs_t.clone());
                let v_loss = unclipped_value_loss(new_v, returns_t.clone());
                let v_loss_val = v_loss.clone().into_scalar().elem::<f32>();
                let grads = v_loss.backward();
                let grads_params = GradientsParams::from_grads(grads, &value);
                let updated_value = self.value_optim.step(lr, value, grads_params);
                self.value = Some(updated_value);

                // Policy-net update: aux-value MSE + β · KL distillation.
                let policy = self.policy.take().expect("policy not restored — earlier panic in learn_step?");
                let aux_v_pred = PpgAuxValueHead::aux_value(&policy, obs_t.clone());
                let aux_v_loss = unclipped_value_loss(aux_v_pred, returns_t);
                let new_logits = PpgAuxValueHead::logits(&policy, obs_t);
                let kl = policy_kl_categorical(old_logits_t, new_logits);
                let aux_v_loss_val = aux_v_loss.clone().into_scalar().elem::<f32>();
                let kl_val = kl.clone().into_scalar().elem::<f32>();
                let total = aux_v_loss + kl.mul_scalar(cfg.beta_clone);
                let grads = total.backward();
                let grads_params = GradientsParams::from_grads(grads, &policy);
                let updated_policy = self.policy_optim.step(lr, policy, grads_params);
                self.policy = Some(updated_policy);

                main_v_acc += v_loss_val;
                aux_v_acc += aux_v_loss_val;
                kl_acc += kl_val;
                mb_count += 1;
            }
        }

        self.aux_buffer.clear();

        let denom = mb_count.max(1) as f32;
        let stats = AuxPhaseStats {
            main_value_loss: main_v_acc / denom,
            aux_value_loss: aux_v_acc / denom,
            policy_kl: kl_acc / denom,
            epochs_run,
            minibatches: mb_count,
        };
        self.last_aux = Some(stats);
        Some(stats)
    }

    /// Computes π_old logits for every step in the auxiliary buffer.
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
                let t: Tensor<B, DO> = self.aux_buffer.obs_at(g).to_tensor(&self.device);
                let data = t.into_data().convert::<f32>();
                obs_flat.extend_from_slice(data.as_slice::<f32>().expect("f32 obs"));
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
