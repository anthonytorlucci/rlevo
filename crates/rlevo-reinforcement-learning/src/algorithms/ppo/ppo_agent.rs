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
use burn::optim::{Adam, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{ElementConversion, Tensor, TensorData};
use rand::Rng;

use rlevo_core::base::{Observation, TensorConvertible};
use rlevo_core::environment::EpisodeStatus;
use crate::metrics::{AgentStats, PerformanceRecord};

use crate::algorithms::ppo::losses::{
    approx_kl, clip_fraction, clipped_surrogate, clipped_value_loss, normalize_advantages,
    unclipped_value_loss,
};
use crate::algorithms::ppo::ppo_config::{PpoTrainingConfig, annealed_learning_rate};
use crate::algorithms::ppo::ppo_policy::PpoPolicy;
use crate::algorithms::ppo::ppo_value::PpoValue;
use crate::algorithms::ppo::rollout::RolloutBuffer;

/// Error variants returned by [`PpoAgent`] operations.
#[derive(Debug, thiserror::Error)]
pub enum PpoAgentError {
    /// A tensor-to-action conversion failed.
    #[error("Tensor conversion failed: {0}")]
    TensorConversionFailed(String),
    /// The config is internally inconsistent (e.g. minibatch_size > batch_size).
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
    /// Last-epoch approx-KL (for `target_kl` gating).
    pub approx_kl: f32,
    /// Mean clip fraction across minibatches.
    pub clip_frac: f32,
    /// Number of update epochs actually completed (≤ `config.update_epochs`).
    pub epochs_run: usize,
}

/// Proximal Policy Optimization agent.
///
/// # Const generics
///
/// - `DO` — rank of a single observation tensor.
/// - `DB` — rank of a batched observation tensor (`= DO + 1`, typically `2`).
pub struct PpoAgent<B, P, V, O, const DO: usize, const DB: usize>
where
    B: AutodiffBackend,
    P: PpoPolicy<B, DB>,
    V: PpoValue<B, DB>,
    O: Observation<DO> + TensorConvertible<DO, B>,
{
    policy: Option<P>,
    value: Option<V>,
    policy_optim: OptimizerAdaptor<Adam, P, B>,
    value_optim: OptimizerAdaptor<Adam, V, B>,
    buffer: RolloutBuffer<B, O>,
    config: PpoTrainingConfig,
    device: B::Device,
    iteration: usize,
    total_iterations: usize,
    step: usize,
    stats: AgentStats<PpoMetrics>,
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
            .finish()
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
    pub fn new(
        policy: P,
        value: V,
        config: PpoTrainingConfig,
        device: B::Device,
        total_iterations: usize,
    ) -> Self {
        assert_eq!(
            config.num_envs, 1,
            "v1 supports sequential rollout only (num_envs == 1)"
        );
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

        Self {
            policy: Some(policy),
            value: Some(value),
            policy_optim,
            value_optim,
            buffer,
            config,
            device,
            iteration: 0,
            total_iterations,
            step: 0,
            stats,
        }
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

    fn policy(&self) -> &P {
        self.policy
            .as_ref()
            .expect("policy not restored — earlier panic in learn_step?")
    }

    fn value(&self) -> &V {
        self.value
            .as_ref()
            .expect("value is populated outside of transient step() calls")
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
    /// Batched rollout is not supported in v1 (num_envs == 1).
    pub fn act(&self, obs: &O, rng: &mut dyn Rng) -> ActOutcome {
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

    /// Finalises the current rollout: computes GAE advantages and bootstrap
    /// returns from `last_obs`, which is the next observation the env would
    /// have produced had the rollout continued.
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
            self.config.gamma,
            self.config.gae_lambda,
        );
    }

    /// Runs `update_epochs × num_minibatches` gradient updates on the current
    /// rollout, applies LR annealing, then clears the buffer. Returns summary
    /// statistics used to populate [`PpoMetrics`].
    pub fn update<R: Rng + ?Sized>(&mut self, rng: &mut R) -> PpoUpdateStats {
        let batch_size = self.buffer.len();
        let mb_size = (batch_size / self.config.num_minibatches.max(1)).max(1);
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

                // Materialise minibatch observations on device.
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
                let policy = self
                    .policy
                    .take()
                    .expect("policy not restored — earlier panic in learn_step?");
                let eval = policy.evaluate(obs_batch.clone(), actions);
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
                kl_sum += kl;
                kl_count += 1;

                let grads = policy_loss.backward();
                let grads_params = GradientsParams::from_grads(grads, &policy);
                let updated_policy = self.policy_optim.step(lr, policy, grads_params);
                self.policy = Some(updated_policy);

                // ----- Value update -----
                let value = self.value.take().expect("value present");
                let new_v = value.forward(obs_batch);
                let v_loss = if self.config.clip_value_loss {
                    clipped_value_loss(new_v, old_v, returns, self.config.clip_coef)
                } else {
                    unclipped_value_loss(new_v, returns)
                };
                let v_loss_scaled = v_loss.clone().mul_scalar(self.config.value_coef);
                let v_loss_val = v_loss.into_scalar().elem::<f32>();
                let grads = v_loss_scaled.backward();
                let grads_params = GradientsParams::from_grads(grads, &value);
                let updated_value = self.value_optim.step(lr, value, grads_params);
                self.value = Some(updated_value);

                // Accumulate.
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

            if let Some(target) = self.config.target_kl
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
}
