//! Hyperparameter configuration for the PPO (Proximal Policy Optimization) algorithm.
//!
//! Field defaults follow CleanRL's [`ppo.py`](https://docs.cleanrl.dev/rl-algorithms/ppo/)
//! and [`ppo_continuous_action.py`]. See
//! [Huang et al. 2022, *The 37 Implementation Details of PPO*](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
//! for the rationale behind each value.
//!
//! [`ppo_continuous_action.py`]: https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy

use burn::grad_clipping::GradientClippingConfig;
use burn::optim::AdamConfig;

/// Configuration for training a PPO agent.
///
/// Covers rollout sizing, optimization, objective weights, and (for continuous
/// action spaces) the policy-head scale. One `PpoAgent` instance is
/// parameterised by the same config regardless of whether the env is discrete
/// or continuous; the continuous-specific fields are simply ignored when the
/// plugged-in policy head is categorical.
#[derive(Clone, Debug)]
pub struct PpoTrainingConfig {
    // ----- rollout sizing -----
    /// Number of environments stepped in parallel.
    ///
    /// v1 only supports sequential rollout (`num_envs == 1`). Vectorised rollout
    /// is deferred to a future release.
    pub num_envs: usize,

    /// Rollout horizon per env (steps collected before each update).
    ///
    /// Default `128` matches CleanRL's `ppo.py`.
    pub num_steps: usize,

    // ----- optimization -----
    /// Number of minibatches the rollout is split into per update epoch.
    pub num_minibatches: usize,

    /// Number of update epochs per rollout.
    pub update_epochs: usize,

    /// Base learning rate passed to Adam.
    pub learning_rate: f64,

    /// When `true`, linearly anneal `learning_rate` to `0` across the total
    /// number of iterations.
    pub anneal_lr: bool,

    /// Global gradient-norm clip applied to each loss.backward() result.
    ///
    /// CleanRL uses `0.5`.
    pub max_grad_norm: f32,

    /// Underlying optimizer config. Adam epsilon defaults to `1e-5` when the
    /// agent is constructed.
    pub optimizer: AdamConfig,

    /// Optional gradient-clipping config. When set, the Burn optimizer wraps
    /// the grads with this clip. Independent of `max_grad_norm`, which is
    /// applied manually in the agent loop.
    pub clip_grad: Option<GradientClippingConfig>,

    // ----- objective -----
    /// Discount factor γ.
    pub gamma: f32,

    /// GAE bootstrap parameter λ.
    pub gae_lambda: f32,

    /// PPO clipping coefficient ε (applied symmetrically as `[1−ε, 1+ε]`).
    pub clip_coef: f32,

    /// When `true`, value-function targets use the clipped loss
    /// `max((v_clipped − R)², (v − R)²)`. CleanRL default: on.
    pub clip_value_loss: bool,

    /// Entropy bonus coefficient. `0.01` is the discrete default; use `0.0`
    /// for continuous envs unless entropy-driven exploration helps.
    pub entropy_coef: f32,

    /// Value-loss coefficient `c_v`.
    pub value_coef: f32,

    /// When `true`, advantages are standardised batch-wise before being
    /// multiplied into the surrogate objective.
    pub normalize_advantages: bool,

    /// Optional early-stop target for the approximate KL divergence. When
    /// `Some(k)`, the update epoch loop aborts as soon as the running
    /// mean-approx-KL exceeds `1.5 · k`.
    pub target_kl: Option<f32>,

    // ----- continuous-only (ignored for categorical policies) -----
    /// Initial value of the state-independent `log_std` parameter used by the
    /// tanh-squashed Gaussian policy head.
    pub action_log_std_init: f32,

    /// Scale applied to the tanh-squashed action before it reaches the
    /// environment. Set to match the env's action-bound magnitude (for
    /// Pendulum-v1 with `max_torque = 2.0`, use `2.0`).
    pub action_scale: f32,
}

impl PpoTrainingConfig {
    /// Size of a single minibatch under the configured rollout.
    ///
    /// `(num_envs · num_steps) / num_minibatches`.
    #[must_use]
    pub fn minibatch_size(&self) -> usize {
        self.batch_size() / self.num_minibatches.max(1)
    }

    /// Total transitions per rollout: `num_envs · num_steps`.
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.num_envs * self.num_steps
    }
}

impl Default for PpoTrainingConfig {
    fn default() -> Self {
        let mut adam = AdamConfig::new();
        adam = adam.with_epsilon(1e-5);
        Self {
            num_envs: 1,
            num_steps: 128,
            num_minibatches: 4,
            update_epochs: 4,
            learning_rate: 2.5e-4,
            anneal_lr: true,
            max_grad_norm: 0.5,
            optimizer: adam,
            clip_grad: None,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_coef: 0.2,
            clip_value_loss: true,
            entropy_coef: 0.01,
            value_coef: 0.5,
            normalize_advantages: true,
            target_kl: None,
            action_log_std_init: 0.0,
            action_scale: 1.0,
        }
    }
}

/// Fluent builder for [`PpoTrainingConfig`]. All unset fields fall back to
/// [`PpoTrainingConfig::default`].
#[derive(Debug)]
pub struct PpoTrainingConfigBuilder {
    config: PpoTrainingConfig,
}

impl Default for PpoTrainingConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PpoTrainingConfigBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: PpoTrainingConfig::default(),
        }
    }

    pub fn num_envs(mut self, num_envs: usize) -> Self {
        self.config.num_envs = num_envs;
        self
    }

    pub fn num_steps(mut self, num_steps: usize) -> Self {
        self.config.num_steps = num_steps;
        self
    }

    pub fn num_minibatches(mut self, num_minibatches: usize) -> Self {
        self.config.num_minibatches = num_minibatches;
        self
    }

    pub fn update_epochs(mut self, update_epochs: usize) -> Self {
        self.config.update_epochs = update_epochs;
        self
    }

    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }

    pub fn anneal_lr(mut self, anneal_lr: bool) -> Self {
        self.config.anneal_lr = anneal_lr;
        self
    }

    pub fn max_grad_norm(mut self, max_grad_norm: f32) -> Self {
        self.config.max_grad_norm = max_grad_norm;
        self
    }

    pub fn optimizer(mut self, optimizer: AdamConfig) -> Self {
        self.config.optimizer = optimizer;
        self
    }

    pub fn clip_grad(mut self, clip_grad: Option<GradientClippingConfig>) -> Self {
        self.config.clip_grad = clip_grad;
        self
    }

    pub fn gamma(mut self, gamma: f32) -> Self {
        self.config.gamma = gamma;
        self
    }

    pub fn gae_lambda(mut self, gae_lambda: f32) -> Self {
        self.config.gae_lambda = gae_lambda;
        self
    }

    pub fn clip_coef(mut self, clip_coef: f32) -> Self {
        self.config.clip_coef = clip_coef;
        self
    }

    pub fn clip_value_loss(mut self, clip_value_loss: bool) -> Self {
        self.config.clip_value_loss = clip_value_loss;
        self
    }

    pub fn entropy_coef(mut self, entropy_coef: f32) -> Self {
        self.config.entropy_coef = entropy_coef;
        self
    }

    pub fn value_coef(mut self, value_coef: f32) -> Self {
        self.config.value_coef = value_coef;
        self
    }

    pub fn normalize_advantages(mut self, normalize_advantages: bool) -> Self {
        self.config.normalize_advantages = normalize_advantages;
        self
    }

    pub fn target_kl(mut self, target_kl: Option<f32>) -> Self {
        self.config.target_kl = target_kl;
        self
    }

    pub fn action_log_std_init(mut self, v: f32) -> Self {
        self.config.action_log_std_init = v;
        self
    }

    pub fn action_scale(mut self, v: f32) -> Self {
        self.config.action_scale = v;
        self
    }

    pub fn build(self) -> PpoTrainingConfig {
        self.config
    }
}

/// Computes the linearly-annealed learning rate at `iteration` out of
/// `total_iterations`.
///
/// At `iteration == 0`, returns `base_lr`; at `iteration == total_iterations`,
/// returns `0`. Linear interpolation between.
#[must_use]
pub fn annealed_learning_rate(base_lr: f64, iteration: usize, total_iterations: usize) -> f64 {
    if total_iterations == 0 {
        return base_lr;
    }
    let frac = 1.0 - (iteration as f64) / (total_iterations as f64);
    base_lr * frac.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_cleanrl() {
        let cfg = PpoTrainingConfig::default();
        assert_eq!(cfg.num_envs, 1);
        assert_eq!(cfg.num_steps, 128);
        assert_eq!(cfg.num_minibatches, 4);
        assert_eq!(cfg.update_epochs, 4);
        assert!((cfg.learning_rate - 2.5e-4).abs() < 1e-12);
        assert_eq!(cfg.clip_coef, 0.2);
        assert_eq!(cfg.gae_lambda, 0.95);
        assert_eq!(cfg.gamma, 0.99);
    }

    #[test]
    fn batch_and_minibatch_sizes() {
        let cfg = PpoTrainingConfigBuilder::new()
            .num_envs(1)
            .num_steps(128)
            .num_minibatches(4)
            .build();
        assert_eq!(cfg.batch_size(), 128);
        assert_eq!(cfg.minibatch_size(), 32);
    }

    #[test]
    fn lr_anneals_to_zero() {
        let total = 100;
        assert!((annealed_learning_rate(1.0, 0, total) - 1.0).abs() < 1e-12);
        assert!((annealed_learning_rate(1.0, 100, total) - 0.0).abs() < 1e-12);
        assert!((annealed_learning_rate(1.0, 50, total) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn lr_anneal_clamped_at_zero_past_end() {
        assert!((annealed_learning_rate(1.0, 200, 100) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn builder_round_trips_fields() {
        let cfg = PpoTrainingConfigBuilder::new()
            .num_steps(256)
            .clip_coef(0.1)
            .entropy_coef(0.0)
            .action_scale(2.0)
            .build();
        assert_eq!(cfg.num_steps, 256);
        assert_eq!(cfg.clip_coef, 0.1);
        assert_eq!(cfg.entropy_coef, 0.0);
        assert_eq!(cfg.action_scale, 2.0);
    }
}
