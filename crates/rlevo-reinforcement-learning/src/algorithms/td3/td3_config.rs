//! Hyperparameter configuration for the TD3 algorithm.
//!
//! Fields and defaults track CleanRL's `td3_continuous_action.py` so that
//! reproducing published Pendulum/MuJoCo numbers is a matter of plugging the
//! same values in. TD3 extends DDPG with two extra knobs on top of
//! [`DdpgTrainingConfig`](crate::algorithms::ddpg::ddpg_config::DdpgTrainingConfig):
//! Gaussian target-policy-smoothing noise σ (`policy_noise`) and its
//! symmetric clip range (`noise_clip`).

use burn::grad_clipping::GradientClippingConfig;
use burn::optim::AdamConfig;

/// Configuration for training a Twin Delayed DDPG (TD3) agent.
#[derive(Clone, Debug)]
pub struct Td3TrainingConfig {
    /// Maximum number of transitions stored in the replay buffer.
    pub buffer_capacity: usize,
    /// Mini-batch size drawn from the replay buffer each learn step.
    pub batch_size: usize,
    /// Number of warm-up env steps before the first gradient update; during
    /// warm-up the agent acts with uniformly random actions on `[low, high]`.
    pub learning_starts: usize,
    /// Learning rate for the actor's Adam optimizer.
    pub actor_lr: f64,
    /// Learning rate for both critics' Adam optimizers (a shared schedule).
    pub critic_lr: f64,
    /// Discount factor γ applied to the bootstrap target.
    pub gamma: f32,
    /// Polyak averaging rate τ for the actor and both critic target networks.
    pub tau: f32,
    /// Standard deviation σ of the Gaussian exploration noise added to the
    /// actor's output at action selection time (before clipping to
    /// `[low, high]`).
    pub exploration_noise: f32,
    /// Standard deviation σ of the Gaussian noise added to the *target*
    /// actor's output during target-Q computation (target-policy smoothing).
    /// `0.2` matches CleanRL's default.
    pub policy_noise: f32,
    /// Symmetric clip applied to the target-policy-smoothing noise: the raw
    /// noise is bounded to `[-noise_clip, +noise_clip]` before being added to
    /// the target action. `0.5` matches CleanRL's default.
    pub noise_clip: f32,
    /// Critic-update cadence at which the policy and all three Polyak
    /// updates run. `policy_frequency = 2` matches CleanRL's default.
    pub policy_frequency: usize,
    /// Optional gradient clipping applied to actor and both critic grads.
    pub clip_grad: Option<GradientClippingConfig>,
    /// Base Adam configuration; cloned for each optimizer so actor and both
    /// critics share β-params but keep independent moment estimates.
    pub optimizer: AdamConfig,
}

impl Default for Td3TrainingConfig {
    /// CleanRL's default hyperparameters for `td3_continuous_action.py`.
    fn default() -> Self {
        Self {
            buffer_capacity: 1_000_000,
            batch_size: 256,
            learning_starts: 25_000,
            actor_lr: 3e-4,
            critic_lr: 3e-4,
            gamma: 0.99,
            tau: 0.005,
            exploration_noise: 0.1,
            policy_noise: 0.2,
            noise_clip: 0.5,
            policy_frequency: 2,
            clip_grad: None,
            optimizer: AdamConfig::new(),
        }
    }
}

/// Fluent builder for [`Td3TrainingConfig`].
///
/// All unset fields default to [`Td3TrainingConfig::default`].
///
/// # Examples
///
/// ```ignore
/// use evorl_rl::algorithms::td3::td3_config::Td3TrainingConfigBuilder;
///
/// let cfg = Td3TrainingConfigBuilder::new()
///     .batch_size(128)
///     .policy_noise(0.2)
///     .noise_clip(0.5)
///     .build();
/// ```
pub struct Td3TrainingConfigBuilder {
    config: Td3TrainingConfig,
}

impl Default for Td3TrainingConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Td3TrainingConfigBuilder {
    /// Creates a new builder initialised with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: Td3TrainingConfig::default(),
        }
    }

    /// Sets the replay-buffer capacity.
    pub fn buffer_capacity(mut self, capacity: usize) -> Self {
        self.config.buffer_capacity = capacity;
        self
    }

    /// Sets the mini-batch size.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }

    /// Sets the number of warm-up steps before learning begins.
    pub fn learning_starts(mut self, learning_starts: usize) -> Self {
        self.config.learning_starts = learning_starts;
        self
    }

    /// Sets the actor learning rate.
    pub fn actor_lr(mut self, lr: f64) -> Self {
        self.config.actor_lr = lr;
        self
    }

    /// Sets the shared critic learning rate.
    pub fn critic_lr(mut self, lr: f64) -> Self {
        self.config.critic_lr = lr;
        self
    }

    /// Sets the discount factor γ.
    pub fn gamma(mut self, gamma: f32) -> Self {
        self.config.gamma = gamma;
        self
    }

    /// Sets the Polyak averaging rate τ.
    pub fn tau(mut self, tau: f32) -> Self {
        self.config.tau = tau;
        self
    }

    /// Sets the action-selection Gaussian exploration-noise σ.
    pub fn exploration_noise(mut self, sigma: f32) -> Self {
        self.config.exploration_noise = sigma;
        self
    }

    /// Sets the target-policy-smoothing Gaussian noise σ.
    pub fn policy_noise(mut self, sigma: f32) -> Self {
        self.config.policy_noise = sigma;
        self
    }

    /// Sets the symmetric clip applied to the target-policy-smoothing noise.
    pub fn noise_clip(mut self, clip: f32) -> Self {
        self.config.noise_clip = clip;
        self
    }

    /// Sets the policy-update cadence (in critic steps).
    pub fn policy_frequency(mut self, frequency: usize) -> Self {
        self.config.policy_frequency = frequency;
        self
    }

    /// Sets the gradient-clipping configuration applied to actor and both
    /// critic gradients.
    pub fn clip_grad(mut self, config: Option<GradientClippingConfig>) -> Self {
        self.config.clip_grad = config;
        self
    }

    /// Overrides the base Adam optimiser configuration.
    pub fn optimizer(mut self, optimizer: AdamConfig) -> Self {
        self.config.optimizer = optimizer;
        self
    }

    /// Consumes the builder and returns the final config.
    pub fn build(self) -> Td3TrainingConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_cleanrl() {
        let cfg = Td3TrainingConfig::default();
        assert_eq!(cfg.buffer_capacity, 1_000_000);
        assert_eq!(cfg.batch_size, 256);
        assert_eq!(cfg.learning_starts, 25_000);
        assert!((cfg.actor_lr - 3e-4).abs() < 1e-12);
        assert!((cfg.critic_lr - 3e-4).abs() < 1e-12);
        assert!((cfg.gamma - 0.99).abs() < 1e-6);
        assert!((cfg.tau - 0.005).abs() < 1e-6);
        assert!((cfg.exploration_noise - 0.1).abs() < 1e-6);
        assert!((cfg.policy_noise - 0.2).abs() < 1e-6);
        assert!((cfg.noise_clip - 0.5).abs() < 1e-6);
        assert_eq!(cfg.policy_frequency, 2);
    }

    #[test]
    fn builder_overrides_propagate() {
        let cfg = Td3TrainingConfigBuilder::new()
            .batch_size(64)
            .policy_noise(0.3)
            .noise_clip(0.4)
            .policy_frequency(4)
            .build();
        assert_eq!(cfg.batch_size, 64);
        assert!((cfg.policy_noise - 0.3).abs() < 1e-6);
        assert!((cfg.noise_clip - 0.4).abs() < 1e-6);
        assert_eq!(cfg.policy_frequency, 4);
        // Untouched fields retain defaults.
        assert!((cfg.exploration_noise - 0.1).abs() < 1e-6);
    }
}
