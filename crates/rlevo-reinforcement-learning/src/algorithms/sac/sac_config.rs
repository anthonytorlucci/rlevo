//! Hyperparameter configuration for the SAC algorithm.
//!
//! Fields and defaults track `CleanRL`'s `sac_continuous_action.py` so that
//! reproducing published Pendulum/MuJoCo numbers reduces to plugging the same
//! values in. Compared to
//! [`Td3TrainingConfig`](crate::algorithms::td3::td3_config::Td3TrainingConfig),
//! SAC drops the deterministic-exploration / target-policy-smoothing knobs
//! (`exploration_noise`, `policy_noise`, `noise_clip`) and adds the
//! entropy-temperature controls (`alpha_lr`, `autotune`, `initial_alpha`,
//! `target_entropy`).
//!
//! The squashed-Gaussian head's `log σ` bounds are **not** here: they live on
//! [`SquashedGaussianPolicyHeadConfig`](crate::algorithms::sac::sac_policy::SquashedGaussianPolicyHeadConfig),
//! the config that is actually consumed to build the head and where the clamp
//! is applied.

use burn::grad_clipping::GradientClippingConfig;
use burn::optim::AdamConfig;
use rlevo_core::config::{self, ConfigError, Validate};

/// Configuration for training a SAC agent.
#[derive(Clone, Debug)]
pub struct SacTrainingConfig {
    /// Maximum number of transitions stored in the replay buffer.
    pub replay_buffer_capacity: usize,
    /// Mini-batch size drawn from the replay buffer each learn step.
    pub batch_size: usize,
    /// Warm-up env steps before the first gradient update. During warm-up
    /// the agent acts with uniformly random actions on `[low, high]`.
    pub learning_starts: usize,
    /// Learning rate for the actor's Adam optimiser.
    pub actor_lr: f64,
    /// Learning rate for both critics' Adam optimisers.
    pub critic_lr: f64,
    /// Learning rate for the `log α` optimiser (ignored when `autotune=false`).
    pub alpha_lr: f64,
    /// Discount factor γ applied to the bootstrap target.
    pub gamma: f32,
    /// Polyak averaging rate τ for both critic target networks. SAC has no
    /// target actor.
    pub tau: f32,
    /// When `true`, `log α` is trained toward `target_entropy`. When `false`,
    /// `α` is frozen at `initial_alpha`.
    pub autotune: bool,
    /// Initial value for α (i.e. `log α = ln(initial_alpha)`). Defaults to
    /// `1.0` so `log α` starts at `0`, matching `CleanRL`.
    pub initial_alpha: f32,
    /// Target entropy H̄. `None` ⇒ `-(A::COMPONENTS as f32)` (the common
    /// heuristic from Haarnoja et al. 2018b, matching `CleanRL`).
    pub target_entropy: Option<f32>,
    /// Critic-update cadence at which the actor and α updates run. `2`
    /// matches `CleanRL`'s `sac_continuous_action.py` default.
    pub policy_frequency: usize,
    /// Critic-update cadence at which the twin critic targets are
    /// Polyak-averaged. `1` matches `CleanRL`'s default.
    pub target_update_frequency: usize,
    /// Optional gradient clipping applied to actor and both critic grads.
    pub clip_grad: Option<GradientClippingConfig>,
    /// Base Adam configuration cloned for each optimiser so the actor, both
    /// critics, and `log α` share β-params but keep independent moment
    /// estimates.
    pub optimizer: AdamConfig,
}

impl Default for SacTrainingConfig {
    /// `CleanRL`'s default hyperparameters for `sac_continuous_action.py`.
    fn default() -> Self {
        Self {
            replay_buffer_capacity: 1_000_000,
            batch_size: 256,
            learning_starts: 5_000,
            actor_lr: 3e-4,
            critic_lr: 1e-3,
            alpha_lr: 1e-3,
            gamma: 0.99,
            tau: 0.005,
            autotune: true,
            initial_alpha: 1.0,
            target_entropy: None,
            policy_frequency: 2,
            target_update_frequency: 1,
            clip_grad: None,
            optimizer: AdamConfig::new(),
        }
    }
}

impl Validate for SacTrainingConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "SacTrainingConfig";
        config::nonzero(C, "replay_buffer_capacity", self.replay_buffer_capacity)?;
        config::nonzero(C, "batch_size", self.batch_size)?;
        config::positive(C, "actor_lr", self.actor_lr)?;
        config::positive(C, "critic_lr", self.critic_lr)?;
        config::positive(C, "alpha_lr", self.alpha_lr)?;
        config::in_range(C, "gamma", 0.0, 1.0, f64::from(self.gamma))?;
        config::in_range(C, "tau", 0.0, 1.0, f64::from(self.tau))?;
        config::positive(C, "initial_alpha", f64::from(self.initial_alpha))?;
        config::at_least(C, "policy_frequency", self.policy_frequency, 1)?;
        config::at_least(
            C,
            "target_update_frequency",
            self.target_update_frequency,
            1,
        )?;
        Ok(())
    }
}

/// Fluent builder for [`SacTrainingConfig`]. All unset fields default to
/// [`SacTrainingConfig::default`].
///
/// # Examples
///
/// ```rust
/// use rlevo_reinforcement_learning::algorithms::sac::sac_config::SacTrainingConfigBuilder;
///
/// let cfg = SacTrainingConfigBuilder::new()
///     .batch_size(128)
///     .autotune(true)
///     .build()
///     .expect("valid config");
/// ```
#[derive(Debug)]
pub struct SacTrainingConfigBuilder {
    config: SacTrainingConfig,
}

impl Default for SacTrainingConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SacTrainingConfigBuilder {
    /// New builder initialised with [`SacTrainingConfig::default`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: SacTrainingConfig::default(),
        }
    }

    /// Sets the replay-buffer capacity.
    #[must_use]
    pub fn replay_buffer_capacity(mut self, capacity: usize) -> Self {
        self.config.replay_buffer_capacity = capacity;
        self
    }

    /// Sets the mini-batch size.
    #[must_use]
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }

    /// Sets the number of warm-up steps before learning begins.
    #[must_use]
    pub fn learning_starts(mut self, learning_starts: usize) -> Self {
        self.config.learning_starts = learning_starts;
        self
    }

    /// Sets the actor learning rate.
    #[must_use]
    pub fn actor_lr(mut self, lr: f64) -> Self {
        self.config.actor_lr = lr;
        self
    }

    /// Sets the shared critic learning rate.
    #[must_use]
    pub fn critic_lr(mut self, lr: f64) -> Self {
        self.config.critic_lr = lr;
        self
    }

    /// Sets the `log α` learning rate.
    #[must_use]
    pub fn alpha_lr(mut self, lr: f64) -> Self {
        self.config.alpha_lr = lr;
        self
    }

    /// Sets the discount factor γ.
    #[must_use]
    pub fn gamma(mut self, gamma: f32) -> Self {
        self.config.gamma = gamma;
        self
    }

    /// Sets the Polyak averaging rate τ.
    #[must_use]
    pub fn tau(mut self, tau: f32) -> Self {
        self.config.tau = tau;
        self
    }

    /// Enables or disables auto-tuning of α.
    #[must_use]
    pub fn autotune(mut self, autotune: bool) -> Self {
        self.config.autotune = autotune;
        self
    }

    /// Sets the initial α (also used as the fixed α when `autotune=false`).
    #[must_use]
    pub fn initial_alpha(mut self, alpha: f32) -> Self {
        self.config.initial_alpha = alpha;
        self
    }

    /// Sets the target entropy H̄. Pass `None` to restore the `-|A|`
    /// heuristic.
    #[must_use]
    pub fn target_entropy(mut self, target: Option<f32>) -> Self {
        self.config.target_entropy = target;
        self
    }

    /// Sets the critic-step cadence at which the actor + α updates run.
    #[must_use]
    pub fn policy_frequency(mut self, frequency: usize) -> Self {
        self.config.policy_frequency = frequency;
        self
    }

    /// Sets the critic-step cadence at which the twin target Polyak updates
    /// run.
    #[must_use]
    pub fn target_update_frequency(mut self, frequency: usize) -> Self {
        self.config.target_update_frequency = frequency;
        self
    }

    /// Sets the gradient-clipping configuration applied to actor and both
    /// critic gradients.
    #[must_use]
    pub fn clip_grad(mut self, config: Option<GradientClippingConfig>) -> Self {
        self.config.clip_grad = config;
        self
    }

    /// Overrides the base Adam optimiser configuration.
    #[must_use]
    pub fn optimizer(mut self, optimizer: AdamConfig) -> Self {
        self.config.optimizer = optimizer;
        self
    }

    /// Consumes the builder and returns the final config.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if the assembled config violates any invariant
    /// checked by [`SacTrainingConfig::validate`] (e.g. a non-positive
    /// `initial_alpha` or a zero `batch_size`).
    pub fn build(self) -> Result<SacTrainingConfig, ConfigError> {
        self.config.validate()?;
        Ok(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_cleanrl() {
        let cfg = SacTrainingConfig::default();
        assert_eq!(cfg.replay_buffer_capacity, 1_000_000);
        assert_eq!(cfg.batch_size, 256);
        assert_eq!(cfg.learning_starts, 5_000);
        assert!((cfg.actor_lr - 3e-4).abs() < 1e-12);
        assert!((cfg.critic_lr - 1e-3).abs() < 1e-12);
        assert!((cfg.alpha_lr - 1e-3).abs() < 1e-12);
        assert!((cfg.gamma - 0.99).abs() < 1e-6);
        assert!((cfg.tau - 0.005).abs() < 1e-6);
        assert!(cfg.autotune);
        assert!((cfg.initial_alpha - 1.0).abs() < 1e-6);
        assert!(cfg.target_entropy.is_none());
        assert_eq!(cfg.policy_frequency, 2);
        assert_eq!(cfg.target_update_frequency, 1);
    }

    #[test]
    fn builder_overrides_propagate() {
        let cfg = SacTrainingConfigBuilder::new()
            .batch_size(64)
            .autotune(false)
            .initial_alpha(0.2)
            .target_entropy(Some(-1.0))
            .policy_frequency(1)
            .build()
            .expect("valid config");
        assert_eq!(cfg.batch_size, 64);
        assert!(!cfg.autotune);
        assert!((cfg.initial_alpha - 0.2).abs() < 1e-6);
        assert_eq!(cfg.target_entropy, Some(-1.0));
        assert_eq!(cfg.policy_frequency, 1);
        // Untouched fields retain defaults.
        assert_eq!(cfg.learning_starts, 5_000);
    }

    #[test]
    fn default_config_is_valid() {
        assert!(SacTrainingConfig::default().validate().is_ok());
    }
}
