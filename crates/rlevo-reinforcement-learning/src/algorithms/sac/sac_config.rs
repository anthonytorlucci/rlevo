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

use crate::target::TargetUpdate;

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
    /// Update rule for both critic target networks: the Polyak coefficient τ
    /// and the cadence at which it fires. SAC has no target actor.
    ///
    /// The cadence counts **gradient (critic) updates**, not environment steps
    /// (ADR 0059) — unlike [`learning_starts`](Self::learning_starts), which is
    /// in env steps. The default `TargetUpdate::polyak(0.005, 1)` is
    /// `CleanRL`'s `tau = 0.005` with a target update on every critic step.
    pub target_update: TargetUpdate,
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
            autotune: true,
            initial_alpha: 1.0,
            target_entropy: None,
            policy_frequency: 2,
            target_update: TargetUpdate::polyak(0.005, 1),
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
        config::positive(C, "initial_alpha", f64::from(self.initial_alpha))?;
        config::at_least(C, "policy_frequency", self.policy_frequency, 1)?;
        // `target_update` carries no check here: `TargetUpdate` is valid by
        // construction (τ ∈ (0, 1], cadence ≥ 1), so the newtype *removes* the
        // paired `config::` lines rather than duplicating them — ADR 0027 §3,
        // ADR 0058 §Consequences.
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

    /// Sets the twin critic targets' update rule: the Polyak coefficient τ and
    /// the critic-update cadence at which it fires.
    ///
    /// ```rust
    /// use rlevo_reinforcement_learning::algorithms::sac::sac_config::SacTrainingConfigBuilder;
    /// use rlevo_reinforcement_learning::target::TargetUpdate;
    ///
    /// let cfg = SacTrainingConfigBuilder::new()
    ///     .target_update(TargetUpdate::polyak(0.02, 1))
    ///     .build()
    ///     .expect("valid config");
    /// assert_eq!(cfg.target_update.every(), 1);
    /// ```
    #[must_use]
    pub fn target_update(mut self, target_update: TargetUpdate) -> Self {
        self.config.target_update = target_update;
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
        assert!(cfg.autotune);
        assert!((cfg.initial_alpha - 1.0).abs() < 1e-6);
        assert!(cfg.target_entropy.is_none());
        assert_eq!(cfg.policy_frequency, 2);
        // `CleanRL`'s `tau = 0.005` and `target_network_frequency = 1`, now one
        // value. Exact equality: both halves are source literals read back.
        assert_eq!(cfg.target_update, TargetUpdate::polyak(0.005, 1));
    }

    /// The pre-ADR-0058 default was `tau: 0.005` + `target_update_frequency: 1`,
    /// consumed as `f64::from(tau)` under an `is_multiple_of(1)` gate. Pinning
    /// both halves separately is what makes "bit-identical at defaults" a
    /// checkable claim rather than a review comment.
    // Bit-exactness *is* the property under test here: τ is a source literal
    // stored verbatim and read back through a widening that is exact for every
    // `f32`, never the result of arithmetic. A tolerance would let a genuine
    // default drift pass.
    #[allow(clippy::float_cmp)]
    #[test]
    fn default_target_update_is_bit_identical_to_the_pre_migration_pair() {
        let cfg = SacTrainingConfig::default();
        assert_eq!(
            cfg.target_update.tau(),
            f64::from(0.005_f32),
            "τ must survive the migration bit-for-bit, including its f32→f64 widening"
        );
        assert_eq!(cfg.target_update.every(), 1);
        // `every = 1` fires on every critic update, exactly as the old
        // `critic_updates.is_multiple_of(1)` gate did.
        for updates in 1_usize..=8 {
            assert!(
                cfg.target_update.fires_at(updates).is_some(),
                "critic update {updates} must fire under the default cadence"
            );
        }
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

    /// The frozen-target state is not *rejected* by `validate` here — it is
    /// unrepresentable, so the assertion belongs at the constructor (ADR 0058
    /// §Consequences). Pinned per family rather than only in `target.rs`,
    /// because the guarantee a SAC reader needs is "no `SacTrainingConfig` can
    /// hold a frozen target", and that is a statement about this config.
    #[test]
    fn frozen_target_is_unreachable_through_the_type() {
        assert!(
            TargetUpdate::try_polyak(0.0, 1).is_err(),
            "τ = 0 fires on schedule and moves nothing — a frozen target"
        );
        assert!(
            TargetUpdate::try_polyak(0.005, 0).is_err(),
            "a cadence of 0 never fires — a frozen target"
        );
        assert!(
            TargetUpdate::try_polyak(0.0, 0).is_err(),
            "both halves frozen at once"
        );
    }

    /// The builder is not the only way in: `target_update` is a `pub` field, so
    /// struct-update syntax bypasses `SacTrainingConfigBuilder` entirely. It is
    /// still safe, and the load-bearing reason is that `TargetUpdate`'s inner
    /// `tau`/`every` fields are **private** (`target.rs`): a caller cannot
    /// construct an invalid `TargetUpdate` *value* to assign into the public
    /// field, so there is nothing for `validate` to catch on this route.
    #[test]
    fn nan_tau_cannot_be_constructed_for_struct_update_syntax() {
        assert!(TargetUpdate::try_polyak(f32::NAN, 1).is_err());
        assert!(TargetUpdate::try_polyak(f32::INFINITY, 1).is_err());
        // Every τ a struct-update config can carry came through `PolyakTau`, so
        // the result is necessarily valid.
        let config = SacTrainingConfig {
            target_update: TargetUpdate::polyak(0.005, 1),
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }
}
