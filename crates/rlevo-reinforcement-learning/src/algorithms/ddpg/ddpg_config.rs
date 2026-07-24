//! Hyperparameter configuration for the DDPG algorithm.
//!
//! Fields and defaults track `CleanRL`'s `ddpg_continuous_action.py` so that
//! reproducing published Pendulum/MuJoCo numbers is a matter of plugging the
//! same values in.

use burn::grad_clipping::GradientClippingConfig;
use burn::optim::AdamConfig;
use rlevo_core::config::{self, ConfigError, Validate};

use crate::target::TargetUpdate;

/// Configuration for training a Deep Deterministic Policy Gradient agent.
#[derive(Clone, Debug)]
pub struct DdpgTrainingConfig {
    /// Maximum number of transitions stored in the replay buffer.
    pub replay_buffer_capacity: usize,
    /// Mini-batch size drawn from the replay buffer each learn step.
    pub batch_size: usize,
    /// Number of warm-up env steps before the first gradient update; during
    /// warm-up the agent acts with uniformly random actions on `[low, high]`.
    pub learning_starts: usize,
    /// Learning rate for the actor's Adam optimizer.
    pub actor_lr: f64,
    /// Learning rate for the critic's Adam optimizer.
    pub critic_lr: f64,
    /// Discount factor γ applied to the bootstrap target.
    pub gamma: f32,
    /// Standard deviation σ of the Gaussian exploration noise added to the
    /// actor's output (before clipping to `[low, high]`).
    pub exploration_noise: f32,
    /// Critic-update cadence at which the **actor** update runs — TD3's delay
    /// `d` (Fujimoto et al. 2018 §5.2), which DDPG inherits here.
    /// `policy_frequency = 2` matches `CleanRL`'s default.
    ///
    /// This governs the actor only. Before ADR 0058 it silently doubled as the
    /// target-network cadence; that role now belongs to
    /// [`target_update`](Self::target_update), so the two are independently
    /// settable.
    pub policy_frequency: usize,
    /// Update rule for the target actor and target critic: the Polyak
    /// coefficient τ and the cadence at which it fires.
    ///
    /// The cadence counts **gradient (critic) updates**, not environment steps
    /// (ADR 0059) — unlike [`learning_starts`](Self::learning_starts), which is
    /// in env steps. The default `TargetUpdate::polyak(0.005, 2)` reproduces
    /// the pre-ADR-0058 behaviour exactly: `CleanRL`'s `tau = 0.005`, fired on
    /// the same cadence the default `policy_frequency = 2` used to impose.
    pub target_update: TargetUpdate,
    /// Optional gradient clipping applied to both actor and critic grads.
    pub clip_grad: Option<GradientClippingConfig>,
    /// Base Adam configuration; cloned for each optimizer so actor and critic
    /// share β-params but keep independent moment estimates.
    pub optimizer: AdamConfig,
}

impl Default for DdpgTrainingConfig {
    /// `CleanRL`'s default hyperparameters for `ddpg_continuous_action.py`.
    fn default() -> Self {
        Self {
            replay_buffer_capacity: 1_000_000,
            batch_size: 256,
            learning_starts: 25_000,
            actor_lr: 3e-4,
            critic_lr: 3e-4,
            gamma: 0.99,
            exploration_noise: 0.1,
            policy_frequency: 2,
            target_update: TargetUpdate::polyak(0.005, 2),
            clip_grad: None,
            optimizer: AdamConfig::new(),
        }
    }
}

impl Validate for DdpgTrainingConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "DdpgTrainingConfig";
        config::nonzero(C, "replay_buffer_capacity", self.replay_buffer_capacity)?;
        config::nonzero(C, "batch_size", self.batch_size)?;
        config::positive(C, "actor_lr", self.actor_lr)?;
        config::positive(C, "critic_lr", self.critic_lr)?;
        config::in_range(C, "gamma", 0.0, 1.0, f64::from(self.gamma))?;
        config::in_range(
            C,
            "exploration_noise",
            0.0,
            f64::INFINITY,
            f64::from(self.exploration_noise),
        )?;
        config::at_least(C, "policy_frequency", self.policy_frequency, 1)?;
        // `target_update` carries no check here: `TargetUpdate` is valid by
        // construction (τ ∈ (0, 1], cadence ≥ 1), so the newtype *removes* the
        // paired `config::in_range(C, "tau", ...)` line rather than duplicating
        // it — ADR 0027 §3, ADR 0058 §Consequences.
        Ok(())
    }
}

/// Fluent builder for [`DdpgTrainingConfig`].
///
/// All unset fields default to [`DdpgTrainingConfig::default`].
///
/// # Examples
///
/// ```rust
/// use rlevo_reinforcement_learning::algorithms::ddpg::ddpg_config::DdpgTrainingConfigBuilder;
///
/// let cfg = DdpgTrainingConfigBuilder::new()
///     .batch_size(128)
///     .actor_lr(1e-4)
///     .critic_lr(1e-3)
///     .build()
///     .expect("valid config");
/// ```
#[derive(Debug)]
pub struct DdpgTrainingConfigBuilder {
    config: DdpgTrainingConfig,
}

impl Default for DdpgTrainingConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DdpgTrainingConfigBuilder {
    /// Creates a new builder initialised with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: DdpgTrainingConfig::default(),
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

    /// Sets the critic learning rate.
    #[must_use]
    pub fn critic_lr(mut self, lr: f64) -> Self {
        self.config.critic_lr = lr;
        self
    }

    /// Sets the discount factor γ.
    #[must_use]
    pub fn gamma(mut self, gamma: f32) -> Self {
        self.config.gamma = gamma;
        self
    }

    /// Sets the Gaussian exploration-noise standard deviation.
    #[must_use]
    pub fn exploration_noise(mut self, sigma: f32) -> Self {
        self.config.exploration_noise = sigma;
        self
    }

    /// Sets the **actor**-update cadence, in critic steps. Since ADR 0058 this
    /// no longer moves the target networks — see
    /// [`target_update`](Self::target_update).
    #[must_use]
    pub fn policy_frequency(mut self, frequency: usize) -> Self {
        self.config.policy_frequency = frequency;
        self
    }

    /// Sets both target networks' update rule: the Polyak coefficient τ and the
    /// critic-update cadence at which it fires.
    ///
    /// ```rust
    /// use rlevo_reinforcement_learning::algorithms::ddpg::ddpg_config::DdpgTrainingConfigBuilder;
    /// use rlevo_reinforcement_learning::target::TargetUpdate;
    ///
    /// // Actor every critic step, targets every second one — a pairing that
    /// // was unexpressible while `policy_frequency` drove both.
    /// let cfg = DdpgTrainingConfigBuilder::new()
    ///     .policy_frequency(1)
    ///     .target_update(TargetUpdate::polyak(0.005, 2))
    ///     .build()
    ///     .expect("valid config");
    /// assert_eq!(cfg.policy_frequency, 1);
    /// assert_eq!(cfg.target_update.every(), 2);
    /// ```
    #[must_use]
    pub fn target_update(mut self, target_update: TargetUpdate) -> Self {
        self.config.target_update = target_update;
        self
    }

    /// Sets the gradient-clipping configuration applied to both actor and
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
    /// checked by [`DdpgTrainingConfig::validate`] (e.g. a zero `batch_size` or
    /// a non-positive learning rate).
    pub fn build(self) -> Result<DdpgTrainingConfig, ConfigError> {
        self.config.validate()?;
        Ok(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_cleanrl() {
        let cfg = DdpgTrainingConfig::default();
        assert_eq!(cfg.replay_buffer_capacity, 1_000_000);
        assert_eq!(cfg.batch_size, 256);
        assert_eq!(cfg.learning_starts, 25_000);
        assert!((cfg.actor_lr - 3e-4).abs() < 1e-12);
        assert!((cfg.critic_lr - 3e-4).abs() < 1e-12);
        assert!((cfg.gamma - 0.99).abs() < 1e-6);
        assert!((cfg.exploration_noise - 0.1).abs() < 1e-6);
        assert_eq!(cfg.policy_frequency, 2);
        // Exact equality: both halves are source literals read back.
        assert_eq!(cfg.target_update, TargetUpdate::polyak(0.005, 2));
    }

    /// Before ADR 0058 the Polyak update lived *inside* the `policy_frequency`
    /// block, so the target cadence was `policy_frequency` by aliasing. The
    /// default `every` must equal the default `policy_frequency`, or the
    /// decoupling silently rescaled every default DDPG run.
    // Bit-exactness *is* the property under test here: τ is a source literal
    // stored verbatim and read back through a widening that is exact for every
    // `f32`, never the result of arithmetic. A tolerance would let a genuine
    // default drift pass.
    #[allow(clippy::float_cmp)]
    #[test]
    fn default_target_update_is_bit_identical_to_the_pre_migration_alias() {
        let cfg = DdpgTrainingConfig::default();
        assert_eq!(
            cfg.target_update.tau(),
            f64::from(0.005_f32),
            "τ must survive the migration bit-for-bit, including its f32→f64 widening"
        );
        assert_eq!(
            cfg.target_update.every(),
            cfg.policy_frequency,
            "at defaults the target cadence must still coincide with the actor delay it \
             used to be aliased to"
        );
        // The two gates therefore agree on every critic-update index.
        for updates in 1_usize..=12 {
            assert_eq!(
                cfg.target_update.fires_at(updates).is_some(),
                updates.is_multiple_of(cfg.policy_frequency),
                "critic update {updates}: the new gate must match the old \
                 `is_multiple_of(policy_frequency)` predicate"
            );
        }
    }

    /// The configuration ADR 0058 exists to make expressible: an actor delay
    /// and a target cadence that differ.
    #[test]
    fn actor_delay_and_target_cadence_are_independently_settable() {
        let cfg = DdpgTrainingConfigBuilder::new()
            .policy_frequency(1)
            .target_update(TargetUpdate::polyak(0.005, 2))
            .build()
            .expect("valid config");
        assert_eq!(cfg.policy_frequency, 1);
        assert_eq!(cfg.target_update.every(), 2);
        assert!(cfg.target_update.fires_at(1).is_none());
        assert!(cfg.target_update.fires_at(2).is_some());
    }

    #[test]
    fn builder_overrides_propagate() {
        let cfg = DdpgTrainingConfigBuilder::new()
            .batch_size(64)
            .actor_lr(1e-4)
            .exploration_noise(0.2)
            .build()
            .expect("valid config");
        assert_eq!(cfg.batch_size, 64);
        assert!((cfg.actor_lr - 1e-4).abs() < 1e-12);
        assert!((cfg.exploration_noise - 0.2).abs() < 1e-6);
        // Untouched fields retain defaults.
        assert_eq!(cfg.policy_frequency, 2);
    }

    #[test]
    fn default_config_is_valid() {
        assert!(DdpgTrainingConfig::default().validate().is_ok());
    }

    #[test]
    fn rejects_zero_batch_size() {
        let err = DdpgTrainingConfigBuilder::new()
            .batch_size(0)
            .build()
            .unwrap_err();
        assert_eq!(err.field, "batch_size");
    }

    /// The frozen-target state is not *rejected* by `validate` here — it is
    /// unrepresentable, so the assertion belongs at the constructor (ADR 0058
    /// §Consequences). Pinned per family rather than only in `target.rs`,
    /// because the guarantee a DDPG reader needs is "no `DdpgTrainingConfig`
    /// can hold a frozen target", and that is a statement about this config.
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
    /// struct-update syntax bypasses `DdpgTrainingConfigBuilder` entirely. It is
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
        let config = DdpgTrainingConfig {
            target_update: TargetUpdate::polyak(0.005, 2),
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }
}
