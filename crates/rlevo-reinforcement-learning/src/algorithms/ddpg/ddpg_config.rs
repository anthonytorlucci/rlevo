//! Hyperparameter configuration for the DDPG algorithm.
//!
//! Fields and defaults track `CleanRL`'s `ddpg_continuous_action.py` so that
//! reproducing published Pendulum/MuJoCo numbers is a matter of plugging the
//! same values in.

use burn::grad_clipping::GradientClippingConfig;
use burn::optim::AdamConfig;
use rlevo_core::config::{self, ConfigError, Validate};

use crate::replay::UniformReplayConfig;
use crate::target::TargetUpdate;

/// Configuration for training a Deep Deterministic Policy Gradient agent.
///
/// # Examples
///
/// Tune it with struct-update syntax; the agent's constructor validates it.
///
/// ```rust
/// use rlevo_core::config::Validate;
/// use rlevo_reinforcement_learning::algorithms::ddpg::ddpg_config::DdpgTrainingConfig;
///
/// let cfg = DdpgTrainingConfig {
///     batch_size: 128,
///     actor_lr: 1e-4,
///     critic_lr: 1e-3,
///     ..Default::default()
/// };
/// assert!(cfg.validate().is_ok());
/// ```
#[derive(Clone, Debug)]
pub struct DdpgTrainingConfig {
    /// Maximum number of transitions stored in the replay buffer.
    ///
    /// Must be non-zero and at most
    /// [`MAX_BUFFER_CAPACITY`](crate::MAX_BUFFER_CAPACITY). The ceiling is not
    /// a taste judgement: a misconfigured value (a unit slip, a `usize::MAX`
    /// sentinel) would otherwise reach `VecDeque::with_capacity` and abort the
    /// process on a failed allocation. Both bounds are checked by
    /// [`UniformReplayConfig`], which `validate` delegates to; see
    /// [`MAX_BUFFER_CAPACITY`](crate::MAX_BUFFER_CAPACITY) for why `$2^{32}$`
    /// is the bound.
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
    /// Discount factor `$\gamma$` applied to the bootstrap target.
    pub gamma: f32,
    /// Standard deviation `$\sigma$` of the Gaussian exploration noise added to the actor's output
    /// (before clipping to `[low, high]`).
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
    /// Update rule for the target actor and target critic: the Polyak coefficient `$\tau$` and the
    /// cadence at which it fires.
    ///
    /// The cadence counts **gradient (critic) updates**, not environment steps
    /// (ADR 0059) — unlike [`learning_starts`](Self::learning_starts), which is
    /// in env steps. The default `TargetUpdate::polyak(0.005, 2)` reproduces
    /// the pre-ADR-0058 behaviour exactly: `CleanRL`'s `tau = 0.005`, fired on
    /// the same cadence the default `policy_frequency = 2` used to impose.
    pub target_update: TargetUpdate,
    /// Optional gradient clipping applied to both actor and critic grads.
    pub clip_grad: Option<GradientClippingConfig>,
    /// Base Adam configuration; cloned for each optimizer so actor and critic share
    /// `$\beta$`-params but keep independent moment estimates.
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
        // `replay_buffer_capacity` carries no `config::` lines of its own: the
        // floor-and-ceiling predicate is single-sourced in
        // `UniformReplayConfig::validate`, which is the same check the buffer
        // this field feeds runs at construction. The error is relabelled to
        // name *this* config's field, so what the caller sees is byte-identical
        // to the two hand-written lines it replaces — only `kind` passes
        // through, and only the predicate's home moved (ADR 0050).
        UniformReplayConfig {
            capacity: self.replay_buffer_capacity,
        }
        .validate()
        .map_err(|e| ConfigError {
            config: C,
            field: "replay_buffer_capacity",
            ..e
        })?;
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
        // `target_update` carries no check here: `TargetUpdate` is valid by construction (`$\tau$`
        // `$\in$` (0, 1], cadence `$\geq$` 1), so the newtype *removes* the paired
        // `config::in_range(C, "tau", ...)` line rather than duplicating it — ADR 0027 §3, ADR 0058
        // §Consequences.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MAX_BUFFER_CAPACITY;
    use rlevo_core::config::ConstraintKind;

    /// `replay_buffer_capacity` had a floor and no ceiling, so a
    /// `usize::MAX` sentinel — a unit slip, a deserialized garbage field —
    /// validated and went straight to `UniformReplay::new`, which aborts the
    /// process on the failed allocation. The rejection must be a **recoverable
    /// `Err`**, which is why this asserts on the returned error rather than
    /// using `#[should_panic]`: a panic here would be the bug, not the fix.
    #[test]
    fn test_ddpg_config_rejects_replay_buffer_capacity_above_ceiling() {
        let err = DdpgTrainingConfig {
            replay_buffer_capacity: usize::MAX,
            ..Default::default()
        }
        .validate()
        .expect_err("usize::MAX capacity must be rejected, not allocated");
        assert_eq!(err.field, "replay_buffer_capacity");
        let max = u64::try_from(MAX_BUFFER_CAPACITY).expect("the ceiling fits in u64");
        let got = u64::try_from(usize::MAX).expect("usize::MAX fits in u64 on this target");
        assert_eq!(err.kind, ConstraintKind::TooLarge { max, got });
    }

    /// The boundary is inclusive: the ceiling itself is a legal (if
    /// unallocatable) capacity, so the guard cannot be off by one.
    #[test]
    fn test_ddpg_config_accepts_replay_buffer_capacity_at_ceiling() {
        let cfg = DdpgTrainingConfig {
            replay_buffer_capacity: MAX_BUFFER_CAPACITY,
            ..Default::default()
        };
        cfg.validate().expect("the ceiling itself must validate");
    }

    #[test]
    fn test_ddpg_config_defaults_match_cleanrl() {
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
    // Bit-exactness *is* the property under test here: `$\tau$` is a source literal stored verbatim
    // and read back unchanged, never the result of arithmetic. A tolerance would let a genuine
    // default drift pass.
    #[allow(clippy::float_cmp)]
    #[test]
    fn test_ddpg_config_default_target_update_is_bit_identical_to_the_pre_migration_alias() {
        let cfg = DdpgTrainingConfig::default();
        assert_eq!(
            cfg.target_update.tau(),
            0.005,
            "τ must survive the migration bit-for-bit"
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
    fn test_ddpg_config_actor_delay_and_target_cadence_are_independently_settable() {
        let cfg = DdpgTrainingConfig {
            policy_frequency: 1,
            target_update: TargetUpdate::polyak(0.005, 2),
            ..Default::default()
        };
        cfg.validate().expect("valid config");
        assert_eq!(cfg.policy_frequency, 1);
        assert_eq!(cfg.target_update.every(), 2);
        assert!(cfg.target_update.fires_at(1).is_none());
        assert!(cfg.target_update.fires_at(2).is_some());
    }

    #[test]
    fn test_ddpg_config_default_is_valid() {
        assert!(DdpgTrainingConfig::default().validate().is_ok());
    }

    #[test]
    fn test_ddpg_config_rejects_zero_batch_size() {
        let err = DdpgTrainingConfig {
            batch_size: 0,
            ..Default::default()
        }
        .validate()
        .unwrap_err();
        assert_eq!(err.field, "batch_size");
    }

    /// The frozen-target state is not *rejected* by `validate` here — it is
    /// unrepresentable, so the assertion belongs at the constructor (ADR 0058
    /// §Consequences). Pinned per family rather than only in `target.rs`,
    /// because the guarantee a DDPG reader needs is "no `DdpgTrainingConfig`
    /// can hold a frozen target", and that is a statement about this config.
    #[test]
    fn test_ddpg_config_frozen_target_is_unreachable_through_the_type() {
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

    /// `target_update` is a `pub` field, so struct-update syntax sets it with no
    /// setter in between. That is safe, and the load-bearing reason is that `TargetUpdate`'s inner
    /// `tau`/`every` fields are **private** (`target.rs`): a caller cannot
    /// construct an invalid `TargetUpdate` *value* to assign into the public
    /// field, so there is nothing for `validate` to catch on this route.
    #[test]
    fn test_ddpg_config_nan_tau_cannot_be_constructed_for_struct_update_syntax() {
        assert!(TargetUpdate::try_polyak(f32::NAN, 1).is_err());
        assert!(TargetUpdate::try_polyak(f32::INFINITY, 1).is_err());
        // Every `$\tau$` a struct-update config can carry came through `PolyakTau`, so the result
        // is necessarily valid.
        let config = DdpgTrainingConfig {
            target_update: TargetUpdate::polyak(0.005, 2),
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }
}
