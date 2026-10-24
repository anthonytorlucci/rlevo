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
//! The squashed-Gaussian head's `$\log \sigma$` bounds are **not** here: they live on
//! [`SquashedGaussianPolicyHeadConfig`](crate::algorithms::sac::sac_policy::SquashedGaussianPolicyHeadConfig),
//! the config that is actually consumed to build the head and where the clamp
//! is applied.

use burn::grad_clipping::GradientClippingConfig;
use burn::optim::AdamConfig;
use rlevo_core::config::{self, ConfigError, Validate};

use crate::replay::UniformReplayConfig;
use crate::target::TargetUpdate;

/// Configuration for training a SAC agent.
///
/// # Examples
///
/// Tune it with struct-update syntax; the agent's constructor validates it.
///
/// ```rust
/// use rlevo_core::config::Validate;
/// use rlevo_reinforcement_learning::algorithms::sac::sac_config::SacTrainingConfig;
///
/// let cfg = SacTrainingConfig {
///     batch_size: 128,
///     autotune: true,
///     ..Default::default()
/// };
/// assert!(cfg.validate().is_ok());
/// ```
#[derive(Clone, Debug)]
pub struct SacTrainingConfig {
    /// Maximum number of transitions stored in the replay buffer.
    ///
    /// Must be non-zero and at most
    /// `MAX_BUFFER_CAPACITY`. The ceiling is not
    /// a taste judgement: a misconfigured value (a unit slip, a `usize::MAX`
    /// sentinel) would otherwise reach `VecDeque::with_capacity` and abort the
    /// process on a failed allocation. Both bounds are checked by
    /// `UniformReplayConfig`, which `validate` delegates to; see
    /// `MAX_BUFFER_CAPACITY` for why `$2^{32}$`
    /// is the bound.
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
    /// Learning rate for the `$\log \alpha$` optimiser (ignored when `autotune=false`).
    pub alpha_lr: f64,
    /// Discount factor `$\gamma$` applied to the bootstrap target.
    pub gamma: f32,
    /// When `true`, `$\log \alpha$` is trained toward `target_entropy`. When `false`,
    /// `$\alpha$` is frozen at `initial_alpha`.
    pub autotune: bool,
    /// Initial value for `$\alpha$` (i.e. `$\log \alpha = \ln(\text{initial\_alpha})$`). Defaults
    /// to `1.0` so `$\log \alpha$` starts at `0`, matching `CleanRL`.
    pub initial_alpha: f32,
    /// Target entropy H̄. `None` `$\Rightarrow$` `-(A::COMPONENTS as f32)` (the common heuristic
    /// from Haarnoja et al. 2018b, matching `CleanRL`).
    pub target_entropy: Option<f32>,
    /// Critic-update cadence at which the actor and `$\alpha$` updates run. `2` matches `CleanRL`'s
    /// `sac_continuous_action.py` default.
    pub policy_frequency: usize,
    /// Update rule for both critic target networks: the Polyak coefficient `$\tau$` and the cadence
    /// at which it fires. SAC has no target actor.
    ///
    /// The cadence counts **gradient (critic) updates**, not environment steps
    /// (ADR 0059) — unlike [`learning_starts`](Self::learning_starts), which is
    /// in env steps. The default `TargetUpdate::polyak(0.005, 1)` is
    /// `CleanRL`'s `tau = 0.005` with a target update on every critic step.
    pub target_update: TargetUpdate,
    /// Optional gradient clipping applied to actor and both critic grads.
    pub clip_grad: Option<GradientClippingConfig>,
    /// Base Adam configuration cloned for each optimiser so the actor, both critics, and
    /// `$\log \alpha$` share `$\beta$`-params but keep independent moment estimates.
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
        config::positive(C, "alpha_lr", self.alpha_lr)?;
        config::in_range(C, "gamma", 0.0, 1.0, f64::from(self.gamma))?;
        config::positive(C, "initial_alpha", f64::from(self.initial_alpha))?;
        config::at_least(C, "policy_frequency", self.policy_frequency, 1)?;
        // `target_update` carries no check here: `TargetUpdate` is valid by construction (`$\tau$`
        // `$\in$` (0, 1], cadence `$\geq$` 1), so the newtype *removes* the paired `config::` lines
        // rather than duplicating them — ADR 0027 §3, ADR 0058 §Consequences.
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
    fn test_sac_config_rejects_replay_buffer_capacity_above_ceiling() {
        let err = SacTrainingConfig {
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
    fn test_sac_config_accepts_replay_buffer_capacity_at_ceiling() {
        let cfg = SacTrainingConfig {
            replay_buffer_capacity: MAX_BUFFER_CAPACITY,
            ..Default::default()
        };
        cfg.validate().expect("the ceiling itself must validate");
    }

    #[test]
    fn test_sac_config_defaults_match_cleanrl() {
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
    // Bit-exactness *is* the property under test here: `$\tau$` is a source literal stored verbatim
    // and read back unchanged, never the result of arithmetic. A tolerance would let a genuine
    // default drift pass.
    #[allow(clippy::float_cmp)]
    #[test]
    fn test_sac_config_default_target_update_is_bit_identical_to_the_pre_migration_pair() {
        let cfg = SacTrainingConfig::default();
        assert_eq!(
            cfg.target_update.tau(),
            0.005,
            "τ must survive the migration bit-for-bit"
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
    fn test_sac_config_default_config_is_valid() {
        assert!(SacTrainingConfig::default().validate().is_ok());
    }

    /// The frozen-target state is not *rejected* by `validate` here — it is
    /// unrepresentable, so the assertion belongs at the constructor (ADR 0058
    /// §Consequences). Pinned per family rather than only in `target.rs`,
    /// because the guarantee a SAC reader needs is "no `SacTrainingConfig` can
    /// hold a frozen target", and that is a statement about this config.
    #[test]
    fn test_sac_config_frozen_target_is_unreachable_through_the_type() {
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
    fn test_sac_config_nan_tau_cannot_be_constructed_for_struct_update_syntax() {
        assert!(TargetUpdate::try_polyak(f32::NAN, 1).is_err());
        assert!(TargetUpdate::try_polyak(f32::INFINITY, 1).is_err());
        // Every `$\tau$` a struct-update config can carry came through `PolyakTau`, so the result
        // is necessarily valid.
        let config = SacTrainingConfig {
            target_update: TargetUpdate::polyak(0.005, 1),
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }
}
