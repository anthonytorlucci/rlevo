//! Hyperparameter configuration for Phasic Policy Gradient (PPG).
//!
//! PPG (Cobbe et al. 2021) augments PPO with a periodic **auxiliary phase**
//! that retrains the value function and an auxiliary value head on the policy
//! network, while distilling the pre-aux-phase policy via KL. The rollout
//! collection and the policy-phase update remain identical to PPO; the extra
//! fields here control only the auxiliary-phase cadence and mix.
//!
//! Defaults follow CleanRL's [`ppg_procgen.py`](https://docs.cleanrl.dev/rl-algorithms/ppg/).
//!
//! The `ppo` field is re-used verbatim (no forking): `PpgAgent` delegates to
//! the PPO loss functions and `RolloutBuffer` from
//! [`crate::algorithms::ppo`].

use crate::algorithms::ppo::ppo_config::PpoTrainingConfig;

/// Training configuration for a PPG agent.
///
/// Extends [`PpoTrainingConfig`] with the four PPG-specific knobs: the
/// auxiliary-phase cadence, the number of auxiliary epochs, the distillation
/// coefficient, and the auxiliary minibatch size.
#[derive(Clone, Debug)]
pub struct PpgConfig {
    /// Policy-phase configuration (rollout sizing, clip coefficient, etc.).
    ///
    /// Consumed by the policy phase exactly as `PpoAgent` would consume it.
    pub ppo: PpoTrainingConfig,

    /// Number of policy-phase iterations between auxiliary phases.
    ///
    /// After `n_iteration` policy phases, the auxiliary buffer is drained
    /// through `e_aux` epochs of auxiliary updates. CleanRL default: `32`.
    pub n_iteration: usize,

    /// Auxiliary epochs per auxiliary phase.
    ///
    /// CleanRL default: `6`.
    pub e_aux: usize,

    /// Behavioral-cloning / distillation coefficient on the KL term.
    ///
    /// Scales the `KL(π_old ‖ π_new)` distillation loss added to the
    /// auxiliary-value loss on the policy network. CleanRL default: `1.0`.
    pub beta_clone: f32,

    /// Minibatch size used in the auxiliary phase.
    ///
    /// Not derived from `ppo.batch_size()` because the auxiliary buffer holds
    /// `n_iteration` rollouts, a much larger pool than one rollout.
    /// CleanRL default: `256`.
    pub aux_batch_size: usize,
}

impl Default for PpgConfig {
    fn default() -> Self {
        Self {
            ppo: PpoTrainingConfig::default(),
            n_iteration: 32,
            e_aux: 6,
            beta_clone: 1.0,
            aux_batch_size: 256,
        }
    }
}

impl PpgConfig {
    /// Transitions per rollout (delegates to the wrapped PPO config).
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.ppo.batch_size()
    }
}

/// Fluent builder for [`PpgConfig`]. Mirrors
/// [`PpoTrainingConfigBuilder`](crate::algorithms::ppo::ppo_config::PpoTrainingConfigBuilder)
/// for the PPG-specific fields; access the inner `PpoTrainingConfig` via
/// [`PpgConfigBuilder::with_ppo`] to tweak policy-phase knobs.
#[derive(Debug)]
pub struct PpgConfigBuilder {
    config: PpgConfig,
}

impl Default for PpgConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PpgConfigBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: PpgConfig::default(),
        }
    }

    /// Replaces the wrapped PPO config wholesale.
    pub fn ppo(mut self, ppo: PpoTrainingConfig) -> Self {
        self.config.ppo = ppo;
        self
    }

    /// Mutates the wrapped PPO config in place.
    pub fn with_ppo<F>(mut self, f: F) -> Self
    where
        F: FnOnce(PpoTrainingConfig) -> PpoTrainingConfig,
    {
        self.config.ppo = f(self.config.ppo);
        self
    }

    pub fn n_iteration(mut self, n: usize) -> Self {
        self.config.n_iteration = n;
        self
    }

    pub fn e_aux(mut self, e: usize) -> Self {
        self.config.e_aux = e;
        self
    }

    pub fn beta_clone(mut self, b: f32) -> Self {
        self.config.beta_clone = b;
        self
    }

    pub fn aux_batch_size(mut self, s: usize) -> Self {
        self.config.aux_batch_size = s;
        self
    }

    pub fn build(self) -> PpgConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_cleanrl() {
        let cfg = PpgConfig::default();
        assert_eq!(cfg.n_iteration, 32);
        assert_eq!(cfg.e_aux, 6);
        assert!((cfg.beta_clone - 1.0).abs() < 1e-6);
        assert_eq!(cfg.aux_batch_size, 256);
        assert_eq!(cfg.ppo.num_steps, 128);
    }

    #[test]
    fn builder_round_trips() {
        let cfg = PpgConfigBuilder::new()
            .n_iteration(8)
            .e_aux(3)
            .beta_clone(0.5)
            .aux_batch_size(64)
            .with_ppo(|p| PpoTrainingConfig {
                num_steps: 64,
                ..p
            })
            .build();
        assert_eq!(cfg.n_iteration, 8);
        assert_eq!(cfg.e_aux, 3);
        assert!((cfg.beta_clone - 0.5).abs() < 1e-6);
        assert_eq!(cfg.aux_batch_size, 64);
        assert_eq!(cfg.ppo.num_steps, 64);
    }

    #[test]
    fn batch_size_delegates() {
        let cfg = PpgConfigBuilder::new()
            .with_ppo(|p| PpoTrainingConfig {
                num_envs: 1,
                num_steps: 128,
                ..p
            })
            .build();
        assert_eq!(cfg.batch_size(), 128);
    }
}
