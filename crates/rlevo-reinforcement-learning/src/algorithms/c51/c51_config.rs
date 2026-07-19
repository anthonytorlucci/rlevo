//! Hyperparameter configuration for the C51 (Categorical DQN) algorithm.
//!
//! Mirrors [`crate::algorithms::dqn::dqn_config::DqnTrainingConfig`] for all
//! shared training hyperparameters and adds the three categorical
//! distributional parameters — `num_atoms`, `v_min`, and `v_max` — that
//! define the fixed support over which return distributions are represented.

use burn::grad_clipping::GradientClippingConfig;
use burn::optim::AdamConfig;
use rlevo_core::config::{self, ConfigError, ConstraintKind, Validate};

use crate::algorithms::c51::projection::atom_spacing;
use crate::replay::PrioritizedReplaySettings;

/// Configuration for training a Categorical DQN (C51) agent.
///
/// Holds all hyperparameters required to initialise and train a
/// [`crate::algorithms::c51::c51_agent::C51Agent`]. The distribution-specific
/// fields are [`num_atoms`](Self::num_atoms), [`v_min`](Self::v_min), and
/// [`v_max`](Self::v_max); the rest are the standard DQN knobs (learning
/// rate, γ, τ, ε schedule, replay capacity, …).
#[derive(Clone, Debug)]
pub struct C51TrainingConfig {
    /// Minibatch size sampled from the replay buffer each learn step.
    pub batch_size: usize,

    /// Discount factor γ in `[0, 1]`.
    pub gamma: f64,

    /// Polyak τ used for the per-step soft update of the target network.
    ///
    /// `target ← (1 − τ) · target + τ · policy`. Set to `0` to disable
    /// soft updates and rely on periodic hard syncs.
    pub tau: f64,

    /// Optimizer learning rate.
    pub learning_rate: f64,

    /// Initial ε value for the ε-greedy exploration schedule.
    pub epsilon_start: f64,

    /// Floor ε value for the exploration schedule.
    pub epsilon_end: f64,

    /// Multiplicative decay applied to ε each env step.
    pub epsilon_decay: f64,

    /// Period, in env steps, between hard syncs of the target network.
    ///
    /// Only meaningful when [`tau`](Self::tau) is `0`; while `tau > 0` the
    /// Polyak soft update is the live mechanism and this field is inert.
    ///
    /// The unit is **environment steps** — not parameter updates and not
    /// gradient updates. The default of `10_000` matches
    /// Stable-Baselines3's `target_update_interval`, which is counted in the
    /// same unit. It is *not* the Nature-DQN figure: Mnih et al. (2015)
    /// specify `C = 10,000` **parameter updates** (Extended Data Table 1),
    /// which under the default [`train_frequency`](Self::train_frequency) of
    /// `4` would be 40,000 environment steps. So 10,000 env steps is roughly
    /// 2,500 parameter updates — four times more frequent than Nature's `C`,
    /// and an exact match to the SB3 convention.
    pub target_update_frequency: usize,

    /// Upper bound on steps allowed per episode (for bookkeeping only —
    /// environments are responsible for enforcing their own step limits).
    pub steps_per_episode: usize,

    /// Maximum number of transitions retained in the replay buffer.
    pub replay_buffer_capacity: usize,

    /// Number of env steps collected before the first gradient update.
    pub learning_starts: usize,

    /// Period, in env steps, between gradient updates.
    pub train_frequency: usize,

    /// Number of atoms `N` in the categorical return distribution.
    ///
    /// The default is 51 — the value from the original C51 paper, from which
    /// the algorithm takes its name.
    pub num_atoms: usize,

    /// Lower bound of the atom support `z_0`.
    pub v_min: f32,

    /// Upper bound of the atom support `z_{N-1}`.
    pub v_max: f32,

    /// Optional gradient-norm / gradient-value clipping.
    pub clip_grad: Option<GradientClippingConfig>,

    /// Optimizer configuration (Adam β's, ε, etc.).
    pub optimizer: AdamConfig,

    /// Opt-in prioritized experience replay (Schaul et al. 2016), `None` by
    /// default (uniform replay).
    ///
    /// The priority signal for C51 is the **KL divergence**
    /// `D_KL(target ‖ pred)`, *not* the cross-entropy the gradient uses — the
    /// two differ by the per-sample target entropy `H(target)` (Rainbow,
    /// "prioritize transitions by the KL loss, since this is what the algorithm
    /// is minimizing"; see
    /// [`categorical_kl_per_sample`](crate::algorithms::c51::loss::categorical_kl_per_sample)).
    ///
    /// **Literature-recommended exponent.** Rainbow pairs the KL priority with
    /// `ω = 0.5` (its `priority_exponent` here), and reports performance is "very
    /// robust to the choice of ω" under the KL priority. The shipped default is
    /// Schaul's `0.6`; set `priority_exponent` to `0.5` on
    /// [`PrioritizedReplaySettings`] to match Rainbow. Buffer capacity comes from
    /// [`replay_buffer_capacity`](Self::replay_buffer_capacity).
    pub prioritized_replay: Option<PrioritizedReplaySettings>,
}

impl C51TrainingConfig {
    /// Spacing between adjacent atoms: `(v_max − v_min) / (num_atoms − 1)`.
    ///
    /// Delegates to [`atom_spacing`] so the support this config builds and the
    /// bin indices the categorical projection computes share one definition of
    /// `Δz` — an ULP of disagreement between the two is enough to push a
    /// scatter index off the end of the support.
    ///
    /// # Returns
    /// The uniform atom spacing, or [`f32::NAN`] for a degenerate
    /// `num_atoms < 2` (a spacing is undefined with fewer than two atoms).
    /// [`Validate`] rejects such a config, but a struct literal can bypass the
    /// builder, so this stays total rather than panicking (see issue #326).
    #[must_use]
    pub fn delta_z(&self) -> f32 {
        atom_spacing(self.v_min, self.v_max, self.num_atoms)
    }
}

impl Default for C51TrainingConfig {
    /// Returns defaults consistent with CleanRL's reference C51 hyperparameters.
    ///
    /// [`target_update_frequency`](Self::target_update_frequency) is
    /// `10_000` environment steps, matching Stable-Baselines3's
    /// `target_update_interval`. It is inert under these defaults because
    /// `tau = 0.005 > 0` selects the Polyak soft update.
    fn default() -> Self {
        Self {
            batch_size: 32,
            gamma: 0.99,
            tau: 0.005,
            learning_rate: 0.001,
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
            target_update_frequency: 10_000,
            steps_per_episode: 1000,
            replay_buffer_capacity: 10_000,
            learning_starts: 1_000,
            train_frequency: 4,
            num_atoms: 51,
            v_min: -10.0,
            v_max: 10.0,
            clip_grad: Some(GradientClippingConfig::Value(100.0)),
            optimizer: AdamConfig::new(),
            prioritized_replay: None,
        }
    }
}

impl Validate for C51TrainingConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "C51TrainingConfig";
        config::nonzero(C, "batch_size", self.batch_size)?;
        config::in_range(C, "gamma", 0.0, 1.0, self.gamma)?;
        config::in_range(C, "tau", 0.0, 1.0, self.tau)?;
        config::positive(C, "learning_rate", self.learning_rate)?;
        config::in_range(C, "epsilon_start", 0.0, 1.0, self.epsilon_start)?;
        config::in_range(C, "epsilon_end", 0.0, 1.0, self.epsilon_end)?;
        config::in_range(C, "epsilon_decay", 0.0, 1.0, self.epsilon_decay)?;
        config::nonzero(C, "replay_buffer_capacity", self.replay_buffer_capacity)?;
        config::nonzero(C, "train_frequency", self.train_frequency)?;
        config::nonzero(C, "steps_per_episode", self.steps_per_episode)?;
        config::at_least(C, "num_atoms", self.num_atoms, 2)?;
        config::distinct(C, "v_max", f64::from(self.v_min), f64::from(self.v_max))?;
        config::ordered(C, "v_max", f64::from(self.v_min), f64::from(self.v_max))?;
        if let Some(per) = &self.prioritized_replay {
            per.validate()?;
        }
        // The target network has exactly two update mechanisms — Polyak soft
        // updates (`tau > 0`) and periodic hard syncs
        // (`target_update_frequency > 0`). Disabling both leaves the target
        // frozen at its initial weights for the whole run, which silently
        // trains against a random bootstrap. `tau` is already range-checked
        // to `[0, 1]` above, so `<= 0.0` here is exactly "τ is zero".
        if self.tau <= 0.0 && self.target_update_frequency == 0 {
            return Err(ConfigError {
                config: C,
                field: "target_update_frequency",
                kind: ConstraintKind::Custom(
                    "target network would never update: set tau > 0 for Polyak \
                     soft updates, or target_update_frequency > 0 for hard syncs",
                ),
            });
        }
        Ok(())
    }
}

/// Fluent builder for [`C51TrainingConfig`]. All unset fields fall back to
/// [`C51TrainingConfig::default`].
#[derive(Debug)]
pub struct C51TrainingConfigBuilder {
    config: C51TrainingConfig,
}

impl Default for C51TrainingConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl C51TrainingConfigBuilder {
    /// Creates a builder pre-populated with [`C51TrainingConfig::default`] values.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: C51TrainingConfig::default(),
        }
    }

    /// Sets [`C51TrainingConfig::batch_size`].
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }

    /// Sets [`C51TrainingConfig::gamma`].
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.config.gamma = gamma;
        self
    }

    /// Sets [`C51TrainingConfig::tau`].
    ///
    /// Pass `0.0` to disable soft updates and rely on periodic hard syncs
    /// instead (see [`C51TrainingConfig::target_update_frequency`]).
    pub fn tau(mut self, tau: f64) -> Self {
        self.config.tau = tau;
        self
    }

    /// Sets [`C51TrainingConfig::learning_rate`].
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }

    /// Sets [`C51TrainingConfig::epsilon_start`].
    pub fn epsilon_start(mut self, epsilon_start: f64) -> Self {
        self.config.epsilon_start = epsilon_start;
        self
    }

    /// Sets [`C51TrainingConfig::epsilon_end`].
    pub fn epsilon_end(mut self, epsilon_end: f64) -> Self {
        self.config.epsilon_end = epsilon_end;
        self
    }

    /// Sets [`C51TrainingConfig::epsilon_decay`].
    pub fn epsilon_decay(mut self, epsilon_decay: f64) -> Self {
        self.config.epsilon_decay = epsilon_decay;
        self
    }

    /// Sets [`C51TrainingConfig::target_update_frequency`].
    pub fn target_update_frequency(mut self, frequency: usize) -> Self {
        self.config.target_update_frequency = frequency;
        self
    }

    /// Sets [`C51TrainingConfig::steps_per_episode`].
    pub fn steps_per_episode(mut self, steps: usize) -> Self {
        self.config.steps_per_episode = steps;
        self
    }

    /// Sets [`C51TrainingConfig::replay_buffer_capacity`].
    pub fn replay_buffer_capacity(mut self, capacity: usize) -> Self {
        self.config.replay_buffer_capacity = capacity;
        self
    }

    /// Sets [`C51TrainingConfig::learning_starts`].
    pub fn learning_starts(mut self, learning_starts: usize) -> Self {
        self.config.learning_starts = learning_starts;
        self
    }

    /// Sets [`C51TrainingConfig::train_frequency`].
    pub fn train_frequency(mut self, train_frequency: usize) -> Self {
        self.config.train_frequency = train_frequency;
        self
    }

    /// Sets [`C51TrainingConfig::num_atoms`].
    ///
    /// The paper default is `51`; smaller values (e.g. `21`) reduce memory and
    /// compute at the cost of distributional resolution.
    pub fn num_atoms(mut self, num_atoms: usize) -> Self {
        self.config.num_atoms = num_atoms;
        self
    }

    /// Sets [`C51TrainingConfig::v_min`].
    pub fn v_min(mut self, v_min: f32) -> Self {
        self.config.v_min = v_min;
        self
    }

    /// Sets [`C51TrainingConfig::v_max`].
    pub fn v_max(mut self, v_max: f32) -> Self {
        self.config.v_max = v_max;
        self
    }

    /// Sets [`C51TrainingConfig::clip_grad`].
    ///
    /// Pass `None` to disable gradient clipping entirely.
    pub fn clip_grad(mut self, config: Option<GradientClippingConfig>) -> Self {
        self.config.clip_grad = config;
        self
    }

    /// Sets [`C51TrainingConfig::optimizer`].
    pub fn optimizer(mut self, optimizer: AdamConfig) -> Self {
        self.config.optimizer = optimizer;
        self
    }

    /// Enables prioritized experience replay with the given settings.
    ///
    /// C51 prioritizes by the KL loss (Rainbow); see
    /// [`C51TrainingConfig::prioritized_replay`] for the literature-recommended
    /// `ω = 0.5`. Leave unset for uniform replay.
    pub fn prioritized_replay(mut self, settings: PrioritizedReplaySettings) -> Self {
        self.config.prioritized_replay = Some(settings);
        self
    }

    /// Consumes the builder and returns the finalised [`C51TrainingConfig`].
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if the assembled config violates any invariant
    /// checked by [`C51TrainingConfig::validate`] (e.g. `v_min >= v_max` or
    /// fewer than two atoms).
    pub fn build(self) -> Result<C51TrainingConfig, ConfigError> {
        self.config.validate()?;
        Ok(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_51_atoms_on_symmetric_support() {
        let cfg = C51TrainingConfig::default();
        assert_eq!(cfg.num_atoms, 51);
        assert_eq!(cfg.v_min, -10.0);
        assert_eq!(cfg.v_max, 10.0);
    }

    #[test]
    fn delta_z_matches_uniform_spacing() {
        let cfg = C51TrainingConfigBuilder::new()
            .num_atoms(51)
            .v_min(-10.0)
            .v_max(10.0)
            .build()
            .expect("valid config");
        // 50 gaps spanning 20.0 → 0.4 per atom.
        assert!((cfg.delta_z() - 0.4).abs() < 1e-6);
    }

    #[test]
    fn builder_round_trips_distributional_fields() {
        let cfg = C51TrainingConfigBuilder::new()
            .num_atoms(21)
            .v_min(-5.0)
            .v_max(5.0)
            .batch_size(128)
            .build()
            .expect("valid config");
        assert_eq!(cfg.num_atoms, 21);
        assert_eq!(cfg.v_min, -5.0);
        assert_eq!(cfg.v_max, 5.0);
        assert_eq!(cfg.batch_size, 128);
        assert!((cfg.delta_z() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn default_config_is_valid() {
        assert!(C51TrainingConfig::default().validate().is_ok());
    }

    #[test]
    fn rejects_config_where_target_never_updates() {
        let err = C51TrainingConfigBuilder::new()
            .tau(0.0)
            .target_update_frequency(0)
            .build()
            .unwrap_err();
        assert_eq!(
            err.field, "target_update_frequency",
            "tau == 0 with no hard-sync period must be rejected: the target would never update"
        );
    }

    #[test]
    fn accepts_either_target_update_mechanism_alone() {
        assert!(
            C51TrainingConfigBuilder::new()
                .tau(0.0)
                .target_update_frequency(100)
                .build()
                .is_ok(),
            "hard sync alone is a valid target-update mechanism"
        );
        assert!(
            C51TrainingConfigBuilder::new()
                .tau(0.005)
                .target_update_frequency(0)
                .build()
                .is_ok(),
            "Polyak soft update alone is a valid target-update mechanism"
        );
        assert!(
            C51TrainingConfigBuilder::new()
                .tau(0.005)
                .target_update_frequency(100)
                .build()
                .is_ok(),
            "both set is legal: tau wins and the frequency is an inert fallback (this is Default)"
        );
    }

    #[test]
    fn rejects_degenerate_support() {
        let err = C51TrainingConfigBuilder::new()
            .v_min(5.0)
            .v_max(5.0)
            .build()
            .unwrap_err();
        assert_eq!(err.field, "v_max");
    }
}
