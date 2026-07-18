//! Hyperparameter configuration for the QR-DQN (Quantile Regression DQN)
//! algorithm.
//!
//! Mirrors [`crate::algorithms::c51::c51_config::C51TrainingConfig`] for all
//! shared training hyperparameters, but replaces the categorical
//! distributional fields (`num_atoms`, `v_min`, `v_max`) with
//! [`num_quantiles`](QrDqnTrainingConfig::num_quantiles) and the Huber
//! threshold [`kappa`](QrDqnTrainingConfig::kappa). QR-DQN learns a
//! fixed-cardinality quantile function rather than a categorical
//! distribution over a fixed support, so no `[v_min, v_max]` range is
//! required.

use burn::grad_clipping::GradientClippingConfig;
use burn::optim::AdamConfig;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use rlevo_core::config::{self, ConfigError, ConstraintKind, Validate};

/// Configuration for training a Quantile Regression DQN (QR-DQN) agent.
///
/// Holds all hyperparameters required to initialise and train a
/// [`crate::algorithms::qrdqn::qrdqn_agent::QrDqnAgent`]. The distribution-
/// specific fields are [`num_quantiles`](Self::num_quantiles) and
/// [`kappa`](Self::kappa); the rest are the standard DQN knobs (learning
/// rate, γ, τ, ε schedule, replay capacity, …).
#[derive(Clone, Debug)]
pub struct QrDqnTrainingConfig {
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
    /// Only meaningful when [`tau`](Self::tau) is `0`. When `tau > 0` the
    /// Polyak soft update is the live mechanism and this field is an inert
    /// fallback —
    /// [`QrDqnAgent::sync_target`](crate::algorithms::qrdqn::qrdqn_agent::QrDqnAgent::sync_target)
    /// is a no-op, so the hard copy can never erase the soft-update lag.
    ///
    /// Setting both this field and `tau` to `0` is rejected by
    /// [`validate`](Validate::validate): the target network would never update
    /// at all.
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

    /// Number of quantiles `N` used to represent the return distribution.
    ///
    /// The default is 200 — the value from Dabney et al. (2018). Quantile
    /// midpoints `τ_i = (i + 0.5) / N` are implied by this field alone.
    pub num_quantiles: usize,

    /// Huber threshold `κ` used by the quantile Huber loss. Values below
    /// `κ` are treated quadratically, above `κ` linearly. Default `1.0`.
    pub kappa: f32,

    /// Optional gradient-norm / gradient-value clipping.
    pub clip_grad: Option<GradientClippingConfig>,

    /// Optimizer configuration (Adam β's, ε, etc.).
    pub optimizer: AdamConfig,
}

impl QrDqnTrainingConfig {
    /// Builds the quantile-midpoint tensor `[(i + 0.5) / N]_{i=0..N-1}` on
    /// the requested backend and device. Shape: `(num_quantiles,)`.
    pub fn quantile_taus<B: Backend>(
        &self,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Tensor<B, 1> {
        let n = self.num_quantiles;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 + 0.5) / n as f32).collect();
        Tensor::from_data(TensorData::new(data, vec![n]), device)
    }
}

impl Default for QrDqnTrainingConfig {
    /// Returns defaults consistent with Dabney et al. 2018 QR-DQN
    /// reference hyperparameters.
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
            num_quantiles: 200,
            kappa: 1.0,
            clip_grad: Some(GradientClippingConfig::Value(100.0)),
            optimizer: AdamConfig::new(),
        }
    }
}

impl Validate for QrDqnTrainingConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "QrDqnTrainingConfig";
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
        config::nonzero(C, "num_quantiles", self.num_quantiles)?;
        config::in_range(C, "kappa", 0.0, f64::INFINITY, f64::from(self.kappa))?;
        // `tau` is already range-checked to `[0, 1]` above, so `<= 0.0` is
        // exactly "soft updates disabled" (and avoids a float `==`).
        if self.tau <= 0.0 && self.target_update_frequency == 0 {
            return Err(ConfigError {
                config: C,
                field: "target_update_frequency",
                kind: ConstraintKind::Custom(
                    "target network would never update: set tau > 0 for Polyak soft updates, \
                     or target_update_frequency > 0 for periodic hard syncs",
                ),
            });
        }
        Ok(())
    }
}

/// Fluent builder for [`QrDqnTrainingConfig`]. All unset fields fall back to
/// [`QrDqnTrainingConfig::default`].
#[derive(Debug)]
pub struct QrDqnTrainingConfigBuilder {
    config: QrDqnTrainingConfig,
}

impl Default for QrDqnTrainingConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl QrDqnTrainingConfigBuilder {
    /// Creates a builder pre-populated with [`QrDqnTrainingConfig::default`] values.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: QrDqnTrainingConfig::default(),
        }
    }

    /// Sets [`QrDqnTrainingConfig::batch_size`].
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }

    /// Sets [`QrDqnTrainingConfig::gamma`] (discount factor γ).
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.config.gamma = gamma;
        self
    }

    /// Sets [`QrDqnTrainingConfig::tau`] (Polyak soft-update coefficient).
    ///
    /// Pass `0.0` to disable soft updates; the target network will then be
    /// refreshed only via periodic hard syncs controlled by
    /// [`target_update_frequency`](Self::target_update_frequency).
    pub fn tau(mut self, tau: f64) -> Self {
        self.config.tau = tau;
        self
    }

    /// Sets [`QrDqnTrainingConfig::learning_rate`].
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }

    /// Sets [`QrDqnTrainingConfig::epsilon_start`] — initial ε for ε-greedy.
    pub fn epsilon_start(mut self, epsilon_start: f64) -> Self {
        self.config.epsilon_start = epsilon_start;
        self
    }

    /// Sets [`QrDqnTrainingConfig::epsilon_end`] — floor ε for ε-greedy.
    pub fn epsilon_end(mut self, epsilon_end: f64) -> Self {
        self.config.epsilon_end = epsilon_end;
        self
    }

    /// Sets [`QrDqnTrainingConfig::epsilon_decay`] — per-step multiplicative decay.
    pub fn epsilon_decay(mut self, epsilon_decay: f64) -> Self {
        self.config.epsilon_decay = epsilon_decay;
        self
    }

    /// Sets [`QrDqnTrainingConfig::target_update_frequency`].
    ///
    /// Only meaningful when [`tau`](Self::tau) is `0.0`; ignored otherwise.
    ///
    /// # Errors
    ///
    /// [`build`](Self::build) rejects `frequency == 0` combined with
    /// `tau == 0.0`, since that leaves the target network frozen forever.
    pub fn target_update_frequency(mut self, frequency: usize) -> Self {
        self.config.target_update_frequency = frequency;
        self
    }

    /// Sets [`QrDqnTrainingConfig::steps_per_episode`].
    pub fn steps_per_episode(mut self, steps: usize) -> Self {
        self.config.steps_per_episode = steps;
        self
    }

    /// Sets [`QrDqnTrainingConfig::replay_buffer_capacity`].
    pub fn replay_buffer_capacity(mut self, capacity: usize) -> Self {
        self.config.replay_buffer_capacity = capacity;
        self
    }

    /// Sets [`QrDqnTrainingConfig::learning_starts`].
    pub fn learning_starts(mut self, learning_starts: usize) -> Self {
        self.config.learning_starts = learning_starts;
        self
    }

    /// Sets [`QrDqnTrainingConfig::train_frequency`].
    pub fn train_frequency(mut self, train_frequency: usize) -> Self {
        self.config.train_frequency = train_frequency;
        self
    }

    /// Sets [`QrDqnTrainingConfig::num_quantiles`] (`N` in Dabney et al. 2018).
    pub fn num_quantiles(mut self, num_quantiles: usize) -> Self {
        self.config.num_quantiles = num_quantiles;
        self
    }

    /// Sets [`QrDqnTrainingConfig::kappa`] (Huber loss threshold κ).
    pub fn kappa(mut self, kappa: f32) -> Self {
        self.config.kappa = kappa;
        self
    }

    /// Sets [`QrDqnTrainingConfig::clip_grad`].
    ///
    /// Pass `None` to disable gradient clipping entirely.
    pub fn clip_grad(mut self, config: Option<GradientClippingConfig>) -> Self {
        self.config.clip_grad = config;
        self
    }

    /// Replaces the default [`AdamConfig`] with a custom optimizer configuration.
    pub fn optimizer(mut self, optimizer: AdamConfig) -> Self {
        self.config.optimizer = optimizer;
        self
    }

    /// Consumes the builder and returns the assembled [`QrDqnTrainingConfig`].
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if the assembled config violates any invariant
    /// checked by [`QrDqnTrainingConfig::validate`] (e.g. zero `num_quantiles`
    /// or a negative `kappa`).
    pub fn build(self) -> Result<QrDqnTrainingConfig, ConfigError> {
        self.config.validate()?;
        Ok(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;

    type B = Flex;

    #[test]
    fn defaults_are_200_quantiles_with_kappa_1() {
        let cfg = QrDqnTrainingConfig::default();
        assert_eq!(cfg.num_quantiles, 200);
        assert!((cfg.kappa - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn builder_round_trips_distributional_fields() {
        let cfg = QrDqnTrainingConfigBuilder::new()
            .num_quantiles(51)
            .kappa(0.5)
            .batch_size(128)
            .build()
            .expect("valid config");
        assert_eq!(cfg.num_quantiles, 51);
        assert!((cfg.kappa - 0.5).abs() < f32::EPSILON);
        assert_eq!(cfg.batch_size, 128);
    }

    #[test]
    fn default_config_is_valid() {
        assert!(QrDqnTrainingConfig::default().validate().is_ok());
    }

    #[test]
    fn rejects_zero_quantiles() {
        let err = QrDqnTrainingConfigBuilder::new()
            .num_quantiles(0)
            .build()
            .unwrap_err();
        assert_eq!(err.field, "num_quantiles");
    }

    #[test]
    fn rejects_frozen_target_when_tau_and_frequency_are_both_zero() {
        let err = QrDqnTrainingConfigBuilder::new()
            .tau(0.0)
            .target_update_frequency(0)
            .build()
            .unwrap_err();
        assert_eq!(
            err.field, "target_update_frequency",
            "a config in which the target network never updates must be rejected"
        );
    }

    #[test]
    fn accepts_soft_updates_with_an_inert_hard_sync_frequency() {
        // The workspace default sets both; under the `sync_target` gate the
        // frequency is simply inert, so this must stay a legal config
        // (ADR 0026: a library default must pass its own `validate`).
        let cfg = QrDqnTrainingConfigBuilder::new()
            .tau(0.005)
            .target_update_frequency(100)
            .build();
        assert!(
            cfg.is_ok(),
            "tau > 0 alongside a nonzero target_update_frequency is legal"
        );
    }

    #[test]
    fn accepts_hard_sync_only_configuration() {
        let cfg = QrDqnTrainingConfigBuilder::new()
            .tau(0.0)
            .target_update_frequency(500)
            .build();
        assert!(cfg.is_ok(), "tau = 0 with periodic hard syncs is legal");
    }

    #[test]
    fn accepts_soft_update_only_configuration() {
        let cfg = QrDqnTrainingConfigBuilder::new()
            .tau(0.01)
            .target_update_frequency(0)
            .build();
        assert!(cfg.is_ok(), "tau > 0 with hard sync disabled is legal");
    }

    #[test]
    fn taus_midpoints_match_dabney_eq_9_for_n_4() {
        // N = 4 ⇒ τ_i = (i + 0.5) / 4 ⇒ {0.125, 0.375, 0.625, 0.875}.
        let device: <B as burn::tensor::backend::BackendTypes>::Device = Default::default();
        let cfg = QrDqnTrainingConfigBuilder::new()
            .num_quantiles(4)
            .build()
            .expect("valid config");
        let taus: Tensor<B, 1> = cfg.quantile_taus::<B>(&device);
        let v: Vec<f32> = taus
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("f32 host read of a tensor this test just built");
        let expected = [0.125_f32, 0.375, 0.625, 0.875];
        for (got, want) in v.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }
}
