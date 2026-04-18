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
    /// Only meaningful when [`tau`](Self::tau) is `0`.
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
    pub fn quantile_taus<B: Backend>(&self, device: &B::Device) -> Tensor<B, 1> {
        let n = self.num_quantiles;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 + 0.5) / n as f32).collect();
        Tensor::from_data(TensorData::new(data, vec![n]), device)
    }
}

impl Default for QrDqnTrainingConfig {
    /// Returns defaults consistent with Dabney et al. 2018 QR-DQN
    /// reference hyperparameters.
    fn default() -> Self {
        Self {
            batch_size: 32,
            gamma: 0.99,
            tau: 0.005,
            learning_rate: 0.001,
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
            target_update_frequency: 100,
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
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: QrDqnTrainingConfig::default(),
        }
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }

    pub fn gamma(mut self, gamma: f64) -> Self {
        self.config.gamma = gamma;
        self
    }

    pub fn tau(mut self, tau: f64) -> Self {
        self.config.tau = tau;
        self
    }

    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }

    pub fn epsilon_start(mut self, epsilon_start: f64) -> Self {
        self.config.epsilon_start = epsilon_start;
        self
    }

    pub fn epsilon_end(mut self, epsilon_end: f64) -> Self {
        self.config.epsilon_end = epsilon_end;
        self
    }

    pub fn epsilon_decay(mut self, epsilon_decay: f64) -> Self {
        self.config.epsilon_decay = epsilon_decay;
        self
    }

    pub fn target_update_frequency(mut self, frequency: usize) -> Self {
        self.config.target_update_frequency = frequency;
        self
    }

    pub fn steps_per_episode(mut self, steps: usize) -> Self {
        self.config.steps_per_episode = steps;
        self
    }

    pub fn replay_buffer_capacity(mut self, capacity: usize) -> Self {
        self.config.replay_buffer_capacity = capacity;
        self
    }

    pub fn learning_starts(mut self, learning_starts: usize) -> Self {
        self.config.learning_starts = learning_starts;
        self
    }

    pub fn train_frequency(mut self, train_frequency: usize) -> Self {
        self.config.train_frequency = train_frequency;
        self
    }

    pub fn num_quantiles(mut self, num_quantiles: usize) -> Self {
        self.config.num_quantiles = num_quantiles;
        self
    }

    pub fn kappa(mut self, kappa: f32) -> Self {
        self.config.kappa = kappa;
        self
    }

    pub fn clip_grad(mut self, config: Option<GradientClippingConfig>) -> Self {
        self.config.clip_grad = config;
        self
    }

    pub fn optimizer(mut self, optimizer: AdamConfig) -> Self {
        self.config.optimizer = optimizer;
        self
    }

    pub fn build(self) -> QrDqnTrainingConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

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
            .build();
        assert_eq!(cfg.num_quantiles, 51);
        assert!((cfg.kappa - 0.5).abs() < f32::EPSILON);
        assert_eq!(cfg.batch_size, 128);
    }

    #[test]
    fn taus_midpoints_match_dabney_eq_9_for_n_4() {
        // N = 4 ⇒ τ_i = (i + 0.5) / 4 ⇒ {0.125, 0.375, 0.625, 0.875}.
        let device: <B as Backend>::Device = Default::default();
        let cfg = QrDqnTrainingConfigBuilder::new().num_quantiles(4).build();
        let taus: Tensor<B, 1> = cfg.quantile_taus::<B>(&device);
        let v: Vec<f32> = taus.into_data().convert::<f32>().into_vec::<f32>().unwrap();
        let expected = [0.125_f32, 0.375, 0.625, 0.875];
        for (got, want) in v.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }
}
