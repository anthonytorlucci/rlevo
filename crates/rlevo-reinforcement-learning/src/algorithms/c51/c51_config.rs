//! Hyperparameter configuration for the C51 (Categorical DQN) algorithm.
//!
//! Mirrors [`crate::algorithms::dqn::dqn_config::DqnTrainingConfig`] for all
//! shared training hyperparameters and adds the three categorical
//! distributional parameters — `num_atoms`, `v_min`, and `v_max` — that
//! define the fixed support over which return distributions are represented.

use burn::grad_clipping::GradientClippingConfig;
use burn::optim::AdamConfig;

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
}

impl C51TrainingConfig {
    /// Spacing between adjacent atoms: `(v_max − v_min) / (num_atoms − 1)`.
    #[must_use]
    pub fn delta_z(&self) -> f32 {
        (self.v_max - self.v_min) / (self.num_atoms.saturating_sub(1) as f32)
    }
}

impl Default for C51TrainingConfig {
    /// Returns defaults consistent with CleanRL's reference C51 hyperparameters.
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
            num_atoms: 51,
            v_min: -10.0,
            v_max: 10.0,
            clip_grad: Some(GradientClippingConfig::Value(100.0)),
            optimizer: AdamConfig::new(),
        }
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
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: C51TrainingConfig::default(),
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

    pub fn num_atoms(mut self, num_atoms: usize) -> Self {
        self.config.num_atoms = num_atoms;
        self
    }

    pub fn v_min(mut self, v_min: f32) -> Self {
        self.config.v_min = v_min;
        self
    }

    pub fn v_max(mut self, v_max: f32) -> Self {
        self.config.v_max = v_max;
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

    pub fn build(self) -> C51TrainingConfig {
        self.config
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
            .build();
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
            .build();
        assert_eq!(cfg.num_atoms, 21);
        assert_eq!(cfg.v_min, -5.0);
        assert_eq!(cfg.v_max, 5.0);
        assert_eq!(cfg.batch_size, 128);
        assert!((cfg.delta_z() - 0.5).abs() < 1e-6);
    }
}
