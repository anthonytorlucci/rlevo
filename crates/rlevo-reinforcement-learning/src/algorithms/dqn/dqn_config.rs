//! Hyperparameter configuration for the DQN algorithm.

use burn::grad_clipping::GradientClippingConfig;
use burn::optim::AdamConfig;
use rlevo_core::config::{self, ConfigError, Validate};

/// Configuration structure for training a Deep Q-Network (DQN).
///
/// This struct holds all hyperparameters and configuration settings required
/// to initialize and train a DQN agent, including learning rates,
/// epsilon-greedy parameters, and optimization settings.
#[derive(Clone, Debug)]
pub struct DqnTrainingConfig {
    /// The number of samples processed before the model is updated.
    pub batch_size: usize,

    /// The discount factor ($\gamma$).
    ///
    /// A value between 0 and 1 that balances the importance of immediate rewards
    /// versus future rewards. A value close to 0 makes the agent short-sighted,
    /// while a value close to 1 makes it strive for long-term high reward.
    pub gamma: f64,

    /// The target network update rate ($\tau$) for soft updates.
    ///
    /// Used to update the weights of the target Q-network slowly to stabilize training.
    /// The formula is $Q_{\text{target}} = (1-\tau) Q_{\text{target}} + \tau Q_{\text{main}}$.
    pub tau: f64,

    /// The learning rate used by the optimizer.
    pub learning_rate: f64,

    /// The starting value for epsilon in the epsilon-greedy exploration strategy.
    ///
    /// Represents the initial probability of choosing a random action over the greedy action.
    pub epsilon_start: f64,

    /// The minimum value that epsilon can decay to.
    ///
    /// Ensures that there is always a small chance of exploration.
    pub epsilon_end: f64,

    /// The decay rate for epsilon.
    ///
    /// Multiplicative factor applied to epsilon after each step or episode to reduce exploration over time.
    pub epsilon_decay: f64,

    /// Interval (in environment steps) between hard target-network syncs.
    ///
    /// When [`tau`](Self::tau) is `0.0`, the target network is updated by
    /// copying the policy network weights wholesale every
    /// `target_update_frequency` steps. When `tau > 0.0`, soft Polyak
    /// averaging is used inside every [`DqnAgent::learn_step`] call and this
    /// field is ignored. Set to `0` to disable hard syncing entirely.
    ///
    /// [`DqnAgent::learn_step`]: crate::algorithms::dqn::dqn_agent::DqnAgent::learn_step
    pub target_update_frequency: usize,

    /// The maximum number of steps allowed per episode.
    pub steps_per_episode: usize,

    /// The maximum number of transitions to store in the replay buffer.
    pub replay_buffer_capacity: usize,

    /// Number of environment steps collected before learning starts.
    ///
    /// Acts as a warm-up period that fills the replay buffer with diverse
    /// transitions before the first gradient update, stabilising early
    /// training.
    pub learning_starts: usize,

    /// How often (in environment steps) a learning update is performed.
    ///
    /// `train_frequency = 4` means one gradient step every four env steps,
    /// matching the Nature-DQN setting.
    pub train_frequency: usize,

    /// If `true`, compute bootstrap targets using Double-DQN
    /// (`a* = argmax_a Q_online(s', a)`, then `y = Q_target(s', a*)`).
    ///
    /// Leave `false` for vanilla DQN.
    pub double_q: bool,

    /// Configuration for gradient clipping.
    ///
    /// Prevents exploding gradients by scaling the gradient vector if its norm exceeds a threshold.
    /// Set to `None` to disable clipping.
    pub clip_grad: Option<GradientClippingConfig>,

    /// Configuration for the optimizer (e.g., Adam).
    ///
    /// This defines the optimization algorithm used to update the network weights.
    pub optimizer: AdamConfig,
}

impl Default for DqnTrainingConfig {
    /// Returns sensible defaults suited to small discrete-action environments.
    ///
    /// Key values: `batch_size = 32`, `gamma = 0.99`, `tau = 0.005`
    /// (soft updates active), `learning_rate = 1e-3`, `epsilon_start = 1.0`
    /// decaying to `0.01` at rate `0.995`, `replay_buffer_capacity = 10_000`,
    /// `learning_starts = 1_000`, `train_frequency = 4`, `double_q = false`,
    /// gradient clipping at norm 100.
    ///
    /// Because `tau > 0.0`, `target_update_frequency` is ignored by the
    /// default configuration — soft updates run every learn step.
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
            replay_buffer_capacity: 10000,
            learning_starts: 1000,
            train_frequency: 4,
            double_q: false,
            clip_grad: Some(GradientClippingConfig::Value(100.0)),
            optimizer: AdamConfig::new(),
        }
    }
}

impl Validate for DqnTrainingConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "DqnTrainingConfig";
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
        Ok(())
    }
}

/// Builder for [`DqnTrainingConfig`] with fluent setters.
///
/// All unset fields default to the values from [`DqnTrainingConfig::default`].
///
/// # Examples
///
/// ```rust
/// use rlevo_reinforcement_learning::algorithms::dqn::dqn_config::DqnTrainingConfigBuilder;
///
/// // Default configuration.
/// let cfg = DqnTrainingConfigBuilder::new().build().expect("valid config");
///
/// // Custom learning rate and batch size.
/// let cfg = DqnTrainingConfigBuilder::new()
///     .learning_rate(0.0005)
///     .batch_size(64)
///     .build()
///     .expect("valid config");
/// ```
pub struct DqnTrainingConfigBuilder {
    config: DqnTrainingConfig,
}

impl Default for DqnTrainingConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DqnTrainingConfigBuilder {
    /// Creates a new builder initialized with default configuration values.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: DqnTrainingConfig::default(),
        }
    }

    /// Sets the batch size.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }

    /// Sets the discount factor (gamma).
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.config.gamma = gamma;
        self
    }

    /// Sets the target network update rate (tau).
    pub fn tau(mut self, tau: f64) -> Self {
        self.config.tau = tau;
        self
    }

    /// Sets the learning rate.
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }

    /// Sets the starting epsilon value for exploration.
    pub fn epsilon_start(mut self, epsilon_start: f64) -> Self {
        self.config.epsilon_start = epsilon_start;
        self
    }

    /// Sets the minimum epsilon value.
    pub fn epsilon_end(mut self, epsilon_end: f64) -> Self {
        self.config.epsilon_end = epsilon_end;
        self
    }

    /// Sets the epsilon decay rate.
    pub fn epsilon_decay(mut self, epsilon_decay: f64) -> Self {
        self.config.epsilon_decay = epsilon_decay;
        self
    }

    /// Sets the target update frequency.
    pub fn target_update_frequency(mut self, frequency: usize) -> Self {
        self.config.target_update_frequency = frequency;
        self
    }

    /// Sets the maximum steps per episode.
    pub fn steps_per_episode(mut self, steps: usize) -> Self {
        self.config.steps_per_episode = steps;
        self
    }

    /// Sets the capacity of the replay buffer.
    pub fn replay_buffer_capacity(mut self, capacity: usize) -> Self {
        self.config.replay_buffer_capacity = capacity;
        self
    }

    /// Sets the number of warm-up steps before learning begins.
    pub fn learning_starts(mut self, learning_starts: usize) -> Self {
        self.config.learning_starts = learning_starts;
        self
    }

    /// Sets how often a learning update runs, in environment steps.
    pub fn train_frequency(mut self, train_frequency: usize) -> Self {
        self.config.train_frequency = train_frequency;
        self
    }

    /// Enables or disables Double-DQN bootstrap targets.
    pub fn double_q(mut self, double_q: bool) -> Self {
        self.config.double_q = double_q;
        self
    }

    /// Sets the gradient clipping configuration.
    pub fn clip_grad(mut self, config: Option<GradientClippingConfig>) -> Self {
        self.config.clip_grad = config;
        self
    }

    /// Sets the optimizer configuration (e.g., specific Adam beta values).
    pub fn optimizer(mut self, optimizer: AdamConfig) -> Self {
        self.config.optimizer = optimizer;
        self
    }

    /// Consumes the builder and returns the final `DqnTrainingConfig`.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if the assembled config violates any invariant
    /// checked by [`DqnTrainingConfig::validate`] (e.g. a zero `batch_size` or a
    /// `gamma` outside `[0, 1]`).
    pub fn build(self) -> Result<DqnTrainingConfig, ConfigError> {
        self.config.validate()?;
        Ok(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        assert!(DqnTrainingConfig::default().validate().is_ok());
    }

    #[test]
    fn rejects_gamma_out_of_range() {
        let err = DqnTrainingConfigBuilder::new()
            .gamma(1.5)
            .build()
            .unwrap_err();
        assert_eq!(err.field, "gamma");
    }

    /// `tau` is a `pub` field, so struct-update syntax (and `Deserialize`)
    /// bypasses the guarded builder entirely and can hand an agent a `NaN`
    /// Polyak coefficient, which would poison every target-net weight on the
    /// first soft update. `DqnAgent::new` is the sole constructor and calls
    /// `config.validate()?`, so pinning the rejection here proves the `NaN`
    /// can never reach an agent.
    #[test]
    fn rejects_nan_tau_from_struct_update_syntax() {
        let config = DqnTrainingConfig {
            tau: f64::NAN,
            ..Default::default()
        };
        let err = config
            .validate()
            .expect_err("NaN tau must be rejected before it can reach DqnAgent::new");
        assert_eq!(err.field, "tau", "NaN tau must be reported against `tau`");
    }
}
