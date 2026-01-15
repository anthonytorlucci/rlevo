use burn::grad_clipping::GradientClippingConfig;
use burn::optim::AdamConfig;

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

    /// The number of steps between strict syncs of the target network (if not using soft updates)
    /// or frequency of logging/evaluation.
    pub target_update_frequency: usize,

    /// The maximum number of steps allowed per episode.
    pub steps_per_episode: usize,

    /// The maximum number of transitions to store in the replay buffer.
    pub replay_buffer_capacity: usize,

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
    /// Creates a configuration with standard default values suitable for many gym environments.
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
            clip_grad: Some(GradientClippingConfig::Value(100.0)),
            optimizer: AdamConfig::new(),
        }
    }
}

/// A Builder for creating `DqnTrainingConfig` instances.
///
/// This allows for setting specific configuration parameters while falling back
/// to reasonable defaults for unspecified values.
pub struct DqnTrainingConfigBuilder {
    config: DqnTrainingConfig,
}

/// Example
/// Create a config using the default parameters via the builder.
/// ```ignore
/// let default_config: DqnTrainingConfig = DqnTrainingConfig::builder()
///     .build();
/// ```
/// Create a config with custom modifications (e.g., changing learning rate and optimizer).
/// ```ignore
/// let custom_config: DqnTrainingConfig = DqnTrainingConfig::builder()
///     .learning_rate(0.0005)
///     .batch_size(64)
///     .optimizer(AdamConfig::new().with_weight_decay(1e-5))
///     .build();
/// ```
impl DqnTrainingConfigBuilder {
    /// Creates a new builder initialized with the default configuration.
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
    pub fn build(self) -> DqnTrainingConfig {
        self.config
    }
}
