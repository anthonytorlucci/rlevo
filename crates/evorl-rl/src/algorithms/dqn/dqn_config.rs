use burn::grad_clipping::GradientClippingConfig;

pub struct DQNTrainingConfig {
    pub batch_size: usize,
    pub gamma: f64, // discount factor; a hyperparameter that balances the importance of immediate rewards versus future rewards. It is a value between 0 and 1
    pub tau: f64, // target network update rate (or soft update parameter); used for updating the weights of the target Q-network in a process called a "soft update". The target network provides stable targets for the main network's training, preventing oscillations. The formula is $Q_{\text{target}} = (1-\tau ) Q_{\text{target} } + \tau Q_{\text{main} }$
    pub learning_rate: f64,
    pub epsilon_start: f64,
    pub epsilon_end: f64,
    pub epsilon_decay: f64,
    pub target_update_frequency: usize,
    pub steps_per_episode: usize,
    pub replay_buffer_capacity: usize,
    pub clip_grad: Option<GradientClippingConfig>,
}

impl Default for DQNTrainingConfig {
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
        }
    }
}
