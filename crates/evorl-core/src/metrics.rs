/// Core performance metrics tracked at any granularity (step, episode, or agent level)
#[derive(Debug, Clone, Copy)]
pub struct CoreMetrics {
    /// Total or average reward for the period
    pub reward: f32,
    /// Loss value if applicable (None for evaluation)
    pub loss: Option<f32>,
    /// Current exploration rate (epsilon)
    pub exploration_rate: f64,
}

impl CoreMetrics {
    /// Create new metrics
    pub fn new(reward: f32, loss: Option<f32>, exploration_rate: f64) -> Self {
        Self {
            reward,
            loss,
            exploration_rate,
        }
    }
}

/// Statistics for a single training episode
#[derive(Debug, Clone)]
pub struct EpisodeStats {
    /// Episode number (0-indexed)
    pub episode_number: usize,
    /// Number of steps in this episode
    pub total_steps: usize,
    /// Cumulative reward for the episode
    pub total_reward: f32,
    /// Core metrics for the episode
    pub metrics: CoreMetrics,
}

impl EpisodeStats {
    /// Calculate average reward per step
    pub fn avg_reward_per_step(&self) -> f32 {
        self.total_reward / self.total_steps.max(1) as f32
    }
}

/// Aggregated statistics for the agent across all training
#[derive(Debug, Clone)]
pub struct AgentStats {
    /// Total number of episodes completed
    pub total_episodes: usize,
    /// Total number of steps across all episodes
    pub total_steps: usize,
    /// Best total reward achieved in any single episode
    pub best_episode_reward: Option<f32>,
    /// Current exploration rate
    pub current_exploration_rate: f64,
    /// Recent episode stats for rolling averages (sliding window)
    pub recent_episodes: Vec<EpisodeStats>,
}

impl AgentStats {
    /// Create new agent stats with a fixed history size
    pub fn new(current_exploration_rate: f64, history_size: usize) -> Self {
        Self {
            total_episodes: 0,
            total_steps: 0,
            best_episode_reward: None,
            current_exploration_rate,
            recent_episodes: Vec::with_capacity(history_size),
        }
    }

    /// Record a completed episode
    pub fn record_episode(&mut self, episode: EpisodeStats) {
        self.total_episodes += 1;
        self.total_steps += episode.total_steps;

        // Update best reward
        self.best_episode_reward = Some(
            self.best_episode_reward
                .map_or(episode.total_reward, |best| best.max(episode.total_reward)),
        );

        // Maintain sliding window of recent episodes
        if self.recent_episodes.len() >= self.recent_episodes.capacity() {
            self.recent_episodes.remove(0);
        }
        self.recent_episodes.push(episode);
    }

    /// Calculate average reward over recent episodes
    pub fn avg_reward(&self) -> Option<f32> {
        if self.recent_episodes.is_empty() {
            None
        } else {
            Some(
                self.recent_episodes
                    .iter()
                    .map(|e| e.total_reward)
                    .sum::<f32>()
                    / self.recent_episodes.len() as f32,
            )
        }
    }

    /// Calculate average loss over recent episodes
    pub fn avg_loss(&self) -> Option<f32> {
        let losses: Vec<f32> = self
            .recent_episodes
            .iter()
            .filter_map(|e| e.metrics.loss)
            .collect();
        if losses.is_empty() {
            None
        } else {
            Some(losses.iter().sum::<f32>() / losses.len() as f32)
        }
    }
}
