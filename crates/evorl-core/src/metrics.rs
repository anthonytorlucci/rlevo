use std::collections::VecDeque;

/// A trait representing the result of a step or an episode.
pub trait PerformanceRecord: std::fmt::Debug + Clone {
    /// The primary metric used for checkpointing/best-model tracking
    fn score(&self) -> f32;

    /// The duration of the event (steps in an episode)
    fn duration(&self) -> usize;
}

#[derive(Debug, Clone)]
pub struct AgentStats<T: PerformanceRecord> {
    /// Global counter of episodes seen
    pub total_episodes: usize,
    /// Global counter of environment steps taken
    pub total_steps: usize,
    /// The highest score observed so far
    pub best_score: Option<f32>, // Renamed from best_episode_reward
    /// Sliding window of the most recent episodes
    pub recent_history: VecDeque<T>,
    ///Maximum capacity of the sliding window
    window_size: usize,
}

impl<T: PerformanceRecord> AgentStats<T> {
    pub fn new(window_size: usize) -> Self {
        Self {
            total_episodes: 0,
            total_steps: 0,
            best_score: None,
            recent_history: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    pub fn record(&mut self, record: T) {
        self.total_episodes += 1;
        self.total_steps += record.duration();

        let score = record.score();
        self.best_score = Some(self.best_score.map_or(score, |b| b.max(score)));

        // Maintain Sliding Window (O(1) with VecDeque)
        if self.recent_history.len() >= self.window_size {
            self.recent_history.pop_front();
        }
        self.recent_history.push_back(record);
    }

    /// Calculate average main score
    pub fn avg_score(&self) -> Option<f32> {
        if self.recent_history.is_empty() {
            None
        } else {
            let sum: f32 = self.recent_history.iter().map(|r| r.score()).sum();
            Some(sum / self.recent_history.len() as f32)
        }
    }
}
