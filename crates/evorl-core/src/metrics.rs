//! Agent performance tracking via episode statistics.
//!
//! This module provides [`PerformanceRecord`] for representing per-episode
//! outcomes and [`AgentStats`] for accumulating them into running statistics.

use std::collections::VecDeque;

/// The outcome of a single episode or step used for performance tracking.
pub trait PerformanceRecord: std::fmt::Debug + Clone {
    /// The primary scalar metric used for checkpointing and best-model tracking.
    fn score(&self) -> f32;

    /// The number of environment steps taken during this episode.
    fn duration(&self) -> usize;
}

/// Accumulates per-episode statistics for a running agent.
///
/// Tracks global counters, best observed score, and a fixed-size sliding
/// window of recent episodes for computing moving averages.
#[derive(Debug, Clone)]
pub struct AgentStats<T: PerformanceRecord> {
    /// Global counter of episodes recorded so far.
    pub total_episodes: usize,
    /// Total environment steps taken across all episodes.
    pub total_steps: usize,
    /// The highest score observed across all episodes.
    pub best_score: Option<f32>,
    /// Fixed-size sliding window of the most recent episodes.
    pub recent_history: VecDeque<T>,
    /// Maximum capacity of the sliding window.
    window_size: usize,
}

impl<T: PerformanceRecord> AgentStats<T> {
    /// Creates a new `AgentStats` with a sliding window of `window_size` episodes.
    pub fn new(window_size: usize) -> Self {
        Self {
            total_episodes: 0,
            total_steps: 0,
            best_score: None,
            recent_history: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    /// Records a completed episode, updating all counters and the sliding window.
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
