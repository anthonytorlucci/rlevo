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
    ///
    /// # Arguments
    /// * `window_size` - Maximum number of recent episodes retained for
    ///   [`Self::avg_score`]. Must be greater than 0.
    ///
    /// # Panics
    /// Panics if `window_size` is 0. A zero-length window cannot hold any
    /// history: [`Self::record`] would evict the previous entry before every
    /// push, pinning `recent_history` at a single episode and making
    /// [`Self::avg_score`] report the latest score rather than a moving
    /// average.
    pub fn new(window_size: usize) -> Self {
        assert!(
            window_size > 0,
            "window_size must be greater than 0; a zero-length window cannot \
             hold history and would make avg_score report the latest score \
             instead of a moving average"
        );
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

    /// Returns the mean [`PerformanceRecord::score`] over the sliding window of
    /// recent episodes, or `None` when no episodes have been recorded yet.
    ///
    /// The average is computed only over the episodes currently held in
    /// `recent_history` (at most `window_size` entries), not over the full
    /// episode history.
    pub fn avg_score(&self) -> Option<f32> {
        if self.recent_history.is_empty() {
            None
        } else {
            let sum: f32 = self.recent_history.iter().map(|r| r.score()).sum();
            Some(sum / self.recent_history.len() as f32)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AgentStats, PerformanceRecord};

    /// Minimal [`PerformanceRecord`] carrying a score and a fixed duration.
    #[derive(Debug, Clone, PartialEq)]
    struct TestRecord {
        score: f32,
        duration: usize,
    }

    impl TestRecord {
        fn new(score: f32) -> Self {
            Self { score, duration: 1 }
        }
    }

    impl PerformanceRecord for TestRecord {
        fn score(&self) -> f32 {
            self.score
        }

        fn duration(&self) -> usize {
            self.duration
        }
    }

    #[test]
    #[should_panic(expected = "window_size must be greater than 0")]
    fn new_rejects_zero_window_size() {
        let _ = AgentStats::<TestRecord>::new(0);
    }

    #[test]
    fn window_retains_n_records_then_evicts_oldest() {
        const N: usize = 3;
        let mut stats = AgentStats::<TestRecord>::new(N);

        // Filling the window retains every record; a zero-length window would
        // pin `len` at 1 and fail here on the second push.
        for i in 0..N {
            stats.record(TestRecord::new(i as f32));
            assert_eq!(stats.recent_history.len(), i + 1);
        }

        let scores: Vec<f32> = stats.recent_history.iter().map(|r| r.score).collect();
        assert_eq!(scores, vec![0.0, 1.0, 2.0]);

        // The N+1th record evicts the oldest, leaving length pinned at N.
        stats.record(TestRecord::new(3.0));
        assert_eq!(stats.recent_history.len(), N);

        let scores: Vec<f32> = stats.recent_history.iter().map(|r| r.score).collect();
        assert_eq!(scores, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn avg_score_averages_the_window_not_the_latest_record() {
        let mut stats = AgentStats::<TestRecord>::new(3);
        assert_eq!(stats.avg_score(), None);

        stats.record(TestRecord::new(0.0));
        stats.record(TestRecord::new(3.0));
        stats.record(TestRecord::new(6.0));

        // A single-entry window would report 6.0 here.
        assert!((stats.avg_score().expect("window is non-empty") - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn window_of_one_keeps_only_the_latest_record() {
        let mut stats = AgentStats::<TestRecord>::new(1);
        stats.record(TestRecord::new(1.0));
        stats.record(TestRecord::new(2.0));

        assert_eq!(stats.recent_history.len(), 1);
        assert_eq!(stats.recent_history[0], TestRecord::new(2.0));
        assert_eq!(stats.total_episodes, 2);
    }
}
