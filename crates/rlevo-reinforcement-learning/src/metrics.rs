//! Agent performance tracking via episode statistics.
//!
//! This module provides [`PerformanceRecord`] for representing per-episode
//! outcomes and [`AgentStats`] for accumulating them into running statistics.

use crate::MAX_BUFFER_CAPACITY;
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
///
/// All state is private: the sliding window upholds the invariant
/// `recent_history().len() <= window_size()`, which only [`Self::record`] is
/// allowed to maintain. Read the accumulated statistics through the accessors.
///
/// # Example
/// ```
/// use rlevo_reinforcement_learning::metrics::{AgentStats, PerformanceRecord};
/// # #[derive(Debug, Clone)]
/// # struct Ep { score: f32, duration: usize }
/// # impl PerformanceRecord for Ep {
/// #   fn score(&self) -> f32 { self.score }
/// #   fn duration(&self) -> usize { self.duration }
/// # }
/// let mut stats = AgentStats::<Ep>::new(16);
/// assert_eq!(stats.avg_score(), None);
///
/// stats.record(Ep { score: 1.0, duration: 10 });
/// stats.record(Ep { score: 3.0, duration: 20 });
///
/// assert_eq!(stats.avg_score(), Some(2.0));
/// assert_eq!(stats.best_score(), Some(3.0));
/// assert_eq!(stats.total_episodes(), 2);
/// assert_eq!(stats.total_steps(), 30);
/// assert_eq!(stats.recent_len(), 2);
/// assert_eq!(stats.window_size(), 16);
/// ```
#[derive(Debug, Clone)]
pub struct AgentStats<T: PerformanceRecord> {
    /// Global counter of episodes recorded so far.
    total_episodes: usize,
    /// Total environment steps taken across all episodes.
    total_steps: usize,
    /// The highest score observed across all episodes.
    best_score: Option<f32>,
    /// Fixed-size sliding window of the most recent episodes.
    recent_history: VecDeque<T>,
    /// Maximum capacity of the sliding window.
    window_size: usize,
}

impl<T: PerformanceRecord> AgentStats<T> {
    /// Creates a new `AgentStats` with a sliding window of `window_size` episodes.
    ///
    /// # Arguments
    /// * `window_size` - Maximum number of recent episodes retained for
    ///   [`Self::avg_score`]. Must be in `1..=`[`MAX_BUFFER_CAPACITY`].
    ///
    /// # Panics
    ///
    /// Panics if `window_size` is 0. A zero-length window cannot hold any
    /// history: [`Self::record`] would evict the previous entry before every
    /// push, pinning `recent_history` at a single episode and making
    /// [`Self::avg_score`] report the latest score rather than a moving
    /// average.
    ///
    /// Panics if `window_size` exceeds [`MAX_BUFFER_CAPACITY`]. The argument is
    /// handed unfiltered to [`VecDeque::with_capacity`], where an out-of-range
    /// request aborts the process on capacity overflow rather than unwinding.
    /// Every in-crate call site passes a hard-coded literal (`100`), so this
    /// bound can only be tripped by a direct API caller — an out-of-range value
    /// is a programming error at that call site, not a runtime condition to
    /// recover from, which is why it panics rather than returning a `Result`.
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        assert!(
            window_size > 0,
            "window_size must be greater than 0; a zero-length window cannot \
             hold history and would make avg_score report the latest score \
             instead of a moving average"
        );
        assert!(
            window_size <= MAX_BUFFER_CAPACITY,
            "window_size must be at most {MAX_BUFFER_CAPACITY}, got \
             {window_size}; this value is passed straight to \
             VecDeque::with_capacity, where an out-of-range request aborts the \
             process instead of unwinding"
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
    ///
    /// Both lifetime counters ([`Self::total_episodes`], [`Self::total_steps`])
    /// saturate at `usize::MAX`: a further `record` leaves them pinned there
    /// rather than wrapping to a small value in release builds or aborting the
    /// run with an "attempt to add with overflow" panic under `overflow-checks`.
    /// A long-lived agent is precisely the caller that reaches the ceiling, and
    /// for it a monotone, clamped counter is a better failure mode than either
    /// a silently rewound total or a crash on the hot path — nothing in this
    /// crate branches on the counters, they are reported statistics.
    pub fn record(&mut self, entry: T) {
        self.total_episodes = self.total_episodes.saturating_add(1);
        self.total_steps = self.total_steps.saturating_add(entry.duration());

        let score = entry.score();
        self.best_score = Some(self.best_score.map_or(score, |b| b.max(score)));

        // Maintain Sliding Window (O(1) with VecDeque)
        if self.recent_history.len() >= self.window_size {
            self.recent_history.pop_front();
        }
        self.recent_history.push_back(entry);
    }

    /// Returns the number of episodes recorded since construction.
    ///
    /// This is a global counter over all of history; it is unaffected by the
    /// sliding window and so may exceed [`Self::window_size`].
    ///
    /// The counter saturates in [`Self::record`], so a returned `usize::MAX`
    /// means "at least this many episodes", not an exact count.
    #[must_use]
    pub fn total_episodes(&self) -> usize {
        self.total_episodes
    }

    /// Returns the sum of [`PerformanceRecord::duration`] over every recorded
    /// episode.
    ///
    /// Like [`Self::total_episodes`], this accumulates over all of history
    /// rather than the sliding window, and saturates: a returned `usize::MAX`
    /// means "at least this many steps", not an exact total.
    #[must_use]
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// Returns the highest [`PerformanceRecord::score`] observed across all
    /// recorded episodes, or `None` when no episodes have been recorded yet.
    ///
    /// Unlike [`Self::avg_score`], the best score is never evicted by the
    /// sliding window.
    #[must_use]
    pub fn best_score(&self) -> Option<f32> {
        self.best_score
    }

    /// Returns the sliding window of the most recent episodes, oldest first.
    ///
    /// The returned deque holds at most [`Self::window_size`] entries
    /// (`recent_history().len() <= window_size()`); once the window is full,
    /// each [`Self::record`] evicts the front (oldest) entry before appending
    /// the new one at the back.
    #[must_use]
    pub fn recent_history(&self) -> &VecDeque<T> {
        &self.recent_history
    }

    /// Returns the current occupancy of the sliding window.
    ///
    /// Always `<= `[`Self::window_size`], and equal to it once at least
    /// `window_size` episodes have been recorded.
    #[must_use]
    pub fn recent_len(&self) -> usize {
        self.recent_history.len()
    }

    /// Returns the configured maximum size of the sliding window.
    ///
    /// This is the `window_size` passed to [`Self::new`] and never changes.
    #[must_use]
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Returns the mean [`PerformanceRecord::score`] over the sliding window of
    /// recent episodes, or `None` when no episodes have been recorded yet.
    ///
    /// The average is computed only over the episodes currently held in
    /// `recent_history` (at most `window_size` entries), not over the full
    /// episode history.
    #[must_use]
    // Divisor/normalizer derived from a count -- batch size, minibatch count,
    // history length, iteration number. All are bounded by configured sizes far
    // below f32's 2^24 (f64's 2^53) exact-integer limit.
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_score(&self) -> Option<f32> {
        if self.recent_history.is_empty() {
            None
        } else {
            let sum: f32 = self
                .recent_history
                .iter()
                .map(PerformanceRecord::score)
                .sum();
            Some(sum / self.recent_history.len() as f32)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AgentStats, MAX_BUFFER_CAPACITY, PerformanceRecord};

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

        /// Builds a record whose duration differs from 1, so that
        /// `total_steps` can never coincide with `total_episodes` and a
        /// transposed accessor is observable.
        fn with_duration(score: f32, duration: usize) -> Self {
            Self { score, duration }
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

    /// The upper end of the same guard. `window_size` reaches
    /// `VecDeque::with_capacity` unfiltered, where an out-of-range request
    /// aborts the process rather than unwinding — so reaching a catchable
    /// panic at all is the property under test.
    #[test]
    #[should_panic(expected = "window_size must be at most")]
    fn new_rejects_window_size_above_ceiling() {
        let _ = AgentStats::<TestRecord>::new(MAX_BUFFER_CAPACITY + 1);
    }

    #[test]
    // Test fixture data: the loop counter and element count are bounded by small
    // constants declared in this test, far below f32's 2^24 exact-integer limit,
    // so every generated value is represented exactly.
    #[allow(clippy::cast_precision_loss)]
    fn window_retains_n_records_then_evicts_oldest() {
        const N: usize = 3;
        let mut stats = AgentStats::<TestRecord>::new(N);

        // Filling the window retains every record; a zero-length window would
        // pin `len` at 1 and fail here on the second push.
        for i in 0..N {
            stats.record(TestRecord::new(i as f32));
            assert_eq!(stats.recent_history().len(), i + 1);
        }

        let scores: Vec<f32> = stats.recent_history().iter().map(|r| r.score).collect();
        assert_eq!(scores, vec![0.0, 1.0, 2.0]);

        // The N+1th record evicts the oldest, leaving length pinned at N.
        stats.record(TestRecord::new(3.0));
        assert_eq!(stats.recent_history().len(), N);

        let scores: Vec<f32> = stats.recent_history().iter().map(|r| r.score).collect();
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
        stats.record(TestRecord::with_duration(1.0, 7));
        stats.record(TestRecord::with_duration(2.0, 11));

        assert_eq!(stats.recent_history().len(), 1);
        assert_eq!(
            stats.recent_history()[0],
            TestRecord::with_duration(2.0, 11)
        );

        // Distinct counters: 2 episodes but 18 steps, so a transposed
        // `total_episodes`/`total_steps` pair cannot satisfy both assertions.
        assert_eq!(stats.total_episodes(), 2);
        assert_eq!(stats.total_steps(), 18);
    }

    /// A freshly constructed `AgentStats` reports empty statistics rather than
    /// zero-valued ones: `avg_score`/`best_score` are `None`, not `Some(0.0)`.
    #[test]
    fn empty_stats_report_no_average_or_best() {
        let stats = AgentStats::<TestRecord>::new(4);

        assert_eq!(stats.avg_score(), None);
        assert_eq!(stats.best_score(), None);
        assert_eq!(stats.total_episodes(), 0);
        assert_eq!(stats.total_steps(), 0);
        assert_eq!(stats.recent_len(), 0);
        // `window_size` is the configured capacity, not the occupancy, so it is
        // non-zero on an empty instance.
        assert_eq!(stats.window_size(), 4);
    }

    /// Drives the eviction boundary purely through the public accessors, with
    /// every observable quantity distinct (window 3, 4 episodes, 4/5/6/7-step
    /// durations, ascending-then-descending scores) so no pair of accessors can
    /// be swapped without failing an assertion.
    #[test]
    fn window_evicts_at_exact_capacity_via_accessors() {
        const WINDOW: usize = 3;
        let mut stats = AgentStats::<TestRecord>::new(WINDOW);

        stats.record(TestRecord::with_duration(2.0, 4));
        stats.record(TestRecord::with_duration(9.0, 5));
        stats.record(TestRecord::with_duration(4.0, 6));

        // Exactly at capacity: nothing has been evicted yet.
        assert_eq!(stats.recent_len(), WINDOW);
        assert_eq!(stats.window_size(), WINDOW);
        assert_eq!(stats.total_episodes(), 3);
        assert_eq!(stats.total_steps(), 15);
        let scores: Vec<f32> = stats.recent_history().iter().map(|r| r.score).collect();
        assert_eq!(scores, vec![2.0, 9.0, 4.0]);

        // One past capacity: the oldest entry (score 2.0) leaves the window,
        // but the global counters keep growing and `best_score` (9.0, from the
        // still-resident middle entry) is unchanged.
        stats.record(TestRecord::with_duration(1.0, 7));

        assert_eq!(stats.recent_len(), WINDOW);
        assert_eq!(stats.total_episodes(), 4);
        assert_eq!(stats.total_steps(), 22);
        let scores: Vec<f32> = stats.recent_history().iter().map(|r| r.score).collect();
        assert_eq!(scores, vec![9.0, 4.0, 1.0]);
        assert_eq!(stats.recent_history().front().map(|r| r.score), Some(9.0));
        assert_eq!(stats.recent_history().back().map(|r| r.score), Some(1.0));

        // best (9.0) != avg (14/3 ~= 4.667): the two cannot be transposed.
        assert_eq!(stats.best_score(), Some(9.0));
        let avg = stats.avg_score().expect("window is non-empty");
        assert!((avg - 14.0 / 3.0).abs() < 1e-6, "avg_score was {avg}");
    }

    /// Pins review row 1.2 ("NaN poisons `best_score` **permanently**") as
    /// **refuted**: `record` folds with `f32::max`, which returns the non-NaN
    /// operand when exactly one side is NaN, so a NaN score is simply skipped
    /// and the next finite score still advances the best. Swapping the fold for
    /// `f32::maximum` (NaN-propagating) would invert this, hence the exact-value
    /// assertions rather than a mere `is_finite` check.
    ///
    /// The final assertion records what is deliberately *not* guarded: review
    /// row 1.2b ("NaN transits `avg_score`") is open and rated Low, because the
    /// NaN clears itself once the window rolls past it. `avg_score` admitting
    /// the NaN is therefore the known, accepted contract — do not "fix" it here.
    ///
    /// This test only exercises the *argument* position of the fold (finite
    /// accumulator, NaN argument). Its sibling
    /// [`nan_as_first_record_does_not_latch_best_score`] covers the *receiver*
    /// position (NaN accumulator, finite argument), which is the only position
    /// where `b.max(score)` and a naive `if score > b` comparison diverge.
    /// Neither test subsumes the other; keep both.
    #[test]
    fn nan_score_does_not_poison_best_score() {
        // Window of 4 so all three records stay resident and `avg_score` is
        // observed over the window that actually contains the NaN.
        let mut stats = AgentStats::<TestRecord>::new(4);

        stats.record(TestRecord::with_duration(5.0, 3));
        assert_eq!(stats.best_score(), Some(5.0));

        // The NaN episode must not become — or destroy — the best score.
        stats.record(TestRecord::with_duration(f32::NAN, 4));
        assert_eq!(stats.best_score(), Some(5.0));

        // The self-healing half: a later, higher finite score still wins, which
        // it could not if the NaN had latched into `best_score`.
        stats.record(TestRecord::with_duration(9.0, 5));
        assert_eq!(stats.best_score(), Some(9.0));

        // Row 1.2b, open by design: the mean is *not* NaN-filtered, so the NaN
        // transits it for as long as that episode sits in the window. `NaN !=
        // NaN`, so this cannot be written as an `assert_eq!`.
        assert!(
            stats.avg_score().is_some_and(f32::is_nan),
            "avg_score should still admit the NaN (row 1.2b is open), got {:?}",
            stats.avg_score()
        );

        // The counters are untouched by the NaN: 3 episodes, 12 steps.
        assert_eq!(stats.total_episodes(), 3);
        assert_eq!(stats.total_steps(), 12);
    }

    /// The receiver-position half of [`nan_score_does_not_poison_best_score`]:
    /// the NaN episode arrives **first**, so `best_score` is seeded through
    /// `map_or`'s `score` branch — the accumulator itself becomes NaN, and the
    /// fold is never given a chance to discard the NaN on the way in.
    ///
    /// This ordering is what makes the test non-redundant. `b.max(score)`
    /// returns the non-NaN operand from *either* side, so `NaN.max(5.0) == 5.0`
    /// and the accumulator heals on the very next finite episode. A naive
    /// `if score > b { score } else { b }` looks equivalent but is not: every
    /// comparison against NaN is `false`, so `5.0 > NaN` keeps the NaN and
    /// `best_score` latches at NaN forever — review row 1.2's original claim,
    /// re-instated. With a finite accumulator (the sibling test) the two forms
    /// agree, which is precisely why that test cannot catch this.
    #[test]
    fn nan_as_first_record_does_not_latch_best_score() {
        let mut stats = AgentStats::<TestRecord>::new(4);

        // NaN first: `best_score` is now `Some(NaN)`, seeded directly rather
        // than folded through `max`.
        stats.record(TestRecord::with_duration(f32::NAN, 3));

        // The healing assertion. `max` yields 5.0; the naive comparison leaves
        // NaN, and `Some(NaN) != Some(5.0)`, so this is the line that kills it.
        stats.record(TestRecord::with_duration(5.0, 4));
        assert_eq!(stats.best_score(), Some(5.0));

        // Once healed, the accumulator behaves normally and still advances.
        stats.record(TestRecord::with_duration(9.0, 5));
        assert_eq!(stats.best_score(), Some(9.0));

        assert_eq!(stats.total_episodes(), 3);
        assert_eq!(stats.total_steps(), 12);
    }

    /// The lifetime step counter clamps at `usize::MAX` instead of wrapping
    /// (release) or panicking on the `overflow-checks` build (test/debug).
    ///
    /// The pre-saturation state is installed by assigning the private field
    /// directly — this module is a child of `metrics`, so no test-only
    /// constructor or setter is needed, and none should be added. Reaching
    /// `usize::MAX` through `record` alone is not feasible.
    ///
    /// The assertion is on the *value* (`usize::MAX`), never on "it did not
    /// panic": a panic-shaped test would pass against the unfixed `+=` under
    /// `cargo test --release`, where the addition silently wraps instead.
    #[test]
    fn total_steps_saturates_instead_of_wrapping() {
        let mut stats = AgentStats::<TestRecord>::new(4);
        stats.total_steps = usize::MAX - 3;

        // A 10-step episode overshoots the remaining headroom of 3.
        stats.record(TestRecord::with_duration(1.0, 10));

        assert_eq!(stats.total_steps(), usize::MAX);
        // Distinct counters: the episode counter is nowhere near saturation, so
        // a transposed accessor pair cannot satisfy both assertions.
        assert_eq!(stats.total_episodes(), 1);
    }

    /// The episode-counter half of
    /// [`total_steps_saturates_instead_of_wrapping`], and not subsumed by it:
    /// `total_episodes` accumulates a literal `1` while `total_steps`
    /// accumulates `entry.duration()`, so the two are separate call sites that
    /// can be fixed — or regress — independently.
    #[test]
    fn total_episodes_saturates_instead_of_wrapping() {
        let mut stats = AgentStats::<TestRecord>::new(4);
        stats.total_episodes = usize::MAX;

        stats.record(TestRecord::with_duration(1.0, 7));

        assert_eq!(stats.total_episodes(), usize::MAX);
        // The step counter starts at 0 and is unaffected: 7 != usize::MAX, so
        // the two assertions cannot be satisfied by a single saturated value.
        assert_eq!(stats.total_steps(), 7);
    }

    /// Eviction must not lower `best_score`: the maximum is global, while
    /// `avg_score` tracks only the window.
    #[test]
    fn best_score_survives_eviction_while_average_does_not() {
        let mut stats = AgentStats::<TestRecord>::new(2);

        stats.record(TestRecord::with_duration(10.0, 3));
        stats.record(TestRecord::with_duration(1.0, 4));
        stats.record(TestRecord::with_duration(3.0, 5));

        // 10.0 has been evicted from the window but remains the best score.
        assert_eq!(stats.recent_len(), 2);
        assert_eq!(stats.best_score(), Some(10.0));
        assert_eq!(stats.avg_score(), Some(2.0));
        assert_eq!(stats.total_episodes(), 3);
        assert_eq!(stats.total_steps(), 12);
    }
}
