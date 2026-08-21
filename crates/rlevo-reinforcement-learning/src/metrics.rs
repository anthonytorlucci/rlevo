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
    /// The highest **finite** score observed across all episodes.
    finite_best_score: Option<f32>,
    /// Lifetime count of recorded episodes whose score was non-finite.
    non_finite_episodes: usize,
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
            finite_best_score: None,
            non_finite_episodes: 0,
            recent_history: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    /// Records a completed episode, updating all counters and the sliding window.
    ///
    /// All three lifetime counters ([`Self::total_episodes`],
    /// [`Self::total_steps`], [`Self::non_finite_episodes`])
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
        if score.is_finite() {
            self.finite_best_score = Some(self.finite_best_score.map_or(score, |b| b.max(score)));
        } else {
            self.non_finite_episodes = self.non_finite_episodes.saturating_add(1);
        }

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
    ///
    /// # A `$+\infty$` score latches here permanently, by decision
    ///
    /// [`Self::record`] folds this with `f32::max`, which **discards** a `NaN`
    /// operand — pinned by the tests `nan_score_does_not_poison_best_score` and
    /// `nan_as_first_record_does_not_latch_best_score`, which cover the argument
    /// and receiver positions of the fold separately — but **propagates** `$+\infty$`,
    /// in either operand position, because `$+\infty$` compares greater than every
    /// finite value.
    ///
    /// Combined with "never evicted" above, a single `$+\infty$` episode pins this
    /// accessor at `$+\infty$` for the entire remaining lifetime of the agent, with no
    /// self-healing of any kind. That is strictly worse than [`Self::avg_score`]'s
    /// transit of a non-finite score, which heals within one
    /// [`Self::window_size`] as the offending episode rolls out of the window.
    ///
    /// **This is not filtered, and must not be.** `$+\infty$` genuinely *is* the true
    /// maximum score observed, and reporting something smaller under the name
    /// "best score" is exactly the fabrication ADR 0061 forbids and ADR 0070
    /// §Rejected alternatives ruled out for the mean.
    ///
    /// Two paths reach it, and both are reachable on an agent that is fully
    /// guarded today:
    ///
    /// - ADR 0065's `FiniteRewardGuard` refuses to *store* a non-finite
    ///   per-step reward, but all eight training loops still accumulate
    ///   `episode_reward += reward_f32` unconditionally and by design (ADR 0065
    ///   §Decision 4), precisely so the bad value still surfaces as a statistic.
    ///   A single `$+\infty$` environment reward therefore reaches [`Self::record`]
    ///   even with every ingestion guard in place.
    /// - A long episode of entirely *finite* rewards can saturate the `f32`
    ///   `episode_reward` accumulator to `$+\infty$` by ordinary accumulation. There is
    ///   no non-finite reward anywhere on that path, `dropped_transitions()`
    ///   reads zero, and no `warn!` fires (ADR 0070 §Context 4).
    ///
    /// `$-\infty$` is harmless here: `f32::max` discards it as the smaller operand at
    /// the first finite episode.
    ///
    /// # When you want the hardened statistic instead
    ///
    /// Use [`Self::finite_best_score`] together with
    /// [`Self::non_finite_episodes`] — the lifetime analogue of the
    /// [`Self::finite_avg_score`] / [`Self::non_finite_recent_len`] pair, and
    /// additive for the same reason (ADR 0071).
    #[must_use]
    pub fn best_score(&self) -> Option<f32> {
        self.best_score
    }

    /// Returns the highest **finite** [`PerformanceRecord::score`] observed
    /// across all recorded episodes, or `None` when no finite score has ever
    /// been recorded.
    ///
    /// This is the hardened counterpart to [`Self::best_score`]: a single `$+\infty$`
    /// episode cannot latch it for the lifetime of the agent. Ship it with
    /// [`Self::non_finite_episodes`], which reports how many episodes this
    /// maximum excluded; a hardened maximum *without* its exclusion count is the
    /// silent sanitization ADR 0065 refuses.
    ///
    /// # Why the predicate is `is_finite`, not `!is_nan`
    ///
    /// The mean side ([`Self::finite_avg_score`]) argues this from `$\pm\infty$`
    /// destroying a mean. The maximum side has a sharper argument of its own:
    /// `f32::max` **already** discards `NaN` for free, so a `!is_nan()`
    /// predicate here would make this method observationally identical to
    /// [`Self::best_score`] at every input *except* the one — `$+\infty$` — that this
    /// method exists for. That is a hardened name over an unhardened statistic,
    /// which is the worst available outcome (ADR 0070 §Decision 3).
    ///
    /// `is_finite()` is additionally what `FiniteRewardGuard::admit` and
    /// [`Self::finite_avg_score`] test for, so all three agree on what "bad"
    /// means rather than each carrying a private definition.
    ///
    /// # Lifetime, and it stores state — unlike its windowed cousin
    ///
    /// This maximum is over all of history, so it **cannot** be derived from
    /// `recent_history` the way [`Self::non_finite_recent_len`] is: the episode
    /// that set it may have been evicted from the window many episodes ago.
    /// [`Self::record`] therefore maintains it, and `AgentStats` carries a
    /// field for it.
    ///
    /// ADR 0070 §Decision 5 declined to store state for the windowed count, and
    /// that objection does **not** transfer here. It was an objection to a
    /// counter that would need **decrementing on `pop_front`** — a new
    /// correctness obligation coupling a counter to a deque eviction, which a
    /// later refactor can break silently. A lifetime maximum and a lifetime
    /// count have no eviction coupling at all: nothing about them changes when
    /// the window rolls, so no new invariant lands on the hot path.
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
    /// let mut stats = AgentStats::<Ep>::new(8);
    /// assert_eq!(stats.finite_best_score(), None);
    ///
    /// stats.record(Ep { score: 5.0, duration: 3 });
    /// stats.record(Ep { score: f32::INFINITY, duration: 4 });
    /// stats.record(Ep { score: 3.0, duration: 5 });
    /// stats.record(Ep { score: 100.0, duration: 6 });
    ///
    /// // The raw maximum latches at `+∞` and never recovers -- by decision.
    /// assert_eq!(stats.best_score(), Some(f32::INFINITY));
    /// // The hardened maximum keeps advancing past the latch.
    /// assert_eq!(stats.finite_best_score(), Some(100.0));
    /// assert_eq!(stats.non_finite_episodes(), 1);
    /// ```
    #[must_use]
    pub fn finite_best_score(&self) -> Option<f32> {
        self.finite_best_score
    }

    /// Returns how many recorded episodes carried a non-finite (`NaN` or `$\pm\infty$`)
    /// [`PerformanceRecord::score`], over all of history.
    ///
    /// This is the count of exactly what [`Self::finite_best_score`] excluded,
    /// and the two are one decision rather than two. The count carries what a
    /// maximum cannot: whether the `$+\infty$` latch on [`Self::best_score`] came from
    /// one episode in a million or from a third of the run. Unlike a windowed
    /// mean, a maximum gives no other signal about how often the bad value
    /// occurred — once latched it never moves again, so its value is identical
    /// in both cases.
    ///
    /// # Lifetime, not windowed — and it does not heal
    ///
    /// Named to parallel [`Self::total_episodes`] so that "lifetime, not
    /// windowed" is audible at the call site. This is a **deliberate** contrast
    /// with [`Self::non_finite_recent_len`], which parallels [`Self::recent_len`],
    /// is derived from the window on every call, and returns to zero once the
    /// window rolls past the offending episode.
    ///
    /// The two will disagree on a run that went bad early and recovered, and
    /// that disagreement is the point: this one answers "did this run *ever* go
    /// bad?", the windowed one answers "is it bad *now*?". It is the same split
    /// ADR 0065 §Decision 1 drew for the agents' monotone
    /// `dropped_transitions()`.
    ///
    /// Always `<= `[`Self::total_episodes`].
    ///
    /// # Saturation
    ///
    /// The counter saturates at `usize::MAX` in [`Self::record`], alongside the
    /// other two lifetime counters, so a returned `usize::MAX` means "at least
    /// this many non-finite episodes", not an exact count.
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
    /// let mut stats = AgentStats::<Ep>::new(2);
    /// stats.record(Ep { score: f32::NAN, duration: 3 });
    ///
    /// // While the bad episode is resident, both counts agree.
    /// assert_eq!(stats.non_finite_recent_len(), 1);
    /// assert_eq!(stats.non_finite_episodes(), 1);
    ///
    /// // Two clean episodes evict it from a window of 2.
    /// stats.record(Ep { score: 1.0, duration: 4 });
    /// stats.record(Ep { score: 3.0, duration: 5 });
    ///
    /// // The windowed count heals; the lifetime count does not.
    /// assert_eq!(stats.non_finite_recent_len(), 0);
    /// assert_eq!(stats.non_finite_episodes(), 1);
    /// assert_eq!(stats.finite_best_score(), Some(3.0));
    /// ```
    #[must_use]
    pub fn non_finite_episodes(&self) -> usize {
        self.non_finite_episodes
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
    ///
    /// # Non-finite scores transit this average, by decision
    ///
    /// A `NaN` or `$\pm\infty$` score is **not** filtered out. It propagates into the
    /// mean and stays there until the sliding window rolls past the offending
    /// episode, so this method can return `Some(NaN)` or `Some(±∞)` even though
    /// [`Self::best_score`] does not — [`Self::record`] folds the best with
    /// `f32::max`, which discards a `NaN` operand rather than latching it.
    ///
    /// That asymmetry is a recorded contract (ADR 0065 §Decision 4,
    /// re-affirmed by ADR 0070), **not an oversight, and it must not be
    /// "fixed" here.** The reasoning: an episode return is the primary
    /// scientific measurement this crate produces. The six off-policy agents
    /// refuse to *store* a non-finite reward at replay ingestion, but every
    /// training loop still accumulates each step's reward into
    /// `episode_reward` unconditionally — precisely so that a non-finite
    /// reward still surfaces here. This is a second, independent detection
    /// channel, and unlike the ingestion drop guard it does not share that
    /// guard's decade-scheduled `warn!` throttling, so a poisoned run is
    /// visible in the reported curve on the very next episode.
    ///
    /// It is also load-bearing for the test harness: `assert_reaches` and
    /// `assert_improves_over_random` in `rlevo-test-support` both gate their
    /// convergence check on the average being finite. Filtering here would let
    /// a `NaN`-poisoned run quietly pass a threshold it never actually met;
    /// admitting the `NaN` makes it fail loudly instead.
    ///
    /// # When you want the hardened statistic instead
    ///
    /// Use [`Self::finite_avg_score`] together with
    /// [`Self::non_finite_recent_len`]. That pair is the direct analogue of
    /// `StrategyMetrics::mean_fitness`/`broken_count` in `rlevo-evolution`
    /// (ADR 0034): a mean over the well-behaved entries, shipped alongside the
    /// count of what it excluded, so nothing is sanitized silently.
    ///
    /// Rule of thumb: reach for the pair when you are **reporting** a learning
    /// curve; keep `avg_score` when you are **detecting** a problem.
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

    /// Returns the mean [`PerformanceRecord::score`] over the **finite**
    /// entries of the sliding window, or `None` when the window holds no
    /// finite entry.
    ///
    /// This is the hardened counterpart to [`Self::avg_score`]: a single
    /// poisoned episode cannot blank the reported curve for the next
    /// `window_size` episodes. `None` is returned in exactly two cases — the
    /// window is empty, or every resident entry is non-finite — which are the
    /// two situations in which "the mean of the good entries" has no answer.
    /// Ship it with [`Self::non_finite_recent_len`], which reports how many
    /// entries this method excluded; a hardened mean *without* that count is
    /// silent sanitization, which ADR 0065 refuses.
    ///
    /// # Why the predicate is `is_finite`, not `!is_nan`
    ///
    /// An earlier design excluded `NaN` only. This method is deliberately
    /// stronger and excludes `$\pm\infty$` as well, for two reasons:
    ///
    /// - `$\pm\infty$` destroys a mean as thoroughly as `NaN` does, and it is what
    ///   ADR 0065's own `FiniteRewardGuard` tests for — a `!is_nan` predicate
    ///   here would disagree with the ingestion guard about what "bad" means.
    /// - `$\pm\infty$` is reachable in a training loop's `episode_reward` from a run of
    ///   entirely *finite* per-step rewards, by `f32` saturation over a long
    ///   episode. No ingestion guard covers that path, because every
    ///   individual reward it inspected was finite; the overflow happens in
    ///   the accumulator afterwards.
    ///
    /// # Numerics
    ///
    /// The sum is accumulated in `f64` and narrowed to `f32` exactly once,
    /// after the division — the same shape as `rlevo-evolution`'s
    /// `sanitized_mean` (ADR 0069 §Decision 1). An `f32` accumulator can
    /// saturate to `$+\infty$` partway through a window of large-but-finite scores
    /// and so manufacture the very non-finite value this method exists to
    /// exclude. On the healthy path the wider accumulator changes nothing the
    /// caller can observe at exactly-representable inputs.
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
    /// let mut stats = AgentStats::<Ep>::new(8);
    /// assert_eq!(stats.finite_avg_score(), None);
    ///
    /// stats.record(Ep { score: 2.0, duration: 3 });
    /// stats.record(Ep { score: f32::NAN, duration: 4 });
    /// stats.record(Ep { score: f32::INFINITY, duration: 5 });
    /// stats.record(Ep { score: 8.0, duration: 6 });
    ///
    /// // The hardened mean ignores both bad episodes; the raw mean does not.
    /// assert_eq!(stats.finite_avg_score(), Some(5.0));
    /// assert_eq!(stats.non_finite_recent_len(), 2);
    /// assert!(!stats.avg_score().expect("window is non-empty").is_finite());
    /// ```
    #[must_use]
    pub fn finite_avg_score(&self) -> Option<f32> {
        let mut sum = 0.0_f64;
        let mut count = 0_usize;
        for entry in &self.recent_history {
            let score = entry.score();
            if score.is_finite() {
                sum += f64::from(score);
                count += 1;
            }
        }
        if count == 0 {
            return None;
        }
        // Divisor derived from a count, bounded by `window_size` and hence by
        // `MAX_BUFFER_CAPACITY` -- far below f64's 2^53 exact-integer limit.
        #[allow(clippy::cast_precision_loss)]
        let n = count as f64;
        // The single narrowing of the whole reduction: every term is a finite
        // `f32`, so their mean is within `f32` range and cannot overflow.
        #[allow(clippy::cast_possible_truncation)]
        let mean = (sum / n) as f32;
        Some(mean)
    }

    /// Returns how many entries of the sliding window carry a non-finite
    /// (`NaN` or `$\pm\infty$`) [`PerformanceRecord::score`].
    ///
    /// This is the count of exactly what [`Self::finite_avg_score`] excluded,
    /// and shipping the two together is the load-bearing half of the pair: a
    /// hardened mean reported on its own is the silent sanitization ADR 0065
    /// refuses. The count also carries information the mean cannot — whether a
    /// stray `NaN` is one episode in a hundred or a third of the window. The
    /// pair mirrors `StrategyMetrics::mean_fitness`/`broken_count` in
    /// `rlevo-evolution` (ADR 0034); see ADR 0070.
    ///
    /// # Windowed, and derived on each call
    ///
    /// Named to parallel [`Self::recent_len`] so that "windowed, not lifetime"
    /// is audible at the call site. Nothing is stored: the count is recomputed
    /// from `recent_history` on every call, so it is self-healing — once the
    /// window rolls past the offending episode the count returns to zero.
    ///
    /// This is a deliberate contrast with the six off-policy agents'
    /// `dropped_transitions()`, which is a **monotone lifetime** counter
    /// (ADR 0065 §Decision 1). The two will disagree on a run that saw a
    /// non-finite value early and recovered, and that disagreement is the
    /// point: one answers "did this run ever go bad?", the other answers "is
    /// it bad *now*?".
    ///
    /// Always `<= `[`Self::recent_len`].
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
    /// let mut stats = AgentStats::<Ep>::new(2);
    /// assert_eq!(stats.non_finite_recent_len(), 0);
    ///
    /// stats.record(Ep { score: f32::NEG_INFINITY, duration: 3 });
    /// assert_eq!(stats.non_finite_recent_len(), 1);
    ///
    /// // Self-healing: two clean episodes evict the bad one from the window.
    /// stats.record(Ep { score: 1.0, duration: 4 });
    /// stats.record(Ep { score: 3.0, duration: 5 });
    /// assert_eq!(stats.non_finite_recent_len(), 0);
    /// assert_eq!(stats.finite_avg_score(), Some(2.0));
    /// ```
    #[must_use]
    pub fn non_finite_recent_len(&self) -> usize {
        self.recent_history
            .iter()
            .filter(|entry| !entry.score().is_finite())
            .count()
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
    /// The final assertion records a **contract, not an unfixed gap**. Review
    /// row 1.2b ("NaN transits `avg_score`") is *decided*, not open pending a
    /// fix: `avg_score` admits the NaN deliberately, as one of ADR 0065
    /// §Decision 4's surfacing channels for a non-finite episode return.
    /// ADR 0070 revisited it and re-affirmed the behaviour rather
    /// than filtering, adding [`AgentStats::finite_avg_score`] and
    /// [`AgentStats::non_finite_recent_len`] alongside for callers that want
    /// the hardened statistic. **Do not "fix" `avg_score` to satisfy this line
    /// — the line is the specification.** Filtering there would also blind
    /// `assert_reaches`/`assert_improves_over_random` in `rlevo-test-support`,
    /// which gate a run's convergence check on the average being finite; a
    /// poisoned run would then pass a threshold it never met.
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

        // Row 1.2b, decided by design (ADR 0065 §Decision 4, re-affirmed by ADR
        // 0070): the mean is *not* NaN-filtered, so the NaN transits it for as
        // long as that episode sits in the window. `NaN != NaN`, so this cannot
        // be written as an `assert_eq!`.
        assert!(
            stats.avg_score().is_some_and(f32::is_nan),
            "avg_score must still admit the NaN (ADR 0065 §D4 / ADR 0070), got {:?}",
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

    /// The test that separates `is_finite()` from the `!is_nan()` predicate
    /// an earlier design originally proposed (ADR 0070).
    ///
    /// Mutant killed: swapping either new method's predicate to `!is_nan()`.
    /// Such an implementation admits `$+\infty$` into the sum and returns
    /// `Some(f32::INFINITY)` with a count of `0`, so the assertions are on
    /// exact values — an `is_finite()` check on the result would pass against
    /// the mutant for the count and only accidentally fail for the mean.
    #[test]
    fn positive_infinity_is_excluded_not_merely_nan() {
        let mut stats = AgentStats::<TestRecord>::new(4);

        stats.record(TestRecord::with_duration(3.0, 3));
        stats.record(TestRecord::with_duration(f32::INFINITY, 4));
        stats.record(TestRecord::with_duration(7.0, 5));

        assert_eq!(stats.finite_avg_score(), Some(5.0));
        assert_eq!(stats.non_finite_recent_len(), 1);
        // The raw mean still admits the infinity, per ADR 0065 §Decision 4.
        assert_eq!(stats.avg_score(), Some(f32::INFINITY));
    }

    /// A window in which *every* resident entry is non-finite has no
    /// mean-over-finite-entries, and says so.
    ///
    /// Mutants killed: returning `Some(0.0)` from a `0/0` guarded to zero, and
    /// returning `Some(NaN)` from an unguarded `0.0 / 0.0`. `assert_eq!` with
    /// `None` kills both — `Some(NaN) != None` holds regardless of NaN's
    /// self-inequality, which is why the assertion is not written as a
    /// `is_none_or(f32::is_nan)`-style check.
    #[test]
    fn all_non_finite_window_has_no_finite_average() {
        let mut stats = AgentStats::<TestRecord>::new(4);

        stats.record(TestRecord::with_duration(f32::NAN, 3));
        stats.record(TestRecord::with_duration(f32::INFINITY, 4));
        stats.record(TestRecord::with_duration(f32::NEG_INFINITY, 5));

        assert_eq!(stats.finite_avg_score(), None);
        // Distinct from the empty-window case: the window is full of entries,
        // all of them bad.
        assert_eq!(stats.non_finite_recent_len(), 3);
        assert_eq!(stats.recent_len(), 3);
        assert_eq!(stats.total_steps(), 12);
    }

    /// All four observable quantities are distinct, so no pair of accessors
    /// can be transposed and still satisfy every assertion.
    ///
    /// Mutants killed: `finite_avg_score` dividing by `recent_len()` rather
    /// than the finite count (that yields `15/5 = 3.0`, not `5.0`);
    /// `non_finite_recent_len` returning the finite count (`3`, deliberately
    /// different from the non-finite count `2`) or the window occupancy (`5`);
    /// and any implementation that also sanitizes `avg_score`.
    #[test]
    fn mixed_window_reports_four_distinct_observables() {
        let mut stats = AgentStats::<TestRecord>::new(5);

        stats.record(TestRecord::with_duration(2.0, 3));
        stats.record(TestRecord::with_duration(f32::NAN, 4));
        stats.record(TestRecord::with_duration(f32::INFINITY, 5));
        stats.record(TestRecord::with_duration(8.0, 6));
        stats.record(TestRecord::with_duration(5.0, 7));

        assert_eq!(stats.finite_avg_score(), Some(5.0));
        assert_eq!(stats.non_finite_recent_len(), 2);
        assert_eq!(stats.recent_len(), 5);
        let avg = stats.avg_score().expect("window is non-empty");
        assert!(
            !avg.is_finite(),
            "avg_score must still admit the non-finite entries, got {avg}"
        );
    }

    /// The window statistic is self-healing: once the bad episode is evicted,
    /// the count returns to zero and the two means agree again.
    ///
    /// Mutant killed: caching the non-finite tally as a monotone field updated
    /// in `record` — the `rlevo`-wide `dropped_transitions()` shape. That
    /// mutant reports `1` after eviction; a derived count reports `0`.
    ///
    /// Mirror image of
    /// [`non_finite_episodes_is_a_lifetime_count_not_a_window_count`], which
    /// runs the same history against the *stored lifetime* counter and expects
    /// the opposite answer. Keep both: together they are the only executable
    /// statement of the windowed/lifetime split.
    #[test]
    fn eviction_heals_the_non_finite_count() {
        let mut stats = AgentStats::<TestRecord>::new(2);

        stats.record(TestRecord::with_duration(f32::NAN, 3));
        assert_eq!(stats.non_finite_recent_len(), 1);

        // Two clean episodes push the NaN out of a window of 2.
        stats.record(TestRecord::with_duration(1.0, 4));
        stats.record(TestRecord::with_duration(3.0, 5));

        assert_eq!(stats.non_finite_recent_len(), 0);
        assert_eq!(stats.finite_avg_score(), Some(2.0));
        assert_eq!(stats.avg_score(), Some(2.0));
        // Lifetime counters are unaffected by the healing: 3 episodes, 12 steps.
        assert_eq!(stats.total_episodes(), 3);
        assert_eq!(stats.total_steps(), 12);
    }

    /// The change is additive: on a healthy window the hardened mean reports
    /// the *same* value as `avg_score`, bit for bit.
    ///
    /// Mutant killed: an `f64` accumulator that perturbs the reported value at
    /// exactly-representable inputs (e.g. narrowing per-term instead of once,
    /// or dividing in `f32` after an `f64` sum). The scores and count here are
    /// exact in both widths, so `assert_eq!` on the two means is meaningful
    /// and no epsilon is warranted.
    #[test]
    fn all_finite_window_matches_avg_score_exactly() {
        let mut stats = AgentStats::<TestRecord>::new(4);

        stats.record(TestRecord::with_duration(1.0, 3));
        stats.record(TestRecord::with_duration(2.0, 4));
        stats.record(TestRecord::with_duration(4.0, 5));
        stats.record(TestRecord::with_duration(9.0, 6));

        assert_eq!(stats.finite_avg_score(), stats.avg_score());
        assert_eq!(stats.finite_avg_score(), Some(4.0));
        assert_eq!(stats.non_finite_recent_len(), 0);
        assert_eq!(stats.recent_len(), 4);
    }

    /// An empty window reports absence, not a zero-valued statistic — matching
    /// [`AgentStats::avg_score`]'s existing `None`.
    ///
    /// Mutant killed: `finite_avg_score` returning `Some(0.0)` on an empty
    /// window, which would read as "the agent scored zero" on a fresh run.
    #[test]
    fn empty_window_has_no_finite_average_and_no_bad_entries() {
        let stats = AgentStats::<TestRecord>::new(4);

        assert_eq!(stats.finite_avg_score(), None);
        assert_eq!(stats.non_finite_recent_len(), 0);
        assert_eq!(stats.avg_score(), None);
        assert_eq!(stats.recent_len(), 0);
    }

    /// The executable form of ADR 0070 §Correction (b) and of the decision that
    /// the latch is *kept*: `best_score` reports `$+\infty$` forever, and the hardened
    /// sibling advances past it.
    ///
    /// Mutants killed:
    /// (a) filtering non-finite scores inside `best_score` itself — the first
    ///     assertion fails, which is what makes the ADR 0061 no-fabrication
    ///     ruling executable rather than prose;
    /// (b) folding the raw score into `finite_best_score` outside the
    ///     `is_finite` guard — the accumulator latches at `$+\infty$` and the second
    ///     assertion reads `Some(f32::INFINITY)`, not `Some(100.0)`;
    /// (c) seeding `finite_best_score` from `best_score` rather than from the
    ///     first finite score.
    ///
    /// The `100.0` episode arrives **after** the `$+\infty$` on purpose: it proves the
    /// finite maximum still advances once the raw one has latched.
    #[test]
    fn positive_infinity_latches_best_score_but_not_finite_best_score() {
        let mut stats = AgentStats::<TestRecord>::new(8);

        stats.record(TestRecord::with_duration(5.0, 3));
        stats.record(TestRecord::with_duration(f32::INFINITY, 4));
        stats.record(TestRecord::with_duration(3.0, 5));
        stats.record(TestRecord::with_duration(100.0, 6));

        assert_eq!(stats.best_score(), Some(f32::INFINITY));
        assert_eq!(stats.finite_best_score(), Some(100.0));
        assert_eq!(stats.non_finite_episodes(), 1);
        assert_eq!(stats.total_episodes(), 4);
    }

    /// `$-\infty$` is the non-finite value `f32::max` handles *correctly* on the raw
    /// side — it is discarded as the smaller operand — which is exactly why the
    /// hardened side must still exclude and count it.
    ///
    /// Mutants killed: a counter predicated on `!is_nan()` (reports `0` here);
    /// a counter predicated on `is_infinite() && is_sign_positive()` (`0`); and
    /// a counter incremented only when `best_score` actually changed, or only
    /// when it changed to a non-finite value (`0` — the raw best is `Some(2.0)`
    /// either way).
    #[test]
    fn negative_infinity_is_excluded_from_finite_best_and_counted() {
        let mut stats = AgentStats::<TestRecord>::new(4);

        stats.record(TestRecord::with_duration(f32::NEG_INFINITY, 3));
        stats.record(TestRecord::with_duration(2.0, 4));

        // The raw fold discards `-∞` as the smaller operand, so raw and
        // hardened agree on the value here -- only the count separates them.
        assert_eq!(stats.best_score(), Some(2.0));
        assert_eq!(stats.finite_best_score(), Some(2.0));
        assert_eq!(stats.non_finite_episodes(), 1);
    }

    /// The `NaN` third of the predicate, in the receiver position (`NaN` first),
    /// where the raw accumulator is itself seeded with the bad value.
    ///
    /// Mutant killed: a counter predicated on `is_infinite()`, which reports `0`
    /// here.
    ///
    /// Note what this test does **not** kill:
    /// [`positive_infinity_latches_best_score_but_not_finite_best_score`]'s
    /// mutant (b) — an unguarded fold into `finite_best_score`. A `NaN`
    /// accumulator heals through `.max` on the very next finite episode, so the
    /// unguarded fold would still report `Some(5.0)` here and pass. That
    /// asymmetry between `NaN` and `$+\infty$` is why both tests exist.
    #[test]
    fn nan_is_excluded_from_finite_best_and_counted() {
        let mut stats = AgentStats::<TestRecord>::new(4);

        stats.record(TestRecord::with_duration(f32::NAN, 3));
        stats.record(TestRecord::with_duration(5.0, 4));

        // Pins the existing, unchanged raw behaviour: `f32::max` discards NaN.
        assert_eq!(stats.best_score(), Some(5.0));
        assert_eq!(stats.finite_best_score(), Some(5.0));
        assert_eq!(stats.non_finite_episodes(), 1);
    }

    /// The single test that pins `is_finite()` as *the* predicate, by making the
    /// three candidate predicates report three different counts over one
    /// history: `!is_nan()` → `1`, `is_infinite()` → `2`, `is_finite()` → `3`.
    ///
    /// The two finite episodes that follow keep the hardened maximum
    /// observable (`7.0`) and distinct from the raw one (`$+\infty$`), so no pair of
    /// accessors can be transposed and still satisfy every assertion.
    #[test]
    fn all_three_non_finite_kinds_are_excluded_and_counted() {
        let mut stats = AgentStats::<TestRecord>::new(8);

        stats.record(TestRecord::with_duration(f32::NAN, 3));
        stats.record(TestRecord::with_duration(f32::INFINITY, 4));
        stats.record(TestRecord::with_duration(f32::NEG_INFINITY, 5));
        stats.record(TestRecord::with_duration(4.0, 6));
        stats.record(TestRecord::with_duration(7.0, 7));

        assert_eq!(stats.finite_best_score(), Some(7.0));
        assert_eq!(stats.non_finite_episodes(), 3);
        assert_eq!(stats.best_score(), Some(f32::INFINITY));
        assert_eq!(stats.total_episodes(), 5);
        assert_eq!(stats.recent_len(), 5);
    }

    /// The most important mutant of the whole change: `finite_best_score`
    /// derived from `recent_history` on each call — the shape ADR 0070
    /// §Decision 5 suggests by analogy with `non_finite_recent_len`, and which
    /// is *wrong* for a lifetime statistic because the record that set the
    /// maximum can be evicted. That mutant reports `Some(3.0)` here.
    ///
    /// Also killed: a `min` fold (`Some(1.0)`), and a "keep the latest finite
    /// score" fold (`Some(3.0)`).
    ///
    /// Sibling of [`best_score_survives_eviction_while_average_does_not`]; the
    /// `avg_score` assertion is carried along to show the window really did
    /// roll past the `10.0`.
    #[test]
    fn finite_best_score_survives_eviction() {
        let mut stats = AgentStats::<TestRecord>::new(2);

        stats.record(TestRecord::with_duration(10.0, 3));
        stats.record(TestRecord::with_duration(1.0, 4));
        stats.record(TestRecord::with_duration(3.0, 5));

        assert_eq!(stats.finite_best_score(), Some(10.0));
        assert_eq!(stats.best_score(), Some(10.0));
        assert_eq!(stats.recent_len(), 2);
        assert_eq!(stats.avg_score(), Some(2.0));
    }

    /// The mirror image of [`eviction_heals_the_non_finite_count`]: the same
    /// history, the same eviction, and the opposite expectation, because the
    /// lifetime counter is stored rather than derived and so does not heal.
    ///
    /// Mutants killed: `non_finite_episodes` delegating to
    /// `non_finite_recent_len` (reports `0` after the eviction); any reset of
    /// the counter on eviction or on a clean episode.
    #[test]
    fn non_finite_episodes_is_a_lifetime_count_not_a_window_count() {
        let mut stats = AgentStats::<TestRecord>::new(2);

        stats.record(TestRecord::with_duration(f32::NAN, 3));
        assert_eq!(stats.non_finite_recent_len(), 1);
        assert_eq!(stats.non_finite_episodes(), 1);

        // Two clean episodes push the NaN out of a window of 2.
        stats.record(TestRecord::with_duration(1.0, 4));
        stats.record(TestRecord::with_duration(3.0, 5));

        assert_eq!(stats.non_finite_recent_len(), 0);
        assert_eq!(stats.non_finite_episodes(), 1);
        assert_eq!(stats.finite_best_score(), Some(3.0));
    }

    /// The third lifetime counter clamps at `usize::MAX` like the two other
    /// lifetime counters, already settled on saturating (not wrapping)
    /// arithmetic, and for the same reason — see
    /// [`total_steps_saturates_instead_of_wrapping`] for the house explanation
    /// of why the pre-saturation state is installed by assigning the private
    /// field directly, and why the assertion is on the *value*.
    ///
    /// Mutant killed: `self.non_finite_episodes += 1`. The assertion is never
    /// "it did not panic": the dev profile enables `overflow-checks`, so a
    /// panic-shaped test would pass against the unfixed `+=` under
    /// `cargo test --release`, where the addition silently wraps to `0` instead.
    #[test]
    fn non_finite_episodes_saturates_instead_of_wrapping() {
        let mut stats = AgentStats::<TestRecord>::new(4);
        stats.non_finite_episodes = usize::MAX;

        stats.record(TestRecord::with_duration(f32::NAN, 7));

        assert_eq!(stats.non_finite_episodes(), usize::MAX);
        // The episode counter starts at 0 and is nowhere near saturation, so a
        // transposed accessor pair cannot satisfy both assertions.
        assert_eq!(stats.total_episodes(), 1);
    }

    /// A history in which *every* episode was non-finite has no
    /// maximum-over-finite-scores, and says so.
    ///
    /// Mutants killed: falling back to `best_score` when nothing finite was
    /// recorded (reports `Some(f32::INFINITY)`); seeding the accumulator with
    /// ADR 0069's `f32::NEG_INFINITY` sentinel, deliberately not adopted in this
    /// crate (ADR 0070 §Decision 4) — that reports `Some(f32::NEG_INFINITY)`,
    /// a claim about the agent rather than about the absence of data; and
    /// `Some(NaN)`, which `assert_eq!` against `None` kills regardless of NaN's
    /// self-inequality.
    #[test]
    fn all_non_finite_history_has_no_finite_best() {
        let mut stats = AgentStats::<TestRecord>::new(4);

        stats.record(TestRecord::with_duration(f32::NAN, 3));
        stats.record(TestRecord::with_duration(f32::INFINITY, 4));
        stats.record(TestRecord::with_duration(f32::NEG_INFINITY, 5));

        assert_eq!(stats.finite_best_score(), None);
        assert_eq!(stats.non_finite_episodes(), 3);
        // The raw maximum still reports the truth about what was observed.
        assert_eq!(stats.best_score(), Some(f32::INFINITY));
        assert_eq!(stats.total_episodes(), 3);
    }

    /// A freshly constructed `AgentStats` reports absence on the new pair too,
    /// matching [`empty_stats_report_no_average_or_best`] rather than
    /// fabricating a zero.
    ///
    /// Mutant killed: `finite_best_score` initialised to `Some(0.0)`, which
    /// would read as "the agent scored zero" on a run that has not started.
    #[test]
    fn empty_stats_report_no_finite_best_and_no_non_finite_count() {
        let stats = AgentStats::<TestRecord>::new(4);

        assert_eq!(stats.finite_best_score(), None);
        assert_eq!(stats.non_finite_episodes(), 0);
        assert_eq!(stats.best_score(), None);
        assert_eq!(stats.total_episodes(), 0);
    }

    /// The additivity claim on the healthy path: with no non-finite episode
    /// anywhere, the hardened maximum is the raw maximum, and the exclusion
    /// count is zero.
    ///
    /// Mutant killed: any implementation that perturbs the reported maximum on
    /// an all-finite history — an off-by-one fold that drops the first or last
    /// episode, or a fold that keeps the latest rather than the greatest score
    /// (the scores ascend, so the two are distinguished only by the `9.0`
    /// arriving last; the count assertion and the exact `Some(9.0)` pin it).
    #[test]
    fn finite_best_score_matches_best_score_on_an_all_finite_history() {
        let mut stats = AgentStats::<TestRecord>::new(4);

        stats.record(TestRecord::with_duration(1.0, 3));
        stats.record(TestRecord::with_duration(2.0, 4));
        stats.record(TestRecord::with_duration(4.0, 5));
        stats.record(TestRecord::with_duration(9.0, 6));

        assert_eq!(stats.finite_best_score(), stats.best_score());
        assert_eq!(stats.finite_best_score(), Some(9.0));
        assert_eq!(stats.best_score(), Some(9.0));
        assert_eq!(stats.non_finite_episodes(), 0);
    }
}
