//! Trajectory storage and history representation for experience replay.
//!
//! This module provides the building blocks for storing and reasoning over
//! sequences of agent–environment interactions:
//! - [`ExperienceTuple`] — a single `(obs, action, reward, next_obs, terminated)` transition
//! - [`History`] — a fixed-capacity FIFO buffer of transitions
//! - [`HistoryRepresentation`] — trait for constructing state summaries from history
//! - [`SufficientStatistic`] — history summary that satisfies the Markov property
//!
//! # Not the replay integration path
//!
//! [`crate::replay::Transition`] is the **live** stored-transition model: it is
//! what every off-policy agent writes into a [`ReplayStrategy`], and it is the
//! type to reach for when wiring a new agent to replay. The four items in this
//! module have **zero consumers** in the workspace today. They are ADR 0003
//! roadmap markers, kept under that ADR's conservative dead-code policy for the
//! POMDP/history-conditioned work they were sketched for. Their survival is an
//! open design decision between two ways of giving a
//! discrete-action RL agent memory on a POMDP (first target: the `SantaFeAnt`
//! environment's one-bit percept): a lighter observation history-framing
//! adapter feeding an existing feedforward agent, versus a heavier recurrent
//! agent with explicit hidden-state threading and sequence replay. Whichever
//! path is chosen picks which of these trait seams (if any) gets its first
//! real implementor; until the decision lands, do not build against them.
//!
//! ADR 0050 §9 records this pointer as its own deliverable: `experience.rs` was
//! deliberately left untouched by the replay-strategy seam, so the module docs
//! carry the signpost instead of the code, and a reader cannot mistake
//! [`ExperienceTuple`] for the integration path.
//!
//! [`ReplayStrategy`]: crate::replay::ReplayStrategy

use crate::MAX_BUFFER_CAPACITY;
use rlevo_core::base::{Action, Observation, Reward};
use rlevo_core::state::MarkovState;
use std::collections::VecDeque;
use std::ops::Index;

/// A single `(obs, action, reward, next_obs, terminated)` transition tuple.
///
/// All five fields are required to compute the Bellman target for off-policy
/// algorithms such as DQN and SAC.
///
/// # Thread safety
///
/// This type holds its observation, action, and reward by value and contains no
/// interior mutability, raw pointers, or reference counting, so the [`Send`] and
/// [`Sync`] auto-traits propagate from its type parameters: an
/// `ExperienceTuple<D, AD, O, A, R>` is [`Send`] exactly when `O`, `A`, and `R`
/// are, and [`Sync`] exactly when `O`, `A`, and `R` are. No explicit bounds are
/// declared on the struct — per the Rust API Guidelines (C-STRUCT-BOUNDS) that
/// would restrict the type without adding guarantees.
#[derive(Clone, Debug)]
pub struct ExperienceTuple<
    const D: usize,
    const AD: usize,
    O: Observation<D>,
    A: Action<AD>,
    R: Reward,
> {
    /// Observation at time *t*.
    pub observation: O,
    /// Action taken at time *t*.
    pub action: A,
    /// Reward received after taking `action`.
    pub reward: R,
    /// Observation at time *t+1*.
    pub next_observation: O,
    /// `true` when this transition ended in an **environmental termination**
    /// — the MDP reached an absorbing state.
    ///
    /// This is the Bellman bootstrap mask, so it must carry
    /// [`Snapshot::is_terminated`], **not** [`Snapshot::is_done`]. A truncation
    /// (time-limit cutoff, [`EpisodeStatus::Truncated`]) ends the *episode* but
    /// not the *MDP*: the continuation state still has a well-defined value, so
    /// it must be recorded as `false` here or the bootstrap is wrongly zeroed
    /// and every Q-value is biased downward. See Pardo et al., "Time Limits in
    /// Reinforcement Learning", ICML 2018, Eq. 6.
    ///
    /// [`Snapshot::is_terminated`]: rlevo_core::environment::Snapshot::is_terminated
    /// [`Snapshot::is_done`]: rlevo_core::environment::Snapshot::is_done
    /// [`EpisodeStatus::Truncated`]: rlevo_core::environment::EpisodeStatus::Truncated
    pub terminated: bool,
}

/// Fixed-capacity FIFO buffer of experience tuples.
///
/// When the buffer is full, the oldest entry is evicted before each new push.
/// This is a thin wrapper around [`VecDeque`] that enforces the capacity
/// contract at the API level.
///
/// # Thread safety
///
/// The buffer is a plain [`VecDeque`] of owned [`ExperienceTuple`]s with no
/// interior mutability, raw pointers, or reference counting, so the [`Send`] and
/// [`Sync`] auto-traits propagate from its type parameters: a
/// `History<D, AD, O, A, R>` is [`Send`] exactly when `O`, `A`, and `R` are, and
/// [`Sync`] exactly when `O`, `A`, and `R` are. This is what lets a history be
/// moved into a worker thread or shared behind a `Mutex` across parallel
/// rollouts. No explicit bounds are declared on the struct — per the Rust API
/// Guidelines (C-STRUCT-BOUNDS) that would restrict the type without adding
/// guarantees.
#[derive(Clone, Debug)]
pub struct History<const D: usize, const AD: usize, O: Observation<D>, A: Action<AD>, R: Reward> {
    trace: VecDeque<ExperienceTuple<D, AD, O, A, R>>,
    capacity: usize,
}

impl<const D: usize, const AD: usize, O: Observation<D>, A: Action<AD>, R: Reward> Index<usize>
    for History<D, AD, O, A, R>
{
    type Output = ExperienceTuple<D, AD, O, A, R>;

    /// Returns the transition stored at `idx`, counted from the oldest
    /// retained entry.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= self.len()`. Indexing is the infallible spelling and
    /// treats an out-of-range index as a programming error; use
    /// [`History::get`] for the fallible alternative, which returns `None`
    /// instead. Note that `len()` shrinks only via [`History::clear`] —
    /// eviction keeps `len()` pinned at `capacity()` — so a stale index taken
    /// before a push remains in range but no longer names the same transition.
    fn index(&self, idx: usize) -> &Self::Output {
        &self.trace[idx]
    }
}

impl<const D: usize, const AD: usize, O: Observation<D>, A: Action<AD>, R: Reward>
    History<D, AD, O, A, R>
{
    /// Creates an empty `History` with the given maximum `capacity`.
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of transitions retained before the oldest
    ///   is evicted. Must be in `1..=`[`MAX_BUFFER_CAPACITY`].
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is 0. A zero-capacity buffer cannot hold any
    /// transitions: [`Self::add`] would evict before every push, pinning
    /// `len()` at 1 while [`Self::capacity`] reports 0 and [`Self::is_full`]
    /// returns `true` from the first insert — a buffer that permanently
    /// violates its own `len() <= capacity` invariant while silently accepting
    /// and discarding every experience the agent collects.
    ///
    /// Panics if `capacity` exceeds [`MAX_BUFFER_CAPACITY`]. The argument
    /// reaches [`VecDeque::with_capacity`] unfiltered, so without this bound a
    /// caller-supplied capacity — a unit slip, a `usize::MAX` sentinel, a
    /// deserialized garbage field — turns straight into an allocation request
    /// for `capacity * size_of::<ExperienceTuple<..>>()` bytes. That aborts the
    /// process on capacity overflow rather than unwinding, which is not a
    /// failure a caller can catch or diagnose. Rejecting the argument up front
    /// converts an abort into a panic that names the offending value.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        assert!(
            capacity > 0,
            "capacity must be greater than 0; a zero-capacity buffer cannot \
             hold transitions and would pin len() at 1 while reporting a \
             capacity of 0, breaking the len() <= capacity invariant"
        );
        assert!(
            capacity <= MAX_BUFFER_CAPACITY,
            "capacity must be at most {MAX_BUFFER_CAPACITY}, got {capacity}; \
             this value is passed straight to VecDeque::with_capacity, where an \
             out-of-range request aborts the process instead of unwinding"
        );
        Self {
            trace: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Returns the configured maximum capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of stored transitions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.trace.len()
    }

    /// Returns `true` when no transitions have been stored yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.trace.is_empty()
    }

    /// Removes all stored transitions without changing capacity.
    pub fn clear(&mut self) {
        self.trace.clear();
    }

    /// Returns a clone of the full transition sequence in insertion order.
    ///
    /// This is an **O(n) deep clone**: every stored [`ExperienceTuple`] is
    /// cloned, which clones two observations, an action, and a reward apiece.
    /// Callers that only need to *read* the stored transitions should use
    /// [`Self::iter`] (or [`Self::get`] / indexing for a single slot) and pay
    /// nothing.
    ///
    /// The name intentionally mirrors the private `trace` field it copies,
    /// rather than following the borrowing-accessor convention — the returned
    /// value is an owned snapshot, not a view.
    #[must_use]
    pub fn trace(&self) -> VecDeque<ExperienceTuple<D, AD, O, A, R>> {
        self.trace.clone()
    }

    /// Appends an experience, evicting the oldest entry when at capacity.
    ///
    /// `terminated` is the Bellman bootstrap mask: pass
    /// [`Snapshot::is_terminated`], **not** [`Snapshot::is_done`]. See
    /// [`ExperienceTuple::terminated`].
    ///
    /// [`Snapshot::is_terminated`]: rlevo_core::environment::Snapshot::is_terminated
    /// [`Snapshot::is_done`]: rlevo_core::environment::Snapshot::is_done
    pub fn add(
        &mut self,
        observation: O,
        action: A,
        reward: R,
        next_observation: O,
        terminated: bool,
    ) {
        // `VecDeque::with_capacity(n)` may allocate more than `n` slots, so we
        // compare against the caller-requested capacity rather than the
        // underlying allocation.
        if self.trace.len() >= self.capacity {
            self.trace.pop_front();
        }
        self.trace.push_back(ExperienceTuple {
            observation,
            action,
            reward,
            next_observation,
            terminated,
        });
    }

    /// Returns `true` when `len() >= capacity`.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.len() >= self.capacity
    }

    /// Returns an iterator over stored transitions in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = &ExperienceTuple<D, AD, O, A, R>> {
        self.trace.iter()
    }

    /// Returns a reference to the transition at `idx`, or `None` if out of bounds.
    #[must_use]
    pub fn get(&self, idx: usize) -> Option<&ExperienceTuple<D, AD, O, A, R>> {
        self.trace.get(idx)
    }
}

/// A summary representation constructed from an interaction history.
///
/// Implementors encode the information in a [`History`] buffer into a
/// compact form that an agent can act on. The trait provides two construction
/// paths:
///
/// - [`from_history`] — builds the representation from scratch by scanning
///   the entire buffer; use this after a reset or when the full trajectory
///   is available.
/// - [`update_with`] — incrementally folds a single new transition into an
///   existing representation without re-scanning the buffer; use this at
///   each step for O(1) updates.
///
/// Implementations must be [`Clone`] so that representations can be stored
/// alongside experience tuples in a replay buffer.
///
/// [`from_history`]: HistoryRepresentation::from_history
/// [`update_with`]: HistoryRepresentation::update_with
pub trait HistoryRepresentation<
    const D: usize,
    const AD: usize,
    O: Observation<D>,
    A: Action<AD>,
    R: Reward,
>: Clone
{
    /// Constructs this representation from the complete interaction history.
    fn from_history(history: &History<D, AD, O, A, R>) -> Self;

    /// Incrementally incorporates one new `(obs, action, reward)` triple.
    fn update_with(&mut self, obs: &O, action: &A, reward: &R);
}

/// A history summary that captures all decision-relevant information.
///
/// A sufficient statistic satisfies the Markov property: conditioning on it
/// renders the future independent of the past, so agents need not retain the
/// full trajectory.
pub trait SufficientStatistic<
    const D: usize,
    const AD: usize,
    O: Observation<D>,
    A: Action<AD>,
    R: Reward,
>: HistoryRepresentation<D, AD, O, A, R> + MarkovState
{
    /// Returns `true` if `self` is a sufficient statistic for `history`.
    fn is_sufficient(&self, history: &History<D, AD, O, A, R>) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    /// Simple scalar reward implementation for testing
    #[derive(Clone, Debug, PartialEq)]
    struct TestReward(f32);

    impl Reward for TestReward {
        fn zero() -> Self {
            TestReward(0.0)
        }
    }

    impl std::ops::Add for TestReward {
        type Output = Self;

        fn add(self, other: Self) -> Self {
            TestReward(self.0 + other.0)
        }
    }

    impl From<TestReward> for f32 {
        fn from(reward: TestReward) -> f32 {
            reward.0
        }
    }

    // ===== Mock Types for Integration Tests =====

    /// Mock observation type for integration tests
    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct TestObs;

    impl Observation<1> for TestObs {
        fn shape() -> [usize; 1] {
            [1]
        }
    }

    /// Mock action type for integration tests
    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct TestAct;

    impl Action<1> for TestAct {
        fn shape() -> [usize; 1] {
            [1]
        }
        fn is_valid(&self) -> bool {
            true
        }
    }

    // ===== History Integration Tests =====

    #[test]
    #[should_panic(expected = "capacity must be greater than 0")]
    fn new_rejects_zero_capacity() {
        let _ = History::<1, 1, TestObs, TestAct, TestReward>::new(0);
    }

    /// The upper end of the same guard: `capacity` reaches
    /// `VecDeque::with_capacity` directly, where an out-of-range request aborts
    /// the process instead of unwinding. `should_panic` only observes an
    /// unwind, so this test passing *is* the evidence that the value was
    /// rejected before it reached the allocator.
    #[test]
    #[should_panic(expected = "capacity must be at most")]
    fn new_rejects_capacity_above_ceiling() {
        let _ = History::<1, 1, TestObs, TestAct, TestReward>::new(MAX_BUFFER_CAPACITY + 1);
    }

    /// Pins the [`Send`]/[`Sync`] auto-trait propagation documented on
    /// [`History`] and [`ExperienceTuple`].
    ///
    /// This is deliberately *not* tautological, despite compiling to nothing:
    /// auto-traits are derived structurally, so adding a single non-`Send` or
    /// non-`Sync` field (an `Rc` for cheap clones, a `Cell` for a lazily
    /// computed statistic) would silently strip `Send`/`Sync` from both types
    /// and break every downstream caller that moves a history into a worker
    /// thread — with nothing else in the suite to catch it. Do not delete.
    #[test]
    fn history_and_experience_tuple_are_send_and_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<History<1, 1, TestObs, TestAct, TestReward>>();
        assert_sync::<History<1, 1, TestObs, TestAct, TestReward>>();
        assert_send::<ExperienceTuple<1, 1, TestObs, TestAct, TestReward>>();
        assert_sync::<ExperienceTuple<1, 1, TestObs, TestAct, TestReward>>();
    }

    /// Test that History can store rewards of different values
    #[test]
    fn test_history_with_rewards() {
        let mut history = History::<1, 1, TestObs, TestAct, TestReward>::new(10);

        history.add(TestObs, TestAct, TestReward(10.0), TestObs, false);
        history.add(TestObs, TestAct, TestReward(20.0), TestObs, false);
        history.add(TestObs, TestAct, TestReward(5.0), TestObs, true);

        assert_eq!(history.len(), 3);
    }

    /// Test `ExperienceTuple` with rewards
    #[test]
    fn test_experience_tuple_creation() {
        let experience = ExperienceTuple {
            observation: TestObs,
            action: TestAct,
            reward: TestReward(50.0),
            next_observation: TestObs,
            terminated: false,
        };

        assert_eq!(experience.reward, TestReward(50.0));
        assert!(!experience.terminated);
    }

    /// Test that `ExperienceTuple` preserves reward through cloning
    #[test]
    fn test_experience_tuple_clone() {
        let original = ExperienceTuple {
            observation: TestObs,
            action: TestAct,
            reward: TestReward(75.0),
            next_observation: TestObs,
            terminated: false,
        };

        let cloned = original.clone();

        assert_eq!(original.reward, cloned.reward);
    }

    /// Test multiple experience tuples with reward accumulation
    #[test]
    fn test_multiple_experience_tuples() {
        let experiences = [
            ExperienceTuple {
                observation: TestObs,
                action: TestAct,
                reward: TestReward(5.0),
                next_observation: TestObs,
                terminated: false,
            },
            ExperienceTuple {
                observation: TestObs,
                action: TestAct,
                reward: TestReward(10.0),
                next_observation: TestObs,
                terminated: false,
            },
            ExperienceTuple {
                observation: TestObs,
                action: TestAct,
                reward: TestReward(15.0),
                next_observation: TestObs,
                terminated: false,
            },
        ];

        let total_reward = experiences
            .iter()
            .fold(TestReward::zero(), |acc, exp| acc + exp.reward.clone());

        assert_eq!(total_reward, TestReward(30.0));
    }

    /// Asserts that the reward at `history[idx]` is `expected`.
    ///
    /// Per-slot, never aggregate: a sum or a length cannot distinguish a
    /// transposition or a dropped entry from a correct buffer.
    fn assert_reward_at(
        history: &History<1, 1, TestObs, TestAct, TestReward>,
        idx: usize,
        expected: f32,
    ) {
        let got = history[idx].reward.0;
        assert!(
            (got - expected).abs() < 1e-6,
            "slot {idx} should hold reward {expected}, got {got}"
        );
    }

    /// Every stored slot holds the reward it was pushed with, at its own
    /// index, in insertion order.
    ///
    /// The reward values are distinct primes so that swapping any two slots or
    /// dropping any one breaks a per-slot assertion — a length check or a sum
    /// would see neither.
    #[test]
    fn test_history_reward_accumulation() {
        let mut history = History::<1, 1, TestObs, TestAct, TestReward>::new(5);

        let rewards = [11.0_f32, 23.0, 37.0, 41.0, 59.0];
        for &reward_val in &rewards {
            history.add(TestObs, TestAct, TestReward(reward_val), TestObs, false);
        }

        assert_eq!(
            history.len(),
            rewards.len(),
            "a below-capacity fill must retain every pushed transition"
        );
        for (idx, &expected) in rewards.iter().enumerate() {
            assert_reward_at(&history, idx, expected);
        }
    }

    /// `terminated` round-trips through `add` unchanged, per slot.
    ///
    /// This flag is the Bellman bootstrap mask (see
    /// [`ExperienceTuple::terminated`] and Pardo et al., ICML 2018): a
    /// `History` that dropped, hardcoded, or inverted it would still store
    /// every reward in the right order and pass every other test in this
    /// module, while biasing every Q-value downward in any agent built on it.
    ///
    /// The pattern `[false, true, true, false, false]` is chosen so that it is
    /// **not** all-true (killing a `terminated: false` hardcode), **not**
    /// all-false (killing a `true` hardcode), and **asymmetric under reversal**
    /// (killing an order-reversing push mutant — the reverse is
    /// `[false, false, true, true, false]`). Pairing each flag with a distinct
    /// prime reward means a transposition of two slots breaks both assertions
    /// when the flags differ, and the reward assertion when they agree.
    #[test]
    fn terminated_round_trips_per_slot() {
        let mut history = History::<1, 1, TestObs, TestAct, TestReward>::new(5);

        let pushed = [
            (13.0_f32, false),
            (29.0, true),
            (43.0, true),
            (61.0, false),
            (71.0, false),
        ];
        for &(reward_val, terminated) in &pushed {
            history.add(
                TestObs,
                TestAct,
                TestReward(reward_val),
                TestObs,
                terminated,
            );
        }

        assert_eq!(
            history.len(),
            pushed.len(),
            "a below-capacity fill must retain every pushed transition"
        );
        for (idx, &(reward_val, terminated)) in pushed.iter().enumerate() {
            assert_reward_at(&history, idx, reward_val);
            assert_eq!(
                history[idx].terminated, terminated,
                "slot {idx} (reward {reward_val}) should carry terminated={terminated}"
            );
        }
    }

    /// At capacity, `add` evicts the **oldest** entry, and the survivors keep
    /// their relative order.
    ///
    /// Pushing five distinct rewards into a capacity-3 buffer must leave the
    /// last three, in order. Asserting slot by slot is what makes this test
    /// sensitive to the two ways eviction goes wrong: evicting from the wrong
    /// end (`pop_back`), and evicting one push too late (`>` instead of `>=`).
    #[test]
    fn add_evicts_oldest_first_at_capacity() {
        let mut history = History::<1, 1, TestObs, TestAct, TestReward>::new(3);

        for &reward_val in &[11.0_f32, 23.0, 37.0, 41.0, 59.0] {
            history.add(TestObs, TestAct, TestReward(reward_val), TestObs, false);
        }

        assert_eq!(
            history.len(),
            3,
            "len() must never exceed the configured capacity of 3"
        );
        assert_reward_at(&history, 0, 37.0);
        assert_reward_at(&history, 1, 41.0);
        assert_reward_at(&history, 2, 59.0);
    }

    /// `is_full()` is `false` below capacity, flips to `true` on reaching it,
    /// and stays `true` across an evicting push.
    #[test]
    fn is_full_transitions_false_to_true() {
        let mut history = History::<1, 1, TestObs, TestAct, TestReward>::new(3);

        history.add(TestObs, TestAct, TestReward(11.0), TestObs, false);
        history.add(TestObs, TestAct, TestReward(23.0), TestObs, false);
        assert!(!history.is_full(), "2 of 3 slots used must not report full");

        history.add(TestObs, TestAct, TestReward(37.0), TestObs, false);
        assert!(history.is_full(), "3 of 3 slots used must report full");

        history.add(TestObs, TestAct, TestReward(41.0), TestObs, false);
        assert!(
            history.is_full(),
            "an evicting push must leave the buffer full, not over- or under-filled"
        );
        assert_eq!(
            history.len(),
            3,
            "an evicting push must hold len() at capacity"
        );
        assert_reward_at(&history, 2, 41.0);
    }

    /// `get()` resolves the last valid index and returns `None` past the end.
    #[test]
    fn get_returns_none_out_of_bounds() {
        let mut history = History::<1, 1, TestObs, TestAct, TestReward>::new(5);
        history.add(TestObs, TestAct, TestReward(11.0), TestObs, false);
        history.add(TestObs, TestAct, TestReward(23.0), TestObs, false);
        history.add(TestObs, TestAct, TestReward(37.0), TestObs, false);

        let last = history
            .get(history.len() - 1)
            .expect("the last valid index must resolve");
        assert!(
            (last.reward.0 - 37.0).abs() < 1e-6,
            "get(len() - 1) must return the most recently pushed transition"
        );

        assert!(
            history.get(history.len()).is_none(),
            "get(len()) is one past the end and must return None"
        );
        assert!(
            history.get(history.len() + 1).is_none(),
            "get(len() + 1) is out of bounds and must return None"
        );
    }

    /// Pins the documented `# Panics` contract on the [`Index`] impl: indexing
    /// past the end is a programming error, and [`History::get`] is the
    /// fallible alternative.
    ///
    /// Deliberately a bare `should_panic` with no `expected` string. The two
    /// `should_panic` tests above pin messages from `assert!` calls this crate
    /// authors in [`History::new`], which are stable by our own contract; the
    /// panic here comes from [`VecDeque`]'s indexing, whose wording is a std
    /// implementation detail that has been reworded before. There is nothing
    /// to disambiguate — the only operation in this body that can panic is the
    /// index — so pinning std's phrasing would only buy failures unrelated to
    /// `History`'s correctness.
    #[test]
    // `clippy::should_panic_without_expect` asks for an `expected` string to
    // prove the test panicked for the intended reason. That guarantee is
    // already structural here — the body has exactly one fallible operation —
    // and the only string available to pin belongs to `VecDeque`, not to this
    // crate, so satisfying the lint would trade a real property for a
    // dependency on std's phrasing. See the doc comment above.
    #[allow(clippy::should_panic_without_expect)]
    #[should_panic]
    fn index_panics_out_of_bounds() {
        let mut history = History::<1, 1, TestObs, TestAct, TestReward>::new(5);
        history.add(TestObs, TestAct, TestReward(11.0), TestObs, false);

        let _ = &history[history.len()];
    }
}
