//! Trajectory storage and history representation for experience replay.
//!
//! This module provides the building blocks for storing and reasoning over
//! sequences of agent–environment interactions:
//! - [`ExperienceTuple`] — a single `(obs, action, reward, next_obs, done)` transition
//! - [`History`] — a fixed-capacity FIFO buffer of transitions
//! - [`HistoryRepresentation`] — trait for constructing state summaries from history
//! - [`SufficientStatistic`] — history summary that satisfies the Markov property

use rlevo_core::base::{Action, Observation, Reward};
use rlevo_core::state::MarkovState;
use std::collections::VecDeque;
use std::ops::Index;

/// A single `(obs, action, reward, next_obs, done)` transition tuple.
///
/// All five fields are required to compute the Bellman target for off-policy
/// algorithms such as DQN and SAC.
#[derive(Clone)]
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
    /// `true` when the episode ended after this transition.
    pub is_done: bool,
}

/// Fixed-capacity FIFO buffer of experience tuples.
///
/// When the buffer is full, the oldest entry is evicted before each new push.
/// This is a thin wrapper around [`VecDeque`] that enforces the capacity
/// contract at the API level.
#[derive(Clone)]
pub struct History<const D: usize, const AD: usize, O: Observation<D>, A: Action<AD>, R: Reward> {
    trace: VecDeque<ExperienceTuple<D, AD, O, A, R>>,
    capacity: usize,
}

impl<const D: usize, const AD: usize, O: Observation<D>, A: Action<AD>, R: Reward> Index<usize>
    for History<D, AD, O, A, R>
{
    type Output = ExperienceTuple<D, AD, O, A, R>;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.trace[idx]
    }
}

impl<const D: usize, const AD: usize, O: Observation<D>, A: Action<AD>, R: Reward>
    History<D, AD, O, A, R>
{
    /// Creates an empty `History` with the given maximum `capacity`.
    pub fn new(capacity: usize) -> Self {
        Self {
            trace: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Returns the configured maximum capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of stored transitions.
    pub fn len(&self) -> usize {
        self.trace.len()
    }

    /// Returns `true` when no transitions have been stored yet.
    pub fn is_empty(&self) -> bool {
        self.trace.is_empty()
    }

    /// Removes all stored transitions without changing capacity.
    pub fn clear(&mut self) {
        self.trace.clear();
    }

    /// Returns a clone of the full transition sequence in insertion order.
    pub fn trace(&self) -> VecDeque<ExperienceTuple<D, AD, O, A, R>> {
        self.trace.clone()
    }

    /// Appends an experience, evicting the oldest entry when at capacity.
    pub fn add(
        &mut self,
        observation: O,
        action: A,
        reward: R,
        next_observation: O,
        is_done: bool,
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
            is_done,
        });
    }

    /// Returns `true` when `len() >= capacity`.
    pub fn is_full(&self) -> bool {
        self.len() >= self.capacity
    }

    /// Returns an iterator over stored transitions in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = &ExperienceTuple<D, AD, O, A, R>> {
        self.trace.iter()
    }

    /// Returns a reference to the transition at `idx`, or `None` if out of bounds.
    pub fn get(&self, idx: usize) -> Option<&ExperienceTuple<D, AD, O, A, R>> {
        self.trace.get(idx)
    }
}

/// A summary representation constructed from an interaction history.
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

    /// Test that History can store rewards of different values
    #[test]
    fn test_history_with_rewards() {
        let mut history = History::<1, 1, TestObs, TestAct, TestReward>::new(10);

        history.add(TestObs, TestAct, TestReward(10.0), TestObs, false);
        history.add(TestObs, TestAct, TestReward(20.0), TestObs, false);
        history.add(TestObs, TestAct, TestReward(5.0), TestObs, true);

        assert_eq!(history.len(), 3);
    }

    /// Test ExperienceTuple with rewards
    #[test]
    fn test_experience_tuple_creation() {
        let experience = ExperienceTuple {
            observation: TestObs,
            action: TestAct,
            reward: TestReward(50.0),
            next_observation: TestObs,
            is_done: false,
        };

        assert_eq!(experience.reward, TestReward(50.0));
        assert!(!experience.is_done)
    }

    /// Test that ExperienceTuple preserves reward through cloning
    #[test]
    fn test_experience_tuple_clone() {
        let original = ExperienceTuple {
            observation: TestObs,
            action: TestAct,
            reward: TestReward(75.0),
            next_observation: TestObs,
            is_done: false,
        };

        let cloned = original.clone();

        assert_eq!(original.reward, cloned.reward);
    }

    /// Test multiple experience tuples with reward accumulation
    #[test]
    fn test_multiple_experience_tuples() {
        let experiences = [ExperienceTuple {
                observation: TestObs,
                action: TestAct,
                reward: TestReward(5.0),
                next_observation: TestObs,
                is_done: false,
            },
            ExperienceTuple {
                observation: TestObs,
                action: TestAct,
                reward: TestReward(10.0),
                next_observation: TestObs,
                is_done: false,
            },
            ExperienceTuple {
                observation: TestObs,
                action: TestAct,
                reward: TestReward(15.0),
                next_observation: TestObs,
                is_done: false,
            }];

        let total_reward = experiences
            .iter()
            .fold(TestReward::zero(), |acc, exp| acc + exp.reward.clone());

        assert_eq!(total_reward, TestReward(30.0));
    }

    /// Test history operations with reward accumulation
    #[test]
    fn test_history_reward_accumulation() {
        let mut history = History::<1, 1, TestObs, TestAct, TestReward>::new(5);

        let rewards = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        for &reward_val in &rewards {
            history.add(TestObs, TestAct, TestReward(reward_val), TestObs, false);
        }

        assert_eq!(history.len(), 5);
    }
}
