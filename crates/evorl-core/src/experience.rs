use crate::base::{Action, Observation, Reward};
use crate::state::MarkovState;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::ops::Index;

/// A single transition/experience in the replay memory.
/// This is fundamentally different from supervised learning, where a batch is just input-label pairs. In RL, you need all five components to compute the Bellman update for Q-learning.
#[derive(Clone)]
pub struct ExperienceTuple<
    const D: usize,
    const AD: usize,
    O: Observation<D>,
    A: Action<AD>,
    R: Reward,
> {
    pub observation: O,
    pub action: A,
    pub reward: R,
    pub next_observation: O,
    pub is_done: bool,
}

/// A history of interactions: sequence of observations, actions, rewards
/// Note that History is an intentional thin-wrapper around VecDeque to enforce capacity.
#[derive(Clone)]
pub struct History<const D: usize, const AD: usize, O: Observation<D>, A: Action<AD>, R: Reward> {
    trace: VecDeque<ExperienceTuple<D, AD, O, A, R>>,
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
    pub fn new(capacity: usize) -> Self {
        Self {
            trace: VecDeque::with_capacity(capacity),
        }
    }

    pub fn len(&self) -> usize {
        self.trace.len()
    }

    pub fn is_empty(&self) -> bool {
        self.trace.is_empty()
    }

    pub fn clear(&mut self) {
        self.trace.clear();
    }

    /// Full history of experiences
    pub fn trace(&self) -> VecDeque<ExperienceTuple<D, AD, O, A, R>> {
        self.trace.clone()
    }

    /// Add an experience, maintaining fixed capacity (FIFO if at capacity)
    pub fn add(
        &mut self,
        observation: O,
        action: A,
        reward: R,
        next_observation: O,
        is_done: bool,
    ) {
        // is this needed since VecDeque is defined as "A double-ended queue implemented with a growable ring buffer." Since it is constructed with capacity, adding to the history should automatically `pop_front`, right?
        if self.trace.len() >= self.trace.capacity() {
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

    pub fn is_full(&self) -> bool {
        self.len() >= self.trace.capacity()
    }

    pub fn iter(&self) -> impl Iterator<Item = &ExperienceTuple<D, AD, O, A, R>> {
        self.trace.iter()
    }

    pub fn get(&self, idx: usize) -> Option<&ExperienceTuple<D, AD, O, A, R>> {
        self.trace.get(idx)
    }
}

/// A representation that can be constructed from history
pub trait HistoryRepresentation<
    const D: usize,
    const AD: usize,
    O: Observation<D>,
    A: Action<AD>,
    R: Reward,
>: Clone
{
    /// Construct representation from complete history
    fn from_history(history: &History<D, AD, O, A, R>) -> Self;

    /// Incrementally update representation with new experience
    fn update_with(&mut self, obs: &O, action: &A, reward: &R);
}

/// Sufficient statistic: contains all decision-relevant information
pub trait SufficientStatistic<
    const D: usize,
    const AD: usize,
    O: Observation<D>,
    A: Action<AD>,
    R: Reward,
>: HistoryRepresentation<D, AD, O, A, R> + MarkovState
{
    /// Verify this is a sufficient statistic for the given history
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
        assert!(experience.is_done)
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
        let experiences = vec![
            ExperienceTuple {
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
            },
        ];

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
