use crate::action::Action;
use crate::state::{MarkovState, Observation, State};
use std::collections::VecDeque;
use std::fmt::Debug;

/// Represents a reward signal
pub trait Reward: Clone + std::ops::Add<Output = Self> + Into<f32> + Debug {
    fn zero() -> Self;
}

/// Update function: how something evolves over time
/// Generic over input and output types
pub trait UpdateFunction<Input, Output> {
    fn update(&self, current: &Output, input: &Input) -> Output;
}

/// Environment transition dynamics: s_{t+1} = f(s_t, a_t)
pub trait TransitionDynamics<const SD: usize, const AD: usize, S: State<SD>, A: Action<AD>> {
    fn transition(&self, state: &S, action: &A) -> S;
}

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
}

/// A history of interactions: sequence of observations, actions, rewards
#[derive(Clone)]
pub struct History<const D: usize, const AD: usize, O: Observation<D>, A: Action<AD>, R: Reward> {
    trace: VecDeque<ExperienceTuple<D, AD, O, A, R>>,
}

impl<const D: usize, const AD: usize, O: Observation<D>, A: Action<AD>, R: Reward>
    History<D, AD, O, A, R>
{
    pub fn new(capacity: usize) -> Self {
        Self {
            trace: VecDeque::with_capacity(capacity),
        }
    }

    pub fn add(&mut self, obs: O, action: A, reward: R) {
        self.trace.push_back(ExperienceTuple {
            observation: obs,
            action,
            reward,
        });
    }

    pub fn len(&self) -> usize {
        self.trace.len()
    }
}

/// Belief state: probability distribution over possible states
pub trait BeliefState<const SD: usize, S: State<SD>>: Clone {
    /// Update belief given a new observation and action
    fn update(&self, action: &S::Observation, observation: &S::Observation) -> Self;

    /// Sample a state from the belief distribution
    fn sample(&self) -> S;

    /// Get probability/weight of a particular state
    fn probability(&self, state: &S) -> f64;
}

/// Hidden/internal state maintained by an agent (e.g., RNN hidden state)
pub trait HiddenState<const D: usize>: Clone {
    type Observation: Observation<D>;

    /// Update hidden state given new observation
    fn update(&mut self, observation: &Self::Observation);

    /// Initialize to default state
    fn reset(&mut self);
}

/// Latent state: learned compact representation
pub trait LatentState<const D: usize, const AD: usize>: Clone {
    type Observation: Observation<D>;

    /// Encode observation into latent representation
    fn encode(observation: &Self::Observation) -> Self;

    /// Predict next latent state given action
    fn predict_next<A: Action<AD>>(&self, action: &A) -> Self;

    /// Decode back to observation space (optional, for world models)
    fn decode(&self) -> Self::Observation;
}

/// State aggregation: mapping from detailed states to abstract states
pub trait StateAggregation<const SD: usize, S: State<SD>> {
    type AbstractState: Clone + Eq;

    /// Map a concrete state to its abstract representative
    fn aggregate(&self, state: &S) -> Self::AbstractState;

    /// Check if two states belong to the same aggregate
    fn same_aggregate(&self, state1: &S, state2: &S) -> bool {
        self.aggregate(state1) == self.aggregate(state2)
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

    // ===== Basic Reward Trait Tests =====

    /// Test that zero() creates a neutral element for addition
    #[test]
    fn test_reward_zero_is_additive_identity() {
        let zero = TestReward::zero();
        let reward = TestReward(42.5);

        // zero + reward should equal reward
        let result = zero.clone() + reward.clone();
        assert_eq!(result, reward);

        // reward + zero should equal reward
        let result = reward.clone() + zero.clone();
        assert_eq!(result, reward);
    }

    /// Test that rewards can be added together
    #[test]
    fn test_reward_addition() {
        let reward1 = TestReward(10.0);
        let reward2 = TestReward(25.5);
        let result = reward1 + reward2;

        assert_eq!(result, TestReward(35.5));
    }

    /// Test that negative rewards can be added
    #[test]
    fn test_reward_negative_addition() {
        let positive = TestReward(100.0);
        let negative = TestReward(-30.0);
        let result = positive + negative;

        assert_eq!(result, TestReward(70.0));
    }

    /// Test that rewards can be converted to f32
    #[test]
    fn test_reward_into_f32() {
        let reward = TestReward(42.5);
        let as_f32: f32 = reward.into();

        assert_eq!(as_f32, 42.5);
    }

    /// Test that zero reward converts to 0.0
    #[test]
    fn test_reward_zero_into_f32() {
        let zero = TestReward::zero();
        let as_f32: f32 = zero.into();

        assert_eq!(as_f32, 0.0);
    }

    /// Test that rewards are cloneable
    #[test]
    fn test_reward_clone() {
        let original = TestReward(123.456);
        let cloned = original.clone();

        assert_eq!(original, cloned);
    }

    /// Test that rewards implement Debug
    #[test]
    fn test_reward_debug() {
        let reward = TestReward(42.0);
        let debug_str = format!("{:?}", reward);

        assert!(!debug_str.is_empty());
        assert!(debug_str.contains("TestReward"));
    }

    // ===== Arithmetic Properties Tests =====

    /// Test accumulated reward through chained additions
    #[test]
    fn test_reward_accumulation() {
        let mut accumulated = TestReward::zero();
        let rewards = vec![TestReward(10.0), TestReward(20.0), TestReward(15.0)];

        for reward in rewards {
            accumulated = accumulated + reward;
        }

        assert_eq!(accumulated, TestReward(45.0));
    }

    /// Test reward trait with floating point precision
    #[test]
    fn test_reward_floating_point_precision() {
        let r1 = TestReward(0.1);
        let r2 = TestReward(0.2);
        let result = r1 + r2;

        // Account for floating point imprecision
        let expected = 0.3;
        let as_f32: f32 = result.into();
        assert!((as_f32 - expected).abs() < 1e-6);
    }

    /// Test addition associativity: (a + b) + c == a + (b + c)
    #[test]
    fn test_reward_addition_associativity() {
        let r1 = TestReward(5.0);
        let r2 = TestReward(10.0);
        let r3 = TestReward(15.0);

        let left = (r1.clone() + r2.clone()) + r3.clone();
        let right = r1 + (r2 + r3);

        assert_eq!(left, right);
    }

    /// Test addition commutativity: a + b == b + a
    #[test]
    fn test_reward_addition_commutativity() {
        let r1 = TestReward(7.5);
        let r2 = TestReward(12.5);

        let left = r1.clone() + r2.clone();
        let right = r2 + r1;

        assert_eq!(left, right);
    }

    // ===== Special Values Tests =====

    /// Test reward arithmetic with large values
    #[test]
    fn test_reward_large_values() {
        let large1 = TestReward(1e6);
        let large2 = TestReward(1e6);

        let result = large1 + large2;
        let result_f32: f32 = result.into();

        assert_eq!(result_f32, 2e6);
    }

    /// Test reward arithmetic with small values
    #[test]
    fn test_reward_small_values() {
        let small1 = TestReward(1e-6);
        let small2 = TestReward(1e-6);

        let result = small1 + small2;
        let result_f32: f32 = result.into();

        assert!((result_f32 - 2e-6).abs() < 1e-7);
    }

    /// Test mixed positive and negative rewards
    #[test]
    fn test_reward_mixed_signs() {
        let positive = TestReward(10.0);
        let negative = TestReward(-5.0);

        let pos_then_neg = positive.clone() + negative.clone();
        let pos_then_neg_f32: f32 = pos_then_neg.into();

        let neg_then_pos = negative.clone() + positive.clone();
        let neg_then_pos_f32: f32 = neg_then_pos.into();

        assert_eq!(pos_then_neg_f32, 5.0);
        assert_eq!(neg_then_pos_f32, 5.0);
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

        history.add(TestObs, TestAct, TestReward(10.0));
        history.add(TestObs, TestAct, TestReward(20.0));
        history.add(TestObs, TestAct, TestReward(5.0));

        assert_eq!(history.len(), 3);
    }

    /// Test ExperienceTuple with rewards
    #[test]
    fn test_experience_tuple_creation() {
        let experience = ExperienceTuple {
            observation: TestObs,
            action: TestAct,
            reward: TestReward(50.0),
        };

        assert_eq!(experience.reward, TestReward(50.0));
    }

    /// Test that ExperienceTuple preserves reward through cloning
    #[test]
    fn test_experience_tuple_clone() {
        let original = ExperienceTuple {
            observation: TestObs,
            action: TestAct,
            reward: TestReward(75.0),
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
            },
            ExperienceTuple {
                observation: TestObs,
                action: TestAct,
                reward: TestReward(10.0),
            },
            ExperienceTuple {
                observation: TestObs,
                action: TestAct,
                reward: TestReward(15.0),
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
            history.add(TestObs, TestAct, TestReward(reward_val));
        }

        assert_eq!(history.len(), 5);
    }
}
