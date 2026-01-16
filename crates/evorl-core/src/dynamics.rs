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
