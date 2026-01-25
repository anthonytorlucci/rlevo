use crate::base::{Action, Observation, State};
use std::fmt::Debug;

/// Error types for state operations
#[derive(Debug, Clone, PartialEq)]
pub enum StateError {
    InvalidShape {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    InvalidData(String),
    InvalidSize {
        expected: usize,
        got: usize,
    },
}

impl std::fmt::Display for StateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StateError::InvalidShape { expected, got } => {
                write!(f, "Invalid shape: expected {:?}, got {:?}", expected, got)
            }
            StateError::InvalidData(msg) => write!(f, "Invalid data: {}", msg),
            StateError::InvalidSize { expected, got } => {
                write!(f, "Invalid size: expected {}, got {}", expected, got)
            }
        }
    }
}

impl std::error::Error for StateError {}

// ----------------------------------------------------------------------------
/// Markov property: future independent of past given present
pub trait MarkovState {
    /// Returns true if this representation satisfies Markov property
    fn is_markov() -> bool {
        true
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
