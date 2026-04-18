//! Advanced state abstraction traits for non-Markovian and latent representations.
//!
//! This module extends the base [`State`] contract with higher-level abstractions
//! needed for POMDPs, recurrent policies, and world-model-based agents:
//! - [`MarkovState`] — verifies the Markov property holds for a representation
//! - [`BeliefState`] — probability distribution over possible states (POMDP)
//! - [`HiddenState`] — recurrent agent memory (e.g., RNN hidden state)
//! - [`LatentState`] — learned compact representation with encode/predict/decode
//! - [`StateAggregation`] — maps concrete states to abstract representatives

use crate::base::{Action, Observation, State};
use std::fmt::Debug;

/// Error type for state validation failures.
#[derive(Debug, Clone, PartialEq)]
pub enum StateError {
    /// Shape dimensions do not match expectations.
    InvalidShape {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    /// Data contents violate invariants.
    InvalidData(String),
    /// Total element count does not match expectations.
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

/// Verifies that a state representation satisfies the Markov property.
///
/// A representation is Markov when the future is conditionally independent of
/// the past given the present. Tabular and neural Q-learning both assume this.
pub trait MarkovState {
    /// Returns `true` if this representation satisfies the Markov property.
    ///
    /// The default implementation returns `true`, which is correct for most
    /// fully-observable environments. Override to return `false` for raw pixel
    /// or partially-observable representations that require history stacking.
    fn is_markov() -> bool {
        true
    }
}

/// A probability distribution over possible environment states (POMDP belief).
///
/// Belief states are used in partially-observable settings where the agent
/// cannot observe the true state directly. The belief is updated via Bayes'
/// rule as new observations arrive.
pub trait BeliefState<const SD: usize, S: State<SD>>: Clone {
    /// Updates the belief distribution given a taken action and new observation.
    fn update(&self, action: &S::Observation, observation: &S::Observation) -> Self;

    /// Draws a state sample from the current belief distribution.
    fn sample(&self) -> S;

    /// Returns the probability (or unnormalized weight) assigned to `state`.
    fn probability(&self, state: &S) -> f64;
}

/// Recurrent agent memory analogous to an RNN hidden state.
pub trait HiddenState<const D: usize>: Clone {
    /// The observation type used to update this hidden state.
    type Observation: Observation<D>;

    /// Incorporates `observation` into the hidden state in-place.
    fn update(&mut self, observation: &Self::Observation);

    /// Resets the hidden state to its initial value at episode start.
    fn reset(&mut self);
}

/// Learned compact representation with encode, predict, and decode steps.
///
/// Used by world-model agents (e.g., DreamerV3) that operate in a learned
/// latent space rather than the raw observation space.
pub trait LatentState<const D: usize, const AD: usize>: Clone {
    /// The observation type this latent state is derived from.
    type Observation: Observation<D>;

    /// Projects `observation` into the latent space.
    fn encode(observation: &Self::Observation) -> Self;

    /// Rolls the latent state forward by one step given `action`.
    fn predict_next<A: Action<AD>>(&self, action: &A) -> Self;

    /// Reconstructs an observation from the latent representation.
    fn decode(&self) -> Self::Observation;
}

/// Maps concrete states to abstract representatives for state aggregation.
///
/// State aggregation is used in function approximation and hierarchical RL to
/// group similar states under a shared abstract representation.
pub trait StateAggregation<const SD: usize, S: State<SD>> {
    /// The abstract state type produced by aggregation.
    type AbstractState: Clone + Eq;

    /// Returns the abstract representative for `state`.
    fn aggregate(&self, state: &S) -> Self::AbstractState;

    /// Returns `true` when `state1` and `state2` map to the same abstract state.
    fn same_aggregate(&self, state1: &S, state2: &S) -> bool {
        self.aggregate(state1) == self.aggregate(state2)
    }
}
