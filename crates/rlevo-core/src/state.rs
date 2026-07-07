//! Advanced state abstraction traits for non-Markovian and latent representations.
//!
//! This module extends the base [`State`] contract with higher-level abstractions
//! needed for POMDPs, recurrent policies, and world-model-based agents:
//! - [`MarkovState`] — verifies the Markov property holds for a representation
//! - [`BeliefState`] — probability distribution over possible states (POMDP)
//! - [`HiddenState`] — recurrent agent memory (e.g., RNN hidden state)
//! - [`LatentState`] — learned compact representation with encode/predict/decode
//! - [`StateAggregation`] — maps concrete states to abstract representatives
//! - [`Observable`] — modality-changing state→observation projection (`OR != SR`)
//!
//! [`State`]: crate::base::State

use crate::base::{Action, Observation, State};

/// Error type for state validation failures.
///
/// Returned by validation logic when a state's shape, contents, or element
/// count do not match the expectations of the calling code.
///
/// # Examples
///
/// ```
/// use rlevo_core::state::StateError;
///
/// let err = StateError::InvalidShape {
///     expected: vec![4, 4],
///     got: vec![4, 3],
/// };
/// assert!(err.to_string().contains("Invalid shape"));
///
/// let err = StateError::InvalidData("NaN in position field".into());
/// assert!(err.to_string().contains("NaN in position field"));
///
/// let err = StateError::InvalidSize { expected: 16, got: 12 };
/// assert!(err.to_string().contains("Invalid size"));
/// ```
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum StateError {
    /// Shape dimensions do not match expectations.
    #[error("Invalid shape: expected {expected:?}, got {got:?}")]
    InvalidShape {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    /// Data contents violate invariants.
    #[error("Invalid data: {0}")]
    InvalidData(String),
    /// Total element count does not match expectations.
    #[error("Invalid size: expected {expected}, got {got}")]
    InvalidSize { expected: usize, got: usize },
}

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
/// rule as the most recent action and new observation arrive.
///
/// # Type Parameters
///
/// - `SR`: Rank of the state space tensor (number of axes).
/// - `AR`: Rank of the action space tensor (number of axes).
/// - `S`: The underlying environment [`State`] type.
/// - `A`: The [`Action`] type taken by the agent.
pub trait BeliefState<const SR: usize, const AR: usize, S: State<SR>, A: Action<AR>>:
    Clone
{
    /// Updates the belief distribution given the last action taken and the
    /// newly received observation.
    fn update(&self, action: &A, observation: &S::Observation) -> Self;

    /// Draws a state sample from the current belief distribution.
    fn sample(&self) -> S;

    /// Returns the probability (or unnormalized weight) assigned to `state`.
    fn probability(&self, state: &S) -> f64;
}

/// Recurrent agent memory analogous to an RNN hidden state.
///
/// Implementations hold the internal summary of past observations (e.g., the
/// `h_t` vector of a GRU or LSTM). The hidden state is updated at each step
/// with the latest [`Observation`] and reset to a
/// zero vector at episode start.
///
/// # Type Parameters
///
/// - `R`: Rank of the observation space tensor used to update this state.
pub trait HiddenState<const R: usize>: Clone {
    /// The observation type used to update this hidden state.
    type Observation: Observation<R>;

    /// Incorporates `observation` into the hidden state in-place.
    fn update(&mut self, observation: &Self::Observation);

    /// Resets the hidden state to its initial value at episode start.
    fn reset(&mut self);
}

/// Learned compact representation with encode, predict, and decode steps.
///
/// Used by world-model agents (e.g., DreamerV3) that operate in a learned
/// latent space rather than the raw observation space.
///
/// # Type Parameters
///
/// - `R`: Rank of the observation space tensor this latent state is derived from.
/// - `AR`: Rank of the action space tensor used in the transition prediction step.
pub trait LatentState<const R: usize, const AR: usize>: Clone {
    /// The observation type this latent state is derived from.
    type Observation: Observation<R>;

    /// Projects `observation` into the latent space.
    fn encode(observation: &Self::Observation) -> Self;

    /// Rolls the latent state forward by one step given `action`.
    fn predict_next<A: Action<AR>>(&self, action: &A) -> Self;

    /// Reconstructs an observation from the latent representation.
    fn decode(&self) -> Self::Observation;
}

/// Maps concrete states to abstract representatives for state aggregation.
///
/// State aggregation is used in function approximation and hierarchical RL to
/// group similar states under a shared abstract representation.
///
/// # Type Parameters
///
/// - `SR`: Rank of the concrete state space tensor.
/// - `S`: The concrete [`State`] type being aggregated.
pub trait StateAggregation<const SR: usize, S: State<SR>> {
    /// The abstract state type produced by aggregation.
    type AbstractState: Clone + Eq;

    /// Returns the abstract representative for `state`.
    fn aggregate(&self, state: &S) -> Self::AbstractState;

    /// Returns `true` when `state1` and `state2` map to the same abstract state.
    fn same_aggregate(&self, state1: &S, state2: &S) -> bool {
        self.aggregate(state1) == self.aggregate(state2)
    }
}

/// Projects a state into an observation whose tensor order may differ from the
/// state's own.
///
/// [`State::observe`] welds its observation to the state's tensor order
/// (`type Observation: Observation<SR>`), so it can never change rank. That is
/// the right model for *information*-reducing partial observability (e.g.
/// dropping velocities from CartPole: a smaller-`shape`, same-order
/// observation). It cannot express a **modality change** — a compact, low-order
/// latent state observed through a higher-order sensor, such as an
/// emulator-RAM byte vector (rank 1) presented to the agent as a pixel image
/// (rank 2 or 3).
///
/// `Observable` is the typed home for that rank-changing projection. It is a
/// **standalone** trait (not a supertrait of [`State`]): a modality-changing
/// environment's state implements [`State<SR>`](State) for its full
/// representation *and* `Observable<OR>` for the projected observation, then
/// builds its snapshots from [`project`](Observable::project) instead of
/// [`observe`](State::observe). [`crate::environment::Environment`] already
/// permits `R != SR` (its observation and state types are independent), so no
/// change to the environment contract is required.
///
/// # Type Parameters
///
/// - `OR`: Tensor order (rank) of the projected observation. Named `OR` rather
///   than `R` — as on the sibling [`HiddenState`]/[`LatentState`] seams — to
///   make the state→observation rank decoupling explicit at every impl site
///   (`impl Observable<2> for Ram`) and to distinguish it from the state order
///   `SR`. This naming is deliberate; do not normalise it to `R`.
///
/// # Invariants
///
/// - **Total over valid states.** `project` is infallible: for any state the
///   environment considers valid, it returns a well-formed observation. The
///   output shape is the compile-time constant `Self::Observation::shape()`,
///   and the projection performs no I/O, so there is no runtime failure mode.
///   (A future emulator that can fail to read a framebuffer models that failure
///   at the [`Environment::step`](crate::environment::Environment::step)
///   boundary via [`EnvironmentError`](crate::environment::EnvironmentError),
///   not here.)
/// - **`OR` is independent of any `State<SR>` order** the type also implements;
///   `OR == SR`, `OR < SR`, and `OR > SR` are all permitted.
///
/// # Examples
///
/// A compact rank-1 state projected into a rank-2 pixel observation:
///
/// ```
/// use rlevo_core::base::Observation;
/// use rlevo_core::state::Observable;
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Debug, Clone)]
/// struct Ram {
///     byte: u8,
/// }
///
/// #[derive(Debug, Clone, Serialize, Deserialize)]
/// struct Pixels([[u8; 2]; 2]);
///
/// impl Observation<2> for Pixels {
///     fn shape() -> [usize; 2] {
///         [2, 2]
///     }
/// }
///
/// impl Observable<2> for Ram {
///     type Observation = Pixels;
///
///     fn project(&self) -> Pixels {
///         let b = self.byte;
///         Pixels([[b & 1, (b >> 1) & 1], [(b >> 2) & 1, (b >> 3) & 1]])
///     }
/// }
///
/// let obs = Ram { byte: 0b1011 }.project();
/// assert_eq!(Pixels::shape(), [2, 2]);
/// assert_eq!(<Pixels as Observation<2>>::RANK, 2);
/// assert_eq!(obs.0, [[1, 1], [0, 1]]);
/// ```
pub trait Observable<const OR: usize> {
    /// The observation produced by this projection, at tensor order `OR`.
    type Observation: Observation<OR>;

    /// Projects `self` into an observation whose order `OR` may differ from any
    /// [`State<SR>`](State) order the type also implements.
    ///
    /// This is the rank-changing analogue of [`State::observe`]: where `observe`
    /// reads out a same-order perception, `project` maps the state into a
    /// possibly different-order observation modality. It is total over valid
    /// states (see the trait [Invariants](Observable#invariants)).
    fn project(&self) -> Self::Observation;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    /// A rank-2 pixel observation: a 2x2 grid of bits.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct MockRamObservation {
        pixels: [[u8; 2]; 2],
    }

    impl Observation<2> for MockRamObservation {
        fn shape() -> [usize; 2] {
            [2, 2]
        }
    }

    /// A trivial rank-1 observation: the raw RAM byte, fully observable.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct MockRamByte {
        byte: u8,
    }

    impl Observation<1> for MockRamByte {
        fn shape() -> [usize; 1] {
            [1]
        }
    }

    /// A compact rank-1 state (one byte of emulator RAM) that is observed two
    /// ways: fully via [`State::observe`] (rank 1), and as a pixel image via
    /// [`Observable::project`] (rank 2) — the modality change `OR != SR`.
    #[derive(Debug, Clone)]
    struct MockRamState {
        byte: u8,
    }

    impl State<1> for MockRamState {
        type Observation = MockRamByte;

        fn shape() -> [usize; 1] {
            [1]
        }

        fn observe(&self) -> Self::Observation {
            MockRamByte { byte: self.byte }
        }

        fn is_valid(&self) -> bool {
            true
        }

        fn numel(&self) -> usize {
            1
        }
    }

    impl Observable<2> for MockRamState {
        type Observation = MockRamObservation;

        fn project(&self) -> Self::Observation {
            let b = self.byte;
            MockRamObservation {
                pixels: [[b & 1, (b >> 1) & 1], [(b >> 2) & 1, (b >> 3) & 1]],
            }
        }
    }

    /// The projected observation is a strictly higher tensor order than the
    /// state — the whole point of the trait.
    #[test]
    fn test_observable_changes_tensor_order() {
        assert_eq!(
            <MockRamState as State<1>>::shape().len(),
            1,
            "state is rank 1"
        );
        assert_eq!(
            <MockRamState as Observable<2>>::Observation::RANK,
            2,
            "projected observation is rank 2"
        );
        assert_ne!(
            <MockRamState as State<1>>::shape().len(),
            <MockRamState as Observable<2>>::Observation::shape().len(),
            "modality change: observation order differs from state order"
        );
    }

    /// `observe()` and `project()` coexist on one type and produce different
    /// modalities from the same state.
    #[test]
    fn test_observable_coexists_with_observe() {
        let state = MockRamState { byte: 0b1011 };

        let full: MockRamByte = state.observe();
        assert_eq!(
            full,
            MockRamByte { byte: 0b1011 },
            "observe is identity here"
        );

        let pixels = state.project();
        assert_eq!(
            pixels.pixels,
            [[1, 1], [0, 1]],
            "project unpacks the byte into a 2x2 pixel grid"
        );
    }

    /// The projected observation's declared shape matches its rank-2 contents.
    #[test]
    fn test_observable_projection_shape() {
        assert_eq!(
            MockRamObservation::shape(),
            [2, 2],
            "projected observation has shape [2, 2]"
        );
        let pixels = MockRamState { byte: 0 }.project();
        assert_eq!(pixels.pixels.len(), 2, "outer axis matches shape[0]");
        assert_eq!(pixels.pixels[0].len(), 2, "inner axis matches shape[1]");
    }
}
