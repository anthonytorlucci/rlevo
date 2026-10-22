//! State abstraction traits beyond the base [`State`] contract.
//!
//! - [`StateError`] — why a value is not a legal state
//! - [`Observable`] — optional pure-projection helper an env-side
//!   [`Sensor`](crate::environment::Sensor) may delegate to (e.g. `OR != SR`)
//!
//! [`State`]: crate::base::State

use crate::base::Observation;

/// Error type for state validation failures.
///
/// Returned by a constructor or conversion when a value's contents do not
/// make a legal state.
///
/// # Examples
///
/// ```
/// use rlevo_core::state::StateError;
///
/// let err = StateError::InvalidData("NaN in position field".into());
/// assert!(err.to_string().contains("NaN in position field"));
/// ```
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum StateError {
    /// Data contents violate invariants.
    #[error("Invalid data: {0}")]
    InvalidData(String),
}

/// Optional pure-projection helper: derives an observation directly from a state
/// value.
///
/// `Observable` is a convenience for the case where an observation is a **pure
/// function of the state** — no world / simulator context is needed. Since ADR
/// 0047 moved observation production off [`State`](crate::base::State) to the env-side
/// [`Sensor`](crate::environment::Sensor), `Observable` is no longer the home
/// for observation and is no longer required: it is a helper an environment's
/// [`Sensor`](crate::environment::Sensor) *may* delegate to when its observation
/// happens to be a pure projection (`sensor.observe(..) == next_state.project()`).
/// The `pixel_grid`
/// environment is the reference for that delegation.
///
/// It remains useful precisely where the projection is total and state-pure,
/// including the **modality-changing** case it was introduced for (ADR 0019): a
/// compact, low-order latent state projected through a higher-order sensor —
/// e.g. an emulator-RAM byte vector (rank 1) presented as a pixel image (rank 2
/// or 3). Because the projected order `OR` is a free parameter, `OR == SR`,
/// `OR < SR`, and `OR > SR` are all permitted.
///
/// # Type Parameters
///
/// - `OR`: Tensor order (rank) of the projected observation. Named `OR` rather
///   than `R` to
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
///   (A world-derived sensor that needs simulator access, or that can fail,
///   belongs on the env-side [`Sensor`](crate::environment::Sensor) reached from
///   [`Environment::step`](crate::environment::Environment::step), not here.)
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
/// assert_eq!(<Pixels as Observation<2>>::shape().len(), 2);
/// assert_eq!(obs.0, [[1, 1], [0, 1]]);
/// ```
pub trait Observable<const OR: usize> {
    /// The observation produced by this projection, at tensor order `OR`.
    type Observation: Observation<OR>;

    /// Projects `self` into an observation whose order `OR` may differ from any
    /// [`State<SR>`](crate::base::State) order the type also implements.
    ///
    /// The projection is a pure function of the state and is total over valid
    /// states (see the trait [Invariants](Observable#invariants)). An
    /// environment whose [`Sensor`](crate::environment::Sensor) observation is
    /// exactly this projection may delegate to it.
    fn project(&self) -> Self::Observation;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::State;

    /// A rank-2 pixel observation: a 2x2 grid of bits.
    ///
    /// **The absence of `Serialize` / `Deserialize` here is load-bearing.** This
    /// mock is the compile-time guard that [`Observation`] carries no serde
    /// supertrait (ADR 0064): its supertrait list is exactly
    /// `Debug + Clone + Send + Sync`, and if anyone re-adds
    /// `Serialize + for<'de> Deserialize<'de>` to it, `rlevo-core`'s own test
    /// build stops compiling on the `impl Observation<2>` below. Do not
    /// "helpfully" derive serde for this type — a consumer that needs to persist
    /// an observation declares the bound at its own seam.
    #[derive(Debug, Clone, PartialEq)]
    struct MockRamObservation {
        pixels: [[u8; 2]; 2],
    }

    impl Observation<2> for MockRamObservation {
        fn shape() -> [usize; 2] {
            [2, 2]
        }
    }

    /// A compact rank-1 state (one byte of emulator RAM) observed as a rank-2
    /// pixel image via [`Observable::project`] — the modality change `OR != SR`
    /// the helper was introduced for (ADR 0019).
    #[derive(Debug, Clone)]
    struct MockRamState {
        byte: u8,
    }

    impl State<1> for MockRamState {
        fn shape() -> [usize; 1] {
            [1]
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
            <MockRamState as Observable<2>>::Observation::shape().len(),
            2,
            "projected observation is rank 2"
        );
        assert_ne!(
            <MockRamState as State<1>>::shape().len(),
            <MockRamState as Observable<2>>::Observation::shape().len(),
            "modality change: observation order differs from state order"
        );
    }

    /// `project()` maps a state value into a higher-order observation modality.
    #[test]
    fn test_observable_projects_state_to_pixels() {
        let state = MockRamState { byte: 0b1011 };

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
