//! Action space abstractions for reinforcement learning environments.
//!
//! This module provides a flexible type system for representing agent actions in RL environments.
//! Actions can be discrete (finite choices) or continuous (real-valued vectors).
//!
//! # Design Philosophy
//!
//! The action traits follow a layered design:
//! - [`Action`](crate::base::Action): Base trait providing validation and cloning semantics
//! - [`DiscreteAction`], [`ContinuousAction`]: Type-specific extensions
//! - [`BoundedAction`]: a [`ContinuousAction`] with per-component bounds
//!
//! # Action Types
//!
//! ## Discrete Actions
//!
//! Discrete actions represent a finite set of mutually exclusive choices (e.g., "move left",
//! "move right", "jump"). They are indexed from `0` to `ACTION_COUNT - 1`.
//!
//! ## Continuous Actions
//!
//! Continuous actions are real-valued vectors, typically used for motor control or
//! parametrized actions (e.g., steering angle, throttle).

use crate::base::Action;
use std::fmt::Debug;

/// Trait for discrete actions with a finite, enumerable set of choices.
///
/// Discrete actions represent mutually exclusive options that can be indexed by
/// integers from `0` to `ACTION_COUNT - 1`. Common examples include:
/// - Game controls (move left/right/jump)
/// - Categorical decisions (buy/hold/sell)
/// - Navigation directions (north/south/east/west)
///
/// # Type Safety
///
/// Implementations should ensure bidirectional conversion between indices and actions:
/// ```math
/// \forall i \in [0, \text{ACTION\_COUNT}): i = \text{from\_index}(i).\text{to\_index}()
/// \forall a: \text{Action}: a = \text{from\_index}(a.\text{to\_index}())
/// ```
pub trait DiscreteAction<const R: usize>: Action<R> {
    /// The total number of distinct actions in this action space.
    ///
    /// This constant defines the cardinality of the action space. It must be
    /// greater than zero and remain constant for the lifetime of the program.
    const ACTION_COUNT: usize;

    /// Constructs an action from its zero-based index.
    ///
    /// This method must be the inverse of [`to_index()`](DiscreteAction::to_index).
    ///
    /// # Panics
    ///
    /// Implementations should panic if `index >= ACTION_COUNT`, as this indicates
    /// a programming error (out-of-bounds access).
    fn from_index(index: usize) -> Self;

    /// Converts this action to its zero-based index.
    ///
    /// The returned index must be in the range `[0, ACTION_COUNT)` and must be
    /// the inverse of [`from_index()`](DiscreteAction::from_index).
    fn to_index(&self) -> usize;
}

/// Trait for continuous-valued actions represented as real-valued vectors.
///
/// Continuous actions are used when the agent's output is a vector of real numbers
/// rather than discrete choices. Common applications include:
/// - Robot motor control (joint angles, torques)
/// - Vehicle control (steering, throttle, brake)
/// - Continuous parameter tuning (learning rates, temperatures)
///
/// # Value Range
///
/// Continuous actions typically have bounded ranges (e.g., `[-1, 1]` or `[0, 1]`).
/// A type whose ranges are fixed declares them with [`BoundedAction`].
pub trait ContinuousAction<const R: usize>: Action<R> {
    /// The number of scalar components in this action's flattened representation —
    /// i.e. the length of the slice returned by [`as_slice`](ContinuousAction::as_slice)
    /// and consumed by [`from_slice`](ContinuousAction::from_slice).
    ///
    /// This is deliberately distinct from the tensor rank `R` and from
    /// `shape().iter().product()`: the workspace carries two encodings of component
    /// count (a rank-1 action with `shape() == [C]`, and a rank-C action with
    /// `shape() == [1; C]`), and neither derived quantity is universally correct.
    /// Implementors MUST declare it explicitly; there is intentionally no default
    /// (a default of `= R` would be wrong for every multi-component rank-1
    /// action). See ADR 0038.
    const COMPONENTS: usize;

    /// Returns a slice view of this action's component values.
    ///
    /// The returned slice must have exactly `COMPONENTS` elements. This is used for
    /// efficient serialization and tensor conversion.
    fn as_slice(&self) -> &[f32];

    /// Constructs an action from a slice of component values.
    ///
    /// The input slice must have exactly
    /// [`COMPONENTS`](ContinuousAction::COMPONENTS) elements — the flattened
    /// scalar count, *not* the tensor rank `R`. This is the inverse operation
    /// of [`as_slice()`](ContinuousAction::as_slice).
    ///
    /// # Panics
    ///
    /// Implementations should panic if `values.len() != Self::COMPONENTS`.
    fn from_slice(values: &[f32]) -> Self;
}

/// A [`ContinuousAction`] with statically-known per-component `[low, high]` bounds.
///
/// DDPG, TD3, SAC and other continuous-control algorithms need the
/// per-component action bounds to scale/shift neural-network outputs, to clip
/// target actions, and to sample uniform warm-up actions.
///
/// Bounds are keyed on [`ContinuousAction::COMPONENTS`] — the flattened scalar
/// count — **not** on the const generic `R`, which is the tensor rank. `R` is
/// retained only to name the [`ContinuousAction<R>`] supertrait; it does not
/// appear in any signature. See ADR 0053.
///
/// # Invariants
///
/// - `low().len() == high().len() == Self::COMPONENTS`.
/// - `low()[i] < high()[i]` for every component `i`.
///
/// These are contract obligations on the implementor, not compile-time
/// guarantees: `&'static [f32]` cannot encode its own length. Consumers that
/// index the bounds should `assert_eq!` the lengths against `Self::COMPONENTS`
/// once, at construction, so a nonconforming impl fails loudly and early.
///
/// # Implementing
///
/// Bounds are almost always compile-time literals, so the usual impl returns a
/// slice literal directly and allocates nothing:
///
/// ```
/// use rlevo_core::action::{BoundedAction, ContinuousAction};
/// use rlevo_core::base::Action;
/// # #[derive(Debug, Clone)]
/// # struct Throttle([f32; 2]);
/// # impl Action<1> for Throttle {
/// #     fn shape() -> [usize; 1] { [2] }
/// #     fn is_valid(&self) -> bool { self.0.iter().all(|v| v.is_finite()) }
/// # }
/// # impl ContinuousAction<1> for Throttle {
/// #     const COMPONENTS: usize = 2;
/// #     fn as_slice(&self) -> &[f32] { &self.0 }
/// #     fn from_slice(values: &[f32]) -> Self {
/// #         assert_eq!(values.len(), Self::COMPONENTS);
/// #         Throttle([values[0], values[1]])
/// #     }
/// # }
/// impl BoundedAction<1> for Throttle {
///     fn low() -> &'static [f32] {
///         &[-1.0, 0.0]
///     }
///
///     fn high() -> &'static [f32] {
///         &[1.0, 1.0]
///     }
/// }
/// # assert_eq!(Throttle::low().len(), Throttle::COMPONENTS);
/// ```
///
/// If bounds genuinely must be *computed* (rather than written as literals),
/// use a [`OnceLock`](std::sync::OnceLock) plus [`Box::leak`] to promote the
/// computed values to `&'static`:
///
/// ```
/// use std::sync::OnceLock;
///
/// fn low() -> &'static [f32] {
///     static LOW: OnceLock<&'static [f32]> = OnceLock::new();
///     LOW.get_or_init(|| {
///         let computed: Vec<f32> = (0..4).map(|i| -1.0 - i as f32).collect();
///         &*Box::leak(computed.into_boxed_slice())
///     })
/// }
/// assert_eq!(low().len(), 4);
/// ```
///
/// The leak is bounded — it happens at most once per type — but it is a leak,
/// so prefer literals. Note that these are *static* methods taking no `self`:
/// they cannot observe any instance and therefore cannot derive bounds from a
/// runtime environment configuration. Per-instance bounds would be a different
/// seam entirely (ADR 0053 §3).
pub trait BoundedAction<const R: usize>: ContinuousAction<R> {
    /// Returns the per-component lower bounds of this action space.
    ///
    /// The returned slice must have exactly [`Self::COMPONENTS`] elements and
    /// satisfy `low()[i] < high()[i]` for every component `i`. See the
    /// trait-level invariants for the full contract.
    ///
    /// [`Self::COMPONENTS`]: ContinuousAction::COMPONENTS
    fn low() -> &'static [f32];

    /// Returns the per-component upper bounds of this action space.
    ///
    /// The returned slice must have exactly [`Self::COMPONENTS`] elements and
    /// satisfy `low()[i] < high()[i]` for every component `i`. See the
    /// trait-level invariants for the full contract.
    ///
    /// [`Self::COMPONENTS`]: ContinuousAction::COMPONENTS
    fn high() -> &'static [f32];
}

/// Error indicating an action violated its type's constraints.
///
/// Returned when an action fails validation or when invalid conversions are
/// attempted (e.g., out-of-bounds indices, non-finite float values).
///
/// # Examples
///
/// ```
/// use rlevo_core::action::InvalidActionError;
///
/// let err = InvalidActionError { message: "index 5 out of bounds for ACTION_COUNT=4".into() };
/// assert!(err.to_string().contains("index 5"));
/// ```
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("Invalid action: {message}")]
pub struct InvalidActionError {
    /// Human-readable description of the constraint that was violated.
    pub message: String,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    // These tests assert exact round-trip of values that are stored and read
    // back without arithmetic, so bit-exact equality is the property under
    // test; an approximate comparison would weaken them.
    // The `as u8` casts below are in fixtures that assert their index is in
    // range first, so the truncation the lint warns about cannot occur.
    #![allow(clippy::float_cmp, clippy::cast_possible_truncation)]
    use super::*;
    use std::error::Error;

    // ========================================================================
    // Test Implementations
    // ========================================================================

    /// Simple discrete action with 4 possible choices.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum SimpleDiscreteAction {
        Left,
        Right,
        Up,
        Down,
    }

    impl Action<1> for SimpleDiscreteAction {
        fn shape() -> [usize; 1] {
            [4]
        }

        fn is_valid(&self) -> bool {
            true // All variants are always valid
        }
    }

    impl DiscreteAction<1> for SimpleDiscreteAction {
        const ACTION_COUNT: usize = 4;

        fn from_index(index: usize) -> Self {
            match index {
                0 => SimpleDiscreteAction::Left,
                1 => SimpleDiscreteAction::Right,
                2 => SimpleDiscreteAction::Up,
                3 => SimpleDiscreteAction::Down,
                _ => panic!("Index out of bounds: {index}"),
            }
        }

        fn to_index(&self) -> usize {
            match self {
                SimpleDiscreteAction::Left => 0,
                SimpleDiscreteAction::Right => 1,
                SimpleDiscreteAction::Up => 2,
                SimpleDiscreteAction::Down => 3,
            }
        }
    }

    /// Continuous action with 3 dimensions (e.g., 3D velocity).
    #[derive(Debug, Clone)]
    struct ContinuousActionTest {
        values: [f32; 3],
    }

    impl Action<3> for ContinuousActionTest {
        fn shape() -> [usize; 3] {
            [1, 1, 1] // Continuous dimensions typically have size 1
        }

        fn is_valid(&self) -> bool {
            self.values.iter().all(|v| v.is_finite())
        }
    }

    impl ContinuousAction<3> for ContinuousActionTest {
        const COMPONENTS: usize = 3;

        fn as_slice(&self) -> &[f32] {
            &self.values
        }

        fn from_slice(values: &[f32]) -> Self {
            assert_eq!(values.len(), 3, "Expected 3 values, got {}", values.len());
            ContinuousActionTest {
                values: [values[0], values[1], values[2]],
            }
        }
    }

    impl BoundedAction<3> for ContinuousActionTest {
        fn low() -> &'static [f32] {
            &[-1.0, -1.0, -1.0]
        }

        fn high() -> &'static [f32] {
            &[1.0, 1.0, 1.0]
        }
    }

    // ========================================================================
    // DiscreteAction Tests
    // ========================================================================

    #[test]
    fn test_discrete_action_shape() {
        assert_eq!(SimpleDiscreteAction::shape(), [4]);
    }

    #[test]
    fn test_discrete_action_count() {
        assert_eq!(SimpleDiscreteAction::ACTION_COUNT, 4);
    }

    #[test]
    fn test_discrete_action_from_index() {
        assert_eq!(
            SimpleDiscreteAction::from_index(0),
            SimpleDiscreteAction::Left
        );
        assert_eq!(
            SimpleDiscreteAction::from_index(1),
            SimpleDiscreteAction::Right
        );
        assert_eq!(
            SimpleDiscreteAction::from_index(2),
            SimpleDiscreteAction::Up
        );
        assert_eq!(
            SimpleDiscreteAction::from_index(3),
            SimpleDiscreteAction::Down
        );
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_discrete_action_from_index_out_of_bounds() {
        SimpleDiscreteAction::from_index(4);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_discrete_action_from_index_negative_like() {
        // Note: usize can't be negative, but we test the boundary
        SimpleDiscreteAction::from_index(100);
    }

    #[test]
    fn test_discrete_action_to_index() {
        assert_eq!(SimpleDiscreteAction::Left.to_index(), 0);
        assert_eq!(SimpleDiscreteAction::Right.to_index(), 1);
        assert_eq!(SimpleDiscreteAction::Up.to_index(), 2);
        assert_eq!(SimpleDiscreteAction::Down.to_index(), 3);
    }

    #[test]
    fn test_discrete_action_roundtrip() {
        // Test bidirectional conversion
        for i in 0..SimpleDiscreteAction::ACTION_COUNT {
            let action = SimpleDiscreteAction::from_index(i);
            assert_eq!(action.to_index(), i);
        }
    }

    #[test]
    fn test_discrete_action_is_valid() {
        for i in 0..SimpleDiscreteAction::ACTION_COUNT {
            let action = SimpleDiscreteAction::from_index(i);
            assert!(action.is_valid());
        }
    }

    // ========================================================================
    // ContinuousAction Tests
    // ========================================================================

    #[test]
    fn test_continuous_action_shape() {
        assert_eq!(ContinuousActionTest::shape(), [1, 1, 1]);
    }

    #[test]
    fn test_continuous_action_as_slice() {
        let action = ContinuousActionTest {
            values: [0.5, -0.3, 1.0],
        };
        let slice = action.as_slice();
        assert_eq!(slice.len(), 3);
        assert_eq!(slice, &[0.5, -0.3, 1.0]);
    }

    #[test]
    fn test_continuous_action_from_slice() {
        let values = [0.1, 0.2, 0.3];
        let action = ContinuousActionTest::from_slice(&values);
        assert_eq!(action.values, values);
    }

    #[test]
    #[should_panic(expected = "Expected 3 values")]
    fn test_continuous_action_from_slice_wrong_size() {
        let values = [0.1, 0.2];
        ContinuousActionTest::from_slice(&values);
    }

    #[test]
    fn test_continuous_action_roundtrip() {
        let original = [0.5, -0.3, 0.9];
        let action = ContinuousActionTest::from_slice(&original);
        assert_eq!(action.as_slice(), &original);
    }

    #[test]
    fn test_continuous_action_is_valid_finite() {
        let action = ContinuousActionTest {
            values: [0.5, -0.3, 1.0],
        };
        assert!(action.is_valid());
    }

    #[test]
    fn test_continuous_action_is_invalid_nan() {
        let action = ContinuousActionTest {
            values: [f32::NAN, 0.5, 1.0],
        };
        assert!(!action.is_valid());
    }

    #[test]
    fn test_continuous_action_is_invalid_inf() {
        let action = ContinuousActionTest {
            values: [f32::INFINITY, 0.5, 1.0],
        };
        assert!(!action.is_valid());

        let action = ContinuousActionTest {
            values: [f32::NEG_INFINITY, 0.5, 1.0],
        };
        assert!(!action.is_valid());
    }

    // ========================================================================
    // InvalidActionError Tests
    // ========================================================================

    #[test]
    fn test_invalid_action_error_creation() {
        let error = InvalidActionError {
            message: String::from("Index out of bounds"),
        };
        assert_eq!(error.message, "Index out of bounds");
    }

    #[test]
    fn test_invalid_action_error_display() {
        let error = InvalidActionError {
            message: String::from("Invalid value"),
        };
        let displayed = format!("{error}");
        assert_eq!(displayed, "Invalid action: Invalid value");
    }

    #[test]
    fn test_invalid_action_error_debug() {
        let error = InvalidActionError {
            message: String::from("Test error"),
        };
        let debug_str = format!("{error:?}");
        assert!(debug_str.contains("Test error"));
    }

    #[test]
    fn test_invalid_action_error_clone() {
        let error = InvalidActionError {
            message: String::from("Original"),
        };
        let cloned = error.clone();
        assert_eq!(error, cloned);
    }

    #[test]
    fn test_invalid_action_error_equality() {
        let error1 = InvalidActionError {
            message: String::from("Same error"),
        };
        let error2 = InvalidActionError {
            message: String::from("Same error"),
        };
        let error3 = InvalidActionError {
            message: String::from("Different error"),
        };

        assert_eq!(error1, error2);
        assert_ne!(error1, error3);
    }

    #[test]
    fn test_invalid_action_error_is_error() {
        let error: Box<dyn Error> = Box::new(InvalidActionError {
            message: String::from("Test"),
        });
        // Should be able to use as std::error::Error trait object
        let _msg = error.to_string();
    }

    // ========================================================================
    // Integration Tests
    // ========================================================================

    #[test]
    fn test_discrete_action_clone_and_debug() {
        let action = SimpleDiscreteAction::Left;
        let cloned = action;
        assert_eq!(action, cloned);

        let debug_str = format!("{action:?}");
        assert!(debug_str.contains("Left"));
    }

    #[test]
    fn test_continuous_action_clone_and_debug() {
        let action = ContinuousActionTest {
            values: [0.1, 0.2, 0.3],
        };
        let cloned = action.clone();
        assert_eq!(action.as_slice(), cloned.as_slice());

        let debug_str = format!("{action:?}");
        assert!(debug_str.contains("values"));
    }

    #[test]
    fn test_large_discrete_action_space() {
        // Test with 256 actions
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        struct LargeDiscreteAction(u8);

        impl Action<1> for LargeDiscreteAction {
            fn shape() -> [usize; 1] {
                [256]
            }

            fn is_valid(&self) -> bool {
                true
            }
        }

        impl DiscreteAction<1> for LargeDiscreteAction {
            const ACTION_COUNT: usize = 256;

            fn from_index(index: usize) -> Self {
                assert!(index < 256);
                LargeDiscreteAction(index as u8)
            }

            fn to_index(&self) -> usize {
                self.0 as usize
            }
        }

        // Verify roundtrip for a few samples
        for i in [0, 1, 127, 255] {
            let action = LargeDiscreteAction::from_index(i);
            assert_eq!(action.to_index(), i);
        }
    }

    #[test]
    fn test_continuous_action_with_zero_values() {
        let action = ContinuousActionTest {
            values: [0.0, 0.0, 0.0],
        };
        assert!(action.is_valid());
        assert_eq!(action.as_slice(), &[0.0, 0.0, 0.0]);
    }

    // ========================================================================
    // BoundedAction Tests
    // ========================================================================

    #[test]
    fn test_bounded_action_bounds_are_component_keyed() {
        // ADR 0053: bound slices are keyed on COMPONENTS (the flattened scalar
        // count), not on the rank `R`. For `ContinuousActionTest` these happen to agree
        // (Convention B: rank 3, 3 components), so assert against COMPONENTS
        // explicitly — that is the contract a multi-component rank-1 impl must
        // also satisfy.
        assert_eq!(
            ContinuousActionTest::low().len(),
            ContinuousActionTest::COMPONENTS,
            "low() length must equal COMPONENTS"
        );
        assert_eq!(
            ContinuousActionTest::high().len(),
            ContinuousActionTest::COMPONENTS,
            "high() length must equal COMPONENTS"
        );
    }

    #[test]
    fn test_bounded_action_low_strictly_below_high() {
        let low = ContinuousActionTest::low();
        let high = ContinuousActionTest::high();
        assert_eq!(low.len(), high.len(), "low()/high() lengths must agree");
        for i in 0..ContinuousActionTest::COMPONENTS {
            assert!(low[i] < high[i], "bound {i}: low >= high");
        }
    }
}
