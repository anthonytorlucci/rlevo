//! Action space abstractions for reinforcement learning environments.
//!
//! This module provides a flexible type system for representing agent actions in RL environments.
//! Actions can be discrete (finite choices), multi-discrete (multiple independent discrete choices),
//! or continuous (real-valued vectors).
//!
//! # Design Philosophy
//!
//! The action traits follow a layered design:
//! - [`Action`]: Base trait providing validation and cloning semantics
//! - [`DiscreteAction`], [`MultiDiscreteAction`], [`ContinuousAction`]: Type-specific extensions
//!
//! # Action Types
//!
//! ## Discrete Actions
//!
//! Discrete actions represent a finite set of mutually exclusive choices (e.g., "move left",
//! "move right", "jump"). They are indexed from `0` to `ACTION_COUNT - 1`.
//!
//! ## Multi-Discrete Actions
//!
//! Multi-discrete actions consist of multiple independent discrete choices, such as selecting
//! both a direction and an attack type simultaneously.
//!
//! ## Continuous Actions
//!
//! Continuous actions are real-valued vectors, typically used for motor control or
//! parametrized actions (e.g., steering angle, throttle).
//!
//! # Test Suite
//!
//! This module includes a comprehensive test suite covering:
//!
//! ## DiscreteAction Tests (10 tests)
//! - `test_discrete_action_shape`: Verifies shape and dimension constants
//! - `test_discrete_action_count`: Checks action count constant
//! - `test_discrete_action_from_index`: Tests index-to-action conversion
//! - `test_discrete_action_from_index_out_of_bounds`: Validates panic on invalid indices
//! - `test_discrete_action_to_index`: Tests action-to-index conversion
//! - `test_discrete_action_roundtrip`: Ensures bidirectional conversion consistency
//! - `test_discrete_action_enumerate`: Verifies all actions are enumerated correctly
//! - `test_discrete_action_random`: Tests random action generation
//! - `test_discrete_action_is_valid`: Validates the `is_valid()` predicate
//! - `test_discrete_action_clone_and_debug`: Tests Debug and Clone trait implementations
//!
//! ## MultiDiscreteAction Tests (11 tests)
//! - `test_multidiscrete_action_shape`: Verifies multi-dimensional shape
//! - `test_multidiscrete_action_from_indices`: Tests multi-index conversion
//! - `test_multidiscrete_action_from_indices_*_out_of_bounds`: Validates panic on invalid indices
//! - `test_multidiscrete_action_to_indices`: Tests reverse conversion
//! - `test_multidiscrete_action_roundtrip`: Ensures bidirectional consistency
//! - `test_multidiscrete_action_enumerate`: Verifies all action combinations are enumerated
//! - `test_multidiscrete_action_enumerate_large_space`: Tests scalability with large action spaces
//! - `test_multidiscrete_action_random`: Tests random sampling
//! - `test_multidiscrete_action_is_valid`: Validates constraints
//! - `test_multidiscrete_action_clone_and_debug`: Tests trait implementations
//!
//! ## ContinuousAction Tests (15 tests)
//! - `test_continuous_action_shape`: Verifies shape specification
//! - `test_continuous_action_as_slice`: Tests slice view access
//! - `test_continuous_action_from_slice`: Tests construction from slice
//! - `test_continuous_action_from_slice_wrong_size`: Validates dimension checking
//! - `test_continuous_action_roundtrip`: Ensures slice conversion consistency
//! - `test_continuous_action_clip_*`: Tests clipping behavior (within, exceeds max/min, mixed, extreme)
//! - `test_continuous_action_clip_chaining`: Verifies method chaining
//! - `test_continuous_action_random`: Tests random action generation
//! - `test_continuous_action_is_valid_*`: Tests validity checking (finite, NaN, Inf)
//! - `test_continuous_action_with_zero_values`: Tests edge case with zero values
//! - `test_continuous_action_clone_and_debug`: Tests trait implementations
//!
//! ## InvalidActionError Tests (6 tests)
//! - `test_invalid_action_error_creation`: Tests error instantiation
//! - `test_invalid_action_error_display`: Tests Display trait formatting
//! - `test_invalid_action_error_debug`: Tests Debug trait formatting
//! - `test_invalid_action_error_clone`: Tests Clone implementation
//! - `test_invalid_action_error_equality`: Tests PartialEq implementation
//! - `test_invalid_action_error_is_error`: Tests std::error::Error trait compatibility
//!
//! ## Integration Tests (4 tests)
//! - `test_large_discrete_action_space`: Tests with 256 actions
//! - `test_continuous_action_extreme_clip_bounds`: Tests edge cases in clipping
//! - Various clone/debug/trait tests across different action types

use crate::base::Action;
use std::error::Error;
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
/// ```text
/// ∀ i ∈ [0, ACTION_COUNT): i == from_index(i).to_index()
/// ∀ a: Action: a == from_index(a.to_index())
/// ```
///
/// # Performance
///
/// For performance-critical code, prefer `from_index()` over `random()` when you
/// already have an index (e.g., from a neural network's argmax). The `random()`
/// method allocates a thread-local RNG on each call.
pub trait DiscreteAction<const D: usize>: Action<D> {
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

    /// Samples a uniformly random action from this action space.
    ///
    /// This is a convenience method for exploration in reinforcement learning.
    /// It uses thread-local RNG state, so it's safe to call from multiple threads
    /// but will produce different sequences per thread.
    ///
    /// # Performance
    ///
    /// If you already have an index from another source (e.g., a neural network
    /// output), use `from_index()` directly instead of this method.
    fn random() -> Self
    where
        Self: Sized,
    {
        use rand::RngExt;
        let mut rng = rand::rng();
        let index = rng.random_range(0..Self::ACTION_COUNT);
        Self::from_index(index)
    }

    /// Returns a vector containing all possible actions in index order.
    ///
    /// This is useful for tabular RL methods (e.g., Q-learning) that need to
    /// iterate over the entire action space. The returned vector has length
    /// `ACTION_COUNT` with actions ordered by their index.
    ///
    /// # Performance
    ///
    /// This allocates a vector of size `ACTION_COUNT`. For large action spaces,
    /// consider using an iterator pattern instead (not currently provided).
    fn enumerate() -> Vec<Self>
    where
        Self: Sized,
    {
        (0..Self::ACTION_COUNT).map(Self::from_index).collect()
    }
}

/// Trait for actions with multiple independent discrete dimensions.
///
/// Multi-discrete actions represent scenarios where an agent must make several
/// independent categorical choices simultaneously. Each dimension can have a
/// different number of options. This is common in:
/// - Strategy games (select unit + select action + select target)
/// - Multi-agent coordination (each agent picks a discrete action)
/// - Parameterized actions (choose action type + intensity level)
///
/// # Dimensionality
///
/// The const generic `D` specifies the number of dimensions. Each dimension
/// can have a different cardinality, defined by [`shape()`](Action::shape).
///
/// The total number of action combinations is the product of all dimension sizes:
/// ```text
/// total_actions = ∏ shape()[i]
/// ```
///
/// # Caution: Combinatorial Explosion
///
/// Be careful with [`enumerate()`](MultiDiscreteAction::enumerate) on large action spaces.
/// A 3D action space with dimensions [10, 10, 10] produces 1000 actions, but
/// [100, 100, 100] produces 1,000,000!
pub trait MultiDiscreteAction<const D: usize>: Action<D> {
    /// Constructs an action from multi-dimensional indices.
    ///
    /// Each index must be in the range `[0, shape()[i])` for dimension `i`.
    ///
    /// # Panics
    ///
    /// Implementations should panic if any index is out of bounds for its dimension.
    fn from_indices(indices: [usize; D]) -> Self;

    /// Converts this action to its multi-dimensional index representation.
    ///
    /// The returned array must satisfy: each element `i` is in `[0, shape()[i])`.
    /// This method must be the inverse of [`from_indices()`](MultiDiscreteAction::from_indices).
    fn to_indices(&self) -> [usize; D];

    /// Samples a uniformly random action from this multi-discrete action space.
    ///
    /// Each dimension is sampled independently and uniformly from its valid range.
    /// This is useful for exploration in reinforcement learning.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let random_action = StrategyAction::random();
    /// assert!(random_action.is_valid());
    /// ```
    fn random() -> Self
    where
        Self: Sized,
    {
        use rand::RngExt;
        let mut rng = rand::rng();
        let space = Self::shape();
        let indices = space.map(|dim| rng.random_range(0..dim));
        Self::from_indices(indices)
    }

    /// Returns all possible action combinations in this space.
    ///
    /// # Warning
    ///
    /// This method generates **all** combinations across all dimensions. The number
    /// of actions grows multiplicatively with the product of dimension sizes:
    ///
    /// - `[10, 10, 10]` → 1,000 actions (manageable)
    /// - `[50, 50, 50]` → 125,000 actions (marginal)
    /// - `[100, 100, 100]` → 1,000,000 actions (likely too large)
    ///
    /// Use this method only when you need to iterate over the entire action space
    /// (e.g., for exact policy evaluation in tabular methods).
    ///
    /// # Panics
    ///
    /// May panic or run out of memory if the action space is too large.
    fn enumerate() -> Vec<Self>
    where
        Self: Sized,
    {
        let space = Self::shape();
        let total: usize = space.iter().product();
        let mut actions = Vec::with_capacity(total);

        fn generate<const D: usize, T: MultiDiscreteAction<D>>(
            space: &[usize; D],
            current: &mut [usize; D],
            dim: usize,
            actions: &mut Vec<T>,
        ) {
            if dim == D {
                actions.push(T::from_indices(*current));
                return;
            }
            for i in 0..space[dim] {
                current[dim] = i;
                generate(space, current, dim + 1, actions);
            }
        }

        let mut current = [0; D];
        generate(&space, &mut current, 0, &mut actions);
        actions
    }
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
/// The [`clip()`](ContinuousAction::clip) method enforces these bounds.
///
/// # Neural Network Integration
///
/// Continuous actions are typically produced by neural networks with `tanh` or
/// `sigmoid` activation functions. Use [`clip()`](ContinuousAction::clip) to
/// ensure outputs stay within valid ranges.
pub trait ContinuousAction<const D: usize>: Action<D> {
    /// Returns a slice view of this action's component values.
    ///
    /// The returned slice must have exactly `DIM` elements. This is used for
    /// efficient serialization and tensor conversion.
    fn as_slice(&self) -> &[f32];

    /// Returns a new action with all components clipped to `[min, max]`.
    ///
    /// This is essential for ensuring neural network outputs (which may exceed
    /// valid ranges due to numerical errors or exploration noise) stay within
    /// acceptable bounds.
    ///
    /// # Common Use Cases
    ///
    /// - Enforcing action space bounds after neural network output
    /// - Adding exploration noise while maintaining validity
    /// - Recovering from numerical instability
    fn clip(&self, min: f32, max: f32) -> Self;

    /// Samples a random action with components uniformly distributed in `[-1, 1]`.
    ///
    /// The default implementation generates uniform random values. Override this
    /// method if you need different sampling behavior (e.g., Gaussian noise,
    /// domain-specific distributions).
    fn random() -> Self
    where
        Self: Sized,
    {
        use rand::RngExt;
        let mut rng = rand::rng();
        // Default implementation - override for custom behavior
        let values: Vec<f32> = (0..Self::DIM)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        Self::from_slice(&values)
    }

    /// Constructs an action from a slice of component values.
    ///
    /// The input slice must have exactly `DIM` elements. This is the inverse
    /// operation of [`as_slice()`](ContinuousAction::as_slice).
    ///
    /// # Panics
    ///
    /// Implementations should panic if `values.len() != DIM`.
    fn from_slice(values: &[f32]) -> Self;
}

/// A [`ContinuousAction`] with statically-known `[low, high]` component bounds.
///
/// DDPG and other continuous-control algorithms need the per-component action
/// bounds to scale/shift neural-network outputs and to sample uniform warm-up
/// actions. Expose them via associated static methods rather than associated
/// constants so implementors can still derive bounds from a runtime env config
/// (e.g. a `max_torque` field) while presenting a uniform API.
///
/// # Invariants
///
/// - `low()[i] < high()[i]` for every component `i`.
/// - [`ContinuousAction::clip`] must be a no-op on an action whose components
///   already lie in `[low, high]`.
pub trait BoundedAction<const D: usize>: ContinuousAction<D> {
    /// Per-component lower bounds.
    fn low() -> [f32; D];
    /// Per-component upper bounds.
    fn high() -> [f32; D];
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
#[derive(Debug, Clone, PartialEq)]
pub struct InvalidActionError {
    /// Human-readable description of the constraint that was violated.
    pub message: String,
}

impl std::fmt::Display for InvalidActionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Invalid action: {}", self.message)
    }
}

impl Error for InvalidActionError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

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
                _ => panic!("Index out of bounds: {}", index),
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

    /// Multi-discrete action with 2 dimensions.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct MultiActionTest {
        direction: usize, // 0-3
        intensity: usize, // 0-2
    }

    impl Action<2> for MultiActionTest {
        fn shape() -> [usize; 2] {
            [4, 3]
        }

        fn is_valid(&self) -> bool {
            self.direction < 4 && self.intensity < 3
        }
    }

    impl MultiDiscreteAction<2> for MultiActionTest {
        fn from_indices(indices: [usize; 2]) -> Self {
            if indices[0] >= 4 {
                panic!("Direction index out of bounds: {}", indices[0]);
            }
            if indices[1] >= 3 {
                panic!("Intensity index out of bounds: {}", indices[1]);
            }
            MultiActionTest {
                direction: indices[0],
                intensity: indices[1],
            }
        }

        fn to_indices(&self) -> [usize; 2] {
            [self.direction, self.intensity]
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
        fn as_slice(&self) -> &[f32] {
            &self.values
        }

        fn clip(&self, min: f32, max: f32) -> Self {
            let clipped = self
                .values
                .iter()
                .map(|&v| v.max(min).min(max))
                .collect::<Vec<_>>();
            ContinuousActionTest {
                values: [clipped[0], clipped[1], clipped[2]],
            }
        }

        fn from_slice(values: &[f32]) -> Self {
            assert_eq!(values.len(), 3, "Expected 3 values, got {}", values.len());
            ContinuousActionTest {
                values: [values[0], values[1], values[2]],
            }
        }
    }

    impl BoundedAction<3> for ContinuousActionTest {
        fn low() -> [f32; 3] {
            [-1.0, -1.0, -1.0]
        }

        fn high() -> [f32; 3] {
            [1.0, 1.0, 1.0]
        }
    }

    // ========================================================================
    // DiscreteAction Tests
    // ========================================================================

    #[test]
    fn test_discrete_action_shape() {
        assert_eq!(SimpleDiscreteAction::shape(), [4]);
        assert_eq!(SimpleDiscreteAction::DIM, 1);
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
    fn test_discrete_action_enumerate() {
        let actions = SimpleDiscreteAction::enumerate();
        assert_eq!(actions.len(), 4);
        assert_eq!(
            actions,
            vec![
                SimpleDiscreteAction::Left,
                SimpleDiscreteAction::Right,
                SimpleDiscreteAction::Up,
                SimpleDiscreteAction::Down
            ]
        );
    }

    #[test]
    fn test_discrete_action_random() {
        for _ in 0..100 {
            let action = SimpleDiscreteAction::random();
            let index = action.to_index();
            assert!(index < SimpleDiscreteAction::ACTION_COUNT);
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
    // MultiDiscreteAction Tests
    // ========================================================================

    #[test]
    fn test_multidiscrete_action_shape() {
        assert_eq!(MultiActionTest::shape(), [4, 3]);
        assert_eq!(MultiActionTest::DIM, 2);
    }

    #[test]
    fn test_multidiscrete_action_from_indices() {
        let action = MultiActionTest::from_indices([0, 0]);
        assert_eq!(action.direction, 0);
        assert_eq!(action.intensity, 0);

        let action = MultiActionTest::from_indices([3, 2]);
        assert_eq!(action.direction, 3);
        assert_eq!(action.intensity, 2);
    }

    #[test]
    #[should_panic(expected = "Direction index out of bounds")]
    fn test_multidiscrete_action_from_indices_direction_out_of_bounds() {
        MultiActionTest::from_indices([4, 0]);
    }

    #[test]
    #[should_panic(expected = "Intensity index out of bounds")]
    fn test_multidiscrete_action_from_indices_intensity_out_of_bounds() {
        MultiActionTest::from_indices([0, 3]);
    }

    #[test]
    fn test_multidiscrete_action_to_indices() {
        let action = MultiActionTest::from_indices([2, 1]);
        assert_eq!(action.to_indices(), [2, 1]);
    }

    #[test]
    fn test_multidiscrete_action_roundtrip() {
        for d in 0..4 {
            for i in 0..3 {
                let action = MultiActionTest::from_indices([d, i]);
                assert_eq!(action.to_indices(), [d, i]);
            }
        }
    }

    #[test]
    fn test_multidiscrete_action_enumerate() {
        let actions = MultiActionTest::enumerate();
        // 4 directions × 3 intensities = 12 total actions
        assert_eq!(actions.len(), 12);

        // Verify all combinations are present
        for (idx, action) in actions.iter().enumerate() {
            let expected_d = idx / 3;
            let expected_i = idx % 3;
            assert_eq!(action.direction, expected_d);
            assert_eq!(action.intensity, expected_i);
        }
    }

    #[test]
    fn test_multidiscrete_action_enumerate_large_space() {
        // Test with 3D space: [5, 5, 5] = 125 total actions
        #[derive(Debug, Clone)]
        struct LargeMultiAction([usize; 3]);

        impl Action<3> for LargeMultiAction {
            fn shape() -> [usize; 3] {
                [5, 5, 5]
            }

            fn is_valid(&self) -> bool {
                self.0.iter().enumerate().all(|(i, &v)| v < [5, 5, 5][i])
            }
        }

        impl MultiDiscreteAction<3> for LargeMultiAction {
            fn from_indices(indices: [usize; 3]) -> Self {
                for (i, &idx) in indices.iter().enumerate() {
                    assert!(idx < 5, "Index {} out of bounds", i);
                }
                LargeMultiAction(indices)
            }

            fn to_indices(&self) -> [usize; 3] {
                self.0
            }
        }

        let actions = LargeMultiAction::enumerate();
        assert_eq!(actions.len(), 125);
    }

    #[test]
    fn test_multidiscrete_action_random() {
        for _ in 0..100 {
            let action = MultiActionTest::random();
            assert!(action.is_valid());
            let indices = action.to_indices();
            assert!(indices[0] < 4);
            assert!(indices[1] < 3);
        }
    }

    #[test]
    fn test_multidiscrete_action_is_valid() {
        // Valid actions
        assert!(MultiActionTest::from_indices([0, 0]).is_valid());
        assert!(MultiActionTest::from_indices([3, 2]).is_valid());

        // Invalid actions created directly
        let invalid = MultiActionTest {
            direction: 5,
            intensity: 0,
        };
        assert!(!invalid.is_valid());

        let invalid = MultiActionTest {
            direction: 0,
            intensity: 5,
        };
        assert!(!invalid.is_valid());
    }

    // ========================================================================
    // ContinuousAction Tests
    // ========================================================================

    #[test]
    fn test_continuous_action_shape() {
        assert_eq!(ContinuousActionTest::shape(), [1, 1, 1]);
        assert_eq!(ContinuousActionTest::DIM, 3);
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
    fn test_continuous_action_clip_within_bounds() {
        let action = ContinuousActionTest {
            values: [0.0, 0.5, -0.5],
        };
        let clipped = action.clip(-1.0, 1.0);
        assert_eq!(clipped.values, [0.0, 0.5, -0.5]);
    }

    #[test]
    fn test_continuous_action_clip_exceeds_max() {
        let action = ContinuousActionTest {
            values: [2.0, 1.5, 3.0],
        };
        let clipped = action.clip(-1.0, 1.0);
        assert_eq!(clipped.values, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_continuous_action_clip_exceeds_min() {
        let action = ContinuousActionTest {
            values: [-2.0, -1.5, -3.0],
        };
        let clipped = action.clip(-1.0, 1.0);
        assert_eq!(clipped.values, [-1.0, -1.0, -1.0]);
    }

    #[test]
    fn test_continuous_action_clip_mixed() {
        let action = ContinuousActionTest {
            values: [2.0, 0.5, -2.0],
        };
        let clipped = action.clip(-1.0, 1.0);
        assert_eq!(clipped.values, [1.0, 0.5, -1.0]);
    }

    #[test]
    fn test_continuous_action_random() {
        for _ in 0..100 {
            let action = ContinuousActionTest::random();
            assert!(action.is_valid());
            for &value in action.as_slice() {
                assert!((-1.0..=1.0).contains(&value));
                assert!(value.is_finite());
            }
        }
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
        let displayed = format!("{}", error);
        assert_eq!(displayed, "Invalid action: Invalid value");
    }

    #[test]
    fn test_invalid_action_error_debug() {
        let error = InvalidActionError {
            message: String::from("Test error"),
        };
        let debug_str = format!("{:?}", error);
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

        let debug_str = format!("{:?}", action);
        assert!(debug_str.contains("Left"));
    }

    #[test]
    fn test_multidiscrete_action_clone_and_debug() {
        let action = MultiActionTest::from_indices([1, 2]);
        let cloned = action;
        assert_eq!(action, cloned);

        let debug_str = format!("{:?}", action);
        assert!(debug_str.contains("direction"));
    }

    #[test]
    fn test_continuous_action_clone_and_debug() {
        let action = ContinuousActionTest {
            values: [0.1, 0.2, 0.3],
        };
        let cloned = action.clone();
        assert_eq!(action.as_slice(), cloned.as_slice());

        let debug_str = format!("{:?}", action);
        assert!(debug_str.contains("values"));
    }

    #[test]
    fn test_continuous_action_clip_chaining() {
        let action = ContinuousActionTest {
            values: [2.0, -3.0, 0.5],
        };
        let clipped = action.clip(-2.0, 2.0).clip(-1.0, 1.0);
        assert_eq!(clipped.values, [1.0, -1.0, 0.5]);
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

        // Enumerate should produce all 256 actions
        let actions = LargeDiscreteAction::enumerate();
        assert_eq!(actions.len(), 256);

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

        let clipped = action.clip(-1.0, 1.0);
        assert_eq!(clipped.values, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_continuous_action_extreme_clip_bounds() {
        let action = ContinuousActionTest {
            values: [100.0, -100.0, 0.0],
        };

        let clipped = action.clip(f32::NEG_INFINITY, f32::INFINITY);
        assert_eq!(clipped.values, [100.0, -100.0, 0.0]);

        let clipped = action.clip(0.0, 0.0);
        assert_eq!(clipped.values, [0.0, 0.0, 0.0]);
    }

    // ========================================================================
    // BoundedAction Tests
    // ========================================================================

    #[test]
    fn test_bounded_action_low_strictly_below_high() {
        let low = ContinuousActionTest::low();
        let high = ContinuousActionTest::high();
        for i in 0..3 {
            assert!(low[i] < high[i], "bound {i}: low >= high");
        }
    }

    #[test]
    fn test_bounded_action_clip_is_noop_inside_bounds() {
        // Construct an action at the low/high bounds: clip(low, high) must
        // return the same components.
        let low = ContinuousActionTest::low();
        let high = ContinuousActionTest::high();
        let at_low = ContinuousActionTest::from_slice(&low);
        let at_high = ContinuousActionTest::from_slice(&high);
        assert_eq!(at_low.clip(low[0], high[0]).as_slice(), &low);
        assert_eq!(at_high.clip(low[0], high[0]).as_slice(), &high);
    }
}
