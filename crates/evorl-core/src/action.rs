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

use std::error::Error;
use std::fmt::Debug;

/// Base trait for all action types in reinforcement learning environments.
///
/// This trait defines the minimal interface that all actions must implement, regardless
/// of their underlying representation (discrete, continuous, or hybrid). It ensures actions
/// are debuggable, clonable, and can validate themselves.
///
/// # Design Rationale
///
/// The `Action` trait is intentionally minimal and framework-agnostic:
/// - `Debug`: Required for logging and debugging agents
/// - `Clone`: Actions may be stored in replay buffers or used multiple times
/// - `Sized`: Enables efficient stack allocation and compile-time optimization
/// - `is_valid()`: Allows runtime validation of action constraints
///
/// # Implementing Action
///
/// When implementing this trait, ensure `is_valid()` checks all constraints:
/// - Range bounds for numeric values
/// - Finiteness for floating-point values
/// - Structural invariants (e.g., array dimensions)
/// - Environment-specific rules (e.g., available moves in a game state)
pub trait Action<const D: usize>: Debug + Clone + Sized {
    /// The number of independent dimensions in this action space.
    ///
    /// This is automatically set to match the const generic parameter `D`.
    const DIM: usize = D;

    /// Returns the cardinality of each dimension in this action space.
    ///
    /// The returned array has length `D`, where each element specifies the number
    /// of possible values for that dimension. All values must be greater than zero.
    fn shape() -> [usize; D];

    /// Validates whether this action satisfies all constraints.
    ///
    /// This method checks if the action is legal according to its type's invariants.
    /// It does **not** check environment-specific legality (e.g., whether a move
    /// is valid in the current game state)—that's the environment's responsibility.
    ///
    /// # Returns
    ///
    /// Returns `true` if the action satisfies all structural constraints, `false` otherwise.
    fn is_valid(&self) -> bool;
}

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
        use rand::Rng;
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
/// can have a different cardinality, defined by [`action_space()`](MultiDiscreteAction::action_space).
///
/// The total number of action combinations is the product of all dimension sizes:
/// ```text
/// total_actions = ∏ action_space()[i]
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
    /// Each index must be in the range `[0, action_space()[i])` for dimension `i`.
    ///
    /// # Panics
    ///
    /// Implementations should panic if any index is out of bounds for its dimension.
    fn from_indices(indices: [usize; D]) -> Self;

    /// Converts this action to its multi-dimensional index representation.
    ///
    /// The returned array must satisfy: each element `i` is in `[0, action_space()[i])`.
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
        use rand::Rng;
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
        use rand::Rng;
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

// ----------------------------------------------------------------------------
/// Error indicating an action violated its type's constraints.
///
/// This error is returned when an action fails validation or when invalid
/// conversions are attempted (e.g., out-of-bounds indices, non-finite values).
///
/// # Examples
///
/// ```rust,ignore
/// use evorl_core::action::InvalidActionError;
///
/// fn validate_action(action: &GameAction) -> Result<(),
#[derive(Debug, Clone, PartialEq)]
pub struct InvalidActionError {
    pub message: String,
}

impl std::fmt::Display for InvalidActionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Invalid action: {}", self.message)
    }
}

impl Error for InvalidActionError {}
