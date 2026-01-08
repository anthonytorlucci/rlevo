//! Action space abstractions for reinforcement learning environments.
//!
//! This module provides a flexible type system for representing agent actions in RL environments.
//! Actions can be discrete (finite choices), multi-discrete (multiple independent discrete choices),
//! or continuous (real-valued vectors). The design keeps action representations framework-agnostic
//! while providing optional Burn tensor conversion.
//!
//! # Design Philosophy
//!
//! The action traits follow a layered design:
//! - [`Action`]: Base trait providing validation and cloning semantics
//! - [`DiscreteAction`], [`MultiDiscreteAction`], [`ContinuousAction`]: Type-specific extensions
//! - [`ActionTensorConvertible`]: Optional framework integration (Burn tensors)
//!
//! This separation allows action types to remain independent of deep learning frameworks while
//! still enabling efficient neural network integration when needed.
//!
//! # Action Types
//!
//! ## Discrete Actions
//!
//! Discrete actions represent a finite set of mutually exclusive choices (e.g., "move left",
//! "move right", "jump"). They are indexed from `0` to `ACTION_COUNT - 1`.
//!
//! ```rust,ignore
//! #[derive(Debug, Clone, Copy, PartialEq, Eq)]
//! enum GameAction {
//!     Left,
//!     Right,
//!     Jump,
//! }
//!
//! impl Action for GameAction {
//!     fn is_valid(&self) -> bool { true }
//! }
//!
//! impl DiscreteAction for GameAction {
//!     const ACTION_COUNT: usize = 3;
//!
//!     fn from_index(index: usize) -> Self {
//!         match index {
//!             0 => GameAction::Left,
//!             1 => GameAction::Right,
//!             2 => GameAction::Jump,
//!             _ => panic!("Invalid index"),
//!         }
//!     }
//!
//!     fn to_index(&self) -> usize {
//!         *self as usize
//!     }
//! }
//! ```
//!
//! ## Multi-Discrete Actions
//!
//! Multi-discrete actions consist of multiple independent discrete choices, such as selecting
//! both a direction and an attack type simultaneously.
//!
//! ```rust,ignore
//! #[derive(Debug, Clone)]
//! struct CombatAction {
//!     direction: u8,  // 0-3: North, East, South, West
//!     attack: u8,     // 0-2: Light, Heavy, Special
//! }
//!
//! impl Action for CombatAction {
//!     fn is_valid(&self) -> bool {
//!         self.direction < 4 && self.attack < 3
//!     }
//! }
//!
//! impl MultiDiscreteAction<2> for CombatAction {
//!     fn action_space() -> [usize; 2] {
//!         [4, 3]  // 4 directions × 3 attacks = 12 total combinations
//!     }
//!
//!     fn from_indices(indices: [usize; 2]) -> Self {
//!         Self { direction: indices[0] as u8, attack: indices[1] as u8 }
//!     }
//!
//!     fn to_indices(&self) -> [usize; 2] {
//!         [self.direction as usize, self.attack as usize]
//!     }
//! }
//! ```
//!
//! ## Continuous Actions
//!
//! Continuous actions are real-valued vectors, typically used for motor control or
//! parametrized actions (e.g., steering angle, throttle).
//!
//! ```rust,ignore
//! #[derive(Debug, Clone)]
//! struct RobotControl {
//!     joint_angles: [f32; 6],
//! }
//!
//! impl Action for RobotControl {
//!     fn is_valid(&self) -> bool {
//!         self.joint_angles.iter().all(|&x| x.is_finite())
//!     }
//! }
//!
//! impl ContinuousAction for RobotControl {
//!     const DIM: usize = 6;
//!
//!     fn as_slice(&self) -> &[f32] {
//!         &self.joint_angles
//!     }
//!
//!     fn clip(&self, min: f32, max: f32) -> Self {
//!         let mut clipped = self.clone();
//!         clipped.joint_angles.iter_mut().for_each(|x| *x = x.clamp(min, max));
//!         clipped
//!     }
//!
//!     fn from_slice(values: &[f32]) -> Self {
//!         let mut angles = [0.0; 6];
//!         angles.copy_from_slice(values);
//!         Self { joint_angles: angles }
//!     }
//! }
//! ```
//!
//! # Tensor Conversion
//!
//! The optional [`ActionTensorConvertible`] trait enables conversion to Burn tensors for
//! neural network processing. This separation keeps action types framework-agnostic.
//!
//! # Examples
//!
//! Generate random actions for exploration:
//!
//! ```rust,ignore
//! use evorl_core::action::{DiscreteAction, ContinuousAction};
//!
//! // Sample a random discrete action
//! let action = GameAction::random();
//!
//! // Sample a random continuous action (uniform distribution)
//! let control = RobotControl::random();
//! ```
//!
//! Enumerate all possible actions for tabular methods:
//!
//! ```rust,ignore
//! let all_actions = GameAction::enumerate();
//! assert_eq!(all_actions.len(), GameAction::ACTION_COUNT);
//! ```

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
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
/// # Examples
///
/// ```rust,ignore
/// use evorl_core::action::Action;
///
/// #[derive(Debug, Clone)]
/// struct MyAction {
///     value: i32,
/// }
///
/// impl Action for MyAction {
///     fn is_valid(&self) -> bool {
///         // Validate action constraints
///         self.value >= 0 && self.value <= 10
///     }
/// }
///
/// let action = MyAction { value: 5 };
/// assert!(action.is_valid());
///
/// let invalid = MyAction { value: -1 };
/// assert!(!invalid.is_valid());
/// ```
///
/// # Implementing Action
///
/// When implementing this trait, ensure `is_valid()` checks all constraints:
/// - Range bounds for numeric values
/// - Finiteness for floating-point values
/// - Structural invariants (e.g., array dimensions)
/// - Environment-specific rules (e.g., available moves in a game state)
pub trait Action: Debug + Clone + Sized {
    /// Validates whether this action satisfies all constraints.
    ///
    /// This method checks if the action is legal according to its type's invariants.
    /// It does **not** check environment-specific legality (e.g., whether a move
    /// is valid in the current game state)—that's the environment's responsibility.
    ///
    /// # Returns
    ///
    /// Returns `true` if the action satisfies all structural constraints, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let action = ContinuousAction::from_slice(&[0.5, 0.3]);
    /// assert!(action.is_valid());  // Values are finite
    ///
    /// let invalid = ContinuousAction::from_slice(&[f32::NAN, 0.0]);
    /// assert!(!invalid.is_valid());  // NaN is not valid
    /// ```
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
/// # Examples
///
/// ```rust,ignore
/// use evorl_core::action::{Action, DiscreteAction};
///
/// #[derive(Debug, Clone, Copy, PartialEq)]
/// enum Move { Up, Down, Left, Right }
///
/// impl Action for Move {
///     fn is_valid(&self) -> bool { true }
/// }
///
/// impl DiscreteAction for Move {
///     const ACTION_COUNT: usize = 4;
///
///     fn from_index(index: usize) -> Self {
///         match index {
///             0 => Move::Up,
///             1 => Move::Down,
///             2 => Move::Left,
///             3 => Move::Right,
///             _ => panic!("Index {} out of bounds", index),
///         }
///     }
///
///     fn to_index(&self) -> usize { *self as usize }
/// }
///
/// // Sample random actions
/// let action = Move::random();
///
/// // Enumerate all possibilities (useful for tabular RL)
/// let all_moves = Move::enumerate();
/// assert_eq!(all_moves.len(), 4);
/// ```
///
/// # Performance
///
/// For performance-critical code, prefer `from_index()` over `random()` when you
/// already have an index (e.g., from a neural network's argmax). The `random()`
/// method allocates a thread-local RNG on each call.
pub trait DiscreteAction: Action {
    /// The total number of distinct actions in this action space.
    ///
    /// This constant defines the cardinality of the action space. It must be
    /// greater than zero and remain constant for the lifetime of the program.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// enum GameAction { Jump, Duck, Left, Right }
    ///
    /// impl DiscreteAction for GameAction {
    ///     const ACTION_COUNT: usize = 4;
    ///     // ... other methods
    /// }
    /// ```
    const ACTION_COUNT: usize;

    /// Constructs an action from its zero-based index.
    ///
    /// This method must be the inverse of [`to_index()`](DiscreteAction::to_index).
    ///
    /// # Panics
    ///
    /// Implementations should panic if `index >= ACTION_COUNT`, as this indicates
    /// a programming error (out-of-bounds access).
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let action = GameAction::from_index(0);  // First action
    /// assert_eq!(action.to_index(), 0);
    /// ```
    fn from_index(index: usize) -> Self;

    /// Converts this action to its zero-based index.
    ///
    /// The returned index must be in the range `[0, ACTION_COUNT)` and must be
    /// the inverse of [`from_index()`](DiscreteAction::from_index).
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let action = GameAction::Jump;
    /// let idx = action.to_index();
    /// assert!(idx < GameAction::ACTION_COUNT);
    /// assert_eq!(GameAction::from_index(idx), action);
    /// ```
    fn to_index(&self) -> usize;

    /// Samples a uniformly random action from this action space.
    ///
    /// This is a convenience method for exploration in reinforcement learning.
    /// It uses thread-local RNG state, so it's safe to call from multiple threads
    /// but will produce different sequences per thread.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // Epsilon-greedy exploration
    /// use rand::Rng;
    ///
    /// let action = if rand::rng().gen_bool(epsilon) {
    ///     GameAction::random()  // Explore
    /// } else {
    ///     policy.select_action(state)  // Exploit
    /// };
    /// ```
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
    /// # Examples
    ///
    /// ```rust,ignore
    /// let actions = GameAction::enumerate();
    /// for (i, action) in actions.iter().enumerate() {
    ///     assert_eq!(action.to_index(), i);
    /// }
    /// ```
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
/// # Examples
///
/// ```rust,ignore
/// use evorl_core::action::{Action, MultiDiscreteAction};
///
/// #[derive(Debug, Clone)]
/// struct StrategyAction {
///     unit_id: u8,    // 0-9: which unit to control
///     action: u8,     // 0-4: move/attack/defend/heal/wait
///     direction: u8,  // 0-3: north/south/east/west
/// }
///
/// impl Action for StrategyAction {
///     fn is_valid(&self) -> bool {
///         self.unit_id < 10 && self.action < 5 && self.direction < 4
///     }
/// }
///
/// impl MultiDiscreteAction<3> for StrategyAction {
///     fn action_space() -> [usize; 3] {
///         [10, 5, 4]  // 10 units × 5 actions × 4 directions = 200 combinations
///     }
///
///     fn from_indices(indices: [usize; 3]) -> Self {
///         Self {
///             unit_id: indices[0] as u8,
///             action: indices[1] as u8,
///             direction: indices[2] as u8,
///         }
///     }
///
///     fn to_indices(&self) -> [usize; 3] {
///         [self.unit_id as usize, self.action as usize, self.direction as usize]
///     }
/// }
/// ```
///
/// # Caution: Combinatorial Explosion
///
/// Be careful with [`enumerate()`](MultiDiscreteAction::enumerate) on large action spaces.
/// A 3D action space with dimensions [10, 10, 10] produces 1000 actions, but
/// [100, 100, 100] produces 1,000,000!
pub trait MultiDiscreteAction<const D: usize>: Action {
    /// The number of independent dimensions in this action space.
    ///
    /// This is automatically set to match the const generic parameter `D`.
    /// Each dimension represents an independent categorical choice.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// impl MultiDiscreteAction<3> for MyAction {
    ///     // DIM is automatically 3
    ///     // ...
    /// }
    ///
    /// assert_eq!(MyAction::DIM, 3);
    /// ```
    const DIM: usize = D;

    /// Returns the cardinality of each dimension in this action space.
    ///
    /// The returned array has length `D`, where each element specifies the number
    /// of possible values for that dimension. All values must be greater than zero.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // Action with 4 directions and 3 intensities
    /// fn action_space() -> [usize; 2] {
    ///     [4, 3]  // Total: 4 × 3 = 12 action combinations
    /// }
    /// ```
    fn action_space() -> [usize; D];

    /// Constructs an action from multi-dimensional indices.
    ///
    /// Each index must be in the range `[0, action_space()[i])` for dimension `i`.
    ///
    /// # Panics
    ///
    /// Implementations should panic if any index is out of bounds for its dimension.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let action = StrategyAction::from_indices([2, 1, 3]);
    /// // unit_id=2, action=1, direction=3
    /// ```
    fn from_indices(indices: [usize; D]) -> Self;

    /// Converts this action to its multi-dimensional index representation.
    ///
    /// The returned array must satisfy: each element `i` is in `[0, action_space()[i])`.
    /// This method must be the inverse of [`from_indices()`](MultiDiscreteAction::from_indices).
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let action = StrategyAction::from_indices([1, 2, 0]);
    /// assert_eq!(action.to_indices(), [1, 2, 0]);
    /// ```
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
        let space = Self::action_space();
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
    /// # Examples
    ///
    /// ```rust,ignore
    /// let all_actions = SmallAction::enumerate();  // [2, 3] = 6 actions
    /// assert_eq!(all_actions.len(), 6);
    /// ```
    ///
    /// # Panics
    ///
    /// May panic or run out of memory if the action space is too large.
    fn enumerate() -> Vec<Self>
    where
        Self: Sized,
    {
        let space = Self::action_space();
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
/// # Examples
///
/// ```rust,ignore
/// use evorl_core::action::{Action, ContinuousAction};
///
/// #[derive(Debug, Clone)]
/// struct VehicleControl {
///     steering: f32,  // -1.0 (left) to 1.0 (right)
///     throttle: f32,  // 0.0 (idle) to 1.0 (full)
/// }
///
/// impl Action for VehicleControl {
///     fn is_valid(&self) -> bool {
///         self.steering.is_finite() &&
///         self.throttle.is_finite() &&
///         self.steering >= -1.0 && self.steering <= 1.0 &&
///         self.throttle >= 0.0 && self.throttle <= 1.0
///     }
/// }
///
/// impl ContinuousAction for VehicleControl {
///     const DIM: usize = 2;
///
///     fn as_slice(&self) -> &[f32] {
///         // Use unsafe to reinterpret struct as slice
///         unsafe { std::slice::from_raw_parts(self as *const _ as *const f32, 2) }
///     }
///
///     fn clip(&self, min: f32, max: f32) -> Self {
///         Self {
///             steering: self.steering.clamp(min, max),
///             throttle: self.throttle.clamp(min, max),
///         }
///     }
///
///     fn from_slice(values: &[f32]) -> Self {
///         Self {
///             steering: values[0],
///             throttle: values[1],
///         }
///     }
/// }
///
/// // Clip to valid range
/// let action = VehicleControl { steering: 1.5, throttle: -0.5 };
/// let clipped = action.clip(-1.0, 1.0);
/// assert_eq!(clipped.steering, 1.0);
/// assert_eq!(clipped.throttle, -1.0);
/// ```
///
/// # Neural Network Integration
///
/// Continuous actions are typically produced by neural networks with `tanh` or
/// `sigmoid` activation functions. Use [`clip()`](ContinuousAction::clip) to
/// ensure outputs stay within valid ranges.
pub trait ContinuousAction: Action {
    /// The number of real-valued components in this action.
    ///
    /// This defines the size of the continuous action vector. It must match
    /// the length of the slice returned by [`as_slice()`](ContinuousAction::as_slice).
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// struct Control { x: f32, y: f32, z: f32 }
    ///
    /// impl ContinuousAction for Control {
    ///     const DIM: usize = 3;  // 3D control vector
    ///     // ...
    /// }
    /// ```
    const DIM: usize;

    /// Returns a slice view of this action's component values.
    ///
    /// The returned slice must have exactly `DIM` elements. This is used for
    /// efficient serialization and tensor conversion.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let action = VehicleControl { steering: 0.5, throttle: 0.8 };
    /// let slice = action.as_slice();
    /// assert_eq!(slice.len(), VehicleControl::DIM);
    /// assert_eq!(slice, &[0.5, 0.8]);
    /// ```
    fn as_slice(&self) -> &[f32];

    /// Returns a new action with all components clipped to `[min, max]`.
    ///
    /// This is essential for ensuring neural network outputs (which may exceed
    /// valid ranges due to numerical errors or exploration noise) stay within
    /// acceptable bounds.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let action = VehicleControl { steering: 1.5, throttle: -0.2 };
    /// let safe = action.clip(-1.0, 1.0);
    /// assert_eq!(safe.steering, 1.0);   // Clipped to max
    /// assert_eq!(safe.throttle, -0.2);  // Within bounds
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // Random exploration
    /// let action = VehicleControl::random();
    /// assert!(action.as_slice().iter().all(|&x| x >= -1.0 && x <= 1.0));
    /// ```
    ///
    /// # Custom Sampling
    ///
    /// ```rust,ignore
    /// // Override for Gaussian sampling
    /// fn random() -> Self {
    ///     use rand_distr::{Normal, Distribution};
    ///     let normal = Normal::new(0.0, 0.5).unwrap();
    ///     let mut rng = rand::rng();
    ///     let values: Vec<f32> = (0..Self::DIM)
    ///         .map(|_| normal.sample(&mut rng) as f32)
    ///         .collect();
    ///     Self::from_slice(&values).clip(-1.0, 1.0)
    /// }
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let values = [0.5, 0.8];
    /// let action = VehicleControl::from_slice(&values);
    /// assert_eq!(action.as_slice(), &values);
    /// ```
    fn from_slice(values: &[f32]) -> Self;
}

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

/// Framework-Specific Conversion Trait
/// Separate trait for converting actions to tensors
/// Keeps core action traits framework-agnostic
pub trait ActionTensorConvertible<const R: usize> {
    // todo! investigate how to include batch_size
    fn to_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, R>;
}
