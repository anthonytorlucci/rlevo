//! State representation abstractions for reinforcement learning environments.
//!
//! This module provides a flexible, type-safe framework for representing environmental states
//! in reinforcement learning systems. It defines a hierarchy of traits that enable different
//! representations and transformations of state data while maintaining framework independence.
//!
//! # Design Philosophy
//!
//! The state traits follow a layered design principle:
//!
//! 1. **Core Abstraction** ([`State`]) - Minimal, framework-agnostic foundation for all states
//! 2. **Numeric Representation** ([`FlattenedState`]) - States that can be efficiently vectorized
//! 3. **Temporal/Sequential** ([`TemporalState`]) - States with historical or time-series data
//! 4. **Framework Integration** ([`StateTensorConvertible`], [`StateIntTensorConvertible`]) -
//!    Bridge to deep learning frameworks like Burn
//!
//! This separation ensures that core state logic remains independent of any particular ML framework,
//! while still providing seamless integration when needed.
//!
//! # State Hierarchy
//!
//! ```text
//! State (base trait)
//!   ├─ FlattenedState (numeric/vectorizable states)
//!   │    └─ TemporalState (time-series states)
//!   │
//!   └─ StateTensorConvertible (framework integration)
//!        └─ StateIntTensorConvertible (discrete state tensors)
//! ```
//!
//! # Framework Integration
//!
//! States can be converted to Burn tensors for neural network processing.
//!
//! # Error Handling
//!
//! State operations that can fail return [`StateError`], which provides detailed information
//! about shape mismatches, size mismatches, or invalid data. All errors implement
//! `std::error::Error` for seamless integration with error handling libraries.
//!
//! # Thread Safety
//!
//! All state types must implement `Clone`, allowing them to be safely shared across threads
//! when combined with appropriate synchronization primitives. States are typically small
//! and cheap to clone, making them suitable for concurrent RL algorithms.

use burn::tensor::backend::Backend;
use burn::tensor::Int;
use burn::tensor::Tensor;
use std::fmt::Debug;
use std::hash::Hash;

/// Core trait for all state types in reinforcement learning environments.
///
/// `State` provides the minimal, framework-agnostic foundation that all environment states
/// must implement. It defines the essential operations needed to represent, validate, and
/// introspect state data without imposing any specific representation or framework dependencies.
///
/// # Design Goals
///
/// - **Framework Independence**: No dependencies on DL frameworks (Burn)
/// - **Type Safety**: Leverage Rust's type system for compile-time guarantees
/// - **Composability**: Serve as a building block for more specialized state traits
/// - **Minimal Overhead**: No required allocations or complex operations
///
/// # Required Trait Bounds
///
/// All state implementations must be:
/// - `Debug`: For debugging and logging
/// - `Clone`: For creating independent copies (common in RL algorithms)
/// - `PartialEq` + `Eq`: For state comparison and deduplication
/// - `Hash`: For use in hash-based data structures (e.g., state visitation counts)
///
/// # Implementation Guidelines
///
/// When implementing `State`:
///
/// 1. **Validation** ([`is_valid`](State::is_valid)): Override if your state has constraints
///    (e.g., grid positions must be within bounds, probabilities must sum to 1.0)
///
/// 2. **Size Consistency** ([`numel`](State::numel), [`shape`](State::shape)): Ensure
///    `numel()` equals the product of dimensions in `shape()`
///
/// 3. **Determinism**: State representations should be deterministic and consistent
///    across program runs for reproducibility
///
/// # Examples
///
/// ## Simple Discrete State
///
/// ```rust
/// use evorl_core::state::State;
///
/// #[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// enum GameState {
///     Menu,
///     Playing { level: u8 },
///     GameOver { score: u32 },
/// }
///
/// impl State for GameState {
///     fn is_valid(&self) -> bool {
///         match self {
///             GameState::Playing { level } => *level > 0 && *level <= 10,
///             _ => true,
///         }
///     }
///
///     fn numel(&self) -> usize {
///         // Encode as 3 features: [state_id, level, score]
///         3
///     }
///
///     fn shape(&self) -> Vec<usize> {
///         vec![3]
///     }
/// }
/// ```
///
/// ## Continuous State with Constraints
///
/// ```rust
/// use evorl_core::state::State;
///
/// #[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// struct RobotPose {
///     x_mm: i32,      // Position in millimeters
///     y_mm: i32,
///     theta_mdeg: i32, // Orientation in millidegrees
/// }
///
/// impl State for RobotPose {
///     fn is_valid(&self) -> bool {
///         // Workspace bounds: 0-1000mm
///         self.x_mm >= 0 && self.x_mm <= 1000 &&
///         self.y_mm >= 0 && self.y_mm <= 1000 &&
///         // Orientation: -180 to 180 degrees
///         self.theta_mdeg >= -180_000 && self.theta_mdeg <= 180_000
///     }
///
///     fn numel(&self) -> usize {
///         3 // x, y, theta
///     }
///
///     fn shape(&self) -> Vec<usize> {
///         vec![3] // Single flat vector
///     }
/// }
/// ```
///
/// # See Also
///
/// - [`FlattenedState`]: For states that can be efficiently vectorized
/// - [`TemporalState`]: For states with sequential/temporal structure
/// - [`StateTensorConvertible`]: For integration with neural network frameworks
pub trait State: Debug + Clone + PartialEq + Eq + Hash {
    /// Validates that this state satisfies all environment constraints.
    ///
    /// The default implementation returns `true`, assuming all representable states are valid.
    /// Override this method if your state has domain-specific constraints (e.g., bounds checks,
    /// probability normalization, structural invariants).
    ///
    /// # Validation Use Cases
    ///
    /// - **Bounds Checking**: Grid positions within map boundaries
    /// - **Physical Constraints**: Robot joints within angle limits
    /// - **Probability Distributions**: Action probabilities sum to 1.0
    /// - **Structural Invariants**: Game states follow legal transition rules
    ///
    /// # Performance Considerations
    ///
    /// This method may be called frequently during environment stepping. Keep validation
    /// logic lightweight. For expensive checks, consider:
    /// - Caching validation results
    /// - Using debug assertions (`debug_assert!`) for development
    /// - Validating only on state construction
    ///
    /// # Examples
    ///
    /// ```rust
    /// use evorl_core::state::State;
    ///
    /// #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    /// struct BoundedPosition {
    ///     x: i32,
    ///     y: i32,
    ///     max_x: i32,
    ///     max_y: i32,
    /// }
    ///
    /// impl State for BoundedPosition {
    ///     fn is_valid(&self) -> bool {
    ///         self.x >= 0 && self.x < self.max_x &&
    ///         self.y >= 0 && self.y < self.max_y
    ///     }
    ///
    ///     fn numel(&self) -> usize { 2 }
    ///     fn shape(&self) -> Vec<usize> { vec![2] }
    /// }
    ///
    /// let valid = BoundedPosition { x: 5, y: 3, max_x: 10, max_y: 10 };
    /// assert!(valid.is_valid());
    ///
    /// let invalid = BoundedPosition { x: 15, y: 3, max_x: 10, max_y: 10 };
    /// assert!(!invalid.is_valid());
    /// ```
    ///
    /// # Returns
    ///
    /// `true` if the state satisfies all constraints, `false` otherwise.
    fn is_valid(&self) -> bool {
        true
    }

    /// Returns the total number of scalar elements in this state's representation.
    ///
    /// This value is critical for:
    /// - Allocating buffers for state serialization
    /// - Determining neural network input layer dimensions
    /// - Validating state transformations (e.g., flattening/unflattening)
    ///
    /// # Relationship to Shape
    ///
    /// For consistency, `numel()` must equal the product of all dimensions returned by
    /// [`shape()`](State::shape):
    ///
    /// ```text
    /// numel() == shape().iter().product()
    /// ```
    ///
    /// # Examples
    ///
    /// ```rust
    /// use evorl_core::state::State;
    ///
    /// #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    /// struct Image {
    ///     width: usize,
    ///     height: usize,
    ///     channels: usize,
    /// }
    ///
    /// impl State for Image {
    ///     fn numel(&self) -> usize {
    ///         self.width * self.height * self.channels
    ///     }
    ///
    ///     fn shape(&self) -> Vec<usize> {
    ///         vec![self.height, self.width, self.channels]
    ///     }
    /// }
    ///
    /// let rgb_image = Image { width: 64, height: 64, channels: 3 };
    /// assert_eq!(rgb_image.numel(), 64 * 64 * 3); // 12,288 pixels
    /// assert_eq!(rgb_image.numel(), rgb_image.shape().iter().product());
    /// ```
    ///
    /// # Returns
    ///
    /// The total number of scalar elements needed to represent this state.
    fn numel(&self) -> usize;

    /// Returns the logical shape (dimensions) of this state's tensor representation.
    ///
    /// The shape describes how the state's scalar elements are organized logically,
    /// independent of any framework-specific tensor format. This is particularly useful
    /// for neural network architectures that expect specific input shapes.
    ///
    /// # Shape Conventions
    ///
    /// - **Scalar States**: `vec![1]` (single value)
    /// - **Vector States**: `vec![n]` (flat array of n elements)
    /// - **Matrix States**: `vec![rows, cols]` (2D grid)
    /// - **Image States**: `vec![height, width, channels]` or `vec![channels, height, width]`
    /// - **Sequential States**: `vec![sequence_length, features]` (time series)
    ///
    /// # Batch Dimensions
    ///
    /// This method returns the shape of a **single** state instance, excluding batch dimensions.
    /// When converting to tensors via [`StateTensorConvertible`], the batch dimension is
    /// typically added automatically by the framework integration layer.
    ///
    /// # Consistency Requirements
    ///
    /// The product of all dimensions must equal [`numel()`](State::numel):
    ///
    /// ```text
    /// shape().iter().product::<usize>() == numel()
    /// ```
    ///
    /// # Examples
    ///
    /// ```rust
    /// use evorl_core::state::State;
    ///
    /// #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    /// struct SensorReadings {
    ///     temperature: i32,
    ///     humidity: i32,
    ///     pressure: i32,
    /// }
    ///
    /// impl State for SensorReadings {
    ///     fn numel(&self) -> usize { 3 }
    ///
    ///     fn shape(&self) -> Vec<usize> {
    ///         vec![3] // Simple 1D vector
    ///     }
    /// }
    /// ```
    ///
    /// # Returns
    ///
    /// A vector describing the logical dimensions of this state, where each element
    /// represents the size of that dimension. The product of all elements equals `numel()`.
    fn shape(&self) -> Vec<usize>;
}

/// Trait for states that can be efficiently represented as contiguous numeric arrays.
///
/// `FlattenedState` extends [`State`] to support conversion to and from flat `f32` vectors,
/// enabling seamless integration with neural networks and numeric computation libraries.
/// This is essential for most deep reinforcement learning algorithms, where states must be
/// fed into neural networks as fixed-size vectors.
///
/// # Design Rationale
///
/// - **Neural Network Compatibility**: Most RL algorithms (DQN, PPO, A3C) require states
///   as flat numeric vectors for input to neural networks
/// - **Efficiency**: Flat representations enable cache-friendly memory access patterns
/// - **Interoperability**: Standard `f32` vectors work across frameworks and serialization formats
/// - **Type Safety**: Bidirectional conversion with error checking prevents data corruption
///
/// # Float Representation
///
/// This trait uses `f32` (32-bit floating point) as the canonical numeric type because:
/// - Most ML frameworks default to `f32` for efficiency and memory usage
/// - Sufficient precision for most RL state representations
/// - Compatible with GPU computation
///
/// For states with discrete values (e.g., enums, small integers), encode them as `f32`:
/// - One-hot encoding for categorical features
/// - Normalized integer values for ordinal features
/// - Consider [`StateIntTensorConvertible`] for truly discrete states
///
/// # Flattening Order
///
/// The order of elements in `flatten()` must be:
/// - **Deterministic**: Same state always produces same ordering
/// - **Consistent**: Order matches what `from_flattened()` expects
/// - **Documented**: Clarify element ordering in implementation docs
///
/// # Implementation Guidelines
///
/// 1. **Preserve Information**: Flattening should be lossless or explicitly document any precision loss
/// 2. **Size Consistency**: `flatten().len()` must equal `numel()`
/// 3. **Normalization**: Consider normalizing values to a standard range (e.g., [0, 1] or [-1, 1])
/// 4. **Error Handling**: `from_flattened()` should validate input size and data constraints
///
/// # Examples
///
/// ## Simple State
///
/// ```rust
/// use evorl_core::state::{State, FlattenedState, StateError};
///
/// #[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// struct Position2D {
///     x: i32,
///     y: i32,
/// }
///
/// impl State for Position2D {
///     fn numel(&self) -> usize { 2 }
///     fn shape(&self) -> Vec<usize> { vec![2] }
/// }
///
/// impl FlattenedState for Position2D {
///     fn flatten(&self) -> Vec<f32> {
///         // Simple conversion to f32
///         vec![self.x as f32, self.y as f32]
///     }
///
///     fn from_flattened(data: Vec<f32>) -> Result<Self, StateError> {
///         if data.len() != 2 {
///             return Err(StateError::InvalidSize {
///                 expected: 2,
///                 got: data.len(),
///             });
///         }
///         Ok(Position2D {
///             x: data[0] as i32,
///             y: data[1] as i32,
///         })
///     }
/// }
/// ```
///
/// ## Normalized Complex State
///
/// ```rust
/// use evorl_core::state::{State, FlattenedState, StateError};
///
/// #[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// struct PlayerState {
///     health: u8,      // 0-100
///     stamina: u8,     // 0-100
///     level: u8,       // 1-50
///     has_sword: bool,
///     has_shield: bool,
/// }
///
/// impl State for PlayerState {
///     fn numel(&self) -> usize { 5 }
///     fn shape(&self) -> Vec<usize> { vec![5] }
/// }
///
/// impl FlattenedState for PlayerState {
///     fn flatten(&self) -> Vec<f32> {
///         vec![
///             self.health as f32 / 100.0,      // Normalize to [0, 1]
///             self.stamina as f32 / 100.0,     // Normalize to [0, 1]
///             (self.level - 1) as f32 / 49.0,  // Normalize to [0, 1]
///             if self.has_sword { 1.0 } else { 0.0 },
///             if self.has_shield { 1.0 } else { 0.0 },
///         ]
///     }
///
///     fn from_flattened(data: Vec<f32>) -> Result<Self, StateError> {
///         if data.len() != 5 {
///             return Err(StateError::InvalidSize {
///                 expected: 5,
///                 got: data.len(),
///             });
///         }
///         Ok(PlayerState {
///             health: (data[0] * 100.0).clamp(0.0, 100.0) as u8,
///             stamina: (data[1] * 100.0).clamp(0.0, 100.0) as u8,
///             level: ((data[2] * 49.0) + 1.0).clamp(1.0, 50.0) as u8,
///             has_sword: data[3] > 0.5,
///             has_shield: data[4] > 0.5,
///         })
///     }
/// }
/// ```
///
/// # Performance Considerations
///
/// - `flatten()` allocates a new `Vec`; consider caching if called frequently
/// - For large states (e.g., images), consider lazy evaluation or memory pooling
/// - Round-trip conversion (`state -> flatten -> from_flattened`) should ideally be lossless
///
/// # See Also
///
/// - [`StateTensorConvertible`]: For direct conversion to framework tensors
/// - [`TemporalState`]: For states with sequential structure
pub trait FlattenedState: State {
    /// Converts this state into a contiguous flat vector of `f32` values.
    ///
    /// This method serializes the state into a format suitable for neural network processing,
    /// numeric computation, or data transmission. The flattened representation must preserve
    /// all information needed to reconstruct the state via [`from_flattened()`](FlattenedState::from_flattened).
    ///
    /// # Guarantees
    ///
    /// - **Length**: The returned vector's length must equal [`numel()`](State::numel)
    /// - **Determinism**: Same state always produces identical vector (same values, same order)
    /// - **Consistency**: Order matches what `from_flattened()` expects
    ///
    /// # Element Ordering
    ///
    /// The ordering of elements is implementation-defined but must be:
    /// 1. **Row-major** for multi-dimensional states (unless explicitly documented otherwise)
    /// 2. **Logical** (e.g., [x, y, z] for 3D positions, not [z, x, y])
    /// 3. **Consistent** across all instances of the state type
    ///
    /// # Normalization
    ///
    /// Consider normalizing values for better neural network training:
    /// - **Continuous values**: Scale to [0, 1] or [-1, 1]
    /// - **Discrete categories**: Use one-hot encoding
    /// - **Boolean flags**: Use 0.0 or 1.0
    /// - **Document** the normalization scheme in implementation docs
    ///
    /// # Performance
    ///
    /// This method allocates a new `Vec<f32>`. For performance-critical code:
    /// - Cache flattened representations if the state is read-only
    /// - Use object pooling for temporary buffers
    /// - Consider implementing `flatten_into(&mut [f32])` as an optimization
    ///
    /// # Examples
    ///
    /// ```rust
    /// use evorl_core::state::{State, FlattenedState, StateError};
    ///
    /// #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    /// struct RGB {
    ///     r: u8,
    ///     g: u8,
    ///     b: u8,
    /// }
    ///
    /// impl State for RGB {
    ///     fn numel(&self) -> usize { 3 }
    ///     fn shape(&self) -> Vec<usize> { vec![3] }
    /// }
    ///
    /// impl FlattenedState for RGB {
    ///     fn flatten(&self) -> Vec<f32> {
    ///         // Normalize to [0, 1]
    ///         vec![
    ///             self.r as f32 / 255.0,
    ///             self.g as f32 / 255.0,
    ///             self.b as f32 / 255.0,
    ///         ]
    ///     }
    ///
    ///     fn from_flattened(data: Vec<f32>) -> Result<Self, StateError> {
    ///         if data.len() != 3 {
    ///             return Err(StateError::InvalidSize {
    ///                 expected: 3,
    ///                 got: data.len(),
    ///             });
    ///         }
    ///         Ok(RGB {
    ///             r: (data[0] * 255.0).round() as u8,
    ///             g: (data[1] * 255.0).round() as u8,
    ///             b: (data[2] * 255.0).round() as u8,
    ///         })
    ///     }
    /// }
    ///
    /// let color = RGB { r: 128, g: 64, b: 255 };
    /// let flat = color.flatten();
    /// assert_eq!(flat.len(), 3);
    /// assert!((flat[0] - 0.502).abs() < 0.01); // 128/255 ≈ 0.502
    /// ```
    ///
    /// # Returns
    ///
    /// A vector of `f32` values representing this state in flattened form.
    /// The length always equals `self.numel()`.
    fn flatten(&self) -> Vec<f32>;

    /// Reconstructs a state from a flattened vector representation.
    ///
    /// This method is the inverse of [`flatten()`](FlattenedState::flatten), deserializing
    /// a flat `f32` vector back into a structured state. It must validate the input data
    /// and return an error if the vector is malformed or contains invalid values.
    ///
    /// # Input Validation
    ///
    /// Implementations must check:
    /// 1. **Size**: `data.len()` must equal the expected [`numel()`](State::numel)
    /// 2. **Value Ranges**: Elements must be within valid bounds for their fields
    /// 3. **Structural Constraints**: Reconstructed state must satisfy [`is_valid()`](State::is_valid)
    ///
    /// # Error Handling
    ///
    /// Return [`StateError`] for:
    /// - **Size Mismatch**: Use [`StateError::InvalidSize`] when `data.len()` is wrong
    /// - **Invalid Values**: Use [`StateError::InvalidData`] for out-of-range or NaN values
    /// - **Shape Mismatch**: Use [`StateError::InvalidShape`] for multi-dimensional states
    ///
    /// # Precision Loss
    ///
    /// Be aware of potential precision loss during round-trip conversion:
    /// - Integer fields may lose precision if values exceed `f32` mantissa bits (~24 bits)
    /// - Consider clamping values to valid ranges rather than rejecting borderline cases
    /// - Document any precision guarantees in implementation docs
    ///
    /// # Examples
    ///
    /// ## Basic Reconstruction
    ///
    /// ```rust
    /// use evorl_core::state::{State, FlattenedState, StateError};
    ///
    /// #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    /// struct Point3D {
    ///     x: i32,
    ///     y: i32,
    ///     z: i32,
    /// }
    ///
    /// impl State for Point3D {
    ///     fn numel(&self) -> usize { 3 }
    ///     fn shape(&self) -> Vec<usize> { vec![3] }
    /// }
    ///
    /// impl FlattenedState for Point3D {
    ///     fn flatten(&self) -> Vec<f32> {
    ///         vec![self.x as f32, self.y as f32, self.z as f32]
    ///     }
    ///
    ///     fn from_flattened(data: Vec<f32>) -> Result<Self, StateError> {
    ///         if data.len() != 3 {
    ///             return Err(StateError::InvalidSize {
    ///                 expected: 3,
    ///                 got: data.len(),
    ///             });
    ///         }
    ///         Ok(Point3D {
    ///             x: data[0] as i32,
    ///             y: data[1] as i32,
    ///             z: data[2] as i32,
    ///         })
    ///     }
    /// }
    ///
    /// // Successful reconstruction
    /// let data = vec![10.0, 20.0, 30.0];
    /// let point = Point3D::from_flattened(data).unwrap();
    /// assert_eq!(point, Point3D { x: 10, y: 20, z: 30 });
    ///
    /// // Size validation
    /// let bad_data = vec![10.0, 20.0]; // Missing z coordinate
    /// assert!(Point3D::from_flattened(bad_data).is_err());
    /// ```
    ///
    /// ## With Value Validation
    ///
    /// ```rust
    /// use evorl_core::state::{State, FlattenedState, StateError};
    ///
    /// #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    /// struct Percentage {
    ///     value: u8, // Must be 0-100
    /// }
    ///
    /// impl State for Percentage {
    ///     fn is_valid(&self) -> bool {
    ///         self.value <= 100
    ///     }
    ///
    ///     fn numel(&self) -> usize { 1 }
    ///     fn shape(&self) -> Vec<usize> { vec![1] }
    /// }
    ///
    /// impl FlattenedState for Percentage {
    ///     fn flatten(&self) -> Vec<f32> {
    ///         vec![self.value as f32 / 100.0] // Normalized to [0, 1]
    ///     }
    ///
    ///     fn from_flattened(data: Vec<f32>) -> Result<Self, StateError> {
    ///         if data.len() != 1 {
    ///             return Err(StateError::InvalidSize {
    ///                 expected: 1,
    ///                 got: data.len(),
    ///             });
    ///         }
    ///
    ///         let val = data[0];
    ///         if !val.is_finite() || val < 0.0 || val > 1.0 {
    ///             return Err(StateError::InvalidData(
    ///                 format!("Value must be in [0, 1], got {}", val)
    ///             ));
    ///         }
    ///
    ///         Ok(Percentage {
    ///             value: (val * 100.0).round() as u8,
    ///         })
    ///     }
    /// }
    /// ```
    ///
    /// # Parameters
    ///
    /// - `data`: A flat vector of `f32` values representing the state. Must have length
    ///   equal to `numel()` and contain valid values for the state type.
    ///
    /// # Returns
    ///
    /// - `Ok(Self)`: Successfully reconstructed state
    /// - `Err(StateError)`: Input data is invalid (wrong size, out-of-range values, etc.)
    ///
    /// # See Also
    ///
    /// - [`flatten()`](FlattenedState::flatten): The inverse operation
    /// - [`StateError`]: Error types for detailed failure information
    fn from_flattened(data: Vec<f32>) -> Result<Self, StateError>;
}

/// Trait for sequential/temporal states with historical observations.
///
/// `TemporalState` extends [`State`] and [`FlattenedState`] to support states that represent
/// time-series data or sequences of observations. This is essential for environments where:
/// - Past observations influence future decisions (e.g., velocity from position history)
/// - Partial observability requires memory (e.g., hidden state inference)
/// - Recurrent networks need sequential input (e.g., LSTM, GRU)
///
/// # Use Cases
///
/// ## 1. Frame Stacking
/// Atari games and similar pixel-based environments often stack the last N frames to
/// provide motion information that a single frame cannot capture:
///
/// ```text
/// [frame_t-3, frame_t-2, frame_t-1, frame_t] → Agent sees motion
/// ```
///
/// ## 2. Sensor History
/// Robot control with noisy sensors benefits from recent history to filter noise:
///
/// ```text
/// [gyro_t-9, gyro_t-8, ..., gyro_t] → Smoother orientation estimate
/// ```
///
/// ## 3. Market State
/// Trading environments track price history for trend analysis:
///
/// ```text
/// [price_t-29, price_t-28, ..., price_t] → 30-minute price window
/// ```
///
/// # Design Pattern
///
/// Temporal states typically use a fixed-size sliding window (ring buffer) pattern:
/// 1. Maintain a fixed number of historical observations (`sequence_length`)
/// 2. Add new observations with [`push_pop()`](TemporalState::push_pop)
/// 3. Automatically discard oldest observation when buffer is full
/// 4. Provide efficient access to recent data via [`latest()`](TemporalState::latest)
///
/// # Memory Efficiency
///
/// For large sequential states (e.g., stacked images):
/// - Consider using circular buffers to avoid allocations on every step
/// - Store observations in row-major or column-major order for cache efficiency
/// - Use lazy evaluation for derived features (e.g., velocity computed on-demand)
///
/// # Examples
///
/// ## Position History for Velocity Estimation
///
/// ```rust
/// use evorl_core::state::{State, FlattenedState, TemporalState, StateError};
///
/// #[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// struct PositionHistory {
///     positions: Vec<i32>, // Last N x-coordinates
///     window_size: usize,
/// }
///
/// impl PositionHistory {
///     fn new(window_size: usize) -> Self {
///         Self {
///             positions: vec![0; window_size],
///             window_size,
///         }
///     }
/// }
///
/// impl State for PositionHistory {
///     fn numel(&self) -> usize {
///         self.window_size
///     }
///
///     fn shape(&self) -> Vec<usize> {
///         vec![self.window_size]
///     }
/// }
///
/// impl FlattenedState for PositionHistory {
///     fn flatten(&self) -> Vec<f32> {
///         self.positions.iter().map(|&p| p as f32).collect()
///     }
///
///     fn from_flattened(data: Vec<f32>) -> Result<Self, StateError> {
///         Ok(PositionHistory {
///             window_size: data.len(),
///             positions: data.iter().map(|&p| p as i32).collect(),
///         })
///     }
/// }
///
/// impl TemporalState for PositionHistory {
///     fn sequence_length(&self) -> usize {
///         self.window_size
///     }
///
///     fn latest(&self) -> &[f32] {
///         // In practice, maintain a separate f32 buffer or convert on-demand
///         &[] // Simplified for example
///     }
///
///     fn push_pop(&self, new_observation: &[f32]) -> Result<Self, StateError> {
///         if new_observation.len() != 1 {
///             return Err(StateError::InvalidSize {
///                 expected: 1,
///                 got: new_observation.len(),
///             });
///         }
///
///         let mut new_positions = self.positions.clone();
///         new_positions.remove(0); // Drop oldest
///         new_positions.push(new_observation[0] as i32); // Add newest
///
///         Ok(PositionHistory {
///             positions: new_positions,
///             window_size: self.window_size,
///         })
///     }
/// }
/// ```
///
/// # Performance Considerations
///
/// - `push_pop()` may be called every environment step; optimize for common case
/// - Consider preallocating buffers and using `Vec::rotate_left()` + assignment
/// - For read-heavy workloads, cache the `flatten()` result until state changes
///
/// # See Also
///
/// - [`State`]: Base trait for all states
/// - [`FlattenedState`]: Required supertrait for numeric conversion
pub trait TemporalState: State {
    /// Returns the number of temporal observations stored in this state.
    ///
    /// This represents the "depth" or "window size" of the historical data. For example:
    /// - Frame stacking with 4 frames: `sequence_length() == 4`
    /// - 30-second sensor history at 10 Hz: `sequence_length() == 300`
    ///
    /// # Relationship to Shape
    ///
    /// For temporal states with shape `[seq_len, features]`, this typically returns `seq_len`.
    /// However, the relationship depends on how you organize your data:
    ///
    /// ```text
    /// // Time-major: sequence_length() == shape()[0]
    /// shape: [10, 3] → 10 timesteps of 3 features each
    ///
    /// // Feature-major: sequence_length() == shape()[1]
    /// shape: [3, 10] → 3 features with 10 historical values each
    /// ```
    ///
    /// # Invariants
    ///
    /// - Must be constant for a given state type's configuration
    /// - Must be > 0 (empty sequences are not valid temporal states)
    /// - Should not change during the lifetime of a state instance
    ///
    /// # Examples
    ///
    /// ```rust
    /// use evorl_core::state::{State, TemporalState};
    ///
    /// #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    /// struct FrameStack {
    ///     frames: Vec<Vec<u8>>, // 4 frames of 84x84 pixels
    /// }
    ///
    /// impl State for FrameStack {
    ///     fn numel(&self) -> usize {
    ///         4 * 84 * 84 // 4 frames, each 84x84
    ///     }
    ///
    ///     fn shape(&self) -> Vec<usize> {
    ///         vec![4, 84, 84] // [frames, height, width]
    ///     }
    /// }
    ///
    /// impl TemporalState for FrameStack {
    ///     fn sequence_length(&self) -> usize {
    ///         4 // Always stack 4 frames
    ///     }
    ///     // ... other methods
    /// #   fn latest(&self) -> &[f32] { &[] }
    /// #   fn push_pop(&self, _: &[f32]) -> Result<Self, evorl_core::state::StateError> { Ok(self.clone()) }
    /// }
    /// ```
    ///
    /// # Returns
    ///
    /// The number of temporal observations (sequence length) in this state.
    fn sequence_length(&self) -> usize;

    /// Returns a reference to the most recent observation(s) in the sequence.
    ///
    /// This provides efficient access to the latest data without reconstructing or flattening
    /// the entire state. Useful for:
    /// - Incremental processing (e.g., checking if latest frame is terminal)
    /// - Debugging (e.g., visualizing current observation)
    /// - Derived computations (e.g., calculating instantaneous velocity)
    ///
    /// # Return Format
    ///
    /// The returned slice should contain the raw values of the most recent observation,
    /// typically in the same format as a single element from `flatten()`. For example:
    ///
    /// ```text
    /// // If state is [obs_t-2, obs_t-1, obs_t] where each obs has 3 features:
    /// flatten() → [a, b, c, d, e, f, g, h, i] (9 elements)
    /// latest()  → [g, h, i] (3 elements, just obs_t)
    /// ```
    ///
    /// # Lifetime and Validity
    ///
    /// The returned reference is valid for the lifetime of `self`. It becomes invalid
    /// if the state is modified (e.g., via `push_pop()`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use evorl_core::state::{State, FlattenedState, TemporalState, StateError};
    ///
    /// #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    /// struct SensorBuffer {
    ///     readings: Vec<f32>, // Flattened: [reading_t-2, reading_t-1, reading_t]
    ///     obs_size: usize,    // Size of each reading
    /// }
    ///
    /// impl State for SensorBuffer {
    ///     fn numel(&self) -> usize {
    ///         self.readings.len()
    ///     }
    ///
    ///     fn shape(&self) -> Vec<usize> {
    ///         vec![self.readings.len() / self.obs_size, self.obs_size]
    ///     }
    /// }
    ///
    /// impl FlattenedState for SensorBuffer {
    ///     fn flatten(&self) -> Vec<f32> {
    ///         self.readings.clone()
    ///     }
    ///
    ///     fn from_flattened(data: Vec<f32>) -> Result<Self, StateError> {
    ///         Ok(SensorBuffer {
    ///             obs_size: 3, // Assume 3 sensors
    ///             readings: data,
    ///         })
    ///     }
    /// }
    ///
    /// impl TemporalState for SensorBuffer {
    ///     fn sequence_length(&self) -> usize {
    ///         self.readings.len() / self.obs_size
    ///     }
    ///
    ///     fn latest(&self) -> &[f32] {
    ///         let start = self.readings.len() - self.obs_size;
    ///         &self.readings[start..]
    ///     }
    ///
    ///     fn push_pop(&self, new_observation: &[f32]) -> Result<Self, StateError> {
    ///         if new_observation.len() != self.obs_size {
    ///             return Err(StateError::InvalidSize {
    ///                 expected: self.obs_size,
    ///                 got: new_observation.len(),
    ///             });
    ///         }
    ///
    ///         let mut new_readings = self.readings[self.obs_size..].to_vec();
    ///         new_readings.extend_from_slice(new_observation);
    ///
    ///         Ok(SensorBuffer {
    ///             readings: new_readings,
    ///             obs_size: self.obs_size,
    ///         })
    ///     }
    /// }
    ///
    /// let state = SensorBuffer {
    ///     readings: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    ///     obs_size: 3,
    /// };
    ///
    /// // Get just the latest reading
    /// assert_eq!(state.latest(), &[7.0, 8.0, 9.0]);
    /// ```
    ///
    /// # Returns
    ///
    /// A slice containing the raw values of the most recent observation in the sequence.
    fn latest(&self) -> &[f32];

    /// Creates a new state by appending a new observation and removing the oldest.
    ///
    /// This implements a sliding window pattern: new observations shift in from the right
    /// while old observations are discarded from the left, maintaining a constant
    /// [`sequence_length()`](TemporalState::sequence_length).
    ///
    /// # Operation
    ///
    /// ```text
    /// Before: [obs_t-3, obs_t-2, obs_t-1, obs_t]
    ///                   ↓        ↓        ↓       + new_obs
    /// After:  [obs_t-2, obs_t-1, obs_t, new_obs]
    /// ```
    ///
    /// # Immutability
    ///
    /// This method returns a **new** state rather than modifying `self` in place,
    /// following Rust's ownership and functional programming principles. This ensures:
    /// - Thread safety (original state unaffected)
    /// - Easier debugging (state history preserved if needed)
    /// - Composability with immutable algorithms
    ///
    /// For performance-critical code with large states, consider:
    /// - Using `Rc<RefCell<State>>` or similar for interior mutability
    /// - Implementing an internal circular buffer that reuses memory
    ///
    /// # Validation
    ///
    /// Implementations must validate:
    /// 1. **Size**: `new_observation.len()` matches the size of one observation
    /// 2. **Values**: Elements are within valid ranges (finite, non-NaN, bounded)
    /// 3. **Compatibility**: New observation has same structure as existing ones
    ///
    /// # Error Handling
    ///
    /// Return [`StateError`] for:
    /// - **Size Mismatch**: [`StateError::InvalidSize`] if observation size is wrong
    /// - **Invalid Values**: [`StateError::InvalidData`] for NaN, infinity, or out-of-bounds
    /// - **Shape Mismatch**: [`StateError::InvalidShape`] for multi-dimensional observations
    ///
    /// # Examples
    ///
    /// ## Simple Scalar History
    ///
    /// ```rust
    /// use evorl_core::state::{State, FlattenedState, TemporalState, StateError};
    ///
    /// #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    /// struct VelocityHistory {
    ///     velocities: Vec<i32>, // Last 5 velocity readings
    /// }
    ///
    /// impl VelocityHistory {
    ///     fn new() -> Self {
    ///         Self { velocities: vec![0; 5] }
    ///     }
    /// }
    ///
    /// impl State for VelocityHistory {
    ///     fn numel(&self) -> usize { 5 }
    ///     fn shape(&self) -> Vec<usize> { vec![5] }
    /// }
    ///
    /// impl FlattenedState for VelocityHistory {
    ///     fn flatten(&self) -> Vec<f32> {
    ///         self.velocities.iter().map(|&v| v as f32).collect()
    ///     }
    ///
    ///     fn from_flattened(data: Vec<f32>) -> Result<Self, StateError> {
    ///         Ok(VelocityHistory {
    ///             velocities: data.iter().map(|&v| v as i32).collect(),
    ///         })
    ///     }
    /// }
    ///
    /// impl TemporalState for VelocityHistory {
    ///     fn sequence_length(&self) -> usize { 5 }
    ///
    ///     fn latest(&self) -> &[f32] { &[] } // Simplified
    ///
    ///     fn push_pop(&self, new_observation: &[f32]) -> Result<Self, StateError> {
    ///         if new_observation.len() != 1 {
    ///             return Err(StateError::InvalidSize {
    ///                 expected: 1,
    ///                 got: new_observation.len(),
    ///             });
    ///         }
    ///
    ///         // Shift window: drop first, append new
    ///         let mut new_vels = self.velocities[1..].to_vec();
    ///         new_vels.push(new_observation[0] as i32);
    ///
    ///         Ok(VelocityHistory { velocities: new_vels })
    ///     }
    /// }
    ///
    /// let state = VelocityHistory::new();
    /// let state = state.push_pop(&[10.0]).unwrap();
    /// let state = state.push_pop(&[15.0]).unwrap();
    /// assert_eq!(state.velocities, vec![0, 0, 0, 10, 15]);
    /// ```
    ///
    /// # Parameters
    ///
    /// - `new_observation`: A slice containing the new observation to append. Its length
    ///   and structure must match the size of individual observations in this state.
    ///
    /// # Returns
    ///
    /// - `Ok(Self)`: A new state with the updated observation window
    /// - `Err(StateError)`: The observation is invalid (wrong size, bad values, etc.)
    ///
    /// # See Also
    ///
    /// - [`sequence_length()`](TemporalState::sequence_length): Window size (unchanged by this operation)
    /// - [`latest()`](TemporalState::latest): Access the most recent observation
    fn push_pop(&self, new_observation: &[f32]) -> Result<Self, StateError>;
}

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

/// Framework-agnostic trait for converting states to framework-specific tensors.
///
/// `StateTensorConvertible` provides the bridge between framework-independent state
/// representations and the tensor types used by deep learning frameworks like Burn.
/// This separation of concerns allows:
///
/// - Core state logic to remain framework-agnostic
/// - Easy migration between ML frameworks
/// - Multiple tensor backends (CPU, CUDA, Metal) without changing state code
/// - Testing state logic without framework dependencies
///
/// # Design Philosophy
///
/// By keeping tensor conversion separate from the core [`State`] trait, we ensure that:
/// 1. States can be used in non-ML contexts (e.g., game logic, visualization)
/// 2. Framework-specific optimizations don't leak into state implementations
/// 3. Different frameworks can provide their own conversion implementations
///
/// # The Rank Parameter `R`
///
/// The const generic `R` specifies the tensor rank (number of dimensions):
/// - `R = 1`: Flat vector `[features]`
/// - `R = 2`: Batched vector `[batch_size, features]`
/// - `R = 3`: Batched sequences `[batch_size, seq_len, features]`
/// - `R = 4`: Batched images `[batch_size, channels, height, width]`
///
/// The rank typically includes a batch dimension for neural network processing,
/// even when converting a single state instance.
///
/// # Backend Independence
///
/// This trait is generic over the Burn [`Backend`](burn::tensor::backend::Backend),
/// allowing the same state to produce:
/// - CPU tensors (`NdArray` backend)
/// - CUDA GPU tensors (`Wgpu` backend)
/// - Metal GPU tensors (macOS)
/// - Auto-differentiation-enabled tensors (for gradient computation)
///
/// # Implementation Guidelines
///
/// 1. **Use `FlattenedState`**: Most implementations will call `self.flatten()` and
///    reshape the result to match the desired rank `R`
///
/// 2. **Handle Batching**: If `R` includes a batch dimension, typically add it as
///    the first dimension with size 1 for single instances
///
/// 3. **Device Placement**: Use the provided `device` parameter to place tensors on
///    the correct hardware (CPU/GPU)
///
/// 4. **Precision**: Use `f32` tensors unless your framework/model specifically requires
///    `f64` or other precisions
///
/// 5. **Validation**: `from_tensor()` should validate tensor shape and values match
///    the state's expectations
///
/// # Examples
///
/// ## Simple 1D State
///
/// ```rust,ignore
/// use burn::tensor::{backend::Backend, Tensor};
/// use evorl_core::state::{State, FlattenedState, StateTensorConvertible, StateError};
///
/// #[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// struct Position { x: i32, y: i32 }
///
/// impl State for Position {
///     fn numel(&self) -> usize { 2 }
///     fn shape(&self) -> Vec<usize> { vec![2] }
/// }
///
/// impl FlattenedState for Position {
///     fn flatten(&self) -> Vec<f32> {
///         vec![self.x as f32, self.y as f32]
///     }
///     fn from_flattened(data: Vec<f32>) -> Result<Self, StateError> {
///         if data.len() != 2 {
///             return Err(StateError::InvalidSize { expected: 2, got: data.len() });
///         }
///         Ok(Position { x: data[0] as i32, y: data[1] as i32 })
///     }
/// }
///
/// // Convert to rank-1 tensor (flat vector)
/// impl StateTensorConvertible<1> for Position {
///     fn to_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 1> {
///         let data = self.flatten();
///         Tensor::from_floats(data.as_slice(), device)
///     }
///
///     fn from_tensor<B: Backend>(tensor: &Tensor<B, 1>) -> Result<Self, StateError> {
///         let shape = tensor.shape();
///         if shape.dims[0] != 2 {
///             return Err(StateError::InvalidSize {
///                 expected: 2,
///                 got: shape.dims[0],
///             });
///         }
///         let data: Vec<f32> = tensor.to_data().to_vec().unwrap();
///         Self::from_flattened(data)
///     }
/// }
/// ```
///
/// ## Batched 2D State
///
/// ```rust,ignore
/// // Convert to rank-2 tensor (with batch dimension)
/// impl StateTensorConvertible<2> for Position {
///     fn to_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 2> {
///         let data = self.flatten();
///         // Create [1, 2] tensor (batch_size=1, features=2)
///         Tensor::from_floats(data.as_slice(), device)
///             .reshape([1, 2])
///     }
///
///     fn from_tensor<B: Backend>(tensor: &Tensor<B, 2>) -> Result<Self, StateError> {
///         let shape = tensor.shape();
///         if shape.dims[0] != 1 || shape.dims[1] != 2 {
///             return Err(StateError::InvalidShape {
///                 expected: vec![1, 2],
///                 got: shape.dims.to_vec(),
///             });
///         }
///         // Flatten batch dimension and convert
///         let data: Vec<f32> = tensor.flatten::<1>(0, 1).to_data().to_vec().unwrap();
///         Self::from_flattened(data)
///     }
/// }
/// ```
///
/// # Performance Considerations
///
/// - Tensor creation may allocate GPU memory; reuse tensors when possible
/// - Device transfers (CPU ↔ GPU) are expensive; batch conversions when feasible
/// - `from_tensor()` extracts data to CPU; avoid in performance-critical inference loops
///
/// # See Also
///
/// - [`FlattenedState`]: Provides the underlying numeric representation
/// - [`StateIntTensorConvertible`]: For discrete/categorical states
pub trait StateTensorConvertible<const R: usize> {
    /// Converts this state into a Burn tensor of rank `R`.
    ///
    /// This method transforms the state into a tensor suitable for neural network processing.
    /// The rank `R` determines the tensor's dimensionality, typically including a batch dimension
    /// even for single state instances.
    ///
    /// # Rank Selection
    ///
    /// Choose `R` based on your model's expected input:
    /// - **R=1**: Flat features `[features]` (rare, usually batched)
    /// - **R=2**: Batched flat `[batch=1, features]` (most common)
    /// - **R=3**: Sequences `[batch=1, seq_len, features]` (RNNs, transformers)
    /// - **R=4**: Images `[batch=1, channels, height, width]` (CNNs)
    ///
    /// # Device Placement
    ///
    /// The `device` parameter specifies where the tensor should be allocated:
    /// - CPU devices: For development, testing, or when GPU unavailable
    /// - GPU devices: For training and inference acceleration
    ///
    /// Ensure the device matches your model's device to avoid transfer overhead.
    ///
    /// # Implementation Pattern
    ///
    /// Typical implementations:
    /// 1. Call `self.flatten()` to get raw data
    /// 2. Create a tensor from the flattened data on the specified device
    /// 3. Reshape to match the desired rank `R`
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use burn::tensor::{backend::Backend, Tensor};
    ///
    /// // Simple rank-1 conversion
    /// fn to_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 1> {
    ///     let data = self.flatten();
    ///     Tensor::from_floats(data.as_slice(), device)
    /// }
    ///
    /// // Rank-2 with explicit batch dimension
    /// fn to_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 2> {
    ///     let data = self.flatten();
    ///     let features = data.len();
    ///     Tensor::from_floats(data.as_slice(), device)
    ///         .reshape([1, features]) // Add batch dimension
    /// }
    /// ```
    ///
    /// # Parameters
    ///
    /// - `device`: The hardware device where the tensor should be allocated
    ///
    /// # Returns
    ///
    /// A tensor of rank `R` containing this state's data, allocated on the specified device.
    fn to_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, R>;

    /// Reconstructs a state from a Burn tensor of rank `R`.
    ///
    /// This is the inverse operation of [`to_tensor()`](StateTensorConvertible::to_tensor),
    /// extracting state data from a tensor back into the structured state type. This is
    /// primarily useful for:
    /// - Verifying round-trip conversion correctness in tests
    /// - Extracting predicted states from model outputs
    /// - Debugging tensor transformations
    ///
    /// # Validation
    ///
    /// Implementations must validate:
    /// 1. **Shape**: Tensor dimensions match expected state structure
    /// 2. **Batch Size**: If tensor is batched, typically expect batch size of 1
    /// 3. **Value Ranges**: Tensor values are valid for the state type
    ///
    /// # Error Handling
    ///
    /// Return [`StateError`] for:
    /// - **Shape Mismatch**: [`StateError::InvalidShape`] if tensor shape is incompatible
    /// - **Size Mismatch**: [`StateError::InvalidSize`] if total elements don't match
    /// - **Invalid Data**: [`StateError::InvalidData`] for out-of-range values
    ///
    /// # Implementation Pattern
    ///
    /// Typical implementations:
    /// 1. Validate tensor shape matches expectations
    /// 2. Flatten tensor to 1D if needed (removing batch dimensions)
    /// 3. Extract data to CPU as `Vec<f32>`
    /// 4. Call `Self::from_flattened()` to reconstruct state
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use burn::tensor::{backend::Backend, Tensor};
    ///
    /// // Rank-1 tensor reconstruction
    /// fn from_tensor<B: Backend>(tensor: &Tensor<B, 1>) -> Result<Self, StateError> {
    ///     let shape = tensor.shape();
    ///     if shape.dims[0] != Self::EXPECTED_SIZE {
    ///         return Err(StateError::InvalidSize {
    ///             expected: Self::EXPECTED_SIZE,
    ///             got: shape.dims[0],
    ///         });
    ///     }
    ///     let data: Vec<f32> = tensor.to_data().to_vec().unwrap();
    ///     Self::from_flattened(data)
    /// }
    ///
    /// // Rank-2 batched tensor reconstruction
    /// fn from_tensor<B: Backend>(tensor: &Tensor<B, 2>) -> Result<Self, StateError> {
    ///     let shape = tensor.shape();
    ///     if shape.dims[0] != 1 {
    ///         return Err(StateError::InvalidData(
    ///             format!("Expected batch size 1, got {}", shape.dims[0])
    ///         ));
    ///     }
    ///     // Remove batch dimension and extract
    ///     let flat = tensor.clone().flatten::<1>(0, 1);
    ///     let data: Vec<f32> = flat.to_data().to_vec().unwrap();
    ///     Self::from_flattened(data)
    /// }
    /// ```
    ///
    /// # Performance Considerations
    ///
    /// - This method typically transfers data from GPU to CPU (expensive)
    /// - Avoid calling in tight loops or during inference
    /// - Consider caching extracted states if used multiple times
    ///
    /// # Parameters
    ///
    /// - `tensor`: A reference to the tensor to extract state data from
    ///
    /// # Returns
    ///
    /// - `Ok(Self)`: Successfully reconstructed state
    /// - `Err(StateError)`: Tensor shape/data is incompatible with this state type
    ///
    /// # See Also
    ///
    /// - [`to_tensor()`](StateTensorConvertible::to_tensor): The inverse operation
    /// - [`FlattenedState::from_flattened()`]: Used internally to reconstruct state
    fn from_tensor<B: Backend>(tensor: &Tensor<B, R>) -> Result<Self, StateError>
    where
        Self: Sized;
}

/// Trait for converting states to integer tensors for discrete/categorical data.
///
/// `StateIntTensorConvertible` is a specialized variant of [`StateTensorConvertible`] for
/// states that are fundamentally discrete or categorical. While `StateTensorConvertible`
/// produces floating-point tensors, this trait produces integer tensors, which are:
/// - More memory efficient for discrete data (e.g., one-hot encodings)
/// - Type-appropriate for embedding lookups in neural networks
/// - Semantically clearer for categorical features
///
/// # Use Cases
///
/// ## 1. Discrete Action Spaces
/// States representing discrete choices (e.g., game moves, menu selections):
/// ```text
/// Action::Left → Tensor<i64, 1> = [0]
/// Action::Right → Tensor<i64, 1> = [1]
/// Action::Jump → Tensor<i64, 1> = [2]
/// ```
///
/// ## 2. Grid Positions
/// Integer coordinates on game boards or grids:
/// ```text
/// GridPos { x: 3, y: 5 } → Tensor<i64, 1> = [3, 5]
/// ```
///
/// ## 3. Categorical Features
/// States with enumerated categories:
/// ```text
/// Color::Red → Tensor<i64, 1> = [0]
/// Color::Green → Tensor<i64, 1> = [1]
/// Color::Blue → Tensor<i64, 1> = [2]
/// ```
///
/// # Design Rationale
///
/// Separating integer conversion from float conversion enables:
/// - **Type Safety**: Prevent accidental mixing of discrete and continuous data
/// - **Memory Efficiency**: `i32`/`i64` tensors use less memory than `f32`
/// - **Semantic Clarity**: Make discrete state structure explicit in type system
/// - **Framework Optimization**: Enable framework-specific integer optimizations (e.g., embedding layers)
///
/// # Integer Type
///
/// The trait uses Burn's `Int` type marker, which typically maps to `i64` (64-bit signed integer).
/// This provides:
/// - Sufficient range for most discrete state spaces
/// - Compatibility with PyTorch conventions (also uses i64 for indices)
/// - Headroom to avoid overflow in large discrete spaces
///
/// # Relationship to StateTensorConvertible
///
/// A state may implement both traits:
/// - `StateTensorConvertible`: For neural network consumption (may normalize, one-hot encode)
/// - `StateIntTensorConvertible`: For discrete representations (raw integer values)
///
/// These serve different purposes and may produce tensors with different shapes/values.
///
/// # Implementation Guidelines
///
/// 1. **Direct Mapping**: Map discrete values directly to integers (no normalization)
/// 2. **Consistency**: Integer values should be deterministic and stable across runs
/// 3. **Range**: Ensure values fit within `i64` range (typically not an issue)
/// 4. **Zero-Based**: By convention, use 0-based indexing for categories
///
/// # Examples
///
/// ## Simple Discrete Action
///
/// ```rust,ignore
/// use burn::tensor::{backend::Backend, Int, Tensor};
///
/// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// enum Action {
///     Up,
///     Down,
///     Left,
///     Right,
/// }
///
/// impl Action {
///     fn to_index(&self) -> i64 {
///         match self {
///             Action::Up => 0,
///             Action::Down => 1,
///             Action::Left => 2,
///             Action::Right => 3,
///         }
///     }
/// }
///
/// impl StateIntTensorConvertible<1> for Action {
///     fn to_int_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 1, Int> {
///         Tensor::from_ints([self.to_index()], device)
///     }
/// }
/// ```
///
/// ## Grid Position
///
/// ```rust,ignore
/// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// struct GridPos {
///     x: i32,
///     y: i32,
/// }
///
/// impl StateIntTensorConvertible<1> for GridPos {
///     fn to_int_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 1, Int> {
///         Tensor::from_ints([self.x as i64, self.y as i64], device)
///     }
/// }
/// ```
///
/// # Performance Considerations
///
/// - Integer tensors are typically smaller than float tensors (i32 vs f32, i64 vs f64)
/// - GPU operations on integers may be faster for certain operations (indexing, masking)
/// - Converting to embeddings (via embedding layers) is more efficient with integer inputs
///
/// # See Also
///
/// - [`StateTensorConvertible`]: For continuous/floating-point tensor conversion
/// - [`State`]: Base trait for all states
pub trait StateIntTensorConvertible<const R: usize> {
    /// Converts this state into a Burn integer tensor of rank `R`.
    ///
    /// This method transforms discrete state data into an integer tensor suitable for
    /// embedding layers, indexing operations, or other discrete neural network operations.
    ///
    /// # Tensor Type
    ///
    /// Returns a `Tensor<B, R, Int>` where `Int` is Burn's integer marker type (typically `i64`).
    /// This is distinct from float tensors (`Tensor<B, R>`) and enables:
    /// - Memory-efficient discrete representations
    /// - Direct use with embedding layers (no conversion needed)
    /// - Type-safe categorical operations
    ///
    /// # Rank Conventions
    ///
    /// Common rank choices:
    /// - **R=1**: Single discrete value or vector of discrete values `[categories]`
    /// - **R=2**: Batched discrete values `[batch=1, categories]`
    /// - **R=3**: Sequences of discrete values `[batch=1, seq_len, categories]`
    ///
    /// # Implementation Pattern
    ///
    /// 1. Convert state fields to `i64` values
    /// 2. Create an array or slice of integers
    /// 3. Use `Tensor::from_ints()` to create the tensor on the specified device
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use burn::tensor::{backend::Backend, Int, Tensor};
    ///
    /// // Single discrete value
    /// fn to_int_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 1, Int> {
    ///     let value = self.to_category_index(); // Returns i64
    ///     Tensor::from_ints([value], device)
    /// }
    ///
    /// // Multiple discrete features
    /// fn to_int_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 1, Int> {
    ///     let values = vec![
    ///         self.x as i64,
    ///         self.y as i64,
    ///         self.category as i64,
    ///     ];
    ///     Tensor::from_ints(values.as_slice(), device)
    /// }
    ///
    /// // With explicit batch dimension (rank 2)
    /// fn to_int_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 2, Int> {
    ///     let value = self.to_category_index();
    ///     Tensor::from_ints([value], device).reshape([1, 1])
    /// }
    /// ```
    ///
    /// # Parameters
    ///
    /// - `device`: The hardware device where the integer tensor should be allocated
    ///
    /// # Returns
    ///
    /// An integer tensor of rank `R` containing this state's discrete values,
    /// allocated on the specified device.
    ///
    /// # See Also
    ///
    /// - [`StateTensorConvertible::to_tensor()`]: For floating-point tensor conversion
    fn to_int_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, R, Int>;

    fn from_int_tensor<B: Backend>(&self, device: &B::Device) -> Result<Self, StateError>
    where
        Self: Sized;
}

// `cargo test -p evorl-core -- state`
#[cfg(test)]
mod tests {
    use super::*;

    /// ========================================================================
    /// GameState example to test the State trait implementation
    /// ========================================================================
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    enum GameState {
        Menu,
        Playing { level: u8 },
        GameOver { score: u32 },
    }

    impl State for GameState {
        fn is_valid(&self) -> bool {
            match self {
                GameState::Playing { level } => *level > 0 && *level <= 10,
                _ => true,
            }
        }

        fn numel(&self) -> usize {
            // Encode as 3 features: [state_id, level, score]
            3
        }

        fn shape(&self) -> Vec<usize> {
            vec![3]
        }
    }

    /// Test state validation for each state variant
    #[test]
    fn test_game_state_validation() {
        // Menu state should always be valid
        let menu_state = GameState::Menu;
        assert!(menu_state.is_valid(), "Menu state should always be valid");

        // GameOver state should always be valid
        let game_over_state = GameState::GameOver { score: 1000 };
        assert!(
            game_over_state.is_valid(),
            "GameOver state should always be valid"
        );

        // Playing state with valid levels should be valid
        for level in 1..=10 {
            let playing_state = GameState::Playing { level };
            assert!(
                playing_state.is_valid(),
                "Playing state with level {} should be valid",
                level
            );
        }

        // Playing state with invalid levels should be invalid
        let invalid_levels = [0, 11, 255];
        for level in invalid_levels {
            let invalid_state = GameState::Playing { level };
            assert!(
                !invalid_state.is_valid(),
                "Playing state with level {} should be invalid",
                level
            );
        }
    }

    /// Test that numel returns 3 for all state variants
    #[test]
    fn test_game_state_numel() {
        let test_states = [
            GameState::Menu,
            GameState::Playing { level: 5 },
            GameState::GameOver { score: 1000 },
        ];

        for state in test_states {
            assert_eq!(
                state.numel(),
                3,
                "Number of elements should be 3 for all states"
            );
        }
    }

    /// Test that shape returns [3] for all state variants
    #[test]
    fn test_game_state_shape() {
        let test_states = [
            GameState::Menu,
            GameState::Playing { level: 5 },
            GameState::GameOver { score: 1000 },
        ];

        for state in test_states {
            assert_eq!(state.shape(), vec![3], "Shape should be [3] for all states");
        }
    }

    /// Test the invariant: numel() should equal product of shape()
    #[test]
    fn test_game_state_consistency() {
        let test_states = [
            GameState::Menu,
            GameState::Playing { level: 5 },
            GameState::GameOver { score: 1000 },
        ];

        for state in test_states {
            let numel = state.numel();
            let shape_product: usize = state.shape().iter().product();
            assert_eq!(
                numel, shape_product,
                "numel({}) should equal shape product({})",
                numel, shape_product
            );
        }
    }

    /// Test that filtering states by validity works correctly
    #[test]
    fn test_game_state_filtering() {
        let states = vec![
            GameState::Menu,
            GameState::Playing { level: 5 },
            GameState::Playing { level: 0 }, // Invalid
            GameState::GameOver { score: 1000 },
        ];

        let valid_states: Vec<_> = states.into_iter().filter(|s| s.is_valid()).collect();

        assert_eq!(
            valid_states.len(),
            3,
            "Should have 3 valid states out of 4 total"
        );
        assert!(
            valid_states.iter().all(|s| s.is_valid()),
            "All filtered states should be valid"
        );

        // Verify the invalid state was filtered out
        assert!(
            !valid_states.contains(&GameState::Playing { level: 0 }),
            "Invalid playing state should be filtered out"
        );
    }

    /// Test edge cases for Playing state level bounds
    #[test]
    fn test_playing_state_edge_cases() {
        // Test boundary values
        let min_valid_level = GameState::Playing { level: 1 };
        assert!(
            min_valid_level.is_valid(),
            "Level 1 should be valid (minimum valid)"
        );

        let max_valid_level = GameState::Playing { level: 10 };
        assert!(
            max_valid_level.is_valid(),
            "Level 10 should be valid (maximum valid)"
        );

        let below_min = GameState::Playing { level: 0 };
        assert!(
            !below_min.is_valid(),
            "Level 0 should be invalid (below minimum)"
        );

        let above_max = GameState::Playing { level: 11 };
        assert!(
            !above_max.is_valid(),
            "Level 11 should be invalid (above maximum)"
        );
    }

    /// ========================================================================
    /// GridPosition example to test the State trait implementation
    /// ========================================================================
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct GridPosition {
        x: i32,
        y: i32,
        max_x: i32,
        max_y: i32,
    }

    impl State for GridPosition {
        fn is_valid(&self) -> bool {
            self.x >= 0 && self.y >= 0 && self.x < self.max_x && self.y < self.max_y
        }

        fn numel(&self) -> usize {
            2 // x and y coordinates
        }

        fn shape(&self) -> Vec<usize> {
            vec![2] // flat 1D representation
        }
    }

    impl FlattenedState for GridPosition {
        fn flatten(&self) -> Vec<f32> {
            vec![
                self.x as f32,
                self.y as f32,
                self.max_x as f32,
                self.max_y as f32,
            ]
        }

        fn from_flattened(data: Vec<f32>) -> Result<Self, StateError> {
            if data.len() != 4 {
                return Err(StateError::InvalidSize {
                    expected: 4,
                    got: data.len(),
                });
            }
            Ok(GridPosition {
                x: data[0] as i32,
                y: data[1] as i32,
                max_x: data[2] as i32,
                max_y: data[3] as i32,
            })
        }
    }

    impl<const R: usize> StateTensorConvertible<R> for GridPosition {
        fn to_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, R> {
            let data = self.flatten();
            Tensor::from_floats(data.as_slice(), device)
        }

        fn from_tensor<B: Backend>(tensor: &Tensor<B, R>) -> Result<Self, StateError> {
            let data = tensor.to_data().to_vec::<f32>().unwrap();
            Self::from_flattened(data)
        }
    }

    /// Test GridPosition validation
    #[test]
    fn test_grid_position_validation() {
        let valid = GridPosition {
            x: 5,
            y: 3,
            max_x: 10,
            max_y: 10,
        };
        assert!(valid.is_valid(), "x, y should be valid.");
        //
        let invalid = GridPosition {
            x: 15,
            y: 3,
            max_x: 10,
            max_y: 10,
        };
        assert!(
            !invalid.is_valid(),
            "x is larger than max_x and therefore invalid."
        );
    }

    // Test GridPosition flatten
    #[test]
    fn test_grid_position_flattening() {
        let pos1 = GridPosition {
            x: 3,
            y: 7,
            max_x: 10,
            max_y: 10,
        };
        let pos2 = GridPosition {
            x: 0,
            y: 0,
            max_x: 10,
            max_y: 10,
        };
        let pos3 = GridPosition {
            x: 9,
            y: 9,
            max_x: 10,
            max_y: 10,
        };
        let flat1 = pos1.flatten();
        let flat2 = pos2.flatten();
        let flat3 = pos3.flatten();

        assert_eq!(flat1, vec![3.0, 7.0, 10.0, 10.0]);
        assert_eq!(flat2, vec![0.0, 0.0, 10.0, 10.0]);
        assert_eq!(flat3, vec![9.0, 9.0, 10.0, 10.0]);
    }
}
