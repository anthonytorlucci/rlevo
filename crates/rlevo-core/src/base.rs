//! Core traits for reinforcement learning abstractions.
//!
//! This module defines the foundational vocabulary used throughout `rlevo-core`:
//! rewards, observations, states, actions, transition dynamics, and tensor
//! conversion. All other modules depend on these primitives.

use burn::tensor::Tensor;
use burn::tensor::TensorData;
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Generic update function: how something evolves over time.
///
/// Parameterized over the input stimulus and the output type it transforms.
pub trait UpdateFunction<Input, Output> {
    /// Computes the next value given the current value and an input.
    fn update(&self, current: &Output, input: &Input) -> Output;
}

/// A scalar reward signal emitted by an environment each step.
pub trait Reward: Clone + std::ops::Add<Output = Self> + Into<f32> + Debug {
    /// Returns the additive identity for this reward type (typically `0.0`).
    fn zero() -> Self;
}

/// The `Observation` trait defines how an agent perceives the world. It
/// represents something that can be observed from the environment.
/// Implements `Serialize` and `Deserialize` for storage in a replay buffer.
pub trait Observation<const R: usize>:
    Debug + Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>
{
    /// The rank of this observation space — i.e. the number of axes (tensor
    /// order), *not* the size of any axis.
    ///
    /// "Rank" here means the count of indices needed to address an element
    /// (NumPy `ndim`, Burn's `Tensor<B, R>`), not matrix rank or CP-decomposition
    /// rank. This is automatically set to match the const generic parameter `R`.
    const RANK: usize = R;

    /// Returns the size of each axis in this observation space.
    ///
    /// The returned array has length `R` (the rank), where each element is the
    /// cardinality of that axis — the number of possible values along it. All
    /// values must be greater than zero.
    fn shape() -> [usize; R];
}

/// The complete state of an environment (Markov property).
///
/// # Observation production has moved off `State`
///
/// A `State` no longer produces its own observation. In the POMDP tuple
/// ⟨S, A, T, R, Ω, O⟩ the emission model `O` is a property of the *environment*,
/// not of a state value, so it lives on the env-side
/// [`Sensor`](crate::environment::Sensor) trait with the canonical signature
/// `O(a, s')`. `State` retains only what genuinely belongs to a point in state
/// space: its rank, [`shape`](State::shape), [`numel`](State::numel), and
/// [`is_valid`](State::is_valid). (ADR 0047, superseding ADR 0019.)
pub trait State<const R: usize>: Debug + Clone + Send + Sync {
    /// The rank of this state space — i.e. the number of axes (tensor order),
    /// *not* the size of any axis.
    ///
    /// "Rank" here means the count of indices needed to address an element
    /// (NumPy `ndim`, Burn's `Tensor<B, R>`), not matrix rank or CP-decomposition
    /// rank. This is automatically set to match the const generic parameter `R`.
    const RANK: usize = R;

    /// Returns the size of each axis in this state space.
    ///
    /// The returned array has length `R` (the rank), where each element is the
    /// cardinality of that axis — the number of possible values along it. All
    /// values must be greater than zero.
    fn shape() -> [usize; R];

    /// Validates whether this state satisfies all constraints.
    ///
    /// This method checks if the state is legal according to its type's invariants.
    /// It does **not** check environment-specific legality - that's the environment's responsibility.
    ///
    /// # Returns
    ///
    /// Returns `true` if the state satisfies all structural constraints, `false` otherwise.
    fn is_valid(&self) -> bool;

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
    /// [`shape()`](State::shape). The default implementation enforces this by computing
    /// the product directly. Override only if the state uses a non-product layout.
    ///
    /// # Returns
    ///
    /// The total number of scalar elements needed to represent this state.
    fn numel(&self) -> usize {
        Self::shape().iter().product()
    }
}

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
pub trait Action<const R: usize>: Debug + Clone + Sized {
    /// The rank of this action space — i.e. the number of axes (tensor order),
    /// *not* the size of any axis.
    ///
    /// "Rank" here means the count of indices needed to address an element
    /// (NumPy `ndim`, Burn's `Tensor<B, R>`), not matrix rank or CP-decomposition
    /// rank. This is automatically set to match the const generic parameter `R`.
    const RANK: usize = R;

    /// Returns the size of each axis in this action space.
    ///
    /// The returned array has length `R` (the rank), where each element is the
    /// cardinality of that axis — the number of possible values along it. All
    /// values must be greater than zero.
    fn shape() -> [usize; R];

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

/// Deterministic environment transition dynamics.
///
/// ```math
/// s_{t+1} = f(s_t, a_t)
/// ```
///
/// This trait covers only **deterministic** transitions. Stochastic dynamics
/// (where the successor state is drawn from a distribution) are not modeled
/// here; environments with stochastic transitions implement that logic internally
/// inside [`crate::environment::Environment::step`].
pub trait TransitionDynamics<const SR: usize, const AR: usize, S: State<SR>, A: Action<AR>> {
    /// Returns the successor state after applying `action` to `state`.
    fn transition(&self, state: &S, action: &A) -> S;
}

/// Error returned when a tensor cannot be converted to or from a domain type.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("Invalid tensor conversion: {message}")]
pub struct TensorConversionError {
    /// Human-readable description of why the conversion failed.
    pub message: String,
}

/// Host-side, backend-independent row serialization.
///
/// This is the layout half of tensor conversion: how a value flattens into a
/// row-major `f32` row, and what shape that row has. Neither answer depends on
/// which Burn backend the row is eventually uploaded to, so this trait carries
/// **no** `B` parameter — a fact the pre-split [`TensorConvertible`] could only
/// state in prose.
///
/// Implement this alongside [`TensorConvertible`], which requires it as a
/// supertrait and derives [`TensorConvertible::to_tensor`] from it. The
/// whole-batch staging helper [`stack_to_tensor`] needs *only* this trait,
/// because staging never touches a device.
///
/// # Type Parameters
///
/// - `R`: Rank of a single row.
///
/// # Invariants
///
/// - A type implements `HostRow` at exactly **one** rank `R`. Implementing it at
///   two ranks makes every unqualified `write_host_row` / `row_shape` call on a
///   concrete value of that type ambiguous (E0283).
/// - `write_host_row` pushes exactly `row_shape().iter().product()` values.
///
/// [`stack_to_tensor`]: crate::base::stack_to_tensor
pub trait HostRow<const R: usize> {
    /// Returns the per-item ("row") shape of the tensor this type serializes to.
    ///
    /// This is the shape of a **single** value — rank `R`, with each axis size
    /// fixed by the domain type (e.g. `[8]` for an 8-feature observation, or
    /// `[H, W, C]` for an image). It is the layout that [`write_host_row`] must
    /// fill, and the shape [`to_tensor`] wraps around the written buffer.
    ///
    /// The product of the returned axes is the number of `f32` scalars a single
    /// row occupies, which is exactly how many values [`write_host_row`] must
    /// push.
    ///
    /// [`write_host_row`]: HostRow::write_host_row
    /// [`to_tensor`]: TensorConvertible::to_tensor
    fn row_shape() -> [usize; R];

    /// Appends the row-major `f32` payload of `self` to `buf`.
    ///
    /// This is the primitive from which both single-item conversion
    /// ([`to_tensor`]) and whole-batch staging ([`stack_to_tensor`]) are
    /// derived, guaranteeing the two can never disagree on element order.
    ///
    /// # Contract
    ///
    /// - Push **exactly** `row_shape().iter().product()` values, in **row-major**
    ///   order matching [`row_shape`].
    /// - Push **plain `f32`** — do *not* pre-convert to a backend element type.
    ///   [`TensorData::new`] performs the element-type conversion at upload time.
    ///   This trait has no backend parameter, so there is nothing to convert to.
    /// - **Append**; never clear or truncate `buf`. Batch staging relies on
    ///   successive rows being concatenated into one contiguous buffer.
    ///
    /// [`row_shape`]: HostRow::row_shape
    /// [`to_tensor`]: TensorConvertible::to_tensor
    /// [`stack_to_tensor`]: crate::base::stack_to_tensor
    fn write_host_row(&self, buf: &mut Vec<f32>);
}

/// Bidirectional conversion between a domain type and a Burn tensor.
///
/// Implementors must round-trip: `from_tensor(x.to_tensor(device))` equals
/// `Ok(x)` for any valid `x`. Strategies and replay buffers rely on this
/// invariant.
///
/// The backend-independent half of the conversion — the row layout — lives on
/// the [`HostRow`] supertrait; this trait adds only the device-facing half.
///
/// # Type Parameters
///
/// - `R`: Rank of the tensor produced.
/// - `B`: Burn backend.
///
/// # Errors
///
/// `from_tensor` returns [`TensorConversionError`] when the tensor's shape,
/// dtype, or contents violate the domain type's invariants (see
/// [`State::is_valid`] / [`Action::is_valid`]).
pub trait TensorConvertible<const R: usize, B: Backend>: HostRow<R> + Sized {
    /// Converts `self` into a tensor on `device`.
    ///
    /// # Do not override
    ///
    /// This method has a default body derived from [`row_shape`] and
    /// [`write_host_row`]: it stages one row into a host `Vec<f32>` and uploads
    /// it with a single [`Tensor::from_data`]. Implementors **must not** provide
    /// their own `to_tensor` — doing so would let the single-item layout drift
    /// from the batched layout produced by [`stack_to_tensor`], defeating the
    /// whole point of the shared row-writer primitive.
    ///
    /// [`row_shape`]: HostRow::row_shape
    /// [`write_host_row`]: HostRow::write_host_row
    /// [`stack_to_tensor`]: crate::base::stack_to_tensor
    fn to_tensor(
        &self,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Tensor<B, R> {
        let row: [usize; R] = Self::row_shape();
        let mut buf: Vec<f32> = Vec::with_capacity(row.iter().product());
        self.write_host_row(&mut buf);
        debug_assert_eq!(buf.len(), row.iter().product::<usize>());
        Tensor::from_data(TensorData::new(buf, row), device)
    }

    /// Reconstructs a value from a tensor.
    ///
    /// # Errors
    ///
    /// Returns [`TensorConversionError`] if the tensor's shape or contents
    /// do not describe a valid instance of `Self`.
    fn from_tensor(tensor: Tensor<B, R>) -> Result<Self, TensorConversionError>;
}

/// Stages a whole batch of rows into one host buffer and uploads it as a single
/// tensor.
///
/// Each item's [`write_host_row`] payload is concatenated into one contiguous
/// `Vec<f32>`, which is then uploaded with a **single** [`Tensor::from_data`]
/// call. This is materially cheaper than converting each item to its own tensor
/// and calling [`Tensor::stack`], which incurs one host→device upload *per item*
/// plus a concatenation kernel. Because both this function and the derived
/// [`TensorConvertible::to_tensor`] draw from the same
/// [`write_host_row`]/[`row_shape`] primitives, the batched layout is guaranteed
/// to match `stack`-ing the individual rows.
///
/// The produced tensor has rank `BR = R + 1` and shape `[items.len(), ..row]`,
/// i.e. a leading batch axis followed by the per-item [`row_shape`].
///
/// # Type Parameters
///
/// - `R`: rank of a single row.
/// - `BR`: rank of the batched tensor; must equal `R + 1`.
/// - `T`: the row type, [`HostRow<R>`]. Note the bound is [`HostRow`], not
///   [`TensorConvertible`]: staging is host-only, so the row type need not name
///   a backend at all.
/// - `B`: Burn backend the assembled buffer is uploaded to.
///
/// # The `BR = R + 1` contract
///
/// Stable Rust cannot express `R + 1` in a const-generic position, so `BR` is a
/// separate parameter checked at runtime. This function is the **single
/// chokepoint** for that invariant: the leading `assert_eq!` runs before the
/// shape array is assembled, which is what makes the subsequent
/// `shape[1..].copy_from_slice(&row)` sound (it would panic on a length
/// mismatch otherwise).
///
/// # Panics
///
/// Panics if `BR != R + 1`.
///
/// # Examples
///
/// ```
/// use burn::backend::Flex;
/// use burn::tensor::Tensor;
/// use rlevo_core::base::{stack_to_tensor, HostRow, TensorConversionError, TensorConvertible};
///
/// #[derive(Clone)]
/// struct Point {
///     x: f32,
///     y: f32,
/// }
///
/// impl HostRow<1> for Point {
///     fn row_shape() -> [usize; 1] {
///         [2]
///     }
///     fn write_host_row(&self, buf: &mut Vec<f32>) {
///         buf.push(self.x);
///         buf.push(self.y);
///     }
/// }
///
/// impl<B: burn::tensor::backend::Backend> TensorConvertible<1, B> for Point {
///     fn from_tensor(_tensor: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
///         unimplemented!()
///     }
/// }
///
/// type B = Flex;
/// let device = Default::default();
/// let items: Vec<Point> = vec![Point { x: 1.0, y: 2.0 }, Point { x: 3.0, y: 4.0 }];
/// let batched: Tensor<B, 2> = stack_to_tensor::<1, 2, Point, B>(&items, &device);
/// assert_eq!(batched.dims(), [2, 2]);
/// ```
///
/// [`write_host_row`]: HostRow::write_host_row
/// [`row_shape`]: HostRow::row_shape
pub fn stack_to_tensor<const R: usize, const BR: usize, T, B>(
    items: &[T],
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<B, BR>
where
    T: HostRow<R>,
    B: Backend,
{
    assert_eq!(BR, R + 1, "batched rank BR must equal row rank R + 1");
    let row: [usize; R] = T::row_shape();
    let row_len: usize = row.iter().product();
    let mut buf: Vec<f32> = Vec::with_capacity(items.len() * row_len);
    for item in items {
        item.write_host_row(&mut buf);
    }
    debug_assert_eq!(buf.len(), items.len() * row_len);
    let mut shape: [usize; BR] = [0usize; BR];
    shape[0] = items.len();
    shape[1..].copy_from_slice(&row); // sound only because the BR == R + 1 assert above ran first — keep the ordering
    Tensor::from_data(TensorData::new(buf, shape), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple scalar reward implementation for testing
    #[derive(Clone, Debug, PartialEq)]
    struct TestReward(f32);

    impl Reward for TestReward {
        fn zero() -> Self {
            TestReward(0.0)
        }
    }

    impl std::ops::Add for TestReward {
        type Output = Self;

        fn add(self, other: Self) -> Self {
            TestReward(self.0 + other.0)
        }
    }

    impl From<TestReward> for f32 {
        fn from(reward: TestReward) -> f32 {
            reward.0
        }
    }

    // ===== Basic Reward Trait Tests =====

    /// Test that zero() creates a neutral element for addition
    #[test]
    fn test_reward_zero_is_additive_identity() {
        let zero = TestReward::zero();
        let reward = TestReward(42.5);

        // zero + reward should equal reward
        let result = zero.clone() + reward.clone();
        assert_eq!(result, reward);

        // reward + zero should equal reward
        let result = reward.clone() + zero.clone();
        assert_eq!(result, reward);
    }

    /// Test that rewards can be added together
    #[test]
    fn test_reward_addition() {
        let reward1 = TestReward(10.0);
        let reward2 = TestReward(25.5);
        let result = reward1 + reward2;

        assert_eq!(result, TestReward(35.5));
    }

    /// Test that negative rewards can be added
    #[test]
    fn test_reward_negative_addition() {
        let positive = TestReward(100.0);
        let negative = TestReward(-30.0);
        let result = positive + negative;

        assert_eq!(result, TestReward(70.0));
    }

    /// Test that rewards can be converted to f32
    #[test]
    fn test_reward_into_f32() {
        let reward = TestReward(42.5);
        let as_f32: f32 = reward.into();

        assert_eq!(as_f32, 42.5);
    }

    /// Test that zero reward converts to 0.0
    #[test]
    fn test_reward_zero_into_f32() {
        let zero = TestReward::zero();
        let as_f32: f32 = zero.into();

        assert_eq!(as_f32, 0.0);
    }

    /// Test that rewards are cloneable
    #[test]
    fn test_reward_clone() {
        let original = TestReward(123.456);
        let cloned = original.clone();

        assert_eq!(original, cloned);
    }

    /// Test that rewards implement Debug
    #[test]
    fn test_reward_debug() {
        let reward = TestReward(42.0);
        let debug_str = format!("{:?}", reward);

        assert!(!debug_str.is_empty());
        assert!(debug_str.contains("TestReward"));
    }

    // ===== Arithmetic Properties Tests =====

    /// Test accumulated reward through chained additions
    #[test]
    fn test_reward_accumulation() {
        let mut accumulated = TestReward::zero();
        let rewards = vec![TestReward(10.0), TestReward(20.0), TestReward(15.0)];

        for reward in rewards {
            accumulated = accumulated + reward;
        }

        assert_eq!(accumulated, TestReward(45.0));
    }

    /// Test reward trait with floating point precision
    #[test]
    fn test_reward_floating_point_precision() {
        let r1 = TestReward(0.1);
        let r2 = TestReward(0.2);
        let result = r1 + r2;

        // Account for floating point imprecision
        let expected = 0.3;
        let as_f32: f32 = result.into();
        assert!((as_f32 - expected).abs() < 1e-6);
    }

    /// Test addition associativity: (a + b) + c == a + (b + c)
    #[test]
    fn test_reward_addition_associativity() {
        let r1 = TestReward(5.0);
        let r2 = TestReward(10.0);
        let r3 = TestReward(15.0);

        let left = (r1.clone() + r2.clone()) + r3.clone();
        let right = r1 + (r2 + r3);

        assert_eq!(left, right);
    }

    /// Test addition commutativity: a + b == b + a
    #[test]
    fn test_reward_addition_commutativity() {
        let r1 = TestReward(7.5);
        let r2 = TestReward(12.5);

        let left = r1.clone() + r2.clone();
        let right = r2 + r1;

        assert_eq!(left, right);
    }

    // ===== Special Values Tests =====

    /// Test reward arithmetic with large values
    #[test]
    fn test_reward_large_values() {
        let large1 = TestReward(1e6);
        let large2 = TestReward(1e6);

        let result = large1 + large2;
        let result_f32: f32 = result.into();

        assert_eq!(result_f32, 2e6);
    }

    /// Test reward arithmetic with small values
    #[test]
    fn test_reward_small_values() {
        let small1 = TestReward(1e-6);
        let small2 = TestReward(1e-6);

        let result = small1 + small2;
        let result_f32: f32 = result.into();

        assert!((result_f32 - 2e-6).abs() < 1e-7);
    }

    /// Test mixed positive and negative rewards
    #[test]
    fn test_reward_mixed_signs() {
        let positive = TestReward(10.0);
        let negative = TestReward(-5.0);

        let pos_then_neg = positive.clone() + negative.clone();
        let pos_then_neg_f32: f32 = pos_then_neg.into();

        let neg_then_pos = negative.clone() + positive.clone();
        let neg_then_pos_f32: f32 = neg_then_pos.into();

        assert_eq!(pos_then_neg_f32, 5.0);
        assert_eq!(neg_then_pos_f32, 5.0);
    }

    /// ========================================================================
    /// GameState example to test the State trait implementation
    /// ========================================================================
    #[derive(Debug, Clone, PartialEq)]
    enum GameState {
        Menu,
        Playing { level: u8 },
        GameOver { score: u32 },
    }

    impl State<1> for GameState {
        fn shape() -> [usize; 1] {
            [3] // 3 features: state_id, level, score
        }

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

        for _state in test_states {
            assert_eq!(
                GameState::shape(),
                [3],
                "Shape should be [3] for all states"
            );
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
            let shape_product: usize = GameState::shape().iter().product();
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
    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct GridPosition {
        x: i32,
        y: i32,
        max_x: i32,
        max_y: i32,
    }

    impl State<1> for GridPosition {
        fn shape() -> [usize; 1] {
            [2] // 2 coordinates: x, y
        }

        fn is_valid(&self) -> bool {
            self.x >= 0 && self.y >= 0 && self.x < self.max_x && self.y < self.max_y
        }

        fn numel(&self) -> usize {
            2 // x and y coordinates
        }
    }

    impl GridPosition {
        /// Flatten the grid position to a vector of f32 values
        fn flatten(&self) -> Vec<f32> {
            vec![
                self.x as f32,
                self.y as f32,
                self.max_x as f32,
                self.max_y as f32,
            ]
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

    /// Test GridPosition flatten
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

    /// Test that GridPosition numel matches shape product
    #[test]
    fn test_grid_position_consistency() {
        let pos = GridPosition {
            x: 5,
            y: 3,
            max_x: 10,
            max_y: 10,
        };
        let numel = pos.numel();
        let shape_product: usize = GridPosition::shape().iter().product();
        assert_eq!(
            numel, shape_product,
            "numel should equal shape product for GridPosition"
        );
        assert_eq!(numel, 2, "GridPosition should have numel of 2");
    }

    /// Test State trait const RANK value
    #[test]
    fn test_state_rank_constant() {
        assert_eq!(
            <GameState as State<1>>::RANK,
            1,
            "GameState should have RANK = 1"
        );
        assert_eq!(
            <GridPosition as State<1>>::RANK,
            1,
            "GridPosition should have RANK = 1"
        );
    }

    // ========================================================================
    // TensorConvertible: derived `to_tensor` + `stack_to_tensor`
    // ========================================================================

    use burn::backend::Flex;

    /// Backend used by the tensor-conversion tests.
    type TcB = Flex;

    /// Rank-1 test row: three scalar features, shape `[3]`.
    #[derive(Clone, Debug)]
    struct Vec3(f32, f32, f32);

    impl HostRow<1> for Vec3 {
        fn row_shape() -> [usize; 1] {
            [3]
        }
        fn write_host_row(&self, buf: &mut Vec<f32>) {
            buf.extend_from_slice(&[self.0, self.1, self.2]);
        }
    }

    impl<B: Backend> TensorConvertible<1, B> for Vec3 {
        fn from_tensor(_tensor: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
            unimplemented!("not exercised by these tests")
        }
    }

    /// Rank-3 test row: a `[2, 2, 1]` image-like payload.
    #[derive(Clone, Debug)]
    struct Img([f32; 4]);

    impl HostRow<3> for Img {
        fn row_shape() -> [usize; 3] {
            [2, 2, 1]
        }
        fn write_host_row(&self, buf: &mut Vec<f32>) {
            buf.extend_from_slice(&self.0);
        }
    }

    impl<B: Backend> TensorConvertible<3, B> for Img {
        fn from_tensor(_tensor: Tensor<B, 3>) -> Result<Self, TensorConversionError> {
            unimplemented!("not exercised by these tests")
        }
    }

    /// `stack_to_tensor` must produce exactly what `Tensor::stack` of the
    /// individually-converted rows produces — bit-identical data and shape.
    #[test]
    fn test_stack_to_tensor_matches_manual_stack() {
        let device: <TcB as burn::tensor::backend::BackendTypes>::Device = Default::default();
        let items: Vec<Vec3> = vec![
            Vec3(1.0, 2.0, 3.0),
            Vec3(4.0, 5.0, 6.0),
            Vec3(7.0, 8.0, 9.0),
        ];

        let batched: Tensor<TcB, 2> = stack_to_tensor::<1, 2, Vec3, TcB>(&items, &device);

        let per_item: Vec<Tensor<TcB, 1>> = items
            .iter()
            .map(|i| <Vec3 as TensorConvertible<1, TcB>>::to_tensor(i, &device))
            .collect();
        let manual: Tensor<TcB, 2> = Tensor::stack(per_item, 0);

        assert_eq!(batched.dims(), manual.dims());
        let batched_v: Vec<f32> = batched
            .into_data()
            .into_vec::<f32>()
            .expect("f32 host read of a tensor this test just built");
        let manual_v: Vec<f32> = manual
            .into_data()
            .into_vec::<f32>()
            .expect("f32 host read of a tensor this test just built");
        assert_eq!(batched_v, manual_v);
    }

    /// `stack_to_tensor` panics when the batched rank does not equal `R + 1`.
    #[test]
    #[should_panic(expected = "batched rank BR must equal row rank R + 1")]
    fn test_stack_to_tensor_wrong_rank_panics() {
        let device: <TcB as burn::tensor::backend::BackendTypes>::Device = Default::default();
        let items: Vec<Vec3> = vec![Vec3(1.0, 2.0, 3.0)];
        // BR = 3, but R + 1 = 2 → must panic.
        let _bad: Tensor<TcB, 3> = stack_to_tensor::<1, 3, Vec3, TcB>(&items, &device);
    }

    /// The derived `to_tensor` produces the same data/shape as the old manual
    /// `Tensor::from_floats` path for a rank-1 row.
    #[test]
    fn test_derived_to_tensor_rank1_matches_manual() {
        let device: <TcB as burn::tensor::backend::BackendTypes>::Device = Default::default();
        let item: Vec3 = Vec3(1.5, -2.5, 3.5);

        let derived: Tensor<TcB, 1> =
            <Vec3 as TensorConvertible<1, TcB>>::to_tensor(&item, &device);
        let manual: Tensor<TcB, 1> = Tensor::from_floats([1.5_f32, -2.5, 3.5], &device);

        assert_eq!(derived.dims(), manual.dims());
        let derived_v: Vec<f32> = derived
            .into_data()
            .into_vec::<f32>()
            .expect("f32 host read of a tensor this test just built");
        let manual_v: Vec<f32> = manual
            .into_data()
            .into_vec::<f32>()
            .expect("f32 host read of a tensor this test just built");
        assert_eq!(derived_v, manual_v);
    }

    /// The derived `to_tensor` produces the same data/shape as the old manual
    /// `TensorData::new` path for a rank-3 row.
    #[test]
    fn test_derived_to_tensor_rank3_matches_manual() {
        let device: <TcB as burn::tensor::backend::BackendTypes>::Device = Default::default();
        let item: Img = Img([0.1, 0.2, 0.3, 0.4]);

        let derived: Tensor<TcB, 3> = <Img as TensorConvertible<3, TcB>>::to_tensor(&item, &device);
        let manual: Tensor<TcB, 3> = Tensor::from_data(
            TensorData::new(vec![0.1_f32, 0.2, 0.3, 0.4], [2, 2, 1]),
            &device,
        );

        assert_eq!(derived.dims(), manual.dims());
        let derived_v: Vec<f32> = derived
            .into_data()
            .into_vec::<f32>()
            .expect("f32 host read of a tensor this test just built");
        let manual_v: Vec<f32> = manual
            .into_data()
            .into_vec::<f32>()
            .expect("f32 host read of a tensor this test just built");
        assert_eq!(derived_v, manual_v);
    }
}
