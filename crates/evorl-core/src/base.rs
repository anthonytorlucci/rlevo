use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt::Debug;

/// Update function: how something evolves over time
/// Generic over input and output types
pub trait UpdateFunction<Input, Output> {
    fn update(&self, current: &Output, input: &Input) -> Output;
}

/// Represents a reward signal
pub trait Reward: Clone + std::ops::Add<Output = Self> + Into<f32> + Debug {
    fn zero() -> Self;
}

/// The `Observation` trait defines how an agent perceives the world. It
/// represents something that can be observed from the environment.
/// Implements `Serialize` and `Deserialize` for storage in a replay buffer.
pub trait Observation<const D: usize>:
    Debug + Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>
{
    /// The number of independent dimensions in this observation space.
    ///
    /// This is automatically set to match the const generic parameter `D`.
    const DIM: usize = D;

    /// Returns the cardinality of each dimension in this observation space.
    ///
    /// The returned array has length `D`, where each element specifies the number
    /// of possible values for that dimension. All values must be greater than zero.
    fn shape() -> [usize; D];
}

/// The complete state of an environment (Markov property)
pub trait State<const D: usize>: Debug + Clone + Send + Sync {
    /// The number of independent dimensions in this state space.
    ///
    /// This is automatically set to match the const generic parameter `D`.
    const DIM: usize = D;

    type Observation: Observation<D>;

    /// Returns the cardinality of each dimension in this state space.
    ///
    /// The returned array has length `D`, where each element specifies the number
    /// of possible values for that dimension. All values must be greater than zero.
    fn shape() -> [usize; D]; // todo! why static method here?

    /// Generate an observation from this state (may be partial)
    fn observe(&self) -> Self::Observation;

    /// Validates whether this state satisfies all constraints.
    ///
    /// This method checks if the state is legal according to its type's invariants.
    /// It does **not** check environment-specific legality - that's the environment's responsibility.
    ///
    /// # Returns
    ///
    /// Returns `true` if the action satisfies all structural constraints, `false` otherwise.
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
    /// [`shape()`](State::shape):
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

/// Environment transition dynamics: s_{t+1} = f(s_t, a_t)
pub trait TransitionDynamics<const SD: usize, const AD: usize, S: State<SD>, A: Action<AD>> {
    fn transition(&self, state: &S, action: &A) -> S;
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorConversionError {
    pub message: String,
}

impl std::fmt::Display for TensorConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Invalid tensor conversion: {}", self.message)
    }
}

impl Error for TensorConversionError {}

/// Bidirectional conversion between a domain type and a Burn tensor.
///
/// Implementors must round-trip: `from_tensor(x.to_tensor(device))` equals
/// `Ok(x)` for any valid `x`. Strategies and replay buffers rely on this
/// invariant.
///
/// # Type Parameters
///
/// - `D`: Rank of the tensor produced.
/// - `B`: Burn backend.
///
/// # Errors
///
/// `from_tensor` returns [`TensorConversionError`] when the tensor's shape,
/// dtype, or contents violate the domain type's invariants (see
/// [`State::is_valid`] / [`Action::is_valid`]).
pub trait TensorConvertible<const D: usize, B: Backend>: Sized {
    /// Converts `self` into a tensor on `device`.
    fn to_tensor(&self, device: &B::Device) -> Tensor<B, D>;

    /// Reconstructs a value from a tensor.
    ///
    /// # Errors
    ///
    /// Returns [`TensorConversionError`] if the tensor's shape or contents
    /// do not describe a valid instance of `Self`.
    fn from_tensor(tensor: Tensor<B, D>) -> Result<Self, TensorConversionError>;
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
    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct GameStateObservation {
        state_id: u8,
        level: u8,
        score: u32,
    }

    impl Observation<1> for GameStateObservation {
        fn shape() -> [usize; 1] {
            [3] // 3 features: state_id, level, score
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    enum GameState {
        Menu,
        Playing { level: u8 },
        GameOver { score: u32 },
    }

    impl State<1> for GameState {
        type Observation = GameStateObservation;

        fn observe(&self) -> Self::Observation {
            match self {
                GameState::Menu => GameStateObservation {
                    state_id: 0,
                    level: 0,
                    score: 0,
                },
                GameState::Playing { level } => GameStateObservation {
                    state_id: 1,
                    level: *level,
                    score: 0,
                },
                GameState::GameOver { score } => GameStateObservation {
                    state_id: 2,
                    level: 0,
                    score: *score,
                },
            }
        }

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

    /// Test that observe() generates correct observations for each state variant
    #[test]
    fn test_game_state_observe() {
        // Test Menu state observation
        let menu_state = GameState::Menu;
        let menu_obs = menu_state.observe();
        assert_eq!(menu_obs.state_id, 0, "Menu state should have state_id 0");
        assert_eq!(menu_obs.level, 0, "Menu state should have level 0");
        assert_eq!(menu_obs.score, 0, "Menu state should have score 0");

        // Test Playing state observation
        let playing_state = GameState::Playing { level: 5 };
        let playing_obs = playing_state.observe();
        assert_eq!(
            playing_obs.state_id, 1,
            "Playing state should have state_id 1"
        );
        assert_eq!(playing_obs.level, 5, "Playing state should preserve level");
        assert_eq!(playing_obs.score, 0, "Playing state should have score 0");

        // Test GameOver state observation
        let game_over_state = GameState::GameOver { score: 1000 };
        let game_over_obs = game_over_state.observe();
        assert_eq!(
            game_over_obs.state_id, 2,
            "GameOver state should have state_id 2"
        );
        assert_eq!(game_over_obs.level, 0, "GameOver state should have level 0");
        assert_eq!(
            game_over_obs.score, 1000,
            "GameOver state should preserve score"
        );
    }

    /// Test GameStateObservation shape
    #[test]
    fn test_game_state_observation_shape() {
        assert_eq!(
            GameStateObservation::shape(),
            [3],
            "GameStateObservation should have shape [3]"
        );
        assert_eq!(
            GameStateObservation::DIM,
            1,
            "GameStateObservation should have dimension 1"
        );
    }

    /// ========================================================================
    /// GridPosition example to test the State trait implementation
    /// ========================================================================
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct GridObservation {
        x: i32,
        y: i32,
    }

    impl Observation<1> for GridObservation {
        fn shape() -> [usize; 1] {
            [2] // 2 coordinates: x, y
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct GridPosition {
        x: i32,
        y: i32,
        max_x: i32,
        max_y: i32,
    }

    impl State<1> for GridPosition {
        type Observation = GridObservation;

        fn observe(&self) -> Self::Observation {
            GridObservation {
                x: self.x,
                y: self.y,
            }
        }

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

    /// Test that observe() generates correct observations for GridPosition
    #[test]
    fn test_grid_position_observe() {
        let pos = GridPosition {
            x: 5,
            y: 3,
            max_x: 10,
            max_y: 10,
        };
        let obs = pos.observe();
        assert_eq!(obs.x, 5, "Observation should preserve x coordinate");
        assert_eq!(obs.y, 3, "Observation should preserve y coordinate");

        // Test with different positions
        let origin = GridPosition {
            x: 0,
            y: 0,
            max_x: 10,
            max_y: 10,
        };
        let origin_obs = origin.observe();
        assert_eq!(origin_obs.x, 0, "Origin observation should have x = 0");
        assert_eq!(origin_obs.y, 0, "Origin observation should have y = 0");

        // Test with edge position
        let edge = GridPosition {
            x: 9,
            y: 9,
            max_x: 10,
            max_y: 10,
        };
        let edge_obs = edge.observe();
        assert_eq!(edge_obs.x, 9, "Edge observation should have x = 9");
        assert_eq!(edge_obs.y, 9, "Edge observation should have y = 9");
    }

    /// Test GridObservation shape
    #[test]
    fn test_grid_observation_shape() {
        assert_eq!(
            GridObservation::shape(),
            [2],
            "GridObservation should have shape [2]"
        );
        assert_eq!(
            GridObservation::DIM,
            1,
            "GridObservation should have dimension 1"
        );
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

    /// Test State trait const DIM value
    #[test]
    fn test_state_dim_constant() {
        assert_eq!(
            <GameState as State<1>>::DIM,
            1,
            "GameState should have DIM = 1"
        );
        assert_eq!(
            <GridPosition as State<1>>::DIM,
            1,
            "GridPosition should have DIM = 1"
        );
    }
}
