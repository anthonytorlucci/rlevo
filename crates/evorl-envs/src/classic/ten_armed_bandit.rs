use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use evorl_core::action::{Action, DiscreteAction};
//use evorl_core::environment::{Environment, EnvironmentError, SnapshotBase};
use evorl_core::state::{State, StateError};
use evorl_core::tensor_convert::TensorConvertible;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

// from Sutton and Barto, 2018, p.25-
// The Problem
// You are faced repeatedly with a choice among k different options, or actions. After each choice you receive a numerical reward chosen from a stationary probability distribution that depends on the action you selected. Your objective is to maximize the expected total reward over some time period, for example, over 1000 action selections, or time steps.

/// Multi-armed bandit state.
///
/// Represents the state in a multi-armed bandit problem where the agent
/// must choose between k=10 different actions to maximize cumulative reward.
/// Since bandit problems are stateless (the optimal action doesn't depend
/// on history), this state is empty.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TenArmedBanditState {}

// impl State for TenArmedBanditState {
//     fn is_valid(&self) -> bool {
//         true
//     }

//     fn numel(&self) -> usize {
//         1
//     }

//     fn shape(&self) -> Vec<usize> {
//         vec![1]
//     }
// }

impl Display for TenArmedBanditState {
    /// Formats the state for human-readable output.
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "TenArmedBanditState")
    }
}

impl<B: Backend> TensorConvertible<1, B> for TenArmedBanditState {
    /// Converts the stateless bandit state to a rank-1 tensor.
    ///
    /// Since the bandit is stateless, this simply returns a tensor with a single
    /// neutral value [0.0].
    fn to_tensor(&self, device: &B::Device) -> Tensor<B, 1> {
        let data = vec![0.0];
        Tensor::from_floats(data.as_slice(), device)
    }

    // /// Reconstructs state from a rank-1 tensor.
    // ///
    // /// Since the state is stateless, this ignores the tensor contents and
    // /// returns a new `TenArmedBanditState`.
    // fn from_tensor<B: Backend>(_tensor: &Tensor<B, 1>) -> Result<Self, StateError>
    // where
    //     Self: Sized,
    // {
    //     Ok(Self {})
    // }
}

// SAFETY: TenArmedBanditState is an empty struct with no fields,
// making it trivially Send and Sync. No synchronization is needed.
unsafe impl Send for TenArmedBanditState {}
unsafe impl Sync for TenArmedBanditState {}

/// Action for the multi-armed bandit problem.
///
/// Represents the agent's choice of which arm to pull.
/// Each action corresponds to selecting one of k=10 arms,
/// indexed from 0 to k-1.
///
/// # Traits Implemented
///
/// - **`Action`**: Core trait defining action validity constraints
/// - **`DiscreteAction`**: Provides enumeration, indexing, and random sampling of actions
/// - **`Display`**: Human-readable formatting (e.g., "TenArmedBanditAction(arm=3)")
/// - **`ActionTensorConvertible<1>`**: Converts to one-hot encoded Burn tensors for neural networks
/// - **`Send + Sync`**: Thread-safe type that can be shared across threads
///
/// # Examples
///
/// ```rust,ignore
/// use evorl_core::action::{Action, DiscreteAction};
///
/// // Create an action by index
/// let action = TenArmedBanditAction::from_index(5);
/// assert!(action.is_valid());
/// assert_eq!(action.to_index(), 5);
///
/// // Enumerate all possible actions
/// let all_actions = TenArmedBanditAction::enumerate();
/// assert_eq!(all_actions.len(), TenArmedBanditAction::ACTION_COUNT);
///
/// // Sample a random action
/// let random_action = TenArmedBanditAction::random();
///
/// // Access the arm index
/// let arm_id = random_action.arm();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TenArmedBanditAction {
    /// The index of the selected arm (0-indexed).
    selected_arm: usize,
}

impl Display for TenArmedBanditAction {
    /// Formats the action for human-readable output.
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "TenArmedBanditAction(arm={})", self.arm())
    }
}

impl TenArmedBanditAction {
    /// Returns the index of the selected arm.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let action = MultiArmedBanditAction::from_index(3);
    /// assert_eq!(action.arm(), 3);
    /// ```
    pub fn arm(&self) -> usize {
        self.selected_arm
    }
}

// impl Action for TenArmedBanditAction {
//     fn is_valid(&self) -> bool {
//         // Action is valid if the arm index is within the valid range.
//         self.selected_arm < Self::ACTION_COUNT
//     }
// }

// impl DiscreteAction for TenArmedBanditAction {
//     /// Standard 10-armed bandit from Sutton & Barto (2018).
//     ///
//     /// This is the classic configuration for multi-armed bandit problems,
//     /// commonly used in reinforcement learning literature and benchmarks.
//     const ACTION_COUNT: usize = 10;

//     fn from_index(index: usize) -> Self {
//         assert!(
//             index < Self::ACTION_COUNT,
//             "Action index {} out of bounds [0, {})",
//             index,
//             Self::ACTION_COUNT
//         );
//         Self {
//             selected_arm: index,
//         }
//     }

//     fn to_index(&self) -> usize {
//         self.selected_arm
//     }

//     fn enumerate() -> Vec<Self>
//     where
//         Self: Sized,
//     {
//         (0..Self::ACTION_COUNT)
//             .map(|selected_arm| Self { selected_arm })
//             .collect()
//     }
// }

// SAFETY: TenArmedBanditAction contains only a usize, which is Copy and thread-safe.
// No synchronization primitives or heap allocations are involved.
unsafe impl Send for TenArmedBanditAction {}
unsafe impl Sync for TenArmedBanditAction {}

// impl<B: Backend> TensorConvertible<1, B> for TenArmedBanditAction {
//     /// Converts this discrete action to a one-hot encoded tensor.
//     ///
//     /// For a 10-armed bandit, this creates a 1D tensor of length 10 where
//     /// the selected arm's position is 1.0 and all others are 0.0.
//     ///
//     /// This conversion enables integration with Burn neural networks for deep
//     /// reinforcement learning applications. The one-hot encoding provides a
//     /// dense representation suitable for network input layers.
//     ///
//     /// # Examples
//     ///
//     /// ```rust,ignore
//     /// use burn::backend::NdArray;
//     /// let action = TenArmedBanditAction::from_index(3);
//     /// let device = NdArray::device();
//     /// let tensor = action.to_tensor::<NdArray>(&device);
//     /// // tensor contains [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
//     /// ```
//     fn to_tensor(&self, device: &B::Device) -> Tensor<B, 1> {
//         // Create one-hot encoding: [0, 0, ..., 1, ..., 0] where 1 is at position self.selected_arm
//         let mut one_hot = vec![0.0_f32; Self::ACTION_COUNT];
//         one_hot[self.selected_arm] = 1.0;
//         Tensor::from_floats(one_hot.as_slice(), device)
//     }
// }

/// 10-armed bandit environment
#[derive(Debug, Clone)]
pub struct TenArmedBandit {
    state: TenArmedBanditState,
    steps: usize,
    done: bool,
    config: TenArmedBanditConfig,
    rng: StdRng,
    // Store the true means q*(a) for the 10 arms
    arm_means: [f32; 10],
}

impl Display for TenArmedBandit {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TenArmedBandit(step={}/{}, done={})",
            self.steps, self.config.max_steps, self.done
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenArmedBanditConfig {
    pub max_steps: usize,
}

impl Default for TenArmedBanditConfig {
    fn default() -> Self {
        Self { max_steps: 500 }
    }
}

// Useful for loading environment configurations from strings (e.g., command line arguments or config files)
impl FromStr for TenArmedBanditConfig {
    type Err = String;

    /// Parses a string into a `TenArmedBanditConfig`.
    ///
    /// Supports two formats:
    /// - A single number: `"500"` → `TenArmedBanditConfig { max_steps: 500 }`
    /// - Key-value format: `"max_steps=500"` → `TenArmedBanditConfig { max_steps: 500 }`
    ///
    /// # Errors
    ///
    /// Returns an error if the string cannot be parsed as either format,
    /// or if the numeric value is invalid.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use std::str::FromStr;
    /// use evorl_envs::classic::TenArmedBanditConfig;
    ///
    /// let config1: TenArmedBanditConfig = "500".parse().unwrap();
    /// assert_eq!(config1.max_steps, 500);
    ///
    /// let config2: TenArmedBanditConfig = "max_steps=1000".parse().unwrap();
    /// assert_eq!(config2.max_steps, 1000);
    /// ```
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed = s.trim();

        // Try parsing as just a number first
        if let Ok(max_steps) = trimmed.parse::<usize>() {
            return Ok(Self { max_steps });
        }

        // Try parsing as "max_steps=value" (with optional whitespace around =)
        if let Some(eq_pos) = trimmed.find('=') {
            let key = trimmed[..eq_pos].trim();
            let value_str = trimmed[eq_pos + 1..].trim();

            if key == "max_steps" {
                let max_steps = value_str
                    .parse::<usize>()
                    .map_err(|e| format!("Failed to parse max_steps value: {}", e))?;
                return Ok(Self { max_steps });
            }
        }

        Err(format!(
            "Invalid TenArmedBanditConfig format. Expected either a number or 'max_steps=<number>', got: {}",
            s
        ))
    }
}

// impl Environment<1, 1> for TenArmedBandit {
//     type StateType = TenArmedBanditState;
//     type ActionType = TenArmedBanditAction;
//     type RewardType = f32;
//     type SnapshotType = SnapshotBase<TenArmedBanditState, f32>;

//     fn new(_render: bool) -> Self {
//         // Fixed seed for reproducibility - matches Sutton & Barto convention
//         let seed = 42;
//         let mut rng = StdRng::seed_from_u64(seed);

//         // Sample q*(a) from normal distribution N(0, 1) for each arm
//         // Each arm's true value is a sample from a normal distribution
//         let normal = Normal::new(0.0, 1.0).unwrap();
//         let mut arm_means = [0.0; 10];
//         for mean in &mut arm_means {
//             *mean = normal.sample(&mut rng);
//         }

//         Self {
//             state: TenArmedBanditState {},
//             steps: 0,
//             done: false,
//             config: TenArmedBanditConfig::default(),
//             rng,
//             arm_means,
//         }
//     }

//     fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
//         self.state = TenArmedBanditState {};
//         self.steps = 0;
//         self.done = false;

//         Ok(SnapshotBase::new(self.state, 0.0, false))
//     }

//     fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> {
//         // Validate the action
//         if !action.is_valid() {
//             return Err(EnvironmentError::InvalidAction(format!(
//                 "Invalid arm index: {}",
//                 action.to_index()
//             )));
//         }

//         // Increment step counter
//         self.steps += 1;

//         // In a real bandit environment, we would sample a reward based on the
//         // arm's probability distribution. For now, we return a deterministic
//         // or random reward based on the arm index.
//         let reward = self.compute_reward(action.to_index());

//         // Check if episode is done
//         self.done = self.steps >= self.config.max_steps;

//         Ok(SnapshotBase::new(self.state, reward, self.done))
//     }
// }

// todo! use evorl_core::dynamics::Reward
impl TenArmedBandit {
    /// Computes a reward for the given arm index.
    fn compute_reward(&mut self, arm_index: usize) -> f32 {
        let mean = self.arm_means[arm_index];
        // Sample from N(mean, 1)
        let normal = Normal::new(mean, 1.0).unwrap();
        normal.sample(&mut self.rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fromstr_simple_number() {
        // Test parsing as a simple number
        let config: TenArmedBanditConfig = "500".parse().expect("Failed to parse");
        assert_eq!(config.max_steps, 500);
    }

    #[test]
    fn test_fromstr_with_whitespace() {
        // Test parsing with surrounding whitespace
        let config: TenArmedBanditConfig = "  750  ".parse().expect("Failed to parse");
        assert_eq!(config.max_steps, 750);
    }

    #[test]
    fn test_fromstr_key_value_format() {
        // Test parsing as key-value format
        let config: TenArmedBanditConfig = "max_steps=1000".parse().expect("Failed to parse");
        assert_eq!(config.max_steps, 1000);
    }

    #[test]
    fn test_fromstr_key_value_with_whitespace() {
        // Test parsing key-value format with whitespace
        let config: TenArmedBanditConfig = "max_steps = 2000".parse().expect("Failed to parse");
        assert_eq!(config.max_steps, 2000);
    }

    #[test]
    fn test_fromstr_zero_steps() {
        // Test parsing zero as max_steps
        let config: TenArmedBanditConfig = "0".parse().expect("Failed to parse");
        assert_eq!(config.max_steps, 0);
    }

    #[test]
    fn test_fromstr_large_number() {
        // Test parsing a large number
        let config: TenArmedBanditConfig = "999999999".parse().expect("Failed to parse");
        assert_eq!(config.max_steps, 999999999);
    }

    #[test]
    fn test_fromstr_invalid_format() {
        // Test that invalid format returns error
        let result: Result<TenArmedBanditConfig, String> = "invalid".parse();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("Invalid TenArmedBanditConfig format"));
    }

    #[test]
    fn test_fromstr_invalid_number() {
        // Test that non-numeric input returns error
        let result: Result<TenArmedBanditConfig, String> = "not_a_number".parse();
        assert!(result.is_err());
    }

    #[test]
    fn test_fromstr_key_value_invalid_number() {
        // Test that invalid number in key-value format returns error
        let result: Result<TenArmedBanditConfig, String> = "max_steps=invalid".parse();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("Failed to parse max_steps value"));
    }

    #[test]
    fn test_fromstr_wrong_key() {
        // Test that wrong key name returns error
        let result: Result<TenArmedBanditConfig, String> = "wrong_key=500".parse();
        assert!(result.is_err());
    }

    #[test]
    fn test_config_default() {
        // Test that default config has expected max_steps
        let config = TenArmedBanditConfig::default();
        assert_eq!(config.max_steps, 500);
    }
}
