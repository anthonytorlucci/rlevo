use crate::action::{Action, ActionTensorConvertible};
use crate::state::{State, StateTensorConvertible};
use std::fmt::Debug;

/// Snapshot trait defines the interface for environment state observations.
///
/// A snapshot captures the state of the environment at a single point in time,
/// including the observed state, reward received, and episode termination flag.
/// This trait allows for custom snapshot implementations while providing sensible defaults.
///
/// # Examples
///
/// ```no_run
/// use evorl_core::environment::Snapshot;
///
/// // Using the default SnapshotBase implementation
/// let snapshot: Box<dyn Snapshot<StateType = MyState, RewardType = f32>> =
///     Box::new(SnapshotBase::new(state, 1.0, false));
///
/// println!("State: {:?}", snapshot.state());
/// println!("Reward: {}", snapshot.reward());
/// println!("Done: {}", snapshot.is_done());
/// ```
pub trait Snapshot: Debug + Clone {
    /// The type of state contained in this snapshot.
    type StateType: State + Debug + Clone;

    /// The type of reward contained in this snapshot.
    type RewardType: Into<f32> + Debug + Clone;

    /// Access the observed state.
    ///
    /// # Returns
    /// A reference to the state captured in this snapshot.
    fn state(&self) -> &Self::StateType;

    /// Access the reward received.
    ///
    /// # Returns
    /// A reference to the reward value.
    fn reward(&self) -> &Self::RewardType;

    /// Check if the episode is terminal.
    ///
    /// # Returns
    /// `true` if this snapshot represents the end of an episode, `false` otherwise.
    fn is_done(&self) -> bool;
}

/// Default snapshot implementation for standard reinforcement learning observations.
///
/// `SnapshotBase` is a generic struct that stores the essential components of an
/// environment observation: the current state, the reward received, and whether
/// the episode is complete. It implements the `Snapshot` trait for ergonomic use.
///
/// # Type Parameters
///
/// * `StateType` - The type of state (must implement `State` and `Clone`)
/// * `RewardType` - The type of reward (must implement `Into<f32>` and `Clone`)
///
/// # Examples
///
/// ```no_run
/// # use evorl_core::environment::SnapshotBase;
/// # struct MyState;
/// # impl evorl_core::state::State for MyState {
/// #     fn numel(&self) -> usize { 0 }
/// #     fn shape(&self) -> Vec<usize> { vec![] }
/// # }
/// # impl Clone for MyState { fn clone(&self) -> Self { MyState } }
/// # impl std::fmt::Debug for MyState { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { Ok(()) } }
/// # impl PartialEq for MyState { fn eq(&self, _: &Self) -> bool { true } }
/// # impl Eq for MyState {}
/// # impl std::hash::Hash for MyState { fn hash<H: std::hash::Hasher>(&self, _: &mut H) {} }
///
/// let state = MyState;
/// let snapshot = SnapshotBase::new(state, 1.5, false);
///
/// assert_eq!(snapshot.reward(), &1.5);
/// assert!(!snapshot.is_done());
/// ```
#[derive(Debug, Clone)]
pub struct SnapshotBase<StateType: State + Debug + Clone, RewardType: Into<f32> + Debug + Clone> {
    /// The observed state of the environment.
    pub state: StateType,
    /// The reward received from the last action.
    pub reward: RewardType,
    /// Whether the episode has terminated.
    pub done: bool,
}

impl<StateType: State + Debug + Clone, RewardType: Into<f32> + Debug + Clone>
    SnapshotBase<StateType, RewardType>
{
    /// Create a new snapshot with the given state, reward, and terminal status.
    ///
    /// # Arguments
    ///
    /// * `state` - The environment state at this observation
    /// * `reward` - The reward received
    /// * `done` - Whether the episode has terminated
    pub fn new(state: StateType, reward: RewardType, done: bool) -> Self {
        Self {
            state,
            reward,
            done,
        }
    }
}

impl<StateType: State + Debug + Clone, RewardType: Into<f32> + Debug + Clone> Snapshot
    for SnapshotBase<StateType, RewardType>
{
    type StateType = StateType;
    type RewardType = RewardType;

    fn state(&self) -> &Self::StateType {
        &self.state
    }

    fn reward(&self) -> &Self::RewardType {
        &self.reward
    }

    fn is_done(&self) -> bool {
        self.done
    }
}

/// The environment trait defines the interaction protocol between an agent and a problem domain.
///
/// An environment encapsulates the dynamics of a problem, processing actions and
/// returning observations (snapshots) along with rewards. Environments are responsible
/// for managing state, computing rewards, and determining episode termination.
///
/// # Type Parameters
///
/// * `S` - The dimensionality of the state tensor representation
/// * `A` - The dimensionality of the action tensor representation
///
/// # Associated Types
///
/// * `StateType` - The concrete state type of this environment
/// * `ActionType` - The concrete action type this environment accepts
/// * `RewardType` - The reward scalar type
/// * `SnapshotType` - The snapshot type returned by reset/step
///
/// # Examples
///
/// ```no_run
/// use evorl_core::environment::Environment;
///
/// # struct MyEnv;
/// # struct MyState;
/// # impl evorl_core::state::State for MyState {
/// #     fn numel(&self) -> usize { 0 }
/// #     fn shape(&self) -> Vec<usize> { vec![] }
/// # }
/// # impl Clone for MyState { fn clone(&self) -> Self { MyState } }
/// # impl std::fmt::Debug for MyState { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { Ok(()) } }
/// # impl PartialEq for MyState { fn eq(&self, _: &Self) -> bool { true } }
/// # impl Eq for MyState {}
/// # impl std::hash::Hash for MyState { fn hash<H: std::hash::Hasher>(&self, _: &mut H) {} }
/// # struct MyAction;
/// # impl evorl_core::action::Action for MyAction {
/// #     fn is_valid(&self) -> bool { true }
/// # }
/// # impl Clone for MyAction { fn clone(&self) -> Self { MyAction } }
/// # impl std::fmt::Debug for MyAction { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { Ok(()) } }
/// # impl Sized for MyAction {}
///
/// impl Environment<1, 1> for MyEnv {
///     type StateType = MyState;
///     type ActionType = MyAction;
///     type RewardType = f32;
///     type SnapshotType = evorl_core::environment::SnapshotBase<MyState, f32>;
///
///     fn new(_render: bool) -> Self { MyEnv }
///     fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
///         Ok(evorl_core::environment::SnapshotBase::new(MyState, 0.0, false))
///     }
///     fn step(&mut self, _action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> {
///         Ok(evorl_core::environment::SnapshotBase::new(MyState, 1.0, true))
///     }
/// }
/// ```
pub trait Environment<const S: usize, const A: usize> {
    /// The concrete state type for this environment.
    type StateType: State + StateTensorConvertible<S> + Debug + Clone;

    /// The concrete action type this environment accepts.
    type ActionType: Action + ActionTensorConvertible<A> + Debug + Clone;

    /// The reward scalar type returned by this environment.
    type RewardType: Into<f32> + Debug + Clone;

    /// The snapshot type returned by reset and step operations.
    type SnapshotType: Snapshot<StateType = Self::StateType, RewardType = Self::RewardType>;

    /// Create a new environment instance.
    ///
    /// # Arguments
    ///
    /// * `render` - Whether to render/display the environment (if supported)
    ///
    /// # Returns
    ///
    /// A new instance of this environment.
    fn new(render: bool) -> Self;

    /// Reset the environment to its initial state.
    ///
    /// This method should reset all state and return an initial observation (snapshot)
    /// of the environment. This is typically called at the start of each episode.
    ///
    /// # Returns
    ///
    /// A snapshot containing the initial state, reward (typically 0), and done=false,
    /// or an error if reset fails.
    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError>;

    /// Execute one step of the environment with the given action.
    ///
    /// This method processes the action, updates internal state, and returns
    /// an observation of the new state along with the reward received.
    ///
    /// # Arguments
    ///
    /// * `action` - The action to execute in the current state
    ///
    /// # Returns
    ///
    /// A snapshot containing the next state, reward, and done flag,
    /// or an error if the step fails.
    fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError>;
}

/// Error type for environment operations.
///
/// `EnvironmentError` captures failures that can occur during environment
/// initialization, reset, or stepping. It provides detailed error messages
/// and supports error chaining via the standard `Error` trait.
///
/// # Variants
///
/// * `InvalidAction` - The provided action is not valid in the current state
/// * `RenderFailed` - Rendering/display operation failed
/// * `IoError` - An I/O operation failed (wrapped std::io::Error)
#[derive(Debug)]
pub enum EnvironmentError {
    /// An invalid or out-of-bounds action was provided.
    InvalidAction(String),
    /// Rendering or display failed.
    RenderFailed(String),
    /// An I/O operation failed (wraps std::io::Error).
    IoError(std::io::Error),
}

impl std::error::Error for EnvironmentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            EnvironmentError::IoError(io_err) => Some(io_err),
            _ => None,
        }
    }
}

impl std::fmt::Display for EnvironmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EnvironmentError::InvalidAction(action_error) => {
                write!(f, "Invalid action: {}", action_error)
            }
            EnvironmentError::RenderFailed(render_error) => {
                write!(f, "Render failed: {}", render_error)
            }
            EnvironmentError::IoError(io_err) => {
                write!(f, "IO operation failed: {}", io_err)
            }
        }
    }
}

impl From<std::io::Error> for EnvironmentError {
    fn from(error: std::io::Error) -> Self {
        EnvironmentError::IoError(error)
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::backend::Backend;
    use burn::tensor::{Tensor, TensorData};

    use super::*;
    use crate::state::StateError;
    use std::hash::Hash;

    // Mock types for testing using Random Walk (1D) environment with 7 states
    // States: 0, 1, 2, 3, 4, 5, 6 (representing positions on a 1D line)
    // Actions: 0 = move left, 1 = move right
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    struct MockState {
        /// The agent's current position in the range [0, 6]
        value: i32,
    }

    impl MockState {
        fn new(value: i32) -> Self {
            Self { value }
        }

        /// Check if position is within valid bounds
        fn is_in_bounds(value: i32) -> bool {
            value >= 0 && value <= 6
        }
    }

    impl State for MockState {
        fn numel(&self) -> usize {
            7
        }

        fn shape(&self) -> Vec<usize> {
            vec![7]
        }

        fn is_valid(&self) -> bool {
            Self::is_in_bounds(self.value)
        }
    }

    impl StateTensorConvertible<1> for MockState {
        fn to_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 1> {
            // Create a one-hot encoded 1D tensor with 7 elements
            // Position `value` is set to 1.0, all others are 0.0
            let mut u: [f32; 7] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            if Self::is_in_bounds(self.value) {
                u[self.value as usize] = 1.0;
            }
            Tensor::<B, 1>::from_data(u, device)
        }

        fn from_tensor<B: Backend>(tensor: &Tensor<B, 1>) -> Result<Self, StateError>
        where
            Self: Sized,
        {
            let tensor_data: TensorData = tensor.to_data();

            if tensor_data.shape[0] != 7 {
                return Err(StateError::InvalidSize {
                    expected: 7,
                    got: tensor_data.shape[0],
                });
            }

            let values: Vec<f32> = tensor_data.to_vec().unwrap();

            // Find the index where the value is 1.0 (one-hot encoding)
            let index = values
                .iter()
                .position(|&v| (v - 1.0).abs() < 1e-6)
                .ok_or_else(|| {
                    StateError::InvalidData(
                        "Expected one-hot encoded tensor with exactly one 1.0 value".to_string(),
                    )
                })?;

            Ok(MockState {
                value: index as i32,
            })
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct MockAction {
        /// 0 -> move left (position -= 1), 1 -> move right (position += 1)
        value: i32,
    }

    impl MockAction {
        /// Create a new action. Value should be 0 (left) or 1 (right).
        fn new(value: i32) -> Self {
            Self { value }
        }
    }

    impl Action for MockAction {
        fn is_valid(&self) -> bool {
            self.value == 0 || self.value == 1
        }
    }

    impl ActionTensorConvertible<1> for MockAction {
        fn to_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 1> {
            // Create a 1D tensor with a single element representing the action
            Tensor::<B, 1>::from_data([self.value as f32], device)
        }
    }

    // Mock environment for testing: 1D random walk with 7 states
    // The agent starts at position 3 (middle) and can move left or right.
    // Episode terminates after 20 steps or when reaching boundaries (state 0 or 6).
    // Reward: +1.0 for reaching the goal (state 6), -1.0 for falling off left (state < 0), 0.0 otherwise.
    struct MockEnvironment {
        current_state: MockState,
        step_count: usize,
        max_steps: usize,
    }

    impl MockEnvironment {
        const START_STATE: i32 = 3;
        const MAX_STEPS: usize = 20;
        const GOAL_STATE: i32 = 6;

        fn with_defaults(_render: bool) -> Self {
            Self {
                current_state: MockState::new(Self::START_STATE),
                step_count: 0,
                max_steps: Self::MAX_STEPS,
            }
        }
    }

    impl Environment<1, 1> for MockEnvironment {
        type StateType = MockState;
        type ActionType = MockAction;
        type RewardType = f32;
        type SnapshotType = SnapshotBase<MockState, f32>;

        fn new(render: bool) -> Self {
            Self::with_defaults(render)
        }

        fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
            self.current_state = MockState::new(Self::START_STATE);
            self.step_count = 0;
            Ok(SnapshotBase::new(self.current_state, 0.0, false))
        }

        fn step(
            &mut self,
            action: Self::ActionType,
        ) -> Result<Self::SnapshotType, EnvironmentError> {
            if !action.is_valid() {
                return Err(EnvironmentError::InvalidAction(format!(
                    "Invalid action value: {}. Expected 0 (left) or 1 (right).",
                    action.value
                )));
            }

            // Update state based on action: 0 = move left, 1 = move right
            let next_position = if action.value == 0 {
                self.current_state.value - 1
            } else {
                self.current_state.value + 1
            };

            // Check boundaries: valid positions are [0, 6]
            let (new_state, reward, done) = if next_position < 0 {
                // Fell off the left boundary
                (MockState::new(0), -1.0, true)
            } else if next_position > 6 {
                // Fell off the right boundary
                (MockState::new(6), -1.0, true)
            } else {
                // Within bounds
                let new_state = MockState::new(next_position);
                let reward = if next_position == Self::GOAL_STATE {
                    1.0 // Reached the goal!
                } else {
                    0.0 // Step cost-free movement
                };
                let episode_done = next_position == Self::GOAL_STATE; // Episode ends upon reaching goal
                (new_state, reward, episode_done)
            };

            self.current_state = new_state;
            self.step_count += 1;

            // Episode also terminates after max steps
            let done = done || (self.step_count >= self.max_steps);

            Ok(SnapshotBase::new(new_state, reward, done))
        }
    }

    // Custom snapshot implementation for advanced testing
    #[derive(Debug, Clone)]
    struct CustomSnapshot {
        state: MockState,
        reward: f32,
        done: bool,
        step_count: usize,
        cumulative_reward: f32,
    }

    impl Snapshot for CustomSnapshot {
        type StateType = MockState;
        type RewardType = f32;

        fn state(&self) -> &Self::StateType {
            &self.state
        }

        fn reward(&self) -> &Self::RewardType {
            &self.reward
        }

        fn is_done(&self) -> bool {
            self.done
        }
    }

    // Tests for Snapshot trait
    #[test]
    fn test_snapshot_base_creation() {
        let state = MockState { value: 42 };
        let snapshot = SnapshotBase::new(state, 1.5, false);

        assert_eq!(snapshot.state(), &state);
        assert_eq!(snapshot.reward(), &1.5);
        assert!(!snapshot.is_done());
    }

    #[test]
    fn test_snapshot_base_terminal() {
        let state = MockState { value: 0 };
        let snapshot = SnapshotBase::new(state, -1.0, true);

        assert!(snapshot.is_done());
        assert_eq!(snapshot.reward(), &-1.0);
    }

    #[test]
    fn test_snapshot_base_clone() {
        let state = MockState { value: 10 };
        let snapshot1 = SnapshotBase::new(state, 0.5, false);
        let snapshot2 = snapshot1.clone();

        assert_eq!(snapshot1.state(), snapshot2.state());
        assert_eq!(snapshot1.reward(), snapshot2.reward());
        assert_eq!(snapshot1.is_done(), snapshot2.is_done());
    }

    #[test]
    fn test_snapshot_debug() {
        let state = MockState { value: 5 };
        let snapshot = SnapshotBase::new(state, 2.0, true);
        let debug_str = format!("{:?}", snapshot);

        assert!(debug_str.contains("SnapshotBase"));
        assert!(debug_str.contains("value: 5"));
        assert!(debug_str.contains("reward: 2.0"));
        assert!(debug_str.contains("done: true"));
    }

    // Tests for custom Snapshot implementations
    #[test]
    fn test_custom_snapshot_trait_impl() {
        let snapshot = CustomSnapshot {
            state: MockState { value: 1 },
            reward: 10.0,
            done: false,
            step_count: 5,
            cumulative_reward: 25.0,
        };

        // Verify trait method access
        assert_eq!(snapshot.state().value, 1);
        assert_eq!(snapshot.reward(), &10.0);
        assert!(!snapshot.is_done());

        // Verify custom fields are accessible
        assert_eq!(snapshot.step_count, 5);
        assert_eq!(snapshot.cumulative_reward, 25.0);
    }

    // Tests for Environment trait
    #[test]
    fn test_environment_creation() {
        let env = MockEnvironment::new(false);
        assert_eq!(env.step_count, 0);
    }

    #[test]
    fn test_environment_reset() {
        let mut env = MockEnvironment::new(false);
        let snapshot = env.reset().expect("Reset should succeed");

        assert_eq!(snapshot.state().value, 3);
        assert_eq!(snapshot.reward(), &0.0);
        assert!(!snapshot.is_done());
    }

    #[test]
    fn test_environment_step_valid_action() {
        let mut env = MockEnvironment::new(false);
        env.reset().expect("Reset should succeed");

        let action = MockAction::new(1);
        let snapshot = env
            .step(action)
            .expect("Step with valid action should succeed");

        assert_eq!(snapshot.state().value, 4);
        assert_eq!(snapshot.reward(), &0.0);
    }

    #[test]
    fn test_environment_step_invalid_action() {
        let mut env = MockEnvironment::new(false);
        env.reset().expect("Reset should succeed");

        let action = MockAction::new(15); // Out of range
        let result = env.step(action);

        assert!(result.is_err());
        match result.unwrap_err() {
            EnvironmentError::InvalidAction(msg) => {
                assert!(msg.contains("Invalid action"));
            }
            _ => panic!("Expected InvalidAction error"),
        }
    }

    #[test]
    fn test_environment_episode_termination() {
        let mut env = MockEnvironment::new(false);
        env.reset().expect("Reset should succeed");

        // Move right toward the goal (state 6)
        for i in 0..6 {
            let action = MockAction::new(1);
            let snapshot = env.step(action).expect("Step should succeed");

            if i < 5 {
                assert!(
                    !snapshot.is_done(),
                    "Episode should not be done before reaching goal"
                );
            } else {
                assert!(
                    snapshot.is_done(),
                    "Episode should be done upon reaching goal"
                );
            }
        }
    }

    #[test]
    fn test_environment_reset_clears_state() {
        let mut env = MockEnvironment::new(false);

        // Run for 5 steps
        env.reset().expect("Reset should succeed");
        for _ in 0..5 {
            let action = MockAction::new(1);
            let _ = env.step(action);
        }

        // Reset and verify state is cleared
        let snapshot = env.reset().expect("Second reset should succeed");
        assert_eq!(snapshot.state().value, 3);
        assert!(!snapshot.is_done());
    }

    #[test]
    fn test_environment_error_display() {
        let error = EnvironmentError::InvalidAction("test action".to_string());
        let display_str = format!("{}", error);
        assert!(display_str.contains("Invalid action"));
        assert!(display_str.contains("test action"));
    }

    #[test]
    fn test_environment_error_io_conversion() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let env_error = EnvironmentError::from(io_error);

        match env_error {
            EnvironmentError::IoError(_) => {
                // Expected
            }
            _ => panic!("Expected IoError variant"),
        }
    }

    #[test]
    fn test_environment_error_source() {
        let io_error = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
        let env_error = EnvironmentError::IoError(io_error);

        use std::error::Error;
        assert!(env_error.source().is_some());
    }

    // #[test]
    // fn test_snapshot_trait_object() {
    //     let state = MockState { value: 7 };
    //     let snapshot: Box<dyn Snapshot<StateType = MockState, RewardType = f32>> =
    //         Box::new(SnapshotBase::new(state, 3.5, false));

    //     assert_eq!(snapshot.state().value, 7);
    //     assert_eq!(snapshot.reward(), &3.5);
    //     assert!(!snapshot.is_done());
    // }

    // #[test]
    // fn test_snapshot_multiple_types() {
    //     let base_snapshot: Box<dyn Snapshot<StateType = MockState, RewardType = f32>> =
    //         Box::new(SnapshotBase::new(MockState { value: 1 }, 1.0, false));

    //     let rich_snapshot: Box<dyn Snapshot<StateType = MockState, RewardType = f32>> =
    //         Box::new(RichSnapshot {
    //             state: MockState { value: 2 },
    //             reward: 2.0,
    //             done: false,
    //             step_count: 0,
    //             cumulative_reward: 2.0,
    //         });

    //     // Both snapshot types implement the trait
    //     assert_eq!(base_snapshot.state().value, 1);
    //     assert_eq!(rich_snapshot.state().value, 2);
    // }

    #[test]
    fn test_environment_multiple_episodes() {
        let mut env = MockEnvironment::new(false);

        for episode in 0..3 {
            let mut snapshot = env.reset().expect("Reset should succeed");
            let mut step = 0;

            while !snapshot.is_done() && step < 5 {
                let action = MockAction::new(1);
                snapshot = env.step(action).expect("Step should succeed");
                step += 1;
            }
        }
    }

    #[test]
    fn test_snapshot_reward_conversion() {
        let state = MockState { value: 1 };
        let snapshot = SnapshotBase::new(state, 42.5_f32, false);

        // RewardType implements Into<f32>
        let reward_as_f32: f32 = snapshot.reward().clone().into();
        assert_eq!(reward_as_f32, 42.5);
    }
}
