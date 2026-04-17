use crate::base::{Action, Observation, Reward, State};
use std::collections::BTreeMap;
use std::fmt::Debug;

/// Describes the lifecycle status of an episode at a given step.
///
/// Separating `Terminated` from `Truncated` allows RL algorithms to correctly
/// bootstrap the value function: a truncated episode still has future value,
/// whereas a terminated one does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpisodeStatus {
    /// The episode is still in progress.
    Running,
    /// The episode ended by reaching a terminal MDP state (goal, failure, etc.).
    Terminated,
    /// The episode ended because an external step limit was reached.
    Truncated,
}

impl EpisodeStatus {
    /// `true` when the episode loop should stop (`Terminated` or `Truncated`).
    pub const fn is_done(self) -> bool {
        matches!(self, Self::Terminated | Self::Truncated)
    }

    /// `true` only for intrinsic MDP termination.
    pub const fn is_terminated(self) -> bool {
        matches!(self, Self::Terminated)
    }

    /// `true` only for extrinsic step-limit truncation.
    pub const fn is_truncated(self) -> bool {
        matches!(self, Self::Truncated)
    }
}

/// Named metadata emitted alongside a snapshot.
///
/// Used for shaped / multi-component reward logging. Keys are `&'static str`
/// constants defined in each per-environment module to avoid magic strings at
/// call sites.
#[derive(Debug, Clone, Default)]
pub struct SnapshotMetadata {
    /// Named reward components (e.g. `"ctrl"`, `"goal"`, `"healthy"`).
    pub components: BTreeMap<&'static str, f32>,
}

impl SnapshotMetadata {
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder-style insert.
    pub fn with(mut self, key: &'static str, value: f32) -> Self {
        self.components.insert(key, value);
        self
    }
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

/// Snapshot trait defines the interface for environment state observations.
///
/// A snapshot captures the state of the environment at a single point in time,
/// including the observed state, reward received, and episode status.
/// The required method is `status()`; `is_done`, `is_terminated`, `is_truncated`,
/// and `metadata` are provided as defaults.
pub trait Snapshot<const D: usize>: Debug {
    type ObservationType: Observation<D>;

    /// The type of reward contained in this snapshot.
    type RewardType: Reward;

    /// Access the observed state.
    fn observation(&self) -> &Self::ObservationType;

    /// Access the reward received.
    fn reward(&self) -> &Self::RewardType;

    /// Episode lifecycle status for this step.
    fn status(&self) -> EpisodeStatus;

    /// `true` when the episode loop should stop.
    fn is_done(&self) -> bool {
        self.status().is_done()
    }

    /// `true` only for intrinsic MDP termination.
    fn is_terminated(&self) -> bool {
        self.status().is_terminated()
    }

    /// `true` only for extrinsic step-limit truncation.
    fn is_truncated(&self) -> bool {
        self.status().is_truncated()
    }

    /// Optional named reward components and position data.
    fn metadata(&self) -> Option<&SnapshotMetadata> {
        None
    }
}

/// Default snapshot implementation for standard reinforcement learning observations.
///
/// `SnapshotBase` stores an observation, reward, and [`EpisodeStatus`].
/// Construct via the named constructors (`running`, `terminated`, `truncated`)
/// rather than the deprecated `new(obs, reward, done: bool)`.
///
/// # Type Parameters
///
/// * `D` - The observation tensor rank
/// * `ObservationType` - The type of observation (must implement `Observation<D>`)
/// * `RewardType` - The type of reward (must implement `Reward`)
#[derive(Debug, Clone)]
pub struct SnapshotBase<const D: usize, ObservationType: Observation<D>, RewardType: Reward> {
    /// The observation derived from the state.
    pub observation: ObservationType,
    /// The reward received from the last action.
    pub reward: RewardType,
    /// Episode lifecycle status.
    pub status: EpisodeStatus,
}

impl<const D: usize, ObservationType: Observation<D>, RewardType: Reward>
    SnapshotBase<D, ObservationType, RewardType>
{
    /// Snapshot for a step where the episode is still running.
    pub fn running(observation: ObservationType, reward: RewardType) -> Self {
        Self { observation, reward, status: EpisodeStatus::Running }
    }

    /// Snapshot for the step on which the MDP reached a terminal state.
    pub fn terminated(observation: ObservationType, reward: RewardType) -> Self {
        Self { observation, reward, status: EpisodeStatus::Terminated }
    }

    /// Snapshot for the step on which an external step limit was reached.
    pub fn truncated(observation: ObservationType, reward: RewardType) -> Self {
        Self { observation, reward, status: EpisodeStatus::Truncated }
    }

    /// Create a snapshot from a raw `done: bool`.
    ///
    /// Prefer the named constructors. This exists for migration compatibility;
    /// `done = true` maps to `Terminated` (not `Truncated`).
    #[deprecated(since = "0.2.0", note = "use SnapshotBase::running / ::terminated / ::truncated")]
    pub fn new(observation: ObservationType, reward: RewardType, done: bool) -> Self {
        let status = if done { EpisodeStatus::Terminated } else { EpisodeStatus::Running };
        Self { observation, reward, status }
    }
}

impl<const D: usize, ObservationType: Observation<D>, RewardType: Reward> Snapshot<D>
    for SnapshotBase<D, ObservationType, RewardType>
{
    type ObservationType = ObservationType;
    type RewardType = RewardType;

    fn observation(&self) -> &Self::ObservationType {
        &self.observation
    }

    fn reward(&self) -> &Self::RewardType {
        &self.reward
    }

    fn status(&self) -> EpisodeStatus {
        self.status
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
pub trait Environment<const D: usize, const SD: usize, const AD: usize> {
    /// The concrete state type for this environment.
    type StateType: State<SD>;

    type ObservationType: Observation<D>;

    /// The concrete action type this environment accepts.
    type ActionType: Action<AD>;

    /// The reward scalar type returned by this environment.
    type RewardType: Reward;

    /// The snapshot type returned by reset and step operations.
    type SnapshotType: Snapshot<
        D,
        ObservationType = Self::ObservationType,
        RewardType = Self::RewardType,
    >;

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

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};

    use super::*;
    use crate::action::DiscreteAction;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub struct MockObservation {
        /// The agent's current position in the range [0, 6]
        position: i32,
    }

    impl Default for MockObservation {
        fn default() -> Self {
            Self { position: 0 }
        }
    }

    impl Observation<1> for MockObservation {
        fn shape() -> [usize; 1] {
            [1]
        }
    }

    // Mock types for testing using Random Walk (1D) environment with 7 states
    // States: 0, 1, 2, 3, 4, 5, 6 (representing positions on a 1D line)
    // Actions: 0 = move left, 1 = move right
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct MockState {
        /// The agent's current position in the range [0, 6]
        position: i32,
    }

    impl MockState {
        fn new(position: i32) -> Self {
            Self { position }
        }

        /// Check if position is within valid bounds
        fn is_in_bounds(position: i32) -> bool {
            position >= 0 && position <= 6
        }
    }

    impl State<1> for MockState {
        type Observation = MockObservation;
        fn numel(&self) -> usize {
            7
        }

        fn shape() -> [usize; 1] {
            [7]
        }

        fn is_valid(&self) -> bool {
            Self::is_in_bounds(self.position)
        }

        fn observe(&self) -> Self::Observation {
            MockObservation {
                position: self.position,
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum MockAction {
        MoveLeft,  // position -= 1
        MoveRight, // position +=1
    }

    impl Action<1> for MockAction {
        fn is_valid(&self) -> bool {
            true // any instance of the enum is a valid action
        }

        fn shape() -> [usize; 1] {
            [1]
        }
    }

    impl DiscreteAction<1> for MockAction {
        const ACTION_COUNT: usize = 2;
        fn from_index(index: usize) -> Self {
            match index {
                0 => MockAction::MoveLeft,
                1 => MockAction::MoveRight,
                _ => panic!("Unknown action index: {}", index),
            }
        }

        fn to_index(&self) -> usize {
            match self {
                MockAction::MoveLeft => 0,
                MockAction::MoveRight => 1,
            }
        }
    }

    use crate::reward::ScalarReward;

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

    impl Environment<1, 1, 1> for MockEnvironment {
        type StateType = MockState;
        type ObservationType = MockObservation;
        type ActionType = MockAction;
        type RewardType = ScalarReward;
        type SnapshotType = SnapshotBase<1, MockObservation, ScalarReward>;

        fn new(render: bool) -> Self {
            Self::with_defaults(render)
        }

        fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
            self.current_state = MockState::new(Self::START_STATE);
            self.step_count = 0;
            Ok(SnapshotBase::running(self.current_state.observe(), ScalarReward(0.0)))
        }

        fn step(
            &mut self,
            action: Self::ActionType,
        ) -> Result<Self::SnapshotType, EnvironmentError> {
            if !action.is_valid() {
                return Err(EnvironmentError::InvalidAction(format!(
                    "Invalid action: {:?}.",
                    action
                )));
            }

            // Update state based on action
            let next_position = if action == MockAction::MoveLeft {
                self.current_state.position - 1 // move left one step
            } else {
                self.current_state.position + 1 // move right one step
            };

            // Check boundaries: valid positions are [0, 6]
            let (new_state, reward, terminated) = if next_position < 0 {
                (MockState::new(0), -1.0, true)
            } else if next_position > 6 {
                (MockState::new(6), -1.0, true)
            } else {
                let new_state = MockState::new(next_position);
                let reward = if next_position == Self::GOAL_STATE { 1.0 } else { 0.0 };
                let done = next_position == Self::GOAL_STATE;
                (new_state, reward, done)
            };

            self.current_state = new_state;
            self.step_count += 1;

            let status = if terminated {
                EpisodeStatus::Terminated
            } else if self.step_count >= self.max_steps {
                EpisodeStatus::Truncated
            } else {
                EpisodeStatus::Running
            };

            Ok(SnapshotBase { observation: new_state.observe(), reward: ScalarReward(reward), status })
        }
    }

    // Custom snapshot implementation for advanced testing
    #[derive(Debug, Clone)]
    pub struct CustomSnapshot {
        observation: MockObservation,
        reward: ScalarReward,
        status: EpisodeStatus,
        step_count: usize,
        cumulative_reward: f32,
    }

    impl Snapshot<1> for CustomSnapshot {
        type ObservationType = MockObservation;
        type RewardType = ScalarReward;

        fn observation(&self) -> &MockObservation {
            &self.observation
        }

        fn reward(&self) -> &ScalarReward {
            &self.reward
        }

        fn status(&self) -> EpisodeStatus {
            self.status
        }
    }

    // Tests for Snapshot trait
    #[test]
    fn test_snapshot_base_creation() {
        let obs = MockObservation { position: 42 };
        let snapshot = SnapshotBase::running(obs, ScalarReward(1.5));

        assert_eq!(snapshot.observation(), &obs);
        assert_eq!(snapshot.reward(), &ScalarReward(1.5));
        assert!(!snapshot.is_done());
        assert_eq!(snapshot.status(), EpisodeStatus::Running);
    }

    #[test]
    fn test_snapshot_base_terminal() {
        let obs = MockObservation { position: 0 };
        let snapshot = SnapshotBase::terminated(obs, ScalarReward(-1.0));

        assert!(snapshot.is_done());
        assert!(snapshot.is_terminated());
        assert!(!snapshot.is_truncated());
        assert_eq!(snapshot.reward(), &ScalarReward(-1.0));
    }

    #[test]
    fn test_snapshot_base_clone() {
        let obs = MockObservation { position: 10 };
        let snapshot1 = SnapshotBase::running(obs, ScalarReward(0.5));
        let snapshot2 = snapshot1.clone();

        assert_eq!(snapshot1.observation(), snapshot2.observation());
        assert_eq!(snapshot1.reward(), snapshot2.reward());
        assert_eq!(snapshot1.is_done(), snapshot2.is_done());
    }

    #[test]
    fn test_snapshot_debug() {
        let obs = MockObservation { position: 5 };
        let snapshot = SnapshotBase::terminated(obs, ScalarReward(2.0));
        let debug_str = format!("{:?}", snapshot);

        assert!(debug_str.contains("SnapshotBase"));
        assert!(debug_str.contains("position: 5"));
        assert!(debug_str.contains("reward: ScalarReward(2.0)"));
        assert!(debug_str.contains("Terminated"));
    }

    // Tests for custom Snapshot implementations
    #[test]
    fn test_custom_snapshot_trait_impl() {
        let snapshot = CustomSnapshot {
            observation: MockObservation { position: 1 },
            reward: ScalarReward(10.0),
            status: EpisodeStatus::Running,
            step_count: 5,
            cumulative_reward: 25.0,
        };

        // Verify trait method access
        assert_eq!(snapshot.observation().position, 1);
        assert_eq!(snapshot.reward(), &ScalarReward(10.0));
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

        assert_eq!(snapshot.observation().position, 3);
        assert_eq!(snapshot.reward(), &ScalarReward(0.0));
        assert!(!snapshot.is_done());
    }

    #[test]
    fn test_environment_step_valid_action() {
        let mut env = MockEnvironment::new(false);
        env.reset().expect("Reset should succeed");

        let action = MockAction::MoveRight;
        let snapshot = env
            .step(action)
            .expect("Step with valid action should succeed");

        assert_eq!(snapshot.observation().position, 4);
        assert_eq!(snapshot.reward(), &ScalarReward(0.0));
    }

    #[test]
    fn test_environment_episode_termination() {
        let mut env = MockEnvironment::new(false);
        env.reset().expect("Reset should succeed");
        env.current_state.position = 0;

        // Move right toward the goal (state 6)
        for i in 0..6 {
            let action = MockAction::MoveRight;
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
            let action = MockAction::MoveRight;
            let _ = env.step(action);
        }

        // Reset and verify state is cleared
        let snapshot = env.reset().expect("Second reset should succeed");
        assert_eq!(snapshot.observation().position, 3);
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
                let action = MockAction::MoveRight;
                snapshot = env.step(action).expect("Step should succeed");
                step += 1;
            }
        }
    }

    #[test]
    fn test_snapshot_reward_conversion() {
        let observation = MockObservation { position: 1 };
        let snapshot = SnapshotBase::new(observation, ScalarReward(42.5), false);

        // RewardType implements Into<f32>
        let reward_as_f32: f32 = snapshot.reward().clone().into();
        assert_eq!(reward_as_f32, 42.5);
    }
}
