//! Environment interaction protocol and snapshot types.
//!
//! This module defines the contract between an agent and a problem domain:
//! - [`Environment`] — core trait with `reset` / `step` methods
//! - [`Snapshot`] — per-step result carrying observation, reward, and status
//! - [`EpisodeStatus`] — distinguishes running, terminated, and truncated episodes
//! - [`SnapshotMetadata`] — optional named reward components and 3D positions
//!
//! [`SnapshotMetadata`]: crate::environment::SnapshotMetadata

use crate::base::{Action, Observation};
use crate::reward::ScalarReward;
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
    #[must_use]
    pub const fn is_done(self) -> bool {
        matches!(self, Self::Terminated | Self::Truncated)
    }

    /// `true` only for intrinsic MDP termination.
    #[must_use]
    pub const fn is_terminated(self) -> bool {
        matches!(self, Self::Terminated)
    }

    /// `true` only for extrinsic step-limit truncation.
    #[must_use]
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
    /// Named 3D positions for analysis (e.g. `"torso"`, `"com"`, `"main_body"`).
    pub positions: BTreeMap<&'static str, [f32; 3]>,
}

impl SnapshotMetadata {
    /// Creates an empty `SnapshotMetadata`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder-style insert for a named reward component.
    #[must_use]
    pub fn with(mut self, key: &'static str, value: f32) -> Self {
        self.components.insert(key, value);
        self
    }

    /// Builder-style insert for a named 3D position.
    #[must_use]
    pub fn with_position(mut self, key: &'static str, xyz: [f32; 3]) -> Self {
        self.positions.insert(key, xyz);
        self
    }
}

/// Error type for environment operations.
///
/// `EnvironmentError` captures failures that can occur during environment
/// initialization, reset, or stepping. It provides detailed error messages
/// and supports error chaining via the standard [`std::error::Error`] trait.
///
/// The enum is `#[non_exhaustive]`: downstream `match` expressions must carry a
/// wildcard arm, so a future variant is not a breaking change.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum EnvironmentError {
    /// An invalid or out-of-bounds action was provided.
    #[error("Invalid action: {0}")]
    InvalidAction(String),
    /// Rendering or display failed.
    #[error("Render failed: {0}")]
    RenderFailed(String),
    /// A configuration-domain invariant failed during a lifecycle operation.
    ///
    /// A `reset()` may re-run construction-time work (e.g. rebuilding a
    /// procedural world), so a config-domain invariant — a [`ConfigError`] —
    /// can surface at reset, not only at construction. This variant is kept
    /// **generic** (not tied to any one environment) so any lifecycle method
    /// that re-validates config-domain state can propagate the failure with
    /// `?`, avoiding a stringly-typed re-wrap.
    ///
    /// [`ConfigError`]: crate::config::ConfigError
    #[error("Configuration error: {0}")]
    Config(#[from] crate::config::ConfigError),
    /// `step()` was called after the episode already ended.
    ///
    /// The action itself was legal; the *call sequence* was not. An episode that
    /// has emitted a snapshot with [`Snapshot::is_done`] `== true` is over — the
    /// only valid next lifecycle call is [`Environment::reset`]. Stepping again
    /// would silently resurrect a finished episode (re-entering the MDP from a
    /// terminal state, emitting rewards on a `Running` snapshot), so it is an
    /// error rather than a no-op.
    ///
    /// The variant carries the [`EpisodeStatus`] that ended the episode, so the
    /// caller can distinguish an intrinsic MDP termination
    /// ([`EpisodeStatus::Terminated`]) from a wrapper-imposed truncation
    /// ([`EpisodeStatus::Truncated`]).
    #[error(
        "step() called after the episode ended ({status:?}); call reset() before stepping again"
    )]
    StepAfterEpisodeEnd {
        /// The status that ended the episode (`Terminated` or `Truncated`).
        status: EpisodeStatus,
    },
}

/// The result of one [`reset`](Environment::reset) or
/// [`step`](Environment::step): what the agent sees, what it earned, and
/// whether the episode goes on.
///
/// A struct, not a trait: every environment returns this one type (ADR 0090),
/// so a wrapper such as `TimeLimit` composes with any environment. `O` is the
/// environment's observation type. Construct with the named constructors
/// [`running`](Self::running), [`terminated`](Self::terminated) or
/// [`truncated`](Self::truncated) — each leaves `metadata` as `None` — and
/// attach metadata with [`with_metadata`](Self::with_metadata) (ADR 0042):
///
/// ```
/// # use rlevo_core::environment::{Snapshot, SnapshotMetadata};
/// # use rlevo_core::reward::ScalarReward;
/// # #[derive(Debug, Clone, Copy)]
/// # struct Obs(f32);
/// let snap = Snapshot::running(Obs(0.0), ScalarReward(1.0))
///     .with_metadata(SnapshotMetadata::new().with("ctrl", -0.25));
/// assert_eq!(snap.metadata().unwrap().components["ctrl"], -0.25);
/// ```
#[derive(Debug, Clone)]
pub struct Snapshot<O> {
    /// The observation the agent receives.
    pub observation: O,
    /// The reward received from the last action.
    pub reward: ScalarReward,
    /// Episode lifecycle status.
    pub status: EpisodeStatus,
    /// Optional named reward components and positions for this step.
    pub metadata: Option<SnapshotMetadata>,
}

impl<O> Snapshot<O> {
    /// Snapshot for a step where the episode is still running.
    pub fn running(observation: O, reward: ScalarReward) -> Self {
        Self {
            observation,
            reward,
            status: EpisodeStatus::Running,
            metadata: None,
        }
    }

    /// Snapshot for the step on which the MDP reached a terminal state.
    pub fn terminated(observation: O, reward: ScalarReward) -> Self {
        Self {
            observation,
            reward,
            status: EpisodeStatus::Terminated,
            metadata: None,
        }
    }

    /// Snapshot for the step on which an external step limit was reached.
    pub fn truncated(observation: O, reward: ScalarReward) -> Self {
        Self {
            observation,
            reward,
            status: EpisodeStatus::Truncated,
            metadata: None,
        }
    }

    /// Builder-style attachment of [`SnapshotMetadata`] to a snapshot.
    ///
    /// Chains off any of the named constructors; replaces any metadata already
    /// present.
    #[must_use]
    pub fn with_metadata(mut self, metadata: SnapshotMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Access the observation.
    pub fn observation(&self) -> &O {
        &self.observation
    }

    /// Consume the snapshot and yield its observation by value.
    pub fn into_observation(self) -> O {
        self.observation
    }

    /// Access the reward received.
    pub fn reward(&self) -> &ScalarReward {
        &self.reward
    }

    /// Episode lifecycle status for this step.
    pub fn status(&self) -> EpisodeStatus {
        self.status
    }

    /// `true` when the episode loop should stop.
    pub fn is_done(&self) -> bool {
        self.status.is_done()
    }

    /// `true` only for intrinsic MDP termination.
    pub fn is_terminated(&self) -> bool {
        self.status.is_terminated()
    }

    /// `true` only for extrinsic step-limit truncation.
    pub fn is_truncated(&self) -> bool {
        self.status.is_truncated()
    }

    /// Optional named reward components and position data.
    pub fn metadata(&self) -> Option<&SnapshotMetadata> {
        self.metadata.as_ref()
    }
}

/// Interaction protocol between an agent and a problem domain.
///
/// An environment encapsulates the dynamics of a problem, processing actions and
/// returning observations (snapshots) along with rewards. Environments are responsible
/// for managing state, computing rewards, and determining episode termination.
///
/// # Type Parameters
///
/// * `OR` - Rank of the observation tensor (matches `Observation<OR>`).
/// * `AR` - Rank of the action tensor (matches `Action<AR>`).
///
/// # Associated Types
///
/// * `ObservationType` - The observation type exposed to the agent.
/// * `ActionType`      - The action type this environment accepts.
///
/// [`reset`](Self::reset) and [`step`](Self::step) return a
/// [`Snapshot`] of the observation type; its reward is a [`ScalarReward`].
///
/// # The environment's state
///
/// The contract names no state type. What an environment keeps between steps
/// (positions, velocities, a physics world, an RNG) is a plain type it owns,
/// read by its own `reset` and `step`; nothing generic reads it, so no trait
/// asks for it (ADR 0090).
///
/// # Building the observation
///
/// How an environment turns its state into what the agent sees (the emission
/// model `O` of a POMDP `$\langle S, A, T, R, \Omega, O \rangle$`) is the
/// environment's own business, and not a trait (ADR 0090). It lives on the
/// environment, not on its state, so it can read world context a state value
/// does not hold (ADR 0047): typically an inherent method that `reset` and
/// `step` call. A stochastic observation is drawn there from the
/// environment's own RNG (ADR 0029). Observation variants of one environment
/// (a vector or pixels) are a type parameter on the environment; noise,
/// latency and history are wrappers beside `TimeLimit`.
///
/// # Construction
///
/// Construction is not part of this trait (ADR 0011). An environment enters
/// the evaluation harness through an `EnvFactory`, a closure from a trial's
/// seed to a fresh environment. The workspace's own environments also
/// implement `Default`.
pub trait Environment<const OR: usize, const AR: usize> {
    /// The observation type exposed to the agent.
    type ObservationType: Observation<OR>;

    /// The concrete action type this environment accepts.
    type ActionType: Action<AR>;

    /// Reset the environment to its initial state and return the first snapshot.
    ///
    /// The returned snapshot carries the initial observation, a reward of zero, and
    /// [`EpisodeStatus::Running`]. Call this at the start of every episode before
    /// calling [`step`](Self::step).
    ///
    /// # Errors
    ///
    /// Returns [`EnvironmentError`] if the environment cannot be initialised (e.g.
    /// a configuration invariant that fails when the episode is rebuilt).
    fn reset(&mut self) -> Result<Snapshot<Self::ObservationType>, EnvironmentError>;

    /// Execute one transition of the environment with the given action.
    ///
    /// Applies `action` to the current state, updates internal state, and
    /// returns a snapshot containing the next observation, the immediate reward,
    /// and the new [`EpisodeStatus`]. When [`Snapshot::is_done`] returns `true`
    /// the episode is over; call [`reset`](Self::reset) to begin a new one.
    ///
    /// # Post-terminal contract
    ///
    /// Implementations **must** return [`EnvironmentError::StepAfterEpisodeEnd`]
    /// when `step()` is called after a snapshot whose [`Snapshot::is_done`] is
    /// `true`. A finished episode is not silently resumed, and the terminal
    /// snapshot is not silently repeated: the call sequence is the caller's bug
    /// and is reported as one. Call [`reset`](Self::reset) to begin a new
    /// episode.
    ///
    /// The check belongs at the **top** of `step()`, before any state mutation
    /// or RNG draw, so a rejected call leaves the environment untouched — an
    /// environment's RNG stream is a persistent, observable part of its state,
    /// so a rejected step must not advance it either.
    ///
    /// [`EpisodeGuard`](crate::episode::EpisodeGuard) implements this: hold
    /// one, `check()` it first, `record()` the status of each snapshot emitted,
    /// and `reset()` it in [`reset`](Self::reset).
    ///
    /// # Errors
    ///
    /// Returns [`EnvironmentError::InvalidAction`] if the action is not legal in
    /// the current state, [`EnvironmentError::StepAfterEpisodeEnd`] if the
    /// episode has already ended (see the post-terminal contract above), or
    /// another [`EnvironmentError`] variant if the step cannot complete (e.g. a
    /// physics simulation failure).
    ///
    /// The reward should be finite; see [`ScalarReward`]'s `# Finiteness`
    /// section.
    fn step(
        &mut self,
        action: Self::ActionType,
    ) -> Result<Snapshot<Self::ObservationType>, EnvironmentError>;
}

#[cfg(test)]
mod tests {
    // These tests assert exact round-trip of values that are stored and read
    // back without arithmetic, so bit-exact equality is the property under
    // test; an approximate comparison would weaken them.
    #![allow(clippy::float_cmp)]
    use serde::{Deserialize, Serialize};

    use super::*;
    use crate::action::DiscreteAction;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
    pub struct MockObservation {
        /// The agent's current position in the range [0, 6]
        position: i32,
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
                _ => panic!("Unknown action index: {index}"),
            }
        }

        fn to_index(&self) -> usize {
            match self {
                MockAction::MoveLeft => 0,
                MockAction::MoveRight => 1,
            }
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

        fn observe(&self) -> MockObservation {
            MockObservation {
                position: self.current_state.position,
            }
        }
    }

    impl Default for MockEnvironment {
        fn default() -> Self {
            Self {
                current_state: MockState::new(Self::START_STATE),
                step_count: 0,
                max_steps: Self::MAX_STEPS,
            }
        }
    }

    impl Environment<1, 1> for MockEnvironment {
        type ObservationType = MockObservation;
        type ActionType = MockAction;

        fn reset(&mut self) -> Result<Snapshot<MockObservation>, EnvironmentError> {
            self.current_state = MockState::new(Self::START_STATE);
            self.step_count = 0;
            Ok(Snapshot::running(self.observe(), ScalarReward(0.0)))
        }

        fn step(
            &mut self,
            action: Self::ActionType,
        ) -> Result<Snapshot<MockObservation>, EnvironmentError> {
            if !action.is_valid() {
                return Err(EnvironmentError::InvalidAction(format!(
                    "Invalid action: {action:?}."
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
                let reward = if next_position == Self::GOAL_STATE {
                    1.0
                } else {
                    0.0
                };
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

            let observation = self.observe();
            Ok(Snapshot {
                observation,
                reward: ScalarReward(reward),
                status,
                metadata: None,
            })
        }
    }

    // Tests for Snapshot
    #[test]
    fn test_snapshot_creation() {
        let obs = MockObservation { position: 42 };
        let snapshot = Snapshot::running(obs, ScalarReward(1.5));

        assert_eq!(snapshot.observation(), &obs);
        assert_eq!(snapshot.reward(), &ScalarReward(1.5));
        assert!(!snapshot.is_done());
        assert_eq!(snapshot.status(), EpisodeStatus::Running);
    }

    #[test]
    fn test_snapshot_terminal() {
        let obs = MockObservation { position: 0 };
        let snapshot = Snapshot::terminated(obs, ScalarReward(-1.0));

        assert!(snapshot.is_done());
        assert!(snapshot.is_terminated());
        assert!(!snapshot.is_truncated());
        assert_eq!(snapshot.reward(), &ScalarReward(-1.0));
    }

    #[test]
    fn test_snapshot_clone() {
        let obs = MockObservation { position: 10 };
        let snapshot1 = Snapshot::running(obs, ScalarReward(0.5));
        let snapshot2 = snapshot1.clone();

        assert_eq!(snapshot1.observation(), snapshot2.observation());
        assert_eq!(snapshot1.reward(), snapshot2.reward());
        assert_eq!(snapshot1.is_done(), snapshot2.is_done());
    }

    #[test]
    fn test_snapshot_debug() {
        let obs = MockObservation { position: 5 };
        let snapshot = Snapshot::terminated(obs, ScalarReward(2.0));
        let debug_str = format!("{snapshot:?}");

        assert!(debug_str.contains("Snapshot"));
        assert!(debug_str.contains("position: 5"));
        assert!(debug_str.contains("reward: ScalarReward(2.0)"));
        assert!(debug_str.contains("Terminated"));
    }

    // Tests for Environment trait
    #[test]
    fn test_environment_creation() {
        let env = MockEnvironment::default();
        assert_eq!(env.step_count, 0);
    }

    #[test]
    fn test_environment_reset() {
        let mut env = MockEnvironment::default();
        let snapshot = env.reset().expect("Reset should succeed");

        assert_eq!(snapshot.observation().position, 3);
        assert_eq!(snapshot.reward(), &ScalarReward(0.0));
        assert!(!snapshot.is_done());
    }

    #[test]
    fn test_environment_step_valid_action() {
        let mut env = MockEnvironment::default();
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
        let mut env = MockEnvironment::default();
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
        let mut env = MockEnvironment::default();

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
        let display_str = format!("{error}");
        assert!(display_str.contains("Invalid action"));
        assert!(display_str.contains("test action"));
    }

    #[test]
    fn test_environment_error_step_after_episode_end_carries_status() {
        for status in [EpisodeStatus::Terminated, EpisodeStatus::Truncated] {
            let error = EnvironmentError::StepAfterEpisodeEnd { status };
            match error {
                EnvironmentError::StepAfterEpisodeEnd { status: carried } => {
                    assert_eq!(
                        carried, status,
                        "StepAfterEpisodeEnd must carry the status that ended the episode"
                    );
                }
                _ => panic!("Expected StepAfterEpisodeEnd variant"),
            }
        }
    }

    #[test]
    fn test_environment_error_step_after_episode_end_display() {
        let error = EnvironmentError::StepAfterEpisodeEnd {
            status: EpisodeStatus::Terminated,
        };
        let display_str = format!("{error}");
        assert!(
            display_str.contains("after the episode ended"),
            "Display must state that the episode had already ended, got: {display_str}"
        );
        assert!(
            display_str.contains("Terminated"),
            "Display must name the ending status, got: {display_str}"
        );
        assert!(
            display_str.contains("reset()"),
            "Display must point the caller at reset(), got: {display_str}"
        );
    }

    #[test]
    fn test_environment_error_source() {
        use std::error::Error;

        let config_error = crate::config::positive("TestConfig", "rate", 0.0).unwrap_err();
        let env_error = EnvironmentError::from(config_error);
        assert!(env_error.source().is_some());
    }

    #[test]
    fn test_environment_multiple_episodes() {
        let mut env = MockEnvironment::default();

        for _episode in 0..3 {
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
        let snapshot = Snapshot::running(observation, ScalarReward(42.5));

        // ScalarReward converts into f32
        let reward_as_f32: f32 = (*snapshot.reward()).into();
        assert_eq!(reward_as_f32, 42.5);
    }

    #[test]
    fn test_snapshot_metadata_defaults_to_none() {
        let obs = MockObservation { position: 1 };
        let snapshot = Snapshot::running(obs, ScalarReward(0.0));
        assert!(
            snapshot.metadata().is_none(),
            "named constructors must leave metadata unset"
        );
    }

    #[test]
    fn test_snapshot_with_metadata_on_every_status() {
        let obs = MockObservation { position: 2 };
        let meta = || SnapshotMetadata::new().with("shaping", -1.5);

        for snapshot in [
            Snapshot::running(obs, ScalarReward(0.0)).with_metadata(meta()),
            Snapshot::terminated(obs, ScalarReward(0.0)).with_metadata(meta()),
            Snapshot::truncated(obs, ScalarReward(0.0)).with_metadata(meta()),
        ] {
            let m = snapshot.metadata().expect("metadata must be Some");
            assert_eq!(m.components.get("shaping"), Some(&-1.5));
        }
    }

    #[test]
    fn test_metadata_default_is_empty() {
        let meta = SnapshotMetadata::default();
        assert!(meta.components.is_empty());
        assert!(meta.positions.is_empty());
    }

    #[test]
    fn test_metadata_builder_components_and_positions() {
        let meta = SnapshotMetadata::new()
            .with("forward", 1.25)
            .with("ctrl", -0.1)
            .with_position("torso", [0.5, 0.0, 1.1])
            .with_position("com", [0.4, 0.0, 0.9]);

        assert_eq!(meta.components.len(), 2);
        assert_eq!(meta.components.get("forward"), Some(&1.25));
        assert_eq!(meta.positions.len(), 2);
        assert_eq!(meta.positions.get("torso"), Some(&[0.5, 0.0, 1.1]));
    }
}
