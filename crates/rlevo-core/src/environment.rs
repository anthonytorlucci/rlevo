//! Environment interaction protocol and snapshot types.
//!
//! This module defines the contract between an agent and a problem domain:
//! - [`Environment`] — core trait with `reset` / `step` methods
//! - [`Snapshot`] — per-step result carrying observation, reward, and status
//! - [`SnapshotBase`] — default `Snapshot` implementation for most environments
//! - [`EpisodeStatus`] — distinguishes running, terminated, and truncated episodes
//! - [`SnapshotMetadata`] — optional named reward components and 3D positions
//!
//! [`SnapshotMetadata`]: crate::environment::SnapshotMetadata

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
    /// An I/O operation failed (wraps `std::io::Error`).
    #[error("IO operation failed: {0}")]
    IoError(#[from] std::io::Error),
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

/// Snapshot trait defines the interface for environment state observations.
///
/// A snapshot captures the state of the environment at a single point in time,
/// including the observed state, reward received, and episode status.
/// The required method is `status()`; `is_done`, `is_terminated`, `is_truncated`,
/// and `metadata` are provided as defaults.
pub trait Snapshot<const R: usize>: Debug {
    /// The observation type exposed to the agent at each step.
    type ObservationType: Observation<R>;

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
/// `SnapshotBase` stores an observation, reward, [`EpisodeStatus`], and an
/// optional [`SnapshotMetadata`]. Construct via the named constructors
/// [`running`](Self::running), [`terminated`](Self::terminated), or
/// [`truncated`](Self::truncated) — each of which leaves `metadata` as `None` —
/// and attach metadata with the fluent [`with_metadata`](Self::with_metadata):
///
/// ```
/// # use rlevo_core::environment::{Snapshot, SnapshotBase, SnapshotMetadata};
/// # use rlevo_core::reward::ScalarReward;
/// # use rlevo_core::base::Observation;
/// # use serde::{Deserialize, Serialize};
/// # #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
/// # struct Obs(f32);
/// # impl Observation<1> for Obs { fn shape() -> [usize; 1] { [1] } }
/// let snap = SnapshotBase::running(Obs(0.0), ScalarReward(1.0))
///     .with_metadata(SnapshotMetadata::new().with("ctrl", -0.25));
/// assert_eq!(snap.metadata().unwrap().components["ctrl"], -0.25);
/// ```
///
/// Because environments that emit metadata still return a `SnapshotBase`, they
/// compose with wrappers such as `TimeLimit` that are bound to this type.
///
/// # Type Parameters
///
/// * `R` - The observation tensor rank
/// * `ObservationType` - The type of observation (must implement `Observation<R>`)
/// * `RewardType` - The type of reward (must implement `Reward`)
#[derive(Debug, Clone)]
pub struct SnapshotBase<const R: usize, ObservationType: Observation<R>, RewardType: Reward> {
    /// The observation derived from the state.
    pub observation: ObservationType,
    /// The reward received from the last action.
    pub reward: RewardType,
    /// Episode lifecycle status.
    pub status: EpisodeStatus,
    /// Optional named reward components and positions for this step.
    pub metadata: Option<SnapshotMetadata>,
}

impl<const R: usize, ObservationType: Observation<R>, RewardType: Reward>
    SnapshotBase<R, ObservationType, RewardType>
{
    /// Snapshot for a step where the episode is still running.
    pub fn running(observation: ObservationType, reward: RewardType) -> Self {
        Self {
            observation,
            reward,
            status: EpisodeStatus::Running,
            metadata: None,
        }
    }

    /// Snapshot for the step on which the MDP reached a terminal state.
    pub fn terminated(observation: ObservationType, reward: RewardType) -> Self {
        Self {
            observation,
            reward,
            status: EpisodeStatus::Terminated,
            metadata: None,
        }
    }

    /// Snapshot for the step on which an external step limit was reached.
    pub fn truncated(observation: ObservationType, reward: RewardType) -> Self {
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
}

impl<const R: usize, ObservationType: Observation<R>, RewardType: Reward> Snapshot<R>
    for SnapshotBase<R, ObservationType, RewardType>
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

    fn metadata(&self) -> Option<&SnapshotMetadata> {
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
/// * `R`  - Rank of the observation tensor (matches `Observation<R>` and `Snapshot<R>`).
/// * `SR` - Rank of the state tensor (matches `State<SR>`).
/// * `AR` - Rank of the action tensor (matches `Action<AR>`).
///
/// # Associated Types
///
/// * `StateType`       - The concrete state type for this environment.
/// * `ObservationType` - The observation type exposed to the agent.
/// * `ActionType`      - The action type this environment accepts.
/// * `RewardType`      - The reward scalar type returned each step.
/// * `SnapshotType`    - The snapshot type returned by `reset` and `step`.
pub trait Environment<const R: usize, const SR: usize, const AR: usize> {
    /// The concrete state type for this environment.
    type StateType: State<SR>;

    /// The observation type exposed to the agent.
    type ObservationType: Observation<R>;

    /// The concrete action type this environment accepts.
    type ActionType: Action<AR>;

    /// The reward scalar type returned by this environment.
    type RewardType: Reward;

    /// The snapshot type returned by reset and step operations.
    type SnapshotType: Snapshot<R, ObservationType = Self::ObservationType, RewardType = Self::RewardType>;

    /// Reset the environment to its initial state and return the first snapshot.
    ///
    /// The returned snapshot carries the initial observation, a reward of zero, and
    /// [`EpisodeStatus::Running`]. Call this at the start of every episode before
    /// calling [`step`](Self::step).
    ///
    /// # Errors
    ///
    /// Returns [`EnvironmentError`] if the environment cannot be initialised (e.g.
    /// an I/O failure when loading level data).
    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError>;

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
    /// The check belongs at the **top** of `step()`, before any state mutation,
    /// so a rejected call leaves the environment untouched.
    ///
    /// **Migration note (alpha).** Only the `toy_text` family and the `TimeLimit`
    /// wrapper currently enforce this. The behaviour of every other environment
    /// after a terminal snapshot is **undefined** — see issue #289 for the
    /// family-by-family rollout. Callers must not rely on it.
    ///
    /// # Errors
    ///
    /// Returns [`EnvironmentError::InvalidAction`] if the action is not legal in
    /// the current state, [`EnvironmentError::StepAfterEpisodeEnd`] if the
    /// episode has already ended (see the post-terminal contract above), or
    /// another [`EnvironmentError`] variant if the step cannot complete (e.g. a
    /// physics simulation failure).
    fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError>;
}

/// The environment-side emission model `O(a, s')` of a POMDP ⟨S, A, T, R, Ω, O⟩.
///
/// A `Sensor` produces the agent's [`Observation`] from the last [`Action`] and
/// the resulting [`State`]. It is implemented on the **environment**, not on the
/// state: `&self` is the environment, so a sensor may read world / simulator
/// context — raycasts against physics geometry, a rendered pixel frame — that a
/// bare `State` value does not own.
///
/// # Why O lives here, not on `State`
///
/// In the POMDP tuple ⟨S, A, T, R, Ω, O⟩ the emission model `O` is a property of
/// the problem, not of a point `s ∈ S` in state space. `State` used to carry a
/// `fn observe(&self) -> Self::Observation`, which welded the observation's
/// tensor order to the state's own order and, having only `&self`, forced
/// world-derived sensors (lidar, pixels) to be cached into the state first.
/// Relocating `O` to this env-side trait removes both problems: the observation
/// rank is independent of the state rank (the [`Environment`] contract already
/// decouples `R` from `SR`), and `&self = env` gives direct world access. (ADR
/// 0047, superseding ADR 0019 and obsoleting ADR 0039's cached-sensor target.)
///
/// # When and how to implement
///
/// Every environment builds the observations in its `reset` / `step` snapshots
/// through a `Sensor` — typically by implementing it on the environment struct
/// itself and calling [`observe`](Sensor::observe) at each step and
/// [`observe_reset`](Sensor::observe_reset) at episode start. When the
/// observation is a pure projection of the state (no world context), the body
/// may delegate to [`Observable::project`](crate::state::Observable) — see the
/// `pixel_grid` environment for the reference "sensor delegates to `Observable`"
/// pattern.
///
/// # Type Parameters
///
/// - `OR`: tensor order (rank) of the produced [`Observation`].
/// - `AR`: tensor order (rank) of the [`Action`].
/// - `SR`: tensor order (rank) of the [`State`].
///
/// The three ranks are independent: an environment whose state is rank `SR` may
/// observe through a sensor of any rank `OR`, which is exactly what lets a
/// compact physics state be observed as a higher-rank pixel frame without
/// inflating the state's own rank.
///
/// # Examples
///
/// A sensor whose observation is a plain function of the resulting state:
///
/// ```
/// use rlevo_core::base::{Action, Observation, State};
/// use rlevo_core::environment::Sensor;
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Debug, Clone)]
/// struct Position {
///     x: i32,
/// }
/// impl State<1> for Position {
///     fn shape() -> [usize; 1] {
///         [1]
///     }
///     fn is_valid(&self) -> bool {
///         true
///     }
/// }
///
/// #[derive(Debug, Clone, Serialize, Deserialize)]
/// struct PositionObs {
///     x: i32,
/// }
/// impl Observation<1> for PositionObs {
///     fn shape() -> [usize; 1] {
///         [1]
///     }
/// }
///
/// #[derive(Debug, Clone, Copy)]
/// struct Nudge;
/// impl Action<1> for Nudge {
///     fn shape() -> [usize; 1] {
///         [1]
///     }
///     fn is_valid(&self) -> bool {
///         true
///     }
/// }
///
/// struct World;
/// impl Sensor<1, 1, 1> for World {
///     type Action = Nudge;
///     type State = Position;
///     type Observation = PositionObs;
///
///     fn observe(&self, _action: &Nudge, next_state: &Position) -> PositionObs {
///         PositionObs { x: next_state.x }
///     }
///     fn observe_reset(&self, state: &Position) -> PositionObs {
///         PositionObs { x: state.x }
///     }
/// }
///
/// let obs = World.observe(&Nudge, &Position { x: 7 });
/// assert_eq!(obs.x, 7);
/// ```
pub trait Sensor<const OR: usize, const AR: usize, const SR: usize> {
    /// The action type consumed by the emission model.
    type Action: Action<AR>;

    /// The state type the observation is produced from.
    type State: State<SR>;

    /// The observation type produced for the agent.
    type Observation: Observation<OR>;

    /// Emission model `O(a, s')`: the observation after taking `action` and
    /// arriving at `next_state`.
    ///
    /// Called once per [`Environment::step`], after the transition has produced
    /// `next_state`. `&self` is the environment, so the observation may read
    /// world / simulator context beyond the state value itself.
    fn observe(&self, action: &Self::Action, next_state: &Self::State) -> Self::Observation;

    /// The initial observation at episode start, before any action exists.
    ///
    /// [`Environment::reset`] has no preceding action, so the stepping form
    /// `O(a, s')` does not apply; this companion produces the first observation
    /// from the initial `state` alone. Implementations whose observation ignores
    /// the action typically forward both methods to the same projection.
    fn observe_reset(&self, state: &Self::State) -> Self::Observation;
}

/// Default-construction factory for environments, lifted off [`Environment`]
/// (ADR-0011).
///
/// Construction is a separate concern from the behavioural [`Environment`]
/// contract (`reset`/`step`). Keeping `new` here means transparent decorators
/// — `RecordingTap`, `TuiEnvTap`, `TimeLimit` — implement only the behaviour
/// they actually forward and are never forced to synthesise a degenerate
/// standalone constructor just to satisfy a trait bound. They are always
/// built from an existing inner environment instead.
///
/// Concrete environments implement this alongside [`Environment`]; generic
/// code that needs to build an environment from nothing (rather than from a
/// caller-supplied factory closure) bounds on `E: ConstructableEnv`.
pub trait ConstructableEnv {
    /// Create a new environment instance.
    ///
    /// `render` controls whether the environment emits display output on each
    /// step; pass `false` for training runs where rendering overhead is
    /// unwanted. Implementations that do not support rendering may ignore the
    /// flag.
    fn new(render: bool) -> Self;
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

        /// Check if position is within valid bounds
        fn is_in_bounds(position: i32) -> bool {
            (0..=6).contains(&position)
        }
    }

    impl State<1> for MockState {
        fn numel(&self) -> usize {
            7
        }

        fn shape() -> [usize; 1] {
            [7]
        }

        fn is_valid(&self) -> bool {
            Self::is_in_bounds(self.position)
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

    impl ConstructableEnv for MockEnvironment {
        fn new(render: bool) -> Self {
            Self::with_defaults(render)
        }
    }

    impl Sensor<1, 1, 1> for MockEnvironment {
        type Action = MockAction;
        type State = MockState;
        type Observation = MockObservation;

        fn observe(&self, _action: &MockAction, next_state: &MockState) -> MockObservation {
            MockObservation {
                position: next_state.position,
            }
        }

        fn observe_reset(&self, state: &MockState) -> MockObservation {
            MockObservation {
                position: state.position,
            }
        }
    }

    impl Environment<1, 1, 1> for MockEnvironment {
        type StateType = MockState;
        type ObservationType = MockObservation;
        type ActionType = MockAction;
        type RewardType = ScalarReward;
        type SnapshotType = SnapshotBase<1, MockObservation, ScalarReward>;

        fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
            self.current_state = MockState::new(Self::START_STATE);
            self.step_count = 0;
            Ok(SnapshotBase::running(
                self.observe_reset(&self.current_state),
                ScalarReward(0.0),
            ))
        }

        fn step(
            &mut self,
            action: Self::ActionType,
        ) -> Result<Self::SnapshotType, EnvironmentError> {
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

            let observation = self.observe(&action, &new_state);
            Ok(SnapshotBase {
                observation,
                reward: ScalarReward(reward),
                status,
                metadata: None,
            })
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
        let debug_str = format!("{snapshot:?}");

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
        let display_str = format!("{error}");
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

        let io_error = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
        let env_error = EnvironmentError::IoError(io_error);
        assert!(env_error.source().is_some());
    }

    #[test]
    fn test_environment_multiple_episodes() {
        let mut env = MockEnvironment::new(false);

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
        let snapshot = SnapshotBase::running(observation, ScalarReward(42.5));

        // RewardType implements Into<f32>
        let reward_as_f32: f32 = (*snapshot.reward()).into();
        assert_eq!(reward_as_f32, 42.5);
    }

    #[test]
    fn test_snapshot_base_metadata_defaults_to_none() {
        let obs = MockObservation { position: 1 };
        let snapshot = SnapshotBase::running(obs, ScalarReward(0.0));
        assert!(
            snapshot.metadata().is_none(),
            "named constructors must leave metadata unset"
        );
    }

    #[test]
    fn test_snapshot_base_with_metadata_on_every_status() {
        let obs = MockObservation { position: 2 };
        let meta = || SnapshotMetadata::new().with("shaping", -1.5);

        for snapshot in [
            SnapshotBase::running(obs, ScalarReward(0.0)).with_metadata(meta()),
            SnapshotBase::terminated(obs, ScalarReward(0.0)).with_metadata(meta()),
            SnapshotBase::truncated(obs, ScalarReward(0.0)).with_metadata(meta()),
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
