//! Episode-lifecycle guard shared by every environment in this crate.
//!
//! An episode that has emitted a snapshot with
//! [`Snapshot::is_done`](rlevo_core::environment::Snapshot::is_done) `== true` is
//! over. Calling
//! [`step`](rlevo_core::environment::Environment::step) again must not silently
//! resurrect it — the contract on `Environment::step` is that the second call
//! returns [`EnvironmentError::StepAfterEpisodeEnd`].
//!
//! [`EpisodeGuard`] is the one-field state machine that implements that
//! contract. It lives at crate level rather than inside any one environment
//! family because both the `toy_text` environments and the
//! [`TimeLimit`](crate::wrappers::time_limit::TimeLimit) wrapper consume it.
//!
//! # Usage
//!
//! Hold a guard in the environment struct, `check()` at the very top of `step()`
//! (before any state mutation, so a rejected call leaves the environment
//! untouched), `record()` the status of every snapshot you emit, and `reset()`
//! the guard in `Environment::reset`:
//!
//! ```rust,ignore
//! fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
//!     self.guard.reset();
//!     // ... rebuild initial state ...
//!     Ok(SnapshotBase::running(obs, ScalarReward(0.0)))
//! }
//!
//! fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> {
//!     self.guard.check()?;
//!     // ... apply the action, build `snapshot` ...
//!     self.guard.record(snapshot.status);
//!     Ok(snapshot)
//! }
//! ```
//!
//! The guard stores an [`EpisodeStatus`], never a `done: bool` — `EpisodeStatus`
//! is the single source of truth for episode termination (`rules.md` §10), and
//! the stored status is what the error carries back to the caller.

use rlevo_core::environment::{EnvironmentError, EpisodeStatus};

/// Tracks whether the current episode has ended, and rejects a post-terminal
/// [`step`](rlevo_core::environment::Environment::step).
///
/// A fresh guard is [`EpisodeStatus::Running`]. [`record`](Self::record) stores
/// the status of the snapshot the environment just emitted;
/// [`check`](Self::check) fails once that stored status is done;
/// [`reset`](Self::reset) re-opens the guard for a new episode.
///
/// The status field is private: it can only be advanced through `record` and
/// cleared through `reset`, so an environment cannot accidentally hand-write a
/// state the snapshot it emitted does not agree with.
///
/// # Examples
///
/// ```
/// use rlevo_core::environment::{EnvironmentError, EpisodeStatus};
/// use rlevo_environments::episode::EpisodeGuard;
///
/// let mut guard = EpisodeGuard::new();
/// assert!(guard.check().is_ok(), "a fresh episode is steppable");
///
/// guard.record(EpisodeStatus::Terminated);
/// let err = guard.check().expect_err("a finished episode is not steppable");
/// assert!(matches!(
///     err,
///     EnvironmentError::StepAfterEpisodeEnd {
///         status: EpisodeStatus::Terminated
///     }
/// ));
///
/// guard.reset();
/// assert!(guard.check().is_ok(), "reset() begins a new episode");
/// ```
#[derive(Debug, Clone, Copy)]
pub struct EpisodeGuard {
    /// Status of the most recently emitted snapshot; `Running` before the first
    /// step of an episode.
    status: EpisodeStatus,
}

impl EpisodeGuard {
    /// Creates a guard for a fresh, running episode.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            status: EpisodeStatus::Running,
        }
    }

    /// Re-opens the guard for a new episode.
    ///
    /// Call from [`Environment::reset`](rlevo_core::environment::Environment::reset).
    pub const fn reset(&mut self) {
        self.status = EpisodeStatus::Running;
    }

    /// Rejects a step taken after the episode already ended.
    ///
    /// Call at the **top** of
    /// [`Environment::step`](rlevo_core::environment::Environment::step), before
    /// any state mutation, so a rejected call leaves the environment untouched.
    ///
    /// # Errors
    ///
    /// Returns [`EnvironmentError::StepAfterEpisodeEnd`], carrying the
    /// [`EpisodeStatus`] that ended the episode, when the recorded status is
    /// done ([`Terminated`](EpisodeStatus::Terminated) or
    /// [`Truncated`](EpisodeStatus::Truncated)).
    pub const fn check(&self) -> Result<(), EnvironmentError> {
        if self.status.is_done() {
            return Err(EnvironmentError::StepAfterEpisodeEnd {
                status: self.status,
            });
        }
        Ok(())
    }

    /// Records the status of the snapshot the environment just emitted.
    ///
    /// Call once per emitted snapshot, with that snapshot's own status — passing
    /// anything else breaks the invariant that the guard and the snapshot agree.
    pub const fn record(&mut self, status: EpisodeStatus) {
        self.status = status;
    }

    /// Status of the most recently emitted snapshot.
    #[must_use]
    pub const fn status(&self) -> EpisodeStatus {
        self.status
    }
}

impl Default for EpisodeGuard {
    fn default() -> Self {
        Self::new()
    }
}

/// Asserts that `env` rejects a `step()` taken after its episode has ended.
///
/// The shared conformance check behind the post-terminal contract on
/// [`Environment::step`](rlevo_core::environment::Environment::step). It:
///
/// 1. drives `env` to a terminal snapshot with `drive_to_done`,
/// 2. asserts that snapshot is in fact done, then
/// 3. asserts a further `step(replay_action)` returns
///    [`EnvironmentError::StepAfterEpisodeEnd`] carrying that same status.
///
/// `drive_to_done` owns the reset — it is handed `&mut env` and must return the
/// terminal snapshot. `replay_action` is the (legal) action replayed after the
/// episode ends; the guard must reject it on call-sequence grounds alone, never
/// on the action's own validity.
///
/// # Panics
///
/// Panics (as a test assertion) if the driven snapshot is not done, if the
/// post-terminal `step` succeeds, or if it fails with any other
/// [`EnvironmentError`] variant.
#[cfg(test)]
pub(crate) fn assert_rejects_post_terminal_step<
    const R: usize,
    const SR: usize,
    const AR: usize,
    E,
    F,
>(
    env: &mut E,
    drive_to_done: F,
    replay_action: E::ActionType,
) where
    E: rlevo_core::environment::Environment<R, SR, AR>,
    F: FnOnce(&mut E) -> E::SnapshotType,
{
    use rlevo_core::environment::Snapshot;

    let terminal = drive_to_done(env);
    let ended = terminal.status();
    assert!(
        ended.is_done(),
        "drive_to_done must leave the environment on a done snapshot, got {ended:?}"
    );

    let err = env
        .step(replay_action)
        .expect_err("step() after a done snapshot must return Err, not a fresh snapshot");

    match err {
        EnvironmentError::StepAfterEpisodeEnd { status } => assert_eq!(
            status, ended,
            "StepAfterEpisodeEnd must carry the status that ended the episode"
        ),
        other => panic!("post-terminal step must fail with StepAfterEpisodeEnd, got {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::{EpisodeGuard, assert_rejects_post_terminal_step};
    use rlevo_core::action::DiscreteAction;
    use rlevo_core::base::{Action, Observation, State};
    use rlevo_core::environment::{
        Environment, EnvironmentError, EpisodeStatus, Snapshot, SnapshotBase,
    };
    use rlevo_core::reward::ScalarReward;
    use serde::{Deserialize, Serialize};

    #[test]
    fn test_episode_guard_check_passes_when_fresh() {
        let guard = EpisodeGuard::new();
        assert_eq!(
            guard.status(),
            EpisodeStatus::Running,
            "a fresh guard must start Running"
        );
        assert!(
            guard.check().is_ok(),
            "a fresh guard must permit the first step"
        );
    }

    #[test]
    fn test_episode_guard_default_matches_new() {
        assert_eq!(
            EpisodeGuard::default().status(),
            EpisodeGuard::new().status(),
            "Default must agree with new(): a running episode"
        );
    }

    #[test]
    fn test_episode_guard_check_rejects_after_terminated() {
        let mut guard = EpisodeGuard::new();
        guard.record(EpisodeStatus::Terminated);

        let err = guard
            .check()
            .expect_err("check() must fail once the episode has terminated");
        match err {
            EnvironmentError::StepAfterEpisodeEnd { status } => assert_eq!(
                status,
                EpisodeStatus::Terminated,
                "the error must carry Terminated, the status that ended the episode"
            ),
            other => panic!("expected StepAfterEpisodeEnd, got {other:?}"),
        }
    }

    #[test]
    fn test_episode_guard_check_rejects_after_truncated() {
        let mut guard = EpisodeGuard::new();
        guard.record(EpisodeStatus::Truncated);

        let err = guard
            .check()
            .expect_err("check() must fail once the episode has been truncated");
        match err {
            EnvironmentError::StepAfterEpisodeEnd { status } => assert_eq!(
                status,
                EpisodeStatus::Truncated,
                "the error must carry Truncated, so the caller can tell it from a termination"
            ),
            other => panic!("expected StepAfterEpisodeEnd, got {other:?}"),
        }
    }

    #[test]
    fn test_episode_guard_check_passes_after_running() {
        let mut guard = EpisodeGuard::new();
        guard.record(EpisodeStatus::Running);
        assert!(
            guard.check().is_ok(),
            "recording a Running snapshot must leave the episode steppable"
        );
    }

    #[test]
    fn test_episode_guard_reset_reopens_done_guard() {
        let mut guard = EpisodeGuard::new();
        guard.record(EpisodeStatus::Terminated);
        guard.reset();

        assert_eq!(
            guard.status(),
            EpisodeStatus::Running,
            "reset() must return the guard to Running"
        );
        assert!(
            guard.check().is_ok(),
            "reset() must re-open a guard that had ended"
        );
    }

    // ── conformance-helper smoke test ────────────────────────────────────────
    //
    // `MockGuardedEnv` is the reference shape the Wave-2 environments take: a
    // guard field, `check()?` at the top of `step`, `record(..)` on the way out.

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    struct MockObservation {
        steps: u8,
    }

    impl Observation<1> for MockObservation {
        fn shape() -> [usize; 1] {
            [1]
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct MockState {
        steps: u8,
    }

    impl State<1> for MockState {
        type Observation = MockObservation;

        fn numel(&self) -> usize {
            1
        }

        fn shape() -> [usize; 1] {
            [1]
        }

        fn is_valid(&self) -> bool {
            true
        }

        fn observe(&self) -> Self::Observation {
            MockObservation { steps: self.steps }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct MockAction;

    impl Action<1> for MockAction {
        fn is_valid(&self) -> bool {
            true
        }

        fn shape() -> [usize; 1] {
            [1]
        }
    }

    impl DiscreteAction<1> for MockAction {
        const ACTION_COUNT: usize = 1;

        fn from_index(index: usize) -> Self {
            assert_eq!(index, 0, "MockAction has a single action");
            MockAction
        }

        fn to_index(&self) -> usize {
            0
        }
    }

    /// Terminates on the second step; guarded by an [`EpisodeGuard`].
    #[derive(Debug, Clone)]
    struct MockGuardedEnv {
        state: MockState,
        guard: EpisodeGuard,
    }

    impl MockGuardedEnv {
        const TERMINAL_STEP: u8 = 2;

        fn new() -> Self {
            Self {
                state: MockState { steps: 0 },
                guard: EpisodeGuard::new(),
            }
        }
    }

    impl Environment<1, 1, 1> for MockGuardedEnv {
        type StateType = MockState;
        type ObservationType = MockObservation;
        type ActionType = MockAction;
        type RewardType = ScalarReward;
        type SnapshotType = SnapshotBase<1, MockObservation, ScalarReward>;

        fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
            self.guard.reset();
            self.state = MockState { steps: 0 };
            Ok(SnapshotBase::running(
                self.state.observe(),
                ScalarReward(0.0),
            ))
        }

        fn step(&mut self, _action: MockAction) -> Result<Self::SnapshotType, EnvironmentError> {
            self.guard.check()?;

            self.state.steps += 1;
            let snapshot = if self.state.steps >= Self::TERMINAL_STEP {
                SnapshotBase::terminated(self.state.observe(), ScalarReward(1.0))
            } else {
                SnapshotBase::running(self.state.observe(), ScalarReward(0.0))
            };

            self.guard.record(snapshot.status());
            Ok(snapshot)
        }
    }

    #[test]
    fn test_episode_guard_rejects_post_terminal_step_in_an_environment() {
        let mut env = MockGuardedEnv::new();
        assert_rejects_post_terminal_step(
            &mut env,
            |env| {
                env.reset().expect("reset must succeed");
                let mut snapshot = env.step(MockAction).expect("first step must succeed");
                while !snapshot.is_done() {
                    snapshot = env.step(MockAction).expect("step must succeed until done");
                }
                snapshot
            },
            MockAction,
        );
    }

    #[test]
    fn test_episode_guard_reset_permits_a_second_episode() {
        let mut env = MockGuardedEnv::new();
        env.reset().expect("reset must succeed");
        for _ in 0..MockGuardedEnv::TERMINAL_STEP {
            env.step(MockAction).expect("step must succeed until done");
        }
        assert!(
            env.step(MockAction).is_err(),
            "the episode has ended; a further step must be rejected"
        );

        env.reset().expect("reset must succeed");
        assert!(
            env.step(MockAction).is_ok(),
            "reset() must re-open the environment for a new episode"
        );
    }
}
