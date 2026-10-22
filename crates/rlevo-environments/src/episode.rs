//! The conformance check for the post-terminal `step` contract, shared by
//! this crate's tests.
//!
//! The contract is `rlevo-core`'s: a
//! [`step`](rlevo_core::environment::Environment::step) taken after a done
//! snapshot returns [`EnvironmentError::StepAfterEpisodeEnd`]. Its state
//! machine, [`EpisodeGuard`](rlevo_core::episode::EpisodeGuard), is
//! `rlevo-core`'s too. What lives here is
//! [`assert_rejects_post_terminal_step`], the assertion each guarded
//! environment's test module calls, with the mock environment that
//! smoke-tests it. The module is compiled for tests only.

use rlevo_core::environment::EnvironmentError;

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
pub(crate) fn assert_rejects_post_terminal_step<const R: usize, const AR: usize, E, F>(
    env: &mut E,
    drive_to_done: F,
    replay_action: E::ActionType,
) where
    E: rlevo_core::environment::Environment<R, AR>,
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

mod tests {
    use super::assert_rejects_post_terminal_step;
    use rlevo_core::action::DiscreteAction;
    use rlevo_core::base::{Action, Observation};
    use rlevo_core::environment::{Environment, EnvironmentError, Snapshot, SnapshotBase};
    use rlevo_core::episode::EpisodeGuard;
    use rlevo_core::reward::ScalarReward;
    use serde::{Deserialize, Serialize};

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

    impl MockGuardedEnv {
        fn observe(&self) -> MockObservation {
            MockObservation {
                steps: self.state.steps,
            }
        }
    }

    impl Environment<1, 1> for MockGuardedEnv {
        type ObservationType = MockObservation;
        type ActionType = MockAction;
        type RewardType = ScalarReward;
        type SnapshotType = SnapshotBase<1, MockObservation, ScalarReward>;

        fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
            self.guard.reset();
            self.state = MockState { steps: 0 };
            Ok(SnapshotBase::running(self.observe(), ScalarReward(0.0)))
        }

        fn step(&mut self, _action: MockAction) -> Result<Self::SnapshotType, EnvironmentError> {
            self.guard.check()?;

            self.state.steps += 1;
            let obs = self.observe();
            let snapshot = if self.state.steps >= Self::TERMINAL_STEP {
                SnapshotBase::terminated(obs, ScalarReward(1.0))
            } else {
                SnapshotBase::running(obs, ScalarReward(0.0))
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
