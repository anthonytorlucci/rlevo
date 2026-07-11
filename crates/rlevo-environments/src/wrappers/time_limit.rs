use rlevo_core::{
    base::{Observation, Reward},
    environment::{ConstructableEnv, Environment, EnvironmentError, EpisodeStatus, SnapshotBase},
    render::{AsciiRenderable, StyledFrame},
};

use crate::episode::EpisodeGuard;

/// Wraps an environment and truncates episodes after `max_steps` steps.
///
/// The inner environment's physics and termination logic are unchanged.
/// When the step counter reaches `max_steps` and the inner environment has
/// not already terminated, the snapshot status is upgraded from `Running`
/// to `Truncated`. This matches the Gymnasium `TimeLimit` wrapper
/// semantics.
///
/// Construct a `TimeLimit` with [`TimeLimit::new`], passing an already-built
/// inner environment and the step budget. Call [`reset`](TimeLimit::reset)
/// before the first [`step`](TimeLimit::step); the step counter resets to
/// zero on every `reset` call.
///
/// `TimeLimit` implements [`Environment`] for any inner env whose
/// `SnapshotType` is [`SnapshotBase`], [`AsciiRenderable`] by forwarding
/// to the wrapped env, and `Classic2DPayloadSource` for structured
/// post-run playback.
///
/// # Post-terminal steps
///
/// `TimeLimit` carries its **own** [`EpisodeGuard`] rather than relying on the
/// inner environment's: it *manufactures* the [`Truncated`](EpisodeStatus::Truncated)
/// status after delegating, so the inner env never learns it was truncated and
/// its guard still reads [`Running`](EpisodeStatus::Running). A guard below the
/// wrapper therefore cannot, by construction, reject a step taken after a
/// *truncation*. Any wrapper that synthesises a terminal status owns the guard
/// for it.
///
/// The guard is checked before the inner `step` is called, so a rejected step
/// never reaches — and never mutates — the wrapped environment. It covers both
/// terminal paths: an inner-env termination and a self-imposed truncation.
/// Call [`reset`](TimeLimit::reset) to begin a new episode.
pub struct TimeLimit<E> {
    inner: E,
    max_steps: usize,
    steps: usize,
    guard: EpisodeGuard,
}

impl<E> TimeLimit<E> {
    /// Wrap `env` with a hard step cap of `max_steps`.
    pub fn new(env: E, max_steps: usize) -> Self {
        Self {
            inner: env,
            max_steps,
            steps: 0,
            guard: EpisodeGuard::new(),
        }
    }

    /// Access the inner environment.
    pub fn inner(&self) -> &E {
        &self.inner
    }

    /// Mutably access the inner environment.
    pub fn inner_mut(&mut self) -> &mut E {
        &mut self.inner
    }

    /// Number of steps taken since the last `reset`.
    pub fn steps(&self) -> usize {
        self.steps
    }
}

impl<E> std::fmt::Debug for TimeLimit<E>
where
    E: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TimeLimit")
            .field("steps", &self.steps)
            .field("max_steps", &self.max_steps)
            .field("guard", &self.guard)
            .field("inner", &self.inner)
            .finish()
    }
}

/// A `TimeLimit` over a constructable inner env is itself constructable: it
/// builds the inner via [`ConstructableEnv`] and wraps it with no limit
/// (`usize::MAX`). Prefer the inherent `TimeLimit::new(env, max)` for real
/// use — this exists so generic `E: ConstructableEnv` code composes.
impl<E: ConstructableEnv> ConstructableEnv for TimeLimit<E> {
    fn new(render: bool) -> Self {
        Self::new(E::new(render), usize::MAX)
    }
}

/// `TimeLimit` implements `Environment` for any inner env whose `SnapshotType`
/// is `SnapshotBase<D, Obs, Rew>`. This constraint lets `step` directly
/// set `snap.status = Truncated` without trait acrobatics.
///
/// `reset` clears the step counter and re-opens the wrapper's [`EpisodeGuard`];
/// `step` consults that guard *before* delegating, then records the status of
/// the snapshot it actually returns — truncation included.
impl<const D: usize, const SD: usize, const AD: usize, E, Obs, Rew> Environment<D, SD, AD>
    for TimeLimit<E>
where
    E: Environment<
            D,
            SD,
            AD,
            ObservationType = Obs,
            RewardType = Rew,
            SnapshotType = SnapshotBase<D, Obs, Rew>,
        >,
    Obs: Observation<D>,
    Rew: Reward,
{
    type StateType = E::StateType;
    type ObservationType = Obs;
    type ActionType = E::ActionType;
    type RewardType = Rew;
    type SnapshotType = SnapshotBase<D, Obs, Rew>;

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        // Delegate first: only a *successful* inner reset actually starts a new
        // episode. Clearing the guard before delegating would re-open a finished
        // episode even when the inner reset failed, leaving the wrapper willing
        // to step an environment that never returned to its initial state.
        let snap = self.inner.reset()?;
        self.steps = 0;
        self.guard.reset();
        Ok(snap)
    }

    fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> {
        // Check *before* delegating: the wrapper manufactures `Truncated` itself,
        // so the inner env's own guard (if any) still reads `Running` after a
        // truncation and provably cannot reject this call. Short-circuiting here
        // also leaves the inner env unmutated on a rejected step.
        self.guard.check()?;

        let mut snap = self.inner.step(action)?;
        self.steps += 1;
        if snap.status == EpisodeStatus::Running && self.steps >= self.max_steps {
            snap.status = EpisodeStatus::Truncated;
        }
        // Record the status of the snapshot actually returned, so the one guard
        // covers both an inner-env `Terminated` and a self-imposed `Truncated`.
        self.guard.record(snap.status);
        Ok(snap)
    }
}

/// Forward [`AsciiRenderable`] through to the wrapped env so wrappers that
/// require it (e.g. `rlevo_benchmarks::env_wrappers::TuiEnvTap`) can compose
/// with `TimeLimit<E>` whenever `E` is itself renderable. Mirrors the
/// forwarding impl on `BenchAdapter`.
impl<E> AsciiRenderable for TimeLimit<E>
where
    E: AsciiRenderable,
{
    fn render_ascii(&self) -> String {
        self.inner.render_ascii()
    }

    fn render_styled(&self) -> StyledFrame {
        self.inner.render_styled()
    }
}

/// Forward the optional `Classic2DPayloadSource` through to the wrapped env,
/// so a `TimeLimit` over a classic-control env stays structurally renderable
/// (ADR-0013) — e.g. when a `RecordingTap` records a `TimeLimit`-wrapped env.
impl<E> rlevo_core::render::payload::Classic2DPayloadSource for TimeLimit<E>
where
    E: rlevo_core::render::payload::Classic2DPayloadSource,
{
    fn classic2d_snapshot(&self) -> rlevo_core::render::payload::Classic2DSnapshot {
        self.inner.classic2d_snapshot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episode::assert_rejects_post_terminal_step;
    use rlevo_core::{
        base::{Action, Observation, State},
        environment::{Environment, EnvironmentError, EpisodeStatus, Snapshot, SnapshotBase},
        reward::ScalarReward,
    };
    use serde::{Deserialize, Serialize};

    // Minimal stub environment: terminates when position reaches GOAL.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    struct StubObs {
        pos: i32,
    }

    impl Observation<1> for StubObs {
        fn shape() -> [usize; 1] {
            [1]
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct StubState {
        pos: i32,
    }

    impl State<1> for StubState {
        type Observation = StubObs;

        fn shape() -> [usize; 1] {
            [1]
        }

        fn is_valid(&self) -> bool {
            true
        }

        fn numel(&self) -> usize {
            1
        }

        fn observe(&self) -> StubObs {
            StubObs { pos: self.pos }
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct StubAction;

    impl Action<1> for StubAction {
        fn shape() -> [usize; 1] {
            [1]
        }

        fn is_valid(&self) -> bool {
            true
        }
    }

    /// Unguarded on purpose: the wrapper's guard, not the inner env's, is what
    /// these tests exercise. `step_calls` counts every `step` that actually
    /// reached the inner env and is *not* cleared by `reset`, so a test can
    /// prove a rejected step never got here.
    ///
    /// `reset_should_fail` makes `reset` fallible on demand. It models
    /// `FrozenLake`, whose `reset` regenerates a random map and returns
    /// `EnvironmentError::RenderFailed` when generation exhausts its retry budget
    /// — the only realistic reset failure in this crate, and one that is
    /// seed-dependent and so cannot be provoked deterministically there. Stubbing
    /// it here is what makes the "a failed reset must not re-open the episode"
    /// invariant testable at all.
    #[derive(Debug, Clone)]
    struct StubEnv {
        pos: i32,
        goal: i32,
        step_calls: usize,
        reset_should_fail: bool,
    }

    impl StubEnv {
        fn new_at_goal(goal: i32) -> Self {
            Self {
                pos: 0,
                goal,
                step_calls: 0,
                reset_should_fail: false,
            }
        }

        /// Number of `step` calls the inner env has seen since construction.
        fn step_calls(&self) -> usize {
            self.step_calls
        }

        /// Arms or disarms the simulated `reset` failure.
        fn set_reset_failure(&mut self, should_fail: bool) {
            self.reset_should_fail = should_fail;
        }
    }

    impl ConstructableEnv for StubEnv {
        fn new(_render: bool) -> Self {
            Self {
                pos: 0,
                goal: i32::MAX,
                step_calls: 0,
                reset_should_fail: false,
            }
        }
    }

    impl Environment<1, 1, 1> for StubEnv {
        type StateType = StubState;
        type ObservationType = StubObs;
        type ActionType = StubAction;
        type RewardType = ScalarReward;
        type SnapshotType = SnapshotBase<1, StubObs, ScalarReward>;

        fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
            if self.reset_should_fail {
                // Fail *before* touching state, exactly as FrozenLake's map
                // regeneration does: a failed reset starts no new episode.
                return Err(EnvironmentError::RenderFailed(
                    "stub reset failure".to_string(),
                ));
            }
            self.pos = 0;
            Ok(SnapshotBase::running(StubObs { pos: 0 }, ScalarReward(0.0)))
        }

        fn step(&mut self, _action: StubAction) -> Result<Self::SnapshotType, EnvironmentError> {
            self.step_calls += 1;
            self.pos += 1;
            if self.pos >= self.goal {
                Ok(SnapshotBase::terminated(
                    StubObs { pos: self.pos },
                    ScalarReward(1.0),
                ))
            } else {
                Ok(SnapshotBase::running(
                    StubObs { pos: self.pos },
                    ScalarReward(0.0),
                ))
            }
        }
    }

    #[test]
    fn truncated_at_step_cap() {
        let env = StubEnv::new_at_goal(100); // goal unreachable in 3 steps
        let mut timed = TimeLimit::new(env, 3);
        timed.reset().unwrap();

        let s1 = timed.step(StubAction).unwrap();
        assert_eq!(s1.status, EpisodeStatus::Running);

        let s2 = timed.step(StubAction).unwrap();
        assert_eq!(s2.status, EpisodeStatus::Running);

        let s3 = timed.step(StubAction).unwrap();
        assert_eq!(s3.status, EpisodeStatus::Truncated);
        assert!(s3.is_truncated());
        assert!(!s3.is_terminated());
        assert!(s3.is_done());
    }

    #[test]
    fn terminated_before_cap() {
        let env = StubEnv::new_at_goal(2); // terminates at step 2
        let mut timed = TimeLimit::new(env, 10);
        timed.reset().unwrap();

        let s1 = timed.step(StubAction).unwrap();
        assert_eq!(s1.status, EpisodeStatus::Running);

        let s2 = timed.step(StubAction).unwrap();
        assert_eq!(s2.status, EpisodeStatus::Terminated);
        assert!(!s2.is_truncated());
    }

    #[test]
    fn reset_clears_step_count() {
        let env = StubEnv::new_at_goal(100);
        let mut timed = TimeLimit::new(env, 2);
        timed.reset().unwrap();

        timed.step(StubAction).unwrap();
        timed.step(StubAction).unwrap();

        // After reset, step count should restart
        timed.reset().unwrap();
        assert_eq!(timed.steps(), 0);

        let s1 = timed.step(StubAction).unwrap();
        assert_eq!(s1.status, EpisodeStatus::Running);
        let s2 = timed.step(StubAction).unwrap();
        assert_eq!(s2.status, EpisodeStatus::Truncated);
    }

    // ── post-terminal guard ──────────────────────────────────────────────────
    //
    // The truncation case is the reason `TimeLimit` owns a guard at all: the
    // wrapper synthesises `Truncated` *after* the inner env has stepped, so the
    // inner env's own guard still reads `Running` and provably cannot reject the
    // next call. Only the wrapper's guard can.

    #[test]
    fn test_time_limit_step_after_truncation_is_rejected() {
        // Goal unreachable within the cap: the episode can only end by truncation.
        let mut timed = TimeLimit::new(StubEnv::new_at_goal(100), 3);

        assert_rejects_post_terminal_step::<1, 1, 1, _, _>(
            &mut timed,
            |env| {
                env.reset().expect("reset must succeed");
                let mut snap = env
                    .step(StubAction)
                    .expect("step must succeed until the cap");
                while !snap.is_done() {
                    snap = env
                        .step(StubAction)
                        .expect("step must succeed until the cap");
                }
                assert_eq!(
                    snap.status,
                    EpisodeStatus::Truncated,
                    "the step cap, not the inner env, must be what ends this episode"
                );
                snap
            },
            StubAction,
        );
    }

    #[test]
    fn test_time_limit_step_after_termination_is_rejected() {
        // Inner env terminates at step 2, well inside the cap of 10.
        let mut timed = TimeLimit::new(StubEnv::new_at_goal(2), 10);

        assert_rejects_post_terminal_step::<1, 1, 1, _, _>(
            &mut timed,
            |env| {
                env.reset().expect("reset must succeed");
                let mut snap = env
                    .step(StubAction)
                    .expect("step must succeed until termination");
                while !snap.is_done() {
                    snap = env
                        .step(StubAction)
                        .expect("step must succeed until termination");
                }
                assert_eq!(
                    snap.status,
                    EpisodeStatus::Terminated,
                    "the inner env, not the cap, must be what ends this episode"
                );
                snap
            },
            StubAction,
        );
    }

    #[test]
    fn test_time_limit_rejected_step_does_not_reach_inner_env() {
        let mut timed = TimeLimit::new(StubEnv::new_at_goal(100), 2);
        timed.reset().expect("reset must succeed");

        timed.step(StubAction).expect("first step must succeed");
        let capped = timed.step(StubAction).expect("second step must succeed");
        assert_eq!(
            capped.status,
            EpisodeStatus::Truncated,
            "the second step reaches the cap and must be truncated"
        );

        let calls_before = timed.inner().step_calls();
        let pos_before = timed.inner().pos;
        let steps_before = timed.steps();

        let err = timed
            .step(StubAction)
            .expect_err("a step after truncation must be rejected");
        assert!(
            matches!(
                err,
                EnvironmentError::StepAfterEpisodeEnd {
                    status: EpisodeStatus::Truncated
                }
            ),
            "the rejection must carry Truncated, the status that ended the episode; got {err:?}"
        );

        assert_eq!(
            timed.inner().step_calls(),
            calls_before,
            "the guard must short-circuit before delegating: the inner env must see no step call"
        );
        assert_eq!(
            timed.inner().pos,
            pos_before,
            "a rejected step must leave the inner env's state untouched"
        );
        assert_eq!(
            timed.steps(),
            steps_before,
            "a rejected step must not advance the wrapper's step counter"
        );
    }

    #[test]
    fn test_time_limit_reset_reopens_after_truncation() {
        let mut timed = TimeLimit::new(StubEnv::new_at_goal(100), 1);
        timed.reset().expect("reset must succeed");

        let truncated = timed.step(StubAction).expect("first step must succeed");
        assert_eq!(
            truncated.status,
            EpisodeStatus::Truncated,
            "a cap of 1 must truncate on the first step"
        );
        assert!(
            timed.step(StubAction).is_err(),
            "the episode is truncated; a further step must be rejected"
        );

        timed.reset().expect("reset must succeed");
        let reopened = timed
            .step(StubAction)
            .expect("reset() must re-open a truncated wrapper for a new episode");
        assert_eq!(
            reopened.status,
            EpisodeStatus::Truncated,
            "the fresh episode gets the full budget again and re-truncates at the cap"
        );
    }

    #[test]
    fn test_time_limit_reset_reopens_after_termination() {
        let mut timed = TimeLimit::new(StubEnv::new_at_goal(1), 10);
        timed.reset().expect("reset must succeed");

        let terminated = timed.step(StubAction).expect("first step must succeed");
        assert_eq!(
            terminated.status,
            EpisodeStatus::Terminated,
            "the inner env terminates on its first step"
        );
        assert!(
            timed.step(StubAction).is_err(),
            "the episode has terminated; a further step must be rejected"
        );

        timed.reset().expect("reset must succeed");
        assert!(
            timed.step(StubAction).is_ok(),
            "reset() must re-open a terminated wrapper for a new episode"
        );
    }

    // ── a failed reset must not re-open the episode ──────────────────────────
    //
    // `TimeLimit::reset` delegates *first* and only clears its guard and step
    // counter on the success path. If it cleared them up front, a failed inner
    // reset would leave the wrapper willing to step an environment that never
    // returned to its initial state — turning a loud, recoverable reset failure
    // into exactly the silent post-terminal stepping the guard exists to stop.
    // `FrozenLake::reset` makes the same promise, but its only failure mode
    // (`MapError::MaxRetriesExceeded`) is seed-dependent and cannot be forced
    // deterministically; `StubEnv`'s switchable failure is where that shared
    // invariant is actually pinned down.

    #[test]
    fn test_time_limit_failed_reset_does_not_reopen_truncated_episode() {
        let mut timed = TimeLimit::new(StubEnv::new_at_goal(100), 2);
        timed.reset().expect("reset must succeed");

        timed.step(StubAction).expect("first step must succeed");
        let truncated = timed.step(StubAction).expect("second step must succeed");
        assert_eq!(
            truncated.status,
            EpisodeStatus::Truncated,
            "the second step reaches the cap of 2 and must be truncated"
        );
        assert!(
            timed.step(StubAction).is_err(),
            "the episode is truncated; a further step must be rejected"
        );

        // Arm the failure and attempt to start a new episode.
        timed.inner_mut().set_reset_failure(true);
        let steps_before = timed.steps();
        let reset_err = timed
            .reset()
            .expect_err("reset() must propagate the inner env's reset failure");
        assert!(
            matches!(reset_err, EnvironmentError::RenderFailed(_)),
            "the wrapper must surface the inner failure unchanged, not swallow it; got {reset_err:?}"
        );

        // The invariant: a reset that failed started no episode, so the guard stays shut.
        let err = timed
            .step(StubAction)
            .expect_err("a failed reset() must not re-open the episode for stepping");
        assert!(
            matches!(
                err,
                EnvironmentError::StepAfterEpisodeEnd {
                    status: EpisodeStatus::Truncated
                }
            ),
            "the guard must still hold the status that ended the episode; got {err:?}"
        );
        assert_eq!(
            timed.steps(),
            steps_before,
            "a failed reset() must not clear the step counter either — the old episode still stands"
        );

        // A *successful* reset is the only thing that re-opens the pair.
        timed.inner_mut().set_reset_failure(false);
        timed
            .reset()
            .expect("reset must succeed once the failure is disarmed");
        assert_eq!(
            timed.steps(),
            0,
            "a successful reset() must clear the step counter"
        );
        assert!(
            timed.step(StubAction).is_ok(),
            "a successful reset() must re-open the wrapper for a new episode"
        );
    }

    #[test]
    fn test_time_limit_failed_reset_does_not_reopen_terminated_episode() {
        let mut timed = TimeLimit::new(StubEnv::new_at_goal(1), 10);
        timed.reset().expect("reset must succeed");

        let terminated = timed.step(StubAction).expect("first step must succeed");
        assert_eq!(
            terminated.status,
            EpisodeStatus::Terminated,
            "the inner env terminates on its first step"
        );

        timed.inner_mut().set_reset_failure(true);
        timed
            .reset()
            .expect_err("reset() must propagate the inner env's reset failure");

        let calls_before = timed.inner().step_calls();
        let err = timed
            .step(StubAction)
            .expect_err("a failed reset() must not re-open a terminated episode");
        assert!(
            matches!(
                err,
                EnvironmentError::StepAfterEpisodeEnd {
                    status: EpisodeStatus::Terminated
                }
            ),
            "the guard must still hold Terminated, the status that ended the episode; got {err:?}"
        );
        assert_eq!(
            timed.inner().step_calls(),
            calls_before,
            "the rejected step must still short-circuit before reaching the inner env"
        );

        timed.inner_mut().set_reset_failure(false);
        timed
            .reset()
            .expect("reset must succeed once the failure is disarmed");
        assert!(
            timed.step(StubAction).is_ok(),
            "a successful reset() must re-open the wrapper for a new episode"
        );
    }
}
