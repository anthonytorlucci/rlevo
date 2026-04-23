use rlevo_core::{
    base::{Observation, Reward},
    environment::{Environment, EnvironmentError, EpisodeStatus, SnapshotBase},
};

/// Wraps an environment and truncates episodes after `max_steps` steps.
///
/// The inner environment's physics and termination logic are unchanged.
/// When the step counter reaches `max_steps` and the inner environment has
/// not already terminated, the snapshot status is upgraded from `Running`
/// to `Truncated`. This implements the Gymnasium `TimeLimit` wrapper
/// semantics in pure Rust.
///
/// # Usage
///
/// ```rust,ignore
/// use evorl_envs::wrappers::TimeLimit;
///
/// let env = CartPole::with_config(CartPoleConfig::default());
/// let mut timed = TimeLimit::new(env, 500);
///
/// let snap = timed.reset().unwrap();
/// assert!(!snap.is_done());
/// ```
pub struct TimeLimit<E> {
    inner: E,
    max_steps: usize,
    steps: usize,
}

impl<E> TimeLimit<E> {
    /// Wrap `env` with a hard step cap of `max_steps`.
    pub fn new(env: E, max_steps: usize) -> Self {
        Self {
            inner: env,
            max_steps,
            steps: 0,
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
            .field("inner", &self.inner)
            .finish()
    }
}

/// `TimeLimit` implements `Environment` for any inner env whose `SnapshotType`
/// is `SnapshotBase<D, Obs, Rew>`. This constraint lets `step` directly
/// set `snap.status = Truncated` without trait acrobatics.
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

    fn new(render: bool) -> Self {
        // TimeLimit has no meaningful standalone constructor; callers use TimeLimit::new(env, max).
        // This satisfies the trait but should not be called directly.
        Self::new(E::new(render), usize::MAX)
    }

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.steps = 0;
        self.inner.reset()
    }

    fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> {
        let mut snap = self.inner.step(action)?;
        self.steps += 1;
        if snap.status == EpisodeStatus::Running && self.steps >= self.max_steps {
            snap.status = EpisodeStatus::Truncated;
        }
        Ok(snap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    struct StubEnv {
        pos: i32,
        goal: i32,
    }

    impl StubEnv {
        fn new_at_goal(goal: i32) -> Self {
            Self { pos: 0, goal }
        }
    }

    impl Environment<1, 1, 1> for StubEnv {
        type StateType = StubState;
        type ObservationType = StubObs;
        type ActionType = StubAction;
        type RewardType = ScalarReward;
        type SnapshotType = SnapshotBase<1, StubObs, ScalarReward>;

        fn new(_render: bool) -> Self {
            Self {
                pos: 0,
                goal: i32::MAX,
            }
        }

        fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
            self.pos = 0;
            Ok(SnapshotBase::running(StubObs { pos: 0 }, ScalarReward(0.0)))
        }

        fn step(&mut self, _action: StubAction) -> Result<Self::SnapshotType, EnvironmentError> {
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
}
