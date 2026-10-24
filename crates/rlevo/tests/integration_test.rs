//! An environment loop over `rlevo-core`'s public contract alone, against a
//! small toy `RandomWalkEnv` defined inline. It can only see the public API —
//! unlike the in-crate `#[cfg(test)]` mocks — so it doubles as a smoke test
//! that the public surface is sufficient to write and drive an environment.

use rlevo_core::action::DiscreteAction;
use rlevo_core::base::{Action, HostRow, Observation};
use rlevo_core::environment::{Environment, EnvironmentError, EpisodeStatus, Snapshot};
use rlevo_core::reward::ScalarReward;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Toy environment: 1-D random walk on positions [0, 6]. Start at 3. Reward +1
// for reaching position 6 (Terminated), -1 for falling below 0 (Terminated),
// 0 otherwise. Truncated after 20 steps.
// ---------------------------------------------------------------------------

/// Agent-visible observation for `RandomWalkEnv`: a single integer position on `[0, 6]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct WalkObservation {
    position: i32,
}

impl Observation<1> for WalkObservation {
    fn shape() -> [usize; 1] {
        [1]
    }
}

impl HostRow<1> for WalkObservation {
    #[allow(clippy::cast_precision_loss)]
    fn write_host_row(&self, buf: &mut Vec<f32>) {
        buf.push(self.position as f32);
    }
}

/// Full internal state for `RandomWalkEnv`: position on the 1-D lattice `[0, 6]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WalkState {
    position: i32,
}

/// Two-move action space for `RandomWalkEnv`: step left (index 0) or right (index 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WalkAction {
    Left,
    Right,
}

impl Action<1> for WalkAction {
    fn shape() -> [usize; 1] {
        [1]
    }
    fn is_valid(&self) -> bool {
        true
    }
}

impl DiscreteAction<1> for WalkAction {
    const ACTION_COUNT: usize = 2;

    fn from_index(index: usize) -> Self {
        match index {
            0 => WalkAction::Left,
            1 => WalkAction::Right,
            _ => panic!("invalid WalkAction index: {index}"),
        }
    }

    fn to_index(&self) -> usize {
        match self {
            WalkAction::Left => 0,
            WalkAction::Right => 1,
        }
    }
}

/// Toy 1-D random-walk environment used across all integration tests.
///
/// The agent starts at position 3 on a lattice `[0, 6]`. Reaching position 6
/// terminates the episode with reward `+1`; falling below 0 terminates with
/// reward `-1`. All other steps yield reward `0`. The episode is truncated
/// after `MAX_STEPS` steps regardless of outcome.
struct RandomWalkEnv {
    state: WalkState,
    steps: usize,
}

impl RandomWalkEnv {
    const START: i32 = 3;
    const GOAL: i32 = 6;
    const MAX_STEPS: usize = 20;

    /// Creates a `RandomWalkEnv` at the start position.
    fn new() -> Self {
        Self {
            state: WalkState {
                position: Self::START,
            },
            steps: 0,
        }
    }
}

impl RandomWalkEnv {
    fn observe(&self) -> WalkObservation {
        WalkObservation {
            position: self.state.position,
        }
    }
}

impl Environment<1, 1> for RandomWalkEnv {
    type ObservationType = WalkObservation;
    type ActionType = WalkAction;

    fn reset(&mut self) -> Result<Snapshot<WalkObservation>, EnvironmentError> {
        self.state = WalkState {
            position: Self::START,
        };
        self.steps = 0;
        Ok(Snapshot::running(self.observe(), ScalarReward::default()))
    }

    fn step(&mut self, action: WalkAction) -> Result<Snapshot<WalkObservation>, EnvironmentError> {
        let delta = match action {
            WalkAction::Left => -1,
            WalkAction::Right => 1,
        };
        let next = self.state.position + delta;
        self.steps += 1;

        if next < 0 {
            self.state = WalkState { position: 0 };
            return Ok(Snapshot::terminated(self.observe(), ScalarReward(-1.0)));
        }
        if next >= Self::GOAL {
            self.state = WalkState {
                position: Self::GOAL,
            };
            return Ok(Snapshot::terminated(self.observe(), ScalarReward(1.0)));
        }

        self.state = WalkState { position: next };

        if self.steps >= Self::MAX_STEPS {
            Ok(Snapshot::truncated(self.observe(), ScalarReward::default()))
        } else {
            Ok(Snapshot::running(self.observe(), ScalarReward::default()))
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Verifies a random-walk episode loop terminates and accumulates a positive total reward.
#[test]
#[allow(clippy::float_cmp)]
fn full_episode_loop_reaches_goal() {
    let mut env = RandomWalkEnv::new();

    let initial = env.reset().expect("reset");
    assert_eq!(initial.status, EpisodeStatus::Running);
    assert_eq!(f32::from(initial.reward), 0.0);
    assert_eq!(initial.observation.position, RandomWalkEnv::START);

    // Three right-moves take position from 3 to 6 (goal).
    let mut last = env.step(WalkAction::Right).expect("step 1");
    assert!(!last.status.is_done());
    last = env.step(WalkAction::Right).expect("step 2");
    assert!(!last.status.is_done());
    last = env.step(WalkAction::Right).expect("step 3 -> goal");

    assert!(last.status.is_done());
    assert!(last.status.is_terminated());
    assert!(!last.status.is_truncated());
    assert_eq!(f32::from(last.reward), 1.0);
    assert_eq!(last.observation.position, RandomWalkEnv::GOAL);
}
