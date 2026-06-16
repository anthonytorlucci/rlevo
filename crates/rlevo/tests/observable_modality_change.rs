//! Cross-crate proof that an [`Environment`] can expose an observation whose
//! tensor order differs from its state's order (`R != SR`), driven by the
//! [`Observable`] projection trait (issue #62, ADR 0019).
//!
//! The environment holds a compact rank-1 `MockRamState` and builds every
//! snapshot from `state.project()` — the rank-2 pixel projection — rather than
//! `state.observe()`. This is the modality-changing POMDP shape (RAM behind
//! pixels) that `State::observe()` structurally cannot express. The whole point
//! is that this compiles and round-trips end-to-end with **no** change to the
//! `Environment`/`Snapshot` contract.

use rlevo_core::base::{Action, Observation, State};
use rlevo_core::environment::{
    Environment, EnvironmentError, EpisodeStatus, Snapshot, SnapshotBase,
};
use rlevo_core::reward::ScalarReward;
use rlevo_core::state::Observable;

use serde::{Deserialize, Serialize};

/// Rank-2 pixel observation: a 2x2 grid of bits (the projected modality).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct MockRamObservation {
    pixels: [[u8; 2]; 2],
}

impl Observation<2> for MockRamObservation {
    fn shape() -> [usize; 2] {
        [2, 2]
    }
}

/// Rank-1 "full" observation: the raw RAM byte. Required because `State<1>`
/// pins its own `observe()` output to rank 1; the modality change lives on the
/// separate `Observable<2>` impl.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct MockRamByte {
    byte: u8,
}

impl Observation<1> for MockRamByte {
    fn shape() -> [usize; 1] {
        [1]
    }
}

/// Compact rank-1 state: one byte of emulator RAM.
#[derive(Debug, Clone)]
struct MockRamState {
    byte: u8,
}

impl State<1> for MockRamState {
    type Observation = MockRamByte;

    fn shape() -> [usize; 1] {
        [1]
    }

    fn observe(&self) -> Self::Observation {
        MockRamByte { byte: self.byte }
    }

    fn is_valid(&self) -> bool {
        true
    }

    fn numel(&self) -> usize {
        1
    }
}

impl Observable<2> for MockRamState {
    type Observation = MockRamObservation;

    fn project(&self) -> Self::Observation {
        let b = self.byte;
        MockRamObservation {
            pixels: [[b & 1, (b >> 1) & 1], [(b >> 2) & 1, (b >> 3) & 1]],
        }
    }
}

/// Minimal rank-1 action: increment or decrement the RAM byte. Core ships only
/// the `Action`/`DiscreteAction` *traits*, so a concrete action mock is needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MockAction {
    Inc,
    Dec,
}

impl Action<1> for MockAction {
    fn shape() -> [usize; 1] {
        [1]
    }

    fn is_valid(&self) -> bool {
        true
    }
}

/// A modality-changing environment: rank-1 state, rank-2 observation, rank-1
/// action — i.e. `Environment<2, 1, 1>`, with `R(2) != SR(1)`.
#[derive(Debug)]
struct ModalityEnv {
    state: MockRamState,
    steps: usize,
}

impl ModalityEnv {
    fn new() -> Self {
        Self {
            state: MockRamState { byte: 0 },
            steps: 0,
        }
    }
}

impl Environment<2, 1, 1> for ModalityEnv {
    type StateType = MockRamState;
    type ObservationType = MockRamObservation;
    type ActionType = MockAction;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<2, MockRamObservation, ScalarReward>;

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.state = MockRamState { byte: 0b0000 };
        self.steps = 0;
        // Build the snapshot from the rank-2 projection, NOT `observe()`.
        Ok(SnapshotBase::running(self.state.project(), ScalarReward(0.0)))
    }

    fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> {
        self.state.byte = match action {
            MockAction::Inc => self.state.byte.wrapping_add(1),
            MockAction::Dec => self.state.byte.wrapping_sub(1),
        };
        self.steps += 1;

        let obs = self.state.project();
        let reward = ScalarReward(1.0);
        let snap = if self.steps >= 4 {
            SnapshotBase::terminated(obs, reward)
        } else {
            SnapshotBase::running(obs, reward)
        };
        Ok(snap)
    }
}

/// The environment's observation is rank 2 while its state is rank 1 — the
/// modality change is visible through the snapshot the agent receives.
#[test]
fn test_modality_env_reset_yields_rank2_observation() {
    let mut env = ModalityEnv::new();
    let snap = env.reset().expect("reset succeeds");

    assert_eq!(snap.status(), EpisodeStatus::Running, "reset is Running");
    assert_eq!(
        MockRamObservation::shape(),
        [2, 2],
        "observation is rank 2, shape [2, 2]"
    );
    assert_eq!(
        <MockRamState as State<1>>::shape(),
        [1],
        "underlying state is rank 1"
    );
    assert_eq!(
        snap.observation().pixels,
        [[0, 0], [0, 0]],
        "byte 0 projects to an all-zero pixel grid"
    );
}

/// A full reset -> step loop round-trips through `project()` and terminates,
/// proving `R != SR` works end-to-end against the unchanged `Environment` API.
#[test]
fn test_modality_env_step_loop_terminates() {
    let mut env = ModalityEnv::new();
    env.reset().expect("reset succeeds");

    // Four increments: byte 0 -> 1 -> 2 -> 3 -> 4.
    let mut last = None;
    for _ in 0..4 {
        last = Some(env.step(MockAction::Inc).expect("step succeeds"));
    }
    let snap = last.expect("at least one step ran");

    assert!(snap.is_done(), "episode terminates after 4 steps");
    assert_eq!(snap.status(), EpisodeStatus::Terminated, "intrinsic termination");
    assert_eq!(
        snap.observation().pixels,
        // byte == 4 == 0b0100 -> bit2 set
        [[0, 0], [1, 0]],
        "byte 4 projects to bit-2 set in the 2x2 grid"
    );
    let reward: f32 = (*snap.reward()).into();
    assert!((reward - 1.0).abs() < 1e-6, "terminal step reward is 1.0");
}

/// A `Dec` from byte 0 wraps to 255, projecting to an all-ones pixel grid —
/// exercises the decrement action and confirms the projection sees the wrap.
#[test]
fn test_modality_env_decrement_wraps() {
    let mut env = ModalityEnv::new();
    env.reset().expect("reset succeeds");

    let snap = env.step(MockAction::Dec).expect("step succeeds");

    assert_eq!(
        snap.observation().pixels,
        // byte 0 -> wrapping_sub(1) -> 255 == 0b11111111 -> low 4 bits all set
        [[1, 1], [1, 1]],
        "decrement wraps to 255, all low-nibble pixels set"
    );
}
