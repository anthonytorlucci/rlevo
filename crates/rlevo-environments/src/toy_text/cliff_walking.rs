//! CliffWalking-v1 environment.
//!
//! A 4×12 grid-world where the agent navigates from a fixed start to a fixed goal while
//! avoiding a cliff that lines the bottom row. The environment is deterministic by default;
//! enable `is_slippery` for stochastic transitions.
//!
//! ## Layout
//!
//! ```text
//! o o o o o o o o o o o o
//! o o o o o o o o o o o o
//! o o o o o o o o o o o o
//! S C C C C C C C C C C G   (row 3)
//! ```
//!
//! - **S** — start `(3, 0)`
//! - **G** — goal `(3, 11)`
//! - **C** — cliff `(3, 1..=10)`
//!
//! ## Reward
//!
//! | Event          | Reward |
//! |----------------|--------|
//! | Step onto cliff | −100 (teleports to start; episode continues) |
//! | Any other step  | −1   |
//! | Goal step       | −1 (terminates) |
//!
//! ## Observation space
//!
//! Integer state id in `[0, 48)` encoded as `row × 12 + col`.
//!
//! ## Action space
//!
//! Four discrete directions via [`CliffWalkingAction`]: `Up` (0), `Right` (1), `Down` (2), `Left` (3).
//!
//! ## Slippery mode
//!
//! When enabled, intended direction succeeds with probability 1/3; each of the two
//! perpendicular directions occurs with probability 1/3.

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rlevo_core::action::DiscreteAction;
use rlevo_core::base::{Action, Observation, State};
use rlevo_core::environment::{Environment, EnvironmentError, SnapshotBase};
use rlevo_core::reward::ScalarReward;
use rlevo_core::state::StateError;
use serde::{Deserialize, Serialize};

const NROW: u8 = 4;
const NCOL: u8 = 12;
const START: (u8, u8) = (3, 0);
const GOAL: (u8, u8) = (3, 11);

// ── config ────────────────────────────────────────────────────────────────────

/// Configuration for the [`CliffWalking`] environment.
///
/// Build with [`CliffWalkingConfig::default`] for the standard deterministic variant, or use
/// [`CliffWalkingConfig::builder`] to customise slipperiness and the RNG seed.
///
/// # Examples
///
/// ```
/// use rlevo_envs::toy_text::cliff_walking::CliffWalkingConfig;
///
/// let cfg = CliffWalkingConfig::builder()
///     .is_slippery(true)
///     .seed(7)
///     .build();
/// ```
#[derive(Debug, Clone, Default)]
pub struct CliffWalkingConfig {
    /// When `true`, intended direction succeeds with probability 1/3; each perpendicular
    /// direction occurs with probability 1/3.
    pub is_slippery: bool,
    /// Seed used to initialise the RNG when the environment is created. Default: `0`.
    pub seed: u64,
}

impl CliffWalkingConfig {
    /// Returns a builder for constructing a `CliffWalkingConfig`.
    pub fn builder() -> CliffWalkingConfigBuilder {
        CliffWalkingConfigBuilder::default()
    }
}

/// Builder for [`CliffWalkingConfig`].
#[derive(Default)]
pub struct CliffWalkingConfigBuilder {
    is_slippery: bool,
    seed: u64,
}

impl CliffWalkingConfigBuilder {
    /// Enables or disables the stochastic slipping behaviour.
    pub fn is_slippery(mut self, v: bool) -> Self {
        self.is_slippery = v;
        self
    }

    /// Sets the RNG seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Builds the [`CliffWalkingConfig`].
    pub fn build(self) -> CliffWalkingConfig {
        CliffWalkingConfig {
            is_slippery: self.is_slippery,
            seed: self.seed,
        }
    }
}

// ── state ─────────────────────────────────────────────────────────────────────

/// Full state: grid position `(row, col)` in a 4×12 grid.
#[derive(Debug, Clone)]
pub struct CliffWalkingState {
    /// Row index in `[0, 4)`, where row 3 contains the cliff and the goal.
    pub row: u8,
    /// Column index in `[0, 12)`.
    pub col: u8,
}

impl CliffWalkingState {
    fn state_id(&self) -> u16 {
        self.row as u16 * NCOL as u16 + self.col as u16
    }
}

impl TryFrom<u16> for CliffWalkingState {
    type Error = StateError;
    fn try_from(id: u16) -> Result<Self, Self::Error> {
        let n = NROW as u16 * NCOL as u16;
        if id >= n {
            return Err(StateError::InvalidData(format!(
                "CliffWalkingState id {id} out of range [0, {n})"
            )));
        }
        Ok(CliffWalkingState {
            row: (id / NCOL as u16) as u8,
            col: (id % NCOL as u16) as u8,
        })
    }
}

impl From<CliffWalkingState> for u16 {
    fn from(s: CliffWalkingState) -> u16 {
        s.state_id()
    }
}

impl State<1> for CliffWalkingState {
    type Observation = CliffWalkingObservation;

    fn shape() -> [usize; 1] {
        [NROW as usize * NCOL as usize]
    }

    fn observe(&self) -> CliffWalkingObservation {
        CliffWalkingObservation {
            state_id: self.state_id(),
        }
    }

    fn is_valid(&self) -> bool {
        self.row < NROW && self.col < NCOL
    }
}

// ── observation ───────────────────────────────────────────────────────────────

/// Agent-visible observation: integer state id in `[0, 48)`.
///
/// Encoded as `row × 12 + col`. Convertible to/from [`CliffWalkingState`] via
/// `TryFrom<u16>` / `From<CliffWalkingState>`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliffWalkingObservation {
    /// Linear grid index: `row × NCOL + col`.
    pub state_id: u16,
}

impl Observation<1> for CliffWalkingObservation {
    fn shape() -> [usize; 1] {
        [NROW as usize * NCOL as usize]
    }
}

// ── action ────────────────────────────────────────────────────────────────────

/// Four-direction action space for grid navigation.
///
/// Movements are clamped at grid boundaries — attempting to leave the grid is a no-op
/// that still costs −1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliffWalkingAction {
    /// Move one row toward row 0 (north).
    Up = 0,
    /// Move one column toward col 11 (east).
    Right = 1,
    /// Move one row toward row 3 (south).
    Down = 2,
    /// Move one column toward col 0 (west).
    Left = 3,
}

impl Action<1> for CliffWalkingAction {
    fn shape() -> [usize; 1] {
        [1]
    }
    fn is_valid(&self) -> bool {
        true
    }
}

impl DiscreteAction<1> for CliffWalkingAction {
    const ACTION_COUNT: usize = 4;

    fn from_index(index: usize) -> Self {
        match index {
            0 => CliffWalkingAction::Up,
            1 => CliffWalkingAction::Right,
            2 => CliffWalkingAction::Down,
            3 => CliffWalkingAction::Left,
            _ => panic!("CliffWalkingAction index {index} out of range [0, 4)"),
        }
    }

    fn to_index(&self) -> usize {
        *self as usize
    }
}

impl CliffWalkingAction {
    fn perpendiculars(self) -> [CliffWalkingAction; 2] {
        match self {
            CliffWalkingAction::Up | CliffWalkingAction::Down => {
                [CliffWalkingAction::Left, CliffWalkingAction::Right]
            }
            CliffWalkingAction::Left | CliffWalkingAction::Right => {
                [CliffWalkingAction::Up, CliffWalkingAction::Down]
            }
        }
    }
}

// ── dynamics helpers ──────────────────────────────────────────────────────────

fn apply_action(row: u8, col: u8, action: CliffWalkingAction) -> (u8, u8) {
    match action {
        CliffWalkingAction::Up => (row.saturating_sub(1), col),
        CliffWalkingAction::Down => ((row + 1).min(NROW - 1), col),
        CliffWalkingAction::Right => (row, (col + 1).min(NCOL - 1)),
        CliffWalkingAction::Left => (row, col.saturating_sub(1)),
    }
}

fn is_cliff(row: u8, col: u8) -> bool {
    row == 3 && (1..=10).contains(&col)
}

// ── environment ───────────────────────────────────────────────────────────────

/// CliffWalking-v1 environment.
///
/// Deterministic by default. Use `TimeLimit` wrapper for a step cap since the
/// environment itself never truncates.
#[derive(Debug)]
pub struct CliffWalking {
    state: CliffWalkingState,
    config: CliffWalkingConfig,
    rng: StdRng,
}

impl CliffWalking {
    /// Creates a [`CliffWalking`] environment with the given configuration.
    pub fn with_config(config: CliffWalkingConfig) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        Self {
            state: CliffWalkingState {
                row: START.0,
                col: START.1,
            },
            config,
            rng,
        }
    }

    fn resolve_action(&mut self, action: CliffWalkingAction) -> CliffWalkingAction {
        if !self.config.is_slippery {
            return action;
        }
        // 1/3 intended, 1/3 each perpendicular.
        let r = self.rng.random_range(0u32..3);
        if r == 0 {
            action
        } else {
            action.perpendiculars()[(r - 1) as usize]
        }
    }
}

impl Environment<1, 1, 1> for CliffWalking {
    type StateType = CliffWalkingState;
    type ObservationType = CliffWalkingObservation;
    type ActionType = CliffWalkingAction;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, CliffWalkingObservation, ScalarReward>;

    fn new(_render: bool) -> Self {
        Self::with_config(CliffWalkingConfig::default())
    }

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.state = CliffWalkingState {
            row: START.0,
            col: START.1,
        };
        Ok(SnapshotBase::running(
            self.state.observe(),
            ScalarReward(0.0),
        ))
    }

    fn step(&mut self, action: CliffWalkingAction) -> Result<Self::SnapshotType, EnvironmentError> {
        let effective = self.resolve_action(action);
        let (nr, nc) = apply_action(self.state.row, self.state.col, effective);

        if is_cliff(nr, nc) {
            self.state.row = START.0;
            self.state.col = START.1;
            return Ok(SnapshotBase::running(
                self.state.observe(),
                ScalarReward(-100.0),
            ));
        }

        self.state.row = nr;
        self.state.col = nc;

        if (nr, nc) == GOAL {
            Ok(SnapshotBase::terminated(
                self.state.observe(),
                ScalarReward(-1.0),
            ))
        } else {
            Ok(SnapshotBase::running(
                self.state.observe(),
                ScalarReward(-1.0),
            ))
        }
    }
}

impl From<(u8, u8)> for CliffWalkingState {
    fn from((row, col): (u8, u8)) -> Self {
        CliffWalkingState { row, col }
    }
}

#[cfg(test)]
/// Unit tests for [`CliffWalking`], covering state encoding, cliff/goal transitions,
/// boundary behaviour, slippery distributions, and RNG determinism.
mod tests {
    use super::*;
    use rlevo_core::action::DiscreteAction;
    use rlevo_core::base::Observation;
    use rlevo_core::environment::Snapshot;

    fn make_env() -> CliffWalking {
        CliffWalking::with_config(CliffWalkingConfig::default())
    }

    #[test]
    /// Verifies the discrete action count matches the four-direction action space.
    fn action_count() {
        assert_eq!(CliffWalkingAction::ACTION_COUNT, 4);
    }

    #[test]
    /// Verifies that state-id encoding and decoding are mutual inverses for all 48 states.
    fn state_id_encoding() {
        let total = NROW as u16 * NCOL as u16;
        for id in 0..total {
            let state = CliffWalkingState::try_from(id).unwrap();
            assert_eq!(u16::from(state), id, "round-trip failed for id {id}");
        }
    }

    #[test]
    /// Verifies the observation shape matches the 48-cell grid.
    fn obs_shape() {
        assert_eq!(CliffWalkingObservation::shape(), [48]);
    }

    #[test]
    /// Verifies a cliff step teleports to start, costs −100, and does not terminate the episode.
    fn cliff_step_teleports_and_costs_100() {
        let mut env = make_env();
        env.reset().unwrap();
        // Start (3,0): move right → (3,1) = cliff.
        let snap = env.step(CliffWalkingAction::Right).unwrap();
        let r: f32 = (*snap.reward()).into();
        assert_eq!(r, -100.0);
        assert!(!snap.is_done(), "cliff must not terminate episode");
        // Agent should be back at start.
        assert_eq!(
            env.state.state_id(),
            CliffWalkingState::from((3u8, 0u8)).state_id()
        );
    }

    #[test]
    /// Verifies stepping onto the goal terminates the episode with reward −1.
    fn goal_step_terminates_with_minus_one() {
        let mut env = make_env();
        env.reset().unwrap();
        // Place agent at (3,10) (one step left of goal, but that's cliff…)
        // Better: place at (2,11) and step down.
        env.state = CliffWalkingState { row: 2, col: 11 };
        let snap = env.step(CliffWalkingAction::Down).unwrap();
        let r: f32 = (*snap.reward()).into();
        assert_eq!(r, -1.0);
        assert!(snap.is_terminated());
    }

    #[test]
    /// Verifies that attempting to move off the grid boundary is a no-op.
    fn off_grid_stays_in_place() {
        let mut env = make_env();
        env.reset().unwrap();
        // Start at (3,0): move Up → (2,0); then move Left → stay (col already 0).
        env.step(CliffWalkingAction::Up).unwrap();
        let snap_before = env.state.state_id();
        env.step(CliffWalkingAction::Left).unwrap();
        assert_eq!(
            env.state.state_id(),
            snap_before,
            "off-grid move must be no-op"
        );
    }

    #[test]
    /// Verifies the optimal (13-step) path yields a total reward of −13.
    fn shortest_path_minus_13() {
        // Optimal: up (1 step) + right×11 + down (1 step) = 13 steps at −1 each = −13.
        let mut env = make_env();
        env.reset().unwrap();
        let mut total = 0.0_f32;
        // One step up from (3,0) → (2,0)
        let snap = env.step(CliffWalkingAction::Up).unwrap();
        {
            let r: f32 = (*snap.reward()).into();
            total += r;
        }
        // Eleven steps right: (2,0) → … → (2,11)
        for _ in 0..11 {
            let snap = env.step(CliffWalkingAction::Right).unwrap();
            let r: f32 = (*snap.reward()).into();
            total += r;
        }
        // One step down (2,11) → (3,11) = GOAL
        let snap = env.step(CliffWalkingAction::Down).unwrap();
        {
            let r: f32 = (*snap.reward()).into();
            total += r;
        }
        assert!(snap.is_done(), "goal must terminate episode");
        assert!(
            (total - (-13.0)).abs() < 1e-5,
            "optimal path must yield -13, got {total}"
        );
    }

    #[test]
    /// Verifies the slippery-mode slip distribution is approximately uniform (1/3 each direction).
    fn slippery_distribution_matches_1_3() {
        let cfg = CliffWalkingConfig::builder()
            .is_slippery(true)
            .seed(7)
            .build();
        let mut env = CliffWalking::with_config(cfg);
        env.reset().unwrap();
        // Place at (1, 6): move Right; count how often we end up at (1,7), (0,6), (2,6).
        let n = 12_000u32;
        let (mut right_count, mut up_count, mut down_count) = (0u32, 0u32, 0u32);
        for _ in 0..n {
            env.state = CliffWalkingState { row: 1, col: 6 };
            env.step(CliffWalkingAction::Right).unwrap();
            match (env.state.row, env.state.col) {
                (1, 7) => right_count += 1,
                (0, 6) => up_count += 1,
                (2, 6) => down_count += 1,
                _ => {}
            }
        }
        let p_right = right_count as f32 / n as f32;
        let p_up = up_count as f32 / n as f32;
        let p_down = down_count as f32 / n as f32;
        let tol = 3.0 * (1.0 / 3.0 * 2.0 / 3.0 / n as f32).sqrt();
        assert!(
            (p_right - 1.0 / 3.0).abs() < tol,
            "intended slip p={p_right}"
        );
        assert!((p_up - 1.0 / 3.0).abs() < tol, "perp-up p={p_up}");
        assert!((p_down - 1.0 / 3.0).abs() < tol, "perp-down p={p_down}");
    }

    #[test]
    /// Verifies that two slippery environments seeded identically produce the same cumulative reward.
    fn determinism() {
        let cfg = CliffWalkingConfig::builder()
            .is_slippery(true)
            .seed(3)
            .build();
        let run = || {
            let mut env = CliffWalking::with_config(cfg.clone());
            env.reset().unwrap();
            let mut total = 0.0_f32;
            for _ in 0..20 {
                let snap = env.step(CliffWalkingAction::Right).unwrap();
                let r: f32 = (*snap.reward()).into();
                total += r;
                if snap.is_done() {
                    break;
                }
            }
            total
        };
        assert!((run() - run()).abs() < 1e-5, "determinism check failed");
    }
}
