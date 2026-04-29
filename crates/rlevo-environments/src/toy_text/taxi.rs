//! Taxi-v3 environment.
//!
//! A 5×5 grid-world where a taxi must navigate to a passenger, pick them up, and drop them off
//! at the correct destination. The grid contains hard-coded east–west walls and four named
//! pickup/dropoff locations.
//!
//! ## State space
//!
//! 500 discrete states encoded as `((row×5 + col)×5 + passenger_loc)×4 + destination`.
//!
//! - **Taxi position**: 5×5 = 25 cells
//! - **Passenger location**: 0–3 (at a named location) or 4 (in the taxi)
//! - **Destination**: 0–3 (R, G, Y, B)
//!
//! ## Named locations
//!
//! | Index | Name | Grid position |
//! |-------|------|---------------|
//! | 0     | R    | (0, 0)        |
//! | 1     | G    | (0, 4)        |
//! | 2     | Y    | (4, 0)        |
//! | 3     | B    | (4, 3)        |
//!
//! ## Action space
//!
//! Six actions via [`TaxiAction`]: `South` (0), `North` (1), `East` (2), `West` (3),
//! `Pickup` (4), `Dropoff` (5).
//!
//! ## Reward
//!
//! | Event              | Reward |
//! |--------------------|--------|
//! | Movement step      | −1     |
//! | Correct dropoff    | +20 (terminates) |
//! | Illegal pickup     | −10    |
//! | Illegal dropoff    | −10    |
//!
//! ## Variants
//!
//! - **Rainy mode** (`is_rainy`): 80% intended direction, 10% each perpendicular.
//! - **Fickle passenger** (`fickle_passenger`): after pickup, the first movement step has a
//!   30% chance to resample the destination.

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rlevo_core::action::DiscreteAction;
use rlevo_core::base::{Action, Observation, State};
use rlevo_core::environment::{Environment, EnvironmentError, SnapshotBase};
use rlevo_core::reward::ScalarReward;
use rlevo_core::state::StateError;
use serde::{Deserialize, Serialize};

// ── map constants ─────────────────────────────────────────────────────────────

/// Named pickup/dropoff locations: R=(0,0), G=(0,4), Y=(4,0), B=(4,3).
const LOCS: [(u8, u8); 4] = [(0, 0), (0, 4), (4, 0), (4, 3)];

/// Walls between adjacent cells (East–West only; all represented as (row, min_col, max_col)).
/// A wall blocks movement between `(row, min_col)` and `(row, max_col)`.
const WALLS: &[(u8, u8, u8)] = &[
    (0, 1, 2),
    (1, 1, 2),
    (3, 0, 1),
    (3, 2, 3),
    (4, 0, 1),
    (4, 2, 3),
];

fn has_wall(row: u8, from_col: u8, to_col: u8) -> bool {
    let (min_c, max_c) = if from_col < to_col {
        (from_col, to_col)
    } else {
        (to_col, from_col)
    };
    WALLS
        .iter()
        .any(|&(r, c0, c1)| r == row && c0 == min_c && c1 == max_c)
}

// ── config ────────────────────────────────────────────────────────────────────

/// Configuration for the [`Taxi`] environment.
///
/// Use [`TaxiConfig::builder`] to enable optional stochastic modes, or
/// [`TaxiConfig::default`] for the standard deterministic variant.
///
/// # Examples
///
/// ```rust
/// use rlevo_environments::toy_text::taxi::TaxiConfig;
///
/// let cfg = TaxiConfig::builder()
///     .is_rainy(true)
///     .fickle_passenger(true)
///     .seed(42)
///     .build();
/// ```
#[derive(Debug, Clone, Default)]
pub struct TaxiConfig {
    /// When `true`, movement directions slip: 80% intended, 10% each perpendicular.
    pub is_rainy: bool,
    /// When `true`, the first movement after pickup has a 30% chance to resample the destination.
    pub fickle_passenger: bool,
    /// Seed used to initialise the RNG when the environment is created. Default: `0`.
    pub seed: u64,
}

impl TaxiConfig {
    /// Returns a builder for constructing a `TaxiConfig`.
    pub fn builder() -> TaxiConfigBuilder {
        TaxiConfigBuilder::default()
    }
}

/// Builder for [`TaxiConfig`].
#[derive(Default)]
pub struct TaxiConfigBuilder {
    is_rainy: bool,
    fickle_passenger: bool,
    seed: u64,
}

impl TaxiConfigBuilder {
    /// Enables or disables rainy stochastic transitions.
    pub fn is_rainy(mut self, v: bool) -> Self {
        self.is_rainy = v;
        self
    }

    /// Enables or disables fickle-passenger destination resampling.
    pub fn fickle_passenger(mut self, v: bool) -> Self {
        self.fickle_passenger = v;
        self
    }

    /// Sets the RNG seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Builds the [`TaxiConfig`].
    pub fn build(self) -> TaxiConfig {
        TaxiConfig {
            is_rainy: self.is_rainy,
            fickle_passenger: self.fickle_passenger,
            seed: self.seed,
        }
    }
}

// ── state ─────────────────────────────────────────────────────────────────────

/// Full state: taxi position, passenger location, and destination.
#[derive(Debug, Clone)]
pub struct TaxiState {
    /// Taxi row in `[0, 5)`.
    pub taxi_row: u8,
    /// Taxi column in `[0, 5)`.
    pub taxi_col: u8,
    /// Passenger location: `0–3` = at named location (index into `LOCS`); `4` = in the taxi.
    pub passenger_loc: u8,
    /// Destination index `0–3` into the named locations array (R, G, Y, B).
    pub destination: u8,
}

impl TaxiState {
    fn encode(&self) -> u16 {
        ((self.taxi_row as u16 * 5 + self.taxi_col as u16) * 5 + self.passenger_loc as u16) * 4
            + self.destination as u16
    }
}

impl TryFrom<u16> for TaxiState {
    type Error = StateError;
    fn try_from(mut id: u16) -> Result<Self, Self::Error> {
        if id >= 500 {
            return Err(StateError::InvalidData(format!(
                "TaxiState id {id} out of range [0, 500)"
            )));
        }
        let destination = (id % 4) as u8;
        id /= 4;
        let passenger_loc = (id % 5) as u8;
        id /= 5;
        let taxi_col = (id % 5) as u8;
        id /= 5;
        let taxi_row = id as u8;
        Ok(TaxiState {
            taxi_row,
            taxi_col,
            passenger_loc,
            destination,
        })
    }
}

impl From<TaxiState> for u16 {
    fn from(s: TaxiState) -> u16 {
        s.encode()
    }
}

impl State<1> for TaxiState {
    type Observation = TaxiObservation;

    fn shape() -> [usize; 1] {
        [500]
    }

    fn observe(&self) -> TaxiObservation {
        TaxiObservation {
            state_id: self.encode(),
        }
    }

    fn is_valid(&self) -> bool {
        self.taxi_row < 5 && self.taxi_col < 5 && self.passenger_loc <= 4 && self.destination < 4
    }
}

// ── observation ───────────────────────────────────────────────────────────────

/// Agent-visible observation: integer state id in `[0, 500)`.
///
/// Encoded as `((taxi_row×5 + taxi_col)×5 + passenger_loc)×4 + destination`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaxiObservation {
    /// Packed state id in the range `[0, 500)`.
    pub state_id: u16,
}

impl Observation<1> for TaxiObservation {
    fn shape() -> [usize; 1] {
        [500]
    }
}

// ── action ────────────────────────────────────────────────────────────────────

/// Six-action Taxi space: four movement directions plus pickup and dropoff.
///
/// East–West movement respects the hard-coded grid walls; blocked moves are no-ops that
/// still cost −1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaxiAction {
    /// Move one row toward row 4 (south).
    South = 0,
    /// Move one row toward row 0 (north).
    North = 1,
    /// Move one column toward col 4 (east), if no wall blocks the move.
    East = 2,
    /// Move one column toward col 0 (west), if no wall blocks the move.
    West = 3,
    /// Pick up the passenger if the taxi is at the passenger's named location; costs −10 otherwise.
    Pickup = 4,
    /// Drop off the passenger if the taxi is at the destination and the passenger is aboard; costs −10 otherwise.
    Dropoff = 5,
}

impl Action<1> for TaxiAction {
    fn shape() -> [usize; 1] {
        [1]
    }
    fn is_valid(&self) -> bool {
        true
    }
}

impl DiscreteAction<1> for TaxiAction {
    const ACTION_COUNT: usize = 6;

    fn from_index(index: usize) -> Self {
        match index {
            0 => TaxiAction::South,
            1 => TaxiAction::North,
            2 => TaxiAction::East,
            3 => TaxiAction::West,
            4 => TaxiAction::Pickup,
            5 => TaxiAction::Dropoff,
            _ => panic!("TaxiAction index {index} out of range [0, 6)"),
        }
    }

    fn to_index(&self) -> usize {
        *self as usize
    }
}

impl TaxiAction {
    fn perpendiculars(self) -> [TaxiAction; 2] {
        match self {
            TaxiAction::South => [TaxiAction::East, TaxiAction::West],
            TaxiAction::North => [TaxiAction::West, TaxiAction::East],
            TaxiAction::East => [TaxiAction::North, TaxiAction::South],
            TaxiAction::West => [TaxiAction::South, TaxiAction::North],
            _ => [TaxiAction::North, TaxiAction::South], // unused
        }
    }
}

// ── movement helpers ──────────────────────────────────────────────────────────

fn attempt_move(row: u8, col: u8, action: TaxiAction) -> (u8, u8) {
    match action {
        TaxiAction::South if row < 4 => (row + 1, col),
        TaxiAction::North if row > 0 => (row - 1, col),
        TaxiAction::East if col < 4 && !has_wall(row, col, col + 1) => (row, col + 1),
        TaxiAction::West if col > 0 && !has_wall(row, col - 1, col) => (row, col - 1),
        _ => (row, col),
    }
}

// ── environment ───────────────────────────────────────────────────────────────

/// Taxi-v3 environment.
#[derive(Debug)]
pub struct Taxi {
    state: TaxiState,
    config: TaxiConfig,
    rng: StdRng,
    /// Tracks whether the fickle destination resample is armed (fickle mode only).
    fickle_armed: bool,
}

impl Taxi {
    /// Creates a [`Taxi`] environment with the given configuration.
    pub fn with_config(config: TaxiConfig) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        Self {
            state: TaxiState {
                taxi_row: 0,
                taxi_col: 0,
                passenger_loc: 0,
                destination: 1,
            },
            config,
            rng,
            fickle_armed: false,
        }
    }

    fn sample_initial_state(&mut self) -> TaxiState {
        let taxi_row = self.rng.random_range(0u8..5);
        let taxi_col = self.rng.random_range(0u8..5);
        let passenger_loc = self.rng.random_range(0u8..4);
        // Destination must differ from passenger_loc.
        let mut destination = self.rng.random_range(0u8..4);
        while destination == passenger_loc {
            destination = self.rng.random_range(0u8..4);
        }
        TaxiState {
            taxi_row,
            taxi_col,
            passenger_loc,
            destination,
        }
    }

    fn resolve_movement(&mut self, action: TaxiAction) -> TaxiAction {
        if !self.config.is_rainy {
            return action;
        }
        let r = self.rng.random_range(0u32..10);
        if r < 8 {
            action
        } else if r == 8 {
            action.perpendiculars()[0]
        } else {
            action.perpendiculars()[1]
        }
    }
}

impl Environment<1, 1, 1> for Taxi {
    type StateType = TaxiState;
    type ObservationType = TaxiObservation;
    type ActionType = TaxiAction;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, TaxiObservation, ScalarReward>;

    fn new(_render: bool) -> Self {
        Self::with_config(TaxiConfig::default())
    }

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.state = self.sample_initial_state();
        self.fickle_armed = false;
        Ok(SnapshotBase::running(
            self.state.observe(),
            ScalarReward(0.0),
        ))
    }

    fn step(&mut self, action: TaxiAction) -> Result<Self::SnapshotType, EnvironmentError> {
        let reward;
        let mut terminated = false;

        match action {
            TaxiAction::South | TaxiAction::North | TaxiAction::East | TaxiAction::West => {
                let effective = self.resolve_movement(action);
                let (nr, nc) = attempt_move(self.state.taxi_row, self.state.taxi_col, effective);
                self.state.taxi_row = nr;
                self.state.taxi_col = nc;

                // Fickle passenger: first movement after pickup resamples destination.
                if self.config.fickle_passenger && self.fickle_armed {
                    self.fickle_armed = false;
                    if self.rng.random_range(0u32..10) < 3 {
                        let mut new_dest = self.rng.random_range(0u8..4);
                        while new_dest == self.state.destination {
                            new_dest = self.rng.random_range(0u8..4);
                        }
                        self.state.destination = new_dest;
                    }
                }

                reward = -1.0;
            }
            TaxiAction::Pickup => {
                let taxi_at = (self.state.taxi_row, self.state.taxi_col);
                let pass = self.state.passenger_loc;
                if pass < 4 && LOCS[pass as usize] == taxi_at {
                    self.state.passenger_loc = 4; // in taxi
                    self.fickle_armed = true;
                    reward = -1.0;
                } else {
                    reward = -10.0;
                }
            }
            TaxiAction::Dropoff => {
                let taxi_at = (self.state.taxi_row, self.state.taxi_col);
                let dest_loc = LOCS[self.state.destination as usize];
                if self.state.passenger_loc == 4 && taxi_at == dest_loc {
                    self.state.passenger_loc = self.state.destination;
                    reward = 20.0;
                    terminated = true;
                } else {
                    reward = -10.0;
                }
            }
        }

        let obs = self.state.observe();
        if terminated {
            Ok(SnapshotBase::terminated(obs, ScalarReward(reward)))
        } else {
            Ok(SnapshotBase::running(obs, ScalarReward(reward)))
        }
    }
}

#[cfg(test)]
/// Unit tests for [`Taxi`], covering state encoding, wall collisions, pickup/dropoff rewards,
/// stochastic modes, and RNG determinism.
mod tests {
    use super::*;
    use rlevo_core::action::DiscreteAction;
    use rlevo_core::base::Observation;
    use rlevo_core::environment::Snapshot;

    fn make_env() -> Taxi {
        Taxi::with_config(TaxiConfig::default())
    }

    #[test]
    /// Verifies the discrete action count matches the six-action Taxi space.
    fn action_count() {
        assert_eq!(TaxiAction::ACTION_COUNT, 6);
    }

    #[test]
    /// Verifies `from_index` and `to_index` are inverses for all valid action indices.
    fn action_roundtrip() {
        for i in 0..TaxiAction::ACTION_COUNT {
            assert_eq!(TaxiAction::from_index(i).to_index(), i);
        }
    }

    #[test]
    /// Verifies the observation shape matches the 500-state space.
    fn obs_shape() {
        assert_eq!(TaxiObservation::shape(), [500]);
    }

    #[test]
    /// Verifies that state-id encoding and decoding are mutual inverses for all 500 states.
    fn state_id_round_trip() {
        // 500 total states; passenger_loc 4 = in-taxi is valid in the encoding.
        for id in 0u16..500 {
            let state = TaxiState::try_from(id).unwrap();
            assert_eq!(u16::from(state), id, "round-trip failed for id {id}");
        }
    }

    #[test]
    /// Verifies that moving into a wall is a no-op and still costs −1.
    fn wall_collision_is_no_op_cost_minus_one() {
        let mut env = make_env();
        env.reset().unwrap();
        // Place taxi at (0, 1): moving East is blocked by the wall between col 1 and col 2.
        env.state = TaxiState {
            taxi_row: 0,
            taxi_col: 1,
            passenger_loc: 0,
            destination: 1,
        };
        let pos_before = (env.state.taxi_row, env.state.taxi_col);
        let snap = env.step(TaxiAction::East).unwrap();
        let r: f32 = (*snap.reward()).into();
        assert_eq!(r, -1.0);
        assert_eq!((env.state.taxi_row, env.state.taxi_col), pos_before);
    }

    #[test]
    /// Verifies that a pickup attempt at the wrong cell costs −10.
    fn pickup_at_wrong_location_costs_ten() {
        let mut env = make_env();
        env.reset().unwrap();
        // Passenger at location 0 (0,0), taxi at (2,2): pickup must fail.
        env.state = TaxiState {
            taxi_row: 2,
            taxi_col: 2,
            passenger_loc: 0,
            destination: 1,
        };
        let snap = env.step(TaxiAction::Pickup).unwrap();
        let r: f32 = (*snap.reward()).into();
        assert_eq!(r, -10.0);
    }

    #[test]
    /// Verifies a correct dropoff yields +20 and terminates the episode.
    fn dropoff_terminates_with_plus_twenty() {
        let mut env = make_env();
        env.reset().unwrap();
        // Passenger in taxi, destination 1 = G = (0,4), taxi at (0,4).
        env.state = TaxiState {
            taxi_row: 0,
            taxi_col: 4,
            passenger_loc: 4,
            destination: 1,
        };
        let snap = env.step(TaxiAction::Dropoff).unwrap();
        let r: f32 = (*snap.reward()).into();
        assert_eq!(r, 20.0);
        assert!(snap.is_terminated());
    }

    #[test]
    /// Verifies the rainy-mode slip distribution: 80% intended, 10% each perpendicular.
    fn is_rainy_slip_distribution() {
        let cfg = TaxiConfig::builder().is_rainy(true).seed(17).build();
        let mut env = Taxi::with_config(cfg);
        env.reset().unwrap();

        let n = 10_000u32;
        let (mut intended, mut perp1, mut perp2) = (0u32, 0u32, 0u32);
        for _ in 0..n {
            // Place taxi at (2,2), move South → intended (3,2), perp East (2,3), West (2,1).
            env.state = TaxiState {
                taxi_row: 2,
                taxi_col: 2,
                passenger_loc: 2,
                destination: 0,
            };
            env.step(TaxiAction::South).unwrap();
            match (env.state.taxi_row, env.state.taxi_col) {
                (3, 2) => intended += 1,
                (2, 3) => perp1 += 1,
                (2, 1) => perp2 += 1,
                _ => {}
            }
        }
        let tol = 3.0 * (0.1_f32 * 0.9 / n as f32).sqrt();
        let p_int = intended as f32 / n as f32;
        let p_p1 = perp1 as f32 / n as f32;
        let p_p2 = perp2 as f32 / n as f32;
        assert!((p_int - 0.8).abs() < tol * 2.0, "intended={p_int}");
        assert!((p_p1 - 0.1).abs() < tol, "perp1={p_p1}");
        assert!((p_p2 - 0.1).abs() < tol, "perp2={p_p2}");
    }

    #[test]
    /// Verifies the fickle-passenger mode resamples the destination with ≈30% probability.
    fn fickle_passenger_30pct() {
        let cfg = TaxiConfig::builder()
            .fickle_passenger(true)
            .seed(31)
            .build();
        let mut env = Taxi::with_config(cfg);
        env.reset().unwrap();

        let n = 10_000u32;
        let mut changed = 0u32;
        for _ in 0..n {
            let orig_dest = 1u8;
            // Position taxi at passenger location 0 (R = (0,0)), destination 1 (G).
            env.state = TaxiState {
                taxi_row: 0,
                taxi_col: 0,
                passenger_loc: 0,
                destination: orig_dest,
            };
            env.fickle_armed = false;
            // Pickup
            env.step(TaxiAction::Pickup).unwrap();
            assert!(env.fickle_armed);
            // One movement step triggers fickle check.
            env.state.taxi_row = 1; // direct state mutation, no wall issues
            env.step(TaxiAction::South).unwrap();
            if env.state.destination != orig_dest {
                changed += 1;
            }
        }
        let p = changed as f32 / n as f32;
        let tol = 3.0 * (0.3_f32 * 0.7 / n as f32).sqrt();
        assert!((p - 0.3).abs() < tol, "fickle p={p}, expected ≈0.3");
    }

    #[test]
    /// Verifies that two rainy-mode environments seeded identically produce the same cumulative reward.
    fn determinism() {
        let cfg = TaxiConfig::builder().is_rainy(true).seed(55).build();
        let run = || {
            let mut env = Taxi::with_config(cfg.clone());
            let mut total = 0.0_f32;
            for _ in 0..3 {
                env.reset().unwrap();
                for _ in 0..10 {
                    let snap = env.step(TaxiAction::South).unwrap();
                    let r: f32 = (*snap.reward()).into();
                    total += r;
                    if snap.is_done() {
                        break;
                    }
                }
            }
            total
        };
        assert!((run() - run()).abs() < 1e-5, "determinism check failed");
    }
}
