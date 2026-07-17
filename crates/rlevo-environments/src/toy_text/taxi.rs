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

use crate::episode::EpisodeGuard;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rlevo_core::action::DiscreteAction;
use rlevo_core::base::{Action, Observation, State};
use rlevo_core::config::{ConfigError, Validate};
use rlevo_core::environment::{
    ConstructableEnv, Environment, EnvironmentError, Sensor, Snapshot, SnapshotBase,
};
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

impl Validate for TaxiConfig {
    /// [`TaxiConfig`] carries only boolean toggles and a seed, so it has no
    /// numeric invariant to check; validation always succeeds.
    fn validate(&self) -> Result<(), ConfigError> {
        Ok(())
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
    fn shape() -> [usize; 1] {
        [500]
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

/// Taxi-v3 environment (5×5 grid, 500 discrete states, 6 actions).
///
/// Each episode begins with the taxi, passenger, and destination placed at randomly sampled
/// positions. The episode ends only on a correct [`TaxiAction::Dropoff`]; there is no step
/// limit built into the environment.
///
/// The RNG advances continuously across episodes — `reset()` does **not** reseed from
/// `config.seed`. Two `Taxi` instances created with the same seed produce identical trajectories.
///
/// Once an episode has ended (a correct [`TaxiAction::Dropoff`]), a further
/// [`step`](Environment::step) is rejected with
/// [`EnvironmentError::StepAfterEpisodeEnd`] rather than driving the taxi on;
/// call [`reset`](Environment::reset) to begin a new episode.
#[derive(Debug)]
pub struct Taxi {
    state: TaxiState,
    config: TaxiConfig,
    rng: StdRng,
    /// Tracks whether the fickle destination resample is armed (fickle mode only).
    fickle_armed: bool,
    /// Rejects a `step()` taken after the episode has already ended.
    guard: EpisodeGuard,
}

impl Taxi {
    /// Creates a [`Taxi`] environment with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `config` fails [`Validate`]. [`TaxiConfig`]
    /// currently has no numeric invariant, so this never fails in practice; the
    /// fallible signature keeps the construction contract uniform across
    /// environments.
    pub fn with_config(config: TaxiConfig) -> Result<Self, ConfigError> {
        config.validate()?;
        let rng = StdRng::seed_from_u64(config.seed);
        Ok(Self {
            state: TaxiState {
                taxi_row: 0,
                taxi_col: 0,
                passenger_loc: 0,
                destination: 1,
            },
            config,
            rng,
            fickle_armed: false,
            guard: EpisodeGuard::new(),
        })
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

impl ConstructableEnv for Taxi {
    fn new(_render: bool) -> Self {
        Self::with_config(TaxiConfig::default()).expect("default config must validate")
    }
}

impl Sensor<1, 1, 1> for Taxi {
    type Action = TaxiAction;
    type State = TaxiState;
    type Observation = TaxiObservation;

    /// Projects the resulting state onto its packed 500-state id; the observation
    /// is a pure function of the taxi/passenger/destination triple and ignores
    /// which action produced it.
    fn observe(&self, _action: &TaxiAction, next_state: &TaxiState) -> TaxiObservation {
        TaxiObservation {
            state_id: next_state.encode(),
        }
    }

    /// Projects the initial sampled state onto its packed state id.
    fn observe_reset(&self, state: &TaxiState) -> TaxiObservation {
        TaxiObservation {
            state_id: state.encode(),
        }
    }
}

impl Environment<1, 1, 1> for Taxi {
    type StateType = TaxiState;
    type ObservationType = TaxiObservation;
    type ActionType = TaxiAction;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, TaxiObservation, ScalarReward>;

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.guard.reset();
        self.state = self.sample_initial_state();
        self.fickle_armed = false;
        Ok(SnapshotBase::running(
            self.observe_reset(&self.state),
            ScalarReward(0.0),
        ))
    }

    fn step(&mut self, action: TaxiAction) -> Result<Self::SnapshotType, EnvironmentError> {
        // Must precede every state mutation *and* every RNG draw: `resolve_movement`
        // and the fickle-passenger resample both consume from `self.rng`, so a
        // rejected call that got past this point would advance the stream and break
        // the determinism guarantee (identical seeds → identical trajectories).
        self.guard.check()?;

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

        let obs = self.observe(&action, &self.state);
        let snapshot = if terminated {
            SnapshotBase::terminated(obs, ScalarReward(reward))
        } else {
            SnapshotBase::running(obs, ScalarReward(reward))
        };

        // Record from the snapshot we are about to emit, so the guard and the
        // snapshot can never disagree on any path.
        self.guard.record(snapshot.status());
        Ok(snapshot)
    }
}

// ---------------------------------------------------------------------------
// ASCII renderer
// ---------------------------------------------------------------------------

const LOC_LABELS: [char; 4] = ['R', 'G', 'Y', 'B'];

impl crate::render::AsciiRenderable for Taxi {
    fn render_ascii(&self) -> String {
        let mut out = String::new();
        let pass_loc = self.state.passenger_loc;
        let dest = self.state.destination as usize;
        let taxi = (self.state.taxi_row, self.state.taxi_col);

        let header = format!(
            "Taxi  pass={}  dest={}\n",
            passenger_label(pass_loc),
            LOC_LABELS[dest]
        );
        out.push_str(&header);

        for row in 0..5u8 {
            for col in 0..5u8 {
                out.push(taxi_cell_char(row, col, taxi, pass_loc, dest));
                if col + 1 < 5 {
                    out.push(if has_wall(row, col, col + 1) {
                        '|'
                    } else {
                        ' '
                    });
                }
            }
            out.push('\n');
        }
        out
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        use crate::render::palette::{AGENT_FG, AGENT_MODIFIER, GOAL_FG, GOAL_MODIFIER, WALL_FG};
        use crate::render::{Color, Modifier, SpanStyle, StyledFrame, StyledLine, StyledSpan};

        let pass_loc = self.state.passenger_loc;
        let dest = self.state.destination as usize;
        let taxi = (self.state.taxi_row, self.state.taxi_col);

        let mut lines: Vec<StyledLine> = Vec::with_capacity(6);

        // Label line — `Taxi` styled as agent.
        let header_label_style = SpanStyle::default()
            .fg(AGENT_FG)
            .with_modifier(AGENT_MODIFIER);
        let header_rest = format!(
            "  pass={}  dest={}",
            passenger_label(pass_loc),
            LOC_LABELS[dest]
        );
        lines.push(StyledLine::from_spans(vec![
            StyledSpan::new("Taxi", header_label_style),
            StyledSpan::raw(header_rest),
        ]));

        let agent_style = SpanStyle::default()
            .fg(AGENT_FG)
            .with_modifier(AGENT_MODIFIER);
        let dest_style = SpanStyle::default()
            .fg(GOAL_FG)
            .with_modifier(GOAL_MODIFIER);
        let passenger_style = SpanStyle::default()
            .fg(Color::Yellow)
            .with_modifier(Modifier::BOLD);
        let wall_style = SpanStyle::default().fg(WALL_FG);

        for row in 0..5u8 {
            let mut spans: Vec<StyledSpan> = Vec::new();
            let mut current_style = SpanStyle::default();
            let mut current_text = String::new();

            for col in 0..5u8 {
                let ch = taxi_cell_char(row, col, taxi, pass_loc, dest);
                let style = if ch == 'T' {
                    agent_style
                } else if ch == 'P' {
                    passenger_style
                } else if ch == 'D' {
                    dest_style
                } else {
                    SpanStyle::default()
                };
                push_styled(&mut spans, &mut current_style, &mut current_text, ch, style);

                if col + 1 < 5 {
                    let sep_ch = if has_wall(row, col, col + 1) {
                        '|'
                    } else {
                        ' '
                    };
                    let sep_style = if sep_ch == '|' {
                        wall_style
                    } else {
                        SpanStyle::default()
                    };
                    push_styled(
                        &mut spans,
                        &mut current_style,
                        &mut current_text,
                        sep_ch,
                        sep_style,
                    );
                }
            }
            if !current_text.is_empty() {
                spans.push(StyledSpan::new(current_text, current_style));
            }
            lines.push(StyledLine::from_spans(spans));
        }
        StyledFrame { lines }
    }
}

fn push_styled(
    spans: &mut Vec<crate::render::StyledSpan>,
    current_style: &mut crate::render::SpanStyle,
    current_text: &mut String,
    ch: char,
    style: crate::render::SpanStyle,
) {
    if style != *current_style && !current_text.is_empty() {
        spans.push(crate::render::StyledSpan::new(
            std::mem::take(current_text),
            *current_style,
        ));
    }
    *current_style = style;
    current_text.push(ch);
}

fn taxi_cell_char(row: u8, col: u8, taxi: (u8, u8), pass_loc: u8, dest: usize) -> char {
    if (row, col) == taxi {
        return 'T';
    }
    // Passenger glyph appears at its named location when not in taxi.
    if pass_loc < 4 && LOCS[pass_loc as usize] == (row, col) {
        return 'P';
    }
    if LOCS[dest] == (row, col) {
        return 'D';
    }
    for (idx, &loc) in LOCS.iter().enumerate() {
        if loc == (row, col) {
            return LOC_LABELS[idx];
        }
    }
    '.'
}

fn passenger_label(pass_loc: u8) -> &'static str {
    match pass_loc {
        0 => "R",
        1 => "G",
        2 => "Y",
        3 => "B",
        _ => "in-taxi",
    }
}

impl rlevo_core::render::payload::TabularPayloadSource for Taxi {
    fn tabular_snapshot(&self) -> rlevo_core::render::payload::TabularSnapshot {
        use rlevo_core::render::payload::{
            TabularCell, TabularGrid, TabularLayout, TabularMarker, TabularMarkerKind,
            TabularSnapshot,
        };
        // Fixed 5×5 grid. Inter-cell walls are omitted from the structured
        // view (a deliberate stopgap, like locomotion's 2D projection); the
        // taxi / passenger / destination markers carry the task dynamics.
        const SIZE: u16 = 5;
        let cells = vec![TabularCell::Empty; (SIZE * SIZE) as usize];
        let mut markers = Vec::with_capacity(3);
        // Taxi (LOCS entries are (row, col)).
        markers.push(TabularMarker {
            x: u16::from(self.state.taxi_col),
            y: u16::from(self.state.taxi_row),
            kind: TabularMarkerKind::Agent,
        });
        // Passenger: at a named location, or riding in the taxi (loc == 4).
        let (prow, pcol) = if self.state.passenger_loc < 4 {
            LOCS[self.state.passenger_loc as usize]
        } else {
            (self.state.taxi_row, self.state.taxi_col)
        };
        markers.push(TabularMarker {
            x: u16::from(pcol),
            y: u16::from(prow),
            kind: TabularMarkerKind::Passenger,
        });
        // Destination.
        let (drow, dcol) = LOCS[self.state.destination as usize];
        markers.push(TabularMarker {
            x: u16::from(dcol),
            y: u16::from(drow),
            kind: TabularMarkerKind::Destination,
        });
        TabularSnapshot {
            layout: TabularLayout::Grid(TabularGrid {
                width: SIZE,
                height: SIZE,
                cells,
                markers,
            }),
        }
    }
}

#[cfg(test)]
/// Unit tests for [`Taxi`], covering state encoding, wall collisions, pickup/dropoff rewards,
/// stochastic modes, and RNG determinism.
mod tests {
    use super::*;
    use crate::episode::assert_rejects_post_terminal_step;
    use rlevo_core::action::DiscreteAction;
    use rlevo_core::base::Observation;
    use rlevo_core::environment::{EpisodeStatus, Snapshot};

    fn make_env() -> Taxi {
        Taxi::with_config(TaxiConfig::default()).expect("valid config")
    }

    /// Passenger aboard, taxi parked on destination `G = LOCS[1] = (0, 4)`:
    /// the next [`TaxiAction::Dropoff`] is correct and terminates the episode.
    fn about_to_drop_off() -> TaxiState {
        TaxiState {
            taxi_row: 0,
            taxi_col: 4,
            passenger_loc: 4,
            destination: 1,
        }
    }

    /// Resets `env`, pins it one correct `Dropoff` from the goal, and takes it —
    /// returning the terminal snapshot. Consumes RNG only inside `reset()`: the
    /// `Dropoff` branch of `step` makes no draw.
    fn drive_to_correct_dropoff(env: &mut Taxi) -> SnapshotBase<1, TaxiObservation, ScalarReward> {
        env.reset().expect("reset must succeed");
        env.state = about_to_drop_off();
        env.step(TaxiAction::Dropoff)
            .expect("a correct dropoff must succeed")
    }

    #[test]
    fn default_config_validates() {
        assert!(TaxiConfig::default().validate().is_ok());
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
        let mut env = Taxi::with_config(cfg).expect("valid config");
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
        let mut env = Taxi::with_config(cfg).expect("valid config");
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
            let mut env = Taxi::with_config(cfg.clone()).expect("valid config");
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

    // ── post-terminal step guard (issue #105) ────────────────────────────────

    #[test]
    /// Verifies that stepping after a correct dropoff is rejected with
    /// `StepAfterEpisodeEnd { status: Terminated }` and leaves the taxi where it was.
    fn test_taxi_step_after_correct_dropoff_is_rejected() {
        let mut env = make_env();
        let terminal = drive_to_correct_dropoff(&mut env);

        let r: f32 = (*terminal.reward()).into();
        assert_eq!(r, 20.0, "a correct dropoff must pay +20");
        assert!(
            terminal.is_terminated(),
            "a correct dropoff must terminate the episode"
        );

        let before = env.state.encode();
        let err = env
            .step(TaxiAction::South)
            .expect_err("the episode has ended; a further step must not drive the taxi on");
        match err {
            EnvironmentError::StepAfterEpisodeEnd { status } => assert_eq!(
                status,
                EpisodeStatus::Terminated,
                "the error must carry Terminated, the status that ended the episode"
            ),
            other => panic!("expected StepAfterEpisodeEnd, got {other:?}"),
        }
        assert_eq!(
            env.state.encode(),
            before,
            "a rejected post-terminal step must leave the state untouched"
        );
    }

    #[test]
    /// Verifies [`Taxi`] satisfies the shared post-terminal-step conformance check.
    fn test_taxi_rejects_post_terminal_step_conformance() {
        let mut env = make_env();
        assert_rejects_post_terminal_step(&mut env, drive_to_correct_dropoff, TaxiAction::North);
    }

    #[test]
    /// Verifies `reset()` re-opens a terminated environment for a new episode.
    fn test_taxi_reset_reopens_env_after_termination() {
        let mut env = make_env();
        drive_to_correct_dropoff(&mut env);
        assert!(
            env.step(TaxiAction::South).is_err(),
            "the episode has ended; a step before reset must be rejected"
        );

        env.reset().expect("reset must succeed after termination");
        let snap = env
            .step(TaxiAction::South)
            .expect("reset() must re-open the environment for a new episode");
        assert!(
            !snap.is_done(),
            "the first step of a fresh episode must not be done"
        );
    }

    #[test]
    /// Verifies a rejected post-terminal step consumes no randomness: the guard's
    /// `check()?` precedes the rainy-slip draw, so two identically seeded envs stay
    /// in lock-step even when one of them is stepped after its episode ended.
    fn test_taxi_rejected_post_terminal_step_does_not_advance_rng() {
        let cfg = TaxiConfig::builder().is_rainy(true).seed(99).build();

        // A full rainy trajectory of state ids, driven purely by the RNG stream.
        fn trajectory(env: &mut Taxi) -> Vec<u16> {
            env.reset().expect("reset must succeed");
            let mut ids = vec![env.state.encode()];
            for _ in 0..20 {
                let snap = env
                    .step(TaxiAction::South)
                    .expect("movement never terminates the episode");
                assert!(!snap.is_done(), "South must not end a Taxi episode");
                ids.push(env.state.encode());
            }
            ids
        }

        let mut probed = Taxi::with_config(cfg.clone()).expect("valid config");
        let mut clean = Taxi::with_config(cfg).expect("valid config");

        drive_to_correct_dropoff(&mut probed);
        drive_to_correct_dropoff(&mut clean);

        // `probed` alone attempts (and is denied) several post-terminal steps.
        for _ in 0..5 {
            let err = probed
                .step(TaxiAction::South)
                .expect_err("post-terminal steps must be rejected");
            assert!(
                matches!(err, EnvironmentError::StepAfterEpisodeEnd { .. }),
                "expected StepAfterEpisodeEnd, got {err:?}"
            );
        }

        assert_eq!(
            trajectory(&mut probed),
            trajectory(&mut clean),
            "rejected post-terminal steps must not draw from the RNG; a probed env must \
             replay the same rainy trajectory as an unprobed one seeded identically"
        );
    }

    #[test]
    fn render_styled_matches_ascii() {
        use crate::render::AsciiRenderable;

        let mut env = Taxi::with_config(TaxiConfig::default()).expect("valid config");
        env.reset().unwrap();
        let plain = env.render_ascii();
        let styled = env.render_styled();
        let plain_no_trailing: String = plain.lines().collect::<Vec<_>>().join("\n");
        assert_eq!(styled.plain_text(), plain_no_trailing);
    }

    #[test]
    fn render_styled_uses_palette_consts() {
        use crate::render::AsciiRenderable;
        use crate::render::palette::{AGENT_FG, AGENT_MODIFIER, GOAL_FG, WALL_FG};

        let mut env = Taxi::with_config(TaxiConfig::default()).expect("valid config");
        env.reset().unwrap();
        // Pin a state with a guaranteed-visible destination (LOCS[1] = (0,4))
        // while the taxi sits at (2, 2) and the passenger waits at LOCS[0].
        env.state = TaxiState {
            taxi_row: 2,
            taxi_col: 2,
            passenger_loc: 0,
            destination: 1,
        };
        let styled = env.render_styled();

        let label = styled.lines[0]
            .spans
            .iter()
            .find(|s| s.text == "Taxi")
            .expect("Taxi label span present");
        assert_eq!(label.style.fg, Some(AGENT_FG));
        assert!(label.style.modifier.contains(AGENT_MODIFIER));

        let mut found_taxi_glyph = false;
        let mut found_dest = false;
        let mut found_wall = false;
        for line in styled.lines.iter().skip(1) {
            for span in &line.spans {
                if span.text.contains('T') {
                    assert_eq!(span.style.fg, Some(AGENT_FG));
                    found_taxi_glyph = true;
                }
                if span.text.contains('D') {
                    assert_eq!(span.style.fg, Some(GOAL_FG));
                    found_dest = true;
                }
                if span.text.contains('|') {
                    assert_eq!(span.style.fg, Some(WALL_FG));
                    found_wall = true;
                }
            }
        }
        assert!(found_taxi_glyph, "taxi T glyph not styled");
        assert!(found_dest, "destination D glyph not styled");
        assert!(found_wall, "wall | glyph not styled");
    }

    #[test]
    fn render_ascii_within_width_budget() {
        use crate::render::AsciiRenderable;

        let mut env = Taxi::with_config(TaxiConfig::default()).expect("valid config");
        env.reset().unwrap();
        for line in env.render_ascii().lines() {
            assert!(
                line.chars().count() <= 80,
                "line exceeds 80 cols: {line:?} ({} chars)",
                line.chars().count()
            );
        }
    }
}
