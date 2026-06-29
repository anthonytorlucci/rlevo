//! Santa Fe Trail artificial-ant environment — a dependency-free POMDP benchmark.
//!
//! The artificial ant (Jefferson et al. 1991; Koza 1992, pp. 147–155) walks a
//! 32×32 **toroidal** grid carrying food pellets laid out along the canonical
//! *Santa Fe Trail*. The ant perceives a single bit — *is there food in the
//! cell directly ahead?* — and may [`Move`](SantaFeAntAction::Move) forward,
//! [`TurnLeft`](SantaFeAntAction::TurnLeft), or
//! [`TurnRight`](SantaFeAntAction::TurnRight). The episode lasts a fixed step
//! budget (default 600); the goal is to eat all 89 pellets.
//!
//! # Why this is a POMDP (and why it needs memory)
//!
//! The one-bit percept is a massive cardinality reduction of the true state at
//! **constant tensor order** — an information-reducing projection exactly like
//! dropping the velocities from CartPole, *not* a modality change. Crucially,
//! perceptual aliasing makes the *optimal* policy provably require internal
//! memory: a memoryless reactive map `{0, 1} → action` cannot cross the trail's
//! single, double, and L-shaped gaps, where the ant must step forward over an
//! empty cell to reach food beyond it. [`SantaFeAntState`] signals this at the
//! type level with [`MarkovState::is_markov`] returning `false`.
//!
//! Because the projection keeps the tensor order at 1, the environment is a
//! plain [`Environment<1, 1, 1>`] and touches **no** new core trait. In
//! particular [`Observable`](rlevo_core::state::Observable) — the rank-changing
//! projection seam — is deliberately **not** used here.
//!
//! # Example
//!
//! ```rust
//! use rlevo_core::environment::{ConstructableEnv, Environment, Snapshot};
//! use rlevo_environments::classic::{SantaFeAnt, SantaFeAntAction};
//!
//! let mut env: SantaFeAnt = <SantaFeAnt as ConstructableEnv>::new(false);
//! let snap = <SantaFeAnt as Environment<1, 1, 1>>::reset(&mut env).expect("reset");
//! assert!(!snap.is_done());
//!
//! // Eating the food directly ahead yields +1.0.
//! let snap = <SantaFeAnt as Environment<1, 1, 1>>::step(&mut env, SantaFeAntAction::Move)
//!     .expect("valid action");
//! assert_eq!(f32::from(*snap.reward()), 1.0);
//! ```

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use rlevo_core::action::DiscreteAction;
use rlevo_core::base::{Action, Observation, State, TensorConversionError, TensorConvertible};
use rlevo_core::environment::{ConstructableEnv, Environment, EnvironmentError, SnapshotBase};
use rlevo_core::reward::ScalarReward;
use rlevo_core::state::MarkovState;
use serde::{Deserialize, Serialize};

use crate::direction::Direction;

/// Side length of the (square, toroidal) trail grid.
pub const GRID_SIZE: usize = 32;

/// Number of motor primitives the ant exposes ([`Move`](SantaFeAntAction::Move),
/// [`TurnLeft`](SantaFeAntAction::TurnLeft), [`TurnRight`](SantaFeAntAction::TurnRight)).
pub const ACTION_COUNT: usize = 3;

/// Total food pellets on the canonical Santa Fe Trail.
pub const TOTAL_PELLETS: u32 = 89;

/// Default per-episode step budget (Koza's 600-move limit).
pub const DEFAULT_MAX_STEPS: usize = 600;

// ---------------------------------------------------------------------------
// Action
// ---------------------------------------------------------------------------

/// The ant's three motor primitives. Each consumes exactly one time step.
///
/// Note the genetic-programming *control* nodes of the classic formulation —
/// `IF-FOOD-AHEAD`, `PROGN2`, `PROGN3` — are **policy** constructs, not
/// environment actions, and are intentionally absent here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SantaFeAntAction {
    /// Step forward one cell (toroidal), eating any food there.
    Move,
    /// Rotate 90° counter-clockwise in place (no translation).
    TurnLeft,
    /// Rotate 90° clockwise in place (no translation).
    TurnRight,
}

impl Action<1> for SantaFeAntAction {
    fn shape() -> [usize; 1] {
        [ACTION_COUNT]
    }

    fn is_valid(&self) -> bool {
        // Every variant is always a legal move.
        true
    }
}

impl DiscreteAction<1> for SantaFeAntAction {
    const ACTION_COUNT: usize = ACTION_COUNT;

    /// Map a flat index to an action: `0 → Move`, `1 → TurnLeft`, `2 → TurnRight`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= ACTION_COUNT` (i.e. `index >= 3`).
    fn from_index(index: usize) -> Self {
        match index {
            0 => Self::Move,
            1 => Self::TurnLeft,
            2 => Self::TurnRight,
            other => panic!("SantaFeAntAction index {other} out of range [0, {ACTION_COUNT})"),
        }
    }

    fn to_index(&self) -> usize {
        match self {
            Self::Move => 0,
            Self::TurnLeft => 1,
            Self::TurnRight => 2,
        }
    }
}

impl<B: Backend> TensorConvertible<1, B> for SantaFeAntAction {
    /// One-hot encoding of the action as a rank-1 tensor of length [`ACTION_COUNT`].
    fn to_tensor(&self, device: &<B as burn::tensor::backend::BackendTypes>::Device) -> Tensor<B, 1> {
        let mut one_hot: [f32; ACTION_COUNT] = [0.0; ACTION_COUNT];
        one_hot[self.to_index()] = 1.0;
        Tensor::from_floats(one_hot, device)
    }

    /// Reconstruct an action from a one-hot tensor by argmax.
    ///
    /// # Errors
    ///
    /// Returns [`TensorConversionError`] if the tensor shape is not
    /// `[ACTION_COUNT]` or its data cannot be read.
    fn from_tensor(tensor: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let dims = tensor.dims();
        if dims.as_slice() != [ACTION_COUNT] {
            return Err(TensorConversionError {
                message: format!("expected shape [{ACTION_COUNT}], got {dims:?}"),
            });
        }
        let data = tensor.into_data();
        let values: Vec<f32> = data.to_vec().map_err(|e| TensorConversionError {
            message: format!("failed to extract tensor data: {e:?}"),
        })?;
        let (argmax, _) = values.iter().enumerate().fold(
            (0_usize, f32::NEG_INFINITY),
            |(i_best, v_best), (i, &v)| {
                if v > v_best { (i, v) } else { (i_best, v_best) }
            },
        );
        Ok(Self::from_index(argmax))
    }
}

// ---------------------------------------------------------------------------
// Observation
// ---------------------------------------------------------------------------

/// The single food-ahead bit: `true` iff the cell directly ahead holds food.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SantaFeAntObservation {
    /// Whether the cell one step ahead (in the current heading) holds a pellet.
    pub food_ahead: bool,
}

impl Observation<1> for SantaFeAntObservation {
    fn shape() -> [usize; 1] {
        [1]
    }
}

impl<B: Backend> TensorConvertible<1, B> for SantaFeAntObservation {
    /// Encode the food-ahead bit as a length-1 tensor (`[1.0]` / `[0.0]`).
    fn to_tensor(&self, device: &<B as burn::tensor::backend::BackendTypes>::Device) -> Tensor<B, 1> {
        let value: f32 = if self.food_ahead { 1.0 } else { 0.0 };
        Tensor::from_floats([value], device)
    }

    /// Decode the food-ahead bit, thresholding the single element at `0.5`.
    ///
    /// # Errors
    ///
    /// Returns [`TensorConversionError`] if the tensor shape is not `[1]` or its
    /// data cannot be read.
    fn from_tensor(tensor: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let dims = tensor.dims();
        if dims.as_slice() != [1] {
            return Err(TensorConversionError {
                message: format!("expected shape [1], got {dims:?}"),
            });
        }
        let data = tensor.into_data();
        let values: Vec<f32> = data.to_vec().map_err(|e| TensorConversionError {
            message: format!("failed to extract tensor data: {e:?}"),
        })?;
        Ok(Self {
            food_ahead: values[0] > 0.5,
        })
    }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// Full environment state: the remaining-food grid plus the ant's pose.
///
/// The 32×32 `food` grid and the pose are **internal** — they are not part of
/// the observation-compatible tensor [`shape`](State::shape), which is `[1]`
/// (the food-ahead bit). This is what keeps the env at order 1 while the true
/// state is far larger, the defining feature of the POMDP.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SantaFeAntState {
    /// Remaining pellets; `food[row][col] == true` means a pellet is present.
    food: [[bool; GRID_SIZE]; GRID_SIZE],
    /// Ant row (toroidal, `0..GRID_SIZE`), increasing downward (south).
    row: usize,
    /// Ant column (toroidal, `0..GRID_SIZE`), increasing rightward (east).
    col: usize,
    /// Heading the ant currently faces.
    heading: Direction,
    /// Pellets still on the grid (`== count(food)`).
    pellets_remaining: u32,
    /// Steps elapsed this episode.
    steps: u32,
}

impl SantaFeAntState {
    /// The `(row, col)` of the cell one step ahead, wrapping toroidally.
    fn cell_ahead(&self) -> (usize, usize) {
        let (dx, dy): (i32, i32) = self.heading.delta();
        let ahead_col: usize = wrap_step(self.col, dx);
        let ahead_row: usize = wrap_step(self.row, dy);
        (ahead_row, ahead_col)
    }

    /// Live count of remaining pellets on the grid.
    fn count_food(&self) -> u32 {
        let count: usize = self.food.iter().flatten().filter(|&&cell| cell).count();
        u32::try_from(count).unwrap_or(u32::MAX)
    }

    /// The ant's `(row, col)` position.
    #[must_use]
    pub fn position(&self) -> (usize, usize) {
        (self.row, self.col)
    }

    /// The ant's current heading.
    #[must_use]
    pub fn heading(&self) -> Direction {
        self.heading
    }

    /// Pellets still remaining on the grid.
    #[must_use]
    pub fn pellets_remaining(&self) -> u32 {
        self.pellets_remaining
    }

    /// Steps elapsed in the current episode.
    #[must_use]
    pub fn steps(&self) -> u32 {
        self.steps
    }
}

impl State<1> for SantaFeAntState {
    type Observation = SantaFeAntObservation;

    fn shape() -> [usize; 1] {
        [1]
    }

    fn observe(&self) -> Self::Observation {
        let (ahead_row, ahead_col): (usize, usize) = self.cell_ahead();
        SantaFeAntObservation {
            food_ahead: self.food[ahead_row][ahead_col],
        }
    }

    fn is_valid(&self) -> bool {
        self.row < GRID_SIZE
            && self.col < GRID_SIZE
            && self.pellets_remaining == self.count_food()
    }
}

/// Non-Markov: the one-bit percept does **not** summarise the history needed to
/// act optimally, so agents must stack observations or carry recurrent memory.
/// This single override is the type-level statement of the environment's purpose.
impl MarkovState for SantaFeAntState {
    fn is_markov() -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Construction parameters for [`SantaFeAnt`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SantaFeAntConfig {
    /// Per-episode step budget; the episode truncates once it is reached.
    pub max_steps: usize,
    /// Whether the env is rendered (reserved; rendering is a planned follow-up).
    pub render: bool,
}

impl Default for SantaFeAntConfig {
    fn default() -> Self {
        Self {
            max_steps: DEFAULT_MAX_STEPS,
            render: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

/// The Santa Fe Trail artificial-ant environment.
///
/// A deterministic [`Environment<1, 1, 1>`] (no RNG in its dynamics): given the
/// same action sequence it reproduces the same trajectory, and
/// [`reset`](Environment::reset) always restores the identical start state.
///
/// # Example: a memoryless reflex policy plateaus below 89
///
/// The pedagogical payoff. The reflex map "if food ahead, move; else turn
/// right" has no memory, so it spins in place at the first gap it cannot see
/// across and eats only a handful of the 89 pellets before the step budget runs
/// out — a concrete demonstration of why the optimal policy needs memory:
///
/// ```rust
/// use rlevo_core::environment::{ConstructableEnv, Environment, Snapshot};
/// use rlevo_environments::classic::{SantaFeAnt, SantaFeAntAction, SantaFeAntObservation};
///
/// let mut env: SantaFeAnt = <SantaFeAnt as ConstructableEnv>::new(false);
/// let mut snap = <SantaFeAnt as Environment<1, 1, 1>>::reset(&mut env).expect("reset");
///
/// let mut eaten: f32 = 0.0;
/// loop {
///     let obs: &SantaFeAntObservation = snap.observation();
///     let action = if obs.food_ahead {
///         SantaFeAntAction::Move
///     } else {
///         SantaFeAntAction::TurnRight
///     };
///     snap = <SantaFeAnt as Environment<1, 1, 1>>::step(&mut env, action).expect("step");
///     eaten += f32::from(*snap.reward());
///     if snap.is_done() {
///         break;
///     }
/// }
/// assert!(eaten < 89.0, "a memoryless reflex cannot clear the trail (got {eaten})");
/// ```
#[derive(Debug, Clone)]
pub struct SantaFeAnt {
    state: SantaFeAntState,
    config: SantaFeAntConfig,
}

impl SantaFeAnt {
    /// Build the env from an explicit [`SantaFeAntConfig`].
    #[must_use]
    pub fn with_config(config: SantaFeAntConfig) -> Self {
        Self {
            state: Self::fresh_state(),
            config,
        }
    }

    /// The pristine start state: the full Santa Fe Trail, ant at the `S` marker
    /// facing east, zero steps elapsed.
    fn fresh_state() -> SantaFeAntState {
        let (food, (row, col)): ([[bool; GRID_SIZE]; GRID_SIZE], (usize, usize)) =
            parse_trail(SANTA_FE_TRAIL);
        let pellets: usize = food.iter().flatten().filter(|&&cell| cell).count();
        let pellets_remaining: u32 = u32::try_from(pellets).unwrap_or(u32::MAX);
        SantaFeAntState {
            food,
            row,
            col,
            heading: Direction::East,
            pellets_remaining,
            steps: 0,
        }
    }

    /// Borrow the current state (pose, remaining food, step count).
    #[must_use]
    pub fn state(&self) -> &SantaFeAntState {
        &self.state
    }

    /// Borrow the configuration.
    #[must_use]
    pub fn config(&self) -> &SantaFeAntConfig {
        &self.config
    }
}

impl ConstructableEnv for SantaFeAnt {
    fn new(render: bool) -> Self {
        Self::with_config(SantaFeAntConfig {
            render,
            ..SantaFeAntConfig::default()
        })
    }
}

impl Environment<1, 1, 1> for SantaFeAnt {
    type StateType = SantaFeAntState;
    type ObservationType = SantaFeAntObservation;
    type ActionType = SantaFeAntAction;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, SantaFeAntObservation, ScalarReward>;

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.state = Self::fresh_state();
        Ok(SnapshotBase::running(
            self.state.observe(),
            ScalarReward::new(0.0),
        ))
    }

    fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> {
        let reward: ScalarReward = match action {
            SantaFeAntAction::TurnLeft => {
                self.state.heading = self.state.heading.left();
                ScalarReward::new(0.0)
            }
            SantaFeAntAction::TurnRight => {
                self.state.heading = self.state.heading.right();
                ScalarReward::new(0.0)
            }
            SantaFeAntAction::Move => {
                let (new_row, new_col): (usize, usize) = self.state.cell_ahead();
                self.state.row = new_row;
                self.state.col = new_col;
                if self.state.food[new_row][new_col] {
                    self.state.food[new_row][new_col] = false;
                    self.state.pellets_remaining -= 1;
                    ScalarReward::new(1.0)
                } else {
                    ScalarReward::new(0.0)
                }
            }
        };
        self.state.steps += 1;

        let obs: SantaFeAntObservation = self.state.observe();
        let limit: u32 = u32::try_from(self.config.max_steps).unwrap_or(u32::MAX);
        let snapshot: Self::SnapshotType = if self.state.pellets_remaining == 0 {
            // Goal reached — the whole trail is cleared.
            SnapshotBase::terminated(obs, reward)
        } else if self.state.steps >= limit {
            // Time-limit reached — truncation, not a true terminal state.
            SnapshotBase::truncated(obs, reward)
        } else {
            SnapshotBase::running(obs, reward)
        };
        Ok(snapshot)
    }
}

// ---------------------------------------------------------------------------
// Trail literal + parsing
// ---------------------------------------------------------------------------

/// One toroidal step of `pos` (in `0..GRID_SIZE`) along one axis.
///
/// `delta` is the per-axis component from [`Direction::delta`], one of
/// `-1`, `0`, or `+1`. Computing the wrap with modular `usize` arithmetic avoids
/// sign-changing casts.
const fn wrap_step(pos: usize, delta: i32) -> usize {
    match delta {
        1 => (pos + 1) % GRID_SIZE,
        -1 => (pos + GRID_SIZE - 1) % GRID_SIZE,
        _ => pos,
    }
}

/// The canonical Santa Fe Trail: 89 pellets on a 32×32 grid.
///
/// Transcribed from DEAP `examples/gp/ant/santafe_trail.txt` (`#` = food,
/// `.` = empty, `S` = start, ant facing east), audited against Koza 1992
/// Fig. 7.1.
///
/// **Defect note (row 24).** The DEAP source file is malformed on this row: it
/// is 33 columns wide with a stray space byte (`0x20`) at column 6
/// (`...##.·.#####....#...`). The canonical Koza/Langton grid has no such cell;
/// the space is dropped here, yielding the correct 32-column
/// `...##..#####....#...............`. This preserves the 89-pellet count and a
/// true 32×32 grid; the invariant tests below lock it in.
const SANTA_FE_TRAIL: &str = "\
S###............................
...#............................
...#.....................###....
...#....................#....#..
...#....................#....#..
...####.#####........##.........
............#................#..
............#.......#...........
............#.......#........#..
............#.......#...........
....................#...........
............#................#..
............#...................
............#.......#.....###...
............#.......#..#........
.................#..............
................................
............#...........#.......
............#...#..........#....
............#...#...............
............#...#...............
............#...#.........#.....
............#..........#........
............#...................
...##..#####....#...............
.#..............#...............
.#..............#...............
.#......#######.................
.#.....#........................
.......#........................
..####..........................
................................";

/// Parse the ASCII trail into a food grid plus the derived `S` start position.
///
/// The `S` cell is treated as empty (no phantom pellet at the start). The start
/// pose is the single source of truth for where the ant begins — never assume
/// `(0, 0)`; it is read from the marker.
///
/// # Panics
///
/// Panics if the literal is not exactly `GRID_SIZE` rows of `GRID_SIZE` columns,
/// contains a character other than `#`/`.`/`S`, or does not contain exactly one
/// `S` marker. These are compile-fixed invariants of [`SANTA_FE_TRAIL`].
fn parse_trail(ascii: &str) -> ([[bool; GRID_SIZE]; GRID_SIZE], (usize, usize)) {
    let mut food: [[bool; GRID_SIZE]; GRID_SIZE] = [[false; GRID_SIZE]; GRID_SIZE];
    let mut start: Option<(usize, usize)> = None;

    let lines: Vec<&str> = ascii.lines().collect();
    assert!(
        lines.len() == GRID_SIZE,
        "trail must have {GRID_SIZE} rows, got {}",
        lines.len()
    );
    for (row, line) in lines.iter().enumerate() {
        assert!(
            line.len() == GRID_SIZE,
            "row {row} must have {GRID_SIZE} columns, got {}",
            line.len()
        );
        for (col, ch) in line.chars().enumerate() {
            match ch {
                '#' => food[row][col] = true,
                'S' => {
                    assert!(start.is_none(), "trail must contain exactly one start marker");
                    start = Some((row, col));
                }
                '.' => {}
                other => panic!("unexpected trail character {other:?} at ({row}, {col})"),
            }
        }
    }

    let start: (usize, usize) = start.expect("trail must contain a start marker 'S'");
    (food, start)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlevo_core::environment::Snapshot;

    type TestBackend = burn::backend::Flex;

    fn env() -> SantaFeAnt {
        <SantaFeAnt as ConstructableEnv>::new(false)
    }

    // -- Trail literal invariants ------------------------------------------

    #[test]
    fn trail_has_exactly_89_pellets_on_32x32() {
        let (food, _start): ([[bool; GRID_SIZE]; GRID_SIZE], (usize, usize)) =
            parse_trail(SANTA_FE_TRAIL);
        // `parse_trail` itself asserts the 32×32 shape (it would panic above
        // otherwise), so reaching here already proves the grid dimensions.
        let count: usize = food.iter().flatten().filter(|&&c| c).count();
        assert_eq!(count, 89);
    }

    #[test]
    fn start_is_derived_at_origin_and_empty() {
        let (food, start): ([[bool; GRID_SIZE]; GRID_SIZE], (usize, usize)) =
            parse_trail(SANTA_FE_TRAIL);
        // Verified against DEAP examples/gp/ant/santafe_trail.txt.
        assert_eq!(start, (0, 0));
        let (row, col): (usize, usize) = start;
        assert!(!food[row][col], "start cell must hold no pellet");
    }

    #[test]
    fn known_gap_cells_are_empty_with_food_neighbours() {
        let (food, _start): ([[bool; GRID_SIZE]; GRID_SIZE], (usize, usize)) =
            parse_trail(SANTA_FE_TRAIL);
        // Row 0: the `###` pellets immediately right of the start.
        assert!(food[0][1] && food[0][2] && food[0][3]);
        // Row 5 single horizontal gap: `...####.#####` — col 7 empty between food.
        assert!(!food[5][7] && food[5][6] && food[5][8]);
        // Row 24 (the corrected DEAP-defect row): cols 5,6 empty, 4 and 7 food.
        assert!(!food[24][5] && !food[24][6]);
        assert!(food[24][4] && food[24][7]);
    }

    // -- Dynamics ----------------------------------------------------------

    #[test]
    fn turns_rotate_without_translation_and_cost_a_step() {
        let mut e = env();
        let start_pos: (usize, usize) = e.state().position();
        assert_eq!(e.state().heading(), Direction::East);

        let _ = <SantaFeAnt as Environment<1, 1, 1>>::step(&mut e, SantaFeAntAction::TurnRight)
            .expect("step");
        assert_eq!(e.state().heading(), Direction::South);
        assert_eq!(e.state().position(), start_pos);
        assert_eq!(e.state().steps(), 1);

        let _ = <SantaFeAnt as Environment<1, 1, 1>>::step(&mut e, SantaFeAntAction::TurnLeft)
            .expect("step");
        assert_eq!(e.state().heading(), Direction::East);
        assert_eq!(e.state().position(), start_pos);
        assert_eq!(e.state().steps(), 2);
    }

    #[test]
    fn move_eats_food_emits_reward_and_decrements() {
        let mut e = env();
        let before: u32 = e.state().pellets_remaining();
        // Facing east from (0,0); (0,1) holds a pellet.
        let snap = <SantaFeAnt as Environment<1, 1, 1>>::step(&mut e, SantaFeAntAction::Move)
            .expect("step");
        assert_eq!(f32::from(*snap.reward()), 1.0);
        assert_eq!(e.state().position(), (0, 1));
        assert_eq!(e.state().pellets_remaining(), before - 1);
    }

    #[test]
    fn re_entering_an_eaten_cell_yields_no_reward() {
        let mut e = env();
        // Eat (0,1) then (0,2) moving east.
        let _ = <SantaFeAnt as Environment<1, 1, 1>>::step(&mut e, SantaFeAntAction::Move);
        let _ = <SantaFeAnt as Environment<1, 1, 1>>::step(&mut e, SantaFeAntAction::Move);
        let pellets: u32 = e.state().pellets_remaining();
        // Turn around (180°) and step back onto the now-eaten (0,1).
        let _ = <SantaFeAnt as Environment<1, 1, 1>>::step(&mut e, SantaFeAntAction::TurnLeft);
        let _ = <SantaFeAnt as Environment<1, 1, 1>>::step(&mut e, SantaFeAntAction::TurnLeft);
        let snap = <SantaFeAnt as Environment<1, 1, 1>>::step(&mut e, SantaFeAntAction::Move)
            .expect("step");
        assert_eq!(e.state().position(), (0, 1));
        assert_eq!(f32::from(*snap.reward()), 0.0);
        assert_eq!(e.state().pellets_remaining(), pellets);
    }

    #[test]
    fn move_into_empty_cell_yields_zero_reward() {
        let mut e = env();
        // Turn to face north; (31,0) wraps and is empty (row 31 is all dots).
        let _ = <SantaFeAnt as Environment<1, 1, 1>>::step(&mut e, SantaFeAntAction::TurnLeft);
        assert_eq!(e.state().heading(), Direction::North);
        let snap = <SantaFeAnt as Environment<1, 1, 1>>::step(&mut e, SantaFeAntAction::Move)
            .expect("step");
        assert_eq!(e.state().position(), (GRID_SIZE - 1, 0));
        assert_eq!(f32::from(*snap.reward()), 0.0);
    }

    // -- Toroidal wrap on both axes ---------------------------------------

    #[test]
    fn move_wraps_west_edge_to_east_edge() {
        let mut e = env();
        // Face west; from (0,0) a forward move wraps the column to GRID_SIZE-1.
        let _ = <SantaFeAnt as Environment<1, 1, 1>>::step(&mut e, SantaFeAntAction::TurnRight);
        let _ = <SantaFeAnt as Environment<1, 1, 1>>::step(&mut e, SantaFeAntAction::TurnRight);
        assert_eq!(e.state().heading(), Direction::West);
        let _ = <SantaFeAnt as Environment<1, 1, 1>>::step(&mut e, SantaFeAntAction::Move)
            .expect("step");
        assert_eq!(e.state().position(), (0, GRID_SIZE - 1));
        assert_eq!(e.state().heading(), Direction::West);
    }

    #[test]
    fn move_wraps_north_edge_to_south_edge() {
        let mut e = env();
        // Face north; from row 0 a forward move wraps the row to GRID_SIZE-1.
        let _ = <SantaFeAnt as Environment<1, 1, 1>>::step(&mut e, SantaFeAntAction::TurnLeft);
        assert_eq!(e.state().heading(), Direction::North);
        let _ = <SantaFeAnt as Environment<1, 1, 1>>::step(&mut e, SantaFeAntAction::Move)
            .expect("step");
        assert_eq!(e.state().position(), (GRID_SIZE - 1, 0));
    }

    // -- Observation under wrap, all four headings -------------------------

    #[test]
    fn observation_matches_cell_ahead_for_all_headings() {
        // From (0,0): east → (0,1) food; south → (1,0) empty;
        // west → (0,31) empty (wrap); north → (31,0) empty (wrap).
        let mut e = env();
        assert!(<SantaFeAnt as Environment<1, 1, 1>>::reset(&mut e)
            .expect("reset")
            .observation()
            .food_ahead); // east, (0,1) is food

        let mut e = env();
        let _ = <SantaFeAnt as Environment<1, 1, 1>>::step(&mut e, SantaFeAntAction::TurnRight);
        assert!(!e.state().observe().food_ahead); // south, (1,0) empty

        let mut e = env();
        let _ = <SantaFeAnt as Environment<1, 1, 1>>::step(&mut e, SantaFeAntAction::TurnRight);
        let _ = <SantaFeAnt as Environment<1, 1, 1>>::step(&mut e, SantaFeAntAction::TurnRight);
        assert!(!e.state().observe().food_ahead); // west, (0,31) empty

        let mut e = env();
        let _ = <SantaFeAnt as Environment<1, 1, 1>>::step(&mut e, SantaFeAntAction::TurnLeft);
        assert!(!e.state().observe().food_ahead); // north, (31,0) empty
    }

    // -- Termination -------------------------------------------------------

    #[test]
    fn episode_truncates_at_max_steps() {
        let mut e = SantaFeAnt::with_config(SantaFeAntConfig {
            max_steps: 5,
            render: false,
        });
        let mut snap = <SantaFeAnt as Environment<1, 1, 1>>::reset(&mut e).expect("reset");
        // Spin in place so no early termination from clearing the trail.
        for _ in 0..5 {
            assert!(!snap.is_done());
            snap = <SantaFeAnt as Environment<1, 1, 1>>::step(&mut e, SantaFeAntAction::TurnRight)
                .expect("step");
        }
        assert!(snap.is_truncated());
        assert!(snap.is_done());
        assert_eq!(e.state().steps(), 5);
    }

    #[test]
    fn episode_terminates_when_last_pellet_eaten() {
        let mut e = env();
        // White-box (same-module access): clear the trail to a single pellet
        // directly ahead of the ant — east of (0,0) is (0,1) — so the next
        // Move drives `pellets_remaining` to zero.
        e.state.food = [[false; GRID_SIZE]; GRID_SIZE];
        e.state.food[0][1] = true;
        e.state.pellets_remaining = 1;
        let snap = <SantaFeAnt as Environment<1, 1, 1>>::step(&mut e, SantaFeAntAction::Move)
            .expect("step");
        assert_eq!(f32::from(*snap.reward()), 1.0);
        assert_eq!(e.state().pellets_remaining(), 0);
        assert!(snap.is_terminated());
        assert!(snap.is_done());
    }

    // -- POMDP marker ------------------------------------------------------

    #[test]
    fn state_is_non_markov() {
        assert!(!<SantaFeAntState as MarkovState>::is_markov());
    }

    // -- TensorConvertible round-trips ------------------------------------

    #[test]
    fn observation_round_trips_through_tensor() {
        let device = Default::default();
        for food_ahead in [false, true] {
            let obs = SantaFeAntObservation { food_ahead };
            let tensor = <SantaFeAntObservation as TensorConvertible<1, TestBackend>>::to_tensor(
                &obs, &device,
            );
            let back =
                <SantaFeAntObservation as TensorConvertible<1, TestBackend>>::from_tensor(tensor)
                    .expect("round-trip");
            assert_eq!(back, obs);
        }
    }

    #[test]
    fn observation_from_tensor_rejects_wrong_shape() {
        use burn::tensor::{Tensor, TensorData as TD};
        let device = Default::default();
        let data = TD::new(vec![0.0_f32, 0.0_f32], [2]);
        let tensor = Tensor::<TestBackend, 1>::from_data(data, &device);
        let err = <SantaFeAntObservation as TensorConvertible<1, TestBackend>>::from_tensor(tensor)
            .expect_err("shape [2] rejected");
        assert!(err.message.contains("expected shape [1]"));
    }

    #[test]
    fn action_round_trips_through_tensor() {
        let device = Default::default();
        for action in [
            SantaFeAntAction::Move,
            SantaFeAntAction::TurnLeft,
            SantaFeAntAction::TurnRight,
        ] {
            let tensor = <SantaFeAntAction as TensorConvertible<1, TestBackend>>::to_tensor(
                &action, &device,
            );
            let back = <SantaFeAntAction as TensorConvertible<1, TestBackend>>::from_tensor(tensor)
                .expect("round-trip");
            assert_eq!(back, action);
        }
    }

    #[test]
    fn action_discrete_index_round_trips() {
        for index in 0..ACTION_COUNT {
            let action = SantaFeAntAction::from_index(index);
            assert_eq!(action.to_index(), index);
        }
        assert_eq!(SantaFeAntAction::ACTION_COUNT, 3);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn action_from_index_panics_out_of_range() {
        let _ = SantaFeAntAction::from_index(ACTION_COUNT);
    }

    // -- Determinism -------------------------------------------------------

    #[test]
    fn reset_reproduces_identical_start_state() {
        let mut e = env();
        let s0: SantaFeAntState = e.state().clone();
        // Perturb, then reset.
        let _ = <SantaFeAnt as Environment<1, 1, 1>>::step(&mut e, SantaFeAntAction::Move);
        let _ = <SantaFeAnt as Environment<1, 1, 1>>::step(&mut e, SantaFeAntAction::TurnRight);
        let _ = <SantaFeAnt as Environment<1, 1, 1>>::reset(&mut e).expect("reset");
        assert_eq!(e.state(), &s0);
    }

    #[test]
    fn same_action_sequence_is_deterministic() {
        let actions = [
            SantaFeAntAction::Move,
            SantaFeAntAction::TurnRight,
            SantaFeAntAction::Move,
            SantaFeAntAction::TurnLeft,
        ];
        let run = || {
            let mut e = env();
            let _ = <SantaFeAnt as Environment<1, 1, 1>>::reset(&mut e).expect("reset");
            for &a in &actions {
                let _ = <SantaFeAnt as Environment<1, 1, 1>>::step(&mut e, a).expect("step");
            }
            e.state().clone()
        };
        assert_eq!(run(), run());
    }
}
