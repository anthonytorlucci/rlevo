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
use rlevo_core::config::{self, ConfigError, Validate};
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
    /// Row shape of the one-hot action encoding: `[ACTION_COUNT]`.
    fn row_shape() -> [usize; 1] {
        [ACTION_COUNT]
    }

    /// One-hot encoding of the action, length [`ACTION_COUNT`].
    fn write_host_row(&self, buf: &mut Vec<f32>) {
        let mut one_hot: [f32; ACTION_COUNT] = [0.0; ACTION_COUNT];
        one_hot[self.to_index()] = 1.0;
        buf.extend_from_slice(&one_hot);
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
    /// Row shape of the food-ahead bit: `[1]`.
    fn row_shape() -> [usize; 1] {
        [1]
    }

    /// Encode the food-ahead bit as a single element (`1.0` / `0.0`).
    fn write_host_row(&self, buf: &mut Vec<f32>) {
        let value: f32 = if self.food_ahead { 1.0 } else { 0.0 };
        buf.push(value);
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
        self.row < GRID_SIZE && self.col < GRID_SIZE && self.pellets_remaining == self.count_food()
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
    /// When `true`, each [`reset`](Environment::reset) /
    /// [`step`](Environment::step) renders the grid to ASCII as a debug side
    /// effect (the text is discarded). The structured report path
    /// ([`GridPayloadSource`](rlevo_core::render::payload::GridPayloadSource)) and
    /// the [`AsciiRenderable`](crate::render::AsciiRenderable) impl are available
    /// regardless of this flag.
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

impl Validate for SantaFeAntConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "SantaFeAntConfig";
        config::nonzero(C, "max_steps", self.max_steps)?;
        Ok(())
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
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `config` fails [`Validate`]
    /// (`max_steps == 0`).
    pub fn with_config(config: SantaFeAntConfig) -> Result<Self, ConfigError> {
        config.validate()?;
        Ok(Self {
            state: Self::fresh_state(),
            config,
        })
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

    /// Render the grid to ASCII as a debug side effect when [`render`] is set.
    /// The string is discarded; callers wanting the text call [`render_ascii`]
    /// directly. Mirrors the grids family's debug-render convention.
    ///
    /// [`render`]: SantaFeAntConfig::render
    /// [`render_ascii`]: crate::render::AsciiRenderable::render_ascii
    fn maybe_render(&self) {
        if self.config.render {
            let _ = crate::render::AsciiRenderable::render_ascii(self);
        }
    }
}

impl ConstructableEnv for SantaFeAnt {
    fn new(render: bool) -> Self {
        Self::with_config(SantaFeAntConfig {
            render,
            ..SantaFeAntConfig::default()
        })
        .expect("default config must validate")
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
        self.maybe_render();
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
        self.maybe_render();

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
                    assert!(
                        start.is_none(),
                        "trail must contain exactly one start marker"
                    );
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

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Heading glyph encoded by **shape**, not colour (accessibility convention):
/// `>` East, `v` South, `<` West, `^` North. Mirrors the grids family's
/// `agent_char`; could be hoisted onto [`Direction`] if a second consumer appears.
const fn heading_glyph(dir: Direction) -> char {
    match dir {
        Direction::East => '>',
        Direction::South => 'v',
        Direction::West => '<',
        Direction::North => '^',
    }
}

/// Map the env-side [`Direction`] onto the wire-neutral grid facing.
const fn heading_to_grid_dir(dir: Direction) -> rlevo_core::render::payload::GridDir {
    use rlevo_core::render::payload::GridDir;
    match dir {
        Direction::East => GridDir::East,
        Direction::South => GridDir::South,
        Direction::West => GridDir::West,
        Direction::North => GridDir::North,
    }
}

/// Structured grid projection for the post-run report (ADR 0013's primary,
/// load-bearing render path). Built directly from the private `food` grid rather
/// than via the shared `grids::core` helper, because the ant does not use the
/// `Grid`/`AgentState` types.
///
/// Tile mapping, distinguishing an **eaten** cell from one that never held food by
/// reconstructing the original trail from the `SANTA_FE_TRAIL` literal:
/// - current food present → [`GridTile::Ball`]`(Green)` — a pellet,
/// - eaten (originally food, now gone) → [`GridTile::Floor`] — carries the
///   distinction losslessly (the report draws `Floor` like `Empty` today),
/// - never food → [`GridTile::Empty`].
///
/// [`GridTile::Ball`]: rlevo_core::render::payload::GridTile::Ball
/// [`GridTile::Floor`]: rlevo_core::render::payload::GridTile::Floor
/// [`GridTile::Empty`]: rlevo_core::render::payload::GridTile::Empty
impl rlevo_core::render::payload::GridPayloadSource for SantaFeAnt {
    #[allow(clippy::cast_possible_truncation)] // GRID_SIZE = 32 and all indices < 32 fit u16
    fn grid_snapshot(&self) -> rlevo_core::render::payload::GridSnapshot {
        use rlevo_core::render::payload::{GridAgentMarker, GridColor, GridSnapshot, GridTile};

        let (original, _): ([[bool; GRID_SIZE]; GRID_SIZE], _) = parse_trail(SANTA_FE_TRAIL);
        let mut tiles: Vec<GridTile> = Vec::with_capacity(GRID_SIZE * GRID_SIZE);
        for (food_row, orig_row) in self.state.food.iter().zip(original.iter()) {
            for (live, orig) in food_row.iter().zip(orig_row.iter()) {
                let tile = if *live {
                    GridTile::Ball(GridColor::Green)
                } else if *orig {
                    GridTile::Floor
                } else {
                    GridTile::Empty
                };
                tiles.push(tile);
            }
        }

        let (row, col): (usize, usize) = self.state.position();
        GridSnapshot {
            width: GRID_SIZE as u16,
            height: GRID_SIZE as u16,
            tiles,
            agent: GridAgentMarker {
                x: col as u16,
                y: row as u16,
                dir: heading_to_grid_dir(self.state.heading),
                carrying: None,
            },
        }
    }
}

/// Optional library-tier debug renderer (ADR 0013 demotes this to an opt-in
/// helper; neither product depends on it). Glyphs are shape-redundant: ant by
/// heading (`> v < ^`), `#` live pellet, `·` eaten cell, `.` never-food. The styled
/// projection reaches only for `palette` constants so the hue-redundant
/// accessibility contract holds.
impl crate::render::AsciiRenderable for SantaFeAnt {
    fn render_ascii(&self) -> String {
        let (original, _): ([[bool; GRID_SIZE]; GRID_SIZE], _) = parse_trail(SANTA_FE_TRAIL);
        let (arow, acol): (usize, usize) = self.state.position();
        let mut out = String::with_capacity(GRID_SIZE * GRID_SIZE * 2);
        for (row, (food_row, orig_row)) in self.state.food.iter().zip(original.iter()).enumerate() {
            for (col, (live, orig)) in food_row.iter().zip(orig_row.iter()).enumerate() {
                let ch = if row == arow && col == acol {
                    heading_glyph(self.state.heading)
                } else if *live {
                    '#'
                } else if *orig {
                    '·'
                } else {
                    '.'
                };
                out.push(ch);
                out.push(' ');
            }
            out.push('\n');
        }
        out
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        use crate::render::palette::{AGENT_FG, AGENT_MODIFIER, GOAL_FG, GOAL_MODIFIER};
        use crate::render::{SpanStyle, StyledFrame, StyledLine, StyledSpan};

        let (original, _): ([[bool; GRID_SIZE]; GRID_SIZE], _) = parse_trail(SANTA_FE_TRAIL);
        let (arow, acol): (usize, usize) = self.state.position();
        let mut lines: Vec<StyledLine> = Vec::with_capacity(GRID_SIZE);
        let mut spans: Vec<StyledSpan> = Vec::new();
        let mut current_style = SpanStyle::default();
        let mut current_text = String::new();
        for (row, (food_row, orig_row)) in self.state.food.iter().zip(original.iter()).enumerate() {
            for (col, (live, orig)) in food_row.iter().zip(orig_row.iter()).enumerate() {
                let (ch, style) = if row == arow && col == acol {
                    (
                        heading_glyph(self.state.heading),
                        SpanStyle::default()
                            .fg(AGENT_FG)
                            .with_modifier(AGENT_MODIFIER),
                    )
                } else if *live {
                    (
                        '#',
                        SpanStyle::default()
                            .fg(GOAL_FG)
                            .with_modifier(GOAL_MODIFIER),
                    )
                } else if *orig {
                    ('·', SpanStyle::default().dim())
                } else {
                    ('.', SpanStyle::default())
                };
                if style != current_style && !current_text.is_empty() {
                    spans.push(StyledSpan::new(
                        std::mem::take(&mut current_text),
                        current_style,
                    ));
                }
                current_style = style;
                current_text.push(ch);
                current_text.push(' ');
            }
            if !current_text.is_empty() {
                spans.push(StyledSpan::new(
                    std::mem::take(&mut current_text),
                    current_style,
                ));
            }
            lines.push(StyledLine::from_spans(std::mem::take(&mut spans)));
            current_style = SpanStyle::default();
        }
        StyledFrame { lines }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlevo_core::environment::Snapshot;

    #[test]
    fn default_config_validates() {
        assert!(SantaFeAntConfig::default().validate().is_ok());
    }

    #[test]
    fn rejects_zero_max_steps() {
        let bad = SantaFeAntConfig {
            max_steps: 0,
            render: false,
        };
        assert!(SantaFeAnt::with_config(bad).is_err());
    }

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
        assert!(
            <SantaFeAnt as Environment<1, 1, 1>>::reset(&mut e)
                .expect("reset")
                .observation()
                .food_ahead
        ); // east, (0,1) is food

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
        })
        .expect("valid config");
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

    // -- Rendering ---------------------------------------------------------
    //
    // The start state has S at (row=0, col=0) facing East; row 0 of the trail
    // is `S###...`, so cells (0,1), (0,2), (0,3) hold food. A single `Move`
    // eats (0,1) and lands the ant there; a second `Move` eats (0,2) and leaves
    // (0,1) as an *eaten* cell with the ant gone.

    use crate::render::AsciiRenderable;
    use rlevo_core::render::payload::{
        GridAgentMarker, GridColor, GridDir, GridPayloadSource, GridTile,
    };

    fn ball_count(snap: &rlevo_core::render::payload::GridSnapshot) -> usize {
        snap.tiles
            .iter()
            .filter(|t| matches!(t, GridTile::Ball(GridColor::Green)))
            .count()
    }

    fn floor_count(snap: &rlevo_core::render::payload::GridSnapshot) -> usize {
        snap.tiles
            .iter()
            .filter(|t| matches!(t, GridTile::Floor))
            .count()
    }

    #[test]
    fn grid_snapshot_is_full_trail_on_reset() {
        let mut e = env();
        e.reset().expect("reset");
        let snap = e.grid_snapshot();

        assert_eq!(snap.width as usize, GRID_SIZE);
        assert_eq!(snap.height as usize, GRID_SIZE);
        assert_eq!(snap.tiles.len(), GRID_SIZE * GRID_SIZE);
        // Every one of the 89 pellets is a green ball; nothing eaten yet.
        assert_eq!(ball_count(&snap), TOTAL_PELLETS as usize);
        assert_eq!(floor_count(&snap), 0);
        // Ant at the origin facing East, carrying nothing.
        assert_eq!(
            snap.agent,
            GridAgentMarker {
                x: 0,
                y: 0,
                dir: GridDir::East,
                carrying: None
            }
        );
    }

    #[test]
    fn grid_snapshot_marks_eaten_cell_as_floor() {
        let mut e = env();
        e.reset().expect("reset");
        e.step(SantaFeAntAction::Move).expect("step"); // eat (0,1), ant -> (0,1)
        e.step(SantaFeAntAction::Move).expect("step"); // eat (0,2), ant -> (0,2)

        let snap = e.grid_snapshot();
        // Two pellets eaten: 87 balls remain.
        assert_eq!(ball_count(&snap), TOTAL_PELLETS as usize - 2);
        // (0,1) is now an eaten Floor (the ant has moved off it); (0,2) carries
        // the agent marker so its tile is also Floor → two Floor cells.
        assert_eq!(floor_count(&snap), 2);
        assert_eq!(
            snap.agent,
            GridAgentMarker {
                x: 2,
                y: 0,
                dir: GridDir::East,
                carrying: None
            }
        );
    }

    #[test]
    fn grid_snapshot_agent_marker_tracks_heading() {
        let mut e = env();
        e.reset().expect("reset");
        e.step(SantaFeAntAction::TurnRight).expect("step"); // East -> South
        assert_eq!(e.grid_snapshot().agent.dir, GridDir::South);
    }

    #[test]
    fn render_ascii_has_grid_dimensions() {
        let mut e = env();
        e.reset().expect("reset");
        let ascii = e.render_ascii();
        let lines: Vec<&str> = ascii.lines().collect();
        assert_eq!(lines.len(), GRID_SIZE);
        for line in &lines {
            // glyph + space per cell, within the 80-col render budget.
            assert_eq!(line.chars().count(), GRID_SIZE * 2);
            assert!(line.chars().count() <= 80);
        }
        // Ant faces East at the origin: first glyph is the heading shape.
        assert_eq!(lines[0].chars().next(), Some('>'));
    }

    #[test]
    fn render_ascii_distinguishes_eaten_from_never_food() {
        let mut e = env();
        e.reset().expect("reset");
        e.step(SantaFeAntAction::Move).expect("step"); // ant -> (0,1)
        e.step(SantaFeAntAction::Move).expect("step"); // ant -> (0,2)
        let row0: Vec<char> = e
            .render_ascii()
            .lines()
            .next()
            .expect("row 0")
            .chars()
            .collect();
        // Cells are (glyph, space) pairs, so column c's glyph is at index 2*c.
        assert_eq!(row0[0], '.', "(0,0): start cell, never had food");
        assert_eq!(row0[2], '·', "(0,1): eaten cell");
        assert_eq!(row0[4], '>', "(0,2): ant, facing East");
        assert_eq!(row0[6], '#', "(0,3): live pellet");
    }

    #[test]
    fn render_styled_matches_ascii() {
        let mut e = env();
        e.reset().expect("reset");
        e.step(SantaFeAntAction::Move).expect("step");
        let plain = e.render_ascii();
        let plain_no_trailing: String = plain.lines().collect::<Vec<_>>().join("\n");
        assert_eq!(e.render_styled().plain_text(), plain_no_trailing);
    }

    #[test]
    fn render_styled_uses_palette_consts() {
        use crate::render::palette::{AGENT_FG, GOAL_FG};
        let mut e = env();
        e.reset().expect("reset");
        let frame = e.render_styled();
        let styles: Vec<_> = frame
            .lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.style))
            .collect();
        assert!(
            styles.iter().any(|s| s.fg == Some(AGENT_FG)),
            "ant span must use the AGENT_FG palette colour"
        );
        assert!(
            styles.iter().any(|s| s.fg == Some(GOAL_FG)),
            "pellet spans must use the GOAL_FG palette colour"
        );
    }
}
