//! FrozenLake-v1 environment.
//!
//! A grid-world where the agent walks across a frozen lake to reach a goal without falling
//! into holes. Supports 4×4 and 8×8 preset maps as well as custom or procedurally generated
//! layouts. Random maps are validated by BFS to ensure the goal is always reachable.
//!
//! ## Tile types
//!
//! | Char | Name   | Effect                                      |
//! |------|--------|---------------------------------------------|
//! | `S`  | Start  | Initial agent position (treated as frozen)  |
//! | `F`  | Frozen | Safe tile; episode continues                |
//! | `H`  | Hole   | Episode terminates with `reward_schedule.hole` |
//! | `G`  | Goal   | Episode terminates with `reward_schedule.goal` |
//!
//! ## Observation space
//!
//! Integer state id `row × ncol + col` in `[0, nrow × ncol)`.
//!
//! ## Action space
//!
//! Four discrete directions via [`FrozenLakeAction`]: `Left` (0), `Down` (1), `Right` (2), `Up` (3).
//!
//! ## Slippery mode
//!
//! When enabled, the intended direction succeeds with probability `success_rate`
//! (default 1/3); each perpendicular direction occurs with probability
//! `(1 − success_rate) / 2`.

use std::collections::VecDeque;

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rlevo_core::action::DiscreteAction;
use rlevo_core::base::{Action, Observation, State};
use rlevo_core::config::{self, ConfigError, Validate};
use rlevo_core::environment::{ConstructableEnv, Environment, EnvironmentError, SnapshotBase};
use rlevo_core::reward::ScalarReward;
use rlevo_core::state::StateError;
use serde::{Deserialize, Serialize};

use crate::episode::EpisodeGuard;
use crate::toy_text::MapError;

// ── tile ──────────────────────────────────────────────────────────────────────

/// A single cell on the frozen-lake grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tile {
    /// Starting cell — treated as frozen during simulation.
    Start,
    /// Safe frozen surface — episode continues with the configured step reward.
    Frozen,
    /// Open hole — episode terminates on entry.
    Hole,
    /// Goal cell — episode terminates on entry with the goal reward.
    Goal,
}

impl TryFrom<char> for Tile {
    type Error = ();
    fn try_from(c: char) -> Result<Self, ()> {
        match c {
            'S' => Ok(Tile::Start),
            'F' => Ok(Tile::Frozen),
            'H' => Ok(Tile::Hole),
            'G' => Ok(Tile::Goal),
            _ => Err(()),
        }
    }
}

// ── preset maps ───────────────────────────────────────────────────────────────

/// Built-in preset map sizes for [`FrozenMapSpec::Preset`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrozenPreset {
    /// The classic 4×4 map: `"SFFF" / "FHFH" / "FFFH" / "HFFG"`.
    Four4x4,
    /// The classic 8×8 map with a longer path and more holes.
    Eight8x8,
}

const MAP_4X4: &[&str] = &["SFFF", "FHFH", "FFFH", "HFFG"];
const MAP_8X8: &[&str] = &[
    "SFFFFFFF", "FFFFFFFF", "FFFHFFFF", "FFFFFHFF", "FFFHFFFF", "FHHFFFHF", "FHFFHFHF", "FFFHFFFG",
];

// ── map spec (C2) ─────────────────────────────────────────────────────────────

/// Source specification for the frozen-lake grid.
///
/// Controls whether the environment uses a built-in preset, a user-supplied layout, or a
/// procedurally generated map. Random maps are regenerated on every call to `reset()`.
#[derive(Debug, Clone)]
pub enum FrozenMapSpec {
    /// One of the two canonical preset maps.
    Preset(FrozenPreset),
    /// A user-defined grid supplied as a `Vec` of row strings using `S`, `F`, `H`, `G`.
    Custom(Vec<String>),
    /// Procedurally generated map with BFS-guaranteed goal reachability.
    Random {
        /// Number of rows.
        nrow: usize,
        /// Number of columns.
        ncol: usize,
        /// Probability that an interior cell is frozen (not a hole). Range: `[0.0, 1.0]`.
        frozen_prob: f32,
    },
}

impl Default for FrozenMapSpec {
    fn default() -> Self {
        FrozenMapSpec::Random {
            nrow: 8,
            ncol: 8,
            frozen_prob: 0.8,
        }
    }
}

// ── reward schedule ───────────────────────────────────────────────────────────

/// Per-tile reward values for [`FrozenLake`].
///
/// The default schedule matches Gymnasium: `+1.0` on goal, `0.0` on hole, `0.0` on frozen.
#[derive(Debug, Clone)]
pub struct RewardSchedule {
    /// Reward issued when the agent reaches the goal tile.
    pub goal: f32,
    /// Reward issued when the agent falls into a hole.
    pub hole: f32,
    /// Reward issued for each step on a frozen tile.
    pub frozen: f32,
}

impl Default for RewardSchedule {
    fn default() -> Self {
        Self {
            goal: 1.0,
            hole: 0.0,
            frozen: 0.0,
        }
    }
}

// ── config ────────────────────────────────────────────────────────────────────

/// Configuration for the [`FrozenLake`] environment.
///
/// Build with [`FrozenLakeConfig::builder`] for full control, or use
/// [`FrozenLakeConfig::default`] for the standard random 8×8 slippery variant.
///
/// # Examples
///
/// ```rust
/// use rlevo_environments::toy_text::frozen_lake::{FrozenLakeConfig, FrozenMapSpec, FrozenPreset};
///
/// let cfg = FrozenLakeConfig::builder()
///     .map(FrozenMapSpec::Preset(FrozenPreset::Four4x4))
///     .is_slippery(false)
///     .seed(0)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct FrozenLakeConfig {
    /// Grid source: preset, custom, or randomly generated.
    pub map: FrozenMapSpec,
    /// When `true`, slip transitions are applied using `success_rate`.
    pub is_slippery: bool,
    /// Probability of moving in the intended direction when `is_slippery` is `true`. Default: `1/3`.
    pub success_rate: f32,
    /// Tile-specific reward values.
    pub reward_schedule: RewardSchedule,
    /// Seed used to initialise the RNG when the environment is created. Default: `0`.
    pub seed: u64,
}

impl Default for FrozenLakeConfig {
    fn default() -> Self {
        Self {
            map: FrozenMapSpec::default(),
            is_slippery: true,
            success_rate: 1.0 / 3.0,
            reward_schedule: RewardSchedule::default(),
            seed: 0,
        }
    }
}

impl FrozenLakeConfig {
    /// Returns a builder for constructing a `FrozenLakeConfig`.
    pub fn builder() -> FrozenLakeConfigBuilder {
        FrozenLakeConfigBuilder::default()
    }
}

impl Validate for FrozenLakeConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "FrozenLakeConfig";
        config::in_range(C, "success_rate", 0.0, 1.0, f64::from(self.success_rate))?;
        Ok(())
    }
}

/// Builder for [`FrozenLakeConfig`].
#[derive(Default)]
pub struct FrozenLakeConfigBuilder {
    map: Option<FrozenMapSpec>,
    is_slippery: bool,
    success_rate: Option<f32>,
    reward_schedule: Option<RewardSchedule>,
    seed: u64,
}

impl FrozenLakeConfigBuilder {
    /// Sets the grid source: preset, custom, or randomly generated.
    pub fn map(mut self, m: FrozenMapSpec) -> Self {
        self.map = Some(m);
        self
    }

    /// Enables or disables stochastic slip transitions.
    pub fn is_slippery(mut self, v: bool) -> Self {
        self.is_slippery = v;
        self
    }

    /// Sets the probability of moving in the intended direction when slippery mode is active.
    ///
    /// The two perpendicular directions each receive probability `(1 − rate) / 2`. Default: `1/3`.
    pub fn success_rate(mut self, r: f32) -> Self {
        self.success_rate = Some(r);
        self
    }

    /// Overrides the per-tile reward values.
    pub fn reward_schedule(mut self, rs: RewardSchedule) -> Self {
        self.reward_schedule = Some(rs);
        self
    }

    /// Sets the RNG seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Builds the [`FrozenLakeConfig`].
    pub fn build(self) -> FrozenLakeConfig {
        FrozenLakeConfig {
            map: self.map.unwrap_or_default(),
            is_slippery: self.is_slippery,
            success_rate: self.success_rate.unwrap_or(1.0 / 3.0),
            reward_schedule: self.reward_schedule.unwrap_or_default(),
            seed: self.seed,
        }
    }
}

// ── resolved map ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct ResolvedMap {
    tiles: Vec<Tile>,
    nrow: usize,
    ncol: usize,
    start_pos: usize,
}

fn parse_map(rows: &[&str]) -> Result<ResolvedMap, MapError> {
    let nrow = rows.len();
    if nrow == 0 {
        return Err(MapError::WrongStartCount(0));
    }
    let ncol = rows[0].len();
    let mut tiles = Vec::with_capacity(nrow * ncol);
    let mut starts = 0usize;
    let mut goals = 0usize;

    for (ri, row) in rows.iter().enumerate() {
        if row.len() != ncol {
            return Err(MapError::RowLengthMismatch {
                row: ri,
                got: row.len(),
                expected: ncol,
            });
        }
        for (ci, ch) in row.chars().enumerate() {
            match Tile::try_from(ch) {
                Ok(t) => {
                    if t == Tile::Start {
                        starts += 1;
                    }
                    if t == Tile::Goal {
                        goals += 1;
                    }
                    tiles.push(t);
                }
                Err(()) => {
                    return Err(MapError::InvalidTile {
                        row: ri,
                        col: ci,
                        ch,
                    });
                }
            }
        }
    }

    if starts != 1 {
        return Err(MapError::WrongStartCount(starts));
    }
    if goals == 0 {
        return Err(MapError::NoGoal(goals));
    }

    let start_pos = tiles.iter().position(|&t| t == Tile::Start).unwrap();
    bfs_reachable(&tiles, nrow, ncol, start_pos)?;

    Ok(ResolvedMap {
        tiles,
        nrow,
        ncol,
        start_pos,
    })
}

fn bfs_reachable(tiles: &[Tile], nrow: usize, ncol: usize, start: usize) -> Result<(), MapError> {
    let mut visited = vec![false; tiles.len()];
    let mut queue = VecDeque::new();
    queue.push_back(start);
    visited[start] = true;
    let mut goal_found = false;

    while let Some(idx) = queue.pop_front() {
        if tiles[idx] == Tile::Goal {
            goal_found = true;
        }
        if tiles[idx] == Tile::Hole {
            continue; // holes are passable in BFS but blocked in simulation
        }
        let row = idx / ncol;
        let col = idx % ncol;
        for (dr, dc) in [(!0usize, 0usize), (1, 0), (0, !0), (0, 1)] {
            let nr = row.wrapping_add(dr);
            let nc = col.wrapping_add(dc);
            if nr < nrow && nc < ncol {
                let ni = nr * ncol + nc;
                if !visited[ni] && tiles[ni] != Tile::Hole {
                    visited[ni] = true;
                    queue.push_back(ni);
                }
            }
        }
    }

    if goal_found {
        Ok(())
    } else {
        Err(MapError::GoalUnreachable)
    }
}

fn generate_random_map(
    nrow: usize,
    ncol: usize,
    frozen_prob: f32,
    rng: &mut StdRng,
) -> Result<ResolvedMap, MapError> {
    const MAX_RETRIES: usize = 1000;
    for _ in 0..MAX_RETRIES {
        let mut tiles = vec![Tile::Frozen; nrow * ncol];
        tiles[0] = Tile::Start;
        tiles[nrow * ncol - 1] = Tile::Goal;
        for tile in tiles[1..nrow * ncol - 1].iter_mut() {
            if rng.random_range(0.0f32..1.0) >= frozen_prob {
                *tile = Tile::Hole;
            }
        }
        let start = 0;
        if bfs_reachable(&tiles, nrow, ncol, start).is_ok() {
            return Ok(ResolvedMap {
                tiles,
                nrow,
                ncol,
                start_pos: start,
            });
        }
    }
    Err(MapError::MaxRetriesExceeded)
}

// ── state / observation / action ──────────────────────────────────────────────

/// Full state: agent position plus the map dimensions needed for id encoding.
#[derive(Debug, Clone)]
pub struct FrozenLakeState {
    /// Current row index in `[0, nrow)`.
    pub row: u8,
    /// Current column index in `[0, ncol)`.
    pub col: u8,
    /// Total number of rows in the active map.
    pub nrow: u8,
    /// Total number of columns in the active map.
    pub ncol: u8,
}

impl FrozenLakeState {
    fn state_id(&self) -> u16 {
        self.row as u16 * self.ncol as u16 + self.col as u16
    }
}

impl TryFrom<(u16, u8, u8)> for FrozenLakeState {
    type Error = StateError;
    fn try_from((id, nrow, ncol): (u16, u8, u8)) -> Result<Self, Self::Error> {
        let n = nrow as u16 * ncol as u16;
        if id >= n {
            return Err(StateError::InvalidData(format!(
                "FrozenLakeState id {id} out of [0,{n})"
            )));
        }
        Ok(FrozenLakeState {
            row: (id / ncol as u16) as u8,
            col: (id % ncol as u16) as u8,
            nrow,
            ncol,
        })
    }
}

impl State<1> for FrozenLakeState {
    type Observation = FrozenLakeObservation;

    fn shape() -> [usize; 1] {
        // Shape is dynamic (depends on map size); return a conservative max.
        [64] // 8×8 max
    }

    fn observe(&self) -> FrozenLakeObservation {
        FrozenLakeObservation {
            state_id: self.state_id(),
        }
    }

    fn is_valid(&self) -> bool {
        self.row < self.nrow && self.col < self.ncol
    }
}

/// Agent-visible observation: integer state id `row × ncol + col`.
///
/// The shape constant is fixed at 64 (8×8 maximum), even for smaller maps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrozenLakeObservation {
    /// Linear index: `row × ncol + col`.
    pub state_id: u16,
}

impl Observation<1> for FrozenLakeObservation {
    fn shape() -> [usize; 1] {
        [64]
    }
}

/// Four-direction action space matching Gymnasium's FrozenLake ordering.
///
/// Movements are clamped at grid boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrozenLakeAction {
    /// Move one column toward col 0 (west).
    Left = 0,
    /// Move one row toward the last row (south).
    Down = 1,
    /// Move one column toward the last column (east).
    Right = 2,
    /// Move one row toward row 0 (north).
    Up = 3,
}

impl Action<1> for FrozenLakeAction {
    fn shape() -> [usize; 1] {
        [1]
    }
    fn is_valid(&self) -> bool {
        true
    }
}

impl DiscreteAction<1> for FrozenLakeAction {
    const ACTION_COUNT: usize = 4;

    fn from_index(index: usize) -> Self {
        match index {
            0 => FrozenLakeAction::Left,
            1 => FrozenLakeAction::Down,
            2 => FrozenLakeAction::Right,
            3 => FrozenLakeAction::Up,
            _ => panic!("FrozenLakeAction index {index} out of range [0, 4)"),
        }
    }

    fn to_index(&self) -> usize {
        *self as usize
    }
}

impl FrozenLakeAction {
    fn perpendiculars(self) -> [FrozenLakeAction; 2] {
        match self {
            FrozenLakeAction::Left | FrozenLakeAction::Right => {
                [FrozenLakeAction::Down, FrozenLakeAction::Up]
            }
            FrozenLakeAction::Down | FrozenLakeAction::Up => {
                [FrozenLakeAction::Left, FrozenLakeAction::Right]
            }
        }
    }
}

fn apply_action(row: u8, col: u8, action: FrozenLakeAction, nrow: u8, ncol: u8) -> (u8, u8) {
    let (nr, nc) = match action {
        FrozenLakeAction::Left => (row, col.saturating_sub(1)),
        FrozenLakeAction::Right => (row, (col + 1).min(ncol - 1)),
        FrozenLakeAction::Down => ((row + 1).min(nrow - 1), col),
        FrozenLakeAction::Up => (row.saturating_sub(1), col),
    };
    (nr, nc)
}

// ── environment ───────────────────────────────────────────────────────────────

/// FrozenLake-v1 environment.
///
/// Construction is infallible via `new()` (uses default random 8×8 map).
/// For custom maps, use `with_config(config)` which may return a [`MapError`].
///
/// # Episode lifecycle
///
/// Entering a `Hole` or `Goal` tile terminates the episode. The agent remains on
/// that tile, so an unguarded `step()` would walk it back onto a frozen
/// neighbour and report `Running` — resurrecting a finished episode. An
/// [`EpisodeGuard`] therefore rejects any `step()` taken after a done snapshot
/// with [`EnvironmentError::StepAfterEpisodeEnd`]; call
/// [`reset`](Environment::reset) to begin a new episode.
#[derive(Debug)]
pub struct FrozenLake {
    state: FrozenLakeState,
    map: ResolvedMap,
    config: FrozenLakeConfig,
    rng: StdRng,
    guard: EpisodeGuard,
}

impl FrozenLake {
    /// Creates a [`FrozenLake`] environment with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns [`MapError`] if the map cannot be constructed:
    ///
    /// - `Custom` maps: [`MapError::RowLengthMismatch`], [`MapError::WrongStartCount`],
    ///   [`MapError::NoGoal`], [`MapError::GoalUnreachable`], or [`MapError::InvalidTile`].
    /// - `Random` maps: [`MapError::MaxRetriesExceeded`] if 1000 attempts all produce
    ///   unreachable goals (unlikely at the default `frozen_prob = 0.8`).
    /// - Any map: [`MapError::InvalidConfig`] if `config` fails [`Validate`]
    ///   (e.g. `success_rate` outside `[0, 1]`).
    pub fn with_config(config: FrozenLakeConfig) -> Result<Self, MapError> {
        config.validate()?;
        let mut rng = StdRng::seed_from_u64(config.seed);
        let map = Self::resolve_map(&config.map, &mut rng)?;
        let state = FrozenLakeState {
            row: (map.start_pos / map.ncol) as u8,
            col: (map.start_pos % map.ncol) as u8,
            nrow: map.nrow as u8,
            ncol: map.ncol as u8,
        };
        Ok(Self {
            state,
            map,
            config,
            rng,
            guard: EpisodeGuard::new(),
        })
    }

    fn resolve_map(spec: &FrozenMapSpec, rng: &mut StdRng) -> Result<ResolvedMap, MapError> {
        match spec {
            FrozenMapSpec::Preset(FrozenPreset::Four4x4) => parse_map(MAP_4X4),
            FrozenMapSpec::Preset(FrozenPreset::Eight8x8) => parse_map(MAP_8X8),
            FrozenMapSpec::Custom(rows) => {
                let refs: Vec<&str> = rows.iter().map(|s| s.as_str()).collect();
                parse_map(&refs)
            }
            FrozenMapSpec::Random {
                nrow,
                ncol,
                frozen_prob,
            } => generate_random_map(*nrow, *ncol, *frozen_prob, rng),
        }
    }

    fn tile_at(&self, row: u8, col: u8) -> Tile {
        self.map.tiles[row as usize * self.map.ncol + col as usize]
    }

    fn resolve_action(&mut self, action: FrozenLakeAction) -> FrozenLakeAction {
        if !self.config.is_slippery {
            return action;
        }
        let r = self.rng.random_range(0.0f32..1.0);
        let sr = self.config.success_rate;
        let perp_each = (1.0 - sr) / 2.0;
        if r < sr {
            action
        } else if r < sr + perp_each {
            action.perpendiculars()[0]
        } else {
            action.perpendiculars()[1]
        }
    }
}

impl ConstructableEnv for FrozenLake {
    fn new(_render: bool) -> Self {
        Self::with_config(FrozenLakeConfig::default()).expect("default random map must succeed")
    }
}

impl Environment<1, 1, 1> for FrozenLake {
    type StateType = FrozenLakeState;
    type ObservationType = FrozenLakeObservation;
    type ActionType = FrozenLakeAction;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, FrozenLakeObservation, ScalarReward>;

    /// Starts a new episode: rebuilds the map (random specs only), places the
    /// agent on the start tile, and re-opens the [`EpisodeGuard`].
    ///
    /// # Errors
    ///
    /// Returns [`EnvironmentError::RenderFailed`] wrapping a [`MapError`] when a
    /// [`FrozenMapSpec::Random`] map cannot be regenerated. The guard is *not*
    /// re-opened in that case: the environment still holds the previous episode's
    /// map and agent position, so re-opening it would let a `step()` walk the
    /// agent off the terminal tile it is standing on — the very defect the guard
    /// exists to prevent. A failed reset leaves the episode closed; the caller
    /// must reset successfully before stepping again.
    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        // Regenerate map for random spec; reuse preset/custom maps.
        if let FrozenMapSpec::Random {
            nrow,
            ncol,
            frozen_prob,
        } = &self.config.map.clone()
        {
            self.map = generate_random_map(*nrow, *ncol, *frozen_prob, &mut self.rng)
                .map_err(|e| EnvironmentError::RenderFailed(e.to_string()))?;
        }
        self.state = FrozenLakeState {
            row: (self.map.start_pos / self.map.ncol) as u8,
            col: (self.map.start_pos % self.map.ncol) as u8,
            nrow: self.map.nrow as u8,
            ncol: self.map.ncol as u8,
        };
        // Only after the reset has actually succeeded.
        self.guard.reset();
        Ok(SnapshotBase::running(
            self.state.observe(),
            ScalarReward(0.0),
        ))
    }

    /// Moves the agent one tile, terminating on a `Hole` or `Goal`.
    ///
    /// # Errors
    ///
    /// Returns [`EnvironmentError::StepAfterEpisodeEnd`] if the episode already
    /// ended. The check is the first statement, before any state mutation and
    /// before `resolve_action` draws from the RNG,
    /// so a rejected call leaves both the agent position and the slip stream
    /// untouched — a rejected step must not perturb a seeded run.
    fn step(&mut self, action: FrozenLakeAction) -> Result<Self::SnapshotType, EnvironmentError> {
        self.guard.check()?;

        let effective = self.resolve_action(action);
        let (nr, nc) = apply_action(
            self.state.row,
            self.state.col,
            effective,
            self.state.nrow,
            self.state.ncol,
        );
        self.state.row = nr;
        self.state.col = nc;

        let tile = self.tile_at(nr, nc);
        let obs = self.state.observe();
        // Build the snapshot once, then record its own status — no `match` arm
        // can return without the guard seeing what it emitted.
        let snapshot = match tile {
            Tile::Hole => {
                SnapshotBase::terminated(obs, ScalarReward(self.config.reward_schedule.hole))
            }
            Tile::Goal => {
                SnapshotBase::terminated(obs, ScalarReward(self.config.reward_schedule.goal))
            }
            _ => SnapshotBase::running(obs, ScalarReward(self.config.reward_schedule.frozen)),
        };
        self.guard.record(snapshot.status);
        Ok(snapshot)
    }
}

// ---------------------------------------------------------------------------
// ASCII renderer
// ---------------------------------------------------------------------------

impl crate::render::AsciiRenderable for FrozenLake {
    fn render_ascii(&self) -> String {
        let mut out = String::with_capacity(self.map.tiles.len() * 2);
        for (idx, tile) in self.map.tiles.iter().enumerate() {
            let row = idx / self.map.ncol;
            let col = idx % self.map.ncol;
            let on_agent = row as u8 == self.state.row && col as u8 == self.state.col;
            let ch = if on_agent { '@' } else { tile_char(*tile) };
            out.push(ch);
            out.push(' ');
            if col + 1 == self.map.ncol {
                out.push('\n');
            }
        }
        out
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        use crate::render::palette::{
            AGENT_FG, AGENT_MODIFIER, GOAL_FG, GOAL_MODIFIER, HAZARD_FG, HAZARD_MODIFIER,
        };
        use crate::render::{Color, Modifier, SpanStyle, StyledFrame, StyledLine, StyledSpan};

        let mut lines = Vec::with_capacity(self.map.nrow);
        let mut spans: Vec<StyledSpan> = Vec::new();
        let mut current_style = SpanStyle::default();
        let mut current_text = String::new();
        for (idx, tile) in self.map.tiles.iter().enumerate() {
            let row = idx / self.map.ncol;
            let col = idx % self.map.ncol;
            let on_agent = row as u8 == self.state.row && col as u8 == self.state.col;
            let (ch, style) = if on_agent {
                (
                    '@',
                    SpanStyle::default()
                        .fg(AGENT_FG)
                        .with_modifier(AGENT_MODIFIER),
                )
            } else {
                match *tile {
                    Tile::Hole => (
                        'H',
                        SpanStyle::default()
                            .fg(HAZARD_FG)
                            .with_modifier(HAZARD_MODIFIER),
                    ),
                    Tile::Goal => (
                        'G',
                        SpanStyle::default()
                            .fg(GOAL_FG)
                            .with_modifier(GOAL_MODIFIER),
                    ),
                    Tile::Start => (
                        'S',
                        SpanStyle::default()
                            .fg(Color::Yellow)
                            .with_modifier(Modifier::BOLD),
                    ),
                    Tile::Frozen => ('F', SpanStyle::default()),
                }
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
            if col + 1 == self.map.ncol {
                if !current_text.is_empty() {
                    spans.push(StyledSpan::new(
                        std::mem::take(&mut current_text),
                        current_style,
                    ));
                }
                lines.push(StyledLine::from_spans(std::mem::take(&mut spans)));
                current_style = SpanStyle::default();
            }
        }
        StyledFrame { lines }
    }
}

const fn tile_char(t: Tile) -> char {
    match t {
        Tile::Start => 'S',
        Tile::Frozen => 'F',
        Tile::Hole => 'H',
        Tile::Goal => 'G',
    }
}

impl rlevo_core::render::payload::TabularPayloadSource for FrozenLake {
    fn tabular_snapshot(&self) -> rlevo_core::render::payload::TabularSnapshot {
        use rlevo_core::render::payload::{
            TabularCell, TabularGrid, TabularLayout, TabularMarker, TabularMarkerKind,
            TabularSnapshot,
        };
        let cells = self
            .map
            .tiles
            .iter()
            .map(|t| match t {
                Tile::Start => TabularCell::Start,
                Tile::Frozen => TabularCell::Frozen,
                Tile::Hole => TabularCell::Hazard,
                Tile::Goal => TabularCell::Goal,
            })
            .collect();
        TabularSnapshot {
            layout: TabularLayout::Grid(TabularGrid {
                width: self.map.ncol as u16,
                height: self.map.nrow as u16,
                cells,
                markers: vec![TabularMarker {
                    x: u16::from(self.state.col),
                    y: u16::from(self.state.row),
                    kind: TabularMarkerKind::Agent,
                }],
            }),
        }
    }
}

#[cfg(test)]
/// Unit tests for [`FrozenLake`], covering map validation, tile transitions,
/// reward customisation, slippery distributions, random map generation, and determinism.
mod tests {
    use super::*;
    use rlevo_core::action::DiscreteAction;
    use rlevo_core::base::Observation;
    use rlevo_core::environment::Snapshot;

    #[test]
    fn default_config_validates() {
        assert!(FrozenLakeConfig::default().validate().is_ok());
    }

    #[test]
    fn rejects_out_of_range_success_rate() {
        let bad = FrozenLakeConfig {
            success_rate: 1.5,
            ..Default::default()
        };
        assert_eq!(bad.validate().unwrap_err().field, "success_rate");
    }

    fn four_env() -> FrozenLake {
        FrozenLake::with_config(
            FrozenLakeConfig::builder()
                .map(FrozenMapSpec::Preset(FrozenPreset::Four4x4))
                .is_slippery(false)
                .seed(0)
                .build(),
        )
        .unwrap()
    }

    #[test]
    /// Verifies the discrete action count matches the four-direction action space.
    fn action_count() {
        assert_eq!(FrozenLakeAction::ACTION_COUNT, 4);
    }

    #[test]
    fn tabular_snapshot_projects_grid_and_agent() {
        use rlevo_core::render::payload::{
            TabularCell, TabularLayout, TabularMarkerKind, TabularPayloadSource,
        };

        let env = four_env();
        let snap = env.tabular_snapshot();
        let TabularLayout::Grid(grid) = snap.layout else {
            panic!("FrozenLake must project a grid layout");
        };
        // The classic 4×4 preset.
        assert_eq!(grid.width, 4);
        assert_eq!(grid.height, 4);
        assert_eq!(grid.cells.len(), 16);
        // "SFFF / FHFH / FFFH / HFFG": start top-left, goal bottom-right, holes present.
        assert_eq!(grid.cells[0], TabularCell::Start);
        assert_eq!(grid.cells[15], TabularCell::Goal);
        assert!(grid.cells.contains(&TabularCell::Hazard));
        // One agent marker at the start cell (0, 0).
        assert_eq!(grid.markers.len(), 1);
        assert_eq!(grid.markers[0].kind, TabularMarkerKind::Agent);
        assert_eq!((grid.markers[0].x, grid.markers[0].y), (0, 0));
    }

    #[test]
    /// Verifies `from_index` and `to_index` are inverses for all valid action indices.
    fn action_roundtrip() {
        for i in 0..FrozenLakeAction::ACTION_COUNT {
            assert_eq!(FrozenLakeAction::from_index(i).to_index(), i);
        }
    }

    #[test]
    /// Verifies the 4×4 preset map has exactly 16 cells.
    fn four_by_four_has_16_states() {
        let env = four_env();
        assert_eq!(env.map.nrow * env.map.ncol, 16);
    }

    #[test]
    /// Verifies the 4×4 preset has start at index 0 and goal at index 15.
    fn default_start_is_0_goal_is_15() {
        let env = four_env();
        assert_eq!(env.map.start_pos, 0);
        assert_eq!(env.map.tiles[15], Tile::Goal);
    }

    #[test]
    /// Verifies the observation shape is fixed at 64 (8×8 max).
    fn obs_shape() {
        assert_eq!(FrozenLakeObservation::shape(), [64]);
    }

    #[test]
    /// Verifies that navigating to the goal terminates the episode with reward +1.
    fn reached_goal_terminates() {
        let mut env = four_env();
        env.reset().unwrap();
        // Navigate to (3,3) from (0,0) in the 4x4 map deterministically.
        // Path: Down×3, Right×3 (avoiding holes at (1,1),(1,3),(2,3),(3,0)).
        // A safe route in SFFF/FHFH/FFFH/HFFG:
        // (0,0)→(1,0)→(2,0)→(2,1)→(2,2)→(3,2)→(3,3)
        let path = [
            FrozenLakeAction::Down,
            FrozenLakeAction::Down,
            FrozenLakeAction::Right,
            FrozenLakeAction::Right,
            FrozenLakeAction::Down,
            FrozenLakeAction::Right,
        ];
        let mut last_snap = None;
        for &a in &path {
            let snap = env.step(a).unwrap();
            if snap.is_done() {
                last_snap = Some(snap);
                break;
            }
            last_snap = Some(snap);
        }
        let snap = last_snap.unwrap();
        assert!(snap.is_terminated(), "goal must terminate");
        let r: f32 = (*snap.reward()).into();
        assert!((r - 1.0).abs() < 1e-6, "goal reward must be 1.0, got {r}");
    }

    #[test]
    /// Verifies that stepping into a hole terminates the episode with the default hole reward.
    fn stepped_into_hole_terminates() {
        let mut env = four_env();
        env.reset().unwrap();
        // (0,0) → Down → (1,0). (1,0) is 'F'. → Right → (1,1) is 'H'.
        env.step(FrozenLakeAction::Down).unwrap();
        let snap = env.step(FrozenLakeAction::Right).unwrap();
        assert!(snap.is_terminated(), "hole must terminate");
        let r: f32 = (*snap.reward()).into();
        assert_eq!(r, 0.0, "default hole reward is 0.0");
    }

    #[test]
    /// Verifies that a custom [`RewardSchedule`] is applied correctly on hole entry.
    fn reward_schedule_customisable() {
        let cfg = FrozenLakeConfig::builder()
            .map(FrozenMapSpec::Preset(FrozenPreset::Four4x4))
            .is_slippery(false)
            .reward_schedule(RewardSchedule {
                goal: 100.0,
                hole: -10.0,
                frozen: 0.0,
            })
            .seed(0)
            .build();
        let mut env = FrozenLake::with_config(cfg).unwrap();
        env.reset().unwrap();
        // Step into hole at (1,1): Down then Right.
        env.step(FrozenLakeAction::Down).unwrap();
        let snap = env.step(FrozenLakeAction::Right).unwrap();
        let r: f32 = (*snap.reward()).into();
        assert!(
            (r - (-10.0)).abs() < 1e-6,
            "custom hole reward -10.0, got {r}"
        );
    }

    #[test]
    /// Verifies that 100 randomly generated 8×8 maps all contain a reachable goal.
    fn generate_random_map_is_solvable() {
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let m = generate_random_map(8, 8, 0.8, &mut rng).unwrap();
            assert_eq!(m.nrow * m.ncol, 64);
        }
    }

    #[test]
    /// Verifies that slippery-mode movement frequency matches `success_rate = 1/3`.
    fn slippery_mean_direction_differs_from_action() {
        let cfg = FrozenLakeConfig::builder()
            .map(FrozenMapSpec::Preset(FrozenPreset::Eight8x8))
            .is_slippery(true)
            .success_rate(1.0 / 3.0)
            .seed(7)
            .build();
        let mut env = FrozenLake::with_config(cfg).unwrap();
        env.reset().unwrap();

        let n = 10_000u32;
        let mut right_count = 0u32;
        for _ in 0..n {
            // This is a slip-distribution harness, not an episode: it places the
            // agent by hand, so it must re-open the episode by hand too. The
            // intended Right lands on the hole at (2,3), which ends the episode
            // — without this the guard would (correctly) reject the next step.
            // `guard.reset()` rather than `reset()` because the latter would also
            // move the agent back to the start tile we are about to overwrite.
            env.guard.reset();
            env.state = FrozenLakeState {
                row: 2,
                col: 2,
                nrow: 8,
                ncol: 8,
            };
            env.step(FrozenLakeAction::Right).unwrap();
            if env.state.col == 3 {
                right_count += 1;
            }
        }
        let p = right_count as f32 / n as f32;
        // With success_rate = 1/3, expected p ≈ 1/3.
        let tol = 3.0 * ((1.0f32 / 3.0) * (2.0 / 3.0) / n as f32).sqrt();
        assert!((p - 1.0 / 3.0).abs() < tol, "slippery p={p}, expected ≈1/3");
    }

    #[test]
    /// Verifies the slip distribution at `success_rate = 0.75` (75% intended, 12.5% each perpendicular).
    fn success_rate_distribution_at_0_75() {
        let cfg = FrozenLakeConfig::builder()
            .map(FrozenMapSpec::Preset(FrozenPreset::Eight8x8))
            .is_slippery(true)
            .success_rate(0.75)
            .seed(13)
            .build();
        let mut env = FrozenLake::with_config(cfg).unwrap();
        env.reset().unwrap();

        let n = 10_000u32;
        let (mut intended, mut perp1, mut perp2) = (0u32, 0u32, 0u32);
        for _ in 0..n {
            // Same hand-placed harness as above; (4,4)'s neighbours happen to be
            // all frozen, but re-opening the episode keeps the harness correct
            // independently of the tiles it lands on.
            env.guard.reset();
            env.state = FrozenLakeState {
                row: 4,
                col: 4,
                nrow: 8,
                ncol: 8,
            };
            env.step(FrozenLakeAction::Right).unwrap();
            match (env.state.row, env.state.col) {
                (4, 5) => intended += 1,
                (3, 4) => perp1 += 1,
                (5, 4) => perp2 += 1,
                _ => {}
            }
        }
        let p_int = intended as f32 / n as f32;
        let p_p1 = perp1 as f32 / n as f32;
        let p_p2 = perp2 as f32 / n as f32;
        let tol = 4.0 * (0.125f32 * 0.875 / n as f32).sqrt();
        assert!((p_int - 0.75).abs() < tol * 2.0, "intended p={p_int}");
        assert!((p_p1 - 0.125).abs() < tol, "perp1 p={p_p1}");
        assert!((p_p2 - 0.125).abs() < tol, "perp2 p={p_p2}");
    }

    #[test]
    /// Verifies that two slippery environments seeded identically produce the same cumulative reward.
    fn determinism() {
        let cfg = FrozenLakeConfig::builder()
            .map(FrozenMapSpec::Preset(FrozenPreset::Four4x4))
            .is_slippery(true)
            .seed(21)
            .build();
        let run = || {
            let mut env = FrozenLake::with_config(cfg.clone()).unwrap();
            let mut total = 0.0_f32;
            for _ in 0..5 {
                env.reset().unwrap();
                for _ in 0..20 {
                    let snap = env.step(FrozenLakeAction::Right).unwrap();
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
    //
    // On termination the agent is left standing *on* the Hole/Goal tile. An
    // unguarded `step()` walks it back onto a frozen neighbour and reports
    // `Running`, resurrecting a finished episode — hence the `EpisodeGuard`.
    // All of these use the non-slippery 4×4 preset so the terminal tile is
    // reached deterministically.

    #[test]
    /// Verifies a `step()` after falling into a hole is rejected, and that the
    /// rejected step leaves the agent on the hole tile.
    fn test_frozen_lake_step_after_hole_is_rejected() {
        use crate::episode::assert_rejects_post_terminal_step;

        let mut env = four_env();
        assert_rejects_post_terminal_step(
            &mut env,
            |env| {
                env.reset().expect("reset on the 4x4 preset must succeed");
                env.step(FrozenLakeAction::Down)
                    .expect("(0,0) → Down → (1,0) is frozen");
                env.step(FrozenLakeAction::Right)
                    .expect("(1,0) → Right → (1,1) is a hole")
            },
            // Legal action, illegal call sequence: Up would walk the agent out of
            // the hole to the frozen (0,1) were the episode not already over.
            FrozenLakeAction::Up,
        );

        assert_eq!(
            (env.state.row, env.state.col),
            (1, 1),
            "the rejected step must leave the agent in the hole, not walk it out"
        );
    }

    #[test]
    /// Verifies a `step()` after reaching the goal is rejected, and that the
    /// rejected step leaves the agent on the goal tile.
    fn test_frozen_lake_step_after_goal_is_rejected() {
        use crate::episode::assert_rejects_post_terminal_step;

        let mut env = four_env();
        assert_rejects_post_terminal_step(
            &mut env,
            |env| {
                env.reset().expect("reset on the 4x4 preset must succeed");
                // (0,0)→(1,0)→(2,0)→(2,1)→(2,2)→(3,2)→(3,3), the goal.
                let path = [
                    FrozenLakeAction::Down,
                    FrozenLakeAction::Down,
                    FrozenLakeAction::Right,
                    FrozenLakeAction::Right,
                    FrozenLakeAction::Down,
                ];
                for &a in &path {
                    let snap = env.step(a).expect("the route to the goal is all frozen");
                    assert!(!snap.is_done(), "the route to the goal must not terminate");
                }
                env.step(FrozenLakeAction::Right)
                    .expect("(3,2) → Right → (3,3) is the goal")
            },
            // Legal action, illegal call sequence: Left would walk the agent off
            // the goal back to the frozen (3,2).
            FrozenLakeAction::Left,
        );

        assert_eq!(
            (env.state.row, env.state.col),
            (3, 3),
            "the rejected step must leave the agent on the goal, not walk it off"
        );
    }

    #[test]
    /// Verifies `reset()` re-opens an environment whose episode has terminated.
    fn test_frozen_lake_reset_reopens_terminated_episode() {
        let mut env = four_env();
        env.reset().expect("reset on the 4x4 preset must succeed");
        env.step(FrozenLakeAction::Down)
            .expect("(0,0) → Down → (1,0) is frozen");
        let snap = env
            .step(FrozenLakeAction::Right)
            .expect("(1,0) → Right → (1,1) is a hole");
        assert!(snap.is_terminated(), "falling into a hole must terminate");
        assert!(
            env.step(FrozenLakeAction::Up).is_err(),
            "the episode has ended; a further step must be rejected"
        );

        env.reset().expect("reset must re-open the environment");
        let snap = env
            .step(FrozenLakeAction::Right)
            .expect("reset() must re-open the episode for stepping");
        assert!(
            !snap.is_done(),
            "a re-opened episode steps onto the frozen (0,1) and keeps running"
        );
        assert_eq!(
            (env.state.row, env.state.col),
            (0, 1),
            "the new episode must start from the start tile, not the hole"
        );
    }

    #[test]
    /// Verifies a rejected post-terminal `step()` draws nothing from the slip RNG.
    ///
    /// `guard.check()?` runs before `resolve_action`, so a rejected call cannot
    /// advance the RNG stream; were the check placed after the draw, a rejected
    /// step would silently desynchronise every subsequent episode of a seeded run.
    fn test_frozen_lake_rejected_step_does_not_advance_rng() {
        fn slippery_env() -> FrozenLake {
            FrozenLake::with_config(
                FrozenLakeConfig::builder()
                    .map(FrozenMapSpec::Preset(FrozenPreset::Four4x4))
                    .is_slippery(true)
                    .seed(21)
                    .build(),
            )
            .expect("the 4x4 preset must build")
        }

        /// Walks Right until the episode ends (slips make this terminate quickly).
        fn drive_to_done(env: &mut FrozenLake) {
            env.reset().expect("reset must succeed");
            for _ in 0..500 {
                let snap = env
                    .step(FrozenLakeAction::Right)
                    .expect("stepping a running episode must succeed");
                if snap.is_done() {
                    return;
                }
            }
            panic!("a slippery 4x4 walk must reach a hole or the goal within 500 steps");
        }

        /// Records the agent's path over one fresh episode.
        fn next_episode_path(env: &mut FrozenLake) -> Vec<(u8, u8)> {
            env.reset().expect("reset must succeed");
            let mut path = Vec::new();
            for _ in 0..500 {
                let snap = env
                    .step(FrozenLakeAction::Right)
                    .expect("stepping a running episode must succeed");
                path.push((env.state.row, env.state.col));
                if snap.is_done() {
                    break;
                }
            }
            path
        }

        // Identical seeds, identically driven to termination: the two RNG streams
        // are in the same position.
        let mut untouched = slippery_env();
        drive_to_done(&mut untouched);

        let mut rejected = slippery_env();
        drive_to_done(&mut rejected);
        assert!(
            rejected.step(FrozenLakeAction::Right).is_err(),
            "the episode has ended; a further step must be rejected"
        );

        // The rejected step is the only difference between the two environments.
        assert_eq!(
            next_episode_path(&mut untouched),
            next_episode_path(&mut rejected),
            "a rejected step must not consume a slip draw: the next episode must replay identically"
        );
    }

    #[test]
    fn render_styled_matches_ascii() {
        use crate::render::AsciiRenderable;

        let mut env = FrozenLake::with_config(FrozenLakeConfig::default()).unwrap();
        env.reset().unwrap();
        let plain = env.render_ascii();
        let styled = env.render_styled();
        // Each map row produces one styled line; the plain output has a
        // trailing newline per row, so compare against the un-trailing form.
        let plain_no_trailing: String = plain.lines().collect::<Vec<_>>().join("\n");
        assert_eq!(styled.plain_text(), plain_no_trailing);
    }

    #[test]
    fn render_styled_uses_palette_consts() {
        use crate::render::AsciiRenderable;
        use crate::render::palette::{AGENT_FG, GOAL_FG, HAZARD_FG};

        let mut env = FrozenLake::with_config(FrozenLakeConfig::default()).unwrap();
        env.reset().unwrap();
        let styled = env.render_styled();

        let mut found_agent = false;
        let mut found_goal = false;
        let mut found_hole = false;
        for line in &styled.lines {
            for span in &line.spans {
                if span.text.starts_with('@') {
                    assert_eq!(span.style.fg, Some(AGENT_FG));
                    found_agent = true;
                }
                if span.text.starts_with('G') {
                    assert_eq!(span.style.fg, Some(GOAL_FG));
                    found_goal = true;
                }
                if span.text.starts_with('H') {
                    assert_eq!(span.style.fg, Some(HAZARD_FG));
                    found_hole = true;
                }
            }
        }
        assert!(found_agent, "agent glyph @ not found in styled output");
        assert!(found_goal, "goal glyph G not found in styled output");
        assert!(found_hole, "hole glyph H not found in default 4x4 map");
    }

    #[test]
    fn render_ascii_within_width_budget() {
        use crate::render::AsciiRenderable;

        let mut env = FrozenLake::with_config(FrozenLakeConfig::default()).unwrap();
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
