//! FrozenLake-v1 environment.
//!
//! Grid MDP with four tile types: Start (`S`), Frozen (`F`), Hole (`H`), Goal (`G`).
//! Stepping onto `H` terminates with `reward_schedule.hole`. Reaching `G` terminates
//! with `reward_schedule.goal`. All other steps yield `reward_schedule.frozen`.
//!
//! Random maps are generated via BFS-verified sampling to guarantee reachability.

use std::collections::VecDeque;

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rlevo_core::action::DiscreteAction;
use rlevo_core::base::{Action, Observation, State};
use rlevo_core::environment::{Environment, EnvironmentError, SnapshotBase};
use rlevo_core::reward::ScalarReward;
use rlevo_core::state::StateError;
use serde::{Deserialize, Serialize};

use crate::toy_text::MapError;

// ── tile ──────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tile {
    Start,
    Frozen,
    Hole,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrozenPreset {
    Four4x4,
    Eight8x8,
}

const MAP_4X4: &[&str] = &["SFFF", "FHFH", "FFFH", "HFFG"];
const MAP_8X8: &[&str] = &[
    "SFFFFFFF", "FFFFFFFF", "FFFHFFFF", "FFFFFHFF", "FFFHFFFF", "FHHFFFHF", "FHFFHFHF", "FFFHFFFG",
];

// ── map spec (C2) ─────────────────────────────────────────────────────────────

/// Map source for [`FrozenLakeConfig`]. Replaces the original `desc + map_name + random` option pair.
#[derive(Debug, Clone)]
pub enum FrozenMapSpec {
    Preset(FrozenPreset),
    Custom(Vec<String>),
    Random {
        nrow: usize,
        ncol: usize,
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

#[derive(Debug, Clone)]
pub struct RewardSchedule {
    pub goal: f32,
    pub hole: f32,
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
#[derive(Debug, Clone)]
pub struct FrozenLakeConfig {
    pub map: FrozenMapSpec,
    /// When `true`, apply slip: `success_rate` intended, `(1-success_rate)/2` each perpendicular.
    pub is_slippery: bool,
    /// Probability of moving in the intended direction when slippery. Default: `1.0/3.0`.
    pub success_rate: f32,
    pub reward_schedule: RewardSchedule,
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
    pub fn builder() -> FrozenLakeConfigBuilder {
        FrozenLakeConfigBuilder::default()
    }
}

#[derive(Default)]
pub struct FrozenLakeConfigBuilder {
    map: Option<FrozenMapSpec>,
    is_slippery: bool,
    success_rate: Option<f32>,
    reward_schedule: Option<RewardSchedule>,
    seed: u64,
}

impl FrozenLakeConfigBuilder {
    pub fn map(mut self, m: FrozenMapSpec) -> Self {
        self.map = Some(m);
        self
    }
    pub fn is_slippery(mut self, v: bool) -> Self {
        self.is_slippery = v;
        self
    }
    pub fn success_rate(mut self, r: f32) -> Self {
        self.success_rate = Some(r);
        self
    }
    pub fn reward_schedule(mut self, rs: RewardSchedule) -> Self {
        self.reward_schedule = Some(rs);
        self
    }
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
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

/// State: grid position (row, col) plus map dimensions.
#[derive(Debug, Clone)]
pub struct FrozenLakeState {
    pub row: u8,
    pub col: u8,
    pub nrow: u8,
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

/// Observation: integer state id `row × ncol + col`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrozenLakeObservation {
    pub state_id: u16,
}

impl Observation<1> for FrozenLakeObservation {
    fn shape() -> [usize; 1] {
        [64]
    }
}

/// Four-direction action space (matches Gymnasium FrozenLake).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrozenLakeAction {
    Left = 0,
    Down = 1,
    Right = 2,
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
#[derive(Debug)]
pub struct FrozenLake {
    state: FrozenLakeState,
    map: ResolvedMap,
    config: FrozenLakeConfig,
    rng: StdRng,
}

impl FrozenLake {
    /// Create with a specific configuration. Returns `Err(MapError)` if a custom map is invalid
    /// or if random map generation fails.
    pub fn with_config(config: FrozenLakeConfig) -> Result<Self, MapError> {
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

impl Environment<1, 1, 1> for FrozenLake {
    type StateType = FrozenLakeState;
    type ObservationType = FrozenLakeObservation;
    type ActionType = FrozenLakeAction;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, FrozenLakeObservation, ScalarReward>;

    fn new(_render: bool) -> Self {
        Self::with_config(FrozenLakeConfig::default()).expect("default random map must succeed")
    }

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
        Ok(SnapshotBase::running(
            self.state.observe(),
            ScalarReward(0.0),
        ))
    }

    fn step(&mut self, action: FrozenLakeAction) -> Result<Self::SnapshotType, EnvironmentError> {
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
        match tile {
            Tile::Hole => Ok(SnapshotBase::terminated(
                obs,
                ScalarReward(self.config.reward_schedule.hole),
            )),
            Tile::Goal => Ok(SnapshotBase::terminated(
                obs,
                ScalarReward(self.config.reward_schedule.goal),
            )),
            _ => Ok(SnapshotBase::running(
                obs,
                ScalarReward(self.config.reward_schedule.frozen),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlevo_core::action::DiscreteAction;
    use rlevo_core::base::Observation;
    use rlevo_core::environment::Snapshot;

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
    fn action_count() {
        assert_eq!(FrozenLakeAction::ACTION_COUNT, 4);
    }

    #[test]
    fn action_roundtrip() {
        for i in 0..FrozenLakeAction::ACTION_COUNT {
            assert_eq!(FrozenLakeAction::from_index(i).to_index(), i);
        }
    }

    #[test]
    fn four_by_four_has_16_states() {
        let env = four_env();
        assert_eq!(env.map.nrow * env.map.ncol, 16);
    }

    #[test]
    fn default_start_is_0_goal_is_15() {
        let env = four_env();
        assert_eq!(env.map.start_pos, 0);
        assert_eq!(env.map.tiles[15], Tile::Goal);
    }

    #[test]
    fn obs_shape() {
        assert_eq!(FrozenLakeObservation::shape(), [64]);
    }

    #[test]
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
    fn generate_random_map_is_solvable() {
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let m = generate_random_map(8, 8, 0.8, &mut rng).unwrap();
            assert_eq!(m.nrow * m.ncol, 64);
        }
    }

    #[test]
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
}
