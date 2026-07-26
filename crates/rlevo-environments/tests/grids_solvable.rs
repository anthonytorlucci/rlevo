//! Integration canary for the `grids` module.
//!
//! One test per environment: construct with a default-ish config, drive an
//! optimal rollout through the public API, and assert that the episode
//! terminates with positive reward. If any refactor breaks a single env, one
//! `cargo test --test grids_solvable` invocation tells you exactly which one.
//!
//! These tests intentionally duplicate the per-env unit tests' optimal
//! rollouts — the goal here is to exercise only items that are
//! re-exported from `rlevo_environments::grids` (public API only), not private
//! helpers accessible from the source modules.
//!
//! ## Three flavours of oracle
//!
//! **Fixed script.** Most grid envs are deterministic given their config, so a
//! hard-coded action sequence *is* the optimal script. [`EmptyEnv`],
//! [`DistShiftEnv`], [`DynamicObstaclesEnv`] (with zero obstacles), [`UnlockEnv`]
//! (#1020) and [`MultiRoomEnv`] (#1021) still use one.
//!
//! **Derived script.** Two envs sample part of the *task* per episode:
//! [`MemoryEnv`] samples its cue (and therefore which fork arm is correct), and
//! [`GoToDoorEnv`] samples its four door colors and its target (ADR 0043). For
//! those, a hard-coded script would be worse than broken — it would be *right by
//! luck* on some seeds and silently wrong on others. Their tests **derive** the
//! script from the environment after `reset` (`MemoryEnv::match_pos` /
//! `GoToDoorEnv::doors` + `mission`) and assert solvability across a whole range
//! of seeds. Do not replace them with a fixed action list.
//!
//! **Planned.** [`DoorKeyEnv`], [`LavaGapEnv`], [`UnlockPickupEnv`],
//! [`CrossingEnv`] (both kinds) and [`FourRoomsEnv`] read their board back from
//! `env.state()` and hand it to [`common::plan`], a BFS over the grid state, which
//! *computes* the action sequence. That sequence is then executed through
//! [`Environment::step`] exactly as a script would be, so nothing about the
//! end-to-end property changes.
//!
//! ### Why planning is a coverage *gain*, not a loss
//!
//! Read this diff as strictly more assertion, not less. A script proves that
//! **one** board is solvable. The planner proves that **every board the env
//! generates** is solvable — it is handed the board and must find a route, with
//! no privileged knowledge of the layout. That is the property these tests were
//! always trying to state, and it is the only form of it that survives the
//! per-episode layout randomization the five planned envs are about to gain
//! (#282). This migration deliberately landed *while the layouts are still
//! deterministic*, so the oracle was validated against a known-good board before
//! the board started moving; a later failure therefore isolates to the generator,
//! not to the oracle.
//!
//! Two assertions the scripted oracles never carried come with it:
//!
//! 1. **Every planned oracle runs at its env's `MIN_SIZE` and at a larger size.**
//!    `MIN_SIZE` is where derived positions collide and sampling ranges
//!    degenerate. This class of bug has already shipped here once —
//!    `tests/config_validation_chokepoint.rs` records
//!    `UnlockPickupConfig { size: 3 }` building a board with no `Door` cell at
//!    all, the key having overwritten it.
//! 2. **No planned oracle contains a coordinate or an action literal.** The board
//!    is read from the env, never asserted against a constant, so the parallel
//!    work that moves those coordinates cannot make this file quietly false.
//!
//! ### Adding randomization later
//!
//! Each planned oracle is shaped so that the *only* edit needed when its env
//! gains a per-episode layout is swapping the `env.reset()` for a
//! `for seed in 0..ORACLE_SEEDS { env.reset_with_seed(seed)?; … }` loop — see
//! [`go_to_door_is_solvable_for_every_seed`] for the shape. None of the five has
//! `reset_with_seed` yet, so all five currently call plain `reset()`.

mod common;

use rlevo_core::environment::{Environment, Snapshot};
use rlevo_core::reward::ScalarReward;
use rlevo_environments::grids::core::GridAction;
use rlevo_environments::grids::{
    CrossingConfig, CrossingEnv, CrossingKind, DistShiftConfig, DistShiftEnv, DistShiftVariant,
    DoorKeyConfig, DoorKeyEnv, DynamicObstaclesConfig, DynamicObstaclesEnv, EmptyConfig, EmptyEnv,
    FourRoomsConfig, FourRoomsEnv, GoToDoorConfig, GoToDoorEnv, LavaGapConfig, LavaGapEnv,
    MemoryConfig, MemoryEnv, MultiRoomConfig, MultiRoomEnv, UnlockConfig, UnlockEnv,
    UnlockPickupConfig, UnlockPickupEnv,
};

/// Number of distinct episodes the seed-driven oracle tests must clear.
const ORACLE_SEEDS: u64 = 32;

/// Every planned oracle runs at its env's `MIN_SIZE` floor and at one larger
/// board, per the degeneracy argument in the module docs.
///
/// The floors are duplicated here as literals because each env's `MIN_SIZE` is a
/// private const. They are pinned from the outside by
/// `tests/config_validation_chokepoint.rs`, which asserts the exact
/// `ConstraintKind::TooSmall { min, .. }` each `with_config` rejects — so a floor
/// that moves without this table moving fails there, loudly, first.
mod sizes {
    /// `DoorKeyEnv`: floor 5 (the split wall needs a sub-room on each side).
    pub const DOOR_KEY: [usize; 2] = [5, 9];
    /// `LavaGapEnv`: floor 5 (one interior cell each side of the lava column).
    pub const LAVA_GAP: [usize; 2] = [5, 9];
    /// `UnlockPickupEnv`: floor 7 (three interior columns per side).
    pub const UNLOCK_PICKUP: [usize; 2] = [7, 9];
    /// `CrossingEnv`: floor 7 (an interior row per strip, plus start and goal).
    pub const CROSSING: [usize; 2] = [7, 9];
    /// `FourRoomsEnv`: floor 11, and the size must stay **odd**.
    pub const FOUR_ROOMS: [usize; 2] = [11, 13];
}

/// The `max_steps` budget every planned oracle uses: the same `4 * size * size`
/// each env's own `Default` picks, so the oracle is not quietly given a budget
/// the shipped default would not have.
const fn budget(size: usize) -> usize {
    4 * size * size
}

/// Run `script` through `env` from a fresh `reset`, then assert the
/// episode terminated and paid a positive reward.
macro_rules! assert_solvable {
    ($env:expr, $script:expr) => {{
        let env = &mut $env;
        env.reset().expect("reset");
        let mut last = None;
        for action in $script {
            last = Some(env.step(action).expect("step"));
        }
        let snap = last.expect("script must contain at least one action");
        assert!(snap.is_done(), "script did not terminate the episode");
        let reward = f32::from(*snap.reward());
        assert!(reward > 0.0, "reward was {reward}, expected > 0.0");
        reward
    }};
}

/// Run `script` on an **already-reset** grid env and return the terminal reward.
///
/// The seed-driven oracle tests must read the environment *after* `reset` to
/// derive their script, so they cannot use [`assert_solvable`], which resets for
/// you. Panics if the script does not terminate the episode.
fn run_script<E>(env: &mut E, script: &[GridAction]) -> f32
where
    E: Environment<3, 3, 1, ActionType = GridAction, RewardType = ScalarReward>,
{
    let mut last = None;
    for &action in script {
        last = Some(env.step(action).expect("step"));
    }
    let snap = last.expect("script must contain at least one action");
    assert!(snap.is_done(), "script did not terminate the episode");
    f32::from(*snap.reward())
}

/// Plan a route across the board `$env` is **currently** holding, execute it
/// through the public `step` API, and assert the episode terminated for reward.
///
/// The env must already be reset. That is deliberate: keeping `reset` at the
/// call site is what makes the later randomization edit a one-line swap to a
/// `reset_with_seed(seed)` loop, with the planning and execution below already
/// layout-agnostic. Everything the failure messages report — the label, the
/// board — is read back from the environment, so this macro carries no knowledge
/// of any layout.
macro_rules! assert_planner_solves {
    ($env:expr, $solved:expr $(,)?) => {{
        let env = &mut $env;
        // `Display` on every grid env prints size / kind / step budget.
        let label = env.to_string();
        let plan = common::plan(env.state(), $solved).unwrap_or_else(|| {
            panic!(
                "{label}: the planner found no route across this board:\n{}",
                env.ascii()
            )
        });
        assert!(
            !plan.is_empty(),
            "{label}: a freshly reset board must not already be solved"
        );
        let reward = run_script(env, &plan);
        assert!(
            reward > 0.0,
            "{label}: the planned {}-action route paid {reward}, expected > 0.0",
            plan.len()
        );
        reward
    }};
}

#[test]
fn empty_is_solvable() {
    let mut env = EmptyEnv::with_config(EmptyConfig::new(5, 100, 0), false).expect("valid config");
    let script = [
        GridAction::Forward,
        GridAction::Forward,
        GridAction::TurnRight,
        GridAction::Forward,
        GridAction::Forward,
    ];
    let reward = assert_solvable!(env, script);
    assert!(reward > 0.9);
}

/// Planned oracle: whatever split column, door row, and key/goal placement the
/// board carries, the key→unlock→goal ordering is achievable.
#[test]
fn door_key_is_solvable() {
    for size in sizes::DOOR_KEY {
        let mut env = DoorKeyEnv::with_config(DoorKeyConfig::new(size, budget(size), 0), false)
            .expect("valid config");
        env.reset().expect("reset");
        assert_planner_solves!(env, common::on_goal);
    }
}

/// Planned oracle: the goal is reachable **without crossing lava** — the planner
/// refuses to expand a `HitLava` successor, so a plan existing at all is the
/// statement that the gap is passable.
#[test]
fn lava_gap_is_solvable() {
    for size in sizes::LAVA_GAP {
        let mut env = LavaGapEnv::with_config(LavaGapConfig::new(size, budget(size), 0), false)
            .expect("valid config");
        env.reset().expect("reset");
        assert_planner_solves!(env, common::on_goal);
    }
}

#[test]
fn unlock_is_solvable() {
    let mut env =
        UnlockEnv::with_config(UnlockConfig::new(5, 100, 0), false).expect("valid config");
    let script = [
        GridAction::Pickup,
        GridAction::TurnLeft,
        GridAction::Toggle,
        GridAction::Toggle,
    ];
    assert_solvable!(env, script);
}

/// Planned oracle: the target box is retrievable. The success predicate is
/// `carrying(env.target())` rather than "stand on a cell", and the target is read
/// off the env — this is the one env in the file whose win condition is an
/// inventory state, and it is the one whose sub-`MIN_SIZE` boards were measured
/// to lose their `Door` cell entirely (see the module docs).
///
/// Note the plan must include a `Drop`: `pickup` is a no-op with a full hand, so
/// the key has to leave the agent's hand before the box can enter it. The
/// planner discovers that ordering; nothing here encodes it.
#[test]
fn unlock_pickup_is_solvable() {
    for size in sizes::UNLOCK_PICKUP {
        let mut env =
            UnlockPickupEnv::with_config(UnlockPickupConfig::new(size, budget(size), 0), false)
                .expect("valid config");
        env.reset().expect("reset");
        let target = env.target();
        assert_planner_solves!(env, common::carrying(target));
    }
}

/// Planned oracle, lava strips: the gap column lines up into a passable corridor.
/// The planner will not route through lava, so a plan is proof of the corridor.
#[test]
fn crossing_lava_is_solvable() {
    for size in sizes::CROSSING {
        let cfg = CrossingConfig::new(size, budget(size), 0, CrossingKind::Lava);
        let mut env = CrossingEnv::with_config(cfg, false).expect("valid config");
        env.reset().expect("reset");
        assert_planner_solves!(env, common::on_goal);
    }
}

/// Planned oracle, wall strips: same geometry, but a mis-step only bumps instead
/// of ending the episode — so this case would still pass with a corridor the lava
/// variant could not survive. Both are asserted for that reason.
#[test]
fn crossing_wall_is_solvable() {
    for size in sizes::CROSSING {
        let cfg = CrossingConfig::new(size, budget(size), 0, CrossingKind::Wall);
        let mut env = CrossingEnv::with_config(cfg, false).expect("valid config");
        env.reset().expect("reset");
        assert_planner_solves!(env, common::on_goal);
    }
}

#[test]
fn dist_shift_variant_one_is_solvable() {
    let mut env =
        DistShiftEnv::with_config(DistShiftConfig::new(DistShiftVariant::One, 100, 0), false)
            .expect("valid config");
    let script = [GridAction::Forward; 6];
    assert_solvable!(env, script);
}

#[test]
fn dist_shift_variant_two_is_solvable() {
    let mut env =
        DistShiftEnv::with_config(DistShiftConfig::new(DistShiftVariant::Two, 100, 0), false)
            .expect("valid config");
    let script = [GridAction::Forward; 6];
    assert_solvable!(env, script);
}

/// Wall indices in [`GoToDoorEnv::doors`] order.
const NORTH: usize = 0;
const EAST: usize = 1;
const SOUTH: usize = 2;
const WEST: usize = 3;

/// Which wall carries the target-colored door this episode.
///
/// The colour→wall map is re-sampled every episode (ADR 0043), so this is the
/// only honest way to script the walk: ask the environment, do not assume.
fn target_wall(env: &GoToDoorEnv) -> usize {
    let target = env.mission().target_color;
    let hits: Vec<usize> = env
        .doors()
        .iter()
        .enumerate()
        .filter(|&(_, &(_, _, color))| color == target)
        .map(|(wall, _)| wall)
        .collect();
    assert_eq!(
        hits.len(),
        1,
        "exactly one of the four doors must wear the mission colour, got {hits:?}"
    );
    hits[0]
}

/// Walk from the fixed start pose `(2, 2)` facing East to a pose *facing* the
/// door in the middle of `wall` on a 6 × 6 grid, ending in `Done`.
fn go_to_door_script(wall: usize) -> &'static [GridAction] {
    use GridAction::{Done, Forward, TurnLeft, TurnRight};
    match wall {
        // (2,2)E → (3,2) → face N → (3,1); the door at (3,0) is now in front.
        NORTH => &[Forward, TurnLeft, Forward, Done],
        // (2,2)E → (3,2) → (4,2) → face S → (4,3) → face E; door at (5,3).
        EAST => &[Forward, Forward, TurnRight, Forward, TurnLeft, Done],
        // (2,2)E → (3,2) → face S → (3,3) → (3,4); door at (3,5) in front.
        SOUTH => &[Forward, TurnRight, Forward, Forward, Done],
        // (2,2)E → face S → (2,3) → face W → (1,3); door at (0,3) in front.
        WEST => &[TurnRight, Forward, TurnRight, Forward, Done],
        _ => unreachable!("wall index must be 0..4, got {wall}"),
    }
}

/// Seed-driven oracle: every episode's target door is reachable, whichever wall
/// the freshly-sampled mission colour happens to land on.
///
/// The old fixed "walk north to the Red door" script is gone with
/// `GoToDoorConfig::target_color` (ADR 0043) — the wall→colour map is now
/// re-sampled per episode, so the answer must be read back from the env.
#[test]
fn go_to_door_is_solvable_for_every_seed() {
    let mut env =
        GoToDoorEnv::with_config(GoToDoorConfig::new(6, 100, 0), false).expect("valid config");
    let mut walls_seen = [0usize; 4];

    for seed in 0..ORACLE_SEEDS {
        env.reset_with_seed(seed).expect("reset");
        let wall = target_wall(&env);
        walls_seen[wall] += 1;
        let reward = run_script(&mut env, go_to_door_script(wall));
        assert!(
            reward > 0.0,
            "seed {seed}: the target door sits on wall {wall} ({:?}), \
             but the scripted walk paid {reward}",
            env.mission().target_color
        );
    }

    // Guard against a degenerate pass: if every seed happened to target the same
    // wall, a hard-coded script would also have passed and this test would prove
    // nothing about the per-episode sampling.
    let distinct = walls_seen.iter().filter(|&&n| n > 0).count();
    assert!(
        distinct > 1,
        "the target wall must vary across {ORACLE_SEEDS} seeds, saw {walls_seen:?}"
    );
}

/// Planned oracle: the four `mid ± 3` openings connect the start quadrant to the
/// goal quadrant. Both sizes are odd, which `FourRoomsConfig::validate` requires
/// (an even size has no single centre row/column for the interior cross).
#[test]
fn four_rooms_is_solvable() {
    for size in sizes::FOUR_ROOMS {
        let mut env = FourRoomsEnv::with_config(FourRoomsConfig::new(size, budget(size), 0), false)
            .expect("valid config");
        env.reset().expect("reset");
        assert_planner_solves!(env, common::on_goal);
    }
}

#[test]
fn multi_room_is_solvable() {
    let mut env =
        MultiRoomEnv::with_config(MultiRoomConfig::default(), false).expect("valid config");
    let script = [
        GridAction::Forward,
        GridAction::Forward,
        GridAction::Forward,
        GridAction::Toggle,
        GridAction::Forward,
        GridAction::Forward,
        GridAction::Forward,
        GridAction::Forward,
        GridAction::Forward,
        GridAction::Toggle,
        GridAction::Forward,
        GridAction::Forward,
        GridAction::Forward,
        GridAction::Forward,
        GridAction::Forward,
    ];
    assert_solvable!(env, script);
}

/// The cue-reading oracle's script for the current episode.
///
/// Everything is derived from the env: the agent starts at `(1, size / 2)`
/// facing East, the fork junction is column `size - 2` on the same row, and the
/// matching object sits two rows above or below it. Walking `size - 3` cells
/// East lands on the junction; turning toward `match_pos` and stepping once puts
/// the matching object directly in front, where `Done` claims the reward.
fn memory_script(env: &MemoryEnv) -> Vec<GridAction> {
    let size = env.size();
    let mid = i32::try_from(size / 2).expect("grid size fits in i32");
    // Start column is 1, the fork column is `size - 2`.
    let forwards = size - 3;

    let mut script = vec![GridAction::Forward; forwards];
    script.push(if env.match_pos().1 < mid {
        GridAction::TurnLeft // face North, toward the upper fork object
    } else {
        GridAction::TurnRight // face South, toward the lower one
    });
    script.push(GridAction::Forward);
    script.push(GridAction::Done);
    script
}

/// Seed-driven oracle: every episode is winnable by an agent that *remembers*
/// the cue, whichever type it was and whichever arm of the fork it landed on.
///
/// This replaces the old `memory_default_is_solvable` / `memory_swapped_is_solvable`
/// pair. `MemoryConfig::swap_fork` is gone (ADR 0043): the fork order is now
/// sampled per episode, so a "swapped" variant is not a thing, and a hard-coded
/// turn direction would be right roughly half the time — passing by luck.
#[test]
fn memory_is_solvable_for_every_seed() {
    let mut env = MemoryEnv::with_config(MemoryConfig::default(), false).expect("valid config");
    let (mut upper_arm, mut lower_arm) = (0usize, 0usize);

    for seed in 0..ORACLE_SEEDS {
        env.reset_with_seed(seed).expect("reset");
        let script = memory_script(&env);
        if script.contains(&GridAction::TurnLeft) {
            upper_arm += 1;
        } else {
            lower_arm += 1;
        }
        let reward = run_script(&mut env, &script);
        assert!(
            reward > 0.0,
            "seed {seed}: cue {:?} matches the fork object at {:?}, \
             but the recall script paid {reward}",
            env.cue(),
            env.match_pos()
        );
    }

    // Guard against a degenerate pass: if every episode put the match on the same
    // arm, a fixed-turn script would pass too and this test would prove nothing.
    assert!(
        upper_arm > 0 && lower_arm > 0,
        "the matching fork arm must vary across {ORACLE_SEEDS} seeds, \
         saw {upper_arm} upper / {lower_arm} lower"
    );
}

#[test]
fn dynamic_obstacles_with_zero_obstacles_is_solvable() {
    let mut env =
        DynamicObstaclesEnv::with_config(DynamicObstaclesConfig::new(5, 0, 100, 0), false)
            .expect("valid config");
    let script = [
        GridAction::Forward,
        GridAction::Forward,
        GridAction::TurnRight,
        GridAction::Forward,
        GridAction::Forward,
    ];
    assert_solvable!(env, script);
}
