//! Integration canary for the `grids` module.
//!
//! One test per environment: construct with a default-ish config, run a
//! scripted optimal rollout through the public API, and assert that the
//! episode terminates with positive reward. If any refactor breaks a
//! single env, one `cargo test --test grids_solvable` invocation tells
//! you exactly which one.
//!
//! These tests intentionally duplicate the per-env unit tests' optimal
//! rollouts — the goal here is to exercise only items that are
//! re-exported from `rlevo_environments::grids` (public API only), not private
//! helpers accessible from the source modules.
//!
//! ## Fixed scripts vs. seed-driven oracles
//!
//! Most grid envs are deterministic given their config, so a hard-coded action
//! sequence *is* the optimal script. Two are not: [`MemoryEnv`] samples its cue
//! (and therefore which fork arm is correct) per episode, and [`GoToDoorEnv`]
//! samples its four door colors and its target per episode (ADR 0043).
//!
//! For those two, a hard-coded script would be worse than broken — it would be
//! *right by luck* on some seeds and silently wrong on others. Their tests
//! instead **derive** the script from the environment after `reset`
//! (`MemoryEnv::match_pos` / `GoToDoorEnv::doors` + `mission`) and assert
//! solvability across a whole range of seeds. Do not replace them with a fixed
//! action list.

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

#[test]
fn door_key_is_solvable() {
    let mut env =
        DoorKeyEnv::with_config(DoorKeyConfig::new(5, 100, 0), false).expect("valid config");
    let script = [
        GridAction::Pickup,
        GridAction::TurnRight,
        GridAction::Toggle,
        GridAction::Toggle,
        GridAction::Forward,
        GridAction::Forward,
        GridAction::TurnRight,
        GridAction::Forward,
    ];
    assert_solvable!(env, script);
}

#[test]
fn lava_gap_is_solvable() {
    let mut env =
        LavaGapEnv::with_config(LavaGapConfig::new(5, 100, 0), false).expect("valid config");
    let script = [
        GridAction::TurnRight,
        GridAction::Forward,
        GridAction::TurnLeft,
        GridAction::Forward,
        GridAction::Forward,
        GridAction::TurnRight,
        GridAction::Forward,
    ];
    assert_solvable!(env, script);
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

#[test]
fn unlock_pickup_is_solvable() {
    let mut env = UnlockPickupEnv::with_config(UnlockPickupConfig::new(7, 196, 0), false)
        .expect("valid config");
    let script = [
        GridAction::Pickup,
        GridAction::TurnRight,
        GridAction::Forward,
        GridAction::TurnRight,
        GridAction::Forward,
        GridAction::TurnLeft,
        GridAction::Toggle,
        GridAction::Toggle,
        GridAction::TurnRight,
        GridAction::Drop,
        GridAction::TurnLeft,
        GridAction::Forward,
        GridAction::Forward,
        GridAction::Pickup,
    ];
    assert_solvable!(env, script);
}

#[test]
fn crossing_lava_is_solvable() {
    let mut env =
        CrossingEnv::with_config(CrossingConfig::new(7, 196, 0, CrossingKind::Lava), false)
            .expect("valid config");
    let script = [
        GridAction::Forward,
        GridAction::Forward,
        GridAction::TurnRight,
        GridAction::Forward,
        GridAction::Forward,
        GridAction::Forward,
        GridAction::Forward,
        GridAction::TurnLeft,
        GridAction::Forward,
        GridAction::Forward,
    ];
    assert_solvable!(env, script);
}

#[test]
fn crossing_wall_is_solvable() {
    let mut env =
        CrossingEnv::with_config(CrossingConfig::new(7, 196, 0, CrossingKind::Wall), false)
            .expect("valid config");
    let script = [
        GridAction::Forward,
        GridAction::Forward,
        GridAction::TurnRight,
        GridAction::Forward,
        GridAction::Forward,
        GridAction::Forward,
        GridAction::Forward,
        GridAction::TurnLeft,
        GridAction::Forward,
        GridAction::Forward,
    ];
    assert_solvable!(env, script);
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

#[test]
fn four_rooms_is_solvable() {
    let mut env =
        FourRoomsEnv::with_config(FourRoomsConfig::new(11, 400, 0), false).expect("valid config");
    let script = [
        GridAction::TurnRight,
        GridAction::Forward,
        GridAction::TurnLeft,
        GridAction::Forward,
        GridAction::Forward,
        GridAction::Forward,
        GridAction::Forward,
        GridAction::Forward,
        GridAction::Forward,
        GridAction::Forward,
        GridAction::TurnRight,
        GridAction::Forward,
        GridAction::Forward,
        GridAction::Forward,
        GridAction::Forward,
        GridAction::Forward,
        GridAction::Forward,
        GridAction::Forward,
        GridAction::TurnLeft,
        GridAction::Forward,
    ];
    assert_solvable!(env, script);
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
