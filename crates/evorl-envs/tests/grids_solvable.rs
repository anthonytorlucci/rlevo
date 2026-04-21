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
//! re-exported from `evorl_envs::grids` (public API only), not private
//! helpers accessible from the source modules.

use evorl_core::environment::{Environment, Snapshot};
use evorl_envs::grids::core::GridAction;
use evorl_envs::grids::{
    Color, CrossingConfig, CrossingEnv, CrossingKind, DistShiftConfig, DistShiftEnv,
    DistShiftVariant, DoorKeyConfig, DoorKeyEnv, DynamicObstaclesConfig, DynamicObstaclesEnv,
    EmptyConfig, EmptyEnv, FourRoomsConfig, FourRoomsEnv, GoToDoorConfig, GoToDoorEnv,
    LavaGapConfig, LavaGapEnv, MemoryConfig, MemoryEnv, MultiRoomConfig, MultiRoomEnv,
    UnlockConfig, UnlockEnv, UnlockPickupConfig, UnlockPickupEnv,
};

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

#[test]
fn empty_is_solvable() {
    let mut env = EmptyEnv::with_config(EmptyConfig::new(5, 100, 0), false);
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
    let mut env = DoorKeyEnv::with_config(DoorKeyConfig::new(5, 100, 0), false);
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
    let mut env = LavaGapEnv::with_config(LavaGapConfig::new(5, 100, 0), false);
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
    let mut env = UnlockEnv::with_config(UnlockConfig::new(5, 100, 0), false);
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
    let mut env = UnlockPickupEnv::with_config(UnlockPickupConfig::new(7, 196, 0), false);
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
    let mut env = CrossingEnv::with_config(
        CrossingConfig::new(7, 196, 0, CrossingKind::Lava),
        false,
    );
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
    let mut env = CrossingEnv::with_config(
        CrossingConfig::new(7, 196, 0, CrossingKind::Wall),
        false,
    );
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
    let mut env = DistShiftEnv::with_config(
        DistShiftConfig::new(DistShiftVariant::One, 100, 0),
        false,
    );
    let script = [GridAction::Forward; 6];
    assert_solvable!(env, script);
}

#[test]
fn dist_shift_variant_two_is_solvable() {
    let mut env = DistShiftEnv::with_config(
        DistShiftConfig::new(DistShiftVariant::Two, 100, 0),
        false,
    );
    let script = [GridAction::Forward; 6];
    assert_solvable!(env, script);
}

#[test]
fn go_to_door_red_is_solvable() {
    let mut env = GoToDoorEnv::with_config(
        GoToDoorConfig::new(6, 100, 0, Color::Red),
        false,
    );
    let script = [
        GridAction::Forward,
        GridAction::TurnLeft,
        GridAction::Forward,
        GridAction::Done,
    ];
    assert_solvable!(env, script);
}

#[test]
fn four_rooms_is_solvable() {
    let mut env = FourRoomsEnv::with_config(FourRoomsConfig::new(11, 400, 0), false);
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
    let mut env = MultiRoomEnv::with_config(MultiRoomConfig::default(), false);
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

#[test]
fn memory_default_is_solvable() {
    let mut env = MemoryEnv::with_config(MemoryConfig::new(140, 0, false), false);
    let script = [
        GridAction::TurnRight,
        GridAction::TurnRight,
        GridAction::Forward,
        GridAction::Forward,
        GridAction::Forward,
        GridAction::TurnLeft,
        GridAction::Done,
    ];
    assert_solvable!(env, script);
}

#[test]
fn memory_swapped_is_solvable() {
    let mut env = MemoryEnv::with_config(MemoryConfig::new(140, 0, true), false);
    let script = [
        GridAction::TurnRight,
        GridAction::TurnRight,
        GridAction::Forward,
        GridAction::Forward,
        GridAction::Forward,
        GridAction::TurnRight,
        GridAction::Done,
    ];
    assert_solvable!(env, script);
}

#[test]
fn dynamic_obstacles_with_zero_obstacles_is_solvable() {
    let mut env = DynamicObstaclesEnv::with_config(
        DynamicObstaclesConfig::new(5, 0, 100, 0),
        false,
    );
    let script = [
        GridAction::Forward,
        GridAction::Forward,
        GridAction::TurnRight,
        GridAction::Forward,
        GridAction::Forward,
    ];
    assert_solvable!(env, script);
}
