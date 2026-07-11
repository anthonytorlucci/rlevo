//! Gridworld environments inspired by Farama Minigrid.
//!
//! This module ports a curated subset of Minigrid environments to Rust. All
//! grid envs share:
//!
//! - A **7×7 egocentric view** where the agent sits at view row `6`, column `3`
//!   and looks toward row `0`. Eleven of the twelve encode it as the shared
//!   3-channel [`GridObservation`](core::GridObservation) (`[type, color,
//!   state]` per cell); [`GoToDoorEnv`] is the one exception — see below.
//! - A **7-action discrete action space**: turn left, turn right, forward,
//!   pickup, drop, toggle, done.
//! - A **`1 - 0.9 * (step / max_steps)` success reward** formula and a
//!   scalar `ScalarReward` type.
//!
//! The shared primitives live in [`core`]; each concrete environment lives
//! in its own module alongside its config struct and test suite.
//!
//! # Environments
//!
//! | Environment | Module | Observation | Key challenge |
//! |---|---|---|---|
//! | [`EmptyEnv`] | [`empty`] | `7×7×3` | Baseline navigation, no obstacles |
//! | [`CrossingEnv`] | [`crossing`] | `7×7×3` | Cross lava or wall strips through a single gap |
//! | [`DistShiftEnv`] | [`dist_shift`] | `7×7×3` | Distribution shift between training and evaluation layouts |
//! | [`DoorKeyEnv`] | [`door_key`] | `7×7×3` | Long-horizon: key pickup → door unlock → goal |
//! | [`DynamicObstaclesEnv`] | [`dynamic_obstacles`] | `7×7×3` | Stochastic ball obstacles that random-walk each step |
//! | [`FourRoomsEnv`] | [`four_rooms`] | `7×7×3` | Multi-room maze; must transit ≥ 2 openings |
//! | [`GoToDoorEnv`] | [`go_to_door`] | **`7×7×4`** | Instruction-conditioned: reach the door whose colour matches the per-episode mission |
//! | [`LavaGapEnv`] | [`lava_gap`] | `7×7×3` | Cross a vertical lava strip through one gap |
//! | [`MemoryEnv`] | [`memory`] | `7×7×3` | POMDP recall: see a cue once, match its *type* at a fork long after it leaves view |
//! | [`MultiRoomEnv`] | [`multi_room`] | `7×7×3` | Toggle doors open across a configurable room count |
//! | [`UnlockEnv`] | [`unlock`] | `7×7×3` | Pick up key, unlock and open a door |
//! | [`UnlockPickupEnv`] | [`unlock_pickup`] | `7×7×3` | Unlock a door then retrieve a box from the far room |
//!
//! # The `GoToDoorEnv` exception: a fourth observation channel
//!
//! [`GoToDoorEnv`] is the family's **only** env that does not emit a
//! [`GridObservation`](core::GridObservation). Its per-episode instruction —
//! "go to the *Blue* door", with the four door colours re-sampled every reset —
//! has to reach the policy somehow, and none of the shared 3 channels can carry
//! it. It therefore emits a bespoke [`GoToDoorObservation`]: `7×7×4`, where
//! channels `0..3` are byte-identical to the shared entity encoding and channel
//! [`MISSION_CHANNEL`] (`3`) broadcasts the mission's colour byte to every cell,
//! in the *same* ordinal encoding the perceived door colour uses in channel 1 —
//! so a network can learn equality between the two. Its snapshot alias is
//! [`GoToDoorSnapshot`], not [`GridSnapshot`](core::GridSnapshot).
//!
//! The rank is still 3, so `Environment<3, 3, 1>` is unchanged — but any harness
//! that runs *one model across all grid envs* must handle both `7×7×3` and
//! `7×7×4`. This split and the reasoning behind it are recorded in ADR 0043,
//! which also makes it the precedent for how any future goal-conditioned rlevo
//! env surfaces its goal to a policy.

pub mod core;
pub mod crossing;
pub mod dist_shift;
pub mod door_key;
pub mod dynamic_obstacles;
pub mod empty;
pub mod four_rooms;
pub mod go_to_door;
pub mod lava_gap;
pub mod memory;
pub mod multi_room;
pub mod unlock;
pub mod unlock_pickup;

pub use core::{Color, GridAction};

pub use crossing::{CrossingConfig, CrossingEnv, CrossingKind};
pub use dist_shift::{DistShiftConfig, DistShiftEnv, DistShiftVariant};
pub use door_key::{DoorKeyConfig, DoorKeyEnv};
pub use dynamic_obstacles::{DynamicObstaclesConfig, DynamicObstaclesEnv};
pub use empty::{EmptyConfig, EmptyEnv};
pub use four_rooms::{FourRoomsConfig, FourRoomsEnv};
pub use go_to_door::{
    DOOR_COUNT, GO_TO_DOOR_OBS_CHANNELS, GoToDoorConfig, GoToDoorEnv, GoToDoorObservation,
    GoToDoorSnapshot, MISSION_CHANNEL, Mission,
};
pub use lava_gap::{LavaGapConfig, LavaGapEnv};
pub use memory::{MemoryConfig, MemoryEnv};
pub use multi_room::{MultiRoomConfig, MultiRoomEnv};
pub use unlock::{UnlockConfig, UnlockEnv};
pub use unlock_pickup::{UnlockPickupConfig, UnlockPickupEnv};
