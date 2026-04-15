//! Gridworld environments inspired by Farama Minigrid.
//!
//! This module ports a curated subset of Minigrid environments to Rust. All
//! grid envs share:
//!
//! - A **7×7×3 egocentric observation** where the agent sits at view
//!   row `6`, column `3` and looks toward row `0`.
//! - A **7-action discrete action space**: turn left, turn right, forward,
//!   pickup, drop, toggle, done.
//! - A **`1 - 0.9 * (step / max_steps)` success reward** formula and a
//!   scalar `ScalarReward` type.
//!
//! The shared primitives live in [`core`]; each concrete environment lives
//! in its own module alongside its config struct and test suite.
//!
//! # Roadmap
//!
//! This module currently ships [`empty::EmptyEnv`]; additional Minigrid
//! ports (DoorKey, LavaGap, FourRooms, ...) land incrementally.

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
pub use go_to_door::{GoToDoorConfig, GoToDoorEnv, Mission};
pub use lava_gap::{LavaGapConfig, LavaGapEnv};
pub use memory::{MemoryConfig, MemoryEnv};
pub use multi_room::{MultiRoomConfig, MultiRoomEnv};
pub use unlock::{UnlockConfig, UnlockEnv};
pub use unlock_pickup::{UnlockPickupConfig, UnlockPickupEnv};
