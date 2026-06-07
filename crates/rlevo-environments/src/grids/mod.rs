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
//! # Environments
//!
//! | Environment | Module | Key challenge |
//! |---|---|---|
//! | [`EmptyEnv`] | [`empty`] | Baseline navigation, no obstacles |
//! | [`CrossingEnv`] | [`crossing`] | Cross lava or wall strips through a single gap |
//! | [`DistShiftEnv`] | [`dist_shift`] | Distribution shift between training and evaluation layouts |
//! | [`DoorKeyEnv`] | [`door_key`] | Long-horizon: key pickup → door unlock → goal |
//! | [`DynamicObstaclesEnv`] | [`dynamic_obstacles`] | Stochastic ball obstacles that random-walk each step |
//! | [`FourRoomsEnv`] | [`four_rooms`] | Multi-room maze; must transit ≥ 2 openings |
//! | [`GoToDoorEnv`] | [`go_to_door`] | Mission-conditioned: navigate to a colored door |
//! | [`LavaGapEnv`] | [`lava_gap`] | Cross a vertical lava strip through one gap |
//! | [`MemoryEnv`] | [`memory`] | Remember a cue object; match it at a fork |
//! | [`MultiRoomEnv`] | [`multi_room`] | Toggle doors open across a configurable room count |
//! | [`UnlockEnv`] | [`unlock`] | Pick up key, unlock and open a door |
//! | [`UnlockPickupEnv`] | [`unlock_pickup`] | Unlock a door then retrieve a box from the far room |

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
