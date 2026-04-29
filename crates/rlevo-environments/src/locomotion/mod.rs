//! # MuJoCo-style locomotion environments
//!
//! Eleven continuous-control environments modelled on the Gymnasium v5 MuJoCo
//! suite, ported to the pure-Rust [`rapier3d`] physics engine. Enable with:
//!
//! ```toml
//! rlevo-environments = { features = ["locomotion"] }
//! ```
//!
//! ## Physics note
//!
//! This module simulates via Rapier3D, **not MuJoCo**. Observation shapes,
//! action dimensions, reward structure, and termination conditions match
//! Gymnasium v5. Absolute reward values, learned policies, and published
//! benchmark scores **will not transfer** to real Gymnasium/MuJoCo without
//! retuning. Treat these environments as MuJoCo-inspired testbeds for
//! in-workspace RL research, not as a Gymnasium-parity port.
//!
//! For a future MuJoCo-parity path, the `mujoco-ffi` cargo feature is
//! reserved. Enabling it today triggers a `compile_error!` pointing at the
//! deferred spec; see [`backend::mujoco_ffi`] once that work lands.
//!
//! ## Envs
//!
//! | Env | Action dim | Obs dim (default) | Terminates? |
//! |---|---|---|---|
//! | Ant | 8 | 105 | z ∉ (0.2, 1.0) |
//! | HalfCheetah | 6 | 17 | never |
//! | Hopper | 3 | 11 | z + angle + state ranges |
//! | Humanoid | 17 | 348 | z ∉ (1.0, 2.0) |
//! | HumanoidStandup | 17 | 348 | never |
//! | InvertedPendulum | 1 | 4 | `|angle| ≥ 0.2` |
//! | InvertedDoublePendulum | 1 | 9 | `y_tip ≤ 1.0` |
//! | Pusher | 7 | 23 | never |
//! | Reacher | 2 | 10 | never |
//! | Swimmer | 2 | 8 | never |
//! | Walker2d | 6 | 17 | z + angle ranges |

#[cfg(feature = "locomotion")]
pub mod backend;

#[cfg(feature = "locomotion")]
pub mod common;

#[cfg(feature = "locomotion")]
pub mod inverted_pendulum;

#[cfg(feature = "locomotion")]
pub mod reacher;

#[cfg(feature = "locomotion")]
pub mod inverted_double_pendulum;

#[cfg(feature = "locomotion")]
pub mod swimmer;
