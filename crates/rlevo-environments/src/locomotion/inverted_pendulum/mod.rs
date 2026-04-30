//! # InvertedPendulum-v5 (Rapier3D-backed)
//!
//! # Physics note
//!
//! This env simulates dynamics via Rapier3D, not MuJoCo. Observation shape,
//! action dimensionality, reward structure, and termination conditions match
//! Gymnasium v5 exactly. **Absolute reward values, learned policies, and
//! trained scores will NOT transfer to real Gymnasium/MuJoCo benchmarks
//! without retuning.**
//!
//! ## Layout
//!
//! * Cart: dynamic body restricted to translation along the world-x axis.
//!   Rotations and y/z translations are locked via rapier3d's per-axis DOF
//!   gate, so the body behaves like a single-DOF slider.
//! * Pole: dynamic capsule attached to the cart by a revolute impulse joint
//!   about the world-y axis. Gravity pulls it down; the agent's job is to
//!   balance it upright by sliding the cart.
//! * Action: `Box(-3, 3, (1,))` — force target; applied as `action · gear`
//!   with `gear = [100]` (Gymnasium XML) directly to the cart.
//! * Observation: `[cart_x, pole_angle, cart_vx, pole_angvel_y]` (4-dim).
//! * Reward: `+1.0` per step while the pole is healthy; `0.0` otherwise.
//! * Termination: `|pole_angle| >= 0.2 rad`, or non-finite state.
//! * Truncation: `max_steps = 1000`.

pub mod action;
pub mod config;
pub mod env;
pub mod observation;
pub mod state;

pub use action::InvertedPendulumAction;
pub use config::InvertedPendulumConfig;
pub use env::{InvertedPendulum, InvertedPendulumRapier, METADATA_KEY_ALIVE};
pub use observation::InvertedPendulumObservation;
pub use state::InvertedPendulumState;
