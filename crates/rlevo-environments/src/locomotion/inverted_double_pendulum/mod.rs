//! # InvertedDoublePendulum-v5 (Rapier3D-backed)
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
//! Cart on a 1D slider plus **two chained poles**, both revolute-y. The agent
//! applies horizontal force to the cart to keep the tip of the upper pole as
//! close as possible to `y_tip = 2` (Gymnasium's convention, which maps to our
//! world-z axis). The pendulum is structurally identical to the shipped
//! [`crate::locomotion::inverted_pendulum`] with a second pole chained on top.
//!
//! * Cart: dynamic body, x-only translation, rotations locked.
//! * Pole1: dynamic capsule, revolute-y joint to cart. Mass from collider density.
//! * Pole2: dynamic capsule, revolute-y joint to pole1's top. Mass from density.
//! * Action: `Box(-1, 1, (1,))` — force target, scaled by `gear = [100]`.
//! * Observation (9-dim):
//!   `[cart_x, sin θ₁, sin θ₂, cos θ₁, cos θ₂, cart_vx, θ̇₁, θ̇₂, F_ext_x]`.
//!   θ₂ is the **relative** elbow angle (pole2 world − pole1 world), wrapped.
//! * Reward:
//!   `alive_bonus − 0.01·x_tip² − (y_tip − 2)² − 1e-3·|ω₁| − 5e-3·|ω₂|`,
//!   with `alive_bonus = 10.0` while healthy and `0` otherwise.
//! * Termination: `y_tip ≤ 1.0`, or non-finite state.
//! * Truncation: `max_steps = 1000`.
//!
//! ## Divergence from Gymnasium
//!
//! * `constraint_force_x` (`obs[8]`) is approximated by reading Rapier's
//!   aggregated contact force on pole2 (`Rapier3DBackend::contact_force`).
//!   MuJoCo's equivalent `cfrc_inv[0]` is a joint reaction force computed in
//!   generalised coordinates. Signs and rough magnitudes follow the same
//!   dynamics; absolute values will differ.
//! * `ω₂` is reported as world-frame angular velocity (not relative to pole1),
//!   matching MuJoCo's `qvel` for the second hinge — i.e. it is the body's
//!   absolute rate, not the rate of the relative joint angle.

pub mod action;
pub mod config;
pub mod env;
pub mod observation;
pub mod state;

pub use action::InvertedDoublePendulumAction;
pub use config::InvertedDoublePendulumConfig;
pub use env::{
    InvertedDoublePendulum, InvertedDoublePendulumRapier, METADATA_KEY_ALIVE,
    METADATA_KEY_DISTANCE, METADATA_KEY_VELOCITY,
};
pub use observation::InvertedDoublePendulumObservation;
pub use state::InvertedDoublePendulumState;
