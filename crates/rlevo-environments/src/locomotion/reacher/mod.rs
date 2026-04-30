//! # Reacher-v5 (Rapier3D-backed)
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
//! A planar 2-link arm in the xy-plane with zero gravity. The shoulder is a
//! fixed anchor at the world origin; the elbow connects `link1` (length 0.1)
//! to `link2` (length 0.11). A small `target` body is placed at a random
//! position in a disk of radius 0.2 at each reset; the agent must reach it
//! with the fingertip.
//!
//! * Shoulder: revolute impulse joint about world-z, root → link1.
//! * Elbow:    revolute impulse joint about world-z, link1 → link2.
//! * Planar constraint: both links have `enabled_translations(true, true, false)`
//!   and `enabled_rotations(false, false, true)`.
//! * Action: `Box(-1, 1, (2,))` — shoulder/elbow torque targets; applied as
//!   `action · gear` with `gear = [200, 200]` (Gymnasium XML).
//! * Observation (10-dim):
//!   `[cos θ₁, cos θ₂, sin θ₁, sin θ₂, target_x, target_y, θ̇₁, θ̇₂,
//!     (finger − target)_x, (finger − target)_y]`.
//!   θ₂ is the **relative** elbow angle (link2 − link1), wrapped to `(-π, π]`.
//! * Reward: `reward_distance + reward_control` with
//!   `reward_distance = −‖finger − target‖` and
//!   `reward_control  = −0.1 · ‖action‖²`; both components ≤ 0.
//! * Termination: never (`TerminationMode::Never`).
//! * Truncation: `max_steps = 50`.
//!
//! ## Fingertip convention
//!
//! Rapier's `capsule_x(half_len, r)` places the capsule symmetric about the
//! body origin, so `link2`'s tip sits at body-local `(+link2_length/2, 0, 0)`.
//! Gymnasium's phrasing `link2_rotation * [link2_length, 0, 0]` is a
//! shorthand — the geometrically correct offset is `link2_length/2`.
//!
//! ## Divergence from Gymnasium-v5 dynamics
//!
//! Gymnasium's reacher XML stabilises the tiny-inertia links via MuJoCo joint
//! `armature` and `damping`, neither of which has a direct Rapier equivalent.
//! With literal gear `[200, 200]` and per-link mass `0.0356`, a **random-policy
//! rollout** drives the arm into highly non-physical velocities and distances;
//! trained / clipped policies stay in a reasonable regime. This is the
//! top-level "Physics note" in concrete form: reward values will not transfer
//! without retuning (e.g. lowering `gear`, raising `link_mass`, or scaling
//! `ctrl_cost_weight`).

pub mod action;
pub mod config;
pub mod env;
pub mod observation;
pub mod state;

pub use action::ReacherAction;
pub use config::ReacherConfig;
pub use env::{METADATA_KEY_REWARD_CONTROL, METADATA_KEY_REWARD_DISTANCE, Reacher, ReacherRapier};
pub use observation::ReacherObservation;
pub use state::ReacherState;
