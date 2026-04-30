//! # Swimmer-v5 (Rapier3D-backed)
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
//! Three cylindrical segments chained in series (front → middle → tail),
//! constrained to the world xy-plane with gravity disabled. Two revolute-z
//! impulse joints actuate the chain; per-segment viscous drag substitutes for
//! MuJoCo's native fluid model.
//!
//! * Segment shape: capsule along body-x, length `0.1`, radius `0.05`, mass
//!   `≈0.0471` (density ≈60 kg/m³ derived from capsule volume). The capsule
//!   stands in for Gymnasium's cylinder — drag depends on COM velocity only,
//!   not collider geometry.
//! * Planar constraint: every segment has `enabled_translations(true, true, false)`
//!   and `enabled_rotations(false, false, true)`, so motion is confined to the
//!   xy-plane and rotations to about-z.
//! * Front ↔ Middle: revolute-z impulse joint, anchor `(+0.05, 0, 0)` on
//!   segment0's back, anchor `(−0.05, 0, 0)` on segment1's front.
//! * Middle ↔ Tail:  revolute-z impulse joint, anchor `(+0.05, 0, 0)` on
//!   segment1's back, anchor `(−0.05, 0, 0)` on segment2's front.
//! * Action: `Box(-1, 1, (2,))` — joint torque targets; applied as
//!   `action · gear` with `gear = [150, 150]` (Gymnasium XML).
//! * Observation (8-dim):
//!   `[body_angle, joint1_angle, joint2_angle, vx_com, vy_com,
//!     ω_body, joint1_dot, joint2_dot]` — matches `qpos[2:5]` + `qvel` from
//!   Gymnasium.
//! * Reward: `forward − ctrl` with `forward = 1.0 · vx_com` and
//!   `ctrl = 1e-4 · ‖action‖²`.
//! * Termination: never (`TerminationMode::Never` implicitly; swimmer has no
//!   healthy gate).
//! * Truncation: `max_steps = 1000`.
//!
//! ## Viscous drag
//!
//! Rapier has no native fluid solver. Each segment accrues a drag force
//! `F = −k · v · ‖v‖` before every physics substep, where `k` is
//! `drag_coefficient` (default `0.1`). The env owns its own `frame_skip`
//! loop so drag is applied on every substep, not once per env step; this is
//! required for numerical stability and to match the Gymnasium frame_skip
//! semantics. If/when a second locomotion env needs viscous drag, the pattern
//! can be promoted to a backend-level hook.
//!
//! ## Divergence from Gymnasium
//!
//! Reward structure and observation layout match Gymnasium v5. Physics
//! parameters diverge where Rapier's reduced-coordinate multibody solver
//! cannot integrate the MuJoCo-native XML at the Gymnasium substep.
//! Specifically (see [`SwimmerConfig::default`] for full reasoning):
//!
//! * **Joints are in `MultibodyJointSet`, not `ImpulseJointSet`.** A
//!   free-floating serial chain with no grounded reference is a stiff
//!   problem for the PGS impulse solver; multibody's Featherstone-style
//!   reduced-coordinate integration keeps the chain conservative.
//! * **`gear = [5, 5]`** instead of Gymnasium's `[150, 150]`. At full gear
//!   the joint angular acceleration (τ/I ≈ 7 500 rad/s²) violates the joint
//!   constraint faster than the solver can resolve it.
//! * **Smaller substep:** `dt = 0.005`, `frame_skip = 8` → env dt 0.04 still
//!   matches Gymnasium; the integration step is halved to keep per-step
//!   Δω tractable.
//! * **`segment_mass = 0.947 kg`** from MuJoCo's body density (1000 kg/m³)
//!   applied to the capsule volume; using 0.0471 kg (as Gymnasium-derived
//!   calculations sometimes produce by crossing body density with MuJoCo's
//!   fluid `<option density>`) gives negligible inertia.
//! * **Linear angular drag** `τ = −k_ang · ω`, not quadratic. Explicit
//!   Euler on quadratic drag overshoots past zero at high |ω| and
//!   diverges within a substep; linear drag is unconditionally stable.
//! * **Capsules, not cylinders** (`capsule_x`, not `cylinder`). The drag
//!   model depends on COM velocity only, not collider geometry.
//!
//! Cumulative effect: observation shape, action shape, reward structure and
//! termination conditions match Gymnasium v5 exactly; absolute reward
//! values, peak velocities, and learned policies will not transfer.

pub mod action;
pub mod config;
pub mod env;
pub mod observation;
pub mod state;

pub use action::SwimmerAction;
pub use config::SwimmerConfig;
pub use env::{METADATA_KEY_CTRL, METADATA_KEY_FORWARD, Swimmer, SwimmerRapier};
pub use observation::SwimmerObservation;
pub use state::SwimmerState;
