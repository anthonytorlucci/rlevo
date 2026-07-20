//! Physics state for the [`super::Reacher`] environment.
//!
//! [`ReacherState`] holds the `Rapier3D` handles needed to read and drive the
//! simulation, plus a small amount of data that the environment caches between
//! steps for observation extraction. The `Rapier3DWorld` itself lives on the
//! [`super::Reacher`] struct because it is not `Clone`.

use rapier3d::prelude::{ImpulseJointHandle, RigidBodyHandle};
use rlevo_core::base::State;

use super::observation::ReacherObservation;

/// Physics state for [`super::Reacher`] — body + joint handles plus the cached
/// target position and last observation. The non-`Clone` world lives on the
/// env struct directly.
#[derive(Debug, Clone)]
pub struct ReacherState {
    /// Handle to the link 1 (upper-arm) rigid body. Its orientation encodes
    /// the shoulder angle θ₁; its angular velocity gives θ̇₁.
    pub link1: RigidBodyHandle,
    /// Handle to the link 2 (forearm) rigid body. Its orientation encodes
    /// the world-frame angle `θ_world2`; the relative elbow angle is
    /// θ₂ = `θ_world2` − θ₁. The fingertip lies at body-local `(+link2_length/2, 0, 0)`.
    pub link2: RigidBodyHandle,
    /// Handle to the fixed target body. Its world-frame translation is
    /// `[target_xy[0], target_xy[1], 0.0]`.
    pub target: RigidBodyHandle,
    /// Impulse-joint handle for the shoulder revolute joint (root → link1,
    /// axis = world-z). Torque `τ[0]` is applied here.
    pub shoulder: ImpulseJointHandle,
    /// Impulse-joint handle for the elbow revolute joint (link1 → link2,
    /// axis = world-z). Torque `τ[1]` is applied here.
    pub elbow: ImpulseJointHandle,
    /// Cached target position `[x, y]` in metres, sampled at reset. Kept
    /// here to avoid querying the physics world on every observation
    /// extraction — the target body is fixed and never moves within an
    /// episode.
    pub target_xy: [f32; 2],
    /// Most recent observation extracted after the last `step` (or `reset`).
    /// Read only by [`State::is_valid`] to detect physics divergence; the
    /// observation itself is produced by the env-side
    /// [`Sensor`](rlevo_core::environment::Sensor) (ADR 0047). Set to
    /// `ReacherObservation::default()` (all zeros) before the first
    /// observation is extracted.
    pub last_obs: ReacherObservation,
}

impl State<1> for ReacherState {
    fn shape() -> [usize; 1] {
        [10]
    }

    fn is_valid(&self) -> bool {
        self.last_obs.is_finite()
    }
}
