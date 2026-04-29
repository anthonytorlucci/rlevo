//! Physics backend abstraction for locomotion environments.
//!
//! Locomotion environments are generic over a [`LocomotionBackend`]. v1 ships
//! only [`Rapier3DBackend`]; a future `mujoco-ffi` backend is reserved behind
//! a cargo feature (see [`mujoco_ffi`]).
//!
//! The trait is deliberately narrow: it exposes just enough to step physics,
//! read poses/velocities, drive joint torques, and pull per-body contact
//! forces. Backend-specific skeleton builders (multibody joint chains, tendon
//! approximations, contact force aggregation) live alongside the backend impl
//! rather than on the trait, since their signatures are backend-coupled.

use std::fmt::Debug;

pub mod rapier3d;

pub use rapier3d::{Rapier3DBackend, Rapier3DJointHandle, Rapier3DWorld};

/// Body pose: position and orientation (unit quaternion, scalar-first `(w, x, y, z)`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose {
    /// `[x, y, z]` world-frame position.
    pub position: [f32; 3],
    /// Unit quaternion `[w, x, y, z]`.
    pub orientation: [f32; 4],
}

/// Body twist: linear and angular velocity in the world frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Twist {
    pub linear: [f32; 3],
    pub angular: [f32; 3],
}

/// Minimal physics backend contract shared by all locomotion envs.
///
/// Keeping this narrow (step + read + actuate + contact force) means backends
/// with very different internal structure (maximal vs generalised coordinates,
/// impulse-based vs analytic solver) can be dropped in without contorting the
/// trait. Env-specific skeleton construction lives in backend-local helpers
/// rather than on the trait.
pub trait LocomotionBackend: 'static {
    /// Opaque per-world simulation state (rigid bodies, joints, pipeline, ...).
    type World: Debug;

    /// Handle referring to a rigid body inside [`Self::World`].
    type BodyHandle: Copy + Debug;

    /// Handle referring to an actuated joint inside [`Self::World`].
    type JointHandle: Copy + Debug;

    /// Advance the simulation by one environment step (including any frame-skip
    /// substepping configured on `world`).
    fn step(world: &mut Self::World);

    /// Read a body's pose from the world.
    fn get_pose(world: &Self::World, body: Self::BodyHandle) -> Pose;

    /// Read a body's twist (linear + angular velocity) from the world.
    fn get_vel(world: &Self::World, body: Self::BodyHandle) -> Twist;

    /// Apply a torque to a joint for the next physics substep.
    ///
    /// Torque is in world-scale units (newton-metres for Rapier3D). Gear
    /// scaling is the caller's responsibility — see [`crate::locomotion::common::Gear`].
    fn apply_joint_torque(world: &mut Self::World, joint: Self::JointHandle, torque: f32);

    /// Aggregate 6D contact force-torque on a body, summed over all contact
    /// manifolds touching it. Layout: `[fx, fy, fz, tx, ty, tz]`. Matches
    /// MuJoCo's `cfrc_ext` layout for observation packing.
    fn contact_force(world: &Self::World, body: Self::BodyHandle) -> [f32; 6];
}
