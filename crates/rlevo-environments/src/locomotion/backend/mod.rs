//! The physics backend of the locomotion environments.
//!
//! The four locomotion environments run on `Rapier3D`. [`Rapier3DBackend`]'s
//! associated functions are the narrow surface they use: step physics, read
//! poses and velocities, drive joint torques, and pull per-body contact
//! forces. Skeleton builders (multibody joint chains, tendon approximations)
//! live beside it on [`Rapier3DWorld`].
//!
//! There is no backend trait: one simulator is supported, and a trait with
//! one implementor is a type (ADR 0090). It returns with a second simulator.

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
    /// `[vx, vy, vz]` world-frame linear velocity (metres per second).
    pub linear: [f32; 3],
    /// `$[\omega_x, \omega_y, \omega_z]$` world-frame angular velocity (radians per second).
    pub angular: [f32; 3],
}

/// Error returned by fallible [`Rapier3DBackend`] operations.
///
/// Currently only [`Rapier3DBackend::apply_joint_torque`] is fallible: driving
/// a joint by a scalar torque is well-defined **only** for a revolute (hinge)
/// joint — a joint with exactly one free angular degree of freedom and no free
/// translation. Other joint kinds (prismatic, spherical, fixed) and stale
/// handles are rejected as errors rather than silently mis-actuated.
///
/// # Examples
///
/// ```
/// use rlevo_environments::locomotion::backend::BackendError;
///
/// let err = BackendError::UnsupportedJoint("prismatic joint has no free angular axis");
/// assert!(err.to_string().contains("prismatic"));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum BackendError {
    /// The supplied joint handle does not refer to a live joint in the world.
    ///
    /// Raised when the handle is stale (the joint was removed), never inserted,
    /// or one of the joint's attached rigid bodies is missing.
    #[error("joint handle does not refer to a live joint (stale/unknown handle or missing body)")]
    InvalidJointHandle,

    /// The joint has no single free angular axis, so a scalar torque has no
    /// well-defined generalized-force interpretation.
    ///
    /// Only revolute (hinge) joints — one free angular DOF, all translation
    /// locked — are actuatable by [`Rapier3DBackend::apply_joint_torque`].
    /// Prismatic (free linear), spherical (three free angular), and fixed
    /// (zero free) joints all fall here. The payload names the offending kind.
    #[error("joint is not actuatable by a scalar torque: {0}")]
    UnsupportedJoint(&'static str),
}
