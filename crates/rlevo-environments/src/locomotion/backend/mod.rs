//! Physics backend abstraction for locomotion environments.
//!
//! Locomotion environments are generic over a [`LocomotionBackend`]. v1 ships
//! only [`Rapier3DBackend`].
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
    /// `[vx, vy, vz]` world-frame linear velocity (metres per second).
    pub linear: [f32; 3],
    /// `[ωx, ωy, ωz]` world-frame angular velocity (radians per second).
    pub angular: [f32; 3],
}

/// Error returned by fallible [`LocomotionBackend`] operations.
///
/// Currently only [`LocomotionBackend::apply_joint_torque`] is fallible: driving
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
    /// locked — are actuatable by [`LocomotionBackend::apply_joint_torque`].
    /// Prismatic (free linear), spherical (three free angular), and fixed
    /// (zero free) joints all fall here. The payload names the offending kind.
    #[error("joint is not actuatable by a scalar torque: {0}")]
    UnsupportedJoint(&'static str),
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

    /// Apply a scalar actuator torque to a revolute joint's single free axis.
    ///
    /// `torque` is the **generalized force** on the joint's one free angular
    /// degree of freedom — the direct analogue of a MuJoCo `motor` actuator on a
    /// hinge joint, where the generalized force is `gear × ctrl`. Torque is in
    /// world-scale units (newton-metres for Rapier3D); **gear scaling stays the
    /// caller's responsibility** — see [`crate::locomotion::common::Gear`].
    ///
    /// # Semantics
    ///
    /// - **Domain — revolute only.** The torque acts on a single free angular
    ///   axis, so the joint must be revolute (one free angular DOF, all
    ///   translation locked). Prismatic, spherical, and fixed joints have no such
    ///   axis and are rejected (see [`# Errors`](#errors)); linear (cart-force)
    ///   actuation is a separate seam that envs still drive via a direct body
    ///   force. See ADR 0041.
    /// - **Sign convention.** A positive `torque` drives `body2` positively about
    ///   the joint's free axis `+â` **relative to** `body1`, where `â` is the
    ///   unit hinge axis mapped into world space by `body1`'s rotation. `body1` /
    ///   `body2` are the first / second bodies of the joint's insertion. (The
    ///   exact extraction of `â` from the joint frame is backend-specific — see
    ///   the implementation.)
    /// - **Lifetime — one physics substep** (ADR 0037). Like `add_force` /
    ///   `add_torque`, the applied torque is integrated once and then cleared, so
    ///   callers that hold an actuator constant across `frame_skip` must
    ///   re-apply it every substep (inside the backend's substep-actuation hook,
    ///   e.g. `Rapier3DWorld::step_actuated`).
    ///
    /// The maximal- vs reduced-coordinate mechanics of *how* the torque reaches
    /// the hinge DOF are backend-specific and documented on the implementation.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::InvalidJointHandle`] if `joint` is stale/unknown or
    /// an attached body is missing, and [`BackendError::UnsupportedJoint`] if the
    /// joint is not revolute (no single free angular axis).
    fn apply_joint_torque(
        world: &mut Self::World,
        joint: Self::JointHandle,
        torque: f32,
    ) -> Result<(), BackendError>;

    /// Aggregate the 6D contact wrench on a body from the **last** physics substep.
    ///
    /// Returns `[fx, fy, fz, tx, ty, tz]` (force first, then torque), summed over
    /// every contact manifold touching any collider attached to `body`, with each
    /// contact's solver impulse divided by the substep `dt`.
    ///
    /// # Semantics
    ///
    /// - **Instantaneous, last-substep.** This is the per-substep *average* force
    ///   (`impulse / substep_dt`) from the **last** solved substep — an
    ///   instantaneous per-timestep quantity, **not** a step-integrated or
    ///   frame-skip-averaged one. It mirrors Gymnasium's read-after-last-substep
    ///   semantics: MuJoCo recomputes `cfrc_ext` via `mj_rnePostConstraint`
    ///   *once*, after `mj_step(nstep = frame_skip)`, so it too reflects only the
    ///   final substep's state. The value is therefore independent of
    ///   `frame_skip`.
    /// - **Sign — force ON the queried body.** The wrench is the external
    ///   contact force-torque acting *on* `body` (the analogue of MuJoCo
    ///   `cfrc_ext` = "external force acting on the body"): a body resting on the
    ///   ground reports a positive upward vertical force. By Newton's third law
    ///   the force part of `contact_force(A)` is the negation of that of
    ///   `contact_force(B)` for a contacting pair (torque parts differ — each is
    ///   taken about its own body's centre of mass). The sign is independent of
    ///   collider insertion order.
    /// - **Layout `[force(3), torque(3)]`** — an intentional deviation from
    ///   MuJoCo's `cfrc_ext`, which is `[torque(3), force(3)]` ("rotation(3),
    ///   translation(3)" per `mj_rnePostConstraint`). Consumers packing an
    ///   observation must map fields accordingly; this layout is **not**
    ///   `cfrc_ext`-compatible.
    /// - **Torque reference point: the body's own centre of mass**, whereas
    ///   MuJoCo references its `cfrc_ext` torque to the *kinematic-subtree* CoM.
    ///   These coincide for a leaf body but differ materially for an internal
    ///   body — a calibration caveat for Ant/Humanoid `contact_cost`.
    /// - **Content: contact-manifold forces only.** MuJoCo's `cfrc_ext` is the
    ///   full Recursive-Newton-Euler post-constraint external wrench (all
    ///   external interactions), a strict superset of contact forces.
    /// - **Self-contacts never contribute.** Two colliders sharing one rigid body
    ///   cannot produce a contact pair in rapier 0.32, matching MuJoCo's
    ///   undisableable same-body geom filter.
    fn contact_force(world: &Self::World, body: Self::BodyHandle) -> [f32; 6];
}
