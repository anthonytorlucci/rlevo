//! `Rapier3D` implementation of [`LocomotionBackend`].
//!
//! Wraps the rapier3d simulation pipeline in [`Rapier3DWorld`] and provides a
//! [`Rapier3DJointHandle`] union covering both impulse and multibody joints.
//! Bodies are kept in maximal coordinates; articulated chains (Ant legs,
//! Humanoid torso/arms/legs, `Walker2D`) are driven via `MultibodyJointSet` so
//! they behave closer to `MuJoCo`'s generalised coordinates than plain impulse
//! joints would.
//!
//! Note on math types: rapier3d 0.32's `Vector`, `Rotation`, and `Isometry`
//! come from parry's glam-backed re-exports (parry3d 0.26 wraps glam — its
//! `Vector` is `glam::Vec3`, `Rotation` is `glam::Quat`). This backend is the
//! **seam** that packs those engine math types into plain `[f32; N]` arrays
//! ([`Pose`], [`Twist`], and the `[f32; 6]` contact wrench), so no glam (or
//! nalgebra) type ever crosses into observation/state code.

use rapier3d::prelude::*;

use super::{BackendError, LocomotionBackend, Pose, Twist};

/// `Rapier3D` scene for one locomotion environment.
///
/// Holds every set the rapier3d pipeline needs, plus env-level bookkeeping
/// (`gravity`, `frame_skip`). Fields are `pub(crate)` so env modules can
/// build skeletons directly; external users interact via the
/// [`LocomotionBackend`] trait.
pub struct Rapier3DWorld {
    pub(crate) bodies: RigidBodySet,
    pub(crate) colliders: ColliderSet,
    pub(crate) impulse_joints: ImpulseJointSet,
    pub(crate) multibody_joints: MultibodyJointSet,
    pub(crate) integration_parameters: IntegrationParameters,
    pub(crate) pipeline: PhysicsPipeline,
    pub(crate) broad_phase: BroadPhaseBvh,
    pub(crate) narrow_phase: NarrowPhase,
    pub(crate) islands: IslandManager,
    pub(crate) ccd_solver: CCDSolver,
    pub(crate) gravity: Vector,
    pub(crate) frame_skip: u32,
}

impl std::fmt::Debug for Rapier3DWorld {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Rapier3DWorld")
            .field("num_bodies", &self.bodies.len())
            .field("num_colliders", &self.colliders.len())
            .field("num_impulse_joints", &self.impulse_joints.len())
            .field("gravity", &[self.gravity.x, self.gravity.y, self.gravity.z])
            .field("dt", &self.integration_parameters.dt)
            .field("frame_skip", &self.frame_skip)
            .finish_non_exhaustive()
    }
}

impl Rapier3DWorld {
    /// Create a world with the given gravity vector, physics substep `dt`, and
    /// `frame_skip` substeps per environment step.
    #[must_use]
    pub fn new(gravity: Vector, dt: Real, frame_skip: u32) -> Self {
        let integration_parameters = IntegrationParameters {
            dt,
            ..IntegrationParameters::default()
        };
        Self {
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            integration_parameters,
            pipeline: PhysicsPipeline::new(),
            broad_phase: BroadPhaseBvh::new(),
            narrow_phase: NarrowPhase::new(),
            islands: IslandManager::new(),
            ccd_solver: CCDSolver::new(),
            gravity,
            frame_skip,
        }
    }

    /// Advance the simulation by `frame_skip` physics substeps.
    ///
    /// External-force lifetime: forces applied via `add_force` / `add_torque`
    /// live **exactly one** substep — each [`step_once`](Self::step_once)
    /// integrates the current accumulator and then clears it. A control input
    /// applied once before this call therefore affects only the *first*
    /// substep. To hold an actuator constant across the frame skip (`MuJoCo`
    /// `ctrl`-held-across-substeps semantics), apply control every substep via
    /// [`step_actuated`](Self::step_actuated) instead. See ADR 0037.
    pub fn step_with_frame_skip(&mut self) {
        for _ in 0..self.frame_skip.max(1) {
            self.step_once();
        }
    }

    /// Advance the simulation by a single physics substep (ignores `frame_skip`).
    ///
    /// External-force lifetime: rapier3d 0.32 `add_force` / `add_torque` are
    /// **additive** (`user_force += force`) and the pipeline never clears the
    /// accumulator on its own (the "auto-cleared each step" folklore is false —
    /// only `reset_forces` / `reset_torques` clear it). To make an external
    /// force live exactly one integration step, this method calls
    /// `reset_external_forces` **after** the
    /// pipeline step: forces applied before a `step_once` are integrated once,
    /// then zeroed. Re-apply control before each `step_once` to sustain it.
    /// See ADR 0037.
    pub fn step_once(&mut self) {
        self.pipeline.step(
            self.gravity,
            &self.integration_parameters,
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd_solver,
            &(),
            &(),
        );
        // Enforce the one-integration-step force lifetime (ADR 0037): the
        // accumulator just integrated is cleared so it cannot carry into the
        // next substep/env step and grow monotonically.
        self.reset_external_forces();
    }

    /// Advance `frame_skip` substeps, applying control fresh before EACH substep.
    ///
    /// rapier 0.32 `add_force`/`add_torque` are additive and consumed+cleared by
    /// [`step_once`](Self::step_once); to hold an actuator constant across the
    /// frame skip (`MuJoCo` `ctrl`-held-across-substeps semantics) the caller MUST
    /// re-apply control every substep. `apply` is invoked with `&mut self`
    /// immediately before each `step_once`. See ADR 0037.
    pub fn step_actuated(&mut self, mut apply: impl FnMut(&mut Self)) {
        for _ in 0..self.frame_skip.max(1) {
            apply(self);
            self.step_once();
        }
    }

    /// Zero every rigid body's accumulated external force and torque.
    ///
    /// Called by [`step_once`](Self::step_once) after the pipeline step so an
    /// external force lives exactly one integration step (ADR 0037). Uses
    /// `wake_up = false`: zeroing the accumulator must not wake a sleeping body
    /// (actuated bodies are already awake from their `add_force(.., true)`
    /// calls).
    fn reset_external_forces(&mut self) {
        for (_handle, body) in self.bodies.iter_mut() {
            body.reset_forces(false);
            body.reset_torques(false);
        }
    }

    // ─── Insertion helpers (used by env skeleton builders) ───────────────────

    /// Insert a rigid body built from `desc` and return its handle.
    pub(crate) fn add_body(&mut self, desc: RigidBodyBuilder) -> RigidBodyHandle {
        self.bodies.insert(desc)
    }

    /// Attach a collider to an existing body and return its handle.
    pub(crate) fn add_collider(
        &mut self,
        desc: ColliderBuilder,
        parent: RigidBodyHandle,
    ) -> ColliderHandle {
        self.colliders
            .insert_with_parent(desc, parent, &mut self.bodies)
    }

    /// Insert a free-standing collider (e.g. the ground plane) with no rigid-body parent.
    #[allow(dead_code)] // v0.2: used by locomotion skeleton builders
    pub(crate) fn add_ground_collider(&mut self, desc: ColliderBuilder) -> ColliderHandle {
        self.colliders.insert(desc)
    }

    /// Link two bodies with an impulse-based joint and return the handle.
    ///
    /// Impulse joints are suitable for contact-rich or high-frequency connections
    /// (e.g. wheels, feet). For serial articulated chains prefer
    /// [`add_multibody_joint`](Self::add_multibody_joint).
    pub(crate) fn add_impulse_joint(
        &mut self,
        b1: RigidBodyHandle,
        b2: RigidBodyHandle,
        joint: impl Into<GenericJoint>,
    ) -> ImpulseJointHandle {
        self.impulse_joints.insert(b1, b2, joint, true)
    }

    /// Link two bodies with a multibody (generalised-coordinate) joint.
    ///
    /// Multibody joints behave closer to `MuJoCo`'s generalised coordinates than
    /// impulse joints and are preferred for articulated limb chains.  Returns
    /// `None` if rapier3d rejects the insertion (e.g. cyclic topology).
    pub(crate) fn add_multibody_joint(
        &mut self,
        b1: RigidBodyHandle,
        b2: RigidBodyHandle,
        joint: impl Into<GenericJoint>,
    ) -> Option<MultibodyJointHandle> {
        self.multibody_joints.insert(b1, b2, joint, true)
    }

    /// Read-only access to the world's rigid-body set.
    #[allow(dead_code)] // v0.2: used by locomotion skeleton builders
    pub(crate) fn bodies(&self) -> &RigidBodySet {
        &self.bodies
    }

    /// Mutable access to the world's rigid-body set (used by per-env skeleton
    /// builders to set initial velocities and apply forces directly).
    pub(crate) fn bodies_mut(&mut self) -> &mut RigidBodySet {
        &mut self.bodies
    }
}

/// Tagged handle covering both rapier joint kinds.
///
/// `Rapier3D` has two disjoint joint stores: [`ImpulseJointSet`] (good for
/// high-frequency / contact-rich joints like wheels and feet) and
/// [`MultibodyJointSet`] (analogous to `MuJoCo`'s generalised coordinates for
/// serial chains). Envs tag each actuated joint so torque application
/// dispatches to the right store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rapier3DJointHandle {
    Impulse(ImpulseJointHandle),
    Multibody(MultibodyJointHandle),
}

impl From<ImpulseJointHandle> for Rapier3DJointHandle {
    fn from(h: ImpulseJointHandle) -> Self {
        Self::Impulse(h)
    }
}

impl From<MultibodyJointHandle> for Rapier3DJointHandle {
    fn from(h: MultibodyJointHandle) -> Self {
        Self::Multibody(h)
    }
}

/// Backend marker type; implements [`LocomotionBackend`] with `Rapier3D` semantics.
#[derive(Debug, Clone, Copy)]
pub struct Rapier3DBackend;

/// True iff `locked` describes a **revolute** (hinge) joint: a single free
/// angular degree of freedom and no free translation.
///
/// A scalar torque is the generalized force on exactly one hinge axis, so it is
/// only well-defined here. This accepts rapier's `LOCKED_REVOLUTE_AXES` (all
/// three linear axes plus two of three angular axes locked, leaving one angular
/// free) and rejects prismatic (free linear), spherical (three free angular),
/// and fixed (zero free) joints.
fn is_revolute_axes(locked: JointAxesMask) -> bool {
    // No free translation …
    let all_lin_locked = locked.contains(JointAxesMask::LIN_AXES);
    // … and exactly one free angular axis.
    let free_ang = JointAxesMask::ANG_AXES.difference(locked);
    all_lin_locked && free_ang.bits().is_power_of_two()
}

impl LocomotionBackend for Rapier3DBackend {
    type World = Rapier3DWorld;
    type BodyHandle = RigidBodyHandle;
    type JointHandle = Rapier3DJointHandle;

    fn step(world: &mut Self::World) {
        world.step_with_frame_skip();
    }

    fn get_pose(world: &Self::World, body: Self::BodyHandle) -> Pose {
        world.bodies.get(body).map_or(
            Pose {
                position: [0.0; 3],
                orientation: [1.0, 0.0, 0.0, 0.0],
            },
            |b| {
                let t = b.translation();
                let q = b.rotation();
                Pose {
                    position: [t.x, t.y, t.z],
                    // Pose stores quaternion as (w, x, y, z) regardless of glam's storage order.
                    orientation: [q.w, q.x, q.y, q.z],
                }
            },
        )
    }

    fn get_vel(world: &Self::World, body: Self::BodyHandle) -> Twist {
        world.bodies.get(body).map_or(
            Twist {
                linear: [0.0; 3],
                angular: [0.0; 3],
            },
            |b| {
                let v = b.linvel();
                let w = b.angvel();
                Twist {
                    linear: [v.x, v.y, v.z],
                    angular: [w.x, w.y, w.z],
                }
            },
        )
    }

    /// Drive a revolute joint's free axis by a scalar torque (`MuJoCo` `motor`
    /// analogue). Dispatches on the [`Rapier3DJointHandle`] kind — the maximal-
    /// vs reduced-coordinate mechanics differ and this split is load-bearing.
    ///
    /// - **`Impulse` (maximal coordinates).** Both bodies are free; the hinge is
    ///   enforced by the solver. Apply **equal-and-opposite** world-axis torques:
    ///   `+τ·â` to `body2`, `−τ·â` to `body1`. The pair injects zero *net*
    ///   external torque, so the generalized force lands on the hinge DOF alone.
    /// - **`Multibody` (reduced coordinates).** The hinge is baked into the
    ///   parameterization, so torque the **child only** (`body2` of the
    ///   parent→child insertion): `+τ·â` on the child projects onto the hinge DOF
    ///   and the solver supplies the parent reaction. Also torquing the parent
    ///   (as the impulse path does) would inject spurious generalized force on
    ///   upstream DOFs — the existing swimmer/env.rs child-only convention.
    ///
    /// `â` is the joint's free hinge axis in world space: `body1`'s world rotation
    /// applied to `local_frame1.rotation · X` (the unit hinge axis in `body1`'s
    /// local frame). `GenericJoint::local_axis1()` is deliberately **avoided** —
    /// under rapier 0.32's glam backend `Pose * Vector` is `transform_point`, so
    /// `local_axis1()` returns `axis + anchor` (non-unit, contaminated). A positive
    /// `τ` drives `body2` positively about `+â`. The torque lives one substep (ADR 0037):
    /// callers hold it across `frame_skip` by re-applying inside
    /// [`Rapier3DWorld::step_actuated`]. See ADR 0041.
    ///
    /// # Errors
    ///
    /// - [`BackendError::InvalidJointHandle`] — the handle is stale/unknown, or a
    ///   joint body (or the multibody parent link) cannot be resolved.
    /// - [`BackendError::UnsupportedJoint`] — the joint is not revolute (it lacks
    ///   a single free angular axis), e.g. prismatic, spherical, or fixed.
    fn apply_joint_torque(
        world: &mut Self::World,
        joint: Self::JointHandle,
        torque: f32,
    ) -> Result<(), BackendError> {
        match joint {
            Rapier3DJointHandle::Impulse(handle) => {
                // Resolve the joint, validate revolute, and compute the world
                // hinge axis under one immutable borrow of disjoint world fields.
                let (body1, body2, axis) = {
                    let joint = world
                        .impulse_joints
                        .get(handle)
                        .ok_or(BackendError::InvalidJointHandle)?;
                    if !is_revolute_axes(joint.data.locked_axes) {
                        return Err(BackendError::UnsupportedJoint(
                            "impulse joint has no single free angular axis (expected revolute)",
                        ));
                    }
                    let body1 = joint.body1;
                    let body2 = joint.body2;
                    // Hinge axis in body1's local frame = local_frame1's rotation
                    // applied to the joint's principal (X) axis. NOTE: do NOT use
                    // `GenericJoint::local_axis1()` here — under rapier 0.32's glam
                    // backend `Pose * Vector` is `transform_point` (adds the frame
                    // translation), so `local_axis1()` returns `axis + anchor`, a
                    // non-unit contaminated vector. Extracting from the rotation
                    // only yields the correct unit axis. See ADR 0041.
                    let local_axis = joint.data.local_frame1.rotation * Vector::X;
                    let rot1 = *world
                        .bodies
                        .get(body1)
                        .ok_or(BackendError::InvalidJointHandle)?
                        .rotation();
                    // â: world-frame hinge axis (unit) from body1's rotation.
                    let axis = rot1 * local_axis;
                    (body1, body2, axis)
                };
                // Equal-and-opposite: in maximal coordinates both bodies are free,
                // so the solver — not this torque pair — enforces the hinge.
                world
                    .bodies
                    .get_mut(body2)
                    .ok_or(BackendError::InvalidJointHandle)?
                    .add_torque(axis * torque, true);
                world
                    .bodies
                    .get_mut(body1)
                    .ok_or(BackendError::InvalidJointHandle)?
                    .add_torque(axis * -torque, true);
                Ok(())
            }
            Rapier3DJointHandle::Multibody(handle) => {
                // Resolve child + parent bodies and the local hinge axis under one
                // immutable borrow of the multibody set.
                let (parent, child, local_axis) = {
                    let (multibody, link_id) = world
                        .multibody_joints
                        .get(handle)
                        .ok_or(BackendError::InvalidJointHandle)?;
                    let link = multibody
                        .link(link_id)
                        .ok_or(BackendError::InvalidJointHandle)?;
                    if !is_revolute_axes(link.joint().data.locked_axes) {
                        return Err(BackendError::UnsupportedJoint(
                            "multibody joint has no single free angular axis (expected revolute)",
                        ));
                    }
                    // Insertion invariant (add_multibody_joint callers): joints go
                    // parent→child (b1 = parent, b2 = child). The handle indexes
                    // the CHILD link; its parent link gives body1.
                    let child = link.rigid_body_handle();
                    let parent_id = link.parent_id().ok_or(BackendError::InvalidJointHandle)?;
                    let parent = multibody
                        .link(parent_id)
                        .ok_or(BackendError::InvalidJointHandle)?
                        .rigid_body_handle();
                    // Hinge axis from the rotation only (see the impulse branch:
                    // `local_axis1()` is anchor-contaminated under the glam backend).
                    let local_axis = link.joint().data.local_frame1.rotation * Vector::X;
                    (parent, child, local_axis)
                };
                // â from the PARENT (body1) rotation, per the sign convention.
                let rot_parent = *world
                    .bodies
                    .get(parent)
                    .ok_or(BackendError::InvalidJointHandle)?
                    .rotation();
                let axis = rot_parent * local_axis;
                // Reduced coordinates: torque the child only; the solver supplies
                // the parent reaction through the hinge parameterization.
                world
                    .bodies
                    .get_mut(child)
                    .ok_or(BackendError::InvalidJointHandle)?
                    .add_torque(axis * torque, true);
                Ok(())
            }
        }
    }

    fn contact_force(world: &Self::World, body: Self::BodyHandle) -> [f32; 6] {
        // Instantaneous last-substep contact wrench (ADR 0041): aggregate the
        // LAST solve's per-contact impulses over every manifold touching any
        // collider of `body`, dividing each by the substep dt to get the average
        // force over that substep — the analogue of MuJoCo's post-`mj_step`
        // `cfrc_ext` read (instantaneous, frame-skip-independent). Torque is r × F
        // about the body's OWN centre of mass (MuJoCo references the subtree CoM
        // — identical for leaf bodies, different for internal ones). This sums
        // contact-manifold forces only, not the full RNE external wrench.
        //
        // Sign: the returned wrench is the external contact force-torque acting
        // ON the queried body (the analogue of MuJoCo cfrc_ext = "external force
        // acting on the body"). A ball resting on the ground therefore reports a
        // positive (upward) vertical force. By Newton's third law the force part
        // of `contact_force(A)` ≈ −that of `contact_force(B)` for a contacting
        // pair (torque parts differ: each is taken about its own body's CoM).
        //
        // Self-contacts cannot appear here: rapier 0.32 unconditionally clears
        // same-parent contact pairs (rapier3d-0.32.0 narrow_phase.rs:841,
        // "Same parents. Ignore collisions."), matching MuJoCo's undisableable
        // same-body geom filter, so two colliders on `body` never contribute.
        let Some(rb) = world.bodies.get(body) else {
            return [0.0; 6];
        };
        let com = rb.center_of_mass();
        let dt = world.integration_parameters.dt.max(f32::EPSILON);

        let mut wrench = [0.0f32; 6];
        for &collider_handle in rb.colliders() {
            for pair in world.narrow_phase.contact_pairs_with(collider_handle) {
                let flipped = pair.collider2 == collider_handle;
                for manifold in &pair.manifolds {
                    // Force ON the queried body. parry's manifold normal points
                    // from collider1 toward collider2 (parry3d-0.26.1
                    // query/contact_manifolds/contact_manifold.rs:449 — "points
                    // from the first shape toward the second shape"). rapier's
                    // solver drives the non-negative `contact.data.impulse`
                    // (contact_pair.rs:34; written back at
                    // solver/contact_constraint/contact_with_coulomb_friction.rs:491)
                    // along `dir1 = -normal` on collider1's body and `+normal`
                    // on collider2's body (dir1 = -normal at
                    // contact_with_coulomb_friction.rs:83, applied to the two
                    // bodies at contact_constraint_element.rs:282/285). So the
                    // contact force exerted ON the queried body is
                    // `-force_mag·normal` when it owns collider1 and
                    // `+force_mag·normal` when it owns collider2. Swapping which
                    // collider is collider1 flips BOTH `normal` and `flipped`,
                    // so the attributed force is insertion-order invariant.
                    let n = if flipped {
                        manifold.data.normal
                    } else {
                        -manifold.data.normal
                    };
                    for contact in &manifold.points {
                        let impulse = contact.data.impulse;
                        if impulse.abs() < f32::EPSILON {
                            continue;
                        }
                        let force_mag = impulse / dt;
                        let force = n * force_mag;
                        let local_p = if flipped {
                            contact.local_p2
                        } else {
                            contact.local_p1
                        };
                        let collider = if flipped {
                            pair.collider2
                        } else {
                            pair.collider1
                        };
                        let contact_world = world
                            .colliders
                            .get(collider)
                            .map_or(local_p, |c| *c.position() * local_p);
                        let r = contact_world - com;
                        let torque = r.cross(force);
                        wrench[0] += force.x;
                        wrench[1] += force.y;
                        wrench[2] += force.z;
                        wrench[3] += torque.x;
                        wrench[4] += torque.y;
                        wrench[5] += torque.z;
                    }
                }
            }
        }
        wrench
    }
}

#[cfg(test)]
mod tests {
    // Exact comparison is intentional throughout this test module: the values
    // are literals or seeds read back without arithmetic, or two identically
    // seeded runs that must agree bit-for-bit. A tolerance would let a real
    // regression pass. Reviewed as a class, not site-by-site.
    #![allow(clippy::float_cmp)]

    use super::*;

    fn make_world() -> Rapier3DWorld {
        Rapier3DWorld::new(Vector::new(0.0, 0.0, -9.81), 1.0 / 60.0, 1)
    }

    #[test]
    fn body_falls_under_gravity() {
        let mut world = make_world();
        let handle = world.add_body(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(0.0, 0.0, 10.0))
                .additional_mass(1.0),
        );
        let z0 = world.bodies.get(handle).unwrap().translation().z;
        for _ in 0..30 {
            world.step_once();
        }
        let z1 = world.bodies.get(handle).unwrap().translation().z;
        assert!(z1 < z0, "body should fall under gravity (z0={z0}, z1={z1})");
    }

    #[test]
    fn get_pose_reads_translation_and_rotation() {
        let mut world = make_world();
        let handle =
            world.add_body(RigidBodyBuilder::fixed().translation(Vector::new(1.0, 2.0, 3.0)));
        let pose = Rapier3DBackend::get_pose(&world, handle);
        assert_eq!(pose.position, [1.0, 2.0, 3.0]);
        // Identity orientation: w=1, x=y=z=0.
        assert!((pose.orientation[0] - 1.0).abs() < 1e-6);
        assert!(pose.orientation[1].abs() < 1e-6);
        assert!(pose.orientation[2].abs() < 1e-6);
        assert!(pose.orientation[3].abs() < 1e-6);
    }

    #[test]
    fn get_vel_reads_linvel_and_angvel() {
        let mut world = make_world();
        let handle = world.add_body(
            RigidBodyBuilder::dynamic()
                .linvel(Vector::new(1.0, -2.0, 0.5))
                .angvel(Vector::new(0.0, 0.1, -0.2)),
        );
        let twist = Rapier3DBackend::get_vel(&world, handle);
        assert_eq!(twist.linear, [1.0, -2.0, 0.5]);
        assert_eq!(twist.angular, [0.0, 0.1, -0.2]);
    }

    #[test]
    fn frame_skip_steps_multiple_times() {
        let mut world = Rapier3DWorld::new(Vector::new(0.0, 0.0, -9.81), 1.0 / 60.0, 5);
        let handle = world.add_body(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(0.0, 0.0, 10.0))
                .additional_mass(1.0),
        );
        let z0 = world.bodies.get(handle).unwrap().translation().z;
        world.step_with_frame_skip();
        let z1 = world.bodies.get(handle).unwrap().translation().z;
        // 5 substeps at dt=1/60 under 9.81 gravity: Δz ≈ -0.5·g·(5·dt)² ≈ -0.034m.
        assert!(
            z1 < z0 - 0.03,
            "frame_skip=5 should drop body noticeably (Δ={})",
            z0 - z1
        );
    }

    #[test]
    fn contact_force_is_zero_for_non_contacting_body() {
        let mut world = make_world();
        let handle =
            world.add_body(RigidBodyBuilder::dynamic().translation(Vector::new(0.0, 0.0, 10.0)));
        world.add_collider(ColliderBuilder::ball(0.5), handle);
        let f = Rapier3DBackend::contact_force(&world, handle);
        assert_eq!(f, [0.0; 6]);
    }

    #[test]
    fn debug_format_does_not_panic() {
        let world = make_world();
        let s = format!("{world:?}");
        assert!(s.contains("Rapier3DWorld"));
    }

    /// A constant force re-applied before every substep must produce a
    /// **stationary** per-step velocity increment. With the pre-ADR-0037 bug
    /// (`user_force` never cleared) the accumulator grows linearly, so Δv grows
    /// linearly too. Zero gravity + a single free body isolates F = m·a.
    #[test]
    fn constant_actuation_gives_stationary_delta_v() {
        let mut world = Rapier3DWorld::new(Vector::new(0.0, 0.0, 0.0), 1.0 / 60.0, 1);
        let handle = world.add_body(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(0.0, 0.0, 0.0))
                .additional_mass(2.0),
        );

        let mut prev_vx = 0.0f32;
        let mut deltas: Vec<f32> = Vec::new();
        for _ in 0..60 {
            world.step_actuated(|w| {
                if let Some(b) = w.bodies.get_mut(handle) {
                    b.add_force(Vector::new(3.0, 0.0, 0.0), true);
                }
            });
            let vx = world.bodies.get(handle).unwrap().linvel().x;
            deltas.push(vx - prev_vx);
            prev_vx = vx;
            assert!(vx.is_finite(), "velocity must stay finite (vx={vx})");
        }

        // Compare the first and last increments: constant force ⇒ constant Δv.
        let first = deltas[1]; // skip step 0 (solver warm-up)
        let last = *deltas.last().unwrap();
        assert!(first > 0.0, "force should accelerate the body (Δv={first})");
        assert!(
            (last - first).abs() < 1e-4,
            "Δv must be stationary under constant force: first={first}, last={last}"
        );
    }

    /// A one-shot force applied before a single `step_once` must not persist:
    /// a following `step_once` with no force applied must leave velocity
    /// essentially unchanged (Δv ≈ 0). With the pre-ADR-0037 bug the leftover
    /// accumulator would keep accelerating the body.
    #[test]
    fn one_shot_force_does_not_persist() {
        let mut world = Rapier3DWorld::new(Vector::new(0.0, 0.0, 0.0), 1.0 / 60.0, 1);
        let handle = world.add_body(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(0.0, 0.0, 0.0))
                .additional_mass(1.0),
        );

        if let Some(b) = world.bodies.get_mut(handle) {
            b.add_force(Vector::new(5.0, 0.0, 0.0), true);
        }
        world.step_once();
        let vx_after_push = world.bodies.get(handle).unwrap().linvel().x;
        assert!(vx_after_push > 0.0, "first step must impart velocity");

        // Second step, no force re-applied: velocity must not change.
        world.step_once();
        let vx_after_coast = world.bodies.get(handle).unwrap().linvel().x;
        assert!(
            (vx_after_coast - vx_after_push).abs() < 1e-5,
            "force must not persist: pushed={vx_after_push}, coasted={vx_after_coast}"
        );
    }

    // ── apply_joint_torque ───────────────────────────────────────────────────

    /// Build a zero-gravity hinge about +Z anchored at the world origin, with
    /// both body COMs on the Z axis so rotation about it induces no translation.
    /// `base_fixed` selects a fixed vs dynamic first body. Returns the world, the
    /// tagged joint handle, and the (body1, body2) handles.
    fn hinge_pair_impulse(
        base_fixed: bool,
    ) -> (
        Rapier3DWorld,
        Rapier3DJointHandle,
        RigidBodyHandle,
        RigidBodyHandle,
    ) {
        let mut world = Rapier3DWorld::new(Vector::ZERO, 1.0 / 60.0, 1);
        let b1_builder = if base_fixed {
            RigidBodyBuilder::fixed()
        } else {
            RigidBodyBuilder::dynamic()
        };
        let body1 = world.add_body(b1_builder.translation(Vector::new(0.0, 0.0, -0.5)));
        if !base_fixed {
            world.add_collider(ColliderBuilder::ball(0.3), body1);
        }
        let body2 =
            world.add_body(RigidBodyBuilder::dynamic().translation(Vector::new(0.0, 0.0, 0.5)));
        world.add_collider(ColliderBuilder::ball(0.3), body2);
        // Revolute hinge about +Z; anchors meet at the origin (both on the axis).
        let joint = RevoluteJointBuilder::new(Vector::Z)
            .local_anchor1(Vector::new(0.0, 0.0, 0.5))
            .local_anchor2(Vector::new(0.0, 0.0, -0.5))
            .build();
        let handle: Rapier3DJointHandle = world.add_impulse_joint(body1, body2, joint).into();
        (world, handle, body1, body2)
    }

    /// Sign pin (impulse): a positive torque about +â drives body2's angular
    /// velocity about +Z positive. Fixed base isolates body2's response.
    #[test]
    fn apply_joint_torque_impulse_positive_spins_body2_positive() {
        let (mut world, handle, _base, arm) = hinge_pair_impulse(true);
        for _ in 0..30 {
            world.step_actuated(|w| {
                Rapier3DBackend::apply_joint_torque(w, handle, 0.5).expect("revolute joint");
            });
        }
        let wz = world.bodies.get(arm).unwrap().angvel().z;
        assert!(
            wz > 0.0,
            "positive torque must spin body2 positively about +Z (ωz={wz})"
        );
    }

    /// Equal-and-opposite (impulse): two identical free bodies, zero gravity.
    /// The `±τ·â` pair injects zero NET external torque, so with symmetric
    /// inertia the bodies counter-rotate and their z-angular-velocities cancel —
    /// i.e. the system's total angular momentum stays ≈ 0.
    #[test]
    fn apply_joint_torque_impulse_is_equal_and_opposite() {
        let (mut world, handle, body1, body2) = hinge_pair_impulse(false);
        for _ in 0..30 {
            world.step_actuated(|w| {
                Rapier3DBackend::apply_joint_torque(w, handle, 0.1).expect("revolute joint");
            });
        }
        let w1 = world.bodies.get(body1).unwrap().angvel().z;
        let w2 = world.bodies.get(body2).unwrap().angvel().z;
        assert!(w2 > 0.0, "body2 must spin positively (ω2z={w2})");
        assert!(w1 < 0.0, "body1 must counter-rotate (ω1z={w1})");
        // Identical inertia ⇒ ω1z ≈ −ω2z ⇒ no net external torque on the system.
        assert!(
            (w1 + w2).abs() < 0.01 * w2.abs(),
            "equal-and-opposite ⇒ ωz cancels (ω1z={w1}, ω2z={w2})"
        );
    }

    /// Build a zero-gravity two-body multibody chain about +Z (same on-axis
    /// geometry as the impulse hinge). Returns the world, the tagged handle, and
    /// the (parent, child) handles.
    fn hinge_chain_multibody() -> (
        Rapier3DWorld,
        Rapier3DJointHandle,
        RigidBodyHandle,
        RigidBodyHandle,
    ) {
        let mut world = Rapier3DWorld::new(Vector::ZERO, 1.0 / 60.0, 1);
        let parent =
            world.add_body(RigidBodyBuilder::dynamic().translation(Vector::new(0.0, 0.0, -0.5)));
        world.add_collider(ColliderBuilder::ball(0.3), parent);
        let child =
            world.add_body(RigidBodyBuilder::dynamic().translation(Vector::new(0.0, 0.0, 0.5)));
        world.add_collider(ColliderBuilder::ball(0.3), child);
        let joint = RevoluteJointBuilder::new(Vector::Z)
            .local_anchor1(Vector::new(0.0, 0.0, 0.5))
            .local_anchor2(Vector::new(0.0, 0.0, -0.5))
            .build();
        let handle: Rapier3DJointHandle = world
            .add_multibody_joint(parent, child, joint)
            .expect("insert multibody joint")
            .into();
        (world, handle, parent, child)
    }

    /// Multibody child-only: a positive torque spins the child about +Z relative
    /// to its parent. The parent stays put here *not* because the root is a fixed
    /// base (rapier's multibody root DOF count follows the actual body type — a
    /// dynamic parent is a free 6-DOF root), but because this test's geometry is
    /// degenerate on-axis: the hinge axis, both anchors, and both bodies' centers
    /// of mass are colinear, so a pure spin torque about that axis transmits zero
    /// joint reaction to the parent regardless of parent mass. The whole hinge
    /// motion therefore shows up as the child's angular velocity about the free
    /// axis — the torque landed on the hinge DOF.
    #[test]
    fn apply_joint_torque_multibody_child_only_spins_child_positive() {
        let (mut world, handle, parent, child) = hinge_chain_multibody();
        for _ in 0..30 {
            world.step_actuated(|w| {
                Rapier3DBackend::apply_joint_torque(w, handle, 0.1).expect("revolute joint");
            });
        }
        let wc = world.bodies.get(child).unwrap().angvel().z;
        let wp = world.bodies.get(parent).unwrap().angvel().z;
        assert!(wc > 0.0, "child must spin positively about +Z (ωc={wc})");
        // On-axis geometry (not a fixed base): with the hinge axis, anchors, and
        // both centers of mass colinear, a pure spin torque transmits zero joint
        // reaction to the parent, so it does not rotate; the torque projects
        // wholly onto the child's hinge DOF (not the parent's upstream DOFs).
        assert!(wp.abs() < 1e-6, "on-axis parent must not rotate (ωp={wp})");
    }

    /// Multibody dispatch equals a manual child-only `add_torque` about the world
    /// hinge axis. Here the parent only ever rotates about +Z, so `â` stays +Z
    /// exactly and the two trajectories must coincide — pinning the multibody arm
    /// to "child body only", not the impulse equal-and-opposite pair.
    #[test]
    fn apply_joint_torque_multibody_matches_manual_child_torque() {
        let tau = 0.1f32;
        let (mut w_trait, h_trait, _p_t, child_trait) = hinge_chain_multibody();
        let (mut w_manual, _h_m, _p_m, child_manual) = hinge_chain_multibody();
        for _ in 0..40 {
            w_trait.step_actuated(|w| {
                Rapier3DBackend::apply_joint_torque(w, h_trait, tau).expect("revolute joint");
            });
            w_manual.step_actuated(|w| {
                if let Some(b) = w.bodies_mut().get_mut(child_manual) {
                    b.add_torque(Vector::new(0.0, 0.0, tau), true);
                }
            });
        }
        let wt = w_trait.bodies.get(child_trait).unwrap().angvel().z;
        let wm = w_manual.bodies.get(child_manual).unwrap().angvel().z;
        assert!(
            (wt - wm).abs() < 1e-5,
            "trait multibody torque must equal manual child-only add_torque (trait={wt}, manual={wm})"
        );
    }

    /// Error path: a stale/default handle of either kind yields `InvalidJointHandle`.
    #[test]
    fn apply_joint_torque_stale_handle_errors() {
        let mut world = make_world();
        let impulse: Rapier3DJointHandle = ImpulseJointHandle::invalid().into();
        assert_eq!(
            Rapier3DBackend::apply_joint_torque(&mut world, impulse, 1.0),
            Err(BackendError::InvalidJointHandle),
            "stale impulse handle must error"
        );
        let multibody: Rapier3DJointHandle = MultibodyJointHandle::invalid().into();
        assert_eq!(
            Rapier3DBackend::apply_joint_torque(&mut world, multibody, 1.0),
            Err(BackendError::InvalidJointHandle),
            "stale multibody handle must error"
        );
    }

    /// Error path: a non-revolute (fixed) joint has no free angular axis, so a
    /// scalar torque is rejected with `UnsupportedJoint`.
    #[test]
    fn apply_joint_torque_fixed_joint_unsupported() {
        let mut world = Rapier3DWorld::new(Vector::ZERO, 1.0 / 60.0, 1);
        let b1 =
            world.add_body(RigidBodyBuilder::dynamic().translation(Vector::new(0.0, 0.0, 0.0)));
        let b2 =
            world.add_body(RigidBodyBuilder::dynamic().translation(Vector::new(1.0, 0.0, 0.0)));
        let joint = FixedJointBuilder::new().build();
        let handle: Rapier3DJointHandle = world.add_impulse_joint(b1, b2, joint).into();
        let result = Rapier3DBackend::apply_joint_torque(&mut world, handle, 1.0);
        assert!(
            matches!(result, Err(BackendError::UnsupportedJoint(_))),
            "fixed joint must be rejected as UnsupportedJoint (got {result:?})"
        );
    }

    // ── contact_force ────────────────────────────────────────────────────────

    /// Insert a large fixed ground collider (top face at z = 0) and a dynamic
    /// ball of exactly `mass` (collider density 0) resting just above it, then
    /// settle. Returns the world and the ball handle.
    fn resting_ball_world(
        frame_skip: u32,
        gravity: f32,
        mass: f32,
    ) -> (Rapier3DWorld, RigidBodyHandle) {
        let radius = 0.25f32;
        let mut world =
            Rapier3DWorld::new(Vector::new(0.0, 0.0, -gravity), 1.0 / 240.0, frame_skip);
        world.add_ground_collider(
            ColliderBuilder::cuboid(10.0, 10.0, 0.5).translation(Vector::new(0.0, 0.0, -0.5)),
        );
        let ball = world.add_body(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(0.0, 0.0, radius + 0.001))
                .additional_mass(mass),
        );
        world.add_collider(ColliderBuilder::ball(radius).density(0.0), ball);
        // Settle: ~600 total substeps regardless of frame_skip.
        for _ in 0..(600 / frame_skip.max(1)) {
            world.step_with_frame_skip();
        }
        (world, ball)
    }

    /// Magnitude + sign pin: a resting ball's vertical contact wrench balances its
    /// weight (`|wrench[2]| ≈ m·g`, an order-of-magnitude larger than any
    /// `1/frame_skip`-scaled value), with lateral force and all torque components
    /// ≈ 0 for an on-axis contact.
    ///
    /// SIGN: the wrench is the external contact force acting ON the ball, so the
    /// ground below pushes UP and `wrench[2]` is **positive** (≈ +m·g). This is
    /// the physically correct convention (`MuJoCo` `cfrc_ext` = "external force
    /// acting on the body"); it is verified insertion-order invariant and
    /// Newton's-third-law antisymmetric by the companion tests below. A slight
    /// `> m·g` magnitude is rapier's steady-state penetration bias.
    #[test]
    fn contact_force_resting_ball_balances_gravity() {
        let g = 9.81f32;
        let m = 1.0f32;
        let (world, ball) = resting_ball_world(1, g, m);
        let wrench = Rapier3DBackend::contact_force(&world, ball);
        let fz = wrench[2];
        assert!(
            fz > 0.0,
            "resting-ball wrench is the force ON the ball: ground pushes UP (fz={fz})"
        );
        assert!(
            (fz.abs() - m * g).abs() < 0.4 * m * g,
            "vertical wrench magnitude must be ≈ m·g={} (|fz|={}), not 1/frame_skip-scaled",
            m * g,
            fz.abs()
        );
        assert!(
            wrench[0].abs() < 0.2 && wrench[1].abs() < 0.2,
            "no lateral contact force (fx={}, fy={})",
            wrench[0],
            wrench[1]
        );
        for (axis, &t) in wrench[3..6].iter().enumerate() {
            assert!(
                t.abs() < 0.5,
                "torque component {axis} must be ≈0 for an on-axis contact (t={t})"
            );
        }
    }

    /// Frame-skip invariance: the steady-state normal force is the same for
    /// `frame_skip = 1` and `frame_skip = 5` — the wrench is an instantaneous
    /// last-substep force, NOT a `1/frame_skip`-scaled quantity.
    #[test]
    fn contact_force_is_frame_skip_invariant() {
        let g = 9.81f32;
        let (w1, b1) = resting_ball_world(1, g, 1.0);
        let (w5, b5) = resting_ball_world(5, g, 1.0);
        let fz1 = Rapier3DBackend::contact_force(&w1, b1)[2];
        let fz5 = Rapier3DBackend::contact_force(&w5, b5)[2];
        assert!(
            (fz1 - fz5).abs() < 0.5,
            "steady-state vertical wrench must be frame-skip invariant (fs1={fz1}, fs5={fz5})"
        );
        assert!(
            fz5.abs() > 5.0,
            "fs5 wrench must be the physical m·g magnitude, not a ~1/5-scaled value (fs5={fz5})"
        );
    }

    /// Same-body invariant: two overlapping colliders on ONE body, zero gravity,
    /// no other body — rapier clears same-parent contact pairs, so the wrench is
    /// exactly zero (pins rapier 0.32 `narrow_phase.rs:841`).
    #[test]
    fn contact_force_same_body_colliders_zero_wrench() {
        let mut world = Rapier3DWorld::new(Vector::ZERO, 1.0 / 60.0, 1);
        let body =
            world.add_body(RigidBodyBuilder::dynamic().translation(Vector::new(0.0, 0.0, 0.0)));
        world.add_collider(ColliderBuilder::ball(0.5), body);
        // Second collider overlaps the first (centres 0.2 apart, radii 0.5).
        world.add_collider(
            ColliderBuilder::ball(0.5).translation(Vector::new(0.2, 0.0, 0.0)),
            body,
        );
        world.step_once();
        let wrench = Rapier3DBackend::contact_force(&world, body);
        assert_eq!(
            wrench, [0.0; 6],
            "same-body collider pairs must contribute zero wrench"
        );
    }

    /// Newton's third law: for the single contact pair between two dynamic
    /// bodies, the force part of `contact_force(A)` is the negation of
    /// `contact_force(B)`. Two dynamic balls are pressed together by a sustained
    /// equal-and-opposite external force (zero gravity, net external force zero
    /// so the pair settles in place) until they reach a steady-state contact —
    /// the horizontal analogue of the resting ball. The solver stores a single
    /// per-contact impulse that acts oppositely on the two bodies
    /// (`-force_mag·normal` on collider1, `+force_mag·normal` on collider2), so
    /// the aggregated force vectors are exactly antisymmetric. (Torque parts are
    /// taken about each body's own `CoM` and need not cancel.)
    #[test]
    fn contact_force_newton_third_law_antisymmetric() {
        let radius = 0.25f32;
        let push = 5.0f32;
        let mut world = Rapier3DWorld::new(Vector::ZERO, 1.0 / 240.0, 1);
        // Start marginally overlapping along +x; the push holds them in contact.
        let a = world.add_body(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(0.0, 0.0, 0.0))
                .additional_mass(1.0),
        );
        world.add_collider(ColliderBuilder::ball(radius).density(0.0), a);
        let b = world.add_body(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(2.0 * radius - 0.01, 0.0, 0.0))
                .additional_mass(1.0),
        );
        world.add_collider(ColliderBuilder::ball(radius).density(0.0), b);
        // Press A toward +x and B toward -x every substep; re-applied because the
        // external-force accumulator lives one substep (ADR 0037). Net force is
        // zero, so the pair stays put and reaches a steady contact.
        for _ in 0..600 {
            world.step_actuated(|w| {
                if let Some(ba) = w.bodies.get_mut(a) {
                    ba.add_force(Vector::new(push, 0.0, 0.0), true);
                }
                if let Some(bb) = w.bodies.get_mut(b) {
                    bb.add_force(Vector::new(-push, 0.0, 0.0), true);
                }
            });
        }

        let fa = Rapier3DBackend::contact_force(&world, a);
        let fb = Rapier3DBackend::contact_force(&world, b);
        // Non-trivial contact, so antisymmetry is not a 0 ≈ −0 tautology.
        let mag = (fa[0] * fa[0] + fa[1] * fa[1] + fa[2] * fa[2]).sqrt();
        assert!(
            mag > 1.0,
            "pressed balls must generate a real contact force (|F_A|={mag})"
        );
        for k in 0..3 {
            assert!(
                (fa[k] + fb[k]).abs() < 1e-3 * mag,
                "force component {k} must be antisymmetric (F_A={}, F_B={})",
                fa[k],
                fb[k]
            );
        }
        // Repulsion pushes each ball away from the other along the contact axis.
        assert!(
            fa[0] < 0.0,
            "A is pushed in −x, away from B (fa_x={})",
            fa[0]
        );
        assert!(
            fb[0] > 0.0,
            "B is pushed in +x, away from A (fb_x={})",
            fb[0]
        );
    }

    /// Build the resting-ball-on-ground scene with a selectable collider
    /// insertion order (ground-first vs ball-first), then settle. Swapping the
    /// order swaps which collider is `collider1` in the contact pair, so this
    /// exercises the insertion-order attribution path.
    fn resting_ball_insertion_order(ground_first: bool) -> (Rapier3DWorld, RigidBodyHandle) {
        let radius = 0.25f32;
        let g = 9.81f32;
        let mut world = Rapier3DWorld::new(Vector::new(0.0, 0.0, -g), 1.0 / 240.0, 1);
        let ground =
            ColliderBuilder::cuboid(10.0, 10.0, 0.5).translation(Vector::new(0.0, 0.0, -0.5));
        let ball;
        if ground_first {
            world.add_ground_collider(ground);
            ball = world.add_body(
                RigidBodyBuilder::dynamic()
                    .translation(Vector::new(0.0, 0.0, radius + 0.001))
                    .additional_mass(1.0),
            );
            world.add_collider(ColliderBuilder::ball(radius).density(0.0), ball);
        } else {
            ball = world.add_body(
                RigidBodyBuilder::dynamic()
                    .translation(Vector::new(0.0, 0.0, radius + 0.001))
                    .additional_mass(1.0),
            );
            world.add_collider(ColliderBuilder::ball(radius).density(0.0), ball);
            world.add_ground_collider(ground);
        }
        for _ in 0..600 {
            world.step_once();
        }
        (world, ball)
    }

    /// Insertion-order robustness: the attributed vertical force on the ball is
    /// positive (ground pushes UP) regardless of whether the ground or the ball
    /// collider was inserted first — the branch that flips `normal` and the
    /// `flipped` flag together must leave the sign (and magnitude) invariant.
    #[test]
    // Justified: paired linear/angular quantities differ by one character by convention.
    #[allow(clippy::similar_names)]
    fn contact_force_insertion_order_robust() {
        let (w_gf, ball_gf) = resting_ball_insertion_order(true);
        let (w_bf, ball_bf) = resting_ball_insertion_order(false);
        let fz_gf = Rapier3DBackend::contact_force(&w_gf, ball_gf)[2];
        let fz_bf = Rapier3DBackend::contact_force(&w_bf, ball_bf)[2];
        assert!(
            fz_gf > 0.0,
            "ground-first: force on ball must be upward (fz={fz_gf})"
        );
        assert!(
            fz_bf > 0.0,
            "ball-first: force on ball must be upward (fz={fz_bf})"
        );
        assert!(
            (fz_gf - fz_bf).abs() < 0.5,
            "attributed vertical force must not depend on insertion order (gf={fz_gf}, bf={fz_bf})"
        );
    }
}
