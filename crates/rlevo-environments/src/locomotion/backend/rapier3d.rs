//! Rapier3D implementation of [`LocomotionBackend`].
//!
//! Wraps the rapier3d simulation pipeline in [`Rapier3DWorld`] and provides a
//! [`Rapier3DJointHandle`] union covering both impulse and multibody joints.
//! Bodies are kept in maximal coordinates; articulated chains (Ant legs,
//! Humanoid torso/arms/legs, Walker2D) are driven via `MultibodyJointSet` so
//! they behave closer to MuJoCo's generalised coordinates than plain impulse
//! joints would.
//!
//! Note: rapier3d 0.32 uses glam (`Vec3`, `Quat`) under the hood via parry's
//! re-exports — there is no nalgebra in observation paths.

use rapier3d::math::{Real, Vector};
use rapier3d::prelude::*;

use super::{LocomotionBackend, Pose, Twist};

/// Rapier3D scene for one locomotion environment.
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
    /// substep. To hold an actuator constant across the frame skip (MuJoCo
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
    /// [`reset_external_forces`](Self::reset_external_forces) **after** the
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
    /// frame skip (MuJoCo `ctrl`-held-across-substeps semantics) the caller MUST
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
    /// Multibody joints behave closer to MuJoCo's generalised coordinates than
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
/// Rapier3D has two disjoint joint stores: [`ImpulseJointSet`] (good for
/// high-frequency / contact-rich joints like wheels and feet) and
/// [`MultibodyJointSet`] (analogous to MuJoCo's generalised coordinates for
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

/// Backend marker type; implements [`LocomotionBackend`] with Rapier3D semantics.
#[derive(Debug, Clone, Copy)]
pub struct Rapier3DBackend;

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

    /// **Not yet implemented — panics at runtime.**
    ///
    /// Rapier3D lacks a first-class "apply torque to joint" primitive. The
    /// planned v0.2 implementation will either (a) apply equal-and-opposite
    /// angular impulses to the two bodies about the joint's free axis, or
    /// (b) use the joint motor API with high damping / zero stiffness.
    /// Until then, per-env skeleton builders apply torque directly via
    /// `RigidBodySet::get_mut(…).add_torque(…)`.
    ///
    /// # Panics
    ///
    /// Always panics with an explanatory message.
    fn apply_joint_torque(_world: &mut Self::World, _joint: Self::JointHandle, _torque: f32) {
        // Rapier 3D doesn't expose a clean "joint torque" primitive — for
        // revolute joints the future implementation will either (a) apply
        // equal-and-opposite angular impulses to the two bodies about the
        // joint's free axis, or (b) use the joint motor API with high
        // damping / zero stiffness. The cleanest mapping depends on whether
        // the joint is impulse-based or multibody. Per-env skeleton builders
        // currently apply torque directly via
        // `RigidBodySet::get_mut(...).add_torque(...)`.
        unimplemented!(
            "Rapier3DBackend::apply_joint_torque is not wired up; \
             per-env skeletons apply torque directly via RigidBodySet::add_torque. \
             Planned for v0.2."
        );
    }

    fn contact_force(world: &Self::World, body: Self::BodyHandle) -> [f32; 6] {
        // Aggregate impulses over all contact manifolds touching any collider
        // attached to `body`. Impulse is in N·s; dividing by dt yields an
        // average force over the substep. Torque is r × F, with r taken from
        // the manifold contact point relative to the body centre of mass.
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
                    // Normal points from body1 to body2; flip for incident body.
                    let n = if flipped {
                        -manifold.data.normal
                    } else {
                        manifold.data.normal
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
}
