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
        let integration_parameters = IntegrationParameters { dt, ..IntegrationParameters::default() };
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
    pub fn step_with_frame_skip(&mut self) {
        for _ in 0..self.frame_skip.max(1) {
            self.step_once();
        }
    }

    /// Advance the simulation by a single physics substep (ignores `frame_skip`).
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
    }

    // ─── Insertion helpers (used by env skeleton builders) ───────────────────

    pub(crate) fn add_body(&mut self, desc: RigidBodyBuilder) -> RigidBodyHandle {
        self.bodies.insert(desc)
    }

    pub(crate) fn add_collider(
        &mut self,
        desc: ColliderBuilder,
        parent: RigidBodyHandle,
    ) -> ColliderHandle {
        self.colliders.insert_with_parent(desc, parent, &mut self.bodies)
    }

    pub(crate) fn add_ground_collider(&mut self, desc: ColliderBuilder) -> ColliderHandle {
        self.colliders.insert(desc)
    }

    pub(crate) fn add_impulse_joint(
        &mut self,
        b1: RigidBodyHandle,
        b2: RigidBodyHandle,
        joint: impl Into<GenericJoint>,
    ) -> ImpulseJointHandle {
        self.impulse_joints.insert(b1, b2, joint, true)
    }

    pub(crate) fn add_multibody_joint(
        &mut self,
        b1: RigidBodyHandle,
        b2: RigidBodyHandle,
        joint: impl Into<GenericJoint>,
    ) -> Option<MultibodyJointHandle> {
        self.multibody_joints.insert(b1, b2, joint, true)
    }

    pub(crate) fn bodies(&self) -> &RigidBodySet {
        &self.bodies
    }

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
            Pose { position: [0.0; 3], orientation: [1.0, 0.0, 0.0, 0.0] },
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
            Twist { linear: [0.0; 3], angular: [0.0; 3] },
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

    fn apply_joint_torque(
        _world: &mut Self::World,
        _joint: Self::JointHandle,
        _torque: f32,
    ) {
        // TODO(locomotion-stage1): Rapier 3D doesn't expose a clean
        // "joint torque" primitive — for revolute joints we'll either
        // (a) apply equal-and-opposite angular impulses to the two bodies
        //     about the joint's free axis, or
        // (b) use the joint motor API with high damping / zero stiffness.
        //
        // The cleanest mapping depends on whether the joint is impulse-based
        // or multibody. Per-env skeleton builders currently apply torque
        // directly via `RigidBodySet::get_mut(...).add_torque(...)` until
        // this is unified. Tracking this gap so I don't ship a half-wired
        // motor and forget.
    }

    fn contact_force(world: &Self::World, body: Self::BodyHandle) -> [f32; 6] {
        // Aggregate impulses over all contact manifolds touching any collider
        // attached to `body`. Impulse is in N·s; dividing by dt yields an
        // average force over the substep. Torque is r × F, with r taken from
        // the manifold contact point relative to the body centre of mass.
        let Some(rb) = world.bodies.get(body) else { return [0.0; 6] };
        let com = rb.center_of_mass();
        let dt = world.integration_parameters.dt.max(f32::EPSILON);

        let mut wrench = [0.0f32; 6];
        for &collider_handle in rb.colliders() {
            for pair in world.narrow_phase.contact_pairs_with(collider_handle) {
                let flipped = pair.collider2 == collider_handle;
                for manifold in &pair.manifolds {
                    // Normal points from body1 to body2; flip for incident body.
                    let n = if flipped { -manifold.data.normal } else { manifold.data.normal };
                    for contact in &manifold.points {
                        let impulse = contact.data.impulse;
                        if impulse.abs() < f32::EPSILON {
                            continue;
                        }
                        let force_mag = impulse / dt;
                        let force = n * force_mag;
                        let local_p = if flipped { contact.local_p2 } else { contact.local_p1 };
                        let collider = if flipped { pair.collider2 } else { pair.collider1 };
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
        let handle = world.add_body(
            RigidBodyBuilder::fixed().translation(Vector::new(1.0, 2.0, 3.0)),
        );
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
        assert!(z1 < z0 - 0.03, "frame_skip=5 should drop body noticeably (Δ={})", z0 - z1);
    }

    #[test]
    fn contact_force_is_zero_for_non_contacting_body() {
        let mut world = make_world();
        let handle = world.add_body(RigidBodyBuilder::dynamic().translation(Vector::new(0.0, 0.0, 10.0)));
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
}
