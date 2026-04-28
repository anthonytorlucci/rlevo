//! Shared Rapier2D physics backend for Box2D-style environments.
//!
//! [`RapierWorld`] wraps the full rapier2d pipeline and exposes a minimal
//! public surface: `step`, `snapshot`, `restore`, body/collider/joint
//! insertion, and ray-casting for lidar. All rapier2d sets are private
//! (design decision D4).
//!
//! **Note**: rapier2d 0.32 uses glam (`Vec2`) for math types, not nalgebra.
//! Use [`rapier2d::math::Vector`] (= `glam::Vec2`) for vectors.

use rapier2d::parry::query::DefaultQueryDispatcher;
use rapier2d::prelude::*;

/// Opaque physics snapshot used for deterministic rollout and restore.
///
/// Stores a flat copy of every rigid body's isometry and velocity so the
/// world can be restored to an earlier state without re-serialising the
/// full broad-phase / narrow-phase state.
#[derive(Debug, Clone)]
pub struct PhysicsSnapshot {
    records: Vec<BodyRecord>,
}

#[derive(Debug, Clone)]
struct BodyRecord {
    handle: RigidBodyHandle,
    pos: [f32; 2],
    rot: f32,
    linvel: [f32; 2],
    angvel: f32,
}

/// Encapsulated Rapier2D simulation world.
///
/// Create one per environment via [`RapierWorld::new`], add bodies and
/// colliders, then call [`RapierWorld::step`] each environment step.
/// Use [`RapierWorld::snapshot`] / [`RapierWorld::restore`] for
/// deterministic rollouts (design decision D4).
pub struct RapierWorld {
    bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    islands: IslandManager,
    broad_phase: BroadPhaseBvh,
    narrow_phase: NarrowPhase,
    pipeline: PhysicsPipeline,
    ccd_solver: CCDSolver,
    gravity: Vector,
    params: IntegrationParameters,
}

impl std::fmt::Debug for RapierWorld {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RapierWorld")
            .field("num_bodies", &self.bodies.len())
            .field("num_colliders", &self.colliders.len())
            .field("gravity", &[self.gravity.x, self.gravity.y])
            .field("dt", &self.params.dt)
            .finish_non_exhaustive()
    }
}

impl RapierWorld {
    /// Create a new world with the given gravity vector and physics timestep.
    ///
    /// Typical gravity for a 2D side-view: `Vector::new(0.0, -9.8)`.
    pub fn new(gravity: Vector, dt: f32) -> Self {
        let params = IntegrationParameters { dt, ..IntegrationParameters::default() };
        Self {
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            islands: IslandManager::new(),
            broad_phase: BroadPhaseBvh::new(),
            narrow_phase: NarrowPhase::new(),
            pipeline: PhysicsPipeline::new(),
            ccd_solver: CCDSolver::new(),
            gravity,
            params,
        }
    }

    /// Advance the simulation by one timestep.
    pub fn step(&mut self) {
        self.pipeline.step(
            self.gravity,
            &self.params,
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

    /// Capture the current state of all rigid bodies.
    pub fn snapshot(&self) -> PhysicsSnapshot {
        let records = self
            .bodies
            .iter()
            .map(|(handle, body)| {
                let pos = body.translation();
                let rot = body.rotation().angle();
                let vel = body.linvel();
                let angvel = body.angvel();
                BodyRecord {
                    handle,
                    pos: [pos.x, pos.y],
                    rot,
                    linvel: [vel.x, vel.y],
                    angvel,
                }
            })
            .collect();
        PhysicsSnapshot { records }
    }

    /// Restore rigid body positions and velocities from a prior snapshot.
    ///
    /// Bodies present at snapshot time but removed since are silently
    /// skipped. Bodies added after the snapshot are not affected.
    pub fn restore(&mut self, snap: &PhysicsSnapshot) {
        for rec in &snap.records {
            if let Some(body) = self.bodies.get_mut(rec.handle) {
                body.set_translation(Vector::new(rec.pos[0], rec.pos[1]), true);
                body.set_rotation(Rotation::new(rec.rot), true);
                body.set_linvel(Vector::new(rec.linvel[0], rec.linvel[1]), true);
                body.set_angvel(rec.angvel, true);
            }
        }
    }

    // ─── Insertion helpers ────────────────────────────────────────────────────

    /// Insert a rigid body and return its handle.
    pub fn add_body(&mut self, desc: RigidBodyBuilder) -> RigidBodyHandle {
        self.bodies.insert(desc)
    }

    /// Insert a collider attached to `parent` and return its handle.
    pub fn add_collider(
        &mut self,
        desc: ColliderBuilder,
        parent: RigidBodyHandle,
    ) -> ColliderHandle {
        self.colliders
            .insert_with_parent(desc, parent, &mut self.bodies)
    }

    /// Insert a free-floating ground collider (not attached to any body).
    pub fn add_ground_collider(&mut self, desc: ColliderBuilder) -> ColliderHandle {
        self.colliders.insert(desc)
    }

    /// Insert an impulse joint between two bodies and return its handle.
    pub fn add_joint(
        &mut self,
        joint: impl Into<GenericJoint>,
        b1: RigidBodyHandle,
        b2: RigidBodyHandle,
        wake_up: bool,
    ) -> ImpulseJointHandle {
        self.impulse_joints.insert(b1, b2, joint, wake_up)
    }

    // ─── Read-only accessors ──────────────────────────────────────────────────

    /// Read-only access to all rigid bodies (for observation extraction).
    pub fn bodies(&self) -> &RigidBodySet {
        &self.bodies
    }

    /// Read-only access to all colliders.
    pub fn colliders(&self) -> &ColliderSet {
        &self.colliders
    }

    /// Mutable access to all rigid bodies (for applying impulses / motor targets).
    pub fn bodies_mut(&mut self) -> &mut RigidBodySet {
        &mut self.bodies
    }

    /// Mutable access to impulse joints (for setting motor targets / velocities).
    pub fn joints_mut(&mut self) -> &mut ImpulseJointSet {
        &mut self.impulse_joints
    }

    // ─── Contact queries ──────────────────────────────────────────────────────

    /// Returns `true` if `collider` is currently in contact with any other collider.
    pub fn is_in_contact(&self, collider: rapier2d::geometry::ColliderHandle) -> bool {
        self.narrow_phase
            .contact_pairs_with(collider)
            .any(|pair| pair.has_any_active_contact())
    }

    // ─── Ray casting ─────────────────────────────────────────────────────────

    /// Cast a ray from `origin` in `dir`, returning the distance to the
    /// first hit collider, or `None` if no hit within `max_toi`.
    pub fn cast_ray(&self, origin: Vector, dir: Vector, max_toi: f32) -> Option<f32> {
        let dispatcher = DefaultQueryDispatcher;
        let qp = self.broad_phase.as_query_pipeline(
            &dispatcher,
            &self.bodies,
            &self.colliders,
            QueryFilter::default(),
        );
        let ray = Ray::new(origin, dir);
        qp.cast_ray(&ray, max_toi, true).map(|(_, toi)| toi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Creates a standard-gravity world at 60 Hz for use in tests.
    fn make_world() -> RapierWorld {
        RapierWorld::new(Vector::new(0.0, -9.8), 1.0 / 60.0)
    }

    #[test]
    fn test_gravity_body_falls() {
        let mut world = make_world();
        let handle = world.add_body(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(0.0, 10.0))
                .additional_mass(1.0),
        );
        let y0 = world.bodies().get(handle).unwrap().translation().y;
        for _ in 0..30 {
            world.step();
        }
        let y1 = world.bodies().get(handle).unwrap().translation().y;
        assert!(y1 < y0, "body should fall under gravity");
    }

    #[test]
    fn test_snapshot_restore_roundtrip() {
        let mut world = make_world();
        let handle =
            world.add_body(RigidBodyBuilder::dynamic().translation(Vector::new(0.0, 10.0)));
        for _ in 0..10 {
            world.step();
        }
        let snap = world.snapshot();
        let pos_at_snap = world.bodies().get(handle).unwrap().translation().y;
        for _ in 0..20 {
            world.step();
        }
        world.restore(&snap);
        let pos_after_restore = world.bodies().get(handle).unwrap().translation().y;
        assert!(
            (pos_at_snap - pos_after_restore).abs() < 1e-5,
            "restore should return body to snapshot position"
        );
    }

    #[test]
    fn test_determinism() {
        let run = |steps: usize| {
            let mut world = make_world();
            let handle =
                world.add_body(RigidBodyBuilder::dynamic().translation(Vector::new(0.0, 10.0)));
            for _ in 0..steps {
                world.step();
            }
            let t = world.bodies().get(handle).unwrap().translation();
            [t.x, t.y]
        };
        let a = run(50);
        let b = run(50);
        assert_eq!(a, b, "same initial conditions must produce identical results");
    }

    #[test]
    fn test_debug_impl() {
        let world = make_world();
        let s = format!("{world:?}");
        assert!(s.contains("RapierWorld"));
    }
}
