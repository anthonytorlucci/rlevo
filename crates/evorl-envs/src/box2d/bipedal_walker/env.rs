//! BipedalWalker environment implementation.

use evorl_core::base::Action;
use evorl_core::environment::{
    Environment, EnvironmentError, EpisodeStatus, SnapshotBase,
};
use evorl_core::reward::ScalarReward;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rapier2d::dynamics::RevoluteJoint;
use rapier2d::geometry::ColliderHandle;
use rapier2d::prelude::*;

use crate::box2d::physics::RapierWorld;

use super::action::BipedalWalkerAction;
use super::config::BipedalWalkerConfig;
use super::observation::BipedalWalkerObservation;
use super::state::BipedalWalkerState;
use super::terrain::{FlatTerrain, TerrainGenerator};

// ─── Physical constants matching Gymnasium BipedalWalker-v3 ──────────────────

/// Half-width of the hull (torso).
const HULL_W: f32 = 24.0 / 2.0;
/// Half-height of the hull.
const HULL_H: f32 = 14.0 / 2.0;
/// Length of the upper leg segment.
const LEG_H: f32 = 34.0 / 2.0;
/// Width of the leg segments.
const LEG_W: f32 = 8.0 / 2.0;
/// Half-height of the lower leg (shin).
const LOWER_H: f32 = 34.0 / 2.0;
/// Scale factor (world units per pixel).
const SCALE: f32 = 30.0;
/// Ground y-level (world units).
const GROUND_Y: f32 = -1.0;

/// BipedalWalker reinforcement learning environment.
///
/// A 2D bipedal robot that learns to walk forward using hip and knee motor
/// targets. Physics are simulated with Rapier2D (enhanced-determinism).
///
/// # Episode lifecycle
///
/// - `reset()` rebuilds the rapier world with fresh terrain.
/// - `step(action)` applies 4 motor targets, advances physics, computes reward.
/// - Terminates when the hull contacts the ground or cumulative reward < −100.
/// - Truncated after `config.max_steps` steps (default: 1600).
///
/// # Observation (24 dims)
///
/// See [`BipedalWalkerObservation`] for the full field mapping.
///
/// # Action (4 dims, D5: must be in `[−1, 1]`)
///
/// `[hip1, knee1, hip2, knee2]` motor velocity targets.
#[derive(Debug)]
pub struct BipedalWalker {
    world: RapierWorld,
    state: BipedalWalkerState,
    ground_handle: ColliderHandle,
    config: BipedalWalkerConfig,
    terrain: Box<dyn TerrainGenerator>,
    rng: StdRng,
    steps: usize,
    /// Running sum of rewards (for the < −100 termination check).
    total_reward: f32,
}

impl BipedalWalker {
    /// Create a new environment with default configuration.
    pub fn with_config(config: BipedalWalkerConfig) -> Self {
        let terrain: Box<dyn TerrainGenerator> = Box::new(FlatTerrain);
        Self::build(config, terrain)
    }

    /// Create with a custom terrain generator (D7).
    pub fn with_terrain(config: BipedalWalkerConfig, terrain: Box<dyn TerrainGenerator>) -> Self {
        Self::build(config, terrain)
    }

    fn build(config: BipedalWalkerConfig, terrain: Box<dyn TerrainGenerator>) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        let mut env = Self {
            world: RapierWorld::new(Vector::new(0.0, config.gravity), config.dt),
            state: BipedalWalkerState {
                hull_handle: RigidBodyHandle::invalid(),
                leg1_upper_handle: RigidBodyHandle::invalid(),
                leg1_lower_handle: RigidBodyHandle::invalid(),
                leg2_upper_handle: RigidBodyHandle::invalid(),
                leg2_lower_handle: RigidBodyHandle::invalid(),
                hip1_joint: ImpulseJointHandle::invalid(),
                knee1_joint: ImpulseJointHandle::invalid(),
                hip2_joint: ImpulseJointHandle::invalid(),
                knee2_joint: ImpulseJointHandle::invalid(),
                leg1_contact: false,
                leg2_contact: false,
                last_obs: BipedalWalkerObservation::default(),
            },
            ground_handle: ColliderHandle::invalid(),
            config,
            terrain,
            rng,
            steps: 0,
            total_reward: 0.0,
        };
        env.rebuild_world();
        env
    }

    /// Tear down and rebuild the rapier world with fresh terrain and walker bodies.
    fn rebuild_world(&mut self) {
        self.world = RapierWorld::new(
            Vector::new(0.0, self.config.gravity),
            self.config.dt,
        );
        let pts = self.terrain.generate(&mut self.rng);
        self.build_ground(&pts);
        self.build_walker();
        // Warm-start: one step so joints settle
        self.world.step();
    }

    fn build_ground(&mut self, pts: &[[f32; 2]]) {
        if pts.len() < 2 {
            return;
        }
        // Build ground as a series of cuboid segments approximating the polyline.
        // For simplicity, use a long flat cuboid for now.
        let ground_rb = self.world.add_body(RigidBodyBuilder::fixed());
        self.ground_handle = self.world.add_collider(
            ColliderBuilder::cuboid(100.0, 0.5)
                .translation(Vector::new(0.0, GROUND_Y - 0.5))
                .friction(self.config.hull_friction),
            ground_rb,
        );
        // Additional segments for rough / hardcore terrain
        if pts.len() >= 2 {
            for w in pts.windows(2) {
                let x0 = w[0][0] / SCALE;
                let y0 = w[0][1] / SCALE;
                let x1 = w[1][0] / SCALE;
                let y1 = w[1][1] / SCALE;
                let mx = (x0 + x1) / 2.0;
                let my = (y0 + y1) / 2.0;
                let dx = x1 - x0;
                let dy = y1 - y0;
                let len = (dx * dx + dy * dy).sqrt() / 2.0;
                let angle = dy.atan2(dx);
                let seg_rb = self.world.add_body(RigidBodyBuilder::fixed());
                self.world.add_collider(
                    ColliderBuilder::cuboid(len, 0.05)
                        .rotation(angle)
                        .translation(Vector::new(mx, my + GROUND_Y))
                        .friction(self.config.hull_friction),
                    seg_rb,
                );
            }
        }
    }

    fn build_walker(&mut self) {
        let spawn_x = 0.0;
        let spawn_y = GROUND_Y + HULL_H / SCALE + LEG_H * 2.0 / SCALE + 0.05;

        // Hull
        let hull_rb = self.world.add_body(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(spawn_x, spawn_y))
                .linear_damping(0.0)
                .angular_damping(0.0),
        );
        self.world.add_collider(
            ColliderBuilder::cuboid(HULL_W / SCALE, HULL_H / SCALE)
                .density(5.0)
                .friction(self.config.hull_friction),
            hull_rb,
        );
        self.state.hull_handle = hull_rb;

        // Legs (offsets from hull centre)
        self.state.leg1_upper_handle = self.build_leg_segment(
            spawn_x - LEG_W / SCALE / 2.0,
            spawn_y - HULL_H / SCALE - LEG_H / SCALE,
            LEG_W / SCALE,
            LEG_H / SCALE,
        );
        self.state.leg1_lower_handle = self.build_leg_segment(
            spawn_x - LEG_W / SCALE / 2.0,
            spawn_y - HULL_H / SCALE - LEG_H * 2.0 / SCALE - LOWER_H / SCALE,
            LEG_W / SCALE,
            LOWER_H / SCALE,
        );
        self.state.leg2_upper_handle = self.build_leg_segment(
            spawn_x + LEG_W / SCALE / 2.0,
            spawn_y - HULL_H / SCALE - LEG_H / SCALE,
            LEG_W / SCALE,
            LEG_H / SCALE,
        );
        self.state.leg2_lower_handle = self.build_leg_segment(
            spawn_x + LEG_W / SCALE / 2.0,
            spawn_y - HULL_H / SCALE - LEG_H * 2.0 / SCALE - LOWER_H / SCALE,
            LEG_W / SCALE,
            LOWER_H / SCALE,
        );

        // Joints
        self.state.hip1_joint = self.attach_revolute(
            hull_rb,
            self.state.leg1_upper_handle,
            Vector::new(-LEG_W / SCALE / 2.0, -HULL_H / SCALE),
            Vector::new(0.0, LEG_H / SCALE),
            self.config.motors_torque,
            self.config.speed_hip,
        );
        self.state.knee1_joint = self.attach_revolute(
            self.state.leg1_upper_handle,
            self.state.leg1_lower_handle,
            Vector::new(0.0, -LEG_H / SCALE),
            Vector::new(0.0, LOWER_H / SCALE),
            self.config.motors_torque,
            self.config.speed_knee,
        );
        self.state.hip2_joint = self.attach_revolute(
            hull_rb,
            self.state.leg2_upper_handle,
            Vector::new(LEG_W / SCALE / 2.0, -HULL_H / SCALE),
            Vector::new(0.0, LEG_H / SCALE),
            self.config.motors_torque,
            self.config.speed_hip,
        );
        self.state.knee2_joint = self.attach_revolute(
            self.state.leg2_upper_handle,
            self.state.leg2_lower_handle,
            Vector::new(0.0, -LEG_H / SCALE),
            Vector::new(0.0, LOWER_H / SCALE),
            self.config.motors_torque,
            self.config.speed_knee,
        );
    }

    fn build_leg_segment(&mut self, cx: f32, cy: f32, hw: f32, hh: f32) -> RigidBodyHandle {
        let rb = self.world.add_body(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(cx, cy))
                .linear_damping(0.0)
                .angular_damping(0.0),
        );
        self.world.add_collider(
            ColliderBuilder::cuboid(hw, hh)
                .density(1.0)
                .friction(self.config.leg_friction),
            rb,
        );
        rb
    }

    #[allow(clippy::too_many_arguments)]
    fn attach_revolute(
        &mut self,
        parent: RigidBodyHandle,
        child: RigidBodyHandle,
        anchor1: Vector,
        anchor2: Vector,
        max_torque: f32,
        _speed: f32,
    ) -> ImpulseJointHandle {
        let mut joint = RevoluteJoint::new();
        joint.set_local_anchor1(anchor1);
        joint.set_local_anchor2(anchor2);
        joint.set_contacts_enabled(false);
        joint.set_motor_max_force(max_torque);
        self.world.add_joint(joint, parent, child, true)
    }

    fn apply_motors(&mut self, action: &BipedalWalkerAction) {
        let [h1, k1, h2, k2] = action.0;
        let torque = self.config.motors_torque;
        let speed_hip = self.config.speed_hip;
        let speed_knee = self.config.speed_knee;

        for (handle, target, speed) in [
            (self.state.hip1_joint, h1, speed_hip),
            (self.state.knee1_joint, k1, speed_knee),
            (self.state.hip2_joint, h2, speed_hip),
            (self.state.knee2_joint, k2, speed_knee),
        ] {
            if let Some(joint) = self.world.joints_mut().get_mut(handle, true) {
                joint
                    .data
                    .set_motor_velocity(JointAxis::AngX, target * speed, torque);
            }
        }
    }

    fn compute_observation(&mut self) -> BipedalWalkerObservation {
        let bodies = self.world.bodies();
        let mut v = [0.0f32; 24];

        if let Some(hull) = bodies.get(self.state.hull_handle) {
            v[0] = hull.rotation().angle();
            v[1] = hull.angvel();
            v[2] = (hull.linvel().x / 10.0).clamp(-1.0, 1.0);
            v[3] = (hull.linvel().y / 10.0).clamp(-1.0, 1.0);
        }

        let joints = [
            (self.state.hip1_joint, 4),
            (self.state.knee1_joint, 6),
            (self.state.hip2_joint, 9),
            (self.state.knee2_joint, 11),
        ];
        for (jhandle, base) in joints {
            if let Some(j) = self.world.joints_mut().get(jhandle) {
                let ang = j.data.local_anchor1().y; // proxy for angle
                let vel = 0.0f32; // motor speed not directly exposed
                v[base] = ang;
                v[base + 1] = vel;
            }
        }
        v[8] = f32::from(self.state.leg1_contact);
        v[13] = f32::from(self.state.leg2_contact);

        // Lidar rays
        let lidar = self.cast_lidar();
        v[14..24].copy_from_slice(&lidar);

        BipedalWalkerObservation::new(v)
    }

    fn cast_lidar(&self) -> [f32; 10] {
        let mut readings = [1.0f32; 10];
        if let Some(hull) = self.world.bodies().get(self.state.hull_handle) {
            let origin = hull.translation();
            for (i, reading) in readings.iter_mut().enumerate() {
                let angle = std::f32::consts::PI
                    * (i as f32 / 9.0 - 0.5); // −90° to +90°
                let dir = Vector::new(angle.cos(), angle.sin());
                if let Some(toi) = self.world.cast_ray(
                    Vector::new(origin.x, origin.y),
                    dir,
                    self.config.lidar_range,
                ) {
                    *reading = (toi / self.config.lidar_range).clamp(0.0, 1.0);
                }
            }
        }
        readings
    }

    fn update_contact_flags(&mut self) {
        let lower1 = self.world.bodies().get(self.state.leg1_lower_handle)
            .and_then(|b| b.colliders().iter().next().copied());
        let lower2 = self.world.bodies().get(self.state.leg2_lower_handle)
            .and_then(|b| b.colliders().iter().next().copied());
        self.state.leg1_contact = lower1.is_some_and(|c| self.world.is_in_contact(c));
        self.state.leg2_contact = lower2.is_some_and(|c| self.world.is_in_contact(c));
    }

    fn hull_touching_ground(&self) -> bool {
        self.world.bodies()
            .get(self.state.hull_handle)
            .and_then(|b| b.colliders().iter().next().copied())
            .is_some_and(|c| self.world.is_in_contact(c))
    }

    fn compute_reward(&self, action: &BipedalWalkerAction, vel_x: f32) -> f32 {
        let ctrl_cost = 0.3 * action.0.iter().map(|a| a * a).sum::<f32>();
        vel_x - ctrl_cost
    }
}

impl Environment<1, 1, 1> for BipedalWalker {
    type StateType = BipedalWalkerState;
    type ObservationType = BipedalWalkerObservation;
    type ActionType = BipedalWalkerAction;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, BipedalWalkerObservation, ScalarReward>;

    fn new(_render: bool) -> Self {
        Self::with_config(BipedalWalkerConfig::default())
    }

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.rebuild_world();
        self.steps = 0;
        self.total_reward = 0.0;
        self.state.leg1_contact = false;
        self.state.leg2_contact = false;
        let obs = self.compute_observation();
        self.state.last_obs = obs.clone();
        Ok(SnapshotBase::running(obs, ScalarReward(0.0)))
    }

    fn step(
        &mut self,
        action: Self::ActionType,
    ) -> Result<Self::SnapshotType, EnvironmentError> {
        if !action.is_valid() {
            return Err(EnvironmentError::InvalidAction(format!(
                "BipedalWalkerAction components must be in [-1, 1], got {:?}",
                action.0
            )));
        }

        self.apply_motors(&action);
        self.world.step();
        self.steps += 1;
        self.update_contact_flags();

        let vel_x = self
            .world.bodies()
            .get(self.state.hull_handle)
            .map_or(0.0, |b| b.linvel().x);
        let reward = self.compute_reward(&action, vel_x);
        self.total_reward += reward;

        let obs = self.compute_observation();
        self.state.last_obs = obs.clone();

        let hull_down = self.hull_touching_ground();
        let terminated = hull_down || self.total_reward < -100.0;
        let status = if terminated {
            EpisodeStatus::Terminated
        } else if self.steps >= self.config.max_steps {
            EpisodeStatus::Truncated
        } else {
            EpisodeStatus::Running
        };

        // Apply fall penalty
        let final_reward = if hull_down { reward - 100.0 } else { reward };
        Ok(SnapshotBase { observation: obs, reward: ScalarReward(final_reward), status })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use evorl_core::base::Observation;
    use evorl_core::environment::Snapshot;

    fn make_env() -> BipedalWalker {
        BipedalWalker::with_config(BipedalWalkerConfig::default())
    }

    #[test]
    fn test_obs_shape() {
        assert_eq!(BipedalWalkerObservation::shape(), [24]);
    }

    #[test]
    fn test_reset_returns_running() {
        let mut env = make_env();
        let snap = env.reset().unwrap();
        assert!(!snap.is_done());
    }

    #[test]
    fn test_step_obs_all_finite() {
        let mut env = make_env();
        env.reset().unwrap();
        for _ in 0..10 {
            let action = BipedalWalkerAction([0.0; 4]);
            let snap = env.step(action).unwrap();
            assert!(
                snap.observation().is_finite(),
                "observation must be all-finite after every step"
            );
        }
    }

    #[test]
    fn test_d5_action_out_of_range() {
        let mut env = make_env();
        env.reset().unwrap();
        let bad_action = BipedalWalkerAction([2.0, 0.0, 0.0, 0.0]);
        assert!(env.step(bad_action).is_err(), "D5: out-of-range action must error");
    }

    #[test]
    fn test_determinism() {
        let cfg = BipedalWalkerConfig::builder().seed(42).build();
        let actions: Vec<BipedalWalkerAction> = (0..20)
            .map(|i| BipedalWalkerAction([(i as f32 * 0.1).sin(); 4]))
            .collect();

        let run = |actions: &[BipedalWalkerAction]| {
            let mut env = BipedalWalker::with_config(cfg.clone());
            env.reset().unwrap();
            let mut last_obs = BipedalWalkerObservation::default();
            for a in actions {
                if let Ok(snap) = env.step(a.clone()) {
                    last_obs = snap.observation().clone();
                }
            }
            last_obs.values
        };

        let a = run(&actions);
        let b = run(&actions);
        assert_eq!(a, b, "D5 determinism: identical seed + actions must give identical observations");
    }

    #[test]
    fn test_terrain_generator_pluggable() {
        use crate::box2d::bipedal_walker::terrain::FlatTerrain;
        let cfg = BipedalWalkerConfig::default();
        let mut env = BipedalWalker::with_terrain(cfg, Box::new(FlatTerrain));
        let snap = env.reset().unwrap();
        assert!(snap.observation().is_finite());
    }
}
