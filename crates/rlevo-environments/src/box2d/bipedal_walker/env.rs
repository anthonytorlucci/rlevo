//! Core [`BipedalWalker`] environment implementation.
//!
//! This module wires together the Rapier2D physics world, terrain generation,
//! motor control, observation computation, and reward shaping into a type that
//! implements [`rlevo_core::environment::Environment`].
//!
//! The walker body is assembled from five rigid bodies (hull + two upper legs +
//! two lower legs) connected by four revolute joints (two hips, two knees).
//! Each joint has a velocity motor whose target speed is set by the action and
//! capped by `motors_torque`. Physics advances one step of `dt` seconds per
//! `step()` call.
//!
//! ## Reward shaping
//!
//! Each step the reward is `vel_x − 0.3 × Σᵢ aᵢ²`, where `vel_x` is the
//! hull's horizontal velocity and `aᵢ` are the four action components.
//! If the hull contacts the ground an additional −100 penalty is subtracted
//! from that step's reward and the episode is terminated.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rapier2d::dynamics::RevoluteJoint;
use rapier2d::geometry::ColliderHandle;
use rapier2d::prelude::*;
use rlevo_core::base::{Action, State};
use rlevo_core::config::{ConfigError, Validate};
use rlevo_core::environment::{
    ConstructableEnv, Environment, EnvironmentError, EpisodeStatus, SnapshotBase,
};
use rlevo_core::reward::ScalarReward;

use crate::box2d::physics::RapierWorld;

use super::action::BipedalWalkerAction;
use super::config::{BipedalTerrain, BipedalWalkerConfig};
use super::observation::BipedalWalkerObservation;
use super::state::BipedalWalkerState;
use super::terrain::{FlatTerrain, HardcoreTerrain, RoughTerrain, TerrainGenerator};

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
/// # Action (4 dims, must be in `[−1, 1]`)
///
/// `[hip1, knee1, hip2, knee2]` motor velocity targets. Components outside
/// the valid range or containing non-finite values cause `step()` to return
/// `Err(InvalidAction)`.
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
    /// Create a new environment from a [`BipedalWalkerConfig`].
    ///
    /// The terrain generator is selected by dispatching on `config.terrain`:
    /// [`BipedalTerrain::Flat`] uses [`FlatTerrain`], [`BipedalTerrain::Rough`]
    /// uses [`RoughTerrain`], and [`BipedalTerrain::Hardcore`] uses
    /// [`HardcoreTerrain`], each with its default parameters. To supply a
    /// custom generator, use [`BipedalWalker::with_terrain`] instead.
    ///
    /// The physics world is fully built and warm-started during construction,
    /// so the environment is ready to receive `reset()` immediately.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `config` fails [`Validate`] (e.g.
    /// non-positive `motors_torque`, `dt`, or `max_steps == 0`).
    pub fn with_config(config: BipedalWalkerConfig) -> Result<Self, ConfigError> {
        let terrain: Box<dyn TerrainGenerator> = match config.terrain {
            BipedalTerrain::Flat => Box::new(FlatTerrain),
            BipedalTerrain::Rough => Box::new(RoughTerrain::default()),
            BipedalTerrain::Hardcore => Box::new(HardcoreTerrain::default()),
        };
        Self::build(config, terrain)
    }

    /// Create with a custom [`TerrainGenerator`], overriding the terrain preset
    /// stored in `config.terrain`.
    ///
    /// Use this to supply a custom terrain implementation or to inject a
    /// seeded generator for reproducible test scenarios.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `config` fails [`Validate`].
    pub fn with_terrain(
        config: BipedalWalkerConfig,
        terrain: Box<dyn TerrainGenerator>,
    ) -> Result<Self, ConfigError> {
        Self::build(config, terrain)
    }

    fn build(
        config: BipedalWalkerConfig,
        terrain: Box<dyn TerrainGenerator>,
    ) -> Result<Self, ConfigError> {
        config.validate()?;
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
        Ok(env)
    }

    /// Tear down and rebuild the rapier world with fresh terrain and walker bodies.
    fn rebuild_world(&mut self) {
        self.world = RapierWorld::new(Vector::new(0.0, self.config.gravity), self.config.dt);
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
        let joints_set = self.world.joints();
        let mut v = [0.0f32; 24];

        if let Some(hull) = bodies.get(self.state.hull_handle) {
            v[0] = hull.rotation().angle();
            v[1] = hull.angvel();
            v[2] = (hull.linvel().x / 10.0).clamp(-1.0, 1.0);
            v[3] = (hull.linvel().y / 10.0).clamp(-1.0, 1.0);
        }

        // Joint observations: relative angle and speed-normalized relative
        // angular velocity between each joint's parent (body1) and child (body2).
        // Mirrors Gymnasium `joints[i].angle` / `joints[i].speed / SPEED_x`, but
        // WITHOUT the +1.0 knee offset — rlevo builds knees at 0 relative
        // rotation, so that constant would be an unphysical bias (deliberate,
        // documented deviation).
        let joints = [
            (self.state.hip1_joint, 4, self.config.speed_hip),
            (self.state.knee1_joint, 6, self.config.speed_knee),
            (self.state.hip2_joint, 9, self.config.speed_hip),
            (self.state.knee2_joint, 11, self.config.speed_knee),
        ];
        for (jhandle, base, speed) in joints {
            if let Some(j) = joints_set.get(jhandle)
                && let (Some(p), Some(c)) = (bodies.get(j.body1), bodies.get(j.body2))
            {
                v[base] = c.rotation().angle() - p.rotation().angle();
                v[base + 1] = (c.angvel() - p.angvel()) / speed;
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
                let angle = std::f32::consts::PI * (i as f32 / 9.0 - 0.5); // −90° to +90°
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
        let lower1 = self
            .world
            .bodies()
            .get(self.state.leg1_lower_handle)
            .and_then(|b| b.colliders().iter().next().copied());
        let lower2 = self
            .world
            .bodies()
            .get(self.state.leg2_lower_handle)
            .and_then(|b| b.colliders().iter().next().copied());
        self.state.leg1_contact = lower1.is_some_and(|c| self.world.is_in_contact(c));
        self.state.leg2_contact = lower2.is_some_and(|c| self.world.is_in_contact(c));
    }

    fn hull_touching_ground(&self) -> bool {
        self.world
            .bodies()
            .get(self.state.hull_handle)
            .and_then(|b| b.colliders().iter().next().copied())
            .is_some_and(|c| self.world.is_in_contact(c))
    }

    /// Compute the per-step reward.
    ///
    /// The formula is:
    ///
    /// ```text
    /// reward = vel_x − 0.3 × (a₀² + a₁² + a₂² + a₃²)
    /// ```
    ///
    /// where `vel_x` is the hull's horizontal velocity (world units per second)
    /// and `aᵢ` are the four action components. The quadratic control penalty
    /// discourages wasteful motor effort; the velocity term rewards forward
    /// progress.
    ///
    /// If the hull contacts the ground the caller in `step()` subtracts an
    /// additional −100 from the value returned here.
    fn compute_reward(&self, action: &BipedalWalkerAction, vel_x: f32) -> f32 {
        let ctrl_cost = 0.3 * action.0.iter().map(|a| a * a).sum::<f32>();
        vel_x - ctrl_cost
    }

    /// Borrow the internal physics state for in-crate invariant tests.
    #[cfg(test)]
    pub(crate) fn state_for_test(&self) -> &BipedalWalkerState {
        &self.state
    }

    /// Debug representation of the active terrain generator, for dispatch tests.
    #[cfg(test)]
    pub(crate) fn terrain_debug(&self) -> String {
        format!("{:?}", self.terrain)
    }
}

impl ConstructableEnv for BipedalWalker {
    fn new(_render: bool) -> Self {
        Self::with_config(BipedalWalkerConfig::default()).expect("default config must validate")
    }
}

impl Environment<1, 1, 1> for BipedalWalker {
    type StateType = BipedalWalkerState;
    type ObservationType = BipedalWalkerObservation;
    type ActionType = BipedalWalkerAction;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, BipedalWalkerObservation, ScalarReward>;

    /// Rebuild the physics world, reset counters, and return the initial
    /// observation with reward 0 and status `Running`.
    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.rebuild_world();
        self.steps = 0;
        self.total_reward = 0.0;
        self.state.leg1_contact = false;
        self.state.leg2_contact = false;
        let obs = self.compute_observation();
        self.state.last_obs = obs.clone();
        debug_assert!(
            self.state.is_valid(),
            "BipedalWalkerState invariant violated after reset"
        );
        Ok(SnapshotBase::running(obs, ScalarReward(0.0)))
    }

    /// Advance the simulation by one timestep and return the resulting snapshot.
    ///
    /// # Errors
    ///
    /// Returns [`EnvironmentError::InvalidAction`] if any component of `action`
    /// is outside `[-1, 1]` or is non-finite.
    fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError> {
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
            .world
            .bodies()
            .get(self.state.hull_handle)
            .map_or(0.0, |b| b.linvel().x);
        let reward = self.compute_reward(&action, vel_x);
        self.total_reward += reward;

        let obs = self.compute_observation();
        self.state.last_obs = obs.clone();
        debug_assert!(
            self.state.is_valid(),
            "BipedalWalkerState invariant violated after step"
        );

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
        Ok(SnapshotBase {
            observation: obs,
            reward: ScalarReward(final_reward),
            status,
        })
    }
}

// ---------------------------------------------------------------------------
// ASCII renderer
// ---------------------------------------------------------------------------

impl crate::render::AsciiRenderable for BipedalWalker {
    fn render_ascii(&self) -> String {
        let bodies = self.collect_bodies();
        let viewport = self.viewport();
        super::super::render::render_box2d_ascii(
            "Walker",
            &bodies,
            viewport,
            Some(GROUND_Y),
            self.steps,
        )
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        let bodies = self.collect_bodies();
        let viewport = self.viewport();
        super::super::render::render_box2d_styled(
            "Walker",
            &bodies,
            viewport,
            Some(GROUND_Y),
            self.steps,
        )
    }
}

impl BipedalWalker {
    /// Collect hull and leg body positions for the ASCII/styled renderer.
    ///
    /// Returns up to five [`Bodyish`](super::super::render::Bodyish) entries:
    /// one `Agent` for the hull (with rotation angle) and up to four `Dynamic`
    /// entries for the leg segments.
    fn collect_bodies(&self) -> Vec<super::super::render::Bodyish> {
        use super::super::render::Bodyish;

        let mut bodies = Vec::with_capacity(5);
        if let Some(hull) = self.world.bodies().get(self.state.hull_handle) {
            let p = hull.translation();
            bodies.push(Bodyish::Agent {
                x: p.x,
                y: p.y,
                angle_rad: hull.rotation().angle(),
            });
        }
        for handle in [
            self.state.leg1_upper_handle,
            self.state.leg1_lower_handle,
            self.state.leg2_upper_handle,
            self.state.leg2_lower_handle,
        ] {
            if let Some(seg) = self.world.bodies().get(handle) {
                let p = seg.translation();
                bodies.push(Bodyish::Dynamic { x: p.x, y: p.y });
            }
        }
        bodies
    }

    /// 10-unit-wide viewport horizontally centred on the hull; vertical
    /// span fixed so the rendered scene shows the ground line plus ~3 m
    /// of headroom above the hull.
    fn viewport(&self) -> super::super::render::Viewport {
        let hull_x = self
            .world
            .bodies()
            .get(self.state.hull_handle)
            .map_or(0.0, |b| b.translation().x);
        super::super::render::Viewport {
            x_min: hull_x - 5.0,
            x_max: hull_x + 5.0,
            y_min: GROUND_Y - 0.5,
            y_max: GROUND_Y + 3.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlevo_core::base::Observation;
    use rlevo_core::environment::Snapshot;

    /// Creates a default flat-terrain environment for use in tests.
    fn make_env() -> BipedalWalker {
        BipedalWalker::with_config(BipedalWalkerConfig::default()).expect("valid config")
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
        assert!(
            env.step(bad_action).is_err(),
            "D5: out-of-range action must error"
        );
    }

    #[test]
    fn test_determinism() {
        let cfg = BipedalWalkerConfig::builder()
            .seed(42)
            .build()
            .expect("valid config");
        let actions: Vec<BipedalWalkerAction> = (0..20)
            .map(|i| BipedalWalkerAction([(i as f32 * 0.1).sin(); 4]))
            .collect();

        let run = |actions: &[BipedalWalkerAction]| {
            let mut env = BipedalWalker::with_config(cfg.clone()).expect("valid config");
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
        assert_eq!(
            a, b,
            "D5 determinism: identical seed + actions must give identical observations"
        );
    }

    #[test]
    fn test_terrain_generator_pluggable() {
        use crate::box2d::bipedal_walker::terrain::FlatTerrain;
        let cfg = BipedalWalkerConfig::default();
        let mut env =
            BipedalWalker::with_terrain(cfg, Box::new(FlatTerrain)).expect("valid config");
        let snap = env.reset().unwrap();
        assert!(snap.observation().is_finite());
    }

    #[test]
    fn render_styled_matches_ascii() {
        use crate::render::AsciiRenderable;

        let mut env =
            BipedalWalker::with_config(BipedalWalkerConfig::default()).expect("valid config");
        env.reset().unwrap();
        let plain_no_trailing: String = env.render_ascii().lines().collect::<Vec<_>>().join("\n");
        assert_eq!(env.render_styled().plain_text(), plain_no_trailing);
    }

    #[test]
    fn render_styled_uses_palette_consts() {
        use crate::render::AsciiRenderable;
        use crate::render::palette::{AGENT_FG, AGENT_MODIFIER};

        let mut env =
            BipedalWalker::with_config(BipedalWalkerConfig::default()).expect("valid config");
        env.reset().unwrap();
        let styled = env.render_styled();
        let label = styled.lines[0]
            .spans
            .iter()
            .find(|s| s.text == "Walker")
            .expect("Walker label span present");
        assert_eq!(label.style.fg, Some(AGENT_FG));
        assert!(label.style.modifier.contains(AGENT_MODIFIER));
    }

    #[test]
    fn test_joint_obs_not_dead() {
        // Regression (#119): joint angle/speed dims must be live, not constants.
        // Dims [4,6,9,11] are joint angles; [5,7,10,12] are joint speeds.
        let mut env = make_env();
        let reset_snap = env.reset().unwrap();
        let reset_obs = reset_snap.observation().values;

        // Drive asymmetric motor targets so joints move differently.
        let mut moved_obs = reset_obs;
        for _ in 0..30 {
            let action = BipedalWalkerAction([1.0, -1.0, -1.0, 1.0]);
            let snap = env.step(action).unwrap();
            moved_obs = snap.observation().values;
        }

        let joint_dims = [4usize, 5, 6, 7, 9, 10, 11, 12];

        // At least one joint-speed dim must be non-zero — proving speeds are
        // read from the physics world rather than hardcoded to 0.0.
        let speed_dims = [5usize, 7, 10, 12];
        assert!(
            speed_dims.iter().any(|&d| moved_obs[d] != 0.0),
            "all joint-speed dims are zero: {:?}",
            speed_dims.map(|d| moved_obs[d])
        );

        // The joint sub-vector must change from its reset-time values —
        // proving angles track posture rather than a fixed anchor constant.
        let reset_joints: Vec<f32> = joint_dims.iter().map(|&d| reset_obs[d]).collect();
        let moved_joints: Vec<f32> = joint_dims.iter().map(|&d| moved_obs[d]).collect();
        assert_ne!(
            reset_joints, moved_joints,
            "joint observation dims did not change after motion (dead obs)"
        );
    }

    #[test]
    fn test_hip_speed_obs_normalized_by_speed_const() {
        // Regression (#119): the joint-speed dims must divide the raw relative
        // angular velocity by the joint's speed constant.
        //
        // `speed_hip` appears in BOTH `apply_motors` (motor target = action *
        // speed) and `compute_observation` (divisor). To isolate the divisor we
        // drive the ZERO action: the motor target becomes `0 * speed = 0`
        // regardless of `speed_hip`, so the physics is byte-identical between two
        // envs that differ only in `speed_hip`. Gravity/settling still induces a
        // non-zero hip relative angvel, and that RAW angvel is identical in both
        // envs. Only the observation divisor differs, so the reported hip-speed
        // dims (v[5], v[10]) must scale as 1/speed_hip: doubling speed_hip halves
        // the reported speed. This fails if the `/speed` division is dropped.
        let run_settle = |speed_hip: f32| -> (f32, f32) {
            let cfg = BipedalWalkerConfig {
                speed_hip,
                seed: 123,
                ..Default::default()
            };
            let mut env = BipedalWalker::with_config(cfg).expect("valid config");
            env.reset().unwrap();
            // Zero action -> motor target 0 for all joints (speed-independent).
            // A few steps of gravity-driven settling build up hip angvel.
            let mut v = [0.0f32; 24];
            for _ in 0..8 {
                let snap = env.step(BipedalWalkerAction([0.0; 4])).unwrap();
                v = snap.observation().values;
            }
            (v[5], v[10])
        };

        let (slow_h1, slow_h2) = run_settle(4.0);
        let (fast_h1, fast_h2) = run_settle(8.0);

        // Guard against the near-zero case: the hips must have moved for the
        // ratio to be meaningful.
        assert!(
            fast_h1.abs() > 1e-4 && fast_h2.abs() > 1e-4,
            "hips did not move; cannot test normalization: {fast_h1}, {fast_h2}"
        );

        // speed_hip doubled -> reported hip speed halved (raw angvel identical).
        approx::assert_relative_eq!(slow_h1 / fast_h1, 2.0, epsilon = 1e-3);
        approx::assert_relative_eq!(slow_h2 / fast_h2, 2.0, epsilon = 1e-3);
    }

    #[test]
    fn test_with_config_dispatches_terrain() {
        use crate::box2d::bipedal_walker::config::BipedalTerrain;

        // Build one env per terrain preset via with_config; each must construct
        // successfully and reset to a finite observation, AND select the matching
        // generator. Asserting the active generator's debug string regression-
        // guards the dispatch: the pre-fix code hardcoded FlatTerrain for every
        // preset, so the Rough/Hardcore expectations below fail against it.
        for (terrain, expected) in [
            (BipedalTerrain::Flat, "FlatTerrain"),
            (BipedalTerrain::Rough, "RoughTerrain"),
            (BipedalTerrain::Hardcore, "HardcoreTerrain"),
        ] {
            let cfg = BipedalWalkerConfig::builder()
                .terrain(terrain)
                .seed(7)
                .build()
                .expect("valid config");
            let mut env = BipedalWalker::with_config(cfg).expect("valid config");
            assert!(
                env.terrain_debug().contains(expected),
                "with_config({terrain:?}) selected {:?}, expected {expected}",
                env.terrain_debug()
            );
            let snap = env.reset().unwrap();
            assert!(
                snap.observation().is_finite(),
                "reset obs must be finite for terrain {terrain:?}"
            );
        }
    }

    #[test]
    fn render_ascii_within_width_budget() {
        use crate::render::AsciiRenderable;

        let mut env =
            BipedalWalker::with_config(BipedalWalkerConfig::default()).expect("valid config");
        env.reset().unwrap();
        for line in env.render_ascii().lines() {
            assert!(
                line.chars().count() <= 80,
                "line exceeds 80 cols: {line:?} ({} chars)",
                line.chars().count()
            );
        }
    }
}
