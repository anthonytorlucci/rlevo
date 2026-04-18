//! # Reacher-v5 (Rapier3D-backed)
//!
//! # Physics note
//!
//! This env simulates dynamics via Rapier3D, not MuJoCo. Observation shape,
//! action dimensionality, reward structure, and termination conditions match
//! Gymnasium v5 exactly. **Absolute reward values, learned policies, and
//! trained scores will NOT transfer to real Gymnasium/MuJoCo benchmarks
//! without retuning.**
//!
//! ## Layout
//!
//! A planar 2-link arm in the xy-plane with zero gravity. The shoulder is a
//! fixed anchor at the world origin; the elbow connects `link1` (length 0.1)
//! to `link2` (length 0.11). A small `target` body is placed at a random
//! position in a disk of radius 0.2 at each reset; the agent must reach it
//! with the fingertip.
//!
//! * Shoulder: revolute impulse joint about world-z, root → link1.
//! * Elbow:    revolute impulse joint about world-z, link1 → link2.
//! * Planar constraint: both links have `enabled_translations(true, true, false)`
//!   and `enabled_rotations(false, false, true)`.
//! * Action: `Box(-1, 1, (2,))` — shoulder/elbow torque targets; applied as
//!   `action · gear` with `gear = [200, 200]` (Gymnasium XML).
//! * Observation (10-dim):
//!   `[cos θ₁, cos θ₂, sin θ₁, sin θ₂, target_x, target_y, θ̇₁, θ̇₂,
//!     (finger − target)_x, (finger − target)_y]`.
//!   θ₂ is the **relative** elbow angle (link2 − link1), wrapped to `(-π, π]`.
//! * Reward: `reward_distance + reward_control` with
//!   `reward_distance = −‖finger − target‖` and
//!   `reward_control  = −0.1 · ‖action‖²`; both components ≤ 0.
//! * Termination: never (`TerminationMode::Never`).
//! * Truncation: `max_steps = 50`.
//!
//! ## Fingertip convention
//!
//! Rapier's `capsule_x(half_len, r)` places the capsule symmetric about the
//! body origin, so `link2`'s tip sits at body-local `(+link2_length/2, 0, 0)`.
//! The spec §9's phrasing `link2_rotation * [link2_length, 0, 0]` is a
//! shorthand — the geometrically correct offset is `link2_length/2`.
//!
//! ## Divergence from Gymnasium-v5 dynamics
//!
//! Gymnasium's reacher XML stabilises the tiny-inertia links via MuJoCo joint
//! `armature` and `damping`, neither of which has a direct Rapier equivalent.
//! With literal gear `[200, 200]` and per-link mass `0.0356`, a **random-policy
//! rollout** drives the arm into highly non-physical velocities and distances;
//! trained / clipped policies stay in a reasonable regime. This is the
//! top-level "Physics note" in concrete form: reward values will not transfer
//! without retuning (e.g. lowering `gear`, raising `link_mass`, or scaling
//! `ctrl_cost_weight`).

use std::marker::PhantomData;

use burn::prelude::{Backend, Tensor};
use evorl_core::action::ContinuousAction;
use evorl_core::base::{Action, Observation, State, TensorConversionError, TensorConvertible};
use evorl_core::environment::{Environment, EnvironmentError, EpisodeStatus, SnapshotMetadata};
use evorl_core::reward::ScalarReward;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rapier3d::math::Vector;
use rapier3d::prelude::*;
use serde::{Deserialize, Serialize};

use super::backend::{LocomotionBackend, Rapier3DBackend, Rapier3DWorld};
use super::common::{Gear, LocomotionSnapshot, ctrl_cost, wrap_to_pi};

/// Reward-component key: `−‖finger − target‖` (≤ 0).
pub const METADATA_KEY_REWARD_DISTANCE: &str = "reward_distance";
/// Reward-component key: `−0.1 · ‖action‖²` (≤ 0).
pub const METADATA_KEY_REWARD_CONTROL: &str = "reward_control";

// ─── Config ───────────────────────────────────────────────────────────────

/// Environment configuration for [`Reacher`].
///
/// Defaults match the Gymnasium v5 reacher XML: gear `[200, 200]`, dt 0.01,
/// frame_skip 2 (env dt = 0.02), reset noise 0.1, ctrl-cost weight 0.1,
/// target-disk radius 0.2, truncation at 50.
#[derive(Debug, Clone)]
pub struct ReacherConfig {
    pub seed: u64,
    pub gear: Gear<2>,
    pub dt: f32,
    pub frame_skip: u32,
    pub reset_noise_scale: f32,
    pub max_steps: usize,
    pub action_clip: (f32, f32),
    pub ctrl_cost_weight: f32,
    pub link1_length: f32,
    pub link2_length: f32,
    pub link_radius: f32,
    pub link_mass: f32,
    pub target_disk_radius: f32,
}

impl Default for ReacherConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            gear: Gear::new([200.0, 200.0]),
            dt: 0.01,
            frame_skip: 2,
            reset_noise_scale: 0.1,
            max_steps: 50,
            action_clip: (-1.0, 1.0),
            ctrl_cost_weight: 0.1,
            link1_length: 0.10,
            link2_length: 0.11,
            link_radius: 0.01,
            link_mass: 0.0356,
            target_disk_radius: 0.2,
        }
    }
}

// ─── Action ──────────────────────────────────────────────────────────────

/// 2D continuous action — `[shoulder, elbow]` torque targets in
/// pre-gear units. Bounds: `[-1.0, 1.0]` per element.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReacherAction(pub [f32; 2]);

impl ReacherAction {
    #[must_use]
    pub const fn new(shoulder: f32, elbow: f32) -> Self {
        Self([shoulder, elbow])
    }
}

impl Action<1> for ReacherAction {
    fn shape() -> [usize; 1] {
        [2]
    }

    fn is_valid(&self) -> bool {
        self.0.iter().all(|v| v.is_finite() && v.abs() <= 1.0)
    }
}

impl ContinuousAction<1> for ReacherAction {
    fn as_slice(&self) -> &[f32] {
        &self.0
    }

    fn clip(&self, min: f32, max: f32) -> Self {
        Self([self.0[0].clamp(min, max), self.0[1].clamp(min, max)])
    }

    fn from_slice(values: &[f32]) -> Self {
        Self([values[0], values[1]])
    }

    fn random() -> Self {
        Self([rand::random::<f32>() * 2.0 - 1.0, rand::random::<f32>() * 2.0 - 1.0])
    }
}

impl<B: Backend> TensorConvertible<1, B> for ReacherAction {
    fn to_tensor(&self, device: &B::Device) -> Tensor<B, 1> {
        Tensor::from_floats(self.0, device)
    }

    fn from_tensor(tensor: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let data = tensor.into_data();
        let slice = data.as_slice::<f32>().map_err(|e| TensorConversionError {
            message: format!("expected f32 action tensor: {e:?}"),
        })?;
        if slice.len() != 2 {
            return Err(TensorConversionError {
                message: format!("expected 2 action elements, got {}", slice.len()),
            });
        }
        Ok(Self([slice[0], slice[1]]))
    }
}

// ─── Observation ─────────────────────────────────────────────────────────

/// 10-dim observation. Layout:
/// `[cos θ₁, cos θ₂, sin θ₁, sin θ₂, target_x, target_y, θ̇₁, θ̇₂,
///   (finger − target)_x, (finger − target)_y]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReacherObservation(pub [f32; 10]);

impl ReacherObservation {
    #[must_use]
    pub const fn theta1_cos(&self) -> f32 {
        self.0[0]
    }
    #[must_use]
    pub const fn theta2_cos(&self) -> f32 {
        self.0[1]
    }
    #[must_use]
    pub const fn theta1_sin(&self) -> f32 {
        self.0[2]
    }
    #[must_use]
    pub const fn theta2_sin(&self) -> f32 {
        self.0[3]
    }
    #[must_use]
    pub const fn target_xy(&self) -> [f32; 2] {
        [self.0[4], self.0[5]]
    }
    #[must_use]
    pub const fn theta1_dot(&self) -> f32 {
        self.0[6]
    }
    #[must_use]
    pub const fn theta2_dot(&self) -> f32 {
        self.0[7]
    }
    #[must_use]
    pub const fn finger_minus_target_xy(&self) -> [f32; 2] {
        [self.0[8], self.0[9]]
    }

    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.0.iter().all(|v| v.is_finite())
    }
}

impl Default for ReacherObservation {
    fn default() -> Self {
        Self([0.0; 10])
    }
}

impl Observation<1> for ReacherObservation {
    fn shape() -> [usize; 1] {
        [10]
    }
}

impl<B: Backend> TensorConvertible<1, B> for ReacherObservation {
    fn to_tensor(&self, device: &B::Device) -> Tensor<B, 1> {
        Tensor::from_floats(self.0, device)
    }

    fn from_tensor(tensor: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let data = tensor.into_data();
        let slice = data.as_slice::<f32>().map_err(|e| TensorConversionError {
            message: format!("expected f32 observation tensor: {e:?}"),
        })?;
        if slice.len() != 10 {
            return Err(TensorConversionError {
                message: format!("expected 10 observation elements, got {}", slice.len()),
            });
        }
        let mut arr = [0.0f32; 10];
        arr.copy_from_slice(slice);
        Ok(Self(arr))
    }
}

// ─── State ───────────────────────────────────────────────────────────────

/// Physics state for [`Reacher`] — body + joint handles plus the cached
/// target position and last observation. The non-`Clone` world lives on the
/// env struct directly.
#[derive(Debug, Clone)]
pub struct ReacherState {
    pub link1: RigidBodyHandle,
    pub link2: RigidBodyHandle,
    pub target: RigidBodyHandle,
    pub shoulder: ImpulseJointHandle,
    pub elbow: ImpulseJointHandle,
    pub target_xy: [f32; 2],
    pub last_obs: ReacherObservation,
}

impl State<1> for ReacherState {
    type Observation = ReacherObservation;

    fn shape() -> [usize; 1] {
        [10]
    }

    fn is_valid(&self) -> bool {
        self.last_obs.is_finite()
    }

    fn observe(&self) -> ReacherObservation {
        self.last_obs
    }
}

// ─── Environment ─────────────────────────────────────────────────────────

/// Reacher — a 2-link planar arm whose fingertip must reach a randomly
/// placed target. Generic in the physics backend; v1 only implements
/// `B = Rapier3DBackend`.
#[derive(Debug)]
pub struct Reacher<B: LocomotionBackend = Rapier3DBackend> {
    world: B::World,
    state: ReacherState,
    config: ReacherConfig,
    rng: StdRng,
    steps: usize,
    _marker: PhantomData<B>,
}

/// Default backend alias.
pub type ReacherRapier = Reacher<Rapier3DBackend>;

impl Reacher<Rapier3DBackend> {
    /// Create with an explicit configuration.
    #[must_use]
    pub fn with_config(config: ReacherConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(config.seed);
        let (world, state) = Self::build_world(&config, &mut rng);
        Self { world, state, config, rng, steps: 0, _marker: PhantomData }
    }

    fn build_world(
        config: &ReacherConfig,
        rng: &mut StdRng,
    ) -> (Rapier3DWorld, ReacherState) {
        // Zero gravity — the arm is planar and must not droop in z (spec §9).
        let mut world = Rapier3DWorld::new(
            Vector::new(0.0, 0.0, 0.0),
            config.dt,
            config.frame_skip,
        );

        let n = config.reset_noise_scale;
        let theta1_init: f32 = rng.random_range(-n..=n);
        let theta2_init: f32 = rng.random_range(-n..=n);
        let theta1_dot_init: f32 = rng.random_range(-n..=n);
        let theta2_dot_init: f32 = rng.random_range(-n..=n);

        // Rejection-sample target position in a disk of radius `target_disk_radius`.
        let r = config.target_disk_radius;
        let (tx, ty) = loop {
            let x: f32 = rng.random_range(-r..=r);
            let y: f32 = rng.random_range(-r..=r);
            if x * x + y * y <= r * r {
                break (x, y);
            }
        };

        let half1 = config.link1_length * 0.5;
        let half2 = config.link2_length * 0.5;

        // Density from mass / capsule-volume so each body has a valid inertia
        // tensor (notes §2 — `additional_mass` would leave angular inertia zero).
        let capsule_volume = |half_len: f32, radius: f32| {
            std::f32::consts::PI * radius.powi(2) * (2.0 * half_len + (4.0 / 3.0) * radius)
        };
        let link1_density = config.link_mass / capsule_volume(half1, config.link_radius).max(f32::EPSILON);
        let link2_density = config.link_mass / capsule_volume(half2, config.link_radius).max(f32::EPSILON);

        // Root anchor — fixed body at the world origin serves as the shoulder pivot.
        let root = world.add_body(RigidBodyBuilder::fixed().translation(Vector::new(0.0, 0.0, 0.0)));

        // link1 position: body origin sits at the capsule midpoint, i.e.
        // shoulder + link1_rotation · (half1, 0, 0) in world frame.
        let c1 = theta1_init.cos();
        let s1 = theta1_init.sin();
        let link1_pos = Vector::new(half1 * c1, half1 * s1, 0.0);
        let link1_builder = RigidBodyBuilder::dynamic()
            .translation(link1_pos)
            .rotation(Vector::new(0.0, 0.0, theta1_init))
            .angvel(Vector::new(0.0, 0.0, theta1_dot_init))
            .enabled_translations(true, true, false)
            .enabled_rotations(false, false, true);
        let link1 = world.add_body(link1_builder);
        world.add_collider(
            ColliderBuilder::capsule_x(half1, config.link_radius).density(link1_density),
            link1,
        );

        // Shoulder revolute joint (impulse), axis = world-z.
        let z_axis: Vector = Vector::new(0.0, 0.0, 1.0);
        let shoulder_joint = RevoluteJointBuilder::new(z_axis)
            .local_anchor1(Vector::new(0.0, 0.0, 0.0))
            .local_anchor2(Vector::new(-half1, 0.0, 0.0))
            .build();
        let shoulder = world.add_impulse_joint(root, link1, shoulder_joint);

        // link2: absolute orientation is θ1 + θ2 (θ2 is the relative elbow
        // angle). Elbow world position = link1_body + link1_rot·(half1, 0, 0).
        let theta2_abs = theta1_init + theta2_init;
        let c12 = theta2_abs.cos();
        let s12 = theta2_abs.sin();
        let elbow_world = Vector::new(2.0 * half1 * c1, 2.0 * half1 * s1, 0.0);
        let link2_pos = Vector::new(
            elbow_world.x + half2 * c12,
            elbow_world.y + half2 * s12,
            0.0,
        );
        let link2_builder = RigidBodyBuilder::dynamic()
            .translation(link2_pos)
            .rotation(Vector::new(0.0, 0.0, theta2_abs))
            .angvel(Vector::new(0.0, 0.0, theta1_dot_init + theta2_dot_init))
            .enabled_translations(true, true, false)
            .enabled_rotations(false, false, true);
        let link2 = world.add_body(link2_builder);
        world.add_collider(
            ColliderBuilder::capsule_x(half2, config.link_radius).density(link2_density),
            link2,
        );

        // Elbow revolute joint (impulse), axis = world-z.
        let elbow_joint = RevoluteJointBuilder::new(z_axis)
            .local_anchor1(Vector::new(half1, 0.0, 0.0))
            .local_anchor2(Vector::new(-half2, 0.0, 0.0))
            .build();
        let elbow = world.add_impulse_joint(link1, link2, elbow_joint);

        // Target body — fixed, small ball at the sampled position.
        let target = world.add_body(
            RigidBodyBuilder::fixed().translation(Vector::new(tx, ty, 0.0)),
        );
        world.add_collider(ColliderBuilder::ball(0.01), target);

        let state = ReacherState {
            link1,
            link2,
            target,
            shoulder,
            elbow,
            target_xy: [tx, ty],
            last_obs: ReacherObservation::default(),
        };
        (world, state)
    }

    fn extract_observation(&self) -> ReacherObservation {
        let p1 = Rapier3DBackend::get_pose(&self.world, self.state.link1);
        let p2 = Rapier3DBackend::get_pose(&self.world, self.state.link2);
        let v1 = Rapier3DBackend::get_vel(&self.world, self.state.link1);
        let v2 = Rapier3DBackend::get_vel(&self.world, self.state.link2);

        // Pure rotation about world-z ⇒ quaternion = (cos(θ/2), 0, 0, sin(θ/2))
        // in [w, x, y, z] order. θ = 2·atan2(qz, qw).
        let [w1, _, _, z1] = p1.orientation;
        let [w2, _, _, z2] = p2.orientation;
        let theta1 = wrap_to_pi(2.0 * z1.atan2(w1));
        let theta2_world = wrap_to_pi(2.0 * z2.atan2(w2));
        let theta2 = wrap_to_pi(theta2_world - theta1); // relative (spec §2)

        // Fingertip in world frame: link2 body origin is the capsule midpoint;
        // the tip sits at body-local (+half2, 0, 0).
        let half2 = self.config.link2_length * 0.5;
        let fx = p2.position[0] + half2 * theta2_world.cos();
        let fy = p2.position[1] + half2 * theta2_world.sin();

        let [tx, ty] = self.state.target_xy;
        ReacherObservation([
            theta1.cos(),
            theta2.cos(),
            theta1.sin(),
            theta2.sin(),
            tx,
            ty,
            v1.angular[2],
            v2.angular[2] - v1.angular[2],
            fx - tx,
            fy - ty,
        ])
    }

    fn apply_action(&mut self, action: &ReacherAction) {
        let (lo, hi) = self.config.action_clip;
        let clipped = [action.0[0].clamp(lo, hi), action.0[1].clamp(lo, hi)];
        let torques = self.config.gear.apply(&clipped);
        // Shoulder torque τ[0] on link1 (root is fixed → no reaction needed).
        // Elbow torque ±τ[1] between link1 and link2 (Newton's third law).
        if let Some(body) = self.world.bodies_mut().get_mut(self.state.link1) {
            body.add_torque(Vector::new(0.0, 0.0, torques[0] - torques[1]), true);
        }
        if let Some(body) = self.world.bodies_mut().get_mut(self.state.link2) {
            body.add_torque(Vector::new(0.0, 0.0, torques[1]), true);
        }
    }
}

impl Environment<1, 1, 1> for Reacher<Rapier3DBackend> {
    type StateType = ReacherState;
    type ObservationType = ReacherObservation;
    type ActionType = ReacherAction;
    type RewardType = ScalarReward;
    type SnapshotType = LocomotionSnapshot<ReacherObservation>;

    fn new(_render: bool) -> Self {
        Self::with_config(ReacherConfig::default())
    }

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.rng = StdRng::seed_from_u64(self.config.seed);
        let (world, mut state) = Self::build_world(&self.config, &mut self.rng);
        self.world = world;
        state.last_obs = ReacherObservation::default();
        self.state = state;
        self.steps = 0;

        let obs = self.extract_observation();
        self.state.last_obs = obs;
        // Zero-reward initial snapshot; emit both reward components at 0.0 so
        // the Σ-components = reward invariant holds at step 0 (no action taken).
        let meta = SnapshotMetadata::new()
            .with(METADATA_KEY_REWARD_DISTANCE, 0.0)
            .with(METADATA_KEY_REWARD_CONTROL, 0.0)
            .with_position("target", [obs.target_xy()[0], obs.target_xy()[1], 0.0]);
        Ok(LocomotionSnapshot::running(obs, ScalarReward(0.0), meta))
    }

    fn step(
        &mut self,
        action: ReacherAction,
    ) -> Result<Self::SnapshotType, EnvironmentError> {
        if !action.0.iter().all(|v| v.is_finite()) {
            return Err(EnvironmentError::InvalidAction(format!(
                "Reacher action must be finite, got {:?}",
                action.0
            )));
        }

        self.apply_action(&action);
        Rapier3DBackend::step(&mut self.world);
        self.steps += 1;

        let obs = self.extract_observation();
        self.state.last_obs = obs;

        let [dx, dy] = obs.finger_minus_target_xy();
        let reward_distance: f32 = -(dx * dx + dy * dy).sqrt();
        // Clip to the bound-enforced action before computing ctrl cost so that
        // unclipped inputs don't inflate the cost.
        let (lo, hi) = self.config.action_clip;
        let clipped = [action.0[0].clamp(lo, hi), action.0[1].clamp(lo, hi)];
        let reward_control: f32 = -ctrl_cost(self.config.ctrl_cost_weight, &clipped);
        let total = reward_distance + reward_control;

        let status = if self.steps >= self.config.max_steps {
            EpisodeStatus::Truncated
        } else {
            EpisodeStatus::Running
        };

        // Fingertip world-xy = (finger − target) + target.
        let [tx, ty] = self.state.target_xy;
        let fx = dx + tx;
        let fy = dy + ty;

        let meta = SnapshotMetadata::new()
            .with(METADATA_KEY_REWARD_DISTANCE, reward_distance)
            .with(METADATA_KEY_REWARD_CONTROL, reward_control)
            .with_position("fingertip", [fx, fy, 0.0])
            .with_position("target", [tx, ty, 0.0]);
        Ok(LocomotionSnapshot::new(obs, ScalarReward(total), status, meta))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use evorl_core::environment::Snapshot;

    fn cfg(seed: u64) -> ReacherConfig {
        ReacherConfig { seed, ..Default::default() }
    }

    #[test]
    fn action_shape_and_validity() {
        assert_eq!(ReacherAction::shape(), [2]);
        assert!(ReacherAction::new(0.0, 0.0).is_valid());
        assert!(ReacherAction::new(1.0, -1.0).is_valid());
        assert!(!ReacherAction::new(1.5, 0.0).is_valid());
        assert!(!ReacherAction::new(f32::NAN, 0.0).is_valid());
    }

    #[test]
    fn observation_shape() {
        assert_eq!(ReacherObservation::shape(), [10]);
    }

    #[test]
    fn reset_returns_running() {
        let mut env = ReacherRapier::with_config(cfg(7));
        let snap = env.reset().unwrap();
        assert!(!snap.is_done());
        assert!(snap.observation().is_finite());
    }

    #[test]
    fn reward_decomposition_sums_to_total() {
        let mut env = ReacherRapier::with_config(cfg(1));
        env.reset().unwrap();
        for i in 0..20 {
            let a = ReacherAction::new(0.3 * (i as f32).sin(), -0.2);
            let snap = env.step(a).unwrap();
            let meta = snap.metadata().unwrap();
            let sum: f32 = meta.components.values().sum();
            assert!(
                (sum - snap.reward().0).abs() < 1e-5,
                "Σ components ({sum}) must equal reward ({}) at step {i}",
                snap.reward().0
            );
        }
    }

    #[test]
    fn ctrl_cost_scales_quadratically() {
        let a = [0.3f32, -0.5];
        let a2 = [0.6f32, -1.0];
        let c1 = ctrl_cost(0.1, &a);
        let c2 = ctrl_cost(0.1, &a2);
        assert!((c2 - 4.0 * c1).abs() < 1e-5);
    }

    #[test]
    fn determinism_across_reset() {
        let rollout = |actions: &[[f32; 2]]| {
            let mut env = ReacherRapier::with_config(cfg(123));
            env.reset().unwrap();
            let mut last = ReacherObservation::default();
            for a in actions {
                if let Ok(snap) = env.step(ReacherAction(*a)) {
                    last = *snap.observation();
                }
            }
            last
        };
        let actions = [[0.1, -0.2], [0.5, 0.3], [-0.4, 0.2], [0.0, 0.0]];
        assert_eq!(rollout(&actions), rollout(&actions));
    }

    #[test]
    fn init_noise_bounded() {
        for seed in 0..50 {
            let env = ReacherRapier::with_config(cfg(seed));
            let obs = env.state.last_obs;
            assert!(obs.is_finite(), "seed {seed} produced non-finite obs");
            let theta1 = obs.theta1_sin().atan2(obs.theta1_cos());
            let theta2 = obs.theta2_sin().atan2(obs.theta2_cos());
            // Reset noise is ±0.1; allow small float slack.
            assert!(theta1.abs() <= 0.1 + 1e-5, "seed {seed}: |θ1|={} > 0.1", theta1.abs());
            assert!(theta2.abs() <= 0.1 + 1e-5, "seed {seed}: |θ2|={} > 0.1", theta2.abs());
        }
    }

    #[test]
    fn truncates_at_max_steps() {
        let mut env = ReacherRapier::with_config(ReacherConfig {
            max_steps: 5,
            ..Default::default()
        });
        env.reset().unwrap();
        let mut status = EpisodeStatus::Running;
        for _ in 0..5 {
            let snap = env.step(ReacherAction::new(0.0, 0.0)).unwrap();
            status = snap.status();
        }
        assert_eq!(status, EpisodeStatus::Truncated);
    }

    #[test]
    fn invalid_action_is_error() {
        let mut env = ReacherRapier::with_config(ReacherConfig::default());
        env.reset().unwrap();
        let bad = ReacherAction::new(f32::NAN, 0.0);
        assert!(env.step(bad).is_err());
    }

    #[test]
    fn obs_is_finite_after_rollout() {
        let mut env = ReacherRapier::with_config(cfg(42));
        env.reset().unwrap();
        for i in 0..50 {
            let a = ReacherAction::new(
                0.5 * (i as f32 * 0.3).sin(),
                0.5 * (i as f32 * 0.4).cos(),
            );
            let snap = env.step(a).unwrap();
            assert!(snap.observation().is_finite(), "non-finite obs at step {i}");
            if snap.is_done() {
                break;
            }
        }
    }

    #[test]
    fn obs_layout_matches_spec() {
        let mut env = ReacherRapier::with_config(cfg(3));
        env.reset().unwrap();
        let snap = env.step(ReacherAction::new(0.2, -0.1)).unwrap();
        let obs = snap.observation().0;
        // cos² + sin² ≈ 1 for each joint
        assert!((obs[0].powi(2) + obs[2].powi(2) - 1.0).abs() < 1e-4);
        assert!((obs[1].powi(2) + obs[3].powi(2) - 1.0).abs() < 1e-4);
        // target_x/y == state cache
        assert!((obs[4] - env.state.target_xy[0]).abs() < 1e-6);
        assert!((obs[5] - env.state.target_xy[1]).abs() < 1e-6);
    }

    #[test]
    fn target_within_disk_on_reset() {
        for seed in 0..200 {
            let env = ReacherRapier::with_config(cfg(seed));
            let [tx, ty] = env.state.target_xy;
            let r2 = tx * tx + ty * ty;
            assert!(
                r2 <= env.config.target_disk_radius.powi(2) + 1e-6,
                "seed {seed}: target ({tx}, {ty}) outside disk radius {}",
                env.config.target_disk_radius
            );
        }
    }

    #[test]
    fn finger_minus_target_matches_positions() {
        let mut env = ReacherRapier::with_config(cfg(5));
        env.reset().unwrap();
        let snap = env.step(ReacherAction::new(0.1, -0.1)).unwrap();
        let obs = snap.observation();

        // Re-derive fingertip from link2's pose and the half-length offset.
        let p2 = Rapier3DBackend::get_pose(&env.world, env.state.link2);
        let theta2_world = wrap_to_pi(2.0 * p2.orientation[3].atan2(p2.orientation[0]));
        let half2 = env.config.link2_length * 0.5;
        let fx = p2.position[0] + half2 * theta2_world.cos();
        let fy = p2.position[1] + half2 * theta2_world.sin();
        let [tx, ty] = env.state.target_xy;

        let [dx, dy] = obs.finger_minus_target_xy();
        assert!((dx - (fx - tx)).abs() < 1e-5, "dx mismatch: obs {dx} vs computed {}", fx - tx);
        assert!((dy - (fy - ty)).abs() < 1e-5, "dy mismatch: obs {dy} vs computed {}", fy - ty);
    }

    #[test]
    fn reward_distance_is_nonpositive() {
        let mut env = ReacherRapier::with_config(cfg(11));
        env.reset().unwrap();
        for i in 0..50 {
            let a = ReacherAction::new(
                0.4 * (i as f32 * 0.17).sin(),
                -0.3 * (i as f32 * 0.23).cos(),
            );
            let snap = env.step(a).unwrap();
            let d = snap.metadata().unwrap().components[METADATA_KEY_REWARD_DISTANCE];
            assert!(d <= 0.0, "reward_distance must be ≤ 0, got {d} at step {i}");
        }
    }

    #[test]
    fn reward_control_is_nonpositive() {
        let mut env = ReacherRapier::with_config(cfg(13));
        env.reset().unwrap();
        for i in 0..50 {
            let a = ReacherAction::new(
                0.6 * (i as f32 * 0.31).cos(),
                0.9 * (i as f32 * 0.11).sin(),
            );
            let snap = env.step(a).unwrap();
            let c = snap.metadata().unwrap().components[METADATA_KEY_REWARD_CONTROL];
            assert!(c <= 0.0, "reward_control must be ≤ 0, got {c} at step {i}");
        }
    }
}
