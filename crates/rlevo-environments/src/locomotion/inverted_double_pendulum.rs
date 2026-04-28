//! # InvertedDoublePendulum-v5 (Rapier3D-backed)
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
//! Cart on a 1D slider plus **two chained poles**, both revolute-y. The agent
//! applies horizontal force to the cart to keep the tip of the upper pole as
//! close as possible to `y_tip = 2` (Gymnasium's convention, which maps to our
//! world-z axis). The pendulum is structurally identical to the shipped
//! [`super::inverted_pendulum`] with a second pole chained on top.
//!
//! * Cart: dynamic body, x-only translation, rotations locked.
//! * Pole1: dynamic capsule, revolute-y joint to cart. Mass from collider density.
//! * Pole2: dynamic capsule, revolute-y joint to pole1's top. Mass from density.
//! * Action: `Box(-1, 1, (1,))` — force target, scaled by `gear = [100]`.
//! * Observation (9-dim):
//!   `[cart_x, sin θ₁, sin θ₂, cos θ₁, cos θ₂, cart_vx, θ̇₁, θ̇₂, F_ext_x]`.
//!   θ₂ is the **relative** elbow angle (pole2 world − pole1 world), wrapped.
//! * Reward:
//!   `alive_bonus − 0.01·x_tip² − (y_tip − 2)² − 1e-3·|ω₁| − 5e-3·|ω₂|`,
//!   with `alive_bonus = 10.0` while healthy and `0` otherwise.
//! * Termination: `y_tip ≤ 1.0`, or non-finite state.
//! * Truncation: `max_steps = 1000`.
//!
//! ## Divergence from Gymnasium
//!
//! * `constraint_force_x` (`obs[8]`) is approximated by reading Rapier's
//!   aggregated contact force on pole2 (`Rapier3DBackend::contact_force`).
//!   MuJoCo's equivalent `cfrc_inv[0]` is a joint reaction force computed in
//!   generalised coordinates. Signs and rough magnitudes follow the same
//!   dynamics; absolute values will differ.
//! * `ω₂` is reported as world-frame angular velocity (not relative to pole1),
//!   matching MuJoCo's `qvel` for the second hinge — i.e. it is the body's
//!   absolute rate, not the rate of the relative joint angle.

use std::marker::PhantomData;

use burn::prelude::{Backend, Tensor};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};
use rapier3d::math::Vector;
use rapier3d::prelude::*;
use rlevo_core::action::ContinuousAction;
use rlevo_core::base::{Action, Observation, State, TensorConversionError, TensorConvertible};
use rlevo_core::environment::{Environment, EnvironmentError, EpisodeStatus, SnapshotMetadata};
use rlevo_core::reward::ScalarReward;
use serde::{Deserialize, Serialize};

use super::backend::{LocomotionBackend, Pose, Rapier3DBackend, Rapier3DWorld};
use super::common::{Gear, HealthyCheck, LocomotionSnapshot, TerminationMode, wrap_to_pi};

/// Reward-component key: alive bonus (`+10` while healthy, `0` otherwise).
pub const METADATA_KEY_ALIVE: &str = "alive";
/// Reward-component key: tip-distance penalty `−0.01·x_tip² − (y_tip − 2)²`.
pub const METADATA_KEY_DISTANCE: &str = "distance";
/// Reward-component key: angular-velocity penalty `−1e-3·|ω₁| − 5e-3·|ω₂|`.
pub const METADATA_KEY_VELOCITY: &str = "velocity";

// ─── Config ───────────────────────────────────────────────────────────────

/// Environment configuration for [`InvertedDoublePendulum`].
///
/// Defaults match the Gymnasium v5 XML: gear 100, dt 0.01, frame_skip 1,
/// reset noise 0.1, truncation at 1000, termination on `y_tip ≤ 1.0`.
#[derive(Debug, Clone)]
pub struct InvertedDoublePendulumConfig {
    pub seed: u64,
    pub gear: Gear<1>,
    pub dt: f32,
    pub frame_skip: u32,
    /// Gate on the tip's world-z (Gymnasium's `y_tip`). Default
    /// `z_range = Some((1.0, ∞))`.
    pub healthy: HealthyCheck,
    pub termination: TerminationMode,
    pub reset_noise_scale: f32,
    pub max_steps: usize,
    pub action_clip: (f32, f32),
    // Physical geometry
    pub cart_mass: f32,
    pub pole_mass: f32,
    /// Total length of one pole (capsule length = 2 · pole_half = `pole_length`).
    pub pole_length: f32,
    pub pole_radius: f32,
    pub cart_half_extents: [f32; 3],
    pub gravity: f32,
    // Reward weights
    pub alive_reward: f32,
    pub x_tip_weight: f32,
    pub y_tip_target: f32,
    pub omega1_weight: f32,
    pub omega2_weight: f32,
}

impl Default for InvertedDoublePendulumConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            gear: Gear::new([100.0]),
            dt: 0.01,
            frame_skip: 1,
            healthy: HealthyCheck {
                z_range: Some((1.0, f32::INFINITY)),
                ..HealthyCheck::none()
            },
            termination: TerminationMode::OnUnhealthy,
            reset_noise_scale: 0.1,
            max_steps: 1000,
            action_clip: (-1.0, 1.0),
            cart_mass: 10.0,
            pole_mass: 0.5,
            pole_length: 0.6,
            pole_radius: 0.045,
            cart_half_extents: [0.15, 0.05, 0.05],
            gravity: -9.81,
            alive_reward: 10.0,
            x_tip_weight: 0.01,
            y_tip_target: 2.0,
            omega1_weight: 1e-3,
            omega2_weight: 5e-3,
        }
    }
}

// ─── Action ──────────────────────────────────────────────────────────────

/// 1D continuous action — horizontal force target on the cart, in pre-gear
/// units. Bounds: `[-1.0, 1.0]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct InvertedDoublePendulumAction(pub [f32; 1]);

impl InvertedDoublePendulumAction {
    #[must_use]
    pub const fn new(force: f32) -> Self {
        Self([force])
    }
}

impl Action<1> for InvertedDoublePendulumAction {
    fn shape() -> [usize; 1] {
        [1]
    }

    fn is_valid(&self) -> bool {
        self.0[0].is_finite() && self.0[0].abs() <= 1.0
    }
}

impl ContinuousAction<1> for InvertedDoublePendulumAction {
    fn as_slice(&self) -> &[f32] {
        &self.0
    }

    fn clip(&self, min: f32, max: f32) -> Self {
        Self([self.0[0].clamp(min, max)])
    }

    fn from_slice(values: &[f32]) -> Self {
        Self([values[0]])
    }

    fn random() -> Self {
        Self([rand::random::<f32>() * 2.0 - 1.0])
    }
}

impl<B: Backend> TensorConvertible<1, B> for InvertedDoublePendulumAction {
    fn to_tensor(&self, device: &B::Device) -> Tensor<B, 1> {
        Tensor::from_floats(self.0, device)
    }

    fn from_tensor(tensor: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let data = tensor.into_data();
        let slice = data.as_slice::<f32>().map_err(|e| TensorConversionError {
            message: format!("expected f32 action tensor: {e:?}"),
        })?;
        if slice.len() != 1 {
            return Err(TensorConversionError {
                message: format!("expected 1 action element, got {}", slice.len()),
            });
        }
        Ok(Self([slice[0]]))
    }
}

// ─── Observation ─────────────────────────────────────────────────────────

/// 9-dim observation: `[cart_x, sin θ₁, sin θ₂, cos θ₁, cos θ₂, cart_vx,
/// θ̇₁, θ̇₂, constraint_force_x]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct InvertedDoublePendulumObservation(pub [f32; 9]);

impl InvertedDoublePendulumObservation {
    #[must_use]
    pub const fn cart_position(&self) -> f32 {
        self.0[0]
    }
    #[must_use]
    pub const fn sin_theta1(&self) -> f32 {
        self.0[1]
    }
    #[must_use]
    pub const fn sin_theta2(&self) -> f32 {
        self.0[2]
    }
    #[must_use]
    pub const fn cos_theta1(&self) -> f32 {
        self.0[3]
    }
    #[must_use]
    pub const fn cos_theta2(&self) -> f32 {
        self.0[4]
    }
    #[must_use]
    pub const fn cart_velocity(&self) -> f32 {
        self.0[5]
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
    pub const fn constraint_force_x(&self) -> f32 {
        self.0[8]
    }

    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.0.iter().all(|v| v.is_finite())
    }
}

impl Default for InvertedDoublePendulumObservation {
    fn default() -> Self {
        Self([0.0; 9])
    }
}

impl Observation<1> for InvertedDoublePendulumObservation {
    fn shape() -> [usize; 1] {
        [9]
    }
}

impl<B: Backend> TensorConvertible<1, B> for InvertedDoublePendulumObservation {
    fn to_tensor(&self, device: &B::Device) -> Tensor<B, 1> {
        Tensor::from_floats(self.0, device)
    }

    fn from_tensor(tensor: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let data = tensor.into_data();
        let slice = data.as_slice::<f32>().map_err(|e| TensorConversionError {
            message: format!("expected f32 observation tensor: {e:?}"),
        })?;
        if slice.len() != 9 {
            return Err(TensorConversionError {
                message: format!("expected 9 observation elements, got {}", slice.len()),
            });
        }
        let mut out = [0.0f32; 9];
        out.copy_from_slice(slice);
        Ok(Self(out))
    }
}

// ─── State ───────────────────────────────────────────────────────────────

/// Physics state: body/joint handles plus the last observation. The world
/// itself (non-`Clone`) lives on [`InvertedDoublePendulum`].
#[derive(Debug, Clone)]
pub struct InvertedDoublePendulumState {
    pub cart: RigidBodyHandle,
    pub pole1: RigidBodyHandle,
    pub pole2: RigidBodyHandle,
    pub joint1: ImpulseJointHandle,
    pub joint2: ImpulseJointHandle,
    pub last_obs: InvertedDoublePendulumObservation,
}

impl State<1> for InvertedDoublePendulumState {
    type Observation = InvertedDoublePendulumObservation;

    fn shape() -> [usize; 1] {
        [9]
    }

    fn is_valid(&self) -> bool {
        self.last_obs.is_finite()
    }

    fn observe(&self) -> InvertedDoublePendulumObservation {
        self.last_obs
    }
}

// ─── Environment ─────────────────────────────────────────────────────────

/// InvertedDoublePendulum — cart-pole-pole balance in 3D. Cart slides along
/// world-x; the two poles rotate about world-y joints (cart→pole1, pole1→pole2).
#[derive(Debug)]
pub struct InvertedDoublePendulum<B: LocomotionBackend = Rapier3DBackend> {
    world: B::World,
    state: InvertedDoublePendulumState,
    config: InvertedDoublePendulumConfig,
    rng: StdRng,
    steps: usize,
    _marker: PhantomData<B>,
}

/// Default backend alias.
pub type InvertedDoublePendulumRapier = InvertedDoublePendulum<Rapier3DBackend>;

impl InvertedDoublePendulum<Rapier3DBackend> {
    /// Create with an explicit configuration.
    #[must_use]
    pub fn with_config(config: InvertedDoublePendulumConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(config.seed);
        let (world, state) = Self::build_world(&config, &mut rng);
        Self {
            world,
            state,
            config,
            rng,
            steps: 0,
            _marker: PhantomData,
        }
    }

    fn build_world(
        config: &InvertedDoublePendulumConfig,
        rng: &mut StdRng,
    ) -> (Rapier3DWorld, InvertedDoublePendulumState) {
        let mut world = Rapier3DWorld::new(
            Vector::new(0.0, 0.0, config.gravity),
            config.dt,
            config.frame_skip,
        );

        // Reset-noise sampling:
        //   qpos (cart_x, θ₁, θ₂) ~ U(-scale, scale)
        //   qvel (cart_vx, θ̇₁, θ̇₂) ~ N(0, scale)
        let n = config.reset_noise_scale;
        let init_cart_x: f32 = rng.random_range(-n..=n);
        let init_theta1: f32 = rng.random_range(-n..=n);
        let init_theta2: f32 = rng.random_range(-n..=n);
        let normal = Normal::new(0.0_f32, n).expect("reset_noise_scale must be finite");
        let init_cart_vx: f32 = normal.sample(rng);
        let init_omega1: f32 = normal.sample(rng);
        let init_omega2: f32 = normal.sample(rng);

        let cart_z = config.cart_half_extents[2];
        let pole_half = config.pole_length * 0.5;

        // Cart — density-driven mass so the inertia tensor is populated even
        // though rotation is DOF-locked (keeps joint reactions well-conditioned).
        let cart_volume = config.cart_half_extents[0]
            * config.cart_half_extents[1]
            * config.cart_half_extents[2]
            * 8.0;
        let cart_density = config.cart_mass / cart_volume.max(f32::EPSILON);
        let cart_builder = RigidBodyBuilder::dynamic()
            .translation(Vector::new(init_cart_x, 0.0, cart_z))
            .linvel(Vector::new(init_cart_vx, 0.0, 0.0))
            .enabled_translations(true, false, false)
            .enabled_rotations(false, false, false);
        let cart = world.add_body(cart_builder);
        world.add_collider(
            ColliderBuilder::cuboid(
                config.cart_half_extents[0],
                config.cart_half_extents[1],
                config.cart_half_extents[2],
            )
            .density(cart_density),
            cart,
        );

        let pole_volume = std::f32::consts::PI
            * config.pole_radius.powi(2)
            * (2.0 * pole_half + (4.0 / 3.0) * config.pole_radius);
        let pole_density = config.pole_mass / pole_volume.max(f32::EPSILON);

        // Pole1 — revolute-y joint to cart top. Initial orientation applied via
        // axis-angle (`Vector::new(0, θ, 0)`). Revolute DOF gated to y-only.
        let cart_half_z = config.cart_half_extents[2];
        let pole1_center_z = cart_z + cart_half_z + pole_half;
        let pole1_builder = RigidBodyBuilder::dynamic()
            .translation(Vector::new(init_cart_x, 0.0, pole1_center_z))
            .rotation(Vector::new(0.0, init_theta1, 0.0))
            .angvel(Vector::new(0.0, init_omega1, 0.0))
            .enabled_translations(true, true, true)
            .enabled_rotations(false, true, false);
        let pole1 = world.add_body(pole1_builder);
        world.add_collider(
            ColliderBuilder::capsule_z(pole_half, config.pole_radius).density(pole_density),
            pole1,
        );

        // Pole2 — revolute-y joint to pole1's top. Initial orientation applied
        // as absolute rotation (θ₁ + θ₂) so the unloaded chain forms a straight
        // arm at θ_rel = θ₂ when both noise samples are zero.
        let pole2_abs_angle = init_theta1 + init_theta2;
        let pole2_center_z = pole1_center_z + 2.0 * pole_half;
        let pole2_builder = RigidBodyBuilder::dynamic()
            .translation(Vector::new(init_cart_x, 0.0, pole2_center_z))
            .rotation(Vector::new(0.0, pole2_abs_angle, 0.0))
            .angvel(Vector::new(0.0, init_omega1 + init_omega2, 0.0))
            .enabled_translations(true, true, true)
            .enabled_rotations(false, true, false);
        let pole2 = world.add_body(pole2_builder);
        world.add_collider(
            ColliderBuilder::capsule_z(pole_half, config.pole_radius).density(pole_density),
            pole2,
        );

        let y_axis: Vector = Vector::new(0.0, 1.0, 0.0);
        let joint1 = RevoluteJointBuilder::new(y_axis)
            .local_anchor1(Vector::new(0.0, 0.0, cart_half_z))
            .local_anchor2(Vector::new(0.0, 0.0, -pole_half))
            .build();
        let joint1_handle = world.add_impulse_joint(cart, pole1, joint1);

        let joint2 = RevoluteJointBuilder::new(y_axis)
            .local_anchor1(Vector::new(0.0, 0.0, pole_half))
            .local_anchor2(Vector::new(0.0, 0.0, -pole_half))
            .build();
        let joint2_handle = world.add_impulse_joint(pole1, pole2, joint2);

        let state = InvertedDoublePendulumState {
            cart,
            pole1,
            pole2,
            joint1: joint1_handle,
            joint2: joint2_handle,
            last_obs: InvertedDoublePendulumObservation::default(),
        };
        (world, state)
    }

    fn extract_observation(&self) -> InvertedDoublePendulumObservation {
        let cart_pose = Rapier3DBackend::get_pose(&self.world, self.state.cart);
        let cart_vel = Rapier3DBackend::get_vel(&self.world, self.state.cart);
        let pole1_pose = Rapier3DBackend::get_pose(&self.world, self.state.pole1);
        let pole1_vel = Rapier3DBackend::get_vel(&self.world, self.state.pole1);
        let pole2_pose = Rapier3DBackend::get_pose(&self.world, self.state.pole2);
        let pole2_vel = Rapier3DBackend::get_vel(&self.world, self.state.pole2);

        let theta1 = pole_y_angle(&pole1_pose);
        let theta2_abs = pole_y_angle(&pole2_pose);
        // θ₂ is the **relative** elbow angle (pole2 − pole1).
        let theta2 = wrap_to_pi(theta2_abs - theta1);

        let cfrc_ext = Rapier3DBackend::contact_force(&self.world, self.state.pole2);

        InvertedDoublePendulumObservation([
            cart_pose.position[0],
            theta1.sin(),
            theta2.sin(),
            theta1.cos(),
            theta2.cos(),
            cart_vel.linear[0],
            pole1_vel.angular[1],
            pole2_vel.angular[1],
            cfrc_ext[0],
        ])
    }

    fn apply_action(&mut self, action: &InvertedDoublePendulumAction) {
        let (lo, hi) = self.config.action_clip;
        let clipped = [action.0[0].clamp(lo, hi)];
        let torques = self.config.gear.apply(&clipped);
        let force = torques[0];
        if let Some(cart) = self.world.bodies_mut().get_mut(self.state.cart) {
            cart.add_force(Vector::new(force, 0.0, 0.0), true);
        }
    }

    /// World-frame position of the tip (upper end of pole2).
    fn tip_position(&self) -> [f32; 3] {
        let pole2_pose = Rapier3DBackend::get_pose(&self.world, self.state.pole2);
        let pole_half = self.config.pole_length * 0.5;
        let local_tip = [0.0f32, 0.0, pole_half];
        let rotated = rotate_by_quat(pole2_pose.orientation, local_tip);
        [
            pole2_pose.position[0] + rotated[0],
            pole2_pose.position[1] + rotated[1],
            pole2_pose.position[2] + rotated[2],
        ]
    }
}

impl Environment<1, 1, 1> for InvertedDoublePendulum<Rapier3DBackend> {
    type StateType = InvertedDoublePendulumState;
    type ObservationType = InvertedDoublePendulumObservation;
    type ActionType = InvertedDoublePendulumAction;
    type RewardType = ScalarReward;
    type SnapshotType = LocomotionSnapshot<InvertedDoublePendulumObservation>;

    fn new(_render: bool) -> Self {
        Self::with_config(InvertedDoublePendulumConfig::default())
    }

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.rng = StdRng::seed_from_u64(self.config.seed);
        let (world, mut state) = Self::build_world(&self.config, &mut self.rng);
        self.world = world;
        state.last_obs = InvertedDoublePendulumObservation::default();
        self.state = state;
        self.steps = 0;

        let obs = self.extract_observation();
        self.state.last_obs = obs;

        let tip = self.tip_position();
        let meta = SnapshotMetadata::new()
            .with(METADATA_KEY_ALIVE, 0.0)
            .with(METADATA_KEY_DISTANCE, 0.0)
            .with(METADATA_KEY_VELOCITY, 0.0)
            .with_position(
                "cart",
                [obs.cart_position(), 0.0, self.config.cart_half_extents[2]],
            )
            .with_position("tip", tip);
        Ok(LocomotionSnapshot::running(obs, ScalarReward(0.0), meta))
    }

    fn step(
        &mut self,
        action: InvertedDoublePendulumAction,
    ) -> Result<Self::SnapshotType, EnvironmentError> {
        if !action.0[0].is_finite() {
            return Err(EnvironmentError::InvalidAction(format!(
                "InvertedDoublePendulum action must be finite, got {}",
                action.0[0]
            )));
        }

        self.apply_action(&action);
        Rapier3DBackend::step(&mut self.world);
        self.steps += 1;

        let obs = self.extract_observation();
        self.state.last_obs = obs;

        let tip = self.tip_position();
        let x_tip = tip[0];
        let y_tip = tip[2];

        // Healthy check: gate on tip-z (Gymnasium's y_tip). Reuse HealthyCheck's
        // `z_range` argument for the tip height — angle / state are unused.
        let healthy = self.config.healthy.is_healthy(y_tip, 0.0, &obs.0);

        let alive = if healthy {
            self.config.alive_reward
        } else {
            0.0
        };
        let distance =
            -self.config.x_tip_weight * x_tip * x_tip - (y_tip - self.config.y_tip_target).powi(2);
        let omega1 = obs.theta1_dot().abs();
        let omega2 = obs.theta2_dot().abs();
        let velocity = -self.config.omega1_weight * omega1 - self.config.omega2_weight * omega2;
        let total = alive + distance + velocity;

        let status = if !healthy && matches!(self.config.termination, TerminationMode::OnUnhealthy)
        {
            EpisodeStatus::Terminated
        } else if self.steps >= self.config.max_steps {
            EpisodeStatus::Truncated
        } else {
            EpisodeStatus::Running
        };

        let meta = SnapshotMetadata::new()
            .with(METADATA_KEY_ALIVE, alive)
            .with(METADATA_KEY_DISTANCE, distance)
            .with(METADATA_KEY_VELOCITY, velocity)
            .with_position(
                "cart",
                [obs.cart_position(), 0.0, self.config.cart_half_extents[2]],
            )
            .with_position("tip", tip);
        Ok(LocomotionSnapshot::new(
            obs,
            ScalarReward(total),
            status,
            meta,
        ))
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────

/// Extract the y-axis rotation angle from a pose whose body is DOF-gated to
/// revolute-y. Quaternion stored as `[w, x, y, z]`.
fn pole_y_angle(pose: &Pose) -> f32 {
    let [w, _, y, _] = pose.orientation;
    wrap_to_pi(2.0 * y.atan2(w))
}

/// Rotate a vector by a unit quaternion `[w, x, y, z]`.
///
/// Uses the standard `v' = v + 2·u × (u × v + w·v)` identity where
/// `u = (x, y, z)`. Avoids pulling glam / nalgebra into the env layer.
fn rotate_by_quat(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let [w, x, y, z] = q;
    let [vx, vy, vz] = v;
    // t = 2 · (u × v)
    let tx = 2.0 * (y * vz - z * vy);
    let ty = 2.0 * (z * vx - x * vz);
    let tz = 2.0 * (x * vy - y * vx);
    // v' = v + w·t + u × t
    [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlevo_core::environment::Snapshot;

    fn deterministic_cfg() -> InvertedDoublePendulumConfig {
        InvertedDoublePendulumConfig {
            seed: 7,
            reset_noise_scale: 0.0,
            ..Default::default()
        }
    }

    #[test]
    fn action_shape_and_validity() {
        assert_eq!(InvertedDoublePendulumAction::shape(), [1]);
        assert!(InvertedDoublePendulumAction::new(0.0).is_valid());
        assert!(InvertedDoublePendulumAction::new(1.0).is_valid());
        assert!(!InvertedDoublePendulumAction::new(1.5).is_valid());
        assert!(!InvertedDoublePendulumAction::new(f32::NAN).is_valid());
    }

    #[test]
    fn observation_shape() {
        assert_eq!(InvertedDoublePendulumObservation::shape(), [9]);
    }

    #[test]
    fn rotate_by_quat_identity() {
        let v = rotate_by_quat([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        assert!((v[0] - 0.0).abs() < 1e-6);
        assert!((v[1] - 0.0).abs() < 1e-6);
        assert!((v[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rotate_by_quat_90deg_about_y() {
        // q = (cos(π/4), 0, sin(π/4), 0) rotates +z → +x.
        let c = (std::f32::consts::FRAC_PI_4).cos();
        let s = (std::f32::consts::FRAC_PI_4).sin();
        let v = rotate_by_quat([c, 0.0, s, 0.0], [0.0, 0.0, 1.0]);
        assert!((v[0] - 1.0).abs() < 1e-5, "expected (1,0,0), got {v:?}");
        assert!(v[1].abs() < 1e-5);
        assert!(v[2].abs() < 1e-5);
    }

    #[test]
    fn reset_returns_running_with_expected_obs_layout() {
        let mut env = InvertedDoublePendulumRapier::with_config(deterministic_cfg());
        let snap = env.reset().unwrap();
        assert!(!snap.is_done());
        let obs = snap.observation();
        // With zero reset noise both poles start upright: sin≈0, cos≈1.
        assert!(obs.cart_position().abs() < 1e-5);
        assert!(obs.sin_theta1().abs() < 1e-5);
        assert!(obs.sin_theta2().abs() < 1e-5);
        assert!((obs.cos_theta1() - 1.0).abs() < 1e-5);
        assert!((obs.cos_theta2() - 1.0).abs() < 1e-5);
        assert!(obs.cart_velocity().abs() < 1e-5);
        assert!(obs.theta1_dot().abs() < 1e-5);
        assert!(obs.theta2_dot().abs() < 1e-5);
        assert!(obs.constraint_force_x().is_finite());
    }

    #[test]
    fn reward_decomposition_sums_to_total() {
        let mut env = InvertedDoublePendulumRapier::with_config(InvertedDoublePendulumConfig {
            seed: 11,
            ..Default::default()
        });
        env.reset().unwrap();
        for i in 0..100 {
            // Deterministic, non-trivial action trace.
            let a = ((i as f32) * 0.17).sin() * 0.5;
            let snap = env.step(InvertedDoublePendulumAction::new(a)).unwrap();
            let meta = snap.metadata().unwrap();
            let sum: f32 = meta.components.values().sum();
            assert!(
                (sum - snap.reward().0).abs() < 1e-5,
                "components sum ({sum}) must equal reward ({}) at step {i}",
                snap.reward().0
            );
            if snap.is_done() {
                break;
            }
        }
    }

    #[test]
    fn alive_bonus_paid_only_while_healthy() {
        let mut env = InvertedDoublePendulumRapier::with_config(deterministic_cfg());
        env.reset().unwrap();
        let snap = env.step(InvertedDoublePendulumAction::new(0.0)).unwrap();
        let alive = snap.metadata().unwrap().components[METADATA_KEY_ALIVE];
        assert!(
            (alive - 10.0).abs() < 1e-5,
            "expected alive=10 while healthy, got {alive}"
        );

        // Drive the cart hard in one direction until the tip falls below 1.0.
        let mut terminated_alive = None;
        for _ in 0..2000 {
            let snap = env.step(InvertedDoublePendulumAction::new(1.0)).unwrap();
            if snap.is_terminated() {
                terminated_alive = Some(snap.metadata().unwrap().components[METADATA_KEY_ALIVE]);
                break;
            }
        }
        assert_eq!(
            terminated_alive,
            Some(0.0),
            "alive must be 0 once unhealthy"
        );
    }

    #[test]
    fn tip_height_terminates() {
        let mut env = InvertedDoublePendulumRapier::with_config(InvertedDoublePendulumConfig {
            reset_noise_scale: 0.0,
            max_steps: 2000,
            ..Default::default()
        });
        env.reset().unwrap();
        let mut terminated = false;
        let mut min_y_tip = f32::INFINITY;
        for _ in 0..500 {
            let snap = env.step(InvertedDoublePendulumAction::new(1.0)).unwrap();
            if let Some(meta) = snap.metadata()
                && let Some(pos) = meta.positions.get("tip")
            {
                min_y_tip = min_y_tip.min(pos[2]);
            }
            if snap.is_terminated() {
                terminated = true;
                break;
            }
        }
        assert!(
            terminated,
            "sustained +1.0 action must drop tip below 1.0 within 500 steps; min y_tip={min_y_tip}"
        );
    }

    #[test]
    fn constraint_force_is_finite() {
        let mut env = InvertedDoublePendulumRapier::with_config(deterministic_cfg());
        env.reset().unwrap();
        for _ in 0..5 {
            let snap = env.step(InvertedDoublePendulumAction::new(0.5)).unwrap();
            assert!(
                snap.observation().constraint_force_x().is_finite(),
                "constraint_force_x must always be finite"
            );
        }
    }

    #[test]
    fn truncates_at_max_steps() {
        let mut env = InvertedDoublePendulumRapier::with_config(InvertedDoublePendulumConfig {
            seed: 1,
            reset_noise_scale: 0.0,
            max_steps: 5,
            termination: TerminationMode::Never,
            ..Default::default()
        });
        env.reset().unwrap();
        let mut status = EpisodeStatus::Running;
        for _ in 0..5 {
            status = env
                .step(InvertedDoublePendulumAction::new(0.0))
                .unwrap()
                .status();
        }
        assert_eq!(status, EpisodeStatus::Truncated);
    }

    #[test]
    fn determinism_across_reset() {
        let cfg = InvertedDoublePendulumConfig {
            seed: 123,
            ..Default::default()
        };
        let rollout = |actions: &[f32]| {
            let mut env = InvertedDoublePendulumRapier::with_config(cfg.clone());
            env.reset().unwrap();
            let mut last = InvertedDoublePendulumObservation::default();
            for &a in actions {
                if let Ok(snap) = env.step(InvertedDoublePendulumAction::new(a)) {
                    last = *snap.observation();
                    if snap.is_done() {
                        break;
                    }
                }
            }
            last
        };
        let actions = [0.0, 0.3, -0.4, 0.2, 0.1];
        assert_eq!(rollout(&actions), rollout(&actions));
    }

    #[test]
    fn invalid_action_is_error() {
        let mut env = InvertedDoublePendulumRapier::with_config(deterministic_cfg());
        env.reset().unwrap();
        assert!(
            env.step(InvertedDoublePendulumAction::new(f32::NAN))
                .is_err()
        );
    }

    #[test]
    fn obs_is_finite_after_rollout() {
        let mut env = InvertedDoublePendulumRapier::with_config(InvertedDoublePendulumConfig {
            seed: 2,
            ..Default::default()
        });
        env.reset().unwrap();
        for _ in 0..50 {
            let snap = env.step(InvertedDoublePendulumAction::new(0.1)).unwrap();
            assert!(snap.observation().is_finite());
            if snap.is_done() {
                break;
            }
        }
    }

    #[test]
    fn action_clip_at_boundaries() {
        let a = InvertedDoublePendulumAction::new(10.0).clip(-1.0, 1.0);
        assert_eq!(a.0[0], 1.0);
        let a = InvertedDoublePendulumAction::new(-10.0).clip(-1.0, 1.0);
        assert_eq!(a.0[0], -1.0);
    }
}
