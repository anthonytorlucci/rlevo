//! # InvertedPendulum-v5 (Rapier3D-backed)
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
//! * Cart: dynamic body restricted to translation along the world-x axis.
//!   Rotations and y/z translations are locked via rapier3d's per-axis DOF
//!   gate, so the body behaves like a single-DOF slider.
//! * Pole: dynamic capsule attached to the cart by a revolute impulse joint
//!   about the world-y axis. Gravity pulls it down; the agent's job is to
//!   balance it upright by sliding the cart.
//! * Action: `Box(-3, 3, (1,))` — force target; applied as `action · gear`
//!   with `gear = [100]` (Gymnasium XML) directly to the cart.
//! * Observation: `[cart_x, pole_angle, cart_vx, pole_angvel_y]` (4-dim).
//! * Reward: `+1.0` per step while the pole is healthy; `0.0` otherwise.
//! * Termination: `|pole_angle| >= 0.2 rad`, or non-finite state.
//! * Truncation: `max_steps = 1000`.

use std::marker::PhantomData;

use burn::prelude::{Backend, Tensor};
use evorl_core::action::ContinuousAction;
use evorl_core::base::{Action, Observation, State, TensorConversionError, TensorConvertible};
use evorl_core::environment::{Environment, EnvironmentError, EpisodeStatus, SnapshotMetadata};
use evorl_core::reward::ScalarReward;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rapier3d::prelude::*;
use serde::{Deserialize, Serialize};

use super::backend::{LocomotionBackend, Rapier3DBackend, Rapier3DWorld};
use super::common::{Gear, HealthyCheck, LocomotionSnapshot, TerminationMode, wrap_to_pi};

/// Reward-component metadata key: `+1` if alive at this step, `0` otherwise.
pub const METADATA_KEY_ALIVE: &str = "alive";

// ─── Config ───────────────────────────────────────────────────────────────

/// Environment configuration for [`InvertedPendulum`].
///
/// Defaults match the Gymnasium v5 XML (gear 100, dt 0.01, frame_skip 1,
/// healthy-angle band `(-0.2, 0.2)`, reset noise 0.01, truncation at 1000).
#[derive(Debug, Clone)]
pub struct InvertedPendulumConfig {
    pub seed: u64,
    pub gear: Gear<1>,
    pub dt: f32,
    pub frame_skip: u32,
    pub healthy: HealthyCheck,
    pub termination: TerminationMode,
    pub reset_noise_scale: f32,
    pub max_steps: usize,
    pub action_clip: (f32, f32),
    // Physical geometry
    pub cart_mass: f32,
    pub pole_mass: f32,
    pub pole_length: f32,
    pub pole_radius: f32,
    pub cart_half_extents: [f32; 3],
    pub gravity: f32,
}

impl Default for InvertedPendulumConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            gear: Gear::new([100.0]),
            dt: 0.01,
            frame_skip: 1,
            healthy: HealthyCheck { angle_range: Some((-0.2, 0.2)), ..HealthyCheck::none() },
            termination: TerminationMode::OnUnhealthy,
            reset_noise_scale: 0.01,
            max_steps: 1000,
            action_clip: (-3.0, 3.0),
            cart_mass: 10.0,
            pole_mass: 1.0,
            pole_length: 0.6,
            pole_radius: 0.05,
            cart_half_extents: [0.15, 0.05, 0.05],
            gravity: -9.81,
        }
    }
}

// ─── Action ──────────────────────────────────────────────────────────────

/// 1D continuous action — horizontal force target on the cart, in Gymnasium's
/// pre-gear units. Bounds: `[-3.0, 3.0]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct InvertedPendulumAction(pub [f32; 1]);

impl InvertedPendulumAction {
    /// Construct from a scalar convenience.
    #[must_use]
    pub const fn new(force: f32) -> Self {
        Self([force])
    }
}

impl Action<1> for InvertedPendulumAction {
    fn shape() -> [usize; 1] {
        [1]
    }

    fn is_valid(&self) -> bool {
        self.0[0].is_finite() && self.0[0].abs() <= 3.0
    }
}

impl ContinuousAction<1> for InvertedPendulumAction {
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
        Self([rand::random::<f32>() * 6.0 - 3.0])
    }
}

impl<B: Backend> TensorConvertible<1, B> for InvertedPendulumAction {
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

/// 4-dim observation: `[cart_x, pole_angle, cart_vx, pole_angvel_y]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct InvertedPendulumObservation(pub [f32; 4]);

impl InvertedPendulumObservation {
    #[must_use]
    pub const fn cart_position(&self) -> f32 {
        self.0[0]
    }
    #[must_use]
    pub const fn pole_angle(&self) -> f32 {
        self.0[1]
    }
    #[must_use]
    pub const fn cart_velocity(&self) -> f32 {
        self.0[2]
    }
    #[must_use]
    pub const fn pole_angular_velocity(&self) -> f32 {
        self.0[3]
    }

    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.0.iter().all(|v| v.is_finite())
    }
}

impl Default for InvertedPendulumObservation {
    fn default() -> Self {
        Self([0.0; 4])
    }
}

impl Observation<1> for InvertedPendulumObservation {
    fn shape() -> [usize; 1] {
        [4]
    }
}

impl<B: Backend> TensorConvertible<1, B> for InvertedPendulumObservation {
    fn to_tensor(&self, device: &B::Device) -> Tensor<B, 1> {
        Tensor::from_floats(self.0, device)
    }

    fn from_tensor(tensor: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let data = tensor.into_data();
        let slice = data.as_slice::<f32>().map_err(|e| TensorConversionError {
            message: format!("expected f32 observation tensor: {e:?}"),
        })?;
        if slice.len() != 4 {
            return Err(TensorConversionError {
                message: format!("expected 4 observation elements, got {}", slice.len()),
            });
        }
        Ok(Self([slice[0], slice[1], slice[2], slice[3]]))
    }
}

// ─── State ───────────────────────────────────────────────────────────────

/// Physics state for [`InvertedPendulum`] — keeps body handles and the last
/// observation, not the (non-`Clone`) world itself.
#[derive(Debug, Clone)]
pub struct InvertedPendulumState {
    pub cart: RigidBodyHandle,
    pub pole: RigidBodyHandle,
    pub joint: ImpulseJointHandle,
    pub last_obs: InvertedPendulumObservation,
}

impl State<1> for InvertedPendulumState {
    type Observation = InvertedPendulumObservation;

    fn shape() -> [usize; 1] {
        [4]
    }

    fn is_valid(&self) -> bool {
        self.last_obs.is_finite()
    }

    fn observe(&self) -> InvertedPendulumObservation {
        self.last_obs
    }
}

// ─── Environment ─────────────────────────────────────────────────────────

/// InvertedPendulum — cart-pole balance in 3D, with the cart restricted to
/// the world-x axis and the pole free to rotate about the world-y axis.
///
/// Generic in the physics backend; v1 only implements `B = Rapier3DBackend`
/// (see [`InvertedPendulumRapier`] for the default type alias).
#[derive(Debug)]
pub struct InvertedPendulum<B: LocomotionBackend = Rapier3DBackend> {
    world: B::World,
    state: InvertedPendulumState,
    config: InvertedPendulumConfig,
    rng: StdRng,
    steps: usize,
    _marker: PhantomData<B>,
}

/// Default backend alias.
pub type InvertedPendulumRapier = InvertedPendulum<Rapier3DBackend>;

impl InvertedPendulum<Rapier3DBackend> {
    /// Create with an explicit configuration.
    #[must_use]
    pub fn with_config(config: InvertedPendulumConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(config.seed);
        let (world, state) = Self::build_world(&config, &mut rng);
        Self { world, state, config, rng, steps: 0, _marker: PhantomData }
    }

    fn build_world(
        config: &InvertedPendulumConfig,
        rng: &mut StdRng,
    ) -> (Rapier3DWorld, InvertedPendulumState) {
        let mut world = Rapier3DWorld::new(
            Vector::new(0.0, 0.0, config.gravity),
            config.dt,
            config.frame_skip,
        );

        // Reset-noise sampling — Gymnasium uses U(-scale, scale) on qpos/qvel.
        let n = config.reset_noise_scale;
        let init_cart_x: f32 = rng.random_range(-n..=n);
        let init_angle: f32 = rng.random_range(-n..=n);
        let init_cart_vx: f32 = rng.random_range(-n..=n);
        let init_pole_angvel: f32 = rng.random_range(-n..=n);

        let cart_z = config.cart_half_extents[2]; // rest cart on z = half-height
        let pole_half = config.pole_length * 0.5;

        // Cart: dynamic, x-only translation, no rotation. Mass is derived from
        // the collider's density × volume so the body has a valid inertia tensor
        // (important for the attached pole's joint reactions).
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

        // Pole: dynamic, only rotation about y is enabled; attached to cart by
        // a revolute joint one frame-anchor length above the cart's origin.
        // Mass comes from the collider density so the inertia tensor is
        // populated (a point mass without a tensor would refuse to rotate).
        let pole_initial_z = cart_z + cart_half_z(config) + pole_half;
        let pole_volume = std::f32::consts::PI
            * config.pole_radius.powi(2)
            * (2.0 * pole_half + (4.0 / 3.0) * config.pole_radius);
        let pole_density = config.pole_mass / pole_volume.max(f32::EPSILON);
        let pole_builder = RigidBodyBuilder::dynamic()
            .translation(Vector::new(init_cart_x, 0.0, pole_initial_z))
            // AxisAngle (scaled-axis form) — rotate about world-y by `init_angle`.
            .rotation(Vector::new(0.0, init_angle, 0.0))
            .angvel(Vector::new(0.0, init_pole_angvel, 0.0))
            .enabled_translations(true, true, true)
            .enabled_rotations(false, true, false);
        let pole = world.add_body(pole_builder);
        world.add_collider(
            ColliderBuilder::capsule_z(pole_half, config.pole_radius).density(pole_density),
            pole,
        );

        // Revolute joint about the y-axis. Local anchor on cart is top face;
        // local anchor on pole is its bottom (i.e. -pole_half along local z).
        let y_axis: Vector = Vector::new(0.0, 1.0, 0.0);
        let joint = RevoluteJointBuilder::new(y_axis)
            .local_anchor1(Vector::new(0.0, 0.0, config.cart_half_extents[2]))
            .local_anchor2(Vector::new(0.0, 0.0, -pole_half))
            .build();
        let joint_handle = world.add_impulse_joint(cart, pole, joint);

        let state = InvertedPendulumState {
            cart,
            pole,
            joint: joint_handle,
            last_obs: InvertedPendulumObservation::default(),
        };
        (world, state)
    }

    fn extract_observation(&self) -> InvertedPendulumObservation {
        let cart_pose = Rapier3DBackend::get_pose(&self.world, self.state.cart);
        let cart_vel = Rapier3DBackend::get_vel(&self.world, self.state.cart);
        let pole_pose = Rapier3DBackend::get_pose(&self.world, self.state.pole);
        let pole_vel = Rapier3DBackend::get_vel(&self.world, self.state.pole);

        // Pole orientation is pure rotation about world-y. Its quaternion is
        // `(cos(θ/2), 0, sin(θ/2), 0)` in `[w, x, y, z]` order. Recover θ:
        let [w, _, y, _] = pole_pose.orientation;
        let pole_angle = 2.0 * y.atan2(w);
        // Normalise to (-π, π].
        let pole_angle = wrap_to_pi(pole_angle);

        InvertedPendulumObservation([
            cart_pose.position[0],
            pole_angle,
            cart_vel.linear[0],
            pole_vel.angular[1],
        ])
    }

    fn apply_action(&mut self, action: &InvertedPendulumAction) {
        let (lo, hi) = self.config.action_clip;
        let clipped = [action.0[0].clamp(lo, hi)];
        let torques = self.config.gear.apply(&clipped);
        let force = torques[0];
        if let Some(cart) = self.world.bodies_mut().get_mut(self.state.cart) {
            cart.add_force(Vector::new(force, 0.0, 0.0), true);
        }
    }
}

impl Environment<1, 1, 1> for InvertedPendulum<Rapier3DBackend> {
    type StateType = InvertedPendulumState;
    type ObservationType = InvertedPendulumObservation;
    type ActionType = InvertedPendulumAction;
    type RewardType = ScalarReward;
    type SnapshotType = LocomotionSnapshot<InvertedPendulumObservation>;

    fn new(_render: bool) -> Self {
        Self::with_config(InvertedPendulumConfig::default())
    }

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.rng = StdRng::seed_from_u64(self.config.seed);
        let (world, mut state) = Self::build_world(&self.config, &mut self.rng);
        self.world = world;
        state.last_obs = InvertedPendulumObservation::default();
        self.state = state;
        self.steps = 0;

        let obs = self.extract_observation();
        self.state.last_obs = obs;
        let meta = SnapshotMetadata::new().with(METADATA_KEY_ALIVE, 0.0);
        Ok(LocomotionSnapshot::running(obs, ScalarReward(0.0), meta))
    }

    fn step(
        &mut self,
        action: InvertedPendulumAction,
    ) -> Result<Self::SnapshotType, EnvironmentError> {
        if !action.0[0].is_finite() {
            return Err(EnvironmentError::InvalidAction(format!(
                "InvertedPendulum action must be finite, got {}",
                action.0[0]
            )));
        }

        self.apply_action(&action);
        Rapier3DBackend::step(&mut self.world);
        self.steps += 1;

        let obs = self.extract_observation();
        self.state.last_obs = obs;

        let healthy = self.config.healthy.is_healthy(
            /* torso_z (unused) */ 0.0,
            obs.pole_angle(),
            &obs.0,
        );
        let alive_bonus = if healthy { 1.0 } else { 0.0 };
        let reward = ScalarReward(alive_bonus);

        let status = if !healthy
            && matches!(self.config.termination, TerminationMode::OnUnhealthy)
        {
            EpisodeStatus::Terminated
        } else if self.steps >= self.config.max_steps {
            EpisodeStatus::Truncated
        } else {
            EpisodeStatus::Running
        };

        let meta = SnapshotMetadata::new()
            .with(METADATA_KEY_ALIVE, alive_bonus)
            .with_position(
                "cart",
                [obs.cart_position(), 0.0, self.config.cart_half_extents[2]],
            );
        Ok(LocomotionSnapshot::new(obs, reward, status, meta))
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────

fn cart_half_z(config: &InvertedPendulumConfig) -> f32 {
    config.cart_half_extents[2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use evorl_core::environment::Snapshot;

    #[test]
    fn action_shape_and_validity() {
        assert_eq!(InvertedPendulumAction::shape(), [1]);
        assert!(InvertedPendulumAction::new(0.0).is_valid());
        assert!(InvertedPendulumAction::new(3.0).is_valid());
        assert!(!InvertedPendulumAction::new(3.5).is_valid());
        assert!(!InvertedPendulumAction::new(f32::NAN).is_valid());
    }

    #[test]
    fn observation_shape() {
        assert_eq!(InvertedPendulumObservation::shape(), [4]);
    }

    #[test]
    fn reset_returns_running_with_near_zero_obs() {
        let mut env = InvertedPendulumRapier::with_config(
            InvertedPendulumConfig { seed: 7, reset_noise_scale: 0.0, ..Default::default() },
        );
        let snap = env.reset().unwrap();
        assert!(!snap.is_done());
        for v in snap.observation().0 {
            assert!(v.abs() < 1e-5, "zero reset noise should give ~zero obs");
        }
    }

    #[test]
    fn ctrl_cost_not_paid() {
        // InvertedPendulum's Gymnasium reward is +1 alive, not a quadratic cost.
        let mut env = InvertedPendulumRapier::with_config(InvertedPendulumConfig::default());
        env.reset().unwrap();
        let snap = env.step(InvertedPendulumAction::new(3.0)).unwrap();
        // Reward is +1 per step while healthy regardless of action magnitude.
        let total: f32 = snap.metadata().unwrap().components.values().sum();
        assert!((total - snap.reward().0).abs() < 1e-5);
    }

    #[test]
    fn reward_roundtrip_matches_components() {
        let mut env = InvertedPendulumRapier::with_config(InvertedPendulumConfig::default());
        env.reset().unwrap();
        for _ in 0..5 {
            let snap = env.step(InvertedPendulumAction::new(0.0)).unwrap();
            let meta = snap.metadata().unwrap();
            let total: f32 = meta.components.values().sum();
            assert!(
                (total - snap.reward().0).abs() < 1e-5,
                "components sum ({total}) must equal reward ({})",
                snap.reward().0
            );
        }
    }

    #[test]
    fn terminates_when_pole_angle_leaves_band() {
        // Start with a mild tilt (within healthy band), no reset noise, then
        // apply force in the tilt direction. Gravity does the rest: the pole
        // must reach |θ| ≥ 0.2 and terminate.
        let mut env = InvertedPendulumRapier::with_config(InvertedPendulumConfig {
            reset_noise_scale: 0.0,
            max_steps: 2000,
            ..Default::default()
        });
        env.reset().unwrap();
        // Kick the pole with one sharp +x impulse on the cart, then let it fall.
        let mut terminated = false;
        let mut max_abs_angle: f32 = 0.0;
        let mut cart_x_max: f32 = 0.0;
        for i in 0..2000 {
            let action = if i < 20 { 3.0 } else { 0.0 };
            let snap = env.step(InvertedPendulumAction::new(action)).unwrap();
            max_abs_angle = max_abs_angle.max(snap.observation().pole_angle().abs());
            cart_x_max = cart_x_max.max(snap.observation().cart_position().abs());
            if snap.is_terminated() {
                terminated = true;
                break;
            }
        }
        assert!(
            terminated,
            "pushing the cart must eventually drop the pole outside (-0.2, 0.2); \
             max |angle| observed = {max_abs_angle}, max |cart_x| = {cart_x_max}"
        );
    }

    #[test]
    fn truncates_at_max_steps() {
        let mut env = InvertedPendulumRapier::with_config(InvertedPendulumConfig {
            max_steps: 5,
            termination: TerminationMode::Never,
            reset_noise_scale: 0.0,
            ..Default::default()
        });
        env.reset().unwrap();
        let mut status = EpisodeStatus::Running;
        for _ in 0..5 {
            let snap = env.step(InvertedPendulumAction::new(0.0)).unwrap();
            status = snap.status();
        }
        assert_eq!(status, EpisodeStatus::Truncated);
    }

    #[test]
    fn determinism_across_reset() {
        let cfg = InvertedPendulumConfig { seed: 123, ..Default::default() };
        let rollout = |actions: &[f32]| {
            let mut env = InvertedPendulumRapier::with_config(cfg.clone());
            env.reset().unwrap();
            let mut last = InvertedPendulumObservation::default();
            for &a in actions {
                if let Ok(snap) = env.step(InvertedPendulumAction::new(a)) {
                    last = *snap.observation();
                }
            }
            last
        };
        let actions = [0.0, 1.0, -1.0, 0.5, 0.0];
        assert_eq!(rollout(&actions), rollout(&actions));
    }

    #[test]
    fn invalid_action_is_error() {
        let mut env = InvertedPendulumRapier::with_config(InvertedPendulumConfig::default());
        env.reset().unwrap();
        let bad = InvertedPendulumAction::new(f32::NAN);
        assert!(env.step(bad).is_err());
    }

    #[test]
    fn action_clip_at_boundaries() {
        let a = InvertedPendulumAction::new(10.0).clip(-3.0, 3.0);
        assert_eq!(a.0[0], 3.0);
        let a = InvertedPendulumAction::new(-10.0).clip(-3.0, 3.0);
        assert_eq!(a.0[0], -3.0);
    }

    #[test]
    fn obs_is_finite_after_rollout() {
        let mut env = InvertedPendulumRapier::with_config(InvertedPendulumConfig::default());
        env.reset().unwrap();
        for _ in 0..50 {
            let snap = env.step(InvertedPendulumAction::new(0.1)).unwrap();
            assert!(snap.observation().is_finite());
            if snap.is_done() {
                break;
            }
        }
    }
}
