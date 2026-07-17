//! [`Swimmer`] environment implementation.
//!
//! This module contains the [`Swimmer`] struct and its [`Environment`] and
//! [`ConstructableEnv`] impls. Physics setup, drag application, and action
//! integration are kept private; the public surface is the two trait methods
//! [`Environment::reset`] and [`Environment::step`].
//!
//! The type alias [`SwimmerRapier`] is the only concretisation shipped at
//! present; the backend type parameter exists to allow a future mock or
//! alternative physics backend without changing the public API.

use std::marker::PhantomData;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rapier3d::math::Vector;
use rapier3d::prelude::*;
use rlevo_core::config::{ConfigError, Validate};
use rlevo_core::environment::{
    ConstructableEnv, Environment, EnvironmentError, EpisodeStatus, Sensor, SnapshotMetadata,
};
use rlevo_core::reward::ScalarReward;

use crate::locomotion::backend::{LocomotionBackend, Rapier3DBackend, Rapier3DWorld};
use crate::locomotion::common::{LocomotionSnapshot, ctrl_cost, wrap_to_pi};

use super::action::SwimmerAction;
use super::config::SwimmerConfig;
use super::observation::SwimmerObservation;
use super::state::SwimmerState;

/// Reward-component key: `forward_reward_weight · vx_com` (≥ 0 when swimming
/// forward, ≤ 0 when drifting backward).
pub const METADATA_KEY_FORWARD: &str = "forward";
/// Reward-component key: `−ctrl_cost_weight · ‖action‖²` (≤ 0).
pub const METADATA_KEY_CTRL: &str = "ctrl";

/// A 3-segment planar swimmer with viscous drag, generic over the physics
/// backend.
///
/// The swimmer chain is described in detail in the
/// [`crate::locomotion::swimmer`] module doc. The only shipped backend is
/// `B = Rapier3DBackend`, accessible through the [`SwimmerRapier`] type alias.
///
/// Construct with [`ConstructableEnv::new`] (uses [`SwimmerConfig::default`])
/// or with [`Swimmer::with_config`] for a custom configuration.
#[derive(Debug)]
pub struct Swimmer<B: LocomotionBackend = Rapier3DBackend> {
    world: B::World,
    state: SwimmerState,
    config: SwimmerConfig,
    rng: StdRng,
    steps: usize,
    _marker: PhantomData<B>,
}

/// Convenience alias: [`Swimmer`] using the Rapier3D backend.
///
/// This is the recommended concrete type for all production use.
pub type SwimmerRapier = Swimmer<Rapier3DBackend>;

impl Swimmer<Rapier3DBackend> {
    /// Create with an explicit configuration.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `config` fails [`Validate`] (e.g.
    /// non-positive `dt`, inverted `action_clip`, or non-positive segment
    /// geometry).
    pub fn with_config(config: SwimmerConfig) -> Result<Self, ConfigError> {
        config.validate()?;
        let mut rng = StdRng::seed_from_u64(config.seed);
        let (world, state) = Self::build_world(&config, &mut rng);
        Ok(Self {
            world,
            state,
            config,
            rng,
            steps: 0,
            _marker: PhantomData,
        })
    }

    /// Re-seed the persistent RNG to `seed`, then [`reset`](Environment::reset).
    ///
    /// Ordinary [`reset`](Environment::reset) advances the persistent stream so
    /// successive episodes differ; use this when you need a *specific* episode
    /// to reproduce bit-for-bit (e.g. replaying a failure). Run-level
    /// reproducibility is already guaranteed by the construction seed.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`reset`](Environment::reset) (currently none).
    pub fn reset_with_seed(
        &mut self,
        seed: u64,
    ) -> Result<LocomotionSnapshot<SwimmerObservation>, EnvironmentError> {
        self.rng = StdRng::seed_from_u64(seed);
        self.reset()
    }

    fn build_world(config: &SwimmerConfig, rng: &mut StdRng) -> (Rapier3DWorld, SwimmerState) {
        // Zero gravity — swimmer floats in open water.
        // The world owns the `frame_skip` substep loop; `step_physics` drives it
        // via `step_actuated`, re-applying actuator torque and drag before each
        // substep (see `step_physics`).
        let mut world =
            Rapier3DWorld::new(Vector::new(0.0, 0.0, 0.0), config.dt, config.frame_skip);

        // Reset-noise sampling: qpos and qvel all ~ U(-s, s).
        let n = config.reset_noise_scale;
        let p0_x: f32 = rng.random_range(-n..=n);
        let p0_y: f32 = rng.random_range(-n..=n);
        let theta_body: f32 = rng.random_range(-n..=n);
        let joint1_init: f32 = rng.random_range(-n..=n);
        let joint2_init: f32 = rng.random_range(-n..=n);
        let vx_init: f32 = rng.random_range(-n..=n);
        let vy_init: f32 = rng.random_range(-n..=n);
        let omega_body_init: f32 = rng.random_range(-n..=n);
        let joint1_dot_init: f32 = rng.random_range(-n..=n);
        let joint2_dot_init: f32 = rng.random_range(-n..=n);

        let half_l = config.segment_length * 0.5;
        let r = config.segment_radius;

        // Capsule volume: cylinder of length 2·half_l plus two hemispherical
        // caps. Density drives the inertia tensor (using `additional_mass`
        // instead would leave angular inertia zero → segments wouldn't rotate).
        let capsule_volume = std::f32::consts::PI * r.powi(2) * (2.0 * half_l + (4.0 / 3.0) * r);
        let density = config.segment_mass / capsule_volume.max(f32::EPSILON);

        // Segment absolute angles: joint angles are relative (child − parent).
        let angle0 = theta_body;
        let angle1 = theta_body + joint1_init;
        let angle2 = theta_body + joint1_init + joint2_init;

        // Chain centres: successive links are placed so anchor1 on the parent
        // back (+half_l, 0) matches anchor2 on the child front (−half_l, 0).
        //   p1 = p0 + half_l·dir(angle0) + half_l·dir(angle1)
        //   p2 = p1 + half_l·dir(angle1) + half_l·dir(angle2)
        let p1_x = p0_x + half_l * angle0.cos() + half_l * angle1.cos();
        let p1_y = p0_y + half_l * angle0.sin() + half_l * angle1.sin();
        let p2_x = p1_x + half_l * angle1.cos() + half_l * angle2.cos();
        let p2_y = p1_y + half_l * angle1.sin() + half_l * angle2.sin();

        // Segment absolute angular velocities: joint_dots are relative rates.
        let w0 = omega_body_init;
        let w1 = omega_body_init + joint1_dot_init;
        let w2 = omega_body_init + joint1_dot_init + joint2_dot_init;

        // Root (segment0) carries the full-system DOF gates so motion stays in
        // the xy-plane and rotation only about z. Children are parameterised
        // by the multibody joints; gating their DOFs directly is redundant
        // (and would over-constrain the reduced-coordinate solver).
        let segment0 = world.add_body(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(p0_x, p0_y, 0.0))
                .rotation(Vector::new(0.0, 0.0, angle0))
                .linvel(Vector::new(vx_init, vy_init, 0.0))
                .angvel(Vector::new(0.0, 0.0, w0))
                .enabled_translations(true, true, false)
                .enabled_rotations(false, false, true),
        );
        world.add_collider(
            ColliderBuilder::capsule_x(half_l, r).density(density),
            segment0,
        );

        let segment1 = world.add_body(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(p1_x, p1_y, 0.0))
                .rotation(Vector::new(0.0, 0.0, angle1))
                .angvel(Vector::new(0.0, 0.0, w1)),
        );
        world.add_collider(
            ColliderBuilder::capsule_x(half_l, r).density(density),
            segment1,
        );

        let segment2 = world.add_body(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(p2_x, p2_y, 0.0))
                .rotation(Vector::new(0.0, 0.0, angle2))
                .angvel(Vector::new(0.0, 0.0, w2)),
        );
        world.add_collider(
            ColliderBuilder::capsule_x(half_l, r).density(density),
            segment2,
        );

        let z_axis: Vector = Vector::new(0.0, 0.0, 1.0);
        // Disable jointed-neighbour contacts: adjacent segment caps overlap at
        // the shared anchor. The narrow phase honours the per-joint flag for
        // multibody joints too (MuJoCo parent–child filter parity, ADR 0041).
        let joint1 = RevoluteJointBuilder::new(z_axis)
            .local_anchor1(Vector::new(half_l, 0.0, 0.0))
            .local_anchor2(Vector::new(-half_l, 0.0, 0.0))
            .contacts_enabled(false)
            .build();
        let joint1_handle = world
            .add_multibody_joint(segment0, segment1, joint1)
            .expect("joint1 must form a tree — segment0 is the multibody root");

        // Disable jointed-neighbour contacts: segment1's and segment2's caps
        // overlap at the shared anchor (MuJoCo parent–child filter parity, ADR 0041).
        let joint2 = RevoluteJointBuilder::new(z_axis)
            .local_anchor1(Vector::new(half_l, 0.0, 0.0))
            .local_anchor2(Vector::new(-half_l, 0.0, 0.0))
            .contacts_enabled(false)
            .build();
        let joint2_handle = world
            .add_multibody_joint(segment1, segment2, joint2)
            .expect("joint2 must form a tree — segment1 is already segment0's child");

        let state = SwimmerState {
            segment0,
            segment1,
            segment2,
            joint1: joint1_handle,
            joint2: joint2_handle,
            last_obs: SwimmerObservation::default(),
        };
        (world, state)
    }

    fn extract_observation(&self) -> SwimmerObservation {
        let p0 = Rapier3DBackend::get_pose(&self.world, self.state.segment0);
        let p1 = Rapier3DBackend::get_pose(&self.world, self.state.segment1);
        let p2 = Rapier3DBackend::get_pose(&self.world, self.state.segment2);
        let v0 = Rapier3DBackend::get_vel(&self.world, self.state.segment0);
        let v1 = Rapier3DBackend::get_vel(&self.world, self.state.segment1);
        let v2 = Rapier3DBackend::get_vel(&self.world, self.state.segment2);

        // Pure rotation about world-z ⇒ quaternion = (cos(θ/2), 0, 0, sin(θ/2))
        // in [w, x, y, z] order. θ = 2·atan2(qz, qw).
        let a0 = segment_z_angle(p0.orientation);
        let a1 = segment_z_angle(p1.orientation);
        let a2 = segment_z_angle(p2.orientation);

        let body_angle = wrap_to_pi(a0);
        let joint1_angle = wrap_to_pi(a1 - a0);
        let joint2_angle = wrap_to_pi(a2 - a1);

        SwimmerObservation([
            body_angle,
            joint1_angle,
            joint2_angle,
            v0.linear[0],
            v0.linear[1],
            v0.angular[2],
            v1.angular[2] - v0.angular[2],
            v2.angular[2] - v1.angular[2],
        ])
    }

    /// Compute the two gear-scaled joint torques for `action` (clip → gear).
    /// Pure: the torques are *applied* inside the `step_physics` substep hook so
    /// they are re-applied fresh each substep (ADR 0037 force-lifetime contract).
    ///
    /// gear is frozen at its current value here; re-tuning it toward the
    /// canonical Swimmer-v5 value (`gear [150, 150]`) is deferred to a follow-up
    /// issue — #98 fixes only force accumulation and must not silently change
    /// gear.
    fn control_torques(&self, action: &SwimmerAction) -> [f32; 2] {
        let (lo, hi): (f32, f32) = self.config.action_clip.into();
        let clipped = [action.0[0].clamp(lo, hi), action.0[1].clamp(lo, hi)];
        self.config.gear.apply(&clipped)
    }

    /// Run the `frame_skip` physics substeps, re-applying the constant actuator
    /// torque **and** recomputing viscous drag before every substep.
    ///
    /// Force lifetime (ADR 0037): rapier3d 0.32 does **not** auto-clear external
    /// forces — the accumulator would otherwise grow monotonically across
    /// substeps and env steps. `Rapier3DWorld::step_once` now clears it after
    /// integrating, so control that must persist has to be re-applied every
    /// substep. Accordingly:
    ///   * the actuator torque (`torques`) is held **constant** across the
    ///     frame skip — the same magnitude re-applied each substep, matching
    ///     MuJoCo's `ctrl`-held-across-substeps semantics;
    ///   * viscous drag is recomputed each substep from the segments' **current**
    ///     velocity, so it tracks the chain as it accelerates within the step.
    ///
    /// Under `MultibodyJointSet`, torquing a child body produces an
    /// equal-and-opposite reaction on the parent through the joint (the
    /// reduced-coordinate solver handles this), so only the child of each joint
    /// is torqued: seg1 for joint1, seg2 for joint2. Double-torquing (also
    /// applying −τ to the parent, as impulse-joint envs like Reacher do) would
    /// inject spurious net torque on the free-floating chain.
    fn step_physics(&mut self, torques: [f32; 2]) {
        let seg0 = self.state.segment0;
        let seg1 = self.state.segment1;
        let seg2 = self.state.segment2;
        let k = self.config.drag_coefficient;
        let k_ang = self.config.angular_drag_coefficient;
        self.world.step_actuated(|w| {
            if let Some(body) = w.bodies_mut().get_mut(seg1) {
                body.add_torque(Vector::new(0.0, 0.0, torques[0]), true);
            }
            if let Some(body) = w.bodies_mut().get_mut(seg2) {
                body.add_torque(Vector::new(0.0, 0.0, torques[1]), true);
            }
            apply_drag_to(w, [seg0, seg1, seg2], k, k_ang);
        });
    }
}

impl Sensor<1, 1, 1> for Swimmer<Rapier3DBackend> {
    type Action = SwimmerAction;
    type State = SwimmerState;
    type Observation = SwimmerObservation;

    /// Emission model: reads the 8-element observation directly from the physics
    /// world through the state's body handles. The action does not enter the
    /// observation, and `next_state` carries the same handles as `self.state`,
    /// so both are unused; the world is the source of truth.
    fn observe(&self, _action: &SwimmerAction, _next_state: &SwimmerState) -> SwimmerObservation {
        self.extract_observation()
    }

    /// Initial observation at episode start, read from the freshly built world.
    fn observe_reset(&self, _state: &SwimmerState) -> SwimmerObservation {
        self.extract_observation()
    }
}

impl ConstructableEnv for Swimmer<Rapier3DBackend> {
    /// Create a [`Swimmer`] with [`SwimmerConfig::default`].
    ///
    /// The `render` parameter is accepted for trait compatibility but has no
    /// effect; this environment has no visual output.
    fn new(_render: bool) -> Self {
        Self::with_config(SwimmerConfig::default()).expect("default config must validate")
    }
}

impl Environment<1, 1, 1> for Swimmer<Rapier3DBackend> {
    type StateType = SwimmerState;
    type ObservationType = SwimmerObservation;
    type ActionType = SwimmerAction;
    type RewardType = ScalarReward;
    type SnapshotType = LocomotionSnapshot<SwimmerObservation>;

    /// Reset the episode to a freshly sampled initial state.
    ///
    /// All generalised positions and velocities are sampled independently
    /// from `U(-reset_noise_scale, reset_noise_scale)`, drawn from the
    /// environment's persistent RNG. The stream **advances** across resets, so
    /// successive episodes see independent initial states. For deterministic
    /// replay of a specific initial state, use [`Swimmer::reset_with_seed`].
    ///
    /// The returned snapshot has `EpisodeStatus::Running`, reward `0.0`, and
    /// metadata components `forward = 0.0` and `ctrl = 0.0`.
    ///
    /// # Errors
    ///
    /// This implementation does not currently return an error; the signature
    /// reflects the `Environment` trait contract.
    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        let (world, mut state) = Self::build_world(&self.config, &mut self.rng);
        self.world = world;
        state.last_obs = SwimmerObservation::default();
        self.state = state;
        self.steps = 0;

        let obs = self.observe_reset(&self.state);
        self.state.last_obs = obs;

        let torso_pos = Rapier3DBackend::get_pose(&self.world, self.state.segment0).position;
        let meta = SnapshotMetadata::new()
            .with(METADATA_KEY_FORWARD, 0.0)
            .with(METADATA_KEY_CTRL, 0.0)
            .with_position("torso", torso_pos)
            .with_position("com", torso_pos)
            .with_position("main_body", torso_pos);
        Ok(LocomotionSnapshot::running(obs, ScalarReward(0.0)).with_metadata(meta))
    }

    /// Advance the simulation by one env step.
    ///
    /// Sequence: clip action → apply gear-scaled torques to joint children →
    /// run `frame_skip` physics substeps (each preceded by viscous drag) →
    /// extract observation → compute reward.
    ///
    /// Reward = `forward_reward_weight × vx_com − ctrl_cost_weight × ‖clipped_action‖²`.
    /// Both components are stored in snapshot metadata under
    /// [`METADATA_KEY_FORWARD`] and [`METADATA_KEY_CTRL`].
    ///
    /// The episode is never terminated on a health condition. Once `steps ≥
    /// max_steps` the status becomes `EpisodeStatus::Truncated`.
    ///
    /// # Errors
    ///
    /// Returns [`EnvironmentError::InvalidAction`] if any element of `action`
    /// is non-finite (`NaN` or `±∞`).
    fn step(&mut self, action: SwimmerAction) -> Result<Self::SnapshotType, EnvironmentError> {
        if !action.0.iter().all(|v| v.is_finite()) {
            return Err(EnvironmentError::InvalidAction(format!(
                "Swimmer action must be finite, got {:?}",
                action.0
            )));
        }

        let torques = self.control_torques(&action);
        self.step_physics(torques);
        self.steps += 1;

        let obs = self.observe(&action, &self.state);
        self.state.last_obs = obs;

        let forward = self.config.forward_reward_weight * obs.vx_com();
        // Clip to the bound-enforced action before computing ctrl cost so that
        // out-of-bound inputs can't inflate the cost beyond the Gymnasium convention.
        let (lo, hi): (f32, f32) = self.config.action_clip.into();
        let clipped = [action.0[0].clamp(lo, hi), action.0[1].clamp(lo, hi)];
        let ctrl = -ctrl_cost(self.config.ctrl_cost_weight, &clipped);
        let total = forward + ctrl;

        let status = if self.steps >= self.config.max_steps {
            EpisodeStatus::Truncated
        } else {
            EpisodeStatus::Running
        };

        let torso_pos = Rapier3DBackend::get_pose(&self.world, self.state.segment0).position;
        let meta = SnapshotMetadata::new()
            .with(METADATA_KEY_FORWARD, forward)
            .with(METADATA_KEY_CTRL, ctrl)
            .with_position("torso", torso_pos)
            .with_position("com", torso_pos)
            .with_position("main_body", torso_pos);
        Ok(LocomotionSnapshot {
            observation: obs,
            reward: ScalarReward(total),
            status,
            metadata: Some(meta),
        })
    }
}

/// Extract the z-axis rotation angle from a pose whose body is DOF-gated to
/// revolute-z. Quaternion stored as `[w, x, y, z]`.
fn segment_z_angle(orientation: [f32; 4]) -> f32 {
    let [w, _, _, z] = orientation;
    2.0 * z.atan2(w)
}

/// Apply per-segment viscous drag to each handle in `handles`:
/// quadratic linear drag `F = −k · v · ‖v‖` plus **linear** angular drag
/// `τ = −k_ang · ω`, read from each body's current velocity.
///
/// The angular term is linear (not quadratic) because Rapier integrates forces
/// with explicit Euler, and the overshoot threshold of an explicit step for
/// quadratic drag is `k_ang · |ω| · dt / I < 1`: at gear=150 the chain reaches
/// `|ω|` in the 100s of rad/s, which drives quadratic drag unstable and diverges
/// to NaN in one substep. Linear drag is unconditionally stable as long as
/// `k_ang · dt / I < 2`. MuJoCo's own swimmer uses quadratic drag but with an
/// implicit integrator; our linear variant is a Rapier-compatibility divergence.
///
/// Free function (not a `&mut self` method) so it can be invoked from inside a
/// `step_actuated` closure that borrows only the world (ADR 0037).
fn apply_drag_to(world: &mut Rapier3DWorld, handles: [RigidBodyHandle; 3], k: f32, k_ang: f32) {
    for handle in handles {
        let twist = Rapier3DBackend::get_vel(world, handle);
        let v = twist.linear;
        let speed = (v[0] * v[0] + v[1] * v[1]).sqrt();
        let fx = -k * v[0] * speed;
        let fy = -k * v[1] * speed;
        let wz = twist.angular[2];
        let tau_z = -k_ang * wz;
        if let Some(body) = world.bodies_mut().get_mut(handle) {
            body.add_force(Vector::new(fx, fy, 0.0), true);
            body.add_torque(Vector::new(0.0, 0.0, tau_z), true);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlevo_core::base::Action;
    use rlevo_core::base::Observation;
    use rlevo_core::environment::Snapshot;

    fn cfg(seed: u64) -> SwimmerConfig {
        SwimmerConfig {
            seed,
            ..Default::default()
        }
    }

    fn deterministic_cfg() -> SwimmerConfig {
        SwimmerConfig {
            seed: 7,
            reset_noise_scale: 0.0,
            ..Default::default()
        }
    }

    #[test]
    fn action_shape_and_validity() {
        assert_eq!(SwimmerAction::shape(), [2]);
        assert!(SwimmerAction::new(0.0, 0.0).is_valid());
        assert!(SwimmerAction::new(1.0, -1.0).is_valid());
        assert!(!SwimmerAction::new(1.5, 0.0).is_valid());
        assert!(!SwimmerAction::new(f32::NAN, 0.0).is_valid());
    }

    #[test]
    fn observation_shape() {
        assert_eq!(SwimmerObservation::shape(), [8]);
    }

    #[test]
    fn action_clip_at_boundaries() {
        use rlevo_core::action::ContinuousAction;
        let a = SwimmerAction::new(10.0, -10.0).clip(-1.0, 1.0);
        assert_eq!(a.0, [1.0, -1.0]);
    }

    #[test]
    fn reset_returns_running() {
        let mut env = SwimmerRapier::with_config(cfg(7)).expect("valid config");
        let snap = env.reset().unwrap();
        assert!(!snap.is_done());
        assert!(snap.observation().is_finite());
    }

    #[test]
    fn reset_with_zero_noise_is_upright() {
        let mut env = SwimmerRapier::with_config(deterministic_cfg()).expect("valid config");
        let snap = env.reset().unwrap();
        let obs = snap.observation();
        assert!(obs.body_angle().abs() < 1e-5);
        assert!(obs.joint1_angle().abs() < 1e-5);
        assert!(obs.joint2_angle().abs() < 1e-5);
        assert!(obs.vx_com().abs() < 1e-5);
        assert!(obs.vy_com().abs() < 1e-5);
        assert!(obs.omega_body().abs() < 1e-5);
        assert!(obs.joint1_dot().abs() < 1e-5);
        assert!(obs.joint2_dot().abs() < 1e-5);
    }

    #[test]
    fn reward_decomposition_sums_to_total() {
        let mut env = SwimmerRapier::with_config(cfg(11)).expect("valid config");
        env.reset().unwrap();
        for i in 0..100 {
            let a =
                SwimmerAction::new(0.5 * (i as f32 * 0.17).sin(), 0.5 * (i as f32 * 0.23).cos());
            let snap = env.step(a).unwrap();
            let meta = snap.metadata().unwrap();
            let sum: f32 = meta.components.values().sum();
            assert!(
                (sum - snap.reward().0).abs() < 1e-5,
                "Σ components ({sum}) must equal reward ({}) at step {i}",
                snap.reward().0
            );
            if snap.is_done() {
                break;
            }
        }
    }

    #[test]
    fn ctrl_cost_scales_quadratically() {
        let a = [0.3f32, -0.5];
        let a2 = [0.6f32, -1.0];
        let c1 = ctrl_cost(1e-4, &a);
        let c2 = ctrl_cost(1e-4, &a2);
        assert!((c2 - 4.0 * c1).abs() < 1e-8);
    }

    #[test]
    fn ctrl_component_nonpositive() {
        let mut env = SwimmerRapier::with_config(cfg(13)).expect("valid config");
        env.reset().unwrap();
        for i in 0..50 {
            let a =
                SwimmerAction::new(0.6 * (i as f32 * 0.31).cos(), 0.9 * (i as f32 * 0.11).sin());
            let snap = env.step(a).unwrap();
            let c = snap.metadata().unwrap().components[METADATA_KEY_CTRL];
            assert!(c <= 0.0, "ctrl must be ≤ 0, got {c} at step {i}");
        }
    }

    #[test]
    fn determinism_across_reset() {
        let rollout = |actions: &[[f32; 2]]| {
            let mut env = SwimmerRapier::with_config(cfg(123)).expect("valid config");
            env.reset().unwrap();
            let mut last = SwimmerObservation::default();
            for a in actions {
                if let Ok(snap) = env.step(SwimmerAction(*a)) {
                    last = *snap.observation();
                    if snap.is_done() {
                        break;
                    }
                }
            }
            last
        };
        let actions = [[0.1, -0.2], [0.5, 0.3], [-0.4, 0.2], [0.0, 0.0]];
        assert_eq!(rollout(&actions), rollout(&actions));
    }

    /// Observation fingerprint used by the RNG-diversity tests.
    fn obs_key(o: &SwimmerObservation) -> [f32; 6] {
        [
            o.body_angle(),
            o.joint1_angle(),
            o.joint2_angle(),
            o.omega_body(),
            o.joint1_dot(),
            o.joint2_dot(),
        ]
    }

    #[test]
    fn two_successive_resets_differ() {
        // The persistent RNG advances across resets (default reset noise > 0),
        // so back-to-back resets must sample independent initial states.
        let mut env = SwimmerRapier::with_config(cfg(7)).expect("valid config");
        let first = *env.reset().unwrap().observation();
        let second = *env.reset().unwrap().observation();
        assert_ne!(
            obs_key(&first),
            obs_key(&second),
            "successive resets must draw independent initial states"
        );
    }

    #[test]
    fn reset_with_seed_is_reproducible() {
        let mut env = SwimmerRapier::with_config(cfg(7)).expect("valid config");
        let a = *env.reset_with_seed(999).unwrap().observation();
        // Advance the stream with an ordinary reset, then re-seed identically.
        env.reset().unwrap();
        let b = *env.reset_with_seed(999).unwrap().observation();
        assert_eq!(
            obs_key(&a),
            obs_key(&b),
            "reset_with_seed must reproduce the same initial state"
        );
    }

    #[test]
    fn init_noise_bounded() {
        for seed in 0..50 {
            let env = SwimmerRapier::with_config(cfg(seed)).expect("valid config");
            let obs = env.state.last_obs;
            assert!(obs.is_finite(), "seed {seed} produced non-finite obs");
            // Reset noise is ±0.1; body_angle and joint angles are pulled
            // directly from the sampled range.
            assert!(
                obs.body_angle().abs() <= 0.1 + 1e-5,
                "seed {seed}: |body_angle|={} > 0.1",
                obs.body_angle().abs()
            );
            assert!(
                obs.joint1_angle().abs() <= 0.1 + 1e-5,
                "seed {seed}: |joint1_angle|={} > 0.1",
                obs.joint1_angle().abs()
            );
            assert!(
                obs.joint2_angle().abs() <= 0.1 + 1e-5,
                "seed {seed}: |joint2_angle|={} > 0.1",
                obs.joint2_angle().abs()
            );
        }
    }

    #[test]
    fn truncates_at_max_steps() {
        let mut env = SwimmerRapier::with_config(SwimmerConfig {
            max_steps: 5,
            ..Default::default()
        })
        .expect("valid config");
        env.reset().unwrap();
        let mut status = EpisodeStatus::Running;
        for step in 0..5 {
            let snap = env.step(SwimmerAction::new(0.0, 0.0)).unwrap();
            status = snap.status();
            if step < 4 {
                assert_eq!(
                    status,
                    EpisodeStatus::Running,
                    "early status at step {step}"
                );
            }
        }
        assert_eq!(status, EpisodeStatus::Truncated);
    }

    #[test]
    fn invalid_action_is_error() {
        let mut env = SwimmerRapier::with_config(SwimmerConfig::default()).expect("valid config");
        env.reset().unwrap();
        let bad = SwimmerAction::new(f32::NAN, 0.0);
        assert!(env.step(bad).is_err());
        let bad = SwimmerAction::new(0.0, f32::INFINITY);
        assert!(env.step(bad).is_err());
    }

    #[test]
    fn obs_is_finite_after_rollout() {
        let mut env = SwimmerRapier::with_config(cfg(42)).expect("valid config");
        env.reset().unwrap();
        for i in 0..100 {
            let a = SwimmerAction::new(0.5 * (i as f32 * 0.3).sin(), 0.5 * (i as f32 * 0.4).cos());
            let snap = env.step(a).unwrap();
            assert!(snap.observation().is_finite(), "non-finite obs at step {i}");
            if snap.is_done() {
                break;
            }
        }
    }

    #[test]
    fn obs_layout_matches_spec() {
        // With zero reset noise then a deliberate angular velocity on each
        // segment, every observation slot is a named function of the state
        // that we can verify independently.
        let mut env = SwimmerRapier::with_config(deterministic_cfg()).expect("valid config");
        env.reset().unwrap();
        // Forcibly set angular velocities on the three segments.
        if let Some(b) = env.world.bodies_mut().get_mut(env.state.segment0) {
            b.set_angvel(Vector::new(0.0, 0.0, 1.0), true);
        }
        if let Some(b) = env.world.bodies_mut().get_mut(env.state.segment1) {
            b.set_angvel(Vector::new(0.0, 0.0, 1.5), true);
        }
        if let Some(b) = env.world.bodies_mut().get_mut(env.state.segment2) {
            b.set_angvel(Vector::new(0.0, 0.0, 1.2), true);
        }
        // Read observation directly; no step, so no constraint reactions.
        let obs = env.extract_observation();
        // body_angle and joint angles: near zero (deterministic_cfg has
        // zero reset noise and we haven't stepped).
        assert!(obs.body_angle().abs() < 1e-5);
        assert!(obs.joint1_angle().abs() < 1e-5);
        assert!(obs.joint2_angle().abs() < 1e-5);
        // Linear velocities: also zero.
        assert!(obs.vx_com().abs() < 1e-5);
        assert!(obs.vy_com().abs() < 1e-5);
        // Angular velocities: ω_body = segment0, joint1_dot = ω1 − ω0 = 0.5,
        // joint2_dot = ω2 − ω1 = -0.3.
        assert!((obs.omega_body() - 1.0).abs() < 1e-5);
        assert!((obs.joint1_dot() - 0.5).abs() < 1e-5);
        assert!((obs.joint2_dot() + 0.3).abs() < 1e-5);
    }

    #[test]
    fn drag_damps_passive_motion() {
        // Drive the chain forward with a short sinusoidal stroke, snapshot the
        // peak vx_com, then cut the actuator and verify the velocity magnitude
        // decays over the next 100 no-action env steps. (We build up motion
        // through actuation rather than `set_linvel` because Rapier's
        // multibody solver tracks the chain's state in reduced coordinates
        // and overrides direct body-level velocity writes.)
        use std::f32::consts::TAU;
        let mut env = SwimmerRapier::with_config(deterministic_cfg()).expect("valid config");
        env.reset().unwrap();
        let mut peak_vx_mag = 0.0f32;
        for i in 0..60 {
            let t = i as f32 * TAU / 10.0;
            let a = SwimmerAction::new(t.sin(), (t - std::f32::consts::FRAC_PI_2).sin());
            let snap = env.step(a).unwrap();
            peak_vx_mag = peak_vx_mag.max(snap.observation().vx_com().abs());
        }
        assert!(
            peak_vx_mag > 0.1,
            "drive stroke must build up measurable |vx|, got {peak_vx_mag}"
        );
        let mut decayed = false;
        let mut final_vx_mag = 0.0f32;
        for _ in 0..100 {
            let snap = env.step(SwimmerAction::new(0.0, 0.0)).unwrap();
            final_vx_mag = snap.observation().vx_com().abs();
            if final_vx_mag < 0.5 * peak_vx_mag {
                decayed = true;
                break;
            }
        }
        assert!(
            decayed,
            "drag must damp vx to < 0.5 of peak {peak_vx_mag}; final |vx| = {final_vx_mag}"
        );
    }

    #[test]
    fn constant_action_does_not_accumulate() {
        // Regression for #98 (ADR 0037): a constant actuator torque must live
        // exactly ONE integration step and must not accumulate across substeps
        // or env steps.
        //
        // We assert the invariant *directly* via the residual `user_torque`
        // accumulator rather than through the chain's emergent angular velocity.
        // The 3-segment swimmer is a driven multi-body chain integrated over
        // many substeps; a hard bound on its emergent |ω| sits only ~1.24×
        // above the measured value (40.4 vs a 50 cap) and depends on nonlinear
        // trajectory details that diverge across CPU architectures (aarch64 vs
        // x86_64 libm/SIMD rounding). The applied torque (`gear × action`) and
        // its running sum are plain IEEE-754 adds, so the residual probe is
        // bit-identical everywhere.
        //
        //   * With the fix, `step_once` calls `reset_external_forces` after the
        //     final substep, so every body's accumulator is exactly 0.
        //   * With the pre-fix bug the accumulator is never cleared and grows
        //     unbounded as `steps × frame_skip × τ` is summed (net of the
        //     opposing angular drag torque) — failing the `< 1e-3` bound.
        //     (Confirmed: reverting `reset_external_forces` leaves segment1 with
        //     a residual of ~86 N·m.)
        const STEPS: usize = 3;
        const RESIDUAL_TOL: f32 = 1e-3;

        let mut env = SwimmerRapier::with_config(deterministic_cfg()).expect("valid config");
        env.reset().unwrap();

        // Clipped action ±1.0 × gear 5 ⇒ ±5 N·m joint torque per substep.
        let mut moved = false;
        for i in 0..STEPS {
            let snap = env.step(SwimmerAction::new(1.0, -1.0)).unwrap();
            let obs = snap.observation();
            assert!(obs.is_finite(), "obs must stay finite at step {i}");
            moved |= obs.joint1_dot().abs() > 0.0 || obs.joint2_dot().abs() > 0.0;
        }

        // segment1/segment2 are the joint children that carry the actuator
        // torque (and per-substep drag); both accumulators must be zeroed after
        // `step`. `user_torque`/`user_force` return glam `Vec3`s, so magnitude
        // is `.length()`.
        let residual = |body: RigidBodyHandle| -> (f32, f32) {
            let rb = env.world.bodies().get(body).expect("segment body exists");
            (rb.user_torque().length(), rb.user_force().length())
        };
        for (name, handle) in [
            ("segment1", env.state.segment1),
            ("segment2", env.state.segment2),
        ] {
            let (torque, force): (f32, f32) = residual(handle);
            assert!(
                torque < RESIDUAL_TOL && force < RESIDUAL_TOL,
                "{name} accumulators must be cleared after each step (ADR 0037); \
                 residual |τ| = {torque} N·m, |F| = {force} N (the accumulation \
                 bug leaves |τ| ~86)"
            );
        }
        assert!(
            moved,
            "constant action should have moved the joints at least once"
        );
    }

    #[test]
    fn forward_reward_positive_for_forward_motion() {
        // A quarter-phase-lagged sinusoidal stroke propagates a travelling
        // wave down the chain — the canonical swimmer gait. Over a 300-step
        // rollout the net forward-reward component must be strictly
        // positive. (A bare alternating-every-env-step gait works too over
        // ~300 steps but the sinusoidal version converges faster and gives
        // the test tighter tolerance.)
        use std::f32::consts::{FRAC_PI_2, TAU};
        let mut env = SwimmerRapier::with_config(deterministic_cfg()).expect("valid config");
        env.reset().unwrap();
        let mut total_forward = 0.0f32;
        for i in 0..300 {
            let t = i as f32 * TAU / 10.0;
            let a = SwimmerAction::new(t.sin(), (t - FRAC_PI_2).sin());
            let snap = env.step(a).unwrap();
            total_forward += snap.metadata().unwrap().components[METADATA_KEY_FORWARD];
        }
        assert!(
            total_forward > 0.0,
            "sinusoidal swim stroke should generate net forward reward, got {total_forward}"
        );
    }
}
