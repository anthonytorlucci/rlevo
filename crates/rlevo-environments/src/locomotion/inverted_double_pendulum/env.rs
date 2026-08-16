//! `InvertedDoublePendulum` environment implementation.

use std::marker::PhantomData;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};
use rapier3d::prelude::*;
use rlevo_core::config::{ConfigError, Validate};
use rlevo_core::environment::{
    ConstructableEnv, Environment, EnvironmentError, EpisodeStatus, Sensor, SnapshotMetadata,
};
use rlevo_core::reward::ScalarReward;

use crate::episode::EpisodeGuard;
use crate::locomotion::backend::{LocomotionBackend, Pose, Rapier3DBackend, Rapier3DWorld};
use crate::locomotion::common::{LocomotionSnapshot, TerminationMode, wrap_to_pi};

use super::action::InvertedDoublePendulumAction;
use super::config::InvertedDoublePendulumConfig;
use super::observation::InvertedDoublePendulumObservation;
use super::state::InvertedDoublePendulumState;

/// Reward-component key: alive bonus (`+10` while healthy, `0` otherwise).
pub const METADATA_KEY_ALIVE: &str = "alive";
/// Reward-component key: tip-distance penalty `$-0.01 \cdot x_{\text{tip}}^2 - (y_{\text{tip}} - 2)^2$`.
pub const METADATA_KEY_DISTANCE: &str = "distance";
/// Reward-component key: angular-velocity penalty `$-1\text{e-}3 \cdot |\omega_1| - 5\text{e-}3 \cdot |\omega_2|$`.
pub const METADATA_KEY_VELOCITY: &str = "velocity";

/// Cart-pole-pole balance task backed by a pluggable physics engine.
///
/// The cart slides along world-x. Pole1 is attached to the cart top via a
/// revolute-y joint, and pole2 is attached to pole1's top via a second
/// revolute-y joint. The agent applies a horizontal force to the cart each
/// step, attempting to keep the tip of pole2 as close as possible to
/// `y_tip = 2.0` (world-z in this coordinate system).
///
/// Use [`InvertedDoublePendulumRapier`] for the default Rapier3D-backed
/// variant, or supply an alternative `B: LocomotionBackend` for testing.
///
/// See the [module documentation](super) for the full observation layout,
/// reward formula, and Gymnasium divergence notes.
///
/// A [`step`](Environment::step) taken after the episode ended is rejected with
/// [`EnvironmentError::StepAfterEpisodeEnd`] — see the [`EpisodeGuard`] field.
#[derive(Debug)]
pub struct InvertedDoublePendulum<B: LocomotionBackend = Rapier3DBackend> {
    world: B::World,
    state: InvertedDoublePendulumState,
    config: InvertedDoublePendulumConfig,
    rng: StdRng,
    steps: usize,
    /// Rejects a `step()` taken after the tip dropped (or the episode was
    /// truncated). Neither status is a latch the physics enforces: the
    /// healthiness test is a live read of the current tip height and the step
    /// counter keeps climbing, so an unguarded post-terminal step integrates
    /// another frame, ticks `steps` and pays a fresh reward — and a chaotic
    /// double pendulum whose tip swings back above the threshold even re-earns
    /// the `+10` alive bonus on a `Running` snapshot, silently resurrecting the
    /// episode.
    guard: EpisodeGuard,
    _marker: PhantomData<B>,
}

/// Convenience alias for the Rapier3D-backed environment. This is the type
/// most callers should use directly.
pub type InvertedDoublePendulumRapier = InvertedDoublePendulum<Rapier3DBackend>;

impl InvertedDoublePendulum<Rapier3DBackend> {
    /// Create an environment with an explicit configuration.
    ///
    /// The Rapier world and initial physics state are built immediately using
    /// `config.seed` and `config.reset_noise_scale`. Call `reset` before the
    /// first `step` to obtain the initial observation snapshot.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `config` fails [`Validate`] (e.g.
    /// non-positive `dt`, inverted `action_clip`, or negative reward weights).
    pub fn with_config(config: InvertedDoublePendulumConfig) -> Result<Self, ConfigError> {
        config.validate()?;
        let mut rng = StdRng::seed_from_u64(config.seed);
        let (world, state) = Self::build_world(&config, &mut rng);
        Ok(Self {
            world,
            state,
            config,
            rng,
            steps: 0,
            guard: EpisodeGuard::new(),
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
    /// Delegates to [`reset`](Environment::reset) for the rebuild, so it
    /// re-opens the [`EpisodeGuard`] on exactly the same terms — there is no
    /// second reset path that could forget to clear it.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`reset`](Environment::reset) (currently none).
    pub fn reset_with_seed(
        &mut self,
        seed: u64,
    ) -> Result<LocomotionSnapshot<InvertedDoublePendulumObservation>, EnvironmentError> {
        self.rng = StdRng::seed_from_u64(seed);
        self.reset()
    }

    // Justified: paired per-joint initial values differ only by joint index.
    #[allow(clippy::similar_names)]
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
        // Disable jointed-neighbour contacts: the cart top face and pole1's
        // bottom cap overlap at the shared anchor, which would otherwise seed
        // permanent internal contacts (MuJoCo parent–child filter parity, ADR 0041).
        let joint1 = RevoluteJointBuilder::new(y_axis)
            .local_anchor1(Vector::new(0.0, 0.0, cart_half_z))
            .local_anchor2(Vector::new(0.0, 0.0, -pole_half))
            .contacts_enabled(false)
            .build();
        let joint1_handle = world.add_impulse_joint(cart, pole1, joint1);

        // Disable jointed-neighbour contacts: pole1's top cap and pole2's
        // bottom cap overlap at the shared anchor, which would otherwise seed
        // permanent internal contacts polluting obs[8] (MuJoCo parent–child
        // filter parity, ADR 0041).
        let joint2 = RevoluteJointBuilder::new(y_axis)
            .local_anchor1(Vector::new(0.0, 0.0, pole_half))
            .local_anchor2(Vector::new(0.0, 0.0, -pole_half))
            .contacts_enabled(false)
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

    /// Sample the current physics state and pack it into a 9-element
    /// observation. θ₂ is the **relative** elbow angle (pole2 world angle
    /// minus pole1 world angle), wrapped to `$(-\pi, \pi]$`.
    ///
    /// `obs[8]` packs `Rapier3DBackend::contact_force(pole2)[0]` — a Cartesian
    /// contact-manifold force on pole2 — where Gymnasium's `InvertedDoublePendulum-v5`
    /// packs `clip(qfrc_constraint, -10, 10)[:1]`, the generalized constraint force on
    /// the cart's slider DOF. These are semantically distinct quantities: this slot is
    /// not a stand-in for the Gymnasium value, it measures something else entirely.
    /// With jointed-neighbour contacts disabled for `MuJoCo` parent–child filter parity
    /// (ADR 0041), pole2 touches nothing in normal operation, so `obs[8]` is `≈ 0` in
    /// practice — a dead observation dimension versus Gymnasium's informative
    /// constraint-force signal. This deviation is currently unresolved: either
    /// `obs[8]` should be re-modeled as the true slider-DOF constraint-force analogue,
    /// or the deviation should be documented as deliberate.
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

    /// Compute the world-x cart force for `action` (clip → gear). Pure: the
    /// force is *applied* inside the `step_actuated` closure so it is re-applied
    /// fresh each substep (ADR 0037 force-lifetime contract).
    fn control_force(&self, action: InvertedDoublePendulumAction) -> f32 {
        let (lo, hi): (f32, f32) = self.config.action_clip.into();
        let clipped = [action.0[0].clamp(lo, hi)];
        let torques = self.config.gear.apply(&clipped);
        torques[0]
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

impl Sensor<1, 1, 1> for InvertedDoublePendulum<Rapier3DBackend> {
    type Action = InvertedDoublePendulumAction;
    type State = InvertedDoublePendulumState;
    type Observation = InvertedDoublePendulumObservation;

    /// Emission model: reads the 9-element observation directly from the physics
    /// world through the state's body handles. The action does not enter the
    /// observation, and `next_state` carries the same handles as `self.state`,
    /// so both are unused; the world is the source of truth.
    fn observe(
        &self,
        _action: &InvertedDoublePendulumAction,
        _next_state: &InvertedDoublePendulumState,
    ) -> InvertedDoublePendulumObservation {
        self.extract_observation()
    }

    /// Initial observation at episode start, read from the freshly built world.
    fn observe_reset(
        &self,
        _state: &InvertedDoublePendulumState,
    ) -> InvertedDoublePendulumObservation {
        self.extract_observation()
    }
}

impl ConstructableEnv for InvertedDoublePendulum<Rapier3DBackend> {
    /// Create an environment with default configuration.
    ///
    /// The `render` flag is accepted for interface compatibility but has no
    /// effect; this environment does not produce any visual output.
    fn new(_render: bool) -> Self {
        Self::with_config(InvertedDoublePendulumConfig::default())
            .expect("default config must validate")
    }
}

impl Environment<1, 1, 1> for InvertedDoublePendulum<Rapier3DBackend> {
    type StateType = InvertedDoublePendulumState;
    type ObservationType = InvertedDoublePendulumObservation;
    type ActionType = InvertedDoublePendulumAction;
    type RewardType = ScalarReward;
    type SnapshotType = LocomotionSnapshot<InvertedDoublePendulumObservation>;

    /// Reset the environment to a freshly sampled initial state and return an
    /// opening snapshot with zero reward.
    ///
    /// Reset perturbations are drawn from the environment's persistent RNG. The
    /// stream **advances** across resets, so successive episodes see
    /// independent initial states. For deterministic replay of a specific
    /// initial state, use [`InvertedDoublePendulum::reset_with_seed`].
    ///
    /// Re-opens the [`EpisodeGuard`], so an environment whose tip dropped (or
    /// whose episode was truncated) becomes steppable again. The guard is
    /// cleared only after the new world is in hand: nothing here is fallible
    /// today, but the ordering keeps the ADR 0044 §6 rule ("clear the guard only
    /// once the rebuild has succeeded") true by construction if `build_world`
    /// ever becomes so.
    ///
    /// # Errors
    ///
    /// This implementation does not currently return an error; the signature
    /// is required by the `Environment` trait.
    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        let (world, mut state) = Self::build_world(&self.config, &mut self.rng);
        self.world = world;
        state.last_obs = InvertedDoublePendulumObservation::default();
        self.state = state;
        self.steps = 0;
        self.guard.reset();

        let obs = self.observe_reset(&self.state);
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
        Ok(LocomotionSnapshot::running(obs, ScalarReward(0.0)).with_metadata(meta))
    }

    /// Advance the simulation by one step (or `frame_skip` Rapier substeps)
    /// and return the resulting snapshot.
    ///
    /// The snapshot status is:
    /// - `Running` — tip is above the healthy threshold and `max_steps` not
    ///   reached.
    /// - `Terminated` — tip fell below `healthy.z_range` lower bound
    ///   (`$y_{\text{tip}} \leq 1.0$` by default) and `termination` is
    ///   `OnUnhealthy`.
    /// - `Truncated` — `steps >= max_steps`.
    ///
    /// # Errors
    ///
    /// Returns [`EnvironmentError::StepAfterEpisodeEnd`] if the episode has
    /// already ended (the tip dropped, or `max_steps` was reached); call
    /// [`reset`](Environment::reset) first. Returns
    /// [`EnvironmentError::InvalidAction`] if the action value is non-finite
    /// (`NaN` or infinity).
    fn step(
        &mut self,
        action: InvertedDoublePendulumAction,
    ) -> Result<Self::SnapshotType, EnvironmentError> {
        // Guard first — ahead of even the finiteness check: the episode being
        // over is a fact about the *call sequence*, independent of whether the
        // action itself is well-formed, so a caller replaying a malformed action
        // past the terminal must hear about the finished episode, not get
        // `InvalidAction` and the wrong diagnosis. It also runs before
        // `step_actuated`, the `steps` tick and the observation refresh, so a
        // rejected call leaves the physics world, the counter and the RNG stream
        // exactly as the terminal step left them (ADR 0029).
        self.guard.check()?;

        if !action.0[0].is_finite() {
            return Err(EnvironmentError::InvalidAction(format!(
                "InvertedDoublePendulum action must be finite, got {}",
                action.0[0]
            )));
        }

        // Re-apply the constant cart force before every substep so it is held
        // across the frame skip and cannot accumulate (ADR 0037). The handle is
        // `Copy` and `force` is precomputed, so the closure borrows only the
        // world — not `self`.
        let force = self.control_force(action);
        let cart_handle = self.state.cart;
        self.world.step_actuated(|w| {
            if let Some(cart) = w.bodies_mut().get_mut(cart_handle) {
                cart.add_force(Vector::new(force, 0.0, 0.0), true);
            }
        });
        self.steps += 1;

        let obs = self.observe(&action, &self.state);
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
        // Single exit: exactly one snapshot is built, and the guard is fed that
        // snapshot's own status, so no branch can forget to record.
        let snapshot = LocomotionSnapshot {
            observation: obs,
            reward: ScalarReward(total),
            status,
            metadata: Some(meta),
        };
        self.guard.record(snapshot.status);
        Ok(snapshot)
    }
}

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
// Justified: single-letter names mirror the reference dynamics equations.
#[allow(clippy::many_single_char_names)]
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
    // Exact comparison is intentional throughout this test module: the values
    // are literals or seeds read back without arithmetic, or two identically
    // seeded runs that must agree bit-for-bit. A tolerance would let a real
    // regression pass. Reviewed as a class, not site-by-site.
    #![allow(clippy::float_cmp)]

    use super::*;
    use crate::episode::assert_rejects_post_terminal_step;
    use rlevo_core::action::ContinuousAction;
    use rlevo_core::base::Action;
    use rlevo_core::base::Observation;
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
        let mut env =
            InvertedDoublePendulumRapier::with_config(deterministic_cfg()).expect("valid config");
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
        })
        .expect("valid config");
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
        let mut env =
            InvertedDoublePendulumRapier::with_config(deterministic_cfg()).expect("valid config");
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
        })
        .expect("valid config");
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
    fn jointed_neighbor_contacts_produce_no_wrench() {
        // Regression for #123 (ADR 0041). Pole1's top cap and pole2's bottom
        // cap both extend `pole_radius` past the shared elbow anchor, so with
        // rapier's default `contacts_enabled = true` the solver generated
        // permanent internal contacts between the jointed neighbours. That
        // internal wrench leaked into `contact_force(pole2)` — packed as
        // observation index 8. MuJoCo's parent–child contact filter excludes
        // exactly these pairs ("avoid permanent contacts within bodies and
        // joints"); disabling jointed-neighbour contacts restores parity and
        // drives the wrench to ~0 in normal operation.
        let mut env =
            InvertedDoublePendulumRapier::with_config(deterministic_cfg()).expect("valid config");
        env.reset().unwrap();
        for _ in 0..10 {
            env.step(InvertedDoublePendulumAction::new(0.0)).unwrap();
        }
        let wrench = Rapier3DBackend::contact_force(&env.world, env.state.pole2);
        let max_abs = wrench.iter().fold(0.0f32, |m, &c| m.max(c.abs()));
        assert!(
            max_abs < 1e-4,
            "pole2 must feel no internal contact wrench once jointed-neighbour \
             contacts are disabled; got {wrench:?} (max |component| = {max_abs})"
        );
    }

    #[test]
    fn constraint_force_is_finite() {
        let mut env =
            InvertedDoublePendulumRapier::with_config(deterministic_cfg()).expect("valid config");
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
        })
        .expect("valid config");
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
            let mut env =
                InvertedDoublePendulumRapier::with_config(cfg.clone()).expect("valid config");
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
        let mut env =
            InvertedDoublePendulumRapier::with_config(deterministic_cfg()).expect("valid config");
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
        })
        .expect("valid config");
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

    #[test]
    fn constant_force_does_not_accumulate() {
        // Regression test (ADR 0037): a constant action must produce a
        // stationary per-step cart-velocity increment. Rapier's `user_force`
        // does not auto-clear each step despite the vendored 0.32 doc comment
        // claiming otherwise; an unguarded `add_force` therefore accumulated
        // across steps, so Δvx grew ~linearly and silently corrupted the
        // control dynamics — existing tests stayed green because they only
        // checked qualitative movement, not whether the applied force stayed
        // bounded. Fixed once in the shared `RapierWorld::step()` (calling
        // `reset_forces`/`reset_torques` each step), with per-env force
        // constants re-tuned since the old values were implicitly tuned
        // around the accumulation bug.
        let mut env = InvertedDoublePendulumRapier::with_config(InvertedDoublePendulumConfig {
            seed: 1,
            reset_noise_scale: 0.0,
            termination: TerminationMode::Never,
            max_steps: 10_000,
            ..Default::default()
        })
        .expect("valid config");
        env.reset().unwrap();

        let mut prev_vx = 0.0f32;
        let mut deltas: Vec<f32> = Vec::new();
        for _ in 0..40 {
            let snap = env.step(InvertedDoublePendulumAction::new(1.0)).unwrap();
            assert!(snap.observation().is_finite(), "obs must stay finite");
            let vx = snap.observation().cart_velocity();
            deltas.push(vx - prev_vx);
            prev_vx = vx;
        }

        let early: f32 = deltas[0..5].iter().sum::<f32>() / 5.0;
        let late: f32 = deltas[35..40].iter().sum::<f32>() / 5.0;
        assert!(
            early > 0.0,
            "force should accelerate the cart (early Δv={early})"
        );
        assert!(
            late < early * 5.0,
            "per-step Δvx must not grow under constant force: early={early}, late={late}"
        );
    }

    #[test]
    fn two_successive_resets_differ() {
        // The persistent RNG advances across resets (default reset noise > 0),
        // so back-to-back resets must sample independent initial states.
        let mut env = InvertedDoublePendulumRapier::with_config(InvertedDoublePendulumConfig {
            seed: 7,
            ..Default::default()
        })
        .expect("valid config");
        let first = env.reset().unwrap().observation().0;
        let second = env.reset().unwrap().observation().0;
        assert_ne!(
            first, second,
            "successive resets must draw independent initial states"
        );
    }

    #[test]
    fn reset_with_seed_is_reproducible() {
        let mut env = InvertedDoublePendulumRapier::with_config(InvertedDoublePendulumConfig {
            seed: 7,
            ..Default::default()
        })
        .expect("valid config");
        let a = env.reset_with_seed(999).unwrap().observation().0;
        // Advance the stream with an ordinary reset, then re-seed identically.
        env.reset().unwrap();
        let b = env.reset_with_seed(999).unwrap().observation().0;
        assert_eq!(
            a, b,
            "reset_with_seed must reproduce the same initial state"
        );
    }

    // ── post-terminal step guard (ADR 0044, issue #292) ──────────────────────
    //
    // Before this guard, `step` only checked `action.0[0].is_finite()` —
    // nothing rejected a call made after the episode had already ended.
    // Since both `Terminated` (unhealthy tip) and `Truncated`
    // (`steps >= max_steps`) hold once reached, an unguarded post-terminal
    // step kept integrating the Rapier physics sim for another
    // `dt * frame_skip`, so the observation drifted further with every extra
    // call instead of staying pinned at the terminal state. The fix:
    // `self.guard.check()?` runs first in `step`, ahead of the
    // action-finiteness check and the physics substeps (see `step`'s doc
    // comment); `self.guard.record(status)` is called on the single
    // snapshot-producing return path; and `reset` clears the guard (ADR 0044
    // §6) so a truncated episode becomes steppable again.
    //
    // The tests below drive a real `Terminated` (tip drops below the healthy
    // `z_range` floor via `drive_to_tip_drop`) and combine the shared
    // conformance check (`crate::episode::assert_rejects_post_terminal_step`)
    // with InvertedDoublePendulum-specific regressions pinning exactly what
    // an unguarded step would have mutated.

    /// Upper bound on the steps the driven pendulum may take before the test
    /// calls it a regression. `max_steps` is set well above it so truncation
    /// cannot pre-empt the termination the test is driving to.
    const FALL_STEP_CAP: usize = 2000;

    /// Config that lets the episode end by a genuine `Terminated`: no reset
    /// noise (deterministic), `OnUnhealthy` termination, and a truncation budget
    /// far beyond `FALL_STEP_CAP`.
    fn terminating_cfg() -> InvertedDoublePendulumConfig {
        InvertedDoublePendulumConfig {
            seed: 7,
            reset_noise_scale: 0.0,
            termination: TerminationMode::OnUnhealthy,
            max_steps: 10_000,
            ..Default::default()
        }
    }

    /// Drives a fresh episode to a real **`Terminated`** — not a truncation: a
    /// sustained +1.0 cart force drops the tip below the healthy `z_range` floor
    /// (`$y_{\text{tip}} \leq 1.0$`), and the `OnUnhealthy` rule ends the episode through the
    /// same physics path an agent would hit.
    fn drive_to_tip_drop(
        env: &mut InvertedDoublePendulum<Rapier3DBackend>,
    ) -> LocomotionSnapshot<InvertedDoublePendulumObservation> {
        env.reset().expect("reset must succeed");
        for _ in 0..FALL_STEP_CAP {
            let snap = env
                .step(InvertedDoublePendulumAction::new(1.0))
                .expect("step must succeed while the episode is running");
            if snap.is_done() {
                assert!(
                    snap.is_terminated(),
                    "this fixture must end by termination, not truncation"
                );
                return snap;
            }
        }
        panic!("a sustained +1.0 action must drop the tip within {FALL_STEP_CAP} steps");
    }

    #[test]
    /// `InvertedDoublePendulum` satisfies the shared post-terminal conformance
    /// check: once the tip has dropped (`Terminated`), a further step with a
    /// *legal* action fails with `StepAfterEpisodeEnd` carrying that same
    /// status. The replayed action is `0.0` — squarely inside the valid range —
    /// so the rejection can only be on call-sequence grounds, never on the
    /// action's own validity.
    fn test_inverted_double_pendulum_rejects_post_terminal_step() {
        let mut env =
            InvertedDoublePendulumRapier::with_config(terminating_cfg()).expect("valid config");
        assert_rejects_post_terminal_step(
            &mut env,
            drive_to_tip_drop,
            InvertedDoublePendulumAction::new(0.0),
        );
    }

    #[test]
    /// A rejected post-terminal step must mutate nothing observable. The
    /// healthiness test is a live read of the current tip height, not a latch,
    /// so before the guard a further step integrated another frame, ticked the
    /// step counter and paid a fresh reward — and a chaotic tip that swings back
    /// above the threshold could re-earn the `+10` alive bonus on a `Running`
    /// snapshot.
    fn test_inverted_double_pendulum_post_terminal_step_does_not_mutate_state() {
        let mut env =
            InvertedDoublePendulumRapier::with_config(terminating_cfg()).expect("valid config");
        let terminal = drive_to_tip_drop(&mut env);
        let ended = terminal.status();

        let obs_at_end = terminal.observation().0;
        let steps_at_end = env.steps;

        let err = env
            .step(InvertedDoublePendulumAction::new(0.0))
            .expect_err("a step after the tip dropped must return Err, not another snapshot");
        match err {
            EnvironmentError::StepAfterEpisodeEnd { status } => assert_eq!(
                status, ended,
                "the error must carry the status that ended the episode"
            ),
            other => panic!("expected StepAfterEpisodeEnd, got {other:?}"),
        }

        assert_eq!(
            env.steps, steps_at_end,
            "a rejected step must not tick the step counter"
        );
        // Recomputed from the *live* physics world, so equality here proves the
        // rejected call never ran `step_actuated`.
        assert_eq!(
            env.extract_observation().0,
            obs_at_end,
            "a rejected step must leave the observation byte-identical"
        );
        assert_eq!(
            env.guard.status(),
            ended,
            "a rejected step must not reopen the episode"
        );
    }

    #[test]
    /// The guard outranks the action check. Replaying a *malformed* action past
    /// the terminal must still report `StepAfterEpisodeEnd`, not
    /// `InvalidAction`: the episode being over is a fact about the call
    /// sequence, and the finished episode is the diagnosis the caller needs.
    fn test_inverted_double_pendulum_post_terminal_step_outranks_the_action_check() {
        let mut env =
            InvertedDoublePendulumRapier::with_config(terminating_cfg()).expect("valid config");
        drive_to_tip_drop(&mut env);

        let err = env
            .step(InvertedDoublePendulumAction::new(f32::NAN))
            .expect_err("a NaN action after termination must still be an error");
        assert!(
            matches!(err, EnvironmentError::StepAfterEpisodeEnd { .. }),
            "the finished episode must be reported before the action's validity, got {err:?}"
        );
    }

    #[test]
    /// A rejected step must not draw from the persistent RNG either (ADR 0029):
    /// the episode that follows the next `reset()` must be the one the caller
    /// would have got had the mis-sequenced call never happened.
    fn test_inverted_double_pendulum_post_terminal_step_does_not_advance_rng() {
        // Non-zero reset noise so the next reset actually consumes the stream.
        let cfg = || InvertedDoublePendulumConfig {
            seed: 7,
            termination: TerminationMode::OnUnhealthy,
            max_steps: 10_000,
            ..Default::default()
        };
        let drive = |env: &mut InvertedDoublePendulumRapier| {
            env.reset().unwrap();
            for _ in 0..FALL_STEP_CAP {
                if env
                    .step(InvertedDoublePendulumAction::new(1.0))
                    .unwrap()
                    .is_done()
                {
                    return;
                }
            }
            panic!("the tip must drop within {FALL_STEP_CAP} steps");
        };

        let with_replay = {
            let mut env = InvertedDoublePendulumRapier::with_config(cfg()).expect("valid config");
            drive(&mut env);
            env.step(InvertedDoublePendulumAction::new(0.0))
                .expect_err("the post-terminal step must be rejected");
            env.reset().unwrap().observation().0
        };
        let without_replay = {
            let mut env = InvertedDoublePendulumRapier::with_config(cfg()).expect("valid config");
            drive(&mut env);
            env.reset().unwrap().observation().0
        };
        assert_eq!(
            with_replay, without_replay,
            "a rejected step must leave the RNG stream untouched"
        );
    }

    #[test]
    /// `reset()` re-opens a finished episode, so a latched guard cannot strand
    /// the environment for the rest of the run.
    fn test_inverted_double_pendulum_reset_reopens_terminated_episode() {
        let mut env =
            InvertedDoublePendulumRapier::with_config(terminating_cfg()).expect("valid config");
        drive_to_tip_drop(&mut env);
        assert!(
            env.step(InvertedDoublePendulumAction::new(0.0)).is_err(),
            "the episode has ended; a step must be rejected before reset()"
        );

        let first = env.reset().expect("reset must succeed after termination");
        assert!(!first.is_done(), "a fresh episode must not start done");
        assert_eq!(
            env.guard.status(),
            EpisodeStatus::Running,
            "reset() must return the guard to Running"
        );
        let snap = env
            .step(InvertedDoublePendulumAction::new(0.0))
            .expect("reset() must re-open the environment for a new episode");
        assert!(
            !snap.is_done(),
            "the first step of a fresh episode must not be done"
        );
    }

    #[test]
    /// The truncation limb of the guard: `reset_with_seed` delegates to
    /// `reset`, so it must clear the guard on the same terms — and a step past
    /// a `Truncated` snapshot must report `Truncated`, not `Terminated`.
    fn test_inverted_double_pendulum_rejects_post_truncation_step_and_reset_with_seed_reopens() {
        let mut env = InvertedDoublePendulumRapier::with_config(InvertedDoublePendulumConfig {
            seed: 1,
            reset_noise_scale: 0.0,
            max_steps: 5,
            termination: TerminationMode::Never,
            ..Default::default()
        })
        .expect("valid config");
        env.reset().unwrap();
        let mut last = None;
        for _ in 0..5 {
            last = Some(env.step(InvertedDoublePendulumAction::new(0.0)).unwrap());
        }
        assert_eq!(
            last.expect("five steps were taken").status(),
            EpisodeStatus::Truncated
        );

        let err = env
            .step(InvertedDoublePendulumAction::new(0.0))
            .expect_err("a step after truncation must be rejected");
        assert!(
            matches!(
                err,
                EnvironmentError::StepAfterEpisodeEnd {
                    status: EpisodeStatus::Truncated
                }
            ),
            "the error must distinguish truncation from termination, got {err:?}"
        );

        env.reset_with_seed(42)
            .expect("reset_with_seed must succeed after truncation");
        assert!(
            env.step(InvertedDoublePendulumAction::new(0.0)).is_ok(),
            "reset_with_seed must re-open the environment for a new episode"
        );
    }
}
