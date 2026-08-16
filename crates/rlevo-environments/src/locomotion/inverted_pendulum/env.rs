//! `InvertedPendulum` environment implementation.

use std::marker::PhantomData;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rapier3d::prelude::*;
use rlevo_core::config::{ConfigError, Validate};
use rlevo_core::environment::{
    ConstructableEnv, Environment, EnvironmentError, EpisodeStatus, Sensor, SnapshotMetadata,
};
use rlevo_core::reward::ScalarReward;

use crate::episode::EpisodeGuard;
use crate::locomotion::backend::{LocomotionBackend, Rapier3DBackend, Rapier3DWorld};
use crate::locomotion::common::{LocomotionSnapshot, TerminationMode, wrap_to_pi};

use super::action::InvertedPendulumAction;
use super::config::InvertedPendulumConfig;
use super::observation::InvertedPendulumObservation;
use super::state::InvertedPendulumState;

/// Reward-component metadata key: `+1` if alive at this step, `0` otherwise.
pub const METADATA_KEY_ALIVE: &str = "alive";

/// `InvertedPendulum` — cart-pole balance in 3D, with the cart restricted to
/// the world-x axis and the pole free to rotate about the world-y axis.
///
/// Generic in the physics backend; v1 only implements `B = Rapier3DBackend`
/// (see [`InvertedPendulumRapier`] for the default type alias).
///
/// A [`step`](Environment::step) taken after the episode ended is rejected with
/// [`EnvironmentError::StepAfterEpisodeEnd`] — see the [`EpisodeGuard`] field.
#[derive(Debug)]
pub struct InvertedPendulum<B: LocomotionBackend = Rapier3DBackend> {
    world: B::World,
    state: InvertedPendulumState,
    config: InvertedPendulumConfig,
    rng: StdRng,
    steps: usize,
    /// Rejects a `step()` taken after the pole fell (or the episode was
    /// truncated). Neither status is a latch the physics enforces: the
    /// healthiness test is a live read of the current pole angle and the step
    /// counter keeps climbing, so an unguarded post-terminal step integrates
    /// another frame, ticks `steps` and emits a fresh alive bonus — a toppled
    /// pole that swings back through `$|\theta| < 0.2$` even re-earns `+1` on a
    /// `Running` snapshot, silently resurrecting the episode.
    guard: EpisodeGuard,
    _marker: PhantomData<B>,
}

/// Default backend alias.
pub type InvertedPendulumRapier = InvertedPendulum<Rapier3DBackend>;

impl InvertedPendulum<Rapier3DBackend> {
    /// Create with an explicit configuration.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `config` fails [`Validate`] (e.g.
    /// non-positive `dt`, inverted `action_clip`, or non-positive body
    /// dimensions).
    pub fn with_config(config: InvertedPendulumConfig) -> Result<Self, ConfigError> {
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
    ) -> Result<LocomotionSnapshot<InvertedPendulumObservation>, EnvironmentError> {
        self.rng = StdRng::seed_from_u64(seed);
        self.reset()
    }

    // Justified: paired per-joint initial values differ only by joint index.
    #[allow(clippy::similar_names)]
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
        // Disable jointed-neighbour contacts: the cart top face and the pole's
        // bottom cap overlap at the shared anchor, which would otherwise seed
        // permanent internal contacts (MuJoCo parent–child filter parity, ADR 0041).
        let y_axis: Vector = Vector::new(0.0, 1.0, 0.0);
        let joint = RevoluteJointBuilder::new(y_axis)
            .local_anchor1(Vector::new(0.0, 0.0, config.cart_half_extents[2]))
            .local_anchor2(Vector::new(0.0, 0.0, -pole_half))
            .contacts_enabled(false)
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
        // `$(\cos(\theta/2), 0, \sin(\theta/2), 0)$` in `[w, x, y, z]` order. Recover θ:
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

    /// Compute the world-x cart force for `action` (clip → gear). Pure: the
    /// force is *applied* inside the `step_actuated` closure so it is re-applied
    /// fresh each substep (ADR 0037 force-lifetime contract).
    fn control_force(&self, action: InvertedPendulumAction) -> f32 {
        let (lo, hi): (f32, f32) = self.config.action_clip.into();
        let clipped = [action.0[0].clamp(lo, hi)];
        let torques = self.config.gear.apply(&clipped);
        torques[0]
    }
}

impl Sensor<1, 1, 1> for InvertedPendulum<Rapier3DBackend> {
    type Action = InvertedPendulumAction;
    type State = InvertedPendulumState;
    type Observation = InvertedPendulumObservation;

    /// Emission model: reads `[cart_x, pole_angle, cart_vx, pole_angvel_y]`
    /// directly from the physics world through the state's body handles. The
    /// action does not enter the observation, and `next_state` carries the same
    /// handles as `self.state`, so both are unused; the world is the source of
    /// truth.
    fn observe(
        &self,
        _action: &InvertedPendulumAction,
        _next_state: &InvertedPendulumState,
    ) -> InvertedPendulumObservation {
        self.extract_observation()
    }

    /// Initial observation at episode start, read from the freshly built world.
    fn observe_reset(&self, _state: &InvertedPendulumState) -> InvertedPendulumObservation {
        self.extract_observation()
    }
}

impl ConstructableEnv for InvertedPendulum<Rapier3DBackend> {
    /// Constructs the environment with [`InvertedPendulumConfig::default`].
    ///
    /// The `render` flag is accepted for interface conformance but has no
    /// effect; this environment has no built-in renderer. Use
    /// [`InvertedPendulum::with_config`] for full control.
    fn new(_render: bool) -> Self {
        Self::with_config(InvertedPendulumConfig::default()).expect("default config must validate")
    }
}

impl Environment<1, 1, 1> for InvertedPendulum<Rapier3DBackend> {
    type StateType = InvertedPendulumState;
    type ObservationType = InvertedPendulumObservation;
    type ActionType = InvertedPendulumAction;
    type RewardType = ScalarReward;
    type SnapshotType = LocomotionSnapshot<InvertedPendulumObservation>;

    /// Resets the environment to a new initial state sampled from
    /// `U(-reset_noise_scale, reset_noise_scale)` on each of the four state
    /// variables, drawn from the environment's persistent RNG. The stream
    /// **advances** across resets, so successive episodes see independent
    /// initial states. For deterministic replay of a specific initial state,
    /// use [`InvertedPendulum::reset_with_seed`].
    ///
    /// Returns a `Running` snapshot with reward `0.0` and the initial
    /// observation. The `METADATA_KEY_ALIVE` component is set to `0.0` on
    /// reset regardless of pole angle.
    ///
    /// Re-opens the [`EpisodeGuard`], so an environment whose pole fell (or
    /// whose episode was truncated) becomes steppable again. The guard is
    /// cleared only after the new world is in hand: nothing here is fallible
    /// today, but the ordering keeps the ADR 0044 §6 rule ("clear the guard only
    /// once the rebuild has succeeded") true by construction if `build_world`
    /// ever becomes so.
    ///
    /// # Errors
    ///
    /// This implementation is currently infallible, but returns `Result` for
    /// trait conformance.
    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        let (world, mut state) = Self::build_world(&self.config, &mut self.rng);
        self.world = world;
        state.last_obs = InvertedPendulumObservation::default();
        self.state = state;
        self.steps = 0;
        self.guard.reset();

        let obs = self.observe_reset(&self.state);
        self.state.last_obs = obs;
        let meta = SnapshotMetadata::new().with(METADATA_KEY_ALIVE, 0.0);
        Ok(LocomotionSnapshot::running(obs, ScalarReward(0.0)).with_metadata(meta))
    }

    /// Advances the simulation by one timestep (`dt * frame_skip` seconds).
    ///
    /// Steps:
    /// 1. Clips the action to `config.action_clip`, multiplies by `config.gear`,
    ///    and applies the result as a world-x force on the cart, re-applied
    ///    fresh before each physics substep (ADR 0037 force-lifetime contract;
    ///    `frame_skip = 1` here so this is one application per env step).
    /// 2. Advances the physics via `Rapier3DWorld::step_actuated`.
    /// 3. Extracts a new observation `[cart_x, pole_angle, cart_vx, pole_angvel_y]`.
    /// 4. Computes reward: `+1.0` if `|pole_angle| < 0.2`, else `0.0`.
    /// 5. Determines episode status:
    ///    - `Terminated` if unhealthy and `termination == OnUnhealthy`.
    ///    - `Truncated` if `steps >= max_steps`.
    ///    - `Running` otherwise.
    ///
    /// The snapshot metadata includes the `"alive"` component (0 or 1) and the
    /// cart's 3-D position keyed as `"cart"`.
    ///
    /// # Errors
    ///
    /// Returns [`EnvironmentError::StepAfterEpisodeEnd`] if the episode has
    /// already ended (the pole fell, or `max_steps` was reached); call
    /// [`reset`](Environment::reset) first. Returns
    /// [`EnvironmentError::InvalidAction`] if the action value is non-finite
    /// (NaN or ±infinity).
    fn step(
        &mut self,
        action: InvertedPendulumAction,
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
                "InvertedPendulum action must be finite, got {}",
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

        let healthy = self.config.healthy.is_healthy(
            /* torso_z (unused) */ 0.0,
            obs.pole_angle(),
            &obs.0,
        );
        let alive_bonus = if healthy { 1.0 } else { 0.0 };
        let reward = ScalarReward(alive_bonus);

        let status = if !healthy && matches!(self.config.termination, TerminationMode::OnUnhealthy)
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
        // Single exit: exactly one snapshot is built, and the guard is fed that
        // snapshot's own status, so no branch can forget to record.
        let snapshot = LocomotionSnapshot {
            observation: obs,
            reward,
            status,
            metadata: Some(meta),
        };
        self.guard.record(snapshot.status);
        Ok(snapshot)
    }
}

fn cart_half_z(config: &InvertedPendulumConfig) -> f32 {
    config.cart_half_extents[2]
}

// ---------------------------------------------------------------------------
// Report-tier payload — sagittal-plane projection.
//
// Joint 0 = cart centre; joint 1 = pole tip. The bone connects them.
// Both points project onto the (x, z) plane: x forward, z up.
// `ground_y` is z = 0 — the floor plane the cart rests on.
// ---------------------------------------------------------------------------

impl rlevo_core::render::Locomotion2DPayloadSource for InvertedPendulum<Rapier3DBackend> {
    /// Returns a sagittal-plane (x–z) projection of the current physics state.
    ///
    /// Joint 0 is the cart centre; joint 1 is the pole tip, approximated as
    /// `cart_centre + 2 * (pole_centre - cart_centre)`. The single bone
    /// connects them. `ground_y` is `0.0` (world-z floor plane). `com` is set
    /// to the pole's centre of mass. No contact points are reported.
    fn locomotion2d_snapshot(&self) -> rlevo_core::render::Locomotion2DSnapshot {
        use rlevo_core::render::{Locomotion2DSnapshot, Point2};

        let cart_pose = Rapier3DBackend::get_pose(&self.world, self.state.cart);
        let pole_pose = Rapier3DBackend::get_pose(&self.world, self.state.pole);

        // Pole tip lies at the far end of the pole body from the joint.
        // The pole's centre is at pole_pose.position; its half-length axis
        // points along the body's local +z, rotated into world frame.
        // For the sagittal projection we approximate the tip with
        // 2*(pole_centre - cart_centre) + cart_centre, which gives the
        // correct visual stick figure for the small-angle regime this env
        // operates in.
        let cx = cart_pose.position[0];
        let cz = cart_pose.position[2];
        let px = pole_pose.position[0];
        let pz = pole_pose.position[2];
        let tip_x = cx + 2.0 * (px - cx);
        let tip_z = cz + 2.0 * (pz - cz);

        Locomotion2DSnapshot {
            joints: vec![Point2::new(cx, cz), Point2::new(tip_x, tip_z)],
            bones: vec![(0, 1)],
            ground_y: 0.0,
            com: Some(Point2::new(px, pz)),
            contacts: vec![],
        }
    }
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
        let mut env = InvertedPendulumRapier::with_config(InvertedPendulumConfig {
            seed: 7,
            reset_noise_scale: 0.0,
            ..Default::default()
        })
        .expect("valid config");
        let snap = env.reset().unwrap();
        assert!(!snap.is_done());
        for v in snap.observation().0 {
            assert!(v.abs() < 1e-5, "zero reset noise should give ~zero obs");
        }
    }

    #[test]
    fn ctrl_cost_not_paid() {
        // InvertedPendulum's Gymnasium reward is +1 alive, not a quadratic cost.
        let mut env = InvertedPendulumRapier::with_config(InvertedPendulumConfig::default())
            .expect("valid config");
        env.reset().unwrap();
        let snap = env.step(InvertedPendulumAction::new(3.0)).unwrap();
        // Reward is +1 per step while healthy regardless of action magnitude.
        let total: f32 = snap.metadata().unwrap().components.values().sum();
        assert!((total - snap.reward().0).abs() < 1e-5);
    }

    #[test]
    fn reward_roundtrip_matches_components() {
        let mut env = InvertedPendulumRapier::with_config(InvertedPendulumConfig::default())
            .expect("valid config");
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
        })
        .expect("valid config");
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
        })
        .expect("valid config");
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
        let cfg = InvertedPendulumConfig {
            seed: 123,
            ..Default::default()
        };
        let rollout = |actions: &[f32]| {
            let mut env = InvertedPendulumRapier::with_config(cfg.clone()).expect("valid config");
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
        let mut env = InvertedPendulumRapier::with_config(InvertedPendulumConfig::default())
            .expect("valid config");
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
        let mut env = InvertedPendulumRapier::with_config(InvertedPendulumConfig::default())
            .expect("valid config");
        env.reset().unwrap();
        for _ in 0..50 {
            let snap = env.step(InvertedPendulumAction::new(0.1)).unwrap();
            assert!(snap.observation().is_finite());
            if snap.is_done() {
                break;
            }
        }
    }

    #[test]
    fn constant_force_does_not_accumulate() {
        // Regression test (ADR 0037): a constant action must produce a
        // stationary per-step cart-velocity increment. Rapier does not
        // auto-clear applied forces each step despite the vendored 0.32 doc
        // comment's claim to the contrary, so an unguarded `add_force` call
        // accumulated across steps and silently corrupted the control
        // dynamics; qualitative "did the joint move" tests stayed green
        // throughout. Fixed once in `RapierWorld::step()` (reset forces and
        // torques every step) with per-env force constants re-tuned
        // afterward. Without that fix, Δvx grows ~linearly instead of
        // holding steady.
        let mut env = InvertedPendulumRapier::with_config(InvertedPendulumConfig {
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
            let snap = env.step(InvertedPendulumAction::new(1.0)).unwrap();
            assert!(snap.observation().is_finite(), "obs must stay finite");
            let vx = snap.observation().cart_velocity();
            deltas.push(vx - prev_vx);
            prev_vx = vx;
        }

        // Early vs late per-step increment. Constant force ⇒ ratio ≈ 1 (plus mild
        // pole coupling); the accumulation bug drives it well above 5×.
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
        let mut env = InvertedPendulumRapier::with_config(InvertedPendulumConfig {
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
        let mut env = InvertedPendulumRapier::with_config(InvertedPendulumConfig {
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

    // ── post-terminal step guard (ADR 0044) ───────────────────────────────────
    //
    // `step` used to check only `action.is_valid()`; nothing rejected a call
    // made after the episode had already ended. A post-terminal step kept
    // integrating the Rapier simulation, so the observation drifted past the
    // point where the episode should have stopped. The fix — shared across
    // the whole locomotion family (`inverted_double_pendulum`, `reacher`,
    // `swimmer`) — holds an `EpisodeGuard` per environment: `step` calls
    // `guard.check()?` before touching the physics world (see the ordering
    // rationale above `step`, ADR 0029), and `guard.record(status)` on the
    // single snapshot-producing exit path; `reset` reopens the guard. The
    // tests below drive that behavior through
    // `crate::episode::assert_rejects_post_terminal_step` plus a direct
    // no-mutation check.

    /// Upper bound on the steps the kicked pendulum may take before the test
    /// calls it a regression. `max_steps` is set well above it so truncation
    /// cannot pre-empt the termination the test is driving to.
    const FALL_STEP_CAP: usize = 2000;

    /// Config that lets the episode end by a genuine `Terminated`: no reset
    /// noise (deterministic tilt), `OnUnhealthy` termination, and a truncation
    /// budget far beyond `FALL_STEP_CAP`.
    fn terminating_cfg() -> InvertedPendulumConfig {
        InvertedPendulumConfig {
            reset_noise_scale: 0.0,
            termination: TerminationMode::OnUnhealthy,
            max_steps: 10_000,
            ..Default::default()
        }
    }

    /// Drives a fresh episode to a real **`Terminated`** — not a truncation:
    /// a short +x kick on the cart topples the pole, gravity carries `$|\theta|$`
    /// past the healthy band `(-0.2, 0.2)`, and the `OnUnhealthy` rule ends
    /// the episode through the same physics path an agent would hit.
    fn drive_to_pole_fall(
        env: &mut InvertedPendulum<Rapier3DBackend>,
    ) -> LocomotionSnapshot<InvertedPendulumObservation> {
        env.reset().expect("reset must succeed");
        for i in 0..FALL_STEP_CAP {
            let force = if i < 20 { 3.0 } else { 0.0 };
            let snap = env
                .step(InvertedPendulumAction::new(force))
                .expect("step must succeed while the episode is running");
            if snap.is_done() {
                assert!(
                    snap.is_terminated(),
                    "this fixture must end by termination, not truncation"
                );
                return snap;
            }
        }
        panic!("a kicked pole must leave the healthy band within {FALL_STEP_CAP} steps");
    }

    #[test]
    /// `InvertedPendulum` satisfies the shared post-terminal conformance check:
    /// once the pole has fallen (`Terminated`), a further step with a *legal*
    /// action fails with `StepAfterEpisodeEnd` carrying that same status. The
    /// replayed action is `0.0` — squarely inside the valid range — so the
    /// rejection can only be on call-sequence grounds, never on the action's own
    /// validity.
    fn test_inverted_pendulum_rejects_post_terminal_step() {
        let mut env = InvertedPendulumRapier::with_config(terminating_cfg()).expect("valid config");
        assert_rejects_post_terminal_step(
            &mut env,
            drive_to_pole_fall,
            InvertedPendulumAction::new(0.0),
        );
    }

    #[test]
    /// A rejected post-terminal step must mutate nothing observable. The
    /// healthiness test is a live read of the current pole angle, not a latch,
    /// so before the guard a further step integrated another frame, ticked the
    /// step counter and could hand back a fresh `+1` alive bonus on a `Running`
    /// snapshot as the toppled pole swung back through the band.
    fn test_inverted_pendulum_post_terminal_step_does_not_mutate_state() {
        let mut env = InvertedPendulumRapier::with_config(terminating_cfg()).expect("valid config");
        let terminal = drive_to_pole_fall(&mut env);
        let ended = terminal.status();

        let obs_at_end = terminal.observation().0;
        let steps_at_end = env.steps;

        let err = env
            .step(InvertedPendulumAction::new(0.0))
            .expect_err("a step after the pole fell must return Err, not another snapshot");
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
    fn test_inverted_pendulum_post_terminal_step_outranks_the_action_check() {
        let mut env = InvertedPendulumRapier::with_config(terminating_cfg()).expect("valid config");
        drive_to_pole_fall(&mut env);

        let err = env
            .step(InvertedPendulumAction::new(f32::NAN))
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
    fn test_inverted_pendulum_post_terminal_step_does_not_advance_rng() {
        let with_replay = {
            let mut env =
                InvertedPendulumRapier::with_config(terminating_cfg()).expect("valid config");
            drive_to_pole_fall(&mut env);
            env.step(InvertedPendulumAction::new(0.0))
                .expect_err("the post-terminal step must be rejected");
            env.reset().unwrap().observation().0
        };
        let without_replay = {
            let mut env =
                InvertedPendulumRapier::with_config(terminating_cfg()).expect("valid config");
            drive_to_pole_fall(&mut env);
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
    fn test_inverted_pendulum_reset_reopens_terminated_episode() {
        let mut env = InvertedPendulumRapier::with_config(terminating_cfg()).expect("valid config");
        drive_to_pole_fall(&mut env);
        assert!(
            env.step(InvertedPendulumAction::new(0.0)).is_err(),
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
            .step(InvertedPendulumAction::new(0.0))
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
    fn test_inverted_pendulum_rejects_post_truncation_step_and_reset_with_seed_reopens() {
        let mut env = InvertedPendulumRapier::with_config(InvertedPendulumConfig {
            max_steps: 5,
            termination: TerminationMode::Never,
            reset_noise_scale: 0.0,
            ..Default::default()
        })
        .expect("valid config");
        env.reset().unwrap();
        let mut last = None;
        for _ in 0..5 {
            last = Some(env.step(InvertedPendulumAction::new(0.0)).unwrap());
        }
        assert_eq!(
            last.expect("five steps were taken").status(),
            EpisodeStatus::Truncated
        );

        let err = env
            .step(InvertedPendulumAction::new(0.0))
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
            env.step(InvertedPendulumAction::new(0.0)).is_ok(),
            "reset_with_seed must re-open the environment for a new episode"
        );
    }
}
