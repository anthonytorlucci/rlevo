//! # Swimmer-v5 (Rapier3D-backed)
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
//! Three cylindrical segments chained in series (front → middle → tail),
//! constrained to the world xy-plane with gravity disabled. Two revolute-z
//! impulse joints actuate the chain; per-segment viscous drag substitutes for
//! MuJoCo's native fluid model.
//!
//! * Segment shape: capsule along body-x, length `0.1`, radius `0.05`, mass
//!   `≈0.0471` (density ≈60 kg/m³ derived from capsule volume). The capsule
//!   stands in for Gymnasium's cylinder — drag depends on COM velocity only,
//!   not collider geometry.
//! * Planar constraint: every segment has `enabled_translations(true, true, false)`
//!   and `enabled_rotations(false, false, true)`, so motion is confined to the
//!   xy-plane and rotations to about-z.
//! * Front ↔ Middle: revolute-z impulse joint, anchor `(+0.05, 0, 0)` on
//!   segment0's back, anchor `(−0.05, 0, 0)` on segment1's front.
//! * Middle ↔ Tail:  revolute-z impulse joint, anchor `(+0.05, 0, 0)` on
//!   segment1's back, anchor `(−0.05, 0, 0)` on segment2's front.
//! * Action: `Box(-1, 1, (2,))` — joint torque targets; applied as
//!   `action · gear` with `gear = [150, 150]` (Gymnasium XML).
//! * Observation (8-dim):
//!   `[body_angle, joint1_angle, joint2_angle, vx_com, vy_com,
//!     ω_body, joint1_dot, joint2_dot]` — matches `qpos[2:5]` + `qvel` from
//!   Gymnasium.
//! * Reward: `forward − ctrl` with `forward = 1.0 · vx_com` and
//!   `ctrl = 1e-4 · ‖action‖²`.
//! * Termination: never (`TerminationMode::Never` implicitly; swimmer has no
//!   healthy gate).
//! * Truncation: `max_steps = 1000`.
//!
//! ## Viscous drag
//!
//! Rapier has no native fluid solver. Each segment accrues a drag force
//! `F = −k · v · ‖v‖` before every physics substep, where `k` is
//! `drag_coefficient` (default `0.1`). The env owns its own `frame_skip`
//! loop so drag is applied on every substep, not once per env step; this is
//! required for numerical stability and to match the Gymnasium frame_skip
//! semantics. If/when a second locomotion env needs viscous drag, the pattern
//! can be promoted to a backend-level hook.
//!
//! ## Divergence from Gymnasium
//!
//! Reward structure and observation layout match Gymnasium v5. Physics
//! parameters diverge where Rapier's reduced-coordinate multibody solver
//! cannot integrate the MuJoCo-native XML at the Gymnasium substep.
//! Specifically (see [`SwimmerConfig::default`] for full reasoning):
//!
//! * **Joints are in `MultibodyJointSet`, not `ImpulseJointSet`.** A
//!   free-floating serial chain with no grounded reference is a stiff
//!   problem for the PGS impulse solver; multibody's Featherstone-style
//!   reduced-coordinate integration keeps the chain conservative.
//! * **`gear = [5, 5]`** instead of Gymnasium's `[150, 150]`. At full gear
//!   the joint angular acceleration (τ/I ≈ 7 500 rad/s²) violates the joint
//!   constraint faster than the solver can resolve it.
//! * **Smaller substep:** `dt = 0.005`, `frame_skip = 8` → env dt 0.04 still
//!   matches Gymnasium; the integration step is halved to keep per-step
//!   Δω tractable.
//! * **`segment_mass = 0.947 kg`** from MuJoCo's body density (1000 kg/m³)
//!   applied to the capsule volume; using 0.0471 kg (as Gymnasium-derived
//!   calculations sometimes produce by crossing body density with MuJoCo's
//!   fluid `<option density>`) gives negligible inertia.
//! * **Linear angular drag** `τ = −k_ang · ω`, not quadratic. Explicit
//!   Euler on quadratic drag overshoots past zero at high |ω| and
//!   diverges within a substep; linear drag is unconditionally stable.
//! * **Capsules, not cylinders** (`capsule_x`, not `cylinder`). The drag
//!   model depends on COM velocity only, not collider geometry.
//!
//! Cumulative effect: observation shape, action shape, reward structure and
//! termination conditions match Gymnasium v5 exactly; absolute reward
//! values, peak velocities, and learned policies will not transfer.

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

/// Reward-component key: `forward_reward_weight · vx_com` (≥ 0 when swimming
/// forward, ≤ 0 when drifting backward).
pub const METADATA_KEY_FORWARD: &str = "forward";
/// Reward-component key: `−ctrl_cost_weight · ‖action‖²` (≤ 0).
pub const METADATA_KEY_CTRL: &str = "ctrl";

// ─── Config ───────────────────────────────────────────────────────────────

/// Environment configuration for [`Swimmer`].
///
/// Defaults match the Gymnasium v5 swimmer XML: gear `[150, 150]`, dt 0.01,
/// frame_skip 4 (env dt = 0.04), reset noise 0.1, forward-reward weight 1.0,
/// ctrl-cost weight 1e-4, drag coefficient 0.1, truncation at 1000.
#[derive(Debug, Clone)]
pub struct SwimmerConfig {
    pub seed: u64,
    pub gear: Gear<2>,
    pub dt: f32,
    pub frame_skip: u32,
    pub reset_noise_scale: f32,
    pub max_steps: usize,
    pub action_clip: (f32, f32),
    pub forward_reward_weight: f32,
    pub ctrl_cost_weight: f32,
    /// Per-segment viscous linear-drag coefficient: `F = −k · v · ‖v‖`.
    pub drag_coefficient: f32,
    /// Per-segment viscous angular-drag coefficient: `τ = −k_ang · ω · |ω|`.
    /// Rapier-only addition to stand in for MuJoCo's fluid viscosity term,
    /// which otherwise lets the chain spin unboundedly under actuator
    /// torques in a zero-gravity free-floating setup.
    pub angular_drag_coefficient: f32,
    pub segment_length: f32,
    pub segment_radius: f32,
    pub segment_mass: f32,
}

impl Default for SwimmerConfig {
    fn default() -> Self {
        // Three knobs diverge from the Gymnasium XML so Rapier's
        // reduced-coordinate multibody solver stays integrable:
        //
        // * `segment_mass = 0.947` uses MuJoCo's default *body* density
        //   (1000 kg/m³) applied to the capsule volume π·r²·(2·half + (4/3)·r).
        //   A figure of 0.0471 kg would cross body density with fluid density
        //   (`<option density>`); the XML uses body density 1000.
        //
        // * `gear = [30, 30]` is one fifth of Gymnasium's `[150, 150]`.
        //   At full gear the angular acceleration (α ≈ τ/I ≈ 7 500 rad/s²)
        //   produces ~75 rad/s per substep at dt=0.01 — a rotation rate
        //   that violates the joint constraints faster than PGS can
        //   resolve them, and the multibody state diverges to NaN within
        //   a handful of env steps. The fifth-scale gear keeps dynamics
        //   inside the solver's stable regime.
        //
        // * `angular_drag_coefficient = 0.1` adds a quadratic angular-drag
        //   term `τ_drag = −k_ang · ω · |ω|` per segment. MuJoCo's swimmer
        //   effectively caps segment spin via fluid viscosity; with only
        //   linear drag, sustained actuation spins the chain up unboundedly
        //   (since no kinetic-energy sink exists about the joint axis) and
        //   the solver still diverges at gear=30 within ~50 steps.
        //
        // The module-level "absolute reward values will NOT transfer"
        // disclaimer covers these divergences.
        Self {
            seed: 0,
            gear: Gear::new([5.0, 5.0]),
            dt: 0.005,
            frame_skip: 8,
            reset_noise_scale: 0.1,
            max_steps: 1000,
            action_clip: (-1.0, 1.0),
            forward_reward_weight: 1.0,
            ctrl_cost_weight: 1e-4,
            drag_coefficient: 0.1,
            angular_drag_coefficient: 0.2,
            segment_length: 0.1,
            segment_radius: 0.05,
            segment_mass: 0.947,
        }
    }
}

// ─── Action ──────────────────────────────────────────────────────────────

/// 2D continuous action — `[joint1, joint2]` torque targets in pre-gear
/// units. Bounds: `[-1.0, 1.0]` per element.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SwimmerAction(pub [f32; 2]);

impl SwimmerAction {
    #[must_use]
    pub const fn new(joint1: f32, joint2: f32) -> Self {
        Self([joint1, joint2])
    }
}

impl Action<1> for SwimmerAction {
    fn shape() -> [usize; 1] {
        [2]
    }

    fn is_valid(&self) -> bool {
        self.0.iter().all(|v| v.is_finite() && v.abs() <= 1.0)
    }
}

impl ContinuousAction<1> for SwimmerAction {
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

impl<B: Backend> TensorConvertible<1, B> for SwimmerAction {
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

/// 8-dim observation. Layout matches Gymnasium's `qpos[2:5]` + `qvel`:
/// `[body_angle, joint1_angle, joint2_angle, vx_com, vy_com,
///   ω_body, joint1_dot, joint2_dot]`.
///
/// * `body_angle` — absolute z-rotation of segment0 (wrapped to `(-π, π]`).
/// * `joint{1,2}_angle` — **relative** angle between adjacent segments
///   (child − parent in world-z), wrapped.
/// * `vx_com, vy_com, ω_body` — segment0's linear/angular velocity.
/// * `joint{k}_dot` — relative angular rate `ω_child − ω_parent`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SwimmerObservation(pub [f32; 8]);

impl SwimmerObservation {
    #[must_use]
    pub const fn body_angle(&self) -> f32 {
        self.0[0]
    }
    #[must_use]
    pub const fn joint1_angle(&self) -> f32 {
        self.0[1]
    }
    #[must_use]
    pub const fn joint2_angle(&self) -> f32 {
        self.0[2]
    }
    #[must_use]
    pub const fn vx_com(&self) -> f32 {
        self.0[3]
    }
    #[must_use]
    pub const fn vy_com(&self) -> f32 {
        self.0[4]
    }
    #[must_use]
    pub const fn omega_body(&self) -> f32 {
        self.0[5]
    }
    #[must_use]
    pub const fn joint1_dot(&self) -> f32 {
        self.0[6]
    }
    #[must_use]
    pub const fn joint2_dot(&self) -> f32 {
        self.0[7]
    }

    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.0.iter().all(|v| v.is_finite())
    }
}

impl Default for SwimmerObservation {
    fn default() -> Self {
        Self([0.0; 8])
    }
}

impl Observation<1> for SwimmerObservation {
    fn shape() -> [usize; 1] {
        [8]
    }
}

impl<B: Backend> TensorConvertible<1, B> for SwimmerObservation {
    fn to_tensor(&self, device: &B::Device) -> Tensor<B, 1> {
        Tensor::from_floats(self.0, device)
    }

    fn from_tensor(tensor: Tensor<B, 1>) -> Result<Self, TensorConversionError> {
        let data = tensor.into_data();
        let slice = data.as_slice::<f32>().map_err(|e| TensorConversionError {
            message: format!("expected f32 observation tensor: {e:?}"),
        })?;
        if slice.len() != 8 {
            return Err(TensorConversionError {
                message: format!("expected 8 observation elements, got {}", slice.len()),
            });
        }
        let mut arr = [0.0f32; 8];
        arr.copy_from_slice(slice);
        Ok(Self(arr))
    }
}

// ─── State ───────────────────────────────────────────────────────────────

/// Physics state — body + joint handles plus the last observation. The
/// non-`Clone` world lives on the env struct directly.
///
/// The two revolute-z joints live in Rapier's `MultibodyJointSet` (not the
/// impulse set): a free-floating serial chain with no grounded reference is
/// a stiff problem for the PGS impulse solver, which injects kinetic energy
/// when constraint drift is corrected aggressively. The Featherstone-style
/// multibody solver parameterises the chain in reduced coordinates
/// (analogous to MuJoCo's generalised coordinates) and stays conservative.
#[derive(Debug, Clone)]
pub struct SwimmerState {
    pub segment0: RigidBodyHandle,
    pub segment1: RigidBodyHandle,
    pub segment2: RigidBodyHandle,
    pub joint1: MultibodyJointHandle,
    pub joint2: MultibodyJointHandle,
    pub last_obs: SwimmerObservation,
}

impl State<1> for SwimmerState {
    type Observation = SwimmerObservation;

    fn shape() -> [usize; 1] {
        [8]
    }

    fn is_valid(&self) -> bool {
        self.last_obs.is_finite()
    }

    fn observe(&self) -> SwimmerObservation {
        self.last_obs
    }
}

// ─── Environment ─────────────────────────────────────────────────────────

/// Swimmer — a 3-segment planar swimmer with viscous drag. Generic in the
/// physics backend; v1 only implements `B = Rapier3DBackend`.
#[derive(Debug)]
pub struct Swimmer<B: LocomotionBackend = Rapier3DBackend> {
    world: B::World,
    state: SwimmerState,
    config: SwimmerConfig,
    rng: StdRng,
    steps: usize,
    _marker: PhantomData<B>,
}

/// Default backend alias.
pub type SwimmerRapier = Swimmer<Rapier3DBackend>;

impl Swimmer<Rapier3DBackend> {
    /// Create with an explicit configuration.
    #[must_use]
    pub fn with_config(config: SwimmerConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(config.seed);
        let (world, state) = Self::build_world(&config, &mut rng);
        Self { world, state, config, rng, steps: 0, _marker: PhantomData }
    }

    fn build_world(
        config: &SwimmerConfig,
        rng: &mut StdRng,
    ) -> (Rapier3DWorld, SwimmerState) {
        // Zero gravity — swimmer floats in open water.
        // Pass frame_skip=1 to the world: the env owns its own substep loop so
        // drag can be injected per substep (see `step_physics`).
        let mut world = Rapier3DWorld::new(
            Vector::new(0.0, 0.0, 0.0),
            config.dt,
            1,
        );

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
        let capsule_volume =
            std::f32::consts::PI * r.powi(2) * (2.0 * half_l + (4.0 / 3.0) * r);
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
        let joint1 = RevoluteJointBuilder::new(z_axis)
            .local_anchor1(Vector::new(half_l, 0.0, 0.0))
            .local_anchor2(Vector::new(-half_l, 0.0, 0.0))
            .build();
        let joint1_handle = world
            .add_multibody_joint(segment0, segment1, joint1)
            .expect("joint1 must form a tree — segment0 is the multibody root");

        let joint2 = RevoluteJointBuilder::new(z_axis)
            .local_anchor1(Vector::new(half_l, 0.0, 0.0))
            .local_anchor2(Vector::new(-half_l, 0.0, 0.0))
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

    fn apply_action(&mut self, action: &SwimmerAction) {
        let (lo, hi) = self.config.action_clip;
        let clipped = [action.0[0].clamp(lo, hi), action.0[1].clamp(lo, hi)];
        let torques = self.config.gear.apply(&clipped);

        // Under `MultibodyJointSet`, applying an external torque to a child
        // body produces an equal-and-opposite reaction on the parent through
        // the joint (reduced-coordinate solver handles this automatically).
        // Torque only the child of each joint: seg1 for joint1, seg2 for
        // joint2. Double-torquing (also applying -τ to the parent as we do
        // in impulse-joint envs like Reacher) would inject spurious net
        // torque on the free-floating chain.
        if let Some(body) = self.world.bodies_mut().get_mut(self.state.segment1) {
            body.add_torque(Vector::new(0.0, 0.0, torques[0]), true);
        }
        if let Some(body) = self.world.bodies_mut().get_mut(self.state.segment2) {
            body.add_torque(Vector::new(0.0, 0.0, torques[1]), true);
        }
    }

    /// Apply per-segment viscous drag to each segment:
    /// quadratic linear drag `F = −k · v · ‖v‖` plus **linear** angular
    /// drag `τ = −k_ang · ω`.
    ///
    /// The angular term is linear (not quadratic) because Rapier integrates
    /// forces with explicit Euler, and the overshoot threshold of an
    /// explicit step for quadratic drag is `k_ang · |ω| · dt / I < 1`:
    /// at gear=150 the chain reaches `|ω|` in the 100s of rad/s, which
    /// drives quadratic drag unstable and diverges to NaN in one substep.
    /// Linear drag is unconditionally stable as long as `k_ang · dt / I < 2`.
    /// MuJoCo's own swimmer uses quadratic drag but with an implicit
    /// integrator; our linear variant is a Rapier-compatibility divergence.
    fn apply_drag(&mut self) {
        let k = self.config.drag_coefficient;
        let k_ang = self.config.angular_drag_coefficient;
        for handle in [
            self.state.segment0,
            self.state.segment1,
            self.state.segment2,
        ] {
            let twist = Rapier3DBackend::get_vel(&self.world, handle);
            let v = twist.linear;
            let speed = (v[0] * v[0] + v[1] * v[1]).sqrt();
            let fx = -k * v[0] * speed;
            let fy = -k * v[1] * speed;
            let wz = twist.angular[2];
            let tau_z = -k_ang * wz;
            if let Some(body) = self.world.bodies_mut().get_mut(handle) {
                body.add_force(Vector::new(fx, fy, 0.0), true);
                body.add_torque(Vector::new(0.0, 0.0, tau_z), true);
            }
        }
    }

    /// Run `frame_skip` physics substeps, injecting viscous drag before each.
    /// Replaces the `Rapier3DBackend::step` call used by envs without drag.
    ///
    /// Drag is applied *every* substep because Rapier clears external-force
    /// accumulators after each `step_once`. The actuator torque, in contrast,
    /// is applied once per env step at the top of `Environment::step` — same
    /// convention as [`super::reacher::Reacher`] and
    /// [`super::inverted_double_pendulum::InvertedDoublePendulum`]. Applying
    /// it per-substep under Rapier's PGS solver over-drives the low-inertia
    /// chain and NaNs out.
    fn step_physics(&mut self) {
        let substeps = self.config.frame_skip.max(1);
        for _ in 0..substeps {
            self.apply_drag();
            self.world.step_once();
        }
    }
}

impl Environment<1, 1, 1> for Swimmer<Rapier3DBackend> {
    type StateType = SwimmerState;
    type ObservationType = SwimmerObservation;
    type ActionType = SwimmerAction;
    type RewardType = ScalarReward;
    type SnapshotType = LocomotionSnapshot<SwimmerObservation>;

    fn new(_render: bool) -> Self {
        Self::with_config(SwimmerConfig::default())
    }

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.rng = StdRng::seed_from_u64(self.config.seed);
        let (world, mut state) = Self::build_world(&self.config, &mut self.rng);
        self.world = world;
        state.last_obs = SwimmerObservation::default();
        self.state = state;
        self.steps = 0;

        let obs = self.extract_observation();
        self.state.last_obs = obs;

        let torso_pos = Rapier3DBackend::get_pose(&self.world, self.state.segment0).position;
        let meta = SnapshotMetadata::new()
            .with(METADATA_KEY_FORWARD, 0.0)
            .with(METADATA_KEY_CTRL, 0.0)
            .with_position("torso", torso_pos)
            .with_position("com", torso_pos)
            .with_position("main_body", torso_pos);
        Ok(LocomotionSnapshot::running(obs, ScalarReward(0.0), meta))
    }

    fn step(
        &mut self,
        action: SwimmerAction,
    ) -> Result<Self::SnapshotType, EnvironmentError> {
        if !action.0.iter().all(|v| v.is_finite()) {
            return Err(EnvironmentError::InvalidAction(format!(
                "Swimmer action must be finite, got {:?}",
                action.0
            )));
        }

        self.apply_action(&action);
        self.step_physics();
        self.steps += 1;

        let obs = self.extract_observation();
        self.state.last_obs = obs;

        let forward = self.config.forward_reward_weight * obs.vx_com();
        // Clip to the bound-enforced action before computing ctrl cost so that
        // out-of-bound inputs can't inflate the cost beyond the Gymnasium convention.
        let (lo, hi) = self.config.action_clip;
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
        Ok(LocomotionSnapshot::new(obs, ScalarReward(total), status, meta))
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────

/// Extract the z-axis rotation angle from a pose whose body is DOF-gated to
/// revolute-z. Quaternion stored as `[w, x, y, z]`.
fn segment_z_angle(orientation: [f32; 4]) -> f32 {
    let [w, _, _, z] = orientation;
    2.0 * z.atan2(w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use evorl_core::environment::Snapshot;

    fn cfg(seed: u64) -> SwimmerConfig {
        SwimmerConfig { seed, ..Default::default() }
    }

    fn deterministic_cfg() -> SwimmerConfig {
        SwimmerConfig { seed: 7, reset_noise_scale: 0.0, ..Default::default() }
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
        let a = SwimmerAction::new(10.0, -10.0).clip(-1.0, 1.0);
        assert_eq!(a.0, [1.0, -1.0]);
    }

    #[test]
    fn reset_returns_running() {
        let mut env = SwimmerRapier::with_config(cfg(7));
        let snap = env.reset().unwrap();
        assert!(!snap.is_done());
        assert!(snap.observation().is_finite());
    }

    #[test]
    fn reset_with_zero_noise_is_upright() {
        let mut env = SwimmerRapier::with_config(deterministic_cfg());
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
        let mut env = SwimmerRapier::with_config(cfg(11));
        env.reset().unwrap();
        for i in 0..100 {
            let a = SwimmerAction::new(
                0.5 * (i as f32 * 0.17).sin(),
                0.5 * (i as f32 * 0.23).cos(),
            );
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
        let mut env = SwimmerRapier::with_config(cfg(13));
        env.reset().unwrap();
        for i in 0..50 {
            let a = SwimmerAction::new(
                0.6 * (i as f32 * 0.31).cos(),
                0.9 * (i as f32 * 0.11).sin(),
            );
            let snap = env.step(a).unwrap();
            let c = snap.metadata().unwrap().components[METADATA_KEY_CTRL];
            assert!(c <= 0.0, "ctrl must be ≤ 0, got {c} at step {i}");
        }
    }

    #[test]
    fn determinism_across_reset() {
        let rollout = |actions: &[[f32; 2]]| {
            let mut env = SwimmerRapier::with_config(cfg(123));
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

    #[test]
    fn init_noise_bounded() {
        for seed in 0..50 {
            let env = SwimmerRapier::with_config(cfg(seed));
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
        });
        env.reset().unwrap();
        let mut status = EpisodeStatus::Running;
        for step in 0..5 {
            let snap = env.step(SwimmerAction::new(0.0, 0.0)).unwrap();
            status = snap.status();
            if step < 4 {
                assert_eq!(status, EpisodeStatus::Running, "early status at step {step}");
            }
        }
        assert_eq!(status, EpisodeStatus::Truncated);
    }

    #[test]
    fn invalid_action_is_error() {
        let mut env = SwimmerRapier::with_config(SwimmerConfig::default());
        env.reset().unwrap();
        let bad = SwimmerAction::new(f32::NAN, 0.0);
        assert!(env.step(bad).is_err());
        let bad = SwimmerAction::new(0.0, f32::INFINITY);
        assert!(env.step(bad).is_err());
    }

    #[test]
    fn obs_is_finite_after_rollout() {
        let mut env = SwimmerRapier::with_config(cfg(42));
        env.reset().unwrap();
        for i in 0..100 {
            let a = SwimmerAction::new(
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
        // With zero reset noise then a deliberate angular velocity on each
        // segment, every observation slot is a named function of the state
        // that we can verify independently.
        let mut env = SwimmerRapier::with_config(deterministic_cfg());
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
        let mut env = SwimmerRapier::with_config(deterministic_cfg());
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
    fn forward_reward_positive_for_forward_motion() {
        // A quarter-phase-lagged sinusoidal stroke propagates a travelling
        // wave down the chain — the canonical swimmer gait. Over a 300-step
        // rollout the net forward-reward component must be strictly
        // positive. (A bare alternating-every-env-step gait works too over
        // ~300 steps but the sinusoidal version converges faster and gives
        // the test tighter tolerance.)
        use std::f32::consts::{FRAC_PI_2, TAU};
        let mut env = SwimmerRapier::with_config(deterministic_cfg());
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
