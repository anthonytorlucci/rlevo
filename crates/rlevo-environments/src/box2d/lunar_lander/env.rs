//! LunarLander environment implementation — discrete and continuous variants.
//!
//! This module contains the two concrete environment types ([`LunarLanderDiscrete`]
//! and [`LunarLanderContinuous`]) and the shared `LunarLanderCore` physics driver
//! that both variants delegate to. Physics are simulated with Rapier2D at a fixed
//! timestep (`config.dt`, default 1/50 s).
//!
//! ## Reward formula
//!
//! Each step applies potential-based shaping plus a control cost:
//!
//! ```text
//! shaping(obs) = -100 * dist_to_helipad
//!              - 100 * speed
//!              - 100 * |angle|
//!              +  10 * leg1_contact
//!              +  10 * leg2_contact
//!
//! reward = shaping(t) - shaping(t-1)   -- potential difference
//!        - 0.3 * (|main| + |lateral|)  -- control cost
//! ```
//!
//! On a terminal step the reward is **set to** +100 (soft landing) or −100
//! (crash / out-of-bounds), replacing that step's shaping delta and control
//! cost — matching Gymnasium LunarLander. See [`LunarLanderSnapshot`] for how
//! the raw shaping value is surfaced through step metadata.

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rapier2d::dynamics::RevoluteJoint;
use rapier2d::geometry::ColliderHandle;
use rapier2d::prelude::*;
use rlevo_core::base::{Action, State};
use rlevo_core::config::{ConfigError, Validate};
use rlevo_core::environment::{ConstructableEnv, Environment, EnvironmentError, EpisodeStatus};
use rlevo_core::reward::ScalarReward;

use crate::box2d::physics::RapierWorld;

use super::action_continuous::LunarLanderContinuousAction;
use super::action_discrete::LunarLanderDiscreteAction;
use super::config::{LunarLanderConfig, WindMode};
use super::observation::LunarLanderObservation;
use super::snapshot::LunarLanderSnapshot;
use super::state::LunarLanderState;

// ─── Physical constants ───────────────────────────────────────────────────────

/// Scale: world units per screen pixel.
const SCALE: f32 = 30.0;
const VIEWPORT_W: f32 = 600.0;
const VIEWPORT_H: f32 = 400.0;
/// Width of the lander hull.
const LANDER_W: f32 = 14.0 / SCALE;
/// Height of the lander hull.
const LANDER_H: f32 = 17.0 / SCALE;
/// Initial spawn y-position (world units).
const INITIAL_Y: f32 = VIEWPORT_H / SCALE / 2.0 + 1.0;
/// Helipad centre x.
const HELIPAD_X: f32 = VIEWPORT_W / SCALE / 2.0;
/// Helipad y-level (world units).
const HELIPAD_Y: f32 = 1.5;

/// Shared physics state factored out of the two lander variants.
#[derive(Debug)]
struct LunarLanderCore {
    world: RapierWorld,
    state: LunarLanderState,
    config: LunarLanderConfig,
    rng: StdRng,
    wind_rng: Option<StdRng>,
    steps: usize,
}

impl LunarLanderCore {
    fn new(config: LunarLanderConfig) -> Result<Self, ConfigError> {
        config.validate()?;
        let rng = StdRng::seed_from_u64(config.seed);
        let wind_rng = if let WindMode::Stochastic { seed, .. } = config.wind_mode {
            Some(StdRng::seed_from_u64(seed))
        } else {
            None
        };
        let mut core = Self {
            world: RapierWorld::new(Vector::new(0.0, config.gravity), config.dt),
            state: LunarLanderState {
                lander_handle: RigidBodyHandle::invalid(),
                leg1_handle: RigidBodyHandle::invalid(),
                leg2_handle: RigidBodyHandle::invalid(),
                ground_handle: ColliderHandle::invalid(),
                leg1_contact: false,
                leg2_contact: false,
                last_obs: LunarLanderObservation::default(),
                prev_shaping: 0.0,
            },
            config,
            rng,
            wind_rng,
            steps: 0,
        };
        core.rebuild();
        Ok(core)
    }

    fn rebuild(&mut self) {
        self.world = RapierWorld::new(Vector::new(0.0, self.config.gravity), self.config.dt);

        // Ground
        let ground_rb = self.world.add_body(RigidBodyBuilder::fixed());
        self.state.ground_handle = self.world.add_collider(
            ColliderBuilder::cuboid(VIEWPORT_W / SCALE, 0.5)
                .translation(Vector::new(VIEWPORT_W / SCALE / 2.0, 0.0))
                .friction(0.1),
            ground_rb,
        );

        // Lander with random initial velocity
        let vx = self
            .rng
            .random_range(-self.config.initial_random..=self.config.initial_random);
        let vy = self
            .rng
            .random_range(-self.config.initial_random..=self.config.initial_random);

        let lander_rb = self.world.add_body(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(HELIPAD_X, INITIAL_Y))
                .linvel(Vector::new(vx, vy))
                .linear_damping(0.5)
                .angular_damping(1.0),
        );
        self.world.add_collider(
            ColliderBuilder::cuboid(LANDER_W / 2.0, LANDER_H / 2.0)
                .density(self.config.lander_density)
                .friction(0.1),
            lander_rb,
        );
        self.state.lander_handle = lander_rb;

        // Landing legs (hinged below the hull)
        for sign in [-1.0_f32, 1.0_f32] {
            let leg_x = HELIPAD_X + sign * LANDER_W / 2.0;
            let leg_y = INITIAL_Y - LANDER_H / 2.0 - 0.2;
            let leg_rb = self
                .world
                .add_body(RigidBodyBuilder::dynamic().translation(Vector::new(leg_x, leg_y)));
            self.world.add_collider(
                ColliderBuilder::cuboid(0.05, 0.3)
                    .density(1.0)
                    .friction(0.5),
                leg_rb,
            );
            // Joint between hull and leg
            let mut joint = RevoluteJoint::new();
            joint.set_local_anchor1(Vector::new(sign * LANDER_W / 2.0, -LANDER_H / 2.0));
            joint.set_local_anchor2(Vector::new(0.0, 0.3));
            joint.set_contacts_enabled(false);
            self.world.add_joint(joint, lander_rb, leg_rb, true);
            if sign < 0.0 {
                self.state.leg1_handle = leg_rb;
            } else {
                self.state.leg2_handle = leg_rb;
            }
        }

        self.steps = 0;
        self.state.leg1_contact = false;
        self.state.leg2_contact = false;
        self.state.prev_shaping = 0.0;
        let obs = self.compute_obs();
        self.state.prev_shaping = self.shaping(&obs);
        self.state.last_obs = obs;

        // Handles are now assigned and `last_obs`/`prev_shaping` written; the
        // state is fully assembled, so the invariant must hold.
        debug_assert!(
            self.state.is_valid(),
            "LunarLanderState invariant violated after reset"
        );
    }

    fn apply_wind(&mut self) {
        let lander_rb = self.state.lander_handle;
        match &self.config.wind_mode {
            WindMode::Off => {}
            WindMode::Constant { force } => {
                let f = *force;
                if let Some(body) = self.world.bodies_mut().get_mut(lander_rb) {
                    body.add_force(Vector::new(f, 0.0), true);
                }
            }
            WindMode::Stochastic { max_force, .. } => {
                let max = *max_force;
                if let Some(rng) = &mut self.wind_rng {
                    let fx = rng.random_range(-max..=max);
                    let fy = rng.random_range(-max * 0.5..=max * 0.5);
                    if let Some(body) = self.world.bodies_mut().get_mut(lander_rb) {
                        body.add_force(Vector::new(fx, fy), true);
                    }
                }
            }
        }
    }

    fn apply_engines(&mut self, main: f32, lateral: f32) {
        let lander = self.state.lander_handle;
        if let Some(body) = self.world.bodies_mut().get_mut(lander) {
            let angle = body.rotation().angle();
            // Main engine (fires downward in body-local frame)
            if main > 0.0 {
                let thrust = main * self.config.main_engine_power;
                let fx = -angle.sin() * thrust;
                let fy = angle.cos() * thrust;
                body.add_force(Vector::new(fx, fy), true);
            }
            // Side engines
            if lateral.abs() > 0.01 {
                let thrust = lateral * self.config.side_engine_power;
                let fx = angle.cos() * thrust;
                let fy = angle.sin() * thrust;
                body.add_force(Vector::new(fx, fy), true);
            }
        }
    }

    fn update_contacts(&mut self) {
        let leg1_col = self
            .world
            .bodies()
            .get(self.state.leg1_handle)
            .and_then(|b| b.colliders().iter().next().copied());
        let leg2_col = self
            .world
            .bodies()
            .get(self.state.leg2_handle)
            .and_then(|b| b.colliders().iter().next().copied());
        self.state.leg1_contact = leg1_col.is_some_and(|c| self.world.is_in_contact(c));
        self.state.leg2_contact = leg2_col.is_some_and(|c| self.world.is_in_contact(c));
    }

    /// Returns `true` if the hull is touching the ground.
    ///
    /// Mirrors [`Self::update_contacts`]: the hull body has a single collider,
    /// fetched via `b.colliders().iter().next().copied()`. Hull↔leg contacts
    /// are disabled at joint setup (`set_contacts_enabled(false)`), so the only
    /// thing the hull can contact is the ground. A missing body degrades to
    /// `false` (never panics) per the never-panic-on-runtime-data rule.
    ///
    /// This is the physics basis for Gym's `game_over` flag: in Gymnasium's
    /// `LunarLander`, a crash is set the instant the hull body begins touching
    /// the ground, unconditionally (leg contact does not suppress it).
    fn hull_in_contact(&self) -> bool {
        let hull_col = self
            .world
            .bodies()
            .get(self.state.lander_handle)
            .and_then(|b| b.colliders().iter().next().copied());
        hull_col.is_some_and(|c| self.world.is_in_contact(c))
    }

    fn compute_obs(&self) -> LunarLanderObservation {
        let bodies = self.world.bodies();
        let (x, y, vx, vy, angle, angvel) =
            bodies
                .get(self.state.lander_handle)
                .map_or((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), |b| {
                    let t = b.translation();
                    let v = b.linvel();
                    (
                        (t.x - HELIPAD_X) / (VIEWPORT_W / SCALE / 2.0),
                        (t.y - HELIPAD_Y) / (VIEWPORT_H / SCALE / 2.0),
                        v.x * (VIEWPORT_W / SCALE / 2.0) / 20.0,
                        v.y * (VIEWPORT_H / SCALE / 2.0) / 20.0,
                        b.rotation().angle(),
                        b.angvel() * 20.0 / SCALE,
                    )
                });
        LunarLanderObservation::new([
            x,
            y,
            vx,
            vy,
            angle,
            angvel,
            f32::from(self.state.leg1_contact),
            f32::from(self.state.leg2_contact),
        ])
    }

    fn shaping(&self, obs: &LunarLanderObservation) -> f32 {
        -100.0 * (obs.x() * obs.x() + obs.y() * obs.y()).sqrt()
            - 100.0 * (obs.vx() * obs.vx() + obs.vy() * obs.vy()).sqrt()
            - 100.0 * obs.angle().abs()
            + 10.0 * obs.leg1_contact()
            + 10.0 * obs.leg2_contact()
    }

    fn step_common(
        &mut self,
        main: f32,
        lateral: f32,
    ) -> (LunarLanderObservation, f32, EpisodeStatus) {
        self.apply_wind();
        self.apply_engines(main, lateral);
        self.world.step();
        self.steps += 1;
        self.update_contacts();

        let obs = self.compute_obs();
        let shaping = self.shaping(&obs);
        let reward_shaping = shaping - self.state.prev_shaping;
        self.state.prev_shaping = shaping;

        let ctrl_cost = 0.3 * (main.abs() + lateral.abs());
        let mut reward = reward_shaping - ctrl_cost;

        let lander = self.world.bodies().get(self.state.lander_handle);
        let pos = lander.map(|b| b.translation()).unwrap_or_default();
        // A crash is a hull–ground contact, mirroring Gymnasium's `game_over`
        // flag (set unconditionally the instant the hull touches the ground;
        // leg contact does NOT suppress it). A clean landing keeps the hull off
        // the ground — the legs extend below the hull and take the load — so
        // crash and landing stay mutually exclusive. This replaces the old
        // positional proxy (`pos.y < 0.1`), which was physically unreachable:
        // a hull resting on the ground settles at `pos.y ≈ 0.78`.
        let is_crashed = self.hull_in_contact();
        let is_out_of_bounds =
            pos.x < 0.0 || pos.x > VIEWPORT_W / SCALE || pos.y > VIEWPORT_H / SCALE;
        let is_landed = self.state.leg1_contact
            && self.state.leg2_contact
            && obs.vx().abs() < 0.1
            && obs.vy().abs() < 0.1
            && obs.angle().abs() < 0.1;

        let status = if is_crashed || is_out_of_bounds {
            // Gym overwrite: a terminal step's reward is set to exactly −100,
            // discarding this step's shaping delta and control cost.
            reward = -100.0;
            EpisodeStatus::Terminated
        } else if is_landed {
            // Gym overwrite: a successful landing's reward is exactly +100.
            reward = 100.0;
            EpisodeStatus::Terminated
        } else if self.steps >= self.config.max_steps {
            EpisodeStatus::Truncated
        } else {
            EpisodeStatus::Running
        };

        self.state.last_obs = obs.clone();

        // `last_obs`/`prev_shaping` are written and handles remain valid; the
        // state is fully assembled, so the invariant must hold.
        debug_assert!(
            self.state.is_valid(),
            "LunarLanderState invariant violated after step"
        );
        (obs, reward, status)
    }

    fn shaping_value(&self) -> f32 {
        self.state.prev_shaping
    }
}

// ─── LunarLanderDiscrete ──────────────────────────────────────────────────────

/// LunarLander with a 4-way discrete action space.
///
/// The step-limit is enforced internally (`config.max_steps`, default 1000).
/// Wrapping with `TimeLimit` is not supported because this environment uses a
/// custom snapshot type ([`LunarLanderSnapshot`]) rather than `SnapshotBase`.
///
/// Actions never return an error; all four [`LunarLanderDiscreteAction`] variants
/// are always valid.
#[derive(Debug)]
pub struct LunarLanderDiscrete {
    core: LunarLanderCore,
}

impl LunarLanderDiscrete {
    /// Construct with the given configuration.
    ///
    /// The Rapier2D world is built immediately; the lander is placed at the spawn
    /// position and the initial shaping value is computed so that the first call
    /// to `reset` returns a valid snapshot without a prior physics step.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `config` fails [`Validate`] (e.g.
    /// non-positive `main_engine_power`, `dt`, or `max_steps == 0`).
    pub fn with_config(config: LunarLanderConfig) -> Result<Self, ConfigError> {
        Ok(Self {
            core: LunarLanderCore::new(config)?,
        })
    }
}

impl ConstructableEnv for LunarLanderDiscrete {
    fn new(_render: bool) -> Self {
        Self::with_config(LunarLanderConfig::default()).expect("default config must validate")
    }
}

impl Environment<1, 1, 1> for LunarLanderDiscrete {
    type StateType = LunarLanderState;
    type ObservationType = LunarLanderObservation;
    type ActionType = LunarLanderDiscreteAction;
    type RewardType = ScalarReward;
    type SnapshotType = LunarLanderSnapshot;

    /// Rebuild the physics world and return the initial observation.
    ///
    /// The reward in the returned snapshot is always `0.0`. The shaping potential
    /// is reset to the value computed from the spawn position so that the first
    /// `step` call produces a meaningful potential difference.
    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.core.rebuild();
        let obs = self.core.state.last_obs.clone();
        Ok(LunarLanderSnapshot::running(
            obs,
            ScalarReward(0.0),
            self.core.shaping_value(),
        ))
    }

    /// Advance the simulation by one timestep and return the resulting snapshot.
    ///
    /// The returned snapshot status is:
    /// - `Running` — episode continues.
    /// - `Terminated` — lander crashed (hull contacts the ground) or flew out
    ///   of bounds (reward −100), or landed softly (reward +100).
    /// - `Truncated` — `config.max_steps` reached without a terminal event.
    ///
    /// This variant never returns `Err`; the result is always `Ok`.
    fn step(
        &mut self,
        action: LunarLanderDiscreteAction,
    ) -> Result<LunarLanderSnapshot, EnvironmentError> {
        let (main, lateral) = match action {
            LunarLanderDiscreteAction::DoNothing => (0.0, 0.0),
            LunarLanderDiscreteAction::LeftEngine => (0.0, -1.0),
            LunarLanderDiscreteAction::MainEngine => (1.0, 0.0),
            LunarLanderDiscreteAction::RightEngine => (0.0, 1.0),
        };
        let (obs, reward, status) = self.core.step_common(main, lateral);
        let shaping = self.core.shaping_value();
        let snap = match status {
            EpisodeStatus::Running => {
                LunarLanderSnapshot::running(obs, ScalarReward(reward), shaping)
            }
            EpisodeStatus::Terminated => {
                LunarLanderSnapshot::terminated(obs, ScalarReward(reward), shaping)
            }
            EpisodeStatus::Truncated => {
                LunarLanderSnapshot::truncated(obs, ScalarReward(reward), shaping)
            }
        };
        Ok(snap)
    }
}

// ─── LunarLanderContinuous ────────────────────────────────────────────────────

/// LunarLander with a 2-dimensional continuous action space.
///
/// Each action is a `[f32; 2]` vector wrapped in [`LunarLanderContinuousAction`].
/// Both components must lie in `[-1, 1]` and be finite; `step` returns
/// `Err(EnvironmentError::InvalidAction)` otherwise (design decision D5).
///
/// The main-engine component maps as follows: values in `[-1, 0]` are treated as
/// off (no thrust); values in `(0, 1]` scale the main engine linearly. The lateral
/// component drives the side engines symmetrically.
///
/// The step-limit is enforced internally (`config.max_steps`, default 1000).
/// Wrapping with `TimeLimit` is not supported because this environment uses a
/// custom snapshot type ([`LunarLanderSnapshot`]) rather than `SnapshotBase`.
#[derive(Debug)]
pub struct LunarLanderContinuous {
    core: LunarLanderCore,
}

impl LunarLanderContinuous {
    /// Construct with the given configuration.
    ///
    /// The Rapier2D world is built immediately; see [`LunarLanderDiscrete::with_config`]
    /// for notes that apply equally here.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `config` fails [`Validate`] (e.g.
    /// non-positive `main_engine_power`, `dt`, or `max_steps == 0`).
    pub fn with_config(config: LunarLanderConfig) -> Result<Self, ConfigError> {
        Ok(Self {
            core: LunarLanderCore::new(config)?,
        })
    }
}

impl ConstructableEnv for LunarLanderContinuous {
    fn new(_render: bool) -> Self {
        Self::with_config(LunarLanderConfig::default()).expect("default config must validate")
    }
}

impl Environment<1, 1, 1> for LunarLanderContinuous {
    type StateType = LunarLanderState;
    type ObservationType = LunarLanderObservation;
    type ActionType = LunarLanderContinuousAction;
    type RewardType = ScalarReward;
    type SnapshotType = LunarLanderSnapshot;

    /// Rebuild the physics world and return the initial observation.
    ///
    /// Identical in behaviour to [`LunarLanderDiscrete::reset`]: reward is `0.0`
    /// and the shaping potential is initialised from the spawn position.
    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.core.rebuild();
        let obs = self.core.state.last_obs.clone();
        Ok(LunarLanderSnapshot::running(
            obs,
            ScalarReward(0.0),
            self.core.shaping_value(),
        ))
    }

    /// Advance the simulation by one timestep and return the resulting snapshot.
    ///
    /// # Errors
    ///
    /// Returns `Err(EnvironmentError::InvalidAction)` if either component of
    /// `action` is outside `[-1, 1]` or is non-finite (design decision D5).
    ///
    /// On success the snapshot status mirrors [`LunarLanderDiscrete::step`]:
    /// `Running`, `Terminated` (crash/landing), or `Truncated` (step limit).
    fn step(
        &mut self,
        action: LunarLanderContinuousAction,
    ) -> Result<LunarLanderSnapshot, EnvironmentError> {
        // D5: validate action bounds
        if !action.is_valid() {
            return Err(EnvironmentError::InvalidAction(format!(
                "LunarLanderContinuousAction components must be in [-1, 1], got {:?}",
                action.0
            )));
        }
        let [main_raw, lateral] = action.0;
        // main engine: [0, 1] throttle when > 0
        let main = main_raw.max(0.0);
        let (obs, reward, status) = self.core.step_common(main, lateral);
        let shaping = self.core.shaping_value();
        let snap = match status {
            EpisodeStatus::Running => {
                LunarLanderSnapshot::running(obs, ScalarReward(reward), shaping)
            }
            EpisodeStatus::Terminated => {
                LunarLanderSnapshot::terminated(obs, ScalarReward(reward), shaping)
            }
            EpisodeStatus::Truncated => {
                LunarLanderSnapshot::truncated(obs, ScalarReward(reward), shaping)
            }
        };
        Ok(snap)
    }
}

// ---------------------------------------------------------------------------
// ASCII renderer
// ---------------------------------------------------------------------------

impl LunarLanderCore {
    fn collect_bodies(&self) -> Vec<super::super::render::Bodyish> {
        use super::super::render::Bodyish;

        let mut bodies = Vec::with_capacity(3);
        let world = &self.world;
        if let Some(lander) = world.bodies().get(self.state.lander_handle) {
            let p = lander.translation();
            bodies.push(Bodyish::Agent {
                x: p.x,
                y: p.y,
                angle_rad: lander.rotation().angle(),
            });
        }
        for handle in [self.state.leg1_handle, self.state.leg2_handle] {
            if let Some(leg) = world.bodies().get(handle) {
                let p = leg.translation();
                bodies.push(Bodyish::Dynamic { x: p.x, y: p.y });
            }
        }
        bodies
    }
}

fn lander_viewport() -> super::super::render::Viewport {
    // Matches the world bounds the env uses for out-of-bounds termination.
    super::super::render::Viewport {
        x_min: 0.0,
        x_max: 20.0,
        y_min: 0.0,
        y_max: 13.3,
    }
}

const LANDER_GROUND_Y: f32 = 0.1;

impl crate::render::AsciiRenderable for LunarLanderDiscrete {
    fn render_ascii(&self) -> String {
        super::super::render::render_box2d_ascii(
            "Lander",
            &self.core.collect_bodies(),
            lander_viewport(),
            Some(LANDER_GROUND_Y),
            self.core.steps,
        )
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        super::super::render::render_box2d_styled(
            "Lander",
            &self.core.collect_bodies(),
            lander_viewport(),
            Some(LANDER_GROUND_Y),
            self.core.steps,
        )
    }
}

// ---------------------------------------------------------------------------
// Report-tier payload — Box2D bodies for the lander, legs, and helipad
// ground.
// ---------------------------------------------------------------------------

impl LunarLanderCore {
    fn box2d_snapshot(&self) -> rlevo_core::render::Box2dSnapshot {
        use rlevo_core::render::{BodyKind, Box2dSnapshot, Point2, RigidBody2D};

        let view = lander_viewport();
        let world = &self.world;

        let mut bodies: Vec<RigidBody2D> = Vec::with_capacity(4);

        // Hull (lander): cuboid with half-extents LANDER_W/2 × LANDER_H/2.
        if let Some(hull) = world.bodies().get(self.state.lander_handle) {
            let p = hull.translation();
            let hw = LANDER_W * 0.5;
            let hh = LANDER_H * 0.5;
            bodies.push(RigidBody2D {
                vertices: vec![
                    Point2::new(-hw, -hh),
                    Point2::new(hw, -hh),
                    Point2::new(hw, hh),
                    Point2::new(-hw, hh),
                ],
                position: Point2::new(p.x, p.y),
                rotation_rad: hull.rotation().angle(),
                kind: BodyKind::Hull,
            });
        }
        // Legs: cuboid 0.05 × 0.3 half-extents.
        for handle in [self.state.leg1_handle, self.state.leg2_handle] {
            if let Some(leg) = world.bodies().get(handle) {
                let p = leg.translation();
                bodies.push(RigidBody2D {
                    vertices: vec![
                        Point2::new(-0.05, -0.3),
                        Point2::new(0.05, -0.3),
                        Point2::new(0.05, 0.3),
                        Point2::new(-0.05, 0.3),
                    ],
                    position: Point2::new(p.x, p.y),
                    rotation_rad: leg.rotation().angle(),
                    kind: BodyKind::Leg,
                });
            }
        }
        // Ground: a thin slab at y = 0 spanning the viewport. Half-height
        // tuned so it reads as a ground line even with rounded corners.
        bodies.push(RigidBody2D {
            vertices: vec![
                Point2::new(-VIEWPORT_W / SCALE / 2.0, -LANDER_GROUND_Y),
                Point2::new(VIEWPORT_W / SCALE / 2.0, -LANDER_GROUND_Y),
                Point2::new(VIEWPORT_W / SCALE / 2.0, LANDER_GROUND_Y),
                Point2::new(-VIEWPORT_W / SCALE / 2.0, LANDER_GROUND_Y),
            ],
            position: Point2::new(VIEWPORT_W / SCALE / 2.0, 0.0),
            rotation_rad: 0.0,
            kind: BodyKind::Ground,
        });

        // Contacts — surfaced from the leg flags; the location is each
        // leg's foot.
        let mut contacts: Vec<Point2> = Vec::new();
        if self.state.leg1_contact
            && let Some(leg) = world.bodies().get(self.state.leg1_handle)
        {
            let p = leg.translation();
            contacts.push(Point2::new(p.x, p.y - 0.3));
        }
        if self.state.leg2_contact
            && let Some(leg) = world.bodies().get(self.state.leg2_handle)
        {
            let p = leg.translation();
            contacts.push(Point2::new(p.x, p.y - 0.3));
        }

        Box2dSnapshot {
            world_bounds: (
                Point2::new(view.x_min, view.y_min),
                Point2::new(view.x_max, view.y_max),
            ),
            bodies,
            contacts,
        }
    }
}

impl rlevo_core::render::Box2dPayloadSource for LunarLanderDiscrete {
    fn box2d_snapshot(&self) -> rlevo_core::render::Box2dSnapshot {
        self.core.box2d_snapshot()
    }
}

impl rlevo_core::render::Box2dPayloadSource for LunarLanderContinuous {
    fn box2d_snapshot(&self) -> rlevo_core::render::Box2dSnapshot {
        self.core.box2d_snapshot()
    }
}

impl crate::render::AsciiRenderable for LunarLanderContinuous {
    fn render_ascii(&self) -> String {
        super::super::render::render_box2d_ascii(
            "Lander",
            &self.core.collect_bodies(),
            lander_viewport(),
            Some(LANDER_GROUND_Y),
            self.core.steps,
        )
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        super::super::render::render_box2d_styled(
            "Lander",
            &self.core.collect_bodies(),
            lander_viewport(),
            Some(LANDER_GROUND_Y),
            self.core.steps,
        )
    }
}

#[cfg(test)]
impl LunarLanderContinuous {
    /// Borrow the shared core's physics state (test-only introspection).
    pub(crate) fn core_state(&self) -> &LunarLanderState {
        &self.core.state
    }

    /// Mutably borrow the shared core's physics state (test-only introspection).
    pub(crate) fn core_state_mut(&mut self) -> &mut LunarLanderState {
        &mut self.core.state
    }
}

#[cfg(test)]
mod tests {
    use super::super::snapshot::METADATA_KEY_SHAPING;
    use super::*;
    use rlevo_core::action::DiscreteAction;
    use rlevo_core::base::Observation;
    use rlevo_core::environment::Snapshot;

    #[test]
    fn test_discrete_action_count() {
        assert_eq!(LunarLanderDiscreteAction::ACTION_COUNT, 4);
    }

    #[test]
    fn test_obs_shape() {
        assert_eq!(LunarLanderObservation::shape(), [8]);
    }

    #[test]
    fn test_reset_returns_running() {
        let mut env =
            LunarLanderDiscrete::with_config(LunarLanderConfig::default()).expect("valid config");
        let snap = env.reset().unwrap();
        assert!(!snap.is_done());
    }

    #[test]
    fn test_shaping_metadata_present() {
        let mut env =
            LunarLanderDiscrete::with_config(LunarLanderConfig::default()).expect("valid config");
        env.reset().unwrap();
        let snap = env.step(LunarLanderDiscreteAction::MainEngine).unwrap();
        let meta = snap.metadata().expect("metadata must be Some");
        assert!(
            meta.components.contains_key(METADATA_KEY_SHAPING),
            "shaping key must be in metadata"
        );
    }

    #[test]
    fn test_d5_continuous_out_of_range() {
        let mut env =
            LunarLanderContinuous::with_config(LunarLanderConfig::default()).expect("valid config");
        env.reset().unwrap();
        let bad = LunarLanderContinuousAction([2.0, 0.0]);
        assert!(env.step(bad).is_err(), "D5: out-of-range action must error");
    }

    #[test]
    fn test_wind_constant_affects_obs() {
        let no_wind_cfg = LunarLanderConfig::builder()
            .seed(1)
            .build()
            .expect("valid config");
        let wind_cfg = LunarLanderConfig::builder()
            .seed(1)
            .wind_mode(WindMode::Constant { force: 5.0 })
            .build()
            .expect("valid config");

        let mut env_no_wind = LunarLanderDiscrete::with_config(no_wind_cfg).expect("valid config");
        let mut env_wind = LunarLanderDiscrete::with_config(wind_cfg).expect("valid config");
        env_no_wind.reset().unwrap();
        env_wind.reset().unwrap();

        for _ in 0..20 {
            let _ = env_no_wind.step(LunarLanderDiscreteAction::DoNothing);
            let _ = env_wind.step(LunarLanderDiscreteAction::DoNothing);
        }
        let obs_no_wind = env_no_wind.core.state.last_obs.values;
        let obs_wind = env_wind.core.state.last_obs.values;
        assert_ne!(
            obs_no_wind, obs_wind,
            "constant wind should cause different trajectory"
        );
    }

    #[test]
    fn test_determinism_no_wind() {
        let cfg = LunarLanderConfig::builder()
            .seed(99)
            .build()
            .expect("valid config");
        let actions = vec![
            LunarLanderDiscreteAction::MainEngine,
            LunarLanderDiscreteAction::DoNothing,
            LunarLanderDiscreteAction::LeftEngine,
            LunarLanderDiscreteAction::RightEngine,
        ];

        let run = |acts: &[LunarLanderDiscreteAction]| {
            let mut env = LunarLanderDiscrete::with_config(cfg.clone()).expect("valid config");
            env.reset().unwrap();
            let mut last = [0.0f32; 8];
            for &a in acts {
                if let Ok(snap) = env.step(a) {
                    last = snap.observation().values;
                }
            }
            last
        };

        assert_eq!(run(&actions), run(&actions));
    }

    /// Regression (#98, ADR 0037): firing the main engine at constant thrust
    /// each step must not integrate a monotonically growing force. The shared
    /// [`RapierWorld::step`] clears external forces after integrating, so the
    /// net (thrust − gravity) force is constant and the per-step Δ(vertical
    /// velocity) stays bounded/decaying under damping. With the accumulation
    /// bug the effective thrust grew each step, so Δvy grew ~linearly.
    #[test]
    fn test_constant_main_engine_delta_vy_does_not_grow() {
        let cfg = LunarLanderConfig::builder()
            .seed(7)
            .build()
            .expect("valid config");
        let mut env = LunarLanderDiscrete::with_config(cfg).expect("valid config");
        env.reset().unwrap();

        let vy = |e: &LunarLanderDiscrete| -> f32 {
            e.core
                .world
                .bodies()
                .get(e.core.state.lander_handle)
                .map_or(0.0, |b| b.linvel().y)
        };

        let mut prev: f32 = vy(&env);
        let mut deltas: Vec<f32> = Vec::new();
        for _ in 0..40 {
            env.step(LunarLanderDiscreteAction::MainEngine).unwrap();
            let v: f32 = vy(&env);
            deltas.push(v - prev);
            prev = v;
        }

        // Under correct (non-accumulating) physics the net force is constant, so
        // |Δvy| is stationary and decays under damping: the second half's mean
        // magnitude must not exceed the first half's. Under the accumulation bug
        // the second half dwarfs the first.
        let mid: usize = deltas.len() / 2;
        let mean_abs = |slice: &[f32]| -> f32 {
            slice.iter().map(|d| d.abs()).sum::<f32>() / slice.len() as f32
        };
        let first_half: f32 = mean_abs(&deltas[..mid]);
        let second_half: f32 = mean_abs(&deltas[mid..]);
        assert!(
            second_half <= first_half + 1e-3,
            "mean |Δvy| grew: first half {first_half}, second half {second_half} \
             (force accumulating?)"
        );
    }

    /// Terminal transition: reaching `max_steps` without a terminal event must
    /// `Truncated`, not `Terminated`. Three `DoNothing` steps under `max_steps
    /// == 3` leave the lander mid-air (it spawns high and falls under gravity),
    /// so the third snapshot truncates.
    #[test]
    fn test_step_truncates_at_max_steps() {
        let cfg = LunarLanderConfig::builder()
            .seed(0)
            .max_steps(3)
            .build()
            .expect("valid config");
        let mut env = LunarLanderDiscrete::with_config(cfg).expect("valid config");
        env.reset().unwrap();

        let snap1 = env.step(LunarLanderDiscreteAction::DoNothing).unwrap();
        assert!(!snap1.is_done(), "step 1 should still be running");
        let snap2 = env.step(LunarLanderDiscreteAction::DoNothing).unwrap();
        assert!(!snap2.is_done(), "step 2 should still be running");
        let snap3 = env.step(LunarLanderDiscreteAction::DoNothing).unwrap();

        assert!(
            snap3.is_truncated(),
            "3rd step at max_steps=3 must truncate"
        );
        assert!(
            !snap3.is_terminated(),
            "step-limit truncation must not report terminated"
        );
    }

    /// Terminal transition: dropping the lander under gravity with no thrust
    /// crashes the hull into the ground, which must `Terminated` (not truncate)
    /// before `max_steps` and apply the −100 crash penalty.
    ///
    /// This is the regression for issue #122: the crash branch was previously
    /// gated on `pos.y < 0.1`, which Rapier's solver never reaches (a hull
    /// resting on the ground settles at `pos.y ≈ 0.78`), so the branch was dead
    /// and the episode silently ran to `Truncated`. Against the old code this
    /// test fails (free-fall never terminates, so `terminal` stays `None` and
    /// the `expect` panics); with the hull-contact check it passes.
    ///
    /// The terminal reward is exactly −100.0: rlevo matches Gymnasium's
    /// **overwrite** semantics, where a terminal step's reward is *set to*
    /// ∓100, discarding that step's shaping delta and control cost.
    #[test]
    fn test_step_terminates_on_hull_crash() {
        let cfg = LunarLanderConfig::builder()
            .seed(0)
            .build()
            .expect("valid config");
        let max_steps = cfg.max_steps;
        let mut env = LunarLanderDiscrete::with_config(cfg).expect("valid config");
        env.reset().unwrap();

        // Capture (steps_taken, terminal_reward).
        let mut terminal: Option<(usize, f32)> = None;
        for i in 0..max_steps {
            let snap = env.step(LunarLanderDiscreteAction::DoNothing).unwrap();
            if snap.is_terminated() {
                terminal = Some((i + 1, snap.reward().0));
                break;
            }
            assert!(
                !snap.is_truncated(),
                "must terminate (crash) before truncating at step {i}"
            );
        }

        let (steps_taken, reward) =
            terminal.expect("free-fall must terminate via hull crash (dead branch under old code)");
        assert!(
            steps_taken < max_steps,
            "crash must terminate before max_steps ({steps_taken} < {max_steps})"
        );
        // Gym overwrite semantics: a crash terminal step's reward is exactly −100.
        approx::assert_relative_eq!(reward, -100.0, epsilon = 1e-4);
    }

    // Landing test (issue #122 §7.1 test 3) intentionally omitted: a soft
    // landing that reports `is_terminated()` with reward ≈ +100 requires a
    // control policy to null out velocity/angle and let the legs settle at
    // rest (leg1_contact && leg2_contact && |vx|,|vy|,|angle| < 0.1). The
    // discrete actions cannot deterministically achieve this within a fixed,
    // hand-written script without flakiness, so the landing branch is left to
    // integration-level policy tests. Tests 1 and 2 above are deterministic and
    // directly exercise the truncation and (newly live) hull-crash branches.

    #[test]
    fn render_styled_matches_ascii() {
        use crate::render::AsciiRenderable;

        let mut env =
            LunarLanderDiscrete::with_config(LunarLanderConfig::default()).expect("valid config");
        env.reset().unwrap();
        let plain_no_trailing: String = env.render_ascii().lines().collect::<Vec<_>>().join("\n");
        assert_eq!(env.render_styled().plain_text(), plain_no_trailing);
    }

    #[test]
    fn render_styled_uses_palette_consts() {
        use crate::render::AsciiRenderable;
        use crate::render::palette::{AGENT_FG, AGENT_MODIFIER};

        let mut env =
            LunarLanderDiscrete::with_config(LunarLanderConfig::default()).expect("valid config");
        env.reset().unwrap();
        let styled = env.render_styled();
        let label = styled.lines[0]
            .spans
            .iter()
            .find(|s| s.text == "Lander")
            .expect("Lander label span present");
        assert_eq!(label.style.fg, Some(AGENT_FG));
        assert!(label.style.modifier.contains(AGENT_MODIFIER));
    }

    #[test]
    fn render_ascii_within_width_budget() {
        use crate::render::AsciiRenderable;

        let mut env =
            LunarLanderDiscrete::with_config(LunarLanderConfig::default()).expect("valid config");
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
