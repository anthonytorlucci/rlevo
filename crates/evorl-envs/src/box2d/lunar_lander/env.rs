//! LunarLander environment implementation (D1: discrete and continuous variants).

use evorl_core::base::Action;
use evorl_core::environment::{Environment, EnvironmentError, EpisodeStatus};
use evorl_core::reward::ScalarReward;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rapier2d::dynamics::RevoluteJoint;
use rapier2d::geometry::ColliderHandle;
use rapier2d::prelude::*;

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
    fn new(config: LunarLanderConfig) -> Self {
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
        core
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
        let vx = self.rng.random_range(-self.config.initial_random..=self.config.initial_random);
        let vy = self.rng.random_range(-self.config.initial_random..=self.config.initial_random);

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
            let leg_rb = self.world.add_body(
                RigidBodyBuilder::dynamic().translation(Vector::new(leg_x, leg_y)),
            );
            self.world.add_collider(
                ColliderBuilder::cuboid(0.05, 0.3).density(1.0).friction(0.5),
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
        let leg1_col = self.world.bodies().get(self.state.leg1_handle)
            .and_then(|b| b.colliders().iter().next().copied());
        let leg2_col = self.world.bodies().get(self.state.leg2_handle)
            .and_then(|b| b.colliders().iter().next().copied());
        self.state.leg1_contact = leg1_col.is_some_and(|c| self.world.is_in_contact(c));
        self.state.leg2_contact = leg2_col.is_some_and(|c| self.world.is_in_contact(c));
    }

    fn compute_obs(&self) -> LunarLanderObservation {
        let bodies = self.world.bodies();
        let (x, y, vx, vy, angle, angvel) =
            bodies.get(self.state.lander_handle).map_or(
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                |b| {
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
                },
            );
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

    fn step_common(&mut self, main: f32, lateral: f32) -> (LunarLanderObservation, f32, EpisodeStatus) {
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
        let is_crashed = pos.y < 0.1 && !self.state.leg1_contact && !self.state.leg2_contact;
        let is_out_of_bounds = pos.x < 0.0
            || pos.x > VIEWPORT_W / SCALE
            || pos.y > VIEWPORT_H / SCALE;
        let is_landed = self.state.leg1_contact
            && self.state.leg2_contact
            && obs.vx().abs() < 0.1
            && obs.vy().abs() < 0.1
            && obs.angle().abs() < 0.1;

        let status = if is_crashed || is_out_of_bounds {
            reward -= 100.0;
            EpisodeStatus::Terminated
        } else if is_landed {
            reward += 100.0;
            EpisodeStatus::Terminated
        } else if self.steps >= self.config.max_steps {
            EpisodeStatus::Truncated
        } else {
            EpisodeStatus::Running
        };

        self.state.last_obs = obs.clone();
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
/// Wrap with `TimeLimit` is not supported for this environment (custom snapshot type).
#[derive(Debug)]
pub struct LunarLanderDiscrete {
    core: LunarLanderCore,
}

impl LunarLanderDiscrete {
    /// Create with default configuration.
    pub fn with_config(config: LunarLanderConfig) -> Self {
        Self { core: LunarLanderCore::new(config) }
    }
}

impl Environment<1, 1, 1> for LunarLanderDiscrete {
    type StateType = LunarLanderState;
    type ObservationType = LunarLanderObservation;
    type ActionType = LunarLanderDiscreteAction;
    type RewardType = ScalarReward;
    type SnapshotType = LunarLanderSnapshot;

    fn new(_render: bool) -> Self {
        Self::with_config(LunarLanderConfig::default())
    }

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.core.rebuild();
        let obs = self.core.state.last_obs.clone();
        Ok(LunarLanderSnapshot::running(obs, ScalarReward(0.0), self.core.shaping_value()))
    }

    fn step(
        &mut self,
        action: LunarLanderDiscreteAction,
    ) -> Result<LunarLanderSnapshot, EnvironmentError> {
        let (main, lateral) = match action {
            LunarLanderDiscreteAction::DoNothing  => (0.0, 0.0),
            LunarLanderDiscreteAction::LeftEngine  => (0.0, -1.0),
            LunarLanderDiscreteAction::MainEngine  => (1.0, 0.0),
            LunarLanderDiscreteAction::RightEngine => (0.0, 1.0),
        };
        let (obs, reward, status) = self.core.step_common(main, lateral);
        let shaping = self.core.shaping_value();
        let snap = match status {
            EpisodeStatus::Running     => LunarLanderSnapshot::running(obs, ScalarReward(reward), shaping),
            EpisodeStatus::Terminated  => LunarLanderSnapshot::terminated(obs, ScalarReward(reward), shaping),
            EpisodeStatus::Truncated   => LunarLanderSnapshot::truncated(obs, ScalarReward(reward), shaping),
        };
        Ok(snap)
    }
}

// ─── LunarLanderContinuous ────────────────────────────────────────────────────

/// LunarLander with a 2-dimensional continuous action space.
///
/// The step-limit is enforced internally (`config.max_steps`, default 1000).
#[derive(Debug)]
pub struct LunarLanderContinuous {
    core: LunarLanderCore,
}

impl LunarLanderContinuous {
    /// Create with default configuration.
    pub fn with_config(config: LunarLanderConfig) -> Self {
        Self { core: LunarLanderCore::new(config) }
    }
}

impl Environment<1, 1, 1> for LunarLanderContinuous {
    type StateType = LunarLanderState;
    type ObservationType = LunarLanderObservation;
    type ActionType = LunarLanderContinuousAction;
    type RewardType = ScalarReward;
    type SnapshotType = LunarLanderSnapshot;

    fn new(_render: bool) -> Self {
        Self::with_config(LunarLanderConfig::default())
    }

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.core.rebuild();
        let obs = self.core.state.last_obs.clone();
        Ok(LunarLanderSnapshot::running(obs, ScalarReward(0.0), self.core.shaping_value()))
    }

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
            EpisodeStatus::Running     => LunarLanderSnapshot::running(obs, ScalarReward(reward), shaping),
            EpisodeStatus::Terminated  => LunarLanderSnapshot::terminated(obs, ScalarReward(reward), shaping),
            EpisodeStatus::Truncated   => LunarLanderSnapshot::truncated(obs, ScalarReward(reward), shaping),
        };
        Ok(snap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::snapshot::METADATA_KEY_SHAPING;
    use evorl_core::action::DiscreteAction;
    use evorl_core::base::Observation;
    use evorl_core::environment::Snapshot;

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
        let mut env = LunarLanderDiscrete::with_config(LunarLanderConfig::default());
        let snap = env.reset().unwrap();
        assert!(!snap.is_done());
    }

    #[test]
    fn test_shaping_metadata_present() {
        let mut env = LunarLanderDiscrete::with_config(LunarLanderConfig::default());
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
        let mut env = LunarLanderContinuous::with_config(LunarLanderConfig::default());
        env.reset().unwrap();
        let bad = LunarLanderContinuousAction([2.0, 0.0]);
        assert!(env.step(bad).is_err(), "D5: out-of-range action must error");
    }

    #[test]
    fn test_wind_constant_affects_obs() {
        let no_wind_cfg = LunarLanderConfig::builder().seed(1).build();
        let wind_cfg = LunarLanderConfig::builder()
            .seed(1)
            .wind_mode(WindMode::Constant { force: 5.0 })
            .build();

        let mut env_no_wind = LunarLanderDiscrete::with_config(no_wind_cfg);
        let mut env_wind = LunarLanderDiscrete::with_config(wind_cfg);
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
        let cfg = LunarLanderConfig::builder().seed(99).build();
        let actions = vec![
            LunarLanderDiscreteAction::MainEngine,
            LunarLanderDiscreteAction::DoNothing,
            LunarLanderDiscreteAction::LeftEngine,
            LunarLanderDiscreteAction::RightEngine,
        ];

        let run = |acts: &[LunarLanderDiscreteAction]| {
            let mut env = LunarLanderDiscrete::with_config(cfg.clone());
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
}
