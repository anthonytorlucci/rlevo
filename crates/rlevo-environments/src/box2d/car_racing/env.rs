//! CarRacing environment: `Environment` and `ConstructableEnv` implementations.
//!
//! This module wires the Rapier2D physics world, the [`Track`] generator, the
//! [`Rasterizer`], and the tile-visit logic into the [`CarRacing`] struct that
//! implements [`Environment<3, 3, 1>`](rlevo_core::environment::Environment).
//!
//! Physics is integrated at a fixed timestep of `config.dt` (default 1/50 s)
//! via [`RapierWorld::step`](crate::box2d::physics::RapierWorld). Car dynamics
//! are approximated by direct force application:
//!
//! - **Gas**: forward force proportional to `gas × 500 × car_density`.
//! - **Brake**: opposing force proportional to the current linear velocity scaled
//!   by `brake × 200 × car_density`.
//! - **Steer**: torque impulse proportional to `steer × speed × 2`, so steering
//!   authority grows with speed.
//! - **Lateral friction**: velocity component perpendicular to the car heading is
//!   damped to prevent unrealistic sliding.
//!
//! Tile visits are detected by a nearest-centre scan on every step, not by
//! physics collision callbacks. A tile can only be counted once per episode.
//!
//! The camera-following ASCII renderer ([`AsciiRenderable`](crate::render::AsciiRenderable))
//! shows a 20-unit-wide window centred on the car body. Full track tile geometry
//! is available only in the report tier via `FamilyPayload::Box2D`.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rapier2d::prelude::*;
use rlevo_core::base::Action;
use rlevo_core::config::{ConfigError, Validate};
use rlevo_core::environment::{
    ConstructableEnv, Environment, EnvironmentError, EpisodeStatus, SnapshotBase,
};
use rlevo_core::reward::ScalarReward;

use crate::box2d::physics::RapierWorld;

use super::action::CarRacingAction;
use super::config::CarRacingConfig;
use super::observation::CarRacingObservation;
use super::rasterizer::{FRAME_SIZE, Rasterizer};
use super::state::CarRacingState;
use super::track::Track;

/// Viewport width in world units.
const VIEWPORT_W: f32 = 600.0 / 30.0;
/// Car half-width.
const CAR_W: f32 = 2.0 / 30.0;
/// Car half-height.
const CAR_H: f32 = 4.0 / 30.0;

/// CarRacing reinforcement learning environment.
///
/// A top-down 2D car racing environment. The agent receives a 96×96 RGB
/// pixel observation and outputs steering, gas, and brake controls.
///
/// # Episode lifecycle
///
/// - `reset()` generates a new procedural track.
/// - `step(action)` applies car controls and advances physics.
/// - `Terminated` when the car visits ≥ 95% of track tiles.
/// - `Truncated` after `config.max_steps` steps.
///
/// # Observation (96×96×3)
///
/// Pixel rendering of the track from a top-down view centred on the car.
///
/// # Action (3 dims, D5: asymmetric bounds)
///
/// `steer ∈ [−1, 1]`, `gas ∈ [0, 1]`, `brake ∈ [0, 1]`.
#[derive(Debug)]
pub struct CarRacing {
    world: RapierWorld,
    state: CarRacingState,
    track: Track,
    config: CarRacingConfig,
    rng: StdRng,
    rasterizer: Rasterizer,
    steps: usize,
}

impl CarRacing {
    /// Create with an explicit configuration.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `config` fails [`Validate`] (e.g.
    /// non-positive `track_width`, `lap_complete_percent` outside `(0, 1]`, or
    /// a non-negative `frame_penalty`).
    pub fn with_config(config: CarRacingConfig) -> Result<Self, ConfigError> {
        config.validate()?;
        let mut rng = StdRng::seed_from_u64(config.seed);
        let track = Track::generate(&config, &mut rng);
        let mut env = Self {
            world: RapierWorld::new(Vector::new(0.0, 0.0), config.dt),
            state: CarRacingState {
                car_handle: RigidBodyHandle::invalid(),
                wheel_handles: [RigidBodyHandle::invalid(); 4],
                current_tile: 0,
                tiles_visited: 0,
                total_tiles: track.tiles.len(),
                lap_complete: false,
                last_obs: CarRacingObservation::default(),
            },
            track,
            config,
            rng,
            rasterizer: Rasterizer::new(),
            steps: 0,
        };
        env.build_world();
        Ok(env)
    }

    fn build_world(&mut self) {
        self.world = RapierWorld::new(Vector::new(0.0, self.config.gravity), self.config.dt);

        let [sx, sy] = self.track.start_pos;
        let angle = self.track.start_angle;

        // Car body
        let car_rb = self.world.add_body(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(sx, sy))
                .rotation(angle)
                .linear_damping(0.3)
                .angular_damping(1.0),
        );
        self.world.add_collider(
            ColliderBuilder::cuboid(CAR_W, CAR_H)
                .density(self.config.car_density)
                .friction(self.config.friction),
            car_rb,
        );
        self.state.car_handle = car_rb;

        // 4 wheels (simplified as small boxes attached to car)
        let offsets = [
            [-CAR_W - 0.05, CAR_H * 0.7],  // FL
            [CAR_W + 0.05, CAR_H * 0.7],   // FR
            [-CAR_W - 0.05, -CAR_H * 0.7], // RL
            [CAR_W + 0.05, -CAR_H * 0.7],  // RR
        ];
        for (i, off) in offsets.iter().enumerate() {
            let wx = sx + off[0] * angle.cos() - off[1] * angle.sin();
            let wy = sy + off[0] * angle.sin() + off[1] * angle.cos();
            let wheel_rb = self.world.add_body(
                RigidBodyBuilder::dynamic()
                    .translation(Vector::new(wx, wy))
                    .rotation(angle),
            );
            self.world.add_collider(
                ColliderBuilder::cuboid(0.02, 0.04)
                    .density(0.0001)
                    .friction(self.config.friction),
                wheel_rb,
            );
            // Fixed joint to car body
            let joint = FixedJointBuilder::new()
                .local_anchor1(Vector::new(off[0], off[1]))
                .local_anchor2(Vector::ZERO);
            self.world.add_joint(joint, car_rb, wheel_rb, true);
            self.state.wheel_handles[i] = wheel_rb;
        }

        // Tile visits are detected via nearest_tile() position query, not physics contacts.
        self.state.current_tile = 0;
        self.state.tiles_visited = 0;
        self.state.lap_complete = false;
        self.steps = 0;
    }

    fn apply_controls(&mut self, action: &CarRacingAction) {
        if let Some(car) = self.world.bodies_mut().get_mut(self.state.car_handle) {
            let angle = car.rotation().angle();
            let forward = Vector::new(-angle.sin(), angle.cos());
            let right = Vector::new(angle.cos(), angle.sin());

            // Gas → forward thrust
            let thrust = action.gas() * 500.0 * self.config.car_density;
            car.add_force(forward * thrust, true);

            // Brake → opposing velocity
            let vel = car.linvel();
            let brake_force = vel * (-action.brake() * 200.0 * self.config.car_density);
            car.add_force(brake_force, true);

            // Steer → torque
            let speed = vel.length();
            let steer_torque = action.steer() * speed * 2.0;
            car.apply_torque_impulse(steer_torque, true);

            // Lateral friction (prevent drifting)
            let lateral_vel = vel.dot(right);
            car.add_force(
                right * (-lateral_vel * 300.0 * self.config.car_density),
                true,
            );
        }
    }

    fn update_tile_visits(&mut self) -> f32 {
        let pos = self
            .world
            .bodies()
            .get(self.state.car_handle)
            .map(|b| [b.translation().x, b.translation().y])
            .unwrap_or([0.0; 2]);

        let nearest = self.track.nearest_tile(pos);
        let mut tile_reward = 0.0;
        if let Some(idx) = nearest {
            if !self.track.tiles[idx].visited {
                self.track.tiles[idx].visited = true;
                self.state.tiles_visited += 1;
                tile_reward = self.config.tile_reward;
            }
            self.state.current_tile = idx;
        }

        let lap_threshold =
            (self.config.lap_complete_percent * self.state.total_tiles as f32) as usize;
        if self.state.tiles_visited >= lap_threshold {
            self.state.lap_complete = true;
        }
        tile_reward
    }

    fn render_frame(&mut self) -> CarRacingObservation {
        let car_pos = self
            .world
            .bodies()
            .get(self.state.car_handle)
            .map(|b| [b.translation().x, b.translation().y])
            .unwrap_or([0.0; 2]);
        let car_angle = self
            .world
            .bodies()
            .get(self.state.car_handle)
            .map(|b| b.rotation().angle())
            .unwrap_or(0.0);

        // Background: green grass
        self.rasterizer.clear([102, 204, 102]);

        let scale = FRAME_SIZE as f32 / VIEWPORT_W;
        let cx = FRAME_SIZE as f32 / 2.0;
        let cy = FRAME_SIZE as f32 / 2.0;

        // Helper: transform world coords to pixel coords (camera follows car)
        let to_pixel = |wx: f32, wy: f32| -> [f32; 2] {
            let dx = wx - car_pos[0];
            let dy = wy - car_pos[1];
            [cx + dx * scale, cy + dy * scale]
        };

        // Draw road tiles
        for tile in &self.track.tiles {
            let px_verts: Vec<[f32; 2]> =
                tile.vertices.iter().map(|v| to_pixel(v[0], v[1])).collect();
            self.rasterizer.fill_polygon(&px_verts, tile.color);
        }

        // Draw car (white rectangle)
        let car_corners_world = [
            [-CAR_W, -CAR_H],
            [CAR_W, -CAR_H],
            [CAR_W, CAR_H],
            [-CAR_W, CAR_H],
        ];
        let car_px: Vec<[f32; 2]> = car_corners_world
            .iter()
            .map(|&[lx, ly]| {
                let wx = car_pos[0] + lx * car_angle.cos() - ly * car_angle.sin();
                let wy = car_pos[1] + lx * car_angle.sin() + ly * car_angle.cos();
                to_pixel(wx, wy)
            })
            .collect();
        self.rasterizer.fill_polygon(&car_px, [255, 255, 255]);

        CarRacingObservation::new(*self.rasterizer.pixels())
    }
}

impl ConstructableEnv for CarRacing {
    /// Construct a `CarRacing` environment with default configuration and seed 0.
    ///
    /// The `render` flag is accepted for interface compatibility but has no
    /// effect; the pixel observation is always produced regardless of this value.
    /// Use [`CarRacing::with_config`] to control the seed and other parameters.
    fn new(_render: bool) -> Self {
        Self::with_config(CarRacingConfig::default()).expect("default config must validate")
    }
}

impl Environment<3, 3, 1> for CarRacing {
    type StateType = CarRacingState;
    type ObservationType = CarRacingObservation;
    type ActionType = CarRacingAction;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<3, CarRacingObservation, ScalarReward>;

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.track = Track::generate(&self.config, &mut self.rng);
        self.state.total_tiles = self.track.tiles.len();
        self.build_world();
        let obs = self.render_frame();
        self.state.last_obs = obs.clone();
        Ok(SnapshotBase::running(obs, ScalarReward(0.0)))
    }

    fn step(&mut self, action: CarRacingAction) -> Result<Self::SnapshotType, EnvironmentError> {
        // D5: validate asymmetric bounds
        if !action.is_valid() {
            return Err(EnvironmentError::InvalidAction(format!(
                "CarRacingAction invalid: steer={}, gas={}, brake={}",
                action.steer(),
                action.gas(),
                action.brake()
            )));
        }

        self.apply_controls(&action);
        self.world.step();
        self.steps += 1;

        let tile_reward = self.update_tile_visits();
        let reward = tile_reward + self.config.frame_penalty;

        let status = if self.state.lap_complete {
            EpisodeStatus::Terminated
        } else if self.steps >= self.config.max_steps {
            EpisodeStatus::Truncated
        } else {
            EpisodeStatus::Running
        };

        let obs = self.render_frame();
        self.state.last_obs = obs.clone();
        Ok(SnapshotBase {
            observation: obs,
            reward: ScalarReward(reward),
            status,
        })
    }
}

// ---------------------------------------------------------------------------
// ASCII renderer
// ---------------------------------------------------------------------------

impl crate::render::AsciiRenderable for CarRacing {
    fn render_ascii(&self) -> String {
        let bodies = self.collect_bodies();
        let viewport = self.viewport();
        super::super::render::render_box2d_ascii("Car", &bodies, viewport, None, self.steps)
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        let bodies = self.collect_bodies();
        let viewport = self.viewport();
        super::super::render::render_box2d_styled("Car", &bodies, viewport, None, self.steps)
    }
}

impl CarRacing {
    fn collect_bodies(&self) -> Vec<super::super::render::Bodyish> {
        use super::super::render::Bodyish;

        let mut bodies = Vec::with_capacity(5);
        if let Some(car) = self.world.bodies().get(self.state.car_handle) {
            let p = car.translation();
            bodies.push(Bodyish::Agent {
                x: p.x,
                y: p.y,
                angle_rad: car.rotation().angle(),
            });
        }
        for handle in self.state.wheel_handles {
            if let Some(wheel) = self.world.bodies().get(handle) {
                let p = wheel.translation();
                bodies.push(Bodyish::Dynamic { x: p.x, y: p.y });
            }
        }
        bodies
    }

    /// Camera-following viewport: 20-unit-wide window centred on the car.
    /// Track tile geometry is not rendered in the library tier; the report
    /// tier owns full track rendering via `FamilyPayload::Box2D`.
    fn viewport(&self) -> super::super::render::Viewport {
        let (cx, cy) = self
            .world
            .bodies()
            .get(self.state.car_handle)
            .map_or((0.0, 0.0), |b| {
                let p = b.translation();
                (p.x, p.y)
            });
        super::super::render::Viewport {
            x_min: cx - 10.0,
            x_max: cx + 10.0,
            y_min: cy - 6.65,
            y_max: cy + 6.65,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlevo_core::base::Observation;
    use rlevo_core::environment::Snapshot;

    /// Creates a default seeded CarRacing environment for use in tests.
    fn make_env() -> CarRacing {
        CarRacing::with_config(CarRacingConfig::default()).expect("valid config")
    }

    #[test]
    fn test_obs_shape() {
        assert_eq!(CarRacingObservation::shape(), [96, 96, 3]);
    }

    #[test]
    fn test_reset_returns_running() {
        let mut env = make_env();
        let snap = env.reset().unwrap();
        assert!(!snap.is_done());
    }

    #[test]
    fn test_d5_negative_gas() {
        let mut env = make_env();
        env.reset().unwrap();
        let bad = CarRacingAction::new(0.0, -0.1, 0.0);
        assert!(env.step(bad).is_err(), "D5: negative gas must error");
    }

    #[test]
    fn test_frame_penalty_every_step() {
        let mut env = make_env();
        env.reset().unwrap();
        let snap = env.step(CarRacingAction::new(0.0, 0.0, 0.0)).unwrap();
        let reward: f32 = (*snap.reward()).into();
        let config = CarRacingConfig::default();
        // Frame penalty is always applied; tile reward may also apply on step 1.
        // Even with one tile: tile_reward + frame_penalty < tile_reward.
        assert!(
            reward < config.tile_reward,
            "frame penalty must reduce reward below tile_reward, got {reward}"
        );
    }

    #[test]
    fn test_track_tile_count() {
        let env = make_env();
        assert!(
            env.track.tiles.len() >= 50,
            "track must have at least 50 tiles"
        );
    }

    #[test]
    fn test_determinism() {
        let cfg = CarRacingConfig::builder()
            .seed(5)
            .build()
            .expect("valid config");
        let action = CarRacingAction::new(0.1, 0.5, 0.0);

        let run = |a: &CarRacingAction| {
            let mut env = CarRacing::with_config(cfg.clone()).expect("valid config");
            env.reset().unwrap();
            let mut reward_sum = 0.0f32;
            for _ in 0..5 {
                if let Ok(snap) = env.step(a.clone()) {
                    let r: f32 = (*snap.reward()).into();
                    reward_sum += r;
                }
            }
            reward_sum
        };

        let a = run(&action);
        let b = run(&action);
        assert!(
            (a - b).abs() < 1e-4,
            "determinism: same seed + actions must give same reward sum"
        );
    }

    /// Regression (#98, ADR 0037): the gas force applied by `apply_controls`
    /// must live exactly one step. A single gas kick followed by an idle step
    /// (gas = 0) must not keep accelerating the car — with the accumulation bug
    /// the forward `user_force` persisted, so the idle step re-applied the full
    /// gas thrust and roughly doubled the speed; with the wrapper clearing
    /// forces the idle step only shows small joint-settling drift.
    ///
    /// The window is deliberately two steps: this ultra-light car has stiff
    /// wheel fixed-joints that make the solver explode once the body carries any
    /// speed (a separate pre-existing property, out of scope for #98), so a
    /// longer constant-throttle rollout is not numerically meaningful here.
    #[test]
    fn test_gas_force_does_not_persist_into_idle_step() {
        let mut env = make_env();
        env.reset().unwrap();

        let speed = |e: &CarRacing| -> f32 {
            e.world
                .bodies()
                .get(e.state.car_handle)
                .map_or(0.0, |b| b.linvel().length())
        };

        // One-shot gas kick → forward motion occurs.
        env.step(CarRacingAction::new(0.0, 0.01, 0.0)).unwrap();
        let kicked: f32 = speed(&env);
        assert!(kicked > 0.0, "gas kick should move the car");

        // One idle step (no gas). The velocity change must be far smaller than
        // the kick itself: with the bug the persisted force would add another
        // ~kicked, roughly doubling speed. Correct physics shows only settling.
        env.step(CarRacingAction::new(0.0, 0.0, 0.0)).unwrap();
        let idle: f32 = speed(&env);
        assert!(
            idle < kicked * 1.5,
            "speed jumped from {kicked} to {idle} on an unactuated step \
             (gas force persisted?)"
        );
    }

    #[test]
    fn render_styled_matches_ascii() {
        use crate::render::AsciiRenderable;

        let mut env = CarRacing::with_config(CarRacingConfig::default()).expect("valid config");
        env.reset().unwrap();
        let plain_no_trailing: String = env.render_ascii().lines().collect::<Vec<_>>().join("\n");
        assert_eq!(env.render_styled().plain_text(), plain_no_trailing);
    }

    #[test]
    fn render_styled_uses_palette_consts() {
        use crate::render::AsciiRenderable;
        use crate::render::palette::{AGENT_FG, AGENT_MODIFIER};

        let mut env = CarRacing::with_config(CarRacingConfig::default()).expect("valid config");
        env.reset().unwrap();
        let styled = env.render_styled();
        let label = styled.lines[0]
            .spans
            .iter()
            .find(|s| s.text == "Car")
            .expect("Car label span present");
        assert_eq!(label.style.fg, Some(AGENT_FG));
        assert!(label.style.modifier.contains(AGENT_MODIFIER));
    }

    #[test]
    fn render_ascii_within_width_budget() {
        use crate::render::AsciiRenderable;

        let mut env = CarRacing::with_config(CarRacingConfig::default()).expect("valid config");
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
