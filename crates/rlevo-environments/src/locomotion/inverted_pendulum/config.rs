//! Configuration for [`super::InvertedPendulum`].

use crate::locomotion::common::{Gear, HealthyCheck, TerminationMode};

/// Environment configuration for [`super::InvertedPendulum`].
///
/// Defaults match the Gymnasium v5 XML (gear 100, dt 0.01, frame_skip 1,
/// healthy-angle band `(-0.2, 0.2)`, reset noise 0.01, truncation at 1000).
#[derive(Debug, Clone)]
pub struct InvertedPendulumConfig {
    /// RNG seed used to initialise the reset-noise sampler.
    pub seed: u64,
    /// Force multiplier applied to the raw action before it is passed to
    /// Rapier. The Gymnasium XML uses `gear = [100]`, so an action of `1.0`
    /// becomes a `100 N` force on the cart.
    pub gear: Gear<1>,
    /// Physics integration timestep in seconds. Gymnasium default: `0.01 s`.
    pub dt: f32,
    /// Number of Rapier substeps taken per `step()` call. Gymnasium default: `1`.
    pub frame_skip: u32,
    /// Healthiness bounds used to determine pole-fall termination. The default
    /// checks only `angle_range = (-0.2, 0.2)` radians.
    pub healthy: HealthyCheck,
    /// Whether an unhealthy state triggers episode termination or is silently
    /// ignored (useful for evaluating without early stopping).
    pub termination: TerminationMode,
    /// Half-width of the uniform noise added to each initial state variable
    /// `(cart_x, pole_angle, cart_vx, pole_angvel_y)` on reset.
    /// Gymnasium default: `0.01`.
    pub reset_noise_scale: f32,
    /// Maximum number of steps before the episode is truncated.
    /// Gymnasium default: `1000`.
    pub max_steps: usize,
    /// Inclusive `(min, max)` bounds to which the raw action scalar is clipped
    /// before the gear multiplier is applied. Default: `(-3.0, 3.0)`.
    pub action_clip: (f32, f32),
    /// Mass of the cart body in kilograms. Default: `10.0 kg`.
    pub cart_mass: f32,
    /// Mass of the pole body in kilograms. Default: `1.0 kg`.
    pub pole_mass: f32,
    /// Full length of the pole from pivot to tip in metres. Default: `0.6 m`.
    pub pole_length: f32,
    /// Capsule radius of the pole collider in metres. Default: `0.05 m`.
    pub pole_radius: f32,
    /// Half-extents `[x, y, z]` of the cart cuboid collider in metres.
    /// Default: `[0.15, 0.05, 0.05]`.
    pub cart_half_extents: [f32; 3],
    /// Gravitational acceleration along the world-z axis in m/s².
    /// Should be negative for downward gravity. Default: `-9.81 m/s²`.
    pub gravity: f32,
}

impl Default for InvertedPendulumConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            gear: Gear::new([100.0]),
            dt: 0.01,
            frame_skip: 1,
            healthy: HealthyCheck {
                angle_range: Some((-0.2, 0.2)),
                ..HealthyCheck::none()
            },
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
