//! Configuration for [`super::InvertedPendulum`].

use crate::locomotion::common::{Gear, HealthyCheck, TerminationMode};

/// Environment configuration for [`super::InvertedPendulum`].
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
