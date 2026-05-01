//! Configuration for [`super::InvertedDoublePendulum`].

use crate::locomotion::common::{Gear, HealthyCheck, TerminationMode};

/// Environment configuration for [`super::InvertedDoublePendulum`].
///
/// Defaults match the Gymnasium v5 XML: gear 100, dt 0.01, frame_skip 1,
/// reset noise 0.1, truncation at 1000, termination on `y_tip ≤ 1.0`.
#[derive(Debug, Clone)]
pub struct InvertedDoublePendulumConfig {
    pub seed: u64,
    pub gear: Gear<1>,
    pub dt: f32,
    pub frame_skip: u32,
    /// Gate on the tip's world-z (Gymnasium's `y_tip`). Default
    /// `z_range = Some((1.0, ∞))`.
    pub healthy: HealthyCheck,
    pub termination: TerminationMode,
    pub reset_noise_scale: f32,
    pub max_steps: usize,
    pub action_clip: (f32, f32),
    // Physical geometry
    pub cart_mass: f32,
    pub pole_mass: f32,
    /// Total length of one pole (capsule length = 2 · pole_half = `pole_length`).
    pub pole_length: f32,
    pub pole_radius: f32,
    pub cart_half_extents: [f32; 3],
    pub gravity: f32,
    // Reward weights
    pub alive_reward: f32,
    pub x_tip_weight: f32,
    pub y_tip_target: f32,
    pub omega1_weight: f32,
    pub omega2_weight: f32,
}

impl Default for InvertedDoublePendulumConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            gear: Gear::new([100.0]),
            dt: 0.01,
            frame_skip: 1,
            healthy: HealthyCheck {
                z_range: Some((1.0, f32::INFINITY)),
                ..HealthyCheck::none()
            },
            termination: TerminationMode::OnUnhealthy,
            reset_noise_scale: 0.1,
            max_steps: 1000,
            action_clip: (-1.0, 1.0),
            cart_mass: 10.0,
            pole_mass: 0.5,
            pole_length: 0.6,
            pole_radius: 0.045,
            cart_half_extents: [0.15, 0.05, 0.05],
            gravity: -9.81,
            alive_reward: 10.0,
            x_tip_weight: 0.01,
            y_tip_target: 2.0,
            omega1_weight: 1e-3,
            omega2_weight: 5e-3,
        }
    }
}
