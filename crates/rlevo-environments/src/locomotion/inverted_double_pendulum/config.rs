//! Configuration for [`super::InvertedDoublePendulum`].

use rlevo_core::bounds::Bounds;
use rlevo_core::config::{self, ConfigError, Validate};

use crate::locomotion::common::{Gear, HealthyCheck, TerminationMode};

/// Environment configuration for [`super::InvertedDoublePendulum`].
///
/// Defaults match the Gymnasium v5 XML: gear 100, dt 0.01, frame_skip 1,
/// reset noise 0.1, truncation at 1000, termination on `y_tip ≤ 1.0`.
#[derive(Debug, Clone)]
pub struct InvertedDoublePendulumConfig {
    /// RNG seed used for both weight initialisation and reset noise sampling.
    /// Each call to `reset` re-seeds from this value, so episodes are
    /// deterministic for a given seed.
    pub seed: u64,
    /// Gear multiplier applied to the clipped action before it becomes a
    /// force (Newtons). Default `[100.0]` matches the Gymnasium v5 XML.
    pub gear: Gear<1>,
    /// Physics timestep in seconds. Default `0.01`.
    pub dt: f32,
    /// Number of Rapier substeps per `Environment::step` call. Default `1`.
    pub frame_skip: u32,
    /// Gate on the tip's world-z (Gymnasium's `y_tip`). Default
    /// `z_range = Some((1.0, ∞))`.
    pub healthy: HealthyCheck,
    /// Whether an unhealthy state triggers `Terminated` or is ignored.
    /// Default `TerminationMode::OnUnhealthy`.
    pub termination: TerminationMode,
    /// Half-width of the uniform distribution used to perturb `cart_x`,
    /// `θ₁`, and `θ₂` at reset; also the standard deviation of the Gaussian
    /// used to perturb velocities. Default `0.1`.
    pub reset_noise_scale: f32,
    /// Maximum number of steps before the episode is truncated. Default
    /// `1000`.
    pub max_steps: usize,
    /// `[min, max]` bounds applied to the raw action before the gear
    /// multiplier. Default `[-1.0, 1.0]`.
    pub action_clip: Bounds,
    // Physical geometry
    /// Total mass of the cart body in kg. Default `10.0`.
    pub cart_mass: f32,
    /// Mass of each pole capsule in kg. Both poles share this value. Default
    /// `0.5`.
    pub pole_mass: f32,
    /// Total length of one pole in metres (capsule height = `pole_length`,
    /// so half-length = `pole_length * 0.5`). Default `0.6`.
    pub pole_length: f32,
    /// Radius of each pole capsule in metres. Default `0.045`.
    pub pole_radius: f32,
    /// Half-extents `[x, y, z]` of the cart cuboid in metres. Default
    /// `[0.15, 0.05, 0.05]`.
    pub cart_half_extents: [f32; 3],
    /// Gravitational acceleration (negative = downward along world-z).
    /// Default `-9.81`.
    pub gravity: f32,
    // Reward weights
    /// Bonus added to the reward each step the tip remains healthy. Default
    /// `10.0`.
    pub alive_reward: f32,
    /// Coefficient of the `x_tip²` penalty. Default `0.01`.
    pub x_tip_weight: f32,
    /// Target height for the pole tip along world-z (Gymnasium's `y_tip`
    /// target). Default `2.0`.
    pub y_tip_target: f32,
    /// Coefficient of the `|ω₁|` angular-velocity penalty for pole1.
    /// Default `1e-3`.
    pub omega1_weight: f32,
    /// Coefficient of the `|ω₂|` angular-velocity penalty for pole2.
    /// Default `5e-3`.
    pub omega2_weight: f32,
}

/// Returns the Gymnasium v5-equivalent defaults.
impl Default for InvertedDoublePendulumConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            gear: Gear::new([100.0]),
            dt: 0.01,
            frame_skip: 1,
            healthy: HealthyCheck {
                z_range: Some(Bounds::new(1.0, f32::INFINITY)),
                ..HealthyCheck::none()
            },
            termination: TerminationMode::OnUnhealthy,
            reset_noise_scale: 0.1,
            max_steps: 1000,
            action_clip: Bounds::new(-1.0, 1.0),
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

impl Validate for InvertedDoublePendulumConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "InvertedDoublePendulumConfig";
        config::positive(C, "dt", f64::from(self.dt))?;
        config::nonzero(C, "frame_skip", self.frame_skip as usize)?;
        config::in_range(
            C,
            "reset_noise_scale",
            0.0,
            f64::INFINITY,
            f64::from(self.reset_noise_scale),
        )?;
        config::nonzero(C, "max_steps", self.max_steps)?;
        config::positive(C, "cart_mass", f64::from(self.cart_mass))?;
        config::positive(C, "pole_mass", f64::from(self.pole_mass))?;
        config::positive(C, "pole_length", f64::from(self.pole_length))?;
        config::positive(C, "pole_radius", f64::from(self.pole_radius))?;
        for extent in self.cart_half_extents {
            config::positive(C, "cart_half_extents", f64::from(extent))?;
        }
        config::in_range(
            C,
            "alive_reward",
            0.0,
            f64::INFINITY,
            f64::from(self.alive_reward),
        )?;
        config::in_range(
            C,
            "x_tip_weight",
            0.0,
            f64::INFINITY,
            f64::from(self.x_tip_weight),
        )?;
        config::in_range(
            C,
            "omega1_weight",
            0.0,
            f64::INFINITY,
            f64::from(self.omega1_weight),
        )?;
        config::in_range(
            C,
            "omega2_weight",
            0.0,
            f64::INFINITY,
            f64::from(self.omega2_weight),
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_validates() {
        assert!(InvertedDoublePendulumConfig::default().validate().is_ok());
    }

    #[test]
    fn rejects_non_positive_pole_length() {
        let bad = InvertedDoublePendulumConfig {
            pole_length: 0.0,
            ..Default::default()
        };
        assert_eq!(bad.validate().unwrap_err().field, "pole_length");
    }
}
