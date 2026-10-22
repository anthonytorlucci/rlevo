//! Configuration for the `BipedalWalker` environment.
//!
//! [`BipedalWalkerConfig`] groups all tunable parameters. Call
//! [`Default::default`] for a flat-terrain, 1600-step episode, and override
//! fields with struct-update syntax.

use rlevo_core::config::{self, ConfigError, Validate};
use serde::{Deserialize, Serialize};

/// Terrain difficulty preset for a [`super::BipedalWalker`] episode.
///
/// The variant is stored in [`BipedalWalkerConfig`] and determines which
/// [`super::terrain::TerrainGenerator`] is used when the environment resets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BipedalTerrain {
    /// Flat ground — no obstacles.
    Flat,
    /// Uneven ground with random height variation.
    Rough,
    /// Obstacles (stumps, pits, stairs) from the Hardcore variant.
    Hardcore,
}

/// Configuration for [`super::BipedalWalker`].
///
/// Override fields of [`Default`] with struct-update syntax. The default is a
/// flat-terrain, seeded-at-zero, 1600-step
/// episode with `motors_torque = 80`, `speed_hip = 4`, `speed_knee = 6`,
/// `lidar_range = 160`, `dt = 1/50 s`, and `gravity = -9.8`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BipedalWalkerConfig {
    /// Terrain difficulty.
    pub terrain: BipedalTerrain,
    /// Hull friction coefficient.
    pub hull_friction: f32,
    /// Leg friction coefficient.
    pub leg_friction: f32,
    /// Maximum motor torque applied to each joint.
    pub motors_torque: f32,
    /// Hip motor speed (rad/s).
    pub speed_hip: f32,
    /// Knee motor speed (rad/s).
    pub speed_knee: f32,
    /// Lidar range (world units).
    pub lidar_range: f32,
    /// RNG seed for terrain generation and initial state.
    pub seed: u64,
    /// Maximum steps per episode (Truncated after this).
    pub max_steps: usize,
    /// Physics timestep (seconds per step).
    pub dt: f32,
    /// Gravity (negative = downward).
    pub gravity: f32,
}

impl Default for BipedalWalkerConfig {
    fn default() -> Self {
        Self {
            terrain: BipedalTerrain::Flat,
            hull_friction: 0.1,
            leg_friction: 0.2,
            motors_torque: 80.0,
            speed_hip: 4.0,
            speed_knee: 6.0,
            lidar_range: 160.0,
            seed: 0,
            max_steps: 1600,
            dt: 1.0 / 50.0,
            gravity: -9.8,
        }
    }
}

impl Validate for BipedalWalkerConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "BipedalWalkerConfig";
        config::in_range(
            C,
            "hull_friction",
            0.0,
            f64::INFINITY,
            f64::from(self.hull_friction),
        )?;
        config::in_range(
            C,
            "leg_friction",
            0.0,
            f64::INFINITY,
            f64::from(self.leg_friction),
        )?;
        config::positive(C, "motors_torque", f64::from(self.motors_torque))?;
        config::positive(C, "speed_hip", f64::from(self.speed_hip))?;
        config::positive(C, "speed_knee", f64::from(self.speed_knee))?;
        config::positive(C, "lidar_range", f64::from(self.lidar_range))?;
        config::nonzero(C, "max_steps", self.max_steps)?;
        config::positive(C, "dt", f64::from(self.dt))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = BipedalWalkerConfig::default();
        assert_eq!(cfg.terrain, BipedalTerrain::Flat);
        assert_eq!(cfg.max_steps, 1600);
        assert!(cfg.dt > 0.0);
    }

    #[test]
    fn default_config_validates() {
        assert!(BipedalWalkerConfig::default().validate().is_ok());
    }

    #[test]
    fn rejects_non_positive_motors_torque() {
        let bad = BipedalWalkerConfig {
            motors_torque: 0.0,
            ..Default::default()
        };
        assert_eq!(bad.validate().unwrap_err().field, "motors_torque");
    }
}
