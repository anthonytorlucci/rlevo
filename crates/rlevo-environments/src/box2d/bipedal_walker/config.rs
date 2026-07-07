//! Configuration for the BipedalWalker environment.
//!
//! [`BipedalWalkerConfig`] groups all tunable parameters. Use
//! [`BipedalWalkerConfig::builder`] for ergonomic construction; call
//! [`Default::default`] for a flat-terrain, 1600-step episode.

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
/// Construct via [`BipedalWalkerConfig::builder`] for named-field ergonomics,
/// or use [`Default`] which produces a flat-terrain, seeded-at-zero, 1600-step
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

impl BipedalWalkerConfig {
    /// Return a builder for configuring a `BipedalWalkerConfig`.
    pub fn builder() -> BipedalWalkerConfigBuilder {
        BipedalWalkerConfigBuilder {
            inner: BipedalWalkerConfig::default(),
        }
    }
}

/// Builder for [`BipedalWalkerConfig`].
#[derive(Debug, Clone)]
pub struct BipedalWalkerConfigBuilder {
    inner: BipedalWalkerConfig,
}

impl BipedalWalkerConfigBuilder {
    /// Sets the terrain difficulty variant.
    pub fn terrain(mut self, terrain: BipedalTerrain) -> Self {
        self.inner.terrain = terrain;
        self
    }

    /// Sets the RNG seed for terrain generation and initial state.
    pub fn seed(mut self, seed: u64) -> Self {
        self.inner.seed = seed;
        self
    }

    /// Sets the maximum steps per episode before truncation.
    pub fn max_steps(mut self, max_steps: usize) -> Self {
        self.inner.max_steps = max_steps;
        self
    }

    /// Sets the maximum torque applied by each leg motor.
    pub fn motors_torque(mut self, torque: f32) -> Self {
        self.inner.motors_torque = torque;
        self
    }

    /// Sets the lidar sensing range in world units.
    pub fn lidar_range(mut self, range: f32) -> Self {
        self.inner.lidar_range = range;
        self
    }

    /// Consumes the builder and returns the configured [`BipedalWalkerConfig`].
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if the assembled config fails [`Validate`]
    /// (e.g. non-positive `motors_torque`, `dt`, or `max_steps == 0`).
    pub fn build(self) -> Result<BipedalWalkerConfig, ConfigError> {
        self.inner.validate()?;
        Ok(self.inner)
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
    fn test_builder() {
        let cfg = BipedalWalkerConfig::builder()
            .terrain(BipedalTerrain::Hardcore)
            .seed(42)
            .max_steps(500)
            .build()
            .expect("valid config");
        assert_eq!(cfg.terrain, BipedalTerrain::Hardcore);
        assert_eq!(cfg.seed, 42);
        assert_eq!(cfg.max_steps, 500);
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
