//! Configuration for the BipedalWalker environment.
//!
//! [`BipedalWalkerConfig`] groups all tunable parameters. Use
//! [`BipedalWalkerConfig::builder`] for ergonomic construction; call
//! [`Default::default`] for a flat-terrain, 1600-step episode.

use serde::{Deserialize, Serialize};

/// Terrain difficulty variants (design decision D3).
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
    pub fn build(self) -> BipedalWalkerConfig {
        self.inner
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
            .build();
        assert_eq!(cfg.terrain, BipedalTerrain::Hardcore);
        assert_eq!(cfg.seed, 42);
        assert_eq!(cfg.max_steps, 500);
    }
}
