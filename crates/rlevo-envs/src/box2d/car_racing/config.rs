//! Configuration for the CarRacing environment.

use serde::{Deserialize, Serialize};

/// Configuration for [`super::CarRacing`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CarRacingConfig {
    /// Number of Bezier control points for track generation.
    pub track_n_checkpoints: usize,
    /// Track turn rate (curvature control).
    pub track_turn_rate: f32,
    /// Half-width of the road surface (world units).
    pub track_width: f32,
    /// Fraction of track tiles that must be visited to complete a lap.
    pub lap_complete_percent: f32,
    /// Reward per new tile visited. Total lap reward ≈ 1000.
    pub tile_reward: f32,
    /// Penalty applied every step regardless of tile visits.
    pub frame_penalty: f32,
    /// Car body density.
    pub car_density: f32,
    /// Road friction coefficient.
    pub friction: f32,
    /// RNG seed for track generation.
    pub seed: u64,
    /// Maximum steps per episode (Truncated after this).
    pub max_steps: usize,
    /// Physics timestep (seconds).
    pub dt: f32,
    /// Gravity (negative = downward, but CarRacing uses top-down view).
    pub gravity: f32,
}

impl Default for CarRacingConfig {
    fn default() -> Self {
        Self {
            track_n_checkpoints: 12,
            track_turn_rate: 0.31,
            track_width: 40.0,
            lap_complete_percent: 0.95,
            tile_reward: 1000.0 / 200.0, // 200 tiles approximated
            frame_penalty: -0.1,
            car_density: 0.001,
            friction: 1.0,
            seed: 0,
            max_steps: 1000,
            dt: 1.0 / 50.0,
            gravity: 0.0, // top-down view: no vertical gravity
        }
    }
}

impl CarRacingConfig {
    pub fn builder() -> CarRacingConfigBuilder {
        CarRacingConfigBuilder { inner: CarRacingConfig::default() }
    }
}

/// Builder for [`CarRacingConfig`].
#[derive(Debug, Clone)]
pub struct CarRacingConfigBuilder {
    inner: CarRacingConfig,
}

impl CarRacingConfigBuilder {
    pub fn seed(mut self, seed: u64) -> Self {
        self.inner.seed = seed;
        self
    }

    pub fn track_n_checkpoints(mut self, n: usize) -> Self {
        self.inner.track_n_checkpoints = n;
        self
    }

    pub fn max_steps(mut self, n: usize) -> Self {
        self.inner.max_steps = n;
        self
    }

    pub fn build(self) -> CarRacingConfig {
        self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let cfg = CarRacingConfig::default();
        assert!(cfg.tile_reward > 0.0);
        assert!(cfg.frame_penalty < 0.0);
        assert_eq!(cfg.track_n_checkpoints, 12);
    }
}
