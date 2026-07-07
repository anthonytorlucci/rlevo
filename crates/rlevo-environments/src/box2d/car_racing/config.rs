//! Configuration for the CarRacing environment.
//!
//! [`CarRacingConfig`] groups all tunable parameters for track generation,
//! physics, and reward shaping. Use [`CarRacingConfig::builder`] for ergonomic
//! construction; call [`Default::default`] for a standard 1000-step episode.

use rlevo_core::config::{self, ConfigError, Validate};
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

impl Validate for CarRacingConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "CarRacingConfig";
        config::nonzero(C, "track_n_checkpoints", self.track_n_checkpoints)?;
        config::positive(C, "track_width", f64::from(self.track_width))?;
        config::positive(
            C,
            "lap_complete_percent",
            f64::from(self.lap_complete_percent),
        )?;
        config::in_range(
            C,
            "lap_complete_percent",
            0.0,
            1.0,
            f64::from(self.lap_complete_percent),
        )?;
        config::positive(C, "tile_reward", f64::from(self.tile_reward))?;
        config::ordered(C, "frame_penalty", f64::from(self.frame_penalty), 0.0)?;
        config::positive(C, "car_density", f64::from(self.car_density))?;
        config::in_range(C, "friction", 0.0, f64::INFINITY, f64::from(self.friction))?;
        config::nonzero(C, "max_steps", self.max_steps)?;
        config::positive(C, "dt", f64::from(self.dt))?;
        Ok(())
    }
}

impl CarRacingConfig {
    /// Returns a builder for configuring a `CarRacingConfig`.
    pub fn builder() -> CarRacingConfigBuilder {
        CarRacingConfigBuilder {
            inner: CarRacingConfig::default(),
        }
    }
}

/// Builder for [`CarRacingConfig`].
#[derive(Debug, Clone)]
pub struct CarRacingConfigBuilder {
    inner: CarRacingConfig,
}

impl CarRacingConfigBuilder {
    /// Sets the RNG seed for procedural track generation.
    pub fn seed(mut self, seed: u64) -> Self {
        self.inner.seed = seed;
        self
    }

    /// Sets the number of Bezier control points (more = more complex track).
    pub fn track_n_checkpoints(mut self, n: usize) -> Self {
        self.inner.track_n_checkpoints = n;
        self
    }

    /// Sets the maximum steps per episode before truncation.
    pub fn max_steps(mut self, n: usize) -> Self {
        self.inner.max_steps = n;
        self
    }

    /// Consumes the builder and returns the configured [`CarRacingConfig`].
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if the assembled config fails [`Validate`]
    /// (e.g. non-positive `track_width`, `lap_complete_percent` outside
    /// `(0, 1]`, or a non-negative `frame_penalty`).
    pub fn build(self) -> Result<CarRacingConfig, ConfigError> {
        self.inner.validate()?;
        Ok(self.inner)
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

    #[test]
    fn default_config_validates() {
        assert!(CarRacingConfig::default().validate().is_ok());
    }

    #[test]
    fn rejects_non_positive_track_width() {
        let bad = CarRacingConfig {
            track_width: 0.0,
            ..Default::default()
        };
        assert_eq!(bad.validate().unwrap_err().field, "track_width");
    }
}
