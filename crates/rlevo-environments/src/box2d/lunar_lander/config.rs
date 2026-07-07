//! Configuration for LunarLander environments.
//!
//! [`LunarLanderConfig`] groups all tunable parameters shared by
//! [`super::LunarLanderDiscrete`] and [`super::LunarLanderContinuous`].
//! Use [`LunarLanderConfig::builder`] for ergonomic construction.

use rlevo_core::config::{self, ConfigError, Validate};
use serde::{Deserialize, Serialize};

/// Wind model (design decision D2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WindMode {
    /// No wind.
    Off,
    /// Constant lateral wind force applied to the lander body.
    Constant {
        /// Lateral wind force (world units). Positive = rightward.
        force: f32,
    },
    /// Stochastic wind whose direction and magnitude vary each step.
    Stochastic {
        /// Seed for the wind RNG (independent of the env seed).
        seed: u64,
        /// Maximum wind force magnitude.
        max_force: f32,
    },
}

/// Configuration for [`super::LunarLanderDiscrete`] and
/// [`super::LunarLanderContinuous`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LunarLanderConfig {
    /// Wind model applied each step.
    pub wind_mode: WindMode,
    /// Gravitational acceleration (negative = downward, default −10.0).
    pub gravity: f32,
    /// Main engine thrust (Newtons, default 13.0).
    pub main_engine_power: f32,
    /// Side engine thrust (Newtons, default 0.6).
    pub side_engine_power: f32,
    /// Random velocity impulse applied at reset (default 1.0).
    pub initial_random: f32,
    /// RNG seed for terrain and reset state.
    pub seed: u64,
    /// Maximum steps before truncation (default 1000).
    pub max_steps: usize,
    /// Physics timestep (seconds, default 1/50).
    pub dt: f32,
    /// Lander body density (default 5.0).
    pub lander_density: f32,
}

impl Default for LunarLanderConfig {
    fn default() -> Self {
        Self {
            wind_mode: WindMode::Off,
            gravity: -10.0,
            main_engine_power: 13.0,
            side_engine_power: 0.6,
            initial_random: 1.0,
            seed: 0,
            max_steps: 1000,
            dt: 1.0 / 50.0,
            lander_density: 5.0,
        }
    }
}

impl Validate for LunarLanderConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "LunarLanderConfig";
        config::positive(C, "main_engine_power", f64::from(self.main_engine_power))?;
        config::positive(C, "side_engine_power", f64::from(self.side_engine_power))?;
        config::in_range(
            C,
            "initial_random",
            0.0,
            f64::INFINITY,
            f64::from(self.initial_random),
        )?;
        config::nonzero(C, "max_steps", self.max_steps)?;
        config::positive(C, "dt", f64::from(self.dt))?;
        config::positive(C, "lander_density", f64::from(self.lander_density))?;
        if let WindMode::Stochastic { max_force, .. } = self.wind_mode {
            config::in_range(
                C,
                "wind_mode.max_force",
                0.0,
                f64::INFINITY,
                f64::from(max_force),
            )?;
        }
        Ok(())
    }
}

impl LunarLanderConfig {
    /// Return a builder for `LunarLanderConfig`.
    pub fn builder() -> LunarLanderConfigBuilder {
        LunarLanderConfigBuilder {
            inner: LunarLanderConfig::default(),
        }
    }
}

/// Builder for [`LunarLanderConfig`].
#[derive(Debug, Clone)]
pub struct LunarLanderConfigBuilder {
    inner: LunarLanderConfig,
}

impl LunarLanderConfigBuilder {
    /// Sets the wind model applied each physics step.
    pub fn wind_mode(mut self, mode: WindMode) -> Self {
        self.inner.wind_mode = mode;
        self
    }

    /// Sets gravitational acceleration (negative = downward).
    pub fn gravity(mut self, g: f32) -> Self {
        self.inner.gravity = g;
        self
    }

    /// Sets the RNG seed for terrain layout and initial velocity.
    pub fn seed(mut self, seed: u64) -> Self {
        self.inner.seed = seed;
        self
    }

    /// Sets the maximum steps per episode before truncation.
    pub fn max_steps(mut self, n: usize) -> Self {
        self.inner.max_steps = n;
        self
    }

    /// Sets the main engine thrust in Newtons.
    pub fn main_engine_power(mut self, p: f32) -> Self {
        self.inner.main_engine_power = p;
        self
    }

    /// Consumes the builder and returns the configured [`LunarLanderConfig`].
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if the assembled config fails [`Validate`]
    /// (e.g. non-positive `main_engine_power`, `dt`, or `max_steps == 0`).
    pub fn build(self) -> Result<LunarLanderConfig, ConfigError> {
        self.inner.validate()?;
        Ok(self.inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = LunarLanderConfig::default();
        assert_eq!(cfg.max_steps, 1000);
        assert!(matches!(cfg.wind_mode, WindMode::Off));
    }

    #[test]
    fn test_builder() {
        let cfg = LunarLanderConfig::builder()
            .seed(7)
            .wind_mode(WindMode::Constant { force: 2.5 })
            .max_steps(500)
            .build()
            .expect("valid config");
        assert_eq!(cfg.seed, 7);
        assert_eq!(cfg.max_steps, 500);
        assert!(matches!(cfg.wind_mode, WindMode::Constant { .. }));
    }

    #[test]
    fn default_config_validates() {
        assert!(LunarLanderConfig::default().validate().is_ok());
    }

    #[test]
    fn rejects_non_positive_main_engine_power() {
        let bad = LunarLanderConfig {
            main_engine_power: 0.0,
            ..Default::default()
        };
        assert_eq!(bad.validate().unwrap_err().field, "main_engine_power");
    }
}
