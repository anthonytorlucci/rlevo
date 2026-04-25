//! Configuration for LunarLander environments.
//!
//! [`LunarLanderConfig`] groups all tunable parameters shared by
//! [`super::LunarLanderDiscrete`] and [`super::LunarLanderContinuous`].
//! Use [`LunarLanderConfig::builder`] for ergonomic construction.

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
    pub fn build(self) -> LunarLanderConfig {
        self.inner
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
            .build();
        assert_eq!(cfg.seed, 7);
        assert_eq!(cfg.max_steps, 500);
        assert!(matches!(cfg.wind_mode, WindMode::Constant { .. }));
    }
}
