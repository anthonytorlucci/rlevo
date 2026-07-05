//! Configuration for the [`super::Reacher`] environment.
//!
//! [`ReacherConfig`] collects every tunable parameter in one place. The
//! [`Default`] impl reproduces the Gymnasium v5 reacher XML values; override
//! individual fields when constructing via [`super::Reacher::with_config`].

use rlevo_core::bounds::Bounds;
use rlevo_core::config::{self, ConfigError, Validate};

use crate::locomotion::common::Gear;

/// Environment configuration for [`super::Reacher`].
///
/// Defaults match the Gymnasium v5 reacher XML: gear `[200, 200]`, dt 0.01,
/// frame_skip 2 (env dt = 0.02), reset noise 0.1, ctrl-cost weight 0.1,
/// target-disk radius 0.2, truncation at 50.
#[derive(Debug, Clone)]
pub struct ReacherConfig {
    /// Seed for the internal `StdRng` used during resets. The same seed
    /// always produces the same sequence of initial states.
    pub seed: u64,
    /// Per-joint gear ratios applied to the clipped action before the
    /// resulting torques are passed to the rigid bodies.
    /// Default: `[200.0, 200.0]` (shoulder, elbow).
    pub gear: Gear<2>,
    /// Physics simulation timestep in seconds. Default: `0.01`.
    pub dt: f32,
    /// Number of physics sub-steps per `Environment::step` call. The
    /// effective environment timestep is `dt * frame_skip`. Default: `2`
    /// (env dt = 0.02 s).
    pub frame_skip: u32,
    /// Half-width of the uniform noise added to each joint angle and joint
    /// velocity at reset. Default: `0.1` rad / (rad s⁻¹).
    pub reset_noise_scale: f32,
    /// Episode length limit. A step that reaches this count returns
    /// `EpisodeStatus::Truncated`. Default: `50`.
    pub max_steps: usize,
    /// Inclusive bounds applied to each action element before gear
    /// multiplication. Default: `[-1.0, 1.0]`.
    pub action_clip: Bounds,
    /// Weight applied to the squared control norm in the reward:
    /// `reward_control = -ctrl_cost_weight · ‖action‖²`. Default: `0.1`.
    pub ctrl_cost_weight: f32,
    /// Full length of link 1 (shoulder to elbow) in metres. Default: `0.10`.
    pub link1_length: f32,
    /// Full length of link 2 (elbow to fingertip) in metres. Default: `0.11`.
    pub link2_length: f32,
    /// Capsule radius for both links in metres. Default: `0.01`.
    pub link_radius: f32,
    /// Mass of each link in kilograms. Default: `0.0356`.
    pub link_mass: f32,
    /// Radius of the disk from which the target position is uniformly
    /// sampled at each reset. Default: `0.2` m.
    pub target_disk_radius: f32,
}

impl Default for ReacherConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            gear: Gear::new([200.0, 200.0]),
            dt: 0.01,
            frame_skip: 2,
            reset_noise_scale: 0.1,
            max_steps: 50,
            action_clip: Bounds::new(-1.0, 1.0),
            ctrl_cost_weight: 0.1,
            link1_length: 0.10,
            link2_length: 0.11,
            link_radius: 0.01,
            link_mass: 0.0356,
            target_disk_radius: 0.2,
        }
    }
}

impl Validate for ReacherConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "ReacherConfig";
        config::positive(C, "dt", f64::from(self.dt))?;
        config::nonzero(C, "frame_skip", self.frame_skip as usize)?;
        config::in_range(C, "reset_noise_scale", 0.0, f64::INFINITY, f64::from(self.reset_noise_scale))?;
        config::nonzero(C, "max_steps", self.max_steps)?;
        config::in_range(C, "ctrl_cost_weight", 0.0, f64::INFINITY, f64::from(self.ctrl_cost_weight))?;
        config::positive(C, "link1_length", f64::from(self.link1_length))?;
        config::positive(C, "link2_length", f64::from(self.link2_length))?;
        config::positive(C, "link_radius", f64::from(self.link_radius))?;
        config::positive(C, "link_mass", f64::from(self.link_mass))?;
        config::positive(C, "target_disk_radius", f64::from(self.target_disk_radius))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_validates() {
        assert!(ReacherConfig::default().validate().is_ok());
    }

    #[test]
    fn rejects_non_positive_link1_length() {
        let bad = ReacherConfig { link1_length: 0.0, ..Default::default() };
        assert_eq!(bad.validate().unwrap_err().field, "link1_length");
    }
}
