//! Configuration for [`super::Reacher`].

use crate::locomotion::common::Gear;

/// Environment configuration for [`super::Reacher`].
///
/// Defaults match the Gymnasium v5 reacher XML: gear `[200, 200]`, dt 0.01,
/// frame_skip 2 (env dt = 0.02), reset noise 0.1, ctrl-cost weight 0.1,
/// target-disk radius 0.2, truncation at 50.
#[derive(Debug, Clone)]
pub struct ReacherConfig {
    pub seed: u64,
    pub gear: Gear<2>,
    pub dt: f32,
    pub frame_skip: u32,
    pub reset_noise_scale: f32,
    pub max_steps: usize,
    pub action_clip: (f32, f32),
    pub ctrl_cost_weight: f32,
    pub link1_length: f32,
    pub link2_length: f32,
    pub link_radius: f32,
    pub link_mass: f32,
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
            action_clip: (-1.0, 1.0),
            ctrl_cost_weight: 0.1,
            link1_length: 0.10,
            link2_length: 0.11,
            link_radius: 0.01,
            link_mass: 0.0356,
            target_disk_radius: 0.2,
        }
    }
}
