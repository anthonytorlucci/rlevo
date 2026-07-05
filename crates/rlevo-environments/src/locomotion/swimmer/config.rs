//! Configuration for [`super::Swimmer`].

use rlevo_core::config::{self, ConfigError, Validate};

use crate::locomotion::common::Gear;

/// Environment configuration for [`super::Swimmer`].
///
/// The Rapier3D-backed default diverges from the Gymnasium v5 XML in three
/// places required for solver stability (see the module-level note and the
/// inline comment in [`Default::default`] for reasoning):
///
/// | Parameter | Gymnasium v5 | This default |
/// |---|---|---|
/// | `gear` | `[150, 150]` | `[5, 5]` |
/// | `dt` | `0.01` | `0.005` |
/// | `segment_mass` | ~0.0471 kg | `0.947 kg` |
///
/// All other parameters (`frame_skip = 8`, env-dt = 0.04, `reset_noise_scale
/// = 0.1`, `forward_reward_weight = 1.0`, `ctrl_cost_weight = 1e-4`,
/// `max_steps = 1000`) match Gymnasium v5.
#[derive(Debug, Clone)]
pub struct SwimmerConfig {
    /// Seed for the episode-level RNG used to sample reset noise.
    pub seed: u64,
    /// Per-joint gear multipliers applied to the `[-1, 1]` action before
    /// torque is added to the physics body. Default `[5, 5]`.
    pub gear: Gear<2>,
    /// Physics substep size in seconds. Default `0.005`.
    pub dt: f32,
    /// Number of physics substeps per `Environment::step` call. Effective
    /// env dt = `dt × frame_skip`. Default `8` → env dt `0.04 s`.
    pub frame_skip: u32,
    /// Half-width of the uniform reset-noise distribution applied to all
    /// generalised positions and velocities at `reset`. Default `0.1`.
    pub reset_noise_scale: f32,
    /// Episode length after which `EpisodeStatus::Truncated` is returned.
    /// Default `1000`.
    pub max_steps: usize,
    /// `(min, max)` bounds applied to each action element before gear
    /// multiplication and ctrl-cost computation. Default `(-1.0, 1.0)`.
    pub action_clip: (f32, f32),
    /// Scale factor on the forward-velocity reward component. Default `1.0`.
    pub forward_reward_weight: f32,
    /// Scale factor on the quadratic control-cost penalty. Default `1e-4`.
    pub ctrl_cost_weight: f32,
    /// Per-segment viscous linear-drag coefficient: `F = −k · v · ‖v‖`.
    /// Default `0.1`.
    pub drag_coefficient: f32,
    /// Per-segment viscous angular-drag coefficient: `τ = −k_ang · ω`.
    ///
    /// The term is **linear** in `ω` (not quadratic) because explicit-Euler
    /// integration of quadratic drag diverges at the angular velocities
    /// reachable under sustained actuation in a zero-gravity free-floating
    /// chain. MuJoCo uses an implicit integrator and can afford quadratic
    /// angular drag; this is a Rapier-compatibility divergence. Default `0.2`.
    pub angular_drag_coefficient: f32,
    /// Half-length of each capsule segment along the body-x axis. Default
    /// `0.1`.
    pub segment_length: f32,
    /// Radius of each capsule segment. Default `0.05`.
    pub segment_radius: f32,
    /// Mass of each segment in kg. Used to compute capsule density so that
    /// the inertia tensor is non-zero. Default `0.947` (MuJoCo body density
    /// 1000 kg/m³ × capsule volume).
    pub segment_mass: f32,
}

impl Default for SwimmerConfig {
    fn default() -> Self {
        // Three knobs diverge from the Gymnasium XML so Rapier's
        // reduced-coordinate multibody solver stays integrable:
        //
        // * `segment_mass = 0.947` uses MuJoCo's default *body* density
        //   (1000 kg/m³) applied to the capsule volume π·r²·(2·half + (4/3)·r).
        //   A figure of 0.0471 kg would cross body density with fluid density
        //   (`<option density>`); the XML uses body density 1000.
        //
        // * `gear = [5, 5]` is one thirtieth of Gymnasium's `[150, 150]`.
        //   At full gear the angular acceleration (α ≈ τ/I ≈ 7 500 rad/s²)
        //   produces ~75 rad/s per substep at dt=0.01 — a rotation rate
        //   that violates the joint constraints faster than PGS can
        //   resolve them, and the multibody state diverges to NaN within
        //   a handful of env steps. The reduced gear keeps dynamics
        //   inside the solver's stable regime.
        //
        // * `angular_drag_coefficient = 0.2` adds a linear angular-drag
        //   term `τ_drag = −k_ang · ω` per segment. MuJoCo's swimmer
        //   effectively caps segment spin via fluid viscosity; with only
        //   linear translational drag, sustained actuation spins the chain
        //   up unboundedly (no kinetic-energy sink about the joint axis)
        //   and the solver diverges even at reduced gear within ~50 steps.
        //   The term is linear in ω (not quadratic) for explicit-Euler
        //   stability; see `angular_drag_coefficient` field doc.
        //
        // The module-level "absolute reward values will NOT transfer"
        // disclaimer covers these divergences.
        Self {
            seed: 0,
            gear: Gear::new([5.0, 5.0]),
            dt: 0.005,
            frame_skip: 8,
            reset_noise_scale: 0.1,
            max_steps: 1000,
            action_clip: (-1.0, 1.0),
            forward_reward_weight: 1.0,
            ctrl_cost_weight: 1e-4,
            drag_coefficient: 0.1,
            angular_drag_coefficient: 0.2,
            segment_length: 0.1,
            segment_radius: 0.05,
            segment_mass: 0.947,
        }
    }
}

impl Validate for SwimmerConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        const C: &str = "SwimmerConfig";
        config::positive(C, "dt", f64::from(self.dt))?;
        config::nonzero(C, "frame_skip", self.frame_skip as usize)?;
        config::in_range(C, "reset_noise_scale", 0.0, f64::INFINITY, f64::from(self.reset_noise_scale))?;
        config::nonzero(C, "max_steps", self.max_steps)?;
        config::ordered(C, "action_clip", f64::from(self.action_clip.0), f64::from(self.action_clip.1))?;
        config::in_range(C, "ctrl_cost_weight", 0.0, f64::INFINITY, f64::from(self.ctrl_cost_weight))?;
        config::in_range(C, "drag_coefficient", 0.0, f64::INFINITY, f64::from(self.drag_coefficient))?;
        config::in_range(C, "angular_drag_coefficient", 0.0, f64::INFINITY, f64::from(self.angular_drag_coefficient))?;
        config::positive(C, "segment_length", f64::from(self.segment_length))?;
        config::positive(C, "segment_radius", f64::from(self.segment_radius))?;
        config::positive(C, "segment_mass", f64::from(self.segment_mass))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_validates() {
        assert!(SwimmerConfig::default().validate().is_ok());
    }

    #[test]
    fn rejects_non_positive_segment_mass() {
        let bad = SwimmerConfig { segment_mass: 0.0, ..Default::default() };
        assert_eq!(bad.validate().unwrap_err().field, "segment_mass");
    }
}
