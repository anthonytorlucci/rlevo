//! Configuration for [`super::Swimmer`].

use crate::locomotion::common::Gear;

/// Environment configuration for [`super::Swimmer`].
///
/// Defaults match the Gymnasium v5 swimmer XML: gear `[150, 150]`, dt 0.01,
/// frame_skip 4 (env dt = 0.04), reset noise 0.1, forward-reward weight 1.0,
/// ctrl-cost weight 1e-4, drag coefficient 0.1, truncation at 1000.
#[derive(Debug, Clone)]
pub struct SwimmerConfig {
    pub seed: u64,
    pub gear: Gear<2>,
    pub dt: f32,
    pub frame_skip: u32,
    pub reset_noise_scale: f32,
    pub max_steps: usize,
    pub action_clip: (f32, f32),
    pub forward_reward_weight: f32,
    pub ctrl_cost_weight: f32,
    /// Per-segment viscous linear-drag coefficient: `F = −k · v · ‖v‖`.
    pub drag_coefficient: f32,
    /// Per-segment viscous angular-drag coefficient: `τ = −k_ang · ω · |ω|`.
    /// Rapier-only addition to stand in for MuJoCo's fluid viscosity term,
    /// which otherwise lets the chain spin unboundedly under actuator
    /// torques in a zero-gravity free-floating setup.
    pub angular_drag_coefficient: f32,
    pub segment_length: f32,
    pub segment_radius: f32,
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
        // * `gear = [30, 30]` is one fifth of Gymnasium's `[150, 150]`.
        //   At full gear the angular acceleration (α ≈ τ/I ≈ 7 500 rad/s²)
        //   produces ~75 rad/s per substep at dt=0.01 — a rotation rate
        //   that violates the joint constraints faster than PGS can
        //   resolve them, and the multibody state diverges to NaN within
        //   a handful of env steps. The fifth-scale gear keeps dynamics
        //   inside the solver's stable regime.
        //
        // * `angular_drag_coefficient = 0.1` adds a quadratic angular-drag
        //   term `τ_drag = −k_ang · ω · |ω|` per segment. MuJoCo's swimmer
        //   effectively caps segment spin via fluid viscosity; with only
        //   linear drag, sustained actuation spins the chain up unboundedly
        //   (since no kinetic-energy sink exists about the joint axis) and
        //   the solver still diverges at gear=30 within ~50 steps.
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
