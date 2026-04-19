//! Shared primitives for MuJoCo-style locomotion environments.
//!
//! * [`ObservationComponents`] bundles the Gymnasium `include_*_in_observation`
//!   and `exclude_current_positions_from_observation` toggles into one struct.
//! * [`HealthyCheck`] and [`TerminationMode`] centralise the healthy-state
//!   termination logic (Hopper's torso z + angle, Walker2d's z + angle, Ant's
//!   z, ...).
//! * [`Gear`] is a type-level gear-ratio vector, size-checked against the
//!   per-env actuator count.
//! * [`LocomotionSnapshot`] is the shared snapshot type every locomotion env
//!   returns; it carries a [`SnapshotMetadata`] (reward components + positions)
//!   and is generic in the per-env observation type.
//! * Small helpers: [`ctrl_cost`] for the quadratic control cost that every
//!   locomotion env pays, and [`is_finite_state`] for the `np.isfinite(state).all()`
//!   gate Gymnasium uses.

use evorl_core::base::Observation;
use evorl_core::environment::{EpisodeStatus, Snapshot, SnapshotMetadata};
use evorl_core::reward::ScalarReward;

/// Which optional components are included in an environment's observation
/// vector.
///
/// Defaults capture the Gymnasium v5 defaults:
/// * Current-step `xy` position of the torso is excluded (`include_xy_position = false`),
///   because including it makes the env non-stationary w.r.t. the forward-reward goal.
/// * Ant includes the 78-dim `cfrc_ext`; Humanoid additionally includes
///   `cinert` / `cvel` / `qfrc_actuator`.
///
/// Envs whose Gymnasium counterpart doesn't expose a given toggle simply
/// ignore the corresponding field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObservationComponents {
    /// Include the torso's world-frame `(x, y)` in the observation.
    pub include_xy_position: bool,
    /// Include per-body centre-of-mass inertia (`cinert`, 10 per body).
    pub include_cinert: bool,
    /// Include per-body centre-of-mass velocity (`cvel`, 6 per body).
    pub include_cvel: bool,
    /// Include per-joint actuator torque (`qfrc_actuator`).
    pub include_qfrc_actuator: bool,
    /// Include per-body external contact force-torque (`cfrc_ext`, 6 per body).
    pub include_cfrc_ext: bool,
}

impl ObservationComponents {
    /// Nothing optional included.
    #[must_use]
    pub const fn minimal() -> Self {
        Self {
            include_xy_position: false,
            include_cinert: false,
            include_cvel: false,
            include_qfrc_actuator: false,
            include_cfrc_ext: false,
        }
    }

    /// Ant-v5 defaults: exclude xy, include cfrc_ext (no cinert / cvel / qfrc).
    #[must_use]
    pub const fn ant_default() -> Self {
        Self {
            include_xy_position: false,
            include_cinert: false,
            include_cvel: false,
            include_qfrc_actuator: false,
            include_cfrc_ext: true,
        }
    }

    /// Humanoid-v5 defaults: exclude xy, include all remaining components.
    #[must_use]
    pub const fn humanoid_default() -> Self {
        Self {
            include_xy_position: false,
            include_cinert: true,
            include_cvel: true,
            include_qfrc_actuator: true,
            include_cfrc_ext: true,
        }
    }
}

impl Default for ObservationComponents {
    fn default() -> Self {
        Self::minimal()
    }
}

/// Healthy-state check for envs with an "unhealthy → terminate" condition.
///
/// Each range is an `Option`; when `None`, that dimension is not checked.
/// Hopper sets all three (z + angle + state); Ant only sets `z_range`;
/// HalfCheetah / Swimmer / Reacher / Pusher / HumanoidStandup set none
/// (they never terminate on unhealthy).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HealthyCheck {
    /// Inclusive `(low, high)` range for the torso's world-frame z (height).
    pub z_range: Option<(f32, f32)>,
    /// Inclusive `(low, high)` range for the torso's pitch angle (radians).
    pub angle_range: Option<(f32, f32)>,
    /// Inclusive `(low, high)` range applied to every element of the packed
    /// state vector (qpos+qvel). Gymnasium uses `(-100, 100)` for Hopper.
    pub state_range: Option<(f32, f32)>,
}

impl HealthyCheck {
    /// Default: no constraints (env is always healthy).
    #[must_use]
    pub const fn none() -> Self {
        Self { z_range: None, angle_range: None, state_range: None }
    }

    /// Evaluate healthiness against the current torso height, pitch, and
    /// packed state. Any range that is `None` is skipped.
    #[must_use]
    pub fn is_healthy(self, torso_z: f32, torso_angle: f32, state: &[f32]) -> bool {
        if !torso_z.is_finite() || !torso_angle.is_finite() {
            return false;
        }
        if let Some((lo, hi)) = self.z_range
            && (torso_z < lo || torso_z > hi)
        {
            return false;
        }
        if let Some((lo, hi)) = self.angle_range
            && (torso_angle < lo || torso_angle > hi)
        {
            return false;
        }
        if let Some((lo, hi)) = self.state_range
            && !state.iter().all(|v| v.is_finite() && *v >= lo && *v <= hi)
        {
            return false;
        }
        true
    }
}

impl Default for HealthyCheck {
    fn default() -> Self {
        Self::none()
    }
}

/// How the env handles an unhealthy state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TerminationMode {
    /// Terminate the episode when [`HealthyCheck::is_healthy`] returns false.
    #[default]
    OnUnhealthy,
    /// Never terminate for health reasons (HalfCheetah, Swimmer, Reacher, ...).
    Never,
}

/// Gear ratios for an N-actuator env: `torque_i = action_i * gear_i`.
///
/// The const generic `N` pins the actuator count at the type level, so
/// swapping a 6-gear HalfCheetah vector for an 8-gear Ant vector is a compile
/// error rather than a runtime mismatch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gear<const N: usize>([f32; N]);

impl<const N: usize> Gear<N> {
    #[must_use]
    pub const fn new(values: [f32; N]) -> Self {
        Self(values)
    }

    #[must_use]
    pub const fn values(&self) -> &[f32; N] {
        &self.0
    }

    /// Multiply a normalised action by per-actuator gear; returns torques.
    #[must_use]
    pub fn apply(&self, action: &[f32; N]) -> [f32; N] {
        let mut torque = [0.0f32; N];
        for i in 0..N {
            torque[i] = action[i] * self.0[i];
        }
        torque
    }
}

/// Quadratic control cost `weight * ‖action‖²`.
#[must_use]
pub fn ctrl_cost<const N: usize>(weight: f32, action: &[f32; N]) -> f32 {
    let mut sum = 0.0f32;
    for a in action {
        sum += a * a;
    }
    weight * sum
}

/// Gymnasium's `is_finite(state).all()` guard — trips the unhealthy flag when
/// the simulator diverges.
#[must_use]
pub fn is_finite_state(state: &[f32]) -> bool {
    state.iter().all(|v| v.is_finite())
}

/// Clamp a contact-cost scalar into the env's allowed range, matching
/// Gymnasium's `np.clip(contact_cost, lo, hi)` pattern.
#[must_use]
pub fn clip_contact_cost(contact_cost: f32, range: (f32, f32)) -> f32 {
    contact_cost.clamp(range.0, range.1)
}

/// Normalise an angle (radians) to the half-open interval `(-π, π]`.
///
/// Used by any env that extracts a joint angle from a quaternion and needs a
/// unique branch cut (e.g. `InvertedPendulum`'s pole angle, `Reacher`'s
/// relative elbow angle).
#[must_use]
pub fn wrap_to_pi(angle: f32) -> f32 {
    let two_pi = std::f32::consts::TAU;
    let mut a = angle % two_pi;
    if a > std::f32::consts::PI {
        a -= two_pi;
    } else if a <= -std::f32::consts::PI {
        a += two_pi;
    }
    a
}

/// Shared snapshot type for every locomotion environment.
///
/// Generic in the per-env [`Observation`] so each env's snapshot has a
/// precise observation type at the type level, while the 11 envs share one
/// `impl Snapshot<1>` rather than duplicating it. Carries a
/// [`SnapshotMetadata`] whose `components` map records decomposed reward
/// contributions and whose `positions` map records torso / centre-of-mass /
/// fingertip coordinates.
#[derive(Debug, Clone)]
pub struct LocomotionSnapshot<O>
where
    O: Observation<1> + Clone,
{
    observation: O,
    reward: ScalarReward,
    status: EpisodeStatus,
    metadata: SnapshotMetadata,
}

impl<O> LocomotionSnapshot<O>
where
    O: Observation<1> + Clone,
{
    /// Build a snapshot directly from its parts.
    #[must_use]
    pub fn new(
        observation: O,
        reward: ScalarReward,
        status: EpisodeStatus,
        metadata: SnapshotMetadata,
    ) -> Self {
        Self { observation, reward, status, metadata }
    }

    /// Convenience constructor — running-episode snapshot.
    #[must_use]
    pub fn running(observation: O, reward: ScalarReward, metadata: SnapshotMetadata) -> Self {
        Self::new(observation, reward, EpisodeStatus::Running, metadata)
    }

    /// Convenience constructor — terminated-episode snapshot (MDP sink).
    #[must_use]
    pub fn terminated(observation: O, reward: ScalarReward, metadata: SnapshotMetadata) -> Self {
        Self::new(observation, reward, EpisodeStatus::Terminated, metadata)
    }

    /// Convenience constructor — truncated-episode snapshot (step-cap hit).
    #[must_use]
    pub fn truncated(observation: O, reward: ScalarReward, metadata: SnapshotMetadata) -> Self {
        Self::new(observation, reward, EpisodeStatus::Truncated, metadata)
    }
}

impl<O> Snapshot<1> for LocomotionSnapshot<O>
where
    O: Observation<1> + Clone,
{
    type ObservationType = O;
    type RewardType = ScalarReward;

    fn observation(&self) -> &O {
        &self.observation
    }

    fn reward(&self) -> &ScalarReward {
        &self.reward
    }

    fn status(&self) -> EpisodeStatus {
        self.status
    }

    fn metadata(&self) -> Option<&SnapshotMetadata> {
        Some(&self.metadata)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_components_ant_default_matches_spec() {
        let comps = ObservationComponents::ant_default();
        assert!(!comps.include_xy_position);
        assert!(comps.include_cfrc_ext);
        assert!(!comps.include_cinert);
    }

    #[test]
    fn observation_components_humanoid_default_includes_everything_but_xy() {
        let comps = ObservationComponents::humanoid_default();
        assert!(!comps.include_xy_position);
        assert!(comps.include_cinert);
        assert!(comps.include_cvel);
        assert!(comps.include_qfrc_actuator);
        assert!(comps.include_cfrc_ext);
    }

    #[test]
    fn healthy_check_none_is_always_healthy() {
        let check = HealthyCheck::none();
        assert!(check.is_healthy(1e9, 1e9, &[1e9]));
    }

    #[test]
    fn healthy_check_z_range_gates_height() {
        let check = HealthyCheck { z_range: Some((0.2, 1.0)), ..HealthyCheck::none() };
        assert!(check.is_healthy(0.5, 0.0, &[]));
        assert!(!check.is_healthy(0.1, 0.0, &[]));
        assert!(!check.is_healthy(1.5, 0.0, &[]));
    }

    #[test]
    fn healthy_check_hopper_style_all_three_ranges() {
        let check = HealthyCheck {
            z_range: Some((0.7, f32::INFINITY)),
            angle_range: Some((-0.2, 0.2)),
            state_range: Some((-100.0, 100.0)),
        };
        assert!(check.is_healthy(1.25, 0.0, &[1.0, -2.0, 3.0]));
        assert!(!check.is_healthy(0.5, 0.0, &[1.0]));          // z too low
        assert!(!check.is_healthy(1.25, 0.5, &[1.0]));         // angle too large
        assert!(!check.is_healthy(1.25, 0.0, &[150.0]));       // state out of range
        assert!(!check.is_healthy(f32::NAN, 0.0, &[]));
    }

    #[test]
    fn gear_apply_scales_quadratically_compatible() {
        let gear = Gear::<3>::new([2.0, 3.0, 4.0]);
        let torque = gear.apply(&[1.0, 2.0, -0.5]);
        assert_eq!(torque, [2.0, 6.0, -2.0]);
    }

    #[test]
    fn ctrl_cost_is_quadratic() {
        let a = [1.0, 2.0, 3.0];
        let c1 = ctrl_cost(0.5, &a);
        let a2 = [2.0, 4.0, 6.0];
        let c2 = ctrl_cost(0.5, &a2);
        assert!((c2 - 4.0 * c1).abs() < 1e-5, "ctrl_cost(2a) must equal 4·ctrl_cost(a); got {c1} vs {c2}");
    }

    #[test]
    fn is_finite_state_trips_on_nan_inf() {
        assert!(is_finite_state(&[0.0, 1.0, -1.0]));
        assert!(!is_finite_state(&[0.0, f32::NAN]));
        assert!(!is_finite_state(&[0.0, f32::INFINITY]));
    }

    #[test]
    fn clip_contact_cost_respects_range() {
        assert_eq!(clip_contact_cost(5.0, (0.0, 10.0)), 5.0);
        assert_eq!(clip_contact_cost(15.0, (0.0, 10.0)), 10.0);
        assert_eq!(clip_contact_cost(-1.0, (0.0, 10.0)), 0.0);
    }

    #[test]
    fn termination_mode_default_is_on_unhealthy() {
        assert_eq!(TerminationMode::default(), TerminationMode::OnUnhealthy);
    }

    #[test]
    fn wrap_to_pi_canonical_values() {
        use std::f32::consts::PI;
        assert!((wrap_to_pi(0.0) - 0.0).abs() < 1e-6);
        assert!((wrap_to_pi(PI) - PI).abs() < 1e-6); // π maps to itself (half-open)
        assert!((wrap_to_pi(-PI) - PI).abs() < 1e-6); // -π → π (excluded below)
        assert!((wrap_to_pi(3.0 * PI / 2.0) - (-PI / 2.0)).abs() < 1e-5);
        assert!((wrap_to_pi(-3.0 * PI / 2.0) - (PI / 2.0)).abs() < 1e-5);
        assert!((wrap_to_pi(4.0 * PI) - 0.0).abs() < 1e-5);
    }
}
