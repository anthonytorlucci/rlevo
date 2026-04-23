//! State type for BipedalWalker.

use rapier2d::dynamics::{ImpulseJointHandle, RigidBodyHandle};
use rlevo_core::base::State;

use super::observation::BipedalWalkerObservation;

/// Physics state for BipedalWalker.
///
/// Stores rapier2d handles for all bodies and joints, plus cached
/// contact flags and the last computed observation.
#[derive(Debug, Clone)]
pub struct BipedalWalkerState {
    /// Hull (torso) rigid body.
    pub hull_handle: RigidBodyHandle,
    /// Upper leg 1 (thigh).
    pub leg1_upper_handle: RigidBodyHandle,
    /// Lower leg 1 (shin).
    pub leg1_lower_handle: RigidBodyHandle,
    /// Upper leg 2 (thigh).
    pub leg2_upper_handle: RigidBodyHandle,
    /// Lower leg 2 (shin).
    pub leg2_lower_handle: RigidBodyHandle,
    /// Hip 1 revolute joint (hull ↔ upper leg 1).
    pub hip1_joint: ImpulseJointHandle,
    /// Knee 1 revolute joint (upper leg 1 ↔ lower leg 1).
    pub knee1_joint: ImpulseJointHandle,
    /// Hip 2 revolute joint (hull ↔ upper leg 2).
    pub hip2_joint: ImpulseJointHandle,
    /// Knee 2 revolute joint (upper leg 2 ↔ lower leg 2).
    pub knee2_joint: ImpulseJointHandle,
    /// Whether leg 1 is in contact with the ground.
    pub leg1_contact: bool,
    /// Whether leg 2 is in contact with the ground.
    pub leg2_contact: bool,
    /// Cached observation from the last `step()` or `reset()`.
    pub last_obs: BipedalWalkerObservation,
}

impl State<1> for BipedalWalkerState {
    type Observation = BipedalWalkerObservation;

    fn shape() -> [usize; 1] {
        [24]
    }

    fn is_valid(&self) -> bool {
        self.last_obs.is_finite()
    }

    fn numel(&self) -> usize {
        24
    }

    fn observe(&self) -> BipedalWalkerObservation {
        self.last_obs.clone()
    }
}
