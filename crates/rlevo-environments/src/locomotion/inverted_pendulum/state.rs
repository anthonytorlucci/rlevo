//! Physics state for [`super::InvertedPendulum`].

use rapier3d::prelude::{ImpulseJointHandle, RigidBodyHandle};
use rlevo_core::base::State;

use super::observation::InvertedPendulumObservation;

/// Physics state for [`super::InvertedPendulum`] — keeps body handles and the
/// last observation, not the (non-`Clone`) world itself.
#[derive(Debug, Clone)]
pub struct InvertedPendulumState {
    pub cart: RigidBodyHandle,
    pub pole: RigidBodyHandle,
    pub joint: ImpulseJointHandle,
    pub last_obs: InvertedPendulumObservation,
}

impl State<1> for InvertedPendulumState {
    type Observation = InvertedPendulumObservation;

    fn shape() -> [usize; 1] {
        [4]
    }

    fn is_valid(&self) -> bool {
        self.last_obs.is_finite()
    }

    fn observe(&self) -> InvertedPendulumObservation {
        self.last_obs
    }
}
