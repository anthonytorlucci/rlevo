//! Physics state for [`super::InvertedDoublePendulum`].

use rapier3d::prelude::{ImpulseJointHandle, RigidBodyHandle};
use rlevo_core::base::State;

use super::observation::InvertedDoublePendulumObservation;

/// Physics state: body/joint handles plus the last observation. The world
/// itself (non-`Clone`) lives on [`super::InvertedDoublePendulum`].
#[derive(Debug, Clone)]
pub struct InvertedDoublePendulumState {
    pub cart: RigidBodyHandle,
    pub pole1: RigidBodyHandle,
    pub pole2: RigidBodyHandle,
    pub joint1: ImpulseJointHandle,
    pub joint2: ImpulseJointHandle,
    pub last_obs: InvertedDoublePendulumObservation,
}

impl State<1> for InvertedDoublePendulumState {
    type Observation = InvertedDoublePendulumObservation;

    fn shape() -> [usize; 1] {
        [9]
    }

    fn is_valid(&self) -> bool {
        self.last_obs.is_finite()
    }

    fn observe(&self) -> InvertedDoublePendulumObservation {
        self.last_obs
    }
}
