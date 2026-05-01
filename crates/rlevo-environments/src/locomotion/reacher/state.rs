//! Physics state for [`super::Reacher`].

use rapier3d::prelude::{ImpulseJointHandle, RigidBodyHandle};
use rlevo_core::base::State;

use super::observation::ReacherObservation;

/// Physics state for [`super::Reacher`] — body + joint handles plus the cached
/// target position and last observation. The non-`Clone` world lives on the
/// env struct directly.
#[derive(Debug, Clone)]
pub struct ReacherState {
    pub link1: RigidBodyHandle,
    pub link2: RigidBodyHandle,
    pub target: RigidBodyHandle,
    pub shoulder: ImpulseJointHandle,
    pub elbow: ImpulseJointHandle,
    pub target_xy: [f32; 2],
    pub last_obs: ReacherObservation,
}

impl State<1> for ReacherState {
    type Observation = ReacherObservation;

    fn shape() -> [usize; 1] {
        [10]
    }

    fn is_valid(&self) -> bool {
        self.last_obs.is_finite()
    }

    fn observe(&self) -> ReacherObservation {
        self.last_obs
    }
}
