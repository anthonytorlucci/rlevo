//! Physics state for [`super::InvertedPendulum`].

use rapier3d::prelude::{ImpulseJointHandle, RigidBodyHandle};
use rlevo_core::base::State;

use super::observation::InvertedPendulumObservation;

/// Physics state for [`super::InvertedPendulum`] — keeps body handles and the
/// last observation, not the (non-`Clone`) world itself.
///
/// Handles are stable identifiers into the Rapier world; they do not carry
/// physical data on their own. All kinematic queries go through
/// `Rapier3DBackend::get_pose` / `get_vel` using these handles.
#[derive(Debug, Clone)]
pub struct InvertedPendulumState {
    /// Rapier rigid-body handle for the cart (dynamic, x-translation only).
    pub cart: RigidBodyHandle,
    /// Rapier rigid-body handle for the pole (dynamic, y-rotation only).
    pub pole: RigidBodyHandle,
    /// Handle for the revolute impulse joint connecting the pole to the cart
    /// about the world-y axis.
    pub joint: ImpulseJointHandle,
    /// Most recently extracted observation; updated after every `step()` and
    /// after `reset()`. Used by [`State::observe`] without an extra world query.
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
