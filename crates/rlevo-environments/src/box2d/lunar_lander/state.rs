//! Physics state for the LunarLander environments.
//!
//! [`LunarLanderState`] holds Rapier2D handles for the lander hull and legs,
//! cached ground-contact flags, the most recent observation, and the previous
//! shaping value needed for potential-based reward computation.

use rapier2d::dynamics::RigidBodyHandle;
use rapier2d::geometry::ColliderHandle;
use rlevo_core::base::State;

use super::observation::LunarLanderObservation;

/// Physics state for LunarLander.
#[derive(Debug, Clone)]
pub struct LunarLanderState {
    /// Main lander body.
    pub lander_handle: RigidBodyHandle,
    /// Left landing leg body.
    pub leg1_handle: RigidBodyHandle,
    /// Right landing leg body.
    pub leg2_handle: RigidBodyHandle,
    /// Ground collider (helipad platform).
    pub ground_handle: ColliderHandle,
    /// Whether left leg is touching the ground.
    pub leg1_contact: bool,
    /// Whether right leg is touching the ground.
    pub leg2_contact: bool,
    /// Cached observation from the last step/reset.
    pub last_obs: LunarLanderObservation,
    /// Previous shaping value (for reward computation).
    pub prev_shaping: f32,
}

impl State<1> for LunarLanderState {
    type Observation = LunarLanderObservation;

    fn shape() -> [usize; 1] {
        [8]
    }

    fn is_valid(&self) -> bool {
        self.last_obs.is_finite()
    }

    fn numel(&self) -> usize {
        8
    }

    fn observe(&self) -> LunarLanderObservation {
        self.last_obs.clone()
    }
}
