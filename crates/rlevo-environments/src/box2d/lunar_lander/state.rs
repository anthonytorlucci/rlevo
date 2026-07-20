//! Physics state for the `LunarLander` environments.
//!
//! [`LunarLanderState`] holds `Rapier2D` handles for the lander hull and legs,
//! cached ground-contact flags, and the previous shaping value needed for
//! potential-based reward computation.

use rapier2d::dynamics::RigidBodyHandle;
use rapier2d::geometry::ColliderHandle;
use rlevo_core::base::State;

/// Physics state for `LunarLander`.
///
/// # Handle-lifetime caveat
///
/// The four handle fields (`lander_handle`, `leg1_handle`, `leg2_handle`,
/// `ground_handle`) are *arena indices* into the [`RapierWorld`] owned by the
/// environment, not values that capture the physics at time *t*. A [`Clone`] is
/// therefore a **non-portable view**: the handles are only meaningful alongside
/// the specific world they were taken from, and dangle after a
/// `reset()`/rebuild swaps that world out. The Markov degrees of freedom (pose,
/// twist) live *behind* the handles, read live out of the world each step —
/// they are not values in this struct.
///
/// This aliasing is resolved by issue #256 (ADR 0039), which moves the handles
/// onto the env core and makes the state own its DOFs as values.
///
/// [`RapierWorld`]: crate::box2d::physics::RapierWorld
#[derive(Debug, Clone)]
pub struct LunarLanderState {
    /// Main lander body.
    pub(crate) lander_handle: RigidBodyHandle,
    /// Left landing leg body.
    pub(crate) leg1_handle: RigidBodyHandle,
    /// Right landing leg body.
    pub(crate) leg2_handle: RigidBodyHandle,
    /// Ground collider (helipad platform).
    pub(crate) ground_handle: ColliderHandle,
    /// Whether left leg is touching the ground.
    pub(crate) leg1_contact: bool,
    /// Whether right leg is touching the ground.
    pub(crate) leg2_contact: bool,
    /// Previous shaping value (for reward computation).
    pub(crate) prev_shaping: f32,
}

impl LunarLanderState {
    /// Handle for the main lander body.
    #[must_use]
    pub fn lander_handle(&self) -> RigidBodyHandle {
        self.lander_handle
    }

    /// Handle for the left landing leg body.
    #[must_use]
    pub fn leg1_handle(&self) -> RigidBodyHandle {
        self.leg1_handle
    }

    /// Handle for the right landing leg body.
    #[must_use]
    pub fn leg2_handle(&self) -> RigidBodyHandle {
        self.leg2_handle
    }

    /// Handle for the ground collider (helipad platform).
    #[must_use]
    pub fn ground_handle(&self) -> ColliderHandle {
        self.ground_handle
    }

    /// Whether the left leg is touching the ground.
    #[must_use]
    pub fn leg1_contact(&self) -> bool {
        self.leg1_contact
    }

    /// Whether the right leg is touching the ground.
    #[must_use]
    pub fn leg2_contact(&self) -> bool {
        self.leg2_contact
    }

    /// Previous shaping value used for potential-based reward computation.
    #[must_use]
    pub fn prev_shaping(&self) -> f32 {
        self.prev_shaping
    }
}

impl State<1> for LunarLanderState {
    fn shape() -> [usize; 1] {
        [8]
    }

    fn is_valid(&self) -> bool {
        self.lander_handle != RigidBodyHandle::invalid()
            && self.leg1_handle != RigidBodyHandle::invalid()
            && self.leg2_handle != RigidBodyHandle::invalid()
            && self.ground_handle != ColliderHandle::invalid()
            && self.prev_shaping.is_finite()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::box2d::lunar_lander::env::LunarLanderContinuous;
    use rlevo_core::environment::{ConstructableEnv, Environment};

    /// A freshly-reset env has a fully-assembled, finite state.
    #[test]
    fn is_valid_true_after_reset() {
        let mut env = LunarLanderContinuous::new(false);
        env.reset().expect("reset must succeed");
        assert!(
            env.core_state().is_valid(),
            "reset state must satisfy is_valid()"
        );
    }

    /// An invalidated handle must fail the invariant.
    #[test]
    fn is_valid_false_on_invalid_handle() {
        let mut env = LunarLanderContinuous::new(false);
        env.reset().expect("reset must succeed");
        let state = env.core_state_mut();
        state.lander_handle = RigidBodyHandle::invalid();
        assert!(
            !state.is_valid(),
            "invalid lander handle must fail is_valid()"
        );
    }

    /// A NaN shaping potential (which feeds the reward) must fail the invariant.
    #[test]
    fn is_valid_false_on_nan_prev_shaping() {
        let mut env = LunarLanderContinuous::new(false);
        env.reset().expect("reset must succeed");
        let state = env.core_state_mut();
        state.prev_shaping = f32::NAN;
        assert!(!state.is_valid(), "NaN prev_shaping must fail is_valid()");
    }
}
