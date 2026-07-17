//! Physics state for the BipedalWalker environment.
//!
//! [`BipedalWalkerState`] holds all Rapier2D handles required to read body
//! kinematics and drive joints each step, plus cached contact flags.

use rapier2d::dynamics::{ImpulseJointHandle, RigidBodyHandle};
use rlevo_core::base::State;

/// Physics state for BipedalWalker.
///
/// Stores rapier2d handles for all bodies and joints, plus cached
/// contact flags.
///
/// # Handle-lifetime caveat
///
/// Every handle field is an *arena index* into the [`RapierWorld`] owned by the
/// enclosing `BipedalWalker` environment — not an owned value capturing state at
/// time *t*. The Markov degrees of freedom (hull pose + twist, per-joint
/// angle/speed, lidar rays) live *behind* these handles in that world; the
/// observation is produced on demand by the env-side
/// [`Sensor`](rlevo_core::environment::Sensor) reading the world (raycasting the
/// lidar against the live physics geometry), not cached on this struct.
///
/// Consequently a [`Clone`] of this state is a **non-portable view**: the cloned
/// handles are only meaningful alongside the exact world they were taken from and
/// **dangle** once the environment calls `reset()` and rebuilds that world. This
/// is a deliberate, contained state model for Scope A (encapsulation +
/// [`is_valid()`](State::is_valid)); relocating the handles onto the env "core"
/// and giving the state genuine owned DOFs is deferred to issue #256 (ADR 0039).
///
/// [`RapierWorld`]: crate::box2d::physics::RapierWorld
#[derive(Debug, Clone)]
pub struct BipedalWalkerState {
    /// Hull (torso) rigid body.
    pub(crate) hull_handle: RigidBodyHandle,
    /// Upper leg 1 (thigh).
    pub(crate) leg1_upper_handle: RigidBodyHandle,
    /// Lower leg 1 (shin).
    pub(crate) leg1_lower_handle: RigidBodyHandle,
    /// Upper leg 2 (thigh).
    pub(crate) leg2_upper_handle: RigidBodyHandle,
    /// Lower leg 2 (shin).
    pub(crate) leg2_lower_handle: RigidBodyHandle,
    /// Hip 1 revolute joint (hull ↔ upper leg 1).
    pub(crate) hip1_joint: ImpulseJointHandle,
    /// Knee 1 revolute joint (upper leg 1 ↔ lower leg 1).
    pub(crate) knee1_joint: ImpulseJointHandle,
    /// Hip 2 revolute joint (hull ↔ upper leg 2).
    pub(crate) hip2_joint: ImpulseJointHandle,
    /// Knee 2 revolute joint (upper leg 2 ↔ lower leg 2).
    pub(crate) knee2_joint: ImpulseJointHandle,
    /// Whether leg 1 is in contact with the ground.
    pub(crate) leg1_contact: bool,
    /// Whether leg 2 is in contact with the ground.
    pub(crate) leg2_contact: bool,
}

impl BipedalWalkerState {
    /// Handle to the hull (torso) rigid body.
    #[must_use]
    pub fn hull_handle(&self) -> RigidBodyHandle {
        self.hull_handle
    }

    /// Handle to the upper leg 1 (thigh) rigid body.
    #[must_use]
    pub fn leg1_upper_handle(&self) -> RigidBodyHandle {
        self.leg1_upper_handle
    }

    /// Handle to the lower leg 1 (shin) rigid body.
    #[must_use]
    pub fn leg1_lower_handle(&self) -> RigidBodyHandle {
        self.leg1_lower_handle
    }

    /// Handle to the upper leg 2 (thigh) rigid body.
    #[must_use]
    pub fn leg2_upper_handle(&self) -> RigidBodyHandle {
        self.leg2_upper_handle
    }

    /// Handle to the lower leg 2 (shin) rigid body.
    #[must_use]
    pub fn leg2_lower_handle(&self) -> RigidBodyHandle {
        self.leg2_lower_handle
    }

    /// Handle to the hip 1 revolute joint (hull ↔ upper leg 1).
    #[must_use]
    pub fn hip1_joint(&self) -> ImpulseJointHandle {
        self.hip1_joint
    }

    /// Handle to the knee 1 revolute joint (upper leg 1 ↔ lower leg 1).
    #[must_use]
    pub fn knee1_joint(&self) -> ImpulseJointHandle {
        self.knee1_joint
    }

    /// Handle to the hip 2 revolute joint (hull ↔ upper leg 2).
    #[must_use]
    pub fn hip2_joint(&self) -> ImpulseJointHandle {
        self.hip2_joint
    }

    /// Handle to the knee 2 revolute joint (upper leg 2 ↔ lower leg 2).
    #[must_use]
    pub fn knee2_joint(&self) -> ImpulseJointHandle {
        self.knee2_joint
    }

    /// Whether leg 1 is in contact with the ground.
    #[must_use]
    pub fn leg1_contact(&self) -> bool {
        self.leg1_contact
    }

    /// Whether leg 2 is in contact with the ground.
    #[must_use]
    pub fn leg2_contact(&self) -> bool {
        self.leg2_contact
    }
}

impl State<1> for BipedalWalkerState {
    /// Returns `[24]` — the flat observation dimension of the state.
    fn shape() -> [usize; 1] {
        [24]
    }

    /// Returns `true` when every handle is live.
    ///
    /// All five [`RigidBodyHandle`]s and all four [`ImpulseJointHandle`]s must
    /// differ from their `::invalid()` sentinels (they are `::invalid()`
    /// placeholders during the incremental world build). A `false` return
    /// signals a partially-assembled state, and the environment should be reset.
    fn is_valid(&self) -> bool {
        self.hull_handle != RigidBodyHandle::invalid()
            && self.leg1_upper_handle != RigidBodyHandle::invalid()
            && self.leg1_lower_handle != RigidBodyHandle::invalid()
            && self.leg2_upper_handle != RigidBodyHandle::invalid()
            && self.leg2_lower_handle != RigidBodyHandle::invalid()
            && self.hip1_joint != ImpulseJointHandle::invalid()
            && self.knee1_joint != ImpulseJointHandle::invalid()
            && self.hip2_joint != ImpulseJointHandle::invalid()
            && self.knee2_joint != ImpulseJointHandle::invalid()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::box2d::bipedal_walker::env::BipedalWalker;
    use rlevo_core::environment::{ConstructableEnv, Environment};

    /// Builds a valid, fully-assembled state by driving a real `reset()`.
    fn reset_state() -> BipedalWalkerState {
        let mut env = BipedalWalker::new(false);
        env.reset().expect("reset must succeed");
        env.state_for_test().clone()
    }

    #[test]
    fn is_valid_true_after_reset() {
        let state = reset_state();
        assert!(
            state.is_valid(),
            "a freshly-reset BipedalWalker state must be valid"
        );
    }

    #[test]
    fn is_valid_false_on_invalid_body_handle() {
        let mut state = reset_state();
        state.hull_handle = RigidBodyHandle::invalid();
        assert!(
            !state.is_valid(),
            "an invalid body handle must invalidate the state"
        );
    }

    #[test]
    fn is_valid_false_on_invalid_joint_handle() {
        let mut state = reset_state();
        state.hip1_joint = ImpulseJointHandle::invalid();
        assert!(
            !state.is_valid(),
            "an invalid joint handle must invalidate the state"
        );
    }
}
