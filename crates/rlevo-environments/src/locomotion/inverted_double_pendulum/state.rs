//! Physics state for [`super::InvertedDoublePendulum`].

use rapier3d::prelude::{ImpulseJointHandle, RigidBodyHandle};
use rlevo_core::base::State;

use super::observation::InvertedDoublePendulumObservation;

/// Physics state: body/joint handles plus the last observation. The world
/// itself (non-`Clone`) lives on [`super::InvertedDoublePendulum`].
///
/// Handles are indices into the `Rapier3DWorld` that owns the rigid-body set.
/// They remain valid for the lifetime of a single episode; after `reset` a
/// new world is constructed and all handles are replaced.
#[derive(Debug, Clone)]
pub struct InvertedDoublePendulumState {
    /// Rapier handle for the sliding cart body.
    pub cart: RigidBodyHandle,
    /// Rapier handle for the lower pole (pole1), revolute-y to the cart top.
    pub pole1: RigidBodyHandle,
    /// Rapier handle for the upper pole (pole2), revolute-y to pole1's top.
    pub pole2: RigidBodyHandle,
    /// Impulse-joint handle connecting the cart to pole1.
    pub joint1: ImpulseJointHandle,
    /// Impulse-joint handle connecting pole1 to pole2.
    pub joint2: ImpulseJointHandle,
    /// Most recent observation extracted from the physics world. Populated by
    /// the env-side [`Sensor`](rlevo_core::environment::Sensor) after each
    /// `reset` or `step`; read only by [`State::is_valid`] to detect physics
    /// divergence (ADR 0047 moved observation production off the state).
    pub last_obs: InvertedDoublePendulumObservation,
}

impl State<1> for InvertedDoublePendulumState {
    fn shape() -> [usize; 1] {
        [9]
    }

    /// Returns `true` when the last extracted observation contains only
    /// finite values. Non-finite values indicate a physics instability or
    /// an uninitialised state.
    fn is_valid(&self) -> bool {
        self.last_obs.is_finite()
    }
}
