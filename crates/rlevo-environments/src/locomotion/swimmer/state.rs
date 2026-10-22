//! Physics state for [`super::Swimmer`].

use rapier3d::prelude::{MultibodyJointHandle, RigidBodyHandle};

use super::observation::SwimmerObservation;

/// Physics state — body + joint handles plus the last observation. The
/// non-`Clone` world lives on the env struct directly.
///
/// The two revolute-z joints live in Rapier's `MultibodyJointSet` (not the
/// impulse set): a free-floating serial chain with no grounded reference is
/// a stiff problem for the PGS impulse solver, which injects kinetic energy
/// when constraint drift is corrected aggressively. The Featherstone-style
/// multibody solver parameterises the chain in reduced coordinates
/// (analogous to `MuJoCo`'s generalised coordinates) and stays conservative.
#[derive(Debug, Clone)]
pub struct SwimmerState {
    pub segment0: RigidBodyHandle,
    pub segment1: RigidBodyHandle,
    pub segment2: RigidBodyHandle,
    pub joint1: MultibodyJointHandle,
    pub joint2: MultibodyJointHandle,
    pub last_obs: SwimmerObservation,
}
