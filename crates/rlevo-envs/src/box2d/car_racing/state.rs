//! State type for CarRacing.

use rapier2d::dynamics::RigidBodyHandle;
use rlevo_core::base::State;

use super::observation::CarRacingObservation;

/// Physics and progress state for CarRacing.
#[derive(Debug, Clone)]
pub struct CarRacingState {
    /// Car body rigid body handle.
    pub car_handle: RigidBodyHandle,
    /// Four wheel rigid body handles (FL, FR, RL, RR).
    pub wheel_handles: [RigidBodyHandle; 4],
    /// Index of the nearest track tile at the start of the last step.
    pub current_tile: usize,
    /// Number of unique track tiles the car has visited.
    pub tiles_visited: usize,
    /// Total number of tiles in the track.
    pub total_tiles: usize,
    /// Whether the lap has been completed.
    pub lap_complete: bool,
    /// Cached last rendered observation.
    pub last_obs: CarRacingObservation,
}

impl State<3> for CarRacingState {
    type Observation = CarRacingObservation;

    fn shape() -> [usize; 3] {
        [96, 96, 3]
    }

    fn is_valid(&self) -> bool {
        true
    }

    fn observe(&self) -> CarRacingObservation {
        self.last_obs.clone()
    }
}
