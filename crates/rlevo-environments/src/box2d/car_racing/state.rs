//! Physics and progress state for the CarRacing environment.
//!
//! [`CarRacingState`] holds the Rapier2D body handles for the car and wheels,
//! tile-visit counters used for lap-completion detection, and the most recent
//! rendered pixel observation.

use rapier2d::dynamics::RigidBodyHandle;
use rlevo_core::base::State;

use super::observation::CarRacingObservation;

/// Physics and progress state for CarRacing.
///
/// # Handle-lifetime caveat
///
/// The [`car_handle`](Self::car_handle) and
/// [`wheel_handles`](Self::wheel_handles) are Rapier arena indices into the
/// [`RapierWorld`](crate::box2d::physics::RapierWorld) owned by the
/// *environment*, not by this state. The car's physics degrees of freedom
/// (pose, twist, per-wheel dynamics) live *behind* those handles in that world;
/// they are not values stored here. Consequently `#[derive(Clone)]` yields a
/// non-portable *view*: a clone's handles alias the arena they were taken from
/// and **dangle** once the world is rebuilt by a `reset()`. The
/// [`shape()`](State::shape) of `[96, 96, 3]` describes the pixel-observation
/// modality (the cached [`last_obs`](Self::last_obs)), not the physics state.
///
/// Making the state genuinely self-contained/Markov (owning its DOFs as values
/// and re-modelling CarRacing as `Environment<3, 1, 1>` + `Observable<3>`) is
/// tracked by issue #255 (ADR 0039); this type only closes the encapsulation
/// and invariant-honesty gap.
#[derive(Debug, Clone)]
pub struct CarRacingState {
    /// Car body rigid body handle.
    pub(crate) car_handle: RigidBodyHandle,
    /// Four wheel rigid body handles (FL, FR, RL, RR).
    pub(crate) wheel_handles: [RigidBodyHandle; 4],
    /// Index of the nearest track tile at the start of the last step, or `None`
    /// if the car has not yet been on a tile this episode.
    pub(crate) current_tile: Option<usize>,
    /// Number of unique track tiles the car has visited.
    pub(crate) tiles_visited: usize,
    /// Total number of tiles in the track.
    pub(crate) total_tiles: usize,
    /// Whether the lap has been completed.
    pub(crate) lap_complete: bool,
    /// Cached last rendered observation.
    pub(crate) last_obs: CarRacingObservation,
}

impl CarRacingState {
    /// Rigid body handle for the car body.
    #[must_use]
    pub fn car_handle(&self) -> RigidBodyHandle {
        self.car_handle
    }

    /// The four wheel rigid body handles (FL, FR, RL, RR).
    #[must_use]
    pub fn wheel_handles(&self) -> [RigidBodyHandle; 4] {
        self.wheel_handles
    }

    /// Index of the nearest track tile at the start of the last step, or `None`
    /// if the car has not yet visited any tile this episode.
    #[must_use]
    pub fn current_tile(&self) -> Option<usize> {
        self.current_tile
    }

    /// Number of unique track tiles the car has visited.
    #[must_use]
    pub fn tiles_visited(&self) -> usize {
        self.tiles_visited
    }

    /// Total number of tiles in the track.
    #[must_use]
    pub fn total_tiles(&self) -> usize {
        self.total_tiles
    }

    /// Whether the lap has been completed.
    #[must_use]
    pub fn lap_complete(&self) -> bool {
        self.lap_complete
    }

    /// The most recent cached pixel observation.
    #[must_use]
    pub fn last_obs(&self) -> &CarRacingObservation {
        &self.last_obs
    }
}

impl State<3> for CarRacingState {
    type Observation = CarRacingObservation;

    fn shape() -> [usize; 3] {
        [96, 96, 3]
    }

    fn is_valid(&self) -> bool {
        self.car_handle != RigidBodyHandle::invalid()
            && self
                .wheel_handles
                .iter()
                .all(|h| *h != RigidBodyHandle::invalid())
            && self.tiles_visited <= self.total_tiles
            && self.current_tile.is_none_or(|i| i < self.total_tiles)
    }

    fn observe(&self) -> CarRacingObservation {
        self.last_obs.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::box2d::car_racing::action::CarRacingAction;
    use crate::box2d::car_racing::env::CarRacing;
    use rlevo_core::environment::{ConstructableEnv, Environment};

    /// Builds a freshly-reset CarRacing env and returns it. The env owns the
    /// world the state's handles index into, so it must outlive any state read.
    fn reset_env() -> CarRacing {
        let mut env = CarRacing::new(false);
        env.reset().expect("reset must succeed");
        env
    }

    #[test]
    fn is_valid_true_after_reset() {
        let env = reset_env();
        assert!(
            env.state_for_test().is_valid(),
            "freshly-reset state must satisfy its invariant"
        );
    }

    #[test]
    fn is_valid_false_on_invalid_car_handle() {
        let mut env = reset_env();
        env.state_for_test_mut().car_handle = RigidBodyHandle::invalid();
        assert!(
            !env.state_for_test().is_valid(),
            "invalid car handle must fail the invariant"
        );
    }

    #[test]
    fn is_valid_false_on_invalid_wheel_handle() {
        let mut env = reset_env();
        env.state_for_test_mut().wheel_handles[2] = RigidBodyHandle::invalid();
        assert!(
            !env.state_for_test().is_valid(),
            "an invalid wheel handle must fail the invariant"
        );
    }

    #[test]
    fn is_valid_false_when_visited_exceeds_total() {
        let mut env = reset_env();
        let total = env.state_for_test().total_tiles();
        env.state_for_test_mut().tiles_visited = total + 1;
        assert!(
            !env.state_for_test().is_valid(),
            "tiles_visited > total_tiles must fail the invariant"
        );
    }

    #[test]
    fn current_tile_none_after_reset_some_after_visit() {
        let mut env = reset_env();
        assert_eq!(
            env.state_for_test().current_tile(),
            None,
            "no tile should be recorded before the first step"
        );

        // Drive the car forward until it registers a tile visit (the start pose
        // sits on the track, so this typically lands on step 1).
        let mut visited = false;
        for _ in 0..8 {
            env.step(CarRacingAction::new(0.0, 0.5, 0.0))
                .expect("valid action must step");
            if env.state_for_test().current_tile().is_some() {
                visited = true;
                break;
            }
        }
        assert!(
            visited,
            "car should register a tile visit within a few steps"
        );
        assert!(
            matches!(env.state_for_test().current_tile(), Some(i) if i < env.state_for_test().total_tiles()),
            "recorded tile index must be within the track"
        );
    }
}
