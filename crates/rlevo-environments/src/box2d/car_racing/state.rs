//! Physics and progress state for the `CarRacing` environment.
//!
//! [`CarRacingState`] holds the `Rapier2D` body handles for the car and wheels
//! and the tile-visit counters used for lap-completion detection.

use rapier2d::dynamics::RigidBodyHandle;
use rlevo_core::base::State;

/// Physics and progress state for `CarRacing`.
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
/// modality — a frame rendered on demand by the env-side
/// [`Sensor`](rlevo_core::environment::Sensor) rasterizing the world, no longer
/// cached on this struct — not the physics state.
///
/// `CarRacing` is a genuine pixels-over-physics POMDP: a single rendered
/// frame is not Markov (recovering angular velocities needs frame-stacking),
/// while the true state is the hull's pose and twist, each wheel's
/// dynamical state (angular velocity, steering phase, slip), fuel, and track
/// progress. The `[96, 96, 3]` shape above is not that state — this struct
/// was typed to the pixel-buffer shape, forcing `CarRacing` into
/// `Environment<3, 3, 1>`, only because
/// [`Observable<OR>`](rlevo_core::state::Observable) (the
/// modality-changing-observation trait) did not exist when this environment
/// was built.
///
/// ADR 0039 targets a compact `State<1>` — this type, refactored to own the
/// DOFs above as values instead of Rapier handles — paired with
/// `Environment<3, 1, 1>` and an `impl Sensor<3, 1, 1> for CarRacing`.
/// ADR 0047 later gave `Sensor` direct `&self` world access, so that target
/// `Sensor` renders the frame straight from the live world rather than a
/// cached `Observable` projection — the on-demand rendering already
/// described above. The handles would move off the exported state onto the
/// environment's core struct, and [`is_valid()`](State::is_valid) would
/// become a real check over the dynamical DOFs rather than handle liveness.
/// As of the most recent audit (2026-08-10) this refactor has not landed:
/// `CarRacingState` is still the handle bundle below, and `env.rs` still
/// implements `Environment<3, 3, 1>`; this type today only closes the
/// encapsulation and invariant-honesty gap around those handles.
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
}

impl State<3> for CarRacingState {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::box2d::car_racing::action::CarRacingAction;
    use crate::box2d::car_racing::env::CarRacing;
    use rlevo_core::environment::{ConstructableEnv, Environment};

    /// Builds a freshly-reset `CarRacing` env and returns it. The env owns the
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
