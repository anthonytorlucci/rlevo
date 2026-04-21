//! Full environment state: a [`Grid`] plus the owning [`AgentState`].

use super::agent::AgentState;
use super::grid::{egocentric_view, Grid};
use super::observation::{GridObservation, OBS_CHANNELS, VIEW_SIZE};
use evorl_core::base::State;

/// The complete state of a grid environment.
///
/// `GridState::shape` reports `[VIEW_SIZE, VIEW_SIZE, OBS_CHANNELS]` —
/// the shape of the egocentric observation that `observe()` emits. The
/// grid itself can be any size at runtime; the static shape is constant
/// across all grid environments so tensor code doesn't have to branch.
#[derive(Debug, Clone)]
pub struct GridState {
    /// The world grid.
    pub grid: Grid,
    /// The agent's position, facing, and carried item.
    pub agent: AgentState,
}

impl GridState {
    /// Construct a [`GridState`] from a grid and agent.
    #[must_use]
    pub const fn new(grid: Grid, agent: AgentState) -> Self {
        Self { grid, agent }
    }
}

impl State<3> for GridState {
    type Observation = GridObservation;

    fn shape() -> [usize; 3] {
        [VIEW_SIZE, VIEW_SIZE, OBS_CHANNELS]
    }

    fn observe(&self) -> Self::Observation {
        let view = egocentric_view(&self.grid, &self.agent);
        GridObservation::from_entity_view(view, self.agent.direction)
    }

    fn is_valid(&self) -> bool {
        self.grid.in_bounds(self.agent.x, self.agent.y)
    }
}

#[cfg(test)]
mod tests {
    use super::super::direction::Direction;
    use super::super::entity::Entity;
    use super::*;

    #[test]
    fn shape_matches_observation_shape() {
        assert_eq!(GridState::shape(), [VIEW_SIZE, VIEW_SIZE, OBS_CHANNELS]);
    }

    #[test]
    fn observe_returns_well_formed_observation() {
        let mut grid = Grid::new(5, 5);
        grid.draw_walls();
        grid.set(3, 3, Entity::Goal);
        let agent = AgentState::new(1, 1, Direction::East);
        let state = GridState::new(grid, agent);
        let obs = state.observe();
        assert_eq!(obs.agent_direction, Direction::East.to_u8());
    }

    #[test]
    fn is_valid_checks_agent_in_grid() {
        let grid = Grid::new(3, 3);
        let agent = AgentState::new(1, 1, Direction::East);
        assert!(GridState::new(grid.clone(), agent).is_valid());
        let out_of_bounds = AgentState::new(5, 5, Direction::East);
        assert!(!GridState::new(grid, out_of_bounds).is_valid());
    }
}
