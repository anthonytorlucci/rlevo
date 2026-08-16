//! Full environment state: a [`Grid`] plus the owning [`AgentState`].

use super::agent::AgentState;
use super::grid::Grid;
use super::observation::{OBS_CHANNELS, VIEW_SIZE};
use rlevo_core::base::State;

/// The complete state of a grid environment.
///
/// `GridState::shape` reports `[VIEW_SIZE, VIEW_SIZE, OBS_CHANNELS]` — the
/// shape of the egocentric observation that
/// [`observe_grid`](super::observe_grid) emits for it. The grid itself can be
/// any size at runtime; the static shape is constant across all grid
/// environments so tensor code doesn't have to branch.
///
/// # Why there is no `Observable` impl
///
/// Observation production deliberately does **not** live on this type. The
/// emission model is parameterised by the owning environment's
/// [`Visibility`](super::Visibility) policy, which a `&self`-only
/// [`project`](rlevo_core::state::Observable::project) cannot read — and an
/// unparameterised projection would be an *unoccluded* view, correct for the
/// see-through environments and silently wrong for the occluded majority.
///
/// That majority case used to be broken: the egocentric view builder read
/// every cell of the rotated view window straight from the [`Grid`], so walls
/// never blocked line of sight and `see_through_walls` was effectively always
/// `true` across all grid environments. The fix ports Minigrid's
/// shadow-casting visibility pass — walls (and other entities that don't
/// "see behind" themselves) occlude cells behind them, and occluded cells are
/// erased from the observation — gated per environment by
/// [`Visibility`](super::Visibility) rather than hard-coded, since canonical
/// environments disagree on the default (e.g. `GoToDoor` is see-through,
/// `Memory` is occluded). Call [`observe_grid`](super::observe_grid) with the
/// environment's policy instead (ADR 0047).
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
    fn shape() -> [usize; 3] {
        [VIEW_SIZE, VIEW_SIZE, OBS_CHANNELS]
    }

    /// Returns `true` when the agent's current position falls inside the grid
    /// bounds. A freshly constructed state is always valid; a state produced
    /// by replaying a corrupted action sequence may not be.
    fn is_valid(&self) -> bool {
        self.grid.in_bounds(self.agent.x, self.agent.y)
    }
}

#[cfg(test)]
mod tests {
    use super::super::entity::Entity;
    use super::super::{Visibility, observe_grid};
    use super::*;
    use crate::direction::Direction;

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
        let obs = observe_grid(&state, Visibility::SeeThrough);
        assert_eq!(obs.agent_direction, Some(Direction::East));
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
