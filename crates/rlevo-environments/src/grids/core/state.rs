//! Full environment state: a [`Grid`] plus the owning [`AgentState`].

use super::agent::AgentState;
use super::grid::Grid;

/// The complete state of a grid environment.
///
/// # Why there is no projection method
///
/// Observation production deliberately does **not** live on this type. The
/// emission model is parameterised by the owning environment's
/// [`Visibility`](super::Visibility) policy, which a `&self`-only
/// `project` method on this type cannot read — and an
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

#[cfg(test)]
mod tests {
    use super::super::entity::Entity;
    use super::super::{Visibility, observe_grid};
    use super::*;
    use crate::direction::Direction;

    #[test]
    fn observe_returns_well_formed_observation() {
        let mut grid = Grid::new(5, 5);
        grid.draw_walls();
        grid.set(3, 3, Entity::Goal);
        let agent = AgentState::new(1, 1, Direction::East);
        let state = GridState::new(grid, agent);
        let obs = observe_grid(&state, Visibility::SeeThrough);
        assert_eq!(obs.agent_direction, Direction::East);
    }
}
