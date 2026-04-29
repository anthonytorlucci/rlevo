//! Mutable per-episode agent state: position, facing, and carried item.

use super::direction::Direction;
use super::entity::Entity;

/// The agent's mutable state within a single episode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AgentState {
    /// World x coordinate.
    pub x: i32,
    /// World y coordinate.
    pub y: i32,
    /// Direction the agent is currently facing.
    pub direction: Direction,
    /// Item the agent is holding, if any.
    pub carrying: Option<Entity>,
}

impl AgentState {
    /// Construct an agent at `(x, y)` facing `direction` with an empty hand.
    #[must_use]
    pub const fn new(x: i32, y: i32, direction: Direction) -> Self {
        Self {
            x,
            y,
            direction,
            carrying: None,
        }
    }

    /// World coordinates of the cell immediately in front of the agent.
    #[must_use]
    pub const fn front(&self) -> (i32, i32) {
        let (dx, dy) = self.direction.delta();
        (self.x + dx, self.y + dy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_has_empty_hand() {
        let a = AgentState::new(3, 4, Direction::North);
        assert_eq!(a.x, 3);
        assert_eq!(a.y, 4);
        assert_eq!(a.direction, Direction::North);
        assert_eq!(a.carrying, None);
    }

    #[test]
    fn front_follows_direction() {
        let a = AgentState::new(5, 5, Direction::East);
        assert_eq!(a.front(), (6, 5));

        let a = AgentState::new(5, 5, Direction::North);
        assert_eq!(a.front(), (5, 4));

        let a = AgentState::new(5, 5, Direction::South);
        assert_eq!(a.front(), (5, 6));

        let a = AgentState::new(5, 5, Direction::West);
        assert_eq!(a.front(), (4, 5));
    }
}
