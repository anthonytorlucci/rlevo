//! Minimal ASCII renderer for grid environments.
//!
//! This is intentionally pure: callers pass a `(Grid, AgentState)` pair
//! and receive a `String`. No terminal control sequences, no curses, no
//! blocking IO — printing is the caller's responsibility.

use super::agent::AgentState;
use super::direction::Direction;
use super::entity::{DoorState, Entity};
use super::grid::Grid;

/// Render the grid and the agent's position to a multi-line ASCII string.
#[must_use]
pub fn render_ascii(grid: &Grid, agent: &AgentState) -> String {
    let mut out = String::with_capacity(grid.width() * grid.height() * 2);
    #[allow(clippy::cast_possible_wrap)]
    let height = grid.height() as i32;
    #[allow(clippy::cast_possible_wrap)]
    let width = grid.width() as i32;
    for y in 0..height {
        for x in 0..width {
            let ch = if x == agent.x && y == agent.y {
                agent_char(agent)
            } else {
                entity_char(grid.get(x, y))
            };
            out.push(ch);
            out.push(' ');
        }
        out.push('\n');
    }
    out
}

const fn agent_char(agent: &AgentState) -> char {
    match agent.direction {
        Direction::East => '>',
        Direction::South => 'v',
        Direction::West => '<',
        Direction::North => '^',
    }
}

const fn entity_char(e: Entity) -> char {
    match e {
        Entity::Empty | Entity::Floor => '.',
        Entity::Wall => '#',
        Entity::Goal => 'G',
        Entity::Lava => 'L',
        Entity::Door(_, DoorState::Open) => '/',
        Entity::Door(_, DoorState::Closed) => '+',
        Entity::Door(_, DoorState::Locked) => '*',
        Entity::Key(_) => 'k',
        Entity::Ball(_) => 'o',
        Entity::Box(_) => '[',
    }
}

#[cfg(test)]
mod tests {
    use super::super::color::Color;
    use super::*;

    #[test]
    fn renders_walls_and_agent() {
        let mut g = Grid::new(3, 3);
        g.draw_walls();
        let a = AgentState::new(1, 1, Direction::East);
        let s = render_ascii(&g, &a);
        // Expected rows: "# # # \n", "# > # \n", "# # # \n"
        assert!(s.contains('>'));
        assert!(s.contains('#'));
        assert_eq!(s.lines().count(), 3);
    }

    #[test]
    fn distinct_chars_for_distinct_entities() {
        let mut g = Grid::new(5, 1);
        g.set(0, 0, Entity::Wall);
        g.set(1, 0, Entity::Goal);
        g.set(2, 0, Entity::Lava);
        g.set(3, 0, Entity::Key(Color::Red));
        g.set(4, 0, Entity::Door(Color::Blue, DoorState::Locked));
        let agent = AgentState::new(100, 100, Direction::East); // off-grid so not drawn
        let s = render_ascii(&g, &agent);
        assert!(s.contains('#'));
        assert!(s.contains('G'));
        assert!(s.contains('L'));
        assert!(s.contains('k'));
        assert!(s.contains('*'));
    }
}
