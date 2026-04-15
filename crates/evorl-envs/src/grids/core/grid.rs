//! Rectangular grid of [`Entity`] cells plus an egocentric view extractor.

use super::agent::AgentState;
use super::direction::Direction;
use super::entity::Entity;
use super::observation::VIEW_SIZE;

/// Rectangular grid of [`Entity`] cells, row-major in `y` then `x`.
#[derive(Debug, Clone)]
pub struct Grid {
    width: usize,
    height: usize,
    cells: Vec<Entity>,
}

impl Grid {
    /// Construct a `width × height` grid initialized with [`Entity::Empty`].
    ///
    /// # Panics
    ///
    /// Panics if either dimension is zero.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        assert!(width > 0 && height > 0, "grid dimensions must be positive");
        Self {
            width,
            height,
            cells: vec![Entity::Empty; width * height],
        }
    }

    /// Grid width in cells.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Grid height in cells.
    #[must_use]
    pub const fn height(&self) -> usize {
        self.height
    }

    /// Whether `(x, y)` falls inside the grid.
    #[must_use]
    pub fn in_bounds(&self, x: i32, y: i32) -> bool {
        x >= 0
            && y >= 0
            && usize::try_from(x).is_ok_and(|ux| ux < self.width)
            && usize::try_from(y).is_ok_and(|uy| uy < self.height)
    }

    /// Read the cell at `(x, y)`. Out-of-bounds reads return [`Entity::Wall`],
    /// which keeps observation extraction total.
    #[must_use]
    pub fn get(&self, x: i32, y: i32) -> Entity {
        if self.in_bounds(x, y) {
            self.cells[self.index(x, y)]
        } else {
            Entity::Wall
        }
    }

    /// Overwrite the cell at `(x, y)`.
    ///
    /// # Panics
    ///
    /// Panics if `(x, y)` is out of bounds.
    pub fn set(&mut self, x: i32, y: i32, entity: Entity) {
        assert!(self.in_bounds(x, y), "Grid::set out of bounds: ({x}, {y})");
        let idx = self.index(x, y);
        self.cells[idx] = entity;
    }

    /// Fill the outermost rows and columns with [`Entity::Wall`], producing
    /// an impassable perimeter.
    pub fn draw_walls(&mut self) {
        #[allow(clippy::cast_possible_wrap)]
        let w = self.width as i32;
        #[allow(clippy::cast_possible_wrap)]
        let h = self.height as i32;
        for x in 0..w {
            self.set(x, 0, Entity::Wall);
            self.set(x, h - 1, Entity::Wall);
        }
        for y in 0..h {
            self.set(0, y, Entity::Wall);
            self.set(w - 1, y, Entity::Wall);
        }
    }

    fn index(&self, x: i32, y: i32) -> usize {
        debug_assert!(self.in_bounds(x, y));
        #[allow(clippy::cast_sign_loss)]
        let ux = x as usize;
        #[allow(clippy::cast_sign_loss)]
        let uy = y as usize;
        uy * self.width + ux
    }
}

/// Extract a [`VIEW_SIZE`]×[`VIEW_SIZE`] egocentric view of the grid.
///
/// The agent sits at view coordinates `(row = VIEW_SIZE - 1, col = VIEW_SIZE / 2)`
/// and looks toward row `0`. Cells outside the grid decode as
/// [`Entity::Wall`], so the returned array is always fully populated.
#[must_use]
pub fn egocentric_view(grid: &Grid, agent: &AgentState) -> [[Entity; VIEW_SIZE]; VIEW_SIZE] {
    let mut view = [[Entity::Wall; VIEW_SIZE]; VIEW_SIZE];
    #[allow(clippy::cast_possible_wrap)]
    let agent_row = (VIEW_SIZE - 1) as i32;
    #[allow(clippy::cast_possible_wrap)]
    let agent_col = (VIEW_SIZE / 2) as i32;
    for (vr, row) in view.iter_mut().enumerate() {
        for (vc, cell) in row.iter_mut().enumerate() {
            #[allow(clippy::cast_possible_wrap)]
            let forward = agent_row - vr as i32; // positive = ahead of agent
            #[allow(clippy::cast_possible_wrap)]
            let right = vc as i32 - agent_col; // positive = to agent's right
            let (wx, wy) = rotate_view_offset(agent.direction, right, forward);
            *cell = grid.get(agent.x + wx, agent.y + wy);
        }
    }
    view
}

/// Rotate an agent-local `(right, forward)` offset into a world `(dx, dy)`
/// offset based on the agent's facing direction.
#[must_use]
const fn rotate_view_offset(dir: Direction, right: i32, forward: i32) -> (i32, i32) {
    match dir {
        // Facing North: forward = -y, right = +x
        Direction::North => (right, -forward),
        // Facing East: forward = +x, right = +y
        Direction::East => (forward, right),
        // Facing South: forward = +y, right = -x
        Direction::South => (-right, forward),
        // Facing West: forward = -x, right = -y
        Direction::West => (-forward, -right),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_fills_empty() {
        let g = Grid::new(3, 4);
        assert_eq!(g.width(), 3);
        assert_eq!(g.height(), 4);
        for y in 0..4 {
            for x in 0..3 {
                assert_eq!(g.get(x, y), Entity::Empty);
            }
        }
    }

    #[test]
    fn in_bounds_limits() {
        let g = Grid::new(3, 3);
        assert!(g.in_bounds(0, 0));
        assert!(g.in_bounds(2, 2));
        assert!(!g.in_bounds(3, 0));
        assert!(!g.in_bounds(0, 3));
        assert!(!g.in_bounds(-1, 0));
        assert!(!g.in_bounds(0, -1));
    }

    #[test]
    fn out_of_bounds_reads_as_wall() {
        let g = Grid::new(3, 3);
        assert_eq!(g.get(-1, 0), Entity::Wall);
        assert_eq!(g.get(3, 0), Entity::Wall);
        assert_eq!(g.get(0, -1), Entity::Wall);
        assert_eq!(g.get(0, 3), Entity::Wall);
    }

    #[test]
    fn draw_walls_wraps_perimeter() {
        let mut g = Grid::new(4, 4);
        g.draw_walls();
        // Corners and edges are walls.
        for x in 0..4 {
            assert_eq!(g.get(x, 0), Entity::Wall);
            assert_eq!(g.get(x, 3), Entity::Wall);
        }
        for y in 0..4 {
            assert_eq!(g.get(0, y), Entity::Wall);
            assert_eq!(g.get(3, y), Entity::Wall);
        }
        // Interior stays empty.
        assert_eq!(g.get(1, 1), Entity::Empty);
        assert_eq!(g.get(2, 2), Entity::Empty);
    }

    #[test]
    fn set_and_get_roundtrip() {
        let mut g = Grid::new(3, 3);
        g.set(1, 1, Entity::Goal);
        assert_eq!(g.get(1, 1), Entity::Goal);
        assert_eq!(g.get(0, 0), Entity::Empty);
    }

    #[test]
    fn egocentric_view_in_front_matches_world() {
        let mut g = Grid::new(5, 5);
        g.draw_walls();
        g.set(3, 1, Entity::Goal); // two cells east of the agent
        let agent = AgentState::new(1, 1, Direction::East);
        let view = egocentric_view(&g, &agent);
        // Agent at view[6][3]. Cell two steps forward is at view[4][3].
        assert_eq!(view[4][3], Entity::Goal);
    }

    #[test]
    fn egocentric_view_rotates_with_direction() {
        let mut g = Grid::new(5, 5);
        g.draw_walls();
        g.set(1, 3, Entity::Lava); // two cells south of the agent
        let agent = AgentState::new(1, 1, Direction::South);
        let view = egocentric_view(&g, &agent);
        // Facing south, the lava is directly in front.
        assert_eq!(view[4][3], Entity::Lava);
    }

    #[test]
    fn egocentric_view_out_of_grid_is_wall() {
        let g = Grid::new(2, 2);
        let agent = AgentState::new(0, 0, Direction::East);
        let view = egocentric_view(&g, &agent);
        let wall_count = view
            .iter()
            .flatten()
            .filter(|&&e| e == Entity::Wall)
            .count();
        // Only the agent's own cell and at most one neighbor are inside the grid.
        assert!(wall_count >= 45);
    }
}
