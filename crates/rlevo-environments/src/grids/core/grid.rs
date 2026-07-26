//! Rectangular grid of [`Entity`] cells, an egocentric view extractor, and the
//! shadow-casting pass that turns that view into a visibility-masked one.

use super::agent::AgentState;
use super::entity::Entity;
use super::observation::VIEW_SIZE;
use crate::direction::Direction;

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

/// View-window row the agent occupies: the bottom row, looking toward row `0`.
///
/// Canonical Minigrid passes `agent_pos=(agent_view_size // 2,
/// agent_view_size - 1)` to `Grid.process_vis`; this is the `row` half of that
/// pair, derived from [`VIEW_SIZE`] rather than hardcoded.
const AGENT_VIEW_ROW: usize = VIEW_SIZE - 1;

/// View-window column the agent occupies: the centre column.
const AGENT_VIEW_COL: usize = VIEW_SIZE / 2;

/// Extract a [`VIEW_SIZE`]×[`VIEW_SIZE`] egocentric view of the grid.
///
/// The agent sits at view coordinates `(row = VIEW_SIZE - 1, col = VIEW_SIZE / 2)`
/// and looks toward row `0`. Cells outside the grid decode as
/// [`Entity::Wall`], so the returned array is always fully populated.
///
/// # Visibility
///
/// The returned view is **unoccluded**: every cell of the window is read
/// straight out of the grid, walls included. That is only a correct emission
/// model for an environment that opted into
/// [`Visibility::SeeThrough`](super::Visibility::SeeThrough), which is why this
/// function is `pub(crate)` — the crate-public entry point is
/// [`observe_grid`](super::observe_grid), which applies the environment's own
/// visibility policy. Callers inside the crate that want the occluded view feed
/// this result to [`process_vis`].
#[must_use]
pub(crate) fn egocentric_view(grid: &Grid, agent: &AgentState) -> [[Entity; VIEW_SIZE]; VIEW_SIZE] {
    let mut view = [[Entity::Wall; VIEW_SIZE]; VIEW_SIZE];
    #[allow(clippy::cast_possible_wrap)]
    let agent_row = AGENT_VIEW_ROW as i32;
    #[allow(clippy::cast_possible_wrap)]
    let agent_col = AGENT_VIEW_COL as i32;
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

/// Apply canonical Minigrid's shadow cast to an already-rotated egocentric
/// `view`, replacing every cell the agent cannot see with `None`.
///
/// This is a direct port of `Grid.process_vis`
/// (`minigrid/core/grid.py`, `master` @ 2026-07-26). The input is the output of
/// [`egocentric_view`], i.e. the window already rotated into the agent's frame,
/// which is the same coordinate system canonical runs `process_vis` in (its
/// `gen_obs_grid` slices, then `rotate_left`s, then occludes).
///
/// # Algorithm
///
/// Visibility is seeded at the agent's own cell and flood-filled *forward*:
/// each row is finalized before the next row out is touched, so a shadow can
/// never leak backward toward the agent. Within a row, a lit transparent cell
/// lights its two horizontal neighbours and the three cells one row ahead of
/// it. An **opaque** cell (`see_behind() == false`) is itself lit — the agent
/// sees the wall that blocks it — but propagates nothing.
///
/// A consequence worth stating, because it looks like a bug and is not: a
/// *single* isolated wall cell casts essentially no shadow, since light routes
/// diagonally around it from the neighbouring columns. Durable shadows require
/// an occluder that also blocks the sideways spread within each row — a wall
/// run, or the view-window edge (out-of-grid cells already read as
/// [`Entity::Wall`] via [`Grid::get`], matching canonical `Grid.slice`).
///
/// # Index transposition
///
/// Canonical indexes its mask `mask[i, j]` with **`i` the column and `j` the
/// row**; rlevo's view array is `view[row][col]`. Every index pair below is
/// therefore deliberately swapped relative to the Python source: canonical's
/// `mask[i + 1, j - 1]` is this port's `mask[row - 1][col + 1]`.
///
/// Getting it wrong is hard to catch, and harder than it looks. Transcribing
/// canonical's `i`/`j` straight into rlevo's two array indices computes
/// `transpose(process_vis(transpose(view)))` — so it agrees with this function
/// on **every** window that composition leaves fixed, which includes the
/// obvious test fixtures: a full-height wall column transposes into a
/// full-width wall row, and both produce the same mask back. A window merely
/// being asymmetric is therefore *not* enough to discriminate the two. The
/// fixture that is, and the transposed transcription it is checked against,
/// live in this module's tests as
/// `process_vis_offset_wall_stub_detects_a_transposed_implementation`.
///
/// # Returns
///
/// A `VIEW_SIZE × VIEW_SIZE` array where a visible cell is `Some(entity)` and a
/// masked cell is `None`.
#[must_use]
pub(crate) fn process_vis(
    view: [[Entity; VIEW_SIZE]; VIEW_SIZE],
) -> [[Option<Entity>; VIEW_SIZE]; VIEW_SIZE] {
    // Canonical: `mask = np.zeros((width, height)); mask[agent_pos] = True`.
    // Transposed to `[row][col]`; the agent sits at the bottom-centre of the
    // window, derived from VIEW_SIZE rather than hardcoded to (3, 6).
    let mut mask = [[false; VIEW_SIZE]; VIEW_SIZE];
    mask[AGENT_VIEW_ROW][AGENT_VIEW_COL] = true;

    // Canonical: `for j in reversed(range(0, self.height))` — from the agent's
    // own row outward toward row 0.
    for row in (0..VIEW_SIZE).rev() {
        // Canonical sub-pass 1: `for i in range(0, self.width - 1)`, writing
        // `i + 1`. The upper bound is short by one precisely because of that
        // write.
        for col in 0..VIEW_SIZE - 1 {
            if !mask[row][col] {
                continue;
            }
            // The occluder is already marked visible above; `continue` here
            // only stops it from *propagating*.
            if !view[row][col].see_behind() {
                continue;
            }
            mask[row][col + 1] = true;
            if row > 0 {
                mask[row - 1][col + 1] = true;
                mask[row - 1][col] = true;
            }
        }

        // Canonical sub-pass 2: `for i in reversed(range(1, self.width))`,
        // writing `i - 1`. Both sub-passes are required — visibility has to
        // spread sideways from the agent's column in *both* directions before
        // the row pushes forward, and a single-direction sweep silently loses
        // cells on one side of an occluder.
        for col in (1..VIEW_SIZE).rev() {
            if !mask[row][col] {
                continue;
            }
            if !view[row][col].see_behind() {
                continue;
            }
            mask[row][col - 1] = true;
            if row > 0 {
                mask[row - 1][col - 1] = true;
                mask[row - 1][col] = true;
            }
        }
    }

    // Canonical erases masked cells with `self.set(i, j, None)`; rlevo models
    // that absence with `Option` rather than an `Entity::Unseen` variant, so
    // the exhaustive `entity_to_tile` match in `render.rs` — which feeds the
    // versioned `GridTile` wire type — stays untouched by a value that can
    // never appear in a recorded world grid.
    std::array::from_fn(|row| std::array::from_fn(|col| mask[row][col].then_some(view[row][col])))
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

    // ---- process_vis -----------------------------------------------------

    use super::super::color::Color;
    use super::super::entity::DoorState;

    /// Shorthand for a visible cell in a hand-written expected mask.
    const V: bool = true;
    /// Shorthand for a masked (unseen) cell in a hand-written expected mask.
    const X: bool = false;

    /// Visible-cell predicate per position, for readable whole-mask asserts.
    fn visibility(
        masked: &[[Option<Entity>; VIEW_SIZE]; VIEW_SIZE],
    ) -> [[bool; VIEW_SIZE]; VIEW_SIZE] {
        std::array::from_fn(|row| std::array::from_fn(|col| masked[row][col].is_some()))
    }

    /// `[[Entity; N]; N]` -> `[[Option<Entity>; N]; N]`, everything visible.
    fn all_visible(
        view: [[Entity; VIEW_SIZE]; VIEW_SIZE],
    ) -> [[Option<Entity>; VIEW_SIZE]; VIEW_SIZE] {
        view.map(|row| row.map(Some))
    }

    /// A wall row one cell ahead of the agent with `centre` punched into the
    /// middle column. Everything else is [`Entity::Empty`].
    ///
    /// This is the discriminating fixture for "does `centre` occlude?": the
    /// flanking walls stop light spreading sideways, so the only route to rows
    /// `0..=4` is straight through `centre`.
    fn wall_row_with_centre(centre: Entity) -> [[Entity; VIEW_SIZE]; VIEW_SIZE] {
        let mut view = [[Entity::Empty; VIEW_SIZE]; VIEW_SIZE];
        view[AGENT_VIEW_ROW - 1] = [Entity::Wall; VIEW_SIZE];
        view[AGENT_VIEW_ROW - 1][AGENT_VIEW_COL] = centre;
        view
    }

    #[test]
    fn process_vis_empty_view_hides_nothing() {
        let view = [[Entity::Empty; VIEW_SIZE]; VIEW_SIZE];
        assert_eq!(
            process_vis(view),
            all_visible(view),
            "with no occluder anywhere, every cell of the window is visible"
        );
    }

    /// Canonical `Grid.process_vis`, transcribed **wrongly** on purpose: `i` and
    /// `j` are used directly as the two array indices, so `mask[i, j]` becomes
    /// `mask[i][j]` rather than `mask[row][col]`.
    ///
    /// This is the concrete bug the port's `# Index transposition` note exists
    /// to prevent, and it is self-consistent — the seed, the mask writes and the
    /// `view` reads are all swapped together, so it compiles, runs, and returns
    /// a plausible mask. Algebraically it computes
    /// `transpose(process_vis(transpose(view)))`, which is why so many fixtures
    /// cannot tell it apart from the real thing.
    ///
    /// It exists only so a fixture claiming to be transpose-detecting can be
    /// *demonstrated* to be, in-tree, instead of asserted to be by a comment.
    fn transposed_process_vis(
        view: [[Entity; VIEW_SIZE]; VIEW_SIZE],
    ) -> [[Option<Entity>; VIEW_SIZE]; VIEW_SIZE] {
        let mut mask = [[false; VIEW_SIZE]; VIEW_SIZE];
        // Canonical `mask[agent_pos] = True` with `agent_pos = (col, row)`.
        mask[AGENT_VIEW_COL][AGENT_VIEW_ROW] = true;

        for j in (0..VIEW_SIZE).rev() {
            for i in 0..VIEW_SIZE - 1 {
                if !mask[i][j] || !view[i][j].see_behind() {
                    continue;
                }
                mask[i + 1][j] = true;
                if j > 0 {
                    mask[i + 1][j - 1] = true;
                    mask[i][j - 1] = true;
                }
            }
            for i in (1..VIEW_SIZE).rev() {
                if !mask[i][j] || !view[i][j].see_behind() {
                    continue;
                }
                mask[i - 1][j] = true;
                if j > 0 {
                    mask[i - 1][j - 1] = true;
                    mask[i][j - 1] = true;
                }
            }
        }

        std::array::from_fn(|row| {
            std::array::from_fn(|col| mask[row][col].then_some(view[row][col]))
        })
    }

    #[test]
    fn process_vis_wall_column_shadows_only_the_far_side() {
        // Layout — agent `A` at (row 6, col 3) looking toward row 0. Column 1
        // is a solid wall run; every other cell is Empty.
        //
        //   col      0  1  2  3  4  5  6
        //   row 0    .  #  .  .  .  .  .
        //   row 1    .  #  .  .  .  .  .
        //   ...
        //   row 6    .  #  .  A  .  .  .
        let mut view = [[Entity::Empty; VIEW_SIZE]; VIEW_SIZE];
        for row in &mut view {
            row[1] = Entity::Wall;
        }

        // Hand-computed and written out literally — NOT derived by running the
        // code under test. Trace: light starts at (6, 3); row 6's right-to-left
        // sub-pass walks it to col 2, which lights col 1 (the wall itself) and
        // then stops, because a wall propagates nothing. Nothing ever reaches
        // col 0. Each row out repeats the shape, since col 2 lights (row-1,
        // col 1) diagonally but the wall never feeds col 0.
        let expected = [[X, V, V, V, V, V, V]; VIEW_SIZE];

        let masked = process_vis(view);
        let actual = visibility(&masked);
        assert_eq!(
            actual, expected,
            "column 0 must lie entirely in the shadow of the wall run in column 1"
        );

        // The occluder is itself visible; only what is behind it is hidden.
        for (row, cells) in masked.iter().enumerate() {
            assert_eq!(
                cells[1],
                Some(Entity::Wall),
                "the agent sees the wall that blocks it, at row {row}"
            );
            assert_eq!(cells[0], None, "col 0 is hidden at row {row}");
        }

        // This fixture is **blind** to an index transposition, and that is
        // recorded here rather than left to be rediscovered. The expected mask
        // above differs from its own transpose, which once passed for a proof
        // that a `mask[col][row]` implementation could not produce it — it is
        // not one. The transposed transcription computes
        // `transpose(process_vis(transpose(view)))`, and transposing a
        // full-height wall column yields a full-width wall row whose mask
        // transposes straight back into this same array. See
        // `process_vis_offset_wall_stub_detects_a_transposed_implementation` for
        // the fixture that does discriminate.
        assert_eq!(
            visibility(&transposed_process_vis(view)),
            expected,
            "a full-height wall column cannot distinguish the two index orders — \
             if it now can, the algorithm changed and this note needs rewriting"
        );
    }

    #[test]
    fn process_vis_offset_wall_stub_detects_a_transposed_implementation() {
        // Layout — agent `A` at (row 6, col 3) looking toward row 0. A
        // three-cell wall stub in column 1, covering the agent's own row and the
        // two rows ahead of it; every other cell is Empty. The occluder is
        // deliberately offset from both the agent's column and the window
        // centre, and stops short of the far edge, so it has no symmetry of any
        // kind — least of all about the transpose.
        //
        //   col      0  1  2  3  4  5  6
        //   row 0    .  .  .  .  .  .  .
        //   row 1    .  .  .  .  .  .  .
        //   row 2    .  .  .  .  .  .  .
        //   row 3    .  .  .  .  .  .  .
        //   row 4    .  #  .  .  .  .  .
        //   row 5    .  #  .  .  .  .  .
        //   row 6    .  #  .  A  .  .  .
        let mut view = [[Entity::Empty; VIEW_SIZE]; VIEW_SIZE];
        for row in &mut view[AGENT_VIEW_ROW - 2..=AGENT_VIEW_ROW] {
            row[1] = Entity::Wall;
        }

        // Hand-computed and written out literally — NOT derived by running the
        // code under test. Trace: in rows 6, 5 and 4 the only routes into col 0
        // are the sideways spread through (row, col 1) and the diagonal from
        // (row + 1, col 1); both are the stub, which is lit but propagates
        // nothing, so col 0 stays dark for exactly as long as the stub lasts. At
        // row 3 the stub has ended: (3, 1) is Empty, is lit diagonally from
        // (4, 2), and its own right-to-left sub-pass lights (3, 0). Rows 2, 1
        // and 0 are then fully transparent and fully lit.
        let expected = [
            [V, V, V, V, V, V, V],
            [V, V, V, V, V, V, V],
            [V, V, V, V, V, V, V],
            [V, V, V, V, V, V, V],
            [X, V, V, V, V, V, V],
            [X, V, V, V, V, V, V],
            [X, V, V, V, V, V, V],
        ];

        let masked = process_vis(view);
        assert_eq!(
            visibility(&masked),
            expected,
            "the stub shadows col 0 only over its own three rows"
        );
        assert_eq!(
            masked[AGENT_VIEW_ROW][1],
            Some(Entity::Wall),
            "the stub is itself visible; only what is behind it is hidden"
        );

        // The whole point of this fixture, **demonstrated** rather than asserted:
        // run the transposed transcription over the identical window and watch
        // it disagree. Transposing the stub turns it into a horizontal run at
        // the far edge of the window, which casts no shadow back here at all —
        // so the buggy implementation reports everything visible.
        let transposed = visibility(&transposed_process_vis(view));
        assert_ne!(
            transposed, expected,
            "this fixture exists to separate `mask[row][col]` from `mask[col][row]`; \
             if the two now agree on it, it has stopped doing its job"
        );
        assert_eq!(
            transposed, [[V; VIEW_SIZE]; VIEW_SIZE],
            "the transposed transcription loses the shadow entirely here"
        );
    }

    #[test]
    fn process_vis_wall_row_ahead_hides_everything_behind_it() {
        let mut view = [[Entity::Empty; VIEW_SIZE]; VIEW_SIZE];
        view[AGENT_VIEW_ROW - 1] = [Entity::Wall; VIEW_SIZE];

        let masked = process_vis(view);

        let wall_row = &masked[AGENT_VIEW_ROW - 1];
        let agent_row = &masked[AGENT_VIEW_ROW];
        for (col, (&on_wall, &on_agent)) in wall_row.iter().zip(agent_row.iter()).enumerate() {
            assert_eq!(
                on_wall,
                Some(Entity::Wall),
                "the wall row itself is visible at col {col}"
            );
            assert_eq!(
                on_agent,
                Some(Entity::Empty),
                "the agent's own row is unaffected at col {col}"
            );
        }
        assert_unseen_beyond_the_wall_row(&masked, "the wall row");
    }

    /// Every cell strictly ahead of the occluding row is masked.
    fn assert_unseen_beyond_the_wall_row(
        masked: &[[Option<Entity>; VIEW_SIZE]; VIEW_SIZE],
        occluder: &str,
    ) {
        for (row, cells) in masked.iter().enumerate().take(AGENT_VIEW_ROW - 1) {
            for (col, &cell) in cells.iter().enumerate() {
                assert_eq!(
                    cell, None,
                    "({row}, {col}) lies behind {occluder} and must be unseen"
                );
            }
        }
    }

    #[test]
    fn process_vis_lone_wall_cell_casts_no_shadow() {
        // Canonical fidelity, and it looks like a bug until traced: a single
        // isolated wall does not hide the cell directly behind it, because the
        // neighbouring columns light it diagonally. Durable shadows need an
        // occluder that also blocks the sideways spread within a row.
        let mut view = [[Entity::Empty; VIEW_SIZE]; VIEW_SIZE];
        view[AGENT_VIEW_ROW - 1][AGENT_VIEW_COL] = Entity::Wall;

        assert_eq!(
            process_vis(view),
            all_visible(view),
            "a lone wall cell occludes nothing — light routes around it"
        );
    }

    #[test]
    fn process_vis_only_open_doors_let_sight_through() {
        let color = Color::Blue;

        let open = process_vis(wall_row_with_centre(Entity::Door(color, DoorState::Open)));
        assert_eq!(
            open[AGENT_VIEW_ROW - 2][AGENT_VIEW_COL],
            Some(Entity::Empty),
            "an open door does not occlude"
        );

        for shut in [DoorState::Closed, DoorState::Locked] {
            let masked = process_vis(wall_row_with_centre(Entity::Door(color, shut)));
            assert_eq!(
                masked[AGENT_VIEW_ROW - 1][AGENT_VIEW_COL],
                Some(Entity::Door(color, shut)),
                "the {shut:?} door itself stays visible"
            );
            assert_unseen_beyond_the_wall_row(&masked, &format!("a {shut:?} door"));
        }
    }

    #[test]
    fn process_vis_objects_and_lava_do_not_occlude() {
        // The counter-intuitive half of canonical `see_behind`: only Wall and
        // shut Doors override it. An agent sees straight past lava, and past a
        // key or ball lying on the floor.
        for transparent in [
            Entity::Lava,
            Entity::Key(Color::Yellow),
            Entity::Ball(Color::Red),
            Entity::Box(Color::Green),
            Entity::Goal,
            Entity::Floor,
        ] {
            let masked = process_vis(wall_row_with_centre(transparent));
            assert_eq!(
                masked[AGENT_VIEW_ROW - 1][AGENT_VIEW_COL],
                Some(transparent),
                "{transparent:?} is visible"
            );
            assert_eq!(
                masked[AGENT_VIEW_ROW - 2][AGENT_VIEW_COL],
                Some(Entity::Empty),
                "{transparent:?} must not occlude the cell behind it"
            );
        }
    }

    #[test]
    fn process_vis_agent_cell_is_always_visible() {
        // Canonical seeds `mask[agent_pos] = True` before reading any cell, so
        // the agent's own cell is visible even when it is opaque.
        let view = [[Entity::Wall; VIEW_SIZE]; VIEW_SIZE];
        let masked = process_vis(view);

        assert_eq!(
            masked[AGENT_VIEW_ROW][AGENT_VIEW_COL],
            Some(Entity::Wall),
            "the agent's own cell is seeded visible unconditionally"
        );
        let visible = masked.iter().flatten().filter(|c| c.is_some()).count();
        assert_eq!(
            visible, 1,
            "walled in on every side, the agent sees exactly its own cell"
        );
    }

    #[test]
    fn process_vis_treats_out_of_grid_cells_as_occluding_walls() {
        // `Grid::get` returns `Entity::Wall` out of bounds (matching canonical
        // `Grid.slice`), so the world edge occludes with no special case in
        // `process_vis`.
        let grid = Grid::new(1, 1);
        let agent = AgentState::new(0, 0, Direction::East);
        let masked = process_vis(egocentric_view(&grid, &agent));

        // Only the agent's cell, its two in-row neighbours (both out-of-grid
        // walls), and the three cells they light one row ahead survive.
        let expected = [
            [X, X, X, X, X, X, X],
            [X, X, X, X, X, X, X],
            [X, X, X, X, X, X, X],
            [X, X, X, X, X, X, X],
            [X, X, X, X, X, X, X],
            [X, X, V, V, V, X, X],
            [X, X, V, V, V, X, X],
        ];
        assert_eq!(
            visibility(&masked),
            expected,
            "the out-of-grid wall ring must occlude everything past it"
        );
        assert_eq!(
            masked[AGENT_VIEW_ROW][AGENT_VIEW_COL],
            Some(Entity::Empty),
            "the one in-grid cell is the agent's own"
        );
        assert_eq!(
            masked[AGENT_VIEW_ROW][AGENT_VIEW_COL - 1],
            Some(Entity::Wall),
            "the out-of-grid neighbour reads as a visible wall"
        );
    }
}
