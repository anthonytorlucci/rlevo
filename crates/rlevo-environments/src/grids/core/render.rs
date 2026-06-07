//! Minimal ASCII renderer for grid environments.
//!
//! This is intentionally pure: callers pass a `(Grid, AgentState)` pair
//! and receive a `String` or `StyledFrame`. No terminal control sequences,
//! no curses, no blocking IO — printing is the caller's responsibility.

use super::agent::AgentState;
use super::color::Color as GridEntityColor;
use super::direction::Direction;
use super::entity::{DoorState, Entity};
use super::grid::Grid;
use rlevo_core::render::payload::{
    GridAgentMarker, GridColor, GridDir, GridDoorState, GridSnapshot, GridTile,
};
use crate::render::palette::{
    AGENT_FG, AGENT_MODIFIER, GOAL_FG, GOAL_MODIFIER, HAZARD_FG, HAZARD_MODIFIER, WALL_FG,
};
use crate::render::{SpanStyle, StyledFrame, StyledLine, StyledSpan};

/// Project a `(Grid, AgentState)` pair into a structured [`GridSnapshot`]
/// for the report tier (ADR-0013). Pure data — the env-side `Entity` /
/// `Color` / `Direction` types map onto the wire-neutral payload enums in
/// `rlevo-core::render::payload`.
///
/// Grid dimensions are stored as `u16`; grids larger than 65 535 cells in
/// either dimension are not supported in practice (all built-in environments
/// are far smaller).
#[must_use]
pub fn grid_snapshot(grid: &Grid, agent: &AgentState) -> GridSnapshot {
    let width = grid.width();
    let height = grid.height();
    let mut tiles = Vec::with_capacity(width * height);
    for y in 0..height as i32 {
        for x in 0..width as i32 {
            tiles.push(entity_to_tile(grid.get(x, y)));
        }
    }
    GridSnapshot {
        width: width as u16,
        height: height as u16,
        tiles,
        agent: GridAgentMarker {
            x: agent.x.max(0) as u16,
            y: agent.y.max(0) as u16,
            dir: dir_to_payload(agent.direction),
            carrying: agent.carrying.map(entity_to_tile),
        },
    }
}

const fn entity_to_tile(entity: Entity) -> GridTile {
    match entity {
        Entity::Empty => GridTile::Empty,
        Entity::Floor => GridTile::Floor,
        Entity::Wall => GridTile::Wall,
        Entity::Goal => GridTile::Goal,
        Entity::Lava => GridTile::Lava,
        Entity::Door(c, s) => GridTile::Door(color_to_payload(c), door_to_payload(s)),
        Entity::Key(c) => GridTile::Key(color_to_payload(c)),
        Entity::Ball(c) => GridTile::Ball(color_to_payload(c)),
        Entity::Box(c) => GridTile::Box(color_to_payload(c)),
    }
}

const fn color_to_payload(c: GridEntityColor) -> GridColor {
    match c {
        GridEntityColor::Red => GridColor::Red,
        GridEntityColor::Green => GridColor::Green,
        GridEntityColor::Blue => GridColor::Blue,
        GridEntityColor::Purple => GridColor::Purple,
        GridEntityColor::Yellow => GridColor::Yellow,
        GridEntityColor::Grey => GridColor::Grey,
    }
}

const fn door_to_payload(s: DoorState) -> GridDoorState {
    match s {
        DoorState::Open => GridDoorState::Open,
        DoorState::Closed => GridDoorState::Closed,
        DoorState::Locked => GridDoorState::Locked,
    }
}

const fn dir_to_payload(d: Direction) -> GridDir {
    match d {
        Direction::East => GridDir::East,
        Direction::South => GridDir::South,
        Direction::West => GridDir::West,
        Direction::North => GridDir::North,
    }
}

/// Render the grid and the agent's position to a multi-line ASCII string.
///
/// Each cell is two characters wide (glyph + space) and each row ends with
/// `'\n'`. The agent's position overrides the underlying entity glyph.
/// Glyph mapping:
///
/// | Glyph | Entity |
/// |-------|--------|
/// | `#`   | Wall |
/// | `.`   | Empty or Floor |
/// | `G`   | Goal |
/// | `L`   | Lava |
/// | `/`   | Door (open) |
/// | `+`   | Door (closed) |
/// | `*`   | Door (locked) |
/// | `k`   | Key |
/// | `o`   | Ball |
/// | `[`   | Box |
/// | `>` `v` `<` `^` | Agent facing East / South / West / North |
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

/// Render the grid and the agent's position into a [`StyledFrame`].
///
/// The output's glyphs are identical to [`render_ascii`] — projecting each
/// styled line back to a plain string via `StyledFrame::plain_text` yields
/// the same characters in the same order. The styling rules are:
///
/// - **walls (`#`)** carry `WALL_FG` so they fade into the background;
/// - **goal (`G`)** carries `GOAL_FG | GOAL_MODIFIER`;
/// - **lava / hazard (`L`)** carries `HAZARD_FG | HAZARD_MODIFIER` — the
///   `REVERSED` modifier ensures the cell flashes as a solid block, giving
///   a hue-redundant signal for deuteranopic users (paired with red/green
///   goal–hazard glyphs);
/// - **agent (`< > ^ v`)** carries `AGENT_FG | AGENT_MODIFIER`;
/// - **interactive entities (doors, keys, balls, boxes) and empty cells**
///   are emitted unstyled — entity colour is preserved in the
///   `family_payload` channel of `FrameRecord` and rendered by the report
///   tier rather than collapsing it into a single ANSI cell here.
#[must_use]
pub fn render_styled(grid: &Grid, agent: &AgentState) -> StyledFrame {
    #[allow(clippy::cast_possible_wrap)]
    let height = grid.height() as i32;
    #[allow(clippy::cast_possible_wrap)]
    let width = grid.width() as i32;

    let mut lines = Vec::with_capacity(grid.height());
    for y in 0..height {
        let mut spans: Vec<StyledSpan> = Vec::new();
        let mut current_style = SpanStyle::default();
        let mut current_text = String::with_capacity(grid.width() * 2);
        for x in 0..width {
            let (ch, style) = if x == agent.x && y == agent.y {
                (agent_char(agent), agent_style())
            } else {
                glyph_for_entity(grid.get(x, y))
            };
            if style != current_style && !current_text.is_empty() {
                spans.push(StyledSpan::new(std::mem::take(&mut current_text), current_style));
            }
            current_style = style;
            current_text.push(ch);
            current_text.push(' ');
        }
        if !current_text.is_empty() {
            spans.push(StyledSpan::new(current_text, current_style));
        }
        lines.push(StyledLine::from_spans(spans));
    }
    StyledFrame { lines }
}

fn agent_style() -> SpanStyle {
    SpanStyle::default()
        .fg(AGENT_FG)
        .with_modifier(AGENT_MODIFIER)
}

fn glyph_for_entity(e: Entity) -> (char, SpanStyle) {
    let ch = entity_char(e);
    let style = match e {
        Entity::Wall => SpanStyle::default().fg(WALL_FG),
        Entity::Goal => SpanStyle::default()
            .fg(GOAL_FG)
            .with_modifier(GOAL_MODIFIER),
        Entity::Lava => SpanStyle::default()
            .fg(HAZARD_FG)
            .with_modifier(HAZARD_MODIFIER),
        Entity::Empty
        | Entity::Floor
        | Entity::Door(_, _)
        | Entity::Key(_)
        | Entity::Ball(_)
        | Entity::Box(_) => SpanStyle::default(),
    };
    (ch, style)
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
    fn grid_snapshot_projects_tiles_and_agent() {
        use rlevo_core::render::payload::{GridColor, GridDir, GridTile};

        let mut g = Grid::new(3, 2);
        g.set(0, 0, Entity::Wall);
        g.set(2, 1, Entity::Goal);
        g.set(1, 0, Entity::Key(Color::Blue));
        let a = AgentState::new(1, 1, Direction::North);

        let snap = grid_snapshot(&g, &a);

        assert_eq!(snap.width, 3);
        assert_eq!(snap.height, 2);
        assert_eq!(snap.tiles.len(), 6);
        // Row-major: (x, y) -> tiles[y * width + x].
        assert_eq!(snap.tiles[0], GridTile::Wall); // (0,0)
        assert_eq!(snap.tiles[1], GridTile::Key(GridColor::Blue)); // (1,0)
        assert_eq!(snap.tiles[2 + 3], GridTile::Goal); // (2,1)
        assert_eq!(snap.agent.x, 1);
        assert_eq!(snap.agent.y, 1);
        assert_eq!(snap.agent.dir, GridDir::North);
        assert_eq!(snap.agent.carrying, None);
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

    #[test]
    fn render_styled_matches_render_ascii() {
        let mut g = Grid::new(5, 3);
        g.draw_walls();
        g.set(2, 1, Entity::Goal);
        let agent = AgentState::new(1, 1, Direction::North);

        let plain = render_ascii(&g, &agent);
        let styled = render_styled(&g, &agent);
        assert_eq!(styled.plain_text(), plain.trim_end_matches('\n'));
    }

    #[test]
    fn render_styled_classifies_glyphs_by_palette() {
        let mut g = Grid::new(5, 1);
        g.set(0, 0, Entity::Wall);
        g.set(1, 0, Entity::Goal);
        g.set(2, 0, Entity::Lava);
        g.set(3, 0, Entity::Empty);
        g.set(4, 0, Entity::Key(Color::Red));
        let agent = AgentState::new(100, 100, Direction::East); // off-grid

        let styled = render_styled(&g, &agent);
        assert_eq!(styled.lines.len(), 1);

        let wall = styled.lines[0]
            .spans
            .iter()
            .find(|s| s.text.starts_with('#'))
            .expect("wall span present");
        assert_eq!(wall.style.fg, Some(WALL_FG));

        let goal = styled.lines[0]
            .spans
            .iter()
            .find(|s| s.text.starts_with('G'))
            .expect("goal span present");
        assert_eq!(goal.style.fg, Some(GOAL_FG));
        assert!(goal.style.modifier.contains(GOAL_MODIFIER));

        let lava = styled.lines[0]
            .spans
            .iter()
            .find(|s| s.text.starts_with('L'))
            .expect("lava span present");
        assert_eq!(lava.style.fg, Some(HAZARD_FG));
        assert!(lava.style.modifier.contains(HAZARD_MODIFIER));
    }

    #[test]
    fn render_styled_agent_glyph_uses_agent_palette() {
        let mut g = Grid::new(3, 3);
        g.draw_walls();
        let agent = AgentState::new(1, 1, Direction::East);

        let styled = render_styled(&g, &agent);
        let agent_span = styled
            .lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .find(|s| s.text.starts_with('>'))
            .expect("agent glyph span present");
        assert_eq!(agent_span.style.fg, Some(AGENT_FG));
        assert!(agent_span.style.modifier.contains(AGENT_MODIFIER));
    }
}
