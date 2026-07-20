//! Shared ASCII / styled renderer for `Box2D` physics envs.
//!
//! Each `Box2D` env (`LunarLander`, `BipedalWalker`, `CarRacing`) holds rapier2d
//! bodies in continuous 2D space. This renderer projects body centres onto
//! a `CELL_COLS × CELL_ROWS` grid spanning a world-space viewport and
//! plots one glyph per body. The agent body additionally carries an
//! 8-direction arrow derived from its rotation angle.
//!
//! The renderer is intentionally lightweight — it shows body *positions*
//! and the agent's orientation, not body geometry. Polygon rasterisation
//! into ASCII is out of scope; the report tier owns full geometric
//! rendering via `FamilyPayload::Box2D` and a richer SVG / canvas
//! adapter.
//!
//! ## Glyph and palette key
//!
//! - **Agent** — one of `→ ↗ ↑ ↖ ← ↙ ↓ ↘` based on rotation; styled
//!   [`AGENT_FG`] + [`AGENT_MODIFIER`]. Agents outside the viewport are
//!   omitted from the grid (no edge-marker fallback).
//! - **Other dynamic bodies** — `o` styled [`Color::Cyan`].
//! - **Ground / static line** — bottom row of `─` styled [`WALL_FG`].
//! - **Empty** — space.

use crate::render::palette::{AGENT_FG, AGENT_MODIFIER, WALL_FG};
use crate::render::{Color, SpanStyle, StyledFrame, StyledLine, StyledSpan};

/// Columns in the rendered viewport.
pub const CELL_COLS: usize = 60;
/// Rows in the rendered viewport (excluding the header line).
pub const CELL_ROWS: usize = 14;

/// World-space rectangle that maps onto the cell grid.
#[derive(Debug, Clone, Copy)]
pub struct Viewport {
    /// Minimum world-X visible at column 0.
    pub x_min: f32,
    /// Maximum world-X visible at column `CELL_COLS - 1`.
    pub x_max: f32,
    /// Minimum world-Y visible at row `CELL_ROWS - 1` (bottom of frame).
    pub y_min: f32,
    /// Maximum world-Y visible at row 0 (top of frame).
    pub y_max: f32,
}

/// A body to be rendered.
#[derive(Debug, Clone, Copy)]
pub enum Bodyish {
    /// The controllable agent. Carries a rotation in radians so the
    /// renderer can pick an arrow glyph.
    Agent { x: f32, y: f32, angle_rad: f32 },
    /// A dynamic body that isn't the agent (e.g., a wheel or leg).
    Dynamic { x: f32, y: f32 },
}

fn project(x: f32, y: f32, vp: Viewport) -> Option<(usize, usize)> {
    if vp.x_max <= vp.x_min || vp.y_max <= vp.y_min {
        return None;
    }
    let tx = (x - vp.x_min) / (vp.x_max - vp.x_min);
    let ty = (y - vp.y_min) / (vp.y_max - vp.y_min);
    if !(0.0..=1.0).contains(&tx) || !(0.0..=1.0).contains(&ty) {
        return None;
    }
    #[allow(clippy::cast_precision_loss)]
    let col = (tx * (CELL_COLS as f32 - 1.0)).round() as usize;
    #[allow(clippy::cast_precision_loss)]
    let row = CELL_ROWS - 1 - (ty * (CELL_ROWS as f32 - 1.0)).round() as usize;
    Some((col.min(CELL_COLS - 1), row.min(CELL_ROWS - 1)))
}

/// Pick one of 8 arrow glyphs based on `angle_rad` (radians, CCW from +X).
#[must_use]
pub fn arrow_glyph(angle_rad: f32) -> char {
    use std::f32::consts::PI;
    let two_pi = 2.0 * PI;
    let mut a = angle_rad % two_pi;
    if a < 0.0 {
        a += two_pi;
    }
    // 8 sectors, each PI/4 wide, centred on the cardinal/diagonal directions.
    let sector = ((a + PI / 8.0) / (PI / 4.0)).floor() as usize % 8;
    match sector {
        0 => '→',
        1 => '↗',
        2 => '↑',
        3 => '↖',
        4 => '←',
        5 => '↙',
        6 => '↓',
        7 => '↘',
        _ => '?',
    }
}

fn header_line(label: &str, agent_x: f32, agent_y: f32, angle_deg: f32, step: usize) -> String {
    format!("{label}  pos=({agent_x:.1}, {agent_y:.1})  angle={angle_deg:>4.0}°  step={step}",)
}

fn rasterise(
    bodies: &[Bodyish],
    viewport: Viewport,
    ground_y: Option<f32>,
) -> Vec<Vec<(char, Glyph)>> {
    let mut grid: Vec<Vec<(char, Glyph)>> = vec![vec![(' ', Glyph::Empty); CELL_COLS]; CELL_ROWS];

    // Ground line (drawn first so bodies overwrite it where they overlap).
    if let Some(gy) = ground_y {
        let world_h = viewport.y_max - viewport.y_min;
        if world_h > 0.0 {
            let ty = (gy - viewport.y_min) / world_h;
            if (0.0..=1.0).contains(&ty) {
                #[allow(clippy::cast_precision_loss)]
                let ground_row = CELL_ROWS - 1 - (ty * (CELL_ROWS as f32 - 1.0)).round() as usize;
                let row = ground_row.min(CELL_ROWS - 1);
                for cell in &mut grid[row] {
                    *cell = ('─', Glyph::Static);
                }
            }
        }
    }

    // Bodies (agent last so it wins overlaps).
    for body in bodies
        .iter()
        .filter(|b| matches!(b, Bodyish::Dynamic { .. }))
    {
        if let Bodyish::Dynamic { x, y } = *body
            && let Some((col, row)) = project(x, y, viewport)
        {
            grid[row][col] = ('o', Glyph::Dynamic);
        }
    }
    for body in bodies.iter().filter(|b| matches!(b, Bodyish::Agent { .. })) {
        if let Bodyish::Agent { x, y, angle_rad } = *body
            && let Some((col, row)) = project(x, y, viewport)
        {
            grid[row][col] = (arrow_glyph(angle_rad), Glyph::Agent);
        }
    }
    grid
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Glyph {
    Empty,
    Static,
    Dynamic,
    Agent,
}

fn glyph_style(g: Glyph) -> SpanStyle {
    match g {
        Glyph::Empty => SpanStyle::default(),
        Glyph::Static => SpanStyle::default().fg(WALL_FG),
        Glyph::Dynamic => SpanStyle::default().fg(Color::Cyan),
        Glyph::Agent => SpanStyle::default()
            .fg(AGENT_FG)
            .with_modifier(AGENT_MODIFIER),
    }
}

/// Render a `Box2D` scene as a plain UTF-8 string.
///
/// Returns a header line followed by [`CELL_ROWS`] grid lines, each
/// [`CELL_COLS`] characters wide, separated by `\n`.
///
/// # Parameters
///
/// - `label` — short environment name shown at the start of the header
///   (e.g., `"LunarLander"`, `"BipedalWalker"`).
/// - `bodies` — slice of [`Bodyish`] values describing every body to
///   render. The first `Agent` variant found determines the header
///   position and orientation readout; if there is no agent body the
///   header defaults to `(0.0, 0.0, 0°)`.
/// - `viewport` — world-space rectangle that maps onto the cell grid.
/// - `ground_y` — if `Some(y)`, draws a horizontal `─` line at the
///   corresponding world-space Y coordinate. Pass `None` to omit it.
/// - `step` — episode step counter appended to the header.
#[must_use]
pub fn render_box2d_ascii(
    label: &str,
    bodies: &[Bodyish],
    viewport: Viewport,
    ground_y: Option<f32>,
    step: usize,
) -> String {
    let agent = bodies
        .iter()
        .find_map(|b| match b {
            Bodyish::Agent { x, y, angle_rad } => Some((*x, *y, angle_rad.to_degrees())),
            Bodyish::Dynamic { .. } => None,
        })
        .unwrap_or((0.0, 0.0, 0.0));

    let mut out = header_line(label, agent.0, agent.1, agent.2, step);
    out.push('\n');

    let grid = rasterise(bodies, viewport, ground_y);
    for (i, row) in grid.iter().enumerate() {
        for (ch, _) in row {
            out.push(*ch);
        }
        if i + 1 < grid.len() {
            out.push('\n');
        }
    }
    out
}

/// Render a `Box2D` scene as a [`StyledFrame`].
///
/// Produces the same layout and content as [`render_box2d_ascii`] but
/// wraps each run of identically styled characters in a [`StyledSpan`]
/// so the caller (e.g., a `ratatui` widget) can apply terminal colours
/// and modifiers without reparsing plain text. The header label is
/// styled with [`AGENT_FG`] + [`AGENT_MODIFIER`]; the position/angle
/// suffix is unstyled.
///
/// Parameters are identical to [`render_box2d_ascii`].
#[must_use]
pub fn render_box2d_styled(
    label: &str,
    bodies: &[Bodyish],
    viewport: Viewport,
    ground_y: Option<f32>,
    step: usize,
) -> StyledFrame {
    let agent = bodies
        .iter()
        .find_map(|b| match b {
            Bodyish::Agent { x, y, angle_rad } => Some((*x, *y, angle_rad.to_degrees())),
            Bodyish::Dynamic { .. } => None,
        })
        .unwrap_or((0.0, 0.0, 0.0));

    let mut lines: Vec<StyledLine> = Vec::with_capacity(CELL_ROWS + 1);

    // Header: label styled as agent.
    let header = header_line(label, agent.0, agent.1, agent.2, step);
    let label_style = SpanStyle::default()
        .fg(AGENT_FG)
        .with_modifier(AGENT_MODIFIER);
    let header_spans = if let Some(rest) = header.strip_prefix(label) {
        vec![
            StyledSpan::new(label, label_style),
            StyledSpan::raw(rest.to_string()),
        ]
    } else {
        vec![StyledSpan::raw(header)]
    };
    lines.push(StyledLine::from_spans(header_spans));

    let grid = rasterise(bodies, viewport, ground_y);
    for row in &grid {
        let mut spans: Vec<StyledSpan> = Vec::new();
        let mut current_style = SpanStyle::default();
        let mut current_text = String::with_capacity(CELL_COLS);
        for (ch, glyph) in row {
            let style = glyph_style(*glyph);
            if style != current_style && !current_text.is_empty() {
                spans.push(StyledSpan::new(
                    std::mem::take(&mut current_text),
                    current_style,
                ));
            }
            current_style = style;
            current_text.push(*ch);
        }
        if !current_text.is_empty() {
            spans.push(StyledSpan::new(current_text, current_style));
        }
        lines.push(StyledLine::from_spans(spans));
    }
    StyledFrame { lines }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn vp() -> Viewport {
        Viewport {
            x_min: 0.0,
            x_max: 20.0,
            y_min: 0.0,
            y_max: 13.3,
        }
    }

    #[test]
    fn agent_centre_projects_into_grid() {
        let bodies = [Bodyish::Agent {
            x: 10.0,
            y: 6.65,
            angle_rad: PI / 2.0,
        }];
        let out = render_box2d_ascii("Lander", &bodies, vp(), Some(0.0), 0);
        // Header + 14 grid rows.
        assert_eq!(out.lines().count(), 1 + CELL_ROWS);
        // Upright arrow somewhere in the grid.
        assert!(out.contains('↑'));
    }

    #[test]
    fn ground_line_drawn_at_y_min() {
        let bodies: [Bodyish; 0] = [];
        let out = render_box2d_ascii("Test", &bodies, vp(), Some(0.0), 0);
        // Last line should be all `─`.
        let last = out.lines().last().unwrap();
        assert!(last.chars().all(|c| c == '─'));
    }

    #[test]
    fn render_styled_matches_ascii() {
        let bodies = [Bodyish::Agent {
            x: 10.0,
            y: 9.7,
            angle_rad: 0.0,
        }];
        let plain = render_box2d_ascii("Lander", &bodies, vp(), Some(0.0), 7);
        let styled = render_box2d_styled("Lander", &bodies, vp(), Some(0.0), 7);
        let plain_no_trailing: String = plain.lines().collect::<Vec<_>>().join("\n");
        assert_eq!(styled.plain_text(), plain_no_trailing);
    }

    #[test]
    fn agent_styled_with_palette() {
        let bodies = [Bodyish::Agent {
            x: 10.0,
            y: 9.7,
            angle_rad: 0.0,
        }];
        let styled = render_box2d_styled("Lander", &bodies, vp(), Some(0.0), 0);
        let label = styled.lines[0]
            .spans
            .iter()
            .find(|s| s.text == "Lander")
            .expect("label present");
        assert_eq!(label.style.fg, Some(AGENT_FG));

        let agent_glyph = styled
            .lines
            .iter()
            .skip(1)
            .flat_map(|l| l.spans.iter())
            .find(|s| s.text.contains('→'))
            .expect("agent arrow present");
        assert_eq!(agent_glyph.style.fg, Some(AGENT_FG));
        assert!(agent_glyph.style.modifier.contains(AGENT_MODIFIER));
    }

    #[test]
    fn off_screen_agent_omitted() {
        let bodies = [Bodyish::Agent {
            x: -100.0,
            y: -100.0,
            angle_rad: 0.0,
        }];
        let out = render_box2d_ascii("Test", &bodies, vp(), None, 0);
        // No arrow glyph in body since the agent is far outside the viewport.
        for line in out.lines().skip(1) {
            for ch in line.chars() {
                assert!(!matches!(ch, '→' | '↗' | '↑' | '↖' | '←' | '↙' | '↓' | '↘'));
            }
        }
    }

    #[test]
    fn arrow_glyph_picks_8_directions() {
        assert_eq!(arrow_glyph(0.0), '→');
        assert_eq!(arrow_glyph(PI / 2.0), '↑');
        assert_eq!(arrow_glyph(PI), '←');
        assert_eq!(arrow_glyph(-PI / 2.0), '↓');
        assert_eq!(arrow_glyph(PI / 4.0), '↗');
    }

    #[test]
    fn header_within_width_budget() {
        let bodies = [Bodyish::Agent {
            x: 10.0,
            y: 9.7,
            angle_rad: PI,
        }];
        let out = render_box2d_ascii("Lander", &bodies, vp(), Some(0.0), 999);
        for line in out.lines() {
            assert!(
                line.chars().count() <= 80,
                "line exceeds 80 cols: {line:?} ({} chars)",
                line.chars().count()
            );
        }
    }
}
