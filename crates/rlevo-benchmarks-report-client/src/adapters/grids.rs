//! Grids adapter — structured SVG render of a Minigrid-style tile grid.
//!
//! Consumes a [`FamilyPayload::Grid`] payload (ADR-0013) and draws one
//! `<rect>` per cell, the agent as a rotated triangle indicating heading,
//! and pickable objects / doors / goal / lava as **shape-distinct** glyphs
//! laid over their cell.
//!
//! Per the project a11y contract every coloured element pairs with a
//! redundant non-colour signal: the agent carries a triangle *shape* whose
//! rotation encodes heading, and every object is a distinct letter glyph
//! (`G` goal, `L` lava, `K` key, `D`/`d`/`/` door locked/closed/open,
//! `o` ball, `\u{25a1}` box) so a B/W screenshot still reads.
//!
//! Falls back to [`super::fallback::render`] for any non-`Grid` payload (e.g.
//! a legacy `Ascii` record from before the structured migration).
//!
//! [`FamilyPayload::Grid`]: crate::wire::FamilyPayload::Grid

use leptos::prelude::*;

use crate::wire::{
    FamilyPayload, FrameRecord, GridColor, GridDir, GridDoorState, GridPayload, GridTile,
};

/// Side length of one grid cell in SVG user units.
const CELL: f32 = 24.0;
/// Padding inside the viewBox on each edge.
const PAD: f32 = 4.0;

/// Renders one grids-family frame into a Leptos [`AnyView`].
///
/// Dispatches on the [`FamilyPayload`] variant:
///
/// - [`FamilyPayload::Grid`] — full structured SVG via `view_with_payload`.
/// - Any other variant — delegates to [`super::fallback::render`], which
///   renders a best-effort text representation and a family label.
///
/// Always returns a valid view; this function is infallible.
#[must_use]
pub fn render(frame: &FrameRecord) -> AnyView {
    match &frame.family_payload {
        FamilyPayload::Grid(payload) => view_with_payload(payload),
        _ => super::fallback::render(crate::wire::EnvFamily::Grids, frame),
    }
    .into_any()
}

/// Background CSS class for a tile (the `<rect>` fill).
const fn tile_class(tile: GridTile) -> &'static str {
    match tile {
        GridTile::Wall => "rlevo-grid-wall",
        GridTile::Goal => "rlevo-grid-goal",
        GridTile::Lava => "rlevo-grid-lava",
        // Floors, objects-on-floor, and empties share the floor background;
        // the object itself is drawn as an overlay glyph.
        _ => "rlevo-grid-floor",
    }
}

/// Maps a Minigrid colour onto its CSS class (paired with a glyph so colour
/// is never the sole signal).
const fn color_class(c: GridColor) -> &'static str {
    match c {
        GridColor::Red => "rlevo-grid-red",
        GridColor::Green => "rlevo-grid-green",
        GridColor::Blue => "rlevo-grid-blue",
        GridColor::Purple => "rlevo-grid-purple",
        GridColor::Yellow => "rlevo-grid-yellow",
        GridColor::Grey => "rlevo-grid-grey",
    }
}

/// Returns the overlay glyph letter and its CSS colour class for a tile, or
/// `None` for tiles that carry no object overlay (floor, wall, empty, and
/// any variant not explicitly listed).
///
/// The glyph is always a distinct shape so colour is never the sole signal
/// (per the project a11y contract in the module doc).
const fn object_glyph(tile: GridTile) -> Option<(&'static str, &'static str)> {
    match tile {
        GridTile::Goal => Some(("G", "rlevo-grid-goal-fg")),
        GridTile::Lava => Some(("L", "rlevo-grid-lava-fg")),
        GridTile::Key(c) => Some(("K", color_class(c))),
        GridTile::Ball(c) => Some(("o", color_class(c))),
        GridTile::Box(c) => Some(("\u{25a1}", color_class(c))),
        GridTile::Door(c, GridDoorState::Locked) => Some(("D", color_class(c))),
        GridTile::Door(c, GridDoorState::Closed) => Some(("d", color_class(c))),
        GridTile::Door(c, GridDoorState::Open) => Some(("/", color_class(c))),
        _ => None,
    }
}

/// Computes the three SVG `points` attribute vertices for an agent triangle
/// centred in grid cell `(ax, ay)` and pointing in `dir`.
///
/// Coordinate convention: SVG origin is top-left; `x` increases east (right)
/// and `y` increases south (down), matching Minigrid's row-major tile layout.
/// `dir` maps to the tip vertex of the triangle — `East` tips right, `South`
/// tips down, and so on — so the triangle visually encodes heading.
///
/// Returns a space-separated `"x1,y1 x2,y2 x3,y3"` string suitable for the
/// `<polygon points="…">` attribute.
fn agent_points(ax: u16, ay: u16, dir: GridDir) -> String {
    let cx = PAD + f32::from(ax) * CELL + CELL / 2.0;
    let cy = PAD + f32::from(ay) * CELL + CELL / 2.0;
    let r = CELL * 0.34;
    // Tip + two base corners, oriented per heading.
    let (tx, ty, b1x, b1y, b2x, b2y) = match dir {
        GridDir::East => (cx + r, cy, cx - r, cy - r, cx - r, cy + r),
        GridDir::West => (cx - r, cy, cx + r, cy - r, cx + r, cy + r),
        GridDir::South => (cx, cy + r, cx - r, cy - r, cx + r, cy - r),
        GridDir::North => (cx, cy - r, cx - r, cy + r, cx + r, cy + r),
    };
    format!("{tx:.2},{ty:.2} {b1x:.2},{b1y:.2} {b2x:.2},{b2y:.2}")
}

/// Builds the full SVG `<figure>` for a [`GridPayload`].
///
/// Returns a `<p class="rlevo-warnings">` error node if the payload is
/// degenerate: either dimension is zero, or `tiles.len()` does not equal
/// `width × height` (which would indicate a serialisation mismatch between
/// the record producer and this client).
///
/// On success produces one `<rect>` per cell, overlay glyphs for objects,
/// the agent `<polygon>` triangle, and a `<figcaption>` legend.
fn view_with_payload(payload: &GridPayload) -> AnyView {
    let w = payload.width;
    let h = payload.height;
    if w == 0 || h == 0 || payload.tiles.len() != usize::from(w) * usize::from(h) {
        return view! {
            <p class="rlevo-warnings">
                "grid payload has inconsistent dimensions — cannot render"
            </p>
        }
        .into_any();
    }

    let vb_w = f32::from(w) * CELL + 2.0 * PAD;
    let vb_h = f32::from(h) * CELL + 2.0 * PAD;
    let view_box = format!("0 0 {vb_w} {vb_h}");

    // One <rect> per cell, plus an overlay <text> glyph for objects.
    let mut cells: Vec<AnyView> = Vec::with_capacity(payload.tiles.len());
    let mut glyphs: Vec<AnyView> = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let tile = payload.tiles[usize::from(y) * usize::from(w) + usize::from(x)];
            let rx = PAD + f32::from(x) * CELL;
            let ry = PAD + f32::from(y) * CELL;
            cells.push(
                view! { <rect x=rx y=ry width=CELL height=CELL class=tile_class(tile) /> }
                    .into_any(),
            );
            if let Some((glyph, cls)) = object_glyph(tile) {
                let gx = rx + CELL / 2.0;
                let gy = ry + CELL / 2.0;
                glyphs.push(
                    view! {
                        <text
                            x=gx
                            y=gy
                            class=format!("rlevo-grid-glyph {cls}")
                            text-anchor="middle"
                            dominant-baseline="central"
                        >
                            {glyph}
                        </text>
                    }
                    .into_any(),
                );
            }
        }
    }

    let agent_pts = agent_points(payload.agent.x, payload.agent.y, payload.agent.dir);

    view! {
        <figure class="rlevo-family-grids">
            <svg
                class="rlevo-svg-frame"
                viewBox=view_box
                role="img"
                aria-label="grid environment view"
            >
                {cells}
                {glyphs}
                <polygon points=agent_pts class="rlevo-grid-agent" />
            </svg>
            <figcaption class="legend">
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-glyph rlevo-grid-agent-fg">"\u{25b6}"</span>
                    " agent (triangle points where it faces)"
                </span>
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-swatch rlevo-grid-wall-swatch" />
                    " wall"
                </span>
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-glyph rlevo-grid-goal-fg">"G"</span>
                    " goal"
                </span>
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-glyph rlevo-grid-lava-fg">"L"</span>
                    " lava (hazard)"
                </span>
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-glyph">"K / D / o / \u{25a1}"</span>
                    " key / door / ball / box (colour = item colour)"
                </span>
            </figcaption>
        </figure>
    }
    .into_any()
}
