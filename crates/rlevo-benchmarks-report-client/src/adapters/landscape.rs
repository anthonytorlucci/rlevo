//! Landscapes adapter (Sphere / Ackley / Rastrigin search visualisation).
//!
//! Consumes a [`FamilyPayload::Landscape2D`] payload and renders the
//! search domain as a bounded SVG viewport with:
//!
//! - the search rectangle (bounds_x × bounds_y);
//! - a trail polyline (most recent candidate positions, oldest first);
//! - the current candidate as a filled circle (cyan + bold stroke);
//! - the best-so-far as an open ring with a cross-hair (green) when
//!   present;
//! - a label badge surfacing the landscape's name (`sphere` / `ackley`
//!   / `rastrigin`) so colour-blind readers identify the surface from
//!   the wording alone.
//!
//! Per the project a11y contract every coloured element pairs with a
//! distinct shape: current = filled disk, best = open ring + cross,
//! trail = dim dashed polyline. A B/W screenshot still reads.
//!
//! [`FamilyPayload::Landscape2D`]: crate::wire::FamilyPayload::Landscape2D

use leptos::prelude::*;

use crate::wire::{FamilyPayload, FrameRecord, Landscape2DPayload};

/// SVG viewport size in user units. The transform maps the landscape
/// bounds onto `[VB_PAD, VB_SIZE - VB_PAD]` along both axes.
const VB_SIZE: f32 = 320.0;
const VB_PAD: f32 = 16.0;

/// Render one landscapes-family frame.
#[must_use]
pub fn render(frame: &FrameRecord) -> AnyView {
    match &frame.family_payload {
        FamilyPayload::Landscape2D(payload) => view_with_payload(payload),
        _ => super::fallback::render(crate::wire::EnvFamily::Landscapes, frame),
    }
    .into_any()
}

fn view_with_payload(payload: &Landscape2DPayload) -> AnyView {
    let (xlo, xhi) = payload.bounds_x;
    let (ylo, yhi) = payload.bounds_y;
    // Guard against degenerate bounds — emit a banner rather than
    // dividing by zero in the affine map.
    if (xhi - xlo).abs() < f32::EPSILON || (yhi - ylo).abs() < f32::EPSILON {
        return view! {
            <p class="rlevo-warnings">
                "landscape payload has degenerate bounds — cannot render"
            </p>
        }
        .into_any();
    }

    let xform = move |p: &crate::wire::Point2| {
        let nx = (p.x - xlo) / (xhi - xlo);
        // Flip y so the +y axis points up in the SVG (canvas y points down).
        let ny = 1.0 - (p.y - ylo) / (yhi - ylo);
        (
            VB_PAD + nx * (VB_SIZE - 2.0 * VB_PAD),
            VB_PAD + ny * (VB_SIZE - 2.0 * VB_PAD),
        )
    };

    let trail_points: Vec<(f32, f32)> = payload.trail.iter().map(xform).collect();
    let trail_str = trail_points
        .iter()
        .map(|(x, y)| format!("{x:.2},{y:.2}"))
        .collect::<Vec<_>>()
        .join(" ");

    let (cx, cy) = xform(&payload.current);
    let best_marker = payload.best.as_ref().map(|b| xform(b));

    let label = payload.label.clone();
    let label_display = label.clone();

    let view_box = format!("0 0 {VB_SIZE} {VB_SIZE}");
    let bg_w = VB_SIZE - 2.0 * VB_PAD;
    let bg_h = VB_SIZE - 2.0 * VB_PAD;

    view! {
        <figure class="rlevo-family-landscape">
            <svg
                class="rlevo-svg-frame"
                viewBox=view_box
                role="img"
                aria-label=format!("landscape search view — {label}")
            >
                <rect
                    x=VB_PAD
                    y=VB_PAD
                    width=bg_w
                    height=bg_h
                    class="rlevo-landscape-bg"
                />
                <polyline
                    points=trail_str
                    class="rlevo-landscape-trail"
                />
                {best_marker.map(|(bx, by)| view! {
                    <g class="rlevo-landscape-best">
                        <circle cx=bx cy=by r=7.0 fill="none" />
                        <line x1=bx - 5.0 y1=by x2=bx + 5.0 y2=by />
                        <line x1=bx y1=by - 5.0 x2=bx y2=by + 5.0 />
                    </g>
                })}
                <circle
                    cx=cx
                    cy=cy
                    r=5.0
                    class="rlevo-landscape-current"
                />
            </svg>
            <figcaption class="legend">
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-glyph rlevo-landscape-name">
                        {label_display}
                    </span>
                    " landscape"
                </span>
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-swatch rlevo-landscape-current-swatch" />
                    " current candidate"
                </span>
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-swatch rlevo-landscape-best-swatch" />
                    " best so far"
                </span>
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-swatch rlevo-landscape-trail-swatch" />
                    " recent trail"
                </span>
                <span class="rlevo-legend-key">
                    <em>"heatmap deferred — only candidate dynamics ship today"</em>
                </span>
            </figcaption>
        </figure>
    }
    .into_any()
}
