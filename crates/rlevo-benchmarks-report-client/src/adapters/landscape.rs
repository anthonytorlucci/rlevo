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

use crate::series::landscape_field;
use crate::wire::{FamilyPayload, FrameRecord, Landscape2DPayload};

/// SVG viewport size in user units (square canvas).
const VB_SIZE: f32 = 320.0;
/// Padding inside the viewBox on each edge; the affine map places the
/// landscape bounds onto `[VB_PAD, VB_SIZE − VB_PAD]` along both axes.
const VB_PAD: f32 = 16.0;
/// Heatmap grid resolution (`HEATMAP_N × HEATMAP_N` cells).
const HEATMAP_N: usize = 24;

/// Renders one landscapes-family frame, dispatching on the payload variant.
///
/// Extracts a [`FamilyPayload::Landscape2D`] payload and forwards it to
/// [`view_with_payload`].  Any other variant falls through to
/// [`super::fallback::render`].
///
/// [`FamilyPayload::Landscape2D`]: crate::wire::FamilyPayload::Landscape2D
#[must_use]
pub fn render(frame: &FrameRecord) -> AnyView {
    match &frame.family_payload {
        FamilyPayload::Landscape2D(payload) => view_with_payload(payload),
        _ => super::fallback::render(crate::wire::EnvFamily::Landscapes, frame),
    }
    .into_any()
}

/// Builds the full SVG figure for a [`Landscape2DPayload`].
///
/// Computes a normalised affine map from landscape coordinates to SVG user
/// units, flipping the y-axis so physics-up becomes SVG-down.  Renders (in
/// paint order): background rectangle, trail polyline, best-so-far open ring
/// with cross-hair (when present), and the current candidate as a filled
/// disk.  Returns an error paragraph if either axis of `bounds` has zero
/// span.
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

    // Closed-form heatmap background for known surfaces (sphere / ackley /
    // rastrigin). Monochrome ramp: basins (low value) bright, ridges dark, so
    // the topography reads in B/W and the candidate markers stay visible on
    // top. Unknown surfaces produce no field and the deferred note shows.
    let field = landscape_field(&payload.label, payload.bounds_x, payload.bounds_y, HEATMAP_N);
    let has_heatmap = field.is_some();
    let heatmap_cells: Vec<AnyView> = field
        .map(|f| {
            let cell = bg_w / f.n as f32;
            let mut out = Vec::with_capacity(f.n * f.n);
            for row in 0..f.n {
                for col in 0..f.n {
                    let t = f.cells[row * f.n + col];
                    // t=0 (best) → bright (~235), t=1 (worst) → dark (~55).
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let g = (235.0 - t * 180.0).clamp(0.0, 255.0) as u8;
                    let fill = format!("#{g:02x}{g:02x}{g:02x}");
                    let x = VB_PAD + col as f32 * cell;
                    let y = VB_PAD + row as f32 * cell;
                    // +0.6 overlap hides hairline seams between cells.
                    out.push(
                        view! {
                            <rect x=x y=y width=cell + 0.6 height=cell + 0.6 fill=fill />
                        }
                        .into_any(),
                    );
                }
            }
            out
        })
        .unwrap_or_default();

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
                <g class="rlevo-landscape-heatmap">{heatmap_cells}</g>
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
                    {if has_heatmap {
                        view! { <em>"heatmap: bright = lower (better) objective value"</em> }.into_any()
                    } else {
                        view! { <em>"heatmap unavailable for this surface — candidate dynamics only"</em> }.into_any()
                    }}
                </span>
            </figcaption>
        </figure>
    }
    .into_any()
}
