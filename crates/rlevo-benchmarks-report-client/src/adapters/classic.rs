//! Classic-control adapter — structured SVG line-art from a
//! [`FamilyPayload::Classic2D`] payload (ADR-0013).
//!
//! Each body is a world-space polyline; the adapter fits an affine map from
//! the payload's viewport `bounds` onto a padded SVG viewBox (flipping y so
//! physics-up renders up). Bodies render by role: `Track` as an open
//! polyline, `Cart` / `Car` as filled polygons, `Pole` / `Link` as thick
//! strokes, `Hinge` (a single point) as a small ring.
//!
//! Per the a11y contract each role pairs colour with a distinct shape /
//! stroke so a B/W screenshot still reads. Falls back to
//! [`super::fallback::render`] for any non-`Classic2D` payload (e.g. a legacy
//! `Ascii` record, or the bandit envs which stay on the ASCII path).
//!
//! [`FamilyPayload::Classic2D`]: crate::wire::FamilyPayload::Classic2D

use leptos::prelude::*;

use crate::wire::{
    Classic2DBody, Classic2DPayload, Classic2DRole, FamilyPayload, FrameRecord, Point2,
};

/// Square SVG viewport size in user units.  The viewBox is always `0 0 VB VB`.
const VB: f32 = 320.0;
/// Padding reserved on each edge of the viewBox, in the same user units as
/// [`VB`].  The drawable inner square is therefore `VB - 2 * PAD` on each
/// side.
const PAD: f32 = 16.0;

/// Renders one classic-family frame as a type-erased Leptos [`AnyView`].
///
/// Dispatches on the payload variant: [`FamilyPayload::Classic2D`] is handled
/// by the SVG path; all other variants fall back to
/// [`super::fallback::render`].  The returned view is ready to be mounted
/// directly into the Leptos component tree.
///
/// # Must use
///
/// `AnyView` is a reactive node that must be returned to the caller and
/// inserted into the view tree; silently dropping it means the frame is never
/// rendered.
///
/// [`FamilyPayload::Classic2D`]: crate::wire::FamilyPayload::Classic2D
#[must_use]
pub fn render(frame: &FrameRecord) -> AnyView {
    match &frame.family_payload {
        FamilyPayload::Classic2D(payload) => view_with_payload(payload),
        _ => super::fallback::render(crate::wire::EnvFamily::Classic, frame),
    }
    .into_any()
}

/// CSS class for a body role.
const fn role_class(role: Classic2DRole) -> &'static str {
    match role {
        Classic2DRole::Track => "rlevo-classic-track",
        Classic2DRole::Cart => "rlevo-classic-cart",
        Classic2DRole::Pole => "rlevo-classic-pole",
        Classic2DRole::Link => "rlevo-classic-link",
        Classic2DRole::Car => "rlevo-classic-car",
        Classic2DRole::Hinge => "rlevo-classic-hinge",
    }
}

/// Builds the SVG figure for a [`Classic2DPayload`].
///
/// Applies a uniform-scale affine map from the payload's world-space `bounds`
/// onto the padded inner square (`VB - 2*PAD`), centering the shorter axis.
/// The y-axis is flipped so physics-up renders as visually up in the SVG.
///
/// Returns a warning paragraph instead of an SVG when `bounds` is degenerate
/// (either dimension is zero or sub-epsilon).
fn view_with_payload(payload: &Classic2DPayload) -> AnyView {
    let (lo, hi) = payload.bounds;
    let (sx, sy) = (hi.x - lo.x, hi.y - lo.y);
    if sx.abs() < f32::EPSILON || sy.abs() < f32::EPSILON {
        return view! {
            <p class="rlevo-warnings">"classic payload has degenerate bounds — cannot render"</p>
        }
        .into_any();
    }
    // Uniform scale so the mechanism keeps its aspect ratio; centre the
    // shorter axis. Flip y (physics-up → SVG-down).
    let span = sx.max(sy);
    let inner = VB - 2.0 * PAD;
    let scale = inner / span;
    let off_x = PAD + (inner - sx * scale) * 0.5;
    let off_y = PAD + (inner - sy * scale) * 0.5;
    let xform = move |p: &Point2| {
        let px = off_x + (p.x - lo.x) * scale;
        let py = off_y + (hi.y - p.y) * scale; // flip
        (px, py)
    };

    let bodies: Vec<AnyView> = payload
        .bodies
        .iter()
        .map(|b| body_view(b, &xform))
        .collect();
    let view_box = format!("0 0 {VB} {VB}");

    view! {
        <figure class="rlevo-family-classic">
            <svg class="rlevo-svg-frame" viewBox=view_box role="img" aria-label="classic control view">
                {bodies}
            </svg>
            <figcaption class="legend">
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-swatch rlevo-classic-cart-swatch" />
                    " cart / car"
                </span>
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-swatch rlevo-classic-pole-swatch" />
                    " pole / link"
                </span>
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-swatch rlevo-classic-track-swatch" />
                    " track / terrain"
                </span>
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-glyph rlevo-classic-hinge-fg">"\u{25cb}"</span>
                    " hinge / pivot"
                </span>
            </figcaption>
        </figure>
    }
    .into_any()
}

/// Renders one body: a ring for a single-point hinge, a polygon when closed,
/// otherwise an open polyline.
fn body_view(body: &Classic2DBody, xform: &impl Fn(&Point2) -> (f32, f32)) -> AnyView {
    let cls = role_class(body.role);
    if body.points.len() == 1 {
        let (cx, cy) = xform(&body.points[0]);
        return view! { <circle cx=cx cy=cy r=4.0 class=cls /> }.into_any();
    }
    let pts = body
        .points
        .iter()
        .map(|p| {
            let (x, y) = xform(p);
            format!("{x:.2},{y:.2}")
        })
        .collect::<Vec<_>>()
        .join(" ");
    if body.closed {
        view! { <polygon points=pts class=cls /> }.into_any()
    } else {
        view! { <polyline points=pts class=cls /> }.into_any()
    }
}
