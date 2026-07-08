//! `Box2D` adapter (`LunarLander`, `BipedalWalker`, `CarRacing`).
//!
//! Consumes a [`FamilyPayload::Box2dBodies`] payload and renders each
//! [`RigidBody2D`] as a transformed SVG `<polygon>`, with per-body
//! styling driven by the [`BodyKind`] discriminant. Contact points
//! sprinkle small `<circle>` markers; the world bounds drive the
//! viewBox so motion stays in frame.
//!
//! Per the project a11y contract, every body kind pairs hue with a
//! distinct stroke pattern: hulls are solid filled polygons, legs are
//! thin solid, wheels are filled circles, ground is dim and dashed,
//! goal is open and dashed. Contacts are open rings.
//!
//! [`FamilyPayload::Box2dBodies`]: crate::wire::FamilyPayload::Box2dBodies
//! [`RigidBody2D`]: crate::wire::RigidBody2D
//! [`BodyKind`]: crate::wire::BodyKind

use leptos::prelude::*;

use crate::wire::{BodyKind, Box2dPayload, FamilyPayload, FrameRecord, Point2, RigidBody2D};

/// SVG viewBox width in user units.
const VB_W: f32 = 480.0;
/// SVG viewBox height in user units.
const VB_H: f32 = 320.0;
/// Padding inside the viewBox on each edge, keeping bodies away from the border.
const VB_PAD: f32 = 12.0;

/// Render one box2d-family frame as an SVG figure.
///
/// Dispatches on [`FamilyPayload::Box2dBodies`]; any other variant falls
/// through to the generic [`super::fallback::render`] placeholder.
///
/// [`FamilyPayload::Box2dBodies`]: crate::wire::FamilyPayload::Box2dBodies
#[must_use]
pub fn render(frame: &FrameRecord) -> AnyView {
    match &frame.family_payload {
        FamilyPayload::Box2dBodies(payload) => view_with_payload(payload),
        _ => super::fallback::render(crate::wire::EnvFamily::Box2d, frame),
    }
    .into_any()
}

/// Maps a [`BodyKind`] to its CSS class name for styling and a11y stroke patterns.
///
/// The returned string is a static CSS class applied directly to the SVG
/// element.  Stylesheet rules keyed on these classes supply both hue and
/// stroke-pattern signals so the rendering is distinguishable without colour.
fn body_class(kind: BodyKind) -> &'static str {
    match kind {
        BodyKind::Hull => "rlevo-box2d-hull",
        BodyKind::Wheel => "rlevo-box2d-wheel",
        BodyKind::Leg => "rlevo-box2d-leg",
        BodyKind::Wing => "rlevo-box2d-wing",
        BodyKind::Ground => "rlevo-box2d-ground",
        BodyKind::Goal => "rlevo-box2d-goal",
        BodyKind::Other => "rlevo-box2d-other",
    }
}

/// Builds the full SVG figure for a [`Box2dPayload`], including bodies, contacts, and legend.
///
/// Computes a uniform scale from `world_bounds` so the entire scene fits
/// inside the viewBox with [`VB_PAD`] margin on each edge, then flips the
/// y-axis so physics-up maps to SVG-down.  Returns an error paragraph if
/// `world_bounds` is degenerate (zero span on either axis).
fn view_with_payload(payload: &Box2dPayload) -> AnyView {
    let (min, max) = payload.world_bounds;
    let span_x = max.x - min.x;
    let span_y = max.y - min.y;
    if span_x.abs() < f32::EPSILON || span_y.abs() < f32::EPSILON {
        return view! {
            <p class="rlevo-warnings">
                "box2d payload has degenerate world bounds — cannot render"
            </p>
        }
        .into_any();
    }

    // Fit world to viewBox preserving aspect ratio.
    let scale = ((VB_W - 2.0 * VB_PAD) / span_x).min((VB_H - 2.0 * VB_PAD) / span_y);
    let xform = move |p: &Point2| {
        let nx = (p.x - min.x) * scale;
        // Flip y so payload y (up) maps to SVG y (down increasing).
        let ny = (max.y - p.y) * scale;
        (VB_PAD + nx, VB_PAD + ny)
    };

    let bodies_svg = payload
        .bodies
        .iter()
        .map(|body| render_body(body, scale, xform))
        .collect::<Vec<_>>();

    let contacts_svg = payload
        .contacts
        .iter()
        .map(|c| {
            let (cx, cy) = xform(c);
            view! {
                <circle cx=cx cy=cy r=4.0 class="rlevo-box2d-contact" />
            }
        })
        .collect::<Vec<_>>();

    let view_box = format!("0 0 {VB_W} {VB_H}");

    view! {
        <figure class="rlevo-family-box2d">
            <svg
                class="rlevo-svg-frame rlevo-svg-box2d"
                viewBox=view_box
                role="img"
                aria-label="box2d world view"
            >
                {bodies_svg}
                {contacts_svg}
            </svg>
            <figcaption class="legend">
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-swatch rlevo-box2d-hull-swatch" />
                    " hull (filled)"
                </span>
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-swatch rlevo-box2d-leg-swatch" />
                    " leg (thin)"
                </span>
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-swatch rlevo-box2d-wheel-swatch" />
                    " wheel"
                </span>
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-swatch rlevo-box2d-ground-swatch" />
                    " ground (dashed)"
                </span>
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-swatch rlevo-box2d-contact-swatch" />
                    " contact"
                </span>
            </figcaption>
        </figure>
    }
    .into_any()
}

/// Renders a single [`RigidBody2D`] as an SVG `<polygon>` (plus a centre disk for wheels).
///
/// Rotates each local-frame vertex by `body.rotation_rad`, translates it by
/// `body.position`, then maps world coordinates to SVG coordinates with
/// `xform`.  Wheel bodies receive an additional filled `<circle>` whose
/// radius is derived from the maximum vertex extent so it never overflows the
/// polygon hull.
///
/// The `'static` bound on `F` is required because Leptos `view!` macros
/// capture closures into reactive nodes that may outlive the calling frame.
fn render_body<F>(body: &RigidBody2D, scale: f32, xform: F) -> AnyView
where
    F: Fn(&Point2) -> (f32, f32) + Copy + 'static,
{
    // Transform each local-frame vertex into world frame, then SVG.
    let (cos_t, sin_t) = (body.rotation_rad.cos(), body.rotation_rad.sin());
    let pts_svg = body
        .vertices
        .iter()
        .map(|v| {
            let wx = body.position.x + cos_t * v.x - sin_t * v.y;
            let wy = body.position.y + sin_t * v.x + cos_t * v.y;
            let (sx, sy) = xform(&Point2::new(wx, wy));
            format!("{sx:.2},{sy:.2}")
        })
        .collect::<Vec<_>>()
        .join(" ");

    let class = body_class(body.kind);

    // For wheels, also render a centre disk so the shape reads as round
    // even at small SVG sizes.
    let extra = if matches!(body.kind, BodyKind::Wheel) {
        let (cx, cy) = xform(&body.position);
        // Derive the wheel radius from the vertex extent so we never
        // overflow the polygon hull.
        let local_r = body
            .vertices
            .iter()
            .map(|v| (v.x * v.x + v.y * v.y).sqrt())
            .fold(0.0_f32, f32::max);
        Some(view! {
            <circle cx=cx cy=cy r=local_r * scale class="rlevo-box2d-wheel-disk" />
        })
    } else {
        None
    };

    view! {
        <g>
            <polygon points=pts_svg class=class />
            {extra}
        </g>
    }
    .into_any()
}
