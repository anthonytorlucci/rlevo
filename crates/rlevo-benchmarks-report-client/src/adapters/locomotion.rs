//! Locomotion adapter — the **canonical view** for the family.
//!
//! Locomotion envs do not implement `rlevo_core::render::AsciiRenderable`;
//! their only
//! rendering pathway in the entire stack is this SVG adapter consuming
//! [`FamilyPayload::Locomotion2D`].
//!
//! The frame is rendered as a sagittal-plane stick figure:
//!
//! - **Bones** as straight lines between connected joints.
//! - **Joints** as filled circles.
//! - **Ground line** at `ground_y`.
//! - **Centre of mass** (optional) as a cross-hair marker.
//! - **Contact points** (optional) as small open rings.
//!
//! Per the project a11y contract, every coloured element pairs with a
//! distinct shape: joints are filled disks, com is a cross, contacts
//! are open rings, ground is a solid horizontal line.
//!
//! [`FamilyPayload::Locomotion2D`]: crate::wire::FamilyPayload::Locomotion2D

use leptos::prelude::*;

use crate::wire::{FamilyPayload, FrameRecord, Locomotion2DPayload, Point2};

/// SVG viewBox width in user units (wider than tall for a sagittal-plane view).
const VB_W: f32 = 480.0;
/// SVG viewBox height in user units.
const VB_H: f32 = 240.0;
/// Padding inside the viewBox on each edge so joints at the boundary stay visible.
const VB_PAD: f32 = 20.0;

/// Renders one locomotion-family frame, dispatching on the payload variant.
///
/// Extracts a [`FamilyPayload::Locomotion2D`] payload and forwards it to
/// `view_with_payload`.  Any other variant falls through to
/// [`super::fallback::render`].  This is the only rendering pathway for
/// locomotion environments — they do not implement `AsciiRenderable`.
///
/// [`FamilyPayload::Locomotion2D`]: crate::wire::FamilyPayload::Locomotion2D
#[must_use]
pub fn render(frame: &FrameRecord) -> AnyView {
    match &frame.family_payload {
        FamilyPayload::Locomotion2D(payload) => view_with_payload(payload),
        _ => super::fallback::render(crate::wire::EnvFamily::Locomotion, frame),
    }
    .into_any()
}

/// Computed axis-aligned viewport in payload coordinates with padding.
/// Returned as `(x_lo, x_hi, y_lo, y_hi)` after `MIN_HALF_RANGE`
/// clamping so a static stick figure still renders visibly.
fn payload_bounds(payload: &Locomotion2DPayload) -> (f32, f32, f32, f32) {
    const MIN_HALF_RANGE: f32 = 0.6;

    let mut xs: Vec<f32> = payload.joints.iter().map(|p| p.x).collect();
    let mut ys: Vec<f32> = payload.joints.iter().map(|p| p.y).collect();
    if let Some(com) = payload.com {
        xs.push(com.x);
        ys.push(com.y);
    }
    for c in &payload.contacts {
        xs.push(c.x);
        ys.push(c.y);
    }
    ys.push(payload.ground_y);

    let (x_min, x_max) = bounds(&xs).unwrap_or((-1.0, 1.0));
    let (y_min, y_max) = bounds(&ys).unwrap_or((0.0, 2.0));
    let cx = (x_min + x_max) * 0.5;
    let cy = (y_min + y_max) * 0.5;
    let hx = ((x_max - x_min) * 0.5).max(MIN_HALF_RANGE);
    let hy = ((y_max - y_min) * 0.5).max(MIN_HALF_RANGE);
    (cx - hx, cx + hx, cy - hy, cy + hy)
}

/// Returns `(min, max)` over finite values in `values`, or `None` if none are finite.
fn bounds(values: &[f32]) -> Option<(f32, f32)> {
    let mut iter = values.iter().copied().filter(|v| v.is_finite());
    let first = iter.next()?;
    let mut lo = first;
    let mut hi = first;
    for v in iter {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    Some((lo, hi))
}

/// Builds the full SVG figure for a [`Locomotion2DPayload`].
///
/// Derives world bounds from all joints, the centre of mass, contact points,
/// and `ground_y` via [`payload_bounds`], then maps world coordinates to SVG
/// user units with a y-flip (physics-up → SVG-down).  Renders in paint order:
/// ground line, bones, joints, contacts, centre-of-mass cross-hair.  Returns
/// an error paragraph if the computed bounds are degenerate on either axis.
// Single paint-ordered SVG scene builder; splitting the passes would only thread
// the shared affine map through helpers.
#[allow(clippy::too_many_lines)]
fn view_with_payload(payload: &Locomotion2DPayload) -> AnyView {
    let (x_lo, x_hi, y_lo, y_hi) = payload_bounds(payload);
    let span_x = x_hi - x_lo;
    let span_y = y_hi - y_lo;
    if span_x.abs() < f32::EPSILON || span_y.abs() < f32::EPSILON {
        return view! {
            <p class="rlevo-warnings">
                "locomotion payload has degenerate bounds — cannot render"
            </p>
        }
        .into_any();
    }

    let xform = move |p: &Point2| {
        let nx = (p.x - x_lo) / span_x;
        // Flip y so payload y (up) maps to SVG y (down increasing).
        let ny = 1.0 - (p.y - y_lo) / span_y;
        (
            VB_PAD + nx * (VB_W - 2.0 * VB_PAD),
            VB_PAD + ny * (VB_H - 2.0 * VB_PAD),
        )
    };

    let joints_xy: Vec<(f32, f32)> = payload.joints.iter().map(xform).collect();

    let bones_svg = payload
        .bones
        .iter()
        .filter_map(|(a, b)| {
            let pa = joints_xy.get(*a as usize)?;
            let pb = joints_xy.get(*b as usize)?;
            Some(view! {
                <line
                    x1=pa.0 y1=pa.1 x2=pb.0 y2=pb.1
                    class="rlevo-locomotion-bone"
                />
            })
        })
        .collect::<Vec<_>>();

    let joints_svg = joints_xy
        .iter()
        .map(|(x, y)| {
            view! {
                <circle cx=*x cy=*y r=5.0 class="rlevo-locomotion-joint" />
            }
        })
        .collect::<Vec<_>>();

    let contacts_svg = payload
        .contacts
        .iter()
        .map(|c| {
            let (cx, cy) = xform(c);
            view! {
                <circle cx=cx cy=cy r=4.0 class="rlevo-locomotion-contact" />
            }
        })
        .collect::<Vec<_>>();

    let com_svg = payload.com.as_ref().map(|com| {
        let (cx, cy) = xform(com);
        view! {
            <g class="rlevo-locomotion-com">
                <line x1=cx - 6.0 y1=cy x2=cx + 6.0 y2=cy />
                <line x1=cx y1=cy - 6.0 x2=cx y2=cy + 6.0 />
            </g>
        }
    });

    // Ground line — payload y is in world units; project just the y.
    let dummy_x = Point2::new(x_lo, payload.ground_y);
    let (_, ground_svg_y) = xform(&dummy_x);

    let view_box = format!("0 0 {VB_W} {VB_H}");

    view! {
        <figure class="rlevo-family-locomotion">
            <svg
                class="rlevo-svg-frame rlevo-svg-locomotion"
                viewBox=view_box
                role="img"
                aria-label="locomotion sagittal-plane stick figure"
            >
                <line
                    x1=VB_PAD y1=ground_svg_y
                    x2=VB_W - VB_PAD y2=ground_svg_y
                    class="rlevo-locomotion-ground"
                />
                {bones_svg}
                {joints_svg}
                {contacts_svg}
                {com_svg}
            </svg>
            <figcaption class="legend">
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-swatch rlevo-locomotion-joint-swatch" />
                    " joint"
                </span>
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-swatch rlevo-locomotion-bone-swatch" />
                    " bone"
                </span>
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-swatch rlevo-locomotion-com-swatch" />
                    " centre of mass"
                </span>
                <span class="rlevo-legend-key">
                    <span class="rlevo-legend-swatch rlevo-locomotion-contact-swatch" />
                    " contact"
                </span>
                <span class="rlevo-legend-key">
                    <em>"canonical view — locomotion has no ASCII pathway"</em>
                </span>
            </figcaption>
        </figure>
    }
    .into_any()
}

#[cfg(test)]
mod tests {
    use super::bounds;

    #[test]
    fn bounds_handles_empty_and_nonfinite() {
        assert_eq!(bounds(&[]), None);
        assert_eq!(bounds(&[f32::NAN, f32::INFINITY]), None);
        assert_eq!(bounds(&[1.0, 2.0, 0.5]), Some((0.5, 2.0)));
        assert_eq!(bounds(&[f32::NAN, 1.0, 3.0]), Some((1.0, 3.0)));
    }
}
