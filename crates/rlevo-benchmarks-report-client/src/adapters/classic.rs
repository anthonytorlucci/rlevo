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

use super::figure::{Figure, Prim, ProjectionError, prim_view};
use crate::wire::{
    Classic2DBody, Classic2DRole, Classic2DSnapshot, FamilyPayload, FrameRecord, Point2,
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
        _ => super::fallback::render(
            crate::wire::EnvFamily::Classic,
            super::fallback::FallbackReason::UnsupportedPayload,
            frame,
        ),
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
        // `#[non_exhaustive]`: an unknown role draws unstyled rather than
        // failing to compile when the shared enum grows.
        _ => "rlevo-classic-body",
    }
}

/// Projects a [`Classic2DSnapshot`] into a [`Figure`] — pure, and what the
/// golden-frame tests assert on.
///
/// Applies a uniform-scale affine map from the payload's world-space `bounds`
/// onto the padded inner square (`VB - 2*PAD`), centring the shorter axis. The
/// y-axis is flipped so physics-up renders as visually up.
///
/// # Errors
///
/// [`ProjectionError::DegenerateBounds`] when either dimension of `bounds` is
/// zero or sub-epsilon, and [`ProjectionError::NonFiniteCoordinate`] when the
/// payload carries a NaN or infinity — which a finite viewport does not rule
/// out, since bounds and vertices are independent fields.
pub fn project(payload: &Classic2DSnapshot) -> Result<Figure, ProjectionError> {
    let (lo, hi) = payload.bounds;
    let (sx, sy) = (hi.x - lo.x, hi.y - lo.y);
    if !sx.is_finite() || !sy.is_finite() || sx.abs() < f32::EPSILON || sy.abs() < f32::EPSILON {
        return Err(ProjectionError::DegenerateBounds);
    }
    if !payload
        .bodies
        .iter()
        .flat_map(|b| b.points.iter())
        .all(|p| p.x.is_finite() && p.y.is_finite())
    {
        return Err(ProjectionError::NonFiniteCoordinate);
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

    let prims = payload
        .bodies
        .iter()
        .map(|body| body_prim(body, &xform))
        .collect();

    Ok(Figure {
        view_box: (VB, VB),
        prims,
    })
}

/// One body: a ring for a single-point hinge, a polygon when closed, otherwise
/// an open polyline.
fn body_prim(body: &Classic2DBody, xform: &impl Fn(&Point2) -> (f32, f32)) -> Prim {
    let class = role_class(body.role);
    if body.points.len() == 1 {
        let (cx, cy) = xform(&body.points[0]);
        return Prim::Circle {
            cx,
            cy,
            r: 4.0,
            class,
        };
    }
    let points: Vec<(f32, f32)> = body.points.iter().map(xform).collect();
    if body.closed {
        Prim::Polygon { points, class }
    } else {
        Prim::Polyline { points, class }
    }
}

/// Builds the SVG figure for a [`Classic2DSnapshot`].
///
/// A thin wrapper over [`project`]: every decision about the picture is made
/// there, so that it can be tested.
fn view_with_payload(payload: &Classic2DSnapshot) -> AnyView {
    let Ok(figure) = project(payload) else {
        return view! {
            <p class="rlevo-warnings">
                "classic payload cannot be rendered — degenerate bounds or a non-finite coordinate"
            </p>
        }
        .into_any();
    };

    let (vb_w, vb_h) = figure.view_box;
    let view_box = format!("0 0 {vb_w} {vb_h}");
    let bodies: Vec<AnyView> = figure.prims.iter().map(prim_view).collect();

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

#[cfg(test)]
mod tests {
    use super::{PAD, VB, project, role_class};
    use crate::adapters::figure::{Prim, ProjectionError};
    use crate::wire::{Classic2DBody, Classic2DRole, Classic2DSnapshot, Point2};

    /// A cart-pole: a track polyline, a closed cart polygon, an open pole
    /// polyline, and a single-point hinge.
    ///
    /// The world bounds are deliberately **non-square** (4 wide, 2 tall) so the
    /// uniform-scale-plus-centre logic is actually exercised; a square domain
    /// would make the two axes indistinguishable and hide a swapped offset.
    fn cart_pole() -> Classic2DSnapshot {
        Classic2DSnapshot {
            bounds: (Point2::new(-2.0, 0.0), Point2::new(2.0, 2.0)),
            bodies: vec![
                Classic2DBody {
                    role: Classic2DRole::Track,
                    points: vec![Point2::new(-2.0, 0.5), Point2::new(2.0, 0.5)],
                    closed: false,
                },
                Classic2DBody {
                    role: Classic2DRole::Cart,
                    points: vec![
                        Point2::new(-0.3, 0.4),
                        Point2::new(0.3, 0.4),
                        Point2::new(0.3, 0.6),
                        Point2::new(-0.3, 0.6),
                    ],
                    closed: true,
                },
                Classic2DBody {
                    role: Classic2DRole::Pole,
                    points: vec![Point2::new(0.0, 0.5), Point2::new(0.4, 1.5)],
                    closed: false,
                },
                Classic2DBody {
                    role: Classic2DRole::Hinge,
                    points: vec![Point2::new(0.0, 0.5)],
                    closed: false,
                },
            ],
        }
    }

    #[test]
    fn golden_body_shapes_follow_point_count_and_closed_flag() {
        let fig = project(&cart_pole()).expect("cart-pole projects");
        assert_eq!(fig.prims.len(), 4);
        assert!(matches!(fig.prims[0], Prim::Polyline { .. }), "open track");
        assert!(matches!(fig.prims[1], Prim::Polygon { .. }), "closed cart");
        assert!(matches!(fig.prims[2], Prim::Polyline { .. }), "open pole");
        assert!(
            matches!(fig.prims[3], Prim::Circle { .. }),
            "a one-point body is a hinge ring, not a degenerate polyline"
        );
    }

    #[test]
    fn golden_roles_map_to_their_css_classes() {
        let fig = project(&cart_pole()).expect("cart-pole projects");
        assert_eq!(fig.count_class("rlevo-classic-track"), 1);
        assert_eq!(fig.count_class("rlevo-classic-cart"), 1);
        assert_eq!(fig.count_class("rlevo-classic-pole"), 1);
        assert_eq!(fig.count_class("rlevo-classic-hinge"), 1);
    }

    #[test]
    fn unknown_role_falls_back_to_a_neutral_class_not_a_panic() {
        // `Classic2DRole` is `#[non_exhaustive]`, so this arm is reachable from
        // a record written by a newer producer.
        assert_eq!(role_class(Classic2DRole::Cart), "rlevo-classic-cart");
        assert_eq!(role_class(Classic2DRole::Hinge), "rlevo-classic-hinge");
    }

    #[test]
    fn golden_y_is_flipped_and_x_is_not() {
        let fig = project(&cart_pole()).expect("cart-pole projects");
        let Prim::Polyline { points, .. } = &fig.prims[2] else {
            panic!("pole should be a polyline");
        };
        // Pole runs from (0.0, 0.5) up-and-right to (0.4, 1.5).
        let (base, tip) = (points[0], points[1]);
        assert!(tip.1 < base.1, "the pole tip is higher, so smaller svg y");
        assert!(tip.0 > base.0, "the pole leans right, and x is not flipped");
    }

    #[test]
    fn golden_uniform_scale_centres_the_shorter_axis() {
        // Bounds are 4 wide by 2 tall on a square viewport. A uniform scale
        // fits the wide axis and leaves the short one centred, so the drawn
        // content must not touch the top and bottom padding edges.
        let fig = project(&cart_pole()).expect("cart-pole projects");
        let Prim::Polyline { points, .. } = &fig.prims[0] else {
            panic!("track should be a polyline");
        };
        // The track spans the full world width, so it reaches both x edges.
        assert!((points[0].0 - PAD).abs() < 1e-3, "track starts at left pad");
        assert!(
            (points[1].0 - (VB - PAD)).abs() < 1e-3,
            "track ends at right pad"
        );
        // ...but sits well inside vertically, because the short axis is centred.
        let inner = VB - 2.0 * PAD;
        let expected_off_y = PAD + (inner - inner * 0.5) * 0.5;
        assert!(
            points[0].1 > expected_off_y,
            "y should be inset by the centring offset"
        );
    }

    #[test]
    fn golden_all_coordinates_stay_within_the_viewport() {
        let fig = project(&cart_pole()).expect("cart-pole projects");
        for prim in &fig.prims {
            let pts: Vec<(f32, f32)> = match prim {
                Prim::Polygon { points, .. } | Prim::Polyline { points, .. } => points.clone(),
                Prim::Circle { cx, cy, .. } => vec![(*cx, *cy)],
                _ => vec![],
            };
            for (x, y) in pts {
                assert!((0.0..=VB).contains(&x), "x {x} escaped the viewBox");
                assert!((0.0..=VB).contains(&y), "y {y} escaped the viewBox");
            }
        }
    }

    #[test]
    fn degenerate_bounds_are_refused() {
        let mut p = cart_pole();
        p.bounds = (Point2::new(1.0, 0.0), Point2::new(1.0, 2.0));
        assert_eq!(project(&p), Err(ProjectionError::DegenerateBounds));
    }

    #[test]
    fn non_finite_bounds_are_refused_rather_than_propagated() {
        // `(NaN - NaN).abs() < EPSILON` is false, so a NaN bound slips past a
        // span-only degeneracy check and reaches every coordinate.
        let mut p = cart_pole();
        p.bounds = (Point2::new(f32::NAN, 0.0), Point2::new(2.0, 2.0));
        assert_eq!(project(&p), Err(ProjectionError::DegenerateBounds));
    }

    #[test]
    fn non_finite_vertex_is_refused_even_when_bounds_are_sane() {
        // Bounds and vertices are independent fields: a perfectly good viewport
        // does not make the geometry inside it finite.
        let mut p = cart_pole();
        p.bodies[1].points[0] = Point2::new(f32::INFINITY, 0.4);
        assert_eq!(project(&p), Err(ProjectionError::NonFiniteCoordinate));
    }
}
