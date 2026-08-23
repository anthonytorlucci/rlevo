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

use super::figure::{Figure, Prim, ProjectionError, prim_view};
use crate::wire::{BodyKind, Box2dSnapshot, FamilyPayload, FrameRecord, Point2, RigidBody2D};

/// SVG viewBox width in user units.
const VB_W: f32 = 480.0;
/// SVG viewBox height in user units.
const VB_H: f32 = 320.0;
/// Padding inside the viewBox on each edge, keeping bodies away from the border.
const VB_PAD: f32 = 12.0;

/// CSS classes named once, so a test asserts the same string the element wears.
const CLASS_CONTACT: &str = "rlevo-box2d-contact";
const CLASS_WHEEL_DISK: &str = "rlevo-box2d-wheel-disk";

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
        _ => super::fallback::render(
            crate::wire::EnvFamily::Box2d,
            super::fallback::FallbackReason::UnsupportedPayload,
            frame,
        ),
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
        // `BodyKind` is `#[non_exhaustive]`, so a variant added upstream lands
        // here rather than breaking the build. Neutral styling is the safe
        // default: an unknown body still draws, just without a semantic class.
        BodyKind::Other | _ => "rlevo-box2d-other",
    }
}

/// Projects a [`Box2dSnapshot`] into a [`Figure`] — pure, and what the
/// golden-frame tests assert on.
///
/// Fits the world bounds into the viewBox preserving aspect ratio, flips y
/// (physics-up → SVG-down), and applies each body's own rotation to its
/// local-frame vertices.
///
/// # Errors
///
/// [`ProjectionError::DegenerateBounds`] when `world_bounds` has zero or
/// non-finite span, and [`ProjectionError::NonFiniteCoordinate`] when a body
/// pose or vertex is not finite. The second is not implied by the first:
/// `CarRacing`'s wheel joints drive bodies to `(NaN, NaN)` while the bounds
/// stay sane, and a NaN then reaches an SVG attribute where the browser drops
/// the element and the panel renders empty with no message.
pub fn project(payload: &Box2dSnapshot) -> Result<Figure, ProjectionError> {
    let (min, max) = payload.world_bounds;
    let span_x = max.x - min.x;
    let span_y = max.y - min.y;
    if !span_x.is_finite() || !span_y.is_finite() {
        return Err(ProjectionError::DegenerateBounds);
    }
    if span_x.abs() < f32::EPSILON || span_y.abs() < f32::EPSILON {
        return Err(ProjectionError::DegenerateBounds);
    }
    let finite_body = |b: &RigidBody2D| {
        b.position.x.is_finite()
            && b.position.y.is_finite()
            && b.rotation_rad.is_finite()
            && b.vertices
                .iter()
                .all(|v| v.x.is_finite() && v.y.is_finite())
    };
    if !payload.bodies.iter().all(finite_body)
        || !payload
            .contacts
            .iter()
            .all(|c| c.x.is_finite() && c.y.is_finite())
    {
        return Err(ProjectionError::NonFiniteCoordinate);
    }

    // Fit world to viewBox preserving aspect ratio.
    let scale = ((VB_W - 2.0 * VB_PAD) / span_x).min((VB_H - 2.0 * VB_PAD) / span_y);
    let xform = move |p: &Point2| {
        let nx = (p.x - min.x) * scale;
        // Flip y so payload y (up) maps to SVG y (down increasing).
        let ny = (max.y - p.y) * scale;
        (VB_PAD + nx, VB_PAD + ny)
    };

    let mut prims: Vec<Prim> = payload
        .bodies
        .iter()
        .map(|body| body_prim(body, scale, xform))
        .collect();

    for c in &payload.contacts {
        let (cx, cy) = xform(c);
        prims.push(Prim::Circle {
            cx,
            cy,
            r: 4.0,
            class: CLASS_CONTACT,
        });
    }

    Ok(Figure {
        view_box: (VB_W, VB_H),
        prims,
    })
}

/// One rigid body: its rotated hull, plus a centre disk for a wheel so the
/// shape reads as round at small sizes.
///
/// Wrapped in a group to match the markup this replaced. No stylesheet rule
/// selects through it — unlike locomotion's centre-of-mass group, which does —
/// but keeping it means the emitted DOM is unchanged.
fn body_prim<F>(body: &RigidBody2D, scale: f32, xform: F) -> Prim
where
    F: Fn(&Point2) -> (f32, f32),
{
    // Transform each local-frame vertex into world frame, then SVG.
    let (cos_t, sin_t) = (body.rotation_rad.cos(), body.rotation_rad.sin());
    let points: Vec<(f32, f32)> = body
        .vertices
        .iter()
        .map(|v| {
            let wx = body.position.x + cos_t * v.x - sin_t * v.y;
            let wy = body.position.y + sin_t * v.x + cos_t * v.y;
            xform(&Point2::new(wx, wy))
        })
        .collect();

    let class = body_class(body.kind);
    let mut children = vec![Prim::Polygon { points, class }];

    if matches!(body.kind, BodyKind::Wheel) {
        let (cx, cy) = xform(&body.position);
        // Derive the wheel radius from the vertex extent so we never overflow
        // the polygon hull.
        let local_r = body
            .vertices
            .iter()
            .map(|v| v.x.hypot(v.y))
            .fold(0.0_f32, f32::max);
        children.push(Prim::Circle {
            cx,
            cy,
            r: local_r * scale,
            class: CLASS_WHEEL_DISK,
        });
    }

    Prim::Group {
        class: "",
        children,
    }
}

/// Builds the full SVG figure for a [`Box2dSnapshot`].
///
/// A thin wrapper over [`project`]: every decision about the picture is made
/// there, so that it can be tested.
fn view_with_payload(payload: &Box2dSnapshot) -> AnyView {
    let Ok(figure) = project(payload) else {
        return view! {
            <p class="rlevo-warnings">
                "box2d payload cannot be rendered — degenerate world bounds or a non-finite coordinate"
            </p>
        }
        .into_any();
    };

    let (vb_w, vb_h) = figure.view_box;
    let view_box = format!("0 0 {vb_w} {vb_h}");
    let body: Vec<AnyView> = figure.prims.iter().map(prim_view).collect();

    view! {
        <figure class="rlevo-family-box2d">
            <svg
                class="rlevo-svg-frame rlevo-svg-box2d"
                viewBox=view_box
                role="img"
                aria-label="box2d world view"
            >
                {body}
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

#[cfg(test)]
mod tests {
    use super::{CLASS_CONTACT, CLASS_WHEEL_DISK, VB_H, VB_W, body_class, project};
    use crate::adapters::figure::{Prim, ProjectionError};
    use crate::wire::{BodyKind, Box2dSnapshot, Point2, RigidBody2D};

    fn body(kind: BodyKind, position: Point2, rotation_rad: f32) -> RigidBody2D {
        RigidBody2D {
            // A unit square about the local origin, so a rotation is visible in
            // the projected vertices rather than being absorbed by symmetry
            // about one axis.
            vertices: vec![
                Point2::new(-1.0, -1.0),
                Point2::new(1.0, -1.0),
                Point2::new(1.0, 1.0),
                Point2::new(-1.0, 1.0),
            ],
            position,
            rotation_rad,
            kind,
        }
    }

    /// A lander: a hull, one wheel, and a ground body, plus a contact point.
    /// Bounds are non-square so the aspect-preserving fit is exercised.
    fn lander() -> Box2dSnapshot {
        Box2dSnapshot {
            world_bounds: (Point2::new(-10.0, 0.0), Point2::new(10.0, 10.0)),
            bodies: vec![
                body(BodyKind::Hull, Point2::new(0.0, 5.0), 0.0),
                body(BodyKind::Wheel, Point2::new(2.0, 3.0), 0.0),
                body(BodyKind::Ground, Point2::new(0.0, 0.5), 0.0),
            ],
            contacts: vec![Point2::new(2.0, 2.0)],
        }
    }

    #[test]
    fn golden_every_body_is_a_group_and_contacts_are_top_level_circles() {
        let fig = project(&lander()).expect("lander projects");
        assert_eq!(fig.prims.len(), 4, "three bodies plus one contact");
        for prim in &fig.prims[..3] {
            assert!(matches!(prim, Prim::Group { .. }), "bodies are groups");
        }
        assert_eq!(fig.count_class(CLASS_CONTACT), 1);
        assert_eq!(fig.view_box, (VB_W, VB_H));
    }

    #[test]
    fn golden_only_a_wheel_gets_a_centre_disk() {
        let fig = project(&lander()).expect("lander projects");
        let disks: usize = fig
            .prims
            .iter()
            .filter_map(|p| match p {
                Prim::Group { children, .. } => Some(children),
                _ => None,
            })
            .flatten()
            .filter(|c| c.class() == CLASS_WHEEL_DISK)
            .count();
        assert_eq!(disks, 1, "exactly the wheel gets a disk");

        // And the hull group carries only its hull polygon.
        let Prim::Group { children, .. } = &fig.prims[0] else {
            panic!("hull should be a group");
        };
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].class(), body_class(BodyKind::Hull));
    }

    #[test]
    fn golden_rotation_actually_rotates_the_hull() {
        // A quarter turn on a unit square maps the corner (-1,-1) to (1,-1) in
        // the body frame. Without the rotation term the projected vertices are
        // identical to the unrotated case, which is what a dropped `sin_t`
        // would produce -- and nothing else in this suite would notice.
        let still = lander();
        let mut turned = lander();
        turned.bodies[0] = body(
            BodyKind::Hull,
            Point2::new(0.0, 5.0),
            core::f32::consts::FRAC_PI_2,
        );

        let a = project(&still).expect("projects");
        let b = project(&turned).expect("projects");
        let (Prim::Group { children: ca, .. }, Prim::Group { children: cb, .. }) =
            (&a.prims[0], &b.prims[0])
        else {
            panic!("hull should be a group");
        };
        let (Prim::Polygon { points: pa, .. }, Prim::Polygon { points: pb, .. }) = (&ca[0], &cb[0])
        else {
            panic!("hull child should be a polygon");
        };
        assert_ne!(pa[0], pb[0], "a quarter turn must move the first vertex");
        // The square is its own image under a quarter turn, so the vertex *set*
        // is preserved while the ordering rotates. That is the precise fact a
        // sign error in the rotation breaks.
        assert!(
            (pa[0].0 - pb[3].0).abs() < 1e-3 && (pa[0].1 - pb[3].1).abs() < 1e-3,
            "a quarter turn should cycle the vertex order, got {pa:?} vs {pb:?}"
        );
    }

    #[test]
    fn golden_y_is_flipped() {
        // The hull sits at world y=5, the ground body at y=0.5. After the flip
        // the hull must have the smaller svg y.
        let fig = project(&lander()).expect("lander projects");
        let centre_y = |i: usize| {
            let Prim::Group { children, .. } = &fig.prims[i] else {
                panic!("group expected");
            };
            let Prim::Polygon { points, .. } = &children[0] else {
                panic!("polygon expected");
            };
            #[expect(
                clippy::cast_precision_loss,
                reason = "a four-vertex polygon's length is exactly representable"
            )]
            let n = points.len() as f32;
            points.iter().map(|p| p.1).sum::<f32>() / n
        };
        assert!(
            centre_y(0) < centre_y(2),
            "hull above ground after the flip"
        );
    }

    #[test]
    fn golden_aspect_ratio_is_preserved_not_stretched() {
        // World is 20 wide by 10 tall into a 480x320 box. The width-limited
        // scale is (480-24)/20 = 22.8; the height-limited one is (320-24)/10 =
        // 29.6. The smaller must win, or the scene is stretched.
        let fig = project(&lander()).expect("lander projects");
        let Prim::Group { children, .. } = &fig.prims[0] else {
            panic!("group expected");
        };
        let Prim::Polygon { points, .. } = &children[0] else {
            panic!("polygon expected");
        };
        // The unit square is 2 world units wide, so at the expected scale its
        // projected width is 2 * 22.8 = 45.6.
        let width = points[1].0 - points[0].0;
        assert!(
            (width - 45.6).abs() < 0.1,
            "hull width should be 45.6 at the aspect-preserving scale, got {width}"
        );
    }

    #[test]
    fn degenerate_world_bounds_are_refused() {
        let mut p = lander();
        p.world_bounds = (Point2::new(0.0, 0.0), Point2::new(0.0, 10.0));
        assert_eq!(project(&p), Err(ProjectionError::DegenerateBounds));
    }

    #[test]
    fn nan_world_bounds_are_refused_rather_than_slipping_past_the_span_check() {
        // `(NaN - NaN).abs() < EPSILON` is false, so a NaN bound passes a
        // span-only degeneracy test and renders an empty panel with no message.
        let mut p = lander();
        p.world_bounds = (
            Point2::new(f32::NAN, f32::NAN),
            Point2::new(f32::NAN, f32::NAN),
        );
        assert_eq!(project(&p), Err(ProjectionError::DegenerateBounds));
    }

    #[test]
    fn a_nan_body_pose_is_refused_even_with_sane_bounds() {
        // This is the CarRacing divergence: wheel fixed-joints drive the body
        // to (NaN, NaN) within ~10 steps while `world_bounds` stays finite.
        let mut p = lander();
        p.bodies[0].position = Point2::new(f32::NAN, f32::NAN);
        assert_eq!(project(&p), Err(ProjectionError::NonFiniteCoordinate));
    }

    #[test]
    fn a_nan_rotation_is_refused() {
        // `cos(NaN)` is NaN, so this reaches every vertex of the body.
        let mut p = lander();
        p.bodies[1].rotation_rad = f32::NAN;
        assert_eq!(project(&p), Err(ProjectionError::NonFiniteCoordinate));
    }

    #[test]
    fn a_nan_contact_is_refused() {
        let mut p = lander();
        p.contacts[0] = Point2::new(f32::NAN, 0.0);
        assert_eq!(project(&p), Err(ProjectionError::NonFiniteCoordinate));
    }
}
