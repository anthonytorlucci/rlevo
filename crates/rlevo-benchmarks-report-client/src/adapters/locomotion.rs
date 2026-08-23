//! Locomotion adapter — the **canonical view** for the family.
//!
//! Locomotion envs do not implement `rlevo_scene::AsciiRenderable`;
//! their only
//! rendering pathway in the entire stack is this SVG adapter consuming
//! [`FamilyPayload::Locomotion2D`].
//!
//! The frame is rendered as a sagittal-plane stick figure:
//!
//! - **Bones** as straight lines between connected joints.
//! - **Joints** as filled circles.
//! - **Ground line** at `ground_y`, omitted entirely when it is `None` —
//!   a top-down, zero-gravity env such as `Swimmer` or `Reacher` has no
//!   floor, and drawing one would put a horizon through the figure.
//! - **Centre of mass** (optional) as a cross-hair marker.
//! - **Contact points** (optional) as small open rings.
//!
//! Per the project a11y contract, every coloured element pairs with a
//! distinct shape: joints are filled disks, com is a cross, contacts
//! are open rings, ground is a solid horizontal line.
//!
//! [`FamilyPayload::Locomotion2D`]: crate::wire::FamilyPayload::Locomotion2D

use leptos::prelude::*;

use super::figure::{Figure, Prim, ProjectionError, prim_view};
use crate::wire::{FamilyPayload, FrameRecord, Locomotion2DSnapshot, Point2};

/// SVG viewBox width in user units (wider than tall for a sagittal-plane view).
const VB_W: f32 = 480.0;
/// SVG viewBox height in user units.
const VB_H: f32 = 240.0;
/// Padding inside the viewBox on each edge so joints at the boundary stay visible.
const VB_PAD: f32 = 20.0;

/// CSS classes, named once so a test asserts the same string the element does.
/// A literal at both sites lets a rename pass the tests it should break.
const CLASS_GROUND: &str = "rlevo-locomotion-ground";
const CLASS_BONE: &str = "rlevo-locomotion-bone";
const CLASS_JOINT: &str = "rlevo-locomotion-joint";
const CLASS_CONTACT: &str = "rlevo-locomotion-contact";
const CLASS_COM: &str = "rlevo-locomotion-com";

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
        _ => super::fallback::render(
            crate::wire::EnvFamily::Locomotion,
            super::fallback::FallbackReason::UnsupportedPayload,
            frame,
        ),
    }
    .into_any()
}

/// Computed axis-aligned viewport in payload coordinates with padding.
/// Returned as `(x_lo, x_hi, y_lo, y_hi)` after `MIN_HALF_RANGE`
/// clamping so a static stick figure still renders visibly.
fn payload_bounds(payload: &Locomotion2DSnapshot) -> (f32, f32, f32, f32) {
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
    // A groundless env contributes no y here: including a placeholder would
    // stretch the viewport toward a plane that is not in the scene.
    if let Some(gy) = payload.ground_y {
        ys.push(gy);
    }

    let (x_min, x_max) = bounds(&xs).unwrap_or((-1.0, 1.0));
    let (y_min, y_max) = bounds(&ys).unwrap_or((0.0, 2.0));
    let cx = (x_min + x_max) * 0.5;
    let cy = (y_min + y_max) * 0.5;
    let hx = ((x_max - x_min) * 0.5).max(MIN_HALF_RANGE);
    let hy = ((y_max - y_min) * 0.5).max(MIN_HALF_RANGE);
    (cx - hx, cx + hx, cy - hy, cy + hy)
}

/// Whether every coordinate the projection would draw is finite.
fn payload_is_finite(payload: &Locomotion2DSnapshot) -> bool {
    let mut pts = payload
        .joints
        .iter()
        .chain(payload.contacts.iter())
        .chain(payload.com.iter());
    pts.all(|p| p.x.is_finite() && p.y.is_finite()) && payload.ground_y.is_none_or(f32::is_finite)
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

/// Projects a [`Locomotion2DSnapshot`] into a [`Figure`] — pure, and the part
/// the golden-frame tests assert on.
///
/// Derives world bounds from all joints, the centre of mass, contact points,
/// and `ground_y` via [`payload_bounds`], then maps world coordinates to SVG
/// user units with a y-flip (physics-up → SVG-down). Emits in paint order:
/// ground line, bones, joints, contacts, centre-of-mass cross-hair.
///
/// # Errors
///
/// [`ProjectionError::DegenerateBounds`] when the computed viewport has zero
/// span on either axis, which would make the affine map divide by zero.
pub fn project(payload: &Locomotion2DSnapshot) -> Result<Figure, ProjectionError> {
    let (x_lo, x_hi, y_lo, y_hi) = payload_bounds(payload);
    let span_x = x_hi - x_lo;
    let span_y = y_hi - y_lo;
    if !span_x.is_finite() || !span_y.is_finite() {
        return Err(ProjectionError::DegenerateBounds);
    }
    if span_x.abs() < f32::EPSILON || span_y.abs() < f32::EPSILON {
        return Err(ProjectionError::DegenerateBounds);
    }
    // Checked before projecting, not after. `payload_bounds` filters non-finite
    // values, so a payload with a NaN joint still produces a perfectly sane
    // viewport -- and the NaN then survives the affine map straight into an SVG
    // attribute. See `ProjectionError::NonFiniteCoordinate`.
    if !payload_is_finite(payload) {
        return Err(ProjectionError::NonFiniteCoordinate);
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
    let mut prims = Vec::new();

    // Ground line first, so the figure paints over it. `None` means the env has
    // no ground plane and no line is emitted at all — a groundless env must not
    // get a floor drawn through it.
    if let Some(gy) = payload.ground_y {
        let (_, gy_svg) = xform(&Point2::new(x_lo, gy));
        prims.push(Prim::Line {
            x1: VB_PAD,
            y1: gy_svg,
            x2: VB_W - VB_PAD,
            y2: gy_svg,
            class: CLASS_GROUND,
        });
    }

    // A bone naming a joint index the payload does not have is skipped rather
    // than panicking: the index is an unenforced cross-field invariant.
    for (a, b) in &payload.bones {
        let (Some(pa), Some(pb)) = (joints_xy.get(*a as usize), joints_xy.get(*b as usize)) else {
            continue;
        };
        prims.push(Prim::Line {
            x1: pa.0,
            y1: pa.1,
            x2: pb.0,
            y2: pb.1,
            class: CLASS_BONE,
        });
    }

    for (x, y) in &joints_xy {
        prims.push(Prim::Circle {
            cx: *x,
            cy: *y,
            r: 5.0,
            class: CLASS_JOINT,
        });
    }

    for c in &payload.contacts {
        let (cx, cy) = xform(c);
        prims.push(Prim::Circle {
            cx,
            cy,
            r: 4.0,
            class: CLASS_CONTACT,
        });
    }

    if let Some(com) = payload.com.as_ref() {
        let (cx, cy) = xform(com);
        // A group, not two bare lines: the stylesheet selects
        // `.rlevo-locomotion-com line`, so flattening this would leave the
        // cross-hair matching no rule.
        prims.push(Prim::Group {
            class: CLASS_COM,
            children: vec![
                Prim::Line {
                    x1: cx - 6.0,
                    y1: cy,
                    x2: cx + 6.0,
                    y2: cy,
                    class: "",
                },
                Prim::Line {
                    x1: cx,
                    y1: cy - 6.0,
                    x2: cx,
                    y2: cy + 6.0,
                    class: "",
                },
            ],
        });
    }

    Ok(Figure {
        view_box: (VB_W, VB_H),
        prims,
    })
}

/// Builds the full SVG figure for a [`Locomotion2DSnapshot`].
///
/// A thin wrapper over [`project`]: everything decided about the picture
/// happens there, so that it can be tested.
fn view_with_payload(payload: &Locomotion2DSnapshot) -> AnyView {
    let Ok(figure) = project(payload) else {
        return view! {
            <p class="rlevo-warnings">
                "locomotion payload cannot be rendered — degenerate bounds or a non-finite coordinate"
            </p>
        }
        .into_any();
    };

    let (vb_w, vb_h) = figure.view_box;
    let view_box = format!("0 0 {vb_w} {vb_h}");
    let body: Vec<AnyView> = figure.prims.iter().map(prim_view).collect();

    view! {
        <figure class="rlevo-family-locomotion">
            <svg
                class="rlevo-svg-frame rlevo-svg-locomotion"
                viewBox=view_box
                role="img"
                aria-label="locomotion sagittal-plane stick figure"
            >
                {body}
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
    use super::{
        CLASS_BONE, CLASS_COM, CLASS_CONTACT, CLASS_GROUND, CLASS_JOINT, VB_H, VB_PAD, VB_W,
        bounds, project,
    };
    use crate::adapters::figure::{Prim, ProjectionError};
    use crate::wire::{Locomotion2DSnapshot, Point2};

    /// A two-bone stick figure with ground, one contact, and a centre of mass.
    ///
    /// Deliberately asymmetric in x and y so a transposed or unflipped axis
    /// changes the numbers. A symmetric fixture is the classic way for a
    /// golden test to pass against a mirrored picture.
    fn walker() -> Locomotion2DSnapshot {
        Locomotion2DSnapshot {
            // Deliberately NOT diagonal. An earlier fixture had x and y both
            // ascending, which is symmetric under transposition -- every
            // ordering assertion held with the axes swapped, and a mutation
            // test proved it. The knee is now right of both hip and foot while
            // sitting between them vertically, so x-order and y-order disagree.
            joints: vec![
                Point2::new(0.2, 2.0), // hip: high, slightly right of the foot
                Point2::new(0.6, 1.0), // knee: furthest right, mid height
                Point2::new(0.1, 0.0), // foot: leftmost, on the ground
            ],
            bones: vec![(0, 1), (1, 2)],
            ground_y: Some(0.0),
            com: Some(Point2::new(0.25, 1.5)),
            contacts: vec![Point2::new(1.0, 0.0)],
        }
    }

    #[test]
    fn bounds_handles_empty_and_nonfinite() {
        assert_eq!(bounds(&[]), None);
        assert_eq!(bounds(&[f32::NAN, f32::INFINITY]), None);
        assert_eq!(bounds(&[1.0, 2.0, 0.5]), Some((0.5, 2.0)));
        assert_eq!(bounds(&[f32::NAN, 1.0, 3.0]), Some((1.0, 3.0)));
    }

    #[test]
    fn golden_node_counts_and_paint_order() {
        let fig = project(&walker()).expect("walker projects");

        assert_eq!(fig.view_box, (VB_W, VB_H));
        assert_eq!(fig.count_class(CLASS_GROUND), 1);
        assert_eq!(fig.count_class(CLASS_BONE), 2, "two bones");
        assert_eq!(fig.count_class(CLASS_JOINT), 3, "three joints");
        assert_eq!(fig.count_class(CLASS_CONTACT), 1);
        assert_eq!(fig.count_class(CLASS_COM), 1);
        assert_eq!(fig.prims.len(), 8);

        // Paint order is a visual fact: the ground must be behind the figure,
        // and the centre-of-mass cross-hair in front of it.
        let classes: Vec<&str> = fig
            .prims
            .iter()
            .map(super::super::figure::Prim::class)
            .collect();
        assert_eq!(classes[0], CLASS_GROUND, "ground paints first");
        assert_eq!(*classes.last().unwrap(), CLASS_COM, "com paints last");
        let first_joint = classes.iter().position(|c| *c == CLASS_JOINT).unwrap();
        let last_bone = classes.iter().rposition(|c| *c == CLASS_BONE).unwrap();
        assert!(last_bone < first_joint, "joints paint over bones");
    }

    #[test]
    fn golden_y_axis_is_flipped_and_x_is_not() {
        let fig = project(&walker()).expect("walker projects");
        let joints: Vec<(f32, f32)> = fig
            .prims
            .iter()
            .filter_map(|p| match p {
                Prim::Circle { cx, cy, class, .. } if *class == CLASS_JOINT => Some((*cx, *cy)),
                _ => None,
            })
            .collect();
        assert_eq!(joints.len(), 3);

        // Payload y increases upward; SVG y increases downward. The hip is the
        // highest joint in the world, so it must have the *smallest* svg y.
        let (hip, knee, foot) = (joints[0], joints[1], joints[2]);
        assert!(hip.1 < knee.1, "hip above knee after the flip");
        assert!(knee.1 < foot.1, "knee above foot after the flip");
        // x is not flipped, and its order differs from y's: the knee is
        // rightmost while sitting between hip and foot vertically. That
        // disagreement is what makes the pair of assertions catch a transpose.
        assert!(foot.0 < hip.0, "foot is leftmost");
        assert!(hip.0 < knee.0, "knee is rightmost");

        // One exact coordinate, so the test pins a value and not merely an
        // order. Derivation, and it is worth following once because it is the
        // whole affine map:
        //
        //   world xs are the joints (0.1, 0.2, 0.6), the com (0.25) and the
        //   contact (1.0), so the extent is 0.1..1.0 -> centre 0.55, half 0.45.
        //   `MIN_HALF_RANGE` (0.6) exceeds that, so the viewport is *widened*
        //   about the centre rather than fitted: x_lo = -0.05, span = 1.2.
        //   The foot at x = 0.1 is therefore 0.125 of the way across, and
        //   20 + 0.125 * (480 - 40) = 75.
        //
        // A change to the padding, the viewport width, or the minimum-range
        // clamp moves this number, and each of those is a visible change.
        assert!(
            (foot.0 - 75.0).abs() < 1e-3,
            "foot x should project to 75.0, got {}",
            foot.0
        );
    }

    #[test]
    fn golden_marker_radii_are_pinned() {
        // Radius is a visible property that no coordinate or count assertion
        // touches -- a mutation changing 5.0 to 9.0 survived the first pass of
        // this suite. Joints read larger than contacts on purpose: the contact
        // is an open ring annotating a joint, not competing with it.
        let fig = project(&walker()).expect("walker projects");
        for prim in &fig.prims {
            match prim {
                Prim::Circle { r, class, .. } if *class == CLASS_JOINT => {
                    assert!((r - 5.0).abs() < f32::EPSILON, "joint radius moved");
                }
                Prim::Circle { r, class, .. } if *class == CLASS_CONTACT => {
                    assert!((r - 4.0).abs() < f32::EPSILON, "contact radius moved");
                }
                _ => {}
            }
        }
    }

    #[test]
    fn golden_figure_fits_inside_the_padded_viewport() {
        let fig = project(&walker()).expect("walker projects");
        for prim in &fig.prims {
            if let Prim::Circle { cx, cy, .. } = prim {
                assert!(
                    (VB_PAD..=VB_W - VB_PAD).contains(cx),
                    "cx {cx} escaped the padded viewport"
                );
                assert!(
                    (VB_PAD..=VB_H - VB_PAD).contains(cy),
                    "cy {cy} escaped the padded viewport"
                );
            }
        }
    }

    #[test]
    fn golden_ground_line_spans_the_full_width_at_one_height() {
        let fig = project(&walker()).expect("walker projects");
        let Some(Prim::Line { x1, y1, x2, y2, .. }) = fig.first_class(CLASS_GROUND) else {
            panic!("expected a ground line");
        };
        assert!((y1 - y2).abs() < f32::EPSILON, "ground is horizontal");
        assert!((*x1 - VB_PAD).abs() < f32::EPSILON);
        assert!((*x2 - (VB_W - VB_PAD)).abs() < f32::EPSILON);
    }

    #[test]
    fn golden_groundless_env_draws_no_floor() {
        // Swimmer and Reacher are top-down and zero-gravity. A line drawn here
        // is a horizon through the figure -- the defect that made `ground_y`
        // an `Option` in the first place.
        let mut p = walker();
        p.ground_y = None;
        let fig = project(&p).expect("groundless walker projects");
        assert_eq!(fig.count_class(CLASS_GROUND), 0);
    }

    #[test]
    fn golden_com_stays_a_group_so_css_can_reach_its_lines() {
        // `.rlevo-locomotion-com line` is the selector. Flattening these into
        // siblings would leave the cross-hair matching no rule -- still drawn,
        // in the browser default, and invisible to a coordinate-only test.
        let fig = project(&walker()).expect("walker projects");
        let Some(Prim::Group { children, .. }) = fig.first_class(CLASS_COM) else {
            panic!("centre of mass must be a <g>, not bare lines");
        };
        assert_eq!(children.len(), 2, "a cross-hair is two lines");
        assert!(
            children.iter().all(|c| matches!(c, Prim::Line { .. })),
            "cross-hair children are lines"
        );
    }

    #[test]
    fn golden_bone_naming_a_missing_joint_is_skipped_not_panicked() {
        // `bones` indexes into `joints` with nothing enforcing the range, so a
        // malformed payload must degrade rather than take the report down.
        let mut p = walker();
        p.bones.push((0, 99));
        let fig = project(&p).expect("out-of-range bone is survivable");
        assert_eq!(fig.count_class(CLASS_BONE), 2, "the bad bone is dropped");
    }

    #[test]
    fn a_single_nan_joint_is_refused_not_projected() {
        // The bug this test found. `payload_bounds` filters non-finite values,
        // so one NaN joint still yields a perfectly sane viewport -- and the
        // NaN then survives the affine map into an SVG attribute, where the
        // browser drops the element and the panel renders empty with no
        // message. Nineteen good joints do not make the twentieth safe.
        let mut p = walker();
        p.joints[1] = Point2::new(f32::NAN, 1.0);
        assert_eq!(project(&p), Err(ProjectionError::NonFiniteCoordinate));
    }

    #[test]
    fn non_finite_is_caught_on_every_coordinate_carrying_field() {
        // One field at a time, because a guard that checks joints and forgets
        // contacts passes any test that only ever poisons a joint.
        for (name, mut p) in [
            ("joint", walker()),
            ("contact", walker()),
            ("com", walker()),
            ("ground_y", walker()),
        ]
        .map(|(n, p)| (n, p))
        {
            match name {
                "joint" => p.joints[0] = Point2::new(0.0, f32::INFINITY),
                "contact" => p.contacts[0] = Point2::new(f32::NAN, 0.0),
                "com" => p.com = Some(Point2::new(0.0, f32::NAN)),
                _ => p.ground_y = Some(f32::NAN),
            }
            assert_eq!(
                project(&p),
                Err(ProjectionError::NonFiniteCoordinate),
                "a non-finite {name} was projected instead of refused"
            );
        }
    }

    #[test]
    fn a_zero_span_viewport_is_refused() {
        // Every drawable at one point, with no ground to widen the range. The
        // `MIN_HALF_RANGE` clamp in `payload_bounds` is what actually stops
        // this, so the assertion is that *something* refuses rather than
        // dividing by zero -- either an error, or a finite figure.
        let p = Locomotion2DSnapshot {
            joints: vec![Point2::new(1.0, 1.0)],
            bones: vec![],
            ground_y: None,
            com: None,
            contacts: vec![],
        };
        match project(&p) {
            Err(_) => {}
            Ok(fig) => {
                for prim in &fig.prims {
                    if let Prim::Circle { cx, cy, .. } = prim {
                        assert!(
                            cx.is_finite() && cy.is_finite(),
                            "a single-point payload produced a non-finite coordinate"
                        );
                    }
                }
            }
        }
    }
}
