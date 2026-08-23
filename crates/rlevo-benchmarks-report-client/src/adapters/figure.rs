//! The projected figure: what an adapter decided to draw, before Leptos.
//!
//! # Why this exists
//!
//! The family adapters turn a payload into an SVG scene, and until now that
//! computation lived inside the `view!` macro. Its output is an `AnyView`, which
//! a unit test cannot inspect without a DOM — so the adapters carried **one
//! test between them**, on a numeric helper, and nothing at all asserted what
//! got drawn.
//!
//! That is the gap ADR 0082's golden-frame prerequisite names: it replaces four
//! adapters totalling 818 lines with one of roughly 350, and a pole drawn at the
//! wrong angle or a lander's legs inverted would ship silently.
//!
//! So the projection is separated from the rendering. `project()` is a pure
//! function from payload to [`Figure`] — real coordinates in final SVG user
//! space, with the CSS class each element will carry — and the `view!` layer is
//! a mechanical map over it. Tests assert on the [`Figure`]: node count, classes
//! present, bounds, key coordinates.
//!
//! # What these tests do and do not cover
//!
//! Stated plainly, because the boundary is easy to forget once the suite is
//! green.
//!
//! **Covered:** every coordinate the viewer sees, every CSS class, element
//! counts, paint order, degenerate-bounds refusal. That is where a migration
//! breaks a picture.
//!
//! **Not covered:** the [`prim_view`] mapping itself, the surrounding
//! `<figure>` / `<figcaption>` chrome, and the stylesheet. If `prim_view`
//! transposed `x1` and `y1`, every test here would still pass. It is kept to a
//! single four-arm match, shared by all three adapters, precisely so that it is
//! small enough to read — one reviewable function against three adapters' worth
//! of tested geometry. Closing that last gap needs a DOM or an SSR renderer,
//! which is a dependency this crate does not have.
//!
//! # `Group` is not cosmetic
//!
//! [`Prim::Group`] exists because the stylesheet reaches through it:
//! `.rlevo-locomotion-com line` styles the centre-of-mass cross-hair, and
//! flattening those two lines into siblings carrying the group's own class
//! would match no rule at all. The cross-hair would still be *drawn*, in the
//! browser default, and no test that checked only coordinates would notice.
//! Keep a group whenever the CSS selects a descendant.

use leptos::prelude::*;

/// One drawable primitive, in final SVG user-space coordinates.
///
/// Coordinates are already projected: no further transform is applied when this
/// becomes an element, so a test asserting on them is asserting on what the
/// viewer sees.
#[derive(Debug, Clone, PartialEq)]
pub enum Prim {
    /// A straight segment.
    Line {
        /// Start x.
        x1: f32,
        /// Start y.
        y1: f32,
        /// End x.
        x2: f32,
        /// End y.
        y2: f32,
        /// CSS class applied to the element.
        class: &'static str,
    },
    /// A circle, used for joints, contacts, and wheels.
    Circle {
        /// Centre x.
        cx: f32,
        /// Centre y.
        cy: f32,
        /// Radius, in SVG user units.
        r: f32,
        /// CSS class applied to the element.
        class: &'static str,
    },
    /// A closed filled shape.
    Polygon {
        /// Vertices in paint order.
        points: Vec<(f32, f32)>,
        /// CSS class applied to the element.
        class: &'static str,
    },
    /// An open stroked path.
    Polyline {
        /// Vertices in paint order.
        points: Vec<(f32, f32)>,
        /// CSS class applied to the element.
        class: &'static str,
    },
    /// An SVG `<g>`. Present when the stylesheet selects a descendant through
    /// it — see the module docs; do not flatten.
    Group {
        /// CSS class applied to the group.
        class: &'static str,
        /// Children, painted in order.
        children: Vec<Prim>,
    },
}

impl Prim {
    /// The CSS class this primitive carries.
    #[must_use]
    pub const fn class(&self) -> &'static str {
        match self {
            Self::Line { class, .. }
            | Self::Circle { class, .. }
            | Self::Polygon { class, .. }
            | Self::Polyline { class, .. }
            | Self::Group { class, .. } => class,
        }
    }
}

/// A projected scene, ready to render.
#[derive(Debug, Clone, PartialEq)]
pub struct Figure {
    /// `viewBox` extent as `(width, height)`; the origin is always `0 0`.
    pub view_box: (f32, f32),
    /// Primitives in paint order — first drawn is furthest back.
    pub prims: Vec<Prim>,
}

impl Figure {
    /// Count of primitives carrying `class`, at the top level only.
    ///
    /// Deliberately not recursive: a test asserting "three bones" should not
    /// silently pass because three same-classed children turned up inside an
    /// unrelated group.
    #[must_use]
    pub fn count_class(&self, class: &str) -> usize {
        self.prims.iter().filter(|p| p.class() == class).count()
    }

    /// The first primitive carrying `class`, if any.
    #[must_use]
    pub fn first_class(&self, class: &str) -> Option<&Prim> {
        self.prims.iter().find(|p| p.class() == class)
    }
}

/// Why a payload could not be projected.
///
/// Adapters render a visible message for this rather than an empty panel; an
/// empty frame with no explanation is the failure mode that made a NaN payload
/// hard to diagnose in the first place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionError {
    /// The computed viewport has zero or non-finite span on an axis, so the
    /// affine map would divide by zero or propagate NaN into every coordinate.
    DegenerateBounds,
    /// The payload carries a non-finite coordinate.
    ///
    /// Refused rather than drawn. A finite viewport does **not** protect
    /// against this: the viewport is derived by filtering non-finite values, so
    /// a payload with one NaN joint still yields sane bounds, and the NaN then
    /// survives the affine map into an element attribute. The browser drops the
    /// element and the panel renders empty with no explanation — the hardest
    /// possible place to notice a diverged simulation.
    NonFiniteCoordinate,
}

/// Renders one primitive. **The one piece of this module its tests cannot
/// reach** — see the module docs.
#[must_use]
pub fn prim_view(prim: &Prim) -> AnyView {
    match prim {
        Prim::Line {
            x1,
            y1,
            x2,
            y2,
            class,
        } => view! { <line x1=*x1 y1=*y1 x2=*x2 y2=*y2 class=*class /> }.into_any(),
        Prim::Circle { cx, cy, r, class } => {
            view! { <circle cx=*cx cy=*cy r=*r class=*class /> }.into_any()
        }
        Prim::Polygon { points, class } => {
            view! { <polygon points=points_attr(points) class=*class /> }.into_any()
        }
        Prim::Polyline { points, class } => {
            view! { <polyline points=points_attr(points) class=*class /> }.into_any()
        }
        Prim::Group { class, children } => {
            let kids: Vec<AnyView> = children.iter().map(prim_view).collect();
            view! { <g class=*class>{kids}</g> }.into_any()
        }
    }
}

/// Formats a point list as an SVG `points` attribute.
fn points_attr(points: &[(f32, f32)]) -> String {
    points
        .iter()
        .map(|(x, y)| format!("{x},{y}"))
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_reads_through_every_variant() {
        assert_eq!(
            Prim::Line {
                x1: 0.0,
                y1: 0.0,
                x2: 1.0,
                y2: 1.0,
                class: "a"
            }
            .class(),
            "a"
        );
        assert_eq!(
            Prim::Group {
                class: "g",
                children: vec![]
            }
            .class(),
            "g"
        );
    }

    #[test]
    fn count_class_does_not_descend_into_groups() {
        // The guard described on `count_class`: a bone nested inside the
        // centre-of-mass group must not be counted as a top-level bone.
        let fig = Figure {
            view_box: (10.0, 10.0),
            prims: vec![
                Prim::Circle {
                    cx: 1.0,
                    cy: 1.0,
                    r: 1.0,
                    class: "joint",
                },
                Prim::Group {
                    class: "com",
                    children: vec![Prim::Circle {
                        cx: 2.0,
                        cy: 2.0,
                        r: 1.0,
                        class: "joint",
                    }],
                },
            ],
        };
        assert_eq!(fig.count_class("joint"), 1);
        assert_eq!(fig.count_class("com"), 1);
    }

    #[test]
    fn points_attr_formats_svg_pairs() {
        assert_eq!(points_attr(&[(1.0, 2.0), (3.5, 4.0)]), "1,2 3.5,4");
    }
}
