//! Per-family structured-rendering surfaces for the M7 report tier.
//!
//! The library and live (TUI) tiers consume [`AsciiRenderable`] /
//! `StyledFrame`; the report tier consumes a richer per-family payload
//! when an env opts in. This module defines:
//!
//! 1. Shared geometry primitives ([`Point2`]).
//! 2. Three per-family **snapshot** types — pure data, owned by the
//!    producer side, free of any wire-format concerns:
//!    - [`Landscape2DSnapshot`] for `landscapes` envs.
//!    - [`Box2dSnapshot`] (with [`RigidBody2D`] / [`BodyKind`]) for
//!      `box2d` envs.
//!    - [`Locomotion2DSnapshot`] for `locomotion` envs (their **canonical
//!      view** — locomotion has no ASCII path).
//! 3. Three opt-in **payload-source** traits — one per family — that an
//!    env implements when it wants the recording layer to capture the
//!    richer payload. Each trait has a single method; envs that do not
//!    implement them keep the M6 behaviour (`FamilyPayload::Ascii`).
//!
//! Wire-format conversion (snapshot → `FamilyPayload`) lives in
//! `rlevo-benchmarks::record` so the wire layer stays owned by the
//! benchmarks crate. `rlevo-core` knows nothing about bincode.
//!
//! [`AsciiRenderable`]: super::AsciiRenderable

use serde::{Deserialize, Serialize};

/// 2D point in the family's natural coordinate frame.
///
/// Each family interprets the frame differently:
/// - landscapes: `(x, y)` in the search domain.
/// - box2d: world-space metres.
/// - locomotion: sagittal-plane projection, `x = forward`, `y = up`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Point2 {
    pub x: f32,
    pub y: f32,
}

impl Point2 {
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

// ---------------------------------------------------------------------------
// Landscape2D
// ---------------------------------------------------------------------------

/// A snapshot of the landscape state at one captured frame.
///
/// The landscape itself (the function evaluated at every grid point) is
/// identified by `label` so the report-tier renderer can reach for a
/// shared, precomputed heatmap rather than embedding one per frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Landscape2DSnapshot {
    /// Search domain along the x axis.
    pub bounds_x: (f32, f32),
    /// Search domain along the y axis.
    pub bounds_y: (f32, f32),
    /// Current candidate position.
    pub current: Point2,
    /// Best candidate seen so far, if tracked.
    pub best: Option<Point2>,
    /// Recent history of `current`, oldest first. Capped by the producer.
    pub trail: Vec<Point2>,
    /// Identifier for the underlying landscape (e.g. `"sphere"`,
    /// `"ackley"`, `"rastrigin"`). The renderer uses this to look up a
    /// shared heatmap; unknown labels fall back to a plain background.
    pub label: String,
}

/// Producer-side trait. An env implements this when it wants its
/// recording to ship a `FamilyPayload::Landscape2D` instead of `Ascii`.
pub trait Landscape2DPayloadSource {
    fn landscape2d_snapshot(&self) -> Landscape2DSnapshot;
}

// ---------------------------------------------------------------------------
// Box2d
// ---------------------------------------------------------------------------

/// Semantic class of a [`RigidBody2D`] — drives the client-side CSS
/// class so colour / stroke / fill choices stay accessible and consistent
/// across all box2d envs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum BodyKind {
    Hull,
    Wheel,
    Leg,
    Wing,
    Ground,
    Goal,
    Other,
}

/// One rigid body's polygon + pose, captured at one frame.
///
/// `vertices` are expressed in the body's local frame; the renderer
/// transforms them via `position` + `rotation_rad` so the wire payload
/// stays compact when a body moves but does not deform.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RigidBody2D {
    pub vertices: Vec<Point2>,
    pub position: Point2,
    pub rotation_rad: f32,
    pub kind: BodyKind,
}

/// All bodies + contact points + world bounds, captured at one frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Box2dSnapshot {
    /// World-space rectangle the renderer fits its viewport to.
    /// `(min, max)` corners.
    pub world_bounds: (Point2, Point2),
    pub bodies: Vec<RigidBody2D>,
    pub contacts: Vec<Point2>,
}

pub trait Box2dPayloadSource {
    fn box2d_snapshot(&self) -> Box2dSnapshot;
}

// ---------------------------------------------------------------------------
// Locomotion2D
// ---------------------------------------------------------------------------

/// Sagittal-plane projection of a locomotion env, captured at one frame.
///
/// **This is locomotion's canonical view** — locomotion envs do not
/// implement [`AsciiRenderable`] per ADR-0008, so this payload is the
/// only rendering pathway in the whole stack.
///
/// `joints[i]` is the i-th joint position; `bones[k] = (a, b)` means
/// joint `a` connects to joint `b` with a rigid bone. `ground_y` is the
/// y-coordinate of the ground line in the same frame. `com` is the
/// projected centre of mass (optional — not every env tracks it).
/// `contacts` are footstep contact points the report tier may sprinkle
/// as small open rings.
///
/// [`AsciiRenderable`]: super::AsciiRenderable
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Locomotion2DSnapshot {
    pub joints: Vec<Point2>,
    pub bones: Vec<(u32, u32)>,
    pub ground_y: f32,
    pub com: Option<Point2>,
    pub contacts: Vec<Point2>,
}

pub trait Locomotion2DPayloadSource {
    fn locomotion2d_snapshot(&self) -> Locomotion2DSnapshot;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point2_const_constructor() {
        const P: Point2 = Point2::new(1.5, -2.5);
        assert!((P.x - 1.5).abs() < f32::EPSILON);
        assert!((P.y + 2.5).abs() < f32::EPSILON);
    }

    #[test]
    fn landscape_snapshot_constructs_and_compares() {
        let snap = Landscape2DSnapshot {
            bounds_x: (-5.0, 5.0),
            bounds_y: (-5.0, 5.0),
            current: Point2::new(0.5, -0.25),
            best: Some(Point2::new(0.0, 0.0)),
            trail: vec![Point2::new(0.1, 0.2), Point2::new(0.3, 0.4)],
            label: "sphere".into(),
        };
        assert_eq!(snap.trail.len(), 2);
        assert_eq!(snap.label, "sphere");
        assert_eq!(snap.clone(), snap);
    }

    #[test]
    fn box2d_snapshot_carries_typed_body_kinds() {
        let snap = Box2dSnapshot {
            world_bounds: (Point2::new(-10.0, -1.0), Point2::new(10.0, 8.0)),
            bodies: vec![
                RigidBody2D {
                    vertices: vec![
                        Point2::new(-0.5, -0.5),
                        Point2::new(0.5, -0.5),
                        Point2::new(0.5, 0.5),
                        Point2::new(-0.5, 0.5),
                    ],
                    position: Point2::new(1.0, 2.0),
                    rotation_rad: 0.25,
                    kind: BodyKind::Hull,
                },
                RigidBody2D {
                    vertices: vec![Point2::new(0.0, 0.0)],
                    position: Point2::new(0.0, 0.0),
                    rotation_rad: 0.0,
                    kind: BodyKind::Ground,
                },
            ],
            contacts: vec![Point2::new(0.0, 0.0)],
        };
        assert_eq!(snap.bodies.len(), 2);
        assert_eq!(snap.bodies[0].kind, BodyKind::Hull);
        assert_eq!(snap.bodies[1].kind, BodyKind::Ground);
    }

    #[test]
    fn locomotion_snapshot_default_ground_and_optional_com() {
        let snap = Locomotion2DSnapshot {
            joints: vec![Point2::new(0.0, 1.0), Point2::new(0.5, 1.5)],
            bones: vec![(0, 1)],
            ground_y: 0.0,
            com: None,
            contacts: vec![],
        };
        assert_eq!(snap.bones, vec![(0u32, 1u32)]);
        assert!(snap.com.is_none());
    }

    /// Sanity: each per-family trait is a "default-free" surface — an
    /// implementor must supply a non-trivial snapshot. Sticking a stub
    /// impl here also guards against unintentional accidental renames.
    struct Stub;
    impl Landscape2DPayloadSource for Stub {
        fn landscape2d_snapshot(&self) -> Landscape2DSnapshot {
            Landscape2DSnapshot {
                bounds_x: (0.0, 1.0),
                bounds_y: (0.0, 1.0),
                current: Point2::default(),
                best: None,
                trail: vec![],
                label: "stub".into(),
            }
        }
    }
    impl Box2dPayloadSource for Stub {
        fn box2d_snapshot(&self) -> Box2dSnapshot {
            Box2dSnapshot {
                world_bounds: (Point2::default(), Point2::new(1.0, 1.0)),
                bodies: vec![],
                contacts: vec![],
            }
        }
    }
    impl Locomotion2DPayloadSource for Stub {
        fn locomotion2d_snapshot(&self) -> Locomotion2DSnapshot {
            Locomotion2DSnapshot {
                joints: vec![],
                bones: vec![],
                ground_y: 0.0,
                com: None,
                contacts: vec![],
            }
        }
    }

    #[test]
    fn payload_source_traits_compose_via_stub() {
        let stub = Stub;
        assert_eq!(stub.landscape2d_snapshot().label, "stub");
        assert_eq!(stub.box2d_snapshot().bodies.len(), 0);
        assert_eq!(stub.locomotion2d_snapshot().joints.len(), 0);
    }
}
