//! Per-family structured-rendering surfaces for the rich report tier.
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
//!    implement them fall back to the default `FamilyPayload::Ascii`.
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
    /// Constructs a new [`Point2`] from the given `x` and `y` coordinates.
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
    /// Returns a [`Landscape2DSnapshot`] capturing the current frame.
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
    /// Polygon corners in the body's **local** frame, counter-clockwise.
    pub vertices: Vec<Point2>,
    /// World-space position of the body's local origin.
    pub position: Point2,
    /// Rotation of the body about its local origin, in radians.
    pub rotation_rad: f32,
    /// Semantic class used by the renderer to choose colour / stroke / fill.
    pub kind: BodyKind,
}

/// All bodies + contact points + world bounds, captured at one frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Box2dSnapshot {
    /// World-space rectangle the renderer fits its viewport to.
    /// `(min, max)` corners.
    pub world_bounds: (Point2, Point2),
    /// All rigid bodies in the scene, in paint order.
    pub bodies: Vec<RigidBody2D>,
    /// Active contact points between bodies this frame.
    pub contacts: Vec<Point2>,
}

/// Producer-side trait. A box2d env implements this when it wants its
/// recording to ship a `FamilyPayload::Box2D` instead of `Ascii`.
pub trait Box2dPayloadSource {
    /// Returns a [`Box2dSnapshot`] capturing the current frame.
    fn box2d_snapshot(&self) -> Box2dSnapshot;
}

// ---------------------------------------------------------------------------
// Locomotion2D
// ---------------------------------------------------------------------------

/// Sagittal-plane projection of a locomotion env, captured at one frame.
///
/// **This is locomotion's canonical view** — locomotion envs do not
/// implement [`AsciiRenderable`], so this payload is the only
/// rendering pathway in the whole stack.
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
    /// Positions of each joint in the sagittal-plane frame.
    pub joints: Vec<Point2>,
    /// Rigid-bone connectivity: each `(a, b)` pair connects `joints[a]` to
    /// `joints[b]`.
    pub bones: Vec<(u32, u32)>,
    /// Y-coordinate of the ground line in the same frame as the joints.
    pub ground_y: f32,
    /// Projected centre of mass. `None` when the env does not track it.
    pub com: Option<Point2>,
    /// Footstep contact points; rendered as small open rings on the report
    /// tier.
    pub contacts: Vec<Point2>,
}

/// Producer-side trait. A locomotion env implements this to supply the only
/// rendering pathway in the stack — locomotion envs do not implement
/// [`AsciiRenderable`], so this payload is the canonical view.
///
/// [`AsciiRenderable`]: super::AsciiRenderable
pub trait Locomotion2DPayloadSource {
    /// Returns a [`Locomotion2DSnapshot`] capturing the current frame.
    fn locomotion2d_snapshot(&self) -> Locomotion2DSnapshot;
}

// ---------------------------------------------------------------------------
// Grid
// ---------------------------------------------------------------------------

/// Cardinal facing of the grid agent. Mirrors the env-side `Direction`
/// (`+x` East, `+y` South); the renderer rotates the agent triangle to match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GridDir {
    East,
    South,
    West,
    North,
}

/// The six Minigrid colours, paired with a redundant non-colour signal
/// (glyph/label) on the report tier per the accessibility contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum GridColor {
    Red,
    Green,
    Blue,
    Purple,
    Yellow,
    Grey,
}

/// Open / closed / locked state of a [`GridTile::Door`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GridDoorState {
    Open,
    Closed,
    Locked,
}

/// One grid cell's contents, projected from the env-side `Entity`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum GridTile {
    /// Empty walkable cell.
    Empty,
    /// Walkable floor (drawn distinctly from `Empty`).
    Floor,
    /// Impassable wall.
    Wall,
    /// Terminal goal cell.
    Goal,
    /// Hazard cell (ends the episode in failure).
    Lava,
    /// Door of the given colour and state.
    Door(GridColor, GridDoorState),
    /// Colored key.
    Key(GridColor),
    /// Colored ball.
    Ball(GridColor),
    /// Colored box.
    Box(GridColor),
}

/// The agent marker: cell position, facing, and any carried item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GridAgentMarker {
    /// Column (0-based, left to right).
    pub x: u16,
    /// Row (0-based, top to bottom).
    pub y: u16,
    /// Direction the agent faces.
    pub dir: GridDir,
    /// Item the agent is holding, if any.
    pub carrying: Option<GridTile>,
}

/// A snapshot of a grid (Minigrid-style) environment at one frame.
///
/// `tiles` is row-major with `tiles.len() == width * height`; cell
/// `(x, y)` is `tiles[y * width + x]`. The renderer draws one `<rect>`
/// per tile, the agent as a rotated triangle, and pickable objects as
/// shape-distinct glyphs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GridSnapshot {
    /// Grid width in cells.
    pub width: u16,
    /// Grid height in cells.
    pub height: u16,
    /// Row-major tiles, `len == width * height`.
    pub tiles: Vec<GridTile>,
    /// The agent marker.
    pub agent: GridAgentMarker,
}

/// Producer-side trait. A grid env implements this so its recording ships
/// a `FamilyPayload::Grid` rendered from structured tile state instead of
/// `Ascii` text.
pub trait GridPayloadSource {
    /// Returns a [`GridSnapshot`] capturing the current frame.
    fn grid_snapshot(&self) -> GridSnapshot;
}

// ---------------------------------------------------------------------------
// TabularText
// ---------------------------------------------------------------------------

/// Background class of a [`TabularGrid`] cell — the union of cell semantics
/// across the grid-shaped toy-text envs (`FrozenLake` / `CliffWalking` / Taxi).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TabularCell {
    /// Plain walkable cell.
    Empty,
    /// Frozen safe surface (`FrozenLake`).
    Frozen,
    /// Episode start cell.
    Start,
    /// Terminal goal cell.
    Goal,
    /// Hazard cell — falling in a hole / stepping off the cliff.
    Hazard,
}

/// Semantic class of a [`TabularMarker`] overlaid on a [`TabularGrid`] cell.
/// Each maps to a shape-distinct glyph on the report tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TabularMarkerKind {
    /// The controllable agent (elf / taxi).
    Agent,
    /// Passenger waiting to be picked up (Taxi).
    Passenger,
    /// Drop-off destination (Taxi).
    Destination,
    /// A named pickup/drop location (Taxi's R/G/Y/B corners).
    Location,
}

/// A point-of-interest overlaid on a grid cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TabularMarker {
    /// Column (0-based, left to right).
    pub x: u16,
    /// Row (0-based, top to bottom).
    pub y: u16,
    /// Semantic role that determines the glyph the renderer draws.
    pub kind: TabularMarkerKind,
}

/// Grid layout for the grid-shaped toy-text envs. `cells` is row-major,
/// `len == width * height`; `markers` overlay agent / passenger / destination.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TabularGrid {
    /// Grid width in cells.
    pub width: u16,
    /// Grid height in cells.
    pub height: u16,
    /// Row-major cells, `len == width * height`; cell `(x, y)` is
    /// `cells[y * width + x]`.
    pub cells: Vec<TabularCell>,
    /// Points-of-interest overlaid on top of the background cells.
    pub markers: Vec<TabularMarker>,
}

/// Card-table layout for Blackjack. Card values are blackjack face values
/// (`1` = ace, `2..=10`, `10` for face cards). `dealer_showing` is the
/// dealer's single up-card while the hole card is concealed during play;
/// `dealer_cards` carries the full hand for post-episode review.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CardTable {
    pub player_cards: Vec<u8>,
    pub player_total: u8,
    pub usable_ace: bool,
    pub dealer_cards: Vec<u8>,
    pub dealer_showing: u8,
}

/// Layout discriminant for [`TabularSnapshot`] — grid-shaped envs vs the
/// Blackjack card table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TabularLayout {
    Grid(TabularGrid),
    Cards(CardTable),
}

/// A snapshot of a tabular (toy-text) environment at one frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TabularSnapshot {
    /// The layout discriminant, carrying either a grid or a card-table view.
    pub layout: TabularLayout,
}

/// Producer-side trait. A toy-text env implements this so its recording
/// ships a `FamilyPayload::TabularText` rendered from structured layout
/// state instead of `Ascii` text.
pub trait TabularPayloadSource {
    /// Returns a [`TabularSnapshot`] capturing the current frame.
    fn tabular_snapshot(&self) -> TabularSnapshot;
}

// ---------------------------------------------------------------------------
// Classic2D
// ---------------------------------------------------------------------------

/// Semantic role of a [`Classic2DBody`], driving the report tier's CSS
/// (colour / stroke / fill) so the parts of each classic-control mechanism
/// stay visually distinct and accessible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Classic2DRole {
    /// The ground line / track / terrain profile.
    Track,
    /// The cart (`CartPole`).
    Cart,
    /// A balancing pole (`CartPole` / Pendulum).
    Pole,
    /// A rigid link of a multi-link arm (Acrobot).
    Link,
    /// The car (`MountainCar`).
    Car,
    /// A pivot / hinge point (drawn as a small marker).
    Hinge,
}

/// One body of a classic-control mechanism, expressed as a **world-space**
/// polyline (already transformed — no separate pose). A single-point body is
/// a marker (e.g. a hinge); `closed = true` makes it a filled polygon.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Classic2DBody {
    /// World-space points in the env's natural frame (`+y` up).
    pub points: Vec<Point2>,
    /// What this body is, for styling.
    pub role: Classic2DRole,
    /// `true` → render as a closed filled polygon; `false` → open polyline.
    pub closed: bool,
}

/// A snapshot of a classic-control env (`CartPole` / Pendulum / `MountainCar` /
/// Acrobot) at one frame: a set of world-space bodies plus the viewport the
/// renderer fits to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Classic2DSnapshot {
    /// Bodies in paint order (track first, moving parts last).
    pub bodies: Vec<Classic2DBody>,
    /// Viewport rectangle the renderer fits to: `(min, max)` corners.
    pub bounds: (Point2, Point2),
}

/// Producer-side trait. A classic-control env implements this so its
/// recording ships a `FamilyPayload::Classic2D` rendered as SVG line-art
/// instead of `Ascii` text.
pub trait Classic2DPayloadSource {
    /// Returns a [`Classic2DSnapshot`] capturing the current frame.
    fn classic2d_snapshot(&self) -> Classic2DSnapshot;
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
