//! The scene payload: 3-D native geometry, recorded once, posed per frame.
//!
//! ADR 0082 decision 4. This module holds the types that replace the
//! per-family snapshot zoo in [`payload`](crate::payload) — one shape for every
//! environment whose rendering is continuous geometry, rather than one payload
//! variant, one client adapter, and one `FamilyPayload` tag per family.
//!
//! # The split that makes this affordable
//!
//! [`SceneDescriptor`] is recorded **once per episode**; [`ScenePose`] is
//! recorded **per frame**. Geometry does not change between frames, and
//! recording it per frame would multiply a 3-D record's size by the frame count
//! for zero information gained. The descriptor reaches the sink through
//! `RecordSink::on_scene`; the pose rides in `FamilyPayload::Scene`.
//!
//! # Poses are keyed, not positional
//!
//! [`ScenePose::transforms`] is a list of `(NodeId, Transform3)` pairs, not a
//! `Vec<Transform3>` in descriptor order. The positional form was the original
//! draft, and it is an implicit parallel-array correspondence **spanning two
//! separately-recorded structures** — the descriptor is written at episode
//! start, the pose per frame — with "skip nodes carrying a `static_transform`"
//! as an extra unwritten precondition.
//!
//! That defect is *worse* than the ones it would have joined, because the
//! standard remedy cannot reach it: a per-value `is_consistent()` check needs
//! both structures, and by the time a pose is written the descriptor is long
//! gone. Keying by [`NodeId`] removes the coupling instead of documenting it —
//! an unknown id is ignorable, a missing id means "unchanged", and reordering
//! the descriptor's nodes is harmless.
//!
//! # What `is_consistent` does and does not promise
//!
//! Every type here that carries a cross-field invariant ships
//! `is_consistent()`, which the recording tier `debug_assert!`s at capture. The
//! target is the failure mode where a malformed payload renders wrongly (or not
//! at all) **only in the WASM report client**, which is the worst place in the
//! stack to debug it. Catching it at the producer, in a debug build, costs
//! nothing in release.
//!
//! These are *structural* checks — index bounds, arity, finiteness, unit-norm
//! quaternions. They do not and cannot verify that the geometry describes the
//! environment it claims to. A skeleton with correct indices and a transposed
//! axis passes every check here.

use alloc::string::String;
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

/// 3-D point in the producer's own world frame.
///
/// 2-D producers set `z = 0` and let the renderer project; there is no separate
/// 2-D point type on the scene path. [`Point2`](crate::payload::Point2) remains
/// for the pre-scene payloads that still use it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Point3 {
    /// Coordinate along the world x axis.
    pub x: f32,
    /// Coordinate along the world y axis.
    pub y: f32,
    /// Coordinate along the world z axis.
    pub z: f32,
}

impl Point3 {
    /// The origin.
    pub const ORIGIN: Self = Self::new(0.0, 0.0, 0.0);

    /// Constructs a point from its three coordinates.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Whether all three coordinates are finite.
    ///
    /// A non-finite coordinate survives every affine map a renderer applies and
    /// lands in an SVG attribute, where the browser drops the element and the
    /// panel renders empty with no explanation.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

/// Rigid placement: translation plus a unit quaternion.
///
/// `orientation` is **scalar-first** `[w, x, y, z]`, matching
/// `rlevo-environments::locomotion::backend::Pose` exactly. That is the whole
/// reason for the convention: locomotion's physics backend already produces
/// orientation in this layout and the sagittal-plane payload discarded it. A
/// scene pose carries it through unchanged, with no component reshuffle for a
/// reader to get wrong.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Transform3 {
    /// World-space position of the node's local origin.
    pub position: Point3,
    /// Unit quaternion `[w, x, y, z]` — scalar first.
    pub orientation: [f32; 4],
}

impl Transform3 {
    /// No translation, no rotation.
    pub const IDENTITY: Self = Self {
        position: Point3::ORIGIN,
        orientation: [1.0, 0.0, 0.0, 0.0],
    };

    /// Tolerance on `|q|² - 1` accepted by [`is_consistent`](Self::is_consistent).
    ///
    /// Generous relative to `f32::EPSILON` because producers accumulate
    /// orientation over thousands of physics steps and renormalise
    /// periodically, not every step. The check is here to catch a quaternion
    /// that was never normalised — an all-zero array, a raw axis-angle triple
    /// written into four slots — not to police numerical drift.
    pub const NORM_TOLERANCE: f32 = 1e-3;

    /// A pure translation, with identity rotation.
    #[must_use]
    pub const fn at(position: Point3) -> Self {
        Self {
            position,
            orientation: [1.0, 0.0, 0.0, 0.0],
        }
    }

    /// Whether the position is finite and the quaternion is finite and
    /// approximately unit-norm.
    ///
    /// A non-unit quaternion does not fail loudly: it scales as well as
    /// rotates, so a body renders subtly the wrong size and stays plausible.
    /// That is precisely the silent mis-render this check exists for.
    #[must_use]
    pub fn is_consistent(&self) -> bool {
        if !self.position.is_finite() || !self.orientation.iter().all(|c| c.is_finite()) {
            return false;
        }
        // Compared as a squared norm so this stays `no_std`-clean: `sqrt` is not
        // available in `core`, and the tolerance is a band either way.
        let norm_sq = self.orientation.iter().map(|c| c * c).sum::<f32>();
        (1.0 - Self::NORM_TOLERANCE..=1.0 + Self::NORM_TOLERANCE).contains(&norm_sq)
    }
}

impl Default for Transform3 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Stable handle for a [`SceneNode`], assigned by the producer.
///
/// Ids need only be unique within one [`SceneDescriptor`]; nothing outside an
/// episode interprets them. They are what makes [`ScenePose`] keyed rather than
/// positional — see the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u32);

/// Endpoints of one bone, as indices into the owning
/// [`Geometry::Skeleton`]'s `vertices`.
///
/// A named struct rather than a bare `(u32, u32)`: the tuple gave `from` and
/// `to` no names at any call site, and two same-typed positional fields
/// transpose silently. Naming them is only worth a wire change alongside a
/// `FORMAT_VERSION` bump, and this variant's introduction is that bump.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bone {
    /// Index into `vertices` of the bone's first endpoint.
    pub from: u32,
    /// Index into `vertices` of the bone's second endpoint.
    pub to: u32,
}

impl Bone {
    /// Constructs a bone between two vertex indices.
    #[must_use]
    pub const fn new(from: u32, to: u32) -> Self {
        Self { from, to }
    }
}

/// What a node is made of, in its own local frame.
///
/// The analytic solids exist so a 3-D renderer never needs a mesh on the wire.
/// A capsule is nine bytes here and a few hundred triangles if tessellated by
/// the producer; tessellation is the renderer's job, and it is the only party
/// that knows the viewport.
///
/// `#[non_exhaustive]`: a mesh variant is the obvious future addition, and a
/// renderer that skips geometry it does not recognise is the correct behaviour
/// for one anyway.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Geometry {
    /// Ordered points. `closed` joins the last to the first and fills.
    Polyline {
        /// Vertices in paint order, in the node's local frame.
        points: Vec<Point3>,
        /// `true` renders a filled polygon; `false` an open stroked path.
        closed: bool,
    },
    /// A shared vertex pool plus bones between them — joints and links without
    /// duplicating a vertex per segment.
    Skeleton {
        /// Joint positions in the node's local frame.
        vertices: Vec<Point3>,
        /// Connectivity, indexing into `vertices`.
        bones: Vec<Bone>,
    },
    /// A single point at the node's origin, drawn as a marker.
    Marker,
    /// An axis-aligned box, half-extents from its local origin.
    Box {
        /// Half-width along each local axis.
        half_extents: Point3,
    },
    /// A sphere centred on the node's local origin.
    Sphere {
        /// Radius in world units.
        radius: f32,
    },
    /// A capsule along the node's local y axis.
    Capsule {
        /// Half the length of the cylindrical section.
        half_height: f32,
        /// Radius of the cylindrical section and the end caps.
        radius: f32,
    },
}

impl Geometry {
    /// Whether this geometry's cross-field invariants hold.
    ///
    /// Checked per variant:
    ///
    /// - `Polyline` — at least two points (three when `closed`), all finite.
    /// - `Skeleton` — every bone endpoint in range of `vertices`, no bone
    ///   joining a vertex to itself, all vertices finite.
    /// - `Marker` — trivially consistent.
    /// - `Box` / `Sphere` / `Capsule` — extents finite and non-negative.
    ///
    /// **Winding is not checked**, though the payload types this replaces
    /// documented it as an invariant. Winding order has no observable effect
    /// under SVG's default `nonzero` fill rule for a simple polygon, and the
    /// points here are 3-D, where "counter-clockwise" is not defined without
    /// naming a viewing direction. Asserting it would be theatre.
    #[must_use]
    pub fn is_consistent(&self) -> bool {
        match self {
            Self::Polyline { points, closed } => {
                let min = if *closed { 3 } else { 2 };
                points.len() >= min && points.iter().all(Point3::is_finite)
            }
            Self::Skeleton { vertices, bones } => {
                let n = vertices.len();
                vertices.iter().all(Point3::is_finite)
                    && bones
                        .iter()
                        .all(|b| b.from != b.to && (b.from as usize) < n && (b.to as usize) < n)
            }
            Self::Marker => true,
            Self::Box { half_extents } => {
                half_extents.is_finite()
                    && half_extents.x >= 0.0
                    && half_extents.y >= 0.0
                    && half_extents.z >= 0.0
            }
            Self::Sphere { radius } => radius.is_finite() && *radius >= 0.0,
            Self::Capsule {
                half_height,
                radius,
            } => {
                half_height.is_finite()
                    && *half_height >= 0.0
                    && radius.is_finite()
                    && *radius >= 0.0
            }
        }
    }
}

/// One drawable, recorded once per episode in a [`SceneDescriptor`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneNode {
    /// Handle a per-frame [`ScenePose`] uses to address this node.
    pub id: NodeId,
    /// What the node is made of, in its local frame.
    pub geometry: Geometry,
    /// **Open** style key, enriched by a renderer-side table. An unrecognised
    /// key gets a neutral default style rather than a compile error, which is
    /// what lets a third party write `"thermometer"` and see something drawn.
    pub role: String,
    /// Set when the node never moves, in which case no per-frame pose need
    /// mention it. `None` means a [`ScenePose`] supplies the placement.
    pub static_transform: Option<Transform3>,
}

/// The static half of a scene: what exists, and the volume to look at.
///
/// Recorded once, at episode start, via `RecordSink::on_scene`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneDescriptor {
    /// Every drawable in the scene.
    pub nodes: Vec<SceneNode>,
    /// The volume the renderer fits its viewport to, as one [`Extent`] per
    /// axis in `(x, y, z)` order. Per-axis rather than a single cube so a
    /// non-square domain is representable.
    pub bounds: (Extent, Extent, Extent),
    /// Named background — a landscape heatmap, a ground plane — looked up by
    /// the renderer. `None` draws no background.
    pub background: Option<String>,
}

impl SceneDescriptor {
    /// Whether every node's geometry is consistent and node ids are unique.
    ///
    /// Uniqueness is the descriptor's own invariant rather than any node's: a
    /// duplicate id makes [`ScenePose`] ambiguous about which node it is
    /// addressing, which is the exact coupling the keyed design exists to
    /// remove.
    ///
    /// Quadratic in node count. This runs behind a `debug_assert!` in the
    /// recording tier, once per episode, over a handful of nodes.
    #[must_use]
    pub fn is_consistent(&self) -> bool {
        if !self.nodes.iter().all(|n| n.geometry.is_consistent()) {
            return false;
        }
        if !self
            .nodes
            .iter()
            .filter_map(|n| n.static_transform.as_ref())
            .all(Transform3::is_consistent)
        {
            return false;
        }
        !self
            .nodes
            .iter()
            .enumerate()
            .any(|(i, n)| self.nodes[..i].iter().any(|m| m.id == n.id))
    }
}

/// The per-frame half of a scene: where things are now.
///
/// This is what `FamilyPayload::Scene` carries. A node absent from
/// `transforms` keeps whatever pose it last had (or its
/// [`SceneNode::static_transform`]); a transform naming an id no scene node
/// declares is ignored.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ScenePose {
    /// Placements this frame, keyed by node id — never positional. See the
    /// module docs for why.
    pub transforms: Vec<(NodeId, Transform3)>,
    /// Transient points with no persistent identity — contacts, centre of
    /// mass, a trail sample — tagged by an open role string as
    /// [`SceneNode::role`] is.
    pub markers: Vec<(String, Point3)>,
}

impl ScenePose {
    /// Whether every transform and marker in this pose is well-formed.
    ///
    /// Deliberately does **not** check that ids resolve against a descriptor:
    /// unknown ids are defined as ignorable, so an unresolved id is not a
    /// defect, and the descriptor is not available here in any case.
    #[must_use]
    pub fn is_consistent(&self) -> bool {
        self.transforms.iter().all(|(_, t)| t.is_consistent())
            && self.markers.iter().all(|(_, p)| p.is_finite())
    }
}

/// An inclusive `[lo, hi]` range over one axis, valid by construction.
///
/// Serialises as a bare `(f32, f32)` through `#[serde(try_from, into)]`, so it
/// costs no wire change over the tuple it replaces — deserialisation simply
/// begins rejecting ranges that were always invalid.
///
/// # Why the type rather than the tuple
///
/// The payloads this replaces had **four** bare bounds pairs between them, and
/// an inverted or NaN range sailed through all four into the report client,
/// whose viewport-fitting then divided by a negative or non-finite span. The
/// client refuses that at projection time now, but refusing it at construction
/// puts the error where the producer can act on it.
///
/// # Not [`Bounds`]
///
/// `rlevo-core`'s `Bounds` carries the same `lo <= hi` invariant. It stays
/// where it is: it is a config-validation primitive named across evolution, RL,
/// and environments, nothing on the wire uses it, and this crate is a
/// dependency root that cannot reach `rlevo-core` anyway. Two types with one
/// invariant is the right outcome here — a config bound constrains a
/// hyperparameter at construction, a viewport extent describes what a renderer
/// should fit, and they share nothing else.
///
/// [`Bounds`]: https://docs.rs/rlevo-core
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "(f32, f32)", into = "(f32, f32)")]
pub struct Extent {
    lo: f32,
    hi: f32,
}

impl Extent {
    /// Constructs an extent, rejecting anything a renderer cannot fit to.
    ///
    /// # Errors
    ///
    /// [`ExtentError::NotFinite`] if either endpoint is NaN or infinite;
    /// [`ExtentError::Inverted`] if `hi < lo`.
    ///
    /// Finiteness is required, not merely `lo <= hi`. `lo <= hi` alone already
    /// excludes NaN — every comparison against NaN is false — but it admits
    /// `(-inf, inf)`, whose span is infinite and whose reciprocal is zero, so
    /// every projected coordinate collapses onto the origin. That is the same
    /// class of silent failure the invariant exists to stop, and it is free to
    /// exclude here.
    pub fn new(lo: f32, hi: f32) -> Result<Self, ExtentError> {
        if !lo.is_finite() || !hi.is_finite() {
            return Err(ExtentError::NotFinite);
        }
        if hi < lo {
            return Err(ExtentError::Inverted);
        }
        Ok(Self { lo, hi })
    }

    /// The lower endpoint.
    #[must_use]
    pub const fn lo(&self) -> f32 {
        self.lo
    }

    /// The upper endpoint.
    #[must_use]
    pub const fn hi(&self) -> f32 {
        self.hi
    }

    /// `hi - lo`. Always finite and non-negative; zero for a degenerate
    /// extent, which is legal here and refused by the renderer, since a
    /// zero-span *range* is meaningful and a zero-span *viewport* is not.
    #[must_use]
    pub const fn span(&self) -> f32 {
        self.hi - self.lo
    }
}

impl TryFrom<(f32, f32)> for Extent {
    type Error = ExtentError;

    fn try_from((lo, hi): (f32, f32)) -> Result<Self, Self::Error> {
        Self::new(lo, hi)
    }
}

impl From<Extent> for (f32, f32) {
    fn from(e: Extent) -> Self {
        (e.lo, e.hi)
    }
}

/// Why a `(lo, hi)` pair is not a valid [`Extent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ExtentError {
    /// An endpoint was NaN or infinite.
    #[error("extent endpoints must be finite")]
    NotFinite,
    /// `hi` was less than `lo`.
    #[error("extent is inverted: hi < lo")]
    Inverted,
}

/// Producer-side trait. An environment implements this to have its recording
/// ship a scene instead of falling back to `FamilyPayload::Ascii`.
///
/// Both methods are needed because the two halves are recorded at different
/// rates: the recording tier calls `scene_descriptor` once at episode start and
/// `scene_pose` on every captured frame. Splitting them across two traits would
/// let an env implement one without the other, which has no coherent meaning —
/// a pose addresses nodes only the descriptor can declare.
pub trait ScenePayloadSource {
    /// The static geometry, captured at episode start.
    ///
    /// Called once per episode. An implementation may build this fresh each
    /// time; it is not on a hot path.
    fn scene_descriptor(&self) -> SceneDescriptor;

    /// Current placements, captured at this frame.
    fn scene_pose(&self) -> ScenePose;
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    use crate::schema::bincode_config;

    fn unit_extent() -> Extent {
        Extent::new(-1.0, 1.0).expect("valid")
    }

    // -- Extent ------------------------------------------------------------

    #[test]
    fn extent_accepts_ordered_finite_ranges_including_degenerate() {
        assert!(Extent::new(-5.0, 5.0).is_ok());
        assert!(Extent::new(0.0, 0.0).is_ok(), "zero span is a valid range");
        assert!((Extent::new(-2.0, 3.0).unwrap().span() - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn extent_rejects_inverted_and_non_finite() {
        assert_eq!(Extent::new(1.0, -1.0), Err(ExtentError::Inverted));
        assert_eq!(Extent::new(f32::NAN, 1.0), Err(ExtentError::NotFinite));
        assert_eq!(Extent::new(0.0, f32::NAN), Err(ExtentError::NotFinite));
        assert_eq!(
            Extent::new(f32::NEG_INFINITY, f32::INFINITY),
            Err(ExtentError::NotFinite),
            "an infinite span projects every coordinate onto the origin",
        );
    }

    /// The `#[serde(try_from, into)]` mechanism is the whole reason `Extent`
    /// costs no wire change: it must encode byte-identically to the `(f32, f32)`
    /// it replaces, and it must *validate* on the way back in. Both halves are
    /// asserted here, under bincode specifically — the format that carries it.
    #[test]
    fn extent_encodes_as_a_bare_pair_and_validates_on_decode() {
        let e = Extent::new(-2.5, 7.5).unwrap();
        let via_extent = bincode::serde::encode_to_vec(e, bincode_config()).unwrap();
        let via_tuple = bincode::serde::encode_to_vec((-2.5f32, 7.5f32), bincode_config()).unwrap();
        assert_eq!(
            via_extent, via_tuple,
            "Extent must be wire-identical to the tuple it replaces",
        );

        let (round_tripped, _): (Extent, usize) =
            bincode::serde::decode_from_slice(&via_extent, bincode_config()).unwrap();
        assert_eq!(round_tripped, e);

        // The same bytes that decode cleanly as a tuple are refused as an
        // Extent. This is the property #942 flagged as unverified under
        // bincode; it holds.
        let inverted = bincode::serde::encode_to_vec((9.0f32, 1.0f32), bincode_config()).unwrap();
        let as_pair: Result<((f32, f32), usize), _> =
            bincode::serde::decode_from_slice(&inverted, bincode_config());
        assert!(as_pair.is_ok(), "control: the bytes are a well-formed pair");
        let as_extent: Result<(Extent, usize), _> =
            bincode::serde::decode_from_slice(&inverted, bincode_config());
        assert!(
            as_extent.is_err(),
            "validation must run on the bincode decode path, not just JSON",
        );
    }

    // -- Transform3 --------------------------------------------------------

    #[test]
    fn identity_transform_is_consistent() {
        assert!(Transform3::IDENTITY.is_consistent());
        assert!(Transform3::at(Point3::new(1.0, 2.0, 3.0)).is_consistent());
    }

    #[test]
    fn transform_rejects_unnormalised_and_non_finite_quaternions() {
        let scaled = Transform3 {
            position: Point3::ORIGIN,
            orientation: [2.0, 0.0, 0.0, 0.0],
        };
        assert!(
            !scaled.is_consistent(),
            "a non-unit quaternion scales as well as rotates — the silent case",
        );
        let zero = Transform3 {
            position: Point3::ORIGIN,
            orientation: [0.0; 4],
        };
        assert!(!zero.is_consistent(), "never-normalised, all-zero");
        let nan_pos = Transform3 {
            position: Point3::new(f32::NAN, 0.0, 0.0),
            ..Transform3::IDENTITY
        };
        assert!(!nan_pos.is_consistent());
    }

    #[test]
    fn transform_tolerates_accumulated_drift() {
        // A quaternion a few ULPs off unit after many physics steps must pass;
        // the check targets never-normalised values, not numerical drift.
        let drifted = Transform3 {
            position: Point3::ORIGIN,
            orientation: [1.0 - 1e-5, 0.0, 0.0, 0.0],
        };
        assert!(drifted.is_consistent());
    }

    // -- Geometry ----------------------------------------------------------

    #[test]
    fn skeleton_rejects_out_of_range_and_self_joining_bones() {
        let vertices = vec![Point3::ORIGIN, Point3::new(1.0, 0.0, 0.0)];
        assert!(
            Geometry::Skeleton {
                vertices: vertices.clone(),
                bones: vec![Bone::new(0, 1)],
            }
            .is_consistent()
        );
        assert!(
            !Geometry::Skeleton {
                vertices: vertices.clone(),
                bones: vec![Bone::new(0, 2)],
            }
            .is_consistent(),
            "index 2 is out of range for two vertices — #944's named defect",
        );
        assert!(
            !Geometry::Skeleton {
                vertices,
                bones: vec![Bone::new(1, 1)],
            }
            .is_consistent(),
            "a bone from a vertex to itself has no length to draw",
        );
    }

    #[test]
    fn skeleton_rejects_non_finite_vertices() {
        assert!(
            !Geometry::Skeleton {
                vertices: vec![Point3::ORIGIN, Point3::new(f32::NAN, 0.0, 0.0)],
                bones: vec![Bone::new(0, 1)],
            }
            .is_consistent()
        );
    }

    #[test]
    fn polyline_arity_depends_on_closed() {
        let two = vec![Point3::ORIGIN, Point3::new(1.0, 0.0, 0.0)];
        assert!(
            Geometry::Polyline {
                points: two.clone(),
                closed: false,
            }
            .is_consistent()
        );
        assert!(
            !Geometry::Polyline {
                points: two,
                closed: true,
            }
            .is_consistent(),
            "two points cannot enclose an area",
        );
        assert!(
            !Geometry::Polyline {
                points: vec![Point3::ORIGIN],
                closed: false,
            }
            .is_consistent(),
            "one point is a Marker, not a polyline",
        );
    }

    #[test]
    fn solids_reject_negative_and_non_finite_extents() {
        assert!(Geometry::Sphere { radius: 0.5 }.is_consistent());
        assert!(!Geometry::Sphere { radius: -0.5 }.is_consistent());
        assert!(!Geometry::Sphere { radius: f32::NAN }.is_consistent());
        assert!(
            Geometry::Capsule {
                half_height: 1.0,
                radius: 0.2
            }
            .is_consistent()
        );
        assert!(
            !Geometry::Capsule {
                half_height: 1.0,
                radius: -0.2
            }
            .is_consistent(),
            "the radius is checked, not just the half-height",
        );
        assert!(
            !Geometry::Capsule {
                half_height: -1.0,
                radius: 0.2
            }
            .is_consistent(),
            "the half-height is checked, not just the radius",
        );
        assert!(
            Geometry::Box {
                half_extents: Point3::new(1.0, 2.0, 3.0)
            }
            .is_consistent()
        );
        assert!(
            !Geometry::Box {
                half_extents: Point3::new(1.0, -2.0, 3.0)
            }
            .is_consistent(),
            "every axis is checked, not only the first",
        );
        assert!(Geometry::Marker.is_consistent());
    }

    // -- Descriptor / pose -------------------------------------------------

    fn node(id: u32, role: &str) -> SceneNode {
        SceneNode {
            id: NodeId(id),
            geometry: Geometry::Marker,
            role: role.into(),
            static_transform: None,
        }
    }

    fn descriptor(nodes: Vec<SceneNode>) -> SceneDescriptor {
        SceneDescriptor {
            nodes,
            bounds: (unit_extent(), unit_extent(), unit_extent()),
            background: None,
        }
    }

    #[test]
    fn descriptor_rejects_duplicate_node_ids() {
        assert!(descriptor(vec![node(0, "a"), node(1, "b")]).is_consistent());
        assert!(
            !descriptor(vec![node(0, "a"), node(0, "b")]).is_consistent(),
            "a duplicate id makes a keyed pose ambiguous",
        );
    }

    #[test]
    fn descriptor_checks_node_geometry_and_static_transforms() {
        let bad_geometry = SceneNode {
            geometry: Geometry::Sphere { radius: -1.0 },
            ..node(0, "a")
        };
        assert!(!descriptor(vec![bad_geometry]).is_consistent());

        let bad_transform = SceneNode {
            static_transform: Some(Transform3 {
                position: Point3::ORIGIN,
                orientation: [0.0; 4],
            }),
            ..node(0, "a")
        };
        assert!(
            !descriptor(vec![bad_transform]).is_consistent(),
            "a static transform is checked too, not only per-frame ones",
        );
    }

    #[test]
    fn pose_ignores_unresolvable_ids_by_design() {
        // The keyed design's whole point: a pose naming a node the descriptor
        // does not declare is not malformed. If this ever starts failing,
        // someone has coupled the two structures back together.
        let pose = ScenePose {
            transforms: vec![(NodeId(999), Transform3::IDENTITY)],
            markers: vec![],
        };
        assert!(pose.is_consistent());
    }

    #[test]
    fn pose_checks_its_transforms_and_markers() {
        let bad = ScenePose {
            transforms: vec![(NodeId(0), Transform3::IDENTITY)],
            markers: vec![("com".into(), Point3::new(0.0, f32::INFINITY, 0.0))],
        };
        assert!(!bad.is_consistent(), "a non-finite marker is refused");
    }

    #[test]
    fn descriptor_and_pose_round_trip_through_bincode() {
        let d = SceneDescriptor {
            nodes: vec![
                SceneNode {
                    id: NodeId(0),
                    geometry: Geometry::Skeleton {
                        vertices: vec![Point3::ORIGIN, Point3::new(0.0, 1.0, 0.0)],
                        bones: vec![Bone::new(0, 1)],
                    },
                    role: "torso".into(),
                    static_transform: None,
                },
                SceneNode {
                    id: NodeId(1),
                    geometry: Geometry::Box {
                        half_extents: Point3::new(10.0, 0.05, 10.0),
                    },
                    role: "ground".into(),
                    static_transform: Some(Transform3::IDENTITY),
                },
            ],
            bounds: (
                Extent::new(-10.0, 10.0).unwrap(),
                Extent::new(0.0, 4.0).unwrap(),
                unit_extent(),
            ),
            background: Some("ground_plane".into()),
        };
        assert!(d.is_consistent());
        let bytes = bincode::serde::encode_to_vec(&d, bincode_config()).unwrap();
        let (back, _): (SceneDescriptor, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode_config()).unwrap();
        assert_eq!(d, back);

        let p = ScenePose {
            transforms: vec![(NodeId(0), Transform3::at(Point3::new(1.0, 2.0, 0.0)))],
            markers: vec![("contact".into(), Point3::ORIGIN)],
        };
        let bytes = bincode::serde::encode_to_vec(&p, bincode_config()).unwrap();
        let (back, _): (ScenePose, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode_config()).unwrap();
        assert_eq!(p, back);
    }
}
