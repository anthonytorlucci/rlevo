//! Rendering abstractions for environment visualization.
//!
//! # Module layout
//!
//! | Submodule          | What it provides                                                    |
//! |--------------------|---------------------------------------------------------------------|
//! | [`ascii`]          | [`AsciiRenderable`] trait and [`AsciiRenderer`] — optional text dump helper (ADR-0013). |
//! | [`styled`]         | [`StyledFrame`] / [`StyledLine`] / [`StyledSpan`] / [`SpanStyle`] / [`Color`] / [`Modifier`] — colour-aware projection consumed by the `ratatui` TUI and the static-HTML report tier. |
//! | [`palette`]        | Project-wide semantic colour constants (`AGENT_FG`, `GOAL_FG`, `HAZARD_FG`, …). Every styled-output implementor should reach for these rather than raw [`Color`] values to satisfy the accessibility contract. |
//! | [`payload`]        | Per-family structured snapshot types ([`Landscape2DSnapshot`], [`Box2dSnapshot`], [`Locomotion2DSnapshot`], …) and the corresponding opt-in payload-source traits consumed by the report tier. |
//!
//! # Renderer trait
//!
//! [`Renderer`] is generic over frame type, so ASCII, image, and no-op
//! renderers all share the same interface. Use [`NullRenderer`] when
//! rendering is not required — `Frame = ()` means no allocation occurs and
//! the compiler eliminates every call site.
//!
//! [`AsciiRenderer`]: crate::render::AsciiRenderer
//! [`StyledFrame`]: crate::render::StyledFrame
//! [`StyledLine`]: crate::render::StyledLine
//! [`StyledSpan`]: crate::render::StyledSpan
//! [`SpanStyle`]: crate::render::SpanStyle
//! [`Color`]: crate::render::Color
//! [`Modifier`]: crate::render::Modifier
//! [`Landscape2DSnapshot`]: crate::render::Landscape2DSnapshot
//! [`Box2dSnapshot`]: crate::render::Box2dSnapshot
//! [`Locomotion2DSnapshot`]: crate::render::Locomotion2DSnapshot
//! [`NullRenderer`]: crate::render::NullRenderer

pub mod ascii;
pub mod palette;
pub mod payload;
pub mod styled;

pub use ascii::{AsciiRenderable, AsciiRenderer};
pub use payload::{
    BodyKind, Box2dPayloadSource, Box2dSnapshot, CardTable, Classic2DBody, Classic2DPayloadSource,
    Classic2DRole, Classic2DSnapshot, GridAgentMarker, GridColor, GridDir, GridDoorState,
    GridPayloadSource, GridSnapshot, GridTile, Landscape2DPayloadSource, Landscape2DSnapshot,
    Locomotion2DPayloadSource, Locomotion2DSnapshot, Point2, RigidBody2D, TabularCell, TabularGrid,
    TabularLayout, TabularMarker, TabularMarkerKind, TabularPayloadSource, TabularSnapshot,
};
pub use styled::{Color, Modifier, SpanStyle, StyledFrame, StyledLine, StyledSpan};

/// A renderer for environment `E`.
///
/// The associated `Frame` type makes rendering zero-cost when `NullRenderer` is
/// used — `Frame = ()` means no allocation occurs and the call optimises away.
///
/// ASCII text renderers pick `Frame = String`; image renderers pick
/// `Frame = Vec<u8>` or `Frame = image::RgbImage`.
pub trait Renderer<E> {
    /// The output produced by one render call.
    type Frame;

    /// Render the current state of `env` and return a frame.
    fn render(&self, env: &E) -> Self::Frame;
}

/// A no-op renderer with `Frame = ()`.
///
/// Use this as the default renderer type when rendering is not needed.
/// The compiler eliminates all calls at zero runtime cost.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullRenderer;

impl<E> Renderer<E> for NullRenderer {
    type Frame = ();

    fn render(&self, _env: &E) {}
}
