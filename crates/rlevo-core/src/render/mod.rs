//! Rendering abstractions for environment visualization.
//!
//! # Module layout
//!
//! | Submodule          | What it provides                                                    |
//! |--------------------|---------------------------------------------------------------------|
//! | [`ascii`]          | [`AsciiRenderable`] trait — optional text dump helper (ADR-0013). |
//! | [`styled`]         | [`StyledFrame`] / [`StyledLine`] / [`StyledSpan`] / [`SpanStyle`] / [`Color`] / [`Modifier`] — colour-aware text projection. The static-HTML report tier renders recorded [`StyledFrame`]s; the `ratatui` TUI maps only [`SpanStyle`] / [`Color`] / [`Modifier`] onto its own styles. |
//! | [`palette`]        | Project-wide semantic colour constants (`AGENT_FG`, `GOAL_FG`, `HAZARD_FG`, …). Every styled-output implementor should reach for these rather than raw [`Color`] values to satisfy the accessibility contract. |
//! | [`payload`]        | Per-family structured snapshot types ([`Landscape2DSnapshot`], [`Box2dSnapshot`], [`Locomotion2DSnapshot`], …) and the corresponding opt-in payload-source traits consumed by the report tier. |
//!
//! [`StyledFrame`]: crate::render::StyledFrame
//! [`StyledLine`]: crate::render::StyledLine
//! [`StyledSpan`]: crate::render::StyledSpan
//! [`SpanStyle`]: crate::render::SpanStyle
//! [`Color`]: crate::render::Color
//! [`Modifier`]: crate::render::Modifier
//! [`Landscape2DSnapshot`]: crate::render::Landscape2DSnapshot
//! [`Box2dSnapshot`]: crate::render::Box2dSnapshot
//! [`Locomotion2DSnapshot`]: crate::render::Locomotion2DSnapshot

pub mod ascii;
pub mod palette;
pub mod payload;
pub mod styled;

pub use ascii::AsciiRenderable;
pub use payload::{
    BodyKind, Box2dPayloadSource, Box2dSnapshot, CardTable, Classic2DBody, Classic2DPayloadSource,
    Classic2DRole, Classic2DSnapshot, GridAgentMarker, GridColor, GridDir, GridDoorState,
    GridPayloadSource, GridSnapshot, GridTile, Landscape2DSnapshot, Locomotion2DPayloadSource,
    Locomotion2DSnapshot, Point2, RigidBody2D, TabularCell, TabularGrid, TabularLayout,
    TabularMarker, TabularMarkerKind, TabularPayloadSource, TabularSnapshot,
};
pub use styled::{Color, Modifier, SpanStyle, StyledFrame, StyledLine, StyledSpan};
