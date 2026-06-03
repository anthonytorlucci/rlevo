//! Rendering abstractions for environment visualization.
//!
//! The [`Renderer`] trait is generic over frame type, so ASCII, image, and
//! no-op renderers all share the same interface. Use [`NullRenderer`] when
//! rendering is not required — all calls are eliminated by the compiler.
//!
//! The [`styled`] submodule defines the colour-aware projection consumed by
//! the live `ratatui` TUI (in `rlevo-benchmarks`) and the static-HTML report
//! tier. The [`palette`] submodule supplies the project-wide semantic
//! colour constants every styled-output implementor should reach for.

pub mod ascii;
pub mod palette;
pub mod payload;
pub mod styled;

pub use ascii::{AsciiRenderable, AsciiRenderer};
pub use payload::{
    Box2dPayloadSource, Box2dSnapshot, BodyKind, GridAgentMarker, GridColor, GridDir,
    GridDoorState, GridPayloadSource, GridSnapshot, GridTile, Landscape2DPayloadSource,
    Landscape2DSnapshot, Locomotion2DPayloadSource, Locomotion2DSnapshot, Point2, RigidBody2D,
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
