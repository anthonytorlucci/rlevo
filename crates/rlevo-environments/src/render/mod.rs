//! Renderers for `rlevo-environments` environments.
//!
//! The primary renderer is [`ascii::AsciiRenderer`], which produces
//! `String` frames for classic-control and toy-text environments.
//! Use [`rlevo_core::render::NullRenderer`] when rendering is not needed.
//!
//! Environments may also implement the [`AsciiRenderable::render_styled`]
//! method to expose a colour-aware projection consumed by the live TUI and
//! the static-HTML report tiers. See [`styled`] for the data types and
//! [`palette`] for the project-wide semantic colour constants.
pub mod ascii;
pub mod palette;
pub mod styled;

pub use ascii::{AsciiRenderable, AsciiRenderer};
pub use styled::{Color, Modifier, SpanStyle, StyledFrame, StyledLine, StyledSpan};
