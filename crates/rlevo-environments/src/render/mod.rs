//! Re-exports of the render surface defined in [`rlevo_core::render`].
//!
//! Concrete `AsciiRenderable` impls live in this crate (one per env), but
//! the trait, the [`StyledFrame`] type set, and the semantic palette all
//! live in `rlevo-core` so that `rlevo-benchmarks` can consume them
//! without a circular package dep. The module is preserved at this path so
//! existing per-env imports (`use rlevo_environments::render::*`) keep
//! working without change.

pub use rlevo_core::render::{
    AsciiRenderable, AsciiRenderer, Color, Modifier, SpanStyle, StyledFrame, StyledLine,
    StyledSpan, ascii, palette, styled,
};
