//! Re-exports of the render surface defined in [`rlevo_scene`].
//!
//! Concrete `AsciiRenderable` impls live in this crate (one per env), but the
//! trait, the [`StyledFrame`] type set, the semantic palette, and the per-family
//! payload shapes all live in `rlevo-scene` so that `rlevo-benchmarks` and the
//! WASM report client can consume them without a circular package dep.
//!
//! The module is preserved at this path so existing per-env imports
//! (`use rlevo_environments::render::*`) keep working without change. That is
//! load bearing: it is why moving the surface out of `rlevo-core` touched one
//! file here rather than the ~270 call sites that reach it through this shim.
//!
//! It moved out of `rlevo-core` because a trait whose only method returns a wire
//! type cannot be separated from that type, and keeping both in core would have
//! forced `rlevo-core` — a dependency root with no first-party deps — to depend
//! on a crate built to serve the report tier. See ADR 0081.

pub use rlevo_scene::{
    AsciiRenderable, AsciiRenderer, Color, Modifier, SpanStyle, StyledFrame, StyledLine,
    StyledSpan, ascii, palette, styled,
};
