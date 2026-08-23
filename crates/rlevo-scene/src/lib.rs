//! Wire types and codec for rlevo's binary episode-record format.
//!
//! This crate is the **single definition** of everything that crosses the
//! recorder-to-report boundary as bincode. It exists so that there is exactly
//! one such definition: before ADR 0081 every payload shape existed three times
//! — as a producer-facing `*Snapshot` in `rlevo-core::render`, as a
//! "bincode-stable mirror" in `rlevo-benchmarks::record::schema`, and again in
//! the WASM client's `wire.rs` — kept honest by a drift-guard test over a
//! hand-written fixture.
//!
//! # Why a separate crate
//!
//! The report client cannot depend on `rlevo-core`: that crate pulls in burn,
//! and burn reaches `rand` and then `getrandom`, which does not build for
//! `wasm32-unknown-unknown` without extra feature gating. But the types
//! themselves never needed burn — the modules moved here import `serde` and
//! little else. The fork existed because of the code's address, not its
//! contents.
//!
//! # What is here, and what is deliberately not
//!
//! The boundary is **the transport**, not the crate diagram:
//!
//! | Crosses as | Keyed by | Drift behaviour | Where it lives |
//! |---|---|---|---|
//! | bincode | position / variant tag | **silent corruption** | here, defined once |
//! | JSON | field name | graceful — unknown ignored, missing become `None` | mirrored, on purpose |
//!
//! `RunManifest` travels to the client as JSON, so it stays in
//! `rlevo-benchmarks` and the client keeps a small serde view of it. That also
//! keeps `ObjectiveSense` in `rlevo-core`, where it belongs: it is an
//! optimisation concept, not a record one. Mirror what is self-describing;
//! share what is positional.
//!
//! # The producer-side traits live here too
//!
//! `AsciiRenderable`, `Renderer`, and the six `*PayloadSource` traits moved here
//! with the shapes they return. ADR 0081 originally put them in
//! `rlevo-core::render` on the reasoning that an obligation on an environment
//! type belongs beside `Environment` — but a trait whose only method returns a
//! wire type cannot be separated from that type without one crate depending on
//! the other, and the direction that would have forced is the wrong one.
//!
//! **`rlevo-core` is a dependency root and must stay one.** It has no
//! first-party dependencies, and neither do `rlevo-metrics-registry` or this
//! crate; those three are the roots the rest of the workspace is built on.
//! Making core depend on a crate created to serve the *report* would invert
//! that, and the precedent ADR 0081 cites — `rlevo-metrics-registry`, ADR
//! 0015 — sits **above** core as a sibling root, exactly as this crate does.
//!
//! Nothing was lost by moving them: `rlevo-core/src/render/` referenced nothing
//! else in core, so it was already a detached subtree living at the wrong
//! address. ADR 0013 also demotes `AsciiRenderable` to *"an optional debug
//! helper — env families are **not** required to implement it"*, so these traits
//! were never part of the core contract.
//!
//! # `no_std`
//!
//! `no_std` with `alloc`: the types need `String` and `Vec`, which is the
//! honest ceiling. The property that matters is unchanged — no burn, no rand,
//! no getrandom, and no transitive path to any of them. `wasm32-unknown-unknown`
//! is a CI target rather than an assumption.
//!
//! # Module layout
//!
//! | Module | What it provides |
//! |---|---|
//! | [`styled`] | [`StyledFrame`] and friends — the colour-aware text projection both the TUI and the report render |
//! | [`palette`] | semantic colour constants over [`Color`] |
//! | [`payload`] | the per-family snapshot shapes a frame can carry, plus the opt-in `*PayloadSource` traits that produce them |
//! | [`ascii`] | [`AsciiRenderable`] / [`AsciiRenderer`] — the optional plain-text tier |
//!
//! # `Bounds` stays in `rlevo-core`
//!
//! ADR 0081 decision 1 listed `Bounds` (ADR 0027) as leaf content, reasoning
//! that ADR 0082 adopts it for `SceneDescriptor` and the leaf would otherwise
//! invert its own dependency. Measured, that trade is wrong: `Bounds` is a
//! **config-validation primitive**, named in 47 files across `rlevo-evolution`,
//! `rlevo-reinforcement-learning`, and `rlevo-environments` — 173 mentions in
//! the first two alone — for things like `$\log \sigma$` ranges that have
//! nothing to do with a record. Nothing on today's wire uses it either: the
//! payload types carry bare `(f32, f32)` and `(Point2, Point2)`.
//!
//! So the leaf does not need it now, and moving a workspace-wide primitive into
//! a record crate to pre-satisfy a later ADR would put it in the wrong place
//! permanently. When ADR 0082 adds `SceneDescriptor`, the viewport range it
//! needs is a *wire* concern and gets its own type here; that a config bound and
//! a viewport extent share an invariant does not make them the same type.

#![no_std]
#![deny(missing_docs)]

extern crate alloc;

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
