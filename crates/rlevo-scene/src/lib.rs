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
//! Producer-side *obligations* also stay in `rlevo-core::render`: the
//! `AsciiRenderable`, `Renderer`, and `*PayloadSource` traits are things an
//! environment implements, and they belong beside `Environment`. **The leaf
//! owns what goes on the wire; core owns what an environment must implement to
//! put it there.**
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
//! | [`payload`] | the per-family snapshot shapes a frame can carry |
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

pub mod palette;
pub mod payload;
pub mod styled;

pub use payload::{
    BodyKind, Box2dSnapshot, CardTable, Classic2DBody, Classic2DRole, Classic2DSnapshot,
    GridAgentMarker, GridColor, GridDir, GridDoorState, GridSnapshot, GridTile,
    Landscape2DSnapshot, Locomotion2DSnapshot, Point2, RigidBody2D, TabularCell, TabularGrid,
    TabularLayout, TabularMarker, TabularMarkerKind, TabularSnapshot,
};
pub use styled::{Color, Modifier, SpanStyle, StyledFrame, StyledLine, StyledSpan};
