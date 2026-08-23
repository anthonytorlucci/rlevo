# rlevo-scene

Wire types and codec for rlevo's binary episode-record format.

This crate is the **single definition** of everything that crosses the
recorder-to-report boundary as bincode. It exists so that there is exactly one
such definition. Before [ADR 0081](../../docs/adr/0081-the-record-format-is-a-leaf-crate.md)
every payload shape existed three times — as a producer-facing `*Snapshot` in
`rlevo-core::render`, as a "bincode-stable mirror" in
`rlevo-benchmarks::record::schema`, and again in the WASM report client's
`wire.rs` — kept honest by a drift-guard test over a hand-written fixture.

## Why a separate crate

The report client cannot depend on `rlevo-core`: that crate pulls in burn, and
burn reaches `rand` and then `getrandom`, which does not build for
`wasm32-unknown-unknown` without extra feature gating. The types themselves
never needed burn. The fork existed because of the code's address, not its
contents.

## The boundary is the transport

| Crosses as | Keyed by | Drift behaviour | Where it lives |
|---|---|---|---|
| bincode | position / variant tag | **silent corruption** | here, defined once |
| JSON | field name | graceful — unknown ignored, missing become `None` | mirrored, on purpose |

`RunManifest` travels to the client as JSON, so it stays in `rlevo-benchmarks`
and the client keeps a small serde view of it. That also keeps `ObjectiveSense`
in `rlevo-core`, where it belongs — it is an optimisation concept, not a record
one.

**Mirror what is self-describing; share what is positional.**

Producer-side *obligations* also stay in `rlevo-core::render`: `AsciiRenderable`,
`Renderer`, and the `*PayloadSource` traits are things an environment
implements, and they belong beside `Environment`. The leaf owns what goes on the
wire; core owns what an environment must implement to put it there.

## Guarantees

- `#![no_std]` with `alloc`. The types need `String` and `Vec`, which is the
  honest ceiling.
- `#![deny(missing_docs)]` from the first commit. There was nothing to
  grandfather, so the class is self-policing.
- No burn, no rand, no getrandom, and no transitive path to any of them.
- Builds for `wasm32-unknown-unknown` with **no feature gating**, and that is a
  CI target rather than an assumption.

## Dependencies

`serde`, `bincode`, and `thiserror`, all taken from
`[workspace.dependencies]` so this crate cannot drift from the writer and the
reader it sits between. That matters most for `bincode`: it is the codec, and a
version skew across the three crates would be a silent wire break — the same
class of defect this crate exists to make impossible for the *types*.

ADR 0081 specified "serde and bincode, nothing else". That was amended during
implementation: `BoundsError` and `DecodeError` both derive `thiserror::Error`,
and the measurement behind the original clause came from grepping `use` lines,
which a fully-qualified derive path never appears on. thiserror 2.0 is
`no_std`-capable, is a proc-macro plus `core`, and already ships in the report
client's wasm32 cone.

## Licence

MIT OR Apache-2.0, as the workspace.
