# rlevo-benchmarks-report-client

Leptos/WASM client for the [rlevo-benchmarks](../rlevo-benchmarks/) static-HTML
report (Milestone 5.1).

This crate is **not** a build-time or runtime dependency of any production
`rlevo` crate. It produces a WASM blob + JS shim + CSS bundle that the
`rlevo-benchmarks::report::html::emit_static_html` function optionally
inlines into a single self-contained `index.html`.

## Build

The build is **out-of-band** — `cargo build -p rlevo-benchmarks` does not
invoke trunk. Run trunk explicitly when you change the client:

```bash
# One-time toolchain setup
rustup target add wasm32-unknown-unknown
cargo install trunk

# Build (run from this crate's directory)
cd crates/rlevo-benchmarks-report-client
trunk build --release
```

`trunk build --release` produces a `dist/` directory containing:

```
dist/
├── rlevo-benchmarks-report-client.js        ← wasm-bindgen JS shim
├── rlevo-benchmarks-report-client_bg.wasm   ← compiled WASM module
├── app.css                                  ← bundled styles
└── index.html                               ← standalone shell (unused by the inliner)
```

The host-side emitter loads these via `ClientAssets::from_trunk_dist(&dir)`.

## Architecture

The client decodes the four stable `<script>` block ids the M5 emitter
inlines into every report:

| Script id | Type | Content |
|-----------|------|---------|
| `rlevo-manifest` | `application/json` | `RunManifest` (run id, env family, seed, hyperparameters) |
| `rlevo-warnings` | `application/json` | `[OpenWarning]` non-fatal load conditions |
| `rlevo-episode-index` | `application/json` | `[EpisodeMeta]` summary per episode |
| `rlevo-episode-NNNNNN` | `application/octet-stream` | Base64-encoded raw `.rec` bytes (16-byte preamble + bincode header + length-prefixed `RecordChunk`s) |

`src/wire.rs` is a **bincode-compatible mirror** of
`rlevo-benchmarks::record::schema`. The cross-crate test
`tests/wire_format_compat.rs` in the host crate round-trips a populated
record through both sides on every `cargo test`, so any field-order or
field-type drift fails the host test suite.

`src/inline_data.rs` is the DOM glue — `read_manifest()` /
`read_episode_index()` / `read_warnings()` / `read_episode_record(id)`.

`src/app.rs` is the Leptos app: manifest header + warnings banner +
interactive episode table + per-episode detail pane.

## M5.1 vs M6+

M5.1 ships the **data-binding skeleton**: the client mounts, decodes,
and renders all four inlined block types end-to-end. It deliberately
ships a generic episode-table UI rather than per-family playback —
those adapters land in M6+ (classic/grids/toy-text first, then
box2d/landscapes/locomotion).

## Standalone development

Trunk can also serve the client standalone for fast iteration:

```bash
cd crates/rlevo-benchmarks-report-client
trunk serve
```

There's no inlined data when served standalone (the script blocks are
absent), so the client renders error banners — useful for debugging
the error-path UI.

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE)
or [MIT License](../../LICENSE-MIT) at your option.
