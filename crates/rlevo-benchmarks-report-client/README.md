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

## M6 — per-family playback adapters

M5.1 shipped a static frame dump; **M6** turns the per-episode detail
pane into a true playback surface:

- **Timeline scrubber** (`<input type="range">`) drives a `frame_idx`
  Leptos signal across `record.frames`.
- **Play / Pause / Restart** with **1× / 2× / 5× / 10×** speed buttons.
  The play loop runs a single `set_interval_with_handle`; it auto-pauses
  at the terminal frame, and `on_cleanup` clears the handle whenever the
  selected episode changes.
- **Per-frame readout**: `frame i/N · step N · reward ±X.XXX`.
- **Styled-frame HTML rendering** (`src/styled.rs`): wire-mirror
  `StyledFrame` → `<pre>` + colour-classed `<span>`s. Pair every colour
  with a hue-redundant signal (`font-weight: 700` for BOLD, CSS
  `currentColor` swap for REVERSED) so a B/W screenshot still conveys
  agent / goal / hazard meaning.
- **Family dispatch** off `manifest.env_family` (`src/adapters/mod.rs`):
  - `Classic` (CartPole / MountainCar / Pendulum / Acrobot).
  - `Grids` (Minigrid-style envs — agent heading glyphs, walls, goals, hazards).
  - `ToyText` (FrozenLake, CliffWalking, Taxi, Blackjack — tile grid + agent).
  - Box2d / Landscapes / Locomotion ship M7 SVG adapters — see below.
  - Any future family hits the generic fallback adapter until a
    dedicated one lands.

The umbrella crate ships paired examples for every covered family:
`record_cartpole` / `record_grids` / `record_toy_text` produce
recordings; `report_cartpole_with_client` / `report_grids_with_client` /
`report_toy_text_with_client` wrap them into single-file HTML reports.

## M7 — per-family SVG adapters for box2d, landscapes, locomotion

M7 lights up the report tier for the three remaining 2D families. The
wire format bumps to **`FORMAT_VERSION = 2`** — M6 records (version 1)
still decode through the same loader because `FamilyPayload::Ascii`
stays at bincode tag 0, so the v1 tag layout is preserved.

- **`Landscape2D`** payload — search-domain bounds, current candidate,
  best-so-far, capped trail, landscape label. The
  `adapters/landscape.rs` SVG renders a bounded rectangle background +
  trail polyline + current-position disk + best-so-far cross-ring.
  Heatmap is deferred to M7.1 — only the candidate dynamics ship in M7.
- **`Locomotion2D`** payload — **the canonical view for the family**,
  since locomotion envs do not implement `AsciiRenderable` per
  ADR-0008. The `adapters/locomotion.rs` SVG renders a sagittal-plane
  stick figure: bones, joint disks, ground line, optional CoM cross,
  contact rings. Auto-fits the viewport with a minimum half-range so a
  static pose still renders visibly.
- **`Box2dBodies`** payload — per-body polygon transforms keyed off the
  `BodyKind` discriminant (hull / wheel / leg / wing / ground / goal /
  other), plus contact points. The `adapters/box2d.rs` SVG transforms
  each body's local-frame vertices into world space via the captured
  `(position, rotation_rad)` pose.

Per the project accessibility contract, every colour pairs with a
distinct shape / stroke style (joints filled disks, CoM cross, contacts
open rings, ground dashed, trail dim-dashed), so a black-and-white
screenshot of the report still reads.

### Producer-side opt-in

Envs become payload sources by implementing one of:

- `rlevo_core::render::Landscape2DPayloadSource`
- `rlevo_core::render::Box2dPayloadSource`
- `rlevo_core::render::Locomotion2DPayloadSource`

`RecordingTap` then exposes per-family convenience constructors —
`with_landscape_payload(env, sink)` / `with_box2d_payload(env, sink)` /
`with_locomotion_payload(env, sink)` — that wire the snapshot into
every captured frame. Locomotion routes through a new
`RecordingTap::new_headless` constructor that drops the
`AsciiRenderable` bound; ASCII / styled fields ship as `None` and the
M7 SVG is the only rendering pathway.

The umbrella crate ships paired M7 examples:

- `record_sphere_landscape` / `report_sphere_landscape_with_client`
  (writes via `RecordSink` directly — landscapes are pure fitness
  functions, not Environments).
- `record_inverted_pendulum` / `report_inverted_pendulum_with_client`.
- `record_lunar_lander` / `report_lunar_lander_with_client`.

## M8 — convergence plots

M8 lights up the report tier's training-diagnostic surface: a new
**Convergence** section between the episode table and the per-episode
playback pane renders per-run scalar charts via
[`leptos-chartistry`](https://docs.rs/leptos-chartistry) (pure-Rust
SVG, no JS interop per umbrella spec §3 constraint #6). Wire format is
unchanged — every panel below consumes data already in the M4-era
record (`MetricSample` stream, per-frame rewards).

Panel inventory (default visible):

- **Episode reward** — sum of `frame.reward` per episode, raw + rolling
  mean (N=50, shrinks to `max(1, len/4)` for short runs).
- **Episode length** — `frames.len()` per episode, same window.
- **Policy loss / Value loss / Loss** — `MetricSample` series, rolling
  mean (N=20). RL-only — absent on harness records that ship without
  `RecordingLayer`.
- **Policy entropy / Approx KL / Clip fraction** — same N=20 window.
- **Best / Mean / Worst fitness / Best fitness (ever)** — EA-only; per
  generation, no smoothing (already aggregated).

Empty panels suppress entirely, so a CartPole harness recording shows
just the two derived panels and a PPO recording shows the full RL
suite. EA runs naturally hide the RL panels and surface the
`*_fitness` series.

The raw / smoothed pair pairs **line width** (1.0 / 2.5) with colour so
a B/W screenshot still distinguishes the two — the project's
hue-redundant a11y contract. Dark-mode override blocks in `app.css`
keep mid-luma hues legible on dark canvases.

### Loading and caching

`src/inline_data.rs::read_all_episode_records()` decodes every episode
block referenced by the index on first call and caches the result in a
`OnceLock` so panel re-renders are zero-cost. Per-record decode
failures are skipped so a single malformed episode does not blank the
panel grid.

### New umbrella example

`record_ppo_cartpole_with_client` runs a headless 12 k-step PPO
training cycle on CartPole, wiring `RecordingTap` for frames and
`RecordingLayer` for the canonical metric stream, then emits the
single-file report in one shot — the natural M8 demo because it
exercises every RL canonical metric in `CANONICAL_METRICS`.

## Deferred to M8.1+

- **Population / lineage panels** ([`evolution-population`](../../docs/specs.md) §3) —
  needs `PopulationSample` in the schema (`FORMAT_VERSION = 3` bump) +
  a `PopulationReporter` producer + EA-loop observer instrumentation.
  Box plot, diversity trace, lineage DAG, selection-pressure indicator
  all wait on that infrastructure.
- **Hybrid scatter** — depends on `PopulationSample` plus a per-individual
  `inner_rl_return` field.
- **Multi-seed aggregation** — `run_group` field on `MetricSample::aux`
  + cross-record loading + ±std band rendering.
- **Static SVG export per panel** — `leptos-chartistry` is already
  SVG-native; the toolbar export button + page-level tile export are
  M8.2 polish.
- **Axis-mode toggle** (`step | episode | wallclock`), **panel reorder**,
  **`localStorage` layout persistence**, **hover crosshair with raw
  values**, **downsampling for >10 k-sample series** — all M8.2.
- **Landscape heatmap background** — sampled per known label
  client-side; M7.1.
- **BipedalWalker / CarRacing / Swimmer / Reacher / DoublePendulum**
  examples — fall out structurally from the shared `Box2dBodies` /
  `Locomotion2D` SVG renderer; M7.1.

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
