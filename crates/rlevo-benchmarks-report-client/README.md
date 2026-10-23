# rlevo-benchmarks-report-client

![Alt Text](https://raw.githubusercontent.com/anthonytorlucci/rlevo/main/rlevo-logo.png)

Leptos/WASM client for the [rlevo-harness](../rlevo-harness/) static-HTML
report.

This crate is **not** a build-time or runtime dependency of any production
`rlevo` crate. It produces a WASM blob + JS shim + CSS bundle that the
`rlevo-harness::report::html::emit_static_html` function optionally
inlines into a single self-contained `index.html`.

## Build

The build is **out-of-band** — `cargo build -p rlevo-harness` does not
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

The client decodes the four stable `<script>` block ids the emitter
inlines into every report:

| Script id | Type | Content |
|-----------|------|---------|
| `rlevo-manifest` | `application/json` | `RunManifest` (run id, env family, seed, hyperparameters, plus ADR-0014 run provenance) |
| `rlevo-warnings` | `application/json` | `[OpenWarning]` non-fatal load conditions |
| `rlevo-episode-index` | `application/json` | `[EpisodeMeta]` summary per episode |
| `rlevo-episode-NNNNNN` | `application/octet-stream` | Base64-encoded raw `.rec` bytes (16-byte preamble + bincode header + length-prefixed `RecordChunk`s) |

`src/wire.rs` is a **bincode-compatible mirror** of
`rlevo-harness::record::schema`. This crate's cross-crate test,
`tests/wire_format_compat.rs`, round-trips a populated record through
both sides, so any field-order or field-type drift fails this crate's
test suite. It reaches the harness through a dev-dependency with the
`record` feature on, so a plain `cargo test` runs it, and the harness's
own test build links no Leptos. The client decodes only records whose
format version equals its own `FORMAT_VERSION`.

`src/inline_data.rs` is the DOM glue — `read_manifest()` /
`read_episode_index()` / `read_warnings()` / `read_episode_record(id)`.

`src/app.rs` is the Leptos app: manifest header + warnings banner +
interactive episode table + per-episode detail pane.

## Per-family playback adapters

The per-episode detail pane is a playback surface:

- **Timeline scrubber** (`<input type="range">`) drives a `frame_idx`
  Leptos signal across `record.frames`.
- **Play / Pause / Restart** with **$1\times$ / $2\times$ / $5\times$ / $10\times$** speed buttons.
  The play loop runs a single `set_interval_with_handle`; it auto-pauses
  at the terminal frame, and `on_cleanup` clears the handle whenever the
  selected episode changes.
- **Per-frame readout**: $\text{frame } i/N \cdot \text{step } N \cdot \text{reward} \pm X.XXX$.
- **Styled-frame HTML rendering** (`src/styled.rs`): wire-mirror
  `StyledFrame` → `<pre>` + colour-classed `<span>`s. Pair every colour
  with a hue-redundant signal (`font-weight: 700` for BOLD, CSS
  `currentColor` swap for REVERSED) so a B/W screenshot still conveys
  agent / goal / hazard meaning.
- **Family dispatch** off `manifest.env_family` (`src/adapters/mod.rs`):
  - `Classic` (CartPole / MountainCar / Pendulum / Acrobot).
  - `Grids` (Minigrid-style envs — agent heading glyphs, walls, goals, hazards).
  - `ToyText` (FrozenLake, CliffWalking, Taxi, Blackjack — tile grid + agent).
  - Box2d / Landscapes / Locomotion — see below.
  - Any future family hits the generic fallback adapter until a
    dedicated one lands.

The umbrella crate ships one combined example per covered family —
`report_ppo_cartpole_with_client`, `report_grids_with_client`,
`report_toy_text_with_client` — each of which records a run and wraps it
into a single-file HTML report in one binary (see `rlevo-examples`).

## Per-family SVG adapters for box2d, landscapes, locomotion

Box2d, landscapes and locomotion each carry a `FamilyPayload` variant
that the family's SVG adapter draws:

- **`Landscape2D`** payload — search-domain bounds, current candidate,
  best-so-far, capped trail, landscape label. The
  `adapters/landscape.rs` SVG renders a closed-form fitness heatmap
  background (sphere / ackley / rastrigin, monochrome low→high ramp) +
  trail polyline + current-position disk + best-so-far cross-ring.
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

- `rlevo_core::render::Box2dPayloadSource`
- `rlevo_core::render::Locomotion2DPayloadSource`

`RecordingTap` then exposes per-family convenience constructors —
`with_box2d_payload(env, sink)` / `with_locomotion_payload(env, sink)` —
that wire the snapshot into every captured frame. A landscape is not an
environment, so it has no source trait and no constructor; a
`Landscape2D` payload is built by the caller. Locomotion routes through the
`RecordingTap::new_headless` constructor that drops the
`AsciiRenderable` bound; ASCII / styled fields ship as `None` and the
SVG adapter is the only rendering pathway.

The umbrella crate ships one combined example per family:

- `report_sphere_landscape_with_client` (writes via `RecordSink` directly
  — landscapes are pure fitness functions, not Environments).
- `report_inverted_pendulum_with_client`.
- `report_lunar_lander_with_client`.

## Convergence plots

The **Convergence** section, between the episode table and the
per-episode playback pane, renders per-run scalar charts. Every chart in
the report is hand-rolled pure-Rust SVG (`src/charts.rs` /
`src/series.rs`), with no JS interop and no charting dependency. The
panels below are computed from the record's frames and its
`MetricSample` stream.

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

### Umbrella example

`report_ppo_cartpole_with_client` runs a headless 12 k-step PPO
training cycle on CartPole, wiring `RecordingTap` for frames and
`RecordingLayer` for the canonical metric stream, then emits the
single-file report in one shot — the natural demo for these panels
because it exercises every RL canonical metric in `CANONICAL_METRICS`.

## Population panels

The **Population** section sits between the Convergence section and the
per-episode playback pane. It draws the
[`PopulationSample`](../rlevo-harness/src/record/schema.rs) chunks
(bincode tag 2) an EA run records once per generation.

Panel inventory (rendered when `population_samples` is non-empty in
the run):

- **Fitness distribution per generation** — hand-rolled SVG box plot.
  Per generation: filled `[Q1, Q3]` rectangle, horizontal median tick,
  vertical whiskers clipped at the Tukey $1.5\times$IQR fence, outliers
  as small open circles. Three overlay polylines (best / median / worst)
  pair colour with distinct dash patterns (`solid` / `4 2` / `1 3`) so
  the trio is distinguishable in a B/W screenshot — the project's
  hue-redundant a11y contract.
- **Diversity trace** — `(generation, diversity)` scalar series, one
  point per sample that carries a diversity value. Suppressed when no
  sample provides one (`EvolutionaryLoop` emits `None` — see
  *Diversity computation* under Still deferred below).
- **Selection-pressure indicator** — `best / median` ratio per
  generation, skipping zero-median degeneracy. Reuses the Convergence
  section's `interactive_line_view`.

### Producer surface

[`rlevo_evolution::PopulationObserver`](../rlevo-evolution/src/observer.rs)
is the EA-side hook the report producer attaches to. Wiring:

```text
EvolutionaryLoop::new(...)
    .with_observer(reporter.clone())
                  ↓ on_population(snapshot)        (per generation,
PopulationReporter                                  post-Strategy::tell)
    ↓ sink.lock().on_population_sample(sample)
Arc<Mutex<dyn RecordSink>>                          → RecordChunk::Population
                                                      → .rec stream
```

`PopulationReporter` is shipped from
[`rlevo_harness::record::population_reporter`](../rlevo-harness/src/record/population_reporter.rs).
The loop pays the device→host transfer of the fitness tensor only
when an observer is attached; observerless runs are unchanged.

### Umbrella example coverage

No shipping example currently wires `PopulationReporter` end-to-end into
a report-client HTML report; `report_sphere_landscape_with_client`
exercises the landscape playback surface but does not attach a
population observer, so its reports have no Population section. Wiring
a GA-on-Sphere example with `.with_observer(reporter.clone())` plus
`RecordingLayer` for the canonical EA metrics (`best_fitness` /
`mean_fitness` / `worst_fitness` / `best_fitness_ever`) is open work.

### Loading and caching

`src/inline_data.rs::read_all_population_samples()` walks the cached
episode records (`read_all_episode_records`) and concatenates every
record's `population_samples` vector behind its own `OnceLock`. RL-only
runs return an empty slice and the Population section `view!`'s out to
an empty span — non-EA reports add nothing to the rendered tree.

## Chart features

Beyond the panels above, `src/charts.rs` / `src/series.rs` provide:

- **Hover crosshair** with an exact raw-sample readout, accurate even on
  a decimated path (`interactive_line_view`).
- **Per-panel static SVG export** — a per-chart download button builds an
  `HtmlAnchorElement` and triggers it.
- **Downsampling for long series** — `downsample_minmax` decimates the
  drawn path (peaks/troughs preserved) while the tooltip still reads the
  full-resolution series.
- **Axis-mode toggle** (`step | episode | wallclock`) via the `AxisMode`
  enum and a reactive UI switch.
- **Multi-seed $\pm$std bands** — `metric_band` / `BandPoint` /
  `distinct_seed_count` feed `band_chart_view`'s filled envelope.
- **Strip-plot overlay** on the box plot, toggleable via an "Individual
  points" button.
- **Threshold horizontal-rule annotation** on the diversity trace
  (`low_diversity_threshold`, user-editable).
- **Landscape heatmap background** (closed-form sphere / ackley /
  rastrigin, monochrome ramp) — `adapters/landscape.rs`.

## Still deferred

- **Lineage DAG** — requires per-Strategy parent tracking.
  `Strategy::tell` returns `(State, StrategyMetrics)` with no parent
  map; threading parent indices through GA + CMA-ES + replacement
  strategies is its own multi-step effort. The schema field
  `parents_of_best` is already in place; the panel renders only when
  non-empty across generations.
- **Hybrid scatter** — depends on a production `rlevo-hybrid` driver
  (`crates/rlevo-hybrid/src/lib.rs` is a skeleton). The schema field
  `inner_rl_returns` is in place; the panel renders only when non-empty.
- **Diversity computation** — `PopulationSnapshot::diversity` is
  emitted as `None` from `EvolutionaryLoop` because the loop has
  no strategy-agnostic geometry over the population tensor. A
  `Strategy::diversity(state) -> Option<f32>` extension is the natural
  follow-up; until it lands the diversity panel (and its threshold rule)
  suppresses cleanly.
- **Panel reorder + `localStorage` layout persistence** — the panel grid
  is still fixed-order and non-persistent across reloads.
- **BipedalWalker / CarRacing / Swimmer / Reacher / DoublePendulum**
  examples — fall out structurally from the shared `Box2dBodies` /
  `Locomotion2D` SVG renderer.

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
