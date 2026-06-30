---
project: rlevo
status: superseded
type: decision
date: 2026-05-26
tags:
  - adr
  - decision
  - architecture
  - visualisation
  - leptos
  - web
  - rlevo
---

# ADR 0006: Leptos-first visualisation; defer Bevy and native 3D

## Status

**Superseded by [0008-three-tier-visualisation-ratatui-live-static-report](0008-three-tier-visualisation-ratatui-live-static-report.md) on 2026-05-27**, which itself was superseded by [0013-metrics-only-live-tui](0013-metrics-only-live-tui.md) on 2026-06-03. The "no Bevy this milestone" conclusion is preserved; the "Leptos web client served by an embedded `axum` server" conclusion is replaced. Retained for historical context.

## Context

`rlevo` reports agent behaviour through `tracing` logs and `AsciiRenderable` text frames (`crates/rlevo-environments/src/render/ascii.rs`). Only the `classic` environment family implements ASCII rendering; `grids`, `toy_text`, `box2d`, `locomotion`, and `landscapes` have no rendering pathway at all. Convergence curves, evolutionary population statistics, and trajectory replays are not visualised — they exist only as numbers in stdout.

Two engine options were evaluated for filling that gap:

1. **Bevy** — pure-Rust ECS game engine with native 2D and 3D rendering, `bevy_egui` for in-window plots. Native window, in-process channels between the training loop and the renderer.
2. **Leptos** — pure-Rust reactive web framework compiled to WASM, served by an `axum` server embedded in the training binary. Browser client, WebSocket transport.

A third shape (Leptos primary + Bevy-in-WASM for 3D panels) was sketched and rejected — see Alternatives.

The choice has four load-bearing inputs:

- **The locomotion family is the only one that genuinely needs 3D.** Every other family (`classic`, `grids`, `toy_text`, `box2d`, `landscapes`) is natively 2D. Picking an engine that excels at 3D in order to serve one of six families is a poor trade.
- **Most debugging value lives in 2D panels.** Rollout scrubbing, convergence curves, action histograms, value heatmaps, population box/violin plots, lineage DAGs — all 2D. Bevy and Leptos are both adequate here; Leptos has a richer plot ecosystem (Plotly, leptos-chartistry, charming).
- **Dissemination matters.** `rlevo` is positioned as a research-velocity tool with a path to a PhD application. A shareable URL ("see my run at `https://…/runs/42`") is materially better for advisors and collaborators than a local-only window.
- **Author familiarity.** The author already builds with Leptos; activation energy for shipping a Leptos UI is lower than for shipping a Bevy UI.

The reason 3D matters at all is the `locomotion` family. The training stack already uses `rapier3d` for physics; what is missing is a 3D *renderer* for the resulting poses. Bevy is the obvious Rust answer. The cost of taking it for that one purpose is a heavy dependency (compile times, binary size), a pre-1.0 engine with breaking minor releases, and a second render stack to maintain in parallel with whatever covers the 2D panels.

## Decision

**Build the visualisation layer as a Leptos web client (`rlevo-viz-web`) served by an `axum` server embedded in the training binary. Do not adopt Bevy in this milestone. Render `locomotion` environments as a 2D sagittal-plane skeleton projection. Re-evaluate 3D rendering as a successor spec after the Leptos stack ships.**

The decision has five concrete parts:

1. **Web client only.** No native window, no `egui`, no `bevy`. The training binary opens a local port; the user opens a browser tab. Multi-user, remote-access, port-forwarding all fall out naturally.
2. **Pure Rust UI source.** The Leptos client compiles to WASM. JS interop is permitted only for the chart libraries that have no acceptable Rust equivalent (Plotly for violin/box, scatter density). No hand-written JS files in the repo.
3. **Transport-agnostic core.** A separate `rlevo-viz-core` crate owns the `Visualize` trait, `EnvFrame` snapshot variants, `ViewerEvent` protocol, `MetricsBuffer`, and recorder/replay. `rlevo-viz-web` consumes that core. Adding a future native viewer (Bevy or otherwise) reuses the core unchanged.
4. **Locomotion gets a 2D projection this milestone.** A sagittal-plane skeleton — joint positions projected to `(x, z)`, bones as line segments, ground line, centre-of-mass marker, footstep contact dots. Enough to diagnose "is it walking, is it falling, is it pronking." Full 3D playback is explicitly deferred.
5. **Browser is read-only.** The client consumes events; it does not pause, step, or steer training. Interactive control is a follow-up that can land after the read-only milestone proves the architecture.

### Reversal criteria

This ADR is reversible. The successor spec for native / 3D rendering should be written if any of the following become true:

- The 2D locomotion projection turns out to be insufficient for debugging the locomotion family in practice, and 3D playback is needed for forward progress on those environments.
- Frame-rate bandwidth over WebSocket becomes the bottleneck for high-fidelity envs (humanoid-style state vectors at 60 Hz prove too costly on the wire, and server-side rendering into a video stream is the cheaper alternative).
- Headless cinematic export (MP4 / GIF for paper figures) becomes a hard requirement, and the browser-via-headless-Chromium path proves too brittle.

In each case the successor adds a *second* viewer that reuses `rlevo-viz-core`; it does not replace the Leptos client.

## Consequences

**Positive**

- **Shareable runs.** A URL is the artifact. Advisors and collaborators see live training without installing anything.
- **One window for env view + plots.** Leptos hosts both via simple layout; no second render stack.
- **Richer plot ecosystem.** Plotly (JS interop) and leptos-chartistry (pure Rust SVG) cover convergence curves, multi-seed bands, violin/box per generation, and scatter with density overlay — uniformly better than `egui_plot`.
- **Smaller render-side surface area.** SVG + Canvas2D, both retained-mode browser primitives, with screenshot-friendly outputs and well-trod accessibility patterns (`prefers-color-scheme`, colourblind palettes).
- **Static-HTML export falls out naturally.** A finished run renders to a single self-contained `.html` file with the timeline functional (spec: rollout-and-replay-web).
- **No Bevy version churn to manage.** Bevy is pre-1.0; pinning and upgrading is real work. We skip it for now.

**Negative / accepted costs**

- **3D fidelity for locomotion is deferred.** The 2D projection is a stopgap. Users who want true 3D playback wait for the successor spec.
- **Live high-bandwidth envs cost network.** A humanoid at 60 Hz with full joint state is non-trivial on the wire. Mitigated by `viz.frame_stride` (server-side decimation) and client-side frame dropping; not eliminated.
- **Headless cinematic export is harder.** "Spin up headless Chromium, screenshot the page" is workable for static images but more work than Bevy's headless render plugin would be for MP4. Cinematic export is out of scope this milestone (spec: rollout-and-replay-web §8).
- **Interactive control round-trips through WS.** Pause / step / reset have latency. The decision to ship a read-only browser sidesteps this for now; interactive control inherits the latency cost when it lands.
- **One JS-interop dependency (Plotly).** Accepted for the 10% of panels (violin, box, scatter density) where pure-Rust SVG charting is genuinely worse. Capped at one library; no `charming`, no ECharts directly.

**Neutral**

- The `Renderer<E>` and `AsciiRenderable` traits in `rlevo-core` and `rlevo-environments` stay as-is. The new `Visualize` trait is additive — environments may implement none, one, or both rendering surfaces.
- `bevy_egui` is no longer relevant. If a future native viewer is built, the engine choice is re-opened from scratch with the experience of the Leptos stack in hand.

## Alternatives considered

**Bevy as the primary UI.** Rejected. The unified 2D + 3D story is real, but it's overkill for a workspace where five of six env families are natively 2D and the dissemination story matters. Bevy's pre-1.0 churn, compile-time cost, and local-window-only deployment lose against Leptos's shareable URLs and richer plot ecosystem. The one place Bevy clearly wins — 3D locomotion playback and headless video export — is exactly the area we are deferring.

**Leptos primary + Bevy-in-WASM for the 3D panel.** Rejected for this milestone. The integration is feasible (Bevy supports targeted-canvas mounting), but it locks us into maintaining two render stacks simultaneously, doubles the WASM bundle size for users who don't need 3D, and increases the surface area for Bevy-version churn. If 3D becomes a hard requirement, this shape returns to the table — but only after the Leptos-only stack has proven that the rest of the spec is right.

**`egui` standalone (with `eframe`) for a native window.** Rejected. Solves the local debugging case adequately but loses the dissemination story entirely, and the plot ecosystem (`egui_plot`) is weaker than Plotly + leptos-chartistry. Adds zero value over the Leptos shape.

**`rerun.io` as the primary visualisation surface.** Rejected as primary. Genuinely strong for ML/robotics replay (3D, plots, scrubbing built in), but it is a separate viewer process, weaker for interactive control, and adopting it as primary would couple `rlevo` tightly to the `rerun` data model and protocol. Worth revisiting as an *optional secondary adapter* for offline analysis — captured in the rlevo-viz-overview Parking Lot.

**Three.js via `wasm-bindgen` for a 3D panel.** Rejected. Hand-rolling Three.js interop in Rust is painful and adds a substantial JS dependency. If 3D returns, evaluate Bevy-in-WASM first.

## References

- rlevo-viz-overview — env-vis umbrella spec, 2026-05-26.
- viz-core — `rlevo-viz-core` crate spec (Visualize trait, event protocol, recorder).
- env-visualization-web — per-family Leptos render adapters.
- training-plots-web — chart library choice and panel inventory.
- evolution-population-web — population, diversity, and lineage views.
- rollout-and-replay-web — trajectory capture, scrubbing, static HTML export.
- [0007-visualisation-crates-isolated-from-production-crates](0007-visualisation-crates-isolated-from-production-crates.md) — companion ADR on crate boundary.
- `crates/rlevo-viz-core/`, `crates/rlevo-viz-web/` — new crates introduced by the umbrella spec.
- `crates/rlevo-core/src/render.rs` — existing `Renderer<E>` / `NullRenderer` traits, preserved.
- `crates/rlevo-environments/src/render/ascii.rs` — existing `AsciiRenderable` trait and `AsciiRenderer`, preserved.
