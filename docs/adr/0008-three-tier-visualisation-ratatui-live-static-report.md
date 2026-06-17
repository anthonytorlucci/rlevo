---
project: rlevo
status: active
type: decision
date: 2026-05-27
tags:
  - adr
  - decision
  - architecture
  - visualisation
  - ratatui
  - leptos
  - tui
  - rlevo
---

# ADR 0008: Three-tier visualisation — ratatui for live, static HTML for replay (supersedes 0006/0007)

## Status
**Superseded by [0013-metrics-only-live-tui](0013-metrics-only-live-tui.md) on 2026-06-03** — the three tiers collapse to two products (live metrics TUI + post-run records/report); the live env panel is removed; the report renders env playback from structured `EpisodeRecord` state; and `AsciiRenderable` is demoted from a library-level invariant to an optional debug helper. The `EpisodeRecord` seam and the production-crate isolation rules established here remain in force.

Adopted 2026-05-27. **Supersedes [0006-leptos-first-visualisation-defer-bevy](0006-leptos-first-visualisation-defer-bevy.md) and [0007-visualisation-crates-isolated-from-production-crates](0007-visualisation-crates-isolated-from-production-crates.md)**. Adopted alongside the v2 revision of env-vis umbrella spec.

## Context

ADR 0006 framed visualisation as a single problem solved by a single technology: a Leptos web client served by an `axum` server embedded in the training binary, streaming env frames and metrics over WebSocket. ADR 0007 followed by establishing strict crate-isolation rules to keep that stack out of production crates' dependency cones.

Two realisations have invalidated that framing:

1. **"Live during training" and "shareable after training" are different products with different customers.** Live is the author debugging a multi-hour run — needs to answer "is it learning?" in the terminal where `cargo run` was issued. Shareable-after is advisors, collaborators, and PhD application reviewers consuming a finished artifact — needs to be a single file, hostable anywhere, with no server. ADR 0006 collapsed both into one Leptos-server product and accepted significant transport-layer complexity to do so: `axum`, WebSocket upgrade, `protocol_version` handshake, broadcast channel, `tokio` runtime in the training binary, "browser is read-only" awkwardness, `cfg(target_arch = "wasm32")` shrapnel.
2. **`rlevo` is a library, not an application.** Users will write their own environments. The library-level rendering surface needs to be cheap to implement, universally applicable, and free of opinionated tooling. `AsciiRenderable` already satisfies all three — it is just under-used (only the `classic` family implements it). The original spec layered an entirely separate `Visualize` trait + `EnvFrame` enum on top of `AsciiRenderable` to feed the web stack; that duplicated the env-rendering concept across two surfaces in service of one new audience.

Separately, **`ratatui`** (https://ratatui.rs) emerged as the natural live-tier technology:

- Consumes `String` frames directly via `Paragraph` — `AsciiRenderable` is exactly the input shape.
- Built-in widgets (`Sparkline`, `Gauge`, `Chart`, scrolling log via `Paragraph` + `Wrap`) cover the live debugging panel inventory without external chart crates.
- Native-only target, no WASM. ~30 transitive deps via `crossterm`. No `tokio` runtime, no `axum`, no WebSocket.
- Works over SSH on remote training boxes without port-forwarding — a real win for HPC and cloud runs that ADR 0006's local-port-only model did not address.
- Owns its own event loop. A keyboard pause/step affordance is locally available if and when we want it; the live tier is not stuck "read-only" by transport.

This rebalances the design. Visualisation is not one product. It is three:

- **Library tier** — every env renders as text via `AsciiRenderable`. Always available, zero deps, the contract a user implements when building their own env.
- **Live tier** — a `ratatui` dashboard wrapping a benchmark run. Env panel sourced from `AsciiRenderable`; metric panels from the existing observer/callback surfaces in `rlevo-reinforcement-learning` and `rlevo-evolution`. Feature-gated in `rlevo-benchmarks`.
- **Report tier** — a static-HTML viewer rendered post-run from a recorded `EpisodeRecord` file. Leptos compiled to WASM, bundled with the data inlined; the output is a single self-contained file. Feature-gated in `rlevo-benchmarks` (or a sibling crate; see Decision §3).

## Decision

**Adopt a three-tier visualisation architecture. Library-level rendering is `AsciiRenderable` in `rlevo-environments`. Live training visualisation is a `ratatui` TUI in `rlevo-benchmarks` behind a `tui` feature. Post-run visualisation is a static-HTML Leptos viewer in `rlevo-benchmarks` (or a sibling `rlevo-benchmarks-report` crate) behind a `report` feature, consuming a recorded `EpisodeRecord` file. There is no embedded server in the training binary, no WebSocket transport, and no `rlevo-viz-core` / `rlevo-viz-web` crate split.**

### Concrete parts

1. **`AsciiRenderable` is promoted to library-level invariant.**
   - Every env family in `rlevo-environments` implements it: `classic` (done), `grids`, `toy_text`, `landscapes`, `box2d` (text approximation), `locomotion` (unicode skeleton, sagittal plane).
   - Stays in `crates/rlevo-environments/src/render/`. No change to its location or trait signature.
   - The original spec's separate `Visualize` trait is **dropped**. There is one env-rendering surface at the library level, not two.

2. **Live tier: `rlevo-benchmarks` `tui` feature.**
   - New module `rlevo-benchmarks/src/tui/` introduced when the feature is enabled.
   - Pulls in `ratatui` + `crossterm`. No other transitive UI deps.
   - Wraps a benchmark run with a terminal dashboard: env panel (rendering `AsciiRenderable`), reward sparkline, loss / entropy / gradient-norm sparklines, generation/episode counter, recent log lines.
   - Consumes metrics from the existing observer/callback surfaces in algorithm crates. No new metric-emission API.
   - Keyboard affordances are out of scope for this milestone; the TUI is read-only by choice (not by transport). A follow-up may add pause/step/reset.

3. **Report tier: `rlevo-benchmarks` `report` feature (or `rlevo-benchmarks-report` sibling crate).**
   - Generates a single self-contained `index.html` from a recorded `EpisodeRecord` file: env playback, timeline scrubber, convergence plots, population/lineage panels.
   - Built with Leptos compiled to WASM, with the data inlined into the HTML. No server at runtime; no hosting required.
   - Crate-location decision (single crate with feature vs. sibling crate) deferred to the v2 umbrella spec — both options preserve isolation; the trade is feature-flag complexity vs. crate count.

4. **`EpisodeRecord` is the integration seam between live and report.**
   - File format: `bincode` with a versioned schema header (CBOR considered; format choice deferred to spec).
   - Lives in `rlevo-benchmarks` (or the sibling report crate's input module). Algorithm crates remain unaware of it.
   - The benchmark runner optionally writes one record per episode (or per generation) when the `record` feature is enabled.
   - Records carry enough state to reconstruct a full run offline. Replay determinism inherits from the existing `SeedStream` contract in `rlevo-core` (ADR 0004).

5. **No dependency from production crates on any viz dep.**
   - `rlevo-core`, `rlevo-environments`, `rlevo-reinforcement-learning`, `rlevo-evolution`, `rlevo-hybrid` do not depend (prod or dev) on `ratatui`, `crossterm`, `leptos`, `axum`, `wasm-bindgen`, or any chart crate.
   - `rlevo-benchmarks` is a leaf crate and is the only production crate that gains the optional viz deps, gated by features.
   - The umbrella `rlevo` crate exposes `viz-tui` and `viz-report` features that forward to `rlevo-benchmarks`'s `tui` and `report` features respectively.

6. **Training runs unaffected when all viz features are off.**
   - Default `cargo build` of any crate compiles zero viz code.
   - No instrumentation hooks in the training-loop hot path beyond the existing observer/callback surfaces.

### Reversal criteria

This ADR is reversible. A successor spec should be written if any of the following become true:

- The terminal becomes insufficient for live debugging in practice (e.g. multi-seed comparison or richer env rendering proves necessary mid-run, and screenshotting the TUI is too lossy). At that point a web *live* tier returns to the table, evaluated against the cost of the transport layer it brings.
- The static-HTML report turns out to be the wrong export format (e.g. paper-figure cinematic export becomes a hard requirement and static HTML cannot carry it). Revisit Bevy, `rerun.io`, or native viewers per ADR 0006's reversal criteria, which remain conceptually valid for the report tier.
- The `EpisodeRecord` file format proves a bottleneck (multi-GB recordings for long runs). At that point evaluate streaming chunked records or selective-channel recording.

## Consequences

**Positive**

- **Three tiers, three audiences, three homes.** No single product trying to be both live-debugger and dissemination artifact.
- **Live tier is materially simpler than ADR 0006's proposal.** No `axum`, no WebSocket, no `protocol_version` handshake, no `tokio` runtime added for transport, no `cfg(target_arch = "wasm32")`. `ratatui` owns its own event loop.
- **Report tier strengthens dissemination.** A static HTML file beats a URL-to-a-local-port: send by email, host on GitHub Pages, embed in a portfolio, archive as a PhD application artifact. Reproducible offline.
- **Library-level rendering is one trait, not two.** `AsciiRenderable` is what users implement; both higher tiers consume it (live directly, report by replaying recorded ASCII frames or via richer per-family adapters internal to the report tier).
- **Production-crate build cones unchanged.** `cargo build -p <any-production-crate>` does not compile `ratatui`, `leptos`, or any viz dep.
- **SSH-friendly out of the box.** Remote training boxes Just Work — the terminal the user already has open is the dashboard.
- **Per-tier version churn contained.** `ratatui`'s churn touches one feature in one leaf crate; Leptos's churn touches one other feature in the same crate (or its sibling). Neither reaches production crates.

**Negative / accepted costs**

- **Two render technologies long-term.** Live uses `ratatui` widgets; report uses Leptos + SVG/canvas. They do not share rendering code. Acceptable — they serve different audiences with different fidelity requirements, and `AsciiRenderable` is the common upstream input.
- **Live tier rendering quality is terminal-cell-bound.** `locomotion` and `box2d` reduce to unicode block-char approximations live. Sufficient for "is it walking?"; insufficient for publication figures. Publication figures live in the report tier by design.
- **No live remote-collaborator viewing.** Two people cannot watch the same training run in different browsers as they could under ADR 0006's server model. Mitigated by the report tier's shareable static artifact, which serves the collaboration use case after the run.
- **`EpisodeRecord` format is a new artifact to maintain.** Versioning, backward compatibility, and migration when the format evolves. Manageable as a leaf-crate concern.
- **Two feature flags on `rlevo-benchmarks` (`tui`, `report`).** A small surface increase on its `Cargo.toml`. Worth it for the isolation.

**Neutral**

- The `Renderer<E>` / `NullRenderer` traits in `rlevo-core` stay as-is. `AsciiRenderable` and `AsciiRenderer` in `rlevo-environments` stay as-is.
- The new crates proposed by ADR 0006 (`rlevo-viz-core`, `rlevo-viz-web`) are **not created**. The transport-agnostic-core abstraction made sense when "live" and "report" shared a transport; with separate technologies for each tier, there is nothing for that core to abstract over.
- ADR 0001's "adapt at the consumer, not the producer" precedent continues to hold: env families are not aware of either viz tier; both tiers adapt to env families internally.

## Alternatives considered

**Keep ADR 0006's Leptos-server live tier as proposed.** Rejected. The transport layer (`axum`, WebSocket, protocol versioning, broadcast channel, `tokio` runtime) is significant complexity in service of a use case (live mid-run peek) that `ratatui` solves with a fraction of the deps and works on a remote box without port-forwarding. The Leptos technology investment is not lost — it moves to the report tier where its strengths (rich charts, SVG, retained-mode browser primitives) are the right fit.

**`ratatui` for live, no report tier at all (just `tracing` + a JSONL log).** Rejected. The dissemination story (advisors, PhD applications) needs a richer artifact than a JSONL file. The report tier costs Leptos as a build-time-only dep and produces a self-contained HTML artifact — cost is bounded, value is real.

**Static HTML report only, no live tier.** Rejected. The multi-hour-run "is it learning?" problem is real and cheap to solve with `ratatui`. Going without it pushes the answer to "wait until the run finishes," which is a meaningful debugging regression for the author.

**Live tier in `rlevo` (umbrella) instead of `rlevo-benchmarks`.** Rejected. The TUI wraps a *run*, and runs are owned by `rlevo-benchmarks` per ADR 0001. Putting the TUI in the umbrella would create a tangled concept (the umbrella driving a benchmark run is now also the home of the dashboard wrapping it). The umbrella exposes the feature; the implementation lives where the runs do.

**`Visualize` trait alongside `AsciiRenderable` in `rlevo-environments`.** Rejected. The original spec introduced `Visualize` to feed the Leptos web stack with typed `EnvFrame` snapshots. With the web-live tier gone and the report tier consuming recorded `EpisodeRecord` files (which can embed ASCII frames or per-family typed snapshots, the latter being a report-tier-internal concern), there is no need for a second env-rendering trait at the library level. `AsciiRenderable` is sufficient. Per-family report-tier renderers consume `EpisodeRecord` variants directly.

**Leptos-in-the-terminal via `crossterm` instead of `ratatui`.** Rejected. Leptos's reactive primitives buy nothing in a terminal context (no DOM, no SSR, no signals-to-text rendering pipeline that beats `ratatui`'s widget model). `ratatui` is purpose-built for this surface; Leptos belongs in the static-HTML tier where its strengths apply.

**`bevy_ratatui` for unified live + render-to-image export.** Rejected. Adds Bevy as a transitive dep for the TUI use case, which is exactly the thing ADR 0006 already concluded is overkill. Per ADR 0006's reversal criteria, Bevy returns to the table only if the report tier or 3D playback needs it; the live tier does not.

## References

- rlevo-viz-overview v2 — env-vis umbrella spec, revised 2026-05-27 to reflect this ADR.
- [0001-keep-environments-and-benchmarks-separate](0001-keep-environments-and-benchmarks-separate.md) — established `rlevo-benchmarks` as the home for run orchestration; this ADR extends that home to cover both live and report visualisation.
- [0004-move-bench-traits-into-rlevo-core](0004-move-bench-traits-into-rlevo-core.md) — `SeedStream` (in `rlevo-core`) underwrites the deterministic-replay contract for `EpisodeRecord`.
- [0005-examples-and-cross-crate-tests-in-umbrella](0005-examples-and-cross-crate-tests-in-umbrella.md) — examples that demonstrate viz live in `crates/rlevo/examples/viz/`, gated by `required-features = ["viz-tui"]` or `["viz-report"]`.
- [0006-leptos-first-visualisation-defer-bevy](0006-leptos-first-visualisation-defer-bevy.md) — **superseded by this ADR on adoption**. The "no Bevy this milestone" conclusion is preserved; the "Leptos web client served by an embedded `axum` server" conclusion is replaced by `ratatui`-for-live + static-HTML-for-report.
- [0007-visualisation-crates-isolated-from-production-crates](0007-visualisation-crates-isolated-from-production-crates.md) — **superseded by this ADR on adoption**. The isolation principle is preserved and strengthened (zero viz deps in production crates; viz lives entirely inside `rlevo-benchmarks` features). The `rlevo-viz-core` / `rlevo-viz-web` crate split is dropped.
- `https://ratatui.rs/` — TUI framework adopted for the live tier.
- `crates/rlevo-environments/src/render/ascii.rs` — `AsciiRenderable` / `AsciiRenderer` (preserved; coverage to be expanded to all env families).
- `crates/rlevo-benchmarks/` — gains optional `tui`, `report`, and `record` features.
