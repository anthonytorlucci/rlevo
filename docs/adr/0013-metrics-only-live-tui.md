---
project: rlevo
status: active
type: decision
date: 2026-06-03
tags:
  - adr
  - decision
  - architecture
  - visualisation
  - ratatui
  - tui
  - report
  - metrics
  - rlevo
---

# ADR 0013: Two-product visualisation — live metrics TUI + post-run report; AsciiRenderable demoted (supersedes 0008)

## Status
Active. Adopted 2026-06-03. **Supersedes [0008-three-tier-visualisation-ratatui-live-static-report](0008-three-tier-visualisation-ratatui-live-static-report.md).** Keeps 0008's `EpisodeRecord` seam and production-crate isolation rules. Collapses 0008's three tiers to **two products**, removes the env panel from the live tier, and demotes `AsciiRenderable` from a library-level invariant to an optional debug helper.

> Note: the filename retains the `metrics-only-live-tui` slug from this ADR's first revision. The scope grew during the same-day design discussion from "narrow the live tier" to "reframe the whole architecture as two products"; the file was revised in place rather than stacking a third same-day visualisation ADR.

## Context

ADR 0008 framed visualisation as **three tiers**: a library tier (`AsciiRenderable`, every env renders as text), a live tier (`ratatui` TUI rendering both env and metrics), and a report tier (static-HTML post-run viewer). Two observations collapse this to two.

**1. The three tiers conflated two different axes.** Two of them are *products with distinct audiences*:

- **Live** — the author at the terminal during a multi-hour run, asking *"is it learning?"* Quantitative, watched in aggregate over thousands of episodes/generations.
- **Post-run** — records + report, for the author and for dissemination (advisors, collaborators, PhD application reviewers), asking *"what did it learn?"* Qualitative, human-watchable, publication-quality.

The third "tier", `AsciiRenderable`, is **not a product or an audience**. It is an *implementation primitive* — the contract an env author implements so that something else can render the env. It was a peer of the other two on the org chart, never in function.

**2. The two products answer two orthogonal questions, and each maps to exactly one product.** "Is it learning?" is answered by live learning curves; "what did it learn?" by post-run playback. Env-render during training answers the qualitative question in the worst possible venue — at training speed (unreadable), competing with the curves for terminal space, and redundant with the report tier's superior scrubable playback. Hence the live tier should be **metrics-only** (this ADR's original conclusion, now retained as one part of the larger reframe).

**3. Once the live tier drops env-render, `AsciiRenderable` loses its load-bearing consumer.** Its only remaining customer would be the report tier — and the report tier does not want ASCII. The report exists to be a *publication-quality, shareable artifact* (the very thing that justified it over a JSONL log in 0008). Monospace ASCII embedded in a browser is the opposite of that. A good report renders real per-family visuals (a drawn cartpole, a lander with thrust vectors) in SVG/canvas, driven by **structured per-family state** that `EpisodeRecord` already must carry ("enough state to reconstruct the run offline", 0008 part 4). 0008 itself listed "per-family report renderers consume `EpisodeRecord` variants directly" as an allowed option; this ADR makes it the chosen one.

With neither product consuming `AsciiRenderable`, its remaining value is a single optional convenience — an ad-hoc debug dump of an env to text — which does not justify a mandatory library-wide invariant.

## Decision

**Adopt a two-product visualisation architecture: (1) a live metrics TUI, (2) a post-run records + report pipeline. The live TUI renders learning metrics only — no env panel. The report tier renders environment playback from structured `EpisodeRecord` state via per-family renderers internal to the report tier. `AsciiRenderable` is demoted from a library-level invariant to an optional debug-helper trait. The `EpisodeRecord` seam and the production-crate isolation rules from ADR 0008 are preserved.**

### Concrete parts

1. **Two products, not three tiers.**
   - **Live product** — `ratatui` TUI in `rlevo-benchmarks` behind the `tui` feature. Answers "is it learning?"
   - **Post-run product** — `EpisodeRecord` recording (`record` feature) feeding a static-HTML report (`report` feature) in `rlevo-benchmarks`. Answers "what did it learn?"
   - `AsciiRenderable` is no longer a third tier; it is an optional helper (part 4).

2. **Live TUI is metrics-only.**
   - Panels: reward/return sparklines (raw + moving average), loss / entropy / value / gradient-norm sparklines, episode/generation counter, throughput (steps/sec), recent log lines. On EA runs: best/mean/worst fitness and population diversity per generation.
   - **No env panel.** The TUI does not consume `AsciiRenderable`.
   - Metrics come from the existing observer/callback surfaces in `rlevo-reinforcement-learning` and `rlevo-evolution`. No new metric-emission API.
   - The `tui` feature therefore has **no dependency on env-render coverage** — any env gets live curves the moment it has a benchmark run.
   - No render-cadence/throttling concern in the run wrapper; metric panels update at observer cadence.

3. **Report renders from structured `EpisodeRecord` state.**
   - `EpisodeRecord` carries per-family structured state (e.g. cart position + pole angle; lander pose + velocity + fuel) sufficient to drive a faithful visual replay.
   - The report tier holds **per-family renderers** that consume those structured variants and draw SVG/canvas — publication-quality, not terminal-cell art.
   - Timeline scrubber, convergence plots, and population/lineage panels are unchanged from 0008.

4. **`AsciiRenderable` is demoted to an optional debug helper.**
   - It is **no longer a mandatory invariant** that every env family implements. The coverage obligation across `grids`, `toy_text`, `landscapes`, `box2d`, `locomotion` is dropped.
   - It remains available as a cheap, opt-in convenience: an env may implement it to support an ad-hoc "dump the env to text" debug affordance. Neither product depends on it.
   - Its location (`rlevo-core::render` per ADR 0009) is unchanged; only its mandate changes.

5. **`EpisodeRecord` seam and isolation rules preserved (0008 parts 4, 5, 6 carry over).**
   - `EpisodeRecord` remains the integration seam between recording and report; replay determinism inherits from `SeedStream` (ADR 0004).
   - No production crate (`rlevo-core`, `-environments`, `-reinforcement-learning`, `-evolution`, `-hybrid`) gains any viz dep. `rlevo-benchmarks` is the only crate with optional viz deps, gated by `tui` / `report` / `record`. The umbrella forwards `viz-tui` / `viz-report`.
   - Default builds compile zero viz code; no hot-path instrumentation beyond existing observer/callback surfaces.

### Reversal criteria

- If a debugging need emerges that the **post-run** product cannot serve *because it is post-run* (an env diverges so fast that waiting for a recorded episode to replay is too slow a loop, and a text dump is too lossy), reintroduce an **opt-in** env panel in the live TUI — off by default, driven by the now-optional `AsciiRenderable`. This was the runner-up in the design discussion and is deliberately kept cheap to revive.
- If the report's structured-state rendering proves too costly to maintain per family (many env families, each needing a bespoke SVG renderer), reconsider recording `AsciiRenderable` text frames as a fallback render path — which would re-promote `AsciiRenderable` toward load-bearing.
- 0008's inherited reversal criteria (terminal insufficient for live, static HTML wrong for report, `EpisodeRecord` a size bottleneck) remain valid.

## Consequences

**Positive**

- **Two products, two questions, one mapping each.** Live answers "is it learning?"; post-run answers "what did it learn?" No product straddles both, and no implementation primitive masquerades as a product tier.
- **One fewer thing to maintain.** The mandatory `AsciiRenderable` coverage obligation across five env families is gone. Env authors implement it only if they want the optional text-dump debug helper.
- **`tui` fully decoupled from env-render.** Live curves are available to any env with a run, with no render coverage and no cadence throttling.
- **Report quality is no longer terminal-bound.** Per-family SVG/canvas from structured state yields genuine publication figures — the dissemination goal that justified the report tier in the first place.
- **Cleaner conceptual model.** "Two products + an optional helper" is easier to hold in the head and to document than "three tiers" where the tiers were heterogeneous.
- **Matches the code's existing drift.** `3dea516 refactor(benchmarks): Refactor TUI rendering for metric separation` already started the metric-only direction; `Metric`/`MetricsProvider` (ADR 0004) and `AgentStats`/`PerformanceRecord` (ADR 0003) are the metric foundation.

**Negative / accepted costs**

- **No live qualitative peek.** During a run the author sees only curves until a recorded episode is available for the report. Accepted: env behaviour at training speed is near-unreadable, an optional `AsciiRenderable` dump remains for spot debugging, and the reversal criteria keep an opt-in panel cheap to revive.
- **`EpisodeRecord` must carry per-family structured state, and the report needs a renderer per family.** This shifts the per-family rendering work from the (now-deleted) live env panel to the report tier — net-neutral in volume, but concentrated where publication quality is actually wanted. New env families need a report renderer to appear in playback (they still get live curves for free).
- **Schema coupling.** `EpisodeRecord`'s per-family variants and the report's per-family renderers must stay in sync as env families evolve. Contained to the leaf crate.

**Neutral**

- `AsciiRenderable` / `AsciiRenderer` stay where ADR 0009 put them (`rlevo-core::render`); only their mandate changes.
- The keyboard pause/step/reset follow-up note from 0008 still applies to the metrics-only TUI.
- `Renderer<E>` / `NullRenderer` in `rlevo-core` are untouched.

## Alternatives considered

**Keep the three tiers as ADR 0008 defined them.** Rejected — the "library tier" was a primitive, not a product, and after the live tier goes metrics-only that primitive has no load-bearing consumer. Three heterogeneous tiers are harder to maintain and reason about than two products plus an optional helper.

**Metrics-only live tier, but keep `AsciiRenderable` as a mandatory library invariant (this ADR's first revision).** Rejected on the same-day reframe — if neither product consumes it, a workspace-wide implementation mandate is unjustified maintenance. Demoting it to optional preserves the genuine debug value at no obligation.

**Report renders by replaying recorded `AsciiRenderable` text frames.** Rejected — produces terminal-style monospace in the browser, defeating the report's publication-quality purpose. Structured-state SVG/canvas is the right fidelity for the dissemination audience. (Retained in the reversal criteria as a fallback if per-family renderers prove too costly.)

**Opt-in env debug panel in the live TUI now.** Deferred, not adopted — keeps the live product single-purpose and the `tui` feature fully decoupled from `AsciiRenderable`. Folded into the reversal criteria so it returns cheaply if a concrete mid-run debugging need appears.

## References

- [0008-three-tier-visualisation-ratatui-live-static-report](0008-three-tier-visualisation-ratatui-live-static-report.md) — **superseded by this ADR on adoption.** `EpisodeRecord` seam and isolation rules preserved; three tiers collapsed to two products; live env panel removed; `AsciiRenderable` demoted.
- [0009-move-styled-render-into-rlevo-core](0009-move-styled-render-into-rlevo-core.md) — `AsciiRenderable`/`AsciiRenderer` location in `rlevo-core::render`; unchanged here, only the mandate changes.
- [0004-move-bench-traits-into-rlevo-core](0004-move-bench-traits-into-rlevo-core.md) — `Metric`/`MetricsProvider` underpin the metric-only live TUI; `SeedStream` underwrites `EpisodeRecord` replay determinism.
- [0003-collapse-rl-modules-into-rlevo-reinforcement-learning](0003-collapse-rl-modules-into-rlevo-reinforcement-learning.md) — `AgentStats`/`PerformanceRecord` are the RL-side metric sources.
- rlevo-viz-overview — env-vis umbrella spec; needs revision to the two-product model (follow-up).
- `crates/rlevo-benchmarks/src/tui/` — live product; env panel to be removed, metric panels retained.
- `3dea516` — `refactor(benchmarks): Refactor TUI rendering for metric separation`, the implementation step this ADR formalises.
