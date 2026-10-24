---
project: rlevo
status: active
type: decision
date: 2026-10-24
tags: [adr, decision, architecture, visualisation, tui, rlevo-harness, rlevo-examples, rlevo-metrics-registry, xtask, public-surface, breaking-change]
---

# ADR 0096: The live metrics TUI is deleted

## Status

**Accepted (2026-10-24).**

Supersedes ADR [0013](0013-metrics-only-live-tui.md)'s live product: the
live half of Decision part 1, part 2 (the metrics-only `ratatui` TUI), the
reversal criterion that would revive an env panel in it, and, in part 5, the
`tui` gate and the umbrella's `viz-tui`. The live tier of ADR
[0008](0008-three-tier-visualisation-ratatui-live-static-report.md), which
0013 had already narrowed, goes with it.

Standing: 0013's post-run product (part 3), `AsciiRenderable`'s demotion
(part 4), and the `EpisodeRecord` seam and isolation rules of part 5; ADR
[0007](0007-visualisation-crates-isolated-from-production-crates.md)'s
isolation principle; ADR
[0009](0009-move-styled-render-into-rlevo-core.md), whose styled types the
environments, the record schema and the report client still use; and ADR
[0010](0010-unify-on-parking-lot-across-viz-stack.md) entire, since
`parking_lot` is a dependency of `record`, not of the TUI.

## Context

ADR 0013 gave visualisation two products: a live metrics-only `ratatui`
dashboard behind `rlevo-harness`'s `tui` feature, answering "is it
learning?", and a post-run record and static-HTML report, answering "what
did it learn?".

The dashboard was an early way to watch the examples train. It became a
large part of the harness for what it did:

- **Size.** 3,576 lines in the harness (the `tui` module, `env_wrappers`
  with `TuiEnvTap`, and `reporter::tui`) and a 247-line example.
- **One caller.** The `tui_ppo_cartpole` example was the only code outside
  the harness that used it. Three other files named it in doc comments.
- **Tests that ran nowhere.** The 89 tests in that code ran in no test
  tier: `cargo xtask test-fast` tests the harness with `record` only, and
  neither `test-heavy` nor `test-gpu` names `tui`. CI compiled the code
  (the viz-examples job and the docs matrix) and ran none of its tests.
- **Dependencies.** With `tui` on, the harness's normal dependency graph
  went from 34 packages to 96. It was the last code in a published crate
  that drew to a terminal.
- **Direction.** The project's rendering is to move out of the library
  crates, behind a sink an adapter crate provides, so the crates can focus
  on the algorithms. A terminal dashboard inside the harness is the
  opposite of that.

`rlevo-harness` is published for the first time in the release this
change lands in (ADR 0088). Shipping `tui` in that release, to delete it
in the next, would make a breaking change one release later than it needs
to be.

## Decision

1. **The live TUI is deleted.** In `rlevo-harness`: the `tui` feature, the
   modules `tui` (`TuiRunner`, `TuiConfig`, `TuiCaptureLayer`,
   `MetricsLayout`, the panels, the theme and the `ratatui` conversions),
   `env_wrappers` (`TuiEnvTap`, its only member) and `reporter::tui`
   (`TuiReporter`, `TuiHandle`, `TuiEvent`), and the optional `ratatui` and
   `crossterm` dependencies. In `rlevo` and `rlevo-examples`: the `viz-tui`
   feature. In `rlevo-examples`: the `tui_ppo_cartpole` example.

2. **No live product replaces it.** Progress during a run is read from
   `tracing`: PPO and PPG log one `info` event per update and
   `EvolutionaryLoop` one per generation, under the metric registry's
   names, and `LoggingReporter` logs each evaluation episode.
   `rlevo_harness::evaluator::train` returns the training episodes'
   summaries. The six off-policy agents log no progress event (ADR 0095,
   Decision 5); this decision does not change that. The dashboard's example
   took its returns from `TuiEnvTap`, not from an agent.

3. **Visualisation is one product until a sink replaces it:** the post-run
   record and static-HTML report.

4. **The rule outlives the code, and tightens.** No workspace manifest
   names `ratatui` or `crossterm`, in any dependency section or behind any
   feature: a new guard, `no-manifest-names-a-terminal-ui`, reads every
   member's dependencies from `cargo metadata`, so the dashboard cannot
   return behind a feature flag. Both stay on the viz-package list the
   reach guard checks. The stricter rule covers the terminal stack only:
   the report client, a workspace member, names `leptos` and
   `wasm-bindgen`.

5. **The registry's reading hints go with their reader.** `Trend`,
   `trend_for`, `hint_for` and `MetricDescriptor`'s `trend` and `hint`
   fields are deleted from `rlevo-metrics-registry`, with the `.reads(..)`
   chains on 20 of its rows. The dashboard was their one reader; the report
   client reads a descriptor's `name`, `kind`, `title` and `unit`, and its
   cadence through `is_per_generation`. A public item needs a consumer (ADR
   [0081](0081-a-public-item-needs-an-implementor-and-a-consumer.md)). The
   one hint that said something the row did not, that `skipped_updates` is
   cumulative, moves into the table's comment.

## Consequences

- **Breaking**, for a user of `rlevo-benchmarks/tui` or `rlevo`'s `viz-tui`
  in 0.5.0: the feature entries no longer resolve, and every `Tui*` item
  is gone. There is no migration but to read the `tracing` events above, or
  to record the run and read its report.
- No manifest in the workspace names `ratatui` or `crossterm`, and a
  guard keeps it so. The harness's graph at all features goes from 180
  packages to 143, and the umbrella's from 456 to 434.
- **Breaking** for `rlevo-metrics-registry`, which shipped `Trend` and the
  hints in 0.5.0. A viewer that wants a reading direction keeps its own
  table, keyed by the metric's name; the registry can take one back when a
  consumer in this workspace reads it.
- The harness's documented feature set is `json`, `record` and `report`;
  the docs CI job documents 19 crate-and-feature combinations, where it
  documented 21.
- No env panel can be revived in a live TUI, as 0013's reversal criterion
  allowed. A live view, if one returns, returns as a sink outside the
  harness.
- `rlevo-core`'s styled types (`StyledFrame`, `SpanStyle`, `Color`,
  `Modifier`) stay; their docs stop naming `ratatui`.

## Alternatives considered

- **Keep the TUI until a replacement sink lands.** It would ship in the
  harness's first release and break in the next, and its tests would go on
  running in no tier.
- **Add its tests to a tier.** That makes 89 tests pay for a product with
  one example, in the direction the project is moving away from.
- **Move it to an unpublished crate.** It keeps the maintenance, and
  `rlevo-examples` would still need a feature to reach it.
- **One manifest guard for every viz package.** The report client names
  `leptos` and `wasm-bindgen` and stays until a sink replaces it, so the
  guard would need exceptions; the terminal stack needs none.
- **Keep `Trend` and the hints for a future viewer.** They would be public
  items with no consumer, and adding them back later breaks nobody.

## Measured

Counts at the parent commit, which do not depend on the machine:
`wc -l` over the deleted files; `rg -c '#\[test\]'` over them;
`cargo tree -e normal -p <crate> --prefix none | sed 's/ (.*//' | sort -u
| wc -l` for the dependency graphs, before and after.

- **Public surface** (`tools/public-surface.sh`, all features). The
  harness's listing goes from 622 lines to 481 and the registry's from 30
  to 20, with no line added to either. `rlevo`'s listing is unchanged,
  since it re-exports the harness as one module; it loses the `viz-tui`
  feature. `rlevo-core`'s is unchanged.

## References

- ADR [0013](0013-metrics-only-live-tui.md): the two products, and the live
  one this decision deletes.
- ADR [0088](0088-the-harness-crate-is-named-rlevo-harness.md): the
  harness's first release under its name.
- ADR [0095](0095-the-agent-seam-is-one-agent-trait-with-defaulted-learning-hooks.md):
  the progress events PPO and PPG log, and the off-policy agents' lack of
  one.
- ADR [0015](0015-shared-typed-metric-registry-crate.md): the shared metric
  registry, which keeps its table and loses the hints.
