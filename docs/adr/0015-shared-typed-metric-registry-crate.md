---
project: rlevo
status: active
type: decision
date: 2026-06-05
tags:
  - adr
  - decision
  - architecture
  - metrics
  - benchmarks
  - report
  - wire-format
  - rlevo
---

# ADR 0015: Shared typed metric registry leaf crate

## Status

Active. Adopted 2026-06-05. Adds one crate (`rlevo-metrics-registry`) to the workspace's
closed crate set (`rules.md §1`). Extends — does not supersede — [0013-metrics-only-live-tui](0013-metrics-only-live-tui.md)
and [0014-record-schema-v6-single-agent-richness-and-provenance](0014-record-schema-v6-single-agent-richness-and-provenance.md): the `EpisodeRecord`
seam, `FORMAT_VERSION = 6`, and production-crate viz isolation are all preserved. Driven by
the architecture review of the post-training-report enhancement effort.

## Context

The canonical training-metric list lived as a flat `pub const CANONICAL_METRICS: &[&str]`
in `crates/rlevo-benchmarks/src/metrics_registry.rs`. Two problems surfaced when planning
the RL-vs-EO report grouping:

1. **The list was forked with no guard.** The report client compiles to `wasm32` and
   cannot depend on `rlevo-benchmarks` (which transitively pulls in `burn → rand →
   getrandom`), so the list was **hand-copied** into
   `rlevo-benchmarks-report-client/src/series.rs`. The doc comment there claimed drift was
   "caught by the cross-crate wire-format compat test," but that test exercises wire
   *types* and `FORMAT_VERSION` only — it never compared the two metric lists. A metric
   could be correctly emitted, correctly recorded, and then **silently demoted** to the
   end of the report's panel grid (un-grouped, un-titled) because the client's copy was
   stale. This breaks the moment anyone adds a metric — which is the whole point of the
   registry.

2. **A flat `&[&str]` cannot carry metric semantics, so the client re-derived them by
   hand in three disconnected places:** `is_per_generation()` (a hardcoded EA-name list),
   `pretty_metric_title()` (a hardcoded title match), and — proposed for this effort — a
   *fourth* hardcoded name list for RL-vs-EO panel grouping. The flat-string abstraction
   had already failed; adding grouping on top would have compounded it.

The root cause is the wasm boundary: the *string list itself* has no native-only
dependency, but it lived in a crate that does.

## Decision

Extract a new `#![no_std]`, zero-dependency leaf crate **`rlevo-metrics-registry`** holding
a single typed table:

```rust
pub enum MetricKind { Rl, Eo, Shared }
pub enum Cadence { PerUpdate, PerGeneration, PerEpisode }
pub struct MetricDescriptor {
    pub name: &'static str,      // exact tracing field name (the wire contract)
    pub kind: MetricKind,        // drives report RL/EO panel grouping
    pub cadence: Cadence,        // PerGeneration ⇒ no rolling-mean overlay
    pub title: &'static str,     // report panel title
    pub unit: Option<&'static str>,
}
pub const CANONICAL_METRICS: &[MetricDescriptor] = &[ /* … */ ];
pub fn descriptor(name) -> Option<&MetricDescriptor>;
pub fn is_canonical_metric(name) -> bool;
pub fn is_per_generation(name) -> bool;
pub fn title_for(name) -> &str;
```

Both `rlevo-benchmarks` and `rlevo-benchmarks-report-client` depend on it. The benchmarks
`metrics_registry` module becomes a thin re-export (the recorder/TUI call
`is_canonical_metric` unchanged). The report client **deletes its fork** and derives
panel grouping (`descriptor(name).kind`), cadence (`is_per_generation`), and titles
(`title_for`) from the table.

### Why a leaf crate (not the mirror-and-guard alternative)

A test-only stopgap (`assert!(registry == client_mirror)` in the compat test) was
considered. It guards drift but keeps two copies and cannot carry the typed semantics
without mirroring a struct table by hand — strictly worse than mirroring a string list.
The leaf crate makes the registry a genuine single source of truth: nothing to mirror,
nothing to guard, and the typed descriptors live in exactly one place. The cost is one
crate added to the closed set, which this ADR authorises.

## Consequences

- **Single source of truth.** Adding a metric = one `MetricDescriptor` row + a matching
  `tracing::info!` field. The field shows up in the live TUI, the on-disk record, and the
  report's RL/EO grouping with no client-side edit. The silent-demotion trap is gone.
- **Grouping derives from data, not code.** RL-vs-EO sectioning in the report reads
  `descriptor.kind`; the three hardcoded shadow-taxonomies collapse to registry lookups.
- **Closed-crate-set change.** `rlevo-metrics-registry` joins the workspace. It is
  `#![no_std]` with no deps, so it builds on every target including `wasm32` and adds no
  weight to non-recording builds.
- **Production-crate isolation preserved.** The algorithm crates still emit metrics purely
  as `tracing::info!` field names; they do **not** depend on the registry crate. The
  field-name string remains the wire contract (the typo-on-emit failure mode is unchanged
  and is covered by producer-side tests).
- **No `FORMAT_VERSION` bump.** This is a code-organisation change; the on-disk record
  format is untouched.

## Alternatives considered

- **Mirror + compat-test guard** (no new crate): rejected — keeps duplication and cannot
  hold the typed table cleanly.
- **Keep the flat `&[&str]` and add a fourth hardcoded grouping list in the client:**
  rejected — entrenches the failed abstraction the architecture review flagged.
- **Make the report client depend on `rlevo-benchmarks` directly:** rejected — violates
  the wasm build constraint (`burn → rand → getrandom`).

## References

- Architecture review (2026-06-05), findings #1 (unguarded fork) and #2 (typed registry).
- `research/2026-06-05-rl-vs-eo-learning.md` §9 — the RL/EO taxonomy the `MetricKind`
  split encodes.
- Implements the registry half of the rlevo-viz-overview report enhancement.
