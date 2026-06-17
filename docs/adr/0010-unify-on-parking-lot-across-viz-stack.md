---
project: rlevo
status: active
type: decision
date: 2026-05-31
tags: [adr, decision, visualisation, concurrency, mutex, parking-lot, rlevo-evolution, rlevo-benchmarks, examples]
---

# ADR 0010: Unify on `parking_lot` across the observer alias

## Status

Accepted. Implements option **A1** from viz-mutex-and-ci-gap; closes gap **G2** in
viz-examples-gaps.

## Context

The recording-tier hardening pass (commit `bde3f69`, branch `fix/vis-record-report`)
discovered that ~10 of the 17 `crates/rlevo/examples/viz/` examples did not compile, and
nobody knew — they are `required-features`-gated, so no `cargo build`/`test`/`clippy`
ever touched them.

The root cause is a lock-type split between two sibling subsystems
(viz-mutex-and-ci-gap §"Root cause A"):

- The record-sink **producers** take `Arc<parking_lot::Mutex<dyn RecordSink>>` after the
  `7d9127b` migration ("better performance in highly contended multi-threaded test
  environments"):
  - `RecordingTap::{new, with_*_payload}` (`record/env_tap.rs`)
  - `RecordingReporter::{new, without_lifecycle}` (`record/reporter.rs`)
  - `RecordingLayer::new` (`record/tracing_layer.rs`)
  - `PopulationReporter::new` (`record/population_reporter.rs`)
- The EA **observer alias** is defined over `std::sync::Mutex`:

  ```rust
  rlevo_evolution::SharedPopulationObserver
    = Arc<std::sync::Mutex<dyn PopulationObserver>>
  ```

An example that does both EA observation *and* recording —
`record_evolution_sphere_with_client` — genuinely needs **two** lock types in scope:

```rust
use std::sync::{Arc, Mutex as StdMutex};   // observer handle
use parking_lot::Mutex;                      // record sink

let sink: Arc<Mutex<dyn RecordSink>> = ...;             // parking_lot
let reporter: Arc<StdMutex<PopulationReporter>> = ...;  // std, coerces to
let observer: SharedPopulationObserver = reporter.clone();
```

That aliased dual-import is the fix that shipped, but it is a **smell**: an example has to
know that two sibling subsystems disagree on their `Mutex` type. `record_sphere_landscape`
and `report_sphere_landscape_with_client` were left on `std::sync::Mutex` because they
drive the sink directly (no producer handoff) — correct today, a latent trap the moment a
producer is added.

## Decision

**Redefine `rlevo_evolution::SharedPopulationObserver` over `parking_lot::Mutex`:**

```rust
pub type SharedPopulationObserver = Arc<parking_lot::Mutex<dyn PopulationObserver>>;
```

so that the observer alias and the record-sink producers agree on one lock type.
Consequences that ride along with the type change:

1. Add `parking_lot` as an **explicit** dependency of `rlevo-evolution` (and of
   `rlevo-benchmarks` where the producers live) — it is already a transitive workspace
   dependency, so no new third-party code enters the tree.
2. Drop the `StdMutex` alias and dual-import from `record_evolution_sphere_with_client`;
   it now uses a single `parking_lot::Mutex`.
3. Converge `record_sphere_landscape` and `report_sphere_landscape_with_client` onto
   `parking_lot::Mutex`, removing the latent footgun.
4. Observer-handle call sites drop `.lock().unwrap()` for the infallible `parking_lot`
   `.lock()` idiom.

Scope is the EA observer surface plus the viz examples; the production RL crates are
unaffected.

## Consequences

### Positive

- **One lock type across the viz stack.** No example needs two `Mutex` types or a careful
  per-example choice; the "always `parking_lot` for sinks" rule becomes uniform.
- Removes the `record_evolution_sphere_with_client` dual-import smell and the two
  sphere-example footguns in one move.
- `parking_lot`'s infallible, non-poisoning, faster-uncontended locks suit non-critical
  viz/observer state better than poison-aware `std::sync`.

### Negative / accepted costs

- **Public API change.** `SharedPopulationObserver` is part of `rlevo-evolution`'s public
  surface; any non-viz consumer constructing the alias over `std::sync::Mutex` must
  update. Acceptable: the project is alpha and the alias has few external consumers.
- `parking_lot` has no lock poisoning, so a panic while the observer lock is held leaks
  partially-mutated state rather than surfacing a `PoisonError`. Accepted — observer
  state is non-critical telemetry, not training data.
- Adds an explicit `parking_lot` dependency to `rlevo-evolution` (already transitive).

## Alternatives Considered

- **A2 — Make producers generic over the lock (`L: lock_api::RawMutex`).** Maximum
  flexibility, but viral generics thread through the entire `RecordSink` trait surface.
  Almost certainly not worth it. Rejected.
- **A3 — Document the split; standardise examples on `parking_lot` except where the
  observer forces `std`.** This is the *current* state after `bde3f69`. Cheap, but leaves
  the footgun and the dual-import smell in place. Rejected.

## Follow-on (not part of this ADR)

The companion CI gap (G1 / viz-mutex-and-ci-gap "Root cause B") — adding an
example build+clippy job so the viz examples can't silently rot again — is tracked
separately and is the higher-leverage fix; A1 makes the eventual green build uniform.

## References

- Research: viz-mutex-and-ci-gap (§"Root cause A", option A1) — the deep dive this ADR enacts.
- Research: viz-examples-gaps (G2) — parent gaps catalog.
- ADR [0008-three-tier-visualisation-ratatui-live-static-report](0008-three-tier-visualisation-ratatui-live-static-report.md) — three-tier viz architecture.
- ADR [0009-move-styled-render-into-rlevo-core](0009-move-styled-render-into-rlevo-core.md) — prior viz-surface relocation.
- Commits: `7d9127b` (introduced the `parking_lot` producer migration), `bde3f69`
  (example repair + recording-tier hardening).
