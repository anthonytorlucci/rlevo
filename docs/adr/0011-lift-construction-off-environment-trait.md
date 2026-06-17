---
project: rlevo
status: accepted
type: decision
date: 2026-05-31
tags: [adr, decision, environment, traits, decorators, recording, visualisation, rlevo-core, rlevo-benchmarks]
---

# ADR 0011: Lift construction (`new`) off the `Environment` trait

## Status

**Accepted (2026-06-03), implemented in PR2** (branch
`migrate/viz-pr2-constructable-env`), bundled with the ADR-0013 two-product
visualisation migration (umbrella rlevo-viz-overview §14 / gap **G9** in
viz-examples-gaps). The reframe made the scope tractable: **M-A already
deleted `RenderTap` and the `TuiEnvTap` frame-capture half**, so by the time
this landed only `RecordingTap` and a slimmed `TuiEnvTap` still carried the
degenerate `new` stub — both removed here.

**Chosen shape: Option 1 (separate factory trait), with one refinement —
`ConstructableEnv` is *not* a supertrait of `Environment`.** It is a
standalone `pub trait ConstructableEnv { fn new(render: bool) -> Self; }` in
`rlevo-core`, so a type can be one without the other (decorators are
`Environment` but not `ConstructableEnv`; nothing forces the pairing). See the
Implementation note below for the divergence from the originally-sketched
`ConstructableEnv: Environment`.

## Context

`rlevo_core::environment::Environment` (`crates/rlevo-core/src/environment.rs`) puts
construction on the behavioural trait:

```rust
pub trait Environment<const D: usize, const SD: usize, const AD: usize> {
    // ... associated types ...
    fn new(render: bool) -> Self;
    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError>;
    fn step(&mut self, action: Self::ActionType) -> Result<Self::SnapshotType, EnvironmentError>;
}
```

`new(render: bool)` conflates two unrelated concerns:

1. **Behaviour** — `reset` / `step`, which a decorator legitimately delegates to its
   inner env.
2. **Construction** — producing an instance from a single `bool`, which a decorator
   *cannot* meaningfully satisfy because it also needs its sink / handle / extractor.

The recording- and TUI-tier env decorators are forced to implement a degenerate
`new` purely to satisfy the trait shape:

- `RecordingTap::new` (`crates/rlevo-benchmarks/src/record/env_tap.rs`) builds a
  local `NullSink` whose every method is a no-op, so frames silently vanish:

  ```rust
  fn new(render: bool) -> Self {
      // No meaningful standalone constructor — callers use
      // `RecordingTap::new(env, sink)`. This impl exists only to
      // satisfy the trait shape.
      struct NullSink; /* impl RecordSink with empty methods */
      Self::new_headless(E::new(render), Arc::new(Mutex::new(NullSink)), |_| FamilyPayload::Ascii)
  }
  ```

- `TuiEnvTap::new` (`crates/rlevo-benchmarks/src/env_wrappers/tui_env_tap.rs`) creates
  a `TuiHandle` channel and immediately drops the receiver, so every render push
  returns `false` into a dead channel.

Both carry comments admitting the impl exists only to satisfy the bound and that
callers should use the real explicit constructor (`Type::new(inner, sink/handle)`).

For contrast, the harness-facing `BenchEnv` trait has **no** `new` method, so its
wrapper `RenderTap` (`env_wrappers/render_tap.rs`) is constructed cleanly inside the
`Suite` env-factory closure — no degenerate stub. That is the shape this ADR wants
for `Environment`.

## Decision (proposed)

Remove `fn new(render: bool) -> Self` from `Environment`. Options for where
construction lands, to be chosen if/when this is scheduled:

- **Option 1 — separate factory trait.** A `ConstructableEnv: Environment { fn new(render: bool) -> Self; }`
  blanket-or-derived for the built-in envs. Decorators implement only `Environment`
  (behaviour) and are built via their explicit constructors; only code that needs
  default construction bounds on `ConstructableEnv`.
- **Option 2 — drop the `bool`, use config constructors.** Most envs already have
  `with_config(...)`; standardise on `Default` + `with_config` and delete the
  `render: bool` parameter entirely (render is a recording/TUI concern, not an env
  concern — consistent with ADR [0007-visualisation-crates-isolated-from-production-crates](0007-visualisation-crates-isolated-from-production-crates.md)).

Either way, the `NullSink` / dead-channel stubs in `RecordingTap` and `TuiEnvTap`
disappear.

## Consequences

### Positive
- Removes two silent-failure stubs (`NullSink`, dead `TuiHandle`) that exist only to
  satisfy the trait.
- Aligns `Environment` with the cleaner `BenchEnv` shape (no construction on the
  behavioural trait).
- Reinforces ADR 0007: render/viz concerns stay off the production env surface.

### Negative / why deferred
- **Wide blast radius.** Every `impl Environment` in `rlevo-environments` (classic,
  grids, toy_text, box2d, locomotion) plus mock envs in tests currently provide
  `new(render)`; all must change. The trait lives in `rlevo-core`, so this is an
  ADR-level public-API change across the workspace.
- Current pain is low: the stubs are documented and unused on the happy path. No
  consumer is presently blocked.
- Touches code far beyond the viz examples that surfaced it — better done as a focused,
  standalone refactor than bundled into example hardening.

## Alternatives Considered
- **Keep the stubs, do nothing.** The status quo. Acceptable short-term (hence this
  ADR is draft/deferred) but leaves two silent-failure seams that contradict the
  recording tier's error-as-API philosophy (cf. G8).
- **Make the stubs panic instead of no-op.** Turns a silent failure into a loud one
  but still pollutes the trait; a half-measure. Rejected in favour of removing `new`.

## Implementation (PR2, 2026-06-03)

- **`rlevo-core`** — removed `fn new(render: bool)` from `Environment`; added a
  standalone `pub trait ConstructableEnv { fn new(render: bool) -> Self; }`.
  Divergence from Option 1's sketch: it is **not** declared
  `ConstructableEnv: Environment`. The supertrait bound bought nothing and
  would have re-coupled the two concerns; keeping it free lets a future
  config-only or non-`Environment` constructable type opt in independently.
- **`rlevo-environments`** — all ~31 concrete env `impl Environment { fn new }`
  bodies moved verbatim into `impl ConstructableEnv for T` blocks (bandits and
  `Acrobot<D>` keep their generic params; `TimeLimit<E>` gains an
  `impl<E: ConstructableEnv> ConstructableEnv for TimeLimit<E>` that builds the
  inner via `E::new` and wraps it unbounded). Mechanical, brace-free bodies, so
  a scripted relocation was safe.
- **`rlevo-benchmarks`** — the `RecordingTap` `NullSink` stub and the
  `TuiEnvTap` dead-channel stub are **deleted**; neither decorator implements
  `ConstructableEnv` (both are always built from an existing inner env). Test
  stub envs (`StubEnv`/`FailEnv`) dropped their now-unused `new(bool)`.
- **Call sites** — concrete `Type::new(false)` calls resolve through
  `ConstructableEnv`, so call sites that name it (cross-crate tests, env
  rustdoc examples) import `rlevo_core::environment::ConstructableEnv`.
  Standalone integration-test envs got an inherent `new` instead (less churn,
  no trait import needed).
- **Note:** `RenderTap` — this ADR's original "clean `BenchEnv` counter-example"
  — no longer exists (deleted by M-A); the `BenchEnv`-has-no-`new` shape it
  illustrated is exactly what `Environment` now matches.

## References
- Research: viz-examples-gaps (G9) — parent gaps catalog.
- ADR [0007-visualisation-crates-isolated-from-production-crates](0007-visualisation-crates-isolated-from-production-crates.md) — render/family
  knowledge stays off `Environment`; this ADR extends that separation to construction.
- ADR [0008-three-tier-visualisation-ratatui-live-static-report](0008-three-tier-visualisation-ratatui-live-static-report.md) — the three-tier
  viz architecture whose decorators surface the smell.
- Code: `crates/rlevo-core/src/environment.rs` (trait), `record/env_tap.rs` +
  `env_wrappers/tui_env_tap.rs` (degenerate `new`), `env_wrappers/render_tap.rs`
  (the clean `BenchEnv` counter-example).
