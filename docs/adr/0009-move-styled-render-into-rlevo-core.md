---
project: rlevo
status: active
type: decision
date: 2026-05-27
tags: [adr, decision, architecture, crates, rlevo-core, rlevo-environments, rlevo-benchmarks, visualisation]
---

# ADR 0009: Hoist styled-output render surface into `rlevo-core`

## Status

Active. Companion to [0008-three-tier-visualisation-ratatui-live-static-report](0008-three-tier-visualisation-ratatui-live-static-report.md); extends the pattern established by [0004-move-bench-traits-into-rlevo-core](0004-move-bench-traits-into-rlevo-core.md) to the render trait surface.

## Context

Milestone 1 of the visualisation roadmap landed `AsciiRenderable` with a styled-output sibling method (`render_styled() -> StyledFrame`) inside `rlevo-environments::render`. Every public 2D env in the crate implements it. The type set — `StyledFrame`, `StyledLine`, `StyledSpan`, `SpanStyle`, `Color`, `Modifier` — plus the project-wide semantic `palette` module, all lived next to the trait under `rlevo-environments::render::{styled, palette}`.

Milestone 2 needed to extend this to a live `ratatui` TUI inside `rlevo-benchmarks`. The new `RenderTap<E>` wrapper (in `rlevo-benchmarks::env_wrappers`) is naturally bounded `E: BenchEnv + AsciiRenderable`, and the `StyledFrame → ratatui::text::Text<'static>` conversion in `rlevo-benchmarks::tui::convert` references `StyledFrame` directly.

That bound forced the question: how does `rlevo-benchmarks` see `AsciiRenderable` and `StyledFrame`?

### The cycle

`rlevo-environments` already declares an **optional** prod dep on `rlevo-benchmarks` (the `bench` feature, established by [0001-keep-environments-and-benchmarks-separate](0001-keep-environments-and-benchmarks-separate.md) and refined by [0004-move-bench-traits-into-rlevo-core](0004-move-bench-traits-into-rlevo-core.md); powers `bench/suites.rs`). Adding `rlevo-benchmarks → rlevo-environments` — even gated behind the optional `tui` feature on the benchmarks side — closes the package-level loop.

We tested this empirically rather than reasoning from the docs. With

```toml
# crates/rlevo-benchmarks/Cargo.toml
rlevo-environments = { path = "../rlevo-environments", optional = true }

[features]
tui = ["dep:ratatui", "dep:crossterm", "dep:rlevo-environments"]
```

`cargo build -p rlevo-benchmarks --features tui` failed with:

```
error: cyclic package dependency: package `rlevo-environments v0.1.0` depends on itself.
Cycle:
package `rlevo-environments`
    ... satisfies path dependency `rlevo-environments` of package `rlevo-benchmarks`
    ... satisfies path dependency `rlevo-benchmarks`  of package `rlevo-environments`
    ... satisfies path dependency `rlevo-environments` of package `rlevo`
```

Cargo's cycle check runs over the `[dependencies]` graph after default-feature resolution but *before* feature gating eliminates optional edges. Optional + optional is still a cycle as far as the checker is concerned. The only Cargo-tolerated cycle is one through `[dev-dependencies]`, because dev-deps are not transitively reachable from downstream builds.

### What's actually env-specific about `AsciiRenderable`?

Inspecting the trait:

```rust
pub trait AsciiRenderable {
    fn render_ascii(&self) -> String;
    fn render_styled(&self) -> StyledFrame {
        StyledFrame::unstyled(self.render_ascii())
    }
}
```

The default `render_styled` body uses only `StyledFrame` (already a candidate to move). The trait itself names no env type, no const generic, no `Environment` or `BenchEnv` bound. It's a structural "thing that produces text"; nothing about it depends on `rlevo-environments`.

The same is true of `AsciiRenderer: Renderer<E>` which sits next to it — `Renderer` is in `rlevo-core::render`, the `AsciiRenderer` impl uses `AsciiRenderable` and `String`. Both could already live next to `Renderer` and `NullRenderer` in `rlevo-core::render`.

### What the ADR 0004 test says

[0004-move-bench-traits-into-rlevo-core](0004-move-bench-traits-into-rlevo-core.md) established the test: **a trait moves to `rlevo-core` when it has ≥1 stable downstream consumer with shared vocabulary**. The styled-render surface qualifies cleanly:

| Crate                                | Uses                                                          |
| ------------------------------------ | ------------------------------------------------------------- |
| `rlevo-environments`                 | Trait *and* per-env impls (the 26 styled renderers from M1)   |
| `rlevo-benchmarks` (`tui` feature)   | Trait bound on `RenderTap`; `StyledFrame` in convert + state  |
| `rlevo-benchmarks` (future `record`) | `StyledFrame` field on `FrameRecord` (M4 spec)                |
| Report tier (future, post-M4)        | Deserialises `StyledFrame` from `EpisodeRecord`               |

Two production consumers today, two more locked in by the umbrella spec. Multiple consumers, shared vocabulary, no env-specific deps. Passes the ADR 0004 test.

## Decision

**Move the styled-output type set, the semantic palette, and `AsciiRenderable` / `AsciiRenderer` from `rlevo-environments::render` to `rlevo-core::render`. Preserve every M1 import path via re-export.**

Concretely:

1. **Promote** `crates/rlevo-core/src/render.rs` to `crates/rlevo-core/src/render/mod.rs` so the render module can host submodules alongside the existing `Renderer` trait and `NullRenderer`.
2. **Move** (verbatim):
   - `crates/rlevo-environments/src/render/styled.rs`  → `crates/rlevo-core/src/render/styled.rs`
   - `crates/rlevo-environments/src/render/palette.rs` → `crates/rlevo-core/src/render/palette.rs`
   - `crates/rlevo-environments/src/render/ascii.rs`   → `crates/rlevo-core/src/render/ascii.rs`
3. **Adjust docstring cross-refs** in `styled.rs` to drop the old `super::AsciiRenderable` rustdoc link (the path is still resolved by `super::` after the move, but the comment around it framed the relationship in env terms).
4. **Reduce** `crates/rlevo-environments/src/render/mod.rs` to a pure re-export shim:

   ```rust
   pub use rlevo_core::render::{
       ascii, palette, styled,
       AsciiRenderable, AsciiRenderer,
       Color, Modifier, SpanStyle, StyledFrame, StyledLine, StyledSpan,
   };
   ```

   Every M1 per-env impl that wrote `use rlevo_environments::render::*` keeps compiling without change. The 26 styled-render impls + the 23-test cross-env coverage suite stayed green throughout the refactor.
5. **Forward** `AsciiRenderable` through `BenchAdapter` in `rlevo-environments::bench::adapter` so `RenderTap<BenchAdapter<E, …>>` composes when `E: AsciiRenderable`. This is glue that becomes possible (and useful) because of the move; bundled into the same refactor commit.

### Scope decision: do not delete `rlevo-environments::render`

The module is preserved as a re-export shim even though every type now lives elsewhere. The shim costs nothing at runtime, keeps the M1 commit history navigable (per-env impls still resolve their imports from the path they were authored against), and gives future readers a hint that `rlevo-environments` is the *consumer* of the render trait, even if not the *home* of it. Matches the v0.1 `rlevo-benchmarks::{env, agent, seed}` shim modules from ADR 0004.

## Consequences

**Positive:**

- **Cycle eliminated.** `rlevo-benchmarks` sees the trait + types through its existing `rlevo-core` dep. No env-side dep needed. The umbrella spec's `rlevo-viz-overview` §3.2 constraint ("`rlevo-benchmarks` is the only production crate gaining optional viz deps") stays honoured.
- **Single canonical home for styled-output vocabulary.** The type set is consumed by three crates at minimum (`rlevo-environments`, `rlevo-benchmarks`, future report-tier). Hosting it in core matches every other "shared vocabulary" decision in the workspace (`Environment`, `Reward`, `BenchEnv`, `Metric`).
- **Source compatibility.** Zero per-env import changes; M1's 26 styled renderers + 23 integration tests pass with no edits. The re-export shim in `rlevo-environments::render` carries the same items it always did.
- **Composes with `BenchAdapter`.** Forwarding `AsciiRenderable` through the adapter (a one-impl change in `bench/adapter.rs`) makes `RenderTap<BenchAdapter<E, …>>` work for every env, with no per-env wrapper code.

**Negative / accepted costs:**

- **Conceptual oddity.** `rlevo-core` now hosts a trait named `AsciiRenderable` and a module called `palette` even though the crate ships no environments and no terminal-side code. Mitigated by the trait's actually-generic shape (it's "anything that can produce text + a styled projection") and by the precedent set by `BenchEnv` in core post-ADR 0004 (a "bench environment" trait hosted in a crate that ships no bench harness). The naming is `AsciiRenderable`, not `EnvironmentRender`, which already reads as structural rather than env-shaped.
- **The `palette` constants are project-wide today.** They sit in core where they could in theory be consumed by non-env code that has no business reaching for an "agent" or "hazard" colour. Accepted: the constants are paired with accessibility-grade modifier companions, and the doc-comment is explicit about their semantic intent. If a future caller misuses them, the misuse is visible at the import site.
- **Reverses the spec's intent on conversion-impl location.** The umbrella spec rlevo-viz-overview §7 said the `From<StyledFrame> for ratatui::text::Text<'_>` impl lives in `rlevo-benchmarks::tui`. With `StyledFrame` now in `rlevo-core` (still foreign to benchmarks), the orphan rule forbids the `From` impl there too. The bridge ships as free functions instead (`frame_to_ratatui`, `frame_to_ratatui_ref`, `color_to_ratatui`, …). Documented in the M2 session log as a deliberate deviation; call-site cost is `frame_to_ratatui(f)` vs `f.into()`. The spec is amended in v2.3 to match.

**Neutral:**

- The `bench` feature on `rlevo-environments` still opt-in pulls `rlevo-benchmarks`. ADR 0004 chose to keep that edge to avoid relocating `bench/suites.rs`; this ADR does not reopen that decision.
- The conceptual boundary between "render vocabulary" (now in core) and "concrete per-env render impls" (in `rlevo-environments`) maps cleanly to ADR 0001's environments-vs-benchmarks split and to ADR 0004's traits-vs-runner split. The workspace dep graph stays a strict DAG, with the M2 additions sitting on `rlevo-core` directly.

## Alternatives considered

**Make the dep cycle work via dev-dependencies on the benchmarks side.** `rlevo-benchmarks` could declare `rlevo-environments` as a *dev*-dep (cycle through dev-deps is tolerated). Rejected:

- The dep is needed for *library* code (`RenderTap` is part of the public surface, not a test helper). A dev-dep would force the `RenderTap` machinery into test-only modules, blocking external users from composing it.
- Even if achievable, the dep cone on test builds would grow by the full `rlevo-environments` cone (rapier2d, rapier3d, burn). Compile-time cost meaningful for `cargo test`-heavy CI.

**Put the conversion in the umbrella `crates/rlevo/` crate instead of `rlevo-benchmarks`.** The umbrella already deps both env and benchmarks. Rejected: the spec mandates the conversion lives in `rlevo-benchmarks::tui` so downstream users of `rlevo-benchmarks` (without the umbrella) can use the live TUI. Hosting the conversion in the umbrella also pushes a chunk of M2 surface into a place reserved for examples and integration tests (per [0005-examples-and-cross-crate-tests-in-umbrella](0005-examples-and-cross-crate-tests-in-umbrella.md)).

**Define a parallel "styled-only" trait in core; keep `AsciiRenderable` in environments with the bound `AsciiRenderable: StyledOnly`.** Rejected: the trait surface is already minimal (two methods). Splitting it in two adds API surface without adding capability. `AsciiRenderable` has no env-specific business, so the split would just rename one of them — no engineering benefit.

**Move `bench/suites.rs` out of `rlevo-environments` so the optional env→benchmarks edge can be deleted entirely.** ADR 0004's "Scope decision" already considered and deferred this; reopening would push suite factories either into the harness (inverting natural data flow — the harness should not know about specific envs) or into the umbrella (one more thing to re-export). Deferred again on the same reasoning.

**Empirically confirm that an optional-only cycle works in some configurations.** We checked. It does not. Cargo's check is feature-blind at the package-graph level, and the project does need `cargo build --all-features` and `cargo build --workspace` to succeed for CI gating. Anything that risks those failing in unusual feature combinations is a non-starter for production code.

## References

- [0001-keep-environments-and-benchmarks-separate](0001-keep-environments-and-benchmarks-separate.md) — established the envs ↔ benchmarks boundary; the source of the cycle this ADR finishes closing.
- [0004-move-bench-traits-into-rlevo-core](0004-move-bench-traits-into-rlevo-core.md) — established the "move shared trait surface to core" pattern this ADR extends.
- [0005-examples-and-cross-crate-tests-in-umbrella](0005-examples-and-cross-crate-tests-in-umbrella.md) — broke the dev-dep half of the envs ↔ benchmarks loop; this ADR breaks the prod half for the render surface.
- [0008-three-tier-visualisation-ratatui-live-static-report](0008-three-tier-visualisation-ratatui-live-static-report.md) — three-tier viz architecture; explains why the conversion needs to live in `rlevo-benchmarks` rather than in any production crate.
- Spec: ascii-renderable-coverage — M1 scope, where the types were originally placed.
- Spec: rlevo-viz-overview §3.2 — production-crate isolation constraint that drove the move.
- Session log: [2026-05-27-milestone-1-complete](2026-05-27-milestone-1-complete.md) — original home for the types.
- Session log: [2026-05-27-milestone-2-complete](2026-05-27-milestone-2-complete.md) — describes the cycle discovery and the move.
- Implementation: commit `bdd1c37` — `refactor(render): hoist styled-output types to rlevo-core`.
- Source locations after move:
  - `crates/rlevo-core/src/render/mod.rs` — `Renderer`, `NullRenderer`, re-exports.
  - `crates/rlevo-core/src/render/styled.rs` — `StyledFrame`, `StyledLine`, `StyledSpan`, `SpanStyle`, `Color`, `Modifier`.
  - `crates/rlevo-core/src/render/palette.rs` — semantic colour constants + accessibility-paired modifiers.
  - `crates/rlevo-core/src/render/ascii.rs` — `AsciiRenderable`, `AsciiRenderer`.
  - `crates/rlevo-environments/src/render/mod.rs` — re-export shim only.
