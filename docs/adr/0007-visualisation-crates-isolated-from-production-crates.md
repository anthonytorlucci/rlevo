---
project: rlevo
status: superseded
type: decision
date: 2026-05-26
tags:
  - adr
  - decision
  - architecture
  - crates
  - visualisation
  - dependency-graph
  - rlevo
---

# ADR 0007: Visualisation crates are isolated from production crates

## Status

Active. Companion to [0006-leptos-first-visualisation-defer-bevy](0006-leptos-first-visualisation-defer-bevy.md). Extends the dependency-graph discipline established by [0001-keep-environments-and-benchmarks-separate](0001-keep-environments-and-benchmarks-separate.md), [0003-collapse-rl-modules-into-rlevo-reinforcement-learning](0003-collapse-rl-modules-into-rlevo-reinforcement-learning.md), [0004-move-bench-traits-into-rlevo-core](0004-move-bench-traits-into-rlevo-core.md), and [0005-examples-and-cross-crate-tests-in-umbrella](0005-examples-and-cross-crate-tests-in-umbrella.md).

## Context

ADR 0006 introduces two new crates:

- `rlevo-viz-core` — transport-agnostic core: `Visualize` trait, `EnvFrame` snapshot enum, `ViewerEvent` protocol, `MetricsBuffer`, recorder/replay.
- `rlevo-viz-web` — Leptos client + `axum` server: per-family render adapters, plot panels, timeline scrubber, static-HTML export.

The naive integration path would pull `rlevo-viz-core` directly into `rlevo-environments` (so each env can `impl Visualize for MyEnv` next to its definition) and into `rlevo-reinforcement-learning` and `rlevo-evolution` (so each algorithm can emit metrics directly). That path is wrong for the same reasons ADR 0001 keeps `rlevo-benchmarks` out of `rlevo-environments` and ADR 0004 inverts `BenchEnv`/`BenchError` into `rlevo-core`:

1. **Dependency cones bloat.** A user who runs `cargo build -p rlevo-environments` should not be paying for `bincode`, `serde_json`, `axum`, `tower`, `tokio`, `leptos`, `wasm-bindgen`, or any of the ~150 transitive deps the visualisation stack brings in.
2. **Compile-time cost reverberates.** Every change to the viz stack would trigger recompilation of every downstream crate. The visualisation stack is a leaf, not a foundation.
3. **WASM target poisons the workspace.** `rlevo-viz-web` builds for `wasm32-unknown-unknown`; the production crates target native only. Mixing the two leads to `cfg(target_arch = "wasm32")` shrapnel scattered across crates that have no business knowing about it.
4. **`Visualize` is an optional surface, not a core invariant.** Unlike `Environment` or `Snapshot`, not every env needs a render adapter. Forcing `rlevo-environments` to depend on `rlevo-viz-core` to provide adapters for the envs that *do* render would couple the unconditional to the optional.
5. **Pre-1.0 churn isolation.** Leptos, axum, and the chart crates are all on rapid release cycles. Confining their version churn to one crate is materially easier than threading it through five.

The reverse coupling — adapters depending on env crates — is unavoidable: an adapter that renders a `CartPoleState` needs to know about `CartPoleState`. The question is where the adapter code lives. ADR 0005 already established a clean precedent: things that integrate multiple crates live in the umbrella `rlevo` crate or in a leaf crate that owns the integration. We extend that precedent to visualisation.

## Decision

**`rlevo-viz-core` depends only on `rlevo-core`. `rlevo-viz-web` depends on `rlevo-viz-core` and (per-family, feature-gated) on `rlevo-environments`. Neither crate is a dependency — prod or dev — of any production crate.**

The rules:

| Crate | May depend on | May NOT depend on |
|---|---|---|
| `rlevo-viz-core` | `rlevo-core` only | every other workspace crate |
| `rlevo-viz-web` | `rlevo-viz-core`, `rlevo-core`, `rlevo-environments` (feature-gated) | `rlevo-reinforcement-learning`, `rlevo-evolution`, `rlevo-hybrid`, `rlevo-benchmarks` |
| `rlevo-core`, `rlevo-environments`, `rlevo-reinforcement-learning`, `rlevo-evolution`, `rlevo-hybrid`, `rlevo-benchmarks` | (unchanged from prior ADRs) | `rlevo-viz-core`, `rlevo-viz-web` — both as prod and dev deps |
| `rlevo` (umbrella) | every workspace crate including viz | — |

Concretely:

1. **`Visualize` lives in `rlevo-viz-core`, not `rlevo-core`.** It is not a supertrait of `Environment`. Environments opt in via a separate `impl Visualize for MyEnv` block — and those impls live in `rlevo-viz-web/src/adapters/<family>.rs`, not in `rlevo-environments`. The env crate compiles without `rlevo-viz-core` in scope.
2. **Per-family adapters are feature-gated in `rlevo-viz-web`.** Each adapter (`adapter-classic`, `adapter-grids`, `adapter-toy-text`, `adapter-box2d`, `adapter-locomotion`, `adapter-landscapes`) pulls in only the env types it needs from `rlevo-environments`. Default features cover everything except `adapter-locomotion` (which is a 2D-projection stopgap per ADR 0006).
3. **`MetricsBuffer` is the integration seam for training loops.** Algorithm crates (`rlevo-reinforcement-learning`, `rlevo-evolution`, `rlevo-hybrid`) emit metrics through their existing observer / callback surfaces. They do **not** import `rlevo-viz-core`. A thin adapter — provided by the example or the user — wires `MetricSample` events from the training loop's callbacks into `MetricsBuffer::push`. Algorithm crates remain visualisation-agnostic.
4. **Examples live in the umbrella.** Per ADR 0005, examples that demonstrate visualisation live in `crates/rlevo/examples/viz/`. The umbrella crate is the only place that simultaneously depends on `rlevo-viz-web` and the algorithm crates.
5. **Training must run with visualisation off.** All instrumentation hooks default to no-op. A user who does not enable the `viz` feature on the umbrella pays zero runtime cost and zero compile-time cost beyond the empty trait definitions.

### Why `rlevo-viz-core` depends on `rlevo-core` (not nothing)

`rlevo-viz-core` needs `EnvironmentError` and a small set of shared scalar types (`RunId` newtype, `SeedStream` for deterministic replay metadata). Reaching into `rlevo-core` for those is consistent with ADR 0004 (shared abstractions live in `rlevo-core`). It does **not** drag in `Environment`, `State`, `Observation`, `Action`, or any tensor-bearing trait — those are not load-bearing for the viz layer.

### Why adapters live in `rlevo-viz-web`, not `rlevo-environments`

The cleaner-feeling shape would be: each env crate provides its own adapter behind a `viz` feature flag. We rejected that because:

- It forces every env crate to take an optional dep on `rlevo-viz-core` and to keep `cfg(feature = "viz")` blocks scattered through env modules.
- It splits the adapter code across six crates, making it harder to keep visual conventions (colour palette, viewport math, overlay rendering) coherent.
- The Leptos client and adapters share enough infrastructure (SVG primitives, layout helpers, colour palette) that co-locating them is materially simpler than wiring shared helpers through every env crate.

Adapters live with the renderer; env crates stay clean. This mirrors the precedent in ADR 0001 (benchmarks adapt to envs via a `bench` feature *inside* the benchmark consumer, not the env producer).

## Consequences

**Positive**

- **Production-crate build cones unchanged.** `cargo build -p rlevo-core` / `-p rlevo-environments` / `-p rlevo-reinforcement-learning` / `-p rlevo-evolution` / `-p rlevo-hybrid` / `-p rlevo-benchmarks` does not compile a single line of `axum`, `leptos`, `bincode`, `wasm-bindgen`, or any chart crate.
- **Strict-DAG dependency graph preserved.** No prod or dev cycles. Visualisation is a strict leaf hanging off the umbrella.
- **WASM concerns contained.** `wasm32-unknown-unknown` builds only ever touch `rlevo-viz-web` and `rlevo-viz-core`. The rest of the workspace remains native-only.
- **Viz version churn is local.** Bumping Leptos / axum / chart crates touches at most two `Cargo.toml` files and one set of source modules. Production crates never see breakage from a Leptos minor release.
- **Algorithm crates stay visualisation-agnostic.** RL and evo crates emit metrics through their existing observer interfaces. Anyone replacing the visualisation layer (a future native viewer, a TensorBoard adapter, a `rerun.io` integration) implements a different `MetricSample` consumer without changing algorithm code.

**Negative / accepted costs**

- **Adapter code is physically separated from env code.** A maintainer adding a new environment must also add an adapter in `rlevo-viz-web`. The two files do not live in the same crate. Acceptable — the alternative is `cfg(feature = "viz")` blocks in every env module and a `rlevo-viz-core` dep on every env crate.
- **Adapter feature flags multiply.** Each env family needs its own feature flag in `rlevo-viz-web`. Manageable — the set is small, fixed, and named after env families that already exist.
- **No `Visualize` blanket impls keyed off `Environment`.** Each env opts in explicitly via its adapter module. We considered a blanket `impl<E: Environment + ToVisualize> Visualize for E { ... }`, but `ToVisualize` would belong in `rlevo-core`, which we've just decided not to do. Explicit beats clever here.

**Neutral**

- The umbrella `rlevo` crate's `[features]` block grows a `viz` feature that activates `rlevo-viz-web` and the adapter set. Localised to one manifest.
- Examples in `crates/rlevo/examples/viz/` are gated by `required-features = ["viz"]`. Default `cargo build -p rlevo` builds without visualisation.

## Alternatives considered

**Make `Visualize` a supertrait of `Environment` in `rlevo-core`.** Rejected. Forces every env to provide an adapter or carry a noisy `impl Visualize for MyEnv { fn frame(&self) -> EnvFrame { unimplemented!() } }`. Adds `EnvFrame` (with its bincode/serde dep cone) to `rlevo-core`, polluting the most foundational crate with a rendering concern. Violates the precedent set by ADR 0004 of keeping `rlevo-core` to load-bearing-only abstractions.

**Co-locate adapters with env crates (feature-gated in `rlevo-environments`).** Rejected. See "Why adapters live in `rlevo-viz-web`" above. The cleaner *feel* loses to the cleaner *graph*.

**Allow algorithm crates to depend on `rlevo-viz-core` for direct metric emission.** Rejected. Algorithm crates emit through their existing observer / callback surfaces and remain visualisation-agnostic. A thin user-side adapter bridges callbacks to `MetricsBuffer`. Replaceable; testable; not a coupling.

**One unified `rlevo-viz` crate that fuses core + web.** Rejected. The transport-agnostic split is the whole point of ADR 0006's "if a future native viewer is built, it reuses the core unchanged." Pre-fusing the core into the Leptos client would lock that door.

## References

- rlevo-viz-overview — env-vis umbrella spec, 2026-05-26.
- viz-core — `rlevo-viz-core` API surface and crate layout.
- env-visualization-web — `rlevo-viz-web` per-family adapters and feature-flag layout.
- [0001-keep-environments-and-benchmarks-separate](0001-keep-environments-and-benchmarks-separate.md) — established the "adapt at the consumer, not the producer" pattern reused here.
- [0003-collapse-rl-modules-into-rlevo-reinforcement-learning](0003-collapse-rl-modules-into-rlevo-reinforcement-learning.md) — established the "RL-only modules belong in the RL crate, not core" pattern.
- [0004-move-bench-traits-into-rlevo-core](0004-move-bench-traits-into-rlevo-core.md) — established the "shared abstractions belong in `rlevo-core`, but only the load-bearing ones" pattern.
- [0005-examples-and-cross-crate-tests-in-umbrella](0005-examples-and-cross-crate-tests-in-umbrella.md) — established that cross-crate integration lives in the umbrella, reused for the viz examples.
- [0006-leptos-first-visualisation-defer-bevy](0006-leptos-first-visualisation-defer-bevy.md) — companion ADR; together they shape the visualisation stack.
- `crates/rlevo-viz-core/Cargo.toml`, `crates/rlevo-viz-web/Cargo.toml` — new manifests introduced by the umbrella spec.
- `crates/rlevo/Cargo.toml` — gains a `viz` feature that pulls in `rlevo-viz-web` and adapter features.
