---
project: rlevo
status: active
type: decision
date: 2026-06-16
tags: [environments, observation, rank, modality, pomdp, pixel, issue-65]
---

# ADR 0020: Synthetic pixel-over-grid environment — first real `Observable<OR>` consumer

## Status

Active. Adopted 2026-06-16. Implements issue #65 per the sub-spec
synthetic-pixel-over-grid-env (under image-observation-over-compact-state-spec).
**Additive**: introduces one new concept module `crates/rlevo-environments/src/pixel_grid.rs`
and changes no existing environment, trait, or manifest. Extends [0019-observable-projection-trait](0019-observable-projection-trait.md)
by giving `Observable<OR>` its first production consumer; the `rlevo-core` contract is byte-unchanged.

## Context

ADR 0019 added `Observable<OR>` — the typed home for modality-changing POMDPs where the
observation tensor order differs from the state order — and proved the `Environment` contract
already permits `R != SR` via the `MockRam` `Environment<2,1,1>` integration test. But the trait
was a **defined-but-unconsumed seam**: no environment in `rlevo-environments` actually projected a
rank-changing observation. Issue #65 asked for the first real consumer.

The open design question was synthetic-vs-Atari. The ALE path was investigated concretely
(canonical-modality-changing-pomdp-benchmarks): ALE's native `ALEInterface` is C++
(`getRAM()` → 128-byte rank-1, `getScreenRGB()` → `[210,160,3]` rank-3), so a real Atari env is
exactly `Environment<3,1,1>` — the *same* `R != SR` wiring. Every Rust binding crate is stale
(`ale` v0.1.3, May 2020), so the maintainable route is fresh C-ABI FFI + a `cmake` `build.rs`,
dragging a C++ toolchain and ROM-redistribution concerns into the lean `rlevo-environments` crate
(rules §1, ADR 0001). Because Atari maps onto the identical shape, the synthetic env loses nothing
by going first.

## Decision

Ship a synthetic **pixel-over-grid** navigation environment as `Environment<3, 1, 1>` in a new
top-level concept module `pixel_grid.rs`. Atari/ALE is deferred to its own milestone.

### 1. Synthetic first, Atari deferred

The synthetic env is dependency-free, fully reproducible (exact known latent), and exercises the
exact `R != SR` projection path Atari would. The Atari backend is a separate milestone behind a
`feature` gate (fresh C-ABI FFI over current ALE).

### 2. Observation rank 3, RGB (`C = 3`) — not grayscale

Image shape is `[20, 20, 3]`. The sub-spec originally resolved on grayscale `C = 1`; the **user
chose RGB `C = 3`** so a future Atari backend differs only in *resolution*, never in rank or
channel count — `getScreenRGB` is already `[H, W, 3]`. RGB cell colors are distinct **hues**
(background black `[0,0,0]`, goal green `[0,128,0]`, agent white `[255,255,255]`), recoverable per
channel rather than by intensity alone. Keeping the trailing channel axis makes the observation
rank 3 regardless of channel count, so the Atari backend needs no rank change.

### 3. Separate concept module, not folded into `grids/`

The `grids/` family shares an *egocentric* `7×7×3` `core/` with `R == SR` and 7-action
turn/forward dynamics. This task is *allocentric*, 4-way Cartesian, and modality-changing — it
reuses none of that core. It lives in its own singular-noun concept module `pixel_grid.rs`
(rules §2). If a second modality-changing env (Atari) lands, promote to a family folder then.

### 4. Fixed dimensions for v1

`GRID_SIDE = 5`, `CELL_PX = 4`, `CHANNELS = 3` → `[20, 20, 3]`, as compile-time constants
(`Observation::shape()` must be const, rules §7; matches the `grids` 7×7×3 precedent). Const-generic
`G`/`S`/`C` parameterization is deferred.

### 5. Dual `State<1>` + `Observable<3>` on one state type

`PixelGridState { agent, goal }` implements `State<1>` (trivial same-order `observe()` →
`LatentObservation`, shape `[2]`) **and** `Observable<3>` (`project()` → `PixelObservation`, shape
`[20,20,3]`). Every snapshot is built from `state.project()`, never `state.observe()` — the
production realization of the ADR-0019 dual-trait pattern.

### 6. `TensorConvertible<3, B>` in scope for the pixel observation

`PixelObservation` round-trips through a Burn `Tensor<B,3>` (bytes normalized to `[0,1]`,
reconstructed by `*255` + round). This makes the "a Burn policy can consume the projected image"
claim real. The latent observation does not get a `TensorConvertible` impl (add on demand).

### 7. Dynamics and reward

4-way Cartesian moves with wall clamping ("bump and hold"); `terminated` on `agent == goal` with
the grids `success_reward` formula (`1 - 0.9·step/max_steps`), `truncated` at `max_steps` with
reward `0.0`. Placement is fixed corners by default (deterministic) or seeded-random distinct cells
via host-side `StdRng` (host-RNG convention — never `B::seed`/global RNG).

## Consequences

**Positive**

- `Observable<OR>` is retired from "defined-but-unconsumed" status: a real production environment
  drives `R(3) != SR(1)` end-to-end through the public `Environment` API.
- A future Atari backend is a drop-in at the same observation rank and channel count — only
  resolution and the RAM/frame source differ.
- The exact-known-latent property is a hook for belief-manifold scoring (information-geometric
  neuroevolution), noted for traceability but out of scope here.

**Negative / accepted costs**

- `pixel_grid.rs` re-implements the `success_reward` formula locally rather than importing
  `grids::core::reward` — a deliberate decoupling so the module shares nothing with `grids`.
- Fixed v1 dimensions; configurable `G`/`S`/`C` deferred.

## Alternatives considered

- **Atari/ALE first.** Rejected for now: stale Rust bindings, a C++/CMake/ROM toolchain against the
  lean-crate posture, and no expressive gain over the synthetic env (identical `Environment<3,1,1>`
  shape). Tracked as a gated follow-up milestone.
- **Grayscale `C = 1`.** Superseded by the user's RGB choice (decision 2) so Atari differs only in
  resolution.
- **Fold into `grids/`.** Rejected (decision 3): no shared `core/`, allocentric vs egocentric.

## References

- synthetic-pixel-over-grid-env — governing sub-spec.
- image-observation-over-compact-state-spec — parent spec.
- [0019-observable-projection-trait](0019-observable-projection-trait.md) — the trait this env consumes.
- canonical-modality-changing-pomdp-benchmarks — the ALE path analysis and synthetic-first call.
- [0001-keep-environments-and-benchmarks-separate](0001-keep-environments-and-benchmarks-separate.md) — the lean-dependency posture behind deferring
  the C++ ALE toolchain.
- `crates/rlevo-environments/src/pixel_grid.rs` — the env + in-source unit tests.
- `crates/rlevo-environments/tests/pixel_grid_modality.rs` — the single-crate `R != SR` proof.
- `crates/rlevo/examples/envs/pixel_grid.rs` — scripted rollout demonstration.
- Issue #65 (this env), #64 (the trait), #62 (design tracker).
