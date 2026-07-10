---
project: rlevo
status: active
type: decision
date: 2026-07-10
tags: [adr, decision, environments, box2d, bipedal-walker, terrain, error-handling, config, validation]
---

# ADR 0040: Environment config error and terrain output contract

## Status

**Accepted (2026-07-10).** Resolves issue #120 (BipedalWalker terrain generates
invalid geometry). Supersedes nothing. Reuses the ADR 0026 `Validate` /
`ConfigError` convention and the ADR 0039 "make invalid states unrepresentable"
posture for the box2d family.

## Context

`HardcoreTerrain::generate` built the walker's ground polyline while advancing
an obstacle cursor in a mixed loop/world coordinate convention (a `-10.0`
offset applied at every push). Three defects followed from that shape:

1. **Non-monotone x (High).** The loop ran `while x < 200.0` (loop space) but a
   pit advanced `x += width` with `width ∈ [2, 5)`, so `x` could overshoot the
   boundary. The function then *unconditionally* pushed a terminal `[190.0, 0.0]`
   point, which could land LEFT of the previous point. `build_ground`'s
   `windows(2)` then built a backwards, overlapping cuboid collider — the
   invalid geometry the issue reports.
2. **Spawn-time collision (High).** The walker spawns at world x = 0, but the
   flat lead-in only covered x ∈ [-10, 0); obstacles and roughness began *at*
   the spawn point. The reference Gymnasium env holds `TERRAIN_STARTPAD = 20`
   flat steps and spawns at the **middle** of the pad, ahead of the hull.
3. **Panic on invalid config (High).** `RoughTerrain::generate` called
   `rng.random_range(-roughness..=roughness)`, which panics on a negative or
   `NaN` `roughness` (empty range). The `roughness` / `step` fields were `pub`
   with no validation gate, so a struct-literal or deserialized value reached
   the panicking draw directly — a `rules.md` §4 violation ("never panic on
   user-supplied runtime data").

Separately, the propagation seam was silent: `build_ground` early-returned on
`pts.len() < 2`, and `rebuild_world` returned `()`, so a malformed generator
degraded quietly instead of surfacing a structured error.

## Decision

### 1. A generic `EnvironmentError::Config(#[from] ConfigError)` variant

Add one variant to `rlevo_core::environment::EnvironmentError`:

```rust
/// A configuration-domain invariant failed during a lifecycle operation.
#[error("Configuration error: {0}")]
Config(#[from] crate::config::ConfigError),
```

It is deliberately **generic**, not terrain-specific:

- `Environment::reset` re-runs construction-time work (rebuilding a procedural
  world), so a config-domain invariant — enforced once at `with_config` — can
  legitimately re-surface at reset. The lifecycle method needs a typed channel
  for that failure.
- `#[from] ConfigError` gives `?` ergonomics: `reset` is `self.rebuild_world()?`
  with no manual re-wrap.
- Keeping it generic (rather than an `InvalidTerrain(String)`) reduces
  stringly-typed error surface and lets any environment whose lifecycle
  re-validates config-domain state reuse the same variant. `ConfigError` is
  already the workspace's structured, allocation-free config-violation type
  (ADR 0026).

### 2. The `TerrainGenerator` output contract, enforced at one chokepoint

`TerrainGenerator::generate` stays `-> Vec<[f32; 2]>` — **infallible and
object-safe** (the env holds a `Box<dyn TerrainGenerator>`). Its output contract
is documented on the trait under `# Invariants`:

- `pts.len() >= 2` — at least one segment.
- `pts[i][0] <= pts[i + 1][0]` — x is **non-decreasing** left-to-right.
- `pts[0][0] < 0.0` — the first point is left of the spawn point (world x = 0).
- Points are in world space (not pre-scaled by `SCALE`).

`BipedalWalker::rebuild_world` is the **single chokepoint** that re-validates
the produced polyline: it rejects `pts.len() < 2` or any decreasing x step with
`ConfigError { config: "BipedalWalker", field: "terrain", kind:
ConstraintKind::Custom("terrain generator produced fewer than 2 non-decreasing
points") }`, and only then builds colliders. `rebuild_world` returns
`Result<(), ConfigError>`; `build` propagates with `?`; `reset` propagates via
the new `EnvironmentError::Config`. The silent `if pts.len() < 2 { return; }`
early-return in `build_ground` is removed — the chokepoint now guarantees the
precondition, so `build_ground` is unconditional.

`HardcoreTerrain::generate` is rewritten to iterate in world space (dropping the
dual `-10.0` offset that hid defect 1) with named constants
(`TERRAIN_END_X = 190`, `WALL_INSET`, `PIT_DEPTH`, `PIT_WIDTH_{MIN,MAX}`,
`STUMP_HEIGHT`, `STUMP_WIDTH`, `SPAWN_PAD = 20`), and appends the terminal cap
**only if the last obstacle did not already reach `TERRAIN_END_X`** — the direct
fix for defect 1. Both `RoughTerrain` and `HardcoreTerrain` hold a flat
`SPAWN_PAD`-wide pad that spans world x = 0 with the walker at its middle — the
fix for defect 2.

### 3. Model B: make invalid config unrepresentable (validate at construction)

Rather than a fallible trait or a standalone `TerrainError` enum, `RoughTerrain`
and `HardcoreTerrain` adopt the ADR 0026 shape:

- Fields become **private**; a `pub fn new(...) -> Result<Self, ConfigError>`
  validates and an `impl Validate` backs it. Getters expose the fields.
- `roughness` is validated **non-negative AND finite** (`roughness == 0.0` is
  legal — it yields flat terrain — so *not* strictly positive); `step` strictly
  positive + finite; `pit_frequency` / `stump_frequency` non-negative + finite,
  and `pit + stump <= 10.0` so the branch partition stays a valid probability
  split. `generate` additionally skips the height draw when `roughness == 0`,
  so no empty-range panic is reachable even on the legal zero case.
- `Default` stays infallible and each type unit-tests that
  `X::default().validate().is_ok()` (ADR 0026 obligation).

Two local helpers (`nonneg_finite`, `positive_finite`) fill the gap the shared
`config` helpers leave: `config::positive` and `config::in_range` with an
infinite bound both accept `+∞`, which terrain geometry cannot use.

## Consequences

### Positive
- The reported invalid geometry is impossible: the terminal-cap guard keeps x
  monotone for every seed (regression-tested over 512 seeds), and the chokepoint
  rejects any future generator that violates the contract.
- No panic path remains for a caller-supplied or deserialized terrain config;
  the failure is a field-named `ConfigError` at construction, or an
  `EnvironmentError::Config` at reset.
- The walker always spawns on flat ground with obstacles ahead of the hull,
  matching the Gymnasium reference.
- One generic error variant serves every environment whose lifecycle
  re-validates config-domain state; no new stringly-typed error type.

### Neutral
- `rebuild_world` and `build` signatures were already `Result`-returning; the
  change is the added `?` and the terrain-output check. `reset`'s signature is
  unchanged (still `Result<_, EnvironmentError>`).
- Terrain struct fields are now private; the only in-crate constructors used the
  factory `::default()` path, so no external call site changed.

### Negative / accepted costs
- `RoughTerrain` / `HardcoreTerrain` struct-literal construction is no longer
  possible outside the module; callers use `new(...)` (fallible) or `default()`.
  Acceptable — it is the whole point of Model B (alpha, no external consumers).

## Alternatives considered

- **Fallible `TerrainGenerator::generate -> Result<Vec<_>, _>`.** Rejected:
  breaks object safety cleanliness and pushes a `Result` through every call
  site for a contract that is cheaply re-checkable at one chokepoint. The
  infallible trait + chokepoint validation keeps `Box<dyn TerrainGenerator>`
  ergonomic.
- **A standalone `TerrainError` enum.** Rejected: duplicates the structured,
  allocation-free `ConfigError` (ADR 0026) for the same class of "a field is out
  of its valid domain" failures.
- **A terrain-specific `EnvironmentError::InvalidTerrain(String)`.** Rejected:
  stringly-typed and single-purpose; the generic `Config(ConfigError)` variant
  is reusable and structured.
- **Keep `pub` fields + validate only in the env.** Rejected: leaves the
  panicking `random_range` reachable via a struct literal and does not name the
  offending field.

## References
- Issue #120 — BipedalWalker terrain invalid geometry generation.
- ADR [0026](0026-shared-config-validation-convention.md) — `Validate` /
  `ConfigError` convention this reuses.
- ADR [0039](0039-box2d-states-own-markov-dofs.md) — the box2d
  make-invalid-states-unrepresentable posture (Model B) this follows.
- Gymnasium `BipedalWalker` — monotone `x = i * TERRAIN_STEP` grid and
  `TERRAIN_STARTPAD = 20` with spawn at the pad middle, the reference geometry.
- Code: `crates/rlevo-environments/src/box2d/bipedal_walker/terrain.rs`
  (generators, `Validate`), `.../bipedal_walker/env.rs` (`rebuild_world`
  chokepoint, `build_ground`), `crates/rlevo-core/src/environment.rs`
  (`EnvironmentError::Config`).
