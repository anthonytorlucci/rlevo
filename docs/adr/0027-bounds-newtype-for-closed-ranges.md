---
project: rlevo
status: active
type: decision
date: 2026-07-05
tags: [adr, decision, bounds, ranges, validation, rlevo-core, conventions]
---

# ADR 0027: `Bounds` newtype for validated closed ranges

## Status

**Accepted (2026-07-05).** Filed to unify issues #99 (environments —
`clamp()`/`clip()` panics on inverted or NaN action/config bounds) and #140
(evolution — a `Bounds`/`SearchBounds` newtype for `(lo, hi)` tuples), which
independently proposed the same fix across two crates. Issue #196 is the
cross-crate parent.

**Chosen shape:** a small `pub struct Bounds` in a new `rlevo_core::bounds`
module — an inclusive range that is **valid by construction**: the whole
invariant is `lo <= hi`, which rejects `lo > hi` and `NaN` (a `NaN` fails the
comparison) while permitting a degenerate single point (`lo == hi`) and a
one-sided infinite range (`[0.7, ∞)`). It *complements* the ADR 0026 `Validate`
convention rather than replacing it: a config field of type `Bounds` is
self-validating, so the config's `validate()` no longer repeats a
`config::ordered(…, "bounds", …)` check for that field. This ADR lands the type
and migrates the existing `(f32, f32)` range/clip fields in `rlevo-evolution`
and `rlevo-environments` to it.

## Context

`(f32, f32)` range tuples appear as bare, unvalidated pairs across the
workspace, and an inverted or `NaN` pair is a live hazard:

- `f32::clamp(min, max)` **panics** when `min > max` or either bound is `NaN`.
  Locomotion action clips flow straight into it — `swimmer/env.rs:228,387`
  (`action.0[i].clamp(lo, hi)`), and `locomotion/common.rs:213`
  (`clip_contact_cost` → `contact_cost.clamp(range.0, range.1)`).
- The evolution search bounds instead use `x.max(lo).min(hi)`
  (`local_search.rs:289`, `clamp_vec`), which does **not** panic but silently
  collapses every coordinate to `hi` when `lo > hi` — a wrong result rather than
  a loud one. The helper's own doc concedes this: *"a degenerate range where
  `lo > hi` collapses every coordinate to `hi` … Callers are expected to supply
  valid bounds."*

ADR 0026 introduced the `Validate` convention, and its first-wave adopters
already call `config::ordered(C, "bounds", …)` in ~22 `Validate` impls (every
population/metaheuristic config, four locomotion `action_clip`s). That fixes the
value **at the config boundary** — but the boundary is not the whole story:

- The validated pair is immediately destructured back to a raw `(f32, f32)` and
  handed to unguarded helpers (`clamp_vec`, `clip_contact_cost`, `affine`) that
  re-carry the "callers must supply valid bounds" caveat. The invariant does not
  travel with the value.
- The same invariant is spelled out ~22 times, once per config, as an
  easy-to-forget line. Three fields were in fact **missed**: the four
  local-search params (`SimulatedAnnealingParams`, `HillClimbingParams`,
  `NelderMeadParams`, `RandomRestartParams`) carry `bounds` but have **no**
  `Validate` impl at all; `HealthyCheck::state_range` is never
  `ordered`-checked; and `clip_contact_cost` takes a bare tuple with no guard.

The evolutionary and environment reviews reached the same conclusion
independently and flagged it ADR-worthy (firefly §3.1, gwo §3.1, woa §3.1,
hill_climbing §3.1, random_restart §3.1). Deciding the newtype's shape once, in
`rlevo-core`, avoids two divergent implementations of one invariant.

### Relationship to existing seams

- This mirrors ADR 0026 (`Validate`) and ADR 0023 (`ObjectiveSense`): a small
  typed primitive in a dedicated `rlevo-core` module that many crates reference
  and none duplicate. `Bounds` is the *type-level* companion to `Validate`'s
  *boundary-level* check — the two compose, they do not compete.
- `Bounds` reuses the ADR 0026 error vocabulary in spirit but not in type: its
  constructor has no config/field name to report, so it returns a dedicated
  `BoundsError`, not a `ConfigError` (see §5).

## Decision

### 1. The `Bounds` type (new `rlevo_core::bounds` module)

```rust
//! Validated inclusive range: the [`Bounds`] newtype and [`BoundsError`].

/// An inclusive range `[lo, hi]` over `f32`, valid by construction.
///
/// A `Bounds` cannot hold an inverted (`lo > hi`) or `NaN` endpoint: the whole
/// invariant is `lo <= hi`, which a `NaN` fails. So the panic-on-`min > max`
/// contract of [`f32::clamp`] and the silent `lo > hi` collapse of
/// `x.max(lo).min(hi)` are both unrepresentable at the call sites that hold a
/// `Bounds`. A degenerate single-point range (`lo == hi`) and a one-sided
/// infinite range (`[0.7, ∞)`, e.g. a "healthy above this height" check) are
/// both **allowed** — `f32::clamp` is well-defined for either.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "(f32, f32)", into = "(f32, f32)")]
pub struct Bounds {
    lo: f32,
    hi: f32,
}

impl Bounds {
    /// Builds a range from compile-time-known endpoints, panicking on an
    /// invalid pair. For literals and `Default`s only (mirrors the ADR 0026
    /// setter-guard exception): the bad value is right there at the call site.
    pub const fn new(lo: f32, hi: f32) -> Self { /* assert!(lo <= hi) */ }

    /// Builds a range from runtime / user-supplied endpoints, returning
    /// [`BoundsError`] on an inverted or `NaN` pair.
    pub fn try_new(lo: f32, hi: f32) -> Result<Self, BoundsError>;

    pub const fn lo(&self) -> f32;
    pub const fn hi(&self) -> f32;
    pub const fn span(&self) -> f32;            // hi - lo
    pub fn clamp(&self, x: f32) -> f32;         // panic-free, min <= max always
    pub fn clamp_slice(&self, xs: &mut [f32]);  // absorbs local_search::clamp_vec
}

impl TryFrom<(f32, f32)> for Bounds { type Error = BoundsError; /* … */ }
impl From<Bounds> for (f32, f32) { /* … */ }   // interop / destructure

/// The single way a [`Bounds`] construction can fail. Allocation-free (`Copy`),
/// carries the offending endpoints.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[error("invalid bounds: lo {lo} must not exceed hi {hi} (and neither may be NaN)")]
pub struct BoundsError {
    pub lo: f32,
    pub hi: f32,
}
```

A one-sided **infinite** endpoint is deliberately permitted: `HealthyCheck`
expresses "healthy above 0.7, no ceiling" as `z_range: Some((0.7, ∞))`, and
`f32::clamp` panics only on `min > max` or `NaN`, never on an infinity. The
`lo <= hi` invariant admits it (`0.7 <= ∞`) while still rejecting `NaN`.

### 2. Inclusive invariant, and the divergence from `config::ordered`

`Bounds` accepts `lo == hi`; `config::ordered` (ADR 0026) requires strict
`lo < hi`. The divergence is deliberate:

- The `Bounds` invariant exists to make **`clamp` safe** — and `clamp` is
  perfectly well-defined on a single point (`x.clamp(c, c) == c`). Rejecting
  `lo == hi` would forbid a harmless, meaningful "pin to a constant" range.
- Every search-space consumer samples with `lo + (hi - lo) * r`
  (`ga.rs:213`, `simulated_annealing.rs:347`, `random_restart.rs:478,503`), so a
  zero-width range yields the constant `lo` — no division, no `NaN`.

`config::ordered` keeps strict `<` for the scalar invariants that genuinely need
it (`v_min < v_max`, where a degenerate support divides by zero); `Bounds` is
for closed clamp/sample ranges, where inclusive is correct. The two are distinct
tools, documented as such.

### 3. Relationship to the ADR 0026 `Validate` convention

A `Bounds` field is self-validating, so adopting it **removes** the paired
`config::ordered(C, "bounds", …)` line from each config's `validate()`. The rest
of every `validate()` is unchanged — `Bounds` narrows one field's invariant into
the type system; it does not discharge the config's other cross-field checks.
Configs keep implementing `Validate`. Where a config still needs the field's
scalars for a *cross-field* check (e.g. mountain-car `goal_position ∈
[pos.lo(), pos.hi()]`), it reads them back through the accessors.

### 4. serde: validated deserialization

`Bounds` derives `Serialize`/`Deserialize` with `#[serde(try_from = "(f32,
f32)")]`, so a range loaded from a file or manifest runs through `try_new` and a
malformed pair is **rejected**, never deserialized into an invalid `Bounds`.
This is required because two adopters — `MountainCarConfig` and
`MountainCarContinuousConfig` — derive `Deserialize`, and rules.md §4 / ADR 0026
forbid trusting deserialized data. It is free otherwise: `rlevo-core` already
depends on `serde` (used by `render::payload`).

### 5. Error surface: a dedicated `BoundsError`, not `ConfigError`

`Bounds::try_new(lo, hi)` has no `config`/`field` name to report, which is the
whole point of `ConfigError`. Returning a `ConfigError` would force a placeholder
field name and couple the primitive to the config module. Instead `try_new`
returns a small `Copy` `BoundsError` carrying the offending endpoints. A config
that builds a `Bounds` from its own scalar fields (only mountain-car does, via
its `Validate`) maps or re-wraps as needed; every other adopter holds a `Bounds`
directly and never sees `BoundsError` after construction.

One documented consequence of the inclusive invariant: `SimulatedAnnealingParams`
and `RandomRestartParams` default `step_size = 0.1 * (hi - lo)`
(`simulated_annealing.rs:79`, `random_restart.rs:84`). A now-constructible
zero-width range makes that default `0.0`, so the search silently cannot move.
This is degenerate-but-safe (no panic, no `NaN`); those two defaults gain a
`debug_assert!(bounds.span() > 0.0, …)` so a zero-width bound is caught in debug
builds rather than read as a mysterious dead run.

### 6. Scope of migration

In scope (this ADR): every `(f32, f32)` / `Option<(f32, f32)>` **range** field
in `rlevo-evolution` (all `bounds`) and `rlevo-environments` (`action_clip`,
`HealthyCheck::{z_range, angle_range, state_range}`), the `clip_contact_cost`
helper, and — folding two scalar pairs into `Bounds` — `MountainCarConfig` /
`MountainCarContinuousConfig` (`min_pos`/`max_pos` → `pos_bounds`,
`min_action`/`max_action` → `action_bounds`). The last is a **breaking** change
to those two configs' public field shape and serialized form; acceptable in
alpha.

Out of scope (documented non-goals): box2d action clips (hardcoded scalar
literals, no config field); `render::payload` `bounds_x`/`bounds_y` and the
`rlevo-benchmarks` record schema `bounds_*` / `Point2` viewport bounds
(serialized visualization wire format, not search ranges); `landscapes/render`
`(f64, f64)` heatmap bounds.

## Consequences

### Positive
- The `lo > hi` / `NaN` clamp hazard is **unrepresentable** wherever a `Bounds`
  is held, not merely checked once at the config boundary. `clamp_vec`,
  `clip_contact_cost`, and `affine` shed their "callers must supply valid bounds"
  caveats.
- ~22 repeated `config::ordered(…, "bounds", …)` lines collapse to a field type.
- Three previously-unguarded sites are fixed for free: the four local-search
  params, `HealthyCheck::state_range`, and `clip_contact_cost`.
- Deserialized ranges are validated (mountain-car), closing the ADR 0026
  loaded-config hole for this field shape.

### Negative / costs
- Cross-crate migration touches ~30 fields plus test/example call sites; the
  `(f32, f32)` literals become `Bounds::new(…)`.
- Mountain-car's public field shape and serialized form change (breaking, alpha).
- A third small typed primitive now coexists with `Validate` and
  `ObjectiveSense`; mitigated by the module split and the §3 rule that `Bounds`
  *replaces* only the one `ordered` line, not `Validate` itself.
- Inclusive `lo == hi` makes a zero-width range constructible; the SA /
  random-restart `step_size` defaults guard it with `debug_assert!` (§5).

### Neutral
- Purely additive to `rlevo-core` (new `bounds` module; existing public types
  unchanged). Reuses the already-present `serde`/`thiserror` dependencies.

## Alternatives considered

- **Keep per-field `config::ordered` only (status quo after ADR 0026).** Leaves
  the invariant at the config boundary; every downstream helper still takes a raw
  tuple and re-documents the caveat, and the three missed sites stay unguarded.
  Rejected — this is exactly the "invariant does not travel with the value" gap.
- **Strict `lo < hi` (parity with `config::ordered`).** Rejected: forbids a
  well-defined single-point clamp range for no safety gain, since every consumer
  is zero-width-safe.
- **Return `ConfigError` from `Bounds::try_new`.** Rejected: the constructor has
  no field name to report, and coupling the primitive to the config module for a
  placeholder buys nothing. A dedicated `Copy` `BoundsError` is simpler (§5).
- **Generic `Bounds<T: Float>` over `f32`/`f64`.** Premature: every in-scope
  range field is `f32`; the only `f64` ranges are the excluded heatmap-render
  helpers. Revisit additively if a numeric consumer appears.
- **Leave mountain-car's scalar `min/max` pairs as-is.** Considered (they are
  already `ordered`-checked and are a serde-shape change). Rejected here in favour
  of one consistent range type across both crates; the breaking change is
  acceptable in alpha.

## References
- Issue #196 — cross-crate parent; unifies #99 (environments) and #140
  (evolution).
- ADR [0026](0026-shared-config-validation-convention.md) — the `Validate`
  convention `Bounds` complements; source of the `config::ordered` lines this
  ADR removes for range fields.
- ADR [0023](0023-objective-sense-and-maximize-convention.md) — small typed
  primitive in a dedicated `rlevo-core` module; the shape this ADR follows.
- `docs/rules.md §4` — Error Handling; the deserialized-data-is-`Result` rule
  that motivates the validated serde (§4).
- Code: `crates/rlevo-evolution/src/local_search.rs:289` (`clamp_vec` — the
  silent-collapse helper), `crates/rlevo-environments/src/locomotion/common.rs:212`
  (`clip_contact_cost` — the unguarded `f32::clamp` site),
  `crates/rlevo-environments/src/locomotion/swimmer/env.rs:228`
  (`action_clip` → `clamp`), `crates/rlevo-evolution/src/algorithms/ga.rs:107`
  (a representative `config::ordered("bounds", …)` line to remove).
