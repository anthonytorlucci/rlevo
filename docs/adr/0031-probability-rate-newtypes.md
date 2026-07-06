---
project: rlevo
status: active
type: decision
date: 2026-07-06
tags: [adr, decision, probability, rate, nan, validation, rlevo-core, evolution, conventions]
---

# ADR 0031: `Probability` and `NonNegativeRate` newtypes for validated rates

## Status

**Accepted (2026-07-06).** Resolves issue #144 ([evo] Probability/rate newtype +
NaN validation, `bug`/`enhancement`/`priority: high`). Follows ADR
[0027](0027-bounds-newtype-for-closed-ranges.md) (the `Bounds` newtype) as its
direct shape template.

**Chosen shape:** two small `Copy` validated newtypes in `rlevo-core` —
`probability::Probability` (invariant `0.0 <= p <= 1.0`) and
`rate::NonNegativeRate` (invariant `is_finite() && r >= 0.0`) — each valid by
construction, threaded into the `rlevo-evolution` operator signatures so a `NaN`
or out-of-range rate cannot reach a mutation/crossover operator.

## Context

The mutation/crossover **rate** scalars (`rate`, `p`, `alpha`, `mutation_rate`,
`crossover_p`, and `WritebackPolicy::Partial`'s probability) were bare `f32`s. A
`NaN`/`Inf` value silently misbehaved in three distinct ways:

- **Comparison operators** (`u.lower_elem(p)`, `mask_rng.random() < p`): `x <
  NaN` is `false` for all `x`, so the Bernoulli mask is all-false and the
  operator degenerates — a no-op reset/flip, or a full clone of one parent
  (`ops/crossover.rs`, `ops/mutation.rs`, `algorithms/memetic.rs`).
- **CGP host comparison** (`rng >= mutation_rate`, `gp_cgp.rs`): the sense is
  inverted, so a `NaN` skips the `continue` and mutates **every** gene — the
  opposite outcome from the tensor operators, an inconsistency in itself.
- **BLX-α arithmetic** (`ops/crossover.rs`): a `NaN`/`Inf` `alpha` multiplies
  into the interval and poisons the entire offspring tensor.

Validation lived only at the config layer via `config::in_range` (which *does*
reject `NaN`, since `NaN >= lo` is `false`) on `BinaryGaConfig`, `CgpConfig`,
and `GaConfig`. It was enforced hard only at harness construction; inside
strategy `init` it was a release-stripped `debug_assert!`. The raw `ops/`
functions accepted unchecked `f32` with zero validation, and
`WritebackPolicy::Partial(f32)` had **no** `Validate` coverage at all — only a
debug-only assert in the `tell` hot path. The invariant did not travel with the
value: exactly the gap ADR 0027 closed for ranges.

### Relationship to existing seams

Mirrors ADR 0027 (`Bounds`), ADR 0026 (`Validate`), and ADR 0023
(`ObjectiveSense`): a small typed primitive in a dedicated `rlevo-core` module
that many crates reference and none duplicate. These newtypes are the
type-level companion to `Validate`'s boundary-level check — they compose. A
newtype field returns a dedicated `Copy` error, not a `ConfigError`, because
construction has no config/field name to report (ADR 0027 §5).

## Decision

### 1. Two newtypes in `rlevo-core`

`rlevo_core::probability::Probability` — invariant `0.0 <= p <= 1.0`; a `NaN` or
`Inf` fails the comparison and is rejected. `rlevo_core::rate::NonNegativeRate` —
invariant `is_finite() && r >= 0.0`; rejects `NaN`, `±Inf`, and negatives. Both
carry the full `Bounds` surface: `const fn new` (panicking, for literals /
`Default`s), `try_new -> Result<_, {Probability,NonNegativeRate}Error>`, `const
fn get`, `#[serde(try_from = "f32", into = "f32")]` + `TryFrom<f32>` / `From`,
and a dedicated `Copy` error struct.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "f32", into = "f32")]
pub struct Probability(f32);      // invariant: 0.0 <= p <= 1.0

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "f32", into = "f32")]
pub struct NonNegativeRate(f32);  // invariant: is_finite() && r >= 0.0
```

Two newtypes rather than one because the constraints genuinely differ: a
probability is a unit interval; BLX `alpha` (an expansion factor, conventionally
0.5 but may exceed 1) and Gaussian `mutation_sigma` (a step size) are
non-negative but **unbounded above**. One shared `NonNegativeRate` covers both
of the latter.

### 2. Threaded into the operator boundary

The raw `ops/` functions take the newtypes directly:
`blx_alpha(alpha: NonNegativeRate)`, `uniform_crossover(p: Probability)`,
`binary_uniform_crossover(p: Probability)`, `gaussian_mutation(sigma:
NonNegativeRate)`, `uniform_reset(p: Probability)` (`lo`/`hi` stay `f32` — a
reset *range*, not a rate; a full `Bounds` migration there is out of scope),
`bit_flip_mutation(p: Probability)`. This is the point of the change: the
operator is valid by construction, not reliant on a caller having validated
first. `gaussian_mutation_per_row` takes a **tensor** of per-row sigmas and is
out of scope (a newtype cannot wrap tensor elements cheaply).

### 3. Config fields become the newtypes; redundant checks drop out

`BinaryGaConfig.{mutation_rate, crossover_p}: Probability`,
`CgpConfig.mutation_rate: Probability`, `GaConfig.mutation_sigma:
NonNegativeRate`, `GaCrossover::BlxAlpha { alpha: NonNegativeRate }` /
`GaCrossover::Uniform { p: Probability }`, and
`WritebackPolicy::Partial(Probability)`. Per ADR 0027 §3, each self-validating
field **removes** its paired `config::in_range(…)` line from `validate()`; the
memetic hot-path `debug_assert!` is deleted (the type now guarantees it). The
rest of every `validate()` is unchanged. `Default`s use the const `new`.

### 4. serde

Both newtypes route `Deserialize` through `try_new` via `#[serde(try_from)]`, so
a rate loaded from a manifest cannot deserialize out of range — free, since
`rlevo-core` already depends on `serde` and `thiserror`. (The four target
configs are not themselves `Deserialize` today, but the primitive is
serde-correct for parity with `Bounds` and future record use.)

## Consequences

### Positive
- The scalar `NaN`/`Inf` rate hazard is **unrepresentable** at every operator,
  not merely checked at the config boundary. The inconsistent CGP-vs-tensor NaN
  behaviour is eliminated at the source.
- `WritebackPolicy::Partial` gains real (type-level) validation where it had
  none; its release-stripped hot-path assert is deleted.
- Redundant `config::in_range` lines collapse into the field types.

### Negative / costs
- A fourth and fifth small typed primitive coexist with `Bounds`, `Validate`,
  `ObjectiveSense`; mitigated by the shared shape (all mirror `Bounds`) and the
  one-file-per-type module split.
- Operator signatures and their call sites change; in-file operator tests and
  every downstream config constructor (tests / examples) migrate their `f32`
  literals to `Probability::new(…)` / `NonNegativeRate::new(…)`.

### Neutral
- Purely additive to `rlevo-core` (two new modules). Reuses the present
  `serde`/`thiserror` deps.

## Deferred (out of scope; separate issue)

The 🔴 crossover.rs §1.3 concern — non-finite **parent gene values** propagating
through `blx_alpha` (and the other operators) — is *not* addressed here. The
newtype fixes the scalar params; a per-element finiteness scan of the parent
tensors on every crossover is a hot GPU-path cost that needs its own design
(candidate: treat parent-gene finiteness as the fitness/eval layer's
responsibility, as CGP already does via `sanitize_fitness`). Filed as #207.

Also out of scope by name: `arch_nas::arch_mutation_rate` and the `gep` config's
`mutation_rate` (not among the five files in #144); their config `in_range`
checks remain until a follow-up migrates them.

## Alternatives considered

- **`Probability` only; leave `alpha`/`sigma` on `config::in_range`.** Rejected:
  leaves the raw `blx_alpha`/`gaussian_mutation` operators NaN-unsafe by
  construction — the very gap this ADR closes.
- **A distinct `BlxAlpha` and `Sigma`/`MutationScale`.** Rejected: identical
  invariant (`finite && >= 0`); one `NonNegativeRate` avoids alias proliferation.
- **Newtype only at the config layer, ops keep `f32`.** Rejected: the operators
  stay callable with `NaN`; the invariant would not travel to the boundary that
  actually consumes it.
- **Return `ConfigError` from `try_new`.** Rejected for the ADR 0027 §5 reason:
  no field name at construction; a dedicated `Copy` error is simpler.

## References
- Issue #144 — this ADR resolves it.
- ADR [0027](0027-bounds-newtype-for-closed-ranges.md) — the `Bounds` newtype;
  direct shape template (`new`/`try_new`, dedicated `Copy` error, validated
  serde, self-validating-field-removes-check rule).
- ADR [0026](0026-shared-config-validation-convention.md) — the `Validate`
  convention these newtypes complement; source of the `config::in_range` lines
  this ADR removes for the migrated rate fields.
- ADR [0023](0023-objective-sense-and-maximize-convention.md) — small typed
  primitive in a dedicated `rlevo-core` module; the pattern this ADR follows.
- `docs/rules.md §4` — Error Handling; the deserialized-data-is-`Result` rule
  that motivates the validated serde (§4).
- Code: `crates/rlevo-core/src/probability.rs`,
  `crates/rlevo-core/src/rate.rs`, `crates/rlevo-evolution/src/ops/crossover.rs`
  (the 🔴 §1.3 BLX-α site), `crates/rlevo-evolution/src/algorithms/gp_cgp.rs`
  (the inverted-sense CGP host comparison).
