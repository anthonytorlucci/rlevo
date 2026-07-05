---
project: rlevo
status: active
type: decision
date: 2026-07-05
tags: [adr, decision, genome, population, tensorgenome, permutation, validation, rlevo-evolution]
---

# ADR 0030: `Permutation` `TensorGenome` opt-in and `Population` non-empty invariant

## Status

**Accepted (2026-07-05).** Filed to resolve issue #158 ("[evo] Core
genome/population type-safety gaps"), which flagged three foundational gaps in
`rlevo-evolution`'s core containers (`genome.rs`, `population.rs`) consumed by
every algorithm in the crate. This ADR records the two decisions in that fix that
touch design of record; the third (a `Send + Sync` bound on the
`TensorGenome::Tensor` GAT) is a straight bug fix under ADR 0022 and needs no new
decision. Extends ADR [0022](0022-tensorgenome-gat-population-storage.md) and
reuses the ADR [0026](0026-shared-config-validation-convention.md) validation
convention.

## Context

ADR 0022 established the `TensorGenome` GAT and `Population<B, K>` storage: a
genome kind opts into on-device rectangular storage by implementing
`TensorGenome`, and only then does `Population<B, K>` type-check. Three gaps in
the shipped code diverged from that design's intent:

1. **`TensorGenome::Tensor` lacked `Send + Sync`.** The GAT was bounded
   `Clone + Debug` only, yet the crate-wide contract is `Strategy: Send + Sync`
   and the sibling bounds in the same file already carry it
   (`GenomeKind: Debug + Copy + Send + Sync + 'static`,
   `type Element: … Send + Sync`). Generic code over `K: TensorGenome` therefore
   could not prove `Population<B, K>: Send + Sync`, even though every concrete
   Burn tensor is `Send + Sync`. This is a straightforward tightening, not a
   design fork.

2. **`Permutation` promised tensor storage it did not have.** Its doc stated
   "Populations are stored as `Tensor<B, 2, Int>`", but `Permutation` implemented
   only `GenomeKind`, so `Population<B, Permutation>` did not type-check. ADR 0022
   anticipated the opt-in ("`Permutation` has a clean opt-in path … the day its
   operators and constructor land") but deferred it. The doc ran ahead of the
   code.

3. **An empty `Population` was never validated at construction.** `new_real` /
   `new_binary` / `new_integer` read `tensor.dims()` and stored them verbatim. A
   zero-row or zero-column tensor built a silently-empty `Population`; the failure
   only surfaced later as an opaque downstream panic (e.g.
   `ops/selection.rs:64`, `assert!(!fitness.is_empty(), …)`) that never named
   `Population` as its source.

`Population<B, K>` has essentially no consumers yet — `Strategy::Genome` is a bare
tensor, not the wrapper — so the constructors' only call sites are the module's
own tests and doc examples. That makes now the cheap moment to harden these
foundational types, before algorithms depend on their current shape.

## Decision

### 1. `Permutation` implements `TensorGenome` now (make the doc true)

Rather than soften the doc to match the deferred code, we advance the opt-in ADR
0022 anticipated:

```rust
impl TensorGenome for Permutation {
    type Tensor<B: Backend> = Tensor<B, 2, Int>;
}
```

A permutation genome *is* rectangular — each row is a length-`n_nodes` integer
vector, all rows equal width — so it genuinely has a device tensor form, unlike
`Tree` (variable-length, host-side ASTs with no rectangular form, which stays a
plain `GenomeKind`). Implementing `TensorGenome` makes `Population<B,
Permutation>` type-check and makes the existing doc accurate. This deliberately
diverges from ADR 0022's "the day its operators land" framing: we land the
*storage seam* (impl + constructor) ahead of the *operators*. The permutation
ACO consumer (`aco_perm`) remains a `todo!()` stub; only its representation is now
first-class.

To keep the type inhabitable through the public API — an implemented
`TensorGenome` with no constructor would leave `Population<B, Permutation>`
type-checkable but unconstructable — we add a matching constructor:

```rust
impl<B: Backend> Population<B, Permutation> {
    pub fn new_permutation(tensor: Tensor<B, 2, Int>) -> Result<Self, ConfigError>;
}
```

**Known limitation (documented, not enforced):** `new_permutation` validates only
shape. It does **not** check that each row is a genuine bijection of
`0..genome_dim`. This mirrors `new_binary`/`new_integer`, which leave gene
*values* unchecked (a binary population is not scanned for stray `2`s, as that
would cost a device round-trip on the hot path). Enforcing the per-row
permutation invariant is deferred to the permutation operators.

### 2. `Population` constructors validate non-empty (ADR 0026 convention)

All four constructors (`new_real`, `new_binary`, `new_integer`,
`new_permutation`) change from `-> Self` to `-> Result<Self, ConfigError>` and
reject an empty tensor at construction, reusing the ADR 0026 helper:

```rust
pub fn new_real(tensor: Tensor<B, 2>) -> Result<Self, ConfigError> {
    let dims = tensor.dims();
    config::nonzero("Population", "pop_size", dims[0])?;
    config::nonzero("Population", "genome_dim", dims[1])?;
    Ok(Self { pop_size: dims[0], genome_dim: dims[1], tensor })
}
```

`config::nonzero` returns `ConstraintKind::Zero`, naming `"Population"` as the
config source and `"pop_size"` / `"genome_dim"` as the offending field. The
failure now names `Population` at the construction boundary instead of surfacing
as an anonymous operator `assert!` several call frames later. This follows ADR
0026 exactly (as `EvolutionaryHarness::new` does), rather than an `assert!` panic:
`Population` construction is a setup boundary, not a hot-path operator, so the
`Result` convention fits. Both zero rows *and* zero columns are rejected — a
zero-width genome is as degenerate as a zero-size population.

The `#[must_use]` attributes drop off (implied by `Result`), and the doc examples
and unit tests thread the `Result` (`?` / `.unwrap()`).

### 3. `Send + Sync` on the GAT (bug fix under ADR 0022)

`type Tensor<B: Backend>: Clone + Debug + Send + Sync;`. Purely additive — every
concrete impl (`Tensor<B, 2>`, `Tensor<B, 2, Int>`) already satisfies it. Recorded
here only for completeness; it introduces no new decision.

## Consequences

### Positive
- `Population<B, Permutation>` is a first-class, constructible type; the
  `Permutation` doc is now true, and the permutation ACO can build populations
  without a further storage change.
- An empty `Population` is rejected where it is created, naming `Population` — the
  confusing downstream panic (#158 §3.1/5.2) is closed.
- `Population<B, K>: Send + Sync` is now provable in generic code, aligning the
  container with the `Strategy: Send + Sync` contract. Two compile-time regression
  guards (`tensor_genome_storage_is_send_sync`, `population_is_send_sync`) would
  catch a future removal of the bound.

### Negative / costs
- The constructors' signature change (`-> Self` → `-> Result<Self, ConfigError>`)
  is technically breaking; blast radius is limited to the module's own tests and
  doc examples, as no external consumer of `Population` exists yet (alpha).
- `Population<B, Permutation>` now ships a public type whose per-row permutation
  invariant nothing enforces at construction — consistent with the unchecked
  element values of `Binary`/`Integer`, but a real, documented gap until the ACO
  operators land.

### Neutral
- Reuses the existing `rlevo_core::config::nonzero` helper and `ConfigError`; no
  new error type. `Tree` is untouched and remains correctly excluded from
  `TensorGenome`.

## Alternatives considered

- **Soften the `Permutation` doc instead of implementing `TensorGenome`**
  (mirror `Tree`'s "no tensor representation" framing, defer the impl to the
  operator release, per ADR 0022's literal wording). Rejected: a permutation is
  genuinely rectangular, so deferring storage documents a limitation rather than
  resolving it; landing the storage seam early costs one impl + one constructor
  and unblocks the future consumer.
- **`assert!`/`debug_assert!` non-empty check, keeping `-> Self`.** Non-breaking,
  but panics rather than returning a structured error — inconsistent with the ADR
  0026 `Result<_, ConfigError>` convention that every other validating constructor
  in the crate follows. Rejected in favour of convention parity.
- **Validate the per-row permutation invariant in `new_permutation`.** Rejected
  for now: it requires a device round-trip (pull to host, check each row is a
  bijection), the same hot-path cost that keeps `Binary`/`Integer` from checking
  values; deferred to the operators that will need the pheromone machinery anyway.

## References
- Issue #158 — "[evo] Core genome/population type-safety gaps" (§6.1 GAT
  `Send + Sync`; §8.1 `Permutation` doc/storage; §3.1/5.2 empty `Population`).
- ADR [0022](0022-tensorgenome-gat-population-storage.md) — the `TensorGenome`
  GAT and `Population` storage this ADR extends; source of the anticipated
  `Permutation` opt-in.
- ADR [0026](0026-shared-config-validation-convention.md) — the `Validate` /
  `ConfigError` convention `new_*` now follows via `config::nonzero`.
- Code: `crates/rlevo-evolution/src/genome.rs` (`TensorGenome`, `Permutation`),
  `crates/rlevo-evolution/src/population.rs` (constructors, `new_permutation`),
  `crates/rlevo-core/src/config.rs:315` (`config::nonzero`).
