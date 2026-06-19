---
project: rlevo
status: active
type: decision
date: 2026-06-19
tags: [evolution, genome, population, gat, trait-design, type-safety]
---

# ADR 0022: `TensorGenome` GAT ties `Population` storage to genome kind

## Status

Active. Adopted 2026-06-19. **Refactor** of an existing, internally-unconsumed
public type (`rlevo_evolution::population::Population`); no other production crate
depends on it today (strategies operate on bare `Tensor`s via
`Strategy::Genome`). Lands alongside the removal of the unused
`GenomeKind::GENOME_LEN` associated const (same session) — both are tightenings of
the `GenomeKind` family discovered while documenting the genome chapter of the
user-book.

## Context

`Population<B, K>` is the typed, shape-checked container strategies reach for at
API boundaries (`ask` produces one, `tell` consumes one); operators themselves
take bare tensors. The genome *kind* `K` is one of the zero-sized markers
(`Real`, `Binary`, `Integer`, plus the roadmap markers `Tree`, `Permutation`)
implementing `GenomeKind`.

The original wrapper stored its payload as **two `Option` tensor fields** — one
float, one integer — and upheld an invariant by *constructor convention*:

```rust
pub struct Population<B: Backend, K> {
    pop_size: usize,
    genome_dim: usize,
    _kind: PhantomData<K>,
    tensor_real: Option<Tensor<B, 2>>,       // Some iff K = Real
    tensor_int: Option<Tensor<B, 2, Int>>,   // Some iff K ∈ {Binary, Integer}
}
```

"Exactly one is `Some`, determined by `K`" was a documented contract, not a
type-level fact. The per-kind `tensor()` accessors `.expect()`ed on the matching
field; a kind/field mismatch was therefore a latent panic guarded only by the
discipline of code in this module. The type parameter `K` was carried purely as
`PhantomData` and contributed nothing to the storage type.

Two further frictions came with that shape:

- The marker trait had no way to say *which* tensor flavour a kind uses, so the
  knowledge lived as a comment and a hand-maintained `match`/`Option` pattern.
- `Population<B, Tree>` was a nameable type even though `Tree` has no rectangular
  tensor form (its genomes are variable-length, host-side ASTs). The
  impossibility was documented, not enforced — there was simply no constructor.

The question this ADR answers: can the kind→storage mapping be moved into the
type system so the wrong-tensor-for-kind state becomes unrepresentable and
non-tensor kinds are excluded by the compiler?

## Decision

Introduce a companion trait carrying a **generic associated type** for the
backing tensor, and make `Population` store a single field of that type.

```rust
// crates/rlevo-evolution/src/genome.rs
pub trait TensorGenome: GenomeKind {
    /// Device tensor storing a whole population of this kind, shape
    /// (pop_size, genome_dim).
    type Tensor<B: Backend>: Clone + Debug;
}

impl TensorGenome for Real    { type Tensor<B: Backend> = Tensor<B, 2>; }
impl TensorGenome for Binary  { type Tensor<B: Backend> = Tensor<B, 2, Int>; }
impl TensorGenome for Integer { type Tensor<B: Backend> = Tensor<B, 2, Int>; }

// crates/rlevo-evolution/src/population.rs
pub struct Population<B: Backend, K: TensorGenome> {
    pop_size: usize,
    genome_dim: usize,
    tensor: K::Tensor<B>,
}
```

`tensor()` returns `&K::Tensor<B>` and `into_tensor()` returns `K::Tensor<B>`,
both **total** — no `Option`, no `.expect()`, no panic path.

### 1. A separate trait, not a `type Tensor<B>` on `GenomeKind`

The GAT lives on a new `TensorGenome: GenomeKind` rather than on `GenomeKind`
itself. Putting it on `GenomeKind` would force *every* kind — including `Tree`,
which has no rectangular tensor representation — to name a tensor type, defeating
the purpose. A separate trait makes tensor-backing **opt-in**: a kind must
declare a `Tensor<B>` to be storable in a `Population`. `Tree` stays a plain
`GenomeKind` and never implements `TensorGenome`, so `Population<B, Tree>` does
not type-check. This mirrors the standalone-trait-not-supertrait pattern of
ADR 0011 and ADR 0019: a separable capability gets its own trait.

### 2. A GAT, because the tensor type depends on the backend `B`

The storage type is a function of two parameters — the kind *and* the backend
(`Real` → `Tensor<B, 2>`). A plain associated type (`type Tensor;`) cannot
express the `B`-dependence; a generic associated type (`type Tensor<B: Backend>`)
is the minimal construct that does. Edition 2024 (Rust ≥ 1.85) is well clear of
GAT stabilisation (1.65), so this carries no MSRV cost.

### 3. `Population<B, K: TensorGenome>` carries the bound on the struct

The bound sits on the struct definition, not only on the impls, so the
non-tensor exclusion (decision 1) is enforced at every mention of the type, and
the single field `tensor: K::Tensor<B>` is well-formed. `K` is now *used* by the
field (via the projection `K::Tensor<B>`), so the `PhantomData<K>` marker is
dropped.

### 4. `Clone + Debug` supertrait bound on the GAT

`type Tensor<B: Backend>: Clone + Debug` lets `Population` keep
`#[derive(Clone, Debug)]`: the derive needs `K::Tensor<B>: Clone + Debug`, which
the supertrait bound supplies for any `K: TensorGenome`. Both `Tensor<B, 2>` and
`Tensor<B, 2, Int>` satisfy it. Removing this bound would break the derives —
flagged as the critical line of the change.

### 5. Per-kind constructors retained; no generic `new`

`new_real`, `new_binary`, `new_integer` stay. Each reads `pop_size`/`genome_dim`
from `tensor.dims()`, which requires the *concrete* tensor type to be in hand. A
single generic `Population::new(tensor: K::Tensor<B>)` would need an extra trait
method (e.g. `fn dims<B>(t: &Self::Tensor<B>) -> [usize; 2]`) to reach `dims()`
through the opaque associated type — more surface for no gain. The named
constructors are also clearer at call sites. The accessors, by contrast, *are*
unified into one generic impl, since returning the field needs no concrete type.

### 6. Drop the dead rank assertion and its `# Panics` clause

The constructors asserted `dims.len() == 2`. Burn's `Tensor<B, D>::dims()`
returns `[usize; D]`, so for the rank-2 tensors these constructors accept,
`dims.len()` is the compile-time constant `2` and the assertion can never fire.
The assertion and the accompanying `# Panics: Panics if tensor is not rank 2`
documentation are removed as unreachable — rank is already guaranteed by the
tensor type. The constructors are now infallible, consistent with the panic
discipline in `docs/rules.md` (reserve panics for reachable programming errors).

## Consequences

**Positive**

- The wrong-tensor-for-kind state is **unrepresentable**: one field, type chosen
  by `K`. The "exactly one `Some`" invariant and the `.expect()` panic path are
  gone — `tensor()`/`into_tensor()` cannot fail.
- Non-rectangular kinds are excluded by the **type checker**: `Population<B, Tree>`
  does not compile, encoding the real domain fact (Tree is host-side,
  variable-length) rather than documenting it.
- The kind→storage mapping is now a single, discoverable place (`impl TensorGenome
  for …`) instead of a comment plus a hand-maintained `Option` pattern.
- `Permutation` has a clean opt-in path: it implements `TensorGenome` (with
  `Tensor<B> = Tensor<B, 2, Int>`) and gains a `Population<B, Permutation>` the
  day its operators and constructor land.

**Negative / accepted costs**

- A GAT now appears in the public API. It is idiomatic on edition 2024 and the
  supertrait bound keeps derives working, but it raises the conceptual floor for
  a contributor reading `Population`.
- `Population` gained a trait bound (`K: TensorGenome`); any future generic code
  over `Population<B, K>` must carry it. Acceptable — that bound is exactly the
  precondition that the type is tensor-backed.
- The kind→tensor table is duplicated between `TensorGenome` impls and the
  user-book chapter; kept in sync by the chapter's verification step.

**Neutral**

- Blast radius today is zero: `Population<B, K>` has no consumers outside its own
  module. Concrete accessor return types at any future call site are unchanged
  (`Population::<B, Real>::tensor()` still yields `&Tensor<B, 2>`).

## Alternatives considered

**Keep two `Option` fields + convention (status quo).** Rejected: the invariant
is unprovable to the compiler, the `.expect()` is a latent panic, and `K` does no
work beyond `PhantomData`.

**An `enum PopData<B> { Real(Tensor<B,2>), Int(Tensor<B,2,Int>) }` single field.**
Rejected: it removes the *pair* of `Option`s but not the core problem — the enum
variant is still not tied to `K`, so the accessor must still `match` and handle
the "wrong variant" arm with `unreachable!`/`expect`. It trades two conventions
for one; it does not make the bad state unrepresentable.

**`type Tensor<B>` directly on `GenomeKind`.** Rejected (decision 1): forces
`Tree` to invent a tensor type it has no use for and leaves `Population<B, Tree>`
nominally valid. The capability is separable, so it gets its own trait.

**A plain associated type `type Tensor;` (no `B`).** Rejected (decision 2): the
storage type genuinely depends on the backend; a non-generic associated type
cannot name `Tensor<B, 2>`.

**One generic `Population::new`.** Rejected (decision 5): requires an extra
dims-reading method on the trait to compensate for the opaque associated type;
per-kind constructors are simpler and read better.

## References

- `crates/rlevo-evolution/src/genome.rs` — `GenomeKind`, `TensorGenome`, markers.
- `crates/rlevo-evolution/src/population.rs` — `Population<B, K>` + in-source tests.
- `docs/user-book/src/part-1-foundations/evolutionary-computation/22-genome.md` —
  user-facing treatment of the GAT design and the `Tree` exclusion.
- [0011-lift-construction-off-environment-trait](0011-lift-construction-off-environment-trait.md),
  [0019-observable-projection-trait](0019-observable-projection-trait.md) — the
  separable-capability-as-standalone-trait pattern this follows.
- Commit `81df270` (this refactor); commit `810bf46` (the related `GENOME_LEN`
  removal that prompted the `GenomeKind` review).
