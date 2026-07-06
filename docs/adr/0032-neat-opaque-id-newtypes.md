---
project: rlevo
status: active
type: decision
date: 2026-07-06
tags: [adr, decision, neuroevolution, neat, newtype, type-safety, evolution, conventions]
---

# ADR 0032: Opaque `NodeId` / `InnovationId` / `SpeciesId` newtypes for NEAT

## Status

**Accepted (2026-07-06).** Resolves issue #142 ([evo] Newtype opaque IDs —
neuroevolution, `enhancement`/`priority: high`). Follows ADR
[0027](0027-bounds-newtype-for-closed-ranges.md) and ADR
[0031](0031-probability-rate-newtypes.md) as its shape precedent, **minus the
validation machinery** (an id has no invariant and is not serialized).

**Chosen shape:** three small `Copy` opaque newtypes over `u64`, defined in place
in the neuroevolution module — `topology::NodeId`, `topology::InnovationId`,
`species::SpeciesId` — each an infallible wrapper (`new`/`get`, plus a
crate-internal `succ`), so id-confusion becomes a compile error.

## Context

The three NEAT id concepts were bare type aliases:

```rust
pub type NodeId = u64;        // topology.rs
pub type InnovationId = u64;  // topology.rs
pub type SpeciesId = u64;     // species.rs
```

A type alias is not a distinct type: `NodeId`, `InnovationId`, `SpeciesId`, and a
raw `u64` are all the *same* type and freely interchangeable. Nothing prevented
passing a `NodeId` where an `InnovationId` was expected —
`InnovationRegistry::register_node_split(split: InnovationId)` would silently
accept a node id, and `ConnectionGene` mixes `source: NodeId` with `innovation:
InnovationId` in one struct. All three independent file reviews (innovation.rs
§3.1, species.rs §3.4, topology.rs §3.2) flagged the same id-confusion surface;
§3.1 rated it 🔴 as a real type-safety gap in the mutation/crossover alignment
logic, where an id mix-up misclassifies genes with **no error**.

### Relationship to existing seams

Mirrors the opaque half of ADR 0027 (`Bounds`) and ADR 0031
(`Probability`/`NonNegativeRate`): a small typed primitive whose invariant (here,
"which id space this integer belongs to") travels with the value. It diverges
from those two on the *validation* axis, because ids differ from rates in two
ways that remove the need for it:

- **No invariant.** Every `u64` is a legal id, so construction cannot fail —
  `new` is infallible (no `try_new`, no error type, no `# Panics`).
- **No persistence.** No NEAT type (`TopologyGenome`, `NodeGene`,
  `ConnectionGene`, `NodeSplit`, `Species`, phenotypes) derives serde, so there
  is no wire representation to validate and no schema to version.

## Decision

### 1. Three opaque newtypes, in place

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(u64);        // topology.rs
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InnovationId(u64);  // topology.rs
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SpeciesId(u64);     // species.rs
```

Each carries: `pub const fn new(u64) -> Self` (infallible), `pub const fn get(self)
-> u64` (the sole public accessor), and `pub(crate) const fn succ(self) -> Self`
(the *only* arithmetic, used exclusively by the `InnovationRegistry` / `speciate`
counters). The field is **private** — no `pub` data-bag field (rules §2), no
`Deref`, no `Display`, matching every existing newtype in the workspace.

The derive set is dictated by the use sites: `Eq + Hash` for the many
`HashMap`/`HashSet<NodeId>` keys (`phenotype.rs`, `innovation.rs` caches) and
`Ord + PartialOrd` for the `BTreeMap<NodeId, _>` (`phenotype.rs`), the
innovation-sorted `connections` invariant, and the ascending-id `sort_*` sites
that reproducibility relies on.

### 2. Kept in place, not hoisted to a new module

The newtypes replace the aliases where they already lived (`NodeId`/`InnovationId`
in `topology.rs`, `SpeciesId` in `species.rs`), so `neuroevolution/mod.rs` and
`lib.rs` re-exports and every import path are unchanged — only the underlying
definition changes. A dedicated `neuroevolution/ids.rs` was considered and
rejected as churn without payoff (see Alternatives).

### 3. Allocation stays behind `succ`

`InnovationRegistry` and `speciate` bump their counters with `succ` (and, for the
two-innovation node split, `InnovationId::new(cur.get() + 2)`); no caller does id
arithmetic. Seed-topology construction (`TopologyGenome::minimal`) and the empty
`compatibility_distance` default wrap their integer expressions in `new`.

## Consequences

### Positive
- Id-confusion is **unrepresentable**: passing a `NodeId` where an
  `InnovationId` is expected is a compile error. A clean build is itself the
  proof the id spaces are threaded correctly.
- The change is contained. Production consumer code (`neat.rs` crossover merge,
  `sort_by_key`, HashMap keys) works unchanged through the derives; only the
  `next_species_id` initializer needed a wrap.

### Negative / costs
- A third id-shaped primitive convention coexists with `Bounds` / the rate
  newtypes; mitigated by the shared opaque shape.
- Test and bench struct literals migrate their integer id literals to
  `NodeId::new(…)` / `InnovationId::new(…)` and compare via `.get()`; churn was
  minimized by wrapping inside existing per-file test helpers so call sites
  stayed terse.

### Neutral
- Purely additive to the public surface (same names, now distinct types); no
  serde, no schema, no downstream-crate impact (`rlevo-hybrid`, `rlevo-examples`,
  the NAS/weight-only genome paths are untouched).

## Alternatives considered

- **Transparent newtype with a `pub` field** (the older `RunId`/`ScalarReward`
  style). Rejected: rules §2 discourages `pub` data-bag fields; the opaque
  `new`/`get` shape is the current convention and keeps the id space sealed.
- **A dedicated `neuroevolution/ids.rs` module** holding all three. Rejected:
  changes intra-module import paths for no functional gain; the aliases already
  had settled homes.
- **Validated newtypes mirroring ADR 0031 exactly** (`try_new` + serde).
  Rejected: an id has no invariant to validate and no persisted form, so the
  validation surface would be dead weight.
- **Leave the aliases, add a lint / naming convention.** Rejected: an alias
  cannot be enforced by the compiler — the id-confusion the reviews flagged
  stays possible.

## References
- Issue #142 — this ADR resolves it.
- ADR [0027](0027-bounds-newtype-for-closed-ranges.md) — the opaque-newtype shape
  template (`new`/`get`, private field, dedicated construction).
- ADR [0031](0031-probability-rate-newtypes.md) — the immediately preceding
  newtype ADR; this one reuses its shape and records the no-validation/no-serde
  divergence.
- `docs/rules.md §2` — Struct Field Encapsulation (private field + accessor).
- Code: `crates/rlevo-evolution/src/neuroevolution/{topology,species,innovation}.rs`
  (definitions + allocators), `.../phenotype.rs` (the `HashMap<NodeId, usize>`
  id→row seam that constrained the derives),
  `.../algorithms/neuroevolution/neat.rs` (the sole production consumer).
