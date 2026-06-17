---
project: rlevo
status: active
type: decision
date: 2026-04-27
tags: [adr, decision, architecture, crates, evolution]
---

# ADR 0002: Collapse `rlevo-core::evolution` traits into `rlevo-evolution`

## Status

Active. Supersedes the `rlevo-core::evolution` module shipped in earlier
alpha snapshots; that module no longer exists after this refactor.

## Context

`rlevo-core` was created to hold shape-erased vocabulary shared across
downstream crates (RL traits like `Agent`, `Environment`, `State`,
`Action`, `Memory`, etc.). Alongside the RL traits, a `rlevo-core::evolution`
module shipped three traits intended for the EA side of the workspace:

- `Fitness` — scalar fitness contract with a `worst()` sentinel,
  `is_finite()`, and `as_f32()`. Blanket-implemented for `f32`/`f64`.
- `MultiFitness` — minimal multi-objective view exposing `objectives()`,
  documented as the contract for a future NSGA family.
- `GenomeKind` — zero-sized marker trait tagging genome categories
  (`Real`, `Binary`, `Integer`, `Tree`, `Permutation`).

A review on 2026-04-27 found that `rlevo-evolution` consumed exactly **one**
of those traits (`GenomeKind`, in `crates/rlevo-evolution/src/genome.rs`),
and **no other workspace crate** consumed any of them. `Fitness` and
`MultiFitness` were aspirational placeholders: their documentation
described an integration with strategies in `rlevo-evolution` that was
never wired up. The `rlevo-evolution` strategy and fitness machinery
(`BatchFitnessFn`, `FitnessFn`, `Strategy`) hardcodes `f32` and
`Tensor<B, 1>` end-to-end and never names `Fitness` once.

The test for whether something belongs in `rlevo-core` is **>1 downstream
consumer with stable shared vocabulary**. The `evolution` module fails
that test. Keeping it there is **premature centralization** — the same
anti-pattern as premature abstraction, lifted to the crate boundary.

## Decision

**Collapse `rlevo-core::evolution` into `rlevo-evolution` and remove the
crate-level dependency.**

Concretely:

1. **Move `GenomeKind`** into `crates/rlevo-evolution/src/genome.rs`,
   co-located with its `Real`/`Binary`/`Integer`/`Tree`/`Permutation`
   impls.
2. **Delete `Fitness` and `MultiFitness`.** They are dead code; git
   history preserves them if they ever come back. Resurrecting a deleted
   trait is cheaper than maintaining speculative API surface.
3. **Delete `crates/rlevo-core/src/evolution.rs`** and the
   `pub mod evolution;` line in `crates/rlevo-core/src/lib.rs`.
4. **Drop `rlevo-core` from `crates/rlevo-evolution/Cargo.toml`.** After
   step 1, `rlevo-evolution` has zero `rlevo_core::` imports and no
   reason to keep the dependency edge.

## Consequences

**Positive:**

- **Tight crate boundary.** `rlevo-core` now holds only RL vocabulary;
  the EA crate stops being an "RL-flavored crate that happens not to use
  RL traits" and becomes a self-contained EA library.
- **Decoupled dep graph.** `rlevo-evolution` no longer depends on
  `rlevo-core` at all. Composition between RL and EA happens at the
  `rlevo-hybrid` layer, which is exactly where it should happen:

  ```
  rlevo-core                   (RL vocabulary)
  rlevo-reinforcement-learning ──→ rlevo-core
  rlevo-environments           ──→ rlevo-core
  rlevo-evolution              (standalone EA library)
  rlevo-hybrid                 ──→ rlevo-core, rlevo-reinforcement-learning, rlevo-evolution
  ```
- **No dead trait surface.** `Fitness` and `MultiFitness` no longer
  appear in `cargo doc --open` or in the rendered `rlevo-core` docs page,
  removing a misleading signal that the RL crate has an EA story.
- **Trait + impls are co-located.** `GenomeKind` and all its impls now
  live in the same file (`crates/rlevo-evolution/src/genome.rs`); the
  orphan rule never had to be navigated for new genome kinds.

**Negative / accepted costs:**

- **Doc churn.** Module-level doc in `rlevo-evolution/src/lib.rs` and
  the `genome` module doc previously referenced
  `rlevo_core::evolution::GenomeKind`; those are rewritten to point at
  the local trait.
- **MultiFitness deletion is irreversible-by-doc.** If/when an NSGA
  family lands, the trait will need to be re-derived from scratch (or
  pulled from git). Acceptable: NSGA isn't on the immediate roadmap and
  the trait was 6 lines.
- **`rlevo-evolution` no longer participates in any RL trait
  contract.** Hybrid algorithms that want both have to depend on
  `rlevo-core` and `rlevo-evolution` separately. This is the right
  layering — `rlevo-hybrid` already does this — but it means the EA
  crate cannot accidentally grow RL-flavored surface area without an
  explicit dep change.

**Neutral:**

- The `evolution` *naming* is gone from `rlevo-core`. If a future
  feature truly needs cross-crate EA vocabulary (e.g. a
  `GenomeKind`-aware `Agent` impl in `rlevo-reinforcement-learning`), promoting the trait
  back to core is a 5-minute refactor — much cheaper than the steady
  cost of a speculative core dependency today.

## Alternatives considered

**Move all three traits into `rlevo-evolution` (keep `Fitness`/
`MultiFitness` instead of deleting).** Rejected: relocating dead code
just transplants the speculative-API debt. The traits had zero
consumers and no concrete plan to acquire any; moving them would
preserve the misleading signal that they are part of the public
contract.

**Keep `GenomeKind` in `rlevo-core` "for future use".** Rejected: the
same premature-centralization argument that motivates this ADR. If a
second consumer materializes, lifting the trait back to core is cheap;
paying the dep edge today for a hypothetical future consumer is not.

**Promote `Fitness` to a real generic parameter on `BatchFitnessFn`
and `Strategy`.** Considered as a counter-proposal: instead of
deleting, wire `Fitness` into the strategy stack so `f64`-precision
strategies become possible. Rejected for v0.1 scope: the on-device hot
path returns `Tensor<B, 1>` (implicitly `f32`), and abstracting that
to `F: Fitness` would push fitness back to host-resident scalars or
require a parallel `Fitness`-typed tensor API. The complexity does not
earn its keep at the current alpha stage; the option remains open as a
future ADR if a real `f64` requirement appears.

## References

- Conversation 2026-04-27 (planning + execution).
- `crates/rlevo-core/src/evolution.rs` — deleted in this ADR.
- `crates/rlevo-evolution/src/genome.rs` — new home of `GenomeKind`.
- `crates/rlevo-evolution/Cargo.toml` — `rlevo-core` dep removed.
- ADR 0001 — `keep-environments-and-benchmarks-separate` — established the
  workspace's preference for disjoint dep cones over speculative
  shared vocabulary; this ADR applies the same principle to the
  RL/EA boundary.
