---
project: rlevo
status: active
type: decision
date: 2026-06-10
tags:
  - adr
  - decision
  - architecture
  - evolution
  - memetic
  - local-search
  - rng
  - rlevo
---

# ADR 0016: Memetic wrapper and host-side local-search seam (phase 3a)

## Status

Active. Adopted 2026-06-10. Implements phase 3a of the advanced-EA roadmap
(`specs/2026-04-27-advanced-evo-algos/memetic-algorithms.md`), with both research
gates closed in the vault: **3a-R1** commits the four-searcher v1 inventory
(`HillClimbing`, `NelderMead`, `SimulatedAnnealing`, `RandomRestart`; BFGS / Powell /
`argmin` deferred) and **3a-R2** commits `WritebackPolicy::Partial(0.5)` as the default
writeback. Purely additive: no changes to the `Strategy` trait, `EvolutionaryHarness`,
or any existing algorithm file. Does not supersede any prior ADR; extends the
host-RNG convention recorded in project_evolution_host_rng_convention and the
`parking_lot` unification of [0010-unify-on-parking-lot-across-viz-stack](0010-unify-on-parking-lot-across-viz-stack.md).

## Context

Memetic algorithms hybridise a population-level evolutionary search with a
per-individual local-refinement step. The phase-3a goal is narrow and concrete:
make local refinement a **composable, zero-cost-to-adopt** upgrade for any existing
`rlevo-evolution` strategy, such that `MemeticWrapper<DE, HillClimbing>` reaches a
target Rastrigin (D=10) fitness in *fewer fitness evaluations* than bare DE,
reproducibly. The design must land without disturbing the frozen public surface.

Four facts about the live API (verified against the current `feature/memetic-algos`
tree) constrain the design:

1. **`Strategy::tell` takes `&self`** (`crates/rlevo-evolution/src/strategy.rs:134`).
   The trait is deliberately pure — its own docs state it is "free of interior
   mutability (so many instances can run in parallel without locks)". A wrapper that
   *is* a `Strategy` therefore cannot mutate any owned `F` from inside `tell` without
   reaching for interior mutability of its own.

2. **Both fitness traits take `&mut self`.**
   `FitnessFn<G>::evaluate_one(&mut self, …)` and
   `BatchFitnessFn<B, G>::evaluate_batch(&mut self, …)`
   (`crates/rlevo-evolution/src/fitness.rs:37,51`). A fitness function is permitted to
   carry mutable state (eval counters, caches). Local search must call `F` *during*
   `tell`, so the wrapper needs `&mut F` from inside a `&self` method.

3. **`EvolutionaryHarness<B, S, F>` owns its own `F: BatchFitnessFn`**
   (`crates/rlevo-evolution/src/strategy.rs:290`) and drives the canonical
   `ask → evaluate_batch → tell` loop. The harness has no knowledge of memetics.

4. **All EA randomness is host-side via `seed_stream(base, generation, SeedPurpose)`**
   (`crates/rlevo-evolution/src/rng.rs`); `B::seed` / `Tensor::random` / `thread_rng`
   are forbidden in new code (process-global RNG mutex races parallel tests). The
   current max purpose tag is `Other = 6`.

Two composition shapes were on the table. **Option (b)** makes the harness
memetic-aware: it would gain a refinement hook and thread the *single* harness-owned
`F` into local search, eliminating the second fitness instance — at the cost of
editing `EvolutionaryHarness` (frozen) and coupling every run path to a memetic
concept it usually does not need. **Option (a)** keeps the harness untouched: the
`MemeticWrapper` is itself a `Strategy`, owns refinement entirely inside its `tell`,
and carries its *own* `F`. The harness then unavoidably holds a *second* `F` instance
to drive its batch loop. This ADR adopts option (a).

## Decision

**Ship phase 3a as a host-side local-search seam plus a `Strategy`-implementing
`MemeticWrapper`. The wrapper owns refinement inside its own `tell` (option (a)),
reconciles `tell(&self)` against `evaluate_*(&mut self)` via a `parking_lot::Mutex<F>`
field, drives refinement through a `LocalSearch<B>` trait over host-side `Vec<f32>`
genomes, and threads two derived RNG streams so the writeback policy is a
stream-position-invariant decision. The harness stays memetic-unaware and holds a
second, independent `F` instance. Four searchers ship in v1; `WritebackPolicy::Partial(0.5)`
and `CoveragePolicy::TopK { k: 1 }` are the defaults. All changes are additive.**

### Concrete parts

1. **`MemeticWrapper<B, S, L, F>` is a `Strategy<B>`; `F` lives behind `Mutex<F>`.**
   The wrapper composes an inner `S: Strategy<B>` with a `L: LocalSearch<B>` and owns
   the fitness function `F: BatchFitnessFn<B, …>` it refines against. Because
   `Strategy::tell` is `&self` but `F` needs `&mut`, `F` is held as
   `parking_lot::Mutex<F>` on the wrapper struct. `tell` locks once at the top of the
   refinement loop and drops the guard before delegating to `S::tell`. `Mutex<F>: Sync`
   needs only `F: Send`, already guaranteed by the `BatchFitnessFn: Send` bound — so
   the wrapper stays `Send + Sync` and satisfies the `Strategy: Send + Sync` supertrait.
   `F` lives on the *struct*, not in `Params` (which is `Clone + Debug` — a `Mutex<F>`
   is neither). `MemeticState` carries only the inner state plus a `generation: u64`.
   The wrapper's `Debug` is hand-written via `finish_non_exhaustive()` (harness
   precedent) to satisfy `missing_debug_implementations` without bounding `F: Debug`.

2. **Composition is option (a): the wrapper owns refinement; the harness holds a
   second `F`.** No edit to `EvolutionaryHarness` or `Strategy`. The standard run is
   `EvolutionaryHarness::new(MemeticWrapper::new(inner, local, fitness_a), …, fitness_b, …)`
   where `fitness_a` (wrapper-owned, refined inside `tell`) and `fitness_b`
   (harness-owned, drives `evaluate_batch` each generation) are **two instances of the
   same fitness function**. For stateless / pure fitness this is invisible. For
   *stateful* fitness (an eval counter, an adaptive cache) the two instances diverge;
   `MemeticWrapper::new` documents this, and callers that must aggregate share state
   across the two via `Arc<…>` interior counters. The headline benchmark does exactly
   this — two `CountingRastrigin` instances over one `Arc<AtomicUsize>` — so total
   evaluation counts (population + refinement) are correct.

3. **Two-stream RNG: one `next_u64()` per `tell`, policy-invariant.** Every `tell`
   draws **exactly one** `rng.next_u64()` from the threaded harness RNG — always,
   regardless of policy or coverage — so the harness RNG's stream position advances
   identically across all wrapper configurations. From that drawn base the wrapper
   derives two *independent* sub-streams:
   - `ls_rng  = seed_stream(base, generation, SeedPurpose::LocalSearch)` — fed to
     `LocalSearch::refine` for the search trajectory.
   - `mask_rng = seed_stream(base, generation, SeedPurpose::Replacement)` — drives the
     per-row Bernoulli writeback mask under `Partial(p)`.

   Because the two purposes index disjoint splitmix64 streams, the writeback mask is
   statistically independent of the refinement trajectory. The payoff is a clean
   identity: **`Partial(1.0)` is bit-identical to `Lamarckian`, and `Partial(0.0)` is
   bit-identical to `Baldwinian`** — the policy is a pure decision overlay that never
   perturbs the search RNG. `SeedPurpose::LocalSearch = 7` is added to `rng.rs`
   (additive; the only exhaustive match is `constant()` in that file). `Replacement`
   is reused for the mask.

4. **`LocalSearch<B>` is a host-side trait over `Vec<f32>` genomes.** The trait's
   `refine` consumes a host genome `Vec<f32>` and a `&mut dyn FitnessFn<Vec<f32>>`
   (the spec-frozen signature), returning the refined `(Vec<f32>, f32)`. The wrapper
   bridges the batch world to this scalar world through a private `RowFitness` adapter
   that wraps `&mut F` + `&Device`, builds a `[1, D]` tensor per `evaluate_one`, calls
   `evaluate_batch`, and pulls the scalar back. The `B` type parameter on
   `LocalSearch<B>` is **carried but unused** by every v1 searcher (all of which work
   purely in host `Vec<f32>` space); it is retained so a future on-device searcher can
   slot in without a trait-signature break. The device→host→device round-trip per
   refined row is **accepted debt** — refinement is deliberately the slow path, gated
   by `CoveragePolicy`.

5. **Four searchers in v1 (3a-R1), each monotone-non-worsening and fresh-fitness.**
   `HillClimbing` (FirstImprovement / BestImprovement, decaying step), `NelderMead`
   (standard α/γ/ρ/σ, budget-counted simplex init, degenerate-budget safety),
   `SimulatedAnnealing` (geometric / linear cooling, Gaussian proposal, returns
   best-so-far not the walker), and `RandomRestart<L>` (run 0 unperturbed for
   monotonicity, runs 1..=k perturbed, argmin). Every searcher's first action is to
   evaluate its input genome and seed a best-so-far tracker updated on every eval, so
   the returned `(genome, fitness)` is always the tracked best with *fresh* fitness —
   structurally guaranteeing monotone non-worsening output. A shared `BudgetedEval`
   wraps `&mut dyn FitnessFn<Vec<f32>>` + a remaining-evals counter so `max_iters`
   means total `evaluate_one` calls; a shared `clamp_vec` enforces bounds. Each
   searcher's params carry `bounds`, `max_iters`, and a `default_for(bounds)`
   constructor mirroring `DeConfig::default_for`.

6. **Policy enums and defaults.**
   `WritebackPolicy { Lamarckian, Baldwinian, Partial(f32) }`, `Default = Partial(0.5)`
   (3a-R2). Lamarckian writes refined genomes back into the population (`slice_assign`
   on refined rows); Baldwinian keeps the genome but hands `S::tell` the refined
   *fitness* (the ask tensor is passed through bit-identically); `Partial(p)` makes the
   choice per row from `mask_rng`. `CoveragePolicy { Full, TopK { k } }`,
   `Default = TopK { k: 1 }` — the cheapest sane default (refine only the single best
   individual). `TopK` selects the `k` smallest fitnesses, ties broken by index.

7. **Module layout.** `src/local_search.rs` holds the `LocalSearch<B>` trait,
   `BudgetedEval`, `clamp_vec`, and the `pub mod` + re-export declarations;
   `src/local_search/{hill_climbing,nelder_mead,simulated_annealing,random_restart}.rs`
   hold one searcher each with in-source tests. The wrapper lives at
   `src/algorithms/memetic.rs` alongside the other strategies (`WritebackPolicy`,
   `CoveragePolicy`, `MemeticParams`, `MemeticState`, private `RowFitness`,
   `MemeticWrapper`). Root re-exports add `LocalSearch`, the four searchers,
   `MemeticWrapper`, `WritebackPolicy`, `CoveragePolicy`, and the Params types.

8. **No manifest churn.** `parking_lot` is already a dependency (ADR 0010); no
   `argmin`, no `Autodiff`, no `Cargo.toml` edit anywhere. `git diff` on
   `strategy.rs` and every `Cargo.toml` must be empty.

### Reversal criteria

- If the **two-fitness-instance** edge bites in practice (a stateful fitness whose two
  copies must agree and `Arc`-sharing proves too clumsy), revisit **option (b)**: add
  an optional refinement hook to `EvolutionaryHarness` that threads the single
  harness-owned `F` into local search. This re-opens the frozen harness, so it is a
  deliberate later decision, not a default.
- If the **per-refine input re-evaluation cost** (every `refine` burns ≥1 eval
  re-scoring its own input because the signature carries no input fitness) dominates a
  real budget, add a `refine_with_known_fitness(genome, fitness, …)` default method to
  `LocalSearch<B>` that skips the seeding eval. This is **additive** — a defaulted
  trait method delegating to `refine` — and breaks nothing already shipped.
- If calibration cannot show the headline margin under the default searcher, the
  documented fallbacks are BestImprovement HC, `TopK{1}` + NelderMead, or (last resort)
  a `refine_every` cadence knob on `MemeticParams`.

## Consequences

**Positive**

- **Zero blast radius on the frozen surface.** `Strategy`, `EvolutionaryHarness`, and
  every existing algorithm file are byte-unchanged; the wrapper is *just another
  `Strategy`*, so it composes with DE, GA, and any future strategy for free, and runs
  through the existing harness and benchmark plumbing unmodified.
- **Reproducibility is structural, not best-effort.** The single-draw two-stream scheme
  makes writeback policy a stream-position-invariant overlay, so `Partial(1.0)≡Lamarckian`
  / `Partial(0.0)≡Baldwinian` hold *bit-identically* and are pinned by test. The whole
  feature stays inside the host-RNG convention; no new global-RNG hazard is introduced.
- **The slow path is opt-in and bounded.** `CoveragePolicy::TopK{1}` default means a
  default memetic run refines exactly one individual per generation; the cost scales
  only as the user widens coverage.
- **The seam extends cleanly.** The unused `B` on `LocalSearch<B>` reserves the
  on-device-searcher escape hatch; the defaulted `refine_with_known_fitness` extension
  is pre-cleared; the `LocalSearch` + `MemeticWrapper` split is the same
  trait-over-host-genomes shape that phase 3b/3c (EDA, CoEA) can reuse without a
  signature break.

**Negative / accepted costs**

- **The wrapper breaks the `Strategy` purity invariant — by design.** Every other
  strategy is lock-free so many instances parallelise without contention; the
  `MemeticWrapper` holds a `Mutex<F>`. This is contained: the lock is wrapper-private,
  taken once per `tell` and dropped before `S::tell`, and the inner strategy stays
  pure. But the wrapper is *not* a drop-in for code that assumed lock-free
  `tell` — it is a composition adapter, and the contention only matters if many
  wrappers over one `F` run concurrently (they do not; each harness owns its wrapper).
- **Two fitness instances.** Option (a) forces a second `F` in the harness. Pure
  fitness: invisible. Stateful fitness: divergent unless the caller shares state via
  interior-mutable handles. Documented at `MemeticWrapper::new`; the headline test
  demonstrates the `Arc<AtomicUsize>` sharing pattern.
- **Per-refine input re-evaluation.** Each `refine` spends ≥1 eval re-scoring its input
  because the frozen signature carries no input fitness. With the `TopK{1}` default the
  waste is one eval/generation; the `refine_with_known_fitness` reversal path is the
  documented mitigation when coverage is wide.
- **The host round-trip is real.** Refining a row pulls it to host and re-uploads a
  `[1,D]` tensor per `evaluate_one`. Accepted: refinement is the deliberately-slow
  path and is budget-gated.
- **Fresh-fitness is exact only for deterministic `F`.** For a stochastic fitness the
  "returned f32 == fresh re-eval" invariant is approximate; documented as such. The
  design never clamps refined fitness to old fitness (that would manufacture stale
  fitness on Lamarckian rows).

**Neutral**

- `SeedPurpose` gains one variant (`LocalSearch = 7`); `Replacement` is reused for the
  writeback mask. Additive; the purpose-stream test is extended.
- Refinement runs on *every* `tell`, including the first (the DE empty-fitness sentinel
  is agnostic; gen-0 refinement is standard MA practice). No special-case asymmetry.
- Tests follow the determinism.rs precedent (inline named fitness structs, no blanket
  `FnMut → FitnessFn` impl), matching the canonical-types-in-tests preference.

## Component diagram

```
                         EvolutionaryHarness<B, MemeticWrapper, F_b>
                         (memetic-unaware; owns the SECOND F instance)
                                          │
                  reset / step:  ask ──► evaluate_batch(F_b) ──► tell
                                          │
                                          ▼
        ┌──────────────────────── MemeticWrapper<B, S, L, F_a> ───────────────────────┐
        │  (IS-A Strategy<B>; owns the FIRST F instance behind a lock)                 │
        │                                                                              │
        │   ask(&self)  ──────────────►  S::ask          (pure delegate, unchanged)   │
        │                                                                              │
        │   tell(&self, pop, fit, …):                                                  │
        │     1. base = rng.next_u64()           (exactly one draw, policy-invariant)  │
        │     2. ls_rng   = seed_stream(base, gen, LocalSearch)                        │
        │        mask_rng = seed_stream(base, gen, Replacement)                        │
        │     3. coverage indices  ◄── CoveragePolicy (TopK{1} default)                │
        │     4. for each covered row:                                                 │
        │            ┌─────────────────────────────────────────────┐                  │
        │            │  L::refine(row: Vec<f32>, &mut RowFitness)    │                 │
        │            │     RowFitness ──► lock ──► Mutex<F_a> ──┐    │                 │
        │            │     evaluate_one ◄── [1,D] tensor ◄──────┘    │                 │
        │            └─────────────────────────────────────────────┘                  │
        │     5. WritebackPolicy (Partial(0.5) default) via mask_rng:                  │
        │            Lamarckian row → slice_assign genome back                         │
        │            Baldwinian row → keep genome, use refined fitness                 │
        │     6. rebuild fitness Tensor<B,1>                                           │
        │     7. ───────────────────────────────►  S::tell  (inner Strategy, pure)     │
        └──────────────────────────────────────────────────────────────────────────────┘

   Frozen / unchanged: Strategy trait, EvolutionaryHarness, DE/GA/all existing algos.
   New: src/local_search.rs (+ 4 searcher files), src/algorithms/memetic.rs,
        SeedPurpose::LocalSearch.
```

## Alternatives considered

**Option (b): make the harness memetic-aware.** Add a refinement hook to
`EvolutionaryHarness` and thread the single harness-owned `F` into local search.
Eliminates the second `F` instance and the `Mutex` (the harness already holds `&mut F`).
Rejected for phase 3a: it edits the frozen `EvolutionaryHarness`, couples every run
path to a memetic concept most runs do not use, and forfeits the "the wrapper is just
another `Strategy`" composability that makes adoption zero-cost. Retained as the
reversal path if the two-instance edge proves painful.

**`RefCell<F>` instead of `Mutex<F>`.** Rejected: `RefCell` is `!Sync`, which would
break the `Strategy: Send + Sync` supertrait and make the wrapper unusable in the
parallel `run_suite` path. `parking_lot::Mutex` is `Sync` with `F: Send`, already a
dependency, and the lock is uncontended in practice.

**Push `F` into `MemeticParams`.** Rejected: `Params` is `Clone + Debug`; a `Mutex<F>`
is neither, and cloning a fitness function per generation is wrong. `F` belongs on the
struct.

**Bake an input-fitness argument into `LocalSearch::refine` now** (skip the seeding
re-eval). Rejected for v1: the spec froze the signature, and the re-eval cost is one
eval/generation under the default coverage. The `refine_with_known_fitness` defaulted
extension is the clean additive path when the cost actually hurts.

**Couple the writeback mask to the refinement RNG** (one stream). Rejected: it would
make `Partial(1.0)` *not* bit-identical to `Lamarckian` (the mask draws would shift the
search stream), losing the clean policy-as-overlay identity and the corresponding
determinism tests.

## References

- `specs/2026-04-27-advanced-evo-algos/memetic-algorithms.md` — phase 3a spec; §4 trait
  signature (corrected against the live API), §6 acceptance items.
- 3a-R1 — closed research gate committing the four-searcher v1 inventory (HillClimbing,
  NelderMead, SimulatedAnnealing, RandomRestart; BFGS / Powell / `argmin` deferred).
- 3a-R2 — closed research gate committing `WritebackPolicy::Partial(0.5)` as default.
- [0010-unify-on-parking-lot-across-viz-stack](0010-unify-on-parking-lot-across-viz-stack.md) — `parking_lot` already a dependency;
  the `Mutex<F>` reuses it with no manifest change.
- project_evolution_host_rng_convention — the host-RNG / `seed_stream` convention
  the two-stream scheme obeys; `B::seed` / `Tensor::random` / `thread_rng` forbidden.
- reference_flex_gemm_nondeterminism — why reproducible runs need a seeded host
  stream + rayon=1; the headline test runs serial and twice.
- `crates/rlevo-evolution/src/strategy.rs` — `Strategy::tell(&self)` (line 134, frozen),
  `EvolutionaryHarness<B, S, F>` (line 290, frozen).
- `crates/rlevo-evolution/src/fitness.rs` — `FitnessFn`/`BatchFitnessFn`, both `&mut self`.
- `crates/rlevo-evolution/src/rng.rs` — `seed_stream` / `SeedPurpose` (gains `LocalSearch`).
- `crates/rlevo-evolution/src/algorithms/de.rs` — reference `Strategy` impl and
  `DeConfig::default_for` pattern the searcher params mirror.
```