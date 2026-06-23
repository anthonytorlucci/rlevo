---
project: rlevo
status: active
type: decision
date: 2026-06-21
tags: [evolution, convention, objective-sense, fitness, reward, multi-objective, core]
---

# ADR 0023: `ObjectiveSense` and the maximise-native convention

## Status

Active. Adopted 2026-06-21. Establishes a single optimisation direction for the
whole library and an explicit per-objective sense contract. **Not purely
additive** — it flips the internal direction of the entire `rlevo-evolution`
engine (a one-time refactor) and bumps the record schema. Touches `rlevo-core`,
`rlevo-evolution`, `rlevo-hybrid`, `rlevo-benchmarks`, the report client,
examples, and docs. **Does not supersede** ADR 0002 — it *resurrects* the MOO
seam that ADR 0002 deliberately deleted (see Decision §5). Extends ADR 0014
(schema). Refines the "negate by hand" wording introduced with ADR 0004's
`FitnessEvaluable`/`Landscape` move.

## Context

`rlevo` spans reinforcement learning (maximise return), evolutionary computation
(maximise fitness — textbook, and the project's own roadmap table), and deep
learning (minimise loss). Before this ADR the codebase did **not** speak one
direction:

- **RL was maximise** (`AgentStats::best_score` uses `.max`; `Reward` itself is
  direction-neutral).
- **The EA engine was minimise**, pervasively: `StrategyMetrics::best_fitness`
  was the smallest value; `from_host_fitness` used `if f < best`; `best_fitness`
  initialised to `f32::INFINITY`; selection/replacement/GA/DE/EP/ES/CMA-ES/EDA
  and every metaheuristic compared with `<` and sorted ascending; `NaN`
  sanitised to `+inf`.
- **NEAT was an explicit exception** ("opposite the crate-wide minimize
  convention … a cost task supplies `−cost`").
- **Three bridges hand-negated in user-visible code:** `EvolutionaryHarness::step`
  (`reward = -best_fitness_ever`), `rlevo-hybrid::rollout_fitness`
  (`fitness = -total/episodes`), and the `FitnessEvaluable`/`Landscape` docs that
  *told the user to negate*.

The EA engine adopted the **deep-learning** (minimise) convention. That was
locally reasonable — the 24 bundled benchmark landscapes are minimise-native cost
surfaces (zero at the optimum) — but it left the library split-brained against
the two fields that name it, forced NEAT into an exception, and pushed sign-flips
into example and bridge code. Multi-objective optimisation (on the research
roadmap; documented in the user-book) makes a single global direction
structurally impossible and forces the issue: dominance is defined *per objective
sense*.

The project is pre-1.0 on a `pre-release` branch with API stability as an
explicit current focus — the right time to fix the contract before it ossifies
and before MOO forces a messier version of the same change.

Constraints that shape the design:

- `Strategy::tell` takes `&self`; all randomness is host-side via `seed_stream`.
  The flip must not disturb either.
- `rlevo-evolution` already depends on `rlevo-core` (ADR 0004 re-added it via the
  bench-trait move), so a shared primitive can live in `rlevo-core`.
- Determinism is load-bearing for the convergence tests (seeded Flex +
  rayon-1-thread).

## Decision

**Adopt a single internal convention — *maximise, higher = better* — across the
whole library, and introduce an explicit `ObjectiveSense { Minimize, Maximize }`
primitive in `rlevo-core` that reconciles cost objectives at exactly one boundary
chokepoint.**

### 1. `ObjectiveSense` in `rlevo-core`

A zero-cost `enum ObjectiveSense { Minimize, Maximize }` with an involutive
`to_canonical(raw) → f32` (negate iff `Minimize`) and `from_canonical` inverse.
Neutral home in `rlevo-core::objective`; consumed by evolution, hybrid, and
benchmarks. It is the **K = 1 atom** of the future multi-objective sense vector
(§5).

### 2. The engine is maximise-native and sense-unaware

Every `Strategy`, operator, shaping rule, local searcher, and metric aggregation
in `rlevo-evolution` works purely in **canonical (maximise) space** and never
sees an `ObjectiveSense`. This is the full internal flip of ~30 strategy files
plus operators/shaping/metrics/harness. NEAT stops being an exception; the
`EvolutionaryHarness` reward negation is deleted.

### 3. Cost is reconciled at one chokepoint: the harness

Sense is declared on the **fitness function** (`BatchFitnessFn::sense`, required —
RD-1); fitness fns and adapters return **natural** values. The
**`EvolutionaryHarness` is the sole canonicaliser**, the only code that knows the
mapping:

- **Ingest:** the harness reads `fitness_fn.sense()` and applies `to_canonical`
  to the fitness tensor before `tell` — a `Minimize` objective is negated so the
  maximise engine optimises `−cost`.
- **Report:** the harness applies `from_canonical` when surfacing
  `best_fitness`/`best()`/records, so a `Minimize` landscape reads as its natural
  cost (Sphere → 0). Because canonical space is already higher-is-better, the old
  `reward = -best_fitness_ever` negation is **deleted** (reward = canonical
  `best_fitness_ever`).

`EvolutionaryHarness::new` is **unchanged** (no sense arg — single source of truth
on the fitness fn). `rlevo-hybrid::rollout_fitness` implements `sense() ->
Maximize` and returns the natural return; its hand-negation is deleted.

### 4. Declaration policy

- `Landscape` is a cost surface by definition → `sense()` defaults to `Minimize`;
  the 24 landscapes need no per-function change.
- Higher-level problems must declare sense **explicitly** (no silent default on
  the problem seam) so a reward/accuracy objective cannot be optimised backwards
  by omission. `BatchFitnessFn::sense()` is required, no default.
- **Landscape tests, benches, and examples pass `ObjectiveSense::Minimize`
  explicitly** (maintainer directive) via the adapters' `with_sense` constructor
  and assert in user space.

### 5. Resurrect — do not block — the MOO seam

`ObjectiveSense` is designed as the K = 1 case of a per-objective sense vector.
The `MultiFitness` trait that ADR 0002 deleted as dead code is the natural home
when NSGA-II/SPEA2 land: `objectives()` + `senses()`, with dominance
canonicalising every objective to maximise then applying "≥ on all, > on one".
This ADR does not build MOO; it guarantees MOO is additive.

### 6. Downstream: schema v7

Add `objective_sense: Option<ObjectiveSense>` (serde-default) to **`RunManifest`**
— *not* `MetricDescriptor`, which is `#![no_std]` zero-dep (ADR 0015) and holds
metric vocabulary, not a run's direction (RD-3). Bump `FORMAT_VERSION` 6→7
(`record/schema.rs` + report-client `wire.rs` + `MIN_SUPPORTED_VERSION` + the
compat test); no migration logic (no pre-1.0 back-compat). `None` ⇒ `Maximize`
(canonical default), so RL and unspecified runs render correctly. The report
client's three direction-hardcoded transforms read the manifest field.

### 7. Resolved sub-decisions (RD-1..RD-4)

- **RD-1.** `ObjectiveSense` is declared on the fitness fn (`BatchFitnessFn::sense`,
  required); harness reads it; `new` unchanged; `Problem` wrapper deferred to
  MOO. Rejected: sense on the harness ctor (second source of truth). *Refinement
  during implementation:* the required `sense()` lives on `BatchFitnessFn` only,
  not on the low-level single-member `FitnessFn` — the local-search seam operates
  entirely in canonical space (the memetic `RowFitness` adapter canonicalises via
  the wrapped `BatchFitnessFn::sense`), so a `sense()` on `FitnessFn` would never
  be consulted (the speculative-surface anti-pattern RD-4 warns against).
- **RD-2.** `StrategyMetrics` is canonical from `tell` (strategies sense-unaware);
  the harness maps to natural space for `latest_metrics`/`best`/records/tracing;
  reward stays canonical. Rejected: sense-carrying metrics.
- **RD-3.** Schema field on `RunManifest`, not the registry (see §6).
- **RD-4.** RL untouched — `Reward` direction-neutral, `AgentStats` `.max`; RL
  already conforms. Rejected: threading sense into `Reward`/RL metrics
  (vestigial, the ADR 0002 speculative-surface anti-pattern).

## Consequences

**Positive**

- **One convention, library-wide.** EA, RL, and NEAT all maximise internally; the
  split-brain and the NEAT exception are gone. A contributor reads `de.rs` /
  `ga.rs` and sees the EC-textbook direction.
- **Hand-negation eliminated from user space.** The only negation left is the
  honest cost↔canonical mapping, confined to two chokepoint layers and
  `grep`-able.
- **Results read naturally.** `best_fitness` is the user's value in the user's
  sense; the report can label direction instead of assuming it.
- **MOO is unblocked, not pre-built.** The scalar contract is literally the K = 1
  restriction of the multi-objective one.
- **Right time.** Pre-1.0, before the convention ossifies or MOO forces it.

**Negative / accepted costs**

- **A large, correctness-sensitive one-time flip** (~30 strategy files + ops +
  shaping + metrics + harness), several sites non-mechanical (CMA-ES/CMSA-ES/ES
  rank-µ recombination ordering, NES shaping utilities, EDA winner/loser and
  truncation, local-search accept rules, the `NaN` sentinel). Mitigated by a
  behaviour-preserving characterization baseline plus targeted unit tests per
  subtle site.
- **Schema break (v6→v7)** with a report-client update and a v6-reader migration
  note.
- **A cheaper alternative was declined** (minimise-native engine + a public
  maximise façade, ~3 files). Rejected because it leaves a permanent internal
  split-brain; the flip buys long-term coherence at a one-time cost (see
  Alternatives).
- **Footgun:** a `Maximize` problem that omits its sense. Mitigated by requiring
  explicit sense on the problem seam (only `Landscape` defaults).

**Neutral**

- RL is untouched (already maximise); the ADR documents that it conforms rather
  than changing it.
- `Reward` stays direction-neutral; `ObjectiveSense` is the typed direction when
  one is needed.

## Component diagram

```
                         user space                         canonical (engine) space
                (the sense the problem declares)          (always maximise, higher better)

  reward / fitness / accuracy ── Maximize: pass through ───────────►  ┐
  cost / loss / landscape     ── Minimize: to_canonical(−x) ───────►  │  Strategy / ops /
                                          (adapter + harness)         │  shaping / metrics
                                                                      │  — sense-unaware,
  best_fitness / records / showcases ◄── from_canonical (−x iff Min) ─┘    maximise-native

  rlevo-core::objective::ObjectiveSense  ── the one primitive (K=1 atom of MOO senses)
  Chokepoint = {FromLandscape, FromFitnessEvaluable, EvolutionaryHarness} ONLY.
  Deleted: NEAT "opposite the convention" exception; harness reward negation;
           rollout_fitness hand-negation; "negate your objective" user docs.
  Deferred (seam only): MultiFitness.objectives()+senses() → Pareto dominance.
```

## Alternatives considered

**Minimise-native engine + a maximise public façade (the cheap chokepoint).**
Keep the ~30 files minimise; flip only the boundary so the *contract* reads
maximise. ~3 files, no risky internal flip. Rejected: it delivers the same
user-visible behaviour but leaves EA minimising while RL/NEAT maximise — a
permanent internal split a cross-cutting reader must carry, and NEAT stays an
exception. The maintainer chose long-term coherence over the cheaper diff.

**Flip globally to maximise with no sense primitive.** Just invert the engine and
hand-negate the landscapes at call sites. Rejected: relocates the hand-negation
onto the most common benchmark path and does not generalise to MOO.

**Keep minimise everywhere (status quo).** Rejected: contradicts RL and the EC
textbook/roadmap; forces the NEAT exception and three hand-negations; and breaks
down entirely under MOO.

**Per-objective sense + full MOO now.** Build `MultiFitness` + dominance
immediately. Rejected for scope: MOO is on the *research* roadmap; the scalar
contract here is its K = 1 case, so deferring costs no rework.

**Put `ObjectiveSense` in `rlevo-evolution` (not core).** Rejected: RL metrics,
hybrid, and benchmarks all reason about direction; the primitive belongs at the
shared base, and the `rlevo-evolution → rlevo-core` dep already exists (ADR 0004).

## References

- ADR 0002 — deleted `Fitness`/`MultiFitness`; the MOO seam resurrected here.
- ADR 0004 — `FitnessEvaluable`/`Landscape` in `rlevo-core`; the dep enabling
  `ObjectiveSense` there.
- ADR 0014 — record schema v6; this bumps to v7.
- ADR 0016 / 0017 / 0021 — `LocalSearch`, EDA, CMA-ES/CMSA-ES: the subtle flip
  sites.
- `crates/rlevo-core/src/objective.rs` — the `ObjectiveSense` primitive.
- `crates/rlevo-evolution/src/strategy.rs`, `.../fitness.rs`,
  `.../ops/{selection,replacement}.rs`, `.../shaping.rs` — the chokepoint + flip
  surface.
- `docs/user-book/src/part-1-foundations/20-evolutionary-computation.md`
  §Multi-Objective Optimisation.
