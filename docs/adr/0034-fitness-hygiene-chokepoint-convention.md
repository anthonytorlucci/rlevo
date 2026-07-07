---
project: rlevo
status: active
type: decision
date: 2026-07-07
tags: [adr, decision, fitness, nan, inf, sanitization, canonical, chokepoint, rlevo-evolution]
---

# ADR 0034: Fitness-hygiene chokepoint convention

## Status

**Accepted (2026-07-07).** Resolves the fitness-hygiene half of issues #131
(metaheuristics), #132 (EA-root), #133 (neuroevolution), and #134 (coevolution),
and generalizes the per-model fixes already shipped for #129 (EDA, PR #214) and
#130 (GEP). **Extends ADR [0023](0023-objective-sense-and-maximize-convention.md)** — it
preserves the `NaN → −∞` rule verbatim and adds an `+∞` rule; ADR 0023 stays
`active`.

**Chosen shape:** one fitness-hygiene primitive in `rlevo-evolution`, applied at
the small number of *driver chokepoints*, backed by the per-site
"sanitize-then-`total_cmp`" convention (`rules.md` §3) as the correctness floor.

## Context

Five issues (#129–#134) each name "NaN/Inf fitness sanitization" across a
different algorithm family. Read together they are not one defect but **three
classes** wearing one label:

1. **Fitness hygiene** — a non-finite fitness poisons ordering, best-so-far
   tracking, species allocation, hall-of-fame champions, and public metrics. Rust's
   `f32::NAN` is a *positive* NaN, so `total_cmp` ranks a raw `NaN` as the
   **maximum** (`rules.md` §3): a `NaN`-fitness member becomes an immortal
   champion. This class *is* addressable at a chokepoint.
2. **Canonical-ordering-direction bugs** — e.g. an onlooker tournament using `<` in
   maximise space, or an argmax initialized from `fitness[0]` instead of `−∞`. A
   perfectly sanitized tensor still selects the wrong individual. **Not**
   chokepoint-fixable.
3. **Parameter-numerics / config guards** — e.g. an unfloored `σ`, `genome_dim == 0
   ⇒ tau = inf`, a dataset silently zero-padded into `NaN` fitness. These are
   construction-time (`Validate`, ADR 0026) or field-newtype (ADR 0027/0031)
   concerns applied *before* fitness exists. **Not** chokepoint-fixable.

Before this ADR, class 1 was handled ad-hoc: a `pub(crate)`, scalar,
**NaN-only** `sanitize_fitness` was applied at some sites (`StrategyMetrics`,
`hof.rs`, `species::remove_stagnant`, local-search `BudgetedEval`, `EdaStrategy::
tell`) and forgotten at most others, so each new family re-discovered the same
poisoning bug in review.

The key structural observation is that the engine already funnels every
`Strategy<B>` run through **one place**: `EvolutionaryHarness::step` evaluates the
batch, canonicalizes via `sense`, and hands the fitness tensor to
`Strategy::tell`. Three other drivers exist outside it — `NeatStrategy::tell`,
`ArchNasStrategy::tell` (both inherent, not `Strategy`), and the coevolution
coupled-fitness path — each its own analogous funnel.

## Decision

1. **One primitive, extended Inf policy.** In `rlevo-evolution/src/fitness.rs`,
   `sanitize_fitness(f: f32) -> f32` applies the single canonical rule:
   - `NaN → f32::NEG_INFINITY` (worst under maximise — preserved from ADR 0023),
   - `+∞ → f32::MAX` (ranks top but **finite**, so it cannot blow a `mean`,
     `variance`, or reward to `+∞`),
   - `−∞` and finite values pass through (`−∞` is the worst-sentinel *and* the
     uninitialized `best_fitness_ever` seed, and must stay non-finite so
     mean-over-finite logic can detect it).

   Add a device-op sibling `sanitize_fitness_tensor<B>(Tensor<B,1>) ->
   Tensor<B,1>` — `is_nan` → `mask_fill(−∞)` → `clamp_max(f32::MAX)` — so the
   tensor-holding chokepoints sanitize on the hot path with no host round-trip.
   Both stay `pub(crate)`: every caller is in `rlevo-evolution`. (Reuse note per
   the "check Burn built-ins" rule: the `is_nan` + `mask_fill` + clamp idiom is
   already used by `EdaStrategy::tell`'s gene backstop; no new hand-rolled op.)

2. **Apply at the four driver chokepoints:**
   - `EvolutionaryHarness::step` — `sanitize_fitness_tensor` on the **canonical**
     tensor (after the `sense` negation). Sanitizing the *natural* tensor before
     `neg()` would flip a `NaN` cost to `+∞` = canonical *best* under `Minimize`;
     "NaN = worst" is only well-defined in maximise space. This one insert covers
     **every** `Strategy<B>` impl — all metaheuristics (#131), all EA-root
     algorithms (#132), EDA, and `WeightOnly`.
   - `NeatStrategy::tell` → before `state.fitness` / `speciate`.
   - `ArchNasStrategy::tell` → before `update_best` (and init best from `−∞`).
   - Coevolution coupled-fitness → `CoEAState` write sites + the `min(best_a,
     best_b)` metric.

3. **Keep the per-site `rules.md` §3 convention as the correctness floor.** The
   chokepoint is an *amortization* (sanitize once, downstream stored fitness is
   clean), **not** a guarantee for callers that bypass the harness (unit tests,
   custom drivers, the three non-`Strategy` drivers). Every ordering / argmax /
   champion-write site still sanitizes locally. The chokepoint does not delete
   these; it makes them redundant on the common path and load-bearing on the rest.

4. **Metrics: mean over finite members.** `StrategyMetrics::from_host_fitness`
   averages the finite members only and reports a `broken_count` of the non-finite
   (`−∞`) ones, so a single broken individual flags the population without
   blanking the mean to `−∞`; a `+∞ → f32::MAX` member is finite and is included.

5. **Contract amendments.** `BatchFitnessFn::evaluate_batch` documents that its
   output may be non-finite and the harness sanitizes; `Strategy::tell` documents
   the harness-scoped finite-or-`−∞` guarantee *and* the bypass caveat;
   `CoupledFitness::evaluate_coupled` and the NEAT/NAS `tell` docs state their own
   sanitization boundary (they are their own drivers, with no harness above them).

**Explicitly out of scope** (separate work-streams, not owned by this ADR):
class-2 ordering-direction bugs and class-3 parameter-numerics/config guards
above; and the GEP #130 per-eval `1e15` bloat clamp, which is a domain modeling
penalty inside `tree.rs`, not fitness hygiene, and is deliberately *not* routed
through this primitive.

## Consequences

### Positive
- **One rule, four inserts, closes a whole defect class.** ~25 per-site fixes the
  individual reviews proposed collapse to a shared primitive plus four chokepoints.
- **New families are safe by default.** Any future `Strategy<B>` inherits the
  hygiene guarantee from the harness with no per-algorithm code.
- **Honest metrics.** `+∞` can no longer masquerade as an infinite mean, and a
  broken member is surfaced (`broken_count`) instead of silently blanking the mean.

### Neutral
- `StrategyMetrics` gains a `broken_count` field and accessor. Field names feed the
  benchmark TUI's canonical-metric registry; the new tracing field is additive.

### Negative / accepted costs
- **Documented bypass hole.** Direct (non-harness) `tell` callers are not covered
  by the chokepoint; correctness there rests on the §3 per-site convention. This is
  stated in the `Strategy::tell` docs. If the hole recurs in practice, the
  sanctioned escalation is a `CanonicalFitness<B>` newtype (below).
- **`−∞` still yields a `−∞` mean when the whole population is broken.** Accepted:
  degenerate but well-defined, and flagged by `broken_count == population_size`.

## Alternatives considered

- **`CanonicalFitness<B>` newtype** wrapping `Tensor<B,1>`, constructible only via
  the sanitizing constructor, as `Strategy::tell`'s parameter — makes "hand raw
  fitness to `tell`" a *compile error*, closing the bypass hole by type (the
  `Probability`/`NonNegativeRate`/opaque-id lineage, ADR 0027/0031/0032). Deferred,
  not rejected: it changes the central trait signature and touches every `Strategy`
  impl. Named here as the fast-follow if a bypass regression recurs.
- **Per-site fixes only (as each issue's review proposed).** Rejected as the
  primary structure: ~25 edit sites, and every future family re-discovers the bug.
  Retained only as the correctness floor (decision 4/§3), not the main mechanism.
- **Move the primitive to `rlevo-core`.** Rejected: `rlevo-core` is a contract crate
  (no implementation logic beyond `util`), and `NaN → −∞ = worst` is the *engine's*
  canonical-space convention, not an abstract objective primitive. All four
  chokepoints are in `rlevo-evolution`. Promote only when a second crate needs it.
- **`assert!(fitness.is_finite())` (proposed in the metaheuristic review).**
  Rejected: `rules.md` §4 forbids panicking on user-supplied runtime data, and a
  landscape returning `0/0` *is* runtime data. Sanitize to a sentinel instead; a
  `debug_assert!` tripwire is acceptable, a hard `assert!` is not.
- **Map all non-finite → −∞ (drop the `+∞ → f32::MAX` distinction).** Rejected: in a
  maximise objective `+∞` means *optimal*; mapping it to *worst* would silently
  delete the best individual (e.g. runaway-weight architectures that legitimately
  peg the objective).

## References
- Issues #131 / #132 / #133 / #134 — fitness-hygiene halves resolved here; their
  class-2/class-3 items split to follow-up issues.
- Issues #129 (PR #214, EDA) / #130 (GEP) — the per-model precedents this
  generalizes.
- ADR [0023](0023-objective-sense-and-maximize-convention.md) — maximise-native canonical
  convention, extended (not superseded) here.
- ADR [0026](0026-shared-config-validation-convention.md) — construction-time `Validate`
  boundary where the class-3 guards belong.
- `rules.md` §Optimisation direction — the per-site sanitize-then-`total_cmp`
  correctness floor.
- Code: `crates/rlevo-evolution/src/fitness.rs` (primitive + tensor sibling),
  `src/strategy.rs` (harness chokepoint + `StrategyMetrics`), `src/neuroevolution/
  species.rs`, `src/algorithms/neuroevolution/{neat,arch_nas}.rs`,
  `src/coevolution/{fitness,competitive,cooperative,harness}.rs`.
