---
project: rlevo
status: active
type: decision
date: 2026-07-08
tags: [adr, decision, coevolution, objective-sense, canonical, chokepoint, maximize, rlevo-evolution]
---

# ADR 0035: `CoupledFitness::sense` and the co-evolution canonicalisation chokepoint

## Status

**Accepted (2026-07-08).** Closes the co-evolution parity half of issue #160.
**Extends ADR [0023](0023-objective-sense-and-maximize-convention.md)** to the
coupled-fitness path and reuses the canonicalise-then-sanitize chokepoint shape
of ADR [0034](0034-fitness-hygiene-chokepoint-convention.md). It **does not**
edit either — ADR 0023 and ADR 0034 stay `active`.

## Context

ADR 0023 made the whole engine maximise-native and confined the cost↔canonical
mapping to one chokepoint: fitness functions return **natural** values, declare
their direction via `sense()`, and the `EvolutionaryHarness` canonicalises once
per generation. `BatchFitnessFn::sense()` is **required, no default**, so a
reward/score objective cannot be optimised backwards by omission.

The co-evolution module was the one path the ADR 0023 flip missed. Before this
ADR, `CoupledFitness::evaluate_coupled` was documented as returning fitness
already in "the crate-wide canonical **maximise convention**" — implementors had
to hand-negate a cost themselves, and there was no `sense()`. Consequences:

- **No single source of truth for direction.** A `Minimize` coupled objective
  had to be pre-negated in user code (the two bundled examples do exactly this),
  re-introducing the hand-negation ADR 0023 deleted from the single-population
  path.
- **`CoEAMetrics` reported canonical values**, diverging from single-population
  `StrategyMetrics`, which reports **natural** values (ADR 0023 RD-2). A
  `Minimize` cost surface's `best_fitness` did not read as its natural cost.
- **The hall of fame argmaxed raw returned fitness.** Correct only if the caller
  pre-negated; a natural `Minimize` cost would have crowned the highest-cost
  (worst) individual as champion.

The fitness-hygiene chokepoint (ADR 0034) already existed in each co-evolution
`step`, but it sanitized the *returned* tensor directly — correct only while that
tensor was assumed canonical.

## Decision

1. **`CoupledFitness::sense()` is required, no default** — mirroring
   `BatchFitnessFn::sense` (ADR 0023 RD-1). `evaluate_coupled` now returns
   **natural** fitness (a cost as its natural cost, a score as its natural value,
   no hand-negation).

2. **The co-evolutionary algorithm `step` is the coupled canonicalisation
   chokepoint** — the analogue of `EvolutionaryHarness::step` for the coupled
   path. It reads `sense()` once, then applies **canonicalise-then-sanitize** to
   each returned population vector: negate iff `Minimize`, **then**
   `sanitize_fitness_tensor` (`NaN → −∞`, `+∞ → f32::MAX`). The ordering is
   load-bearing and identical to the golden reference: "NaN = worst" is only
   well-defined in maximise space, so sanitizing before `neg()` would flip a
   `NaN` cost to `+∞` = canonical *best* under `Minimize`. The `CoEAState`
   best/mean trackers stay canonical.

3. **`CoEAMetrics` display fields are natural; `binding_fitness` is canonical.**
   The four `best_fitness_{a,b}` / `mean_fitness_{a,b}` fields are mapped back to
   the objective's natural sense via `from_canonical` (parity with
   `StrategyMetrics`). A new `binding_fitness` field carries the **canonical**
   `min(best_a, best_b)` — the weaker population binds — and is the harness
   reward, read from the dedicated field rather than re-derived off the (now
   natural) display fields.

4. **The hall of fame canonicalises champion selection.**
   `HallOfFameFitness::evaluate_coupled` returns natural blended fitness (blend is
   affine and `to_canonical` is negation, so canonicalising the blend equals
   blending the canonicalised terms — the algorithm canonicalises it downstream),
   but it canonicalises the current-gen fitness **unconditionally** (even at blend
   weight `0`) before handing it to `HallOfFame::update`, whose argmax/eviction
   are sense-blind maximise-space operations. `HallOfFame::update`'s contract is
   amended to state the caller must pass canonical fitness.

5. **Uniform-common-direction constraint.** `sense()` applies to **every**
   returned population vector — all populations are reported in one common
   direction. A truly asymmetric / zero-sum objective (population A maximises a
   score, B minimises the same score) has **no** compliant single-`sense()`
   representation and must be pre-expressed by the implementor (return A's score
   and B's negated score, both `Maximize`). A per-population sense vector is
   **out of scope for v1** — the same deferral posture ADR 0023 takes for the MOO
   sense vector.

**Also in this change (co-located, not direction-related):** the
`HallOfFameFitness` mutex critical section is narrowed to a cheap archive
*snapshot* plus the final `update`, releasing the lock across the expensive inner
`evaluate_coupled` calls (the wrapper is logically serial per instance).
`CooperativeState` caches its `dims_b` complement (frozen at `init`) so `step`
never re-derives the split — state-caching, not a `DimSplit` newtype, since the
`Validate` impl already enforces the split invariant at the harness chokepoint
(ADR 0026).

## Consequences

### Positive
- **One direction convention, library-wide.** The coupled path now matches the
  single-population path (ADR 0023); a `Minimize` coupled objective is optimised
  correctly with no user-side hand-negation, and its `best_fitness` reads as its
  natural cost.
- **The hall of fame is correct under `Minimize`** — champions are the best
  (lowest-cost) individuals, not the worst.
- **Honest metrics.** `CoEAMetrics` display values read in user space; the
  reward's canonical binding value lives in its own field.
- **Narrower lock, no per-generation `dims_b` alloc.**

### Neutral
- `CoEAMetrics` gains a `binding_fitness` field (additive).

### Negative / accepted costs
- **Atomic trait change.** Adding a required `sense()` breaks every
  `CoupledFitness` impl until it declares its direction; the crate does not
  compile until all impls (production, examples, tests) are updated in one pass.
  Accepted — this is the same required-no-default posture ADR 0023 chose for
  `BatchFitnessFn::sense`, and it prevents silent backwards optimisation.
- **Asymmetric zero-sum objectives are not first-class in v1** — they must be
  pre-expressed by the implementor (documented on the trait). A per-population
  sense vector is deferred, mirroring the MOO deferral.

## Alternatives considered

- **Default `sense() -> Maximize`.** Rejected: it re-opens the exact footgun
  ADR 0023 closed for `BatchFitnessFn` — a cost objective silently optimised
  backwards by omission. The required-no-default rule is the whole point.
- **Keep `CoEAMetrics` canonical.** Rejected: diverges from single-population
  `StrategyMetrics` (ADR 0023 RD-2); a `Minimize` cost would not read as its
  natural cost. The `binding_fitness` field resolves the one genuinely-canonical
  consumer (the reward) without leaking canonical values into display.
- **`DimSplit` newtype for the cooperative split.** Rejected for v1: it changes
  the public `CooperativeCoEAParams` shape, and the `Validate` impl already
  enforces the invariant at the ADR 0026 chokepoint. State-caching is minimal and
  non-breaking.
- **Per-population sense vector now.** Rejected for scope, mirroring ADR 0023's
  MOO deferral; the uniform-direction constraint plus a pre-negation escape hatch
  covers v1.

## References
- Issue #160 — co-evolution ADR 0023 parity and robustness (resolved here).
- ADR [0023](0023-objective-sense-and-maximize-convention.md) — maximise-native
  convention and the single-population chokepoint, extended (not superseded) here.
- ADR [0034](0034-fitness-hygiene-chokepoint-convention.md) — the
  canonicalise-then-sanitize chokepoint shape reused here.
- ADR [0026](0026-shared-config-validation-convention.md) — the `Validate`
  boundary the `dims_b` state-cache relies on.
- Code: `crates/rlevo-evolution/src/coevolution/{fitness,competitive,cooperative,
  harness,hof}.rs`; `crates/rlevo-evolution/src/strategy.rs`
  (`EvolutionaryHarness::step`, the golden reference).
</content>
</invoke>
