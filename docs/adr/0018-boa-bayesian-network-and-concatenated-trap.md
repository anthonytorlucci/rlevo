---
project: rlevo
status: active
type: decision
date: 2026-06-12
tags: [evolution, eda, boa, bayesian-network, landscape, trap, phase-3b]
---

# ADR 0018: BOA (`BayesianNetwork`) EDA and the `ConcatenatedTrap` landscape

## Status

Active. Adopted 2026-06-12. Implements issue #37 (the BOA item descoped from
issue #31 / PR #38) per the sub-spec
boa-bayesian-network-eda. **Additive**: extends
[0017-probability-model-trait-and-eda-strategy](0017-probability-model-trait-and-eda-strategy.md) without superseding it — the
`ProbabilityModel<B>` trait, `EdaStrategy`, `Strategy`, the harness, and every
manifest are byte-unchanged. No new dependency.

## Context

The four shipped EDA models (UMDA, PBIL, cGA, MIMIC) factorise over single
genes or a first-order chain. On problems with higher-order deceptive linkage
— where the cost of a block of bits depends on the whole block — such models
are driven *away* from the optimum: in a random population, lower-unitation
blocks are cheaper on average, so per-gene statistics push every bit toward
`0`. BOA (Pelikan, Goldberg & Cantú-Paz 1999) learns a Bayesian network over
the genes each generation, capturing the `k`-way dependencies needed to sample
and preserve coupled blocks.

Issue #37's acceptance bar requires a multivariate-dependency landscape where
BOA beats both UMDA and MIMIC. No discrete deceptive landscape existed (all 22
landscapes in `rlevo-environments` are continuous), so the benchmark ships
first: `ConcatenatedTrap`, the canonical Deb & Goldberg (1992) deceptive trap
(`cost(u) = 0` if `u == k` else `u + 1`, summed over contiguous
`block_size`-bit blocks; global optimum all-ones, deceptive basin all-zeros).

Research note eda-probability-model-taxonomy (3b-R1) left two BOA
questions open: whether structure learning needs a trait hook, and how to
regularise spurious edges (the shipped MIMIC was strictly worse than UMDA
until a `|r| < 2/√k` significance filter was added). Both are resolved here.

## Decision

### 1. DAG-in-`State`; no trait hook (resolves 3b-R1 Q1)

The learned network lives entirely in `BayesianNetworkState`:

```rust
pub struct BayesianNetworkState {
    pub order: Vec<usize>,        // topological sampling order
    pub parents: Vec<Vec<usize>>, // parents[node], sorted ascending
    pub cpt: Vec<Vec<f32>>,       // cpt[node][config] = P(node = 1 | config)
}
```

Structure learning happens inside the single `fit()` call, exactly like
`DependencyChain` embeds its `chain: Vec<usize>`. The two-method trait is
sufficient; **no trait change, no superseding ADR.**

### 2. BIC scoring; K2 rejected (resolves 3b-R1 Q2)

Edges are scored with BIC, computed in `f64` on **raw MLE counts**:

```text
score(v, Pa) = Σ_c Σ_{x∈{0,1}} N(c,x)·ln(N(c,x)/N(c))  −  (ln N / 2)·2^|Pa|
```

- `0·ln 0 := 0` (terms with `N(c,x) = 0` are skipped — no `ln(0)` path
  exists); configs with `N(c) = 0` contribute zero likelihood but still count
  toward the `2^|Pa|` penalty.
- The `½·ln(N)·2^q` complexity penalty **is** the spurious-edge guard — the
  structural analogue of MIMIC's `2/√k` significance filter, baked into the
  metric rather than bolted on (per feedback_resolve_dont_just_document).
  K2 has no built-in penalty and would need an ad-hoc minimum-gain threshold
  grafted on; rejected.
- Greedy edge addition: lexicographic `(u, v)` candidate scan, strict-`>`
  best-gain selection (first candidate wins ties → deterministic), stop when
  no strictly positive gain remains. Acyclicity by upward DFS through
  `parents[]`. A `d × d` gain cache is recomputed only for the just-modified
  child, keeping fit at `O(D²·N·κ)` — necessary because the cross-crate gate
  runs unoptimised.
- Smoothed counts are **never** used for scoring: injecting pseudo-counts into
  the likelihood would bias the score toward uniform CPTs and
  double-regularise (the penalty already guards overfitting).

### 3. `max_parents` default 3

CPT size is bounded at `2^κ`. κ = 3 suffices for the trap-5 gate (the
calibration sweep showed κ = 3 vs κ = 4 bit-identical at every failing config
— the cap never bound; at the winning config κ = 3 solves 10/10) and halves
worst-case CPT size versus κ = 4.

### 4. `smoothing_count` default 1; CPT estimation only

```text
cpt[v][c] = (N(c,1) + s) / (N(c) + 2s)
```

With `s ≥ 1` every sampling probability is strictly interior `(0, 1)` and
unseen parent configs default to `0.5`. Guard: `s = 0` with `N(c) = 0` falls
back to `init_prob`. Smoothing exists solely so ancestral sampling never
degenerates to 0/1 point masses; it plays no role in structure selection
(part 2).

### 5. Non-incremental `fit`

`prev: Option<&State>` is consumed only as the bootstrap signal (`None` ⇒
edgeless prior at `init_prob`); the network and CPTs are relearned from
scratch each generation, matching canonical BOA. The incremental seam remains
available (it is what PBIL/cGA/CMA-ES use) but is deliberately unused.

### 6. Convergence-gate configuration (empirically pinned)

The sub-spec left pop size / selection ratio / budget open. Calibration sweep
(trap-5 × 4, dim 20, 60 generations, 10 candidate seeds
`[11, 22, 33, 44, 55, 66, 77, 88, 99, 110]`, BOA `max_parents = 3`):

| pop  | ratio | BOA solved | notes |
|------|-------|-----------|-------|
| 300  | 0.5   | 0/10 | κ=3 and κ=4 bit-identical (cap never binds) |
| 600  | 0.5   | 0/10 | κ-insensitive |
| 1000 | 0.5   | 0/10 | κ-insensitive |
| 2000 | 0.5   | 0/10 | finals 1–4 |
| **2000** | **0.3** | **10/10** | **pinned gate config** |
| 4000 | 0.5   | 4/10 | finals 0–1 |
| 4000 | 0.3   | 10/10 | 2× runtime of 2000/0.3 for no gain |

Why both knobs are load-bearing (diagnosed with a per-generation structure
probe):

- **Population.** At `pop = 600` (300 selected rows) the pairwise MI between
  same-block bits in the gen-0 selected set is ≈ 0.0016 nats, so the BIC gain
  (≈ 0.5) cannot clear the penalty (≈ 2.85): the network stays near-edgeless
  while the deceptive per-gene gradient collapses mean unitation from ~10 to
  ~0.05 within nine generations, after which there is nothing left to learn.
  The gain grows ∝ N while the penalty grows ∝ ln N, so scale fixes it.
- **Selection ratio.** `0.3` truncation enriches solved-block carriers fast
  enough that intra-block edges are learned inside the early window while
  diversity still exists; at `0.5` the deceptive slope wins the race even at
  `pop = 4000`.

Baselines at the same pinned budget (pop 2000, ratio 0.3, 60 gens, symmetric
prior `init_mean = 0.5`, `init_std = 0.5`, bounds `(0, 1)`): UMDA median 3.0,
MIMIC median 3.0 — both deceived, as the gate requires. The committed test
(`crates/rlevo/tests/eda_convergence.rs::boa_solves_trap_where_umda_and_mimic_stall`)
uses seeds `[11, 22, 33, 44, 55]` and asserts BOA median `== 0.0`, UMDA and
MIMIC medians `>= 2.0`, and BOA strictly below both.

**Recorded surprise:** at this large-population config the *incremental*
binary models partially escape the trap — PBIL final costs 0–2 (median 1) and
cGA solved 9/10. Their damped probability-vector updates (and cGA's
winner/loser comparisons inside the elite set) resist the deceptive average
gradient that defeats the full-refit models. This does not affect the gate
(the spec names UMDA and MIMIC), but it falsifies the earlier session-log
assumption that a trap demo would show "PBIL/cGA failing"; the showcase
example therefore compares the full-refit trio (UMDA / MIMIC / BOA).

## Consequences

**Positive**

- The trait held: BOA needed zero changes to `ProbabilityModel<B>`,
  `EdaStrategy`, or any manifest — evidence the ADR 0017 surface generalises
  to learned-structure models.
- `ConcatenatedTrap` is the workspace's first discrete deceptive landscape;
  it is independently useful for any future linkage-learning work (hBOA,
  LTGA, EBNA per the parent spec's out-of-scope list).
- The BIC penalty replaces an ad-hoc filter with a principled one; the same
  raw-counts-vs-smoothed separation is the template for future
  count-based models.

**Negative / accepted costs**

- Host-side sequential `fit` at `O(D²·N·κ)`; with the gain cache this is
  ~12 s for the full 5-seed gate in debug. Large-D structure learning is a
  future CubeCL item (same escape hatch ADR 0017 reserved for sampling).
- The gate needs `pop = 2000` — an order of magnitude above the other EDA
  gates. Accepted: BOA's population scaling on order-5 traps is a literature
  result (Pelikan), not an implementation artifact, and the runtime stays
  inside the test budget.
- Block-aligned (tight) encoding only; a bit-permuted (loose) trap variant is
  a stronger structure-learning test and stays deferred (spec §7).

**Neutral**

- `BayesianNetworkState` is `pub` like the sibling model states; the showcase
  example reads `parents`/`order`/`cpt` to print the learned DAG.
- cGA's unexpected trap competence at large populations is recorded above and
  in the session log; no action taken.

## Alternatives considered

**K2 scoring.** Rejected (part 2): no built-in complexity penalty; would
require grafting a minimum-gain threshold — exactly the bolt-on regularisation
the MIMIC experience argues against.

**Scoring on Laplace-smoothed counts.** Rejected (part 2): biases the
likelihood toward uniform CPTs, shrinking real edges' apparent gain, and
double-regularises on top of the BIC penalty. The `0·ln 0 := 0` convention
already makes raw-count scoring total.

**A structure-fit trait hook.** Rejected (part 1): `DependencyChain` proved
learned structure belongs in `State`; a third trait method would break the
two-method surface budget ADR 0017 froze, for no expressive gain.

**Incremental network refinement via `prev`.** Rejected (part 5): canonical
BOA relearns per generation; carrying stale edges across generations couples
the model to population drift and complicates determinism reasoning. The seam
stays available if a damped-CPT variant is ever needed.

**Elitist replacement to rescue small populations.** Out of scope: Pelikan's
BOA replaces the worst half of the population, but `EdaStrategy` (frozen, ADR
0017) is full-generational. Scaling the population achieves the gate within
the frozen surface; replacement policy changes would need their own ADR.

## References

- Pelikan, Goldberg & Cantú-Paz (1999), *BOA: The Bayesian Optimization
  Algorithm.*
- Deb & Goldberg (1992), *Analyzing deception in trap functions.*
- Schwarz (1978), *Estimating the dimension of a model* (BIC).
- boa-bayesian-network-eda — governing sub-spec (type designs §3.3/§4.2).
- [0017-probability-model-trait-and-eda-strategy](0017-probability-model-trait-and-eda-strategy.md) — the frozen trait this
  ADR extends; host-RNG and `SeedPurpose::EdaSampling` conventions.
- eda-probability-model-taxonomy — 3b-R1; its OPEN BOA items are resolved
  by parts 1–2.
- `crates/rlevo-evolution/src/algorithms/eda/bayesian_network.rs` — the model.
- `crates/rlevo-environments/src/landscapes/concatenated_trap.rs` — the
  landscape.
- `crates/rlevo/tests/eda_convergence.rs` — the discriminating gate.
