---
project: rlevo
status: active
type: decision
date: 2026-06-18
tags: [evolution, cma-es, cmsa-es, strategy, probability-model, placement, phase-3]
---

# ADR 0021: CMA-ES / CMSA-ES placement and self-contained `Strategy` (not `ProbabilityModel`)

## Status

Active. Adopted 2026-06-18. Implements the deferred CMA-ES half of the phase-3
advanced-EA roadmap, the co-author partner of ADR 0017. Ships two
covariance-matrix evolution strategies — `CmaEs` (Hansen & Ostermeier 2001) and
`CmsaEs` (Beyer & Sendhoff 2008) — each a self-contained `Strategy<B>`. Purely
additive: no change to `Strategy`, `EvolutionaryHarness`, `ProbabilityModel`,
`EdaStrategy`, any existing algorithm file, or any manifest; no new dependency.
**Does not supersede ADR 0017** — it *refines* the forward-looking expectation in
ADR 0017 §Decision part 9 (see Decision part 2 below). Tracks issue #59.

## Context

CMA-ES is the canonical black-box optimiser for continuous, non-separable,
ill-conditioned landscapes; CMSA-ES is its evolution-path-free sibling. Two
design questions need an immutable record before code:

1. **Placement.** Issue #59 says add `pub mod cma_es;` / `pub mod cmsa_es;`
   "alongside the existing entries" in `algorithms/mod.rs` (flat). The research
   note `eda-vs-cma-es-boundary` (3b-R2, a *draft* note — never an ADR) tentatively
   proposed an `es_advanced/` submodule "because evolution paths are the
   identity-defining machinery." ADR 0017 §9 deferred placement to "elsewhere
   (`algorithms/cma_es.rs`)". The two readings (flat vs submodule) must be settled.

2. **Trait reuse.** ADR 0017 was co-designed so "CMA-ES can reuse
   [`ProbabilityModel<B>`] later without a signature break," and its §Decision
   part 9 states CMA-ES "will live elsewhere … but **reuse `ProbabilityModel`**:
   its multivariate Gaussian implements the trait, and the evolution-path /
   step-size machinery is layered into CMA-ES's own `tell`." That was an
   *anticipation* recorded before the CMA-ES design pass; this ADR is that design
   pass and must either confirm or refine it.

Three live-API facts constrain the design (the same frozen surface ADR 0016/0017
verified):

- **`Strategy::tell` takes `&self`**, no interior mutability — instances
  parallelise without locks. CMA-ES/CMSA-ES carry all adaptive state (`C`, σ,
  paths) in `State`, so this holds trivially.
- **All randomness is host-side via `seed_stream(base, generation, SeedPurpose)`**
  (`crates/rlevo-evolution/src/rng.rs`); `B::seed` / `Tensor::random` forbidden.
  ADR 0017 raised the max purpose tag to `EdaSampling = 8`.
- **Burn 0.21 ships no Cholesky or symmetric eigendecomposition**, and the
  workspace pulls no `nalgebra` (verified in research note
  `cma-es-sampling-and-numerics`). The decomposition must be host-side hand-rolled
  or a new dependency.

## Decision

**Ship `CmaEs` and `CmsaEs` as two flat, self-contained `Strategy<B>`
implementations in `crates/rlevo-evolution/src/algorithms/cma_es.rs` and
`cmsa_es.rs`. They do *not* instantiate `ProbabilityModel<B>`; the trait remains
available but deliberately unused for covariance-matrix ES. All changes are
additive.**

### 1. Placement — flat `algorithms/{cma_es,cmsa_es}.rs`, not `es_advanced/`

- Honours issue #59's explicit instruction.
- Matches the tree's *actual* convention: submodules (`eda/`, `gep/`,
  `metaheuristic/`, `neuroevolution/`) exist only for **multi-file** families;
  single-file families are flat (`es_classical.rs`, `de.rs`, `ep.rs`,
  `gp_cgp.rs`). Two sibling files have the same cardinality as
  `ga.rs` + `ga_binary.rs`, which are flat siblings — not a `ga/` directory.
- The `es_advanced/` proposal in `eda-vs-cma-es-boundary` was motivated by
  *intellectual* separation from `eda/`, not by file count. That separation is
  real and is carried by **module rustdoc** in `cma_es.rs` citing the EDA/ES
  boundary, not by a directory that would currently hold exactly two files.
- **This resolves the boundary note's open module-layout question.** The note is a
  draft research artefact; on acceptance of this ADR its `es_advanced/` lines are
  struck as superseded.
- If a third+ covariance-ES variant lands later (Separable-CMA, Cholesky-CMA,
  LM-CMA — all deferred), promoting the pair into `es_advanced/` becomes a
  justified mechanical move at *that* point.

### 2. Self-contained `Strategy<B>` — `ProbabilityModel<B>` available but unused

CMA-ES and CMSA-ES implement `Strategy<B>` directly and do **not** route through
`ProbabilityModel<B>`. This **refines** the anticipation in ADR 0017 §9.

- **The fit→sample shape does not fit covariance-matrix ES.**
  `ProbabilityModel::fit(population, fitness, prev) → State` /
  `sample(state) → population` is a pure density-estimation seam. CMA-ES's
  *identity* is the machinery that is **not** density estimation: cumulative
  step-size adaptation (CSA), the two evolution paths `p_σ`/`p_c`, and the
  *learning-rate-blended* rank-1 + rank-μ covariance update. CMSA-ES's identity is
  per-individual log-normal σ self-adaptation. Forcing either through `fit`/`sample`
  would (a) leak ES strategy state (`σ`, paths, per-individual σ_i) into the model
  `State`, and (b) split one algorithm awkwardly across `fit` and a `tell` overlay
  for no expressive gain.
- **The boundary note's own heuristic points here:** "EDA uses the model *only* to
  sample; CMA-ES layers paths on top" — i.e. the right seam is the *`Strategy`*,
  not the *model*.
- **ADR 0017's reuse goal is satisfied in its correct sense.** The EDA spec's open
  acceptance criterion 3 ("CMA-ES spec demonstrates trait reuse — no *divergent*
  parallel implementation") holds: there is **no divergent `ProbabilityModel`**,
  the trait was genuinely co-designed with CMA-ES in view, and the
  `prev: Option<&State>` incremental seam (ADR 0017 part 1) **exists and remains
  available** for path/rank-μ updates. What this ADR declines is the stronger
  reading "CMA-ES must *instantiate* the trait." This mirrors the
  **"available but deliberately unused"** precedent ADR 0018 set for the same seam.
- **Clean future bridge (out of scope).** A `MultivariateGaussian` model (≈ EMNA,
  the pure-MLE full-covariance EDA) *could* implement `ProbabilityModel<B>` in
  `eda/` and would genuinely exercise the trait at the multivariate-Gaussian end of
  the continuum — demonstrating reuse in the strong form **without** contorting
  CMA-ES. Noted as the clean way to close criterion 3 strongly; not built here.

### 3. Host-side numerics, hand-rolled Jacobi — no new dependency

Resolved in research note `cma-es-sampling-and-numerics` (§L4); recorded here for
the additive/no-dep claim. Decomposition + sampling are host-side `Vec<f32>`; the
eigensolver is a hand-rolled cyclic **Jacobi** routine (it ports the canonical
`pycma` `eigen()` for bit-validation, needs no dependency, and has better
small-eigenvalue accuracy than QR — which matters for the narrow-axis-collapse
failure mode). `nalgebra` rejected for v1 (new dep, a logged 4×4 correctness bug
dimforge/nalgebra #1109, strong perf only via non-portable `nalgebra-lapack`,
negligible gain at D≤30). CMSA-ES needs only a Cholesky factor (no `C^{-1/2}`);
CMA-ES reuses the Jacobi eigendecomposition for both sampling and `C^{-1/2}`. The
device→host fitness sync per generation is the only GPU-specific cost and is
bounded by λ — cheap on both PCIe and unified-memory backends.

### 4. `SeedPurpose::CmaSampling = 11` — one additive variant

Multivariate sampling (and CMSA-ES's per-individual log-normal σ draw) draw from
`seed_stream(base, generation, SeedPurpose::CmaSampling)`, disjoint from every
other operator stream. Additive; the only exhaustive match is `constant()` in
`rng.rs`, extended by one arm, plus one assertion in the purpose-stream test —
continuing the numbering ADR 0017 left at `EdaSampling = 8` (`Representative = 9`,
`Transposition = 10` shipped since).

### 5. CMSA-ES reuses the `es_classical` σ rule

CMSA-ES's log-normal self-adaptation uses `τ = 1/√(2D)`, the same rule and
default as `EsConfig::default_for` (Beyer & Schwefel 2002). The two σ-adaptation
families (CSA in CMA-ES, self-adaptation in CMSA-ES/`es_classical`) thus share one
formula across the crate.

## Consequences

**Positive**

- **Zero blast radius on the frozen surface.** `Strategy`, `EvolutionaryHarness`,
  `ProbabilityModel`, `EdaStrategy`, every existing algorithm file, and every
  `Cargo.toml` are byte-unchanged. `CmaEs`/`CmsaEs` are *just more `Strategy`s*;
  they compose with the harness, benchmark suites, and record/report plumbing for
  free.
- **No new dependency.** Hand-rolled Jacobi keeps the dep graph and the
  backend-agnostic / wasm posture intact; the "check built-ins before
  hand-rolling" rule is satisfied with a documented *reject* (Burn has none).
- **Each algorithm reads as one unit.** Keeping CSA + paths + covariance in one
  `Strategy::tell` (rather than fractured across a model `fit` and an overlay)
  matches the reference literature line-for-line and eases validation against
  `pycma`.
- **The trait stays honestly general.** Declining to force CMA-ES through it
  *confirms* the boundary (ADR 0017 part 9 / the research note) rather than blurs
  it; the `prev` seam remains the real reuse point for a future EMNA model.

**Negative / accepted costs**

- **ADR 0017 §9's prose over-promised.** It said CMA-ES "will reuse
  `ProbabilityModel`"; this ADR declines instantiation. Accepted and explicitly
  recorded as a *refinement* (the seam remains available-but-unused, ADR 0018
  precedent), not a silent reversal. ADR 0017 is immutable and unchanged.
- **EDA-spec criterion 3 closes only in the weak sense.** "No divergent
  implementation" holds; "CMA-ES instantiates the trait" does not. Strong-form
  closure waits on the optional `MultivariateGaussian`/EMNA model.
- **Host-side decomposition forces one device→host sync per generation** on GPU
  backends. Accepted: payload is λ floats (bytes), dominated by the host O(D³)
  decomposition, and free on the `ndarray` backend the convergence tests use.

**Neutral**

- `SeedPurpose` gains one variant (`CmaSampling = 11`); the purpose-stream test is
  extended by one arm.
- Tensorised / on-device covariance path, Separable/Cholesky/LM-CMA variants, and
  restart strategies (BIPOP/IPOP) remain deferred to a future GPU/advanced spec;
  promoting the pair into `es_advanced/` is the natural trigger.

## Component diagram

```
        EvolutionaryHarness<B, CmaEs<B>, F>      EvolutionaryHarness<B, CmsaEs<B>, F>
        (CMA-unaware; ask ─► evaluate_batch ─► tell)
                          │
                          ▼
   rlevo-evolution
   ┌──────────────────────────────────────────────────────────────────────┐
   │  algorithms/cma_es.rs        CmaEs  IS-A Strategy<B>   (self-contained)│
   │    ask:  eig(C)=BΛBᵀ; z~N(0,I) [CmaSampling]; x = m + σ·B√Λ z          │
   │    tell: ⟨y⟩_w → p_σ (CSA) → σ → p_c → C' (rank-1 + rank-μ)            │
   │                                                                        │
   │  algorithms/cmsa_es.rs       CmsaEs IS-A Strategy<B>   (self-contained)│
   │    ask:  chol(C)=A; σ_l = σ̄·exp(τξ_l); x_l = m + σ_l·A z_l            │
   │    tell: m' = Σwᵢxᵢ; σ̄' = Σwᵢσᵢ; C' = (1−1/τc)C + (1/τc)Σwᵢ sᵢsᵢᵀ    │
   │                                                                        │
   │  probability_model.rs  trait ProbabilityModel<B> { fit(.., prev); .. } │
   │     └─ AVAILABLE BUT UNUSED by CMA-ES/CMSA-ES (ADR 0018 precedent);    │
   │        prev: Option<&State> seam reserved for a future EMNA model.     │
   │  algorithms/eda/   EdaStrategy<B, M>   (unchanged — the trait's users) │
   └──────────────────────────────────────────────────────────────────────┘

   Frozen / unchanged: Strategy, EvolutionaryHarness, ProbabilityModel, EdaStrategy,
                       all existing algos, every Cargo.toml.
   New: src/algorithms/{cma_es,cmsa_es}.rs, hand-rolled host Jacobi/Cholesky,
        SeedPurpose::CmaSampling = 11.
```

## Alternatives considered

**`es_advanced/{cma_es,cmsa_es}.rs` submodule (the boundary-note proposal).**
Rejected for v1: the tree reserves submodules for multi-file families; two files
is a flat-sibling cardinality (cf. `ga.rs`/`ga_binary.rs`). The intellectual
separation is carried by module rustdoc. Revisit on the third covariance-ES
variant.

**CMA-ES instantiates `ProbabilityModel<B>` (a `MultivariateGaussian` model) with
a `tell` overlay (ADR 0017 §9's letter).** Rejected: CSA, evolution paths, and the
learning-rate-blended covariance update are not density estimation; threading them
through `fit`/`sample` leaks ES state into the model and fractures one algorithm
across two seams. The `Strategy` is the correct seam (the boundary note's own
heuristic). The trait stays available for a genuine EMNA model instead.

**A new dependency (`nalgebra` / `nalgebra-lapack`) for the eigensolver.**
Rejected for v1: new dep, a logged small-matrix correctness bug (#1109), strong
perf only via non-portable LAPACK bindings, negligible gain at D≤30. Hand-rolled
Jacobi ports the `pycma` reference and adds nothing to the dep graph.

**Reuse `SeedPurpose::EdaSampling` or `Other` for the multivariate draw.**
Rejected: `EdaSampling` is semantically the EDA model-sampling stream and `Other`
is a catch-all; a dedicated `CmaSampling` keeps the covariance-ES draw disjoint and
self-documenting, matching the per-purpose isolation rationale in `rng.rs`.

## References

- Issue #59 — `feat(evo): implement CMA-ES and CMSA-ES`; specifies flat
  `pub mod cma_es; pub mod cmsa_es;` and the host-RNG/determinism constraints.
- Spec `covariance-matrix-evolution-strategies` (advanced-evo-algos) — OQ-1/OQ-2
  resolved here.
- Research note `cma-es-module-placement` (cma-R2) — placement + trait-reuse.
- Research note `cma-es-sampling-and-numerics` (cma-R1) — Jacobi vs `nalgebra`,
  host/GPU cost analysis, `CmaSampling` proposal.
- Research note `eda-vs-cma-es-boundary` (3b-R2) — the `es_advanced/` proposal this
  ADR resolves; the "EDA samples only / CMA-ES layers paths" heuristic.
- [0017-probability-model-trait-and-eda-strategy](0017-probability-model-trait-and-eda-strategy.md)
  — co-author partner; §Decision part 1 (the `prev: Option<&State>` seam) and
  part 9 (the CMA-ES-reuse anticipation this ADR refines).
- [0018-boa-bayesian-network-and-concatenated-trap](0018-boa-bayesian-network-and-concatenated-trap.md)
  — the "available but deliberately unused" seam precedent.
- [0016-memetic-wrapper-and-local-search-seam](0016-memetic-wrapper-and-local-search-seam.md)
  — additive-seam + `SeedPurpose` numbering lineage.
- Hansen (2016), *The CMA Evolution Strategy: A Tutorial* (arXiv:1604.00772);
  Beyer & Sendhoff (2008), *Covariance matrix adaptation revisited — CMSA-ES*.
- `crates/rlevo-evolution/src/rng.rs` — `seed_stream` / `SeedPurpose` (gains
  `CmaSampling = 11`); the `project_evolution_host_rng_convention` host-RNG rule.
