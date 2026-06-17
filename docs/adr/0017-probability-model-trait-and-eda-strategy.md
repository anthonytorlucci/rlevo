---
project: rlevo
status: active
type: decision
date: 2026-06-12
tags: [evolution, eda, probability-model, phase-3b]
---

# ADR 0017: `ProbabilityModel` trait and `EdaStrategy` seam (phase 3b)

## Status

Active. Adopted 2026-06-12. Implements phase 3b (issue #31) of the advanced-EA
roadmap: estimation-of-distribution algorithms. Ships the `ProbabilityModel<B>`
trait and a generic `EdaStrategy<B, M>` over the frozen `Strategy<B>` ask/tell
contract, plus four concrete univariate / chain models. BOA (Bayesian
optimization algorithm) is deferred to a follow-up issue. Purely additive: no
change to `Strategy`, `EvolutionaryHarness`, any existing algorithm file, or any
manifest; no new dependency. Does not supersede a prior ADR; extends ADR 0016's
phase-3 pattern (additive seams on `Strategy`, host-RNG sampling) and reuses the
NaN chokepoint shipped with it.

## Context

An EDA replaces the crossover/mutation operators of a classical EA with an
explicit *probability model*: each generation fits a distribution to the
selected (fittest) individuals, then samples the next population from it. The
phase-3b goal is to make that fit→sample loop a first-class `Strategy<B>` so EDAs
run through the existing harness and benchmark plumbing unmodified, and to ship a
trait general enough that CMA-ES can reuse it later without a signature break.

Three facts about the live API constrain the design (same frozen surface ADR
0016 verified):

1. **`Strategy::tell` takes `&self`** (`crates/rlevo-evolution/src/strategy.rs`)
   and the trait is documented as free of interior mutability so instances
   parallelise without locks. Unlike the memetic wrapper, an EDA needs no `&mut F`
   inside `tell` — the model fit consumes the *already-evaluated* population
   tensor the harness hands in, so `EdaStrategy` stays genuinely pure and
   lock-free.
2. **All EA randomness is host-side via `seed_stream(base, generation,
   SeedPurpose)`** (`crates/rlevo-evolution/src/rng.rs`); `B::seed` /
   `Tensor::random` / `thread_rng` are forbidden (process-global RNG mutex races
   parallel tests). ADR 0016 raised the max purpose tag to `LocalSearch = 7`.
3. **`Strategy<B>` already threads `&mut dyn rand::Rng`** through ask/tell. The
   phase-3b spec (§4) sketched a placeholder `ProbabilityModel` over `RngCore`
   and a stateless `fit`; issue #31 §1 overrides both. This ADR records the final
   trait against the live convention.

Issue #31 constraint **C9.3** ("must not add a `rlevo-core` dep to
`rlevo-evolution`") has a stale premise: `rlevo-core` is *already* a prod dep of
`rlevo-evolution` on `main` (re-added by ADR 0004 for the `BenchEnv` trait
surface). This change adds nothing to the dep graph — no manifest edit anywhere.

## Decision

**Ship phase 3b as a `ProbabilityModel<B>` trait in
`rlevo-evolution/src/probability_model.rs` (re-exported at crate root) and a
generic `EdaStrategy<B, M: ProbabilityModel<B>>` in
`rlevo_evolution::algorithms::eda` that implements the frozen `Strategy<B>`.
Sampling is host-side only via `rand_distr`; Burn's PRNG kernels are explicitly
rejected. Four concrete models ship in v1; BOA is deferred. All changes are
additive.**

### The trait (final — supersedes the spec §4 placeholder)

```rust
pub trait ProbabilityModel<B: Backend>: Send + Sync {
    type Params: Clone + Debug + Send + Sync;
    type State:  Clone + Debug + Send + Sync;

    /// `prev = None` ⇒ build prior purely from `params`
    /// (population/fitness ignored; passed as 0×0 / 0-length tensors).
    fn fit(&self, params: &Self::Params, prev: Option<&Self::State>,
           population: Tensor<B, 2>, fitness: Tensor<B, 1>,
           device: &<B as burn::tensor::backend::BackendTypes>::Device) -> Self::State;

    fn sample(&self, state: &Self::State, n: usize, rng: &mut dyn Rng,
              device: &<B as burn::tensor::backend::BackendTypes>::Device) -> Tensor<B, 2>;
}
```

### Concrete parts

1. **`prev: Option<&Self::State>` in `fit` — the incremental seam.** The spec
   placeholder was stateless. PBIL and cGA are *incremental*: each generation
   interpolates the probability vector toward the selected sample, never
   overwriting it, so `fit` must read the prior state. Rather than add a third
   `init_state` method (the two-method cap is a deliberate trait-surface budget),
   `prev` is `Option`: `None` is the bootstrap path `EdaStrategy::init` calls
   (population/fitness arrive as 0×0 / 0-length tensors and the model builds its
   prior purely from `params`); `Some` is the per-generation incremental path.
   This is the *same seam CMA-ES needs* for its rank-μ and evolution-path updates
   — the prior covariance/path lives in `State` and is read back each `fit`. It
   resolves umbrella-spec §9 Q1 ("does fitness belong in `fit`?") affirmatively:
   `fitness` is a parameter; univariate models that do not weight by it ignore it
   with `let _ = fitness;`.

2. **`rng: &mut dyn rand::Rng`, not `RngCore` (issue #31 §1 over spec §4).**
   `Rng` is the dyn-safe core trait in rand 0.10 and is exactly what `Strategy<B>`
   already threads through ask/tell, so the model RNG and the strategy RNG are the
   same type. No `RngCore`-vs-`Rng` impedance at the `EdaStrategy::sample` call
   boundary.

3. **`genome_dim` lives in each model's `Params`, never in `EdaParams`.**
   `EdaParams` carries exactly `pop_size`, `selection_ratio`,
   `bounds: Option<(f32, f32)>`, and the wrapped `M::Params`. The genome
   dimension is a *model* concern (a Gaussian's mean length, a Bernoulli vector's
   length), and the `prev = None` bootstrap means `EdaStrategy` itself never needs
   the dimension to produce the first population — the model's prior knows its own
   shape. This keeps `EdaParams` model-agnostic.

4. **Binary models emit raw {0, 1} genes; `bounds` clamp is a documented no-op
   for them.** PBIL and cGA sample Bernoulli genes into a `[n, D]` tensor of 0.0 /
   1.0. `EdaParams.bounds` (a real-genome convenience for the Gaussian) is applied
   uniformly but is a no-op on already-in-range binary genes; this is documented,
   not special-cased. On the Sphere landscape the all-zeros genome scores exactly
   `0.0`, so the convergence gate stays meaningful for the binary models' smoke
   tests.

5. **Host-RNG sampling only; Burn PRNG kernels explicitly rejected.** Burn
   0.21's `burn-cubecl` PRNG kernels (`random_bernoulli` / `random_normal` /
   `random_uniform`) take **no seed or stream argument**; their only seeding path
   is the process-global `Backend::seed()` → `cubek::random::seed()`
   (`burn-cubecl/src/backend.rs:61-62`). That is incompatible with the per-purpose
   `seed_stream` determinism scheme (rules.md §8), races parallel tests, and is
   CubeCL-only. `burn-backend::Distribution::sampler` (host-side) *is*
   convention-compliant but is a thin wrapper over the same `rand_distr`
   primitives; this ADR samples `rand_distr` directly, matching the
   `ops/mutation.rs` precedent. **Future work:** large-N GPU sampling via a custom
   CubeCL kernel that takes an explicit seed argument (the Firefly / Lévy kernel
   pattern under the `custom-kernels` feature) — out of scope for phase 3b.

6. **New `SeedPurpose::EdaSampling` variant (discriminant 8).** Model sampling
   draws from `seed_stream(base, generation, SeedPurpose::EdaSampling)`, a stream
   disjoint from every other operator's. Additive; the only exhaustive match is
   `constant()` in `rng.rs`, extended by one arm.

7. **NaN chokepoint reuse.** `EdaStrategy::tell` sanitizes host fitness through
   the existing `crate::local_search::sanitize_fitness` (NaN → +inf) before
   best-tracking and truncation selection. No new sanitizer; reuses the chokepoint
   ADR 0016 shipped.

8. **Selection is deterministic truncation.** Take the top
   `⌈selection_ratio · pop_size⌉` rows (minimum 2) by the `(fitness, index)` total
   order. Selected rows reach `fit` in *ascending-fitness* order; this is
   documented, and models must not rely on it beyond reproducibility (a future
   fitness-weighted model may consume the order, but the ordering guarantee exists
   only to pin determinism, not as a ranking contract).

9. **EDA vs CMA-ES boundary (research note `eda-vs-cma-es-boundary`).** Pure
   fit-then-sample algorithms — no per-generation path/step-size adaptation
   outside the model — live in `algorithms/eda/`. CMA-ES will live elsewhere
   (`algorithms/cma_es.rs`) but **reuse `ProbabilityModel`**: its multivariate
   Gaussian implements the trait, and the evolution-path / step-size machinery is
   layered into CMA-ES's own `tell` around the model rather than baked into
   `EdaStrategy`. The `prev = Some` seam (part 1) is precisely what lets the path
   state survive between fits.

### Module layout

`src/probability_model.rs` holds the trait (re-exported at crate root).
`src/algorithms/eda/mod.rs` holds `EdaStrategy`, `EdaParams`, `EdaState`, and the
`pub mod` + re-export declarations. The four models sit one-per-file under
`src/algorithms/eda/`:

| File                    | Type                 | Algorithm |
| ----------------------- | -------------------- | --------- |
| `univariate_gaussian.rs`| `UnivariateGaussian` | UMDA      |
| `univariate_bernoulli.rs`| `UnivariateBernoulli`| PBIL     |
| `compact_genetic.rs`    | `CompactGenetic`     | cGA       |
| `dependency_chain.rs`   | `DependencyChain`    | MIMIC     |

## Consequences

**Positive**

- **Zero blast radius on the frozen surface.** `Strategy`,
  `EvolutionaryHarness`, every existing algorithm file, and every `Cargo.toml`
  are byte-unchanged. `EdaStrategy` is *just another `Strategy`*; it composes with
  the harness, the benchmark suites, and the record/report plumbing for free.
- **`EdaStrategy` stays pure and lock-free.** Unlike the memetic wrapper (ADR
  0016, which needed a `Mutex<F>`), the EDA consumes the already-evaluated
  population the harness hands `tell`; no `&mut F` is needed, so the
  `Strategy: Send + Sync` purity invariant holds without interior mutability.
- **The trait is CMA-ES-ready.** The `prev: Option<&State>` seam carries
  exactly the prior-state thread CMA-ES needs; adding CMA-ES later is a new model
  + a thin `tell` overlay, not a trait change.
- **Determinism is structural.** Host-RNG sampling under a dedicated
  `EdaSampling` purpose, deterministic truncation selection, and the reused NaN
  chokepoint mean runs reproduce bit-for-bit under the rules.md §8 scheme.

**Negative / accepted costs**

- **`bounds` is a partial-truth knob.** `EdaParams.bounds` clamps real genomes
  but is a documented no-op for the binary models — one parameter with
  model-dependent meaning. Accepted over splitting `EdaParams` per genome kind.
- **Host-side sampling only (no GPU PRNG).** Sampling round-trips through host
  `rand_distr`; for large N this is slower than an on-device kernel. Accepted:
  Burn's seedless kernels cannot satisfy the determinism scheme, and the
  custom-kernel escape hatch is reserved for future work.
- **Selection order is a guarantee callers might over-read.** The
  ascending-fitness arrival order exists only to pin determinism; documenting it
  risks a model accidentally depending on it as a ranking. Mitigated by the
  explicit "must not rely on beyond determinism" note.

**Neutral**

- `SeedPurpose` gains one variant (`EdaSampling = 8`); the purpose-stream test is
  extended by one arm.
- BOA is deferred to a follow-up issue; the trait is general enough to host it
  (a Bayesian network is a `State`, structure-learning is `fit`) without a
  signature change.

## Component diagram

```
                  EvolutionaryHarness<B, EdaStrategy<B, M>, F>
                  (EDA-unaware; ask ──► evaluate_batch ──► tell)
                                    │
                                    ▼
   rlevo-evolution
   ┌──────────────────────────────────────────────────────────────────────┐
   │  probability_model.rs                                                  │
   │    trait ProbabilityModel<B> { type Params; type State;               │
   │                                fn fit(.., prev: Option<&State>, ..);   │
   │                                fn sample(..); }                        │
   │                         ▲                                              │
   │                         │ impls                                        │
   │  algorithms/eda/        │                                              │
   │  ┌──────────────────────┴───────────────────────────────────────────┐ │
   │  │ mod.rs:  EdaStrategy<B, M: ProbabilityModel<B>>  IS-A Strategy<B> │ │
   │  │            init:  M::fit(params, prev = None, 0×0, ..) ──► sample │ │
   │  │            tell:  sanitize_fitness → truncation select →          │ │
   │  │                   M::fit(params, prev = Some(state), pop, fit) →  │ │
   │  │                   M::sample(.., EdaSampling stream)               │ │
   │  │                                                                   │ │
   │  │  univariate_gaussian.rs   UnivariateGaussian  (UMDA)             │ │
   │  │  univariate_bernoulli.rs  UnivariateBernoulli (PBIL, incremental)│ │
   │  │  compact_genetic.rs       CompactGenetic      (cGA,  incremental)│ │
   │  │  dependency_chain.rs      DependencyChain     (MIMIC)            │ │
   │  └──────────────────────────────────────────────────────────────────┘ │
   │                                                                        │
   │  algorithms/cma_es.rs   (future)                                       │
   │    CMA-ES reuses ProbabilityModel (multivariate Gaussian) and layers  │
   │    evolution-path / step-size adaptation in its OWN tell, reading      │
   │    prior state via the prev = Some(..) seam.                           │
   └──────────────────────────────────────────────────────────────────────┘

   Frozen / unchanged: Strategy trait, EvolutionaryHarness, DE/GA/memetic/all algos,
                       every Cargo.toml.
   New: src/probability_model.rs, src/algorithms/eda/ (mod + 4 models),
        SeedPurpose::EdaSampling = 8.
```

## Alternatives considered

**Stateless `fit` (the spec §4 placeholder).** Rejected: PBIL and cGA are
incremental and cannot be expressed without reading the prior probability vector,
and CMA-ES's path machinery needs the same prior thread. A stateless signature
would force a third `init_state` method or an out-of-band state channel.

**A third `init_state` method instead of `prev: Option<&State>`.** Rejected: it
widens the trait surface past the two-method cap for no expressive gain — the
`None` arm of `fit` already *is* the bootstrap, and folding it into one method
keeps the bootstrap and incremental paths from drifting.

**Burn `Distribution::sampler` or `burn-cubecl` PRNG kernels for sampling.**
Rejected: the CubeCL kernels are seedless (process-global `Backend::seed()` only),
incompatible with the per-purpose `seed_stream` scheme, and CubeCL-only;
`Distribution::sampler` is convention-compliant but a thin wrapper over the same
`rand_distr` primitives this ADR calls directly (matching `ops/mutation.rs`).

**`genome_dim` in `EdaParams`.** Rejected: the dimension is a model concern, and
the `prev = None` bootstrap means `EdaStrategy` never needs it — keeping it out of
`EdaParams` leaves that struct model-agnostic.

**Ship CMA-ES inside `algorithms/eda/`.** Rejected: CMA-ES is not pure
fit-then-sample (it adapts step size and an evolution path per generation). The
research note draws the boundary at "pure fit→sample lives in `eda/`"; CMA-ES
reuses the trait but lives elsewhere with its own `tell` overlay.

## References

- Issue #31 — phase-3b EDA spec; §1 overrides the spec §4 trait signature
  (`Rng` over `RngCore`); constraint C9.3 has a stale premise (see Context).
- Umbrella spec §4 (placeholder `ProbabilityModel`) — superseded by the trait in
  this ADR; §9 Q1 (fitness in `fit`?) resolved affirmatively.
- Research note `eda-vs-cma-es-boundary` — the `eda/` vs CMA-ES placement
  resolution adopted in part 9.
- [0016-memetic-wrapper-and-local-search-seam](0016-memetic-wrapper-and-local-search-seam.md) — phase-3a; this ADR extends its
  additive-seam-on-`Strategy` pattern, reuses its `sanitize_fitness` NaN
  chokepoint, and continues the `SeedPurpose` numbering (`LocalSearch = 7` →
  `EdaSampling = 8`).
- [0004-move-bench-traits-into-rlevo-core](0004-move-bench-traits-into-rlevo-core.md) — re-added the
  `rlevo-evolution → rlevo-core` edge for `BenchEnv`; the reason C9.3's premise is
  stale (the dep already exists on `main`).
- project_evolution_host_rng_convention — the host-RNG / `seed_stream`
  convention the sampling obeys; `B::seed` / `Tensor::random` / `thread_rng`
  forbidden.
- `crates/rlevo-evolution/src/strategy.rs` — `Strategy::tell(&self)` (frozen),
  the `&mut dyn Rng` thread `EdaStrategy::sample` matches.
- `crates/rlevo-evolution/src/rng.rs` — `seed_stream` / `SeedPurpose` (gains
  `EdaSampling = 8`).
- `crates/rlevo-evolution/src/ops/mutation.rs` — the direct-`rand_distr`
  host-sampling precedent.
- `burn-cubecl/src/backend.rs:61-62` — the seedless `Backend::seed()` →
  `cubek::random::seed()` path that rules out Burn's PRNG kernels.
