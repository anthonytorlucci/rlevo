---
project: rlevo
status: active
type: decision
date: 2026-07-12
tags: [adr, decision, environments, landscapes, benchmarks, bounds, search-space, issue-113]
---

# ADR 0045: Landscape `bounds()` is a search box with two obligations, not a domain

## Status

**Accepted (2026-07-12).** Resolves issue #113 ("[env] Landscapes: incorrect
`bounds()` for asymmetric-domain functions"). Supersedes nothing. Builds on ADR
[0027](0027-bounds-newtype-for-closed-ranges.md) — the single scalar `Bounds`
every EA config holds is precisely what makes a per-axis rectangle
unrepresentable, and therefore what forces this decision.

## Context

`Branin::bounds()` returned `(-5.0, 10.0)`. Branin's published domain is
**asymmetric** — `x₁ ∈ [-5, 10]`, `x₂ ∈ [0, 15]` — and the returned pair was the
`x₁` range. But a `bounds()` pair is applied to **every** coordinate, so the
effective search box was `[-5, 10]²`, which **excludes the certified global
optimum `(−π, 12.275)`** (`x₂ = 12.275 > 10`). One of Branin's three global
minima was outside the search space: unreachable by any optimiser, at any budget,
forever.

`Trefethen::bounds()` had the identical defect in the other direction — it
returned the `x₂` range `(-4.5, 4.5)`, clipping `x₁ ∈ [-6.5, 6.5]`.

`Bukin6` had independently got it **right**: its domain is `x₁ ∈ [-15, -5]`,
`x₂ ∈ [-3, 3]`, and it returned the square hull `(-15.0, 3.0)`, with in-source
tests asserting the box reaches the optimum and admits nothing better. But its
convention was never written down anywhere — not in `rules.md`, not in an ADR,
not in the `Landscape` trait docs. So two of its siblings diverged from it and
nothing caught them.

Two facts explain why the bug was invisible for as long as it was:

1. **The convention was unrecorded.** Three landscapes with asymmetric domains,
   three independent judgment calls, one of them correct. There was no rule to
   cite in review.
2. **No test asserted the box could reach the optima.** Every landscape had
   value tests (`evaluate(optimum) ≈ f*`) — evaluating *at* a point says nothing
   about whether a search can *reach* it. Bukin6 had the reachability test; the
   other two did not.

### Why a per-axis box is not an option today

`bounds() -> (f64, f64)` is an **inherent** method on all 23 landscape structs;
it is deliberately **not** on the `rlevo_core::fitness::Landscape` trait (which
carries only `evaluate` + `sense`). Its consumers cannot use anything richer than
a scalar pair:

- Every EA config in `rlevo-evolution` — `GaConfig`, `EsConfig`, `DeConfig`,
  `PsoConfig`, the metaheuristics, the local searchers, the coevolution halves:
  ~20 of them, 52 fields — holds a **single** `rlevo_core::bounds::Bounds`
  (ADR 0027), applied to every coordinate through
  `clamp_vec(&mut [f32], Bounds)`.
- The showcase harness (`crates/rlevo/examples/evo/common.rs`) takes one
  `bounds: (f64, f64)` and fans it out to every algorithm's single `bounds`
  field.

A per-axis search **rectangle** is therefore not representable anywhere in the
engine, and making one representable is out of scope for this ADR (see Open
questions).

## Decision

### 1. `bounds()` returns the recommended per-coordinate *search box*

Not the function's mathematical domain. The name is retained; the **meaning is
pinned**, in the rustdoc of every landscape and in this record.

It is the search box because the search box is the only thing its consumers can
consume. The true per-axis domain of an asymmetric function is real, and it is
**documentation only**: a module-level `# Domain` section with its literature
citation. `bounds()` is what a search actually runs inside.

### 2. Two obligations, each pinned by a test

A conforming `bounds()` satisfies both:

- **O1 — reachability.** The box contains every certified global optimum, on
  **every** coordinate. This is the obligation Branin and Trefethen violated.
- **O2 — no spurious optimum.** The box contains **no point with `f < f*`**.
  This is the obligation that makes widening safe — and it must be **discharged
  analytically**, in a code comment on `bounds()`, not assumed.

Every landscape with a certified optimum carries an in-source unit test for O1.
O2 is regression-tested by a deterministic dense grid sweep over the box
asserting no sample beats `f*`; the sweep is a guard against a later edit, not
the proof — the proof is the comment.

### 3. Two conforming shapes, neither an exception to the other

**Square hull of an asymmetric domain.** Branin `(-5, 15)` (hull of `[-5,10] ×
[0,15]`), Trefethen `(-6.5, 6.5)`, Bukin6 `(-15, 3)`. A hull **does** admit
points outside the published rectangle — Bukin6's hull admits `x₁ = +2`, well
outside its `[-15, -5]` — and that is exactly what **O2 certifies as harmless**:
the extra area contains nothing better than `f*`, so it costs search effort, not
correctness.

**Cited reduced range of an impractically large canonical domain.**
RosenbrockFlat `(-30, 30)`, reduced from the canonical `[-2000, 2000]ⁿ`
(Monismith 2010) after Chen (1997); Six-Hump Camel `(-2, 2)`, reduced from the
canonical `[-5, 5]²` (Al-Roomi) to the window that actually shows the six-hump
structure. Same two obligations, same citation requirement.

These are **two legitimate shapes of the same rule**, not a rule and an
exception. A landscape whose domain is already a symmetric box (Sphere, Ackley,
Rastrigin, …) satisfies both obligations trivially and needs no comment beyond
the citation.

### 4. No `domain()` accessor and no per-axis renderer ship here

Both were tempting; both are deliberately deferred, and the reasons are worth
recording because they will be re-proposed.

- **The renderer no longer clips anything.** After the O1 fix, every landscape's
  square box **contains** its full true domain. A per-axis renderer would trim
  excess area off a heatmap — cosmetic, not correctness.
- **It would introduce a second, colliding rectangle convention.**
  `rlevo-core/src/render/payload.rs` documents `Landscape2DSnapshot`'s
  `bounds_x` / `bounds_y` as the **search domain**, and the planned
  candidate-overlay work (`landscapes/render.rs` names it explicitly) puts
  candidate markers in **search-box space**. A true-domain ASCII frame and a
  search-box report frame would then disagree about the same landscape, in the
  same run — with candidates plotted against the wrong rectangle.

The true domain therefore stays in the docs until per-axis *search* boxes are
representable, at which point one rectangle type serves both (Open questions).

## Consequences

### Positive

- Branin's third global minimum `(−π, 12.275)` is inside the search box for the
  first time; Trefethen's `x₁` range is no longer clipped. Both landscapes are
  now solvable in principle, which they were not.
- The rule Bukin6 was already following is now **written down**, with a
  falsifiable test for each of its two obligations. The next asymmetric landscape
  has a convention to conform to and a reviewer has a clause to cite.
- O2 turns "is this box safe?" from a judgment call into an obligation with a
  written proof and a regression sweep.

### Negative / accepted costs

- **Published-baseline incomparability.** Results on **Branin, Trefethen, Bukin6
  and Six-Hump Camel** are obtained over the square hull / reduced range, **not**
  over the literature rectangle. They are therefore **not directly comparable to
  published results that search the rectangle** — the box is larger (hull) or
  smaller (reduced range), so success rates and evaluation counts differ. This is
  disclosed in each landscape's docs and is the price of representable bounds.
- **Widening a search box is a research hazard in general.** A wider box could in
  principle admit a point better than `f*` — which would silently invalidate
  *every* result ever reported on that landscape, since the "optimum" the runs
  are scored against would no longer be the optimum. **O2 is the guard, and it
  must be discharged analytically, not assumed.** For the record, both widenings
  in this ADR are provably safe: Branin's `f* = 10/(8π)` is the global infimum
  over all of ℝ² (`a² ≥ 0`, `cos x₁ ≥ −1`), so no box can beat it; and any point
  beating Trefethen's `f*` must lie within radius ≈ 0.817 of the origin, well
  inside the hull.
- **Showcase mutation strength shifts.** The `mutation_sigma` constants in
  `crates/rlevo/examples/evo/*_showcase.rs` are **absolute**, not box-relative,
  so changing a landscape's box changes the effective mutation strength of that
  landscape's demo. The affected showcases were re-tuned by hand; there is no
  mechanism preventing the next box change from silently detuning one.

### Neutral

- No API change: `bounds()` keeps its signature, stays inherent, stays off the
  `Landscape` trait. The `Landscape2DSnapshot` wire format is untouched, and no
  record `FORMAT_VERSION` bump is needed.
- The true per-axis domains were never encoded anywhere in the first place, so
  keeping them as documentation loses nothing that existed.

### Open questions (recorded, not resolved)

- **Nothing enforces that a search runs inside the landscape's box.** `bounds()`
  is not on the `Landscape` trait, so the wiring from a landscape to its
  optimiser's `bounds` field is done **by hand, 23 times**, in the showcases.
  That missing chokepoint is precisely why a wrong constant was invisible: no
  single place ever compared a landscape's box against the search it fed. Closing
  it is the natural job of a future constrained-objective design (a bounded
  objective seam that carries its own feasible region), not of a fix to three
  constants.
- **If per-axis search boxes ever become representable, this decision is
  revisited.** That is the moment to design a per-axis rectangle type — and it
  must be designed for *search* first. A render-only per-axis type introduced
  earlier would pre-empt that design with a rectangle that has no bearing on
  where the optimiser is allowed to look.

## Alternatives considered

- **Return the true per-axis domain from `bounds()`** (i.e. keep `(-5, 10)` for
  Branin and let the consumer sort it out). Rejected: it is what the code did,
  and it is unsound — a scalar pair applied per-coordinate *is* a square box, so
  returning one axis of an asymmetric rectangle silently excludes part of the
  domain. The type cannot express what this option claims to return.
- **Add a `domain() -> [(f64, f64); N]` accessor alongside `bounds()`.**
  Rejected for now: it has no consumer. The EA configs cannot hold it (ADR 0027 —
  one scalar `Bounds`), and after the O1 fix the renderer no longer needs it
  (§4). Shipping it would add a second rectangle convention colliding with
  `Landscape2DSnapshot`'s search-domain `bounds_x`/`bounds_y` and the planned
  candidate overlay.
- **Ship a per-axis renderer only.** Rejected on the same collision: a
  true-domain ASCII frame and a search-box report frame would disagree about the
  same landscape, and candidate markers live in search-box space. Purely cosmetic
  gain (trimming excess heatmap area), real semantic cost.
- **Clamp inside `evaluate()` so out-of-domain points are impossible.**
  Rejected: the landscapes are pure evaluators and the harness owns domain
  enforcement (already stated in the landscape docs). Clamping inside `evaluate`
  would flatten the objective outside the box, manufacturing a plateau of
  spurious global optima on the boundary — a far worse defect than the one being
  fixed.
- **Widen the per-axis `Bounds` type to a rectangle across `rlevo-evolution`.**
  Rejected as out of scope: it touches ~20 configs and 52 fields, changes
  `clamp_vec`'s contract, and is a search-space design change, not a fix for
  three wrong constants. Recorded as the Open question that would supersede this
  ADR.
- **Leave `bounds()` alone and document the exclusion.** Rejected outright: a
  documented unreachable optimum is still an unreachable optimum, and every
  Branin result in the library would be scored against an `f*` no run could
  attain.

## References

- Issue #113 — "[env] Landscapes: incorrect `bounds()` for asymmetric-domain
  functions."
- ADR [0027](0027-bounds-newtype-for-closed-ranges.md) — the single scalar
  `Bounds` newtype every EA config holds, applied per-coordinate via
  `clamp_vec`; the constraint that makes a per-axis search rectangle
  unrepresentable and therefore forces the square-box convention recorded here.
  (§Scope of migration also explicitly excluded the `landscapes/render`
  `(f64, f64)` heatmap bounds from that migration — they are still raw pairs.)
- ADR [0023](0023-objective-sense-and-maximize-convention.md) — `Landscape`
  defaults to `ObjectiveSense::Minimize`; a landscape is a cost surface, which is
  why O2 is phrased as "no point with `f < f*`".
- ADR [0043](0043-grid-observation-contract.md) — the precedent for recording a
  *stated invariant* (Invariant M) plus its derived numeric bound, rather than
  leaving a magic constant a later contributor can "simplify" back. O1/O2 follow
  the same shape.
- Code: `crates/rlevo-environments/src/landscapes/branin.rs`,
  `.../trefethen.rs`, `.../bukin6.rs`, `.../six_hump_camel.rs`,
  `.../rosenbrock_flat.rs` (the five landscapes whose boxes are hulls or reduced
  ranges); `crates/rlevo-environments/src/landscapes/render.rs` (samples
  `bounds()` along the first two coordinates; documents the deferred
  candidate overlay); `crates/rlevo-core/src/render/payload.rs:59-62`
  (`Landscape2DSnapshot::bounds_x`/`bounds_y`, documented as the **search
  domain**); `crates/rlevo-core/src/fitness.rs:111` (the `Landscape` trait —
  `evaluate` + `sense`, no `bounds`); `crates/rlevo-evolution/src/local_search.rs:272`
  (`clamp_vec` — one `Bounds`, every coordinate);
  `crates/rlevo/examples/evo/common.rs:98` (`showcase(title, dim, bounds,
  mutation_sigma, landscape)` — the hand-wired, absolute `mutation_sigma`).
- Dixon, L.C.W. & Szegő, G.P. (1978), "The global optimization problem: an
  introduction", *Towards Global Optimization 2*, pp. 1–15, North-Holland — the
  source of Branin's asymmetric domain `[-5,10] × [0,15]` and its three certified
  minima `(−π, 12.275)`, `(π, 2.275)`, `(3π, 2.475)`.
- Trefethen, L.N. (2002), "A Hundred-dollar, Hundred-digit Challenge",
  *SIAM News* 35(1) — Problem 4. Cited for the **function and its certified
  `f* ≈ −3.306868647475` at `(−0.024403, 0.210612)`**, and for what it does *not*
  say: the original problem is **unconstrained** (well-posed anyway — the
  `(x₁²+x₂²)/4` term is coercive), so it is not the source of any box.
- Al-Roomi, A.R. (2015), *Unconstrained Single-Objective Benchmark Functions
  Repository*, propagated via Gavana's `benchmark_functions` (2013) — source of
  Trefethen's **asymmetric benchmark box** `x₁ ∈ [-6.5, 6.5]`, `x₂ ∈ [-4.5,
  4.5]`, whose square hull `(-6.5, 6.5)` this ADR adopts. Al-Roomi's page lists
  Mishra, S. (2006), *"Some New Test Functions for Global Optimization and
  Performance of Repulsive Particle Swarm Method"*, MPRA Paper 2718, among its
  references, but Mishra's paper does not itself discuss the Trefethen
  function or these bounds — the box is attributed here to Al-Roomi/Gavana,
  not to Mishra.
- Monismith, D. (2010) — the canonical `[-2000, 2000]ⁿ` side constraints of the
  modified (flat) Rosenbrock; Chen, Y. (1997) — the reduced `[-30, 30]` range
  adopted here under §3.
- Al-Roomi, A.R. (2015), *Unconstrained Single-Objective Benchmark Functions
  Repository* — the canonical `[-5, 5]²` domain of Six-Hump Camel-Back
  (function #23), reduced here to `(-2, 2)` under §3.
