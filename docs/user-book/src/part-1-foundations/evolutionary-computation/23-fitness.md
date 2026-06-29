# Fitness Evaluation

The [genome chapter](22-genome.md) covered the *representation* — how a candidate
solution becomes a row of a population tensor. But a row of numbers is inert until
something *scores* it, and that score is what every operator in the previous two
chapters was sorting, comparing, and selecting on. This chapter is about where
that score comes from.

One idea carries the whole chapter, so we put it first: the engine **maximises** —
higher is always better — and you never hand-negate a cost; you declare your
objective's natural direction with `ObjectiveSense` and let one chokepoint
reconcile it. The [operators chapter](21-ops.md) *asserted* that convention; here
is where it is justified. Around that spine sit four supporting pieces: who *calls*
the fitness function (the harness, never the strategy), the two **trait shapes**
`rlevo` evaluates through, the **adapters** that wire a plain objective into the
engine, and the **shaping** transforms that condition the raw signal before an
ES-style update consumes it.

## The harness calls the objective, not the strategy

A point worth making first, because it shapes every trait below: **strategies
never call the fitness function.** A `Strategy` proposes a population in `ask` and
consumes a fitness *tensor* in `tell` — but the call that turns one into the other
happens in the harness, between those two steps:

```text
state ──ask──▶ population ──[harness evaluates]──▶ fitness ──tell──▶ next state
```

This separation is what lets one strategy implementation run against *any*
objective — a synthetic landscape, a simulator, a neural-network rollout — without
a line of change. The strategy knows the *shape* of fitness (a `(pop_size,)`
vector) but nothing about its *meaning*. The fitness traits in this chapter are
the contract at that seam.

## Two evaluation shapes

`rlevo` models fitness with two traits, one per calling pattern. Neither is
implemented by a strategy — they are implemented by the **objective** (a landscape
adapter, a module evaluator, a policy-rollout scorer) and consumed by the
*harness*, not the strategy. The strategy only ever sees the resulting tensor.

**`BatchFitnessFn<B, G>`** evaluates an *entire population* in one call and
returns a device-resident tensor. This is the hot path — the harness calls it once
per generation, between `ask` and `tell`:

```rust
pub trait BatchFitnessFn<B: Backend, G>: Send {
    fn evaluate_batch(&mut self, population: &G, device: &B::Device) -> Tensor<B, 1>;
}
```

The returned `Tensor<B, 1>` has shape `(pop_size,)` on the supplied device, and
one invariant is load-bearing: **row order is preserved.** `fitness[i]` is the
score of the individual at row `i` of `population`. Every downstream
operator — tournament selection gathering winner rows, elitism pairing parents
with their costs — relies on that index alignment. Break it and the algorithm
silently optimises the wrong correspondence.

This is the only evaluation path the `EvolutionaryHarness` uses: a strategy's
`Genome` is whatever `ask` produces (a `Tensor<B, 2>` for the real-coded GA), and
the harness's stored `fitness_fn: F: BatchFitnessFn<B, S::Genome>` turns it into
the fitness tensor that `tell` consumes. Batched-by-default is a throughput choice:
scoring a whole generation in one call lets a purpose-built landscape keep the
population on device and compute every fitness in a single kernel sweep, no per-row
round-trip. The host-side adapters below ([`FromLandscape`](#bridging-a-host-objective-the-two-adapters)
and friends) still satisfy this batched signature — they just loop internally.

**`FitnessFn<G>`** evaluates a *single* member and returns a scalar. It takes
`&mut self` so an evaluator can keep state — most usefully an evaluation counter:

```rust
pub trait FitnessFn<G>: Send {
    fn evaluate_one(&mut self, member: &G) -> f32;
}
```

This is *not* the per-generation path — the harness never calls it. Its production
role is **single-point evaluation for memetic local search**: the hill-climbing /
Nelder–Mead / simulated-annealing refiners probe one candidate at a time, and the
`RowFitness` adapter in `memetic.rs` wraps a `BatchFitnessFn` into this shape (a
one-row `evaluate_batch` per probe). It is also the convenient shape for
unit-testing operators, where evaluating one genome at a time is clearer than
batching.

## The engine maximises — and you declare your objective's sense

An evolutionary algorithm is, in principle, direction-agnostic: fitness can be a
reward to *maximise* or a cost to *minimise*, and the algorithm itself does not
care which. `rlevo` makes a deliberate choice and commits to it everywhere — **the
evolution engine maximises.** Inside `rlevo-evolution`, the fitness a strategy sees
is *canonical*: **higher is always better.** This is the convention reinforcement
learning and the evolutionary-computation literature already speak, so a
contributor reading `de.rs` or `cma_es.rs` sees the textbook direction.

This is an **internal contract of the engine**, baked structurally into every
layer: the same `f32::NEG_INFINITY` worst-value sentinel, the same rolling
`.max()`, the same `>` improvement test recur in every strategy:

- `StrategyMetrics::best_fitness` is the **largest** value in a generation
  (canonical space);
- `best_fitness_ever` is a **rolling maximum** across all generations
  (`previous_best.max(current_best)`);
- a fresh state initialises `best_fitness` to `f32::NEG_INFINITY`, so the first
  real evaluation can only improve it;
- truncation keeps the highest `top_k`, tournaments keep the largest of each draw,
  elitism carries the fittest parents — exactly as the
  [operators chapter](21-ops.md) described.

But most of `rlevo`'s synthetic landscapes — Sphere, Ackley, Rastrigin — are
**cost** surfaces: each is zero at its global optimum and positive everywhere else,
so "drive fitness toward zero" means *minimise*. Rather than force you to
hand-negate every cost (the old footgun) or thread a "minimise or maximise?" flag
through every operator, we reconcile the two directions at **exactly one place**.

### `ObjectiveSense` — declared once, reconciled at one chokepoint

You declare your objective's natural direction with
[`ObjectiveSense`](https://docs.rs/rlevo-core) — a zero-cost
`enum ObjectiveSense { Minimize, Maximize }` in `rlevo-core::objective`. Your
fitness function returns its **natural** value (a Sphere returns \\( \sum_i x_i^2
\\); a policy-return objective returns its mean return) — you never hand-negate.
The [`EvolutionaryHarness`](https://docs.rs/rlevo-evolution) is the *sole*
canonicaliser: it reads the sense, negates a `Minimize` objective into the engine's
maximise space before `tell`, and maps the metrics back to your declared sense for
reporting. A `Minimize` landscape's `best_fitness` reads as its natural cost
(Sphere → 0), even though the engine internally maximised \\( -\sum_i x_i^2 \\).

`ObjectiveSense` lives on the **fitness function** —
[`BatchFitnessFn::sense`](https://docs.rs/rlevo-evolution), required with no
default, so a reward or accuracy objective cannot be optimised backwards by
omission. The two landscape adapters carry it for you:

| Mechanism | Where | What pins the direction |
| --------- | ----- | ----------------------- |
| Worst-value sentinel | every strategy: `algorithms/ga.rs`, `ep.rs`, `de.rs`, `es_classical.rs`, `eda/mod.rs`, `gp_cgp.rs` | `best_fitness: f32::NEG_INFINITY` — only a *higher* value can replace it |
| Rolling maximum | `strategy.rs` (`StrategyMetrics::from_host_fitness`) | `best_fitness_ever.max(best)` |
| Improvement test | every `tell`; hill-climbing `local_search/hill_climbing.rs` | `if f > best` — never `<` |
| **Sense declaration** (your objective) | `BatchFitnessFn::sense`; `Landscape::sense` (defaults `Minimize`) | required on the fitness fn; only a landscape defaults |
| **Canonicalisation** (the chokepoint) | `strategy.rs` (`EvolutionaryHarness::step`); `fitness.rs` (`FromLandscape`/`FromFitnessEvaluable`) | `Minimize` → negate before `tell`; `from_canonical` for reporting |
| Convergence assertions | `algorithms/*.rs` tests; `crates/rlevo/tests/rastrigin_run_suite.rs` | `assert!(best < tolerance)` on zero-at-optimum landscapes — in *natural* space |

A [`Landscape`](https://docs.rs/rlevo-core) is a cost surface by definition, so its
`sense()` defaults to `ObjectiveSense::Minimize` — the bundled landscapes need no
per-type change. When you wrap one for a strategy, spell the sense out at the
construction site so intent is visible:

```rust,ignore
use rlevo_core::objective::ObjectiveSense;
use rlevo_evolution::fitness::FromLandscape;

// A cost surface: declare Minimize explicitly (the adapter would default to it).
let fitness = FromLandscape::with_sense(SphereLandscape::new(dim), ObjectiveSense::Minimize);
// ... run the harness ...
// best_fitness_ever reads in your declared sense — the natural cost.
assert!(harness.latest_metrics().unwrap().best_fitness_ever < 1e-3); // Sphere → 0
```

> **The reward needs no sign-flip any more.** Because canonical space is already
> higher-is-better, the `EvolutionaryHarness` emits the canonical
> `best_fitness_ever` directly as the per-step reward — monotone non-decreasing, so
> its cumulative sum still integrates the optimisation trajectory. The old
> `reward = -best_fitness_ever` negation is gone, and so is the "negate your
> maximisation objective" advice: a reward objective declares
> `ObjectiveSense::Maximize` and is passed straight through.

## Bridging a host objective: the two adapters

Most objectives are easiest to write as a plain scalar function on the host —
`fn(&[f64]) -> f64`. Two adapters lift such a function into a
`BatchFitnessFn<B, Tensor<B, 2>>` so it can drive a real-coded strategy:

- **`FromLandscape<L>`** wraps a self-evaluating
  `rlevo_core::fitness::Landscape` (one that carries its own
  `evaluate(&[f64]) -> f64`). Reach for it when the landscape *is* the fitness
  function — Sphere, Ackley, Rastrigin — and a separate evaluator would add
  nothing.
- **`FromFitnessEvaluable<FE, L>`** wraps an
  `rlevo_core::fitness::FitnessEvaluable<Individual = Vec<f64>, Landscape = L>` —
  the case where the *scoring procedure* (`FE`) and the *thing scored against*
  (`L`) are separate types. The trait's `Landscape` associated type carries **no
  bound** (`rlevo-core/src/fitness.rs`), so `L` is whatever context the evaluator
  needs — a real benchmark landscape, or a bare marker.

That second adapter has two idioms in the codebase, and the difference is where
the arithmetic lives.

**Evaluator delegates to a real landscape.** The cross-crate Rastrigin suite
(`crates/rlevo/tests/rastrigin_run_suite.rs`) drives a GA, ES, EP, and DE off one
evaluator that forwards to the `Rastrigin` landscape from `rlevo-environments`:

```rust
use rlevo_core::fitness::FitnessEvaluable;
use rlevo_environments::landscapes::rastrigin::Rastrigin;
use rlevo_evolution::fitness::FromFitnessEvaluable;

struct Minimizer;
impl FitnessEvaluable for Minimizer {
    type Individual = Vec<f64>;
    type Landscape = Rastrigin;
    fn evaluate(&self, x: &Self::Individual, l: &Self::Landscape) -> f64 {
        l.evaluate(x) // the geometry lives in Rastrigin; the evaluator forwards
    }
}

// 10-dimensional Rastrigin, ready to hand to an EvolutionaryHarness.
let fitness_fn = FromFitnessEvaluable::new(Minimizer, Rastrigin::new(DIM));
```

**Evaluator carries the scoring; the landscape is a marker.** The per-strategy
unit tests (`algorithms/ga.rs`, `metaheuristic/pso.rs`, `algorithms/ep.rs`, and
their siblings) skip the separate landscape type entirely. The `Landscape`
associated type is an empty tag, ignored in the body, and the score is computed
inline:

```rust
struct Sphere; // a marker — deliberately not a Landscape impl
struct SphereFit;
impl FitnessEvaluable for SphereFit {
    type Individual = Vec<f64>;
    type Landscape = Sphere;
    fn evaluate(&self, x: &Self::Individual, _: &Self::Landscape) -> f64 {
        x.iter().map(|v| v * v).sum() // sum of squares, scored in the evaluator
    }
}

let fitness_fn = FromFitnessEvaluable::new(SphereFit, Sphere);
```

Choose between the two by asking where the objective is most naturally written.
If a reusable landscape already exists (the `rlevo-environments::landscapes`
family — Sphere, Ackley, Rastrigin), delegate to it; if the score is a one-off
defined right next to the test or experiment, fold it into the evaluator and let
the landscape collapse to a marker. When the landscape *is* the whole objective
and no scoring shim adds anything, skip `FromFitnessEvaluable` and reach for
`FromLandscape` instead — `crates/rlevo-examples/examples/book/ch01_sphere_ga.rs`
does exactly that with `FromLandscape::new(Sphere::new(DIM))`.

Both adapters follow the same recipe per generation: pull each population row to
host as `f32`, widen to `f64`, evaluate on the CPU row-by-row, and re-upload the
results as one `Tensor<B, 1>` — preserving row order. They implement
`BatchFitnessFn` only for `Tensor<B, 2>` (real genomes); discrete-genome
objectives implement the trait directly.

Two caveats live in the adapters, both inherent to the host round-trip:

- **Precision.** Rows are read as `f32` and widened to `f64` for the call; the
  `f64` result is narrowed back to `f32` before upload. An objective relying on
  values outside `f32` range, or on sub-ulp precision, loses information at the
  narrowing step.
- **The round-trip itself.** Pulling the population to host and back costs a
  device transfer every generation. A purpose-built landscape that stays on
  device and implements `BatchFitnessFn` directly avoids it — the adapters are a
  convenience for host-side objectives, not the fast path.

Both panic if the population tensor is not rank 2, or if its data cannot be read
as `f32` (e.g. an integer backend) — programming errors, documented as `# Panics`
clauses in the spirit of `docs/rules.md`.

> **The full adapter catalogue.** Beyond these two host adapters, `rlevo` ships
> `ModuleEvalFn` (score a neural-network module per genome) and `RolloutFitness`
> (run a policy in an environment), and you can implement `BatchFitnessFn`
> directly for an on-device objective. [Wiring an Objective: Adapters and
> Evaluators](../../appendix-d-suppl/objective-adapters.md) maps all of them with a
> decision guide and a bring-your-own-objective walkthrough — start there if the
> generics feel dense.

> **Both `rlevo-core` traits return natural values.** `Landscape::evaluate` and
> `FitnessEvaluable::evaluate` each return the objective's *natural* value — you
> never hand-negate. A `Landscape` is a cost surface by definition, so
> `Landscape::sense()` defaults to `ObjectiveSense::Minimize`; the adapters
> (`FromLandscape`, `FromFitnessEvaluable`) carry that sense to the harness, which
> does the one negation. Every shipped landscape is written this way (the bundled
> `Sphere` returns a plain sum of squares), and a genuinely maximisation-style
> objective just declares `ObjectiveSense::Maximize` — no negation anywhere.

## Shaping the signal

Raw fitness is sometimes a poor *training signal* even when it is a fine *score*.
A handful of outlier costs can dominate a step; scale can drift wildly between
generations. The `shaping` module offers two monotone, **RNG-free** transforms
that operate directly on the `(pop_size,)` fitness tensor:

| Transform | What it does | Why |
| --------- | ------------ | --- |
| `z_score` | centres to mean 0 and divides by population std-dev (floored at `1e-8`) | normalises scale across generations; degenerate all-equal populations map to zeros, not NaNs |
| `centered_rank` | replaces each fitness with its rank, linearly spaced so the largest maps to `+0.5` and the smallest to `-0.5` | discards outlier *magnitudes*, keeping only order — the standard signal in modern ES (e.g. OpenAI-ES) |

Both are pure functions of the fitness vector, so they compose freely and never
touch the host RNG. They are *signal conditioning*, applied by strategies that
consume fitness as a gradient-like update (the ES family); the comparison-based
operators — tournament, truncation, elitist replacement — need only the ordering
and use the raw costs directly.

## Reading the result: `StrategyMetrics`

`tell` returns a `StrategyMetrics` snapshot summarising the generation that just
finished. The harness reports these in **your declared sense** (a `Minimize`
landscape reads as its natural cost), so the four fitness fields tell the story in
the space you expect:

- `best_fitness` — the best cost/score **this** generation;
- `mean_fitness` — the generation's average;
- `worst_fitness` — the worst this generation;
- `best_fitness_ever` — the best across **all** generations.

The most informative reading is the *gap* between `best_fitness_ever` and
`mean_fitness` in the final generation. A large gap signals **premature
convergence** — a few elites found a good basin while the rest of the population is
still scattered. A small gap means the whole population has settled near the same
optimum. For landscapes with a known optimum (Ackley → 0), `best_fitness_ever`
doubles as a direct "how close did we get" readout.

## Putting it together

Fitness is the contract between a genome and the objective: the harness evaluates
a proposed population through a `BatchFitnessFn`, returns a row-aligned
`(pop_size,)` cost tensor, and the strategy consumes it — always treating *lower
as better*. Host-side objectives reach the engine through the `FromLandscape` and
`FromFitnessEvaluable` adapters; ES-style strategies condition the raw signal with
rank or z-score shaping; and `StrategyMetrics` reports the per-generation summary,
which the harness maps back into the objective's declared sense. With representation
([genome](22-genome.md)), variation ([operators](21-ops.md)), and evaluation
(this chapter) all on the table, the last piece is the **strategy** itself — the
`ask`/`tell` choreography that assembles selection, crossover, mutation, and
replacement into a running generation.

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
