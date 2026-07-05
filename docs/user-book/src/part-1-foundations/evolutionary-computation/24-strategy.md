# Strategies: the ask/tell Loop

This is the chapter where the three ingredients become one loop. We have a
[genome](22-genome.md) — a row of a population tensor; the [operators](21-ops.md)
that select and vary those rows; and the [fitness](23-fitness.md) that scores them
in the engine's canonical maximise space (a cost objective declaring
`ObjectiveSense::Minimize`). Each has been a thing you *call*; none of them, on its
own, runs. A **strategy** is what assembles them into a running generation.

The [parent chapter](../20-evolutionary-computation.md) introduced the `Strategy<B>`
trait as the choreography behind the five-step evolutionary skeleton. We walk it
here in two movements: first the **four methods** of the trait in the abstract,
then a single complete, real strategy — the Genetic Algorithm — traced from `init`
to convergence with every one of those methods made concrete.

## The four methods

Every algorithm in `rlevo::evo` implements one trait. It has three associated
types — the static run config, the generation-to-generation state, and the genome
container `ask` produces — and four methods:

```rust
pub trait Strategy<B: Backend>: Send + Sync {
    type Params: Clone + Debug + Send + Sync;  // static run config
    type State:  Clone + Debug + Send;         // carried across generations
    type Genome: Clone + Send;                 // produced by ask, consumed by tell

    fn init(&self, params: &Self::Params, rng: &mut dyn Rng, device: &B::Device) -> Self::State;
    fn ask (&self, params: &Self::Params, state: &Self::State, rng: &mut dyn Rng, device: &B::Device) -> (Self::Genome, Self::State);
    fn tell(&self, params: &Self::Params, population: Self::Genome, fitness: Tensor<B, 1>, state: Self::State, rng: &mut dyn Rng) -> (Self::State, StrategyMetrics);
    fn best(&self, state: &Self::State) -> Option<(Self::Genome, f32)>;
}
```

Reading them in order:

- **`init`** builds the starting state — samples the initial population, primes any
  adaptive quantities (step-sizes, covariance), and zeroes the generation counter.
- **`ask`** proposes the next population to evaluate, returning it alongside an
  updated state. The state can carry pre-computed bookkeeping forward — e.g. the
  parent indices a selection step sampled — so `tell` need not recompute them.
- **`tell`** consumes that population *together with its fitness tensor* and
  produces the next state plus a `StrategyMetrics` snapshot of the generation that
  just finished.
- **`best`** is the best-so-far accessor: `Some((genome, fitness))`, or `None`
  before the first `tell`.

Two things are deliberate across all four. First, the **RNG is a parameter**, never
a field: strategies carry *no* internal PRNG state, so the caller owns every
source of stochasticity (the reason why is the host-RNG convention from the
[operators chapter](21-ops.md)). Second, the split between `ask` and `tell` is
exactly the seam where the harness slots evaluation — the strategy proposes, the
harness scores, the strategy learns. A strategy never calls the fitness function
itself.

## The warm-up generation

There is one subtlety in the contract worth surfacing before the walkthrough,
because it trips up first readings of the code. The very first `ask` has nothing
to select *from* — no fitness has been computed yet. So the first turn of the loop
is an **evaluation-only pass**:

- the first `ask` returns the freshly sampled seed population **unchanged** (it
  detects the empty fitness cache in the state);
- the first `tell` simply **caches** that seed population's fitness and bumps the
  generation counter — no selection, no replacement.

From the second generation on, the loop runs in full. This is why a run of `N`
generations performs `N` evaluations but only `N − 1` rounds of variation: the
first generation pays for scoring the random seed.

## A strategy in full: the Genetic Algorithm

`GeneticAlgorithm<B>` is the canonical real-coded GA, and it is small enough to
read end to end. Its `Params` is a plain config struct whose operator choices are
**enum-selected** rather than generic — a deliberate move to avoid a combinatorial
explosion of type parameters while still letting a run mix and match:

```rust
pub struct GaConfig {
    pub pop_size: usize,
    pub genome_dim: usize,
    pub bounds: Bounds,              // initial-sample range and clamp range
    pub mutation_sigma: f32,
    pub selection:   GaSelection,    // Tournament { size }
    pub crossover:   GaCrossover,    // BlxAlpha { alpha } | Uniform { p }
    pub replacement: GaReplacement,  // Generational | Elitist { elitism_k }
}
```

Its `State` carries the live population, a host-side cache of the current
fitness, the best-so-far genome and its canonical fitness, and the generation counter:

```rust
pub struct GaState<B: Backend> {
    pub population:   Tensor<B, 2>,
    pub fitness:      Vec<f32>,       // empty until the first tell
    pub best_genome:  Option<Tensor<B, 2>>,
    pub best_fitness: f32,            // f32::NEG_INFINITY until the first tell
    pub generation:   usize,
}
```

The `bounds` field is a [`Bounds`](https://docs.rs/rlevo-core) — a small newtype
over a `(lo, hi)` pair that we validate at construction (it rejects an inverted
`lo > hi` or a `NaN` endpoint) so an invalid search box can never reach the
sampler or the clamp. You build one with `Bounds::new(lo, hi)`; the population
sampler and the per-generation clamp both read `bounds.lo()` and `bounds.hi()`.

**`init`** samples an `(pop_size, genome_dim)` population uniformly within
`bounds`, leaves the fitness cache empty, and sets `best_fitness` to
`f32::NEG_INFINITY` (the worst value under the maximise convention) so the first
real evaluation can only improve it.

**`ask`** is where the operators chapter pays off. After the warm-up short-circuit,
the body reads as the short recipe that chapter promised — select, recombine,
mutate, clamp:

```rust
// (warm-up: if state.fitness is empty, return the seed population unchanged)

let parents_a = tournament_select(&state.population, &state.fitness, size, pop_size, &mut sel_rng, device);
let parents_b = tournament_select(&state.population, &state.fitness, size, pop_size, &mut sel_rng, device);

let offspring = match crossover {
    GaCrossover::BlxAlpha { alpha } => blx_alpha(parents_a, parents_b, alpha, &mut xover_rng, device),
    GaCrossover::Uniform  { p }     => uniform_crossover(parents_a, parents_b, p, &mut xover_rng, device),
};

let offspring = gaussian_mutation(offspring, mutation_sigma, &mut mut_rng, device);
let offspring = offspring.clamp(lo, hi);   // keep genes inside bounds
```

Two independent tournaments produce two parent sets; crossover blends them; Gaussian
mutation perturbs the result; the final `clamp` keeps every gene inside the
configured box. Nothing here is virtual dispatch — each operator is the free
function from the [operators chapter](21-ops.md), called directly.

**`tell`** pulls the fitness tensor to host and applies the replacement policy —
unless it is the warm-up call, which only caches the seed's fitness:

```rust
match replacement {
    GaReplacement::Generational    => generational(parents, &parent_fit, offspring, offspring_fit),
    GaReplacement::Elitist { k }   => elitist(parents, &parent_fit, offspring, &offspring_fit, k, device),
}
```

It then refreshes the best-so-far (a rolling maximum — the new best is kept only if
it is *higher* than the incumbent, honouring the canonical-maximise convention),
stores the next population and its fitness, increments the generation, and returns a
`StrategyMetrics` snapshot. **`best`** simply hands back the cached best genome and
its canonical fitness. That `f32::NEG_INFINITY` init and `>` test are not local to
the GA — they recur in every strategy; the
[fitness chapter](23-fitness.md#the-engine-maximises--and-you-declare-your-objectives-sense)
tabulates exactly where the maximise convention is pinned across the engine.

That is the whole algorithm. Swapping tournament for a different selector, or
elitist for generational replacement, is a one-line change to the config enum — the
loop above is unchanged.

## Reproducible randomness

The three RNGs in `ask` — `sel_rng`, `xover_rng`, `mut_rng` — are not the harness
RNG used directly. Each is an independent sub-stream derived from it, keyed so that
selection, crossover, and mutation draws can never collide and every generation is
individually replayable (the initial population is sampled the same way). Combined
with the host-sampling discipline from the [operators chapter](21-ops.md) — every
operator draws on the host, never touching Burn's process-global tensor RNG — this
is what lets a single seed reproduce an entire run. The exact derivation
(`seed_stream`), and what it buys for research reproducibility, are the subject of
[The Ask/Tell Contract](../../appendix-d-suppl/ask-tell-contract.md).

## Driving it: the `EvolutionaryHarness`

A strategy is inert on its own — it proposes and learns but never evaluates. The
`EvolutionaryHarness` is the driver that closes the loop, gluing a strategy to a
`BatchFitnessFn` and exposing the whole thing as a benchmarkable environment:

```rust
let mut harness = EvolutionaryHarness::<B, _, _>::new(
    GeneticAlgorithm::<B>::new(),                 // strategy
    GaConfig::default_for(32, 5),                 // params
    FromFitnessEvaluable::new(SphereFit, Sphere), // fitness fn
    seed, device, max_generations,
).expect("valid config");
harness.reset();
while !harness.step(()).done {}
```

`reset` reseeds the harness RNG from the base seed and materialises the initial
state via `init` — so two runs with the same seed start identically. Each `step`
runs exactly one generation:

```text
ask ──▶ population ──▶ evaluate_batch ──▶ fitness ──▶ tell ──▶ (next state, metrics)
```

and returns once the generation counter reaches `max_generations`. The reward it
reports back is the **canonical** `best_fitness_ever` directly — no sign flip,
because canonical space is already "higher = better" (the harness canonicalises a
`Minimize` objective on the way in, as the [fitness chapter](23-fitness.md)
describes). It stays monotone non-decreasing, so its cumulative sum integrates the
optimisation trajectory. An optional `with_observer` hook surfaces the full per-generation
fitness vector for recorders that need more than the scalar metric stream, at the
cost of one device→host transfer per generation.

## Putting it together

A strategy is the choreography that the operators, genomes, and fitness functions
dance to: `init` seeds a population, `ask` proposes the next one by selecting and
varying the current generation, the harness scores it, and `tell` folds the result
back into state — all driven by reproducible seed streams and read out as
`StrategyMetrics`. With this chapter the Part I tour of `rlevo`'s evolutionary
machinery is complete: representation, variation, evaluation, and orchestration.
For the full GA and ES pseudocode that the trait formalises — and the families
beyond the GA — see [Appendix A](../../appendix-a-ec-algorithms/index.md); to watch
a strategy actually optimise a function end to end, head to the
[guided tour](../../part-2-guided-tour/10-optimizing-a-function.md).

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
