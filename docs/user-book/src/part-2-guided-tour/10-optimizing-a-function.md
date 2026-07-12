# Optimising a function

The smallest interesting problem in optimisation is: **find the bottom of a
bowl.** Mathematically, minimise the *sphere function*

```math
f(\mathbf{x}) = \sum_i x_i^2
```

whose minimum is obviously \\(\mathbf{x} = \mathbf{0}\\). It's a terrible problem
to need an algorithm for — which is exactly why it's the right place to learn
one. There are no neural networks in this chapter and no reward signal; just a
population of candidate vectors getting better each generation.

## Step 1 — the problem is a `Landscape`

In `rlevo`, a static function you want to optimise is a **`Landscape`**: anything
that can score a point in \\(\mathbb{R}^n\\).

```rust
pub trait Landscape: Send + Sync {
    fn evaluate(&self, x: &[f64]) -> f64;
}
```

The sphere is built in:

```rust,no_run
use rlevo::envs::landscapes::sphere::Sphere;

let problem = Sphere::new(/* dim = */ 8).expect("dim >= 1");
assert_eq!(problem.evaluate(&[0.0; 8]), 0.0);   // the global minimum
```

Construction is fallible on purpose: at `dim == 0`, \\(\sum_i x_i^2\\) is an empty
sum, which evaluates to exactly `0.0` — the sphere's own global optimum — so an
unguarded zero-dimensional instance would silently report every run "solved"
rather than fail loudly at setup. Every landscape in `rlevo::envs::landscapes`
rejects its degenerate dimensionality this way; we `.expect` here because `8` is
a compile-time constant we control, not a value read from user input.

`Sphere` also tells you a sensible search box via `problem.bounds()`, which
returns `(-5.12, 5.12)` — the conventional range for this benchmark.

Read that name carefully, because it promises less than it appears to. `bounds()`
returns a **recommended search box, not the landscape's mathematical domain**. It
is a single \\( (lo, hi) \\) pair, and every consumer applies it to *each*
coordinate — so for `Sphere`, whose domain is the symmetric hypercube
\\( [-5.12, 5.12]^D \\), the two coincide and the distinction costs you nothing.
It starts to matter for the landscapes whose true domain is a *rectangle*.
`Branin` lives on \\( x_1 \in [-5, 10] \\), \\( x_2 \in [0, 15] \\), which no
single pair can express, so its `bounds()` returns the **square hull**
\\( (-5, 15) \\) of that rectangle. We guarantee two things about whatever box a
landscape hands you: it can **reach every certified global optimum** on every
axis, and it contains **no point better than the true optimum** \\( f^* \\). The
first is what lets you trust a converged run; the second is what stops a
generous box from inventing an optimum that isn't there. The trade is that the
hull may admit points outside the published domain — harmless, precisely because
of the second guarantee. ADR 0045 records the reasoning, and each landscape's
module docs cite the source for its domain.

## Step 2 — the searcher is a `Strategy`

A **`Strategy`** proposes candidate populations and updates itself from their
fitness. The classic one is a **genetic algorithm**: keep the fittest, mate them,
mutate the children. We configure it with `GaConfig`:

```rust,no_run
use rlevo::core::bounds::Bounds;
use rlevo::core::rate::NonNegativeRate;
use rlevo::evo::algorithms::ga::{
    GaConfig, GaCrossover, GaReplacement, GaSelection, GeneticAlgorithm,
};

let config = GaConfig {
    pop_size: 64,
    genome_dim: 8,
    bounds: Bounds::new(-5.12, 5.12),
    mutation_sigma: NonNegativeRate::new(0.3),
    selection: GaSelection::Tournament { size: 3 },
    crossover: GaCrossover::BlxAlpha { alpha: NonNegativeRate::new(0.5) },
    replacement: GaReplacement::Elitist { elitism_k: 2 },
};
```

Read it top to bottom: keep **64** candidates of length **8**, each gene clamped
to the box; **tournament selection** picks parents by holding 3-way contests;
**BLX-α crossover** blends two parents; and **elitism** carries the best 2 into
the next generation untouched so progress never regresses. Don't memorise these —
each is one enum variant and the others are listed in the API docs.

You may have noticed we wrap the scalars: the search box is a
[`Bounds`](https://docs.rs/rlevo-core), the mutation step \\(\sigma\\) and the
BLX-α \\(\alpha\\) are a [`NonNegativeRate`](https://docs.rs/rlevo-core), and a
probability like a crossover \\(p\\) is a
[`Probability`](https://docs.rs/rlevo-core). These are validated newtypes — they
reject a `NaN`, an infinity, or an out-of-range value *at construction*, so a
malformed rate can never silently degenerate an operator half a run later. You
build one with `NonNegativeRate::new(0.3)`; pass user-supplied values through the
fallible `try_new` instead.

## Step 3 — turning the crank with the harness

The `Strategy` follows an **ask/tell** contract — it *asks* you for the next
population, you score it, you *tell* it the fitnesses. For a static landscape
that loop is so standard that `rlevo` ships a driver, `EvolutionaryHarness`, that
runs it for you:

```rust,no_run
{{#rustdoc_include ../../../../crates/rlevo-examples/examples/book/ch01_sphere_ga.rs:harness}}
```

That loop is the whole run. Each `step` does one generation: the harness asks the
GA for a population, evaluates every member against the sphere, and tells the GA
the scores.

> **Fitness convention.** `rlevo`'s engine is **maximise-native** (higher is
> better), but you declare a cost objective's direction rather than negating it.
> Sphere is a cost, so its `Landscape::sense()` is `ObjectiveSense::Minimize`; the
> harness reconciles direction at one chokepoint and reports `best_fitness_ever` in
> that natural sense. For Sphere, `best_fitness_ever = 0.0` is perfect.

Because we passed `seed = 42`, re-running gives an identical trajectory. `rlevo`
derives every random draw from that seed through a `seed_stream`, never from
process-global RNG, so parallel runs and tests stay reproducible (more on this in
[The Ask/Tell Contract](../appendix-d-suppl/ask-tell-contract.md) and the
contributor book).

## Step 4 — what you should see

Run it:

```bash
cargo run -p rlevo-examples --example ch01_sphere_ga
```

The best fitness falls steeply for the first ~50 generations, then crawls toward
zero as the population concentrates near the origin:

```text
gen   0   best = 4.17e1
gen  25   best = 2.31e0
gen  50   best = 1.80e-1
gen 100   best = 4.00e-3
gen 200   best = 6.00e-5
```

You just ran a complete evolutionary optimisation. Three pieces did all the work:

- a **`Landscape`** (your problem),
- a **`Strategy`** (the searcher),
- the **ask/tell** loop tying them together (here, driven by the harness).

## Make it yours

The fastest way to internalise this: change one thing and re-run.

- **Swap the problem.** `rlevo::envs::landscapes` has harder bowls —
  `Rastrigin` (a minefield of local minima) and `Rosenbrock` (a banana-shaped
  valley). Drop one in place of `Sphere` and watch the GA struggle — that
  struggle is exactly what the stronger strategies in
  [Appendix A](../appendix-a-ec-algorithms/index.md) (CMA-ES, differential
  evolution, the EDAs) are built to handle.

> **Reach for CMA-ES on the hard bowls.** When the GA stalls on `Rosenbrock`'s
> bent valley or `Rastrigin`'s ill-conditioned ripples, the standard next step is
> [CMA-ES](../appendix-a-ec-algorithms/cma-es.md). It adapts a full covariance
> matrix to the landscape's *shape* — rotating and stretching the search
> distribution to follow a bent valley instead of fighting the coordinate axes —
> which makes it the strongest general-purpose baseline for low-to-medium `D`
> (≲ 30). On multimodal landscapes, a larger population (`λ`) helps it find the
> right basin. Its path-free cousin **CMSA-ES** is the simpler alternative when
> you want fewer moving parts. Both are drop-in `Strategy` swaps: same harness,
> same ask/tell loop, only the config type changes.
- **Crank the population down** to `pop_size: 8` and watch it stall — too few
  candidates can't cover an 8-D box.
- **Kill elitism** (`GaReplacement::Generational`) and watch the best score
  bounce instead of monotonically improving.

## Where this is going

You optimised a *function*. But the candidates were just vectors — nothing said
they couldn't be the **weights of a neural network**, or that the score had to
come from a formula instead of an **agent acting in a world**.

The [next section](20-classic-control.md) brings in a real `Environment`, a
reward signal, and a gradient-based RL agent. If you first want to see exactly
how the harness drove the strategy here — and how to write that ask/tell loop by
hand when the harness doesn't fit — read
[The Ask/Tell Contract](../appendix-d-suppl/ask-tell-contract.md) in the
supplementary material.

> **Foundations link.** The exploitation–exploration tension you just observed
> (population too small → stalls; no elitism → regresses) is discussed
> conceptually in [What Is Optimisation?](../part-1-foundations/10-optimisation.md).
> The GA operators used here — tournament selection, BLX-α crossover, Gaussian
> mutation — are derived in
> [Appendix A](../appendix-a-ec-algorithms/index.md).

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
