# Wiring an Objective: Adapters and Evaluators

The [Fitness Evaluation](../part-1-foundations/evolutionary-computation/23-fitness.md)
chapter introduced the trait the engine consumes — `BatchFitnessFn` — and the two
host-side adapters that feed it. This page is the reference behind that chapter: a
map of *every* way to get an objective into `rlevo`, why there is more than one,
and a worked example that takes a brand-new objective from nothing to a running
optimisation.

It exists because the fitness layer is the part of `rlevo` most likely to look
like a thicket of generics on first contact: `BatchFitnessFn<B, G>`,
`FromLandscape<L>`, `FromFitnessEvaluable<FE, L>`, `ModuleEvalFn`,
`RolloutFitness`. The thicket dissolves once you see that **they are not five
competing concepts — they are one trait and four on-ramps to it.**

> **Coming from Python?** In DEAP, PyGAD, or a hand-rolled numpy loop, "the
> objective" is a single callable: `def fitness(x): return ...`. `rlevo` splits
> that one idea into *a contract* (the trait the engine calls) and *adapters*
> (small wrappers that turn the shape your objective already has into that
> contract). You are never writing five objectives — you are picking which on-ramp
> matches the objective you already have.

---

## Part 1 — The decision guide

### The one trait that matters

The engine — specifically the `EvolutionaryHarness` — consumes exactly one trait:

```rust
pub trait BatchFitnessFn<B: Backend, G>: Send {
    /// Whole population in, one cost per row out. Row order is preserved:
    /// `fitness[i]` scores the individual at row `i`.
    fn evaluate_batch(&mut self, population: &G, device: &B::Device) -> Tensor<B, 1>;
}
```

Everything else on this page is a type that *implements* `BatchFitnessFn` so you
do not have to write `evaluate_batch` by hand. The two generic parameters are less
exotic than they look:

- **`B`** is the Burn **backend** — the compute target (`Flex`, `Wgpu`, `NdArray`).
  It rides along on every tensor type; you almost never name it explicitly, because
  the harness infers it.
- **`G`** is the **genome container** — the *shape of one population*. For the
  real-coded algorithms (GA, ES, EP, DE) it is `Tensor<B, 2>`: a `(pop_size,
  genome_dim)` block of `f32`. For discrete algorithms it is `Tensor<B, 2, Int>`.

> **The `B`/`G` pair in one sentence.** `B` says *where the numbers live* (which
> device/backend); `G` says *what one genome looks like* (real vs. integer
> tensor). Neither encodes the objective — that is the body of `evaluate_batch`.

### Pick your on-ramp

You rarely implement `BatchFitnessFn` directly. Instead you choose the adapter
whose input shape matches the objective you *already* have:

| You already have… | Reach for | What the adapter does for you |
| ----------------- | --------- | ----------------------------- |
| a pure scalar function `f(&[f64]) -> f64` | **`FromLandscape<L>`** | pulls each genome row to host, calls `f`, re-uploads the costs |
| an evaluator type and a *separate* landscape type | **`FromFitnessEvaluable<FE, L>`** | same round-trip, for when scorer and problem are defined apart |
| a neural-network **module** plus a way to score it | **`ModuleEvalFn`** | unflattens each genome row into a Burn `Module`, runs your scorer |
| a **policy** to run inside an environment | **`RolloutFitness`** (in `rlevo-hybrid`) | rolls out episodes per genome, returns the negated mean return |
| a bespoke, fully on-device objective | **`impl BatchFitnessFn` yourself** | the fast path — no host transfer, one kernel sweep |

Two things worth internalising from the table:

- **The first four are convenience; the last is control.** The adapters all do a
  host round-trip (device → CPU → device) per generation. That is the right default
  for an objective you can only express on the CPU (a simulator, a `Vec<f64>`
  formula). When throughput matters and your objective is expressible in tensor
  ops, implement the trait directly and keep the whole generation on device.
- **`FromLandscape` vs. `FromFitnessEvaluable` is a code-organisation choice, not
  a capability one.** Use `FromLandscape` when the problem *is* the function
  (`Sphere`, `Ackley`). Use `FromFitnessEvaluable` when your design already keeps a
  reusable evaluator separate from a marker landscape type. They produce identical
  per-row behaviour.

### The single-member trait, and where it fits

There is a second, smaller trait — `FitnessFn<G>`, with `evaluate_one(&mut self,
member: &G) -> f32`. **The harness never calls it.** Its job is *single-point*
evaluation for **memetic local search**: the hill-climbing / Nelder–Mead /
simulated-annealing refiners probe one candidate at a time. The `RowFitness`
adapter wraps a `BatchFitnessFn` into a `FitnessFn` (a one-row `evaluate_batch`
per probe) so local search and the population loop share a single objective
definition. Unless you are writing a memetic algorithm, you can ignore it.

### How it all reaches the strategy

Worth restating, because it is the crux: **strategies never see any of these
types.** The data flow is

```text
harness.fitness_fn : impl BatchFitnessFn<B, G>
        │
   strategy.ask ──▶ population (G) ──▶ evaluate_batch ──▶ fitness Tensor<B,1>
                                                              │
                                              strategy.tell(population, fitness)
```

The strategy proposes a population and later consumes a *fitness tensor*. The
adapter you chose lives in the harness, called once per generation between `ask`
and `tell`. This is why the same `GeneticAlgorithm` runs unchanged against a
synthetic landscape, a neural-network rollout, or your custom objective — it is
decoupled from all of them by the tensor seam.

---

## Part 2 — Bring your own objective

Let us make it concrete. Suppose you want to minimise a custom function — say a
shifted, weighted sum of squares, \\(f(x) = \sum_i w_i (x_i - c_i)^2\\) — that is
not one of the bundled landscapes. We will take it from a plain formula to a
running GA, choosing the on-ramp as we go.

### Step 1 — Write the objective in its most natural shape

It is a pure scalar function of a point, so the natural shape is the host-side
`Landscape` trait (`rlevo_core::fitness::Landscape`):

```rust,no_run
use rlevo_core::fitness::Landscape;

struct WeightedSphere {
    weights: Vec<f64>,
    centre:  Vec<f64>,
}

impl Landscape for WeightedSphere {
    // Minimisation convention: lower is better. Zero at x == centre.
    fn evaluate(&self, x: &[f64]) -> f64 {
        x.iter()
            .zip(&self.centre)
            .zip(&self.weights)
            .map(|((xi, ci), wi)| wi * (xi - ci).powi(2))
            .sum()
    }
}
```

> **Python parallel.** This is your `def fitness(x): ...` — and nothing more. The
> only `rlevo`-specific obligations are the **minimisation convention** (return a
> *cost*; negate a reward-style score) and that `x` arrives as `&[f64]`. See the
> [enforcement table](../part-1-foundations/evolutionary-computation/23-fitness.md#where-its-enforced--and-where-its-your-responsibility)
> for why the direction matters.

### Step 2 — Choose the on-ramp

We have a self-contained `f(&[f64]) -> f64`, so the table points straight at
`FromLandscape`. No separate evaluator, no module, no environment — the simplest
adapter applies:

```rust,no_run
use rlevo_evolution::fitness::FromLandscape;

let objective = WeightedSphere {
    weights: vec![1.0, 4.0, 9.0],
    centre:  vec![0.5, -0.5, 1.0],
};
let fitness_fn = FromLandscape::new(objective);   // now an impl BatchFitnessFn
```

That single `::new` is the entire bridge from "a Rust function" to "something the
engine can call on a whole population."

### Step 3 — Hand it to the harness

From here it is identical to the bundled-landscape path in
[Optimising a Function](../part-2-guided-tour/10-optimizing-a-function.md) — the
objective is the only thing that changed:

```rust,no_run
use burn::backend::Flex;
use rlevo_evolution::algorithms::ga::{GaConfig, GeneticAlgorithm};
use rlevo_evolution::strategy::EvolutionaryHarness;

type B = Flex;

let device = Default::default();
let strategy = GeneticAlgorithm::<B>::new();
let config = GaConfig::default_for(/* pop = */ 64, /* dim = */ 3);

let mut harness =
    EvolutionaryHarness::<B, _, _>::new(strategy, config, fitness_fn, 42, device, 200);

harness.reset();
while !harness.step(()).done {}
println!("best = {:.3e}", harness.latest_metrics().unwrap().best_fitness_ever);
```

The harness infers `B` and `G` from the strategy and the adapter, which is why the
turbofish is just `::<B, _, _>` — you never spell out the genome type.

### When the simple on-ramp is the wrong one

The `FromLandscape` path round-trips the population to the host every generation.
That is fine for a cheap formula, but if your objective is itself tensor
arithmetic — a batched matrix expression, a quadratic form — you are paying for a
device transfer you do not need. Then skip the adapter and implement the contract
directly, keeping everything on device:

```rust,no_run
use burn::tensor::{backend::Backend, Tensor};
use rlevo_evolution::fitness::BatchFitnessFn;

struct OnDeviceWeightedSphere<B: Backend> {
    weights: Tensor<B, 1>,   // (genome_dim,)
    centre:  Tensor<B, 1>,   // (genome_dim,)
}

impl<B: Backend> BatchFitnessFn<B, Tensor<B, 2>> for OnDeviceWeightedSphere<B> {
    fn evaluate_batch(&mut self, population: &Tensor<B, 2>, _device: &B::Device) -> Tensor<B, 1> {
        let centre = self.centre.clone().unsqueeze_dim::<2>(0);    // (1, dim) — broadcasts
        let weights = self.weights.clone().unsqueeze_dim::<2>(0);  // (1, dim)
        let centred = population.clone() - centre;                 // (pop, dim)
        let weighted = centred.clone() * centred * weights;        // (pop, dim)
        weighted.sum_dim(1).squeeze_dim::<1>(1)   // (pop,) — one cost per row, order preserved
    }
}
```

Same objective, same minimisation convention, no host round-trip — and the `pop`
rows are scored in a single kernel sweep. The harness cannot tell the difference:
it still just calls `evaluate_batch`.

> **The decision in one line.** Express it on the host and wrap it
> (`FromLandscape`) when the objective is naturally a CPU formula or a simulator;
> implement `BatchFitnessFn` directly when it is naturally tensor math and you want
> the throughput. Everything else — `FromFitnessEvaluable`, `ModuleEvalFn`,
> `RolloutFitness` — is the same choice specialised to objectives that arrive as an
> evaluator, a network, or a policy.

---

## Where this connects

- [Fitness Evaluation](../part-1-foundations/evolutionary-computation/23-fitness.md)
  — the Part I chapter this page expands; start there for the conceptual picture
  and the minimisation contract.
- [The Ask/Tell Contract](ask-tell-contract.md) — how the harness drives the
  `ask` → `evaluate_batch` → `tell` loop, and how to run it by hand.
- [Genome Representation](../part-1-foundations/evolutionary-computation/22-genome.md)
  — what the `G` in `BatchFitnessFn<B, G>` actually is, and the `ParamReshaper`
  bridge that `ModuleEvalFn` and `RolloutFitness` rely on for neuroevolution.

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
