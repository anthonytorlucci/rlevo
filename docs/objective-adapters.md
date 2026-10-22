# Wiring an Objective: Adapters and Evaluators

This page is a map of *every* way to get an objective into `rlevo`, why there is more than one,
and a worked example that takes a brand-new objective from nothing to a running
optimisation.

It exists because the fitness layer is the part of `rlevo` most likely to look
like a thicket of generics on first contact: `BatchFitnessFn<B, G>`,
`FromLandscape<L>`, `ModuleEvalFn`, `RolloutFitness`. The thicket dissolves
once you see that **they are not four competing concepts — they are one trait
and three on-ramps to it.**

> **Coming from Python?** In DEAP, PyGAD, or a hand-rolled numpy loop, "the
> objective" is a single callable: `def fitness(x): return ...`. `rlevo` splits
> that one idea into *a contract* (the trait the engine calls) and *adapters*
> (small wrappers that turn the shape your objective already has into that
> contract). You are never writing five objectives — you are picking which on-ramp
> matches the objective you already have.

> **Running the examples.** They need one dependency, `rlevo` at its default
> features. Burn is reached through it, as `rlevo::burn`.

---

## Part 1 — The decision guide

### The one trait that matters

The engine — specifically the `EvolutionaryLoop` — consumes exactly one trait:

```rust,ignore
pub trait BatchFitnessFn<B: Backend, G>: Send {
    /// Whole population in, one fitness value per row out, in the objective's
    /// own direction: a cost as a cost, a score as a score. Row order is
    /// preserved: `fitness[i]` scores the individual at row `i`.
    fn evaluate_batch(&mut self, population: &G, device: &B::Device) -> Tensor<B, 1>;

    /// Which way is better. Required, with no default, so that a score cannot
    /// be minimised by omission.
    fn sense(&self) -> ObjectiveSense;
}
```

Everything else on this page is a type that *implements* `BatchFitnessFn` so you
do not have to write it by hand. The two generic parameters are less exotic than
they look:

- **`B`** is the Burn **backend** — the compute target (`Flex`, `Wgpu`, `NdArray`).
  It rides along on every tensor type. You name it once, in the loop's turbofish
  (`EvolutionaryLoop::<B, _, _>::new`); the strategy's and the adapter's types are
  inferred from the arguments.
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
| a neural-network **module** plus a way to score it | **`ModuleEvalFn`** | unflattens each genome row into a Burn `Module`, runs your scorer |
| a **policy** to run inside an environment | **`RolloutFitness`** (in `rlevo-hybrid`) | rolls out episodes per genome, returns the natural mean return (declares `Maximize`) |
| a bespoke, fully on-device objective | **`impl BatchFitnessFn` yourself** | the fast path — no host transfer, one kernel sweep |

Two things worth internalising from the table:

- **The first four are convenience; the last is control.** The adapters all do a
  host round-trip (device → CPU → device) per generation. That is the right default
  for an objective you can only express on the CPU (a simulator, a `Vec<f64>`
  formula). When throughput matters and your objective is expressible in tensor
  ops, implement the trait directly and keep the whole generation on device.

### Single-member evaluation, and where it fits

**Memetic local search** scores one candidate at a time: the hill-climbing /
Nelder–Mead / simulated-annealing refiners take their objective as a closure,
`&mut dyn FnMut(&[f32]) -> f32`. **`EvolutionaryLoop` never calls it.**
`MemeticWrapper` builds that closure itself, scoring each probe as a one-row
`evaluate_batch`, so local search and the population loop share a single
objective definition and you write only the batch one. Unless you are writing a
local-search operator of your own, you never write the closure.

### How it all reaches the strategy

Worth restating, because it is the crux: **strategies never see any of these
types.** The data flow is

```text
evo_loop.fitness_fn : impl BatchFitnessFn<B, G>
        │
   strategy.ask ──▶ population (G) ──▶ evaluate_batch ──▶ fitness Tensor<B,1>
                                                              │
                                              strategy.tell(population, fitness)
```

The strategy proposes a population and later consumes a *fitness tensor*. The
adapter you chose lives in the loop, called once per generation between `ask`
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
`Landscape` trait (`rlevo::core::fitness::Landscape`):

```rust,no_run
use rlevo::core::fitness::Landscape;

struct WeightedSphere {
    weights: Vec<f64>,
    centre:  Vec<f64>,
}

impl Landscape for WeightedSphere {
    // Return the natural cost; zero at x == centre. A Landscape is a cost
    // surface, so Landscape::sense() defaults to ObjectiveSense::Minimize.
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
> only `rlevo`-specific obligations are to return the objective's **natural** value
> (you never hand-negate — a `Landscape` is a cost surface, so its `sense()`
> defaults to `ObjectiveSense::Minimize` and the loop reconciles direction) and
> that `x` arrives as `&[f64]`. See
> [The Ask/Tell Contract](ask-tell-contract.md#lineage-evosax-and-where-rlevo-diverges)
> for why the direction matters and exactly where the chokepoint lives
> (`EvolutionaryLoop::step`, which negates a `Minimize` objective before
> `tell` and maps metrics back to your declared sense).

### Step 2 — Choose the on-ramp

We have a self-contained `f(&[f64]) -> f64`, so the table points straight at
`FromLandscape`. No separate evaluator, no module, no environment — the simplest
adapter applies:

```rust,no_run
use rlevo::core::objective::ObjectiveSense;
use rlevo::evo::fitness::FromLandscape;

let objective = WeightedSphere {
    weights: vec![1.0, 4.0, 9.0],
    centre:  vec![0.5, -0.5, 1.0],
};
// `new` would take the landscape's default sense (Minimize); we spell it out so
// the direction is visible at the call site.
let fitness_fn = FromLandscape::with_sense(objective, ObjectiveSense::Minimize);
```

That single constructor is the entire bridge from "a Rust function" to "something
the engine can call on a whole population."

### Step 3 — Hand it to `EvolutionaryLoop`

From here it is identical to the bundled-landscape path in
[The Ask/Tell Contract](ask-tell-contract.md#the-same-run-with-evolutionaryloop) —
the objective is the only thing that changed:

```rust,no_run
use rlevo::burn::backend::Flex;
use rlevo::evo::algorithms::ga::{GaConfig, GeneticAlgorithm};
use rlevo::evo::strategy::EvolutionaryLoop;

type B = Flex;

let device = Default::default();
let strategy = GeneticAlgorithm::<B>::new();
// `default_for` searches [-5.12, 5.12] in every dimension; set `bounds` on
// the config for a different domain, or the optimum may lie outside the box.
let config = GaConfig::default_for(/* pop = */ 64, /* dim = */ 3);

let mut evo_loop =
    EvolutionaryLoop::<B, _, _>::new(strategy, config, fitness_fn, 42, device, 200).expect("valid config");

evo_loop.reset();
while !evo_loop.is_done() {
    evo_loop.step();
}
println!("best = {:.3e}", evo_loop.latest_metrics().unwrap().best_fitness_ever());

// The minimiser itself: the best genome so far, a `(1, genome_dim)` tensor, and
// its cost in the objective's own direction.
if let Some((genome, cost)) = evo_loop.best() {
    let x: Vec<f32> = genome.into_data().into_vec().expect("f32 genome");
    println!("x* = {x:?}, f(x*) = {cost:.3e}");
}
```

You name the backend `B`; the strategy's and the adapter's types are inferred,
which is why the turbofish is just `::<B, _, _>` — you never spell out the
genome type.

`reset()` starts the run, `step()` runs one generation and `is_done()` says
whether the 200-generation budget is spent.
[The Ask/Tell Contract](ask-tell-contract.md#the-same-run-with-evolutionaryloop)
explains the three calls and why they are separate.

### When the simple on-ramp is the wrong one

The `FromLandscape` path round-trips the population to the host every generation.
That is fine for a cheap formula, but if your objective is itself tensor
arithmetic — a batched matrix expression, a quadratic form — you are paying for a
device transfer you do not need. Then skip the adapter and implement the contract
directly, keeping everything on device:

```rust,no_run
use rlevo::burn::tensor::{backend::Backend, Tensor};
use rlevo::core::objective::ObjectiveSense;
use rlevo::evo::fitness::BatchFitnessFn;

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

    // Implementing the trait directly means declaring the sense yourself —
    // there is no Landscape default to inherit. This is a cost surface.
    fn sense(&self) -> ObjectiveSense { ObjectiveSense::Minimize }
}
```

Build it on the device the loop will use, and hand it over exactly as before:

```rust,ignore
let device = Default::default();
let fitness_fn = OnDeviceWeightedSphere::<B> {
    weights: Tensor::from_floats([1.0, 4.0, 9.0], &device),
    centre: Tensor::from_floats([0.5, -0.5, 1.0], &device),
};
```

Same objective, the same declared sense, no host round-trip — and the `pop` rows
are scored in a single kernel sweep. The loop cannot tell the difference: it
still just calls `evaluate_batch` and reads `sense()`.

> **Precision.** Genomes and fitness are `f32` tensors throughout. `FromLandscape`
> widens each gene to `f64` for your function and narrows
> its result back to `f32`, and every metric is an `f32`. A comparison at
> tolerances near `1e-7` relative is measuring the tensor type, not the optimiser.

> **The decision in one line.** Express it on the host and wrap it
> (`FromLandscape`) when the objective is naturally a CPU formula or a simulator;
> implement `BatchFitnessFn` directly when it is naturally tensor math and you want
> the throughput. Everything else — `ModuleEvalFn`, `RolloutFitness` — is the same
> choice specialised to objectives that arrive as a network or a policy.

---

## Where this connects

- [The Ask/Tell Contract](ask-tell-contract.md) — how `EvolutionaryLoop` drives the
  `ask` → `evaluate_batch` → `tell` loop, how to run it by hand, and the
  maximise convention / `ObjectiveSense` contract this page assumes.
- The genome type `G` in `BatchFitnessFn<B, G>` is the bare population
  tensor — `Tensor<B, 2>` for real-valued genomes, `Tensor<B, 2, Int>` for
  binary/integer ones. For weight-only neuroevolution, `ModuleReshaper` is the
  bridge `ModuleEvalFn` and `RolloutFitness` rely on: `flatten` walks a Burn
  `Module`'s float leaves in declaration order into one `Tensor<B, 1>` genome,
  and `unflatten` clones a template module and refills its leaves from a flat
  vector in the same order, so $\text{unflatten}(\text{flatten}(m)) \approx m$.
