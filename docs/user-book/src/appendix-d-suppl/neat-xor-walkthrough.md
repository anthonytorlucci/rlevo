# Worked Example: NEAT on XOR

The [neuroevolution chapter](../part-1-foundations/40-neuroevolution.md) and its
four sections develop NEAT one mechanism at a time — the
[graph genome and innovation numbers](../part-1-foundations/neuroevolution/42-neat-genome.md),
the [speciation](../part-1-foundations/neuroevolution/43-speciation.md) that
protects new structure, and the
[two phenotype seams](../part-1-foundations/neuroevolution/44-phenotypes.md) that
run a genome. This page assembles all of them into one runnable program: evolving
a network that computes **XOR**, from an empty population to a verified solver. It
is the appendix-tier complement to those sections — where they explain *why* each
piece exists, this shows the whole machine turning over.

XOR is the canonical NEAT proving task for a precise reason, and that reason is
the spine of this walkthrough: it is **not linearly separable**, so no
perceptron — no direct input→output network — can solve it. A solution *requires*
at least one hidden node, which means a solution requires an **add-node
mutation**. And the same structural divergence that an add-node introduces is
exactly what the [compatibility distance](../part-1-foundations/neuroevolution/43-speciation.md)
reads to split off a new species. So on XOR, *reaching a solution* and *sustaining
more than one species* are not two goals — they are two views of the same event.
The complete program is the `test_neat_solves_xor_with_speciation` integration
test; the sections below take it apart.

## The problem and its fitness

XOR maps two binary inputs to one output: `0 ⊕ 0 = 0`, `0 ⊕ 1 = 1`, `1 ⊕ 0 = 1`,
`1 ⊕ 1 = 0`. The four input rows are a single `[4, 2]` batch, the four targets a
length-4 vector:

| `in₀` | `in₁` | target |
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

NEAT [maximises](../part-1-foundations/neuroevolution/43-speciation.md), so the
objective must be framed as a reward, higher-is-better. The standard XOR fitness
is the squared-error gap below a perfect score of four — one point per input row,
less the sum of squared errors:

```math
f = 4 - \sum_{i=1}^{4} (o_i - t_i)^2
```

A perfect solver scores exactly \\(4.0\\); the run treats \\(f \ge 3.9\\) as
solved, leaving a small tolerance for the steepened-sigmoid output never quite
saturating to a hard `0`/`1`. Because every term is squared, \\(f\\) is
non-negative across the whole population — which matters, because the
fitness-sharing and offspring-apportionment arithmetic of speciation
[assumes non-negative fitness](../part-1-foundations/neuroevolution/43-speciation.md).

## Implementing the fitness seam

NEAT does not score genomes through the tensor `BatchFitnessFn` the rectangular
strategies use — its genomes are graphs, so it scores them through the
`GraphFitnessFn` seam, which hands you a phenotype *builder* and lets you decide
how to turn each genome into numbers:

```rust
pub trait GraphFitnessFn<B: Backend>: Send + Sync {
    /// One maximisation fitness per genome, in order. Build each genome's
    /// network with `builder` on `device`.
    fn evaluate(
        &self,
        population: &[TopologyGenome],
        builder: &dyn PhenotypeBuilder<B>,
        device: &B::Device,
    ) -> Vec<f32>;
}
```

The XOR objective holds its input batch and targets, builds each genome into an
interpreted phenotype, runs the forward pass, and reduces the four outputs against
the four targets:

```rust
struct XorFitness {
    inputs: Tensor<B, 2>,   // the [4, 2] XOR input batch
    targets: [f32; 4],      // [0.0, 1.0, 1.0, 0.0]
}

impl GraphFitnessFn<B> for XorFitness {
    fn evaluate(
        &self,
        population: &[TopologyGenome],
        builder: &dyn PhenotypeBuilder<B>,
        device: &B::Device,
    ) -> Vec<f32> {
        population
            .iter()
            .map(|genome| {
                let phenotype = builder.build(genome, device);
                let out = phenotype.forward(self.inputs.clone());
                let values = out.into_data().into_vec::<f32>().unwrap();
                let sse: f32 = values
                    .iter()
                    .zip(self.targets.iter())
                    .map(|(o, t)| (o - t) * (o - t))
                    .sum();
                4.0 - sse
            })
            .collect()
    }
}
```

The `builder` passed in is the harness's chosen
[phenotype builder](../part-1-foundations/neuroevolution/44-phenotypes.md) — in
this run the per-genome `InterpretedBuilder`, which compiles each `TopologyGenome`
into a host-side feedforward evaluator. The fitness function never names a concrete
builder; it programs against the trait, so the same objective works unchanged
whichever builder the harness supplies.

## Configuring the run

`NeatParams::default_for(pop_size, num_inputs, num_outputs)` fills in the canonical
defaults from Stanley and Miikkulainen (2002)
[[Stanley and Miikkulainen, 2002]](../bibliography.md):

| Parameter | Default | Role |
| --- | --- | --- |
| `c1`, `c2`, `c3` | `1.0`, `1.0`, `0.4` | excess / disjoint / weight coefficients in the distance |
| `compat_threshold` | `3.0` | distance below which a genome joins a species |
| `stagnation_limit` | `15` | generations of no improvement before a species is culled |
| `survival_threshold` | `0.2` | top fraction of a species eligible to breed |
| `p_add_node` | `0.03` | per-genome add-node rate |
| `p_add_connection` | `0.05` | per-genome add-connection rate |

These canonical values are tuned for larger, longer runs. For a fast, reliable XOR
demonstration the example raises the structural-mutation rates and lowers the
species threshold — and the reasoning is the coupling described above:

```rust
fn xor_params(pop: usize) -> NeatParams {
    let mut params = NeatParams::default_for(pop, 2, 1);
    params.p_add_node = 0.15;          // 5× the default: surface a hidden node early
    params.p_add_connection = 0.3;     // 6× the default: enrich the topology faster
    params.p_toggle_enable = 0.03;
    params.compat_threshold = 2.5;     // below 3.0: split species more readily
    params.weight_perturb_std = 0.8;   // wider weight search per mutation
    params
}
```

Raising `p_add_node` makes the one mutation XOR *requires* appear within a small
generation budget rather than after hundreds of generations of waiting; lowering
`compat_threshold` to `2.5` means the structural divergence that mutation creates
crosses the species boundary sooner. The chapter's claim that the
structural-mutation rates and the compatibility threshold are *the knobs worth
tuning* is exactly this configuration in practice — every other field keeps its
canonical default.

## The generational loop

NEAT's harness, `NeatStrategy`, deliberately does **not** implement the crate's
`Strategy` trait — its genome is a `Vec<TopologyGenome>`, not a tensor — so the run
is an explicit `init → loop { ask → evaluate → tell }`, threading the `NeatState`
by hand:

```rust
let device = Default::default();
let params = xor_params(150);
let strat = NeatStrategy::<B>::new();
let builder = InterpretedBuilder;
let fitness_fn = XorFitness::new(&device);

let mut rng = StdRng::seed_from_u64(42);
let mut state = strat.init(&params, &mut rng, &device);

for _ in 0..MAX_GENERATIONS {
    let (population, next) = strat.ask(&params, &state, &mut rng);
    let fitness = fitness_fn.evaluate(&population, &builder, &device);
    state = strat.tell(&params, population, fitness, next, &mut rng);

    if state.best_fitness >= SOLVE_THRESHOLD { /* solved */ }
}
```

Each of the three calls is one of the mechanisms from the chapter sections, and
seeing them in sequence is the point of this page:

- **`init`** creates the per-run `Arc<InnovationRegistry>` and `pop_size` minimal
  genomes — two inputs fully connected to one output, *no hidden node* — all
  sharing the same
  [innovation ids](../part-1-foundations/neuroevolution/42-neat-genome.md) but with
  per-individual random weights. The population starts as simple as it can be.
- **`ask`** reproduces last generation's
  [species](../part-1-foundations/neuroevolution/43-speciation.md): it prunes
  stagnant species, apportions offspring by size-adjusted fitness, copies each
  large species' champion through unchanged, and breeds the rest with crossover and
  the structural and weight mutations. This is where add-node fires.
- **`tell`** installs the scores and
  [re-speciates](../part-1-foundations/neuroevolution/43-speciation.md): it
  partitions the new population by compatibility distance, updates each species'
  best fitness and stagnation clock, and records the global champion. This is where
  a freshly grown topology either founds or joins a species.

The harness call between `ask` and `tell` is *your* code — `fitness_fn.evaluate`,
which [builds each phenotype](../part-1-foundations/neuroevolution/44-phenotypes.md)
and scores it. The harness owns reproduction and speciation; you own evaluation.

## Why solving and speciating are one event

The run asserts two things, and the example stops only once *both* hold:

1. **Solve** — `best_fitness` reaches \\(\ge 3.9\\) within 300 generations.
2. **Speciation** — more than one species exists for more than half the
   generations run.

These look independent but are mechanically the same event, and tracing why is the
real lesson of XOR. The seed population is all minimal direct-connection networks;
they are structurally identical, so the compatibility distance between any two is
just their mean weight difference — small — and they all fall into **one species**.
A direct network cannot represent XOR, so fitness plateaus below the solve
threshold. The only escape is an **add-node mutation**: splitting a connection
inserts a hidden node, the single structural element XOR needs. The moment that
node appears, the mutant's genome carries excess and disjoint genes the rest of the
population lacks — its compatibility distance to the seed representatives jumps past
`compat_threshold`, and it **founds a second species**. So the first genome capable
of *progressing toward a solution* is, by construction, the first genome to
*create a second species*. AC-solve and AC-speciation are two readings of the same
add-node event, which is precisely why XOR is the canonical NEAT task: it cannot be
solved without exercising the topology-growth-plus-speciation machinery that is the
whole point of the algorithm.

## Reading and verifying the champion

`best` returns the highest-fitness genome seen across the whole run, not merely the
current generation's best:

```rust
let (best_genome, best_fitness) = strat.best(&state).expect("a best genome exists");

// Re-run the champion through the fitness function to confirm it really solves XOR.
let recomputed = fitness_fn.evaluate(std::slice::from_ref(best_genome), &builder, &device);
assert!(recomputed[0] >= SOLVE_THRESHOLD, "champion re-evaluates as a solver");
```

Re-evaluating the recovered champion is a genuine check, not a formality: it
confirms the genome the harness *stored* as best reproduces the fitness the harness
*recorded* for it — that the phenotype build is deterministic and the champion is a
real solver, not a logging artefact.

## Reproducibility

Two runs with the same seed produce identical results — same best fitness, same
species count, same generation count:

```rust
let run = || {
    let mut rng = StdRng::seed_from_u64(7);
    let mut state = strat.init(&params, &mut rng, &device);
    for _ in 0..30 {
        let (population, next) = strat.ask(&params, &state, &mut rng);
        let fitness = fitness_fn.evaluate(&population, &builder, &device);
        state = strat.tell(&params, population, fitness, next, &mut rng);
    }
    (state.best_fitness, state.species.len(), state.generation)
};
assert_eq!(run().1, run().1, "species count is reproducible");
```

This determinism is not luck; it is the
[innovation registry](../part-1-foundations/neuroevolution/42-neat-genome.md) and
the seed-stream convention working together. Structural mutations are applied
sequentially host-side under seeded substreams, so the order in which innovation
ids are first issued is seed-fixed, and the registry's caches make the *result* of
a repeated mutation order-independent. Same seed, same mutation sequence, same
genomes — generation after generation.

## Swapping in the batched evaluator

Because the objective programs against `GraphFitnessFn` and not a concrete builder,
the same XOR fitness can be driven through the
[batched evaluation seam](../part-1-foundations/neuroevolution/44-phenotypes.md)
instead. `BatchGraphFitness` wraps a `DensePaddedEvaluator`, runs the whole
population through one device-resident forward pass, and reduces each genome's
output slab with the same `4 − Σ(out − target)²` closure:

```rust
let batched = BatchGraphFitness::new(
    DensePaddedEvaluator::default(),
    obs,
    move |slab: &[f32]| {
        let sse: f32 = slab.iter().zip(targets.iter())
            .map(|(o, t)| (o - t) * (o - t)).sum();
        4.0 - sse
    },
);
```

The `test_batch_graph_fitness_matches_interpreted_fitness` test confirms this
produces the *same* fitness vector as the interpreted path to float epsilon — so on
XOR, where networks stay tiny, the choice is purely about throughput and changes
nothing about the run's outcome. As the
[phenotypes section](../part-1-foundations/neuroevolution/44-phenotypes.md)
explains, the batched path only begins to pay once the topologies grow wide; at XOR
scale the two run at par, and a generation's cost is dominated by host-side
reproduction, not the forward pass.

## Where to go next

This example is pure neuroevolution end to end: a population of graphs, scored by
their behaviour, growing the one hidden node XOR demands and speciating the moment
it appears. The same `GraphFitnessFn` seam scales to any task expressible as a
forward pass over an observation batch — swap the inputs, targets, and reduction,
and the harness, speciation, and phenotype machinery are unchanged. For the
weight-only and architecture-search approaches that share this chapter, the
[parameter bridge](../part-1-foundations/neuroevolution/41-param-bridge.md) is the
analogous entry point; for combining a population with gradient-based refinement,
[Why Combine Them?](../part-1-foundations/50-why-combine.md) is where the two halves
of `rlevo` meet.

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
