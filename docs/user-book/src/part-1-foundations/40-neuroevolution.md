# Neuroevolution

Neuroevolution applies the evolutionary machinery of the previous chapters to the
object this library ultimately cares about — a **neural network** — and evolves it
directly, with no gradients and no backpropagation. It is the bridge between the
two halves of `rlevo`: evolution and reinforcement learning optimise the *same*
object (a policy network), but from opposite directions. Gradient-based RL
differentiates through the network; neuroevolution treats it as a black box and
searches its parameter — or *topology* — space with a population.

That black-box stance is exactly what makes it worth having. Where a policy is
non-differentiable, the reward signal is sparse or deceptive, or the network
*structure* itself is a search variable, a population of networks scored by
episodic return is a credible alternative to gradient descent. Moriarty and
Miikkulainen (1996) [[Moriarty and Miikkulainen, 1996]](#bibliography) first
showed GAs evolving network weights for control; two decades later Such et al.
(2017) [[Such et al., 2017]](#bibliography) demonstrated that a plain GA on the
weights — no gradients at all — matches DQN and A3C on many Atari games while
training far faster thanks to trivial parallelism. Neuroevolution is a serious
competitor to gradient-based RL, not a niche curiosity.

`rlevo` ships three approaches, distinguished by *how much structure is fixed*
before the search begins: fix the architecture and evolve only weights
(`WeightOnly`), choose among a menu of architectures while evolving their weights
(`ArchNas`), or grow the topology itself from a minimal seed (`Neat`). All three
live under `rlevo::evo::algorithms::neuroevolution`.

## Evolving weights: `WeightOnly`

The simplest approach fixes the architecture and evolves the flattened weight
vector. Because the genome is then just a real vector, **any** flat-genome
strategy from the previous chapters can drive it — GA, classical ES, CMA-ES, or
DE — with no change to the strategy itself.

`WeightOnly<B, S, M>` is a thin adapter: it wraps an inner strategy `S` and a
template Burn module `M`, and is itself a `Strategy<B>`.

```rust
pub struct WeightOnly<B, S, M> { /* inner strategy + ModuleReshaper */ }

// S: Strategy<B, Genome = Tensor<B, 2>>,  M: Module<B>
let strategy = WeightOnly::new(GeneticAlgorithm::<B>::new(), template);
let n = strategy.num_params(); // flat parameter count, read from the template
```

The bridge between a network's nested parameter tree and a flat genome row is the
`ParamReshaper` trait (concrete impl `ModuleReshaper<B, M>`), the same bridge
introduced in the [genome chapter](evolutionary-computation/22-genome.md). It is a
two-way map:

- `flatten(module, device) -> Tensor<B, 1>` walks the module's float leaves in a
  deterministic order and concatenates them into one vector;
- `unflatten(flat) -> Module` clones the template and writes the slices back into
  the matching leaves.

Leaf order is kept consistent by Burn's `#[derive(Module)]` `visit`/`map`
traversals, so flatten and unflatten always agree. Crucially, `WeightOnly`'s
`init`/`ask`/`tell` delegate *verbatim* to the inner strategy — the population
stays a flat `Tensor<B, 2>` of shape `(pop_size, num_params)` end to end, and the
reshaping into runnable modules happens **only at the fitness boundary**, inside
`ModuleEvalFn`. Evolution never pays for a tensor→module round-trip except where
it must: at evaluation. The whole thing runs through the standard
`EvolutionaryHarness` like any other strategy.

## Evolution strategies as a scalable alternative to RL

The weight-only view leads directly to one of the results that put neuroevolution
back on the map. Salimans et al. (2017) [[Salimans et al., 2017]](#bibliography)
reframed RL as black-box optimisation over the policy parameters \\(\theta\\) and
estimated a search gradient purely from returns:

```math
\nabla_\theta \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)}
  \left[ F(\theta + \sigma \epsilon) \right]
  \approx \frac{1}{n\sigma} \sum_{i=1}^n F(\theta + \sigma \epsilon_i)\, \epsilon_i,
```

where \\(F(\theta)\\) is the total undiscounted return of a rollout. No
backpropagation appears — only function evaluations. With antithetic sampling
(evaluating \\(\epsilon\\) and \\(-\epsilon\\) in pairs) and virtual batch
normalisation, OpenAI ES matched A3C on MuJoCo locomotion using a thousand CPUs
with near-linear speedup. The lesson is a systems one: at scale, broadcasting
scalar returns across workers can be cheaper than exchanging gradients, and the
embarrassingly parallel population evaluation that `WeightOnly` performs is what
makes that trade pay off.

## Searching architectures: `ArchNas`

Fixing the architecture begs the question of *which* architecture. `ArchNas`
co-evolves that choice alongside the weights — a lightweight neural architecture
search over a **closed menu** of fixed-topology variants the user declares up
front.

Variants are registered through `ArchNasBuilder::add_variant`, each paired with
its own scorer; `build` returns the `NasParams` config and an `ArchNasFitnessFn`:

```rust
let mut builder = ArchNasBuilder::<B>::new();
builder
    .add_variant(ShallowMlp::new(&device), |m| shallow_scorer(m))
    .add_variant(DeepMlp::new(&device),    |m| deep_scorer(m));
let (params, fitness) = builder.build(config);
```

The genome, `NasGenome<B>`, carries a categorical choice per individual plus a
weight matrix:

- `arch_ids: Vec<usize>` — each individual's architecture, an index into the menu;
- `weights: Tensor<B, 2>` of shape `(pop_size, max_param_count)`, zero-padded
  beyond the active variant's parameter count so a ragged population still rides a
  single rectangular tensor.

The load-bearing invariant is that **`arch_id` equals the registration index** in
both the strategy state and the fitness evaluator, because both are produced from
the same ordered variant list — so row `i` is always scored by the network it
actually encodes. `ArchNasStrategy` runs an elitist tournament loop: same-variant
parents blend their weights, cross-variant pairs fall back to copying, and an
individual either mutates its architecture (re-initialising weights for the new
shape) or perturbs its existing weights. Like the rest of the engine it
**minimises**, so wire the scorer as a cost. `ArchNas` drives its own
`init`/`ask`/`tell` loop directly rather than going through `EvolutionaryHarness`,
because its genome is not a bare tensor.

## Growing topologies: `Neat`

The most ambitious approach evolves the network *structure* itself. **NEAT**
(NeuroEvolution of Augmenting Topologies; Stanley and Miikkulainen, 2002)
[[Stanley and Miikkulainen, 2002]](#bibliography) starts every run from a minimal
fully-connected input→output graph and grows complexity only as it earns its
keep. `rlevo` implements it as `NeatStrategy<B>`, configured by
`NeatParams::default_for(pop_size, num_inputs, num_outputs)`, which fills in the
canonical defaults (compatibility coefficients \\(c_1 = c_2 = 1.0\\),
\\(c_3 = 0.4\\); compatibility threshold \\(3.0\\); stagnation limit \\(15\\)
generations; survival fraction \\(0.2\\)).

NEAT departs from every other strategy in the library in two structural ways, and
both are deliberate:

- **The genome is a graph, host-side, not a device tensor.** `TopologyGenome`
  holds a `Vec<NodeGene>` and a `Vec<ConnectionGene>` kept sorted by *innovation
  number*. There is no rank-2 population tensor here — variable-topology genomes
  do not fit that mould — so NEAT carries a `Vec<TopologyGenome>` and is evaluated
  through a separate `GraphFitnessFn` seam rather than the tensor `BatchFitnessFn`.
  A `PhenotypeBuilder` turns each genome into a runnable network: `InterpretedBuilder`
  is the reference per-genome evaluator, `DensePaddedEvaluator` the batched
  population-level one.
- **NEAT maximises.** Higher fitness is better here, the opposite of the engine's
  global minimise-cost convention — a direct consequence of NEAT's fitness-sharing
  and speciation arithmetic, which are framed in terms of reward.

Two ideas make topology evolution work, and both hinge on **innovation numbers** —
historical markers that timestamp when a given structural element first appeared:

- **Crossover alignment.** When two networks of different shapes recombine, genes
  that share an innovation number are *homologous* and inherited from either
  parent; disjoint and excess genes are taken from the fitter parent. Without the
  historical marking there is no principled way to line up two different graphs.
  A shared, per-run `Arc<InnovationRegistry>` hands out these numbers and caches
  them, so the same structural mutation (the same `(source, target)` edge, or the
  same connection split) always receives the same id across the whole population.
- **Speciation.** Structural innovations are initially unfit — a freshly added
  node rarely helps before its weights are tuned. NEAT protects them by grouping
  genomes into species via a `compatibility_distance` over excess genes, disjoint
  genes, and mean matching-weight difference, then sharing fitness within each
  species so a new structure competes against its own kind rather than the whole
  population. Stagnant species (no improvement within the stagnation limit) are
  culled, with the top performers protected from a population wipeout.

Structural variation comes from two mutations layered on top of ordinary weight
perturbation: *add-connection* wires two previously unconnected nodes, and
*add-node* splits an existing connection into two, inserting a node in the middle
in a function-preserving way. Both register their innovations through the shared
registry, and the feedforward (acyclic) invariant is maintained across enabled
*and* disabled edges so toggling a connection can never introduce a cycle.

## From evolving networks to combining paradigms

Everything in this chapter is *pure* neuroevolution: evolution alone, optimising a
network. It is one half of what `rlevo` is for. The other half is gradient-based
reinforcement learning — and the most interesting algorithms run **both at once**,
letting a population explore while gradients refine. Why you would do that, and
how the `Strategy`, `Environment`, and replay-buffer seams make it possible, is
the subject of [Why Combine Them?](50-why-combine.md).

<a name="bibliography"></a>
*References on this page are collected in the [Bibliography](../bibliography.md).*

---

*Co-Authored-By: Anthropic Claude Opus 4.8*\
*Reviewed-By: (Human) Anthony Torlucci*
