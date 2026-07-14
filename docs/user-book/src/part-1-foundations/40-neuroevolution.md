# Neuroevolution

The last two chapters described two ways to get better at a task — a population
that searches by selection, and an agent that learns from the fine structure of a
single trajectory. **Neuroevolution** is where the first one is pointed straight
at the object this whole library exists to train: a **neural network**. It takes
the evolutionary machinery wholesale and evolves the network directly — no
gradients, no backpropagation.

That makes it the bridge between `rlevo`'s two halves. Evolution and
reinforcement learning optimise the *same* object — a policy network — but from
opposite directions: gradient-based RL differentiates *through* the network,
while neuroevolution treats it as a black box and searches its parameter (or
*topology*) space with a population. The black-box stance is exactly what earns
it a place. When a policy is non-differentiable, when the reward is sparse or
deceptive, or when the network *structure* is itself a thing you want to search,
a population of networks scored by episodic return is a credible rival to gradient
descent. This is not a niche curiosity: Moriarty and Miikkulainen (1996)
[[Moriarty and Miikkulainen, 1996]](#bibliography) evolved network weights
for control, and two decades on Such et al. (2017)
[[Such et al., 2017]](#bibliography) showed a *plain GA on the weights* — no
gradients at all — matching DQN and A3C on many Atari games while training far
faster, thanks to how trivially a population parallelises.

The thread for this chapter is a single dial: **how much structure do you fix
before the search begins, and how much do you let it discover?** `rlevo` ships
three approaches arranged along exactly that dial — fix the architecture and
evolve only the weights (`WeightOnly`), choose among a menu of architectures
while evolving their weights (`ArchNas`), or grow the topology itself from a
minimal seed (`Neat`). Each rung hands the search more freedom and asks more of
its machinery. All three live under
`rlevo::evo::algorithms::neuroevolution`, and we climb them in order.

## Evolving weights: `WeightOnly`

Start at the bottom rung, where the most is fixed. Pin the architecture down
completely and evolve only the flattened weight vector. The pedagogical payoff of
starting here is that, once the genome is just a real vector, **nothing new is
needed** — *any* flat-genome strategy from the evolution chapter can drive it (GA,
classical ES, CMA-ES, DE) with no change to the strategy itself. Neuroevolution,
at this rung, is the evolution you already know with a network bolted onto the
fitness function.

`WeightOnly<B, S, M>` is the thin adapter that makes that literally true: it wraps
an inner strategy `S` and a template Burn module `M`, and is itself a
`Strategy<B>`.

```rust
pub struct WeightOnly<B, S, M> { /* inner strategy + ModuleReshaper */ }

// S: Strategy<B, Genome = Tensor<B, 2>>,  M: Module<B>
let strategy = WeightOnly::new(GeneticAlgorithm::<B>::new(), template);
let n = strategy.num_params(); // flat parameter count, read from the template
```

The one genuinely new piece is the bridge between a network's nested parameter
tree and a flat genome row — the `ParamReshaper` trait (concrete impl
`ModuleReshaper<B, M>`), the same bridge introduced in the [genome
chapter](evolutionary-computation/22-genome.md). It is a two-way map:

- `flatten(module, device) -> Tensor<B, 1>` walks the module's float leaves in a
  deterministic order and concatenates them into one vector;
- `unflatten(flat) -> Module` clones the template and writes the slices back into
  the matching leaves.

Burn's `#[derive(Module)]` `visit`/`map` traversals keep leaf order consistent,
so `flatten` and `unflatten` always agree. The design choice worth noticing is
*where* the bridge runs: `WeightOnly`'s `init`/`ask`/`tell` delegate **verbatim**
to the inner strategy — the population stays a flat `Tensor<B, 2>` of shape
`(pop_size, num_params)` end to end — and the reshaping into runnable modules
happens **only at the fitness boundary**, inside `ModuleEvalFn`. Evolution never
pays for a tensor→module round-trip except where it truly must, at evaluation, so
the whole thing runs through the standard `EvolutionaryHarness` like any other
strategy.

Made concrete: a `GeneticAlgorithm` driving a `1→16→1` MLP — 49 flattened
parameters, the count `ModuleReshaper` reads straight off the template — descends
an MSE fitness toward a noisy sine target, the population's best error improving
generation over generation with no gradient ever computed. [The Parameter Bridge:
Weights as a Flat Genome](neuroevolution/41-param-bridge.md) develops the adapter
layer this rests on — the fitness-boundary confinement, the loop-over-N
evaluation, and how the same bridge generalises to `ArchNas` one rung up.

## Evolution strategies as a scalable alternative to RL

Before climbing higher, it is worth pausing on what the weight-only view bought
us, because it produced one of the results that put neuroevolution back on the
map. Salimans et al. (2017) [[Salimans et al., 2017]](#bibliography) reframed RL
itself as black-box optimisation over the policy parameters \\(\theta\\), and
estimated a search gradient purely from returns:

```math
\nabla_\theta \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)}
  \left[ F(\theta + \sigma \epsilon) \right]
  \approx \frac{1}{n\sigma} \sum_{i=1}^n F(\theta + \sigma \epsilon_i)\, \epsilon_i,
```

where \\(F(\theta)\\) is the total undiscounted return of a rollout. No
backpropagation appears anywhere — only function evaluations. With antithetic
sampling (evaluating \\(\epsilon\\) and \\(-\epsilon\\) in pairs) and virtual
batch normalisation, OpenAI ES matched A3C on MuJoCo locomotion using a thousand
CPUs with near-linear speedup. The lesson is a *systems* one, and it is why we
care about the fitness-boundary confinement above: at scale, broadcasting scalar
returns across workers can be cheaper than exchanging gradients, and the
embarrassingly parallel population evaluation `WeightOnly` performs is exactly
what makes that trade pay off.

## Searching architectures: `ArchNas`

Fixing the architecture only raises the next question — *which* architecture? The
second rung, `ArchNas`, refuses to answer it by hand and co-evolves that choice
alongside the weights: a lightweight neural architecture search over a **closed
menu** of fixed-topology variants you declare up front. You still bound the search
(it cannot invent a topology that is not on the menu), but you no longer have to
pick the winner yourself.

You register variants through `ArchNasBuilder::add_variant`, each paired with its
own scorer; `build` returns the `NasParams` config and an `ArchNasFitnessFn`:

```rust
let mut builder = ArchNasBuilder::<B>::new();
builder
    .add_variant(ShallowMlp::new(&device), |m| shallow_scorer(m))
    .add_variant(DeepMlp::new(&device),    |m| deep_scorer(m));
let (params, fitness) = builder.build(config);
```

The interesting engineering problem here is that the population is now *ragged* —
different individuals encode different-sized networks — yet we still want it to
ride a single rectangular tensor for throughput. The genome, `NasGenome<B>`,
solves that by carrying a categorical choice per individual plus a padded weight
matrix:

- `arch_ids: Vec<usize>` — each individual's architecture, an index into the menu;
- `weights: Tensor<B, 2>` of shape `(pop_size, max_param_count)`, zero-padded
  beyond the active variant's parameter count so a ragged population still fits one
  rectangular tensor.

A three-variant MLP menu at depths `2→8→1`, `2→16→8→1`, and `2→32→16→8→1`, for
instance, registers `per_variant_params` of 33, 193, and 769; every row of the
population tensor is sized to the `max_param_count` of 769 regardless of which
variant it currently encodes, with the tail zero-padded.

The load-bearing invariant is that **`arch_id` equals the registration index** in
both the strategy state and the fitness evaluator — both are produced from the
same ordered variant list — so row `i` is always scored by the network it
actually encodes. Get that wrong and individuals would be graded as networks they
are not. `ArchNasStrategy` then runs an elitist tournament loop: same-variant
parents blend their weights, cross-variant pairs fall back to copying, and an
individual either mutates its architecture (re-initialising weights for the new
shape) or perturbs its existing weights. Like the rest of the engine it
**maximises** a canonical fitness, so its scorer returns a value where higher is
better — and because `ArchNas` drives its own `init`/`ask`/`tell` loop directly
rather than going through `EvolutionaryHarness`, there is no chokepoint to
reconcile a cost, so you wire a cost scorer as `−cost` yourself. Its genome is not
a bare tensor, which is precisely why it bypasses the harness.

## Growing topologies: `Neat`

The top rung fixes the least and discovers the most: it evolves the network
*structure* itself, with no menu at all. **NEAT** (NeuroEvolution of Augmenting
Topologies; Stanley and Miikkulainen, 2002)
[[Stanley and Miikkulainen, 2002]](#bibliography) starts every run from a minimal
fully-connected input→output graph and grows complexity only as it earns its
keep. `rlevo` implements it as `NeatStrategy<B>`, configured by
`NeatParams::default_for(pop_size, num_inputs, num_outputs)`, which fills in the
canonical defaults (compatibility coefficients \\(c_1 = c_2 = 1.0\\),
\\(c_3 = 0.4\\); compatibility threshold \\(3.0\\); stagnation limit \\(15\\)
generations; survival fraction \\(0.2\\)).

Granting the search this much freedom forces NEAT to break from every other
strategy in the library in two structural ways, and both are deliberate
consequences of a genome that no longer has a fixed shape:

- **The genome is a graph, host-side, not a device tensor.** `TopologyGenome`
  holds a `Vec<NodeGene>` and a `Vec<ConnectionGene>` kept sorted by *innovation
  number*. There is no rank-2 population tensor here — variable-topology genomes
  do not fit that mould — so NEAT carries a `Vec<TopologyGenome>` and is evaluated
  through a separate `GraphFitnessFn` seam rather than the tensor
  `BatchFitnessFn`. The graph data model and the innovation-number bookkeeping
  that recombines two differently-shaped networks are the subject of [The NEAT
  Genome and Innovation Numbers](neuroevolution/42-neat-genome.md). Genomes reach
  a forward pass through one of two seams — a per-genome `PhenotypeBuilder` and a
  population-batched `BatchPhenotypeEvaluator` that agree to float epsilon, so the
  choice between them is purely about throughput. [Phenotypes: Compiling a Genome
  into a Network](neuroevolution/44-phenotypes.md) develops both, including why a
  NEAT phenotype carries no Burn `Module` at all.
- **NEAT maximises.** Higher fitness is better here — the same direction as the
  rest of the engine's canonical-maximise convention. Its fitness-sharing and
  speciation arithmetic additionally assume *non-negative* fitness, a
  NEAT-specific precondition orthogonal to objective sense (a cost is reconciled
  into canonical space by the harness/adapter chokepoint, not by hand here).

What actually makes topology evolution work is two ideas, and both hinge on
**innovation numbers** — historical markers that timestamp when a given structural
element first appeared:

- **Crossover alignment.** When two networks of different shapes recombine, genes
  that share an innovation number are *homologous* and inherited from either
  parent; disjoint and excess genes are taken from the fitter parent. Without the
  historical marking there is simply no principled way to line up two different
  graphs. A shared, per-run `Arc<InnovationRegistry>` hands out these numbers and
  caches them, so the same structural mutation (the same `(source, target)` edge,
  or the same connection split) always receives the same id across the whole
  population. [The NEAT Genome and Innovation
  Numbers](neuroevolution/42-neat-genome.md) works through the registry and the
  matching/disjoint/excess crossover it enables.
- **Speciation.** A freshly added node rarely helps before its weights are tuned,
  so structural innovations start out *unfit* and would be culled immediately by a
  population that judged them on raw fitness. NEAT protects them by grouping
  genomes into species via a `compatibility_distance` over excess genes, disjoint
  genes, and mean matching-weight difference, then sharing fitness within each
  species so a new structure competes against its own kind rather than the whole
  population. Stagnant species (no improvement within the stagnation limit) are
  culled, with the top performers shielded from a population wipeout. [Speciation:
  Protecting Structural Innovation](neuroevolution/43-speciation.md) works through
  the distance, fitness sharing, and offspring apportionment.

Structural variation itself comes from two mutations layered on top of ordinary
weight perturbation: *add-connection* wires two previously unconnected nodes, and
*add-node* splits an existing connection in two, inserting a node in the middle in
a function-preserving way. Both register their innovations through the shared
registry, and the feedforward (acyclic) invariant is maintained across enabled
*and* disabled edges, so toggling a connection can never sneak in a cycle.

Here is the detail that makes NEAT click as a *system* rather than two bolted-on
mechanisms: growing the topology and splitting the population into species are
**coupled**, not independent. XOR, the canonical proving task, is not linearly
separable, so it cannot be solved without at least one add-node mutation — and
that same structural divergence is exactly what `compatibility_distance` reads to
spin off a new species. On XOR, then, *reaching a solution* and *sustaining more
than one species* are two views of the same event, which is why the
structural-mutation rates (`p_add_node`, `p_add_connection`) are the knobs worth
tuning; `NeatParams::default_for` only seeds canonical starting values.

## From evolving networks to combining paradigms

Everything in this chapter has been *pure* neuroevolution — evolution alone,
optimising a network, all the way up the ladder from fixed weights to grown
topologies. That is one half of what `rlevo` is for. The other half is
gradient-based reinforcement learning, and the most interesting algorithms refuse
to choose: they run **both at once**, letting a population explore while gradients
refine. Why you would want that, and how the `Strategy`, `Environment`, and
replay-buffer seams make it possible, is the subject of [Why Combine
Them?](50-why-combine.md).

<a name="bibliography"></a>
*References on this page are collected in the [Bibliography](../bibliography.md).*

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
