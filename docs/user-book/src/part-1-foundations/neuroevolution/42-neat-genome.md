# The NEAT Genome and Innovation Numbers

The [neuroevolution overview](../40-neuroevolution.md) made one claim about NEAT
in passing and then moved on: its genome *is a graph, host-side, not a device
tensor*. That single sentence is the reason NEAT looks unlike every other
strategy in `rlevo`, and it is worth unpacking properly. This section is about
the **representation layer** of topology-evolving neuroevolution: how a network's
structure is written down as genes, why those genes carry *innovation numbers*,
and how that one bookkeeping device makes recombining two differently-shaped
networks a principled operation rather than a guess.

The reader is assumed to know NEAT at the level of the original paper (Stanley
and Miikkulainen, 2002) [[Stanley and Miikkulainen, 2002]](../../bibliography.md) —
that it grows topologies from a minimal seed, that it protects innovations by
speciation, that it aligns genes by historical markings. What that paper does not
tell you is how any of it is realised in this crate. That is the gap this section
fills. Speciation gets its [own section](43-speciation.md); here the subject is
the genome and the markings crossover reads.

## Genotype and phenotype are separate objects

Every classical strategy in the previous chapters keeps its whole population in a
single rank-2 tensor — the [genome chapter](../evolutionary-computation/22-genome.md)
made the population *be* the matrix and a genome *be* a row. Variable-topology
networks do not fit that mould: two genomes in the same population have different
numbers of nodes and edges, so there is no rectangular `(pop_size, num_params)`
shape to pack them into. NEAT therefore steps off the device entirely. The
genotype is plain host-side data:

```rust
pub struct TopologyGenome {
    pub nodes: Vec<NodeGene>,
    pub connections: Vec<ConnectionGene>,
}
```

Two consequences follow immediately, and both distinguish `TopologyGenome` from
the tensor genomes elsewhere in the crate. First, it **is `Clone`** — it is
ordinary owned data with no Burn-tensor storage to alias, so duplicating a genome
for mutation is a plain memory copy. Second, a population is a
`Vec<TopologyGenome>`, scored through a graph-shaped fitness seam
(`GraphFitnessFn`) rather than the tensor `BatchFitnessFn` the rectangular
strategies use.

The `TopologyGenome` is the **genotype** — the heritable description. The network
actually run on inputs is the **phenotype**, compiled from the genotype by the
machinery in the [phenotypes section](44-phenotypes.md). Keeping the two separate
is what lets mutation and crossover operate on cheap host-side gene lists while
evaluation happens wherever it is fastest. This section stays entirely on the
genotype side.

## The two gene types

A `TopologyGenome` is built from two flat gene lists. A **node gene** names a
neuron — its identity, its role in the network, its activation, and a per-node
bias:

```rust
pub struct NodeGene {
    pub id: NodeId,                 // stable within a run
    pub kind: NodeKind,             // Input | Output | Hidden | Bias
    pub activation: ActivationFn,   // Sigmoid | Tanh | Relu | Linear
    pub bias: f32,
}
```

A **connection gene** is a weighted directed edge, tagged with the historical
marker that the rest of this section is about:

```rust
pub struct ConnectionGene {
    pub innovation: InnovationId,   // the historical marker
    pub source: NodeId,
    pub target: NodeId,
    pub weight: f32,
    pub enabled: bool,
}
```

Three details on these fields earn their place. The bias lives **on the node**,
not as a phantom always-on input edge — and because a bias is functionally just a
weight, the same weight-perturbation operator that jitters edge weights mutates
it. The canonical NEAT activation is a *steepened* logistic sigmoid with gain
\\(4.9\\); inputs use `Linear` and fresh outputs `Sigmoid` by convention. And the
`enabled` flag is subtle: a disabled gene carries no signal in the phenotype, yet
it is **still counted** when measuring how far apart two genomes are and when
aligning them for crossover. A connection that has been switched off is part of a
genome's history, not erased from it.

`NodeId` and `InnovationId` are both `u64`, monotone within a run. They are
deliberately distinct *types* in intent even though they share a representation:
a node id names a neuron, an innovation id names the historical event of an edge
first appearing. Conflating them is a category error the next subsection exists
to prevent.

## The minimal seed and its id convention

NEAT's defining principle is to *start minimal and complexify only as it pays*.
A run therefore begins with no hidden nodes at all — every input wired directly
to every output:

```rust
let genome = TopologyGenome::minimal(num_inputs, num_outputs, &registry, rng, std);
```

The id layout of that seed is fixed by convention, and the convention matters
because every genome in the run inherits it:

- input node ids are `0..num_inputs`;
- output node ids are `num_inputs..num_inputs + num_outputs`;
- the initial fully-connected edges take innovations
  `input_index * num_outputs + output_index`, i.e. the range
  `0..num_inputs * num_outputs`.

For a 2-input, 1-output seed — the XOR shape — that is nodes `0, 1` (inputs) and
`2` (output), with two connection genes carrying innovations `0` and `1`. Weights
and the output bias are drawn per-individual from a zero-mean normal, so a whole
initial population shares one *structure* and one id-assignment while differing in
its *weights*. That shared structure is not a happy accident; it is the
precondition for the next idea to work at all.

## Innovation numbers: the historical marking

Here is the problem NEAT exists to solve. Two networks that have each grown a few
mutations no longer have the same shape. To recombine them you must decide which
gene in one corresponds to which gene in the other — and aligning two arbitrary
graphs by their topology is expensive and ambiguous. NEAT's insight is to *avoid
the graph-matching problem entirely* by recording, at the moment each structural
element first appears, a global counter value: its **innovation number**. Two
genes that share an innovation number descend from the *same* historical mutation
and are therefore homologous by construction — no graph comparison needed.

For this to hold, the same structural mutation must receive the same innovation
number wherever and whenever it happens within a run. That guarantee is the sole
job of the `InnovationRegistry`:

```rust
let registry = Arc::new(InnovationRegistry::new(num_seed_nodes, num_seed_innovations));
```

One registry is created per run and shared across the whole NEAT harness via
`Arc`. It is the **only** allocator of innovation ids and of hidden node ids, and
it caches every allocation it makes:

- `register_connection(source, target)` returns the innovation for an
  add-connection mutation. The first time a given `(source, target)` edge is seen
  it gets a fresh id; every later request for the *same* edge — in any genome,
  any generation — returns the cached id.
- `register_node_split(split_innovation)` handles add-node mutations, which split
  an existing connection in two. Keyed on the **innovation of the split edge**
  (not on its endpoints), it returns a `NodeSplit` bundling the new hidden node id
  and the two new edge innovations, cached so that splitting "the same place"
  always yields the same node even after the surrounding topology has diverged.

```rust
pub struct NodeSplit {
    pub new_node: NodeId,    // the inserted hidden node
    pub in_innov: InnovationId,   // source -> new_node, canonical weight 1.0
    pub out_innov: InnovationId,  // new_node -> target, inherits the old weight
}
```

The registry's counters are constructed to start *after* the minimal seed — a
2-input, 1-output run creates `InnovationRegistry::new(3, 2)`, so the first hidden
node it ever allocates is id `3` and the first new edge innovation is `2`,
slotting in directly above the seed without ever colliding with it.

## How crossover reads the markings

With innovation numbers in hand, recombination becomes a linear merge of two
sorted lists rather than a graph alignment. This is why `TopologyGenome` keeps an
invariant the next subsection states in full — its `connections` are **sorted by
innovation**. Walking the two parents' connection lists in lockstep classifies
every gene into one of three buckets, exactly as in the canonical algorithm:

| Classification | Condition | Inherited from |
| --- | --- | --- |
| **matching** | same innovation in both parents | a randomly chosen parent |
| **disjoint** | innovation present in one parent, within the other's range | the fitter parent |
| **excess** | innovation beyond the other parent's highest | the fitter parent |

When the two parents have equal fitness, disjoint and excess genes are taken from
*both*. A matching gene that is disabled in either parent has a chance
(`p_disable_inherited`) of staying disabled in the child, preserving the
literature's treatment of toggled connections. The crate's `crossover` is a
faithful single-pass implementation of this scheme: it advances two indices over
the innovation-sorted lists, and the `Ordering::Equal` / `Less` / `Greater` arms
are precisely the matching / disjoint / excess cases above.

One `rlevo`-specific guard sits on top of the canonical scheme. Combining genes
from two divergent parents can, in principle, assemble a child that contains both
`A→B` and `B→A` — a cycle, which the feedforward invariant forbids. So the
collected candidate genes are filtered in innovation order through a cycle check,
dropping any edge that would close a loop over the edges accepted so far. Such
drops are rare, but the check is what keeps recombination from ever producing a
non-feedforward child.

## The two invariants

Everything above leans on two properties of a `TopologyGenome` that the type
maintains for itself. State them plainly, because the rest of the NEAT machinery
takes them as given:

- **Innovation-sorted connections.** `connections` is kept strictly increasing by
  `innovation`. Mutations only ever append the largest-so-far innovation, so
  `insert_connection_sorted` keeps the order with a cheap binary-search insertion,
  and both crossover and the [compatibility distance](43-speciation.md) used by
  speciation reduce to `O(n)` merges of two sorted lists. `is_innovation_sorted`
  checks the property; the tests assert it survives
  every mutation and every crossover.
- **Acyclicity over *all* edges.** The directed graph is acyclic — the feedforward
  invariant — and `would_create_cycle` enforces it against **enabled and disabled
  edges alike**. Checking disabled edges too is what makes the invariant stable
  under toggling: a connection can be switched off and back on without ever
  introducing a cycle, because the acyclicity check never stopped accounting for
  it.

These are not decorative. The sorted invariant is what turns NEAT's central
operation — aligning two different graphs — from a quadratic graph-matching
problem into a linear scan, and the all-edges acyclicity invariant is what lets
the [phenotype builder](44-phenotypes.md) assume a clean topological order without
re-validating the graph on every forward pass.

## Determinism falls out of the design

A subtle payoff of routing every structural id through one cached allocator is
that NEAT runs are reproducible without any extra machinery. Mutation is applied
sequentially host-side under the seeded `seed_stream` RNG, so the order in which
ids are first issued is fixed by the seed; the registry's caches then make the
*result* of a repeated mutation order-independent, since only first-issue
assignment depends on order and that order is seeded. Replaying the same seed and
the same mutation sequence yields an identical innovation-and-node-id sequence —
the property the `test_innovation_numbering_is_deterministic` and
`test_independent_registries_replay_identical_ids` tests pin down. Independent
trials with different seeds each build their own registry and so occupy isolated
innovation spaces, which is correct: aligning genes *across* unrelated runs would
be meaningless.

With the genome and its markings established, two things become tractable that
the overview only gestured at: [grouping genomes into species](43-speciation.md)
by how far apart their genes sit, and
[compiling a genome into a runnable network](44-phenotypes.md).

---

*Co-Authored-By: Anthropic Claude Opus 4.8*\
*Reviewed-By: (Human) Anthony Torlucci*
