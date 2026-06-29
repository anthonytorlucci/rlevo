# Phenotypes: Compiling a Genome into a Network

The two preceding sections stayed entirely on the **genotype** side. The
[genome section](42-neat-genome.md) built `TopologyGenome` from gene lists and the
innovation numbers that recombine them; the [speciation section](43-speciation.md)
grouped those genomes and apportioned offspring. Neither ever ran a network. This
section closes that gap: how a `TopologyGenome` — a pair of gene lists, not a
layer stack — becomes a callable network that maps inputs to outputs, and why
`rlevo` ships *two* ways to do it.

If you have followed NEAT this far, you know a genome *describes* a network
without *being* one. The genotype is a recipe; the phenotype is the dish. What is
specific to `rlevo` is that the recipe is compiled to bare tensor arithmetic with
**no Burn `Module`** anywhere in sight — and that the same genome can be run one
at a time or a whole population at once, through two deliberately separate seams
that agree to float epsilon.

## Two seams, two traits

The single most important thing to get right here — and the easiest to get wrong —
is that phenotype evaluation is **two distinct traits**, not one trait with two
implementations. They answer different questions and have different shapes.

The **per-genome** seam compiles one genome into one callable network:

```rust
pub trait PhenotypeBuilder<B: Backend> {
    fn build(&self, genome: &TopologyGenome, device: &B::Device) -> Box<dyn Phenotype<B>>;
}

pub trait Phenotype<B: Backend>: Send + Sync {
    /// `[batch, num_inputs]` -> `[batch, num_outputs]`.
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2>;
}
```

The **population-batched** seam evaluates an entire population on one shared input
batch in a single pass:

```rust
pub trait BatchPhenotypeEvaluator<B: Backend>: Send + Sync {
    /// `[batch, obs_dim]` -> `[pop, batch, action_dim]`.
    fn evaluate_population(
        &self,
        genomes: &[TopologyGenome],
        obs: Tensor<B, 2>,
        device: &B::Device,
    ) -> Tensor<B, 3>;
}
```

`rlevo` provides one implementation of each: `InterpretedBuilder` (producing an
`InterpretedPhenotype`) for the per-genome seam, and `DensePaddedEvaluator` for
the batched seam. It is worth being explicit, because the pairing is *not*
diagonal: `DensePaddedEvaluator` implements `BatchPhenotypeEvaluator`, **not**
`PhenotypeBuilder`. The per-genome interpreted path is the correctness oracle; the
batched dense path is the throughput option. The rest of this section takes them
in that order.

## Why there is no Burn `Module`

Every other network in `rlevo` — every weight-only policy, every NAS variant — is
a Burn `Module`, flattened and unflattened through the
[parameter bridge](../evolutionary-computation/22-genome.md). NEAT phenotypes are
not, and the reason is structural. Burn's `#[derive(Module)]` needs a **static
field structure** known at compile time: a fixed set of named parameter fields it
can generate `visit`/`map` traversals over. A NEAT topology is *data-defined* —
its nodes and edges exist only at runtime, different for every genome and every
generation — so there is no static field layout to derive against.

That sounds like a limitation; it is the opposite. A `Module` buys autodiff,
recorders, and gradient plumbing — every one of which neuroevolution discards,
because it searches weight space directly and never backpropagates. A NEAT
phenotype needs exactly one capability: a forward pass. Skipping `Module` and
building the forward pass from bare `Tensor` arithmetic therefore costs nothing
and removes a pile of machinery that would never be used.

## The interpreted phenotype: the oracle

`InterpretedPhenotype` is a host-side feedforward evaluator. Construction compiles
the genome once into an **evaluation plan** — for each non-input node, its enabled
incoming edges as `(source, weight)` pairs, its bias, and its activation — with
the nodes laid out in **topological order**. Crucially it stores *no Burn tensors*,
only this host-side metadata, so it is trivially `Send + Sync` and cheap to build;
the compute device is taken from the input tensor at forward time.

The topological order is computed over **all** structural edges — enabled and
disabled alike — and this is exactly where the genome's
[acyclicity invariant](42-neat-genome.md) pays off. Because the all-edges graph is
a guaranteed DAG, one Kahn sort (ties broken by ascending node id, for
reproducibility) yields an order valid for the *enabled* subgraph the forward pass
actually walks. The phenotype never re-validates the graph; the invariant already
did.

The forward pass is then a plain walk in that order: seed input nodes with the
input columns, and for each subsequent node accumulate `Σ weight · source_value`,
add the bias, and apply the activation. The activations mirror the host-side
`ActivationFn::apply` *exactly* — including the steepened sigmoid gain — so a
hand-computed truth table matches the tensor output to float epsilon. Two tests
pin this down: `test_interpreted_phenotype_reproduces_truth_table` checks a
hand-built `in0, in1 → relu hidden → linear out` network against its closed-form
outputs, and `test_interpreted_phenotype_skips_disabled_edges` confirms a disabled
edge carries no signal — the output collapses to the bias, exactly as a skipped
gene should behave.

This path is the **reference semantics** for what a NEAT genome *means* as a
function. Everything else must agree with it.

## The dense-padded evaluator: the throughput path

Running the interpreted phenotype in a host-side loop over a whole population is a
correctness baseline, not a fast one — each genome is evaluated independently, on
host, one at a time. `DensePaddedEvaluator` collapses the population into a single
device-resident forward pass.

Each call compiles the `P` genomes into padded dense tensors over a node budget
`N = max_nodes` (the largest genome's node count): a `(P, N, N)` weight tensor
where `weights[p, i, j]` is the weight of enabled edge `j → i`, a `(P, N)` bias
tensor, and one boolean mask per activation variant. Genomes smaller than `N` are
padded with id-free rows that carry no edges and no activation, so they stay zero
and never feed a real node. NEAT fixes the input/output node set across the whole
population, so input rows are always `0..num_inputs` and output rows always
`num_inputs..num_inputs + num_outputs` — uniform across `P`, which is what lets
seeding and output-slicing skip all per-genome index arithmetic.

The forward pass is **synchronous update**: seed the input rows with the
observation (held fixed every step), then repeatedly compute
`acc = W · values + bias` — a single batched `(P, N, N) × (P, N, batch)` matmul —
and apply each node's activation. Heterogeneous per-node activations are handled
by running one masked pass per variant (`mask_where`) and reusing the interpreted
formulas verbatim, including the same sigmoid gain, so the two paths cannot drift.

Two design points make this exact and affordable:

- **It is exact because NEAT is feedforward.** After `d` synchronous updates,
  every node at topological depth `d` has settled. So iterating for the
  population's **deepest enabled path** resolves every genome exactly — no
  fixed-point iteration, no convergence tolerance. A recurrent network would need
  a different story; v1 NEAT's feedforward invariant is what licenses this one.
- **The iteration count is depth-bounded, not `N − 1`.** The safe static worst
  case is `N − 1` steps, but NEAT topologies are typically sparse and shallow, and
  each step is a full dense `N × N` matmul — so over-iterating dominates runtime at
  scale. The evaluator instead iterates the *actual* longest enabled path across
  the population (floored at `1` so a bias-only node still activates). This single
  choice is what makes the dense path competitive; a naïve `N − 1` bound erases
  its advantage entirely.

Memory is dominated by the weight tensor — `(256, 50) → 2.56 MB`,
`(256, 200) → 41 MB` in f32 — so a `max_nodes_cap` (default `512`) guards against
a runaway topology silently allocating an oversized tensor; a population exceeding
it panics rather than thrashes.

## Parity, and when each path wins

The two seams are kept numerically interchangeable by design. The
`test_dense_padded_matches_interpreted_population` test runs a fixed population —
all four activations, a two-hidden-layer chain, a disabled edge, and padding —
through both paths and asserts they agree within float epsilon. Because they
agree, the choice between them is **purely about throughput**, never about
correctness.

And throughput depends entirely on scale, which the `neat_xor` benchmarks measure
rather than assume:

- **At XOR scale** (`N ≈ 5–10`), the two run at par — roughly 580 µs versus 590 µs
  for a full generation at `pop_size = 64`. The workload is too small for batching
  to amortise, and a generation's cost is dominated by host-side reproduction
  (selection, crossover, mutation), not by the forward pass at all.
- **As networks widen**, the batched path pulls ahead. On a synthetic wide
  feedforward population at `P = 256`, the dense evaluator measures roughly 1.7×
  the interpreted loop at hidden width `10` and 1.9× at width `100` (Flex/CPU) —
  and the margin keeps growing, precisely because the depth-bounded iteration count
  amortises one padded matmul across the whole population.

The guidance falls out cleanly. The interpreted phenotype is the oracle and the
right default for small networks and for verifying behaviour; the dense-padded
evaluator is what you reach for once the topologies grow wide enough that the
forward pass, rather than reproduction, dominates the generation. They are two
views of one function, and the genome — with its innovation markings and the
species that protect it — is what both of them run.

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
