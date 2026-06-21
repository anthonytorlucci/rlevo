# The Parameter Bridge: Weights as a Flat Genome

The three sections that follow this one are about NEAT, where the genome is a
graph. This one is about the other two approaches — `WeightOnly` and `ArchNas` —
where the network's architecture is fixed (or chosen from a menu) and only its
*weights* evolve. There the genome is the simplest thing it could be: a flat row
of every trainable float in the network. The question this section answers is the
plumbing one. A Burn `Module` is a nested tree of parameter tensors; an
evolutionary strategy wants a rectangular `(pop_size, num_params)` population of
plain numbers. What converts between the two, and — more importantly for the
design — *where* in the pipeline that conversion is allowed to happen.

The conversion itself, the `ParamReshaper` trait and its `ModuleReshaper`
implementation, was introduced in the
[genome chapter](../evolutionary-computation/22-genome.md): `flatten` walks a
module's float leaves in a deterministic order and concatenates them into a 1-D
tensor, `unflatten` clones a template and refills its leaves from a flat vector in
the same order, and the round-trip holds because Burn's `#[derive(Module)]`
generates `visit`/`map` traversals that agree leaf-for-leaf. That mechanism is not
repeated here. This section is about the **adapter layer** built on top of it: how
weight-only neuroevolution keeps the reshaper out of the hot path, confines it to
a single boundary, and wires the genome width through to the strategy.

## The reshaper is the single source of truth for genome width

A weight-only run has a chicken-and-egg parameter: the inner strategy needs to
know `num_params` — the genome width — before it can initialise a population, but
that count is a property of the *network*, not of the strategy. `ModuleReshaper`
resolves it. Constructed from a template module, it counts that module's float
leaves once and caches the total:

```rust
let template = MyMlp::<B>::new(&device);
let strategy = WeightOnly::new(GeneticAlgorithm::<B>::new(), template.clone());
let n = strategy.num_params();          // read straight off the template
let params = GaConfig::default_for(64, n);
```

`num_params` is read *off the template*, never hand-counted — a `1→16→1` MLP
reports `49`, the two-layer `Linear(3→4)→ReLU→Linear(4→2)` of the reshaper's own
test reports `26`, and any miscount between the strategy's genome width and the
network's leaf count is caught at the evaluation boundary rather than silently
producing a misaligned network. The template is the authority; everything
downstream reads from it.

## Reshaping is confined to the fitness boundary

Here is the load-bearing design choice, and it is what makes the weight-only path
efficient. Reshaping a flat row back into a module is not free — it allocates a
network. If evolution paid that cost on every operation, the saving of working in
flat tensor space would evaporate. So `rlevo` confines reshaping to **exactly one
place**: the fitness boundary.

`WeightOnly<B, S, M>` is a `Strategy` wrapper. It owns the inner strategy `S`
(whose genome is a flat `Tensor<B, 2>`) and the `ModuleReshaper`, but its
`init`/`ask`/`tell` methods **delegate verbatim** to the inner strategy and do no
reshaping at all. The population stays a flat `(pop_size, num_params)` tensor end
to end; selection, crossover, and mutation all operate on plain numbers, exactly
as they would for any real-valued GA, ES, or DE. This follows the same wrapper
pattern as `MemeticWrapper` (ADR 0016): the inner strategy owns all population
state, and the wrapper adds one orthogonal concern — here, the module binding.

The tensor→module round-trip happens only when a genome must actually be *scored*,
inside `ModuleEvalFn`:

```rust
pub struct ModuleEvalFn<B, R, F> { /* reshaper R + scorer F */ }
// F: Fn(&R::Module) -> f32   — MSE, accuracy, negative episode return, …
```

`ModuleEvalFn` implements `BatchFitnessFn`: it takes the flat population, and for
each row unflattens it into a module via the reshaper, hands that module to a
host-side scorer, and collects the scalar into the fitness tensor. Its own module
doc states the rule plainly — it is *the only place flatten/unflatten happens in
the weight-only pipeline*. Evolution never pays for a tensor→module conversion
except where it is unavoidable: at the moment of evaluation.

One detail of `BatchFitnessFn` matters for wiring objectives correctly: it follows
the engine's **minimisation** convention — lower fitness is better. A scorer must
therefore be a *cost* (MSE, negative return), the same direction the rest of the
non-NEAT engine assumes. This is the opposite of the maximisation convention NEAT
uses, and the contrast is worth holding onto.

## Evaluation is loop-over-N, for now

`ModuleEvalFn` scores its population by unflattening and running **one row at a
time**. This is a deliberate, documented limitation rather than an oversight: Burn
0.21 has no `vmap` or batched-forward primitive that would let a single forward
pass carry a whole population of *distinct* parameter sets through one network. So
each member is reconstructed and scored individually, and a batched forward path
is a future addition.

It is worth contrasting this with NEAT's batched evaluator from the
[phenotypes section](44-phenotypes.md). That path batches a population because it
*builds the forward pass itself* from bare tensor arithmetic, so it can pack `P`
distinct topologies into padded weight tensors and run one matmul. The weight-only
path delegates the forward pass to an opaque Burn `Module`, which has no such
population dimension — hence the per-member loop. The two make opposite trade-offs
deliberately: the interpreted NEAT phenotype gives up `Module` to gain batching;
weight-only keeps `Module` and gives up batching.

## Gradient isolation is enforced by the type

Evolution searches weight space directly and never backpropagates, and the bridge
makes that structural rather than conventional. Both `ModuleReshaper` and
`ModuleEvalFn` are generic over `B: Backend`, **not** `AutodiffBackend`. Tensors
produced by `unflatten` carry no gradient tape, and a caller holding an autodiff
module must call `.valid()` before constructing the wrapper. The absence of a
gradient tape is therefore a compile-time property of the types involved, not a
discipline the caller has to remember — there is no path by which an autodiff
graph leaks into the evolutionary loop.

The one sharp edge inherited from the underlying reshaper is worth recalling from
the [genome chapter](../evolutionary-computation/22-genome.md): Burn's traversal
touches *every* float leaf, including a `BatchNorm` layer's non-trainable running
mean and variance (a `BatchNorm` over `d` features exposes `4·d` leaves, not
`2·d`). Evolution will therefore flatten and perturb those running statistics like
any weight. Fixed MLP policies — the v1 target for both `WeightOnly` and
`ArchNas` — have no such buffers, so it is moot in practice; callers evolving
batch-normalised networks should reset running stats after `unflatten`.

## From one network to a menu: `ArchNas`

`ArchNas` is the same bridge applied to a *set* of templates instead of one. The
[overview](../40-neuroevolution.md) covers its genome — a categorical architecture
choice per individual plus a zero-padded weight matrix sized to the widest
variant — but the parameter-bridge mechanics are identical underneath: each
registered variant has its own float-leaf count (`per_variant_params`), read off
that variant's template exactly as `num_params` is for `WeightOnly`, and the
population tensor is padded to the menu's `max_param_count` so a ragged population
of differently-sized networks still rides one rectangular tensor. The reshaper
concept does not change; it is simply instantiated once per variant, and the
load-bearing invariant becomes that an individual's `arch_id` selects the matching
template when its row is unflattened for scoring.

So the parameter bridge is what lets the entire flat-genome half of the previous
chapters — GA, ES, CMA-ES, DE, and the operators that drive them — apply to neural
networks unchanged. Upstream of the fitness boundary a genome is just a row of
floats the operators already know how to mutate and recombine; downstream,
`unflatten` turns each row into a network the scorer can run. Neither side needs
to know the other exists, and that clean seam is the whole point.

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
