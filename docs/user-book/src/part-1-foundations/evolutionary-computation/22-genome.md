# Genome Representation

The [operators chapter](21-ops.md) worked on populations as raw tensors —
`Tensor<B, 2>` for real-valued genomes, `Tensor<B, 2, Int>` for binary ones —
and leaned on a single sentence to justify the split: *the element type is the
kind-specialisation*. This chapter unpacks that sentence. It is about the
**representation layer**, and the question that organises it is one of degree:
*how much does the compiler know about a genome, and how does it come to know
it?*

We answer that question in four rungs of increasing type-level knowledge, each
one closing a gap the rung below leaves open:

1. **A genome is a row of a population tensor** — the element type already
   separates real from discrete.
2. **A zero-sized `GenomeKind` marker** tags what the element type cannot:
   binary versus integer, both `i32`, want different operators.
3. **`Population<B, K>`** welds that marker to its storage tensor so the
   wrong-tensor-for-this-kind state is unrepresentable.
4. **`ParamReshaper`** reaches past hand-written rows entirely, turning a neural
   network's weights into an evolvable genome.

Climb those four rungs and you have the whole representation layer. We start at
the bottom.

## A population is a rank-2 tensor

Every classical strategy in `rlevo` represents its whole population as a single
rank-2 tensor of shape `(N, D)`:

- `N` rows — one per **individual** (a candidate solution);
- `D` columns — one per **gene** (a single decision variable).

There is no per-individual struct, no `Vec<Genome>`. The population *is* the
matrix, and a "genome" is one of its rows. This is what lets the operators be
pure tensor algebra: tournament selection is a gather over rows, Gaussian
mutation is an element-wise add, BLX-α is a handful of broadcast ops. Keeping the
entire population in one device allocation is also what makes the math
GPU-friendly — the whole generation moves through a kernel at once, never
row-by-row.

The **element type** of that tensor is not incidental — it carries the genome's
*kind*:

| Genome kind | Tensor type | Element | Meaning of a gene |
| ----------- | ----------- | ------- | ----------------- |
| real-valued | `Tensor<B, 2>` | `f32` | a continuous decision variable |
| binary | `Tensor<B, 2, Int>` | `i32` ∈ {0, 1} | a single bit |
| integer | `Tensor<B, 2, Int>` | `i32` ≥ 0 | a bounded index (CGP node, symbol) |

Because the float and integer tensor types are distinct, the type system already
distinguishes real genomes from discrete ones for free: `gaussian_mutation` only
typechecks against `Tensor<B, 2>`, `bit_flip_mutation` only against
`Tensor<B, 2, Int>`. Handing a binary genome to a real-valued operator is a
compile error, not a runtime surprise — the point made in passing last chapter,
now with the *why* behind it.

But `f32` versus `i32` is a coarse distinction. Binary and integer genomes are
*both* `Tensor<B, 2, Int>`, yet they want different operators — bit-flip makes
sense for a bitstring, not for a vector of node indices. The element type alone
cannot tell them apart. That is the gap the genome-kind marker fills.

## Genome kinds: a type-level tag

`rlevo` discriminates genome kinds with a set of **zero-sized marker types** that
implement one trait:

```rust
pub trait GenomeKind: Debug + Copy + Send + Sync + 'static {
    /// Element type of the genome (typically `f32`, `i32`, or `bool`).
    type Element: Copy + Debug + Send + Sync + 'static;
}
```

The markers themselves hold no data — they exist purely to discriminate trait
impls and to parameterise strategies:

```rust
pub struct Real;    // Element = f32  — ES, real-coded GA, EP, DE
pub struct Binary;  // Element = i32  — binary-coded GA, binary EDAs
pub struct Integer; // Element = i32  — Cartesian GP, discrete search
```

A marker is a *compile-time label*. A strategy that is generic over
`K: GenomeKind` can pick its operator set from the marker without ever inspecting
a value at runtime — there is no enum to match on, no tag byte stored anywhere.
`Real` and `Binary` both erase to nothing in the compiled binary; their only job
is to make `Population<B, Real>` and `Population<B, Binary>` *different types* so
the right `impl` block applies.

Notice what the trait does *not* carry: a genome length. That is deliberate.
Genome **width is a runtime property, never a type-level one** — you choose `D`
when you build a population, and read it back from `Population::genome_dim()`.
The same `Real` marker backs an 8-gene sphere problem and a 100 000-gene neural
weight vector; `Binary` backs a 32-bit string and a 1024-bit one. The kind fixes
the *element semantics* and the operator set; it says nothing about how many
genes a row holds.

The two roadmap kinds make the point sharper, because they have no fixed width
even at runtime. A `Tree` genome's length varies *from individual to individual*
within one population — that variability is the whole point of genetic
programming — and a `Permutation`'s length is the problem instance's node count
(the number of cities in a TSP). Width simply is not a property `GenomeKind` is
in a position to promise.

> **A type-level length, if it were ever earned.** One could imagine a kind whose
> width is welded to its *type* — a quaternion-rotation gene that is *always*
> four floats, an RGB-triple that is *always* three channels — where a
> compile-time length constant would document the invariant and let const-generic
> buffers size themselves from it. No kind in `rlevo` is of that sort, so the
> trait carries no such constant; an associated const with a default value is a
> non-breaking addition, so it can be introduced the day a structurally-fixed
> kind actually needs it, and not before.

## The `Population<B, K>` wrapper

Operators take bare tensors, but strategies pass populations across method
boundaries (`ask` produces one, `tell` consumes one), and at those boundaries it
helps to carry validated shape metadata alongside the data.
`Population<B, K>` is that container:

```rust
pub struct Population<B: Backend, K: TensorGenome> {
    pop_size: usize,
    genome_dim: usize,
    tensor: K::Tensor<B>,
}
```

The single tensor field is the point. Its type is **not** fixed to one flavour —
it is `K::Tensor<B>`, an associated type chosen by the genome kind through a
companion trait:

```rust
pub trait TensorGenome: GenomeKind {
    type Tensor<B: Backend>: Clone + Debug + Send + Sync;
}

impl TensorGenome for Real        { type Tensor<B: Backend> = Tensor<B, 2>; }
impl TensorGenome for Binary      { type Tensor<B: Backend> = Tensor<B, 2, Int>; }
impl TensorGenome for Integer     { type Tensor<B: Backend> = Tensor<B, 2, Int>; }
impl TensorGenome for Permutation { type Tensor<B: Backend> = Tensor<B, 2, Int>; }
```

The `Send + Sync` on the associated type is not decoration: every
[`Strategy`](https://docs.rs/rlevo-evolution) is `Send + Sync`, and a
`Population<B, K>` is only shareable across threads if its stored tensor is too.
Bounding the tensor here lets generic code over `K: TensorGenome` rely on that
without re-stating it at every call site.

So `Population<B, Real>` *is* a population whose field is a `Tensor<B, 2>`, and
`Population<B, Binary>` *is* one whose field is a `Tensor<B, 2, Int>` — the kind
and its storage type are welded together at compile time. Two consequences fall
out, both of which tighten the earlier design:

- **The wrong-tensor-for-this-kind state is unrepresentable.** There is no pair
  of `Option` fields to keep in sync and no "exactly one is `Some`" invariant to
  uphold by convention. The single `tensor()` accessor returns `&K::Tensor<B>`
  directly — it is *total*, with no `.expect()` and no panic path, because the
  type system already guarantees the field holds the right flavour.
- **Kinds without a rectangular tensor form are excluded by the type checker.**
  The `K: TensorGenome` bound means a kind must *opt in* by naming a tensor type.
  `Tree` (a host-side, variable-length AST) does not implement `TensorGenome`, so
  `Population<B, Tree>` does not type-check at all — the impossibility is enforced,
  not just documented.

Each kind still gets an explicit constructor — `new_real`, `new_binary`,
`new_integer`, `new_permutation` — because reading `pop_size`/`genome_dim` from
`tensor.dims()` happens where the concrete tensor type is in hand. Each returns a
`Result`: we reject an empty (`0`-row or `0`-column) tensor at construction so the
failure names `Population` as its source, rather than surfacing later as an opaque
operator panic deep in a selection kernel.

```rust
let pop = Population::<B, Real>::new_real(tensor)?;
assert_eq!(pop.pop_size(), 4);                          // rows
assert_eq!(pop.genome_dim(), 3);                        // columns

let bits = Population::<B, Binary>::new_binary(int_tensor)?;
```

The wrapper exists for one reason: to give operators and strategies a **single
shape contract** to validate against. Rather than every call site re-deriving `N`
and `D` from `tensor.dims()`, the population computes them once at construction
and exposes `pop_size()` / `genome_dim()`. It is a thin convenience, not a
smart container — it does not validate gene *values* (a binary population is not
checked for stray `2`s; that would cost a device round-trip on the hot path).

### Two layers: `Genome` the tensor, `Population` the boundary

There is a subtlety worth stating plainly, because the two layers coexist. The
`Strategy` trait has an associated `type Genome`, and in every concrete strategy
that type is the **bare tensor**, not the wrapper:

```rust
impl<B: Backend> Strategy<B> for GeneticAlgorithm<B> {
    type Genome = Tensor<B, 2>;        // real-coded GA
    // ...
}
impl<B: Backend> Strategy<B> for BinaryGeneticAlgorithm<B> {
    type Genome = Tensor<B, 2, Int>;   // binary GA
    // ...
}
```

So the flow of data inside a generation is all raw tensors — that is what the
operators consume and return, with no wrapping or unwrapping on the hot path.
`Population<B, K>` is the **typed, shape-checked container** you reach for when
constructing a population from host data or handing one across an API boundary
where the kind and dimensions should travel with the data. Think of the marker
`K` as the contract and the tensor as the payload: the wrapper bundles them when
that bundling buys clarity, and the strategies unwrap to the payload when they
just need to compute.

## Bridging to neural networks: `ParamReshaper`

Everything so far assumes you can *write down* a genome as a row of numbers. For
weight-only neuroevolution — evolving the parameters of a fixed-topology network
— the genome is exactly that row: a flat vector of every trainable float in the
network, laid out so a population is a `(pop_size, num_params)` real tensor and
each row is one network's complete weight set. The piece that converts between a
live Burn `Module` and that flat row is the `ParamReshaper` trait:

```rust
pub trait ParamReshaper<B: Backend>: Send + Sync {
    type Module: Module<B>;

    fn num_params(&self) -> usize;
    fn flatten(&self, module: &Self::Module, device: &B::Device) -> Tensor<B, 1>;
    fn unflatten(&self, flat: Tensor<B, 1>) -> Self::Module;
}
```

- `flatten` walks the module's float leaves in a deterministic order and
  concatenates them into one 1-D tensor — *encoding* a network as a genome.
- `unflatten` clones a template module and refills its leaves from a flat vector,
  in the *same* order — *decoding* a genome back into a scorable network.

The round-trip `unflatten(flatten(m)) ≈ m` holds because Burn's
`#[derive(Module)]` generates `visit`/`map` traversals that walk fields in
declaration order, recursively, so the encoder and decoder agree leaf-for-leaf.
The concrete `ModuleReshaper` clones a template once, caches its float-leaf count
as `num_params`, and uses it for every conversion.

Two sharp edges are worth flagging, both already pinned down by the
implementation:

- **`BatchNorm` running statistics are evolvable.** Burn's traversal touches
  *every* float leaf, including non-trainable running mean/variance (they are
  `RunningState` modules whose `Module` impl forwards to `visit_float`/`map_float`,
  so they are visited exactly like a `Param`). A `BatchNorm` over `d` features
  therefore exposes **four** float-leaf tensors — `gamma`, `beta`, `running_mean`,
  `running_var`, each of length `d`, i.e. `4·d` scalar parameters — all of which
  evolution will flatten and perturb. The crate verifies this empirically:
  `param_reshaper.rs`'s `batchnorm_running_stats_are_traversed` test asserts
  `num_params() == 4·d` (not `2·d`). Fixed MLP policies have no such buffers, so
  this is moot for the v1 target; callers evolving batch-normalised networks should
  reset running stats after `unflatten`.
- **Gradient isolation.** The reshaper is generic over `B: Backend`, *not*
  `AutodiffBackend`. Tensors produced by `unflatten` do not require gradients,
  and callers holding an autodiff module call `.valid()` first — so the absence
  of a gradient tape is enforced at the type level, not by convention. Evolution
  searches the weight space directly; it never backpropagates.

This is the seam where the evolutionary side of `rlevo` meets the neural side:
upstream, a genome is just a row of `f32`s the operators already know how to
mutate and recombine; downstream, `unflatten` turns each row into a network the
fitness function can run. Neither side needs to know about the other.

## Kinds on the roadmap

Two further markers exist so the API can name representations whose *operators*
have not fully landed yet. They sit on opposite sides of the tensor seam, and
that difference is exactly what `TensorGenome` captures.

- **`Permutation`** — each row a permutation of `0..n`, for ordering problems
  (TSP, QAP) driven by Ant Colony Optimization. It *is* rectangular — each row is
  a length-`n` integer vector — so it implements `TensorGenome` today
  (`Tensor<B, 2, Int>`) and `Population<B, Permutation>` type-checks and
  constructs via `new_permutation`. What is still stubbed is the *operator* set:
  a stubbed consumer ships now, and the full pheromone/tour-construction
  machinery is planned. Note that `new_permutation` validates only shape — it
  does **not** check that each row is a genuine bijection of `0..n`, mirroring
  how the binary and integer constructors leave gene *values* unchecked.
- **`Tree`** — variable-length ASTs for classical Koza-style genetic programming.
  Tree genomes cannot be batched into a rectangular GPU tensor, so they have no
  tensor representation in this crate: `Tree` *stays* a plain `GenomeKind` and
  never implements `TensorGenome`, so `Population<B, Tree>` does not type-check.
  The marker reserves the name for a host-side implementation in a future release.

Both implement `GenomeKind`, so strategies and operators can already name them.
Listing them here is deliberate: it marks the seams where the representation layer
is designed to grow — `Permutation` already across the tensor seam and awaiting
operators, `Tree` deliberately off-device — without disturbing the kinds that
already work.

## Putting it together

A genome in `rlevo` is a row of a rank-2 population tensor; its *kind* — real,
binary, or integer — is a zero-sized marker that lets the compiler hand each
strategy the right operators with no runtime dispatch. `Population<B, K>` bundles
that tensor with its shape metadata and kind tag for use at API boundaries, while
strategies compute on the bare tensor for speed. And for neuroevolution,
`ParamReshaper` makes a network's weights *be* a genome — flatten to evolve,
unflatten to score. With representation (this chapter) and operators
([the last](21-ops.md)) on the table, the remaining ingredient is how a genome
earns its keep: the **fitness** that scores it, and the **strategy** that
assembles selection, variation, and replacement into a running generation.

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
