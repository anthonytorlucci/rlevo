# Genome Representation

The [operators chapter](21-ops.md) worked on populations as raw tensors —
`Tensor<B, 2>` for real-valued genomes, `Tensor<B, 2, Int>` for binary ones —
and leaned on a single sentence to justify the split: *the element type is the
kind-specialisation*. This chapter unpacks that sentence. It is about the
**representation layer**: how a candidate solution becomes a row in a device
tensor, how `rlevo` tags the *kind* of genome at the type level so the compiler
picks the right operators, and how the wrapper that carries shape metadata fits
in. It also covers the one place a genome is *not* a hand-written tensor — the
bridge that turns a neural network's weights into an evolvable parameter vector.

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
    /// Compile-time genome length (number of genes), or `0` for
    /// variable-length kinds.
    const GENOME_LEN: usize;

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

`GENOME_LEN` records the genome length at the type level *when it is known at
compile time*. For the tensor-backed kinds the population's `D` is a runtime
value (you choose it when you build the population), so they set
`GENOME_LEN = 0` — read it as "length not fixed by the type." The constant earns
its keep only for representations whose width is structurally fixed.

## The `Population<B, K>` wrapper

Operators take bare tensors, but strategies pass populations across method
boundaries (`ask` produces one, `tell` consumes one), and at those boundaries it
helps to carry validated shape metadata alongside the data.
`Population<B, K>` is that container:

```rust
pub struct Population<B: Backend, K> {
    pop_size: usize,
    genome_dim: usize,
    _kind: PhantomData<K>,
    tensor_real: Option<Tensor<B, 2>>,
    tensor_int: Option<Tensor<B, 2, Int>>,
}
```

Two things are worth reading closely.

First, the **kind is a `PhantomData<K>`**, not a stored value — the marker lives
only in the type, exactly as the previous section described. `Population<B, Real>`
and `Population<B, Binary>` are distinct types that compile to the same runtime
layout.

Second, the wrapper holds **two `Option` tensor fields**, one float and one
integer, and the public constructors maintain a strict invariant: for any
population they produce, *exactly one* is `Some`, determined by `K`. `Real`
populates `tensor_real`; `Binary` and `Integer` populate `tensor_int`. Each kind
has its own constructor and its own `tensor()` accessor:

```rust
let pop = Population::<B, Real>::new_real(tensor);      // tensor_real = Some
assert_eq!(pop.pop_size(), 4);                          // rows
assert_eq!(pop.genome_dim(), 3);                        // columns

let bits = Population::<B, Binary>::new_binary(int_tensor);  // tensor_int = Some
```

The accessors `.expect()` on their matching field, which looks risky but is not:
the constructor contract pins the invariant, so a mismatch would be a bug *in
this module*, not something a caller can trigger. The constructors do assert the
tensor is rank 2 — that *is* a programming error worth catching loudly, and it is
documented as a `# Panics` clause (consistent with the panic discipline recorded
in `docs/rules.md`).

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

Two further markers exist as **type-level placeholders** for representations that
do not fit the rectangular-tensor mould, so the API can name them before the
machinery lands:

- **`Tree`** — variable-length ASTs for classical Koza-style genetic programming.
  Tree genomes cannot be batched into a rectangular GPU tensor, so they have no
  tensor representation in this crate; the marker reserves the name for a
  host-side implementation in a future release.
- **`Permutation`** — each row a permutation of `0..n`, for ordering problems
  (TSP, QAP) driven by Ant Colony Optimization. A stubbed consumer ships today;
  the full operator set is planned.

Both implement `GenomeKind` (with `GENOME_LEN = 0`, since their length is not
fixed by the type) so they slot into the same `Population<B, K>` and strategy
machinery once their operators exist. Listing them here is deliberate: it marks
the seams where the representation layer is designed to grow without disturbing
the kinds that already work.

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

*Co-Authored-By: Anthropic Claude Opus 4.8*\
*Reviewed-By: (Human) Anthony Torlucci*
