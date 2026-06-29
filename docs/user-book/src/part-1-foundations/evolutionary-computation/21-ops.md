# Evolutionary Operators

The [previous chapter](../20-evolutionary-computation.md) introduced the
five-step evolutionary skeleton — maintain a population, evaluate, select,
vary, repeat — and the `Strategy<B>` trait that maps it onto `ask`/`tell`.
A strategy is the *choreography*; this chapter is about the *moves*.

Strip a generation down to its mechanics and you find four questions, asked in
order:

1. **Who gets to reproduce?** — *selection*
2. **How do two parents combine into a child?** — *crossover*
3. **Where does fresh novelty come from?** — *mutation*
4. **Who survives into the next generation?** — *replacement*

Every population-based algorithm in `rlevo` is some answer to those four
questions, and `rlevo` ships the answers as a small, composable toolkit of free
functions in `rlevo::evo::ops`. One module per question:

| Module | Question it answers | Operators |
| ------ | ---- | --------- |
| `ops::selection` | **who reproduces** | tournament, truncation |
| `ops::crossover` | **how parents combine** | BLX-α, uniform, binary uniform |
| `ops::mutation` | **where novelty comes from** | Gaussian (scalar and per-row), uniform-reset, bit-flip |
| `ops::replacement` | **who carries over** | generational, elitist, \\((\mu+\lambda)\\), \\((\mu,\lambda)\\) |

Here is what one generation looks like with the four moves laid end to end —
read it as the table above, run forward:

```rust
let parents   = selection::tournament_select(&pop, &fitness, 3, lambda, rng, device);
let offspring = crossover::blx_alpha(parent_a, parent_b, 0.5, rng, device);
let mutated   = mutation::gaussian_mutation(offspring, sigma, rng, device);
let (next, next_fitness) =
    replacement::mu_plus_lambda(pop, &fitness, mutated, &child_fitness, mu, device);
```

That recipe is the whole chapter in four lines. The rest of it is the catalogue
behind each call — and, first, three conventions that cut across all four
modules. They are worth internalising up front, because they explain why every
signature below looks the way it does.

## Three conventions that run through everything

### Operators are free functions, not methods

Notice what the recipe above did *not* do: it never constructed an operator
object, boxed a `dyn Mutation`, or registered anything. There is no `Operator`
trait at all. Each operator is a plain generic function over a backend `B`,
taking a population tensor and returning a *new* one — the input is left
untouched:

```rust
pub fn gaussian_mutation<B: Backend>(
    population: Tensor<B, 2>,
    sigma: f32,
    rng: &mut dyn Rng,
    device: &B::Device,
) -> Tensor<B, 2>;
```

We chose this deliberately. A `Strategy` calls exactly the operators it needs,
in the order it needs, with no virtual dispatch and no trait machinery sitting
between the algorithm and the tensor math — which is why composing a generation
reads as nothing more than calling functions, the way the four-line recipe did.

The data those functions move is uniform. Populations live on-device as rank-2
tensors of shape `(N, D)` — `N` individuals, each a `D`-gene genome. Real-valued
genomes are `Tensor<B, 2>`; binary genomes are `Tensor<B, 2, Int>` with values
in `{0, 1}`. The element type *is* the kind-specialisation: `gaussian_mutation`
only typechecks for the float tensor, `bit_flip_mutation` only for the Int
tensor, so handing a binary genome to a real-valued operator is a compile error,
not a runtime surprise three hours into a run.

### Fitness is canonical — higher is better

The classical evolutionary-computation tradition treats *fitness* as a quantity
to **maximise**, with the *fittest* individual scoring *highest* — and `rlevo`
follows it. Throughout `ops` the operators work in **canonical** space: the *best*
individual has the *highest* fitness. Read every "highest fitness", "largest
fitness", and "top-k" in this chapter as "closest to the optimum we're driving
toward."

That single convention runs through every operator. Tournament selection keeps
the *largest* fitness in each draw; truncation returns the `top_k` *highest*
values; elitism preserves the fittest parents. You never hand-negate a cost: you
declare your objective's direction with
[`ObjectiveSense`](https://docs.rs/rlevo-core) (a cost is `Minimize`), and the
`EvolutionaryHarness` reconciles it at one chokepoint *before* the operators ever
see the fitness. Fixing the direction once, globally, means no operator needs a
"maximise or minimise?" flag and the strategies that call them stay simple.

### The harness owns randomness — the host-RNG convention

Look back at every signature so far and you will see the same parameter:
`rng: &mut dyn Rng`. Every operator that draws random numbers takes that
explicit handle and threads *all* its stochasticity through it. Operators
**never** call `B::seed` or `Tensor::random`. This is the single most important
invariant in the crate, and it is not stylistic:

> Burn's backend RNG (the kernel path behind `Tensor::random`) is **process-global**.
> Under the parallel test runner — or any setup that evolves several
> populations concurrently — those kernel draws interleave non-deterministically
> across threads, so a seed no longer reproduces a run. Sampling on the host with
> a caller-owned `Rng` and lifting the draws onto the device with
> `Tensor::from_data` sidesteps the shared mutex entirely: the same seed yields
> the same population regardless of thread schedule.

Concretely, the stochastic operators all follow a **host-sample → device-load**
pattern: build a `Vec<f32>` of the right size from the host `rng`, wrap it in
`TensorData`, and load it onto the device in one shot. The selection operators
go one step further with a **host-decide → single-gather** pattern, which the
next section unpacks.

With those three conventions in hand, we can walk the four moves in order.

## Selection — who gets to reproduce

Selection turns a fitness vector into *parents*. Both selectors in `rlevo`
compute winner **indices** on the host from a `&[f32]` fitness slice, then
perform a single `Tensor::select` gather on the device to pull those rows out of
the population. Each comes in two layers:

- a `*_indices_host` function — the pure, testable core that returns `Vec<i32>`
  winner indices;
- a `*_select` wrapper — the convenience that runs the host core and does the
  one device gather, returning a `(n_winners, D)` tensor.

The wrapper is deliberately thin. `tournament_select` runs the host core to get
winner indices, lifts that `Vec<i32>` onto the device as a rank-1 `Int` tensor,
and issues a *single* `Tensor::select` along axis 0 to pull the winning rows out
of the population:

```rust
pub fn tournament_select<B: Backend>(
    population: &Tensor<B, 2>,
    fitness: &[f32],
    tournament_size: usize,
    n_winners: usize,
    rng: &mut dyn Rng,
    device: &B::Device,
) -> Tensor<B, 2> {
    let winners = tournament_indices_host(fitness, tournament_size, n_winners, rng);
    let indices = Tensor::<B, 1, Int>::from_data(TensorData::new(winners, [n_winners]), device);
    population.clone().select(0, indices)
}
```

Three things in those three lines are the whole **host-decide → single-gather**
pattern:

- **All the *deciding* happens on the host.** Every random draw and every
  comparison lives inside `tournament_indices_host`, working on the `&[f32]`
  fitness slice — never a tensor. That is what keeps the operator reproducible
  under the host-RNG convention: no `Tensor::random`, no backend RNG, just the
  caller's `rng`.
- **The device is touched exactly once.** The winner indices cross to the device
  in a single `from_data`, and a single `select` does all the gathering. There is
  no per-winner indexing loop and no host round-trip mid-selection — selecting
  `n_winners` parents from an `N`-member population costs one gather regardless of
  how many tournaments ran.
- **`select` indexes rows, so duplicates are free.** Tournament selection draws
  *with replacement*, so the same index can win several tournaments;
  `select(0, indices)` simply emits that row once per occurrence, which is exactly
  the multiset of parents the algorithm wants.

`truncation_select` has the identical shape — it forwards to
`truncation_indices_host` and gathers — the only difference being that its core
is deterministic and so takes no `rng`. Strategies almost always call the
`*_select` wrappers; the bare `*_indices_host` cores exist for when you need the
indices themselves (logging a lineage, gathering a *second* tensor by the same
winners) or want to unit-test selection without a device in play.

### Tournament selection

\\(k\\)-ary tournament: draw `tournament_size` candidate indices uniformly at random
*with replacement*, keep the one with the highest fitness, and repeat
`n_winners` times.

```rust
pub fn tournament_indices_host(
    fitness: &[f32],
    tournament_size: usize,
    n_winners: usize,
    rng: &mut dyn Rng,
) -> Vec<i32>;
```

`tournament_size` is the **selection-pressure knob**: larger tournaments make it
ever more likely the best individuals win, driving the population toward the
current best faster (and risking premature convergence); `tournament_size = 2`
is the gentle default. For a population of size `N` and binary tournaments, the
unique best member wins any single draw with probability

```math
P(\text{best wins}) = 1 - \left(\frac{N-1}{N}\right)^{2},
```

which the in-source tests verify empirically — the contributor book walks
through that test and the operator-testing patterns behind it in
[Adding an evolutionary strategy](https://github.com/anthonytorlucci/rlevo/blob/main/docs/contributor-book/src/ch06-adding-an-evolutionary-strategy.md).
Panics if `fitness` is empty or `tournament_size < 2` (a tournament needs at
least two contenders).

### Truncation selection

Deterministic: sort by fitness descending and take the `top_k` highest, returned
best-first. Ties break by `f32::partial_cmp` with `NaN` sorted last (so a stray
`NaN` fitness can never masquerade as a good solution).

```rust
pub fn truncation_indices_host(fitness: &[f32], top_k: usize) -> Vec<i32>;
```

This is the workhorse behind the \\((\mu+\lambda)\\) and \\((\mu,\lambda)\\) replacement strategies as
well as ES-style parent selection. Panics if `top_k > fitness.len()` or
`fitness` is empty.

## Crossover — how two parents combine

With parents chosen, the second move recombines them. Crossover operators
consume two parent tensors of shape `(N, D)` and produce one offspring tensor of
the same shape, drawing their randomness from the host `rng`. `rlevo` ships two
real-valued recombinations plus a binary counterpart.

> **What's *not* here yet.** The parent chapter mentions single-point crossover
> and Simulated Binary Crossover (SBX) as part of the GA tradition. Those are
> textbook context, not the current `ops` surface — today the module implements
> **BLX-α** and **uniform** crossover. The catalogue below describes what you
> can call now.

### BLX-α (blend crossover)

For each gene, the child is drawn uniformly from an interval that *extends
beyond* the two parents in proportion to their distance:

```math
\text{child}_i \sim \mathcal{U}\!\left(\min(a_i, b_i) - \alpha\,|a_i - b_i|,\ \ \max(a_i, b_i) + \alpha\,|a_i - b_i|\right)
```

With \\(\alpha = 0\\) the child lies strictly inside the parents' bounding box (pure
interpolation); \\(\alpha = 0.5\\) is the conventional default and allows mild
extrapolation past either parent, which keeps the search from collapsing into
the convex hull of the population. Internally it is pure tensor algebra —
`min_pair`/`max_pair` to bracket the parents, then `lo + u * (hi - lo)` with a
host-sampled uniform `u` — so it runs entirely on-device after the one data
load. Panics if the two parents differ in shape.

### Uniform crossover

A per-gene Bernoulli swap: each gene takes parent A's allele (value) with
probability \\(p\\), parent B's otherwise. No blending happens, so the set of gene
values in the offspring is exactly the union of the parents' values — the
marginal distribution per locus is preserved. \\(p = 0.5\\) is an unbiased mix; \\(p = 1.0\\)
clones A; \\(p = 0.0\\) clones B. Implemented with a host-sampled uniform mask and
`mask_where`.

`binary_uniform_crossover` is the same operation on `Tensor<B, 2, Int>` genomes
— a pure bitwise swap for binary-coded GAs and EDAs.

## Mutation — where novelty comes from

Crossover only ever shuffles and blends genes the parents already carry; it can
never introduce a value the population has lost. Mutation is the move that
*does* — it injects the diversity that selection then exploits. Each operator
consumes a population tensor and returns a perturbed copy, with all noise drawn
from the host `rng`.

| Operator | Genome | What it does |
| -------- | ------ | ------------ |
| `gaussian_mutation` | real | adds isotropic \\(\sigma \cdot \mathcal{N}(0, 1)\\) noise to every gene |
| `gaussian_mutation_per_row` | real | per-individual \\(\sigma\\) — a `(N,)` tensor of step-sizes |
| `uniform_reset` | real / bounded | with probability \\(p\\), resets a gene to \\(\mathcal{U}(lo, hi)\\) |
| `bit_flip_mutation` | binary | flips each gene with probability \\(p\\) |

**Gaussian mutation** is the staple of continuous evolution. A scalar \\(\sigma\\)
controls the step size globally; \\(\sigma = 0\\) is an exact identity (useful as a
no-op in tests and ablations). Its sibling `gaussian_mutation_per_row` takes a
`(N,)` tensor of per-individual step-sizes — isotropic *within* a row but
varying *across* the population — which is exactly the shape self-adaptive
\\((1+1)\\)-ES and CMA-ES warm-starts want when each individual carries its own \\(\sigma\\).

**Uniform-reset** replaces a gene outright with a fresh draw from `U(lo, hi)`
rather than nudging it — well suited to bounded or integer-coded genomes where
additive noise would drift out of range. \\(p = 0\\) is identity; \\(p = 1\\)
reinitialises the whole population.

**Bit-flip** is the binary analogue, computing the flip arithmetically as
\\(1 - x\\) under a Bernoulli mask. The textbook rate is \\(1/D\\) (one expected flip per
individual), but the function imposes no particular rate — that is the caller's
to choose. The input must hold only `{0, 1}`; values outside that set produce
out-of-range results silently.

## Replacement — who survives to the next generation

The final move closes the generation: replacement (survivor selection) decides
which individuals form the next generation's starting population. Unlike the
others, **these operators draw no random numbers** — they are fully
deterministic given their inputs, operating on host-side fitness slices and
lifting winners back with a single gather. Each takes the current generation
plus the offspring and returns the `(population, fitness)` pair to carry forward.

| Strategy | Parent survival | Reach for it when… |
| -------- | --------------- | ------------------ |
| `generational` | none | offspring quality is trusted — classic GA, CMA-ES |
| `elitist` | top-\\(k\\) | losing the best-so-far would hurt |
| `mu_plus_lambda` | best of the merged \\(\mu+\lambda\\) pool | strong elitism — ES, DE |
| `mu_comma_lambda` | none (offspring only) | deliberate forgetting — tracking moving optima |

**Generational** is the simplest: discard the parents wholesale and let the
offspring become the next generation. **Elitist** softens that by carrying the
\\(k\\) fittest (highest-fitness) parents forward and backfilling with the best `pop_size − k`
offspring — a direct implementation of De Jong's elitism, which guarantees the
best solution found so far never regresses.

The two ES-style strategies differ only in whether parents compete.
**\\((\mu+\lambda)\\)** merges the \\(\mu\\) parents and \\(\lambda\\) offspring into one pool and keeps the \\(\mu\\)
best overall, so a strong parent can survive indefinitely — maximal selection
pressure and elitism. **\\((\mu,\lambda)\\)** ignores the parents entirely and keeps the \\(\mu\\)
best *offspring* (requiring \\(\lambda \geq \mu\\)); deliberately forgetting good parents lets
the population escape local optima and track optima that move over time. Both
return rows ordered best-first.

## Looking ahead: fused kernels

The operators above follow the host-sample → device-load pattern, which keeps
them backend-agnostic and reproducible but pays for a host round-trip on the
random draws. The `ops::kernels` submodule is the reserved home for custom
`CubeCL` fused kernels that would generate noise and apply it in a single device
pass, gated behind the `custom-kernels` crate feature. It is empty for now — the
correctness-and-reproducibility baseline comes first; the fused fast path is
tracked follow-up work.

## Putting it together

We have now walked all four moves: **select** parents, **recombine** them,
**mutate** the result, and choose **survivors**. That is exactly the four-line
recipe this chapter opened with, and now every call in it has a catalogue behind
it. Because the operators are free functions over `Tensor<B, _>` with a shared
canonical-maximise convention and a shared host-RNG discipline, a `Strategy`
implementation reads as a short, readable recipe — and swapping tournament for
truncation, or \\((\mu+\lambda)\\) for \\((\mu,\lambda)\\), is a one-line change. The full GA and ES
pseudocode that assembles these operators end-to-end lives in
[Appendix A](../../appendix-a-ec-algorithms/index.md).

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
