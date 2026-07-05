# Gene Expression Programming

<!-- source: crates/rlevo-evolution/src/algorithms/gep/ -->
<!-- source: crates/rlevo-evolution/src/function_set.rs -->

**Gene Expression Programming** (GEP, Ferreira 2001) evolves expression trees
through a *linear, fixed-length* chromosome. It sits between the two extremes of
program evolution: tree-based GP mutates the tree directly and must guard every
operator against producing an ill-formed program, while
[Cartesian GP](cartesian-genetic-programming.md) evolves a fixed-grid integer
graph. GEP keeps a fixed-length string genotype — so its operators are simple
array edits like a GA's — yet decodes that string to a variable-shape tree
phenotype, the natural representation for symbolic expressions.

The headline property is **repair-free validity**: the chromosome is split into
a `head` (any symbol) and a `tail` (terminals only), with the tail sized so that
*every* chromosome respecting that split decodes to a complete tree. No operator
ever needs a repair pass, and unlike CGP and the
[Evolution Strategies](evolution-strategies.md), GEP has real **crossover**.

`rlevo` implements GEP as `GepStrategy<B, F>` under
`rlevo::evo::algorithms::gep`, generic over a `FunctionSet F` and sharing the
`ArithmeticFunctionSet` and `Symbol` machinery with CGP. The genome is
`Tensor<B, 2, Int>` of shape `(pop_size, head_len + tail_len)`; decoding and
evaluation run host-side, per chromosome.

## The head/tail chromosome

A chromosome is `head_len + tail_len` integer symbol ids. The `head` may hold
any symbol — functions or terminals — while the `tail` holds **terminals only**.
The tail length is not a free parameter; `GepConfig::new` derives it from the
head length and the function set's maximum arity \\(a_{\max}\\):

```math
T = H\,(a_{\max} - 1) + 1
```

for head length \\(H\\). This is the minimum tail that guarantees a repair-free
decode. The worst case is a head of all-maximum-arity functions: \\(H\\) functions
of arity \\(a_{\max}\\) demand \\(H\,(a_{\max} - 1) + 1\\) terminals to close every
open branch, and a tail of exactly that length supplies them. Any shorter tail
could leave a branch unfilled; any chromosome with a terminal-only tail of this
length always closes. For the default `ArithmeticFunctionSet`
(\\(a_{\max} = 2\\)) and a head of 7, the tail is \\(7 \cdot 1 + 1 = 8\\) and the
genome length 15.

Only the chromosome's *coding prefix* — the open reading frame — contributes to
the phenotype; the rest is **non-coding "junk DNA"** carried silently in the
genome. As in CGP's inactive grid nodes, this redundancy is the substrate that
lets neutral variation accumulate without disturbing fitness.

## The symbol alphabet

`Alphabet<F>` wraps a `FunctionSet` with the GEP-only terminal layer (input
variables and problem constants) that CGP does not need. Every symbol gets a
contiguous non-negative `i32` id in three blocks, so the head and tail sampling
ranges are plain integer intervals:

| Block | Id range | Decodes to |
|---|---|---|
| Functions | \\(0 \ldots n_f\\) | opcode from `F`, arity from the function set |
| Variables | \\(n_f \ldots n_f + n_v\\) | input variable, `input_index = id - n_f` |
| Constants | \\(n_f + n_v \ldots \texttt{len}\\) | `constants[id - n_f - n_v]` |

where \\(n_f\\) is the function count and \\(n_v\\) the variable count. A
**terminal** is any arity-0 symbol: the arity-0 *functions* (the constant
\\(1.0\\) opcode), the variables, and the problem constants. `terminal_range`
returns the half-open id interval covering all of them. This is contiguous only
if the function set's arity-0 functions are its trailing ids — true for
`ArithmeticFunctionSet`, whose sole zero-arity opcode (`const 1.0`) is last — and
`Alphabet::new` debug-asserts the property. Head loci sample uniformly from
\\([0,\ \texttt{len})\\); tail loci sample from `terminal_range`.

The eight-opcode `ArithmeticFunctionSet` is the same one
[CGP documents](cartesian-genetic-programming.md#function-set): add, sub, mul,
`protected_div` (guarded at \\(\lvert b\rvert < 10^{-6}\\)), sin, cos, tanh, and
the constant \\(1.0\\). GEP adds variables and optional problem constants on top.

## Decoding: open reading frame, then breadth-first tree

`GepDecoder` implements the canonical Ferreira decode in two passes.

**1. ORF-length scan.** A single left-to-right pass tracks the number of
still-unfilled child slots, starting at 1 (the root). Reading a symbol \\(s\\)
fills one slot and opens `arity(s)` new ones:

```math
\texttt{needed} \leftarrow \texttt{needed} - 1 + \operatorname{arity}(s),
\qquad \texttt{needed} = 1 \text{ initially},
```

and the coding region ends the moment `needed` reaches \\(0\\). The tail-length
constraint guarantees this happens within the chromosome.

**2. BFS child assignment.** Because the coding prefix is already in level
order, a node's children are simply the next unread symbols. A single running
read cursor assigns each node `arity` contiguous children — array order *is* BFS
order, so no explicit queue is needed. The result is an `ExpressionTree` stored
in level-order layout, where every node's children occupy a contiguous,
strictly-higher index range.

An all-terminal head decodes to a single-node tree; a head of nested functions
decodes to a deeper tree whose size is the ORF length, never exceeding
`genome_len`.

## Phenotype evaluation

`ExpressionTree::eval` scores the tree on one input row in a single
**right-to-left sweep**. Because children always have higher indices than their
parent, walking indices from last to first guarantees a node's children are
already resolved when it is reached — no recursion, no stack:

- A **variable** node resolves to `inputs[input_index]` (a missing index reads
  as \\(0.0\\)).
- A **constant** node resolves to its stored value.
- A **function** node calls `FunctionSet::apply` on its children's
  already-computed results. A non-finite result collapses to \\(0.0\\), the same
  robustness rule the [CGP evaluator](cartesian-genetic-programming.md#phenotype-evaluation)
  uses.

The root's value (`results[0]`) is the program output. Evaluation is host-side
and, in the fitness function, row-parallel across the population with `rayon`;
because decoding is deterministic, the parallel order never affects results.

## Genetic operators

Every operator preserves the head/tail position-class invariant and the fixed
chromosome length, so offspring **always decode to a complete tree** — verified
by property tests (1000 mutations leave zero tail violations; 500 trials per
operator all decode complete). Operators act on host-side `&mut [Symbol]`
chromosomes, co-located with the host-side roulette selection they share a
device round-trip with.

**Point mutation** is the one operator that needs a sampling constraint, and it
enforces the invariant directly: each locus mutates independently with
probability `mutation_rate`; a head locus is replaced by any symbol, a tail
locus by a terminal drawn from `terminal_range`. The tail therefore stays
terminal-only without a repair pass.

**IS transposition** (insertion sequence) copies a short sequence — length \\(1\\)
to \\(3\\) — and inserts it at a non-root head locus, shifting the rest of the
head right and dropping the overflow off the head end. The tail is untouched, so
it cannot break validity (any symbol is legal in the head).

**RIS transposition** (root insertion sequence) scans the head from a random
offset for the first function symbol, copies the sequence starting there to head
position \\(0\\), and shifts right with overflow dropped. It guarantees the root
becomes a function, deepening the tree. The tail is untouched.

**One- and two-point crossover** swap class-aligned segments between two
equal-layout parents — a suffix past one cut, or the middle segment between two
cuts. Because both parents share the same head/tail boundary, a tail locus
always exchanges with a tail locus, so the tail stays terminal-only by
construction. This is the recombination operator CGP and ES lack.

## The generational engine

`GepStrategy` is a generational `Strategy` with roulette-wheel selection and
elitism (Ferreira 2001). Each `ask` after the first breeds a new generation from
the cached one:

1. **Roulette selection** of `pop_size` parents. The weight of an individual
   with error \\(\texttt{mse}\\) is \\(1 / (1 + \texttt{mse})\\), so a smaller error
   yields a larger weight; non-finite fitness contributes zero weight, and a
   degenerate all-diverged population falls back to uniform sampling.
2. **Crossover** over consecutive offspring pairs — one-point then two-point,
   each gated by its own rate.
3. **Transposition then point mutation** per individual — IS and RIS gated by
   their rates, then locus-class point mutation.
4. **Elitism.** The best-so-far genome is copied unchanged into row \\(0\\), so
   the incumbent can never be lost to a bad generation.

All randomness is host-side. A single base seed per generation spawns four
independent `seed_stream`s — `Selection`, `Crossover`, `Transposition`,
`Mutation` — each keyed by the generation counter, so two runs with the same
seed and config are identical. Decoding and evaluation are deliberately **not**
part of the strategy: they live in the `BatchFitnessFn` (`GepSymRegression`),
keeping the ask/tell loop free of program interpretation. `tell` caches the
population and its fitness for the next roulette draw and updates the best-so-far
record on strict improvement.

## Configuration

`GepConfig::new(head_len, max_arity, n_vars, pop_size)` derives the tail length
and fills the operator rates with canonical Ferreira defaults; mutate the public
fields afterwards to override.

```rust,no_run
use rlevo::evo::algorithms::gep::GepConfig;
use rlevo::evo::function_set::{ArithmeticFunctionSet, FunctionSet};

// head 7, max_arity 2 -> tail 8, genome 15:
let mut cfg = GepConfig::new(7, ArithmeticFunctionSet.max_arity(), 1, 100);
cfg.crossover_1p_rate = 0.4; // override a default if desired
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `head_len` | `usize` | problem-defined | Loci that may hold any symbol; bounds program size |
| `tail_len` | `usize` | derived | \\(H(a_{\max} - 1) + 1\\); not set directly |
| `pop_size` | `usize` | problem-defined | Individuals per generation |
| `n_vars` | `usize` | problem-defined | Input variables; must match the alphabet |
| `mutation_rate` | `f32` | \\(2 / L\\) | Per-locus point-mutation probability (~2 genes per chromosome) |
| `is_transpose_rate` | `f32` | 0.1 | Per-individual IS transposition probability |
| `ris_transpose_rate` | `f32` | 0.1 | Per-individual RIS transposition probability |
| `crossover_1p_rate` | `f32` | 0.3 | Per-pair one-point crossover probability |
| `crossover_2p_rate` | `f32` | 0.3 | Per-pair two-point crossover probability |

`new` panics if `head_len`, `max_arity`, `n_vars`, or `pop_size` is zero — each
would make the genome layout or tree decode degenerate. A larger `head_len`
admits larger expressions at the cost of a longer genome and deeper trees.

## Fitness convention

All strategies in `rlevo::evo` maximise a **canonical** fitness — higher is better. You declare a cost objective's direction with [`ObjectiveSense::Minimize`](https://docs.rs/rlevo-core) and the harness reconciles it at one chokepoint, so you never hand-negate. `GepSymRegression` scores each program by mean squared error against the targets: declare `ObjectiveSense::Minimize` and the harness reconciles — MSE is **not** hand-negated. The roulette weight \\(1/(1+\texttt{mse})\\) is an internal sampling convenience and does not affect what the harness maximises.

## Minimal example

Symbolic regression recovering \\(x^2 + x + 1\\) over twenty points in
\\([-1, 1]\\). The alphabet is built once and shared (by clone) between the
strategy and the fitness function.

```rust,no_run
use burn::backend::Flex;
use rlevo::evo::algorithms::gep::{Alphabet, GepConfig, GepStrategy, GepSymRegression};
use rlevo::evo::function_set::{ArithmeticFunctionSet, FunctionSet};
use rlevo::evo::strategy::EvolutionaryHarness;

type B = Flex;

fn main() {
    let device = Default::default();

    // 20 evenly spaced x ∈ [−1, 1] and targets f(x) = x² + x + 1.
    let xs: Vec<f32> = (0..20).map(|i| -1.0 + 2.0 * (i as f32) / 19.0).collect();
    let inputs: Vec<Vec<f32>> = xs.iter().map(|&x| vec![x]).collect();
    let targets: Vec<f32> = xs.iter().map(|&x| x * x + x + 1.0).collect();

    // One variable (x), no extra constants beyond the function set's const 1.0.
    let alphabet = Alphabet::new(ArithmeticFunctionSet, /* n_vars */ 1, vec![]);
    let cfg = GepConfig::new(/* head_len */ 7, alphabet.max_arity(), /* n_vars */ 1, /* pop */ 100);
    let genome_len = cfg.genome_len();

    let strategy = GepStrategy::<B, _>::new(alphabet.clone());
    let fitness = GepSymRegression::new(alphabet, genome_len, inputs, targets);

    let mut harness = EvolutionaryHarness::<B, _, _>::new(
        strategy,
        cfg,
        fitness,
        /* seed */ 11,
        device,
        /* max_generations */ 500,
    ).expect("valid config");

    harness.reset();
    loop {
        if harness.step(()).done {
            break;
        }
    }

    let best = harness.latest_metrics().unwrap().best_fitness_ever;
    println!("best MSE = {best:.4e}"); // converges to a near-exact fit (≤ 1e-2)
}
```

For two-variable targets such as \\(x^2 + y^2\\), build the alphabet and config
with `n_vars = 2` and supply each input row as `vec![x, y]`.

## Implementation notes

**Repair-free by construction, not by check.** Validity rests on two invariants
the operators preserve mechanically: the tail holds only terminals, and the
chromosome length is fixed. Point mutation enforces the first by sampling tail
loci from `terminal_range`; transposition rewrites only the head; crossover is
class-aligned. Nothing ever validates or repairs an offspring — the property
tests confirm the invariants hold rather than guard against their violation.

**Tail length is derived, not configured.** `tail_len` is a read-only
consequence of `head_len` and \\(a_{\max}\\). Passing the wrong `max_arity` to
`GepConfig::new` (smaller than the function set's true maximum) would size the
tail too short and could yield an incomplete decode, so derive it from the
alphabet — `alphabet.max_arity()` — rather than hard-coding a literal.

**Integer genome, host decode.** Like CGP, the genotype travels with the
population as `Tensor<B, 2, Int>` but is pulled to host (`into_vec::<i32>`) for
mutation and decoding. The device round-trip keeps storage uniform with the
real-valued strategies; the per-node tree walk never touches the device.

**Out-of-range symbols are inert.** A symbol id outside the alphabet classifies
as an inert zero-arity function rather than panicking, mirroring CGP's index
clamping. Combined with the non-finite collapse in `eval`, every integer
chromosome decodes to a finite program.

**Reproducibility.** The four operator streams derive from one base seed per
generation, keyed by the generation counter under distinct `SeedPurpose`s
(`Selection`, `Crossover`, `Transposition`, `Mutation`); initial sampling uses
`SeedPurpose::Init`. Changing any operator rate, the head length, or the
population reshapes the draw sequence, so seeds are not portable across configs.

## When to use

| Situation | Recommendation |
|---|---|
| Symbolic regression wanting tree-shaped expressions | Gene Expression Programming — linear genome, tree phenotype, repair-free operators |
| Want crossover / recombination on programs | GEP — its class-aligned crossover never produces an invalid program |
| Graph-structured reuse of subexpressions | [Cartesian GP](cartesian-genetic-programming.md) — a node can feed many others |
| GPU-heavy workload with a fixed computation grid | [Cartesian GP](cartesian-genetic-programming.md) — genotype lives on device, regular grid sweep |
| Continuous parameter-vector optimisation | [Evolution Strategies](evolution-strategies.md) or [CMA-ES](cma-es.md) — an integer program genome is the wrong tool |
| Binary or real GA-style search | [Binary GA](binary-encoded-genetic-algorithm.md) / [Real-Valued GA](real-valued-genetic-algorithm.md) |
| Maximum expressive freedom, CPU budget acceptable | Tree-based GP (deferred) — unconstrained depth and shape, at the cost of closure/type-safe operators |

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
