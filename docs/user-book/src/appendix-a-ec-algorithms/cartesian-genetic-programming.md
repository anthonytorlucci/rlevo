# Cartesian Genetic Programming

<!-- source: crates/rlevo-evolution/src/algorithms/gp_cgp.rs -->
<!-- source: crates/rlevo-evolution/src/function_set.rs -->

**Cartesian Genetic Programming** (CGP) evolves *programs* rather than vectors.
Where the [Real-Valued GA](real-valued-genetic-algorithm.md) and
[Evolution Strategies](evolution-strategies.md) search a fixed-length real
genome, CGP searches a fixed-length *integer* genome that decodes to a directed
acyclic computation graph on a `rows × cols` grid. Each node names a function
and the wires feeding it; evolution rewires the graph and swaps operators to fit
a target mapping. The canonical application is symbolic regression — recovering
a closed-form expression that explains a dataset.

`rlevo` implements classical CGP as `CartesianGeneticProgramming` under
`rlevo::evo::algorithms::gp_cgp`. Two design choices set it apart from the
real-valued strategies and motivate the rest of this page:

- The genome is `Tensor<B, 2, Int>`, not a float tensor. Populations are integer
  index vectors, so the genotype storage stays on-device like every other
  strategy while phenotype evaluation runs on the host.
- The engine is a hand-written `(1 + λ)` Evolution Strategy — point mutation,
  **no crossover** — specialised to CGP genome semantics rather than reusing
  [`es_classical`](evolution-strategies.md). The specialisation is the
  feed-forward connection constraint: a mutated wire must still point backwards
  in the graph, which the generic ES has no notion of.

## Genome layout

The genome is a flat integer vector of length

```math
L = R \cdot C \cdot 3 + 1
```

for an \\(R \times C\\) grid (`rows`, `cols`). Each node occupies three
consecutive genes — `GENES_PER_NODE = 3` — laid out
\\((\texttt{function\_id},\ \texttt{input\_0},\ \texttt{input\_1})\\), and a
single trailing **output gene** (`OUTPUT_GENES = 1`) names the node whose value
is taken as the program's result. `CgpConfig::genome_len` returns \\(L\\).

Wires are addressed in a single index space shared by graph inputs and node
outputs. For `n_inputs` independent variables and \\(R \cdot C\\) nodes:

| Index range | Refers to |
|---|---|
| \\(0 \ldots \texttt{n\_inputs} - 1\\) | the graph inputs (independent variables) |
| \\(\texttt{n\_inputs} \ldots \texttt{n\_inputs} + R\,C - 1\\) | node outputs, in column-major grid order |

So `input_0 = 0` reads the first graph input, while
`input_0 = n_inputs` reads the first node's output. The output gene draws from
the same space, \\([0,\ \texttt{n\_inputs} + R\,C)\\), and may therefore select a
graph input directly (an identity program) or any node.

**Active vs. inactive genes.** Only the nodes on a path from the output gene
back to the inputs contribute to the result; the rest are *inactive* (non-coding)
but still carried in the genome. This redundancy is not waste — it is the
substrate for the neutral drift discussed under [Selection](#selection-and-neutral-drift).

## Function set

The v1 function set is the shared `ArithmeticFunctionSet`, eight opcodes with
ids \\(0 \ldots 7\\):

| id | op | arity | result |
|---|---|---|---|
| 0 | add | 2 | \\(a + b\\) |
| 1 | sub | 2 | \\(a - b\\) |
| 2 | mul | 2 | \\(a \cdot b\\) |
| 3 | `protected_div` | 2 | \\(a / b\\), or \\(a\\) if \\(\lvert b\rvert < 10^{-6}\\) |
| 4 | sin | 1 | \\(\sin a\\) |
| 5 | cos | 1 | \\(\cos a\\) |
| 6 | tanh | 1 | \\(\tanh a\\) |
| 7 | const | 0 | \\(1.0\\) |

Opcode dispatch goes through the `FunctionSet` trait, not a hard-coded match, so
the same CGP engine runs any function set. `evaluate_cgp` is a thin wrapper that
calls the generic `evaluate_cgp_with` against `ArithmeticFunctionSet`; the
`&F` is monomorphised (never `&dyn FunctionSet`) so `apply` inlines in the
per-node × per-sample inner loop. The `FUNCTION_ARITIES` array
\\([2,2,2,2,1,1,1,0]\\) and `NUM_FUNCTIONS = 8` are retained because the mutation
logic samples function ids from \\(0 \ldots \texttt{NUM\_FUNCTIONS}\\).

The **protection** is deliberate and worth stating as a convention: division
guards against a near-zero denominator by returning the numerator, and any node
whose value comes back non-finite (an `inf` or `NaN` leaking through, say, a
chained division) collapses to \\(0.0\\). Combined with index clamping (below),
this means *every* integer genome decodes to a finite program — there is no such
thing as an invalid CGP genotype, so mutation never needs a repair pass.

## The `(1 + λ)` engine

CGP runs Rechenberg's `(1 + λ)` Evolution Strategy. One parent produces
\\(\lambda\\) offspring per generation by point mutation; the parent is then
replaced by the best offspring under the tie-break rule below. There is no
crossover and no σ self-adaptation — the only operator is per-gene mutation at a
fixed rate.

**Point mutation.** Each gene is independently mutated with probability
`mutation_rate`. The new value depends on which gene it is:

- **Function gene** (`within == 0`): resampled uniformly from \\(0 \ldots
  \texttt{NUM\_FUNCTIONS}\\).
- **Input gene** (`within ∈ {1, 2}`): resampled from the node's *feed-forward
  pool* — see below. A wire can only ever point backwards.
- **Output gene** (the trailing gene): resampled uniformly from \\([0,\
  \texttt{n\_inputs} + R\,C)\\).

The default `mutation_rate` is tuned to flip roughly three genes per genome:
\\(\texttt{mutation\_rate} = 3 / L\\). On the default \\(1 \times 30\\) grid that
is \\(3/91 \approx 0.033\\).

**Feed-forward constraint and levels-back.** A node in column \\(c\\) may only
draw inputs from the graph inputs or from node outputs in earlier columns. The
`levels_back` parameter bounds *how many* earlier columns: a node in column
\\(c\\) connects to columns \\([\,c - \texttt{levels\_back},\ c\,)\\). The default
is `usize::MAX` ("any previous column"), which with the default single-row grid
makes the graph a free feed-forward chain of 30 columns. Restricting
`levels_back` localises connectivity and is the classical lever for controlling
how reusable intermediate sub-results are.

Because mutation resamples input genes from this same pool, the acyclicity
invariant is preserved by construction — there is no separate validity check.

## Phenotype evaluation

`evaluate_cgp` decodes a host-side genome against a batch of input rows and
returns one output per row. Evaluation is **on the host**: the per-node opcode
dispatch is a poor fit for dense tensor kernels, so only the genotype storage
stays on-device. A scratch buffer of length \\(\texttt{n\_inputs} + R\,C\\) holds
the graph inputs followed by each node's value; nodes are computed in
topological order (left to right across columns), reading already-computed
slots:

```math
\texttt{buf}[\texttt{n\_inputs} + i]
  = f_i\bigl(\texttt{buf}[a_i],\ \texttt{buf}[b_i]\bigr),
\qquad i = 0, 1, \ldots, R\,C - 1
```

where \\(f_i\\) is node \\(i\\)'s opcode and \\(a_i, b_i\\) its two wire genes. The
opcode's arity selects how many arguments reach `apply`: arity-2 ops receive
both slots, arity-1 ops only the first, zero-arity constants an empty slice. The
program output is \\(\texttt{buf}[\texttt{output\_idx}]\\).

Two robustness rules keep evaluation total over *any* integer genome:

- **Index clamping.** An out-of-range wire or output index is clamped to the
  last buffer slot rather than panicking. This is what lets mutation skip a
  repair pass — the only structural cost is that a wildly out-of-range gene
  reads a fixed slot instead of erroring.
- **Non-finite collapse.** A node value that is not finite is stored as
  \\(0.0\\), so divisions and unbounded compositions cannot poison downstream
  nodes with `NaN`.

`evaluate_cgp` panics only on an *empty* genome (the output gene must exist).

## Selection and neutral drift

`tell` applies `(1 + λ)` selection with the **canonical CGP tie-break**: the
best offspring replaces the parent when its fitness is *less than or equal to*
the parent's,

```math
\text{replace parent} \iff \min_j f_{\text{offspring}, j} \le f_{\text{parent}}.
```

The non-strict \\(\le\\) is the entire point. A neutral mutation — one that lands
in the inactive portion of the genome and leaves fitness unchanged — is accepted
rather than rejected. Over many generations this lets the population drift
sideways across fitness-neutral genotypes, quietly accumulating inactive
material that a later mutation can activate. Empirically this neutral drift is a
large part of why CGP escapes local optima despite having no crossover and a
population of one; rejecting ties (\\(<\\)) measurably hurts search.

The best-so-far tracker is separate and uses *strict* improvement
(\\(<\\), ties broken to the lowest index), so `best` reports the true incumbent
even while the parent drifts neutrally. As with the other strategies, the first
`ask` returns the single parent unchanged for evaluation and the first `tell`
bootstraps the parent fitness without running selection.

## Configuration

```rust,no_run
use rlevo::evo::algorithms::gp_cgp::CgpConfig;

// Explicit construction:
let config = CgpConfig {
    lambda:        4,
    n_inputs:      1,
    rows:          1,
    cols:          30,
    mutation_rate: 3.0 / 91.0,   // ~3 genes per genome
    levels_back:   usize::MAX,   // any previous column
};

// Or use the defaults for a given input arity:
let config = CgpConfig::default_for(1);
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `lambda` | `usize` | 4 | Offspring per generation (the \\(\lambda\\) in \\((1 + \lambda)\\)) |
| `n_inputs` | `usize` | problem-defined | Number of independent variables the program reads |
| `rows` | `usize` | 1 | Grid rows; 1 makes the graph a linear column chain |
| `cols` | `usize` | 30 | Grid columns; bounds program length and depth |
| `mutation_rate` | `f32` | \\(3 / L\\) | Per-gene mutation probability; default flips ~3 genes |
| `levels_back` | `usize` | `usize::MAX` | Earlier-column reach of a node's wires; `MAX` = unrestricted |

The grid size \\(R \times C\\) caps the program's complexity: more nodes admit
larger expressions but enlarge the genome and slow per-genome evaluation. The
single-row default is the common starting point for symbolic regression; widen
`rows` only when the problem benefits from parallel sub-graphs feeding a common
output.

## Fitness convention

All strategies in `rlevo::evo` treat fitness as **cost** — lower is better.
Symbolic regression therefore minimises an error such as mean squared error
between the program's outputs and the targets; no negation is needed.
Maximisation objectives must be negated before they reach the harness.

## Minimal example

Symbolic regression recovering \\(x^2 + 1\\) over twenty points in
\\([-1, 1]\\). The fitness function decodes each genome with `evaluate_cgp` and
returns the mean squared error against the targets.

```rust,no_run
use burn::backend::Flex;
use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use rlevo::evo::algorithms::gp_cgp::{
    CartesianGeneticProgramming, CgpConfig, evaluate_cgp,
};
use rlevo::evo::fitness::BatchFitnessFn;
use rlevo::evo::strategy::EvolutionaryHarness;

type B = Flex;

/// Symbolic regression on f(x) = x² + 1 over 20 evenly spaced x ∈ [−1, 1].
struct SymRegression {
    params: CgpConfig,
    xs: Vec<f32>,
    ys: Vec<f32>,
}

impl SymRegression {
    fn new(params: CgpConfig) -> Self {
        let xs: Vec<f32> = (0..20).map(|i| -1.0 + 2.0 * (i as f32) / 19.0).collect();
        let ys: Vec<f32> = xs.iter().map(|x| x * x + 1.0).collect();
        Self { params, xs, ys }
    }
}

impl BatchFitnessFn<B, Tensor<B, 2, Int>> for SymRegression {
    fn evaluate_batch(
        &mut self,
        population: &Tensor<B, 2, Int>,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Tensor<B, 1> {
        let pop_size = population.dims()[0];
        let data: Vec<i64> = population
            .clone()
            .into_data()
            .into_vec::<i32>()
            .unwrap_or_default()
            .into_iter()
            .map(i64::from)
            .collect();
        let gl = self.params.genome_len();
        let inputs: Vec<Vec<f32>> = self.xs.iter().map(|&x| vec![x]).collect();
        let mut fitness = Vec::with_capacity(pop_size);
        for row in 0..pop_size {
            let genome = &data[row * gl..(row + 1) * gl];
            let preds = evaluate_cgp(genome, &self.params, &inputs);
            let mse: f32 = preds
                .iter()
                .zip(self.ys.iter())
                .map(|(p, y)| (p - y).powi(2))
                .sum::<f32>()
                / (self.ys.len() as f32);
            fitness.push(mse);
        }
        Tensor::<B, 1>::from_data(TensorData::new(fitness, [pop_size]), device)
    }
}

fn main() {
    let device = Default::default();
    let params = CgpConfig::default_for(1);

    let mut harness = EvolutionaryHarness::<B, _, _>::new(
        CartesianGeneticProgramming::<B>::new(),
        params.clone(),
        SymRegression::new(params),
        /* seed */ 21,
        device,
        /* max_generations */ 2000,
    );

    harness.reset();
    loop {
        if harness.step(()).done {
            break;
        }
    }

    let best = harness.latest_metrics().unwrap().best_fitness_ever;
    println!("best MSE = {best:.4e}"); // drives well below a constant-1 fit
}
```

## Implementation notes

**Integer genome, host evaluation.** The genotype is `Tensor<B, 2, Int>` so it
travels with the population like every other strategy, but `ask`/`tell` pull it
to the host (`into_vec::<i32>`, widened to `i64`) for mutation and decoding. The
device round-trip is the price of keeping the storage uniform; the per-node loop
itself never touches the device.

**No repair, ever.** Index clamping and non-finite collapse make every integer
vector a valid program, so mutation resamples genes freely without a
correctness check. The only invariant mutation must respect is feed-forward
acyclicity, and it gets that for free by resampling input genes from the same
backward-only pool the initial genome was built from.

**Reproducibility.** Offspring mutation draws from a `seed_stream` keyed by the
generation counter under `SeedPurpose::Mutation` (host-RNG convention). Two runs
with the same seed and the same `CgpConfig` produce identical trajectories;
changing `lambda`, the grid, or `levels_back` changes the draw sequence, so
seeds are not portable across configurations.

**Own engine, not `es_classical`.** Although the algorithm is a `(1 + λ)` ES,
the module re-implements that loop directly rather than calling
[`es_classical`](evolution-strategies.md). The connection constraints,
function-id range, and output-gene semantics are not expressible through the
real-valued ES's Gaussian mutation, so sharing the engine would cost more in
adapters than it saves.

## When to use

| Situation | Recommendation |
|---|---|
| Symbolic regression / want a readable closed-form program | Cartesian GP — the genome *is* the expression graph |
| Discrete program / graph structure search | Cartesian GP — fixed-length integer genome, no bloat |
| Continuous parameter vector optimisation | [Evolution Strategies](evolution-strategies.md) or [CMA-ES](cma-es.md) — CGP's integer genome is the wrong tool |
| Binary or real GA-style search | [Binary GA](binary-encoded-genetic-algorithm.md) / [Real-Valued GA](real-valued-genetic-algorithm.md) |
| Need crossover / linear chromosome program encoding | Gene Expression Programming (`gep`) — head/tail chromosome decoded to a tree |
| Very large grids / deep programs | Raise `cols` and cap `levels_back`; expect slower host evaluation per genome |

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
