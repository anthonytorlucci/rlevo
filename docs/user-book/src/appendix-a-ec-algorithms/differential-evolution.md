# Differential Evolution

<!-- source: crates/rlevo-evolution/src/algorithms/de.rs -->

Differential Evolution (DE) is a population-based continuous optimiser
(Storn & Price, 1997) that constructs new candidate solutions by combining
existing population members arithmetically — no crossover in the genetic sense,
no parent selection before mutation. Its key insight: the scaled difference
between two randomly chosen individuals produces a step that is automatically
calibrated to the current spread of the population.

DE is often competitive with or superior to the [real-valued GA](real-valued-genetic-algorithm.md)
and classical [Evolution Strategies](evolution-strategies.md) on multi-modal
continuous landscapes, and requires only two tuning parameters (`F` and `CR`).

## Variants

`DifferentialEvolution` selects its mutation and crossover strategy through the
`DeVariant` enum. All five variants apply the same greedy per-slot replacement
in `tell`.

| Variant | Mutation formula | Crossover | Notes |
|---|---|---|---|
| `Rand1Bin` | `x_{r1} + F·(x_{r2} − x_{r3})` | Binomial | **Recommended default.** Balanced exploration; converges to optimum on Sphere |
| `Rand1Exp` | `x_{r1} + F·(x_{r2} − x_{r3})` | Exponential | Same mutation as `Rand1Bin`; contiguous gene mask instead of independent flips |
| `Rand2Bin` | `x_{r1} + F·(x_{r2}−x_{r3}) + F·(x_{r4}−x_{r5})` | Binomial | Higher variance; two difference vectors; needs pop ≥ 6; slower convergence |
| `Best1Bin` | `x_{best} + F·(x_{r2} − x_{r3})` | Binomial | Strong exploitation; **prone to premature convergence** (see below) |
| `CurrentToBest1Bin` | `x_i + F·(x_{best}−x_i) + F·(x_{r1}−x_{r2})` | Binomial | Hybrid; useful on multimodal problems; still susceptible to premature convergence |

**Binomial crossover** independently includes each gene from the mutant vector
with probability `CR`, with a `j_rand` guarantee that at least one gene always
comes from the mutant. **Exponential crossover** selects a contiguous block of
genes starting at a random position, extending while `U[0,1) < CR`; the block
wraps around the genome. Both enforce the `j_rand` guarantee.

**Premature convergence in best-biased variants.** `Best1Bin` and
`CurrentToBest1Bin` use the current best individual as the mutation base.
As the population converges around the best, the difference vectors
`F·(xᵣ₂ − xᵣ₃)` shrink toward zero and exploration stops. On Sphere-D10 with
500 generations, `Rand1Bin` reaches `< 1e-20` while `Best1Bin` stalls near 1.
Use best-biased variants when fast exploitation of a known promising region is
more important than finding the global optimum.

## Configuration

```rust,no_run
use rlevo::evo::algorithms::de::{DeConfig, DeVariant};

// Explicit construction:
let config = DeConfig {
    pop_size:   30,
    genome_dim: 10,
    bounds:     (-5.12, 5.12),
    f:          0.5,
    cr:         0.9,
    variant:    DeVariant::Rand1Bin,
};

// Or use the defaults (Rand1Bin, F = 0.5, CR = 0.9):
let config = DeConfig::default_for(30, 10);
```

| Field | Type | Typical range | Notes |
|---|---|---|---|
| `pop_size` | `usize` | 10·D – 30·D | DE literature recommends 10×D as a starting point; `Rand2Bin` needs ≥ 6 |
| `genome_dim` | `usize` | problem-defined | Dimensionality of the search space |
| `bounds` | `(f32, f32)` | problem-defined | Initial population sampled here; trial vectors clamped after crossover |
| `f` | `f32` | 0.4 – 0.9 | Differential weight; higher F → more exploration, slower convergence |
| `cr` | `f32` | 0.1 – 0.9 | Crossover probability; higher CR → more genes from mutant vector |
| `variant` | `DeVariant` | — | See table above; `Rand1Bin` is the canonical starting point |

A common rule of thumb: start with `F = 0.5`, `CR = 0.9`, `pop_size = 10·D`,
`Rand1Bin`. Tune `F` first — it is the more sensitive parameter.

## Fitness convention

All strategies in `rlevo::evo` treat fitness as **cost** — lower is better.
Maximization problems must be negated. Greedy replacement (`trial ≤ current`)
uses the same convention: the trial survives if its cost is no worse.

## Minimal example

```rust,no_run
use burn::backend::Flex;
use burn::tensor::{Tensor, TensorData, backend::Backend};
use rlevo::evo::algorithms::de::{DeConfig, DeVariant, DifferentialEvolution};
use rlevo::evo::fitness::BatchFitnessFn;
use rlevo::evo::strategy::EvolutionaryHarness;

type B = Flex;

/// Sphere function: f(x) = Σ xᵢ², minimum 0 at the origin.
struct SphereCost;

impl BatchFitnessFn<B, Tensor<B, 2>> for SphereCost {
    fn evaluate_batch(
        &mut self,
        population: &Tensor<B, 2>,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Tensor<B, 1> {
        let [pop_size, _] = population.dims();
        let data = population
            .clone()
            .into_data()
            .into_vec::<f32>()
            .unwrap_or_default();
        let dim = data.len() / pop_size;
        let fitness: Vec<f32> = (0..pop_size)
            .map(|row| (0..dim).map(|col| data[row * dim + col].powi(2)).sum::<f32>())
            .collect();
        Tensor::<B, 1>::from_data(TensorData::new(fitness, [pop_size]), device)
    }
}

fn main() {
    let device = Default::default();
    let config = DeConfig::default_for(30, 10);   // Rand1Bin, F=0.5, CR=0.9

    let mut harness = EvolutionaryHarness::<B, _, _>::new(
        DifferentialEvolution::<B>::new(),
        config,
        SphereCost,
        /* seed */ 42,
        device,
        /* max_generations */ 500,
    );

    harness.reset();
    loop {
        let result = harness.step(());
        if result.done {
            break;
        }
    }

    let best = harness.latest_metrics().unwrap().best_fitness_ever;
    println!("best cost = {best:.4e}");   // converges near 0
}
```

## Implementation notes

**Greedy per-slot replacement.** Unlike GA or ES, DE does not rank the whole
population and select survivors globally. Each trial individual `u_i` is
compared only against its corresponding current individual `x_i`: if
`f(u_i) ≤ f(x_i)` the slot is updated, otherwise `x_i` is kept. This means
DE is fully population-diverse — no individual is ever displaced by someone
from a different slot.

**`j_rand` guarantee.** Both crossover schemes guarantee that at least one gene
per trial individual comes from the mutant vector. Without this, when `CR` is
low, an individual could pass through `tell` entirely unchanged (equal to its
current self), making the trial evaluation pointless. The `j_rand` gene index
is drawn host-side from `seed_stream`.

**Minimum population size.** `Rand1Bin` and `Rand1Exp` require 4 distinct
indices (i, r1, r2, r3), so `pop_size ≥ 4`. `Rand2Bin` requires 6
(`i, r1, r2, r3, r4, r5`). The implementation panics in debug mode if
`pop_size ≤ k` (the required number of distinct indices).

**First-call behaviour.** Same pattern as all other strategies: the first
`ask` returns the initial population for evaluation; the first `tell`
populates the fitness cache and initializes `best_genome` without running
replacement. All subsequent cycles produce and evaluate trial vectors.

**Reproducibility.** Mutation indices (`SeedPurpose::Trial`) and crossover
masks (`SeedPurpose::Crossover`) are drawn from independent host-side
sub-streams via `seed_stream`. Two runs with the same seed produce identical
trajectories.

## When to use

| Situation | Recommendation |
|---|---|
| Continuous optimisation, general case | `Rand1Bin` is an excellent default |
| Multi-modal landscape | `Rand1Bin` or `Rand2Bin`; increase `pop_size` |
| Fast exploitation of a known region | `Best1Bin` or `CurrentToBest1Bin` (accept premature convergence risk) |
| High dimensionality (D > 50) | Increase `pop_size` to ~10·D; CMA-ES may scale better once implemented |
| Binary or combinatorial search space | [Binary GA](binary-encoded-genetic-algorithm.md); DE is defined over real-valued differences |
| Strong variable dependencies | CMA-ES (adapts covariance); DE's random-pair differences assume weak separability |

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
