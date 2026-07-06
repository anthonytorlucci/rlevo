# Binary Encoded Genetic Algorithm

<!-- source: crates/rlevo-evolution/src/algorithms/ga_binary.rs -->

The binary genetic algorithm (binary GA) evolves a population of fixed-length
bit strings — each gene is `0` or `1`. It is the canonical form of Holland's
original GA (1975) and remains the right tool when the search space is
naturally discrete or combinatorial: feature selection, combinatorial
optimisation, deceptive benchmark problems (`OneMax`, `ConcatenatedTrap`).

For continuous optimisation use the [real-valued GA](index.md) instead; binary
encoding continuous variables with Gray codes adds unnecessary complexity when
Gaussian mutation is available.

## Operators

`BinaryGeneticAlgorithm` composes four fixed operator choices. The config
parameters (below) tune them; the operators themselves are not swappable at the
type level (unlike the real-valued GA, which exposes `GaSelection`,
`GaCrossover`, `GaReplacement` enums).

| Stage | Operator | Config knob |
|---|---|---|
| Selection | k-tournament (host-side) | `tournament_size` |
| Crossover | Uniform crossover | `crossover_p` |
| Mutation | Per-gene bit-flip | `mutation_rate` |
| Replacement | Elitist (truncation) | `elitism_k` |

**Uniform crossover** treats each gene position independently: with probability
`crossover_p` the offspring takes the bit from parent A; otherwise from parent
B. This is more disruptive than single-point crossover and is generally
preferred when the number of relevant schemata is unknown.

**Bit-flip mutation** independently flips each gene with probability
`mutation_rate`. The standard rule of thumb from the binary-GA literature is
`mutation_rate = 1 / D` (one expected flip per genome), which
`BinaryGaConfig::default_for` sets automatically.

## Configuration

```rust,no_run
use rlevo::core::probability::Probability;
use rlevo::evo::algorithms::ga_binary::BinaryGaConfig;

// Explicit construction:
let config = BinaryGaConfig {
    pop_size:       64,
    genome_dim:     20,
    mutation_rate:  Probability::new(1.0 / 20.0),   // 1/D rule
    crossover_p:    Probability::new(0.5),
    tournament_size: 2,
    elitism_k:      1,
};

// Or use the defaults (sets mutation_rate = 1/D, crossover_p = 0.5,
// tournament_size = 2, elitism_k = 1):
let config = BinaryGaConfig::default_for(64, 20);
```

| Field | Type | Typical range | Notes |
|---|---|---|---|
| `pop_size` | `usize` | 32–512 | Larger populations slow per-generation cost but reduce drift |
| `genome_dim` | `usize` | problem-defined | Number of binary genes |
| `mutation_rate` | `Probability` | `1/D`–`5/D` | Higher rates help deceptive problems; too high destroys convergence |
| `crossover_p` | `Probability` | 0.5–0.7 | Probability each gene comes from parent A in uniform crossover |
| `tournament_size` | `usize` | 2–5 | Larger k → stronger selection pressure |
| `elitism_k` | `usize` | 1–3 | Elites copied verbatim; set 0 for fully generational replacement |

## Fitness convention

All strategies in `rlevo::evo` maximise a **canonical** fitness — higher is better. You declare a cost objective's direction with [`ObjectiveSense::Minimize`](https://docs.rs/rlevo-core) and the harness reconciles it at one chokepoint, so you never hand-negate. `OneMax` (maximise the number of `1` bits) is a native maximisation objective: return `count_ones(genome)` directly and declare `ObjectiveSense::Maximize`. The optimum is `fitness = D` (all bits set).

## Minimal example

This example uses `EvolutionaryHarness` — the recommended entry point for
single-strategy runs. See [The Ask/Tell Contract](../appendix-d-suppl/ask-tell-contract.md)
for the manual loop if you need custom logging or early stopping.

```rust,no_run
use burn::backend::Flex;
use rlevo::evo::algorithms::ga_binary::{BinaryGaConfig, BinaryGeneticAlgorithm};
use rlevo::evo::fitness::BatchFitnessFn;
use rlevo::evo::strategy::EvolutionaryHarness;
use burn::tensor::{Int, Tensor, TensorData, backend::Backend};

type B = Flex;

/// OneMax as a native maximisation objective: fitness = count_ones.
/// Declare ObjectiveSense::Maximize — no negation needed.
struct OneMax {
    dim: usize,
}

impl BatchFitnessFn<B, Tensor<B, 2, Int>> for OneMax {
    fn evaluate_batch(
        &mut self,
        population: &Tensor<B, 2, Int>,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Tensor<B, 1> {
        let [pop_size, _] = population.dims();
        let data = population
            .clone()
            .into_data()
            .into_vec::<i32>()
            .unwrap_or_default();
        let fitness: Vec<f32> = (0..pop_size)
            .map(|row| {
                let ones: u32 = (0..self.dim)
                    .filter(|&col| data[row * self.dim + col] != 0)
                    .count() as u32;
                ones as f32
            })
            .collect();
        Tensor::<B, 1>::from_data(TensorData::new(fitness, [pop_size]), device)
    }
}

fn main() {
    let device = Default::default();
    let dim = 20;
    let config = BinaryGaConfig::default_for(64, dim);

    let mut harness = EvolutionaryHarness::<B, _, _>::new(
        BinaryGeneticAlgorithm::<B>::new(),
        config,
        OneMax { dim },
        /* seed */ 42,
        device,
        /* max_generations */ 300,
    ).expect("valid config");

    harness.reset();
    loop {
        let result = harness.step(());
        if result.done {
            break;
        }
    }

    let best = harness.latest_metrics().unwrap().best_fitness_ever;
    println!("best fitness = {best}");   // D (= 20.0) at the OneMax optimum
}
```

## Implementation notes

**First-call behaviour.** On the very first `ask`, `state.fitness` is empty
because the seed population has not been evaluated yet. In this case `ask`
returns the seed population unchanged so the harness can evaluate it and pass
results to `tell`. The first `tell` populates `state.fitness`; all subsequent
`ask` calls run the full selection → crossover → mutation pipeline.

**Reproducibility.** Every random draw — initial population, tournament draws,
crossover coin flips, bit-flip decisions — comes from host-side sub-streams
derived from the root `rng` via `seed_stream`. The Burn backend's GPU PRNG
kernels (`Tensor::random`, `B::seed`) are never used. This means two runs with
the same seed produce identical trajectories regardless of how many threads
evaluated the population between `ask` and `tell`.

**Replacement policy.** Only elitist replacement is implemented. If you need
generational (non-elitist) replacement, build a manual ask/tell loop and omit
the elite re-insertion step.

## When to use

| Situation | Recommendation |
|---|---|
| Search space is inherently binary | Binary GA is the natural fit |
| Deceptive problem (`ConcatenatedTrap`, `OneMax` with building blocks) | Binary GA or BOA; consider larger `pop_size` |
| Continuous optimisation | Real-valued GA or CMA-ES |
| Strong variable dependencies | BOA (learns pairwise and higher-order dependencies) |
| Large genome (D > 500) | EDA variants (UMDA, PBIL) scale better; GA crossover loses schema pressure at high D |

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
