# Real-Valued Genetic Algorithm

<!-- source: crates/rlevo-evolution/src/algorithms/ga.rs -->

The real-valued (or floating-point) genetic algorithm evolves a population of
continuous genome vectors. Each individual is a `Vec` of `f32` values rather
than bits, which avoids the Hamming cliff problem that plagues binary-encoded
representations of continuous variables: small phenotypic changes no longer
require flipping many bits simultaneously.

It is the right first choice for continuous black-box optimisation —
hyperparameter tuning, weight initialisation search, or any problem where the
search space is a bounded region of \\(\mathbb{R}^D\\).

## Operators

Unlike the [Binary GA](binary-encoded-genetic-algorithm.md), the operator
choices are enum-selectable at runtime via `GaConfig`. This lets you mix and
match without changing the algorithm type.

| Stage | Variants | Config field |
|---|---|---|
| Selection | `Tournament { size }` | `selection: GaSelection` |
| Crossover | `BlxAlpha { alpha }`, `Uniform { p }` | `crossover: GaCrossover` |
| Mutation | Isotropic Gaussian (always) | `mutation_sigma` |
| Replacement | `Generational`, `Elitist { elitism_k }` | `replacement: GaReplacement` |
| Post-mutation | Bounds clamp | `bounds` |

**BLX-α crossover** (Eshelman & Schaffer, 1993) is the default. Given two
parents \\(a\\) and \\(b\\) with \\(d = |a_i - b_i|\\), each offspring gene is
sampled uniformly from \\([\min(a_i, b_i) - \alpha \cdot d,\ \max(a_i, b_i) +
\alpha \cdot d]\\). The \\(\alpha\\) parameter controls how far outside the
parental range offspring can land: `alpha = 0.0` restricts offspring to the
parental interval; `alpha = 0.5` (the default) allows 50% extrapolation on
each side and is the value that minimises selection pressure on schema length
(Deb & Agrawal, 1995).

**Uniform crossover** (`GaCrossover::Uniform { p }`) swaps each gene
independently with probability `p`, identical in structure to binary uniform
crossover but on real values. Prefer BLX-α for most continuous problems; use
`Uniform` if you specifically want maximum schema disruption.

**Gaussian mutation** adds \\(\mathcal{N}(0, \sigma^2)\\) to each gene
independently. There is no per-gene on/off probability — every gene is
perturbed every generation. After mutation, offspring are clamped to `bounds`.

## Configuration

```rust,no_run
use rlevo::evo::algorithms::ga::{
    GaConfig, GaCrossover, GaReplacement, GaSelection,
};

// Explicit construction:
let config = GaConfig {
    pop_size:       64,
    genome_dim:     10,
    bounds:         (-5.12, 5.12),
    mutation_sigma: 0.3,
    selection:      GaSelection::Tournament { size: 2 },
    crossover:      GaCrossover::BlxAlpha { alpha: 0.5 },
    replacement:    GaReplacement::Elitist { elitism_k: 1 },
};

// Or use the defaults (same values as above):
let config = GaConfig::default_for(64, 10);
```

| Field | Type | Typical range | Notes |
|---|---|---|---|
| `pop_size` | `usize` | 32–512 | Larger slows each generation; helps on multi-modal landscapes |
| `genome_dim` | `usize` | problem-defined | Dimensionality of the search space |
| `bounds` | `Bounds` | problem-defined | Initial population sampled uniformly here; offspring clamped here after mutation |
| `mutation_sigma` | `f32` | 0.05–1.0 | Scale relative to `bounds` width; σ ≈ 0.3 / D for high-D problems |
| `GaSelection::Tournament { size }` | `usize` | 2–5 | k = 2 is low pressure; k = pop_size is truncation selection |
| `GaCrossover::BlxAlpha { alpha }` | `f32` | 0.0–1.0 | 0.5 is the standard choice; increase for more exploration |
| `GaCrossover::Uniform { p }` | `f32` | 0.3–0.7 | Per-gene swap probability |
| `GaReplacement::Elitist { elitism_k }` | `usize` | 1–5 | Elites copied verbatim; higher k slows diversity loss |
| `GaReplacement::Generational` | — | — | Full population replacement; faster convergence, less stable |

## Fitness convention

All strategies in `rlevo::evo` maximise a **canonical** fitness — higher is better. You declare a cost objective's direction with [`ObjectiveSense::Minimize`](https://docs.rs/rlevo-core) and the harness reconciles it at one chokepoint, so you never hand-negate. A maximisation objective plugs in directly; a cost objective such as Sphere declares `ObjectiveSense::Minimize`.

## Minimal example

This example uses `EvolutionaryHarness` — the recommended entry point for
single-strategy runs. See [The Ask/Tell Contract](../appendix-d-suppl/ask-tell-contract.md)
for the manual loop if you need custom logging or early stopping.

```rust,no_run
use burn::backend::Flex;
use burn::tensor::{Tensor, TensorData, backend::Backend};
use rlevo::evo::algorithms::ga::{
    GaConfig, GaCrossover, GaReplacement, GaSelection, GeneticAlgorithm,
};
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
        // Sum of squares per row — no GPU kernel needed for this demo.
        let [pop_size, _] = population.dims();
        let data = population
            .clone()
            .into_data()
            .into_vec::<f32>()
            .unwrap_or_default();
        let dim = data.len() / pop_size;
        let fitness: Vec<f32> = (0..pop_size)
            .map(|row| {
                (0..dim)
                    .map(|col| data[row * dim + col].powi(2))
                    .sum::<f32>()
            })
            .collect();
        Tensor::<B, 1>::from_data(TensorData::new(fitness, [pop_size]), device)
    }
}

fn main() {
    let device = Default::default();
    let dim = 10;
    let config = GaConfig {
        pop_size:       64,
        genome_dim:     dim,
        bounds:         (-5.12, 5.12),
        mutation_sigma: 0.3,
        selection:      GaSelection::Tournament { size: 2 },
        crossover:      GaCrossover::BlxAlpha { alpha: 0.5 },
        replacement:    GaReplacement::Elitist { elitism_k: 1 },
    };

    let mut harness = EvolutionaryHarness::<B, _, _>::new(
        GeneticAlgorithm::<B>::new(),
        config,
        SphereCost,
        /* seed */ 42,
        device,
        /* max_generations */ 500,
    ).expect("valid config");

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

**First-call behaviour.** On the very first `ask`, `state.fitness` is empty
because the seed population has not been evaluated yet. `ask` returns the seed
population unchanged so the harness evaluates it and passes results to `tell`.
The first `tell` populates `state.fitness`; all subsequent `ask` calls run the
full selection → crossover → mutation pipeline. This is the same pattern as the
[Binary GA](binary-encoded-genetic-algorithm.md).

**Bounds clamping.** After Gaussian mutation, every offspring gene is clamped to
`params.bounds`. This prevents genetic drift into regions the initial population
never covered, but it also creates a probability mass at the boundary. If the
optimum is near a boundary, consider widening `bounds` slightly so clamping does
not artificially pile up individuals there.

**Replacement tradeoff.** `Generational` replacement converges faster but is
more susceptible to diversity loss on multi-modal landscapes. `Elitist` with
`elitism_k = 1` guarantees the best individual is never lost, at the cost of
slightly slower exploration. For most problems, `elitism_k = 1` or `2` is a
safe default.

**Reproducibility.** All random draws — initial population, tournament indices,
crossover, mutation — come from host-side sub-streams derived from the root
`rng` via `seed_stream`. The Burn backend's GPU PRNG kernels are never called.
Two runs with the same seed produce identical trajectories.

## When to use

| Situation | Recommendation |
|---|---|
| Continuous bounded search space | Real-valued GA is a good starting point |
| Dimensionality D ≤ ~30 | GA handles well; scales adequately to ~100 |
| Multi-modal landscape | Increase `pop_size`; try `tournament_size = 2` for lower pressure |
| High dimensionality (D > 100) | CMA-ES adapts step sizes per dimension; GA's isotropic σ struggles |
| Binary or combinatorial search | [Binary GA](binary-encoded-genetic-algorithm.md) |
| Strong variable dependencies | CMA-ES captures covariance; GA crossover assumes separability |
| Need fast convergence, unimodal | Evolution Strategy (ES) with self-adaptive σ is tighter |

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
