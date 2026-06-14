# Evolutionary Programming

<!-- source: crates/rlevo-evolution/src/algorithms/ep.rs -->

Evolutionary Programming (EP) is a continuous optimiser in the ES family tree,
introduced by Fogel (1994). Like ES it uses no crossover — each parent produces
exactly one offspring by Gaussian mutation — and each individual carries its own
self-adaptive step size σ. The distinguishing feature of EP is its **survivor
selection**: rather than truncation (keep the μ best), EP uses a
**q-tournament** over the combined `(μ + μ)` pool, giving weaker individuals a
stochastic chance to survive and maintaining more diversity at the cost of
slower convergence.

Compared to the [classical ES variants](evolution-strategies.md), EP occupies a
middle ground: it shares log-normal σ adaptation with `(μ+λ)` ES but replaces
deterministic truncation with probabilistic tournament selection.

## Operators

| Stage | Mechanism | Config knob |
|---|---|---|
| Reproduction | Each parent → one offspring (always μ + μ) | `mu` |
| σ adaptation | Log-normal: `σ' = σ · exp(τ · N(0,1))` per individual | `tau` |
| Mutation | Per-row Gaussian with individual σ | `initial_sigma` |
| Survivor selection | q-tournament over the `2μ` combined pool | `tournament_q` |
| Post-mutation | Bounds clamp | `bounds` |

**q-tournament selection.** Each of the `2μ` individuals (parents + offspring)
plays `tournament_q` randomly drawn opponents. An individual wins a bout if its
fitness is strictly lower (cost convention). The μ individuals with the most
wins survive; ties are broken by fitness. Higher `tournament_q` increases
selection pressure toward better individuals; lower values preserve more
diversity. The default `tournament_q = 10` is the value from Fogel (1994).

**σ adaptation timing.** In EP, σ is adapted *before* mutation (the updated σ'
is what drives the offspring perturbation). In the multi-parent ES variants σ
is adapted at the same time but the per-individual σ update is applied to
produce offspring σ values that survive only if their carrier survives. The
practical difference is small, but it means EP's σ values reflect the *current
generation's* spread rather than the previous generation's.

## Configuration

```rust,no_run
use rlevo::evo::algorithms::ep::EpConfig;

// Explicit construction:
let config = EpConfig {
    mu:            30,
    genome_dim:    10,
    bounds:        (-5.12, 5.12),
    initial_sigma: 1.0,
    tau:           1.0 / (2.0 * (10.0_f32).sqrt()).sqrt(),
    tournament_q:  10,
};

// Or use the defaults (initial_sigma = 1.0, tournament_q = 10,
// tau = 1 / sqrt(2·sqrt(D))):
let config = EpConfig::default_for(30, 10);
```

| Field | Type | Typical range | Notes |
|---|---|---|---|
| `mu` | `usize` | 10–100 | Parent population size; offspring count equals `mu` (always `μ + μ`) |
| `genome_dim` | `usize` | problem-defined | Dimensionality of the search space |
| `bounds` | `(f32, f32)` | problem-defined | Initial population sampled here; offspring clamped after mutation |
| `initial_sigma` | `f32` | 0.1–3.0 | Initial step size; adapts per-individual during the run |
| `tau` | `f32` | `1/sqrt(2·sqrt(D))` | Log-normal learning rate; `default_for` computes the standard value |
| `tournament_q` | `usize` | 5–20 | Opponents per bout; higher → stronger selection pressure |

## Fitness convention

All strategies in `rlevo::evo` treat fitness as **cost** — lower is better.
The q-tournament defines "winning a bout" as having strictly lower cost than
the opponent.

## Minimal example

```rust,no_run
use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData, backend::Backend};
use rlevo::evo::algorithms::ep::{EpConfig, EvolutionaryProgramming};
use rlevo::evo::fitness::BatchFitnessFn;
use rlevo::evo::strategy::EvolutionaryHarness;

type B = NdArray;

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
    let config = EpConfig::default_for(30, 10);

    let mut harness = EvolutionaryHarness::<B, _, _>::new(
        EvolutionaryProgramming::<B>::new(),
        config,
        SphereCost,
        /* seed */ 42,
        device,
        /* max_generations */ 2000,
    );

    harness.reset();
    loop {
        let result = harness.step(());
        if result.done {
            break;
        }
    }

    let best = harness.latest_metrics().unwrap().best_fitness_ever;
    println!("best cost = {best:.4e}");
}
```

EP converges more slowly than ES on unimodal problems — the D=10 Sphere test
requires 2000 generations to reach `< 1e-4`, whereas `(μ+λ)` ES needs ~1500
to reach `< 1e-6`. The q-tournament's diversity maintenance is the tradeoff:
it costs convergence speed in exchange for resilience on multi-modal landscapes.

## Implementation notes

**σ scratchpad between `ask` and `tell`.** EP shares the same scratchpad
pattern as the [multi-parent ES variants](evolution-strategies.md#implementation-notes):
after `ask` returns, `state.sigmas` has shape `(2μ,)` — the first μ entries
are the parent σ values, the last μ are the offspring σ values just derived
by log-normal adaptation. `tell` uses this combined vector to select survivor
σ values at the same tournament indices as their genomes. After `tell`
completes, `state.sigmas` is back to shape `(μ,)`. Do not read `state.sigmas`
between `ask` and `tell` expecting shape `(μ,)`.

**First-call behaviour.** On the first `ask`, `state.parent_fitness` is empty
and the initial parents are returned unchanged for evaluation. The first `tell`
populates the fitness cache and resets σ to `initial_sigma` before any
tournament runs. This matches the first-call pattern in the other strategies.

**Reproducibility.** σ noise (`SeedPurpose::Other`), mutation draws
(`SeedPurpose::Mutation`), and tournament opponent draws
(`SeedPurpose::Selection`) each come from independent `seed_stream`
sub-streams. Two runs with the same seed produce identical trajectories.

## When to use

| Situation | Recommendation |
|---|---|
| Continuous optimisation, moderate D | ES `(μ+λ)` is faster to converge; use EP when diversity matters more |
| Multi-modal landscape with many shallow basins | EP's q-tournament preserves more diversity than truncation selection |
| Noisy fitness evaluations | Stochastic tournament is more robust to noise than deterministic truncation |
| Tight convergence budget | Prefer `(μ+λ)` ES or DE/rand/1/bin; EP needs more generations for the same precision |
| Binary or combinatorial space | [Binary GA](binary-encoded-genetic-algorithm.md) |
