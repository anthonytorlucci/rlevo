# Evolution Strategies

<!-- source: crates/rlevo-evolution/src/algorithms/es_classical.rs -->

Evolution Strategies (ES) are a family of continuous optimisation algorithms
that differ from genetic algorithms in two key ways: there is **no crossover**
(offspring come from a single parent by mutation only), and **step sizes are
part of the genome** — the strategy adapts its own mutation strength during
the run rather than keeping σ fixed.

`rlevo` implements four classical variants under a single `EvolutionStrategy`
type, selected by the `EsKind` enum in `EsConfig`. All four operate on
real-valued genomes.

## Variants

| Variant | `EsKind` | σ adaptation | Selection pressure |
|---|---|---|---|
| `(1+1)` | `OnePlusOne` | Rechenberg 1/5th rule | Greedy (parent vs. offspring) |
| `(1+λ)` | `OnePlusLambda { lambda }` | None (σ carried over) | Best offspring vs. parent |
| `(μ,λ)` | `MuCommaLambda { mu, lambda }` | Log-normal per-individual | μ highest-fitness offspring; parents discarded |
| `(μ+λ)` | `MuPlusLambda { mu, lambda }` | Log-normal per-individual | μ highest-fitness of combined pool |

**`(1+1)` — Rechenberg's 1/5th rule.** One parent produces one offspring per
generation. If the offspring is better, it replaces the parent. Every
\\(10 \cdot D\\) generations, the success rate over that window is measured: if
more than 1/5 of trials succeeded, σ is multiplied by 1.22 (larger steps);
if fewer than 1/5 succeeded, σ is divided by 1.22 (smaller steps). This keeps
the algorithm near the empirically optimal 20% success rate.

**`(1+λ)` — one parent, many offspring.** Produces λ offspring from a single
parent, keeps the best offspring only if it improves on the parent. σ is not
adapted — this variant is useful when the landscape is cheap to evaluate and
you want a simple baseline. It is also used internally by Cartesian GP.

**`(μ,λ)` — comma selection.** μ parents each produce λ/μ offspring; the
μ highest-fitness *offspring* become the new parents. The parent pool is **discarded**
each generation. This forces re-evaluation of parental quality and prevents
stagnation on flat plateaux, but the best individual seen so far can be lost
between generations. Require λ > μ (typically λ ≥ 7μ).

**`(μ+λ)` — plus selection.** Same as `(μ,λ)` but survivors are drawn from
the **combined** parent + offspring pool. Elitism is implicit — the best
individual is never lost. Slower to escape local optima than comma selection
but safer on expensive landscapes where losing good solutions matters.

<a name="log-normal-sigma-adaptation"></a>
**Log-normal σ adaptation.** In both multi-parent variants, each individual
carries its own step size σ as part of its genome — there is no external rule
adjusting it (contrast the `(1+1)` 1/5th rule above). σ is *self-adapted*:
selection scores only the mutated genes, but a well-scaled σ tends to produce
fitter offspring, so good step sizes survive indirectly. The update is a
two-step mutation — σ is perturbed first, then the *new* σ drives the gene
mutation:

```math
\sigma' = \sigma \cdot \exp(\tau \cdot \mathcal{N}(0, 1)),
\qquad
\mathbf{x}' = \mathbf{x} + \sigma' \cdot \mathcal{N}(0, \mathbf{I})
```

where \\(\tau = 1 / \sqrt{2\sqrt{D}}\\) is the standard learning rate
(Beyer & Schwefel, 2002). Larger populations learn a better σ faster because
more diverse trials give a richer gradient signal.

Three properties motivate this exact form:

- **Order.** Mutating σ before the genes ties each step size to the step it
  produced, so selection judges the σ that actually generated the surviving
  offspring. Perturbing σ afterwards would select a σ that was never tested.
- **Positivity.** The multiplicative \\(\exp(\cdot)\\) factor keeps σ strictly
  positive; an additive Gaussian update could drive it negative.
- **Unbiased in log-space.** A factor-of-\\(c\\) increase and a
  factor-of-\\(1/c\\) decrease are equally likely, and \\(\mathbb{E}[\ln
  \sigma'] = \ln \sigma\\). Absent selection σ neither grows nor shrinks
  systematically, so selection alone shapes it.

`rlevo` adapts a single σ per individual. The classical Schwefel scheme
generalises this to a per-coordinate \\(\sigma_i\\) with a global and a
coordinate-local term, \\(\sigma_i' = \sigma_i \cdot \exp(\tau' \mathcal{N}(0,1)
+ \tau \, \mathcal{N}_i(0,1))\\), trading more learnable parameters for the
ability to rescale each axis independently.

> **Shared mechanism, different constant.** This same log-normal
> σ-self-adaptation drives [CMSA-ES](cma-es.md), which reuses the rule with its
> own learning rate \\(\tau = 1/\sqrt{2D}\\) rather than the classical
> \\(1/\sqrt{2\sqrt{D}}\\) above. Full [CMA-ES](cma-es.md) instead replaces
> self-adaptation with cumulative step-size adaptation (CSA) and adds a full
> covariance matrix — see the [CMA-ES and CMSA-ES](cma-es.md) appendix for both.

## Configuration

```rust,no_run
use rlevo::evo::algorithms::es_classical::{EsConfig, EsKind};

// Explicit construction:
let config = EsConfig {
    kind:          EsKind::MuPlusLambda { mu: 5, lambda: 20 },
    genome_dim:    10,
    bounds:        (-5.12, 5.12),
    initial_sigma: 1.0,
    tau:           1.0 / (2.0 * (10.0_f32).sqrt()).sqrt(),
};

// Or use the defaults (bounds = (-5.12, 5.12), initial_sigma = 1.0,
// tau = 1 / sqrt(2·sqrt(D))):
let config = EsConfig::default_for(EsKind::MuPlusLambda { mu: 5, lambda: 20 }, 10);
```

| Field | Type | Typical range | Notes |
|---|---|---|---|
| `kind` | `EsKind` | — | Selects the variant; see table above |
| `genome_dim` | `usize` | problem-defined | Dimensionality of the search space |
| `bounds` | `(f32, f32)` | problem-defined | Initial population sampled here; offspring clamped after mutation |
| `initial_sigma` | `f32` | 0.1–3.0 | Relative to the `bounds` width; 1.0 is a reasonable start |
| `tau` | `f32` | `1/sqrt(2·sqrt(D))` | Learning rate for log-normal σ update; `default_for` computes the standard value |

For `(μ,λ)` and `(μ+λ)`, set λ ≥ 7μ as a starting point. A common choice
for medium-D problems is `mu = 5, lambda = 20` or `mu = 10, lambda = 50`.

## Fitness convention

All strategies in `rlevo::evo` maximise a **canonical** fitness — higher is better. You declare a cost objective's direction with [`ObjectiveSense::Minimize`](https://docs.rs/rlevo-core) and the harness reconciles it at one chokepoint, so you never hand-negate. The Sphere function is a cost surface: declare `ObjectiveSense::Minimize` and the harness maximises \\(-\sum x_i^2\\) internally; `best_fitness` still reads as the natural cost (→ 0).

## Minimal example

```rust,no_run
use burn::backend::Flex;
use burn::tensor::{Tensor, TensorData, backend::Backend};
use rlevo::evo::algorithms::es_classical::{EsConfig, EsKind, EvolutionStrategy};
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
    let config = EsConfig::default_for(
        EsKind::MuPlusLambda { mu: 5, lambda: 20 },
        10,
    );

    let mut harness = EvolutionaryHarness::<B, _, _>::new(
        EvolutionStrategy::<B>::new(),
        config,
        SphereCost,
        /* seed */ 42,
        device,
        /* max_generations */ 1500,
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

**σ scratchpad between `ask` and `tell`.** In the multi-parent variants,
offspring carry their own adapted σ values. `ask` appends the offspring σ
vector to `state.sigmas` before returning, so after `ask` returns,
`state.sigmas` has shape `(μ + λ,)`: the first μ entries are the unchanged
parent σ values, the last λ are the per-offspring σ values. `tell` slices
them apart by index:

- `(μ,λ)`: survivor σ comes from the offspring slice (parent σ discarded).
- `(μ+λ)`: survivor σ comes from the combined slice at the same truncation
  indices used to select genomes.

This avoids adding a separate pending field to `EsState` at the cost of a
briefly non-obvious tensor shape. Do not read `state.sigmas` between `ask`
and `tell` expecting shape `(μ,)` — it is `(μ + λ,)` in that window.

**First-call behaviour.** On the very first `ask` (when `state.parent_fitness`
is empty), `ask` returns the initial parent population unchanged so the harness
evaluates it and passes results to `tell`. The first `tell` bootstraps
`state.parent_fitness` without running selection. This matches the pattern in
the [Binary GA](binary-encoded-genetic-algorithm.md) and
[Real-Valued GA](real-valued-genetic-algorithm.md).

**1/5th rule window.** For `(1+1)`, the success-rate window resets every
\\(\max(1, 10 \cdot D)\\) generations. On low-D problems this fires frequently;
on high-D problems (D = 100) it fires every 1,000 generations. The σ change
factor is fixed at 1.22 (Rechenberg's original recommendation).

**`(μ,λ)` convergence guarantee.** Because parents are discarded, `(μ,λ)` has
no monotone convergence guarantee — it is possible (though rare with λ ≥ 7μ)
for the highest fitness in the current generation to temporarily fall below the
previous generation's best. The harness's `best_fitness_ever` tracker is
unaffected, but `latest_metrics().best_fitness` may dip. Use `(μ+λ)` if you
need elitist monotone improvement.

**Reproducibility.** All random draws use `seed_stream` (host-RNG convention).
Two runs with the same seed and the same `EsKind` produce identical
trajectories. Changing the variant changes the draw sequence, so seeds are
not portable across variants.

## When to use

| Situation | Recommendation |
|---|---|
| Continuous optimisation, low to medium D (≤ 30) | `(μ+λ)` with `mu = 5, lambda = 20` is a strong default |
| Very cheap function evaluations | `(1+1)` — minimal overhead per generation |
| Noisy fitness evaluations | `(μ,λ)` — discarding parents avoids locking to a noisy incumbent |
| Ill-conditioned / rotated landscape | [CMA-ES](cma-es.md) learns the covariance, so it searches along the landscape's natural axes |
| Want the strongest general-purpose default | [CMA-ES](cma-es.md) for D ≤ 30; [CMSA-ES](cma-es.md) for a simpler, path-free variant |
| High dimensionality (D ≫ 30) | Classical ES degrades; [CMA-ES](cma-es.md)'s covariance also grows costly (\\(O(D^3)\\)) — a separable variant is the right tool (deferred) |
| Multi-modal landscape | Increase λ; consider `(μ,λ)` for more exploration |
| Need crossover / schema recombination | [Real-Valued GA](real-valued-genetic-algorithm.md) |
| Discrete / binary search space | [Binary GA](binary-encoded-genetic-algorithm.md) |

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
