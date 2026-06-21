# Whale Optimization Algorithm

<!-- source: crates/rlevo-evolution/src/algorithms/metaheuristic/woa.rs -->

The Whale Optimization Algorithm (WOA; Mirjalili & Lewis, 2016) is a swarm
metaheuristic cast as humpback "bubble-net" hunting. Each generation every whale
picks one of three moves by a coin flip \\(p\\) and a coefficient \\(\lvert
A\rvert\\): shrink-encircle the current best, drift towards a random whale, or
spiral in towards the best along a logarithmic curve. As in
[GWO](grey-wolf-optimizer.md), a single coefficient \\(a\\) anneals from \\(2\\) to
\\(0\\) to trade exploration for exploitation over the budget.

`rlevo` ships WOA as a **legacy comparator**. The spiral and encircle operators
compose, in expectation, to a weighted PSO-style pull towards the incumbent best
(Camacho-Villalón et al., 2023); and Deng & Liu (2023) show WOA's reported edge on
standard benchmarks is partly an artefact of an **origin bias** — it favours
solutions near the centre of the search box, which flatters benchmarks whose
optima sit there. It is provided for API coverage and as a comparator, not as a
recommendation: for general continuous work, [PSO](particle-swarm-optimization.md)
is the default in `rlevo`, and [CMA-ES](cma-es.md) the choice when precision
matters.

`rlevo` implements it as `WhaleOptimization` under
`rlevo::evo::algorithms::metaheuristic::woa`, operating on real-valued positions
of shape `(pop_size, D)`.

## The three moves

`ask` draws, per whale, scalars \\(p, l \sim \mathcal{U}\\) and coefficients

```math
A = 2 a\, r_a - a, \qquad C = 2\, r_c, \qquad r_a, r_c \sim \mathcal{U}[0,1),
```

then selects a move. With \\(X^*\\) the current best (the "food source") and
\\(X_{\text{rand}}\\) a different randomly chosen whale:

```math
X' =
\begin{cases}
X^* - A\,\lvert C X^* - X\rvert & p < 0.5,\ \lvert A\rvert < 1 \quad\text{(encircle best)}\\[4pt]
X_{\text{rand}} - A\,\lvert C X_{\text{rand}} - X\rvert & p < 0.5,\ \lvert A\rvert \ge 1 \quad\text{(search)}\\[4pt]
\lvert X^* - X\rvert\, e^{b l}\cos(2\pi l) + X^* & p \ge 0.5 \quad\text{(spiral)}
\end{cases}
```

with \\(l \sim \mathcal{U}[-1,1]\\) and the result clamped to the search box.
The \\(\lvert A\rvert\\) switch is what couples exploration to the \\(a\\)
schedule: early on \\(a\\) is large so \\(\lvert A\rvert \ge 1\\) is common and
whales chase *random* peers (exploration); as \\(a \to 0\\), \\(\lvert A\rvert <
1\\) dominates and whales converge on \\(X^*\\) (exploitation). The spiral arm,
chosen half the time, walks in towards \\(X^*\\) along a logarithmic spiral whose
shape is set by \\(b\\).

> **\\(A\\) and \\(C\\) are per-whale *scalars* here, unlike
> [GWO](grey-wolf-optimizer.md).** Each whale draws one \\(A\\) and one \\(C\\)
> per generation, broadcast across all \\(D\\) coordinates. The 2016 paper states
> them as vectors; `rlevo` uses the common per-individual-scalar reading, which
> keeps the three candidate tensors cheap and the masking branch-free.

## The exploration coefficient

As in GWO, \\(a\\) is scheduled from the generation counter:

```math
a = 2\left(1 - \min\!\left(\tfrac{t}{T},\ 1\right)\right), \qquad T = \texttt{max\_generations}.
```

`max_generations` paces this anneal; it is **not** a stopping criterion (the
harness owns the stop condition). Set it to the budget over which you want the
exploration-to-exploitation transition to complete.

## Configuration

```rust,no_run
use rlevo::evo::algorithms::metaheuristic::woa::WoaConfig;

// Explicit construction:
let config = WoaConfig {
    pop_size:        32,
    genome_dim:      10,
    bounds:          (-5.12, 5.12),
    max_generations: 500,   // paces a = 2(1 − t/T); not a stop condition
    b:               1.0,   // logarithmic-spiral shape constant
};

// Or use the defaults (bounds = (-5.12, 5.12), max_generations = 500, b = 1.0):
let config = WoaConfig::default_for(32, 10);
```

| Field | Type | Typical range | Notes |
|---|---|---|---|
| `pop_size` | `usize` | 20 – 50 | Number of whales |
| `genome_dim` | `usize` | problem-defined | Dimensionality of each whale |
| `bounds` | `(f32, f32)` | problem-defined | Initial positions sampled here; steps clamped here. Mind the origin bias — see implementation notes |
| `max_generations` | `usize` | match your budget | Annealing horizon for \\(a\\); paces exploration→exploitation only |
| `b` (\\(b\\)) | `f32` | ~1.0 | Spiral shape; Mirjalili's canonical \\(1.0\\) |

## Fitness convention

All strategies in `rlevo::evo` treat fitness as **cost** — lower is better.
Maximisation problems must be negated. The food source \\(X^*\\) is the
lowest-cost whale seen so far, and the best-so-far tracker is an argmin.

## Minimal example

```rust,no_run
use burn::backend::Flex;
use burn::tensor::{Tensor, TensorData, backend::Backend};
use rlevo::evo::algorithms::metaheuristic::woa::{WhaleOptimization, WoaConfig};
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
    // The Sphere optimum sits at the origin — exactly the case WOA's
    // origin bias flatters. Treat the strong result here with that caveat.
    let config = WoaConfig::default_for(32, 10);

    let mut harness = EvolutionaryHarness::<B, _, _>::new(
        WhaleOptimization::<B>::new(),
        config,
        SphereCost,
        /* seed */ 5,
        device,
        /* max_generations */ 600,
    );

    harness.reset();
    while !harness.step(()).done {}

    let best = harness.latest_metrics().unwrap().best_fitness_ever;
    println!("best cost = {best:.4e}");
}
```

## Implementation notes

**Branch-free composition.** The three candidate tensors (encircle-best,
search-random, spiral) are computed in full, then selected with two boolean
masks: `spiral.mask_where(p < 0.5, encircle)` where `encircle =
enc_rand.mask_where(|A| < 1, enc_best)`. No whale takes a divergent kernel path,
so the per-whale random choice costs three tensor evaluations and two masks
rather than a host-side loop.

**Host-sampled per-whale scalars.** \\(r_a, r_c, p, l\\) and the random-peer index
are host-sampled from a single `seed_stream` (`SeedPurpose::Other`, keyed by the
generation counter) and uploaded; the strategy never calls `Tensor::random` or
`B::seed` (the host-RNG convention). The search-branch peer is forced to differ
from the whale itself.

**Origin bias.** Deng & Liu (2023) show WOA implicitly favours the centre of the
search box, so results on origin-centred benchmarks (like the Sphere example
above) overstate its general performance. On problems whose optima are far from
the box centre, expect weaker behaviour; do not read a strong Sphere result as
evidence WOA will transfer.

**Two-cycle bootstrap.** WOA needs a fitness cache and a food source before it can
move. The first `ask` returns the initial positions unchanged (it detects an empty
fitness cache) and the first `tell` seeds the cache and sets `best_genome`. The
three-move update goes live from the **second `ask`**; calling `ask` twice without
an intervening `tell` panics, since the food source would be unset.
[`EvolutionaryHarness`](../appendix-d-suppl/ask-tell-contract.md) handles the
bootstrap transparently.

**Reproducibility.** Initial positions (`SeedPurpose::Init`) and the
per-generation per-whale scalars (`SeedPurpose::Other`, keyed by the generation
counter) come from independent host sub-streams. Two runs with the same seed and
config produce identical trajectories on a fixed backend.

## When to use

| Situation | Recommendation |
|---|---|
| General continuous optimisation | Prefer [PSO](particle-swarm-optimization.md) — WOA composes to a PSO-style pull with an origin-bias caveat |
| Optimum known to sit far from the box centre | Avoid WOA — the origin bias works against you; use [CMA-ES](cma-es.md) or [DE](differential-evolution.md) |
| Need high-precision convergence | [CMA-ES](cma-es.md) — WOA converges strongly on centred problems but is not a precision optimiser in general |
| Strong variable dependencies | [CMA-ES](cma-es.md) — WOA's scalar \\(A\\)/\\(C\\) carry no covariance model |
| You specifically need a WOA baseline | This implementation — faithful to the 2016 spiral/encircle composition |
| Binary / combinatorial search | [Binary GA](binary-encoded-genetic-algorithm.md) — WOA is defined over continuous positions |

The Whale Optimization Algorithm is one of the metaphor-based methods examined by
Camacho-Villalón et al. (2023), with a complementary empirical critique by Deng &
Liu (2023); prefer it as a comparator, not a workhorse.

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
