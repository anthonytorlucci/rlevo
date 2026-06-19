# Salp Swarm Algorithm

<!-- source: crates/rlevo-evolution/src/algorithms/metaheuristic/salp.rs -->

The Salp Swarm Algorithm (SSA; Mirjalili et al., 2017) is a swarm metaheuristic
modelled on the chains that salps form when swimming. The population is split into
a *leader* block and a *follower* chain: leaders take a signed, decaying step
towards the best-known position (the "food source"), and each follower simply
averages its position with the salp directly ahead of it, propagating the leaders'
motion down the chain.

`rlevo` ships SSA as a **legacy comparator**, but with a *different* caveat from
the other four. The leader update is a biased random walk towards the best and the
follower rule is a stencil average — neither a novel search dynamic over PSO or DE.
More pointedly, Castelli et al. (2022) show the original leader update is
**shift-variant**: it references the absolute search bounds, so the algorithm's
behaviour changes if you translate the search domain away from the origin. This is
a mathematical-correctness critique, not the PSO-equivalence argument levelled at
[GWO](grey-wolf-optimizer.md)/[WOA](whale-optimization-algorithm.md)/[Bat](bat-algorithm.md).
It is provided for API coverage; for general continuous work,
[PSO](particle-swarm-optimization.md) is the default in `rlevo`, and
[CMA-ES](cma-es.md) the choice when precision matters.

`rlevo` implements it as `SalpSwarm` under
`rlevo::evo::algorithms::metaheuristic::salp`, operating on real-valued swarms of
shape `(pop_size, D)`. The swarm must hold at least two salps.

## The leader update

`ask` treats the first half of the swarm as leaders. Each leader moves relative to
the food source \\(F\\) by a signed, bound-scaled step whose magnitude is set by a
time-decayed coefficient \\(c_1\\). Per leader, per coordinate, two uniform draws
\\(c_2, c_3 \sim \mathcal{U}[0,1]\\) give:

```math
X_i = F \pm c_1\bigl((\text{ub} - \text{lb})\,c_2 + \text{lb}\bigr),
\qquad
\pm = \begin{cases} + & c_3 \ge 0.5\\ - & c_3 < 0.5 \end{cases}
```

clamped to the search box. The bracketed term is a uniform sample *over the
search range itself*, scaled by \\(c_1\\) and given a random sign — so a leader's
step is a decaying, symmetric jitter around the food source.

> **The \\(+\,\text{lb}\\) term is the shift-variance Castelli et al. (2022)
> flag.** The leader step depends on the *absolute* bounds, not on a displacement
> relative to the current position, so translating the domain changes the
> dynamics. `rlevo`'s default \\([-5.12, 5.12]\\) is the benign, origin-symmetric
> case; on asymmetric or shifted bounds the step is biased. If exact behaviour on
> a shifted domain matters, prefer a shift-invariant strategy such as
> [CMA-ES](cma-es.md) or [DE](differential-evolution.md).

The coefficient \\(c_1\\) is scheduled from the generation counter:

```math
c_1 = 2\,\exp\!\left(-\left(\frac{4t}{T}\right)^{2}\right), \qquad T = \texttt{max\_generations},
```

with \\(4t/T\\) clamped at \\(4\\). It starts near \\(2\\) and decays rapidly — the
**squared** exponent is essential; the unsquared form decays far more slowly and
is a common mis-citation. `max_generations` paces this anneal and is not a
stopping criterion.

## The follower chain

The second half are followers. Each follower averages its current position with
the salp directly ahead:

```math
X_i \leftarrow \frac{X_i + X_{i-1}}{2}.
```

This is realised as a **parallel stencil**, not a sequential cascade: the "ahead"
positions are gathered in one shot from the pre-update follower block, with the
first follower reading the *updated* last leader at the boundary. So one `ask`
advances the chain by a single averaging step; the leaders' new motion reaches
deeper followers over successive generations rather than instantly.

## Configuration

```rust,no_run
use rlevo::evo::algorithms::metaheuristic::salp::SalpConfig;

// Explicit construction:
let config = SalpConfig {
    pop_size:        40,
    genome_dim:      10,
    bounds:          (-5.12, 5.12),
    max_generations: 500,   // paces c1 = 2·exp(−(4t/T)²); not a stop condition
};

// Or use the defaults (bounds = (-5.12, 5.12), max_generations = 500):
let config = SalpConfig::default_for(40, 10);
```

| Field | Type | Typical range | Notes |
|---|---|---|---|
| `pop_size` | `usize` | ≥ 2 | Swarm size; split into `pop_size/2` leaders and the rest followers. **Panics below 2** |
| `genome_dim` | `usize` | problem-defined | Dimensionality of each salp |
| `bounds` | `(f32, f32)` | problem-defined | Initial positions sampled here; leader step scales with these bounds (see the shift-variance note) |
| `max_generations` | `usize` | match your budget | Annealing horizon for \\(c_1\\); paces the leader step decay only |

> **The leader/follower split is `pop_size/2`, not the paper's single leader.**
> Mirjalili et al. (2017) use one leader and \\(N-1\\) followers; `rlevo` splits
> the swarm in half. This gives more leaders driving towards the food source and
> a shorter follower chain — a deliberate, documented deviation.

## Fitness convention

All strategies in `rlevo::evo` treat fitness as **cost** — lower is better.
Maximisation problems must be negated. The food source \\(F\\) is the lowest-cost
salp seen so far, and the best-so-far tracker is an argmin.

## Minimal example

```rust,no_run
use burn::backend::Flex;
use burn::tensor::{Tensor, TensorData, backend::Backend};
use rlevo::evo::algorithms::metaheuristic::salp::{SalpConfig, SalpSwarm};
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
    // Origin-symmetric bounds: the benign case for SSA's leader step.
    let config = SalpConfig::default_for(40, 10);

    let mut harness = EvolutionaryHarness::<B, _, _>::new(
        SalpSwarm::<B>::new(),
        config,
        SphereCost,
        /* seed */ 3,
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

**Parallel follower stencil.** The chain average is computed without a host-side
loop: the swarm is reassembled as `[new_leaders; old_followers]`, and each
follower gathers the row directly ahead with one `select`. The boundary follower
reads the updated last leader; the rest read the pre-update follower ahead of
them. This keeps the update GPU-friendly and deterministic.

**Host-sampled leader steps.** \\(c_2\\) and \\(c_3\\) are host-sampled per leader
per coordinate from a `seed_stream` (`SeedPurpose::Other`, keyed by the generation
counter) and uploaded; the strategy never calls `Tensor::random` or `B::seed`
(the host-RNG convention), so the leader jitter is reproducible across thread
schedules.

**Two-cycle bootstrap.** SSA needs a food source before leaders can move. The
first `ask` returns the initial swarm unchanged (it detects an empty fitness
cache) and the first `tell` seeds the cache and sets the food source. The leader
and follower updates go live from the **second `ask`**; calling `ask` twice
without an intervening `tell` panics, since the food source would be unset.
[`EvolutionaryHarness`](../appendix-d-suppl/ask-tell-contract.md) handles this
transparently.

**Reproducibility.** Initial positions (`SeedPurpose::Init`) and the
per-generation leader draws (`SeedPurpose::Other`, keyed by the generation
counter) come from independent host sub-streams. Two runs with the same seed and
config produce identical trajectories on a fixed backend.

## When to use

| Situation | Recommendation |
|---|---|
| General continuous optimisation | Prefer [PSO](particle-swarm-optimization.md) — SSA is a biased walk plus a chain average, with fewer studied guarantees |
| Shifted or asymmetric search domain | Avoid SSA — the leader step is shift-variant (Castelli et al., 2022); use [CMA-ES](cma-es.md) or [DE](differential-evolution.md) |
| Need high-precision convergence | [CMA-ES](cma-es.md) — SSA reduces strongly on centred problems but plateaus short of machine precision |
| Strong variable dependencies | [CMA-ES](cma-es.md) — SSA perturbs each coordinate independently |
| You specifically need an SSA baseline | This implementation — note the `pop_size/2` leader split deviates from the paper's single leader |
| Binary / combinatorial search | [Binary GA](binary-encoded-genetic-algorithm.md) — SSA is defined over continuous swarms |

The Salp Swarm Algorithm sits within the broader critique of metaphor-based
metaheuristics (Sörensen, 2015), with a specific shift-variance correction from
Castelli et al. (2022); prefer it as a comparator, not a workhorse.

---

*Co-Authored-By: Anthropic Claude Opus 4.8*\
*Reviewed-By: (Human) Anthony Torlucci*
