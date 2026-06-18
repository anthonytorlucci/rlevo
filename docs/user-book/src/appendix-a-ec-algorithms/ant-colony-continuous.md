# Ant Colony Optimization for Continuous Domains (ACOR)

<!-- source: crates/rlevo-evolution/src/algorithms/metaheuristic/aco_r.rs -->

Classical Ant Colony Optimization solves *discrete* problems: ants lay pheromone
on the edges of a graph, and a probability table over those edges biases future
path choices. ACOR (Socha & Dorigo, 2008) carries that idea into
**continuous** search spaces by replacing the discrete pheromone table with a
**solution archive** — the `k` best real-valued solutions seen so far — and
replacing edge-selection probabilities with **Gaussian sampling kernels** built
from that archive.

Its key insight: the archive *is* the pheromone. Good solutions persist in it and
bias where new samples are drawn (rank-weighted selection); the spread of the
archive sets how widely those samples scatter (the kernel width). Both adapt
automatically as the archive converges — no separate step-size schedule.

## The solution archive as a pheromone model

The colony state is an archive of `k` solutions, kept **sorted by cost**
(best first). Each generation:

1. Draw `m` offspring by sampling from Gaussian kernels centred on archive
   members.
2. Evaluate the offspring.
3. Merge them into the archive, re-sort, and keep only the top `k`.

Because step 3 is a strict top-`k` truncation over the combined `k + m` pool, the
archive is **elitist** — its quality is monotonic, it can never get worse from
one generation to the next. This is the continuous analogue of pheromone
evaporation-and-reinforcement: weak solutions fall out of the archive, strong
ones stay and keep guiding the search.

## Sampling offspring

Two quantities drive the sampling, both derived entirely from the current
archive.

**Rank weights.** Every archive member `l` (with `l = 0` the best,
`l = k − 1` the worst) gets a Gaussian-over-rank weight:

```math
w_l \;=\; \frac{1}{q\,k\,\sqrt{2\pi}}\;
\exp\!\left(-\frac{l^{2}}{2\,q^{2}\,k^{2}}\right),
\qquad l = 0, 1, \dots, k-1
```

The weights are then normalised to sum to one. The decay parameter `q` controls
how sharply selection concentrates on the top of the archive: a small `q` (the
paper's sharp setting, `≈ 0.01`) makes the colony almost always sample around the
very best solutions — strong exploitation — while a larger `q` (`≈ 0.5`) flattens
the weights toward uniform, spreading effort across the whole archive.

**Per-dimension kernel width.** The standard deviation used when sampling around
archive member `l` in dimension `d` is the mean absolute spread of the archive
from that member along that axis, scaled by the exploration factor `ξ`:

```math
\sigma_{l,d} \;=\; \xi \cdot \frac{1}{k-1}\sum_{e=1}^{k}\bigl|\,x_{e,d} - x_{l,d}\,\bigr|
```

As the archive collapses toward an optimum the distances shrink, so `σ` shrinks
with it — the search self-narrows from exploration to refinement. `ξ` is the one
global knob on that rate.

**The draw.** For each offspring `i` and each dimension `d`, the implementation
selects an archive member by rank-weighted roulette and samples one coordinate
from its kernel:

```math
x_{i,d} \sim \mathcal{N}\!\bigl(x_{l,d},\ \sigma_{l,d}\bigr),
\qquad l \sim \mathrm{Categorical}(w)
```

> **Per-coordinate kernel selection.** Note that `l` is drawn **independently for
> every dimension**, so a single offspring's coordinates can be guided by
> *different* archive members. This differs from Socha & Dorigo's canonical
> ACOR, where each ant picks one guiding solution and samples all
> dimensions from it. The per-coordinate variant mixes archive members more
> aggressively; like the coordinate-wise step in
> [ABC](artificial-bee-colony.md), it suits *separable* problems and does not
> preserve correlations between coordinates — for strongly coupled variables an
> approach that models covariance is preferable.

## Configuration

```rust,no_run
use rlevo::evo::algorithms::metaheuristic::aco_r::AcoRConfig;

// Explicit construction:
let config = AcoRConfig {
    archive_size: 50,            // k — the pheromone memory
    m:            30,            // offspring sampled per generation
    genome_dim:   10,
    bounds:       (-5.12, 5.12),
    xi:           0.85,          // exploration scale
    q:            0.1,           // rank-weight decay
};

// Or use the canonical defaults (ξ = 0.85, q = 0.1):
let config = AcoRConfig::default_for(/* k */ 50, /* m */ 30, /* dim */ 10);
```

| Field | Type | Typical range | Notes |
|---|---|---|---|
| `archive_size` (`k`) | `usize` | 20 – 100 | Pheromone memory; **≥ 2 required** (σ needs a pairwise distance). Paper recommends 50 |
| `m` | `usize` | k/2 – 2k | Offspring per generation; **≥ 1 required**. Larger → more exploration per generation |
| `genome_dim` | `usize` | problem-defined | Dimensionality of the search space |
| `bounds` | `(f32, f32)` | problem-defined | Initial archive sampled here; offspring clamped here after sampling |
| `xi` (`ξ`) | `f32` | 0.5 – 1.0 | Exploration scale on `σ`; higher → wider kernels, slower convergence |
| `q` | `f32` | 0.01 – 0.5 | Rank-weight decay; **small → sharp exploitation**, large → flat exploration |

`q` and `ξ` are the explore/exploit dials and the parameters most worth tuning:
`q` decides *which* archive members guide the search, `ξ` decides *how far* the
offspring scatter around them.

## Fitness convention

All strategies in `rlevo::evo` treat fitness as **cost** — lower is better.
Maximisation problems must be negated. The archive is sorted ascending, so
`archive[0]` is always the incumbent best and the top-`k` truncation keeps the
lowest-cost solutions.

## Minimal example

```rust,no_run
use burn::backend::Flex;
use burn::tensor::{Tensor, TensorData, backend::Backend};
use rlevo::evo::algorithms::metaheuristic::aco_r::{AcoRConfig, AntColonyReal};
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
    let config = AcoRConfig::default_for(50, 30, 10);   // ξ = 0.85, q = 0.1

    let mut harness = EvolutionaryHarness::<B, _, _>::new(
        AntColonyReal::<B>::new(),
        config,
        SphereCost,
        /* seed */ 17,
        device,
        /* max_generations */ 400,
    );

    harness.reset();
    while !harness.step(()).done {}

    let best = harness.latest_metrics().unwrap().best_fitness_ever;
    println!("best cost = {best:.4e}");   // converges below 1e-3
}
```

## Implementation notes

**On-device σ via a 3-D broadcast.** The per-member, per-dimension spread
\\(\sigma_{l,d}\\) is a `(k, d)` matrix computed without a host loop: the archive
is unsqueezed to `(1, k, d)` and `(k, 1, d)`, broadcast-subtracted and `abs`'d to
a `(k, k, d)` distance cube, summed over the `e` axis, and scaled by
`ξ / (k − 1)`. One batched expression yields every kernel width at once.

**Host-side weighted selection and Gaussian draws.** Index selection is a CDF
walk: the normalised `weights` are accumulated into a cumulative table once, and
each of the `m · d` draws picks the first archive index whose CDF entry exceeds a
uniform `u`. The Gaussian sample itself uses `rand_distr::Normal` so that the draw
stays on the same `splitmix`-derived host stream as everything else. A `1e-12`
floor on `σ` guards against a degenerate zero-width kernel when the archive has
collapsed in some dimension.

**Elitist top-`k` merge.** `tell` concatenates `[archive; offspring]` into a
`(k + m, D)` tensor, argsorts the combined host-side cost vector, truncates to `k`
indices, and `select`s the survivors in a single gather. The archive's cost
vector is carried alongside in sorted order, so `archive[0]` is the incumbent and
feeds `best_genome`.

**Asymmetric evaluation budget.** The first `ask` returns the **full initial
archive** (`k` rows) for scoring, and the first `tell` simply sorts it — no
sampling happens yet. Every generation after that evaluates only `m` offspring.
So generation 0 costs `k` evaluations and each later generation costs `m`;
`AcoRConfig::steady_state_pop_size()` reports the latter. Budget accordingly when
comparing against fixed-population strategies.

**Reproducibility.** Initialisation (`SeedPurpose::Init`), archive-index roulette
(`SeedPurpose::Selection`), and the Gaussian offspring draws
(`SeedPurpose::Mutation`) each come from an independent host-side sub-stream via
`seed_stream`, never `Tensor::random` or `B::seed`. Two runs with the same root
seed produce identical trajectories. The archive weights are cached on the state
and recomputed only when `q` or `k` change.

## When to use

| Situation | Recommendation |
|---|---|
| Continuous optimisation with useful elite memory | ACOR keeps an explicit archive of the best solutions |
| Separable landscape | Good fit; per-coordinate kernels match separable structure |
| Strongly non-separable / rotated problem | Prefer CMA-ES (covariance) once implemented; per-coordinate sampling ignores variable coupling |
| Need an explicit explore/exploit dial | `q` (which members guide) and `ξ` (how far offspring scatter) give direct control |
| Comparison against [DE](differential-evolution.md) / [ABC](artificial-bee-colony.md) | All three are difference/archive-based; ACOR's distinctive feature is the rank-weighted Gaussian kernel over a sorted archive |
| Binary / combinatorial search | [Binary GA](binary-encoded-genetic-algorithm.md); ACOR samples real-valued Gaussians (discrete ACO is a separate model) |

---

*Co-Authored-By: Anthropic Claude Opus 4.8*\
*Reviewed-By: (Human) Anthony Torlucci*
