# Firefly Algorithm

<!-- source: crates/rlevo-evolution/src/algorithms/metaheuristic/firefly.rs -->

The Firefly Algorithm (FA; Yang, 2008) is a swarm metaheuristic built on a
bioluminescence metaphor: each firefly is a candidate solution whose *brightness*
encodes fitness, and every firefly is drawn towards every **brighter** one, with
attraction falling off exponentially in the squared distance between them. A
firefly with no brighter neighbour performs a random walk instead. The result is
a multi-attractor flock: rather than a single global pull as in
[PSO](particle-swarm-optimization.md), each individual feels a weighted sum of
pulls towards all the brighter individuals it can "see", which in principle lets
several basins be tracked at once.

`rlevo` ships FA as a **legacy comparator**. Camacho-Villalón, Stützle and Dorigo
(2020) — extended in their 2023 survey — analyse it component-by-component and
show its pairwise attraction reduces to PSO-style attractor moves under a new
metaphor, introducing no algorithmic mechanism absent from the prior swarm
literature. It is provided because it is widely cited and because the multi-pull
structure is occasionally informative on multimodal landscapes, not because it is
recommended: for general continuous work, [PSO](particle-swarm-optimization.md)
is the default in `rlevo`, and [CMA-ES](cma-es.md) the choice when precision
matters.

`rlevo` implements it as `FireflyAlgorithm` under
`rlevo::evo::algorithms::metaheuristic::firefly`, operating on real-valued
positions of shape `(pop_size, D)`.

## The attraction update

`ask` moves every firefly towards each brighter firefly in a single pass.
Attractiveness decays with the **squared** Euclidean distance \\(r_{ij} = \lVert
\mathbf{x}_i - \mathbf{x}_j\rVert\\):

```math
\beta(r_{ij}) = \beta_0 \, e^{-\gamma\, r_{ij}^2}.
```

Firefly \\(i\\) sums a displacement towards every firefly \\(j\\) that is strictly
brighter (lower cost), plus a uniform random kick:

```math
\Delta\mathbf{x}_i = \sum_{j\, :\, f(\mathbf{x}_j) < f(\mathbf{x}_i)}
\beta(r_{ij})\,(\mathbf{x}_j - \mathbf{x}_i)
\;+\; \alpha\,\bigl(\mathcal{U}[0,1]^D - \tfrac{1}{2}\bigr),
\qquad
\mathbf{x}_i' = \operatorname{clamp}(\mathbf{x}_i + \Delta\mathbf{x}_i,\ \text{bounds}).
```

The brightness test is strict (\\(<\\)), so a firefly with no brighter neighbour
receives no attraction term and moves on the noise alone. The sum runs over *all*
brighter fireflies, not just the brightest — this is the distinguishing feature
of FA over a single-attractor swarm, and the reason the update is canonically
\\(O(N^2 D)\\).

The base attractiveness \\(\beta_0\\) (default \\(1.0\\)) sets the pull strength at
zero distance; the absorption coefficient \\(\gamma\\) (default \\(0.01\\)) sets the
range over which fireflies see each other; the noise scale \\(\alpha\\) (default
\\(0.2\\)) sets the exploration floor.

> **\\(\gamma\\) is scaled to the search box, not Yang's canonical \\(1.0\\).**
> The original paper assumes positions normalised to \\([0,1]\\), where
> \\(\gamma = 1\\) is appropriate. `rlevo`'s default domain is \\([-5.12, 5.12]\\)
> (width \\(L \approx 10.24\\)), so distances — and hence \\(r^2\\) — are far
> larger; the default \\(\gamma = 0.01 \approx 1/L^2\\) keeps \\(e^{-\gamma r^2}\\)
> from collapsing to zero across typical pairs. Retune \\(\gamma\\) if you change
> `bounds`.

## The `O(N²D)` cost and the population cap

The attraction sum needs the full pairwise displacement tensor \\(\mathbf{x}_j -
\mathbf{x}_i\\) of shape `(N, N, D)`, materialised on device. That is cubic in the
swarm size, so the pure-tensor path is **hard-capped at `pop_size ≤ 128`**:

- Without the `custom-kernels` feature, `init` asserts the cap and panics above
  it (`FIREFLY_PURE_TENSOR_CAP = 128`).
- With the feature on, the cap is a `debug_assert!` only, because the fused
  kernel that would lift it (`pairwise_attract_cube`, designed to stream over the
  neighbour axis at \\(O(ND)\\) memory) is a placeholder — the pure-tensor path
  still runs.

Until that kernel lands, treat 128 as the working ceiling on swarm size.

## Configuration

```rust,no_run
use rlevo::evo::algorithms::metaheuristic::firefly::FireflyConfig;

// Explicit construction:
let config = FireflyConfig {
    pop_size:   32,
    genome_dim: 10,
    bounds:     (-5.12, 5.12),
    beta0:      1.0,    // attractiveness at zero distance
    gamma:      0.01,   // absorption; scale ≈ 1/L² to your box width
    alpha:      0.2,    // random-walk noise scale
};

// Or use the defaults (bounds = (-5.12, 5.12), β₀ = 1.0, γ = 0.01, α = 0.2):
let config = FireflyConfig::default_for(32, 10);
```

| Field | Type | Typical range | Notes |
|---|---|---|---|
| `pop_size` | `usize` | 15 – 128 | Number of fireflies; **capped at 128** on the pure-tensor path |
| `genome_dim` | `usize` | problem-defined | Dimensionality of each firefly |
| `bounds` | `(f32, f32)` | problem-defined | Initial positions sampled here; steps clamped here |
| `beta0` (\\(\beta_0\\)) | `f32` | ~1.0 | Pull strength at zero distance |
| `gamma` (\\(\gamma\\)) | `f32` | \\(\sim 1/L^2\\) | Absorption; smaller → longer sight range. Retune with `bounds` |
| `alpha` (\\(\alpha\\)) | `f32` | 0.01 – 0.5 | Random-walk noise; the exploration floor |

## Fitness convention

All strategies in `rlevo::evo` treat fitness as **cost** — lower is better.
Maximisation problems must be negated. "Brighter" therefore means *lower-cost*:
the attraction mask pulls firefly \\(i\\) towards every \\(j\\) with
\\(f(\mathbf{x}_j) < f(\mathbf{x}_i)\\), and the best-so-far tracker is an argmin.

## Minimal example

```rust,no_run
use burn::backend::Flex;
use burn::tensor::{Tensor, TensorData, backend::Backend};
use rlevo::evo::algorithms::metaheuristic::firefly::{FireflyAlgorithm, FireflyConfig};
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
    // Keep pop_size ≤ 128: the pure-tensor attraction is O(N²D).
    let config = FireflyConfig::default_for(24, 10);

    let mut harness = EvolutionaryHarness::<B, _, _>::new(
        FireflyAlgorithm::<B>::new(),
        config,
        SphereCost,
        /* seed */ 29,
        device,
        /* max_generations */ 500,
    );

    harness.reset();
    while !harness.step(()).done {}

    let best = harness.latest_metrics().unwrap().best_fitness_ever;
    println!("best cost = {best:.4e}");
}
```

## Implementation notes

**Host-side brightness mask and noise.** The strictly-brighter test
\\(f(\mathbf{x}_j) < f(\mathbf{x}_i)\\) is built on the host into an `(N, N)`
integer mask and uploaded, then used to zero out non-brighter pairs in
\\(\beta\\) before the displacement is summed over the neighbour axis. The
random-walk kick is host-sampled from a `seed_stream` and uploaded with
`Tensor::from_data`; the strategy never calls `Tensor::random` or `B::seed` (the
host-RNG convention), so draws are reproducible across thread schedules.

**Squared-distance shortcut.** Pairwise \\(r_{ij}^2\\) is read off the same
`(N, N, D)` difference tensor the displacement needs, so no separate distance
pass is required; the cost is dominated by allocating that cubic tensor, which is
exactly what the population cap bounds.

**Two-cycle bootstrap.** Like the other swarm strategies, FA needs a fitness
cache before it can move. The first `ask` returns the initial positions unchanged
(it detects an empty fitness cache) and the first `tell` seeds the cache and the
best-so-far record. Attraction goes live from the **second `ask`**.
[`EvolutionaryHarness`](../appendix-d-suppl/ask-tell-contract.md) handles this
transparently; a hand-written loop must run the bootstrap cycle first.

**CubeCL kernel is deferred.** The fused `pairwise_attract_cube` kernel (gated
behind the `custom-kernels` feature) that would stream the attraction at
\\(O(ND)\\) memory and remove the 128-firefly cap is designed but not yet
implemented; FA uses the pure-tensor path above until it lands.

**Reproducibility.** Initial positions (`SeedPurpose::Init`) and the
per-generation random walk (`SeedPurpose::Mutation`, keyed by the generation
counter) come from independent host sub-streams. Two runs with the same seed and
config produce identical trajectories on a fixed backend.

## When to use

| Situation | Recommendation |
|---|---|
| General continuous optimisation | Prefer [PSO](particle-swarm-optimization.md) — a single well-studied attractor, fewer caveats, no population cap |
| Multimodal landscape, several basins worth tracking | Firefly's multi-pull sum is its reason to exist; lower \\(\gamma\\) to widen sight range, but mind the 128-firefly cap |
| Need high-precision convergence on a smooth landscape | [CMA-ES](cma-es.md) or [DE](differential-evolution.md) — FA reduces to a PSO-style attractor and plateaus early |
| Large swarm (`pop_size > 128`) | Not supported on the pure-tensor path; the fused kernel that would lift the cap is not yet implemented |
| Strong variable dependencies | [CMA-ES](cma-es.md) — FA perturbs and attracts each coordinate independently |
| Binary / combinatorial search | [Binary GA](binary-encoded-genetic-algorithm.md) — FA is defined over continuous positions |

The Firefly Algorithm is one of the metaphor-based methods examined by
Camacho-Villalón et al. (2020, 2023) and the broader critique of Sörensen (2015);
prefer it as a comparator, not a workhorse.

---

*Co-Authored-By: Anthropic Claude Opus 4.8*\
*Reviewed-By: (Human) Anthony Torlucci*
