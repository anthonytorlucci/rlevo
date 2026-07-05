# Particle Swarm Optimization

<!-- source: crates/rlevo-evolution/src/algorithms/metaheuristic/pso.rs -->

Particle Swarm Optimization (PSO; Kennedy & Eberhart, 1995) optimises by flying a
swarm of particles through the search space. Each particle carries a **position**
`x` (a candidate solution) and a **velocity** `v`, and remembers the best
position it has personally visited (`pbest`). The swarm as a whole tracks the
best position any particle has found (`gbest`). Every step nudges each particle's
velocity toward its own best and toward the swarm's best, then moves it.

Its key insight is the balance of two pulls: a **cognitive** term draws a particle
back toward its own past success, and a **social** term draws it toward the
swarm's collective success. The interplay — momentum carrying a particle forward,
two attractors reining it in — produces a search that explores broadly early
(particles are spread out, velocities large) and refines late (the swarm
contracts around `gbest`). Unlike the genetic algorithms, there is no selection
or recombination: the population is never culled, it *moves*.

## The velocity update

PSO's entire behaviour lives in how velocity is updated each generation. `rlevo`
offers the two canonical forms, selected by `PsoVariant`. Both use the
*global-best* topology (every particle sees the single swarm-wide best).

**Inertia weight** (Shi & Eberhart, 1998):

```math
\mathbf{v}_i \leftarrow \omega\,\mathbf{v}_i
\;+\; c_1\,\mathbf{r}_1 \odot (\mathbf{p}_i - \mathbf{x}_i)
\;+\; c_2\,\mathbf{r}_2 \odot (\mathbf{g} - \mathbf{x}_i)
```

**Constriction factor** (Clerc & Kennedy, 2002):

```math
\mathbf{v}_i \leftarrow \chi\left(\mathbf{v}_i
\;+\; c_1\,\mathbf{r}_1 \odot (\mathbf{p}_i - \mathbf{x}_i)
\;+\; c_2\,\mathbf{r}_2 \odot (\mathbf{g} - \mathbf{x}_i)\right)
```

In both, \\(\mathbf{p}_i\\) is particle `i`'s personal best, \\(\mathbf{g}\\) the
global best, and \\(\odot\\) is element-wise multiplication. The three
contributions are the **inertia/momentum** term (carry the current heading), the
**cognitive** term (pull to `pbest`, weighted by `c1`), and the **social** term
(pull to `gbest`, weighted by `c2`). After the update the velocity is clamped to
\\([-v_{\max}, v_{\max}]\\) and the new position is

```math
\mathbf{x}_i \leftarrow \mathrm{clamp}\bigl(\mathbf{x}_i + \mathbf{v}_i,\ \text{bounds}\bigr).
```

> **`r1` and `r2` are matrices, not scalars.** Each is an independent
> \\((\text{pop} \times D)\\) draw from \\(U[0,1)\\), so *every coordinate of
> every particle* gets its own random cognitive and social weighting. This is the
> standard PSO formulation: the stochasticity is per-dimension, which decouples
> the pull across axes rather than scaling a whole vector by one random number.

## The constriction factor

The constriction factor is a closed form that guarantees the swarm contracts
rather than explodes:

```math
\chi = \frac{2}{\left|\,2 - \varphi - \sqrt{\varphi^2 - 4\varphi}\,\right|},
\qquad \varphi = c_1 + c_2.
```

Clerc & Kennedy's analysis requires \\(\varphi > 4\\) for the discriminant to be
real. The implementation enforces this with a `debug_assert!`; if a release build
supplies \\(\varphi \le 4\\), the discriminant is clamped to zero and `χ` falls
back to `1.0` (no contraction) so the strategy stays numerically well-defined
rather than producing a `NaN`.

**Why the two variants agree under the defaults.** The default configuration
(`ω = 0.7298`, `c1 = c2 = 1.49618`, `Inertia`) is not arbitrary — it is the
*expansion* of the constriction form at `φ = 4.1`. With `c1 = c2 = 2.05`
(`φ = 4.1`) the factor is `χ ≈ 0.7298`, and distributing it gives
`χ·v + (χ·2.05)·r⊙(…) = 0.7298·v + 1.49618·r⊙(…)`. So the default inertia weight
*is* `χ` and the default coefficients *are* `χ · 2.05`: the two variants describe
the same dynamics at the canonical settings, and differ only when you retune.

## Configuration

```rust,no_run
use rlevo::evo::algorithms::metaheuristic::pso::{PsoConfig, PsoVariant};

// Explicit construction:
let config = PsoConfig {
    pop_size:   32,
    genome_dim: 10,
    bounds:     (-5.12, 5.12),
    inertia:    0.7298,            // ω — only used by the Inertia variant
    c1:         1.49618,           // cognitive (pbest) coefficient
    c2:         1.49618,           // social (gbest) coefficient
    v_max:      5.12,              // per-dimension velocity clamp
    variant:    PsoVariant::Inertia,
};

// Or use the canonical defaults (ω = 0.7298, c1 = c2 = 1.49618, Inertia):
let config = PsoConfig::default_for(32, 10);
```

| Field | Type | Typical range | Notes |
|---|---|---|---|
| `pop_size` | `usize` | 20 – 50 | Swarm size; one evaluation per particle per generation |
| `genome_dim` | `usize` | problem-defined | Position dimensionality |
| `bounds` | `(f32, f32)` | problem-defined | Initial positions sampled here; positions clamped here each step |
| `inertia` (`ω`) | `f32` | 0.4 – 0.9 | Momentum; **Inertia variant only**. Higher → more exploration |
| `c1` | `f32` | ~1.5 – 2.05 | Cognitive pull toward `pbest` |
| `c2` | `f32` | ~1.5 – 2.05 | Social pull toward `gbest` |
| `v_max` | `f32` | ~½ search extent | Velocity clamp; caps step size to prevent swarm explosion |
| `variant` | `PsoVariant` | — | `Inertia` or `Constriction` |

> **Switching to `Constriction` requires retuning `c1`/`c2`.** The defaults give
> `φ = c1 + c2 ≈ 2.99`, which **violates** Clerc & Kennedy's `φ > 4` requirement.
> Set `c1 = c2 = 2.05` (`φ = 4.1`, `χ ≈ 0.7298`) when selecting the constriction
> variant, or `χ` silently falls back to `1.0`.

## Fitness convention

All strategies in `rlevo::evo` maximise a **canonical** fitness — higher is better. You declare a cost objective's direction with [`ObjectiveSense::Minimize`](https://docs.rs/rlevo-core) and the harness reconciles it at one chokepoint, so you never hand-negate. A particle updates its `pbest` only when the new fitness is strictly higher, and `gbest` tracks the argmax over all personal bests.

## Minimal example

```rust,no_run
use burn::backend::Flex;
use burn::tensor::{Tensor, TensorData, backend::Backend};
use rlevo::evo::algorithms::metaheuristic::pso::{ParticleSwarm, PsoConfig};
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
    let config = PsoConfig::default_for(32, 10);   // ω = 0.7298, Inertia

    let mut harness = EvolutionaryHarness::<B, _, _>::new(
        ParticleSwarm::<B>::new(),
        config,
        SphereCost,
        /* seed */ 42,
        device,
        /* max_generations */ 500,
    ).expect("valid config");

    harness.reset();
    while !harness.step(()).done {}

    let best = harness.latest_metrics().unwrap().best_fitness_ever;
    println!("best cost = {best:.4e}");   // converges below 1e-6
}
```

## Implementation notes

**Zero velocity initialisation.** Velocities start at exactly **zero**, not at a
random slice of \\([-v_{\max}, v_{\max}]\\) as some references do. The first move
is therefore driven purely by the cognitive and social pulls. This converges
slightly faster on Sphere and — more importantly here — makes the initial swarm
bit-reproducible independent of the velocity clamp.

**Global-best topology only.** The swarm is fully connected: every particle is
attracted to the single best position found anywhere in the swarm. Ring and other
local neighbourhood topologies — which slow information spread and resist
premature convergence on multimodal problems — are *not* implemented. This is the
main caveat when applying PSO here to rugged landscapes.

**Greedy `pbest`, vectorised.** `tell` compares each particle's new fitness to its
`pbest` host-side, builds a `(pop)` improvement mask, broadcasts it to
`(pop, D)`, and uses `mask_where` to update only the improved rows in one tensor
op. The global best is then the `argmax` over the refreshed personal-best cache,
and is only re-pointed when it strictly improves.

**Boundary handling.** A position that would leave `bounds` is clamped, but the
particle keeps its (clamped) velocity — so it may carry momentum back across the
boundary on the next step rather than sticking to the wall.

**Two-cycle bootstrap.** Like the other swarm strategies, PSO needs a fitness
cache before it can move. The first `ask` returns the initial positions unchanged
(it detects an empty `personal_best_fitness`), and the first `tell` seeds the
personal bests and the global best. The velocity update goes live from the
**second `ask`**. [`EvolutionaryHarness`](../appendix-d-suppl/ask-tell-contract.md)
handles this transparently; a hand-written loop must run the bootstrap cycle
first.

**Reproducibility.** Initial positions (`SeedPurpose::Init`) and the per-generation
`r1` / `r2` matrices (`SeedPurpose::Other` and `SeedPurpose::Mutation`) are drawn
from independent host-side sub-streams via `seed_stream`, never `Tensor::random`
or `B::seed`. Two runs with the same root seed produce identical trajectories.

## When to use

| Situation | Recommendation |
|---|---|
| General continuous optimisation | PSO is the recommended default metaheuristic in `rlevo` — few parameters, fast convergence |
| Smooth / unimodal landscape | Excellent fit; the global-best topology converges quickly |
| Highly multimodal landscape | Global-best PSO is **prone to premature convergence** (no local topology); consider [DE](differential-evolution.md) (`Rand1Bin`) or raise `v_max`/`ω` for more exploration |
| Want theoretical stability guarantees | `Constriction` (with `φ = c1 + c2 > 4`); mathematically equivalent to the inertia default at `φ = 4.1` |
| Strong variable dependencies | CMA-ES (adapts covariance) once implemented; PSO's per-dimension stochasticity assumes weak coupling |
| Binary / combinatorial search | [Binary GA](binary-encoded-genetic-algorithm.md); this PSO is defined over continuous positions and velocities |

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
