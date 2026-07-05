# Grey Wolf Optimizer

<!-- source: crates/rlevo-evolution/src/algorithms/metaheuristic/gwo.rs -->

The Grey Wolf Optimizer (GWO; Mirjalili, Mirjalili & Lewis, 2014) is a swarm
metaheuristic that dresses a weighted-attractor update in a pack-hunting metaphor.
Each generation the pack is ranked; the three fittest wolves are named \\(\alpha\\),
\\(\beta\\), \\(\delta\\), and every wolf is pulled towards the average of three
positions, one computed relative to each leader. A single coefficient \\(a\\),
annealed linearly from \\(2\\) to \\(0\\), widens the steps early (exploration) and
shrinks them late (exploitation).

`rlevo` ships GWO as a **legacy comparator**. Camacho-Villalón, Dorigo and Stützle
(2020; 2023) show its update is algorithmically a PSO-style weighted attractor
with no novel search mechanism over Kennedy & Eberhart (1995) — the "leaders" play
the role of attractors and the \\(a\\) schedule the role of an inertia decay. It is
provided because it is widely cited and exercises the `Strategy<B>` trait across a
wider design space, not because it is recommended: for general continuous work,
[PSO](particle-swarm-optimization.md) is the default in `rlevo`, and
[CMA-ES](cma-es.md) the choice when precision matters.

`rlevo` implements it as `GreyWolfOptimizer` under
`rlevo::evo::algorithms::metaheuristic::gwo`, operating on real-valued packs of
shape `(pop_size, D)`. The pack must hold at least three wolves.

## The pack update

`ask` ranks the pack by fitness and selects the three highest as leaders
\\(X_\alpha, X_\beta, X_\delta\\). For each leader \\(k\\) it forms two
**element-wise** coefficient tensors of shape `(pop_size, D)` from uniform draws
\\(r_1, r_2 \sim \mathcal{U}[0,1)\\):

```math
A_k = 2 a\, r_1 - a, \qquad C_k = 2\, r_2,
```

then a per-wolf distance to that leader and the leader-relative candidate:

```math
D_k = \lvert C_k \odot X_k - X \rvert, \qquad
X_k' = X_k - A_k \odot D_k.
```

The new position of every wolf is the **equal-weight** mean of the three
candidates, clamped to the search box:

```math
X' = \operatorname{clamp}\!\left(\frac{X_\alpha' + X_\beta' + X_\delta'}{3},\ \text{bounds}\right).
```

\\(C_k\\) is a stochastic emphasis on the leader coordinate (it can over- or
under-shoot the leader); \\(A_k\\) is the step gain. Because \\(A_k \in [-a, a]\\)
element-wise, when \\(a\\) is large early on \\(\lvert A_k\rvert\\) frequently
exceeds \\(1\\) and wolves can move *away* from a leader (exploration); as \\(a \to
0\\) the steps collapse onto the leaders (exploitation).

> **\\(r_1\\) and \\(r_2\\) are matrices, not scalars.** Each is drawn fresh per
> leader at full `(pop_size, D)` resolution, so \\(A_k\\) and \\(C_k\\) vary per
> wolf *and* per coordinate. This is the canonical GWO formulation; a
> scalar-per-leader simplification would behave differently.

## The exploration coefficient

The coefficient \\(a\\) is scheduled from the generation counter:

```math
a = 2\left(1 - \min\!\left(\tfrac{t}{T},\ 1\right)\right),
```

where \\(t\\) is the current generation and \\(T = \texttt{max\_generations}\\).

> **`max_generations` paces \\(a\\), it is not a stopping criterion.** The
> strategy is memoryless with respect to the harness budget: `max_generations`
> only sets how fast \\(a\\) anneals to zero. It need not equal the harness's
> own generation limit — set it to the budget over which you want the
> exploration-to-exploitation transition to complete, then clamp keeps \\(a\\)
> at \\(0\\) afterwards.

## Configuration

```rust,no_run
use rlevo::evo::algorithms::metaheuristic::gwo::GwoConfig;

// Explicit construction:
let config = GwoConfig {
    pop_size:        32,
    genome_dim:      10,
    bounds:          (-5.12, 5.12),
    max_generations: 500,   // paces a = 2(1 − t/T); not a stop condition
};

// Or use the defaults (bounds = (-5.12, 5.12), max_generations = 500):
let config = GwoConfig::default_for(32, 10);
```

| Field | Type | Typical range | Notes |
|---|---|---|---|
| `pop_size` | `usize` | ≥ 3 | Pack size; **panics below 3** (needs three leaders) |
| `genome_dim` | `usize` | problem-defined | Dimensionality of each wolf |
| `bounds` | `Bounds` | problem-defined | Initial positions sampled here; steps clamped here |
| `max_generations` | `usize` | match your budget | Annealing horizon for \\(a\\); paces exploration→exploitation only |

## Fitness convention

All strategies in `rlevo::evo` maximise a **canonical** fitness — higher is better. You declare a cost objective's direction with [`ObjectiveSense::Minimize`](https://docs.rs/rlevo-core) and the harness reconciles it at one chokepoint, so you never hand-negate. The three leaders are the three highest-fitness wolves, and the best-so-far tracker (the reported \\(\alpha\\)) is an argmax over the evaluated pack.

## Minimal example

```rust,no_run
use burn::backend::Flex;
use burn::tensor::{Tensor, TensorData, backend::Backend};
use rlevo::evo::algorithms::metaheuristic::gwo::{GreyWolfOptimizer, GwoConfig};
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
    // Set max_generations to the budget so a anneals to 0 by the end.
    let config = GwoConfig::default_for(32, 10);

    let mut harness = EvolutionaryHarness::<B, _, _>::new(
        GreyWolfOptimizer::<B>::new(),
        config,
        SphereCost,
        /* seed */ 11,
        device,
        /* max_generations */ 600,
    ).expect("valid config");

    harness.reset();
    while !harness.step(()).done {}

    let best = harness.latest_metrics().unwrap().best_fitness_ever;
    println!("best cost = {best:.4e}");
}
```

## Implementation notes

**Host-side ranking.** The pack is ranked on the host (`argtop3_max` finds the
three highest-fitness individuals in one pass), avoiding backend-specific `argsort` quirks; the
\\(O(N\log N)\\) cost is negligible against the device tensor ops. The three
leaders are then gathered with a single `select`.

**Host-sampled coefficients.** For each of the three leaders, \\(r_1\\) and
\\(r_2\\) are host-sampled from independent `seed_stream`s
(`SeedPurpose::Other` and `SeedPurpose::Mutation`, keyed by `generation·3 + k`)
and uploaded; the strategy never calls `Tensor::random` or `B::seed` (the
host-RNG convention), so draws are reproducible across thread schedules.

**No pack elitism.** The update overwrites every wolf — including the leaders —
with its averaged candidate, so the \\(\alpha\\) position is not carried forward
inside the pack. The best-so-far genome is tracked *separately* in `tell` (a
strict-improvement argmax), so the reported best never regresses even though the
pack itself can.

**Two-cycle bootstrap.** GWO needs a fitness cache before it can rank. The first
`ask` returns the initial pack unchanged (it detects an empty fitness cache) and
the first `tell` seeds the cache and the best-so-far record. The leader update
goes live from the **second `ask`**.
[`EvolutionaryHarness`](../appendix-d-suppl/ask-tell-contract.md) handles this
transparently; a hand-written loop must run the bootstrap cycle first.

**Reproducibility.** Initial positions (`SeedPurpose::Init`) and the
per-generation, per-leader coefficient draws come from independent host
sub-streams keyed by the generation counter. Two runs with the same seed and
config produce identical trajectories on a fixed backend.

## When to use

| Situation | Recommendation |
|---|---|
| General continuous optimisation | Prefer [PSO](particle-swarm-optimization.md) — GWO is a PSO-style attractor with extra metaphor and fewer studied guarantees |
| Need high-precision convergence on a smooth landscape | [CMA-ES](cma-es.md) or [DE](differential-evolution.md) — GWO converges strongly but plateaus short of machine precision |
| Strong variable dependencies | [CMA-ES](cma-es.md) — GWO's per-coordinate coefficients carry no covariance model |
| You specifically need a GWO baseline for comparison | This implementation — it follows the 2014 formulation (matrix \\(A\\)/\\(C\\), equal-weight three-leader mean) faithfully |
| Binary / combinatorial search | [Binary GA](binary-encoded-genetic-algorithm.md) — GWO is defined over continuous packs |

The Grey Wolf Optimizer is one of the metaphor-based methods examined by
Camacho-Villalón et al. (2020, 2023) and the broader critique of Sörensen (2015);
prefer it as a comparator, not a workhorse.

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
