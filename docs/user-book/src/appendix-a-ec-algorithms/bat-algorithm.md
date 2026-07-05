# Bat Algorithm

<!-- source: crates/rlevo-evolution/src/algorithms/metaheuristic/bat.rs -->

The Bat Algorithm (BA; Yang, 2010) is a swarm metaheuristic built on an
echolocation metaphor. Each bat carries a position, a velocity, an emission
*frequency* \\(f\\), a *loudness* \\(A\\), and a *pulse rate* \\(r\\). Each
generation a bat flies by a frequency-tuned velocity update (a PSO-like move
relative to the global best), occasionally substitutes a short random walk around
the best, and then accepts the new position only probabilistically and only if it
improves — a simulated-annealing-style gate. As a bat closes in, its loudness
decays and its pulse rate rises, throttling late-stage exploration.

`rlevo` ships BA as a **legacy comparator**. Its velocity/position update is
structurally a PSO variant towards the global best, and the probabilistic
acceptance is a thin annealing layer; Camacho-Villalón et al. (2020; 2023) find no
search mechanism not already present in simpler algorithms. It is provided for API
coverage, not as a recommendation: for general continuous work,
[PSO](particle-swarm-optimization.md) is the default in `rlevo`, and
[CMA-ES](cma-es.md) the choice when precision matters.

`rlevo` implements it as `BatAlgorithm` under
`rlevo::evo::algorithms::metaheuristic::bat`, operating on real-valued positions
of shape `(pop_size, D)`, with per-bat loudness and pulse rate held host-side.

## The move

`ask` proposes one candidate per bat. First a frequency is drawn per bat from
\\(\beta \sim \mathcal{U}[0,1]\\):

```math
f_i = f_{\min} + (f_{\max} - f_{\min})\,\beta_i,
```

then the velocity and global move, with \\(x^*\\) the current best:

```math
v_i \leftarrow v_i + (x_i - x^*)\, f_i, \qquad x_i' = x_i + v_i.
```

With probability \\(1 - r_i\\) (i.e. when a uniform draw exceeds the pulse rate)
the global move is replaced by a **local walk** around the best, scaled by the
colony's mean loudness \\(\bar{A}\\):

```math
x_i' = x^* + \varepsilon\,\bar{A}, \qquad \varepsilon \sim \mathcal{U}[-1,1].
```

The candidate is then clamped to the search box. Both arms are computed as full
tensors and selected with a boolean mask — no per-bat branching on device.

> **The velocity term keeps Yang's original sign, \\((x_i - x^*)\\).** `rlevo`
> implements the 2010 paper literally; many later write-ups silently reverse it to
> \\((x^* - x_i)\\). With \\(f_i \ge 0\\) (since \\(f_{\min} = 0\\)) the literal
> term is diffusive rather than strictly attractive, so the convergence pressure
> comes mainly from the **local walk around \\(x^*\\)** and the greedy acceptance
> gate below — not from the velocity update. If you port results from a
> reversed-sign implementation, expect different trajectories.

## Acceptance, loudness, and pulse rate

The accept decision is split across the two-phase contract: `ask` records, per
bat, whether a uniform draw fell below that bat's *current* loudness; `tell`
applies the gate once fitness is known. Candidate \\(i\\) replaces position
\\(i\\) iff

```math
\underbrace{u_i < A_i}_{\text{recorded in } ask} \quad\text{and}\quad
\underbrace{f(x_i') \ge f(x_i)}_{\text{checked in } tell},
```

a per-slot \\((1+1)\\) gate: a bat competes only against its own previous
position. On acceptance, loudness decays and pulse rate grows:

```math
A_i \leftarrow \alpha\, A_i, \qquad r_i = r_0\bigl(1 - e^{-\gamma t}\bigr),
```

so an accepting bat becomes quieter (less likely to accept future moves) and
emits faster (more likely to take the local walk), both of which damp exploration
as \\(t\\) grows.

The first generation is a **bootstrap**: `tell` unconditionally accepts all
initial positions, seeds the fitness cache and the best, and skips the
loudness/pulse updates. The acceptance gate goes live from the second generation.

## Configuration

```rust,no_run
use rlevo::evo::algorithms::metaheuristic::bat::BatConfig;

// Explicit construction:
let config = BatConfig {
    pop_size:   40,
    genome_dim: 10,
    bounds:     (-5.12, 5.12),
    f_min:      0.0,    // minimum echolocation frequency
    f_max:      2.0,    // maximum echolocation frequency
    a0:         1.0,    // initial loudness
    r0:         0.5,    // initial pulse rate
    alpha:      0.9,    // loudness decay (0 < α ≤ 1)
    gamma:      0.9,    // pulse-rate growth (γ > 0)
};

// Or use the defaults (f∈[0,2], A₀=1.0, r₀=0.5, α=γ=0.9, bounds = (-5.12, 5.12)):
let config = BatConfig::default_for(40, 10);
```

| Field | Type | Typical range | Notes |
|---|---|---|---|
| `pop_size` | `usize` | 20 – 50 | Number of bats |
| `genome_dim` | `usize` | problem-defined | Dimensionality of each bat |
| `bounds` | `(f32, f32)` | problem-defined | Initial positions sampled here; candidates clamped here |
| `f_min`, `f_max` (\\(f_{\min}, f_{\max}\\)) | `f32` | \\(0\\) – \\(O(1)\\) | Frequency range; sets the velocity step scale |
| `a0` (\\(A_0\\)) | `f32` | ~1.0 | Initial loudness; gates acceptance, decays by \\(\alpha\\) |
| `r0` (\\(r_0\\)) | `f32` | 0 – 1 | Initial pulse rate; gates the local walk, grows with \\(t\\) |
| `alpha` (\\(\alpha\\)) | `f32` | (0, 1], ~0.9 | Loudness decay per acceptance |
| `gamma` (\\(\gamma\\)) | `f32` | > 0, ~0.9 | Pulse-rate growth rate |

## Fitness convention

All strategies in `rlevo::evo` maximise a **canonical** fitness — higher is better. You declare a cost objective's direction with [`ObjectiveSense::Minimize`](https://docs.rs/rlevo-core) and the harness reconciles it at one chokepoint, so you never hand-negate. The acceptance gate keeps a candidate only when its fitness is no worse than the incumbent (\\(f(x_i') \ge f(x_i)\\)), and the best-so-far tracker is an argmax over the colony.

## Minimal example

```rust,no_run
use burn::backend::Flex;
use burn::tensor::{Tensor, TensorData, backend::Backend};
use rlevo::evo::algorithms::metaheuristic::bat::{BatAlgorithm, BatConfig};
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
    let config = BatConfig::default_for(40, 10);

    let mut harness = EvolutionaryHarness::<B, _, _>::new(
        BatAlgorithm::<B>::new(),
        config,
        SphereCost,
        /* seed */ 23,
        device,
        /* max_generations */ 800,
    ).expect("valid config");

    harness.reset();
    while !harness.step(()).done {}

    let best = harness.latest_metrics().unwrap().best_fitness_ever;
    // The loudness decay throttles late progress: expect a strong
    // reduction from the random baseline, not machine zero.
    println!("best cost = {best:.4e}");
}
```

## Implementation notes

**Split-phase acceptance draw.** The loudness gate \\(u_i < A_i\\) is sampled in
`ask` (against the loudness *before* any decay) and stored in
`state.pending_accept`; `tell` combines it with the improvement test. Recording
the draw in `ask` keeps the random stream aligned with the rest of the
generation's host sampling, so acceptance is reproducible rather than dependent on
when `tell` happens to draw.

**Host-sampled randomness.** \\(\beta\\), the pulse check, the acceptance draw, and
the local-walk \\(\varepsilon\\) are all host-sampled from one `seed_stream`
(`SeedPurpose::Other`, keyed by the generation counter) and uploaded; the strategy
never calls `Tensor::random` or `B::seed` (the host-RNG convention). BA's draws
are uniform, so it has none of [Cuckoo Search](cuckoo-search.md)'s fractional-power
backend-parity caveat.

**Greedy gather.** Acceptance is realised by concatenating `[positions;
candidates]` and selecting row \\(i\\) or row \\(\texttt{pop}+i\\) per bat — a
single `select`, no host loop over the position tensor.

**Two-cycle bootstrap.** The first `ask` returns the initial colony unchanged (it
detects an empty fitness cache); the first `tell` accepts all of it
unconditionally and seeds the best. The frequency/velocity update and the
acceptance gate go live from the **second `ask`**; calling `ask` twice without an
intervening `tell` panics, since the best would be unset.
[`EvolutionaryHarness`](../appendix-d-suppl/ask-tell-contract.md) handles this
transparently.

**Reproducibility.** Initial positions (`SeedPurpose::Init`) and the
per-generation draws (`SeedPurpose::Other`, keyed by the generation counter) come
from independent host sub-streams. Two runs with the same seed and config produce
identical trajectories on a fixed backend.

## When to use

| Situation | Recommendation |
|---|---|
| General continuous optimisation | Prefer [PSO](particle-swarm-optimization.md) — BA is a PSO variant plus an annealing gate, with more knobs and fewer guarantees |
| Need high-precision convergence | [CMA-ES](cma-es.md) or [DE](differential-evolution.md) — BA's loudness decay throttles late progress and it plateaus early |
| Strong variable dependencies | [CMA-ES](cma-es.md) — BA perturbs each coordinate independently |
| You specifically need a Bat baseline | This implementation — faithful to Yang's 2010 formulation, including the original velocity sign |
| Porting results from another BA implementation | Check the velocity sign convention first — `rlevo` uses \\((x_i - x^*)\\); reversed-sign ports will not match |
| Binary / combinatorial search | [Binary GA](binary-encoded-genetic-algorithm.md) — BA is defined over continuous positions |

The Bat Algorithm is one of the metaphor-based methods examined by
Camacho-Villalón et al. (2020, 2023) and the broader critique of Sörensen (2015);
prefer it as a comparator, not a workhorse.

---

*Drafted, Edited, and Reviewed By: (Human) Anthony Torlucci*\
*Co-Authored-By: Anthropic Claude Opus 4.8*
