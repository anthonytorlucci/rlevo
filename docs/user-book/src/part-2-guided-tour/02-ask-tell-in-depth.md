# The Ask/Tell Contract

The previous section used `EvolutionaryHarness` to run the GA — it handled the
loop for you. This section opens the harness and shows what is actually happening,
because you will need to understand it when:

- you want to add custom logging or early stopping,
- you are combining evolution with RL (where the loop is more complex), or
- the harness's assumptions don't match your problem.

## The contract

Every `Strategy` in `rlevo` exposes exactly two methods:

```rust,no_run
pub trait Strategy<B: Backend> {
    type Genome;

    /// Produce the next population of candidates.
    fn ask(&mut self, rng: &mut dyn RngCore) -> Vec<Self::Genome>;

    /// Update internal state given how each candidate scored.
    fn tell(&mut self, population: Vec<Self::Genome>, fitnesses: Vec<f64>);
}
```

`ask` and `tell` are always paired — you call `ask`, evaluate every returned
candidate against your objective, then call `tell` with the *same* candidates in
the *same* order alongside their scores. The strategy never evaluates anything
itself; it only proposes and learns.

This separation is not incidental. It means:

- **You control evaluation.** Parallelize it, cache it, run it on a cluster, or
  evaluate against a human — the strategy does not care.
- **The strategy is stateless between `ask` and `tell`.** You own the candidates
  while they are being evaluated. If your evaluation takes hours, that is fine;
  the strategy waits.
- **Composition is natural.** Any code that calls `ask`/`tell` can drive any
  strategy. The harness, your custom loop, and the ERL hybrid all speak the same
  language.

## Sequence diagram

```text
   your code          Strategy              Landscape / Env
       │                  │                       │
       │──── ask() ──────►│                       │
       │◄── population ───│                       │
       │                  │                       │
       │──── evaluate(x) ─────────────────────────►│
       │◄──── f(x) ────────────────────────────────│
       │  (for each x)                             │
       │                  │                       │
       │──── tell(pop, fitnesses) ────────────────►│  (no-op on env)
       │                  │◄── update internal ───│
       │                  │    state              │
       │  (repeat)        │                       │
```

## Writing the loop yourself

Here is the GA sphere example from the previous section, rewritten without the
harness:

```rust,no_run
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rlevo::envs::landscapes::sphere::Sphere;
use rlevo::evo::algorithms::ga::{GaConfig, GeneticAlgorithm};
use rlevo::evo::strategy::Strategy;

let problem = Sphere::new(8);
let config = GaConfig { pop_size: 64, genome_dim: 8, bounds: (-5.12, 5.12), .. };
let mut ga = GeneticAlgorithm::new(config);
let mut rng = ChaCha8Rng::seed_from_u64(42);

let mut best = f64::MAX;

for gen in 0..300 {
    // 1. Ask for the next population.
    let population = ga.ask(&mut rng);

    // 2. Evaluate every candidate — this is your code.
    let fitnesses: Vec<f64> = population
        .iter()
        .map(|genome| problem.evaluate(genome))
        .collect();

    // 3. Track progress however you like.
    let gen_best = fitnesses.iter().cloned().fold(f64::MAX, f64::min);
    if gen_best < best {
        best = gen_best;
        println!("gen {:>4}   best = {:.3e}", gen, best);
    }

    // 4. Tell the strategy the results.
    ga.tell(population, fitnesses);
}
```

Notice that nothing here is magic. The harness is just this loop with some
bookkeeping (metrics, early stopping, observer callbacks) wrapped around it.

## The `seed_stream` and reproducibility

The `rng` you pass to `ask` is the only source of randomness in the entire
evolutionary run. `rlevo` derives every stochastic draw — initial population,
selection, crossover, mutation — from this single stream via a `seed_stream`
helper that produces independent, non-overlapping sub-streams from a root seed.

This means:

- Given the same root seed, two runs produce identical trajectories regardless
  of how many threads evaluated the population in between.
- Parallel population evaluation does not affect the trajectory, because
  evaluation results are deterministic given the genome (for a deterministic
  landscape) or are averaged/handled independently of the RNG stream.
- You can reproduce any result from a single integer seed logged at the start of
  a run.

> **Why this matters for research.** The canonical failure mode of irreproducible
> RL results — where "we ran 5 seeds and reported the mean" hides high variance —
> is partially mitigated by making every seed's trajectory exactly reproducible.
> If a result is surprising, you can re-run it exactly and confirm it was not an
> artifact of a specific random draw.

The host-RNG convention (all randomness through `seed_stream`, never via
`Backend::seed` or `Tensor::random`) is enforced by convention throughout
`rlevo::evo`. The contributor book documents why.

## What the harness adds

The `EvolutionaryHarness` wraps the above loop with:

- **Observer callbacks** — emit per-generation metrics to a `PopulationObserver`
  for TUI display or structured record export.
- **Convergence checks** — stop early if best fitness has not improved for
  \\(N\\) generations.
- **Parallel evaluation** — if your landscape implements `Send + Sync`, the
  harness uses `rayon` to evaluate the population in parallel automatically.

For most use cases, the harness is the right choice. Writing your own loop is
the right choice when you need control the harness does not expose, or when you
are building a hybrid (the ERL loop in Part III cannot be expressed as a single
harness call).

## Up next

The next section brings in a real environment, a reward signal, and a DQN agent.
The ask/tell vocabulary carries over — but instead of a static `Landscape`, the
score comes from an `Environment` that the agent acts inside over multiple
timesteps.

> **Foundations link.** The exploitation–exploration trade-off that drove our
> choice of tournament size and elitism is discussed in
> [What Is Optimization?](../part-1-foundations/01-optimization.md). The GA
> operators that `ask` uses internally — and their convergence properties — are
> derived in [Appendix A](../appendix-a-ec-algorithms/index.md).
