# The Ask/Tell Contract

The previous section used `EvolutionaryHarness` to run the GA — it handled the
loop for you. This section opens the harness and shows what is actually happening,
because you will need to understand it when:

- you want to add custom logging or early stopping,
- you are combining evolution with RL (where the loop is more complex), or
- the harness's assumptions don't match your problem.

## The contract

This is the same `Strategy` trait you saw in
[Part I](../part-1-foundations/20-evolutionary-computation.md). Here is the full
signature, because this time we are going to call it by hand:

```rust,no_run
pub trait Strategy<B: Backend>: Send + Sync {
    type Params: Clone + Debug + Send + Sync;  // static run config
    type State:  Clone + Debug + Send;          // generation-to-generation state
    type Genome: Clone + Send;                  // genome container produced by ask

    /// Build the initial state (samples the first population).
    fn init(&self, params: &Self::Params, rng: &mut dyn Rng, device: &B::Device) -> Self::State;

    /// Propose the next population; returns it together with an updated state.
    fn ask(&self, params: &Self::Params, state: &Self::State, rng: &mut dyn Rng, device: &B::Device)
        -> (Self::Genome, Self::State);

    /// Consume the population and its fitness; returns the next state and metrics.
    fn tell(&self, params: &Self::Params, population: Self::Genome, fitness: Tensor<B, 1>,
            state: Self::State, rng: &mut dyn Rng) -> (Self::State, StrategyMetrics);

    /// Best-so-far accessor — `None` before the first `tell`.
    fn best(&self, state: &Self::State) -> Option<(Self::Genome, f32)>;
}
```

Two things look different from a textbook ask/tell, and both are deliberate:

- **The strategy is pure.** `ask` and `tell` take `&self`, not `&mut self`, and
  thread the run state through return values rather than mutating in place. A
  `Strategy` owns no per-run state and no RNG — `init` produces the first
  `State`, and each call returns the next one. That is what lets many strategy
  instances run in parallel without locks, and makes `Clone`-based checkpointing
  trivial.
- **Fitness is a tensor, not a `Vec`.** `tell` takes a `Tensor<B, 1>` of shape
  `(pop_size,)` living on the same device as the population, so the strategy can
  do its selection arithmetic on-device without a host round-trip.

`ask` and `tell` are always paired: you call `ask`, evaluate every member of the
returned population against your objective, then call `tell` with that *same*
population alongside its fitness. The strategy never evaluates anything itself;
it only proposes and learns. That separation is what makes the rest possible:

- **You control evaluation.** Parallelize it, cache it, run it on a cluster, or
  evaluate against a human — the strategy does not care.
- **You own the candidates between `ask` and `tell`.** If your evaluation takes
  hours, that is fine; the strategy holds no lock on them.
- **Composition is natural.** Any code that calls `init`/`ask`/`tell` can drive
  any strategy. The harness, your custom loop, and the ERL hybrid all speak the
  same language.

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

Here is the GA sphere run from the previous section, rewritten without the
harness. We reuse the `Sphere` `Landscape` from Chapter 1; `FromLandscape`
adapts it into the `BatchFitnessFn` the strategy expects — a function that takes
the whole population at once and returns a `Tensor<B, 1>` of per-individual costs:

```rust,no_run
use burn::backend::Flex;
use rand::{SeedableRng, rngs::StdRng};
use rlevo::envs::landscapes::sphere::Sphere;
use rlevo::evo::algorithms::ga::{GaConfig, GeneticAlgorithm};
use rlevo::evo::fitness::{BatchFitnessFn, FromLandscape};
use rlevo::evo::strategy::Strategy;

type B = Flex;

fn main() {
    let device = Default::default();
    let strategy = GeneticAlgorithm::<B>::new();
    let params = GaConfig::default_for(/* pop = */ 64, /* dim = */ 8);
    let mut rng = StdRng::seed_from_u64(42);
    let mut fitness_fn = FromLandscape::new(Sphere::new(8));

    // init builds the first state and samples the initial population.
    let mut state = strategy.init(&params, &mut rng, &device);

    for gen in 0..300 {
        // 1. Ask: propose the next population and the state that goes with it.
        let (population, next_state) = strategy.ask(&params, &state, &mut rng, &device);

        // 2. Evaluate — this is your code. Returns a Tensor<B, 1> of (pop_size,).
        let fitness = fitness_fn.evaluate_batch(&population, &device);

        // 3. Tell: hand back the same population and its fitness; get the next
        //    state plus a metrics snapshot for this generation.
        let (new_state, metrics) = strategy.tell(&params, population, fitness, next_state, &mut rng);
        state = new_state;

        if gen % 25 == 0 {
            println!("gen {:>4}   best = {:.3e}", gen, metrics.best_fitness_ever);
        }
    }
}
```

> **Why `Flex`?** `Flex` is the Burn backend `rlevo`'s own examples, tests, and
> benches run on, so the book uses it too. Any `Backend` works —
> `init`/`ask`/`tell` are generic over `B` — but matching the codebase means the
> snippets above line up with what you'll see in `crates/rlevo-examples`.

A few things worth pointing out:

- **`init` then `ask`/`tell` in a loop.** `init` does the work the harness's
  `reset()` does; the body of the loop is exactly what one harness `step()` does.
- **State threads through the returns.** `ask` hands you `next_state`, which you
  pass straight into `tell`; `tell` hands you `new_state`, which becomes the
  `state` for the next iteration. Nothing is stored on the strategy.
- **The first cycle bootstraps.** On generation 0, `ask` returns the freshly
  sampled population unchanged (there is no fitness yet to select on); the first
  `tell` records it and primes `best_fitness_ever`. Every later `ask` runs the
  full selection → crossover → mutation pipeline.

Nothing here is magic. The harness is just this loop with bookkeeping (metrics
plumbing, early stopping, observer callbacks, parallel evaluation) wrapped
around it.

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
> [What Is Optimization?](../part-1-foundations/10-optimization.md). The GA
> operators that `ask` uses internally — and their convergence properties — are
> derived in [Appendix A](../appendix-a-ec-algorithms/index.md).

---

*Co-Authored-By: Anthropic Claude Opus 4.8*\
*Reviewed-By: (Human) Anthony Torlucci*
