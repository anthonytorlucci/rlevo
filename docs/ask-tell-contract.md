# The Ask/Tell Contract

Every evolutionary `Strategy` in `rlevo` speaks a single protocol:
**ask/tell**. Most programs never write it out, because `EvolutionaryLoop`
drives it for them. This page opens the protocol up, then shows the same run
through `EvolutionaryLoop` and through the evaluation harness, and says when
each is the right tool. Read it when you need to:

- understand what one generation is, and what `EvolutionaryLoop` does on your
  behalf,
- write the loop yourself, because your evaluation cannot run inside one
  synchronous call or because you are combining evolution with RL, or
- understand exactly why a run is — or isn't — reproducible.

> **Running the examples.** They need one dependency, `rlevo` at its default
> features, plus `rand` for the example that seeds an RNG by hand. Burn is
> reached through `rlevo::burn`.

## The contract

This is the same `Strategy` trait every algorithm in `rlevo::evo` implements —
the four-method contract (`init`/`ask`/`tell`/`best`) that maps the five-step
evolutionary loop (population → evaluate → select → vary → repeat) onto the
`Params`/`State`/`Genome` associated types, shared by the GA, ES, EP, DE, EDA,
and CGP families. Here is the full signature, because this time we are going to
call it by hand:

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
  any strategy. `EvolutionaryLoop`, your own loop, and the ERL hybrid all speak the
  same language.

## Lineage: evosax, and where `rlevo` diverges

This shape is not invented here. The direct inspiration is
[**evosax**](https://github.com/RobertTLange/evosax), the JAX library of
evolution strategies, which popularised a *functional* ask/tell interface:
`state = strategy.initialize(...)`, then `x, state = strategy.ask(rng, state,
params)` and `state = strategy.tell(x, fitness, state, params)`. The strategy is
a set of pure functions; all mutable run state and configuration live *outside*
it and thread through the calls. `rlevo`'s `&self` methods and the
`init → (ask, tell)*` state-threading are a deliberate port of that design to
Rust — the pure-strategy property above is exactly what makes an evosax strategy
`jax.jit`- and `vmap`-friendly, and what makes an `rlevo` strategy lock-free to
clone and run in parallel.

Three differences are worth calling out, because they are where the Rust/Burn
setting changes the contract rather than just the syntax:

- **One contract spans the whole EA taxonomy, not just ES.** evosax is
  organised around evolution strategies over real vectors. `rlevo`'s `Strategy`
  is generic over the genome *kind* — `Real`, `Binary`, `Integer`, and the
  roadmap `Permutation`/`Tree` kinds are zero-sized marker types implementing a
  `GenomeKind` trait, so operators specialise at compile time (Gaussian
  mutation only typechecks against a real genome, bit-flip only against a
  binary one) with no runtime dispatch or tag byte. A `Population<B, K>`
  wrapper welds that marker to its storage tensor so a wrong-tensor-for-this-kind
  population is unrepresentable. That kind marker is what varies across the
  *same* ask/tell trait backing the GA, ES, EP, DE, EDA, and CGP families — the
  associated `type Genome` changes, not the protocol.
- **Fitness is a backend tensor, with a fixed direction.** `tell` consumes a
  Burn `Tensor<B, 1>` on whatever backend the run uses, rather than a host
  array — selection arithmetic stays on-device. And where evosax leaves the
  optimisation direction to the caller, `rlevo` pins a **maximise-native contract**
  end-to-end (higher is better): a fitness function declares `ObjectiveSense`
  (required, no default — a reward or accuracy objective cannot be optimised
  backwards by omission), a cost objective spells out `ObjectiveSense::Minimize`
  explicitly, and `EvolutionaryLoop::step` is the sole chokepoint that
  negates a `Minimize` objective into the engine's maximise space before `tell`
  and maps metrics back to the declared sense for reporting. That same
  chokepoint also **sanitises** the tensor (ADR 0034): $\text{NaN} \to -\infty$ (so a broken
  individual can never masquerade as the champion — Rust's `f32::NAN` is
  positive and would otherwise rank as the maximum) and $+\infty \to \text{f32::MAX}$ (a
  genuinely optimal individual still ranks top, but stays finite). The fitness
  a `tell` driven by `EvolutionaryLoop` receives is therefore always finite or $-\infty$ — never a
  `NaN` or $+\infty$. Drive a strategy directly, bypassing the loop, and you lose
  that guarantee. You then do both jobs yourself before `tell`, in this order:
  negate a `Minimize` objective, then sanitise with
  `rlevo::evo::fitness::sanitize_fitness_tensor`, the function the loop itself
  calls. Keep that order. Sanitise before negating, and a `NaN` cost becomes
  the best individual in the population.
- **Randomness is a host stream, not a threaded key.** evosax splits and threads
  explicit JAX `PRNGKey`s. `rlevo` passes `&mut dyn Rng` and derives every draw
  from a single host `seed_stream` (below) — different plumbing for the same
  goal of seed-level reproducibility.

## Sequence diagram

```text
   your code                          Strategy                 fitness function
       │                                  │                           │
       │── init(params, rng) ────────────►│                           │
       │◄── state ────────────────────────│                           │
       │                                  │                           │
       │── ask(params, &state, rng) ─────►│                           │  ┐
       │◄── population, state ────────────│                           │  │
       │                                                              │  │ one
       │── evaluate_batch(&population) ──────────────────────────────►│  │ generation
       │◄── fitness ──────────────────────────────────────────────────│  │
       │                                  │                           │  │
       │── tell(params, population, ─────►│                           │  │
       │        fitness, state, rng)      │                           │  │
       │◄── state, metrics ───────────────│                           │  ┘
       │
       │  (repeat from ask)
```

The state travels in the arguments and the returns. Nothing is kept on the
strategy between calls, which is why something else has to hold it: your own
loop in the next section, or `EvolutionaryLoop` in the one after.

## Writing the loop yourself

Here is a genetic algorithm minimising the Sphere function, written without
`EvolutionaryLoop`. `Sphere` is a shipped `Landscape`; `FromLandscape` adapts it
into the `BatchFitnessFn` the strategy expects — a function that takes the
whole population at once and returns a `Tensor<B, 1>` of per-individual costs:

```rust,no_run
use rlevo::burn::backend::Flex;
use rand::{SeedableRng, rngs::StdRng};
use rlevo::envs::landscapes::sphere::Sphere;
use rlevo::evo::algorithms::ga::{GaConfig, GeneticAlgorithm};
use rlevo::evo::fitness::{BatchFitnessFn, FromLandscape, sanitize_fitness_tensor};
use rlevo::evo::strategy::Strategy;

type B = Flex;

fn main() {
    let device = Default::default();
    let strategy = GeneticAlgorithm::<B>::new();
    // `default_for` searches [-5.12, 5.12] in every dimension.
    let params = GaConfig::default_for(/* pop = */ 64, /* dim = */ 8);
    let mut rng = StdRng::seed_from_u64(42);
    let mut fitness_fn = FromLandscape::new(Sphere::new(8).expect("dim >= 1"));

    // init builds the first state and samples the initial population.
    let mut state = strategy.init(&params, &mut rng, &device);

    for generation in 0..300 {
        // 1. Ask: propose the next population and the state that goes with it.
        let (population, next_state) = strategy.ask(&params, &state, &mut rng, &device);

        // 2. Evaluate — this is your code. Returns a Tensor<B, 1> of (pop_size,):
        //    here, each individual's Sphere cost.
        let cost = fitness_fn.evaluate_batch(&population, &device);

        // 3. Canonicalise. A strategy maximises, and Sphere is a cost, so negate
        //    it. Leave this out and the GA climbs to the worst corner of the
        //    domain instead of descending to the origin. Then sanitise, so that
        //    a `NaN` or infinite score can never be ranked best. Negate first.
        let fitness = sanitize_fitness_tensor(cost.neg());

        // 4. Tell: hand back the same population and its fitness; get the next
        //    state plus a metrics snapshot for this generation.
        let (new_state, metrics) = strategy.tell(&params, population, fitness, next_state, &mut rng);
        state = new_state;

        if generation % 25 == 0 {
            // `metrics` is in the space `tell` saw, so negate again to read a cost.
            println!("gen {:>4}   best cost = {:.3e}", generation, -metrics.best_fitness_ever());
        }
    }
}
```

> **Why `Flex`?** `Flex` is the Burn backend `rlevo`'s own examples, tests, and
> benches run on, so this page uses it too. Any `Backend` works —
> `init`/`ask`/`tell` are generic over `B` — but matching the codebase means the
> snippets above line up with the examples in the repository.

A few things worth pointing out:

- **`init` then `ask`/`tell` in a loop.** `init` does the work `EvolutionaryLoop`'s
  `reset()` does, and the body of the loop is what one `EvolutionaryLoop::step()`
  does.
- **The direction is yours here.** Step 3 is the line a hand-written loop most
  easily forgets, and nothing fails when it is missing: the run completes and
  optimises the wrong way. `EvolutionaryLoop` reads the direction from the
  fitness function and does it for you. The sanitising on the same line is
  the loop's other job: Sphere never returns a `NaN`, so it changes nothing
  here, and it is what a fitness function that can diverge needs.
- **State threads through the returns.** `ask` hands you `next_state`, which you
  pass straight into `tell`; `tell` hands you `new_state`, which becomes the
  `state` for the next iteration. Nothing is stored on the strategy.
- **The first cycle bootstraps.** On generation 0, `ask` returns the freshly
  sampled population unchanged (there is no fitness yet to select on); the first
  `tell` records it and primes `best_fitness_ever`. Every later `ask` runs the
  full selection → crossover → mutation pipeline.

Nothing here is magic. `EvolutionaryLoop` is just this loop with bookkeeping
(the direction and sanitising step, metrics, observer callbacks, a generation
budget) wrapped around it.

## The `seed_stream` and reproducibility

The `rng` you pass to `ask` is the only source of randomness in the entire
evolutionary run. `rlevo` derives every stochastic draw — initial population,
selection, crossover, mutation — from this single stream via a `seed_stream`
helper that produces independent, non-overlapping sub-streams from one seed.

This means:

- Given the same seed, two runs produce identical trajectories regardless
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
`rlevo::evo`. `docs/rules.md` gives the reason, under "Host-RNG seeding
convention".

> **Determinism when trials run in parallel.** Because every draw comes from the
> host `seed_stream` and none from Burn's process-global generator, a loop's
> trajectory cannot be reached from another thread. On the `Flex` backend, seeded
> trials give the same scalars, bit for bit, whether the harness runs them on one
> thread or on a pool; a test in the repository pins it
> (`parallel_trials_match_serial_trials`). You do not need to pin the thread
> count to reproduce an evolution run. Two things can still break it: a fitness
> function of your own that draws from the backend generator (`Tensor::random`),
> and the `Wgpu` backend, where a run follows the `Flex` trajectory only to within
> the GPU's floating-point differences.

## The same run with `EvolutionaryLoop`

The hand-written loop above keeps four things alive between calls: the
strategy's state, the RNG, the fitness function and the generation count.
`EvolutionaryLoop` owns all four and reduces the run to three calls:

```rust,no_run
use rlevo::burn::backend::Flex;
use rlevo::envs::landscapes::sphere::Sphere;
use rlevo::evo::algorithms::ga::{GaConfig, GeneticAlgorithm};
use rlevo::evo::fitness::FromLandscape;
use rlevo::evo::strategy::EvolutionaryLoop;

type B = Flex;

let device = Default::default();
let mut evo_loop = EvolutionaryLoop::<B, _, _>::new(
    GeneticAlgorithm::<B>::new(),
    GaConfig::default_for(/* pop = */ 64, /* dim = */ 8),
    FromLandscape::new(Sphere::new(8).expect("dim >= 1")),
    /* seed = */ 42,
    device,
    /* max_generations = */ 300,
)
.expect("valid config");

evo_loop.reset();
while !evo_loop.is_done() {
    evo_loop.step();
}
let metrics = evo_loop.latest_metrics().expect("a generation has run");
println!("best = {:.3e}", metrics.best_fitness_ever());
```

From the same seed this follows the same trajectory as the hand-written loop
above, generation for generation. It is the same sequence of calls:

| Call | What it does | In the hand-written loop |
|---|---|---|
| `reset()` | seeds the RNG, zeroes the generation count, builds the first state | `StdRng::seed_from_u64(42)` and `strategy.init(..)` |
| `step()` | one generation: ask, score, negate the cost, sanitise, tell | the body of the `for` loop |
| `is_done()` | whether `max_generations` generations have run | the `0..300` range |

**Why `step()` returns nothing.** Advancing a run and asking whether it is over
are two calls on purpose. A `step()` that returned a flag would be written
`while !evo_loop.step() {}`, and the reader would have to remember which way the
flag points. With the question separate, the loop reads as what it does, and it
asks *before* it runs, so a budget of zero runs no generation. `is_done()` is
also the word `rlevo` uses to end an episode (`Snapshot::is_done()`), so both
kinds of run stop on the same question.

**The budget is advisory to `step()`.** `is_done()` turns `true` when the budget
is spent; it is your loop that stops there. `step()` runs a generation every
time it is called. That is deliberate. A `step()` that refused would have to
either do nothing, which hides a mistake behind a run that looks complete, or
panic, which forbids carrying a run on a little longer when you mean to.
`is_done()` reports the budget and nothing else: it does not say the search has
converged.

**`reset()` is a call of its own.** `new` only checks the configuration and
stores the parts. `reset()` builds the first state, and it is also how a run is
replayed: call it again and the same seed produces the same run. Skip it and the
first `step()` panics. The seed is fixed at `new`, so a different seed means a
new loop, not another `reset()`.

**Stopping on your own condition.** The loop stops on nothing but the budget.
There is no early stopping built in. Read the metrics after a step and leave:

```rust,ignore
while !evo_loop.is_done() {
    evo_loop.step();
    let metrics = evo_loop.latest_metrics().expect("a generation has run");
    if metrics.best_fitness_ever() < 1e-6 {
        break;
    }
}
```

### What it adds to the hand-written loop

- **Direction and hygiene** — the `ObjectiveSense` reconciliation and the
  sanitising step described under the lineage section above. They happen in
  `step()` and nowhere else, which is why a strategy never sees a `NaN` and you
  never negate a cost.
- **Validation** — `new` checks the strategy's configuration and returns an
  error for an invalid one, before any tensor is built.
- **Replay** — `reset()` re-seeds from the loop's seed, so two runs of one loop
  are identical.
- **Metrics and observers** — `latest_metrics()` after every step, a `tracing`
  event per generation, and an optional `PopulationObserver` for per-individual
  fitness.

`EvolutionaryLoop` calls your `BatchFitnessFn` once per generation; it does not
parallelize the population evaluation itself (the shipped `FromLandscape` and
`FromFitnessEvaluable` adapters evaluate a batch with a plain, serial loop —
`Send + Sync` on your landscape is a bound the trait asks for, not a signal
that triggers `rayon`). If you need per-individual parallelism, implement it
inside your `BatchFitnessFn`.

### Many runs: give the evaluation harness a loop per trial

One run is `reset`, `step`, `is_done`. For many seeded runs with a report at
the end, do not write that loop at all. The evaluation harness, `rlevo::harness`,
runs the trials for you. You give it a closure that builds one loop from the
seed it is handed; the harness calls the closure once per trial and drives each
loop to its budget.

```rust,no_run
use rlevo::burn::backend::Flex;
use rlevo::envs::landscapes::sphere::Sphere;
use rlevo::evo::algorithms::ga::{GaConfig, GeneticAlgorithm};
use rlevo::evo::fitness::FromLandscape;
use rlevo::evo::strategy::EvolutionaryLoop;
use rlevo::harness::evaluator::{Evaluator, EvaluatorConfig, GenerationTrial};
use rlevo::harness::reporter::logging::LoggingReporter;

type B = Flex;
const GENERATIONS: usize = 300;

let config = EvaluatorConfig {
    base_seed: 2026,         // the suite's seed; each trial's is derived from it
    num_trials_per_env: 10,  // ten trials, so ten different loop seeds
    max_steps: GENERATIONS,  // must be at least the loop's budget; see below
    ..EvaluatorConfig::default()
};
let mut reporter = LoggingReporter::new();
let report = Evaluator::new(config).run_trials(
    "sphere-ga",
    &["sphere-d8".to_string()],
    // Called once per trial with the seeds the harness derived for that trial.
    // The first is the one a loop takes: build a new loop from it each time.
    |_key, seed, _agent_seed| GenerationTrial {
        probe: EvolutionaryLoop::<B, _, _>::new(
            GeneticAlgorithm::<B>::new(),
            GaConfig::default_for(/* pop = */ 64, /* dim = */ 8),
            FromLandscape::new(Sphere::new(8).expect("dim >= 1")),
            seed,
            Default::default(),
            GENERATIONS,
        )
        .expect("valid config"),
    },
    &mut reporter,
);

for trial in &report.trials {
    // The last generation's metrics, in the objective's own direction.
    println!(
        "trial {:>2}: best cost {:.3e} after {} generations",
        trial.key.trial_idx,
        trial.scalars["ea/best_fitness_ever"],
        trial.scalars["generations"],
    );
}
```

Three things decide whether such a run is the one you meant:

- **One loop per trial, each with its own seed.** There are two numbers here
  and they are not the same one. `EvaluatorConfig::base_seed` is the suite's:
  one number, from which the harness derives a different seed for every trial.
  The `seed` the closure receives is one of those derived seeds, and it becomes
  the loop's seed. Give ten trials one `base_seed` and they are ten different
  runs; give the suite the same `base_seed` again and it repeats all ten. A
  loop's seed is fixed when it is built, so the closure builds a new loop each
  time. Resetting one loop ten times would replay the same run ten times.
- **`max_steps` caps a generation trial.** The harness stops a trial at the
  loop's `max_generations` or at `EvaluatorConfig::max_steps`, whichever comes
  first, and `max_steps` defaults to 1000. Set `max_steps` at least as large as
  the loop's budget. If the cap ends a trial that had budget left, the harness
  logs a warning that names the trial and the cap, and the `generations` scalar
  in that trial's report is the cap, not the budget. The warning is a `tracing`
  event, as `LoggingReporter`'s own lines are: a program that installs no
  `tracing` subscriber sees neither, and has only the `generations` scalar to
  go by.
- **Threads.** Trials run concurrently on a thread pool
  (`EvaluatorConfig::num_threads`). On `Flex` the results do not depend on the
  thread count; see the determinism note above for the two exceptions.

The harness drives each loop through a trait, `GenerationProbe`, whose two
methods are built from the three calls above: `begin` is `reset`, and `advance`
is "if `is_done`, stop; otherwise `step` and hand back the metrics". You never
call `begin` or `advance` yourself. They are the harness's side of the seam, and
`EvolutionaryLoop` already implements them. You would implement `GenerationProbe`
only for a generation loop of your own that the harness should run.

| Who drives | Start | One generation | Finished when |
|---|---|---|---|
| your own code | `reset()` | `step()` | `is_done()` |
| `rlevo::harness` | `begin()` | `advance()` | `advance()` returns `None` |

The harness has no early stopping: a trial runs to the budget. To stop each of
several seeds on a condition of your own, drive a loop per seed yourself.

### Which to use

- **`EvolutionaryLoop`** for almost everything: one optimisation run, with your
  own logging and your own stopping rule.
- **The harness** when you want several seeds and a comparison.
- **Your own `init`/`ask`/`tell` loop** when evaluation cannot happen inside one
  synchronous `evaluate_batch` call (a cluster job, a human, an evaluation that
  takes hours), or when one loop has to drive several strategies or populations
  at once, as a hybrid of evolution and RL does. You then own everything the
  loop did for you: the RNG and its seed, the generation count, the direction
  (negate a cost), the sanitising of non-finite fitness
  (`sanitize_fitness_tensor`, after the negation), the validation of the
  configuration, and turning the metrics back into your objective's own
  direction when you read them. To let the harness run a loop of your own,
  implement `GenerationProbe` for it: `begin` re-seeds and starts over, and
  `advance` checks your budget, runs one generation and returns its metrics,
  or `None` once the budget is spent.

## Where this connects

- [Objective Adapters](objective-adapters.md) shows how to wire a function of
  your own into the `BatchFitnessFn` that every example on this page takes.
- The RL loop that trains a CartPole agent with DQN is *not* ask/tell — the
  agent acts sequentially, learning after individual steps rather than after a
  full population evaluation. The vocabulary returns when evolution and RL
  combine: the EA `ask`s for a population of policies, evaluates each by
  running a full episode, and `tell`s the scores back.

> **Exploration vs. exploitation.** The tournament-size and elitism choices in
> the code above trade off *exploitation* (concentrating on candidates already
> known to be good) against *exploration* (probing unknown regions that might
> be better) — the same tension that appears as the exploration–exploitation
> dilemma in RL and the bias–variance trade-off in statistics. Turning
> tournament size up exploits harder; dropping elitism (favouring
> `GaReplacement::Generational`) trades monotonic improvement for more
> exploration. The GA operators used in the code above — tournament selection,
> BLX-$\alpha$ crossover, elitist replacement — are the same operators, with the same
> convergence behaviour, that back every real-coded strategy in `rlevo::evo`.
