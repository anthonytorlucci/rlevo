# Local refinement: memetic algorithms

> **Status:** stub — prose and tested include coming in a follow-up PR.

**The problem.** A plain GA explores broadly but wastes evaluations near the
optimum: it has no memory of gradient-like local structure. A **memetic
algorithm** bolts a local searcher onto the evolutionary loop so each individual
can hill-climb a few steps before being scored.

**Learning goal.** `MemeticWrapper` as a transparent `Strategy` wrapper: keep
any existing strategy, add a local search policy, and watch convergence
accelerate on smooth landscapes — without changing the harness or any other code.

## The new seams

- `MemeticWrapper<B, S, L, F>` — wraps an inner `Strategy<B>` and a
  `LocalSearch<B>` implementor (ADR-0016).
- `LocalSearch<B>` trait — `Vec<f32>` genomes, `&mut dyn FitnessFn<Vec<f32>>`,
  `&mut dyn Rng`. Concrete implementations:
  - `HillClimbing`
  - `NelderMead`
  - `SimulatedAnnealing`
  - `RandomRestart`
- `WritebackPolicy` — `Lamarckian` (write back to genome), `Baldwinian` (score
  only, don't overwrite), `Partial(f)` (interpolate).

## Story beat

On Rastrigin-D10, the memetic GA with `SimulatedAnnealing` as the local searcher
closes the gap that a plain GA leaves after 200 generations. The insight: local
search is *free* perspective on the neighbourhood the population has already
paid to reach.

## Outline

1. Wrapping `GeneticAlgorithm` in `MemeticWrapper` — two lines of change.
2. Choosing a local searcher — `HillClimbing` for smooth bowls, `SimulatedAnnealing`
   for rough ones.
3. `WritebackPolicy` — Lamarckian vs Baldwinian and what it means for inheritance.
4. Running the same Sphere and Rastrigin benchmarks as Ch1 with the memetic GA.
5. Make it yours — try `NelderMead` on Rosenbrock.

## Example

```bash
cargo run -p rlevo-examples --example ch05_memetic_ga
```

<!-- TODO: {{#rustdoc_include ../../../crates/rlevo-examples/examples/book/ch05_memetic_ga.rs}} -->
